"""Audio-projector-only alignment, mirroring train_align.py.

Freezes the LLM, SigLIP2, AND the visual projector (audio rows reference the
chart image, so the image path must stay fixed at the visual align output),
and trains only the audio projector so the frozen LLM can turn whisper
features into useful answer tokens. Audio features are precomputed npy files
(data/precompute_audio_features.py) — no whisper inference at train time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import ModelConfig
from model.mm_minimind import MMConfig, MMForCausalLM
from model.projector import VisionProjector
from model.vision_encoder import FrozenSigLIP2
from train.mm_dataset import MMDataset, collate_mm, project_vision_features
from train.amp import autocast_context, configure_precision


def main() -> None:
    ap = argparse.ArgumentParser(description="audio-projector-only alignment")
    ap.add_argument("--data", type=Path, required=True, help="audio train manifest (with audio_features npy)")
    ap.add_argument("--checkpoint", type=Path, required=True, help="project1 pretrain checkpoint")
    ap.add_argument("--vision-model", required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--visual-projector", type=Path, required=True, help="out/align/projector.pt (frozen)")
    ap.add_argument("--whisper-model", default="openai/whisper-small")
    ap.add_argument("--audio-hidden-size", type=int, default=768)
    ap.add_argument("--save", type=Path, default=Path("out/align_audio/audio_projector.pt"))
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    cfg = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, scaler = configure_precision(device, args.precision)
    print(f"[align-audio] device={device} precision={args.precision}")
    image_id = tokenizer.token_to_id("<|image_pad|>")
    video_id = tokenizer.token_to_id("<|video_pad|>")
    audio_id = tokenizer.token_to_id("<|audio_pad|>")
    mm_cfg = MMConfig(
        vocab_size=len(tokenizer.get_vocab()),
        max_position_embeddings=cfg.max_seq_len,
        image_token_id=12003 if image_id is None else int(image_id),
        video_token_id=12004 if video_id is None else int(video_id),
        audio_token_id=12005 if audio_id is None else int(audio_id),
        image_token_len=cfg.image_token_len,
        audio_token_len=cfg.audio_token_len,
    )
    lm = MMForCausalLM.from_project1_checkpoint(mm_cfg, str(args.checkpoint), device)
    for p in lm.parameters():
        p.requires_grad_(False)
    vision = FrozenSigLIP2(args.vision_model, device=device, dtype=torch.float32).load()
    visual_projector = VisionProjector(vision.hidden_size, cfg.hidden_size, cfg.projector_hidden_size).to(device)
    visual_projector.load_state_dict(torch.load(args.visual_projector, map_location="cpu", weights_only=False), strict=False)
    for p in visual_projector.parameters():
        p.requires_grad_(False)
    audio_projector = VisionProjector(args.audio_hidden_size, cfg.hidden_size, cfg.projector_hidden_size).to(device)
    # The audio manifest keeps image paths relative to figure_math/ (audio
    # rows still reference the chart); resolve every asset to an absolute
    # path exactly like build_sft_jsonl.load does, through a temp manifest.
    from data.build_sft_jsonl import load as resolve_rows
    rows = resolve_rows(args.data, "audio_math")
    tmp_manifest = Path("/tmp") / f"align_audio_{args.data.stem}.jsonl"
    tmp_manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    ds = MMDataset(tmp_manifest, tokenizer, cfg.max_seq_len, vision.processor, cfg.image_token_len, cfg.audio_token_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mm)
    opt = torch.optim.AdamW(audio_projector.parameters(), lr=args.lr)
    lm.eval(); vision.model.eval()
    audio_projector.train()
    step = 0
    empty_batches = 0
    while step < args.steps:
        for batch in loader:
            if batch["audio_features"] is None:
                empty_batches += 1
                if empty_batches > max(100, args.steps * 3):
                    raise SystemExit(
                        f"no audio samples in the last {empty_batches} batches; "
                        "check that --data contains audio rows with precomputed features"
                    )
                continue
            empty_batches = 0
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.no_grad():
                raw_audio = batch["audio_features"].to(device)
                visual = None
                if batch["pixel_values"] is not None:
                    raw_vis = vision.encode(batch["pixel_values"].to(device))
                    visual = project_vision_features(raw_vis, visual_projector, cfg.image_token_len)
            with autocast_context(device, amp_dtype):
                audio_features = project_vision_features(raw_audio, audio_projector, cfg.audio_token_len)
                # Same frozen-LLM alignment signal as train_align.py: the audio
                # projector must make audio slots useful for the answer tokens.
                _, loss, _ = lm(ids, labels=labels, vision_features=visual, audio_features=audio_features)
            opt.zero_grad(set_to_none=True)
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(audio_projector.parameters(), 1.0)
            if scaler is None:
                opt.step()
            else:
                scaler.step(opt); scaler.update()
            step += 1
            if step % 10 == 0:
                print(f"[align-audio] step={step} loss={loss.item():.5f}")
            if step >= args.steps:
                break
    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save(audio_projector.state_dict(), args.save)
    print(f"[align-audio] saved={args.save}")


if __name__ == "__main__":
    main()
