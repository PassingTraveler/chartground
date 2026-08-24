from __future__ import annotations

import argparse
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
    ap = argparse.ArgumentParser(description="projector-only image alignment")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--vision-model", required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--save", type=Path, default=Path("out/align/projector.pt"))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    cfg = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, scaler = configure_precision(device, args.precision)
    print(f"[align] device={device} precision={args.precision}")
    image_id = tokenizer.token_to_id("<|image_pad|>")
    video_id = tokenizer.token_to_id("<|video_pad|>")
    mm_cfg = MMConfig(
        vocab_size=len(tokenizer.get_vocab()),
        max_position_embeddings=cfg.max_seq_len,
        image_token_id=12003 if image_id is None else int(image_id),
        video_token_id=12004 if video_id is None else int(video_id),
        image_token_len=cfg.image_token_len,
    )
    lm = MMForCausalLM.from_project1_checkpoint(mm_cfg, str(args.checkpoint), device)
    for p in lm.parameters():
        p.requires_grad_(False)
    vision = FrozenSigLIP2(args.vision_model, device=device, dtype=torch.float32).load()
    projector = VisionProjector(vision.hidden_size, cfg.hidden_size, cfg.projector_hidden_size).to(device)
    ds = MMDataset(args.data, tokenizer, cfg.max_seq_len, vision.processor, cfg.image_token_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mm)
    opt = torch.optim.AdamW(projector.parameters(), lr=args.lr)
    lm.eval()
    projector.train()
    step = 0
    empty_batches = 0
    while step < args.steps:
        for batch in loader:
            if batch["pixel_values"] is None:
                # Guard against infinite idle spinning when the manifest has no
                # image rows at all (or every row failed to decode).
                empty_batches += 1
                if empty_batches > max(100, args.steps * 3):
                    raise SystemExit(
                        f"no image samples in the last {empty_batches} batches; "
                        "check that --data contains image rows and the images exist"
                    )
                continue
            empty_batches = 0
            pixels = batch["pixel_values"].to(device)
            with torch.no_grad():
                raw = vision.encode(pixels)
            with autocast_context(device, amp_dtype):
                features = project_vision_features(raw, projector, cfg.image_token_len)
                ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                # Use the frozen language model as the alignment signal: the projector
                # must make image slots useful for predicting the answer tokens.
                # This is a real task loss, while keeping the stage cheap and auditable.
                _, loss, _ = lm(ids, labels=labels, vision_features=features)
            opt.zero_grad(set_to_none=True)
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            if scaler is None:
                opt.step()
            else:
                scaler.step(opt); scaler.update()
            step += 1
            if step % 10 == 0:
                print(f"[align] step={step} loss={loss.item():.5f}")
            if step >= args.steps:
                break
    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save(projector.state_dict(), args.save)
    print(f"[align] saved={args.save}")


if __name__ == "__main__":
    main()
