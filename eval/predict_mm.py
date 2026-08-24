"""Shardable prediction generator for the multimodal eval suite.

Splits a manifest by row across M shards (--shard i/M) so 4 GPUs can each
process a disjoint slice, then --merge concatenates the shards into one
JSONL. With --no-image, visual rows are evaluated without their image
(identical prompt) to support the paired visual-gain (Δvis) protocol.

Output JSONL rows: {"example_id", "prediction", "answer", "prompt"}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer

from model.mm_minimind import MMConfig, MMForCausalLM
from model.mm_template import MMTemplate
from model.projector import VisionProjector
from model.vision_encoder import FrozenSigLIP2
from train.mm_dataset import project_vision_features


def load_checkpoint(path: Path, tokenizer_size: int, device: torch.device):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in raw:
        raise SystemExit(f"{path} is not an MM checkpoint (no 'model' key); align/SFT output required")
    cfg_data = raw.get("config", {})
    cfg = MMConfig(vocab_size=tokenizer_size, **{k: v for k, v in cfg_data.items() if k in MMConfig.__dataclass_fields__ and k != "vocab_size"})
    model = MMForCausalLM(cfg).to(device)
    missing, unexpected = model.load_state_dict(raw["model"], strict=False)
    print(f"[predict] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return model, raw.get("projector"), raw.get("audio_projector")


def predict(manifest: Path, checkpoint: Path, vision_model: str, tokenizer_path: Path,
            out: Path, shard: tuple[int, int] | None = None, no_image: bool = False,
            max_new_tokens: int = 128, audio: bool = False, audio_hidden_size: int = 768,
            image_tokens: int | None = None, audio_include_question: bool = False) -> int:
    rows = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    if shard is not None:
        i, m = shard
        rows = rows[i::m]
    if not rows:
        return 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model, projector_state, audio_projector_state = load_checkpoint(checkpoint, len(tokenizer.get_vocab()), device)
    if image_tokens:
        # Zero-train resolution probe: the projector is a per-token MLP, so
        # the interpolate frame count (and prompt marker count) can be
        # changed at inference time without touching weights.
        model.config.image_token_len = image_tokens
    model.eval()
    projector = VisionProjector(model.config.image_hidden_size if hasattr(model.config, "image_hidden_size") else 768,
                                model.config.hidden_size, 1024).to(device)
    if projector_state:
        projector.load_state_dict(projector_state, strict=False)
    projector.eval()
    audio_projector = None
    if audio:
        if audio_projector_state is None:
            raise SystemExit("--audio requires a checkpoint with an 'audio_projector' "
                             "(train with --audio-projector); the align/SFT checkpoint lacks one")
        audio_projector = VisionProjector(audio_hidden_size, model.config.hidden_size, 1024).to(device)
        audio_projector.load_state_dict(audio_projector_state, strict=False)
        audio_projector.eval()
    vision = None
    if not no_image:
        vision = FrozenSigLIP2(vision_model, device=device, dtype=torch.float32).load()
    template = MMTemplate()
    results = []
    for row in rows:
        modality = row.get("modality", "text")
        question = row.get("prompt") or ""
        is_audio_row = audio and (modality == "audio" or row.get("audio"))
        features = None
        audio_features = None
        if is_audio_row:
            # Audio arm: question heard via the whisper-feature span (no text);
            # the chart stays visible because bar/line rows put every number
            # in the figure. Control arm = default path below (written
            # question + same image, since audio rows keep the image field).
            text = template.prompt_audio(question, 1, tokens_per_image=model.config.image_token_len,
                                         n_audio_tokens=model.config.audio_token_len,
                                         include_question=audio_include_question)
            feat_path = Path(row["audio_features"])
            if not feat_path.is_absolute():
                feat_path = manifest.parent / feat_path
            raw_audio = torch.from_numpy(np.ascontiguousarray(np.load(feat_path), dtype=np.float32))
            with torch.no_grad():
                audio_features = project_vision_features(
                    raw_audio.unsqueeze(0).to(device), audio_projector, model.config.audio_token_len)
        elif modality == "image" or row.get("image"):
            if no_image:
                text = template.prompt(question, "text")
                features = None
            else:
                text = template.prompt(question, "image", 1, tokens_per_image=model.config.image_token_len)
                features = None
        else:
            text = template.prompt(question, "text")
            features = None
        if features is None and (is_audio_row or (modality == "image" or row.get("image"))) and not no_image:
            image_path = Path(row["image"])
            if not image_path.is_absolute():
                image_path = manifest.parent / image_path
            with Image.open(image_path).convert("RGB") as img:
                pixels = vision.processor(images=img, return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                raw = vision.encode(pixels)
                features = project_vision_features(raw, projector, model.config.image_token_len)
        ids = torch.tensor([tokenizer.encode(text, add_special_tokens=False).ids], dtype=torch.long, device=device)
        with torch.no_grad():
            output = model.generate(ids, vision_features=features, audio_features=audio_features,
                                    max_new_tokens=max_new_tokens, eos_id=2)
        results.append({
            "example_id": row.get("example_id"),
            "prediction": tokenizer.decode(output[0, ids.size(1):].tolist()).strip(),
            "answer": row.get("answer", ""),
            "prompt": question,
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(results)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--vision-model", required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="JSONL output (or shard prefix with --shard)")
    ap.add_argument("--shard", help="i/M to process only rows i, i+M, ...")
    ap.add_argument("--no-image", action="store_true", help="evaluate visual rows without the image (Δvis control)")
    ap.add_argument("--audio", action="store_true",
                    help="audio arm: use the whisper-feature question span instead of text (Δaudio)")
    ap.add_argument("--audio-hidden-size", type=int, default=768, help="whisper encoder d_model (small=768/base=512)")
    ap.add_argument("--audio-include-question", action="store_true",
                    help="diagnostic arm: whisper-feature span PLUS the written question "
                         "(decomposes Δaudio into perception loss vs alignment loss)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--image-tokens", type=int, default=None,
                    help="override image_token_len at inference (zero-train resolution probe)")
    args = ap.parse_args()
    if args.shard:
        i, m = (int(x) for x in args.shard.split("/"))
        out = args.out.with_name(f"{args.out.stem}.shard{i}of{m}{args.out.suffix}")
        n = predict(args.manifest, args.checkpoint, args.vision_model, args.tokenizer,
                    out, (i, m), args.no_image, args.max_new_tokens, args.audio, args.audio_hidden_size,
                    args.image_tokens, args.audio_include_question)
        print(json.dumps({"shard": f"{i}/{m}", "rows": n, "out": str(out)}))
    else:
        n = predict(args.manifest, args.checkpoint, args.vision_model, args.tokenizer,
                    args.out, None, args.no_image, args.max_new_tokens, args.audio, args.audio_hidden_size,
                    args.image_tokens, args.audio_include_question)
        print(json.dumps({"rows": n, "out": str(args.out)}))


if __name__ == "__main__":
    main()
