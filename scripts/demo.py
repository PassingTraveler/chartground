"""Minimal single-image inference entry point; UI layers can wrap this function."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from tokenizers import Tokenizer

from model.mm_minimind import MMConfig, MMForCausalLM
from model.projector import VisionProjector
from model.vision_encoder import FrozenSigLIP2
from train.mm_dataset import project_vision_features
from model.mm_template import MMTemplate


def load_checkpoint(path: Path, tokenizer_size: int, device: torch.device):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in raw:
        # A bare pretrain checkpoint has no projector; the caller would get a
        # randomly initialized projector and silently wrong answers.
        raise SystemExit(
            f"{path} is not an MM checkpoint (no 'model' key). Run train_align.py / "
            "train_mm_sft.py first, or point --checkpoint at their output."
        )
    cfg_data = raw.get("config", {})
    cfg = MMConfig(vocab_size=tokenizer_size, **{k: v for k, v in cfg_data.items() if k in MMConfig.__dataclass_fields__ and k != "vocab_size"})
    model = MMForCausalLM(cfg).to(device)
    model.load_state_dict(raw["model"], strict=False)
    return model, raw.get("projector")


def answer(checkpoint: Path, image_path: Path, question: str, vision_model: str, tokenizer_path: Path,
           max_new_tokens: int = 128) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model, projector_state = load_checkpoint(checkpoint, len(tokenizer.get_vocab()), device)
    vision = FrozenSigLIP2(vision_model, device=device, dtype=torch.float32).load()
    projector = VisionProjector(vision.hidden_size, model.config.hidden_size, 1024).to(device)
    if projector_state:
        projector.load_state_dict(projector_state, strict=False)
    template = MMTemplate()
    prompt = template.prompt(question, "image", 1, tokens_per_image=model.config.image_token_len)
    ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False).ids], dtype=torch.long, device=device)
    with Image.open(image_path).convert("RGB") as image:
        pixels = vision.processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        features = project_vision_features(vision.encode(pixels), projector, model.config.image_token_len)
        output = model.generate(ids, vision_features=features, max_new_tokens=max_new_tokens, eos_id=2)
    return tokenizer.decode(output[0, ids.size(1):].tolist()).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--vision-model", required=True)
    ap.add_argument("--tokenizer", type=Path, default=Path("assets/project1/tokenizer/best_mm.json"))
    args = ap.parse_args()
    print(answer(args.checkpoint, args.image, args.question, args.vision_model, args.tokenizer))


if __name__ == "__main__":
    main()
