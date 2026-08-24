"""Small deterministic model smoke test.

The optional vision branch uses synthetic projected features, so it validates
the marker protocol without requiring a multi-gigabyte SigLIP checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Resolve `model` imports when run as a plain script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--with-vision", action="store_true")
    args = ap.parse_args()
    try:
        import torch
    except ImportError as e:
        raise SystemExit("PyTorch is not installed in this environment; data smoke can still run via scripts/run_data_smoke.ps1") from e
    from tokenizers import Tokenizer
    from model.mm_minimind import MMConfig, MMForCausalLM
    tok = Tokenizer.from_file(str(args.tokenizer))
    image_id = tok.token_to_id("<|image_pad|>") or 12003
    cfg = MMConfig(vocab_size=len(tok.get_vocab()), image_token_id=image_id, num_hidden_layers=2, hidden_size=128, intermediate_size=256, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    model = MMForCausalLM(cfg).cuda() if torch.cuda.is_available() else MMForCausalLM(cfg)
    ids = torch.tensor([[1, 4, 5, 2]], dtype=torch.long, device=next(model.parameters()).device)
    labels = ids.clone()
    _, loss, _ = model(ids, labels=labels)
    if not torch.isfinite(loss):
        raise SystemExit("non-finite text loss")
    result = {"status": "ok", "device": str(next(model.parameters()).device), "text_loss": float(loss)}
    if args.with_vision:
        # A mixed batch is intentional: row 0 is text-only, row 1 contains four
        # image markers. This catches the most common collate/injection bug.
        image_ids = torch.tensor([[1, image_id, image_id, image_id, image_id, 2]], dtype=torch.long, device=ids.device)
        mixed_ids = torch.cat([torch.nn.functional.pad(ids, (0, 2), value=0), image_ids], dim=0)
        mixed_labels = mixed_ids.clone()
        features = torch.randn(2, 4, cfg.hidden_size, device=ids.device)
        _, image_loss, _ = model(mixed_ids, labels=mixed_labels, vision_features=features)
        if not torch.isfinite(image_loss):
            raise SystemExit("non-finite multimodal loss")
        result["multimodal_loss"] = float(image_loss)
        result["marker_count"] = 4
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
