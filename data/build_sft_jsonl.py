"""Merge visual math, image VQA, no-image math and text replay records."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .schema import read_jsonl, write_jsonl


def load(path: Path, role: str) -> list[dict]:
    rows = read_jsonl(path)
    for row in rows:
        row.setdefault("metadata", {})["mix_role"] = role
        for field in ("image", "video", "audio", "audio_features"):
            if row.get(field):
                asset = Path(row[field])
                if not asset.is_absolute():
                    row[field] = str((path.parent / asset).resolve())
    return rows


def _row_text(row: dict[str, Any]) -> str:
    prompt = row.get("prompt") or row.get("messages", [{}])[0].get("content", "")
    answer = row.get("answer") or row.get("messages", [{}, {"content": ""}])[-1].get("content", "")
    return f"{prompt}\n{answer}"


def _row_tokens(row: dict[str, Any], tokenizer=None) -> int:
    text = _row_text(row)
    if tokenizer is not None:
        cost = max(1, len(tokenizer.encode(text, add_special_tokens=False).ids))
    else:
        cost = max(1, len(text) // 2)
    # Image rows also carry 64 <|image_pad|> tokens in the training sequence;
    # video rows carry 4 frames x 64 (see model/mm_template.py). Counting only
    # text tokens systematically underweights image data in the token mix.
    if row.get("image"):
        cost += 64
    if row.get("video"):
        cost += 4 * 64
    # Audio rows carry the <|audio_start|> + 256 <|audio_pad|> + <|audio_end|>
    # span in the training sequence (model/mm_template.py prompt_audio).
    if row.get("audio") or row.get("audio_features"):
        cost += 256 + 2
    return cost


def _mix_ratios() -> dict[str, float]:
    try:
        from config import DataMixConfig
        cfg = DataMixConfig()
        cfg.validate()
        return {
            "general_image": cfg.general_image_ratio,
            "text_math_sft": cfg.text_math_ratio,
            "visual_math": cfg.visual_math_ratio,
            "text_replay": cfg.text_replay_ratio,
            "audio_math": cfg.audio_math_ratio,
        }
    except ImportError:
        # Fall back to README §4.3 defaults when config.py is not importable.
        return {"general_image": 0.45, "text_math_sft": 0.15, "visual_math": 0.25,
                "text_replay": 0.15, "audio_math": 0.0}


def build(paths: dict[str, Path], output: Path, seed: int = 42, max_rows: int | None = None,
          max_tokens: int | None = None, tokenizer=None) -> int:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for role, path in paths.items():
        if path is None:
            continue
        if not path.exists():
            raise FileNotFoundError(f"input for mix role '{role}' does not exist: {path}")
        groups[role] = load(path, role)
    if not groups:
        raise FileNotFoundError("no input JSONL exists")
    for rows in groups.values():
        rng.shuffle(rows)
    ratios = _mix_ratios()
    merged = []
    if max_tokens is not None:
        for role, ratio in ratios.items():
            rows = groups.get(role, [])
            if not rows:
                continue
            budget = max(1, int(max_tokens * ratio))
            used = 0
            selected = 0
            for row in rows:
                cost = _row_tokens(row, tokenizer)
                if cost > budget:
                    # Row alone exceeds the whole budget; never force it in.
                    continue
                if used + cost > budget:
                    # Skip rows that do not fit, keep filling with shorter rows.
                    continue
                merged.append(row)
                used += cost
                selected += 1
            print(f"mix_role={role} rows={selected} tokens~={used} budget={budget}")
    else:
        target = max_rows or max(len(x) for x in groups.values())
        for role, ratio in ratios.items():
            rows = groups.get(role, [])
            if not rows:
                continue
            take = min(len(rows), max(1, int(target * ratio)))
            merged.extend(rows[:take])
    rng.shuffle(merged)
    return write_jsonl(output, merged)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual-math", type=Path)
    ap.add_argument("--general-image", type=Path)
    ap.add_argument("--text-math", type=Path)
    ap.add_argument("--text-replay", type=Path)
    ap.add_argument("--audio-math", type=Path, help="audio_math stream (data/generated/figure_audio/train.jsonl)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--max-tokens", type=int, help="token budget; takes precedence over --max-rows")
    ap.add_argument("--tokenizer", type=Path, help="tokenizers JSON for exact token estimates")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    tokenizer = None
    if args.tokenizer:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
    paths = {"visual_math": args.visual_math, "general_image": args.general_image,
             "text_math_sft": args.text_math, "text_replay": args.text_replay, "audio_math": args.audio_math}
    print(f"written={build(paths, args.output, args.seed, args.max_rows, args.max_tokens, tokenizer)}")


if __name__ == "__main__":
    main()
