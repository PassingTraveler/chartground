"""Build paired IID/OOD evaluation manifests and text-retention metadata."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from .schema import read_jsonl, write_jsonl


def build_visual_eval(source: Path, output: Path, limit: int = 500, seed: int = 20260814) -> int:
    rows = read_jsonl(source)
    rng = random.Random(seed)
    rng.shuffle(rows)
    chosen = []
    for row in rows[:limit]:
        row = dict(row)
        row["split"] = "eval_iid"
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            row["metadata"] = metadata
        metadata["eval_protocol"] = "paired_visual_gain"
        # The generated prompts carry no numeric values (numbers live in the
        # image only), so the same prompt without image input is a valid
        # no-image pair for the Δvis protocol: Acc(with image) - Acc(no image).
        metadata["no_image_pair"] = True
        # Resolve assets relative to the source manifest so write_jsonl can
        # re-relativize them against the output manifest directory; consumers
        # resolve relative paths against the manifest directory.
        for field in ("image", "video"):
            if row.get(field):
                asset = Path(row[field])
                if not asset.is_absolute():
                    row[field] = str((source.parent / asset).resolve())
        chosen.append(row)
    return write_jsonl(output, chosen)


def build_text_eval(source: Path, output: Path, limit: int = 500, seed: int = 42) -> int:
    rows = read_jsonl(source)
    rng = random.Random(seed)
    rng.shuffle(rows)
    chosen = []
    for row in rows[:limit]:
        obj = dict(row)
        obj["split"] = "text_retention"
        metadata = obj.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            obj["metadata"] = metadata
        metadata["eval_protocol"] = "text_retention"
        chosen.append(obj)
    return write_jsonl(output, chosen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual-source", type=Path, required=True)
    ap.add_argument("--visual-output", type=Path, required=True)
    ap.add_argument("--text-source", type=Path)
    ap.add_argument("--text-output", type=Path)
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()
    print(f"visual_eval={build_visual_eval(args.visual_source, args.visual_output, args.limit)}")
    if args.text_source and args.text_output:
        print(f"text_eval={build_text_eval(args.text_source, args.text_output, args.limit)}")


if __name__ == "__main__":
    main()
