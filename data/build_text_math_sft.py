"""Build the no-image math instruction slice separately from text replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import write_jsonl


def build(input_path: Path, output_path: Path, limit: int | None = None) -> int:
    rows = []
    skipped = 0
    with input_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            conv = obj.get("conversations", [])
            if not isinstance(conv, list):
                skipped += 1
                continue
            # Select by role, not by position: a leading system message or an
            # extra user turn must not become the instruction.
            prompt = next((m.get("content", "").strip() for m in conv
                           if isinstance(m, dict) and m.get("role") in {"user", "human"} and m.get("content")), "")
            answer = next((m.get("content", "").strip() for m in reversed(conv)
                           if isinstance(m, dict) and m.get("role") in {"assistant", "gpt", "bot"} and m.get("content")), "")
            if not prompt or not answer:
                skipped += 1
                continue
            rows.append({
                "example_id": f"text_math_{i:07d}",
                "modality": "text",
                "prompt": prompt,
                "answer": answer,
                "split": "train",
                "source": "project1_math_sft",
                "task": "text_math_instruction",
                "visual_required": False,
                "metadata": {"mix_role": "text_math_sft"},
            })
    if skipped:
        import sys
        print(f"skipped {skipped} unparsable/skipped rows", file=sys.stderr)
    return write_jsonl(output_path, rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    print(f"written={build(args.input, args.output, args.limit)}")


if __name__ == "__main__":
    main()

