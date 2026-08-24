"""Build the text-replay stream from project-one's math SFT corpus.

The replay stream protects the frozen-text abilities of `pretrain_02b` from
catastrophic forgetting during multimodal SFT (README §4.3).  It is NOT the
same as the no-image math SFT stream: the no-image math stream *establishes*
the math instruction format, the replay stream *preserves* it.

Rows are taken from project-one `sft_v4_combined.jsonl` (conversations format)
and converted to the shared Example schema.  Rows whose user prompt already
appears in the no-image math SFT stream are dropped so the two text streams
do not double-count the same problem, and rows are deduplicated within the
source by prompt.  A uniform sample of the remainder is written out; the
token-budgeted mixer (`data.build_sft_jsonl`) shuffles and fills its 15% quota
from this file, so the sample only needs to be comfortably larger than the
quota.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from .schema import Example, read_jsonl, write_jsonl


def _user_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="project-one sft jsonl (conversations format)")
    ap.add_argument("--text-math", type=Path, required=True, help="no-image math SFT jsonl whose prompts must not be duplicated")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=12000, help="uniform sample size after dedup")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Prompts already served by the no-image math stream.
    text_math_prompts = {_user_hash(row["prompt"]) for row in read_jsonl(args.text_math)}
    print(f"[text_replay] text_math prompts={len(text_math_prompts)}")

    seen: set[str] = set()
    kept: list[Example] = []
    with open(args.input, encoding="utf-8") as f:
        for row_no, line in enumerate(f):
            row = json.loads(line)
            convs = row.get("conversations") or []
            if len(convs) < 2:
                continue
            prompt = convs[0].get("content", "")
            answer = convs[-1].get("content", "")
            h = _user_hash(prompt)
            if not prompt or h in seen or h in text_math_prompts:
                continue
            seen.add(h)
            kept.append(Example(
                example_id=f"text_replay_{row_no:07d}",
                modality="text",
                prompt=prompt,
                answer=answer,
                split="train",
                source="project1_sft_v4_combined",
                task="text_replay",
                visual_required=False,
                metadata={"mix_role": "text_replay", "source_row": row_no},
            ))
    print(f"[text_replay] deduped={len(kept)}")
    rng = random.Random(args.seed)
    rng.shuffle(kept)
    sampled = kept[: args.limit]
    written = write_jsonl(args.output, sampled)
    print(f"[text_replay] written={written} -> {args.output}")


if __name__ == "__main__":
    main()
