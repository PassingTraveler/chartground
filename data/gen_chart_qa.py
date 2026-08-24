"""Generate a small deterministic chart-QA slice with executable answers."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    from .schema import Example, write_jsonl
except ImportError:  # direct script execution
    from schema import Example, write_jsonl


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def generate(count: int, out_dir: Path, manifest: Path, seed: int, split: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    labels = ["A", "B", "C", "D"]
    for i in range(count):
        # Keep the max unique so "which category is largest" has a single
        # unambiguous answer (a tie would be ambiguous visually).
        while True:
            values = [rng.randint(15, 95) for _ in labels]
            if len(set(values)) == len(values):
                break
        image = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(image)
        draw.line((30, 20, 30, 220), fill="black", width=2)
        draw.line((30, 220, 240, 220), fill="black", width=2)
        for tick in range(0, 101, 20):
            y = 220 - int(tick * 1.8)
            draw.line((26, y, 30, y), fill="black")
            draw.text((2, y - 5), str(tick), fill="black", font=_font(8))
        for j, (label, value) in enumerate(zip(labels, values)):
            x = 48 + j * 46
            y = 220 - int(value * 1.8)
            draw.rectangle((x, y, x + 27, 220), fill=(68, 114, 196), outline="black")
            draw.text((x + 9, 224), label, fill="black", font=_font(10))
        example_id = f"chart_qa_{split}_{seed}_{i:06d}"
        image_path = out_dir / f"{example_id}.png"
        image.save(image_path, format="PNG", optimize=True)
        best = labels[values.index(max(values))]
        rows.append(Example(
            example_id=example_id,
            modality="image",
            image=str(image_path.resolve()),
            prompt="柱状图中哪个类别的数值最大？只输出类别字母。",
            answer=best,
            split=split,
            source="programmatic_pillow",
            task="chart_qa",
            visual_required=True,
            metadata={"labels": labels, "values": values, "seed": seed},
        ))
    return write_jsonl(manifest, rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--out-dir", type=Path, default=Path("data/generated/chart_qa/images"))
    ap.add_argument("--manifest", type=Path, default=Path("data/generated/chart_qa/train.jsonl"))
    ap.add_argument("--seed", type=int, default=11000)
    ap.add_argument("--split", choices=["train", "eval", "ood"], default="train")
    args = ap.parse_args()
    print(f"generated={generate(args.count, args.out_dir, args.manifest, args.seed, args.split)} manifest={args.manifest}")


if __name__ == "__main__":
    main()
