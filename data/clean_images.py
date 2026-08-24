"""Image validation, exact deduplication and a lightweight perceptual hash."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image, ImageStat


def average_hash(path: Path, size: int = 16) -> int:
    with Image.open(path).convert("L") as im:
        im = im.resize((size, size))
        getter = getattr(im, "get_flattened_data", im.getdata)
        pixels = list(getter())
    mean = sum(pixels) / len(pixels)
    value = 0
    for p in pixels:
        value = (value << 1) | int(p >= mean)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def clean(input_dir: Path, output_dir: Path, report: Path, min_size: int = 64, max_hash_distance: int = 4) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_sha: set[str] = set()
    seen_hashes: list[int] = []
    rows, stats = [], {"seen": 0, "kept": 0, "bad": 0, "exact_duplicate": 0, "near_duplicate": 0}
    for src in sorted(input_dir.rglob("*")):
        if src.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        stats["seen"] += 1
        try:
            raw = src.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            with Image.open(src) as im:
                im.verify()
            with Image.open(src).convert("RGB") as im:
                if min(im.size) < min_size:
                    stats["bad"] += 1
                    continue
                stat = ImageStat.Stat(im.resize((32, 32)))
                # Only reject genuinely blank/placeholder images (uniform color:
                # every channel has near-zero spread). The old channel-difference
                # test wrongly dropped valid grayscale photos (R≈G≈B always).
                if max(stat.stddev) < 4.0:
                    stats["bad"] += 1
                    continue
                ah = average_hash(src)
                if sha in seen_sha:
                    stats["exact_duplicate"] += 1
                    continue
                if any(hamming(ah, old) <= max_hash_distance for old in seen_hashes):
                    stats["near_duplicate"] += 1
                    continue
                dst = output_dir / f"{sha[:16]}.jpg"
                im.save(dst, format="JPEG", quality=95)
            seen_sha.add(sha)
            seen_hashes.append(ah)
            stats["kept"] += 1
            rows.append({
                "source": Path(os.path.relpath(src, report.parent)).as_posix(),
                "image": Path(os.path.relpath(dst, report.parent)).as_posix(),
                "sha256": sha,
                "ahash": ah,
            })
        except Exception as e:
            stats["bad"] += 1
            rows.append({"source": Path(os.path.relpath(src, report.parent)).as_posix(), "error": repr(e)})
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"stats": stats, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--min-size", type=int, default=64)
    ap.add_argument("--max-hash-distance", type=int, default=4)
    args = ap.parse_args()
    print(json.dumps(clean(args.input_dir, args.output_dir, args.report, args.min_size, args.max_hash_distance), ensure_ascii=False))


if __name__ == "__main__":
    main()
