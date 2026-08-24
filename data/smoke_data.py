"""Dependency-light data smoke test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .schema import read_jsonl


def check(manifest: Path, expected_images: int | None = None) -> dict:
    rows = read_jsonl(manifest)
    if not rows:
        raise AssertionError("manifest is empty")
    seen = set()
    image_count = 0
    for row in rows:
        for key in ("example_id", "prompt", "answer", "modality"):
            if not row.get(key):
                raise AssertionError(f"missing {key}: {row}")
        if row["example_id"] in seen:
            raise AssertionError(f"duplicate example_id: {row['example_id']}")
        seen.add(row["example_id"])
        if row.get("image"):
            image_count += 1
            image_path = Path(row["image"])
            if not image_path.is_absolute():
                image_path = manifest.parent / image_path
            with Image.open(image_path) as im:
                if min(im.size) < 64:
                    raise AssertionError(f"image too small: {image_path}")
                # Image.open is lazy; verify() catches truncated/corrupt files.
                im.verify()
    if expected_images is not None and image_count != expected_images:
        raise AssertionError(f"expected {expected_images} images, got {image_count}")
    return {"rows": len(rows), "images": image_count, "unique_ids": len(seen)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--expected-images", type=int)
    args = ap.parse_args()
    print(json.dumps(check(args.manifest, args.expected_images), ensure_ascii=False))


if __name__ == "__main__":
    main()
