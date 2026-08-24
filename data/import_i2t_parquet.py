"""Convert an I2T parquet shard into this project's image-path JSONL schema.

Designed for server-side preparation of MiniMind-V-style data. The parquet
stays under DATA_ROOT/raw; only the selected images and a JSONL manifest are
materialized under DATA_ROOT/processed and DATA_ROOT/images.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from .schema import write_jsonl


def _as_bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, dict):
        payload = value.get("bytes") or value.get("data")
        return bytes(payload) if payload is not None else None
    return None


def _conversation(value: Any) -> tuple[str, str] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, list):
        return None
    user, assistant = "", ""
    for message in value:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower()
        content = message.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content)
        content = str(content).replace("<image>", "").replace("<|image_pad|>", "").strip()
        if role in {"user", "human"} and not user and content:
            user = content
        if role in {"assistant", "gpt", "bot"} and content:
            assistant = content
    return (user, assistant) if user and assistant else None


def convert(input_path: Path, output_path: Path, image_dir: Path, limit: int = 20000,
            batch_size: int = 256, min_size: int = 64) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("server conversion requires pyarrow; install requirements-server.txt") from exc
    parquet = pq.ParquetFile(input_path)
    columns = set(parquet.schema_arrow.names)
    required = {"conversations", "image_bytes"}
    missing = required - columns
    if missing:
        raise ValueError(f"parquet is missing columns: {sorted(missing)}")
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    scanned = 0
    for batch in parquet.iter_batches(columns=["conversations", "image_bytes"], batch_size=batch_size):
        for item in batch.to_pylist():
            scanned += 1
            parsed = _conversation(item.get("conversations"))
            raw = _as_bytes(item.get("image_bytes"))
            if parsed is None or raw is None:
                continue
            digest = hashlib.sha256(raw).hexdigest()
            if digest in seen:
                continue
            try:
                with Image.open(io.BytesIO(raw)).convert("RGB") as image:
                    if min(image.size) < min_size:
                        continue
                    stat = ImageStat.Stat(image.resize((32, 32)))
                    # MiniMind-V includes black placeholders for text-only rows.
                    # A placeholder is uniform color; the old channel-difference
                    # test also dropped valid grayscale photos (R≈G≈B always).
                    if max(stat.stddev) < 4.0:
                        continue
                    image_path = image_dir / f"{digest[:20]}.jpg"
                    if not image_path.exists():
                        image.save(image_path, format="JPEG", quality=95)
            except Exception:
                continue
            seen.add(digest)
            prompt, answer = parsed
            rows.append({
                "example_id": f"minimind_v_i2t_{len(rows):07d}",
                "modality": "image",
                "image": str(image_path.resolve()),
                "prompt": prompt,
                "answer": answer,
                "split": "train",
                "source": "minimind_v_sft_i2t",
                "task": "general_image",
                "visual_required": True,
                "metadata": {"mix_role": "general_image", "source_row": scanned},
            })
            if len(rows) >= limit:
                return write_jsonl(output_path, rows)
    return write_jsonl(output_path, rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    print(f"written={convert(args.input, args.output, args.image_dir, args.limit, args.batch_size)}")


if __name__ == "__main__":
    main()
