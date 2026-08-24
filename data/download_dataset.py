"""Resumable, manifest-first dataset downloader.

Usage examples:
  python data/download_dataset.py --url URL --out-dir data/raw/public --license "CC-BY-4.0"
  python data/download_dataset.py --manifest data/raw/manifest.jsonl --out-dir data/raw/public

The script deliberately does not silently scrape arbitrary websites. Every
download is recorded with URL, license, SHA256 and timestamp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, out_dir: Path, license_name: str, expected_sha256: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = url.split("?")[0].rstrip("/").split("/")[-1] or "download.bin"
    target = out_dir / name
    part = target.with_suffix(target.suffix + ".part")
    headers = {"User-Agent": "minimind-proj2-data-pipeline/0.1"}
    request = urllib.request.Request(url, headers=headers)
    mode = "ab" if part.exists() else "wb"
    start = part.stat().st_size if part.exists() else 0
    if start:
        request.add_header("Range", f"bytes={start}-")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, part.open(mode) as f:
            shutil.copyfileobj(response, f, length=1024 * 1024)
    except Exception:
        if not part.exists():
            raise
        raise RuntimeError(f"download interrupted; rerun to resume: {part}")
    digest = sha256(part)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"sha256 mismatch for {url}: {digest} != {expected_sha256}")
    part.replace(target)
    extracted = []
    if target.suffix.lower() == ".zip":
        extract_dir = out_dir / target.stem
        with zipfile.ZipFile(target) as z:
            z.extractall(extract_dir)
        extracted = [str(extract_dir.resolve())]
    elif target.name.endswith((".tar.gz", ".tgz", ".tar")):
        extract_dir = out_dir / target.name.split(".tar")[0]
        extract_dir.mkdir(exist_ok=True)
        with tarfile.open(target) as t:
            t.extractall(extract_dir)
        extracted = [str(extract_dir.resolve())]
    return {"url": url, "file": str(target.resolve()), "sha256": digest, "license": license_name, "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "extracted": extracted}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", default=[])
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("data/raw/public"))
    ap.add_argument("--license", default="UNSPECIFIED")
    ap.add_argument("--expected-sha256")
    ap.add_argument("--records", type=Path, default=Path("data/raw/download_records.jsonl"))
    args = ap.parse_args()
    items = [{"url": u, "license": args.license} for u in args.url]
    if args.manifest:
        for line in args.manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append(json.loads(line))
    if not items:
        raise SystemExit("provide --url or --manifest")
    args.records.parent.mkdir(parents=True, exist_ok=True)
    with args.records.open("a", encoding="utf-8") as f:
        for item in items:
            record = download(item["url"], args.out_dir, item.get("license", args.license), item.get("sha256", args.expected_sha256))
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()

