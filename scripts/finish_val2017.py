"""Finalize the aria2-downloaded COCO val2017 archive: wait, verify, install.

1. wait until the .aria2 control file disappears (aria2 finished)
2. size check (expect 815585330 bytes, the known-good target size)
3. zipfile.testzip() CRC check (no extraction needed)
4. rename to val2017.zip, extract to data/raw/public/val2017/
5. append one record to data/raw/download_records.jsonl
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "data" / "raw" / "public"
SRC = PUBLIC / "val2017_fresh.zip"
TARGET = PUBLIC / "val2017.zip"
RECORDS = ROOT / "data" / "raw" / "download_records.jsonl"
EXPECTED_SIZE = 815585330


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record_exists(url: str) -> bool:
    if not RECORDS.exists():
        return False
    for line in RECORDS.read_text(encoding="utf-8").splitlines():
        if line.strip() and f'"url": "{url}"' in line:
            return True
    return False


def main() -> None:
    url = "http://images.cocodataset.org/zips/val2017.zip"

    if TARGET.exists() and TARGET.stat().st_size == EXPECTED_SIZE and record_exists(url):
        print(f"{TARGET.name} already installed and recorded; nothing to do")
        return

    # 1. wait until aria2 finished (unless the archive is already installed)
    control = PUBLIC / "val2017_fresh.zip.aria2"
    if not (TARGET.exists() and TARGET.stat().st_size == EXPECTED_SIZE):
        deadline = time.time() + 60 * 60
        while control.exists():
            if time.time() > deadline:
                sys.exit(f"timeout: {control} still present after 60 min; run: aria2c -c ...")
            time.sleep(20)

    src = TARGET if TARGET.exists() and TARGET.stat().st_size == EXPECTED_SIZE else SRC
    size = src.stat().st_size
    if size != EXPECTED_SIZE:
        sys.exit(f"unexpected size {size} != {EXPECTED_SIZE}; keep fresh name, resume aria2 with -c")
    print(f"size ok: {size}")

    digest = sha256(src)
    print(f"sha256: {digest}")

    with zipfile.ZipFile(src) as z:
        bad = z.testzip()
    if bad is not None:
        sys.exit(f"CRC failure in {bad}; keep fresh name, resume aria2 with -c")
    print("zip CRC ok")

    if src != TARGET:
        TARGET.unlink(missing_ok=True)
        SRC.rename(TARGET)
        print(f"renamed -> {TARGET.name}")

    extract_dir = PUBLIC / "val2017"
    if not (extract_dir.exists() and len(list(extract_dir.glob("*.jpg"))) == 5000):
        with zipfile.ZipFile(TARGET) as z:
            z.extractall(extract_dir)
        n = len(list(extract_dir.glob("*.jpg")))
        print(f"extracted {n} jpgs -> {extract_dir}")

    if not record_exists(url):
        record = {
            "url": url,
            "file": str(TARGET.resolve()),
            "sha256": digest,
            "license": "COCO Terms of Use",
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "extracted": [str(extract_dir.resolve())],
        }
        with RECORDS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("record appended")
    else:
        print("record already present; skipped")


if __name__ == "__main__":
    main()
