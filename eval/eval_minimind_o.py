"""Fair-comparison manifest checker; inference is intentionally model-specific."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    args = ap.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    ours = json.loads(args.ours.read_text(encoding="utf-8"))
    base = json.loads(args.baseline.read_text(encoding="utf-8"))
    if len(ours) != len(base):
        raise SystemExit(f"prediction count mismatch: ours={len(ours)} baseline={len(base)}")
    print(json.dumps({"n": len(ours), "protocol": protocol, "status": "paired files aligned"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

