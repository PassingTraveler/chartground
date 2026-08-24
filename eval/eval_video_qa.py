from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import exact_acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    args = ap.parse_args()
    rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    print(json.dumps({"n": len(rows), "video_qa_acc": exact_acc([x["prediction"] for x in rows], [x["answer"] for x in rows])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

