from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import exact_acc


def read_predictions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    args = ap.parse_args()
    rows = read_predictions(args.predictions)
    acc = exact_acc([r["prediction"] for r in rows], [r["answer"] for r in rows])
    print(json.dumps({"n": len(rows), "text_acc": acc}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

