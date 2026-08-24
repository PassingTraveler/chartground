from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import exact_acc, paired_gain


def read_predictions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="aggregate paired visual math predictions")
    ap.add_argument("--pred-with-image", type=Path, required=True)
    ap.add_argument("--pred-without-image", type=Path, required=True)
    args = ap.parse_args()
    with_image = read_predictions(args.pred_with_image)
    without_image = read_predictions(args.pred_without_image)
    if len(with_image) != len(without_image):
        raise SystemExit(
            f"paired prediction files differ in length: {len(with_image)} vs {len(without_image)}"
        )
    ids_a = [x.get("example_id") for x in with_image]
    ids_b = [x.get("example_id") for x in without_image]
    if ids_a != ids_b:
        mismatch = next((i for i, (a, b) in enumerate(zip(ids_a, ids_b)) if a != b), -1)
        raise SystemExit(f"paired prediction order/id mismatch at row {mismatch}: {ids_a[mismatch]} vs {ids_b[mismatch]}")
    golds = [x["answer"] for x in with_image]
    preds_a = [x["prediction"] for x in with_image]
    preds_b = [x["prediction"] for x in without_image]
    result = paired_gain(preds_a, preds_b, golds)
    result["n"] = len(golds)
    result["with_image_exact"] = exact_acc(preds_a, golds)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
