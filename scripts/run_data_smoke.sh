#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m data.gen_figure_math --count 12 --out-dir data/generated/smoke/images --manifest data/generated/smoke/train.jsonl --seed 10000 --split train
"$PYTHON_BIN" -m data.smoke_data --manifest data/generated/smoke/train.jsonl --expected-images 12
"$PYTHON_BIN" -m data.gen_figure_math --count 8 --out-dir data/generated/smoke/eval_images --manifest data/generated/smoke/eval.jsonl --seed 20260814 --split eval
"$PYTHON_BIN" -m data.smoke_data --manifest data/generated/smoke/eval.jsonl --expected-images 8
"$PYTHON_BIN" -m data.clean_images --input-dir data/generated/smoke/images --output-dir data/generated/smoke/clean_images --report data/generated/smoke/clean_report.json
echo DATA_SMOKE_OK
