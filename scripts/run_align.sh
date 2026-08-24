#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${ALIGN_GPU:-0}" "$PYTHON_BIN" -m train.train_align \
  --data "${DATA:-data/processed/mm_sft_v6.jsonl}" \
  --checkpoint "${BASE_MODEL:?set BASE_MODEL to project-one pretrain checkpoint}" \
  --vision-model "${VISION_MODEL:?set VISION_MODEL to local SigLIP2 checkpoint}" \
  --tokenizer "${TOKENIZER:-assets/project1/tokenizer/best_mm.json}" \
  --save "${ALIGN_SAVE:-out/align/projector.pt}" --steps "${STEPS:-1000}" \
  --precision "${PRECISION:-bf16}"
