#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
# Single-GPU GRPO (the trainer refuses WORLD_SIZE != 1); defaults tuned for the
# smooth-reward loop: larger groups + temperature 1.0 + top-k 50 diversify the
# rollout, uniform groups are re-rolled from other rows (--resample-attempts),
# cosine LR with warmup, and an intermediate checkpoint every --save-interval.
"$PYTHON_BIN" -m train.train_mm_grpo \
  --data "${DATA:?set DATA to a visual-math JSONL}" \
  --checkpoint "${CHECKPOINT:?set CHECKPOINT to out/mm_sft/mm_sft.pt}" \
  --vision-model "${VISION_MODEL:?set VISION_MODEL to local SigLIP2 checkpoint}" \
  --tokenizer "${TOKENIZER:-assets/project1/tokenizer/best_mm.json}" \
  --save-dir "${SAVE_DIR:-out/mm_grpo}" --steps "${STEPS:-500}" \
  --group-size "${GROUP_SIZE:-8}" --max-new-tokens "${MAX_NEW_TOKENS:-128}" \
  --temperature "${TEMPERATURE:-1.0}" --top-k "${TOP_K:-50}" \
  --lr "${LR:-1e-6}" --kl-beta "${KL_BETA:-0.01}" \
  --warmup-steps "${WARMUP_STEPS:-50}" --save-interval "${SAVE_INTERVAL:-100}" \
  --resample-attempts "${RESAMPLE_ATTEMPTS:-1}" \
  --precision "${PRECISION:-bf16}"
