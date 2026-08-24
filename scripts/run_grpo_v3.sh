#!/usr/bin/env bash
set -euo pipefail

# GRPO v3: audio+visual mixed RL on top of the 5-stream SFT checkpoint.
# Audio rows roll out on the whisper-feature span (chart still visible), so
# the audio projector is trained by the same group-relative signal as the
# visual projector — no separate reward model. Single GPU by design; the user
# convention pins this to cuda:3.
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" "$PYTHON_BIN" -m train.train_mm_grpo \
  --data "${DATA:-data/processed/visual_math_v5_train.jsonl}" \
  --checkpoint "${CHECKPOINT:-out/mm_sft_v6/mm_sft.pt}" \
  --vision-model "${VISION_MODEL:-model/siglip2-base-patch16-256}" \
  --tokenizer "${TOKENIZER:-assets/project1/tokenizer/best_mm.json}" \
  --save-dir "${SAVE_DIR:-out/mm_grpo_v3}" --steps "${STEPS:-500}" \
  --group-size "${GROUP_SIZE:-8}" --max-new-tokens "${MAX_NEW_TOKENS:-128}" \
  --temperature "${TEMPERATURE:-1.0}" --top-k "${TOP_K:-50}" \
  --lr "${LR:-1e-6}" --kl-beta "${KL_BETA:-0.01}" \
  --warmup-steps "${WARMUP_STEPS:-50}" --save-interval "${SAVE_INTERVAL:-100}" \
  --resample-attempts "${RESAMPLE_ATTEMPTS:-5}" \
  --precision "${PRECISION:-bf16}"
