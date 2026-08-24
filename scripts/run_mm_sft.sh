#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
PROJECTOR_ARGS=()
if [[ -n "${PROJECTOR:-}" ]]; then
  PROJECTOR_ARGS+=(--projector "$PROJECTOR")
fi
AUDIO_PROJECTOR_ARGS=()
if [[ -n "${AUDIO_PROJECTOR:-}" ]]; then
  AUDIO_PROJECTOR_ARGS+=(--audio-projector "$AUDIO_PROJECTOR")
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" "$TORCHRUN_BIN" --nproc_per_node=4 -m train.train_mm_sft \
  --data "${DATA:-data/processed/mm_sft_v6.jsonl}" \
  --checkpoint "${BASE_MODEL:?set BASE_MODEL to project-one pretrain checkpoint}" \
  --vision-model "${VISION_MODEL:?set VISION_MODEL to local SigLIP2 checkpoint}" \
  --tokenizer "${TOKENIZER:-assets/project1/tokenizer/best_mm.json}" \
  --save-dir "${SAVE_DIR:-out/mm_sft}" --steps "${STEPS:-3000}" \
  --batch-size "${BATCH_SIZE:-1}" --grad-accum "${GRAD_ACCUM:-8}" \
  --precision "${PRECISION:-bf16}" "${PROJECTOR_ARGS[@]}" "${AUDIO_PROJECTOR_ARGS[@]}"
