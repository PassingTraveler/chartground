#!/usr/bin/env bash
set -euo pipefail

# Audio-projector-only alignment: frozen LLM + frozen SigLIP2 + frozen visual
# projector (out/align/projector.pt), trains only the audio projector so the
# LLM can read whisper features. ~10-20 min on one 3090.
#
# Usage: bash scripts/run_align_audio.sh   (env overrides below)
# Run AFTER: data/gen_audio_tts.py (omni-tts env) + data/precompute_audio_features.py
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"

: "${AUDIO_MANIFEST:=data/generated/figure_audio_v5/train.jsonl}"
: "${CHECKPOINT:=assets/project1/out/pretrain_02b/pretrain_h1024_l24.pth}"
: "${VISION_MODEL:=model/siglip2-base-patch16-256}"
: "${TOKENIZER:=assets/project1/tokenizer/best_mm.json}"
: "${VISUAL_PROJECTOR:=out/align/projector.pt}"
: "${WHISPER_MODEL:=model/whisper-small}"
: "${STEPS:=1000}"
: "${LR:=1e-4}"
: "${SAVE:=out/align_audio/audio_projector.pt}"
: "${PRECISION:=bf16}"
: "${GPU:=3}"

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" -m train.train_align_audio \
  --data "$AUDIO_MANIFEST" \
  --checkpoint "$CHECKPOINT" \
  --vision-model "$VISION_MODEL" \
  --tokenizer "$TOKENIZER" \
  --visual-projector "$VISUAL_PROJECTOR" \
  --whisper-model "$WHISPER_MODEL" \
  --steps "$STEPS" \
  --lr "$LR" \
  --save "$SAVE" \
  --precision "$PRECISION"
