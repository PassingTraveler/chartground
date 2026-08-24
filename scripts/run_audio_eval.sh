#!/usr/bin/env bash
set -euo pipefail

# 4-GPU sharded audio-math eval: the --audio arm feeds the whisper-feature
# question span (no written question, chart visible) and the default arm
# feeds the written question + chart; eval_mm_math's paired gain reports
# Δaudio = acc(audio) - acc(text) with bootstrap CI over the 500 eval rows.
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"

: "${CHECKPOINT:?set CHECKPOINT to the MM-SFT checkpoint (must contain audio_projector)}"
: "${VISION_MODEL:?set VISION_MODEL to the SigLIP2 path}"
: "${TOKENIZER:?set TOKENIZER (assets/project1/tokenizer/best_mm.json)}"
: "${AUDIO_EVAL_MANIFEST:?set AUDIO_EVAL_MANIFEST to the audio eval jsonl (data/generated/figure_audio/eval.jsonl)}"
OUT_DIR="${OUT_DIR:-out/eval_audio}"
N_SHARDS="${N_SHARDS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
# Audio eval runs on top of the training/eval GPUs; shard i -> GPU (GPU_BASE + i).
# All stages use cuda 0-3 per user convention; override with GPU_BASE if needed.
GPU_BASE="${GPU_BASE:-0}"
# Training env carries torch; bare conda base does not.
mkdir -p "$OUT_DIR"

echo "== predict with audio question span (${N_SHARDS} shards) =="
for i in $(seq 0 $((N_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=$((GPU_BASE + i)) "$PYTHON_BIN" -m eval.predict_mm \
    --manifest "$AUDIO_EVAL_MANIFEST" \
    --checkpoint "$CHECKPOINT" \
    --vision-model "$VISION_MODEL" \
    --tokenizer "$TOKENIZER" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --audio \
    --out "$OUT_DIR/pred_with_audio.jsonl" \
    --shard "$i/$N_SHARDS" > "$OUT_DIR/pred_with_audio.shard$i.log" 2>&1 &
done
wait

echo "== predict with written question (control) =="
for i in $(seq 0 $((N_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=$((GPU_BASE + i)) "$PYTHON_BIN" -m eval.predict_mm \
    --manifest "$AUDIO_EVAL_MANIFEST" \
    --checkpoint "$CHECKPOINT" \
    --vision-model "$VISION_MODEL" \
    --tokenizer "$TOKENIZER" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --out "$OUT_DIR/pred_with_text.jsonl" \
    --shard "$i/$N_SHARDS" > "$OUT_DIR/pred_with_text.shard$i.log" 2>&1 &
done
wait

echo "== merge shards =="
: > "$OUT_DIR/pred_with_audio.jsonl"
: > "$OUT_DIR/pred_with_text.jsonl"
for i in $(seq 0 $((N_SHARDS - 1))); do
  cat "$OUT_DIR/pred_with_audio.shard${i}of${N_SHARDS}.jsonl" >> "$OUT_DIR/pred_with_audio.jsonl"
  cat "$OUT_DIR/pred_with_text.shard${i}of${N_SHARDS}.jsonl" >> "$OUT_DIR/pred_with_text.jsonl"
done
rm -f "$OUT_DIR"/pred_with_audio.shard*of*.jsonl "$OUT_DIR"/pred_with_text.shard*of*.jsonl

echo "== aggregate (Δaudio) =="
"$PYTHON_BIN" -m eval.eval_mm_math \
  --pred-with-image "$OUT_DIR/pred_with_audio.jsonl" \
  --pred-without-image "$OUT_DIR/pred_with_text.jsonl"
