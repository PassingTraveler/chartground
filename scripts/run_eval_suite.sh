#!/usr/bin/env bash
set -euo pipefail

# 4-GPU sharded visual-math eval: each GPU predicts a disjoint row slice
# (predict_mm --shard i/4), the shards are merged, then the paired
# (with-image / without-image) aggregate reports Δvis with bootstrap CI.
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"

: "${CHECKPOINT:?set CHECKPOINT to the MM-SFT checkpoint}"
: "${VISION_MODEL:?set VISION_MODEL to the SigLIP2 path}"
: "${TOKENIZER:?set TOKENIZER (assets/project1/tokenizer/best_mm.json)}"
: "${EVAL_MANIFEST:?set EVAL_MANIFEST to the visual eval jsonl}"
OUT_DIR="${OUT_DIR:-out/eval}"
N_SHARDS="${N_SHARDS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
# GPU offset so shard i runs on GPU (GPU_BASE + i); default 0 keeps the
# historical behavior of using cuda 0..N-1.
GPU_BASE="${GPU_BASE:-0}"
# Training env carries torch; bare conda base does not.
mkdir -p "$OUT_DIR"

echo "== predict with image (${N_SHARDS} shards) =="
for i in $(seq 0 $((N_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=$((GPU_BASE + i)) "$PYTHON_BIN" -m eval.predict_mm \
    --manifest "$EVAL_MANIFEST" \
    --checkpoint "$CHECKPOINT" \
    --vision-model "$VISION_MODEL" \
    --tokenizer "$TOKENIZER" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --out "$OUT_DIR/pred_with_image.jsonl" \
    --shard "$i/$N_SHARDS" > "$OUT_DIR/pred_with_image.shard$i.log" 2>&1 &
done
wait

echo "== predict without image (same prompt, no image) =="
for i in $(seq 0 $((N_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=$((GPU_BASE + i)) "$PYTHON_BIN" -m eval.predict_mm \
    --manifest "$EVAL_MANIFEST" \
    --checkpoint "$CHECKPOINT" \
    --vision-model "$VISION_MODEL" \
    --tokenizer "$TOKENIZER" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --no-image \
    --out "$OUT_DIR/pred_without_image.jsonl" \
    --shard "$i/$N_SHARDS" > "$OUT_DIR/pred_without_image.shard$i.log" 2>&1 &
done
wait

echo "== merge shards =="
: > "$OUT_DIR/pred_with_image.jsonl"
: > "$OUT_DIR/pred_without_image.jsonl"
for i in $(seq 0 $((N_SHARDS - 1))); do
  cat "$OUT_DIR/pred_with_image.shard${i}of${N_SHARDS}.jsonl" >> "$OUT_DIR/pred_with_image.jsonl"
  cat "$OUT_DIR/pred_without_image.shard${i}of${N_SHARDS}.jsonl" >> "$OUT_DIR/pred_without_image.jsonl"
done
rm -f "$OUT_DIR"/pred_with_image.shard*of*.jsonl "$OUT_DIR"/pred_without_image.shard*of*.jsonl

echo "== aggregate =="
"$PYTHON_BIN" -m eval.eval_mm_math \
  --pred-with-image "$OUT_DIR/pred_with_image.jsonl" \
  --pred-without-image "$OUT_DIR/pred_without_image.jsonl"

if [[ -n "${TEXT_PREDICTIONS:-}" ]]; then
  "$PYTHON_BIN" -m eval.eval_text_retention --predictions "$TEXT_PREDICTIONS"
fi
if [[ -n "${VQA_PREDICTIONS:-}" ]]; then
  "$PYTHON_BIN" -m eval.eval_vqa --predictions "$VQA_PREDICTIONS"
fi
