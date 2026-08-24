#!/usr/bin/env bash
set -euo pipefail

# Run this file on the 4×RTX 3090 server from any working directory.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
HF_BIN="${HF_BIN:-hf}"
PROJECT1_ROOT="${PROJECT1_ROOT:-$PROJECT_ROOT/assets/project1}"
PROJECT1_SOURCE="${PROJECT1_SOURCE:-$PROJECT_ROOT/../upload_proj_1}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/server}"
RAW_DIR="$DATA_ROOT/raw/minimind_v"
IMAGE_DIR="$DATA_ROOT/images/general_i2t"
PROCESSED_DIR="$DATA_ROOT/processed"
LIMIT="${GENERAL_I2T_LIMIT:-20000}"
# 12.6M total keeps the full 20,000-row general_i2t stream (5.63M tokens) at
# its 45% quota; smaller budgets silently drop i2t rows in the mixer.
MAX_TOKENS="${SFT_MAX_TOKENS:-12600000}"
VISUAL_MATH="${VISUAL_MATH:-$PROJECT_ROOT/data/generated/figure_math/train.jsonl}"
AUDIO_MATH="${AUDIO_MATH:-$PROJECT_ROOT/data/generated/figure_audio_v5/train.jsonl}"
INCLUDE_AUDIO="${INCLUDE_AUDIO:-auto}"

mkdir -p "$RAW_DIR" "$IMAGE_DIR" "$PROCESSED_DIR"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Make the project self-contained after the first run.  Only the assets that
# are actually consumed by proj_2 are copied; the original project-one tree
# remains an optional source location and is never used by the training code
# once these files exist under PROJECT1_ROOT.
mkdir -p "$PROJECT1_ROOT/tokenizer" "$PROJECT1_ROOT/out/pretrain_02b" "$PROJECT1_ROOT/data/sft"
if [[ ! -f "$PROJECT1_ROOT/tokenizer/best_mm.json" ]]; then
  test -f "$PROJECT1_SOURCE/tokenizer/best_mm.json" || {
    echo "Missing project-one tokenizer. Set PROJECT1_SOURCE to the old project-one directory." >&2
    exit 1
  }
  cp "$PROJECT1_SOURCE/tokenizer/best_mm.json" "$PROJECT1_ROOT/tokenizer/best_mm.json"
fi
if [[ ! -f "$PROJECT1_ROOT/out/pretrain_02b/pretrain_h1024_l24.pth" ]]; then
  test -f "$PROJECT1_SOURCE/out/pretrain_02b/pretrain_h1024_l24.pth" || {
    echo "Missing project-one pretrain checkpoint under PROJECT1_SOURCE." >&2
    exit 1
  }
  cp "$PROJECT1_SOURCE/out/pretrain_02b/pretrain_h1024_l24.pth" "$PROJECT1_ROOT/out/pretrain_02b/pretrain_h1024_l24.pth"
fi
if [[ ! -f "$PROJECT1_ROOT/data/sft/sft_v4_combined.jsonl" ]]; then
  test -f "$PROJECT1_SOURCE/data/sft/sft_v4_combined.jsonl" || {
    echo "Missing project-one math SFT JSONL under PROJECT1_SOURCE." >&2
    exit 1
  }
  cp "$PROJECT1_SOURCE/data/sft/sft_v4_combined.jsonl" "$PROJECT1_ROOT/data/sft/sft_v4_combined.jsonl"
fi
if [[ -f "$PROJECT1_SOURCE/out/agent_sft_v3/sft_h1024_l24.pth" && ! -f "$PROJECT1_ROOT/out/agent_sft_v3/sft_h1024_l24.pth" ]]; then
  mkdir -p "$PROJECT1_ROOT/out/agent_sft_v3"
  cp "$PROJECT1_SOURCE/out/agent_sft_v3/sft_h1024_l24.pth" "$PROJECT1_ROOT/out/agent_sft_v3/sft_h1024_l24.pth"
fi

"$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements.txt" -r "$PROJECT_ROOT/requirements-server.txt"

# MiniMind-V's released sft_i2t parquet is used as the general image bridge.
# The large parquet remains under DATA_ROOT/raw; only LIMIT rows are converted.
"$HF_BIN" download jingyaogong/minimind-v_dataset sft_i2t.parquet \
  --repo-type dataset --local-dir "$RAW_DIR"

"$PYTHON_BIN" -m data.import_i2t_parquet \
  --input "$RAW_DIR/sft_i2t.parquet" \
  --output "$PROCESSED_DIR/general_i2t.jsonl" \
  --image-dir "$IMAGE_DIR" \
  --limit "$LIMIT"

# Project-one math data is now read from the self-contained assets directory.
"$PYTHON_BIN" -m data.build_text_math_sft \
  --input "${PROJECT1_SFT:-$PROJECT1_ROOT/data/sft/sft_v4_combined.jsonl}" \
  --output "$PROCESSED_DIR/text_math_sft.jsonl" \
  --limit "${TEXT_MATH_LIMIT:-100000}"

if [[ ! -f "$VISUAL_MATH" ]]; then
  "$PYTHON_BIN" -m data.gen_figure_math \
    --count "${VISUAL_MATH_COUNT:-10000}" \
    --out-dir "$PROJECT_ROOT/data/generated/figure_math/images" \
    --manifest "$VISUAL_MATH" \
    --seed 10000 --split train
fi

# Text replay: project-one math SFT minus whatever the no-image math stream
# already serves, deduplicated by prompt and uniformly sampled (README §4.3).
TEXT_REPLAY="${TEXT_REPLAY:-$PROCESSED_DIR/text_replay.jsonl}"
"$PYTHON_BIN" -m data.build_text_replay \
  --input "$PROJECT1_ROOT/data/sft/sft_v4_combined.jsonl" \
  --text-math "$PROCESSED_DIR/text_math_sft.jsonl" \
  --output "$TEXT_REPLAY" \
  --limit "${TEXT_REPLAY_LIMIT:-12000}"

MIX_ARGS=(
  --general-image "$PROCESSED_DIR/general_i2t.jsonl"
  --text-math "$PROCESSED_DIR/text_math_sft.jsonl"
  --visual-math "$VISUAL_MATH"
  --text-replay "$TEXT_REPLAY"
  --output "$PROCESSED_DIR/mm_sft.jsonl"
  --max-tokens "$MAX_TOKENS"
  --tokenizer "${PROJECT1_TOKENIZER:-$PROJECT1_ROOT/tokenizer/best_mm.json}"
)
if [[ "$INCLUDE_AUDIO" == "1" ]]; then
  test -f "$AUDIO_MATH" || {
    echo "INCLUDE_AUDIO=1 but AUDIO_MATH does not exist: $AUDIO_MATH" >&2
    exit 1
  }
  MIX_ARGS+=(--audio-math "$AUDIO_MATH")
elif [[ "$INCLUDE_AUDIO" == "auto" && -f "$AUDIO_MATH" ]]; then
  MIX_ARGS+=(--audio-math "$AUDIO_MATH")
else
  echo "AUDIO_STREAM=disabled; set AUDIO_MATH and INCLUDE_AUDIO=1 for the final five-stream mix" >&2
fi
"$PYTHON_BIN" -m data.build_sft_jsonl "${MIX_ARGS[@]}"

"$PYTHON_BIN" -m data.smoke_data --manifest "$PROCESSED_DIR/general_i2t.jsonl"
"$PYTHON_BIN" -m data.smoke_data --manifest "$PROCESSED_DIR/mm_sft.jsonl"
echo "SERVER_DATA_READY: $PROCESSED_DIR/mm_sft.jsonl"
