# Reproduction Guide

This project is released as a code-and-experiment artifact. Large datasets,
encoder weights, checkpoints, generated images, audio files and feature caches
are intentionally excluded from the Git repository. Place them under the paths
shown below, or provide equivalent paths through environment variables.

## 1. Environment

Use Linux with 4×RTX 3090, CUDA-enabled PyTorch, and Python 3.10 or newer.
Install the project dependencies:

```bash
python -m pip install -r requirements.txt -r requirements-server.txt
```

The training entry points use `python` and `torchrun` from `PATH`. Override
them when necessary:

```bash
export PYTHON_BIN=/opt/conda/envs/vlm/bin/python
export TORCHRUN_BIN=/opt/conda/envs/vlm/bin/torchrun
```

## 2. Data preparation

The server helper downloads the general image-text bridge and builds the text
and visual streams. The project-one assets are expected at `PROJECT1_SOURCE`.

```bash
PROJECT1_SOURCE=/path/to/project1 \
SFT_MAX_TOKENS=12600000 \
GENERAL_I2T_LIMIT=20000 \
bash scripts/prepare_server_data.sh
```

To reproduce the final five-stream mixture, provide the generated audio
manifest and enable the audio stream explicitly:

```bash
AUDIO_MATH=/path/to/figure_audio_v5/train.jsonl \
INCLUDE_AUDIO=1 \
PROJECT1_SOURCE=/path/to/project1 \
bash scripts/prepare_server_data.sh
```

The resulting manifest is written to `data/server/processed/mm_sft.jsonl`.
Always run the data smoke check before training.

## 3. Four-GPU SFT

```bash
DATA=data/processed/mm_sft_v6.jsonl \
BASE_MODEL=assets/project1/out/pretrain_02b/pretrain_h1024_l24.pth \
VISION_MODEL=model/siglip2-base-patch16-256 \
TOKENIZER=assets/project1/tokenizer/best_mm.json \
SAVE_DIR=out/mm_sft_v6 \
PRECISION=bf16 \
bash scripts/run_mm_sft.sh
```

Use `mm_sft_v7.jsonl` and a different `SAVE_DIR` for the curriculum-SFT
comparison. Audio training additionally requires `AUDIO_PROJECTOR` and
precomputed Whisper features referenced by the manifest.

## 4. Evaluation

Set the checkpoint, tokenizer, vision encoder and evaluation manifest before
running the four-shard evaluation harness:

```bash
CHECKPOINT=out/mm_sft_v6/mm_sft.pt \
VISION_MODEL=model/siglip2-base-patch16-256 \
TOKENIZER=assets/project1/tokenizer/best_mm.json \
EVAL_MANIFEST=data/generated/figure_math_v5/eval.jsonl \
bash scripts/run_eval_suite.sh
```

The main metric is the paired visual gain:

```text
Δvis = Acc(with image) - Acc(without image on the paired text version)
```

## 5. Scope and limitations

The model is a compact research system for controlled visual-math and speech-
question experiments. The speech response demo uses an external TTS layer;
it is not native audio-token generation. The synthetic visual benchmark is
designed for verifiable ablations and should not be presented as a general
vision benchmark.
