"""Synthesize Chinese audio math questions with ChatTTS.

Reads figure_math manifests (train + eval), converts math symbols to spoken
Chinese, synthesizes each question to 16k mono wav, and writes an audio
manifest that keeps the ORIGINAL image field: bar/line questions put all
numbers in the figure, so the audio arm must still see the chart. The audio
span replaces the question text (MMTemplate.prompt_audio omits it).

Run ONLY inside the `omni-tts` conda env (chattts pins transformers==4.53.2,
which must never touch the omni training env).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

# Both ChatTTS and the training env export `data`; import after heavy modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.schema import Example, read_jsonl, write_jsonl  # noqa: E402


_CN_DIGITS = "零一二三四五六七八九"


def _num_to_cn(m: re.Match) -> str:
    """0-99 -> 中文（题目里都是小整数）；大于 99 保持原样。"""
    n = int(m.group(0))
    if n < 0 or n > 99:
        return m.group(0)
    if n < 10:
        return _CN_DIGITS[n]
    if n < 20:
        return "十" + (_CN_DIGITS[n - 10] if n % 10 else "")
    tens, ones = divmod(n, 10)
    return _CN_DIGITS[tens] + "十" + (_CN_DIGITS[ones] if ones else "")


def normalize_for_tts(text: str) -> str:
    """Math symbols -> spoken words; numerals -> 中文; full-width QM dropped.

    ChatTTS's refine model rejects arabic digits ("found invalid characters:
    {'4'}"), so numerals are converted here instead of trusting the built-in
    normalization.
    """
    text = text.replace("×", "乘以").replace("÷", "除以")
    text = text.replace("+", "加").replace("−", "减").replace("-", "减")
    text = text.replace("=", "等于").replace("％", "百分之").replace("%", "百分之")
    # ChatTTS rejects the full-width question mark (U+FF1F) outright
    # ("found invalid characters: {'？'}"); 句号 reads fine for a question.
    text = text.replace("？", "。")
    text = re.sub(r"\d+", _num_to_cn, text)
    return text


def synthesize(chat, text: str, speed: float, pitch_st: float, device: str) -> np.ndarray | None:
    """One 24k waveform (ChatTTS output) or None on failure; one retry.

    Natural Mandarin speech is ~4-5 chars/s; ChatTTS's default ([speed_5])
    reads ~5.3 chars/s. speed=0.65 -> [speed_3] = 5.0 chars/s (normal
    speaking pace), and pitch_st=-2 drops the fundamental by 2 semitones
    (torchaudio phase vocoder) for a normal speaking register.
    Note: early "X chars/s" numbers measured on the buggy wav[::3] output
    were 2x the true rate — the synth itself was never that fast.
    """
    # Newer ChatTTS dropped the `speed` kwarg; the prompt marker [speed_N] is
    # the only control (N=5 is the default, so speed=1.2 -> [speed_6]).
    import ChatTTS
    infer_params = ChatTTS.Chat.InferCodeParams(prompt=f"[speed_{min(9, max(1, int(round(speed * 5))))}]")
    for _ in range(2):
        try:
            # skip_refine_text: the refine model's vocab lacks digits and
            # raises "found invalid characters"; skipping it only drops
            # interjection/breath marks, content is unaffected.
            wavs = chat.infer([text], use_decoder=True, stream=False,
                              skip_refine_text=True, params_infer_code=infer_params)
            wav = np.asarray(wavs[0])
            if wav.ndim == 2:  # [T, ch] -> mono
                wav = wav.mean(axis=1)
            if pitch_st:
                import torch
                import torchaudio.functional as AF
                t = torch.from_numpy(wav.copy()).float()
                wav = AF.pitch_shift(t.unsqueeze(0), 24000, n_steps=float(pitch_st)).squeeze(0).numpy()
            if wav.size > 0 and 0.5 < wav.size / 24000 < 30.0:
                return wav.astype(np.float32)
        except Exception:
            pass
    return None


def merge_shards(out_dir: Path) -> None:
    """Concatenate train.shard{i}.jsonl / eval.shard{i}.jsonl into merged manifests."""
    for split in ("train", "eval"):
        parts = sorted((out_dir).glob(f"{split}.shard*.jsonl"))
        if not parts:
            print(f"[merge] no {split} shards found")
            continue
        rows = [r for p in parts for r in read_jsonl(p)]
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        print(f"[merge] {split}: {len(parts)} shards, {len(rows)} rows -> {out_dir / (split + '.jsonl')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-input", type=Path, required=True, help="figure_math train.jsonl")
    ap.add_argument("--eval-input", type=Path, required=True, help="figure_math eval.jsonl")
    ap.add_argument("--limit-train", type=int, default=3000)
    ap.add_argument("--out-dir", type=Path, default=Path("data/generated/figure_audio"))
    ap.add_argument("--speed", type=float, default=0.65,
                    help="ChatTTS speed factor; 0.65 = [speed_3] 5.0 chars/s (normal speech 4-5). "
                         "Measured on the 24k synth output: [speed_5]=5.3, [speed_7]=5.8, [speed_9]=6.7 chars/s")
    ap.add_argument("--pitch-semitones", type=float, default=-2.0,
                    help="post-hoc pitch shift in semitones (negative = lower voice); 0 disables")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", help="i/M: process only rows i, i+M, ... (parallel TTS on multiple GPUs; "
                                    "each shard writes train.shard{i}.jsonl, merge them afterwards)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--merge-shards", action="store_true",
                    help="merge train/eval.shard{i}.jsonl into train/eval.jsonl, then exit")
    args = ap.parse_args()
    if args.merge_shards:
        merge_shards(args.out_dir)
        return
    shard_i, shard_m = (0, 1)
    if args.shard:
        shard_i, shard_m = (int(x) for x in args.shard.split("/"))

    import torch
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    import ChatTTS
    chat = ChatTTS.Chat()
    chat.load(compile=False, source="huggingface", device=args.device)
    if chat.device == "cpu":
        raise SystemExit("ChatTTS failed to load on CUDA; check torch build")

    wav_dir = args.out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = ([(r, args.train_input.parent.name) for r in read_jsonl(args.train_input)[: args.limit_train]]
            + [(r, args.eval_input.parent.name) for r in read_jsonl(args.eval_input)])
    rows = rows[shard_i::shard_m]
    existing = set()
    if args.skip_existing:
        # Merged manifests plus every shard manifest already on disk.
        for path in [args.out_dir / "train.jsonl", args.out_dir / "eval.jsonl"]:
            if path.exists():
                existing |= {r["example_id"] for r in read_jsonl(path)}
        for path in sorted((args.out_dir).glob("train.shard*.jsonl")) + sorted((args.out_dir).glob("eval.shard*.jsonl")):
            existing |= {r["example_id"] for r in read_jsonl(path)}

    durations: list[float] = []
    out_train, out_eval = [], []
    t0 = time.time()
    for i, (row, src_name) in enumerate(rows):
        eid = row["example_id"]
        if eid in existing:
            continue
        wav_path = wav_dir / f"{eid}.wav"
        wav = synthesize(chat, normalize_for_tts(row["prompt"]), args.speed, args.pitch_semitones, args.device)
        if wav is None:
            print(f"[tts] FAIL {eid}: {row['prompt'][:40]!r}", flush=True)
            continue
        import soundfile as sf
        # 24k -> 16k to match whisper's input convention. wav[::3] was a bug:
        # writing every 3rd 24k sample at 16k plays the audio 2x too fast
        # (the correct decimation ratio is 24/16 = 1.5) — that is why every
        # batch sounded "too fast": the synthesis was fine, the downmix
        # sped it up. Resample properly instead.
        import torch
        import torchaudio.functional as AF
        wav16 = AF.resample(torch.from_numpy(wav.copy()).float().unsqueeze(0), 24000, 16000).squeeze(0).numpy()
        sf.write(wav_path, wav16, 16000)
        durations.append(len(wav16) / 16000)
        example = Example(
            example_id=eid,
            modality="audio",
            # The source figure_math manifest stores image paths relative to
            # ITS OWN directory; this manifest lives one level deeper, so the
            # image field must point back to the source dir (v5 lesson: a
            # hardcoded ../figure_math/ prefix broke when the source dir was
            # figure_math_v5/ and pointed at a nonexistent eval_images/).
            image=f"../{src_name}/{row['image']}",
            audio=str(wav_path.relative_to(args.out_dir)).replace("\\", "/"),
            prompt=row["prompt"],
            answer=row["answer"],
            split=row["split"],
            source="chattts_synthesized",
            task="audio_math",
            visual_required=True,
            metadata={**(row.get("metadata") or {}), "audio_sec": round(len(wav16) / 16000, 2)},
        )
        (out_train if row["split"] == "train" else out_eval).append(example)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"[tts] {i + 1}/{len(rows)} ok={len(out_train) + len(out_eval)} "
                  f"elapsed={el:.0f}s est_remaining={el / max(i + 1, 1) * (len(rows) - i - 1):.0f}s", flush=True)
    if shard_m > 1:
        # Sharded run: each process owns a disjoint row slice, so the wavs
        # never collide; only the manifests are shard-specific. Merge them
        # after all shards finish (see scripts/merge_tts_shards or below).
        n_train = write_jsonl(args.out_dir / f"train.shard{shard_i}.jsonl", out_train)
        n_eval = write_jsonl(args.out_dir / f"eval.shard{shard_i}.jsonl", out_eval)
    else:
        n_train = write_jsonl(args.out_dir / "train.jsonl", out_train)
        n_eval = write_jsonl(args.out_dir / "eval.jsonl", out_eval)
    if durations:
        ds = np.array(durations)
        print(f"[tts] done train={n_train} eval={n_eval} duration sec: "
              f"mean={ds.mean():.2f} p90={np.percentile(ds, 90):.2f} max={ds.max():.2f}")


if __name__ == "__main__":
    main()
