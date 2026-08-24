"""Synthesize spoken answers with ChatTTS — the voice-output half of the loop.

Reads model predictions (JSONL rows with a text field, or the demo JSON array
with per-row "pred_audio"), collapses degenerate repeated clauses (the 0.25B
occasionally loops one phrase until max_new_tokens), and synthesizes each
answer to 16k mono wav with the same voice settings as the question TTS.

Run ONLY inside the `omni-tts` conda env (same constraints as gen_audio_tts:
chattts pins transformers, must never touch the omni training env).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.gen_audio_tts import normalize_for_tts, synthesize  # noqa: E402

# Clause split on the CoT separators; a trailing clause without a separator
# is kept as its own unit.
_CLAUSE = re.compile(r"[^。；]*[。；]?|[^。；]+$")


def collapse_loops(text: str) -> str:
    """Drop consecutive duplicated clauses (degenerate repeat loops).

    A correct CoT chains distinct comparisons; a failed rollout often loops
    one phrase until max_new_tokens. Speaking the loop once is honest and
    keeps the utterance short; speaking it 10x adds nothing.
    """
    units = [u for u in _CLAUSE.findall(text) if u.strip()]
    out = []
    for u in units:
        if not out or out[-1] != u:
            out.append(u)
    return "".join(out)


def final_sentence(text: str) -> str:
    """Speak only the concluding clause ("第X季度的柱最高，所以答案是X。")."""
    units = [u for u in _CLAUSE.findall(text) if u.strip()]
    return units[-1] if units else text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="JSONL of predictions, or the demo JSON array (see --field)")
    ap.add_argument("--field", default="prediction",
                    help="text field to read from each row (demo JSON: pred_audio)")
    ap.add_argument("--out-dir", type=Path, default=Path("demo/out/answer_wavs"))
    ap.add_argument("--manifest-out", type=Path, default=Path("demo/out/answer_tts.jsonl"))
    ap.add_argument("--mode", choices=["full", "final"], default="full",
                    help="full: speak the whole generated answer (reasoning aloud); "
                         "final: speak only the concluding clause")
    ap.add_argument("--speed", type=float, default=0.65,
                    help="same voice settings as the question TTS (0.65 = [speed_3])")
    ap.add_argument("--pitch-semitones", type=float, default=-2.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    raw = args.input.read_text(encoding="utf-8").lstrip()
    rows = json.loads(raw) if raw.startswith("[") else [
        json.loads(x) for x in raw.splitlines() if x.strip()]

    import torch
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    import ChatTTS
    chat = ChatTTS.Chat()
    chat.load(compile=False, source="huggingface", device=args.device)
    if chat.device == "cpu":
        raise SystemExit("ChatTTS failed to load on CUDA; check torch build")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    existing = set()
    if args.skip_existing:
        for p in sorted(args.out_dir.glob("*.wav")):
            existing.add(p.stem)

    import soundfile as sf
    import torchaudio.functional as AF

    durations: list[float] = []
    manifest = []
    t0 = time.time()
    for i, row in enumerate(rows):
        eid = row.get("example_id") or row.get("id") or f"row{i:05d}"
        if eid in existing:
            continue
        text = row.get(args.field) or ""
        text = collapse_loops(text)
        if args.mode == "final":
            text = final_sentence(text)
        if not text.strip():
            print(f"[ans-tts] SKIP {eid}: empty prediction", flush=True)
            continue
        wav = synthesize(chat, normalize_for_tts(text), args.speed, args.pitch_semitones, args.device)
        if wav is None:
            print(f"[ans-tts] FAIL {eid}: {text[:40]!r}", flush=True)
            continue
        wav16 = AF.resample(torch.from_numpy(wav.copy()).float().unsqueeze(0),
                            24000, 16000).squeeze(0).numpy()
        wav_path = args.out_dir / f"{eid}.wav"
        sf.write(wav_path, wav16, 16000)
        sec = len(wav16) / 16000
        durations.append(sec)
        manifest.append({"example_id": eid, "text": text, "wav": str(wav_path),
                         "audio_sec": round(sec, 2)})
        if (i + 1) % 5 == 0:
            el = time.time() - t0
            print(f"[ans-tts] {i + 1}/{len(rows)} elapsed={el:.0f}s "
                  f"est_remaining={el / (i + 1) * (len(rows) - i - 1):.0f}s", flush=True)

    args.manifest_out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest), encoding="utf-8")
    if durations:
        ds = np.array(durations)
        print(f"[ans-tts] done n={len(manifest)} duration sec: mean={ds.mean():.2f} "
              f"p90={np.percentile(ds, 90):.2f} max={ds.max():.2f}")


if __name__ == "__main__":
    main()
