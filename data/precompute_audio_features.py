"""Precompute whisper encoder features for synthesized audio manifests.

Reads wavs (16k mono), runs the frozen whisper-small encoder, interpolates
the frame axis to the fixed audio-token budget (256 by default — same as
`project_vision_features` does at train time, so collate needs zero logic),
and stores [256, d] float32 npy files next to the wavs. The manifest rows
gain an `audio_features` field pointing at the npy.

Changing the token budget or the whisper checkpoint requires regenerating
the npy files (~10-20 min for 3500 clips with 4 GPUs; the whisper encoder
asserts a 3000-frame mel input, so every clip costs one full 30s pass).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.audio_encoder import FrozenWhisperEncoder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--whisper-model", default="openai/whisper-small")
    ap.add_argument("--audio-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", help="i/M: process only rows i, i+M, ... (parallel on multiple GPUs)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    shard_i, shard_m = (0, 1)
    if args.shard:
        shard_i, shard_m = (int(x) for x in args.shard.split("/"))

    import soundfile as sf

    rows = [json.loads(x) for x in args.manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    out_root = args.manifest.parent / "features"
    out_root.mkdir(parents=True, exist_ok=True)
    enc = FrozenWhisperEncoder(args.whisper_model, device=args.device, dtype=torch.float32).load()
    todo = [r for r in rows if r.get("audio") and r.get("audio_features") is None]
    todo = todo[shard_i::shard_m]
    if args.skip_existing:
        todo = [r for r in todo if not (out_root / f"{r['example_id']}.npy").exists()]
    print(f"[precompute] shard={shard_i}/{shard_m} {len(todo)} clips, tokens={args.audio_tokens}, device={args.device}")

    t0 = time.time()
    done = 0
    for start in range(0, len(todo), args.batch_size):
        chunk = todo[start: start + args.batch_size]
        wavs, paths = [], []
        for r in chunk:
            wav_path = Path(r["audio"])
            if not wav_path.is_absolute():
                wav_path = args.manifest.parent / wav_path
            wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                raise SystemExit(f"unexpected sample rate {sr} in {wav_path}")
            wavs.append(wav.astype(np.float32))
            paths.append(out_root / f"{r['example_id']}.npy")
        feats, mask = enc.preprocess(wavs)
        with torch.no_grad():
            raw = enc.encode(feats, mask)  # [B, 1500, d]
        # Interpolate the frame axis to the fixed token budget (mirrors
        # train/mm_dataset.project_vision_features).
        if raw.size(1) != args.audio_tokens:
            raw = F.interpolate(raw.transpose(1, 2), size=args.audio_tokens,
                                mode="linear", align_corners=False).transpose(1, 2)
        for r, path, fea in zip(chunk, paths, raw):
            np.save(path, fea.float().cpu().numpy())
            r["audio_features"] = str(path.relative_to(args.manifest.parent)).replace("\\", "/")
            done += 1
        if done % 100 < args.batch_size:
            el = time.time() - t0
            print(f"[precompute] {done}/{len(todo)} elapsed={el:.0f}s "
                  f"est_remaining={el / max(done, 1) * (len(todo) - done):.0f}s", flush=True)
    # Rewrite the manifest with the audio_features field filled in. Sharded
    # runs patch a FRESH read of the manifest with only this shard's rows so
    # parallel processes never clobber each other's fields (read-modify-write
    # against the latest state; process exit times differ so the window is
    # negligible). The npy files themselves are per-example_id and never race.
    # Sharded runs patch a FRESH read of the manifest under an flock so the
    # read-modify-write can never interleave across shards (a bare RMW loses
    # whichever shard wrote between our read and our write).
    if shard_m > 1:
        import fcntl
        with open(f"{args.manifest}.lock", "w", encoding="utf-8") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            fresh = [json.loads(x) for x in args.manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
            patch = {r["example_id"]: r["audio_features"] for r in rows if r.get("audio_features")}
            for r in fresh:
                if r["example_id"] in patch:
                    r["audio_features"] = patch[r["example_id"]]
            args.manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in fresh), encoding="utf-8")
            fcntl.flock(lf, fcntl.LOCK_UN)
    else:
        args.manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"[precompute] shard={shard_i}/{shard_m} done={done} manifest={args.manifest}")


if __name__ == "__main__":
    main()
