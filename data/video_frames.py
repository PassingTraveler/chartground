"""Deterministic video frame extraction with an auditable JSONL manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import write_jsonl


def extract(video: Path, out_dir: Path, n_frames: int = 4) -> list[str]:
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("video frame extraction requires opencv-python") from e
    cap = cv2.VideoCapture(str(video))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"cannot read frame count: {video}")
        # Never duplicate indices when the video has fewer frames than asked.
        n = min(n_frames, total)
        indices = [round(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
        # One sub-directory per video so same-named videos never overwrite.
        frame_dir = out_dir / video.stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for j, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"failed to read frame {idx} from {video}")
            path = frame_dir / f"frame_{idx:05d}.jpg"
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            paths.append(str(path.resolve()))
        return paths
    finally:
        cap.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--prompt", default="请观看视频中的画面，用中文回答视频内容相关的问题。")
    args = ap.parse_args()
    paths = extract(args.input, args.out_dir, args.frames)
    # One JSONL row per video, schema-compatible (Example-like); multi-frame
    # layout lives in metadata["frames"] for the video pipeline stage.
    row = {
        "example_id": f"video_{args.input.stem}",
        "modality": "video",
        "video": str(args.input.resolve()),
        "prompt": args.prompt,
        "answer": "",
        "split": "train",
        "source": "video_frames",
        "task": "video_qa",
        "visual_required": True,
        "metadata": {"n_frames": len(paths), "frames": paths},
    }
    n = write_jsonl(args.manifest, [row])
    print(json.dumps({"n_frames": len(paths), "manifest": str(args.manifest), "rows": n}))


if __name__ == "__main__":
    main()
