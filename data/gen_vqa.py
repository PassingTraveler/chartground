"""Teacher-VLM annotation pipeline with explicit no-fake-label safeguards.

Input rows must already contain image paths. Without --model, the command only
supports --copy-existing-answer; it never invents labels for an unannotated image.

Concurrent (thread-pool) teacher inference with per-row failure tolerance and
resume: already-annotated rows (by example_id in the output file) are skipped,
bad rows are recorded as failures instead of aborting the whole run.
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from .schema import read_jsonl, write_jsonl


def _resolve_image(image_path: str, input_path: Path) -> Path | None:
    p = Path(image_path)
    # Relative paths are resolved against the manifest directory, matching
    # the project-wide convention (see data/schema.py).
    if not p.is_absolute():
        p = input_path.parent / p
    return p if p.is_file() else None


def _annotate_row(row: dict, input_path: Path, processor, model, device,
                  copy_existing_answer: bool, max_new_tokens: int) -> tuple[dict | None, str | None]:
    """Return (annotated row, failure reason). A row is never half-written."""
    image_path = _resolve_image(row.get("image", ""), input_path)
    if image_path is None:
        return None, f"missing image: {row.get('image')}"
    answer = row.get("answer", "").strip() if copy_existing_answer else ""
    prompt = row.get("prompt", "请用中文简洁描述图片中的主要内容，并给出一个可核验的事实。")
    if model is not None:
        with Image.open(image_path).convert("RGB") as image:
            inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
        answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    if not answer:
        return None, f"empty teacher answer for {image_path}"
    obj = dict(row)
    # Keep the row's original task/source; only fill them in when absent.
    obj.setdefault("task", "vqa")
    obj.setdefault("source", f"teacher:{model_path_repr(model, copy_existing_answer)}")
    metadata = obj.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        obj["metadata"] = metadata
    metadata["teacher_prompt"] = prompt
    obj["answer"] = answer
    return obj, None


def model_path_repr(model, copy_existing_answer: bool) -> str:
    if copy_existing_answer or model is None:
        return "existing"
    return getattr(model, "name_or_path", "teacher_vlm")


def annotate(input_path: Path, output_path: Path, model_path: str | None, limit: int | None,
             copy_existing_answer: bool, max_new_tokens: int, workers: int = 1) -> dict:
    rows = read_jsonl(input_path)
    rows = rows[:limit] if limit is not None else rows
    if not copy_existing_answer and not model_path:
        raise ValueError("provide --model for teacher annotation, or --copy-existing-answer for pre-labeled rows")
    # Resume: rows already present in the output manifest are not re-annotated.
    done_ids: set[str] = set()
    if output_path.exists():
        done_ids = {r.get("example_id") for r in read_jsonl(output_path)}
        print(f"resume: {len(done_ids)} rows already annotated in {output_path}", flush=True)
    pending = [r for r in rows if r.get("example_id") not in done_ids]
    if not pending:
        return {"input": len(rows), "annotated": 0, "skipped": len(rows), "failed": 0}

    model = processor = device = None
    if model_path:
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("teacher annotation requires torch and transformers") from exc
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype="auto").to(device).eval()

    out_path = output_path if output_path.exists() else None
    lock = threading.Lock()
    annotated = failed = 0
    failures: list[str] = []

    def worker(row: dict) -> tuple[dict | None, str | None]:
        return _annotate_row(row, input_path, processor, model, device,
                             copy_existing_answer, max_new_tokens)

    def flush(rows_out: list[dict]) -> None:
        nonlocal out_path
        from .schema import write_jsonl
        write_jsonl(output_path, rows_out, append=out_path is not None)
        out_path = output_path

    batch: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(worker, r): r for r in pending}
        for future in as_completed(futures):
            row, reason = future.result()
            with lock:
                if row is None:
                    failed += 1
                    failures.append(reason or "unknown")
                    print(f"FAIL {futures[future].get('example_id')}: {reason}", flush=True)
                else:
                    annotated += 1
                    batch.append(row)
                    if len(batch) >= 100:
                        flush(batch)
                        batch = []
    if batch:
        flush(batch)
    return {"input": len(rows), "annotated": annotated, "skipped": len(rows) - len(pending), "failed": failed,
            "failure_samples": failures[:10]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", help="local transformers vision-language checkpoint")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--copy-existing-answer", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--workers", type=int, default=1, help="concurrent teacher-inference workers")
    args = ap.parse_args()
    print(json.dumps(annotate(args.input, args.output, args.model, args.limit,
                              args.copy_existing_answer, args.max_new_tokens, args.workers),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
