"""External VLM baseline eval on our probe grid + new-type Δvis.

Runs a HuggingFace image-text-to-text model zero-shot on:
  1) the 15-cell probe grid (with-image only, 200 rows each)
  2) figure_math_v5/eval.jsonl with-image + without-image arms (paired Δvis)
Same prompts/images/last-number scorer as our own eval, so numbers are
directly comparable with out/probe_v6|v7 and out/eval_mm_sft_v6_new.

Shared helpers (run_rows / score_vis_eval / acc_of / PROBE_CELLS) are also
imported by eval_baseline_minimind.py; a model adapter just supplies
predict_fn(prompt, image_or_none) -> str.

Usage:
  python -m eval.eval_baseline_vlm --model-id Qwen/Qwen2.5-VL-3B-Instruct \
      --local-dir <hf snapshot> --device cuda:0 --out-dir out/baseline_qwen3b
Resumable: rows whose example_id already exists in the output are skipped.
"""
import argparse
import json
import os
import random
import re
from pathlib import Path

import torch
from PIL import Image

PROBE_CELLS = [
    ("probe_grid", "n2_argmax"), ("probe_grid", "n2_named"),
    ("probe_grid", "n3_argmax"), ("probe_grid", "n3_named"),
    ("probe_grid", "n4_argmax"), ("probe_grid", "n4_argmin"),
    ("probe_grid", "n4_named"), ("probe_grid", "n4_2nd"),
    ("probe_grid", "n4_sum2"), ("probe_grid", "n4_sum2v1"),
    ("probe_grid", "n4_groupsum"), ("probe_grid", "n4_sum2tall"),
    ("probe_ablation", "n2_whichbar_max"),
    ("probe_ablation", "n4_whichbar_max"),
    ("probe_ablation", "n4_whichbar_min"),
]


def last_num(s: str):
    m = re.findall(r"[-+]?\d+(?:\.\d+)?", str(s).replace(",", "").replace("，", ""))
    return m[-1] if m else None


def run_rows(predict_fn, rows, image_base, out_path, max_new_tokens=128):
    """predict_fn(prompt, image_or_none) -> str; appends per-row results, resumable."""
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["example_id"] for l in open(out_path)}
    with open(out_path, "a", encoding="utf-8") as f:
        for r in rows:
            if r["example_id"] in done:
                continue
            img = None
            if r.get("image"):
                p = Path(r["image"])
                if not p.is_absolute():
                    p = image_base / p
                img = Image.open(p).convert("RGB")
            try:
                pred = predict_fn(r["prompt"], img)
            except Exception as e:
                pred = f"<ERROR {type(e).__name__}>"
                print(f"  ERR {r['example_id']}: {e}", flush=True)
            f.write(json.dumps(
                {"example_id": r["example_id"], "prediction": pred, "answer": r["answer"]},
                ensure_ascii=False) + "\n")
            f.flush()
    return out_path


def acc_of(path):
    rows = [json.loads(l) for l in open(path)]
    hit = sum(1 for r in rows if last_num(r["prediction"]) is not None
              and last_num(r["prediction"]) == last_num(r["answer"]))
    return hit, len(rows), (hit / len(rows) if rows else 0.0)


def score_vis_eval(with_path, noimg_path):
    """Paired Δvis + bootstrap CI over the two arms (aligned row order)."""
    with_rows = [json.loads(l) for l in open(with_path)]
    without_rows = [json.loads(l) for l in open(noimg_path)]
    assert len(with_rows) == len(without_rows), "arms misaligned"
    deltas, hit_w, hit_wo = [], 0, 0
    for w, wo in zip(with_rows, without_rows):
        cw = last_num(w["prediction"]) == last_num(w["answer"])
        cwo = last_num(wo["prediction"]) == last_num(wo["answer"])
        hit_w += cw
        hit_wo += cwo
        deltas.append(int(cw) - int(cwo))
    n = len(deltas)
    rng = random.Random(0)
    boots = []
    for _ in range(1000):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(deltas[i] for i in idx) / n)
    boots.sort()
    return {"with_image": hit_w / n, "without_image": hit_wo / n,
            "delta_vis": hit_w / n - hit_wo / n,
            "ci_low": boots[25], "ci_high": boots[975], "n": n}


def load_model(local_dir: str, device: str):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=False)
    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            local_dir, torch_dtype=torch.bfloat16, device_map=device,
            attn_implementation="sdpa")
    except Exception:
        # AutoModelForVision2Seq was removed in transformers 5.x; generic fallback
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            local_dir, torch_dtype=torch.bfloat16, device_map=device,
            attn_implementation="sdpa")
    model.eval()
    return processor, model


def build_inputs(processor, family: str, prompt: str, image):
    """Return kwargs for model.generate."""
    if family == "qwen":
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        return processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
    elif family == "smolvlm":
        content = []
        if image is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        return processor(text=text, images=[image] if image is not None else None,
                         return_tensors="pt")
    else:
        raise ValueError(f"unknown family {family}")


def generate(processor, model, inputs, max_new_tokens: int) -> str:
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if hasattr(out, "sequences"):  # GenerateOutput
        out = out.sequences
    if isinstance(out, list):
        out = out[0] if len(out) == 1 else torch.cat(out, dim=0)
    tok = getattr(processor, "tokenizer", processor)
    # decode row 0 directly (some VLM processors' batch_decode chokes on lists),
    # slicing off the input prompt
    in_len = inputs["input_ids"].shape[1]
    return tok.decode(out[0][in_len:], skip_special_tokens=True,
                      clean_up_tokenization_spaces=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--local-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    family = "qwen" if args.model_id.lower().startswith("qwen") else "smolvlm"
    print(f"[baseline] {args.model_id} family={family} device={args.device}", flush=True)
    processor, model = load_model(args.local_dir, args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    def predict_fn(prompt, image):
        inputs = build_inputs(processor, family, prompt, image)
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        return generate(processor, model, inputs, args.max_new_tokens)

    summary = {}
    # 1) probe cells (with-image)
    probe_dir = os.path.join(args.out_dir, "probe")
    os.makedirs(probe_dir, exist_ok=True)
    for sub, cell in PROBE_CELLS:
        manifest = Path(f"data/generated/{sub}/{cell}.jsonl")
        rows = [json.loads(l) for l in open(manifest)]
        out = os.path.join(probe_dir, f"{cell}.jsonl")
        run_rows(predict_fn, rows, manifest.parent, out, args.max_new_tokens)
        hit, n, acc = acc_of(out)
        summary[f"probe:{cell}"] = f"{acc:.3f} ({hit}/{n})"
        print(f"[probe] {cell}: {acc:.3f} ({hit}/{n})", flush=True)

    # 2) new-type Δvis paired
    eval_manifest = Path("data/generated/figure_math_v5/eval.jsonl")
    rows = [json.loads(l) for l in open(eval_manifest)]
    with_path = os.path.join(args.out_dir, "vis_eval_with_image.jsonl")
    run_rows(predict_fn, rows, eval_manifest.parent, with_path, args.max_new_tokens)
    rows_noimg = [dict(r, image=None) for r in rows]
    noimg_path = os.path.join(args.out_dir, "vis_eval_without_image.jsonl")
    run_rows(predict_fn, rows_noimg, eval_manifest.parent, noimg_path, args.max_new_tokens)

    vis = score_vis_eval(with_path, noimg_path)
    summary["vis_eval"] = vis
    print(f"[vis] with {vis['with_image']:.3f} without {vis['without_image']:.3f} "
          f"Δvis {vis['delta_vis']:.3f} CI [{vis['ci_low']:.3f}, {vis['ci_high']:.3f}]",
          flush=True)

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("[baseline] DONE", flush=True)


if __name__ == "__main__":
    main()
