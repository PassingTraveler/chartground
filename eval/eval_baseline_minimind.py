"""minimind-3v-moe baseline eval (same probe grid + Δvis protocol).

The model ships its own non-transformers code (model_minimind.py / model_vlm.py)
in the HF snapshot, so this adapter imports it via importlib and builds the
prompt manually: '<|im_start|>user\n' + '<|image_pad|>'*64 + '\n' + question.

Usage:
  python -m eval.eval_baseline_minimind --local-dir <snapshot> \
      --vision-dir <siglip2-base-p32-256-ve snapshot> --device cuda:0 \
      --out-dir out/baseline_minimind3v
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

from eval.eval_baseline_vlm import PROBE_CELLS, acc_of, run_rows, score_vis_eval

IMAGE_PAD = "<|image_pad|>"
N_IMAGE_TOKENS = 64


def load_minimind(local_dir: str, vision_dir: str, device: str):
    # model_vlm.py does `from .model_minimind import *` — assemble a temp package
    import shutil
    import tempfile
    pkg_dir = tempfile.mkdtemp(prefix="mm_pkg_")
    pkg_name = os.path.basename(pkg_dir)
    os.makedirs(os.path.join(pkg_dir, pkg_name), exist_ok=True)
    for fname in ("model_minimind.py", "model_vlm.py"):
        shutil.copy(os.path.join(local_dir, fname), os.path.join(pkg_dir, pkg_name, fname))
    open(os.path.join(pkg_dir, pkg_name, "__init__.py"), "w").close()
    sys.path.insert(0, pkg_dir)
    model_vlm = importlib.import_module(f"{pkg_name}.model_vlm")

    cfg_dict = json.load(open(os.path.join(local_dir, "config.json")))
    cfg = model_vlm.VLMConfig(**cfg_dict)
    # The LLM/projector params load as fp32 (bf16 sd upcast), but their
    # get_vision_model picks up the repo's fp16 torch_dtype for SigLIP —
    # mixed precision breaks the projector LayerNorm. Force fp32 everywhere.
    from transformers import SiglipImageProcessor, SiglipVisionModel

    @staticmethod
    def _patched_get_vision_model(model_path):
        m = SiglipVisionModel.from_pretrained(model_path, torch_dtype=torch.float32)
        for p in m.parameters():
            p.requires_grad = False
        return m.eval(), SiglipImageProcessor.from_pretrained(model_path)

    model_vlm.MiniMindVLM.get_vision_model = _patched_get_vision_model
    model = model_vlm.MiniMindVLM(cfg, vision_model_path=vision_dir)
    sd = torch.load(os.path.join(local_dir, "pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # vision_encoder params are loaded separately from vision_model_path
    missing = [k for k in missing if not k.startswith("vision_encoder.")]
    assert not missing, f"missing keys: {missing[:5]}"
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    model.to(device).eval()

    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(local_dir, "tokenizer.json"))
    tok_cfg = json.load(open(os.path.join(local_dir, "tokenizer_config.json")))
    added = tok_cfg.get("added_tokens_decoder", {})
    tokens = [v["content"] for v in added.values() if v.get("content")]
    if tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", required=True)
    ap.add_argument("--vision-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    print(f"[minimind] load from {args.local_dir} device={args.device}", flush=True)
    model, tokenizer = load_minimind(args.local_dir, args.vision_dir, args.device)
    pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
    assert pad_id != tokenizer.unk_token_id, "image_pad token missing"
    print(f"[minimind] image_pad id={pad_id} vocab={len(tokenizer)}", flush=True)

    def predict_fn(prompt, image):
        text = ("<|im_start|>user\n" +
                (IMAGE_PAD * N_IMAGE_TOKENS if image is not None else "") +
                ("\n" if image is not None else "") + prompt +
                "<|im_end|>\n<|im_start|>assistant\n")
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert pad_id in ids if image is not None else True, "pad token not preserved"
        input_ids = torch.tensor([ids], device=args.device, dtype=torch.long)
        mask = torch.ones_like(input_ids)
        kwargs = {}
        if image is not None:
            # pass the whole feature dict: model.forward routes it through the
            # dict branch of get_image_embeddings (a bare tensor breaks on **kwargs)
            pv = model.processor(images=image, return_tensors="pt")
            kwargs["pixel_values"] = {k: v.to(args.device, dtype=model.dtype)
                                      for k, v in pv.items()}
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=mask,
                                 do_sample=False, max_new_tokens=args.max_new_tokens,
                                 **kwargs)
        return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}
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

    eval_manifest = Path("data/generated/figure_math_v5/eval.jsonl")
    rows = [json.loads(l) for l in open(eval_manifest)]
    with_path = os.path.join(args.out_dir, "vis_eval_with_image.jsonl")
    run_rows(predict_fn, rows, eval_manifest.parent, with_path, args.max_new_tokens)
    rows_noimg = [dict(r, image=None) for r in rows]
    noimg_path = os.path.join(args.out_dir, "vis_eval_without_image.jsonl")
    run_rows(predict_fn, rows_noimg, eval_manifest.parent, noimg_path,
             args.max_new_tokens)

    vis = score_vis_eval(with_path, noimg_path)
    summary["vis_eval"] = vis
    print(f"[vis] with {vis['with_image']:.3f} without {vis['without_image']:.3f} "
          f"Δvis {vis['delta_vis']:.3f} CI [{vis['ci_low']:.3f}, {vis['ci_high']:.3f}]",
          flush=True)

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("[minimind] DONE", flush=True)


if __name__ == "__main__":
    main()
