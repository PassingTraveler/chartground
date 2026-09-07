#!/usr/bin/env python
"""ChartGround 实时图像推理 Web Demo。

一次启动时加载语言模型、SigLIP2 和 VisionProjector，之后每次上传图片
只执行预处理、视觉特征注入和生成，不重复加载 1GB 级 checkpoint。

运行：
  python demo/web_demo.py --checkpoint out/mm_sft_v6/mm_sft.pt \
    --vision-model model/siglip2-base-patch16-256 --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.mm_template import MMTemplate  # noqa: E402
from model.projector import VisionProjector  # noqa: E402
from model.vision_encoder import FrozenSigLIP2  # noqa: E402
from train.mm_dataset import project_vision_features  # noqa: E402
from scripts.demo import load_checkpoint  # noqa: E402

COMPONENTS = None


def load_components(checkpoint: Path, vision_model: Path, tokenizer_path: Path, device_name: str):
    device = torch.device(device_name if not device_name.startswith("cuda") or torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model, projector_state = load_checkpoint(checkpoint, len(tokenizer.get_vocab()), device)
    model.eval()
    vision = FrozenSigLIP2(vision_model, device=device, dtype=torch.float32).load()
    projector = VisionProjector(vision.hidden_size, model.config.hidden_size, 1024).to(device).eval()
    if projector_state is None:
        raise RuntimeError("checkpoint 缺少 projector 权重，请使用 MM SFT/GRPO checkpoint")
    projector.load_state_dict(projector_state, strict=False)
    return model, vision, projector, tokenizer, MMTemplate(), device


def infer(image, question: str, max_new_tokens: int, temperature: float) -> str:
    if image is None:
        return "请先上传一张图表。"
    question = (question or "").strip()
    if not question:
        return "请输入图表问题。"
    if COMPONENTS is None:
        return "模型尚未加载完成。"
    model, vision, projector, tokenizer, template, device = COMPONENTS
    start = time.perf_counter()
    prompt = template.prompt(question, "image", 1, tokens_per_image=model.config.image_token_len)
    ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False).ids], dtype=torch.long, device=device)
    pixels = vision.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].to(device)
    with torch.inference_mode():
        raw_features = vision.encode(pixels)
        visual_features = project_vision_features(raw_features, projector, model.config.image_token_len)
        output = model.generate(
            ids,
            vision_features=visual_features,
            max_new_tokens=int(max_new_tokens),
            eos_id=2,
            temperature=float(temperature),
            top_k=50 if temperature > 0 else 0,
        )
    answer = tokenizer.decode(output[0, ids.size(1):].tolist()).strip()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return f"{answer}\n\n---\n推理耗时：{elapsed_ms:.0f} ms · 图像 token：{model.config.image_token_len} · device：{device}"


def main() -> None:
    parser = argparse.ArgumentParser(description="ChartGround realtime visual math demo")
    parser.add_argument("--checkpoint", type=Path, default=Path("out/mm_sft_v6/mm_sft.pt"))
    parser.add_argument("--vision-model", type=Path, default=Path("model/siglip2-base-patch16-256"))
    parser.add_argument("--tokenizer", type=Path, default=Path("assets/project1/tokenizer/best_mm.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7864)
    args = parser.parse_args()

    global COMPONENTS
    COMPONENTS = load_components(args.checkpoint, args.vision_model, args.tokenizer, args.device)
    import gradio as gr

    with gr.Blocks(title="ChartGround · 视觉数学 Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """# ChartGround · 视觉数学 Demo
上传图表并提问，模型会将 SigLIP2 视觉特征通过 VisionProjector 注入 0.25B 语言骨干。

> 这是实时单图推理入口；SFT v6 的正式 paired Δvis=0.413（95% CI [0.357, 0.471]）来自独立评测协议，不由单个 Demo 样例替代。"""
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="上传图表")
                question = gr.Textbox(
                    label="问题",
                    value="图中四个季度的销量分别由四根柱表示。销量最高的是第几个季度？只输出编号（1到4）。",
                    lines=4,
                )
                with gr.Row():
                    max_tokens = gr.Slider(16, 256, value=128, step=16, label="最大生成 token")
                    temperature = gr.Slider(0, 1, value=0, step=0.1, label="温度")
                run = gr.Button("开始视觉推理", variant="primary")
            with gr.Column(scale=1):
                answer = gr.Markdown("等待上传图像。", label="模型回答")
                gr.Markdown(
                    """### 推理链
1. 图像 → 冻结 SigLIP2 视觉编码器
2. 视觉 token → VisionProjector
3. 注入 `<|image_pad|>` → Decoder-only 语言模型
4. 生成答案并显示单次耗时"""
                )
        run.click(infer, [image, question, max_tokens, temperature], answer)

    demo.launch(server_name=args.host, server_port=args.port, show_error=True, quiet=True)


if __name__ == "__main__":
    main()
