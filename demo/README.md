# ChartGround Demo

```bash
python demo/app.py
```

打开 `http://127.0.0.1:7862`。页面回放 `curriculum_eval_v7` 的真实保存预测与同一评测图像，正确和错误样例都会展示。它不加载模型、不伪造在线多模态回答。

实时图像 Web Demo（需要 Gradio、Torch、模型权重和 CUDA）：

```bash
python -m pip install -r requirements.txt -r requirements-demo.txt
python demo/web_demo.py --checkpoint out/mm_sft_v6/mm_sft.pt --vision-model model/siglip2-base-patch16-256 --device cuda:0
```

打开 `http://127.0.0.1:7864`，上传图表即可进行真实推理。

实时单图推理：

```bash
python scripts/demo.py --checkpoint out/mm_sft_v6/mm_sft.pt --image <chart.png> --question "<问题>" --vision-model <siglip2_path>
```

语音题面 + 图表 + 语音回答的自包含演示：

```bash
python demo/demo_audio_math.py --reuse-json --out-json demo/out/demo_audio_math_v5.json
```

请区分：SFT v6 的正式 paired Δvis 为 0.413（95% CI [0.357, 0.471]）；本页面的 v7 样例仅用于展示模型输出和失败边界。
