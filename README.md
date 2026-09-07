# ChartGround

基于 0.25B 基座的图表视觉数学与语音题面问答实验系统。

ChartGround 在独立预训练的 0.25B 语言模型上接入冻结的 SigLIP2 视觉编码器和 Whisper 音频编码器，自建程序化可验证数据管线，完成 projector 对齐、五流混合 SFT 和 GRPO 训练，研究小模型能从视觉与听觉输入中吸收什么、以及吸收不了什么。语音回答由外部 ChatTTS 层合成，作为交互闭环演示，不是原生音频 token 生成。

> 这是一个研究归档与复现工程，不是通用 VLM。主指标是 paired Δvis——同一题、同一模型、同一解码配置下带图与无图文本版准确率之差；不用绝对准确率对齐通用模型。

## Results

在相同题目、相同解码协议下：

- 新题型（SFT v6）：Δvis = **0.413**，95% CI [0.357, 0.471]；with_image 0.583，without 0.170
- 分题型：named 比较题带图 1.000；相邻下降段计数 0.605；无锚点 argmax/argmin 与两组求和比较接近随机，定位出 0.25B 在「题面形式 → 操作选择」映射上的能力边界
- 外部对照（同 prompt 同图）：minimind-3v-moe Δvis 0.044、SmolVLM-256M Δvis 0.028、Qwen2.5-VL-3B 零样本 Δvis 0.557；基线为零样本、本项目为训练后，口径已在报告中注明
- GRPO v6/v7：RL 期间 without 臂持续上升（0.080→0.152、0.232→0.267），with_image 持平略降——RL 学到的是「输出更像答案」而非「读得更对」，方向关闭
- SFT v7 增加 13k 条操作分解课程数据后探针仍接近随机，形成可复现的负结果证据
- 语音题面新题型听题准确率 0.328；该结果与图像输入共同测量，不能解释为通用语音理解能力
- 文本保持 0.016，无灾难性遗忘

完整数字、判别实验和失败案例见 [REPORT.md](REPORT.md)。

## Demo

- `python demo/app.py`：零依赖本地页面，按同一 ID 回放 `curriculum_eval_v7` 的真实保存预测与评测图像，正确与错误样例都展示；不加载模型、不伪造在线生成。
- `python demo/web_demo.py --checkpoint out/mm_sft_v6/mm_sft.pt --vision-model model/siglip2-base-patch16-256 --device cuda:0`：需 checkpoint 与 CUDA 的 Gradio 图像推理页面，默认端口 `7864`。
- `python scripts/demo.py ...`：加载 checkpoint 的单图命令行推理。
- `python demo/demo_audio_math.py --reuse-json --out-json demo/out/demo_audio_math_v5.json`：语音题面 + 图表 + 语音回答的自包含演示。

运行方式与口径边界见 [demo/README.md](demo/README.md)。

## What is implemented

### Model and multimodal architecture

- 0.25B decoder-only 语言模型（独立预训练权重，hidden 1024，24 层，词表 12014）
- 冻结 SigLIP2-base-patch16-256 视觉编码器，per-token MLP VisionProjector 输出 64 个连续视觉 token
- 冻结 Whisper-small 音频编码器，同构 AudioProjector 输出 256 个音频 token
- prompt 布局：图像标记 64 + 音频标记 256 + 文本，1024 位置预算内不含题面文字（Δaudio 协议要求）
- ChatTTS 语音输出层，支持朗读完整 CoT 或仅结论句

### Data pipeline

- 程序化图表数学题：Pillow 渲染，题面数字与答案同源 RNG，metadata 携带真值程序，答案可机械验证，零 API 成本
- ChatTTS 合成语音题面 + Whisper 特征预计算（4 卡分片）
- 五流混合 SFT（约 88k 行）：通用图文 0.40、无图数学 0.13、视觉数学 0.22、文本 replay 0.15、语音题面 0.10
- 评估集与训练集像素哈希零重叠

### Training

- 冻结 LLM 与编码器的 projector 预对齐（视觉、音频各一阶段）
- 4×RTX 3090 DDP 混合 SFT，bf16
- GRPO：group-relative advantage、KL 约束、平滑奖励，单卡可审计实现

### Evaluation

- paired Δvis 协议：Acc(带图) − Acc(无图同题文本版)，bootstrap 95% CI
- Δaudio 分解臂与零样本听题迁移
- 文本保持集、15 格探针矩阵、readable 渲染 / 视觉 token 数消融
- 三个外部 VLM 基线（同 prompt、同图、同一评分器）

## Main experiment chain

当前最终结果对应的主线为：

```text
pretrain_02b（独立预训练权重）
    ↓
align / align_audio_v2        # 双 projector 预对齐
    ↓
mm_sft_v4                    # 旧题型五流混合 SFT
    ↓
mm_sft_v5                    # 新题型（paircmp / turns / groupsum）重建
    ↓
mm_sft_v6                    # 音频流换入新题型，最终模型
    ↓
mm_grpo_v6 / mm_grpo_v7      # GRPO 对照（负结果）
    ↓
mm_sft_v7                    # 操作分解课程（负结果）
```

对应权重目录：

```text
out/align/projector.pt
out/align_audio_v2/audio_projector.pt
out/mm_sft_v5/mm_sft.pt
out/mm_sft_v6/mm_sft.pt
out/mm_sft_v7/mm_sft.pt
```

## Repository layout

```text
ChartGround/
├── data/       # 数据生成、清洗、混合与音频特征预计算
├── model/      # 多模态模型、视觉/音频编码器加载
├── train/      # 对齐、混合 SFT、GRPO
├── eval/       # Δvis/Δaudio paired 评测、探针、文本保持
├── scripts/    # 数据冒烟、训练与评测入口
├── demo/       # 回放与实时演示
├── docs/       # 复现说明
├── REPORT.md   # 完整实验报告
└── out/        # 本地权重与评测输出；GitHub 版本默认忽略
```

## Reproduction

### 1. Environment

建议使用 Linux + CUDA、Python 3.10+、4 张 RTX 3090。依赖安装：

```bash
pip install -r requirements.txt
```

Windows 开发环境无 GPU 时只能完成数据冒烟和静态检查；4 卡训练在 Linux/CUDA 主机执行。

### 2. Smoke test

```bash
python -m compileall -q .
bash scripts/run_data_smoke.sh
```

### 3. Current training entrypoints

```bash
# 视觉 projector 预对齐（冻结 LLM 与 SigLIP2）
BASE_MODEL=assets/project1/out/pretrain_02b/pretrain_h1024_l24.pth \
VISION_MODEL=model/siglip2-base-patch16-256 \
bash scripts/run_align.sh

# 4 卡五流混合 SFT
BASE_MODEL=assets/project1/out/pretrain_02b/pretrain_h1024_l24.pth \
VISION_MODEL=model/siglip2-base-patch16-256 \
PROJECTOR=out/align/projector.pt \
AUDIO_PROJECTOR=out/align_audio_v2/audio_projector.pt \
DATA=data/processed/mm_sft_v5.jsonl SAVE_DIR=out/mm_sft_v5 STEPS=3000 \
bash scripts/run_mm_sft.sh

# GRPO（单卡，不使用 torchrun）
CUDA_VISIBLE_DEVICES=0 python -m train.train_mm_grpo \
  --data data/processed/grpo_v7_newtypes.jsonl \
  --checkpoint out/mm_sft_v5/mm_sft.pt --save-dir out/mm_grpo_v7 \
  --steps 400 --group-size 8 --max-new-tokens 128
```

### 4. Evaluation

```bash
bash scripts/run_eval_suite.sh   # Δvis
bash scripts/run_audio_eval.sh   # Δaudio
```

生成数据全集、基座权重、编码器权重、图片、音频和 checkpoint 不随仓库发布；完整数据链、语音数据准备和评测细节见 [docs/REPRODUCTION.md](docs/REPRODUCTION.md) 与 [REPORT.md](REPORT.md)。

## Scientific scope

- Δvis 是主验收口径：同一题、同一模型、同一 prompt 与解码配置下带图与无图文本版之差，配 bootstrap 置信区间；不用绝对 acc 与独立数学 Agent 项目的结果对齐。
- 能力边界由四联判别实验支撑（readable 渲染、64→128 token、256 token 全保真、粗粒度步进读数）：精确数值读数超出 0.25B + SigLIP2-256 的感知边界，是模型规模问题而非管道缺陷。
- GRPO 是单卡、可审计的 group-relative 实现，不是多卡在线 RL 的完整复刻；其负结果是双重确认后的受控结论。
- 语音输出是外部 TTS 层；听题准确率与图像输入共同测量，跨模态的题型泛化不存在（AudioProjector 只训过旧题型语音时，新题型听题低于随机）。

## Limitations

- 无点名锚的最值、求和比较超出 0.25B + SigLIP2-256 的能力边界，相关题型接近随机；语言改写、模板、稀释、CoT 操作分解四条路线均未能解锁
- 题型覆盖比较与计数类，不能外推为通用图表理解或 OCR 能力
- RL 在当前规模与奖励密度下没有带来稳定的视觉增益
- 语音输出非原生音频 token 生成；原生路线见 REPORT.md 的讨论
- 仓库不含生成数据全集、基座与第三方编码器权重、服务器日志

## License and data

本仓库中的代码、数据处理脚本和评测脚本应分别遵守第三方数据集、SigLIP2/Whisper/ChatTTS 组件和基座模型的许可证。公开仓库前请按 docs/ 与 REPORT.md 审核数据来源、许可证和模型权重，不要直接上传原始语料、第三方权重或私有服务器日志。
