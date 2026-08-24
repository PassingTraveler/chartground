# ChartGround

**Compact Visual Math and Speech-QA Research System**

独立搭建的数据管线、多模态模型结构与训练评测框架，面向图表视觉数学和语音题面问答。模型接收文本、图像与语音题面输入；语音回答由外部 TTS 层生成，仅作为交互 Demo，不代表原生音频 token 生成。

## Project snapshot

- 0.25B 级解码式语言骨干，接入冻结视觉编码器和 Whisper 音频编码器。
- 自研 Vision/Audio Projector、连续 image/audio token 注入、混合 SFT 与 GRPO 训练代码。
- 10 万级五流训练数据：通用图文、无图数学、视觉数学、文本 replay 和语音题面。
- 支持 4×RTX 3090 DDP、bf16/fp16 混合精度、梯度累积、KV Cache 和分片评测。

## Results

- 视觉数学新题型：SFT v6 的 paired `Δvis=0.413`，95% CI `[0.357, 0.471]`。
- named 比较题接近满分；无锚点 argmax/argmin/sum 探针接近随机，定位出小模型的视觉操作选择瓶颈。
- SFT v7 增加 13k 条操作分解课程数据后，IID holdout 仍接近 chance，形成可复现的负结果证据。
- 语音题面新题型听题准确率 0.328；该结果与图像输入共同测量，不能解释为通用语音理解能力。

## Quick start

大体量数据、模型权重、生成图片、音频和实验输出不会提交到 Git；服务器准备、四卡训练和评测命令见 [docs/REPRODUCTION.md](docs/REPRODUCTION.md)。完整实验数字、失败案例和方法讨论见 [REPORT.md](REPORT.md)。

```bash
python -m compileall -q .
bash scripts/run_data_smoke.sh
```

## Repository layout

```text
data/       数据生成、清洗、混合与特征预计算
model/      多模态模型、视觉/音频编码器与模板
train/      对齐、SFT、GRPO 和训练工具
eval/       paired 增益、文本保持、探针和 baseline 评测
scripts/    数据准备、四卡训练与分片评测入口
docs/       复现说明与发布文档
```

## Research notes

以下内容保留完整的设计决策、数据职责划分和实验演进记录。

---

## 详细设计与实验记录

## 一、先确定基座：不再单独做数学文字基座

**结论**:项目二不需要再训练一个“数学文字模型”。主干使用项目一的**预训练权重** `out/pretrain_02b/pretrain_h1024_l24.pth`;项目一的数学 SFT/Agent 权重只作为对照实验和数据/奖励经验来源，不作为多模态主模型的初始化。

原因有三点:

1. 多模态主模型首先需要稳定的中文语言生成、指令跟随和视觉问答输出能力；数学专用 SFT 已经明显偏向算术模板和工具格式，直接作为主干会把文本分布带窄。
2. 项目一的 99.7% 结果依赖工具协议，且无工具文本能力明显较弱。把 `agent_sft_v3` 直接接视觉模块，无法区分收益来自视觉理解、数学模板还是工具格式。
3. 项目二的研究问题应是“视觉信息能否被小语言模型吸收，并在可验证视觉任务上受益于 RL”，而不是再次证明数学文本模型可以做算术。

数学模型保留为三种用途:

- **文本数学对照**:比较 `pretrain_02b` 与 `agent_sft_v3` 初始化对视觉数学 SFT/RL 的影响;
- **数据构造经验**:复用项目一的程序化生成、真值程序和答案校验;
- **负结果解释**:如果视觉 RL 不涨，分析是视觉 grounding 失败、数学规划不足，还是 RL 信号不足。

多模态 SFT 中把两类文本数据分开统计，不能把它们都叫作 replay:

- **无图数学指令 SFT 约 15%**:使用题面 + CoT/答案的纯文本数学题，先让 `pretrain_02b` 学会数学指令模板、答案格式和基本推理表达；这是“建立”数学文字能力，不是保持已有能力;
- **通用文本 replay 约 10%～15%**:通用中文指令、知识和少量项目一数学文本，用于防止多模态训练造成灾难性遗忘。

因此首版建议的 SFT token 配比为:通用图文 40%～50%、无图数学 15%、视觉数学 20%～30%、通用文本 replay 10%～15%；视频数据只有在视频阶段开启时再加入。训练后必须同时报告文本保持、图像理解和视觉数学结果。

## 二、定位与差异化(对照 minimind-o)

本地 [minimind-o-master](../../minimind-o-master) 已确认其能力边界(**2026-05-05 更新为 0.1B dense / 0.3B-A0.1B MoE 双版本 + 技术报告 arXiv:2605.03937,Apache-2.0**,但仍无视频输入、无 RL、视觉评估弱),本项目逐一打差异化:

| 能力 | minimind-3o(2026-05 版) | 本项目 |
|---|---|---|
| 输入模态 | 文本 + 语音 + 图像 | 文本 + 图像 + **语音输入(whisper-small 特征)**(视频已砍) |
| 训练 | 仅 SFT(T2A/A2A/I2T 三段) | SFT + **GRPO/DAPO 多模态 RLVR** |
| 视觉数据 | 复用公开数据(MiniMind-V 等) | **自建可追溯 pipeline**(程序化生成 + 授权图集 + 教师 VLM 标注) |
| 评估 | 技术报告仅报 CER/音色相似度,视觉评估仍弱 | **中文多模态评估集**(视觉数学/图表主评估 + 文本保持 + OOD；视频为可选) |
| 基座 | 0.1B dense / 0.3B-A0.1B MoE | **项目一预训练 0.25B**(238.8M);数学 Agent 权重只作消融 |
| 语音输出 | 有(Talker + MTP + 音色克隆) | **已实现**(ChatTTS 朗读模型回答,端到端语音闭环;非原生音频 token 生成,原生路线见 REPORT.md §8) |

**核心论点(面试 30 秒)**:minimind-o 只有 SFT 没有 RL、没有视频输入、视觉评估约等于没有——这三个空档就是本项目的位置。

## 三、资产复用清单(复用接口与经验,不复制数学模型偏置)

| 资产 | 来源 | 用法 |
|---|---|---|
| 0.25B 基座 | 项目一 `out/pretrain_02b/pretrain_h1024_l24.pth` | 多模态 LLM 主干(纯文本能力已验收) |
| 多模态特殊 token | 项目一 tokenizer `best_mm.json` | 14 个是 token 类型，不是 14 个视觉槽位；使用 `<|image_pad|>` 连续重复 64 次，视频按帧重复；不重新训练词表，但不能声称这些 embedding 已被视觉预训练定型 |
| 数学 SFT/Agent 权重 | 项目一 `out/agent_sft_v3` | 只做“预训练基座 vs 数学专用基座”消融，不作为主线初始化 |
| GRPO/DAPO 算法经验 | 项目一 `train/train_grpo.py` | 只复用 advantage、clip-higher、overlong shaping 和监控思路；必须重写多模态输入、视觉缓存、KV cache、reference log-prob |
| RL 四指标监控 | 项目一 acc/reward/长度/KL 曲线 | 继承监控维度；β、奖励平滑和 rollout 长度在视觉任务上重新做小规模 sweep |
| 程序化生成 + 真值程序 | 项目一 36k 算术模板的经验 | 视觉数学题照搬此范式:图 + 题 + 真值程序 |
| 评估框架 | 项目一双协议 4 卡分片 harness | 改造成多模态输入，并新增文本保持、图像置换、模板 OOD 对照 |
| minimind-o 基线 | 本地 `minimind-o-master` | 同题对比 + 能力边界清单 |

## 四、数据来源与构造(自建 pipeline,主体使用可追溯数据)

### 4.1 视觉数学题(RL 核心,程序化生成 + 真值程序)

**设计**:继承项目一「程序化生成 + 真值程序验证」范式——图由 matplotlib 画出,题面数字与答案由同一 RNG 流生成,天然可验证、无泄漏、零 API 成本。

| 题型 | 图 | 题目 | 真值程序 | 难度 |
|---|---|---|---|---|
| 图表读数 | 条形图/折线图/饼图 | "第三季度销量比第一季度多多少?" | 生成器自带答案 | 易(RL 起点) |
| 图表统计 | 多序列柱状图 | "哪两个月的合计超过 X?" | 生成器自带答案 | 中 |
| 几何计数 | 三角形/矩形网格图 | "图中有几个三角形?" | 程序计数 | 中 |
| 几何计算 | 带标注角度的几何图 | "∠AOB 的度数?" | 程序计算 | 难 |
| 图像应用 | 商品图与题面分离 | 算术应用题(复用项目一模板,避免第一版把答案文字画进图里) | 项目一真值程序 | 中 |

配比建议 3:3:2:1:1;IID 训练/评估使用独立 RNG 流，另行保留训练未出现的模板和视觉风格作为 OOD，不再只用数字零重叠证明泛化。

**规模修订**:5,000 条可以作为第一轮 pilot 和 RL 核心集，但不能承担通用视觉训练。当前生成器实际只有条形图、折线图和网格图三类固定布局，5,000 条的“有效视觉多样性”低于表面条数。主线建议扩展到 8,000～15,000 条视觉数学 SFT，其中保留 3,000～5,000 条独立题作为 RL；同时增加题型、字体、颜色、布局、问题改写和遮挡扰动。

### 4.2 图文 caption/VQA(通用视觉理解)

| 来源 | 数量 | 处理 |
|---|---|---|
| 明确授权的公开图集/ MiniMind-V 风格 I2T 数据(网络爬取只作补充,phash 去重) | 主线 2～5 万张；先取 2 万张跑通 | 尺寸/格式/来源过滤 → 教师 VLM 生成中文 caption + VQA |
| 教师 VLM:Qwen2.5-VL-7B(AWQ,单卡本地跑,国内可下载)或 DashScope API | — | 生成"图片→问题→答案",自写规则验证(答案非空/长度/与 caption 一致性) |
| 清洗 | — | 复用 proj_1 清洗管线思想:来源登记、去重、语言/长度过滤、人工抽检；phash 不能替代语义去重 |

MiniMind-V 的公开 `sft_i2t` 规模约 290 万条，其中既有图像指令/描述，也混有用于保持语言能力的纯文本样本；项目二不需要照搬全集，建议只抽取 2～5 万条通用图文作为视觉桥接，并按许可和来源做 source-level split。MiniMind-O 的 full pipeline 也把 `sft_i2t` 作为独立视觉阶段，而不是用少量合成任务替代通用视觉数据。

### 4.3 无图数学 SFT 与文本 replay(两者目的不同)

- **无图数学指令 SFT 约 15%**:题面 + CoT/答案，不带图像；它负责让 `pretrain_02b` 先学会数学指令模板、答案格式和基本推理表达，这是“建立”能力，不是 replay。
- **通用文本 replay 约 10%～15%**:通用中文指令、知识和少量项目一数学文本，用于防止多模态训练造成灾难性遗忘；它负责“保持”能力。
- **独立统计**:数据构造和 loss 日志分别记录无图数学、通用 replay、图文通用、视觉数学四类 token 数和 loss，不能把无图数学 SFT 的收益误记成视觉收益。

首版默认 SFT **token** 配比为:通用图文 45%、无图数学 15%、视觉数学 25%、通用文本 replay 15%；必须按 token 统计而不是按 JSONL 行数统计，因为项目一数学 CoT 明显比短 caption 更长。若暂时没有通用图文数据，当前 5,000 条视觉数学 + 无图数学只能命名为“视觉数学原型”，不能宣称通用 VLM。

建议按 MiniMind-O 的训练顺序执行:①通用图文上冻结 LLM 只训 vision projector；②低学习率混合 SFT，让通用图文建立 grounding、无图数学建立模板、视觉数学建立跨模态推理；③只在视觉数学子集上做 GRPO/RL；④用文本保持和通用图文验证集检查遗忘。音频数据不需要引入，复用的是阶段顺序和数据职责，不是整套 Omni 模态。**（⚠️ 已推翻：语音模态后来成为主线——whisper 特征 + AudioProjector + 语音题面数据，见 REPORT.md §1-2。）**

若文本 replay 后仍明显遗忘，优先降低视觉训练学习率或冻结部分 LLM 层，不通过把数学 Agent 权重直接当基座来掩盖问题。

### 4.4 视频问答(stretch,时间紧可砍)

| 来源 | 处理 |
|---|---|
| 有明确许可的公开视频 QA 或自录屏 | 首版 4 帧×64 token；8 帧×64 只做实验 | 先验证多帧是否优于单帧；时间戳 token 只是位置信息，不等于已经具备时序建模能力 |

### 4.5 评估集(与训练完全隔离,自建 + OOD + 外部对照)

- 主评估:视觉数学 500 题、图表问答 500 题；视频评估仅在视频阶段完成后加入 500 题;
- 合成题分成 IID、模板 OOD、风格 OOD 三组。独立 RNG 只用于 IID 防数字重复，不能把“数字零重叠”当成完整防泄漏证明;
- 图表文字/OCR 单独标记为一个 slice；第一版非 OCR 视觉数学题不把文字识别能力混入主指标;
- 增加文本保持集:通用中文 300～500 题 + 数学文本 300～500 题;
- 视觉数学主指标增加**视觉增益**:在关键数值/几何信息只存在于图像的“视觉必要题”上，对同一题使用同一模型、同一 prompt 和同一解码配置，计算 `Δvis = Acc(带图) - Acc(无图同题文本版)`；主验收看 `Δvis` 是否为正且置信区间是否跨过 0，不用项目一的 99.7% 作为视觉模型绝对目标;
- minimind-o 对比必须固定可加载的模型 checkpoint、视觉编码器、processor、prompt、解码参数和最大输出长度；本地只有代码时不得直接宣称已完成公平对比;
- 报告 bootstrap 置信区间，并增加图像置换/置空图像对照，证明答案确实使用了图像。

## 五、代码结构(规划)

```
proj_2/
├── README.md                    # 本文档
├── config.py                    # 路径/模型规模/训练超参
├── data/
│   ├── gen_figure_math.py       # 程序化生成视觉数学题(matplotlib + 真值,独立 RNG 流)
│   ├── gen_chart_qa.py          # 图表问答生成
│   ├── gen_vqa.py               # 教师 VLM caption/VQA 生成(并发+失败重试+断点续写)
│   ├── clean_images.py          # phash 去重 + 尺寸/格式/质量过滤
│   ├── video_frames.py          # 视频帧采样 + 时间戳标注
│   ├── import_i2t_parquet.py    # 服务器端 MiniMind-V 风格 I2T parquet → JSONL/图片
│   ├── build_text_math_sft.py   # 无图数学指令 SFT(题面 + CoT/答案,独立统计)
│   ├── build_sft_jsonl.py       # 组装多模态 SFT jsonl(图像路径 + 对话 + 两类文本数据)
│   └── build_eval.py            # IID/OOD/文本保持评估集构建
├── assets/project1/             # 首次服务器准备时复制的项目一最小必需资产
│   ├── tokenizer/best_mm.json
│   ├── out/pretrain_02b/pretrain_h1024_l24.pth
│   └── data/sft/sft_v4_combined.jsonl
├── model/
│   ├── vision_encoder.py        # SigLIP2 加载(冻结,256×256→64 token,与 minimind-o 同思路)
│   ├── projector.py             # 自研 MM projector(2 层 MLP,隐层 1024)
│   ├── mm_minimind.py           # 0.25B LLM + image_pad 连续槽位注入视觉特征
│   └── mm_template.py           # 图像/视频布局、长度预算、marker 对齐与 assert
├── train/
│   ├── mm_dataset.py            # 多模态数据集(图像懒加载 + 磁盘缓存 + text replay)
│   ├── train_align.py           # 阶段 1:仅训 projector(冻结 LLM + 视觉编码器)
│   ├── train_mm_sft.py          # 阶段 2:混合 SFT(视觉数据 + 15% 无图数学 + 10%~15% 文本 replay)
│   ├── train_mm_grpo.py         # 阶段 3:视觉 GRPO/DAPO(KV cache + vision cache)
│   └── smoke_multimodal.py      # 小模型 text/image marker 与有限值冒烟
├── eval/
│   ├── eval_mm_math.py          # 视觉数学评估(rule reward)
│   ├── eval_vqa.py              # 图文问答评估
│   ├── eval_video_qa.py         # 视频问答评估
│   ├── eval_text_retention.py   # 文本保持与灾难性遗忘评估
│   ├── eval_minimind_o.py       # minimind-o 同题对比
│   └── run_eval_suite.sh        # 4 卡分片评估(复用项目一框架)
├── scripts/
│   ├── prepare_server_data.sh    # 服务器下载/转换通用图文数据
│   ├── run_align.sh / run_mm_sft.sh / run_mm_grpo.sh
│   └── demo.py                  # 单图 CLI 推理入口，可被 Gradio 包装
├── requirements.txt              # 通用运行/训练依赖
├── requirements-server.txt       # 服务器端 hf/pyarrow 数据转换依赖
└── out/                          # 权重/日志/评估证据
```

## 六、实施阶段与时间线(主线 7~8 周,视频另算)

| 阶段 | 内容 | 产出 | 时间 |
|---|---|---|---|
| 1 数据与协议 | 程序化视觉数学 + 授权图集清洗 + text replay + 数据 schema | 数据统计、来源登记、shape/marker 协议 | 1.5~2 周 |
| 2 单图结构 | SigLIP2 冻结 + projector + 64 image tokens；不做视频 | 单图 forward、32 条过拟合、置空图像对照 | 1 周 |
| 3 混合 SFT | projector 对齐 → 视觉数据 + 15% 无图数学 SFT + 10%~15% 通用文本 replay | loss 曲线、视觉 demo、文本保持结果 | 2 周 |
| 4 评测与基线 | IID/OOD/置换图像/文本保持 + 固定 minimind-o checkpoint | 可复现实验表、错误案例 | 1 周 |
| 5 条件视觉 RL | 仅视觉数学 GRPO/DAPO；先完成 KV cache 和 vision cache | acc/reward/长度/KL、RL 前后消融 | 1.5~2 周 |
| 6 视频 stretch | 4 帧优先，验证多帧增益后再尝试 8 帧 | 视频 forward、时序消融、视频评测 | 额外 2~3 周 |

**主线交付是“单图 + 混合 SFT + 可验证视觉 RL + 文本保持评估”，约 7~8 周。视频不再与主线并行，时间不足直接砍掉。**

### 6.1 Stop/Go 门槛

- **Go 1**:单图 32 条样本可以过拟合；置空图像或打乱图像后准确率明显下降，证明模型确实使用视觉输入。
- **Go 2**:混合 SFT 在视觉必要题上产生正的视觉增益 `Δvis = Acc(带图) - Acc(无图同题文本版)`，同时文本保持集没有出现不可接受的灾难性遗忘；不以绝对 acc 对齐项目一的数学 Agent 结果。
- **Go 3**:RL 训练前，视觉数学训练分布上已经有足够的正确/错误混合样本；单 prompt group 能完成一次 finite 的 policy/ref 更新，且无死锁、无 NaN。
- **No-Go**:没有 KV cache、没有视觉 embedding cache、RL group 长期全对或全错时，不进入大规模视觉 RL，改做 SFT 消融和失败归因。

## 七、可行性分析

### 7.1 显存账(4×3090,容量可行但不能忽略激活与时间)

| 组件 | 显存 | 说明 |
|---|---|---|
| 0.25B LLM(bf16) | ~0.5GB | 主干 |
| SigLIP2-base(冻结) | ~0.2GB | 视觉编码器,不反传 |
| 全参数 SFT(权重+梯度+AdamW) | 约 4GB 级基础开销 + 激活 | 0.25B 在 24GB 卡上可行，但 batch/序列长度需实测；建议 gradient checkpointing，不能固定写成 ~2GB/卡 |
| SigLIP2 + projector | 约 0.2~0.5GB 级，取决于 dtype/实现 | 冻结视觉编码器，每卡各放一份即可 |
| 视觉 GRPO rollout | 不是“轻量” | 先用 group=4、max_new_tokens=128~256；必须缓存视觉 embedding 和 KV，否则序列变长后耗时不可控 |

**结论:显存不是第一瓶颈，RL 计算量、I/O 和工程闭环才是。** 4×3090 可以承载主线，但不意味着视频 + 长输出 + 多组 rollout 可以在原项目一时间内完成。

### 7.2 风险与对策

| 风险 | 等级 | 对策 |
|---|---|---|
| 数学专用基座导致语言/视觉分布变窄 | 高 | 主线使用 `pretrain_02b`;数学 Agent 权重只做消融；加入 15% 无图数学 SFT + 10%~15% 通用 text replay，并报告文本遗忘 |
| 0.25B 视觉理解上限(复杂几何推理可能学不会) | 高 | 先做图表读数→图表统计→几何计数；几何计算为 stretch；用图像置换、模板 OOD 和公开子集确认不是背模板 |
| 视觉 RL 不涨或采样太慢 | 高 | 真值程序提供高密度 reward，但仍需先完成 KV/vision cache；低步数、group=4；无效则保留为受控负结果 |
| Go 2 未通过、视觉增益接近 0 | 中 | 先保留“视觉 grounding 边界”负结果；若项目周期允许，再升级到 0.5B 自训预训练基座或可加载的 Qwen3.5-0.8B 级小模型做对照，不把升级路径写成当前主线必交付 |
| 教师 VLM 标注质量/成本 | 中 | 本地模型只作标注器；保留来源、人工抽检、答案一致性和重复样本审计 |
| 视频数据难自建、时序能力不足 | 中 | 视频独立排期；先 4 帧；没有时序增益证据就不纳入主结论 |
| 图像里的文字(OCR 需求) | 中 | 非 OCR 与 OCR slice 分开报告；不要用“题目文字在 prompt”包装成通用图表理解 |

### 7.3 与项目一的衔接

- 主干复用:`out/pretrain_02b` 权重 + `best_mm.json` 词表；不使用 `agent_sft_v3` 作为主线多模态初始化;
- 数学模型只作为消融:`pretrain_02b → MM-SFT` 对比 `agent_sft_v3 → MM-SFT`，观察语言保持、视觉 grounding 和视觉数学迁移;
- RL 代码只复用算法经验，必须重写多模态 batch、视觉特征缓存、KV cache 和 reference log-prob;
- 评估叙事复用项目一的机制归因方法，但增加文本保持、图像置换和模板 OOD，避免只讲“数字零重叠”;
- **王牌图双生**:文本 RL 曲线(项目一)+ 视觉 RL 曲线(本项目)同源同款,面试一页讲完。

### 7.4 差异化定位

项目二的差异化不依赖“某方向文献无人占位”或 SOTA 承诺，而建立在可复现实验:

文献核查可以作为叙事钩子，但不应写成强断言：已核查的代表性工作 [Video-R1](https://arxiv.org/abs/2503.21776) 和 [Video-Thinker](https://arxiv.org/abs/2510.23473) 都以 7B 级模型为主要实验对象；Video-Thinker 使用了 10K 级数据做 SFT+GRPO。因而本项目的 `0.25B + 约 5k 条视觉数学 RL` 可以作为“在相近小数据预算下做规模下探”的定位，但不能预先承诺 SOTA，也不能把所有视频 RL 工作概括为 3k～10k 数据。

- **基座选择对照**:通用预训练基座 vs 数学专用基座，验证窄域能力是否有利于视觉数学迁移;
- **视觉信息是否真正被使用**:图像置空/置换、单帧/多帧和模板 OOD;
- **小模型视觉 RL 的边界**:在真值程序 reward、低步数和缓存优化下，准确率是否上涨；不涨也报告机制归因;
- **能力保持**:视觉能力提升不能以文本能力大幅遗忘为代价。

收尾后可以整理成技术报告，但不把 arXiv、workshop 或 SOTA 作为项目完成条件。

---

## 八、里程碑

- [x] M1(第 1~2 周):数据 schema、image marker/长度预算、视觉数学题（最终 30000+500，比较题型）、语音题面 3000+500 完成
- [x] M2(第 3 周):单图 forward 跑通；过拟合验证；置空/置换图像对照完成
- [x] M3(第 5 周):混合 SFT 完成——旧题型 `Δvis = 0.166`（CI [0.120, 0.212] 下界 > 0）；新题型 SFT v5 `Δvis = 0.395 [0.341, 0.445]`、with_image 0.627，验收通过；文本保持 0.016 无遗忘
- [x] M4(第 6 周):IID/OOD/文本保持评测完成（minimind-o 固定 checkpoint 对比未做——差异化为自建评估协议，放弃外部对比）
- [x] M5(第 7~8 周):条件视觉 RL 完成——GRPO v1-v5 双 bug 归因（REPORT.md §6）；v6/v7 修复后重试仍为**受控负结果**：RL 只提升 without 猜测（0.068→0.152→0.267）、with_image 持平略降，RL 方向正式关闭（REPORT.md §4.3）
- [ ] M6(可选,额外 2~3 周):4 帧视频输入——**已砍掉**，未投入
- [x] M7(收尾):技术报告与 demo 完成——[REPORT.md](REPORT.md) + [demo/out/demo_audio_math_v5.html](demo/out/demo_audio_math_v5.html)

**与规划的偏差**（详见 REPORT.md）：① 语音输入模态加入（规划期排除，后成为主线）；② 视频砍掉；③ 题型从精算改为比较（判别实验证明精确读数超出 0.25B 能力边界），进而用难度梯度题型（paircmp/turns/groupsum）找到能力边界内的可训练区间；④ minimind-o 外部对比未执行；⑤ minimind-o 对照表中"语音输出砍掉"已推翻；⑥ RL 主线以受控负结果收尾（v1-v5 双 bug 归因 + v6/v7 修复重试仍不涨），方向关闭、转题型课程。

## 九、已实现入口与运行顺序

当前目录已开始落地为“图像主线 + 视频/RL 可扩展接口”。建议在项目二目录执行:

```powershell
# 1) 不依赖 PyTorch 的本地数据冒烟：生成、schema 校验、图片清洗
./scripts/run_data_smoke.ps1

# 2) 从项目一数学 SFT 数据中单列无图数学指令数据
./scripts/prepare_project1_data.ps1

# 2.5) 下载已登记、明确许可的数据；不会隐式爬取网页
python -m data.download_dataset --manifest scripts/sources.example.jsonl --out-dir data/raw/public

# 3) 生成正式视觉数学训练/评估数据（先用小数量验证）
python -m data.gen_figure_math --count 5000 --out-dir data/generated/figure_math/images --manifest data/generated/figure_math/train.jsonl --seed 10000 --split train
python -m data.gen_figure_math --count 500 --out-dir data/generated/figure_math/eval_images --manifest data/generated/figure_math/eval.jsonl --seed 20260814 --split eval

# 4) 先准备通用图文 I2T 与 text replay；以下文件是主线必需，只有视觉数学/无图数学时仅算 pilot
#    data/processed/general_i2t.jsonl
#    data/processed/text_replay.jsonl   <- 由 data.build_text_replay 从项目一 SFT 生成
#
# 5) 在具备 CUDA PyTorch、SigLIP2 和项目一权重的 Linux 服务器上组装混合 SFT
#    12.6M token 预算保证 20,000 条 general_i2t 全量装入 45% 配额
#    （sft_v4_combined 中与无图数学重复的题面已被 build_text_replay 去重）
python -m data.build_text_replay --input assets/project1/data/sft/sft_v4_combined.jsonl --text-math data/processed/text_math_sft.jsonl --output data/processed/text_replay.jsonl --limit 12000
python -m data.build_sft_jsonl \
  --visual-math data/generated/figure_math/train.jsonl \
  --general-image data/processed/general_i2t.jsonl \
  --text-math data/processed/text_math_sft.jsonl \
  --text-replay data/processed/text_replay.jsonl \
  --output data/processed/mm_sft.jsonl \
  --max-tokens 12600000 --tokenizer assets/project1/tokenizer/best_mm.json
# 服务端一键版：bash scripts/prepare_server_data.sh（含以上所有步骤）

# 6) 可选：先在单卡训练 projector；PROJECTOR 传给后续 SFT
BASE_MODEL=assets/project1/out/pretrain_02b/pretrain_h1024_l24.pth \
VISION_MODEL=/path/to/siglip2-base-patch16-256 \
bash scripts/run_align.sh

# 7) 4 卡训练；BASE_MODEL/VISION_MODEL 指向实际服务器路径
BASE_MODEL=assets/project1/out/pretrain_02b/pretrain_h1024_l24.pth \
VISION_MODEL=/path/to/siglip2-base-patch16-256 \
PROJECTOR=out/align/projector.pt \
bash scripts/run_mm_sft.sh
```

`run_mm_sft.sh` 是 4 卡 DDP 主线：语言模型由 DDP 同步，外置 projector 的梯度也显式 all-reduce。`run_align.sh` 保持单卡 projector 预对齐；`train_mm_grpo.py` 当前是单卡、可审计的 group-relative RL 基线，明确禁止误用 `torchrun`，后续若需要再扩展 rollout 分片。当前 Windows 开发环境若没有 PyTorch，只能完成数据冒烟和 Python 静态检查；SFT 的 4 卡运行需在 Linux/CUDA 主机执行。

### 9.1 服务器数据准备（不在本地下载大数据）

通用图文数据在 4×3090 服务器下载。默认情况下，原始 parquet、抽取图片、处理后 manifest 以及项目一最小资产都放进 `proj_2`；只有磁盘空间不足时才显式把 `DATA_ROOT` 指到外部数据盘。脚本不依赖当前工作目录:

```bash
cd /path/to/minimind-master/my_minimind/proj_2

# PROJECT1_SOURCE 是旧项目一目录，仅首次运行用于复制 tokenizer/基座/数学 SFT；
# 运行完成后，训练只读取 proj_2/assets/project1
PROJECT1_SOURCE=/data/minimind/proj1 \
GENERAL_I2T_LIMIT=20000 \
SFT_MAX_TOKENS=5000000 \
bash scripts/prepare_server_data.sh
```

脚本会在服务器完成以下操作:

- 将项目一 tokenizer、预训练权重和数学 SFT 复制到 `$PROJECT_ROOT/assets/project1`;
- 使用 `hf download` 下载 MiniMind-V 风格的 `sft_i2t.parquet` 到 `$DATA_ROOT/raw/minimind_v`;
- 只将前 `GENERAL_I2T_LIMIT` 条有效图像转换到 `$DATA_ROOT/images/general_i2t`，过滤黑色文本占位图、坏图和重复图;
- 在 `$DATA_ROOT/processed` 生成 `general_i2t.jsonl`、`text_math_sft.jsonl` 和按 token 配额混合的 `mm_sft.jsonl`;
- manifest 中写入图片绝对路径；如果外部数据 manifest 使用相对路径，训练集、RL 和 smoke 校验会相对于 manifest 所在目录解析。

下载前需在服务器安装 `requirements-server.txt`；如果使用 Hugging Face 私有镜像或需要登录，可先设置 `HF_ENDPOINT`、执行 `hf auth login`，再运行脚本。公开数据的许可证和源数据条款仍需在正式训练前核查。

生成混合 manifest 后，训练命令使用绝对路径，不受 `cd` 位置影响:

```bash
BASE_MODEL=/path/to/proj_2/assets/project1/out/pretrain_02b/pretrain_h1024_l24.pth \
VISION_MODEL=/data/minimind/proj2_models/siglip2-base-patch16-256 \
DATA=/path/to/proj_2/data/server/processed/mm_sft.jsonl \
TOKENIZER=/path/to/proj_2/assets/project1/tokenizer/best_mm.json \
SAVE_DIR=/path/to/proj_2/out/mm_sft \
bash scripts/run_mm_sft.sh
```

> **终态复现命令以 [REPORT.md](REPORT.md) §7 为准**——本节为规划期草案（实际数据为 figure_math_v3/figure_audio_v3，模型为 out/mm_sft_v4）。
