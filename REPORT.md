# proj_2 项目报告：0.25B 多模态语音数学问答

**最终状态**（2026-08-22）。本报告记录项目全部最终数字、架构、负结果与教训；规划期文档见 [README.md](README.md)。

**一句话定位**：「基于项目一预训练 0.25B 基座构建四模态小模型（文本+图像+语音输入+语音输出）：自建程序化可验证视觉数学数据，SigLIP2 与 whisper 双编码器对齐、5 流混合 SFT、GRPO RLVR；用 paired Δvis/Δaudio 协议 + bootstrap CI + 像素哈希防泄露刻画视觉/听觉真实增益，并通过判别实验刻画小模型多模态能力边界。语音数学问答闭环：听中文题面 + 看图表 → 说出答案。」

---

## 1. 最终系统架构

```
语音题面 wav ──► FrozenWhisper-small (d=768, 冻结)
                     │ 预计算特征 [256, 768] npy
                     ▼
               AudioProjector（复用 VisionProjector 类：LN→Linear→GELU→Linear，per-token MLP）
                     │
图表 PNG ────► FrozenSigLIP2-base-patch16-256 (d=768, 冻结, 16×16=256 patch)
                     │
               VisionProjector（同一 MLP 类，帧数无关 → 64 个视觉 token）
                     │
                     ▼
              MMForCausalLM（0.25B, hidden 1024, 24 层, 词表 12014）
                     │ 文本答案（CoT 尾数字）
                     ▼
               ChatTTS（omni-tts env，speed 0.65 / pitch −2）──► 语音答案 wav
```

- prompt 布局：图标记(64) + `<|audio_start|>` + 256×`<|audio_pad|>` + `<|audio_end|>`，**不含问题文字**（Δaudio 协议要求：含文字则模型可忽略音频）
- 1024 max_pos 预算：256 音频 + 64 图像 + 文本 ≈ 370，安全
- 音频行保留图像字段——bar/line 题面数字全在图里，听题必须看图

## 2. 数据管线（自建、可验证、零 API 成本）

- **figure_math_v3**（Pillow 渲染，`data/gen_figure_math.py`）：train 30000 + eval 500。题型（v5 重设计后）：bar/line 为**找最值/比较 4 选 1**（柱值两两不同保证最值唯一；CoT 为相对高度比较链，不出现数值）；grid 为 4/5/6 网格三角形计数。metadata 带真值程序，答案可机械验证
- **figure_audio_v3**（ChatTTS 合成题面）：train 3000 + eval 500，16k mono wav + whisper 预计算特征 npy
- **5 流混合 SFT**（`mm_sft_v4.jsonl`，82727 行）：general_image 0.40 / text_math 0.13 / visual_math 0.22 / text_replay 0.15 / audio_math 0.10；音频行与视觉行同题重叠是跨模态迁移的正向设计
- **评估集**：eval 500 行与 train **像素哈希 0 重叠**（grid 3×3 只有 512 种配置 → 撞图记忆的泄露教训，见 §6）

## 3. 训练历程

| 阶段 | 内容 |
|---|---|
| align | 冻结 LLM+SigLIP，训 VisionProjector |
| align_audio | 冻结 LLM+SigLIP+视觉 projector，训 AudioProjector |
| SFT v4 | 4×3090 3000 步 5 流混合（17 min）——旧题型最终模型 `out/mm_sft_v4/mm_sft.pt` |
| GRPO v1-v5 | 完整 RL 探索线，最终因两个隐藏缺陷放弃（见 §6），RL 收益历史为 +1pp/500 步 |
| SFT v5 | v4 基座 + 新题型 5 流重建（新题型 10057 行 / 总 82727 行）——**最终模型 `out/mm_sft_v5/mm_sft.pt`** |
| GRPO v6 | v4 基座 + 旧题型 RL 400 步（`out/mm_grpo_v6/`）：without 猜测 0.08→0.152，with 终点 0.230 回到 SFT 水平 |
| GRPO v7 | v5 基座 + 新题型 RL 400 步（turns 4000 + groupsum 2000，`out/mm_grpo_v7/`）：Δvis 0.347→0.335，**未超 SFT v5 基线 0.395** |
| SFT v6 | v5 同款 5 流重建，音频流换成旧 3000 + **新题型 6000 行**（新题型 TTS 备料 + whisper 特征，总 88727 行，28 min）——**最终模型 `out/mm_sft_v6/mm_sft.pt`** |
| SFT v7 | v6 混合 + **13k 操作分解课程行**（序数桥接 + sum dominance，总 101727 行，17 min，`out/mm_sft_v7/`）——课程 CoT 被学成话语形式而非接地计算，**探针 Tier 1 未解锁，0.25B 硬上限坐实**（§4.6） |

## 4. 最终数字（干净评估，500/501 行 paired bootstrap CI）

| 指标 | 旧题型（SFT v4，最终） | **新题型（SFT v5，最终）** |
|---|---|---|
| Δvis = Acc(带图) − Acc(无图) | 0.166 [0.120, 0.212]（旧 eval） | **0.395 [0.341, 0.445]**（新 eval） |
| with_image | 0.234（without 0.068） | **0.627**（without 0.232） |
| 旧 eval 复测（v5 模型） | — | 0.232 / 0.132 / Δvis 0.100 |
| Δaudio = Acc(听题+图) − Acc(读题+图) | −0.018 [−0.06, 0.02] | **−0.010**（v5 模型，旧题型音频） |
| 文本保持（held-out 500） | 0.008 | 0.016（持平，无遗忘） |

**结论**：① 在 0.25B 能力边界内存在可训练题型（§4.1）：v5 新题型把 Δvis 拉到 0.395、with_image 0.627——paircmp 满血、turns 过半，证明"比较两柱"类任务在模型容量内；② 旧题型复测 without 0.068→0.132（新题型训练教了模型"更会猜"），with 不变——旧题型 Δvis 收窄是新题型转移的副产品，不是旧题型本身学会了；③ 文本保持无退化。

### 4.1 新题型难度梯度（SFT v5，n=501，with_image）

| 类型 | 题目 | acc（with / without） | 对照 |
|---|---|---|---|
| bar_paircmp | 指定两柱比高 | **1.000 / 0.072** | 4 选 1 随机 = 0.25 |
| line_turns | 相邻下降段计数（答案 0-3） | **0.605 / 0.431** | 随机 0.25 |
| bar_groupsum | 两组柱和比较 | 0.275 / 0.192 | ≈ 随机 |
| bar-argmax / bar-argmin（旧） | 四柱找最值 | 0.270 / 0.256 | ≈ 随机 |

**难度梯度** paircmp 1.000 > turns 0.605 > groupsum 0.275 ≈ 旧 argmax/argmin——模型能精确比两柱、能数下降段，但"和比较"与"四选一最值"仍超界。gap≥10 受控探针（值两两间距 ≥10，n=500）：argmax 0.259（v4）/ 0.239（v5）、argmin 0.237/0.237——**间距不是瓶颈**，比较本身超界。without 臂里 RL 最先教会模型的就是"猜"（0.068→0.152→0.267，见 §4.3）。

### 4.2 能力边界判别实验（四联，换题型的依据）

| 实验 | 结果 |
|---|---|
| readable 渲染（14px 刻度+网格线） | 无改善 → 不是渲染清晰度问题 |
| 64→128 token 零训练 | 更差（分布外）→ 不是压缩率问题 |
| 256 token 全保真 | 精确读数 exact=0 → 不是信息丢失问题 |
| 步进 10 粗粒度读数 | 8.4% < 随机 25% → 精确读数超能力边界 |

**结论：精确数值读数超出 0.25B + SigLIP2-256 的感知能力边界**（不是管道缺陷）。这是本项目最重要的科学结论之一，直接驱动了题型重设计。

### 4.3 GRPO RLVR 两连：v6（旧题型）+ v7（新题型）

| 轮换（with / without / Δvis） | step100 | step200 | step300 | step400 |
|---|---|---|---|---|
| GRPO v6（SFT v4 基座，旧题型） | 0.192 / 0.080 / 0.112 | 0.192 / 0.152 / 0.040 | 0.222 / 0.152 / 0.070 | 0.230 / 0.152 / **0.078** |
| GRPO v7（SFT v5 基座，新题型） | 0.607 / 0.259 / 0.347 | （被 uniform-skip 吞） | 0.599 / 0.253 / 0.345 | 0.603 / 0.267 / **0.335** |

- **v6**：旧题型接近随机，RL 把 without 猜测从 0.08 拉到 0.152（平滑奖励只奖励"输出更像答案"的分布），with 终点 0.230 ≈ SFT v4 基线 0.234——RL 没恢复任何视觉能力。终点 Δaudio −0.026（音频臂 0.204，SFT 基线 0.222）——旧题型 RL 对听题也无效。
- **v7**：新题型可训练（SFT 已到 0.627），但 400 步 RL 后 with 0.603 持平略降、without 0.232→0.267——**RL 依然只提升猜测**。分类型终点：turns with 0.605→0.533（唯一受损）、groupsum without 0.192→0.299。文本保持 0.016 无退化。

**结论**：GRPO RLVR 在可训练与不可训练题型上都不能替代 SFT 已学到的视觉技能；RL 梯度最先落点在"输出更像答案"而非"读得更对"。v6+v7 双重确认后，RL 方向正式关闭（§8）。

### 4.4 语音模态：Δaudio 分解 + 零样本听题迁移

| 实验（旧题型，n=500） | audio | text | audio+text |
|---|---|---|---|
| Δaudio 分解臂（SFT v4） | 0.216 | 0.234 | **0.262**（+0.028 vs text，CI [−0.026, 0.082]） |

audio+text 臂最高——whisper 特征确实携带题面信息（听+读 > 只读），Δaudio 的差距来自单模态承载题面的难度（听题更费认知），不是对齐失败。

| 零样本听题迁移（新题型，n=501） | audio+图 | text+图 | Δ |
|---|---|---|---|
| SFT v5（音频只训过旧题型） | **0.164** | 0.627 | **−0.463 [−0.517, −0.411]** |
| GRPO v7 终点（RL 数据无音频） | 0.164 | 0.603 | −0.439 |

新题型听题精度 0.164 **低于随机 0.25**——模型在只见过旧题型语音的 AudioProjector 上听不懂新题型题面（系统性地选错），而读题 0.627。**结论：跨模态的题型泛化不存在**——AudioProjector 学到的是具体题型模式而非"听题"能力；音频模态每加一种题型都要进 SFT 音频流。

### 4.5 SFT v6：新题型音频流 + 题面形式探针矩阵（2026-08-23）

**v6 = v5 同款 5 流 + 音频流换入新旧合并 9000 行（新题型 6000 行 TTS 备料）。全量数字：**

| 指标 | v5 | **v6** |
|---|---|---|
| Δvis 新题型 | 0.395 [0.341, 0.445] | **0.413 [0.357, 0.471]**（with 0.583 / without 0.170） |
| Δvis 旧 v3 | 0.100 [0.052, 0.154] | 0.072 [0.024, 0.122]（with 0.206） |
| Δaudio 新题型（n=500） | 0.164（零样本） | **0.328**；Δaudio = −0.254 [−0.302, −0.204]（text 臂 0.582） |
| 文本保持 | 0.016 | 0.008（该指标自 v5 起即退化，两者同区间） |

- **新题型音频有效但远不足**：听题 0.164→0.328（越过 chance 0.25），6000 音频行把"零样本听不懂"修复到"听得半懂"，仍比读题低 0.25——whisper 特征里长题面（groupsum 选项列表）仍是瓶颈，且音频行只占混合 ~7%。
- Δvis 新题型微涨（CI 上移），旧题型小回退——音频权重上升的代价，可接受。

**题面形式探针矩阵（12+3 格 × 200 行，零训练 with-image；现为永久 CI 能力测试，v5/v6 对比）：**

| 格 | v5 | v6 | 格 | v5 | v6 |
|---|---|---|---|---|---|
| n2_named | 1.000 | 1.000 | n4_2nd | 0.250 | 0.235 |
| n3_named | 1.000 | 1.000 | n4_groupsum | 0.270 | 0.225 |
| n4_named | 1.000 | 1.000 | n4_sum2 | 0.030 | 0.515 |
| n2_argmax | 0.505 | 0.505 | n4_sum2v1 | 0.000 | 0.055 |
| n3_argmax | 0.330 | 0.330 | n4_sum2tall | 0.235 | 0.340 |
| n4_argmax | 0.295 | 0.295 | n2_whichbar | 0.000 | 0.505 |
| n4_argmin | 0.255 | 0.255 | n4_whichbar_max | 0.215 | 0.245 |
| n4_whichbar_min | 0.320 | 0.215 | | | |

**核心发现（定位失败层级）**：
1. **named 点名比较族 n=2/3/4 全 1.000**——模型看柱做单次比较的能力完美；
2. **argmax 族全 chance，连 n=2 也只有 0.505**——"第几个季度最高/最低"的序数指称绑定就是瓶颈，~3 万行 argmax 训练也没建立；
3. **消融**："哪根柱最高"（语言更简单）n=2 只有 **0.000**——200/200 条全部背诵同一段 4 柱训练模板（"…第3季度的柱最低，所以答案是3"），问"最高"答"最低"。**语言改写路线否决：失败在"无点名锚的最值操作"本身**，不在中文序数语言；
4. **sum 类整体模板坍缩**（v5 的 n4_sum2 0.030、185/200 末位"3"）；v6 训练后坍缩被稀释（n4_sum2 0.515、n2_whichbar 0.505）但回到的是 **chance 而非能力**——模板先验被冲淡，操作仍未学会。

**结论：模型学到的是窄形式→模板映射，不是"读柱→比较"的通用技能**。技能锁死在训练见过的表面形式里；任何无点名锚的最值/求和题面超分布即退回背诵。课程方向 = **操作分解**（named 比较 + 显式转换的桥接 CoT），13k 课程数据已备（§8）。

### 4.6 SFT v7：操作分解课程 + 结论性负结果（2026-08-23，项目收官）

**v7 = v6 混合 88727 行 + 13k 课程行（序数桥接 argmax/argmin/2nd n=2→4 递进 + sum dominance 子集，CoT = 逐对 named 比较 + 显式转换）。全量数字：**

| 指标 | v6 | **v7** |
|---|---|---|
| Δvis 新题型 | 0.413 [0.357, 0.471]（with 0.583 / without 0.170） | **0.305 [0.257, 0.355]**（with **0.615** / without **0.309**） |
| Δvis 旧 v3 | 0.072（with 0.206） | 0.044（with 0.222） |
| Δaudio 新题型 | 听 0.328 / 读 0.582，Δ = −0.254 | 听 0.358 / 读 0.614，Δ = −0.256 [−0.306, −0.204] |
| 文本保持 | 0.008 | 0.008（无退化） |

- with_image 新题型 0.583→**0.615**（课程行对整体视觉数学有 +3pp 正贡献），但 without 0.170→0.309——课程 CoT 的答案格式先验泄入纯文本臂（"更会猜"），Δvis 反而收窄。两臂均如实报告。
- Δaudio 双臂各 +3pp（听 0.358 / 读 0.614），差距 −0.256 与 v6 持平。

**课程 holdout（IID，1400 行，训练同分布未见值，with-image）——全部 chance：**

| 形式（训练各 1500-2000 行） | holdout | chance |
|---|---|---|
| argmax_bridge_2 | 0.500 | 0.50 |
| argmax_bridge_3 | 0.285 | 0.33 |
| argmax_bridge_4 | 0.255 | 0.25 |
| argmin_bridge_4 | 0.250 | 0.25 |
| 2nd_bridge | 0.250 | 0.25 |
| sum2_dom | 0.550 | 0.50 |
| sum2tall_dom | 0.485 | 0.50 |

**15 格探针 CI（v5/v6/v7 三列）——决定性读数：**

| 格 | v5 | v6 | v7 | 格 | v5 | v6 | v7 |
|---|---|---|---|---|---|---|---|
| n2_named | 1.000 | 1.000 | **0.990** | n4_2nd | 0.250 | 0.235 | **0.255** |
| n3_named | 1.000 | 1.000 | **1.000** | n4_groupsum | 0.270 | 0.225 | **0.295** |
| n4_named | 1.000 | 1.000 | **1.000** | n4_sum2 | 0.030 | 0.515 | **0.485** |
| n2_argmax | 0.505 | 0.505 | **0.495** | n4_sum2v1 | 0.000 | 0.055 | **0.280** |
| n3_argmax | 0.330 | 0.330 | **0.330** | n4_sum2tall | 0.235 | 0.340 | **0.525** |
| n4_argmax | 0.295 | 0.295 | **0.295** | n2_whichbar | 0.000 | 0.505 | **0.495** |
| n4_argmin | 0.255 | 0.255 | **0.255** | n4_whichbar_max | 0.215 | 0.245 | **0.280** |
| n4_whichbar_min | 0.320 | 0.215 | **0.290** | | | | |

**核心发现——课程 CoT 被学成了话语形式，不是接地计算：**

1. **序数桥接 Tier 1 未解锁**（n2/n3/n4_argmax、n4_2nd 全 chance，与 v5/v6 无差）。13k 行、每形式 1500-2000 行、CoT 逐对分解——训完连 IID holdout 都全 chance。
2. **逐字证据**：n4_argmax 探针 **200/200 条全部吐同一条记忆链**（"第4季度的柱比第2季度的柱低；第2季度的柱比第3季度的柱低；第3季度的柱比第1季度的柱低"），首断言与真值链一致仅 **18/200**（低于瞎猜方向）。模型输出结构完美、措辞流利的 CoT，但内容与图无关——**"比较→推理→转换"整个链条被学成一个不可拆解的言语模式，内部的比较步骤从未接地到图像**。
3. **瓶颈最终定位**：named 族 1.000（题目点名比哪两根，模型就会看柱比较）vs n2_argmax 0.495（n=2 时唯一可比的组合就是 1 vs 2，语义上与 named 全等，仍 chance）——**模型无法把"最高"这种无点名锚的题面翻译成"去比较哪一对"的操作选择**。瓶颈不在比较本身、不在序数绑定、不在链式长度，而在"题面形式 → 操作选择"的映射。
4. **n4_sum2tall 0.340→0.525 是模板残留不是能力**：对应课程形式 sum2tall_dom 的 IID holdout 只有 0.485（chance 0.50）——探针上抬是模板先验漂移，无操作。

**结论（项目收官判读）**：0.25B + SigLIP2-256 在"无点名锚的最值/求和操作"上是**硬上限**——语言改写（消融探针）、模板（v5）、稀释（v6）、CoT 操作分解（v7）四条路线全部失败，且 v7 给出了机制级解释：模型学得会"比较的话语"（named 族满血），但无法把话语中的操作步骤接地到视觉输入，多步操作退化为背诵整段文本。**打破该上限的唯一明确方向是放大模型（§8-1，已获 §4.7 外部验证）**；项目其余成果（Δvis 0.615 with_image 难度梯度、Δaudio 听题 0.358 越过 chance、语音闭环 demo）不受影响。

### 4.7 外部基线对比：同规模模型 + 3B 参考（2026-08-23）

三外部基线零样本跑同一 15 格探针 + 新题型 Δvis（同 prompt/图/last-num scorer/greedy；我们 v6/v7 为训练后，公平性注明见下）：

| 格 | 我们 v6 | 我们 v7 | minimind-3v-moe | SmolVLM-256M | Qwen2.5-VL-3B | chance |
|---|---|---|---|---|---|---|
| n2_argmax | 0.505 | 0.495 | 0.010 | 0.490 | **1.000** | 0.50 |
| n2_named | **1.000** | 0.990 | 0.130 | 0.185 | 0.990 | 0.50 |
| n3_argmax | 0.330 | 0.330 | 0.000 | 0.480 | **1.000** | 0.33 |
| n3_named | **1.000** | **1.000** | 0.110 | 0.075 | 0.735 | 0.33 |
| n4_argmax | 0.295 | 0.295 | 0.015 | 0.370 | **0.995** | 0.25 |
| n4_argmin | 0.255 | 0.255 | 0.000 | 0.205 | **1.000** | 0.25 |
| n4_named | **1.000** | **1.000** | 0.090 | 0.235 | 0.685 | 0.25 |
| n4_2nd | 0.235 | 0.255 | 0.010 | 0.215 | **0.460** | 0.25 |
| n4_sum2 | 0.515 | 0.485 | 0.100 | 0.005 | 0.485 | 0.50 |
| n4_sum2v1 | 0.055 | 0.280 | 0.025 | 0.000 | **0.945** | 0.50 |
| n4_groupsum | 0.225 | 0.295 | 0.170 | 0.225 | **0.920** | 0.25 |
| n4_sum2tall | 0.340 | 0.525 | 0.025 | 0.000 | 0.425 | 0.50 |
| n2_whichbar_max | 0.505 | 0.495 | 0.170 | 0.505 | **1.000** | 0.50 |
| n4_whichbar_max | 0.245 | 0.280 | 0.055 | 0.255 | **0.995** | 0.25 |
| n4_whichbar_min | 0.215 | 0.290 | 0.040 | 0.205 | **0.980** | 0.25 |

| 新题型 Δvis（paired，501） | with | without | Δvis |
|---|---|---|---|
| 我们 v6 | 0.583 | 0.170 | 0.413 [0.357, 0.473] |
| 我们 v7 | **0.615** | 0.309 | 0.305 [0.257, 0.355] |
| minimind-3v-moe（0.29B） | 0.080 | 0.036 | 0.044 [0.016, 0.072] |
| SmolVLM-256M | 0.160 | 0.132 | 0.028 [−0.012, 0.074] |
| Qwen2.5-VL-3B | **0.685** | 0.128 | **0.557** [0.513, 0.601] |

**结论：**
1. **"无点名锚最值"不是所有模型的共性天花板，而是 0.25B 级的硬上限——放大即解锁**。Qwen2.5-VL-3B（12× 规模）零样本在同 prompt 同图上把 argmax/argmin/whichbar 打到 0.980-1.000，n4_2nd 0.460、sum2v1 0.945、groupsum 0.920；而三个 ≤0.3B 模型（我们、minimind、SmolVLM）无一越过 chance。§4.6 的四路线失败 + 3B 零样本成功 = 放大模型路线的外部双重验证。注意 3B 也非全能：n4_sum2/n4_sum2tall 两格仍 chance（成对和比较对 3B 也难）、n3/n4_named 反而不如我们（0.685-0.735 vs 1.000，窄题面专用 SFT 胜在 trained-form 内）。
2. **README 声称的对标差异化首次获得数字支撑**：minimind-3v-moe 在所有图表数学形式上远落后于我们（named 0.090-0.130 vs 1.000、with_image 0.080 vs 0.583、Δvis 0.044 vs 0.413）——它的 ALLaVA-4V 描述式训练不产生任何读图比较能力；SmolVLM-256M 的 Δvis 0.028≈0（通用描述 VLM 在图上看不到题面信息），我们 0.413/0.305 的 Δvis 明确来自图表数学 SFT 而非通用 VLM 自带能力。
3. **公平性注明**：我们 v6/v7 见过这些题面形式（训练分布内），三基线是零样本；本表判读的是"各模型在相同形式下的可达水平"，不是"谁的学习效率更高"。零样本 vs 训练后不构成对基线的贬低——它恰好回答"该形式在什么规模上可解"。

## 5. 语音输出闭环

`data/gen_answer_tts.py`（omni-tts env）：模型预测文本 → 折叠退化循环 → ChatTTS 合成语音答案（与题面同 voice 参数），支持 full（朗读完整 CoT）/ final（只读结论句）模式。

**演示**：[demo/out/demo_audio_math_v5.html](demo/out/demo_audio_math_v5.html)（7.6MB 自包含，浏览器直接打开）——8 样例闭环展示：语音题面可播放 → whisper 转写对照 → 模型听题看图 → 文字答案 → 🔊 语音答案可播放 → 对错判定（音频臂 7/8 正确，含 1 个设计内错例）。

## 6. 负结果与教训（项目最有价值的部分）

1. **eval 撞图泄露**：grid 2×2 仅 16 种配置、3×3 仅 512 种 → eval 与 train 必然撞图，模型纯记忆作答（假 Δvis 0.224 → 真 0.036）。修复：4/5/6 网格（配置空间 65536/33M/68B）+ 像素哈希去重。**教训：合成数据评估必须做配置空间检查，数值零重叠≠防泄露。**
2. **GRPO uniform resample 放大**（v5 放弃根因）：4 值答案空间 + group 16 → uniform 组常态化 → resample ×5 → 每步 rollout 5×16×160 tokens ≈ 14min/步。**教训：答案空间越小，resample 放大越严重；题型设计与 GRPO 机制要一起设计。**
3. **smooth_reward 对 CoT gold 全零**（隐藏 bug）：旧版 `float(gold)` 遇 CoT 文本抛异常静默返回 0.0——GRPO v5 即使不撞问题 2，奖励也全是零（v4 时代 gold 为纯数字从未暴露）。已修复：gold 走与 pred 相同的尾数字提取。**教训：奖励函数要对答案字段格式变更做回归测试。**
4. **wav[::3] 下采样 bug**：ChatTTS 24k 写 16k 时取每第 3 个样本 = 2x 加速（正确比例 1.5）。用户三次"语速太快"投诉的根因。修复：torchaudio.functional.resample。
5. **GRPO projector 梯度丢失**：rollout 里 policy projector 投影必须在 no_grad 外（token_logprobs 复用特征回传梯度），只有 reference 包 no_grad——否则 projector 白训练。验证方法：对比 checkpoint 前后各子模块权重 Δ。
6. **模板行为**：模型在精算题型上输出固定 CoT 模板（连算术都错），167 行仅 11 种预测——小模型失败模式以模板化呈现，评估要看预测多样性。
7. **编排教训**：运行中的 bash 脚本绝不能 Edit（逐块读文件错位）；pkill 模式不能与自身命令行相同；env 变量传子进程要 export。
8. **GRPO uniform-skip 吞 checkpoint**（v7 step200/400 消失根因）：保存检查写在 uniform 组 `continue` 之后——step 边界恰逢 uniform 组就永远不写 checkpoint，轮换脚本死等。修复：uniform 分支内也做边界保存。**教训：循环里的副作用（保存/日志）不能放在任何 `continue` 之后。**
9. **gen_audio_tts 硬编码 `../figure_math/` 前缀**：v5 目录改名 figure_math_v5 后，音频 manifest 的 image 字段指向不存在的旧目录，--audio 臂全部 FileNotFoundError（重跑才发现）。修复：前缀从输入 manifest 目录名动态推导。**教训：跨目录 manifest 重写必须用相对路径动态推导，且改目录名后要有下游回归测试。**

## 7. 复现指南

```bash
export PATH=/media/data2/tangzc/miniconda3/envs/omni/bin:$PATH   # 训练/评估 env
export VISION_MODEL=model/siglip2-base-patch16-256
export TOKENIZER=assets/project1/tokenizer/best_mm.json

# 数据（v5 新题型：paircmp / turns / groupsum / gap 探针；旧题型为 v3 比较题）
python -m data.gen_figure_math --count 15000 --out-dir data/generated/figure_math_v5 \
  --manifest data/generated/figure_math_v5/train.jsonl --seed 989898 --split train
python -m data.gen_figure_math --count 500 ... --seed 919191 --split eval \
  --dedup-against data/generated/figure_math_v5/train.jsonl

# 语音题面（omni-tts env）+ 特征（omni env）
#   data/gen_audio_tts.py → data/precompute_audio_features.py（4 卡分片）

# 混合 SFT（v5：新题型）
python -m data.build_sft_jsonl --visual-math data/generated/figure_math_v5/train.jsonl \
  --general-image data/server/processed/general_i2t.jsonl \
  --text-math data/processed/text_math_sft.jsonl --text-replay data/processed/text_replay.jsonl \
  --audio-math data/generated/figure_audio_v3/train.jsonl \
  --output data/processed/mm_sft_v5.jsonl --max-rows 136364 --seed 42

# 训练（4 卡）
BASE_MODEL=assets/project1/out/pretrain_02b/pretrain_h1024_l24.pth \
PROJECTOR=out/align/projector.pt AUDIO_PROJECTOR=out/align_audio_v2/audio_projector.pt \
DATA=data/processed/mm_sft_v5.jsonl SAVE_DIR=out/mm_sft_v5 STEPS=3000 \
bash scripts/run_mm_sft.sh

# GRPO（v6/v7 已跑完并结论：RL 不超 SFT；smooth_reward + resample-attempts=1 + cosine LR）
CUDA_VISIBLE_DEVICES=0 python -m train.train_mm_grpo --data .../grpo_v7_newtypes.jsonl \
  --checkpoint out/mm_sft_v5/mm_sft.pt --save-dir out/mm_grpo_v7 \
  --steps 400 --group-size 8 --max-new-tokens 128 --temperature 1.0 --top-k 50 \
  --lr 1e-6 --kl-beta 0.01 --warmup-steps 50 --save-interval 100 --resample-attempts 1

# 评估：scripts/run_eval_suite.sh（Δvis）+ scripts/run_audio_eval.sh（Δaudio）
#   + eval.predict_mm 4 卡分片文本保持；全部 GPU_BASE=4

# 闭环 demo
python demo/demo_audio_math.py --rows 8                # omni env 推理
python data/gen_answer_tts.py --input demo/out/demo_audio_math_v5.json --field pred_audio  # omni-tts env
python demo/demo_audio_math.py --reuse-json           # 轻量渲染，任意 python
```

## 8. 未来方向（按性价比排序）

1. **模型放大 0.5B/1B**：§4.6 收官判读后**唯一明确**能打破"无点名锚最值/求和"能力边界的方向——四条路线（语言改写/模板/稀释/CoT 操作分解）在 0.25B 全部失败，瓶颈机制已定位在"题面形式→操作选择"映射的接地。8×3090 可行；15 格探针矩阵是现成的跨规模能力判读 CI。
2. **音频流加深**：v6/v7 把新题型听题拉到 0.328/0.358（>chance）但仍低读题 ~0.25——提高 audio_math 占比（0.10→0.15+）、或把 groupsum 长选项题面拆分/简化后再 TTS
3. **数据侧改进**：柱值强制最小间距（gap 探针已证明无用，可跳过）、更多题型多样性——边际收益，受 0.25B 容量限制
4. **Mimi 编解码器**：把"文字→ChatTTS"桥接换成 LLM 直接输出音频 token（Moshi/GLM-4-Voice 路线）。只适合项目定位转向"原生语音对话"时投入；对数学精度零帮助
