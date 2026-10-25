# EAQuant — arXiv:2506.13329

> MoE 做**权重 + 激活**量化（W4A4 及更低）时有三个 dense 模型没有的麻烦：
> (1) SmoothQuant 式通道缩放得每个专家一套，但前面的 RMSNorm 只能吸收一个向量；
> (2) router 对量化扰动极敏感，logit 稍偏 top-k 就换人；(3) 冷门专家分到的校准 token 太少。
> EAQuant 在 DuQuant 上加三件事：**EA-SA** 把各专家和 router 的平滑系数逐通道取最大值合成一个向量并进 RMSNorm（零开销）；
> **EA-RCA** 校准 router 自身的量化参数时，目标 = logit MSE + 只在 top-k 专家上算的 KL（KL-Top）；
> **EA-CDB** 统计每个专家分到的 token，低于平均值 2 倍的就从训练数据里补抽，直到够数。

本文件是这篇 paper 的**解读总入口**。radar-native 分析在下方，各角度详读见导航。

## 论文

- **标题**：EAQuant: Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization
- **链接**：[abs](https://arxiv.org/abs/2506.13329) · [PDF](https://arxiv.org/pdf/2506.13329v3) · [code](https://github.com/darren-fzq1/EAQuant)
- **作者**：Zhongqian Fu, Tianyi Zhao, Ning Ding, Xianzhi Yu, Xiaosong Li, Yehui Tang, Yunhe Wang
- **发表**：2025-06-16（v1）；本解读基于 **v3** · cs.CL
- **许可**：arXiv 非独占许可（非 CC-BY）——中文翻译只做逐节复述，不是逐句翻译

## 解读导航

| 角度 | 文件 | 内容 |
|---|---|---|
| 原理故事 | [paper.org](paper.org) | 七拍故事版 |
| 中文伴读 | [reading.org](reading.org) | 选择性精读 |
| 中文翻译 | [translation_zh.md](translation_zh.md) | 全文中文复述（逐节释义，含公式/表/图） |
| 溯源倒读 | [../../paper-river/EAQuant-2506.13329.org](../../paper-river/EAQuant-2506.13329.org)（中）· [_en](../../paper-river/EAQuant-2506.13329_en.org)（英） | 本次新生成 |

数据来源：radar 无打分记录（见下）；解读基于 arXiv HTML v3 全文。

## Radar 记录

- **未收录**：2025-06 发表，早于 radar 归档起点（2025-11），没有 composite / bucket / 摘要。
- **未 triage**。按内容应落在 `ptq` 桶。
- [VSRAQ (2606.05688)](../VSRAQ-2606.05688/README.md) 把它列为最近的前辈，但**没有跑它做基线**。

## 同类对比 / novelty

| 论文 | 关系 |
|---|---|
| DuQuant | **宿主框架**，所有主表都是"DuQuant vs DuQuant + EAQuant 三件套"；没有与 QuaRot / SpinQuant / FlatQuant 并排 |
| SmoothQuant | EA-SA 的出发点；新东西只是"多专家 + router 的系数取逐通道最大值"这个聚合 |
| [MoEQuant (2505.03804)](../MoEQuant-2505.03804/README.md) | **直接对手**。两者都处理冷门专家校准不足：MoEQuant 让模型自生成校准文本，EAQuant 认为这让对比不公平，改从训练数据补抽——但补抽同样让 EAQuant 用了比基线更多的真实数据，"公平性"批评对自己也成立 |
| EAC-MoE（ACL 2025） | 同期的 router 校准工作（TopK-MSE），EAQuant 未对比 |
| OstQuant（Hu et al. 2025b） | KL-Top 损失的来源，EAQuant 直接借用 |
| [VSRAQ (2606.05688)](../VSRAQ-2606.05688/README.md) | 后继的路由对齐线。VSRAQ 批评 EAQuant"全维度对齐会浪费在低排名专家上"——但 **v3 的 KL-Top 默认 α=0，只在 top-k 上算**，这条批评对 v3 不成立；v1 的公式写的是全专家 KL，但 v1 的消融已经选了只算 top-k，所以批评只对得上 v1 的公式写法或全专家计算的 logit MSE 项（溯源 agent 核对 v1/v3） |

判断：**三件套里只有 EA-SA 是有分量的新东西**（解决"多专家平滑系数无法并进 RMSNorm"这个真实工程问题）；
EA-RCA 和 EA-CDB 都是已有想法（router 对齐、专家均衡校准）的工程化组合。消融也支持这个判断（见下）。

## 实验与工程可落地性

- **设置**：DuQuant 基线；per-channel 权重 + per-token 激活，非对称，**无分组**；**router 层统一 W8A8**；校准 WikiText2 128 段；模型 OLMoE-7B、DeepSeek-MoE-16B、Mixtral-8x7B。评估 PPL + 5 个零样本选择题（ARC-C/E、BoolQ、PIQA、WinoGrande），**没有生成式 / 推理任务**。
- **W4A4 平均准确率**：OLMoE 66.69 → 68.06（FP 70.28）；DeepSeek 65.87 → 67.02（FP 69.44）；Mixtral 73.06 → 74.21（FP 76.58），提升 1.15–1.37。
- **更低比特**：W3A3 Mixtral 53.05 → 62.46；W2A4 是 DuQuant 崩溃（PPL 7000+）对比 EAQuant"没崩"——但 EAQuant W2A4 的 PPL 仍是 21.9（Mixtral）/ 78.6（OLMoE）/ 138（DeepSeek），实际不可用。摘要的"+13.81%"就来自 Mixtral W2A4 这一格。
- **消融（W3A4）**：DeepSeek 上 EA-RCA 单独 +0.40、EA-CDB 单独 +0.11，三件全上 +1.33；OLMoE 上 EA-SA 单独 +2.12，全上 +2.28。**收益基本来自 EA-SA**。
  EA-SA 的聚合算子：正文说 `maximum` 最好（OLMoE +1.20），但 DeepSeek 上 `router_logits`（+1.05）和 Mixtral 上 `expert_frequency`（+0.92）都比 `maximum`（+0.42 / +0.46）好，正文结论只对 OLMoE 成立。
- **EA-RCA 只校准 router 自己的权重量化参数**：式 10 的 MSE 和 KL 都用同一个输入 `x̃` 算，所以它只管 router 权重量化误差，**不管上游层量化导致的 router 输入偏移**（后者才是 VSRAQ 想解决的）；而主实验里 router 是 W8A8，这部分误差本来就小。
- **与 MoEQuant 的对比（表 7）不是同口径**：MoEQuant 的数字与 MoEQuant 原文表格完全一致（6.94 / 9.32、5100 / 4924、4.12 / 7.34…），应为直接抄录；而两篇的 FP 基线不同（DeepSeek C4：EAQuant 9.10 vs MoEQuant 9.04），评估设置不一致。只比 PPL，不比任务准确率。
- **EA-CDB 补抽的"训练数据"是哪个数据集，正文没说**。在 SA+RCA 之上再加 CDB 只多 +0.09（OLMoE）/ +0.27（DeepSeek），"三者互补"的说法偏弱。
- **表 2 有笔误**：OLMoE W3A4 "只加 EA-RCA" 一行标 +1.90，但平均 64.20 − 63.30 = +0.90（我已核对，表 4 也写 +0.90）。主表用的是哪种聚合算子，论文没说。
- 翻译 agent 还核出：表 6 说 56/64 个专家误差降低，按附录表 12 逐个数是 61/64；表 7 的 2-bit 对比是 EAQuant 2.0 bit 对 MCMoE 2.2 / 2.05 bit（我没逐项复算）。
- **推理开销**：EA-SA 并进 RMSNorm、EA-RCA/CDB 只在校准阶段，推理零额外开销；但 W4A4 的实际加速需要 INT4 激活 kernel，论文**没有任何速度 / 显存测量**。
- **与 AMD / Quark**：EA-SA 是最值得借鉴的一招——任何 SmoothQuant 类流程在 MoE 上都会遇到"每专家系数没法融合"的问题，取 max 聚合并进 norm 是零成本方案；在 FP8 / MXFP4 激活量化上是否同样有效，论文没测。

## 趋势定位

属于 2025 年中 MoE PTQ 起步期的一批（与 MoEQuant、MoQa、MC-MoE 同期），是其中**少数做激活量化**的。
后续 radar 里的 MoE 工作大多转向权重量化 + 位宽分配 / 路由对齐，W-A 低比特方向跟进较少（印象判断，未系统统计）。值得读 §2.2（EA-SA）；EA-RCA 的设计后来被 VSRAQ 一类工作取代。

## Triage 结论

未 triage（radar 未收录）。若入 seed：理由是"MoE 激活量化下多专家平滑系数融合的零开销方案 + 有开源代码"；
注意：评估只有选择题、无速度数据、与 MoEQuant 的对比不同口径。
