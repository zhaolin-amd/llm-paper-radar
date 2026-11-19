# FragCal-DC — When Calibration Data Matter in Sub-4-Bit LM Quantization

arXiv:2611.11079 · Jiang, Chen, Tian · cs.LG/cs.CL · 2026-11-08
[abs](https://arxiv.org/abs/2611.11079) · [pdf](https://arxiv.org/pdf/2611.11079v1) · 无官方代码链接

一句话：这是一篇"测量型"论文——固定量化器的更新规则，只改 calibration 数据（Pile + 低保留率任务族的 benchmark 训练文档），系统比较 GPTQ / AWQ / OmniQuant / AQLM / SignRound V2 在 2–4 bit 的响应。

## 导航
- [中文翻译（全文复述）](translation_zh.md)
- [中文精读](reading.org) · [原理与故事](paper.org)
- paper-river 溯源：[zh](../../paper-river/FragCal-DC-2611.11079.org) · [en](../../paper-river/FragCal-DC-2611.11079_en.org)
- radar 数据：[data/summarized/2026-11-08.json](../../data/summarized/2026-11-08.json) · [digest](../../digests/2026-11-08.md)

## Radar-native 分析 (D)

**Triage 状态**：2026-11-08 digest 已收录（composite 9，bucket `low_bits`，topic_relevance 5，practicality 4）；未被 accept/reject，尚未进 seeds。

**核心结论（论文自述，已对照正文）**
- 2-bit：SignRound V2 六模型 +1.83 pp（6/6 正），OmniQuant 五模型 +1.78 pp（Mistral-7B 因 W2 baseline 不可用被排除）；AQLM +0.73 pp，CI 含 0。
- 3-bit：GPTQ/AWQ 均值响应小，CI 跨 0；4-bit 基本 ±0.1 pp，无响应。
- Llama-3.1-70B W2：GPTQ +2.33 pp（gap closure 6.6%），SR-V2 +0.86 pp（GC 13.8%）——绝对增益与 GC 排序相反。
- 对照：错误答案标注保留约 90% 增益，不给答案标注保留约 55%；增益集中在被纳入 calibration 的任务族；重构误差变化与精度变化不一致（GPTQ 误差 6 格中 5 格上升，但精度 3 格上升；AQLM 误差 6/6 上升，精度 5/6 上升）。

**对 radar 摘要的修正/需要警惕的点**
- radar 记 "Llama-3-70B"，正文为 **Llama-3.1-70B**。
- radar highlight 写"calibration 文本语义而非标签质量是关键"——论文只证明"答案指定"不关键；MMLU/ARC/OBQA 仍保留全部候选选项（含正确项），不等于去掉了答案信息。
- **评测泄漏未排除**：calibration 用的是 benchmark 训练集，论文自己的 overlap 检查在评测题干和 HellaSwag completion 上有匹配，"full-item 验证与去除后精度影响仍未解决"。"任务知情 calibration" 的收益有多少是任务相关激活覆盖、多少是 benchmark 污染，无法从本文分开。
- 优化器类别结论（迭代式 vs 单 pass）论文自己说证据不足：W3 的 SR-V2 vs AWQ 增益差 +0.40 pp，CI [-0.10, +0.90]；类别对比 +0.06 pp，sign-flip p=0.875。radar 摘要里"迭代式受益更显著"是 W2 baseline 可用性与 bit 宽耦合的结果，不是干净的 optimizer 效应。
- FragCal-DC 本身（208 文档：128 Pile + 32 MMLU aux-train + 32 ARC/OBQA + 16 HellaSwag）相对 random QA 混合只多 +0.47 pp（+1.93 vs +1.46），且对照没有隔离"fragility 排序"。

**同类对比 / novelty**
- 同 bucket 的 CodeGEMM / StrataFlux / DuoRate / Tetra / LC-QAT 都在动格式、kernel 或 QAT；本文是 bucket 内唯一专门研究 *calibration 数据* 的，与它们正交。最近邻是 LC-QAT（2-bit 数据高效训练）：同样关注 2-bit 下数据用量，但路径是 QAT。
- 想法本身（task-specific / in-domain calibration 提升 PTQ）并不新，新在 cross-method、cross-bit 的受控测量与"重构误差不是可靠代理"的负结果。

**趋势 / timing**：`low_bits` 桶 ~1 篇/天，平稳；calibration 数据方向在桶内偏冷门。值得精读的是第 6 节控制实验，不是方法本身。

**工程可落地性**：practicality 4 偏乐观。calibration 成本低、可嫁接现有 pipeline，但：(1) 收益只出现在 2-bit 且 SR-V2/OmniQuant，这类迭代方法本身 calibration 成本更高；(2) 只验证了 MCQ 类 benchmark，代码生成、多轮对话未测；(3) 对"部署模型的目标任务"需要有对应训练文档，才能复用该思路；(4) 无端到端速度/峰值显存数据（unknown）。在 AMD/Quark 侧：可作为 AutoRound/SignRound 校准集构造的参考，但先要自己的 held-out 去重。

**待 PDF 核查**：附录 Table 59 的 overlap 比例；Table 11 逐模型结果；Mistral-7B 的 W2 baseline 为何不可用。
