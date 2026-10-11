# UltraQuant: 4-bit KV Caching for Context-Heavy Agents

[arXiv:2606.20474](https://arxiv.org/abs/2606.20474) · [PDF](https://arxiv.org/pdf/2606.20474.pdf) · 代码：论文未提供仓库链接

AMD 团队的 4-bit KV cache 压缩工作：面向 agent 式长会话场景，用硬件原生的 FP4+UE8M0 网格替代 TurboQuant 式的码本查表，让反量化直接折进 CDNA4 的 scaled-MFMA 指令，在长上下文高并发下逼近/反超 FP8 KV cache 吞吐量，同时显存减半。

## Radar 原生分析

**同类对比（同 bucket: `kv_cache`）**

| 相似论文 | 核心区别 |
|---|---|
| [QuantMLA (2609.36760)](../../data/summarized/2026-09-11.json) | 针对 MLA 双路径做 INT4/INT2，本文针对标准 MHA/GQA 做 FP4，硬件绑定方式不同 |
| [RoPE-Aware Bit Allocation (2606.24033)](../../data/summarized/2026-09-11.json) | 按 RoPE 位置做比特分配的算法侧工作，本文是系统侧的硬件格式选择，互补而非竞争 |
| [Tailoring Quantization Space for 1-Bit KV (2610.03027)](../../data/summarized/2026-09-11.json) | 走向更激进的 1-bit，本文守在 4-bit 但换格式——两者解的是"要多激进"和"怎么落地"两个不同轴 |

本文的核心新意不在"4-bit KV cache"这个方向本身（这个方向上论文很多），而在一个具体的反转结论：消融实验显示，决定精度的关键是 per-block absmax 定标，而不是码本精细程度——这个发现在同 bucket 的其他论文里没有被单独拎出来验证过，值得作为一个可迁移的设计原则记一笔。

**趋势定位**：`kv_cache` bucket 过去一个月（09-11 至 10-08）非 gated 论文产出量从每天 1-2 篇涨到 4-6 篇，方向在持续升温，但本文被 scorer 打了 `hard_gate`，不计入这个热度曲线——它本身依赖的 CDNA4 scaled-MFMA 是定制硬件特性，不是这波"算法侧"热潮里可直接对比的论文。

**工程可落地性**：`practicality=0`（hard_gate）。核心瓶颈不是算法复杂度，是硬件依赖——scaled-F8F6F4 MFMA 指令目前只存在于 AMD CDNA4（MI350/MI450 级别），标准 NVIDIA 推理栈或更早的 AMD 架构都无法直接复用这条路径。精度方面峰值显存减半（KV bytes 减半）是实打实的收益，但 AIME25 掉 10-13pp 说明在数值推理密集的任务上不是"无损"压缩。校准成本未知——论文只说常数 c 和码本都是离线标定，没有给出标定耗时的量化数字。

**Triage 状态**：尚未 triage（既未 accept 也未 reject）。scorer 的 hard_gate 理由：无标准 LLM 精度 benchmark（MMLU/GSM8K/HumanEval 等）+ 依赖非量产定制硬件（CDNA4 scaled-MFMA）= 判定为不满足"可在标准推理栈复现"的门槛。这个判断本身合理——如果你在 NVIDIA GPU 或更早的 AMD 架构上做推理优化，这篇论文的硬件路径不可直接移植；但其中 per-block absmax 定标这个算法设计原则，是跟硬件无关的，值得单独摘出来看。

## 导航

- [原理故事（七拍叙事）](paper.org) — 从 agent 显存压力讲到"硬件原生格式优先于理论最优表示"这一核心洞见
- [中文精读 / 伴读](reading.org) — 逐段翻译 + 注疏 + 术语表，含待压力测试的核心假设
- [中文翻译（全文复述）](translation_zh.md) — 章节级全文覆盖，含全部 9 张图表
- [Paper-river 溯源](../../paper-river/UltraQuant-2606.20474.org)（[英文版](../../paper-river/UltraQuant-2606.20474_en.org)）— 倒读法追溯 KIVI/KVQuant → QJL → TurboQuant 的算法谱系，交叉 PagedAttention/vLLM FP8 → UE8M0/CDNA4 的部署谱系
- [Radar 原始评分数据](../../data/summarized/2026-09-11.json)
