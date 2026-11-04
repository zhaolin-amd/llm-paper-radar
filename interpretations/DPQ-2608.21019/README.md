# DPQ — Target-Aware Calibration Data Selection for Preserving Uncertainty in Quantized LMs

arXiv:[2608.21019](https://arxiv.org/abs/2608.21019) · [PDF](https://arxiv.org/pdf/2608.21019v1) · [Code](https://github.com/xi-xiaoran/DPQ) · 2026-08-21 · cs.CL/cs.AI · License CC BY 4.0
作者：Zhen Yang (Yale), Sizai Hou (HKUST) 等 · 8 个 LM（Qwen2.5 0.5–7B、Llama-3.2/3.1 1–8B、Mistral-7B）· 9 个 NLP 基准 · 22 个对比方法

一句话：量化后 top-1 不变，但置信度/margin/可答性判断会漂移；本文把 GPTQ 的校准集选择当成"保留哪种 FP 不确定性行为"的问题，提出 DPQ——用 FP 模型的低 margin（high-doubt）样本 + 通用 anchor 按比例 r 混合，量化器本身不动。结论是没有通吃的配方：SQuAD2 可答性边界用 r=75 最好，广义 MCQA 用 r=50 / 单信号（confidence/entropy-only）更好。

## 导航

| 内容 | 文件 |
|---|---|
| 中文精读（A） | [reading.org](reading.org) |
| 原理故事（B） | [paper.org](paper.org) |
| 中文翻译（E，复述体） | [translation_zh.md](translation_zh.md) |
| Paper-river 溯源（C） | [../../paper-river/DPQ-2608.21019.org](../../paper-river/DPQ-2608.21019.org) · [_en](../../paper-river/DPQ-2608.21019_en.org) |
| Radar 数据 | [../../data/summarized/2026-08-21.json](../../data/summarized/2026-08-21.json) · [digest zh](../../digests/2026-08-21.md#p-2608-21019) |

## Radar 原生分析（D）

**Radar 状态**：已在 2026-08-21 日报推送（survey 桶，composite 8/10，topic 4 / practicality 4；周榜第 9、月榜第 56）。尚未 triage（未 accept / reject）。

**核心机制（三句话）**
1. doubt 分数 `b(x) = 1 − (p₁ − p₂)`，取 FP top-two margin 最小的 `⌊s·r⌋` 条，其余用 WikiText / RandomQA 作 anchor，然后原封不动跑 GPTQ。
2. 理论只有两个命题：Prop.1（margin γ 小的样本在 ‖Δ‖∞ ≥ γ/2 时会翻转），Prop.2（目标分布 T_θ 与校准分布 q_r 的 TV 距离 = |θ−r|，所以最优 r*=θ）。后者在 q_bdry、q_anchor 支撑不相交的假设下成立，是设计直觉，不是预测器（作者自己也这么说）。
3. 候选池只有 512 条 ARC-Challenge train + 512 条 SQuAD2 answerability（目标对齐的池子），评测集不参与。

**关键数字**（4-bit GPTQ，8 模型平均）
- SQuAD2 边界：DPQ-r75 对 WikiText，boundary-acc 偏差 0.4553→0.2125，answerable-rate 偏差 0.2271→0.1000，JSD 0.0387→0.0158；r100 反而变差（12.44 名），说明 anchor 必要。
- 广义 MCQA（Table 4）：DPQ-r75 排倒数第二（11.88），而 BNB-NF4 的 Agr./JSD 最好（0.8637 / 0.0241）；confidence-only 8.96 领先，但各方法差距很小（Agr. 0.818–0.836）。
- 后验校准（Table 5）：温度缩放不改 acc/agreement 但 JSD 0.0320→0.0445；矩阵/Dirichlet/isotonic 提高 acc 但 agreement 掉到 ~0.77。结论：后验校准与 FP 行为保留是不同目标。
- AWQ：7B/8B 上 DPQ-r75 mean rank 1.97，但分数类指标在部分数据集仍偏向通用校准——只部分迁移。

**同类对比 / novelty**
- 与 FAQ 2601.11200（量化，标题为"再生成校准数据 + family-aware 量化"，radar 无摘要，仅据标题判断）：同样在改校准数据，但看起来目标是降量化误差；DPQ 的选择信号来自 FP 预测。未读全文，需核对。
- 与 2606.03328（高稀疏**剪枝**，15 个校准源、4 个能力维度）：同样得出"没有单一校准源保住所有能力"，DPQ 在量化上用 margin/可答性维度得出类似结论。
- 与 2610.15578（MoE 专家**剪枝**，gpt-oss-20b，校准集规模/语言/分布）：问题同族，对象是剪枝。
- 本文的"新"主要在问题定义（把 uncertainty preservation 当校准选择目标）和 mixture-mismatch 的论证框架；算法本身是 margin 排序 + 混合，很轻。survey 桶的归类（scorer 说是方法论比较）我认为偏宽——它其实是一个小方法 + 大规模消融。
- survey 桶里其他篇（Broken on Arrival 2609.05881、Don't Skip the Baseline 2610.28369 等）都是 deployment/benchmark 类，与本文不构成方法层面的竞争。

**趋势 / timing**：survey 桶近两周日均 ≤1 篇，没有升温；"校准数据怎么选"这条线在 radar 里是慢变量（FAQ、两篇剪枝校准集研究 2606.03328 / 2610.15578、本文），值得作为一个小专题跟踪，但不急着精读全文。

**工程可落地性（AMD / 生产推理视角）**
- 优点：零推理开销，不改量化 kernel；只需 FP 前向一次给 1024 条候选打分，成本可忽略；可直接放进 Quark 的 GPTQ 校准数据入口。
- 限制（均来自论文本身）：
  - 只测 4-bit weight-only（GPTQ/BNB-NF4，AWQ 作检查），最大模型 8B；没有 W4A8 / FP8 / MXFP4 / 70B+ / MoE。
  - 不确定性只用 option scoring（MCQA + SQuAD2 二选一）衡量，没有生成式、verbalized confidence、长上下文。
  - 候选池是与评测目标对齐的（ARC + SQuAD2 train），"r75 最优"很可能依赖这一点；跨域池没测。
  - 没有给出如何在部署时预先知道 θ（目标 boundary 占比）——论文建议用验证集指标挑 recipe，等于要多量化几次。
- 需在 PDF 里核对：校准样本数 s=128 的固定性、seed 方差（附录 A.1 有 stability，未细看）、BNB-NF4 为何在 broad MCQA 上 Agr./JSD 反而最好。

**Triage 建议**：不是算法突破，但对"量化后需要 abstain / rerank / 打分的下游"有直接的校准集选择启示；若要 accept 进 seeds，理由应写成"校准集选择 × uncertainty preservation 的评测框架"，而不是方法。
