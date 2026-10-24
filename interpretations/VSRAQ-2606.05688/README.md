# VSRAQ — arXiv:2606.05688

> MoE 量化的特殊毛病：权重/激活的一点点扰动会让 router 的 top-k 选择翻转，token 直接走进另一批专家，
> 这不是连续误差而是计算路径变了。VSRAQ 在 AutoRound 这类"块重构"校准上，**多加一个 router 对齐损失**：
> value 项对齐被选中专家的 logit（外加对 sigmoid 路由敏感区加权），structure 项保持被选专家之间的排序和第 k 名与第 k+1 名之间的间隔。
> 只改校准目标，推理图不变，零推理开销。

本文件是这篇 paper 的**解读总入口**。radar-native 分析在下方，各角度详读见导航。

## 论文

- **标题**：Value-and-Structure Alignment for Routing-Consistent Quantization of Mixture-of-Experts Models
- **链接**：[abs](https://arxiv.org/abs/2606.05688) · [PDF](https://arxiv.org/pdf/2606.05688v1) · code：全文未给出代码链接
- **作者**：Hancheol Park, Geonho Lee, Tairen Piao, Tae-Ho Kim（Nota Inc.，韩国）
- **发表**：2026-06-04 · cs.CL / cs.AI（ICML 模板）
- **许可**：arXiv 非独占许可（非 CC-BY）——中文翻译只做逐节复述，不是逐句翻译

## 解读导航

| 角度 | 文件 | 内容 |
|---|---|---|
| 原理故事 | [paper.org](paper.org) | 七拍故事版 |
| 中文伴读 | [reading.org](reading.org) | 选择性精读 |
| 中文翻译 | [translation_zh.md](translation_zh.md) | 全文中文复述（逐节释义，不是逐句翻译，含公式/表/图） |
| 溯源倒读 | [../../paper-river/VSRAQ-2606.05688.org](../../paper-river/VSRAQ-2606.05688.org)（中）· [_en](../../paper-river/VSRAQ-2606.05688_en.org)（英） | 已有，沿用 |

数据来源：[`data/summarized/2026-06-04.json`](../../data/summarized/2026-06-04.json)（radar 摘要与打分）+ arXiv HTML 全文。

## Radar 记录

- composite 8，bucket `ptq`，topic_relevance 4，practicality 4，未触发 hard gate。
- **未 triage**（既非 seed 也非 rejected）。
- radar 打分时只看了摘要，"最大模型 / 位宽 / benchmark 未知"这几项现在可以补上：**Solar-Open-100B（102B）**、**Nemotron-3-Nano-30B-A3B（Mamba-Transformer 混合 MoE）**；位宽只有 **W4A16 和 NVFP4**；基础框架 AutoRound；校准集 512 条推理数据（OpenCodeReasoning / OpenScienceReasoning-2 / OpenMathReasoning）。

## 同类对比 / novelty

方法线是"router 感知的 MoE 量化"，直接前辈是 EAC-MoE（TopK-MSE 损失）和 EAQuant（对齐 router logits/概率）。论文自己只对比了 AutoRound 原目标和"AutoRound + TopK-MSE"。

| 论文 | 关系 |
|---|---|
| EAC-MoE（ACL 2025）/ EAQuant（2506.13329） | **最近的前辈**。VSRAQ 的 value 项里 `L_TopK` 就是直接沿用 EAC-MoE 的 TopK-MSE；新增的只有 sigmoid 敏感度加权 + 两个 structure 项。论文没跑 EAQuant，只跑了 TopK-MSE 变体 |
| [AlphaQ (2606.04980)](https://arxiv.org/abs/2606.04980) | 同样是 MoE PTQ，但改**位宽分配**而非校准目标，两者正交，理论上可叠加（论文没测） |
| [Colla-Q (2609.18131)](https://arxiv.org/abs/2609.18131)、[Q-Strata (2608.30564)](https://arxiv.org/abs/2608.30564)、[DuoRate (2610.13328)](https://arxiv.org/abs/2610.13328) | 都是 MoE 混合精度分配线（据标题/摘要，未读全文），与 VSRAQ 的"校准目标"线互补 |

判断：**增量式改进，不是新范式**。核心洞察（量化会翻转 top-k，所以对齐 router）EAC-MoE / EAQuant 已经提出；
本文的新东西是把"对齐值"推广到"对齐结构"（排序 + 边界间隔）以及对 sigmoid 路由的敏感度加权。
消融（表 5）显示增益并不单调：Solar W4A16 上 `+Sigmoid` 的 PPL 6.72 反而比完整 VSRAQ 的 6.81 好，加 structure 项只在 NVFP4 上带来 PPL 改进；Nemotron 上 `+TopK-MSE` / `+Sigmoid` 几乎没动（10.78 / 10.79 vs 基线 10.79）。

## 实验与工程可落地性

- **精度**（WikiText-2 PPL / 生成式推理平均 / 似然式选择题平均）：
  - Solar-Open-100B W4A16：FP 6.06 / 73.62 / 76.92；AutoRound 7.12 / 72.19 / 76.44；VSRAQ 6.81 / 73.22 / 76.76。
  - Solar-Open-100B NVFP4：AutoRound 7.22 / 61.78 / 75.59；VSRAQ 6.90 / 64.43 / 75.63（FP 为 6.06 / 73.62 / 76.92）。
  - Nemotron-3-Nano-30B NVFP4：AutoRound 10.79 / 69.10 / 69.96；VSRAQ 10.70 / 70.92 / 70.18（FP 10.36 / 73.64 / 70.33）。
- **收益有限**：生成式任务有 +0.4 ~ +2.7 分（NVFP4 上最明显），似然式任务基本是噪声级；Solar NVFP4 生成式仍比 FP 低 ~9 分，VSRAQ 只拿回其中约 2.7 分。
- **一致性证据**：Jaccard（616,759 tokens，仅 W4A16 的逐层曲线）最大处相对提升 11.29%；论文没给绝对值和 NVFP4 的曲线。
- **成本**：无推理开销，因为只改校准损失；校准阶段的额外耗时（router logits 的前向 + 排序）论文**未报告**。
- **与 AMD / 生产推理**：产物就是 AutoRound 的 W4A16 / NVFP4 checkpoint，不需要新 kernel，落地门槛低；对 Quark 一类框架而言，等价于"在块重构损失里加一项 router 对齐"。但需要能拿到每个 MoE 块里 router 的 logits，且路由形式（softmax vs sigmoid）要匹配。
- **待确认**（论文没说）：无代码；只测 4-bit（作者自己承认 2/3-bit 才可能有大收益，未验证）；没有对比 EAQuant / MoEQuant；`λ_router` 在两个模型上取值不同（0.01 vs 0.1），说明要调参；每个设置只一个种子，没有方差。

## 趋势定位

`ptq` 桶里 MoE 专用 PTQ 近半年明显升温（本仓库 2026-02 ~ 2026-10 有十余篇，见 [AlphaQ](../AlphaQ-2606.04980/README.md)、[TileQ](../TileQ-2605.09281/README.md)）。
VSRAQ 属于其中"校准目标改良"这一支，且是少数在 **100B 级 MoE + NVFP4 + 混合 Mamba MoE** 上验证的工作。
值得读 §3.3–3.4（损失设计只有几行）；对已有 AutoRound / NVFP4 流水线的团队，实现成本很低，可以小规模试一下，但别指望大幅收益。

## Triage 结论

未 triage。若入 seed：理由是"router 感知 MoE PTQ 的最新一跳，100B 规模 + NVFP4 有实证"；
不建议高优先级——增益是 PPL 0.1~0.3、生成式 1~3 分，且无代码。
