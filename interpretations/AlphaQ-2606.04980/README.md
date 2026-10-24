# AlphaQ — arXiv:2606.04980

> MoE 混合精度量化要决定"哪个专家该多给几个 bit"，现有做法（如 PMQ）靠校准数据估计专家重要性，
> 但前沿 MoE 的训练分布拿不到，校准集只是替身，会误判专家使用率。AlphaQ 改成**只看权重**：
> 用 Heavy-Tailed Self-Regularization（HT-SR）理论里的权重谱重尾指数（PL_Alpha_Hill）衡量每个权重矩阵"训练得好不好"，
> 重尾越明显越重要、给更高位宽，再配上量化噪声项，在全局 bit 预算下解一个整数规划。完全 data-free。

本文件是这篇 paper 的**解读总入口**。radar-native 分析在下方，各角度详读见导航。

## 论文

- **标题**：AlphaQ: Calibration-Free Bit Allocation for Mixture-of-Experts Quantization
- **链接**：[abs](https://arxiv.org/abs/2606.04980) · [PDF](https://arxiv.org/pdf/2606.04980v1) · [code](https://github.com/Superone77/AlphaQ)
- **作者**：Wanqi Yang, Yuexiao Ma, Alexander Conzelmann, Xiawu Zheng, Michael W. Mahoney, T. Konstantin Rusch, Shiwei Liu
- **发表**：2026-06-03 · cs.LG
- **许可**：arXiv 非独占许可（非 CC-BY）——中文翻译只做逐节复述，不是逐句翻译

## 解读导航

| 角度 | 文件 | 内容 |
|---|---|---|
| 原理故事 | [paper.org](paper.org) | 七拍故事版 |
| 中文伴读 | [reading.org](reading.org) | 选择性精读 |
| 中文翻译 | [translation_zh.md](translation_zh.md) | 全文中文复述（逐节释义，不是逐句翻译，含公式/表/图） |
| 溯源倒读 | [../../paper-river/AlphaQ-2606.04980.org](../../paper-river/AlphaQ-2606.04980.org)（中）· [_en](../../paper-river/AlphaQ-2606.04980_en.org)（英） | 已有，沿用 |

数据来源：[`data/summarized/2026-06-03.json`](../../data/summarized/2026-06-03.json)（radar 摘要与打分）+ arXiv HTML 全文。

## Radar 记录

- composite 8，bucket `ptq`，topic_relevance 4，practicality 4，未触发 hard gate。
- **未 triage**。
- radar 记的"最大模型 Qwen1.5-MoE"不准：全文测了 OLMoE-1B-7B、Qwen1.5-MoE、DeepSeek-V2-Lite、Mixtral-8×7B、**Qwen3-30B-A3B**（最大，只测 2.5-bit 预算）。

## 同类对比 / novelty

| 论文 | 关系 |
|---|---|
| PMQ（Huang et al. 2024） | **主对比**：靠校准数据估专家重要性。AlphaQ 在匹配预算下全面更好，并专门做了多领域校准（C4 / CEval-dev / MATH 及其混合）实验来说明 PMQ 对校准域敏感 |
| AFG、DynaMo | 前者作为 Qwen1.5-MoE 2.5-bit 的对照（PPL 未报）；DynaMo 只在附录 A.9 提到"AlphaQ 在 OLMoE 上更好"，主文无数字 |
| [Colla-Q (2609.18131)](https://arxiv.org/abs/2609.18131)、[Q-Strata (2608.30564)](https://arxiv.org/abs/2608.30564)、[DuoRate (2610.13328)](https://arxiv.org/abs/2610.13328)、[Contrast Calibration (2610.20353)](https://arxiv.org/abs/2610.20353) | 同属 MoE 混合精度分配线，晚于 AlphaQ（据标题/摘要，未读全文）；论文自然没对比 |
| [VSRAQ (2606.05688)](../VSRAQ-2606.05688/README.md) | 正交：VSRAQ 改校准**损失**，AlphaQ 改**位宽分配** |
| [TileQ (2605.09281)](../TileQ-2605.09281/README.md) | 同为 MoE 压缩，但走低秩共享结构，不是分配位宽 |

判断：**新点是把 HT-SR 的 PL_Alpha_Hill 搬到 MoE 位宽分配上，并且做成 data-free**；这个指标本身来自前人（模型质量预测），
论文自己的贡献是"用它 + 量化噪声项解全局 ILP"和 MoE 上的系统验证。不是 re-skin，但"重尾即训练得好"的假设是借用的，在 MoE 专家上的因果性靠实验间接支持。

## 实验与工程可落地性

- **预算与精度**：Qwen1.5-MoE 3.5-bit，AlphaQ 平均精度 68.04（BF16 67.93）、PPL 7.38（BF16 7.22），PMQ 67.75 / 7.42，均匀 65.94 / 8.13；2.5-bit 时 AlphaQ 65.17 vs PMQ 62.18 vs 均匀 51.99。
  Qwen3-30B-A3B 2.5-bit：AlphaQ 69.56 / PPL 9.27，均匀 68.76 / 10.29，PMQ 65.88 / 10.94（PMQ 反而差于均匀）。
- **消融（OLMoE-1B-7B，3-bit）**：均匀 PPL 11.78；只用噪声 11.22；只用 PL_Alpha_Hill 10.03；直接相乘 9.56；论文的组合 9.19。说明谱指标是主信号，噪声项是补充，**组合方式也有影响**。
- **粒度**：全局预算优于逐块预算；**层级（专家内各线性层）优于专家级**分配（如 Mixtral 3-bit PPL 4.37 vs 4.72）。所以实际是"每个矩阵一个位宽"，不是严格意义上的"每专家一个位宽"，标题略窄于实质。
- **效率**：A40 上，Mixtral-8×7B 2-bit 加速 1.68×，参数显存 96.8 GB → 13.4 GB；3.5-bit 为 22.0 GB（4.4×）。后端复用 PMQ 风格的 HQQ 反量化，加 Triton 融合 kernel 做 decode 的专家层。**加速数字依赖他们自写的混合位宽 kernel**。
- **成本**：PL_Alpha_Hill 一遍约 3 分钟（OLMoE，3152 个模块）/ 9 分钟（Qwen3-30B-A3B，12448 个模块），一次性离线，外加 ILP 求解。
- **与 AMD / 生产推理**：混合位宽意味着同一层里不同矩阵 bit 数不同，需要 kernel 支持逐矩阵位宽（论文靠 HQQ + Triton 自行解决）；在固定 W4 / FP4 格式的生产栈里，只能当"决定哪些层降到 3-bit 或保 4-bit"的离线策略用，落地成本取决于运行时是否支持混合位宽。
- **与 Quark**：分配结果（每矩阵位宽）可以作为配置输入；实际量化由 GPTQ 完成（据精读代笔 agent 对全文的核对），推理后端才是 HQQ 反量化 + Triton，所以"calibration-free"只指位宽分配这一步，量化本身仍要校准数据。
- **补充**：只用噪声项分配比均匀还差（OLMoE 3-bit 平均准确率 63.74 vs 64.73），收益主要来自谱指标，同时"层级粒度 + 全局预算"这两个结构选择也贡献了一部分，论文没有把它们与 alpha 信号分开。
- **待确认**：只用 PPL 与少数 zero-shot 任务，没有生成式 / 推理任务（VSRAQ 一类的评估显示 4-bit 下生成式任务才是敏感项）；最大模型 30B，没有 100B 级；没有 Mamba 混合 MoE；各项为单次运行、无方差。

## 趋势定位

MoE 混合精度分配是 `ptq` 桶里目前最拥挤的一支（Colla-Q、Q-Strata、DuoRate、Contrast Calibration 等都在 2026-08 ~ 10 扎堆出现）。
AlphaQ 是这一支里**较早**且**data-free** 的代表，卖点（无需校准集）在后续论文里多被替换成"校准更稳健"的叙述，
说明这个切入点有区分度。值得深读 §3 的指标定义和 ILP 形式，以及多领域校准的对照实验设计。

## Triage 结论

未 triage。若入 seed：理由是"data-free、有开源代码、实验覆盖 5 个 MoE，与后续一批 MoE 混合精度工作同路线的起点"，适合作 `ptq` 桶的 MoE 参考锚点。
