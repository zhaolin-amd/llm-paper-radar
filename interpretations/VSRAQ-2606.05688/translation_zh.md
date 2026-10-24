> 中文翻译（按章节复述，非逐句直译）：VSRAQ: Value-and-Structure Alignment for Routing-Consistent Quantization of Mixture-of-Experts Models，arXiv:2606.05688v1 [cs.CL]，2026-06-04。作者：Hancheol Park, Geonho Lee, Tairen Piao, Tae-Ho Kim（Nota Inc., 韩国）。原文：https://arxiv.org/pdf/2606.05688v1 。本文依 arXiv 默认许可，用自己的话逐节复述，公式、表格与数字保持原样。论文无附录（正文之后只有 Impact Statement 与参考文献）。

# 摘要

MoE 模型每个 token 只激活一部分专家，因而能高效扩展，但专家参数总量巨大，部署时仍必须量化。与稠密模型不同，MoE 对路由不稳定很敏感：量化带来的微小扰动就可能改变 top-$`k`$ 专家的选择，进而改变计算路径、拉低模型质量。作者提出 VSRAQ（Value-and-Structure Routing Alignment for Quantization），一种面向 MoE 的 PTQ 目标，让量化后保持量化前的专家选择行为。它由两部分互补的目标组成：值对齐（匹配与路由相关的 logits 或分数）和结构对齐（保持专家排序与 top-$`k`$ 决策边界）。保持路由一致性之后，量化退化减少，且推理时没有任何额外开销，也能嵌入现有量化框架。在近期 MoE 基础模型上的实验表明，VSRAQ 提高了专家选择一致性，并稳定优于仅做重建的基线和已有的路由感知基线。

关键词：Machine Learning, ICML

# 1 引言

MoE 通过扩大模型容量、同时每个 token 只激活部分参数，成为扩展大语言模型的有效路线（Shazeer et al., 2017；Fedus et al., 2022；Chen et al., 2025）。典型 MoE 层里，路由器根据输入隐状态算出各专家得分，挑出 top-$`k`$ 个专家处理该 token。稀疏激活让激活参数量保持较小，推理很高效。

但部署仍是难题：虽然每个 token 只用少数专家，全部专家参数通常都要常驻内存，所以内存和带宽开销很大。因此 PTQ 对在内存与延迟约束下部署 MoE 很重要。GPTQ（Frantar et al., 2023）、SmoothQuant（Xiao et al., 2023）、AWQ（Lin et al., 2024）以及基于学习的取整方法（Cheng et al., 2024）在稠密 Transformer 上表现很好。

问题在于常规量化目标是为"固定计算图上的数值保真"设计的，而 MoE 层依赖输入相关的路由。量化后模型的行为不仅取决于权重、激活或块输出重建得多准，还取决于选出的专家是否相同。隐状态或路由 logits 的小扰动就能改变 top-$`k`$ 选择；这种重路由不是连续的数值误差，而是让 token 走了不同的专家，改变了计算路径本身。所以需要显式保持路由行为的量化目标。

近期工作已开始处理 MoE 量化的专门问题：一类关注 MoE 层的校准集构造（平衡专家激活、建模 token-专家亲和度；Hu et al., 2025；Fu et al., 2025）；一类研究结构感知的混合精度，按敏感度给不同 MoE 组件分配不同位宽（Li et al., 2024）；最相关的一类致力于保持量化前后的路由行为，例如 EAC-MoE（Chen et al., 2025）通过校准路由器缓解低比特量化造成的专家选择变化，EAQuant（Fu et al., 2025）通过匹配路由 logits 或路由概率分布做路由一致性对齐。

这些工作说明路由行为对稳健的 MoE 量化至关重要。但现有路由感知目标主要保持"路由值"，即匹配量化前后原本被选中的 top-$`k`$ 专家的 logits、概率或分数。而 top-$`k`$ 决策不仅由数值决定，还由结构关系决定：被选中专家之间的相对顺序影响混合成分；第 $`k`$ 个被选专家与相邻未选专家之间的间隔，决定了选择边界在扰动下有多稳。量化模型可能分数匹配得还不错，却仍出现排名反转或越过边界而改变专家选择。

为此提出 VSRAQ，通过两个互补部分保持路由行为：值对齐，匹配与路由相关的 logits 或分数，并额外强调路由函数的敏感区域；结构对齐，保持专家排序以及被选专家与邻近未选专家之间的决策边界。VSRAQ 可作为额外的校准目标嵌入现有 PTQ 框架，无推理开销。

贡献总结：

- 提出 VSRAQ，一个同时保持路由值与结构关系的 MoE 量化目标，包含敏感度感知的值对齐、被选专家排序、以及 top-$`k`$ 边界间隔。
- 说明 VSRAQ 是现有 PTQ 流程的即插即用校准目标，不引入推理开销。
- 在近期 MoE 基础模型上验证，专家选择一致性提高，量化引起的性能下降减小。

# 2 相关工作

**MoE 专用量化。** 近期工作围绕 MoE 稀疏、输入相关的计算做量化。由于专家使用往往极不均衡，校准数据必须对各专家有足够覆盖，否则量化会退化。MoEQuant（Hu et al., 2025）构造专家均衡的校准集，并把 token-专家亲和度纳入量化；EAQuant（Fu et al., 2025）同样在专家层面平衡校准数据来改进稀疏专家的校准。除校准之外，QuantMoE-Bench（Li et al., 2024）表明注意力层、共享专家、路由专家、MoE 块等不同组件对量化的敏感度不同，从而启发按组件或按专家的混合精度。这些工作说明 MoE 量化应考虑稀疏专家激活、校准不均衡和组件级敏感度。

**聚焦路由器的 MoE 量化。** 另一类工作盯住决定稀疏计算路径的路由器。由于量化扰动会改变被选的 top-$`k`$ 专家，路由感知方法用辅助损失优化量化参数，使其保持全精度模型的路由行为。EAQuant 对齐量化前后的路由 logits 与专家选择概率，但其全维度对齐可能把优化精力花在很少影响 top-$`k`$ 决策的低排名专家上；EAC-MoE（Chen et al., 2025）则把校准集中在 top-$`n`$ （记作 $`nk`$ ）专家上以缓解低比特量化后的专家选择变化。

VSRAQ 在此基础上进一步细化路由对齐，联合保持路由值与路由结构：第一，敏感度感知的值对齐，强调那些小的 logit 扰动会引起较大路由分数变化的区域；第二，保持路由输出之间的结构关系，包括被选专家的排序，以及被选专家与邻近未选专家之间的间隔。这为路由稳定的 MoE 量化提供更有针对性的目标。

# 3 方法：Value-and-Structure Routing Alignment for Quantization

VSRAQ 不是独立的量化算法，而是一个可以嵌入现有量化框架的校准目标。本文把它实例化在一个基于学习的 PTQ 框架之上：裁剪范围、缩放因子、零点、取整参数等量化参数，用一小份校准数据来优化。

## 3.1 基于重建的量化

设 $`F`$ 为未量化的解码器块， $`\hat{F}`$ 为其量化版本。给定输入隐状态 $`\mathbf{x}`$ ，块输出为

```math
\mathbf{y}^{\mathrm{fp}}=F(\mathbf{x}),\qquad\mathbf{y}^{q}=\hat{F}(\mathbf{x}).
```
（式 1）

传统基于重建的量化通过最小化块输出误差来优化量化参数：

```math
\mathcal{L}_{\mathrm{recon}}=\left\|\mathbf{y}^{q}-\mathbf{y}^{\mathrm{fp}}\right\|_{2}^{2}.
```
（式 2）

该目标对稠密模型有效，但不显式保持 MoE 层的路由行为。MoE 里解码器块的输出强烈依赖路由器选出的专家，所以只最小化重建损失，可能得到"块输出数值上接近、但量化后选中的专家变了"的解。VSRAQ 因此在重建目标上增加路由对齐损失：

```math
\mathcal{L}=\mathcal{L}_{\mathrm{recon}}+\lambda_{\mathrm{router}}\mathcal{L}_{\mathrm{router}},
```
（式 3）

其中 $`\lambda_{\mathrm{router}}`$ 控制路由对齐的强度。

## 3.2 路由 logits

设 $`\mathbf{z}^{\mathrm{fp}}\in\mathbb{R}^{E}`$ 为未量化模型的路由 logits， $`E`$ 为专家数； $`\mathbf{z}^{q}\in\mathbb{R}^{E}`$ 为量化模型的路由 logits。按 $`\mathbf{z}^{\mathrm{fp}}`$ 对专家排序，用 $`\pi(i)`$ 表示未量化路由 logits 下排名第 $`i`$ 的专家下标， $`\pi`$ 是满足下式的专家下标排列：

```math
z^{\mathrm{fp}}_{\pi(1)}\geq z^{\mathrm{fp}}_{\pi(2)}\geq\cdots\geq z^{\mathrm{fp}}_{\pi(E)}.
```
（式 4）

被选中的 top-$`k`$ 专家集合定义为

```math
\mathcal{T}_{k}=\lbrace\pi(1),\pi(2),\ldots,\pi(k)\rbrace.
```
（式 5）

VSRAQ 把路由对齐分解为值对齐与结构对齐：

```math
\mathcal{L}_{\mathrm{router}}=\mathcal{L}_{\mathrm{value}}+\mathcal{L}_{\mathrm{struct}}.
```
（式 6）

## 3.3 值对齐

值对齐损失负责保持量化前后与路由相关的 logits 或分数。VSRAQ 把它分成两部分：直接的 top-$`k`$ logit 保持，以及 sigmoid 感知的分数保持。

**Top-$`k`$ logit 保持。** 最自然的做法，是对未量化路由器所选的那些专家，直接让量化路由 logits 去匹配全精度模型的 logits。沿用 EAC-MoE（Chen et al., 2025），定义 TopK-MSE 损失：

```math
\mathcal{L}_{\mathrm{TopK}}=\frac{1}{k}\sum_{i=1}^{k}\left(z^{q}_{\pi(i)}-z^{\mathrm{fp}}_{\pi(i)}\right)^{2}.
```
（式 7）

**Sigmoid 感知的分数保持。** 直接匹配 logit 没有考虑近期 MoE 采用的路由形式。DeepSeek-V3（DeepSeek-AI, 2024）、Nemotron-3-Nano（NVIDIA et al., 2025）、GLM-4.5（GLM-4.5 Team, 2025）等现代 MoE 采用基于 sigmoid 的路由。softmax 路由在所有专家上归一化，专家之间直接竞争；sigmoid 路由则在 top-$`k`$ 选择之前独立估计每个 token-专家亲和度，通常只对被选专家做归一化。由于 sigmoid 的 S 形曲线，接近 $`0.5`$ 的分数对输入 logit 扰动的变化，比接近 $`0`$ 或 $`1`$ 的饱和分数剧烈得多。因此量化误差在这一区域会引起更大的路由分数变化，可能改变专家选择。VSRAQ 通过保持量化前后基于 sigmoid 的路由分数来应对这一点，引入 sigmoid 感知的值对齐损失。

对每个被选专家 $`\pi(i)`$ ，计算

```math
p^{\mathrm{fp}}_{\pi(i)}=\sigma\left(z^{\mathrm{fp}}_{\pi(i)}\right),\qquad p^{q}_{\pi(i)}=\sigma\left(z^{q}_{\pi(i)}\right),
```
（式 8）

其中 $`\sigma(\cdot)`$ 为 sigmoid 函数。敏感度权重定义为

```math
w_{\pi(i)}=\left(p^{\mathrm{fp}}_{\pi(i)}\left(1-p^{\mathrm{fp}}_{\pi(i)}\right)\right)^{\gamma}+\epsilon_{w},
```
（式 9）

$`\gamma`$ 控制加权的尖锐程度， $`\epsilon_{w}`$ 是为数值稳定而设的一个很小的最小权重。因为 $`p(1-p)`$ 在 $`p=0.5`$ 处最大，这个权重给 sigmoid 敏感区域内的 logits 更大的重要性。sigmoid 感知的对齐损失为

```math
\mathcal{L}_{\mathrm{sig}}=\frac{1}{k}\sum_{i=1}^{k}w_{\pi(i)}\left(p^{q}_{\pi(i)}-p^{\mathrm{fp}}_{\pi(i)}\right)^{2}.
```
（式 10）

敏感度权重的作用是防止这一项把量化优化约束得过死。VSRAQ 在多个目标下优化量化参数，以保持量化前的路由决策；若强迫模型精确匹配对路由决策影响有限的饱和分数，就会消耗校准容量，并干扰更关键的路由目标。该权重压低低敏感度的分数差异，把校准信号集中到分数变化更可能影响路由的区域。

最终的值对齐损失为

```math
\mathcal{L}_{\mathrm{value}}=(1-\alpha)\mathcal{L}_{\mathrm{TopK}}+\alpha\mathcal{L}_{\mathrm{sig}},
```
（式 11）

其中 $`\alpha\in[0,1]`$ 平衡直接 logit 匹配与 sigmoid 感知分数匹配。实践中 $`\alpha`$ 取较小值，因为保持原先被选中 top-$`k`$ 专家的 logits 是路由一致性的主要驱动力，sigmoid 感知项只是针对敏感路由区域的辅助细化。

## 3.4 结构对齐

值对齐促使 top-$`k`$ 路由 logits 或分数在量化前后保持接近，但路由稳定性不仅取决于数值本身，也取决于它们的结构关系。VSRAQ 保持两种与路由相关的结构：被选专家之间的顺序，以及 top-$`k`$ 选择边界附近的间隔。

**Top-$`k`$ 顺序保持。** 全精度路由器对被选专家的排序为

```math
z^{\mathrm{fp}}_{\pi(1)}\geq z^{\mathrm{fp}}_{\pi(2)}\geq\cdots\geq z^{\mathrm{fp}}_{\pi(k)}.
```
（式 12）

VSRAQ 鼓励量化后的路由器保持这一顺序。对相邻的被选专家，定义排名保持损失：

```math
\mathcal{L}_{\mathrm{rank}}=\frac{1}{k-1}\sum_{i=1}^{k-1}\mathrm{softplus}\left(m_{\mathrm{rank}}-\left(z^{q}_{\pi(i)}-z^{q}_{\pi(i+1)}\right)\right),
```
（式 13）

$`m_{\mathrm{rank}}`$ 是排名间隔。当 $`m_{\mathrm{rank}}=0`$ 时，该损失惩罚排名反转；当 $`m_{\mathrm{rank}}>0`$ 时，还额外鼓励相邻被选专家之间保持正的间隔。

**边界保持。** 重路由往往发生在 top-$`k`$ 截断边界附近，尤其是第 $`k`$ 个被选专家与邻近未选专家之间。设 $`b`$ 为截断之后所考虑的邻近未选专家个数。对 $`j=1,\ldots,b`$ ，定义全精度与量化后的边界间隔：

```math
\Delta^{\mathrm{fp}}_{j}=z^{\mathrm{fp}}_{\pi(k)}-z^{\mathrm{fp}}_{\pi(k+j)},\qquad\Delta^{q}_{j}=z^{q}_{\pi(k)}-z^{q}_{\pi(k+j)}.
```
（式 14）

边界间隔保持损失为

```math
\mathcal{L}_{\mathrm{margin}}=\frac{1}{b}\sum_{j=1}^{b}\left(\Delta^{q}_{j}-\Delta^{\mathrm{fp}}_{j}\right)^{2}.
```
（式 15）

最终结构对齐损失为

```math
\mathcal{L}_{\mathrm{struct}}=\beta_{\mathrm{rank}}\mathcal{L}_{\mathrm{rank}}+\beta_{\mathrm{margin}}\mathcal{L}_{\mathrm{margin}},
```
（式 16）

$`\beta_{\mathrm{rank}}`$ 与 $`\beta_{\mathrm{margin}}`$ 分别控制顺序保持与边界保持的相对强度。

## 3.5 总体目标

合并值项与结构项，最终的路由对齐损失为

```math
\mathcal{L}_{\mathrm{router}}=\mathcal{L}_{\mathrm{value}}+\mathcal{L}_{\mathrm{struct}}.
```
（式 17）

总校准目标由式 3 给出。由于 VSRAQ 只改校准目标，不改变量化模型的推理计算图，因此没有额外的推理开销。

# 4 实验

## 4.1 实验设置

在两个近期开源 MoE LLM 上评估：Solar-Open-100B（Park et al., 2026）与 Nemotron-3-Nano-30B-A3B（NVIDIA et al., 2025）。Solar-Open-100B 是 102B 参数的 MoE 语言模型，作为大规模纯 Transformer MoE 的评估对象；Nemotron-3-Nano-30B-A3B 采用 Mamba-Transformer 混合 MoE 架构，用来检验 VSRAQ 在纯 Transformer MoE 之外是否依然有效。对 Solar-Open-100B 评估 W4A16 仅权重量化和 NVFP4 量化（W4A16 只量化权重、激活保持较高精度；NVFP4 同时量化权重与激活）；对 Nemotron-3-Nano-30B-A3B 评估 NVFP4。

基础 PTQ 框架用 AutoRound（Cheng et al., 2024），它在一小份校准集上通过最小化块级重建损失优化量化参数。基线即使用原始重建目标的 AutoRound。VSRAQ 的实现方式，是在该重建目标上加上提出的路由对齐损失，使量化参数同时保持块输出与路由行为。还对比了一个变体：把 EAC-MoE 的 TopK-MSE 损失作为 AutoRound 的辅助损失。

校准集使用一个紧凑的、偏推理的校准集，称为 NVIDIA Reasoning 校准集，由 OpenCodeReasoning（Ahmad et al., 2025）、OpenScienceReasoning-2（NVIDIA Corporation, 2025）和 OpenMathReasoning（Moshkov et al., 2025）构成。每个模型从该混合集中随机抽 512 个样本；最大校准序列长度对 Solar-Open-100B 为 2,048 token，对 Nemotron-3-Nano-30B-A3B 为 1,024 token。

使用 lm-evaluation-harness v0.4.11（Sutawika et al., 2026），从三个角度评估：

1. WikiText-2（Merity et al., 2017）上的困惑度。
2. 偏推理的生成式评估，采用基于生成的答案抽取（模型自由生成回答，再从文本里抽取最终答案），包括 MMLU-Pro（Wang et al., 2024）、GPQA-Diamond（Rein et al., 2024）、GSM-8K（Cobbe et al., 2021）。MMLU-Pro 与 GPQA-Diamond 使用 8,192 token 的思考预算。
3. 通用短文本理解，采用基于似然的多选评估（模型对每个候选答案按对数似然打分并选最高者），包括 ARC-C、ARC-E（Clark et al., 2018）、BoolQ（Clark et al., 2019）、HellaSwag（Zellers et al., 2019）、MMLU（Hendrycks et al., 2021）、PIQA（Bisk et al., 2020）、TruthfulQA（Lin et al., 2022）、WinoGrande（Sakaguchi et al., 2020）。

每类基准报告对应任务的平均分。所有生成式答案抽取评估都用贪心解码而非采样：目标不是通过解码调参去刷高分，而是衡量量化相对原模型引入的性能下降；贪心解码去除了采样带来的随机性，使量化效应的评估更可复现。

表 1：VSRAQ 超参数配置。

| 超参数 | 取值 |
|---|---|
| $`b`$ | 1 |
| $`\alpha`$ | 0.3 |
| $`\gamma`$ | 1.0 |
| $`\epsilon_{w}`$ | $`10^{-3}`$ |
| $`m_{\mathrm{rank}}`$ | 0.0 |
| $`\beta_{\mathrm{rank}}`$ | 0.01 |
| $`\beta_{\mathrm{margin}}`$ | 0.05 |
| $`\lambda_{\mathrm{router}}`$ | $`\lbrace 0.1, 0.01 \rbrace`$ |

超参数的设定原则，是让路由对齐损失与重建损失量级相当，并在校准期间产生相近量级的梯度。除路由损失系数 $`\lambda_{\mathrm{router}}`$ 外，所有模型和量化设置都用同一套超参：Solar-Open-100B 取 $`\lambda_{\mathrm{router}}=0.01`$ ，Nemotron-3-Nano-30B-A3B 取 $`\lambda_{\mathrm{router}}=0.1`$ 。配置汇总于表 1。VSRAQ 只在校准阶段使用，推理时不增加任何计算；因此除生成质量差异外，它与同精度的 AutoRound 量化模型有相同的内存占用与推理延迟。

表 2：Solar-Open-100B 在 W4A16 量化下的基准结果。PPL 为困惑度，Gen.-based Extract. 为基于生成的答案抽取，Likelihood-based MC 为基于似然的多选评估。PPL 越低越好，基准分越高越好。粗体表示量化方法中的最佳结果。

| Method | PPL | Gen.-based Extract. | Likelihood-based MC |
|---|---|---|---|
| Unquantized | 6.06 | 73.62 | 76.92 |
| AutoRound | 7.12 | 72.19 | 76.44 |
| TopK-MSE | 6.87 | 71.66 | 76.72 |
| VSRAQ | 6.81 | 73.22 | 76.76 |

表 3：Solar-Open-100B 在 NVFP4 量化下的基准结果。

| Method | PPL | Gen.-based Extract. | Likelihood-based MC |
|---|---|---|---|
| Unquantized | 6.06 | 73.62 | 76.92 |
| AutoRound | 7.22 | 61.78 | 75.59 |
| TopK-MSE | 7.10 | 63.86 | 75.21 |
| VSRAQ | 6.90 | 64.43 | 75.63 |

表 4：Nemotron-3-Nano-30B-A3B 在 NVFP4 量化下的基准结果。

| Method | PPL | Gen.-based Extract. | Likelihood-based MC |
|---|---|---|---|
| Unquantized | 10.36 | 73.64 | 70.33 |
| AutoRound | 10.79 | 69.10 | 69.96 |
| TopK-MSE | 10.78 | 69.47 | 69.59 |
| VSRAQ | 10.70 | 70.92 | 70.18 |

## 4.2 结果

表 2、3、4 汇总了实验结果。作者称 VSRAQ 在所有设置下都在量化方法中取得最佳表现。

对于基于似然的多选评估，4 比特量化相对未量化模型只有很小的性能下降，所以相对基线的绝对提升有限，但 VSRAQ 仍取得最高平均分，说明即便评估对小的路由差异不太敏感，所提出的路由感知目标依然有益。

相比之下，基于生成的答案抽取任务上的提升明显得多。这类任务要求模型先自由推理再抽取最终答案，对解码过程中累积的误差更敏感。在 MoE 模型中，每个生成步的路由决定了由哪些专家处理隐状态，而得到的隐状态又影响后续 token 的预测，所以量化引入的小幅专家选择失配会在长生成中累积放大。VSRAQ 在校准时显式保持路由行为，从而缓解这一效应，相对 AutoRound 和 TopK-MSE 变体取得更清晰的增益。

# 5 讨论

## 5.1 各个 VSRAQ 组件的作用

先分析 VSRAQ 的各组件是否对量化质量有贡献。表 5 给出逐步加入各组件时 WikiText-2 上的 PPL。

表 5：WikiText-2 PPL 的逐步消融。"+ Sigmoid" 在 TopK-MSE 之上加入 sigmoid 感知的值对齐，"+ Structure" 再加入结构对齐。越低越好。

| Method | Solar W4A16 | Solar NVFP4 | Nemotron NVFP4 |
|---|---|---|---|
| Unquantized | 6.06 | 6.06 | 10.36 |
| AutoRound | 7.12 | 7.22 | 10.79 |
| + TopK-MSE | 6.87 | 7.10 | 10.78 |
| + Sigmoid | 6.72 | 6.98 | 10.79 |
| + Structure | 6.81 | 6.90 | 10.70 |

TopK-MSE 损失稳定地优于基线，说明保持与路由相关的 logits 对 MoE 量化有用。加入 sigmoid 感知对齐后，Solar-Open-100B 的 PPL 进一步改善，暗示敏感度加权能在基于 sigmoid 的路由中更好地保持路由分数。完整的 VSRAQ 目标（另含结构对齐）在 NVFP4 设置下给出最佳 PPL，并且在所有情形下都显著好于基线。虽然各组件的效果并非在每种设置中严格单调，但整体趋势表明，值对齐与结构对齐都有助于更稳健的 MoE 量化。

## 5.2 专家选择一致性

为直接衡量 VSRAQ 是否保持路由决策，计算全精度模型与量化模型所选 top-$`k`$ 专家之间的一致程度。对每个 token 与每个 MoE 层，计算两组被选专家集合的 Jaccard 相似度：

```math
\mathrm{Jaccard}\left(\mathcal{T}^{\mathrm{fp}}_{k},\mathcal{T}^{q}_{k}\right)=\frac{\left|\mathcal{T}^{\mathrm{fp}}_{k}\cap\mathcal{T}^{q}_{k}\right|}{\left|\mathcal{T}^{\mathrm{fp}}_{k}\cup\mathcal{T}^{q}_{k}\right|}.
```
（式 18）

数据使用 Nemotron-Post-Training-Dataset-v2 的 stem、chat、math、code 子集中的 616,759 个 token（https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2）。

VSRAQ 在各层上都稳定提高了量化前后专家选择的一致程度；在差距最大处，其专家选择一致度相对基线的提升超过 11.29%。这一结果支持 VSRAQ 的核心动机：显式保持路由值与路由结构，会让量化后的专家选择更稳定。

![图 1：W4A16 量化下，全精度模型与量化模型所选 top-$`k`$ 专家集合的逐层 Jaccard 相似度。VSRAQ 比 AutoRound 与 TopK-MSE 更好地保持了专家选择一致性。](images/fig1_jaccard.png)

图中三条曲线（AutoRound、TopK-MSE、VSRAQ）横轴为层号（1 到 48），纵轴 Jaccard 相似度大致在 0.84 到 0.91 之间；AutoRound 在若干层（如第 2 层与第 8 层、第 13 层附近）明显偏低，TopK-MSE 与 VSRAQ 基本在其上方，VSRAQ 在前几层略高。

# 6 结论

本文提出 VSRAQ，一个面向 MoE 的 PTQ 目标，联合保持路由值与路由结构。相比仅重建和 TopK-MSE 基线，VSRAQ 改善了困惑度、下游基准表现和专家选择一致性，在路由失配会逐步累积的基于生成的评估上增益更明显。

作者指出，实验只覆盖 4 比特设置（W4A16 与 NVFP4），其量化噪声相对有限，路由感知校准的作用可能偏温和；预期在更激进的 2 比特或 3 比特场景里，专家选择的不稳定会更严重，收益也更大。该扩展以及对更多 MoE 架构与量化框架的更广泛评估留作未来工作。

# Impact Statement

本文目标是通过 MoE 专用 PTQ 推进 MoE 语言模型的高效部署。减少低比特量化带来的性能下降，可能让大型 MoE 模型以更低的内存与计算成本部署，有助于提高可及性与能效。但与其他大模型量化方法一样，量化引起的输出变化是否会影响不良行为（不道德文本、有偏输出、幻觉等）还需进一步研究；量化 MoE 模型的实际部署应当配合适当的安全性、鲁棒性与公平性评估。

# 参考文献

原文列有参考文献（Shazeer et al., 2017；Fedus et al., 2022；Frantar et al., 2023；Xiao et al., 2023；Lin et al., 2024；Cheng et al., 2024；Hu et al., 2025；Fu et al., 2025；Li et al., 2024；Chen et al., 2025；DeepSeek-AI, 2024；NVIDIA et al., 2025；GLM-4.5 Team, 2025 等），此处不再逐条复述，请见原文。
