> 中文翻译（全文）：Pengxiang Zhao 等，*SoScale: Second-Order Scale Search for FP4 Quantization*，arXiv:2611.16426v1 [cs.LG]，2026-11-12。原文链接：<https://arxiv.org/abs/2611.16426>（CC BY 4.0）。本文为忠实翻译，常用技术术语保留英文。

# SoScale：面向 FP4 量化的二阶 Scale 搜索

Pengxiang Zhao、Xing Li、Hui-Ling Zhen、Weizhe Lin、Xianzhi Yu、Zhenhua Dong

Huawei Technologies Co., Ltd.

## 摘要

低比特量化可以降低大语言模型（LLM）的显存占用和推理成本，但精确的 FP4 量化仍然具有挑战性。MXFP4、NVFP4 和 HiF4 等实用格式把 4 比特浮点 payload 与分块（block-wise）或分层（hierarchical）的 scale code 配对使用，使得精度对每个局部分布如何缩放高度敏感。默认的最大绝对值（maximum-absolute-value）缩放可以保留动态范围，但可能把稀缺的 FP4 取值层级浪费在极少被占用的离群值区域，使稠密的张量取值分辨率不足。现有的 scale 搜索方法最小化局部逐元素重建误差，优化的是张量保真度而非层输出保真度。我们提出 **SoScale**，一种面向分层 FP4 量化的二阶权重 scale code 搜索方法。SoScale 在 Hessian 感知的层重建目标下搜索硬件合法的 scale code，并将该搜索与 GPTQ 风格的误差恢复（error recovery）耦合，依据 scale 对层输出的影响来选择 scale。在 Qwen3 系列模型上，SoScale 在 MXFP4、NVFP4 和 HiF4 下一致地降低了下游任务的量化精度损失。

## 1 引言

大语言模型（LLM）已在广泛的自然语言处理任务上展现出强大能力（OpenAI 2026；DeepSeek-AI 2026；Qwen Team 2026），但其部署仍受制于巨大的显存占用和推理成本。因此，训练后量化（post-training quantization）已成为一项重要技术，可在无需昂贵重训练的情况下缩小模型体积并提升服务效率（Frantar et al. 2023；Xiao et al. 2023；Lin et al. 2024；Shao et al. 2024；Zhao and Yuan 2025）。在近期的低比特格式中，FP4 对 LLM 推理颇具吸引力：它把每个数值压缩到四比特，同时保留基于指数的动态范围，并支持低比特浮点计算的硬件加速。MXFP4（Rouhani et al. 2023）、NVFP4（Alvarez et al. 2025）和 HiF4（Luo et al. 2026）等实用 FP4 格式进一步引入了分层缩放机制，以更好地适应 LLM 权重和激活的非均匀分布。

尽管前景可观，采用分层 FP4 格式的精确量化对 scale code 的选择仍然高度敏感。由于 FP4 payload 只提供 16 个可表示的取值，block scale 决定了这些量化层级在实数轴上的位置。当一个 block 含有离群值时，标准的最大绝对值缩放规则会扩展网格以覆盖幅值最大的元素。这样虽保留了范围，却可能把层级花在极少被占用的区域，并降低张量取值稠密部分附近的分辨率。图 1(a) 在一个真实的 Qwen3-8B MXFP4 权重 block 上展示了这一效应。在默认缩放下，网格被离群值拉伸，但最外侧的两个可表示层级没有被分配到任何数值，因为离群值被舍入到了更靠内的层级。这使得零点附近真正有用的层级变少；而搜索得到的 scale 把网格向被占用区域收缩，提高了稠密取值的分辨率。图 1(b) 提供了互补的误差视角：对 block 内取值排序后，两端尾部的绝对误差几乎不变，而中间区域（对应接近零的取值）的误差在搜索得到的 scale 下显著下降。

![图 1：Scale code 对齐与模型层面的影响。（a）Scale code 对齐；（b）逐元素误差；（c）PPL 对比。在一个真实的 Qwen3-8B 权重 block 上，scale 搜索把 MXFP4 的 scale 从 0.125 改为 0.0625。（a）默认缩放保留了更宽的范围，但最外侧层级未被使用，而搜索得到的 scale 把层级集中到稠密的张量取值附近。（b）搜索得到的 scale 降低了大多数元素的逐元素重建误差。（c）局部逐元素 scale 搜索并不能稳定降低困惑度（perplexity），而 SoScale 可以。](images/fig1_scale_alignment.png)

上述例子表明，调整 scale 是降低 FP4 量化误差的有效手段，已有工作也探索了若干相关方向。一些方法在量化前重塑张量分布（Xiao et al. 2023；Ashkboos et al. 2024；Liu et al. 2025；Shao et al. 2025）；另一些方法在量化之后或量化过程中补偿残余的量化误差（Frantar et al. 2023；Meng et al. 2026）；还有 FP4 专用方法借助格式相关的启发式规则或局部失真目标来调整 block scale（Cook et al. 2026；Gupta et al. 2026）。这些方法表明，感知分布的 FP4 量化十分重要，但大多数 scale 选择目标仍然是局部的：它们依据某个 scale code 对一个 block 的重建效果来选择，尽管该 code 最终是通过对层输出的影响起作用的。如图 1(c) 所示，降低局部逐元素误差并不一定降低模型层面的困惑度。这促使我们把 scale 选择视为层重建问题，而不是局部张量拟合问题。

在这项工作中，我们提出 **SoScale**，一种二阶训练后量化（PTQ）方法，用于分层 FP4 量化中权重和激活 scale code 的选择。SoScale 作用于固定的预训练模型，无需重训练，也不修改目标 FP4 格式；它在该格式的合法 scale code 空间内搜索，并通过最小化 Hessian 加权的层重建损失来选择 code。在量化时，SoScale 把 scale 选择表述为一个顺序的二阶重建问题：每个 block 被确定（commit）之后，其引入的误差通过逆 Hessian 因子传播，使后续 block 在经过补偿的参数空间中优化，而不是针对原始权重优化（Frantar et al. 2023）。这种候选集视角使该方法与格式无关：给定某分层 FP4 格式的合法 scale code 集合和量化器，同样的 Hessian 感知搜索与恢复流程无需针对格式重新设计即可应用。由于权重量化离线完成，且得到的表示保持目标 FP4 编码，SoScale 不会引入服务时（serving-time）的搜索，并与已支持相应格式的硬件推理路径保持兼容。我们在 MXFP4、NVFP4 和 HiF4 上实例化这一通用表述，并且它与块 Hadamard 旋转（Shao et al. 2025）等预处理方法互补。

我们在 Qwen3-8B 和 Qwen3-14B（Yang et al. 2025）上，针对三种分层 FP4 格式（MXFP4、NVFP4 和 HiF4）评估 SoScale。在各下游基准上，最佳的 SoScale 变体在每个模型-格式组中都取得了最高的平均分，优于 RTN、GPTQ（Frantar et al. 2023）、局部 ScaleSearch（Gupta et al. 2026）和 BlockRotation（Ashkboos et al. 2024）基线。在 Qwen3-14B 上，这些变体在 MXFP4 下保留了 BF16 平均精度的 97.17%，NVFP4 下为 99.20%，HiF4 下为 99.40%。这些结果表明，利用二阶层信息优化 scale code，可以恢复局部张量级目标所遗漏的精度。

我们的贡献总结如下：

- 我们指出分层 FP4 量化中由离群值引起的 scale 失配：默认的保范围 scale 会把稀缺的 FP4 层级预留给没有被分配任何数值的区域，从而降低稠密权重取值的分辨率。
- 我们提出 **SoScale**，一种离线 scale code 优化器：它保持目标 FP4 格式不变，把 Hessian 加权的层重建与量化误差补偿结合起来，并保持与现有 FP4 硬件推理路径的兼容。
- 我们在两个 Qwen3 模型和三种 FP4 格式上，相对代表性基线展示了一致的提升。

## 2 相关工作

**分层 FP4 格式。** 低比特浮点格式在保留基于指数的动态范围的同时降低 LLM 推理成本，并支持硬件加速。包括 MXFP4（Rouhani et al. 2023）、NVFP4（Alvarez et al. 2025）和 HiF4（Luo et al. 2026）在内的实用 FP4 格式，都把 4 比特 payload 与分块或分层的 scale 配对；它们主要的结构差异总结于表 1。尽管在 scale 层级、粒度和码本上存在差异，这些格式共享同一个抽象：scale code 为每个 block 在实数轴上安放 FP4 payload 的层级。因此，当 block 分布呈重尾或含有离群值时，scale 选择就成为精度损失的主要来源。

| | MXFP4 | NVFP4 | HiF4 |
|---|---|---|---|
| Element format | E2M1 | E2M1 | E1M2 |
| Scale hierarchy | E8M0 | FP32 + E4M3 | E6M2 + E1 + E1 |
| Granularity | 32 | 16 + tensor | 64 / 8 / 4 |
| Bits / element | 4.25 | ≳ 4.50 | 4.50 |

表 1：本文所考虑的分层 FP4 格式的简要对比。

**训练后量化方法。** PTQ 在无需昂贵重训练的情况下降低预训练 LLM 的显存和计算成本。简单的就近舍入（round-to-nearest，RTN）量化是被广泛使用的高效基线，但在极低比特宽度下往往会降低精度。基于校准的 PTQ 方法借助激活统计、显著性估计或二阶近似来保持层的行为，从而提升低比特精度（Frantar et al. 2023；Lin et al. 2024；Shao et al. 2024；Zhao and Yuan 2025）。另一条互补路线是在量化前对权重和激活做平滑、旋转或重新平衡，以减轻由离群值带来的难度（Xiao et al. 2023；Ashkboos et al. 2024；Liu et al. 2025）。一条共同的经验是：量化误差应当按其对层输出的影响来衡量，而不仅仅是逐元素的张量失真。SoScale 采用这种以层为中心的视角，但将其应用于分层 FP4 格式的离散 scale code。

**FP4 方法。** 近期的 FP4 方法通常通过预处理、补偿或 scale 调整来处理对离群值的敏感性。块 Hadamard 旋转在量化前缩小每个 MXFP4 block，但依赖下游的 scale 规则（Shao et al. 2025）。ARCQuant 通过重排序和残差通道修复 NVFP4 的误差，增加的是补偿而非优化 scale（Meng et al. 2026）。最接近的工作直接调整 scale：Four Over Six 在两种 NVFP4 规则之间切换（Cook et al. 2026），ScaleSearch 则在 block scale 上最小化局部均方误差（MSE）（Gupta et al. 2026）。这些方法确立了 scale 选择的重要性，但仍受制于格式相关的规则或局部张量损失。SoScale 使用 Hessian 加权的层重建损失，在 MXFP4、NVFP4 和 HiF4 上搜索合法 scale code，并配合 GPTQ 恢复。

## 3 方法

本节介绍 **SoScale**，一种面向分层 FP4 量化的权重 scale code 搜索方法。对每个线性层，SoScale 用 Hessian 加权的层重建目标选择合法 scale code，并把分块量化表述为带误差恢复的顺序二阶重建过程。我们先给出 scale code 搜索的形式化，再描述误差恢复，最后总结完整流程。

### 3.1 Hessian 感知的 Scale Code 搜索

现代 LLM 以注意力和前馈模块中的线性投影为主，因此层重建是训练后量化的自然目标。考虑一个线性层，其校准输入为 $`\boldsymbol{X}\in\mathbb{R}^{T\times n}`$ ，全精度权重为 $`\boldsymbol{W}\in\mathbb{R}^{m\times n}`$ ：

```math
\boldsymbol{Y}=\boldsymbol{X}\boldsymbol{W}^{\top},\qquad(1)
```

其中 $`T`$ 表示校准 token 数， $`n`$ 和 $`m`$ 表示输入和输出维度。给定量化权重 $`\widehat{\boldsymbol{W}}`$ ，SoScale 用基于校准数据的层重建误差来衡量量化质量：

```math
\begin{aligned}
\mathcal{L}(\boldsymbol{W},\widehat{\boldsymbol{W}})&=\left\|\boldsymbol{X}\boldsymbol{W}^{\top}-\boldsymbol{X}\widehat{\boldsymbol{W}}^{\top}\right\|_{F}^{2}\\
&=\sum_{i=1}^{m}\boldsymbol{e}_{i}^{\top}\boldsymbol{H}\boldsymbol{e}_{i},
\end{aligned}\qquad(2)
```

其中 $`\boldsymbol{e}_{i}=\boldsymbol{w}_{i}-\widehat{\boldsymbol{w}}_{i}`$ ， $`\boldsymbol{w}_{i}`$ 和 $`\widehat{\boldsymbol{w}}_{i}`$ 分别是 $`\boldsymbol{W}`$ 和 $`\widehat{\boldsymbol{W}}`$ 的第 $`i`$ 行， $`\boldsymbol{H}=\boldsymbol{X}^{\top}\boldsymbol{X}`$ 为经验 Hessian。与逐元素失真度量不同，该目标并不平等对待所有权重误差：会引起校准后层输出更大变化的误差受到更大的惩罚。

重建目标刻画了期望的层行为，而目标 FP4 格式则定义了 $`\widehat{\boldsymbol{W}}`$ 的离散可行集。在分层 FP4 量化中， $`\boldsymbol{W}`$ 的每一行沿输入维度被划分为若干 block，每个 block 通过将低比特 FP4 payload 与一个或多个 scale code 结合来量化（如表 1 所总结）。令 $`\boldsymbol{w}_{i,b}\in\mathbb{R}^{B}`$ 表示第 $`i`$ 行的第 $`b`$ 个 block，其中 $`B`$ 是格式相关的 block 大小。对于目标格式 $`\mathcal{F}`$ ，我们把量化后的 block 写作

```math
\widehat{\boldsymbol{w}}_{i,b}(\boldsymbol{c})=Q_{\mathcal{F}}(\boldsymbol{w}_{i,b};\boldsymbol{c}),\qquad(3)
```

其中 $`Q_{\mathcal{F}}`$ 是格式相关的 FP4 量化器， $`\boldsymbol{c}\in\mathbb{C}_{\mathcal{F}}`$ 表示一个合法的 scale 配置。对每个候选 $`\boldsymbol{c}`$ ，payload 数值按目标格式的标准舍入与截断规则得到；SoScale 只改变使用哪些合法 scale code。为具体起见，考虑 MXFP4。每个 block 使用 E2M1 payload 和 E8M0 block-scale 指数。令 $`c\in\mathbb{C}_{\mathrm{MX}}`$ 表示指数 code， $`s(c)=2^{c}`$ 为对应的 block scale。量化后的 block 为

```math
\widehat{\boldsymbol{w}}_{i,b}(c)=s(c)\,\Pi_{\mathrm{E2M1}}\left(\frac{\boldsymbol{w}_{i,b}}{s(c)}\right),\qquad(4)
```

其中 $`\Pi_{\mathrm{E2M1}}(\cdot)`$ 表示向 E2M1 payload 码本的舍入与截断。对固定的 E2M1 码本，指数 code $`c`$ 诱导出缩放后的可表示集合 $`s(c)\mathcal{C}_{\mathrm{E2M1}}`$ ：增大 $`s(c)`$ 会扩大覆盖范围但使层级间距变粗，而减小 $`s(c)`$ 则收缩范围并提高较小幅值的分辨率，如图 1(a) 所示。同样的记号也适用于本文考虑的其他分层格式：对 NVFP4，SoScale 搜索 E4M3 block-scale code，同时保持 FP32 tensor scale 不变；对 HiF4，则搜索 E6M2 scale code，并据此推导 E1 scale。

在固定 FP4 payload 和合法 scale 空间之后，scale code 优化归结为为每个 block 选择一个 code。理想的选择是在所有 scale code 上最小化上述层重建损失，但完整 Hessian 会耦合不同 block 之间的误差，使联合搜索成为组合问题。因此 SoScale 采用与 FP4 布局对齐的 block 局部二阶代理目标。令 $`\mathbb{I}_{b}`$ 表示 block $`b`$ 的输入索引，并令

```math
\boldsymbol{H}_{b}=\boldsymbol{H}_{\mathbb{I}_{b},\mathbb{I}_{b}}\qquad(5)
```

为对应的 Hessian 子矩阵。对候选 scale 配置 $`\boldsymbol{c}`$ ，定义行-block 误差

```math
\boldsymbol{e}_{i,b}(\boldsymbol{c})=\boldsymbol{w}_{i,b}-\widehat{\boldsymbol{w}}_{i,b}(\boldsymbol{c}).\qquad(6)
```

SoScale 用下式对 $`\boldsymbol{c}`$ 打分：

```math
\ell_{i,b}(\boldsymbol{c})=\boldsymbol{e}_{i,b}(\boldsymbol{c})^{\top}\boldsymbol{H}_{b}\boldsymbol{e}_{i,b}(\boldsymbol{c}),\qquad(7)
```

矩阵 $`\boldsymbol{H}_{b}`$ 是该打分区别于局部 MSE 的关键：若 $`\boldsymbol{H}_{b}`$ 为单位矩阵，则 $`\ell_{i,b}`$ 退化为 block 误差的平方，而经验 Hessian 则根据校准激活对误差及其相关性加权。对每个行-block 对 $`(i,b)`$ ，SoScale 选择

```math
\boldsymbol{c}_{i,b}^{\star}=\arg\min_{\boldsymbol{c}\in\mathbb{C}_{i,b}}\ell_{i,b}(\boldsymbol{c}),\qquad(8)
```

其中 $`\mathbb{C}_{i,b}`$ 是在默认 scale code 附近取一个小的、格式相关的偏移窗口，并把无效 code 截断到目标格式合法范围内而形成的候选集。

### 3.2 二阶误差恢复

![图 2：SoScale 对单个线性层的工作流程。校准阶段构建经验 Hessian 和带阻尼的逆 Hessian 因子。在 2.1 中，SoScale 用 Hessian 感知的层损失搜索合法 FP4 scale code，而 ScaleSearch 和 Four Over Six 等仅调整 scale 的基线使用不含 Hessian 信息的局部 MSE 式目标；在 2.2 中，它把所选 block 的误差传播给后续 block。推理只使用存储的 FP4 payload、合法 scale code 和轻量的逐 token 激活 scale。](images/fig2_workflow.png)

上一小节定义了如何为固定的一个权重 block 选择 scale code。然而在顺序量化中，确定一个 block 会引入误差，而这一误差可以被剩余未量化的 block 部分吸收。如果后续 block 仍在原始权重上搜索，它们的 scale 选择就会忽略这种补偿。这种耦合对分层 FP4 格式是自然的，因为 scale 选择和误差恢复都在 block 层面进行。因此 SoScale 遵循 GPTQ 的原则（Frantar et al. 2023）：每个 block 量化之后，向后续 block 传播二阶修正，并在经补偿的权重上进行下一次 scale code 搜索。

令 $`\widetilde{\boldsymbol{W}}`$ 表示权重的工作副本，初始化为 $`\widetilde{\boldsymbol{W}}=\boldsymbol{W}`$ 。与 GPTQ 一致，我们计算带阻尼逆 Hessian 的上三角因子 $`\boldsymbol{U}`$ ，

```math
\boldsymbol{U}^{\top}\boldsymbol{U}=\left(\boldsymbol{H}+\lambda\boldsymbol{I}\right)^{-1},\qquad(9)
```

其中 $`\lambda`$ 是阻尼系数。在某一顺序步骤，令 $`\mathbb{Q}\subset\lbrace 1,\ldots,n\rbrace`$ 表示当前 FP4 block 的输入列索引， $`\mathbb{F}`$ 表示属于剩余后续 block 的列。切片 $`\widetilde{\boldsymbol{W}}_{:,\mathbb{Q}}`$ 包含所有输出行的当前 block。切片中的每一行有各自的 scale code 或 scale 配置，我们把这些逐行的 code 汇总为 $`\boldsymbol{c}_{\mathbb{Q}}`$ 。在确定该 block 之前，SoScale 在经补偿的权重上搜索这些 code：

```math
\boldsymbol{c}_{\mathbb{Q}}^{\star}=\arg\min_{\boldsymbol{c}_{\mathbb{Q}}\in\mathbb{C}_{\mathbb{Q}}}\ell_{\mathbb{Q}}(\boldsymbol{c}_{\mathbb{Q}}).\qquad(10)
```

这里 $`\ell_{\mathbb{Q}}`$ 是上一小节的行-block 分数在所有输出行上的求和。使用选出的 scale code，SoScale 确定量化 block

```math
\widehat{\boldsymbol{W}}_{:,\mathbb{Q}}=Q_{\mathcal{F}}\left(\widetilde{\boldsymbol{W}}_{:,\mathbb{Q}};\boldsymbol{c}_{\mathbb{Q}}^{\star}\right),\qquad(11)
```

并得到 block 误差

```math
\boldsymbol{E}_{\mathbb{Q}}=\widetilde{\boldsymbol{W}}_{:,\mathbb{Q}}-\widehat{\boldsymbol{W}}_{:,\mathbb{Q}}.\qquad(12)
```

随后通过回代（back-substitution）把 block 误差传播给后续 block：

```math
\widetilde{\boldsymbol{W}}_{:,\mathbb{F}}\leftarrow\widetilde{\boldsymbol{W}}_{:,\mathbb{F}}-\boldsymbol{E}_{\mathbb{Q}}\boldsymbol{U}_{\mathbb{Q},\mathbb{Q}}^{-1}\boldsymbol{U}_{\mathbb{Q},\mathbb{F}}.\qquad(13)
```

这里 $`\boldsymbol{U}_{\mathbb{Q},\mathbb{Q}}`$ 和 $`\boldsymbol{U}_{\mathbb{Q},\mathbb{F}}`$ 是 $`\boldsymbol{U}`$ 中由当前 block 与后续输入维度索引的子矩阵。该更新只改变用于后续 block 的工作副本 $`\widetilde{\boldsymbol{W}}`$ ；已确定的 block $`\widehat{\boldsymbol{W}}_{:,\mathbb{Q}}`$ 保持不变。因此，scale code 搜索与误差恢复交错进行：每个 block 都在已包含此前量化误差的二阶补偿的权重上选择 scale。

### 3.3 动态逐 token 激活量化

SoScale 优化的是权重 block 的 scale code。激活则由目标格式下的在线动态逐 token FP4 规则处理。对 token $`t`$ ，令 $`\boldsymbol{x}_{t}`$ 表示其激活向量， $`\boldsymbol{x}_{t,b}`$ 表示目标格式下的第 $`b`$ 个 block。每个激活 block 按下式量化：

```math
\widehat{\boldsymbol{x}}_{t,b}=Q_{\mathcal{F}}^{\mathrm{act}}\left(\boldsymbol{x}_{t,b};c^{\mathrm{act}}_{t,b}\right),\qquad c^{\mathrm{act}}_{t,b}=D_{\mathcal{F}}(\boldsymbol{x}_{t,b}),\qquad(14)
```

其中 $`D_{\mathcal{F}}`$ 是格式相关的激活动态 scale 规则。由于激活 scale 要对每个 token 在线计算， $`D_{\mathcal{F}}`$ 保持轻量，并使用逐元素 MSE 准则为每个 token block 选择 $`c^{\mathrm{act}}_{t,b}`$ 。

### 3.4 SoScale 整体框架

**算法 1：单个线性层的 SoScale**

- 输入：全精度权重 $`\boldsymbol{W}`$ ，校准输入 $`\boldsymbol{X}`$ ，目标 FP4 格式 $`\mathcal{F}`$ ，阻尼 $`\lambda`$ ，可选变换 $`\boldsymbol{R}`$
- 输出：量化权重 $`\widehat{\boldsymbol{W}}`$ 和 scale code 设置 $`\boldsymbol{c}^{\star}`$

1. 计算经验 Hessian $`\boldsymbol{H}\leftarrow\boldsymbol{X}^{\top}\boldsymbol{X}`$
2. 若使用预处理变换 $`\boldsymbol{R}`$ ：
3. 　变换权重和 Hessian： $`\boldsymbol{W}\leftarrow\boldsymbol{W}\boldsymbol{R}`$ ， $`\boldsymbol{H}\leftarrow\boldsymbol{R}^{\top}\boldsymbol{H}\boldsymbol{R}`$
4. 结束 if
5. 初始化工作权重 $`\widetilde{\boldsymbol{W}}\leftarrow\boldsymbol{W}`$ 、量化权重 $`\widehat{\boldsymbol{W}}`$ 和 scale code $`\boldsymbol{c}^{\star}`$
6. 计算上三角因子 $`\boldsymbol{U}^{\top}\boldsymbol{U}=(\boldsymbol{H}+\lambda\boldsymbol{I})^{-1}`$
7. 对每个 FP4 block $`\mathbb{Q}`$ 及其后续 block 索引集 $`\mathbb{F}`$ ：
8. 　构造合法候选集 $`\mathbb{C}_{\mathbb{Q}}`$
9. 　在权重上搜索： $`\boldsymbol{c}_{\mathbb{Q}}^{\star}\leftarrow\arg\min_{\boldsymbol{c}_{\mathbb{Q}}\in\mathbb{C}_{\mathbb{Q}}}\ell_{\mathbb{Q}}(\boldsymbol{c}_{\mathbb{Q}})`$
10. 　确定量化 block： $`\widehat{\boldsymbol{W}}_{:,\mathbb{Q}}\leftarrow Q_{\mathcal{F}}(\widetilde{\boldsymbol{W}}_{:,\mathbb{Q}};\boldsymbol{c}_{\mathbb{Q}}^{\star})`$
11. 　计算 block 误差 $`\boldsymbol{E}_{\mathbb{Q}}\leftarrow\widetilde{\boldsymbol{W}}_{:,\mathbb{Q}}-\widehat{\boldsymbol{W}}_{:,\mathbb{Q}}`$
12. 　把误差传播给后续 block： $`\widetilde{\boldsymbol{W}}_{:,\mathbb{F}}\leftarrow\widetilde{\boldsymbol{W}}_{:,\mathbb{F}}-\boldsymbol{E}_{\mathbb{Q}}\boldsymbol{U}_{\mathbb{Q},\mathbb{Q}}^{-1}\boldsymbol{U}_{\mathbb{Q},\mathbb{F}}`$
13. 　把 $`\boldsymbol{c}_{\mathbb{Q}}^{\star}`$ 存入 $`\boldsymbol{c}^{\star}`$
14. 结束 for
15. 返回 $`\widehat{\boldsymbol{W}}`$ 和 $`\boldsymbol{c}^{\star}`$

算法 1 总结了单个线性层的 SoScale，图 2 通过把 Hessian 感知搜索（2.1）与误差恢复（2.2）分开，将同一工作流程可视化。该算法利用校准激活构建经验 Hessian，可选地对权重和 Hessian 同时施加预处理变换，然后顺序地处理 FP4 输入 block。对每个 block，在当前经补偿的权重上搜索 scale code，确定所选的量化 block，并通过回代把其误差传播给后续 block。ScaleSearch（Gupta et al. 2026）和 Four Over Six（Cook et al. 2026）等仅调整 scale 的基线主要对应于 2.1 的搜索阶段，且优化的是不含 Hessian 信息的局部 MSE 式目标；SoScale 则用 Hessian 感知的层损失对合法 scale code 打分，然后应用 2.2 的恢复来更新后续 block。输出是目标格式的 FP4 表示，由量化 payload 和合法 scale code 组成。该权重量化过程在校准期间离线完成；推理只使用存储的 FP4 payload 和 scale code。

## 4 实验

我们在 W4A4 量化下，针对三种分层 FP4 格式（MXFP4、NVFP4 和 HiF4），在 Qwen3 系列 LLM 上评估 SoScale，并报告相对代表性基线的下游基准精度。

### 4.1 实验设置

**模型。** 我们在 Qwen3-8B 和 Qwen3-14B（Yang et al. 2025）上评估 SoScale。

**量化格式。** 我们评估 MXFP4、NVFP4 和 HiF4，即表 1 所总结的三种分层 FP4 格式。主要设置为 W4A4：权重由参与比较的权重量化器量化，激活使用方法部分所述的在线逐 token FP4 规则。我们量化 transformer block 中的线性层，而 token embedding、LM head 和归一化层保持为 BF16。我们还报告了 scale 搜索方法的直接激活（direct-activation）设置，其中激活使用直接 RTN 量化；这样可以隔离权重侧的贡献，同时使激活路径更接近硬件原生的 FP4 执行。

**基线。** 我们使用 **BF16** 作为全精度参考。**RTN** 表示默认的、格式相关的就近舍入 FP4 量化器。**GPTQ** 表示 GPTQ 风格的二阶权重量化（Frantar et al. 2023）。由于 GPTQ 原本是仅权重的 PTQ 方法，在主要的 W4A4 对比中，我们把它量化得到的权重与上述在线逐 token 激活规则配对使用。**ScaleSearch** 表示最小化逐元素 MSE 的局部 block-scale 搜索（Gupta et al. 2026）。**BlockRotation** 表示为 MXFP4 提出的独立块 Hadamard 预处理方法（Shao et al. 2025）；在我们的实验中，对 MXFP4、NVFP4 和 HiF4 都采用同样的预处理。在表格中，后缀 **w/ Had.** 表示在相应量化方法之前应用块 Hadamard 预处理，遵循 QuaRot（Ashkboos et al. 2024）和 block rotation（Shao et al. 2025）。我们在结果表中一致地使用这些名称。

**校准与实现。** 对需要校准的方法，我们从 WikiText 的第一个分片（Merity et al. 2016）中采样 128 条、每条 2,048 个 token 的序列，遵循广泛使用的 PTQ 设置（Frantar et al. 2023）。对 SoScale，这些校准激活用于估计经验 Hessian，以进行 Hessian 感知的 scale code 搜索。我们从公开的 Hugging Face 仓库加载 Qwen3 checkpoint 和 tokenizer，使用 `datasets`（Lhoest et al. 2021）和 `transformers`（Wolf et al. 2020）进行数据预处理和模型执行，并用 PyTorch（Paszke et al. 2019）实现所有量化流程。

**评测基准。** 我们使用 Language Model Evaluation Harness（Gao et al. 2024），以 batch size 64 在 GSM8K（Cobbe et al. 2021）、HellaSwag（Zellers et al. 2019）、MMLU（Hendrycks et al. 2021）、ARC-Challenge 和 ARC-Easy（Clark et al. 2018）、BoolQ（Clark et al. 2019）以及 PIQA（Bisk et al. 2020）上评估量化模型。GSM8K 报告 exact-match 精度，其余任务报告 accuracy。除非另有说明，GSM8K 使用 5-shot 设置，其他基准使用 zero-shot 设置。我们报告各任务的平均分及其相对于 BF16 参考的比值。

### 4.2 主要结果

| Model | Bit | Format | Method | GSM8K | HellaSwag | MMLU | ARC-C | ARC-E | BoolQ | PIQA | Avg. | Rel. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | – | BF16 | – | 0.8810 | 0.5713 | 0.7292 | 0.5538 | 0.8371 | 0.8663 | 0.7628 | 0.7431 | 1.0000 |
| Qwen3-8B | W4A4 | MXFP4 | RTN | 0.8211 | 0.5292 | 0.6718 | 0.4923 | 0.7735 | 0.8425 | 0.7492 | 0.6971 | 0.9381 |
| Qwen3-8B | W4A4 | MXFP4 | GPTQ | 0.8362 | 0.5329 | 0.6862 | 0.5068 | 0.7971 | 0.8520 | 0.7514 | 0.7089 | 0.9540 |
| Qwen3-8B | W4A4 | MXFP4 | BlockRotation | 0.7839 | 0.5329 | 0.6772 | 0.5077 | 0.7921 | 0.8440 | 0.7421 | 0.6971 | 0.9381 |
| Qwen3-8B | W4A4 | MXFP4 | ScaleSearch | 0.8317 | 0.5303 | 0.6802 | 0.5171 | 0.7837 | 0.8505 | 0.7459 | 0.7056 | 0.9496 |
| Qwen3-8B | W4A4 | MXFP4 | SoScale | 0.8575 | 0.5334 | 0.6924 | 0.5000 | 0.7875 | 0.8517 | 0.7421 | 0.7092 | 0.9545 |
| Qwen3-8B | W4A4 | MXFP4 | SoScale w/ Had. | 0.8271 | 0.5455 | 0.7010 | 0.5137 | 0.8072 | 0.8636 | 0.7584 | **0.7166** | 0.9644 |
| Qwen3-8B | W4A4 | NVFP4 | RTN | 0.8544 | 0.5542 | 0.7029 | 0.5401 | 0.8178 | 0.8560 | 0.7579 | 0.7262 | 0.9773 |
| Qwen3-8B | W4A4 | NVFP4 | GPTQ | 0.8582 | 0.5515 | 0.7117 | 0.5247 | 0.8178 | 0.8584 | 0.7573 | 0.7257 | 0.9765 |
| Qwen3-8B | W4A4 | NVFP4 | BlockRotation | 0.8423 | 0.5439 | 0.6966 | 0.5187 | 0.8093 | 0.8523 | 0.7600 | 0.7176 | 0.9657 |
| Qwen3-8B | W4A4 | NVFP4 | ScaleSearch | 0.8620 | 0.5551 | 0.7100 | 0.5324 | 0.8215 | 0.8621 | 0.7535 | 0.7281 | 0.9798 |
| Qwen3-8B | W4A4 | NVFP4 | SoScale | 0.8787 | 0.5556 | 0.7153 | 0.5401 | 0.8349 | 0.8664 | 0.7579 | **0.7356** | 0.9899 |
| Qwen3-8B | W4A4 | NVFP4 | SoScale w/ Had. | 0.8582 | 0.5543 | 0.7204 | 0.5384 | 0.8161 | 0.8651 | 0.7628 | 0.7308 | 0.9834 |
| Qwen3-8B | W4A4 | HiF4 | RTN | 0.8529 | 0.5555 | 0.7053 | 0.5119 | 0.8056 | 0.8541 | 0.7622 | 0.7211 | 0.9704 |
| Qwen3-8B | W4A4 | HiF4 | GPTQ | 0.8643 | 0.5577 | 0.7109 | 0.5273 | 0.8173 | 0.8630 | 0.7617 | 0.7289 | 0.9809 |
| Qwen3-8B | W4A4 | HiF4 | BlockRotation | 0.8567 | 0.5475 | 0.7058 | 0.5154 | 0.8081 | 0.8587 | 0.7584 | 0.7215 | 0.9710 |
| Qwen3-8B | W4A4 | HiF4 | ScaleSearch | 0.8552 | 0.5587 | 0.7093 | 0.5213 | 0.8081 | 0.8554 | 0.7617 | 0.7242 | 0.9747 |
| Qwen3-8B | W4A4 | HiF4 | SoScale | 0.8696 | 0.5582 | 0.7154 | 0.5299 | 0.8165 | 0.8590 | 0.7579 | 0.7295 | 0.9817 |
| Qwen3-8B | W4A4 | HiF4 | SoScale w/ Had. | 0.8605 | 0.5563 | 0.7163 | 0.5367 | 0.8161 | 0.8694 | 0.7617 | **0.7310** | 0.9838 |
| Qwen3-14B | – | BF16 | – | 0.8863 | 0.6094 | 0.7710 | 0.5845 | 0.8418 | 0.8936 | 0.8020 | 0.7698 | 1.0000 |
| Qwen3-14B | W4A4 | MXFP4 | RTN | 0.8408 | 0.5792 | 0.7243 | 0.5623 | 0.8186 | 0.8728 | 0.7671 | 0.7379 | 0.9585 |
| Qwen3-14B | W4A4 | MXFP4 | GPTQ | 0.8711 | 0.5794 | 0.7313 | 0.5546 | 0.8157 | 0.8804 | 0.7753 | 0.7440 | 0.9664 |
| Qwen3-14B | W4A4 | MXFP4 | BlockRotation | 0.8400 | 0.5759 | 0.7281 | 0.5469 | 0.8186 | 0.8798 | 0.7780 | 0.7382 | 0.9589 |
| Qwen3-14B | W4A4 | MXFP4 | ScaleSearch | 0.8522 | 0.5839 | 0.7334 | 0.5631 | 0.8119 | 0.8789 | 0.7677 | 0.7416 | 0.9633 |
| Qwen3-14B | W4A4 | MXFP4 | SoScale | 0.8726 | 0.5811 | 0.7321 | 0.5691 | 0.8215 | 0.8865 | 0.7726 | 0.7479 | 0.9716 |
| Qwen3-14B | W4A4 | MXFP4 | SoScale w/ Had. | 0.8658 | 0.5898 | 0.7414 | 0.5546 | 0.8203 | 0.8847 | 0.7797 | **0.7480** | 0.9717 |
| Qwen3-14B | W4A4 | NVFP4 | RTN | 0.8757 | 0.5937 | 0.7496 | 0.5529 | 0.8224 | 0.8887 | 0.7791 | 0.7517 | 0.9765 |
| Qwen3-14B | W4A4 | NVFP4 | GPTQ | 0.8719 | 0.5993 | 0.7554 | 0.5546 | 0.8338 | 0.8850 | 0.7922 | 0.7560 | 0.9821 |
| Qwen3-14B | W4A4 | NVFP4 | BlockRotation | 0.8817 | 0.5847 | 0.7418 | 0.5478 | 0.8047 | 0.8826 | 0.7927 | 0.7480 | 0.9717 |
| Qwen3-14B | W4A4 | NVFP4 | ScaleSearch | 0.8772 | 0.5976 | 0.7550 | 0.5683 | 0.8253 | 0.8856 | 0.7862 | 0.7565 | 0.9827 |
| Qwen3-14B | W4A4 | NVFP4 | SoScale | 0.8901 | 0.5991 | 0.7580 | 0.5811 | 0.8291 | 0.8890 | 0.7851 | 0.7616 | 0.9894 |
| Qwen3-14B | W4A4 | NVFP4 | SoScale w/ Had. | 0.8954 | 0.5956 | 0.7615 | 0.5802 | 0.8338 | 0.8896 | 0.7894 | **0.7636** | 0.9920 |
| Qwen3-14B | W4A4 | HiF4 | RTN | 0.8749 | 0.5987 | 0.7564 | 0.5657 | 0.8333 | 0.8936 | 0.7922 | 0.7593 | 0.9863 |
| Qwen3-14B | W4A4 | HiF4 | GPTQ | 0.8946 | 0.5993 | 0.7564 | 0.5700 | 0.8304 | 0.8896 | 0.7889 | 0.7613 | 0.9890 |
| Qwen3-14B | W4A4 | HiF4 | BlockRotation | 0.8795 | 0.5927 | 0.7512 | 0.5794 | 0.8367 | 0.8893 | 0.7905 | 0.7599 | 0.9871 |
| Qwen3-14B | W4A4 | HiF4 | ScaleSearch | 0.8772 | 0.6007 | 0.7558 | 0.5700 | 0.8375 | 0.8899 | 0.7927 | 0.7605 | 0.9880 |
| Qwen3-14B | W4A4 | HiF4 | SoScale | 0.8840 | 0.5965 | 0.7559 | 0.5922 | 0.8422 | 0.8911 | 0.7943 | **0.7652** | 0.9940 |
| Qwen3-14B | W4A4 | HiF4 | SoScale w/ Had. | 0.8954 | 0.6000 | 0.7595 | 0.5700 | 0.8392 | 0.8890 | 0.7884 | 0.7631 | 0.9913 |

表 2：W4A4 分层 FP4 量化下的下游精度；Avg. 为各任务的平均值，Rel. 为相对于同一模型 BF16 平均值的归一化比值。原文中 SoScale 变体行有高亮，每个模型-格式组内 Avg. 最优者加粗，次优者加下划线（此处仅保留加粗，不保留高亮与下划线）。

**SoScale 提升下游精度。** 表 2 给出主要 W4A4 设置下的下游精度。在两个 Qwen3 模型和全部三种 FP4 格式上，每个模型-格式组中都有一个 SoScale 变体取得最佳平均分，在 MXFP4 下保留 BF16 平均精度的 96.44%–97.17%，NVFP4 下为 98.99%–99.20%，HiF4 下为 98.38%–99.40%。与每组中最强的非 SoScale 基线相比，最佳 SoScale 变体将平均分提高 0.21 到 0.77 个百分点；相对于局部 ScaleSearch，提升为 0.47 到 1.10 个点。MXFP4 下提升最大，因为其 E8M0 scale code 只允许粗粒度的 2 的幂调整：离群值可能迫使 FP4 网格变粗，使稠密取值分辨率不足，从而为 Hessian 感知的 scale 选择留下更大空间。NVFP4 和 HiF4 的基线更强、提升空间更小，但 SoScale 仍然改进了它们，这表明即使 scale 空间更精细，层级敏感度依然有用。

**Hadamard 的收益取决于格式。** 单独的 BlockRotation 带来的改进好坏参半：它在某些模型-格式组中有帮助，但并不总是优于 ScaleSearch，有时甚至落后于 RTN。把 Hadamard 预处理与 SoScale 结合，对 MXFP4 最有用，在 Qwen3-8B 和 Qwen3-14B 上都取得最佳平均分，在 Qwen3-14B 的 NVFP4 上也取得最佳结果。然而，最优行并不总是 Hadamard 变体：Qwen3-8B 的 NVFP4 和 Qwen3-14B 的 HiF4 都是单独的 SoScale 最好。这种对格式的依赖与近期的微缩放（microscaling）FP4 分析一致，该分析强调了 MXFP4 的 2 的幂 scale 误差以及 NVFP4 独特的小分组行为（Egiazarian et al. 2026）。总体而言，预处理改变的是呈现给量化器的分布，而 SoScale 选择的是最终 FP4 表示中使用的硬件合法 scale code。这解释了为什么旋转可以帮助 MXFP4，却无法取代 Hessian 感知的 scale 选择。

| Model | Bit | Format | Method | GSM8K | HellaSwag | MMLU | ARC-C | ARC-E | BoolQ | PIQA | Avg. | Rel. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | W4A4 | MXFP4 | ScaleSearch-w | 0.8241 | 0.5273 | 0.6738 | 0.5009 | 0.7774 | 0.8492 | 0.7492 | 0.7003 | 0.9424 |
| Qwen3-8B | W4A4 | MXFP4 | SoScale-w | 0.8317 | 0.5292 | 0.6856 | 0.5000 | 0.7997 | 0.8508 | 0.7405 | 0.7054 | 0.9492 |
| Qwen3-8B | W4A4 | MXFP4 | SoScale-w w/ Had. | 0.8461 | 0.5438 | 0.6966 | 0.5205 | 0.8064 | 0.8635 | 0.7601 | **0.7195** | 0.9683 |
| Qwen3-8B | W4A4 | NVFP4 | ScaleSearch-w | 0.8605 | 0.5533 | 0.7124 | 0.5341 | 0.8114 | 0.8575 | 0.7655 | 0.7278 | 0.9795 |
| Qwen3-8B | W4A4 | NVFP4 | SoScale-w | 0.8726 | 0.5542 | 0.7082 | 0.5384 | 0.8157 | 0.8602 | 0.7459 | 0.7279 | 0.9796 |
| Qwen3-8B | W4A4 | NVFP4 | SoScale-w w/ Had. | 0.8560 | 0.5541 | 0.7121 | 0.5410 | 0.8136 | 0.8657 | 0.7579 | **0.7286** | 0.9806 |
| Qwen3-8B | W4A4 | HiF4 | ScaleSearch-w | 0.8400 | 0.5567 | 0.7074 | 0.5060 | 0.8035 | 0.8514 | 0.7650 | 0.7186 | 0.9670 |
| Qwen3-8B | W4A4 | HiF4 | SoScale-w | 0.8613 | 0.5574 | 0.7124 | 0.5230 | 0.8144 | 0.8636 | 0.7557 | 0.7268 | 0.9781 |
| Qwen3-8B | W4A4 | HiF4 | SoScale-w w/ Had. | 0.8582 | 0.5591 | 0.7140 | 0.5444 | 0.8186 | 0.8700 | 0.7655 | **0.7328** | 0.9861 |
| Qwen3-14B | W4A4 | MXFP4 | ScaleSearch-w | 0.8567 | 0.5795 | 0.7328 | 0.5469 | 0.8140 | 0.8670 | 0.7584 | 0.7365 | 0.9567 |
| Qwen3-14B | W4A4 | MXFP4 | SoScale-w | 0.8749 | 0.5762 | 0.7318 | 0.5700 | 0.8249 | 0.8813 | 0.7715 | 0.7472 | 0.9707 |
| Qwen3-14B | W4A4 | MXFP4 | SoScale-w w/ Had. | 0.8673 | 0.5877 | 0.7421 | 0.5555 | 0.8236 | 0.8853 | 0.7829 | **0.7492** | 0.9732 |
| Qwen3-14B | W4A4 | NVFP4 | ScaleSearch-w | 0.8901 | 0.5970 | 0.7569 | 0.5631 | 0.8237 | 0.8829 | 0.7753 | 0.7556 | 0.9815 |
| Qwen3-14B | W4A4 | NVFP4 | SoScale-w | 0.8878 | 0.5971 | 0.7584 | 0.5794 | 0.8249 | 0.8902 | 0.7829 | 0.7601 | 0.9874 |
| Qwen3-14B | W4A4 | NVFP4 | SoScale-w w/ Had. | 0.8878 | 0.5975 | 0.7587 | 0.5768 | 0.8333 | 0.8911 | 0.7813 | **0.7609** | 0.9885 |
| Qwen3-14B | W4A4 | HiF4 | ScaleSearch-w | 0.8825 | 0.6002 | 0.7528 | 0.5717 | 0.8316 | 0.8838 | 0.7835 | 0.7580 | 0.9847 |
| Qwen3-14B | W4A4 | HiF4 | SoScale-w | 0.8893 | 0.5954 | 0.7578 | 0.5657 | 0.8413 | 0.8917 | 0.7905 | 0.7617 | 0.9894 |
| Qwen3-14B | W4A4 | HiF4 | SoScale-w w/ Had. | 0.8946 | 0.6008 | 0.7609 | 0.5836 | 0.8380 | 0.8902 | 0.7954 | **0.7662** | 0.9953 |

表 3：采用直接 RTN 激活量化的 W4A4 对比。后缀“-w”表示使用权重侧 scale 搜索、同时激活走直接 RTN 路径的变体；w/ Had. 表示在权重量化之前进行块 Hadamard 预处理。原文中 SoScale 变体的单元格有高亮，每个模型-格式组内最佳平均精度加粗（此处仅保留加粗）。

**直接激活 RTN 依然可行。** 表 3 评估一个面向部署的变体，其中激活使用直接 RTN 量化，而不是激活侧的 scale 搜索。该设置使激活路径与硬件支持的 FP4 格式保持一致，并避免在运行时引入格式相关的激活搜索。在这种直接激活路径下，最佳 SoScale-w 变体在每个模型-格式组中仍然优于 ScaleSearch-w，平均精度提升范围为 0.08 到 1.92 个百分点。Hadamard 预处理在此依然有用，因为分别施加在权重和激活上的成对正交变换保持了线性映射不变，同时在乘法两侧都分散了离群值。当激活走直接 RTN 路径、没有激活侧搜索来纠正由离群值引起的 scale 选择时，这一点尤其有帮助；最明显的提升同样出现在 MXFP4 上。一些直接 RTN 行也超过了表 2 中对应的行。这并不意味着直接激活 RTN 总是更好；而是因为在线激活规则是局部启发式，没有与权重 scale code 或下游损失联合优化。当激活已经与目标 FP4 格式匹配良好时，避免这种额外的激活侧调整可能是有益的，而 SoScale 继续解决占主导地位的权重侧 scale code 误差。

| Format | Variant | Loss | Rec. | Avg. | Rel. |
|---|---|---|---|---|---|
| MXFP4 | RTN | N/A | No | 0.6971 | 0.9381 |
| MXFP4 | ScaleSearch | MSE | No | 0.7056 | 0.9496 |
| MXFP4 | MSE + recovery | MSE | ✓ | 0.7061 | 0.9503 |
| MXFP4 | SoScale | Hessian | ✓ | **0.7092** | 0.9545 |
| NVFP4 | RTN | N/A | No | 0.7262 | 0.9773 |
| NVFP4 | ScaleSearch | MSE | No | 0.7281 | 0.9798 |
| NVFP4 | MSE + recovery | MSE | ✓ | 0.7270 | 0.9784 |
| NVFP4 | SoScale | Hessian | ✓ | **0.7356** | 0.9899 |
| HiF4 | RTN | N/A | No | 0.7211 | 0.9704 |
| HiF4 | ScaleSearch | MSE | No | 0.7242 | 0.9747 |
| HiF4 | MSE + recovery | MSE | ✓ | 0.7263 | 0.9774 |
| HiF4 | SoScale | Hessian | ✓ | **0.7295** | 0.9817 |

表 4：Qwen3-8B 在 W4A4 量化下的消融；Rec. 表示图 2 中 2.2 的误差传播阶段。每个格式组内 Avg. 最优者加粗。

**对搜索与恢复的消融。** 表 4 在固定 FP4 格式下的 Qwen3-8B 上，比较 MSE scale 搜索、带 block 级恢复的 MSE 搜索以及完整的 SoScale。给 MSE 目标加上恢复，相对于 ScaleSearch 的收益好坏参半：在 MXFP4 下提升很小，在 NVFP4 下反而下降，在 HiF4 下有所提升。相比之下，完整的 SoScale 在三种格式上都是最优，说明恢复与 Hessian 感知的 scale code 选择耦合时最有效，而不是与局部 MSE 目标耦合。

## 5 结论

我们提出了 **SoScale**，一种面向分层 FP4 量化的二阶 scale code 搜索方法。SoScale 解决了低比特浮点格式的一个实际失效模式：保范围的或局部 MSE 最优的 scale 可能使 FP4 网格相对于层输出而言放置不当，尤其是当离群值与离散的 scale code 相互作用时。通过在 Hessian 感知的层重建目标下优化硬件合法的权重 scale code，并将该搜索与 GPTQ 风格的误差恢复交错进行，SoScale 在不改变目标 FP4 表示的前提下改进了 W4A4 量化。在 Qwen3-8B 和 Qwen3-14B 上、跨 MXFP4、NVFP4 和 HiF4 的实验表明，其相对 RTN、GPTQ、局部 ScaleSearch 和 BlockRotation 基线取得了一致的提升，MXFP4 的提升空间最大，对更强的格式也有持续的增益。直接激活设置进一步表明，SoScale 可以在保持激活路径简单且与硬件对齐的同时改进权重量化。这些结果表明，对于精确的低比特浮点 LLM 推理，scale code 的选择应当被视为一个层感知的优化问题，而不仅仅是局部张量拟合规则。

## 附录

原文（arXiv v1）无附录。

## 参考文献

以下保留原文参考文献的作者与标题（英文原样）。

- Alvarez et al. (2025). E. Alvarez, O. Almog, E. Chung, S. Layton, D. Stosic, R. Krashinsky, K. Aubrey. Introducing NVFP4 for efficient and accurate low-precision inference. NVIDIA Technical Blog. https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- Ashkboos et al. (2024). S. Ashkboos, A. Mohtashami, M. L. Croci, B. Li, P. Cameron, M. Jaggi, D. Alistarh, T. Hoefler, J. Hensman. QuaRot: outlier-free 4-bit inference in rotated LLMs.
- Bisk et al. (2020). Y. Bisk, R. Zellers, R. Le Bras, J. Gao, Y. Choi. PIQA: reasoning about physical commonsense in natural language. AAAI, pp. 7432–7439.
- Clark et al. (2019). C. Clark, K. Lee, M. Chang, T. Kwiatkowski, M. Collins, K. Toutanova. BoolQ: exploring the surprising difficulty of natural yes/no questions. NAACL, pp. 2924–2936.
- Clark et al. (2018). P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, O. Tafjord. Think you have solved question answering? Try ARC, the AI2 reasoning challenge.
- Cobbe et al. (2021). K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, J. Schulman. Training verifiers to solve math word problems.
- Cook et al. (2026). J. Cook, J. Guo, G. Xiao, Y. Lin, K. Wyss, M. Nazemi, A. Mishra, C. del Mundo, T. Blankevoort, S. Han. Four over six: more accurate NVFP4 quantization with adaptive block scaling.
- DeepSeek-AI (2026). DeepSeek-V4: towards highly efficient million-token context intelligence.
- Egiazarian et al. (2026). V. Egiazarian, R. L. Castro, D. Kuznedelev, A. Panferov, E. Kurtic, S. Pandit, A. Marques, M. Kurtz, S. Ashkboos, T. Hoefler, D. Alistarh. Bridging the gap between promise and performance for microscaling FP4 quantization.
- Frantar et al. (2023). E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. GPTQ: accurate post-training quantization for generative pre-trained transformers.
- Gao et al. (2024). L. Gao, J. Tow, B. Abbasi, S. Biderman, et al. The language model evaluation harness. Zenodo.
- Gupta et al. (2026). T. Gupta, H. Prairie, X. Wu, R. Abhyankar, Q. Wu, A. Silveria, P. Ponnusamy, J. Wang, B. Athiwaratkun, L. Song, T. Dao, D. Y. Fu, C. D. Sa. Search your block floating point scales!
- Hendrycks et al. (2021). D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, J. Steinhardt. Measuring massive multitask language understanding. ICLR.
- Lhoest et al. (2021). Q. Lhoest, A. Villanova del Moral, Y. Jernite, et al. Datasets: a community library for natural language processing. EMNLP System Demonstrations, pp. 175–184.
- Lin et al. (2024). J. Lin, J. Tang, H. Tang, S. Yang, W. Chen, W. Wang, G. Xiao, X. Dang, C. Gan, S. Han. AWQ: activation-aware weight quantization for on-device LLM compression and acceleration. MLSys, pp. 87–100.
- Liu et al. (2025). Z. Liu, C. Zhao, I. Fedorov, B. Soran, D. Choudhary, R. Krishnamoorthi, V. Chandra, Y. Tian, T. Blankevoort. SpinQuant: LLM quantization with learned rotations.
- Luo et al. (2026). Y. Luo, J. Huang, Y. Cheng, Z. Yu, K. Tang, X. Ma, X. Wang, A. Tong, G. Hu, Y. Xu, M. Taghian, P. Wu, G. Li, Y. Peng, T. Hu, M. Chen, M. B. Mi, H. Liu, X. Zhou, J. Wang, Q. Lin, H. Liao. HiFloat4 format for language model inference.
- Meng et al. (2026). H. Meng, Y. Luo, Y. Zhao, W. Liu, P. Zhang, X. Ma. ARCQuant: boosting NVFP4 quantization with augmented residual channels for LLMs.
- Merity et al. (2016). S. Merity, C. Xiong, J. Bradbury, R. Socher. Pointer sentinel mixture models.
- OpenAI (2026). ChatGPT.
- Paszke et al. (2019). A. Paszke, S. Gross, F. Massa, A. Lerer, et al. PyTorch: an imperative style, high-performance deep learning library. NeurIPS.
- Qwen Team (2026). Qwen3.5: towards native multimodal agents.
- Rouhani et al. (2023). B. D. Rouhani, R. Zhao, A. More, M. Hall, et al. Microscaling data formats for deep learning. arXiv:2310.10537.
- Shao et al. (2024). W. Shao, M. Chen, Z. Zhang, P. Xu, L. Zhao, Z. Li, K. Zhang, G. Peng, Y. Qiao, P. Luo. OmniQuant: omnidirectionally calibrated quantization for large language models. ICLR.
- Shao et al. (2025). Y. Shao, P. Wang, Y. Chen, C. Xu, Z. Wei, J. Cheng. Block rotation is all you need for MXFP4 quantization.
- Wolf et al. (2020). T. Wolf, L. Debut, V. Sanh, et al. Transformers: state-of-the-art natural language processing. EMNLP System Demonstrations, pp. 38–45.
- Xiao et al. (2023). G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, S. Han. SmoothQuant: accurate and efficient post-training quantization for large language models. ICML, pp. 38087–38099.
- Yang et al. (2025). A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, et al. Qwen3 technical report. arXiv:2505.09388.
- Zellers et al. (2019). R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, Y. Choi. HellaSwag: can a machine really finish your sentence? ACL, pp. 4791–4800.
- Zhao and Yuan (2025). P. Zhao, X. Yuan. GANQ: GPU-adaptive non-uniform quantization for large language models. ICML, pp. 77911–77925.
