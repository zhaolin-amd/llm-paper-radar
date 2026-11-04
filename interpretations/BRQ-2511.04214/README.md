# BRQ — Block Rotation is All You Need for MXFP4 Quantization

arXiv:[2511.04214](https://arxiv.org/abs/2511.04214) · [PDF](https://arxiv.org/pdf/2511.04214v1) · 2025-11-06 · cs.LG/cs.CL · ICML 2026 poster · arXiv 非独占许可
作者：Yuantian Shao, Peisong Wang, Yuanteng Chen, Chang Xu, Zhihui Wei, Jian Cheng（中科院自动化所 CDL / UCAS / Sydney / NJUST）

一句话：QuaRot / SpinQuant 这类全局旋转在 INT4 上有效，在 MXFP4 上反而变差。作者定位到原因（MXFP4 的 2 的幂（PoT）block scale 对大值重建差，而全局旋转把离群值能量摊给所有"小值"block，使大量 regular block 的 scale 变大），提出 BRQ：旋转矩阵做成按 32 通道分块的块对角阵，每个 MX block 内各自旋转。

## 导航

| 内容 | 文件 |
|---|---|
| 中文精读（A） | [reading.org](reading.org) |
| 原理故事（B） | [paper.org](paper.org) |
| 中文翻译（E，复述体） | [translation_zh.md](translation_zh.md) |
| Paper-river 溯源（C） | [../../paper-river/BRQ-2511.04214.org](../../paper-river/BRQ-2511.04214.org) · [_en](../../paper-river/BRQ-2511.04214_en.org) |

## Radar 原生分析（D）

**Radar 状态**：不在推荐列表。radar 没有这篇的打分记录，没出现在任何日报、周报或月报（论文发表于 2025-11-06，也许早于当时的抓取窗口，未核实）；未 triage。数据源为 arXiv HTML v1 全文。

**核心论点与证据**
- Benchmark（Table 1，6 个模型，模拟 MXFP4，microxcaling）：GPTQ 最稳；QuaRot+RTN 在 LLaMA-2 7B 上 PPL 7.08→13.09（比 RTN 还差），QuaRot+GPTQ 增益不稳定，SpinQuant 只比 QuaRot+ 略好（Mistral 7B 5.73→5.68）。
- 机理（§4）：PoT scale 在大值处粒度粗，outlier block 里 outlier 的量化损失最高约为 regular block 最大损失的 5 倍；全局旋转保 L2 能量，把 ~1% 超过 3 的激活削掉，但把超过 1.5 的比例从约 5% 抬到 11%；regular block 数量远多于 outlier block，累计误差占主导。原文 findings C–E：组量化格式（含 FP16 scale 的 BINT4/BFP4）下随机旋转都会掉点，MX（PoT scale）最严重，PoT scale 相对 FP16 scale 本身就多一份损失，旋转后差距更大。
- BRQ（Table 2，随机块 Hadamard + GPTQ）：LLaMA-3 8B PPL 7.14（QuaRot+ 7.68，BINT4 7.40），LLaMA-3.2 1B 11.95（QuaRot+ 12.78），Qwen2.5 1.5B 12.15（12.80）；除 LLaMA-3.2 3B（9.41 vs BINT4 9.29）外都优于 BINT4。
- 无需优化即超过优化后的全局旋转 SpinQuant（Table 3，LLaMA-3 8B 7.62→7.14）；在 SpinQuant 框架里优化块旋转（BRQ-Spin）收益有限（LLaMA-3.2 3B 9.41→9.08，LLaMA-3 8B 7.14→7.13）。
- 旋转维度扫描（Fig. 8）：最优维度恰等于 MX block size 32。
- 开销（Table 4，LLaMA-2 7B prefill，A800 + 软件仿真）：相对 MXFP4，QuaRot 额外 4.6–7.8%，BRQ 1.3–4.3%，旋转带来的额外延迟约降 40%。

**同类对比 / novelty**
- 算法本身几乎没有新东西：块对角 Hadamard 早已是常见思路。价值在"为什么全局旋转在 MX 上坏"的诊断和"旋转维度 = block size"的结论。
- radar 里已有 MXFP4 / 旋转相关的多篇解读（OptRot-2512.24124、NativeMXFP4-2605.09825、MixQuant-2605.20315、DiRotQ-2605.16732、MXFPBench-2601.09555）。这些我这次没有重读，所以不对它们与 BRQ 的具体差异下判断；paper-river 的前向部分（见溯源文件）会覆盖。
- 溯源 agent 通过检索确认：MR-GPTQ（arXiv 2509.23202，块级 Hadamard + GPTQ）早于 BRQ 且 BRQ 未引用，更像并行发现，所以"块旋转"的优先权不要全记在 BRQ 头上；本人未读 MR-GPTQ 原文。
- 与本仓库另一篇 MixQuant（块 Hadamard + permutation）方向相近，需要对读才知道谁在谁基础上；未核实。

**趋势 / timing**：MXFP4 W4A4 + rotation 在 2026 年是活跃线，BRQ 是较早的"块旋转适配 MX"的诊断论文，后续工作大概率引用它；对 AMD（Quark 的 MXFP4 流程、MI350 系列原生 MXFP4）有直接相关性，原文也引用了 AMD Quark 2025。

**工程落地性**
- 好：仅把旋转改成块对角，可接入 QuaRot / SpinQuant 现有框架；在线 R4（down_proj 前）从 O(N²) 降到 O(N×32)；不需要优化即可用。
- 限制（论文自述或可从表中看出）：
  - 精度全部是 microxcaling 软件仿真，延迟在 A800（不支持 FP4）上测，没有真实 MXFP4 硬件数字。
  - 模型最大 LLaMA-2 70B（附录 Table 5），没有 MoE、没有长上下文或推理（reasoning）类评测，指标只有 WikiText2 PPL 和 5 个 zero-shot 任务。
  - 个别设置下 BRQ 仍不如 BINT4（LLaMA-3.2 3B）；W4A4 与 FP16 仍有差距（LLaMA-3 8B 6.14 vs 7.14）。
  - 文中数字有小瑕疵：Table 2 与 Table 3 里 Qwen2.5 1.5B 的 BRQ Avg 不一致（54.83 vs 54.17）；正文把 SpinQuant 的 LLaMA-3 8B PPL 写成 7.68，而 Table 3 是 7.62；正文把 OmniQuant 的对比写成"14.32 vs. 3.35"（应为 13.35）。引用时以表为准，需回 PDF 核对。
- 若想在 Quark 里验证：只需要在已有 rotation 入口把 R1/R2/R4 的维度对齐到 32 并做块对角化，注意 MXFP4 kernel 实际是否与 microxcaling 的舍入一致。

**Triage 建议**：值得 accept 进 seeds——理由是"MXFP4 下旋转失效的机理诊断"，不是算法新颖性。
