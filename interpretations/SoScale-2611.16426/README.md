# SoScale — Second-Order Scale Search for FP4 Quantization

arXiv:2611.16426 · Zhao, Li, Zhen, Lin, Yu, Dong (Huawei) · cs.LG/cs.AI · 2026-11-12
[abs](https://arxiv.org/abs/2611.16426) · [pdf](https://arxiv.org/pdf/2611.16426v1) · 无官方代码链接

一句话：FP4 微缩放格式（MXFP4 / NVFP4 / HiF4）里，每块的 scale code 是硬件规定的离散值，默认用 max-abs 选。SoScale 在合法 scale code 集合里，用 Hessian 加权的层输出误差来选 code，并把 GPTQ 的误差回填串进逐块量化流程。W4A4 下在 Qwen3-8B/14B 上全部 6 个 (模型, 格式) 组里均分最高。

## 导航
- [中文翻译（全文）](translation_zh.md)
- [中文精读](reading.org) · [原理与故事](paper.org)
- paper-river 溯源：[zh](../../paper-river/SoScale-2611.16426.org) · [en](../../paper-river/SoScale-2611.16426_en.org)
- radar 数据：[data/summarized/2026-11-12.json](../../data/summarized/2026-11-12.json) · [digest](../../digests/2026-11-12.md#p-2611-16426)

## Radar-native 分析 (D)

**Triage 状态**：2026-11-12 digest 已收录（composite 7，bucket `ptq`，topic_relevance 4，practicality 3）；未 accept / reject，未进 seeds。

**核心结论（已对照正文表格）**
- Table 2（W4A4，7 个任务均分相对 BF16）：Qwen3-14B 上 SoScale 系最佳为 MXFP4 97.17% / NVFP4 99.20% / HiF4 99.40%；Qwen3-8B 为 96.44% / 98.99% / 98.38%。
- 对最强非 SoScale 基线 +0.21 ~ +0.77 pp，对 ScaleSearch +0.47 ~ +1.10 pp（论文自述，与表一致）。
- 消融（Table 4，Qwen3-8B）：Hessian 目标 + 回填 vs MSE 目标 + 回填，均分差 MXFP4 +0.31 / NVFP4 +0.86 / HiF4 +0.32 pp。MSE + 回填本身收益不稳定（NVFP4 反而比纯 ScaleSearch 低）。这是全文最能支持"Hessian 目标才是关键"的证据。

**对 radar 摘要的修正 / 需要警惕的点**
- radar 记 "Qwen3 (size unknown)"，正文是 **Qwen3-8B 和 Qwen3-14B**，仅这两个。没有 7B 以上更大模型，也没有 MoE。
- 与 GPTQ 的差距很小。Qwen3-8B MXFP4：GPTQ 0.7089 vs SoScale 0.7092（+0.03 pp）；Qwen3-14B MXFP4：+0.39 pp；Qwen3-14B NVFP4：+0.56 pp。"持续优于 GPTQ" 成立，但幅度在噪声量级。
- 单次 calibration（128×2048 WikiText），无多 seed、无置信区间。均分里 GSM8K 方差最大，如 8B MXFP4 的 SoScale 均分优势大半来自 GSM8K（0.8575 vs GPTQ 0.8362）；SoScale w/ Had. 在同一格里 GSM8K 反而掉到 0.8271。
- 图 1(c) 用 PPL 论证"局部 MSE 不等于低 PPL"，但正文没有 PPL 表，主结果只有 7 个 MCQ/GSM8K 任务。PPL 的论证只在一张图里。
- 全文没有校准耗时、峰值显存、kernel 速度数据。"无 serving 侧开销"是结构性论断（只改离线选的 scale code），合理，但搜索成本是 unknown。
- Table 3（权重侧搜索 + 激活直接 RTN）里，SoScale-w 对 ScaleSearch-w 的增益 0.08 ~ 1.92 pp，且部分行高于 Table 2 的在线激活规则版本。论文承认在线 per-token 激活规则只是局部启发式。也就是说 Table 2 的激活路径本身未必是最优设置。
- HiF4 是华为格式，NVIDIA / AMD 硬件上没有原生支持；MXFP4 / NVFP4 的结论才与 AMD 部署直接相关。

**同类对比 / novelty**（同 bucket `ptq` 的 radar 一句话摘要，未读原文）
- 想法本身（Hessian 加权选量化参数 + GPTQ 回填）并不新：GPTQ 的回填、AWQ/OmniQuant 的层输出感知都是前例。新点是把它落到"hardware-valid 的离散 scale code 集合"上，并且格式无关（同一套流程覆盖三种格式）。
- 最近邻：2609.36654 *Replay the Curvature*（NVFP4，名字暗示曲率/二阶信息，可能与本文最重叠）；2611.00486 *Integer-Exact Microscaling*（MXFP4 在 INT8 上跑，正交）；2610.00983 *Reconstruction Loss Scale*（重构损失尺度，同属"目标函数选择"）。*Replay the Curvature* 与 SoScale 的具体差异需要读原文才能下结论，此处未核实。
- 论文自己列的直接对手是 ScaleSearch（Gupta 2026）和 Four Over Six（Cook 2026）；两者本文均只作为基线，没有对 Four Over Six 做实验对比（表里没有）。

**趋势 / timing**：`ptq` 桶约 1~4 篇/天，近期平稳。FP4（NVFP4/MXFP4）相关在桶内持续出现（SoloQ、Replay the Curvature、Integer-Exact MX、本文），属于热方向，但"格式 + 缩放码"这类小改动的边际收益在缩小，NVFP4 / HiF4 已接近 BF16（99%+）。

**工程可落地性**：practicality 3 合理。
- 优点：产物仍是标准 FP4 payload + 合法 scale code，推理路径不变；offline 量化一次。
- 代价：需要 Hessian（$X^\top X$）+ GPTQ 式逐块回填 + 每块候选 code 窗口内的搜索，校准成本高于 RTN / ScaleSearch；论文未给数字。
- 对 AMD / Quark：MXFP4 路径可直接对应（E8M0 scale，窗口内搜整数指数）。可与 Quark 现有 GPTQ + MXFP4 实现对接：把 GPTQ 逐列流程里"scale 由 max-abs 决定"换成"每块在 $c_0\pm2$ 内按 Hessian 子矩阵打分选 code"。是否有明显收益取决于 8B 以上模型和 MoE，本文没有验证。
- 待 PDF 核查：窗口大小（图 2 画的是 $c_0\pm2$，正文说 "small format-specific offset window"，具体值）；HiF4 的 E1 scale 如何派生；Hadamard 版本的块大小；附录是否有 PPL / 耗时。
