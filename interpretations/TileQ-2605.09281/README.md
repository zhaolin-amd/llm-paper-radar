# TileQ — arXiv:2605.09281

> 低秩量化 = 量化权重 + 一个低秩补偿项 `UV`。放到 MoE 上，每个专家各带一对低秩因子，额外显存大，
> 而且几十上百个小专家各算各的低秩分支，kernel 碎、推理慢。TileQ 把同层专家权重按 **2D 瓦片**组织，
> **在输入和输出两个维度上共享低秩因子**（同层专家的权重有相似结构），再把所有专家的低秩计算融合成一次 GEMM。
> fine-tuning-free PTQ，后端量化器可以是 GPTQ（标量版 TileQ-s）或 GPTVQ（向量版 TileQ-v）。

本文件是这篇 paper 的**解读总入口**。radar-native 分析在下方，各角度详读见导航。

## 论文

- **标题**：TileQ: Efficient Low-Rank Quantization of Mixture-of-Experts with 2D Tiling
- **链接**：[abs](https://arxiv.org/abs/2605.09281) · [PDF](https://arxiv.org/pdf/2605.09281v1) · code：仅给了匿名仓库 `anonymous.4open.science/r/TileQ-BE20`（审稿用，非正式开源）
- **作者**：Hongyaoxing Gu, Xinzhe Chen, Lijuan Hu, Fangfang Liu（中科院软件所）
- **发表**：2026-05-10 · cs.LG
- **许可**：**CC BY 4.0**（允许逐句全文翻译；本目录的中文翻译仍按统一做法写成逐节复述，需要逐句版可再生成）

## 解读导航

| 角度 | 文件 | 内容 |
|---|---|---|
| 原理故事 | [paper.org](paper.org) | 七拍故事版 |
| 中文伴读 | [reading.org](reading.org) | 选择性精读 |
| 中文翻译 | [translation_zh.md](translation_zh.md) | 全文中文复述（逐节释义，含公式/表/图） |
| 溯源倒读 | [../../paper-river/TileQ-2605.09281.org](../../paper-river/TileQ-2605.09281.org)（中）· [_en](../../paper-river/TileQ-2605.09281_en.org)（英） | 已有，沿用 |

数据来源：[`data/summarized/2026-05-10.json`](../../data/summarized/2026-05-10.json)（radar 摘要与打分）+ arXiv HTML 全文（含附录目录）。

## Radar 记录

- composite 9（三篇里最高），bucket `ptq`，topic_relevance 5，practicality 4，未触发 hard gate。
- **未 triage**。
- radar 记"最大模型未知"：全文测了 Qwen1.5-MoE-A2.7B、Qwen3-30B-A3B、**Qwen3-Next-80B-A3B**（512 专家）、Mixtral-8×7B、DeepSeek-MoE-16B；位宽 **2-bit 和 3-bit**。

## 同类对比 / novelty

| 论文 | 关系 |
|---|---|
| GPTQ / GPTVQ | 它们既是对比基线，也是 TileQ 的后端量化器（TileQ-s 用 GPTQ、TileQ-v 用 GPTVQ），所以结果 = "加了 2D 低秩共享"的增益 |
| LoPRo（Gu et al. 2026，低秩 + 旋转；作者与本文一作同姓，疑为同组前作，未核实） | **最近的前辈**。逐专家（element-wise）低秩；TileQ 精度"相当或略好"，但额外位宽从 ~0.32 降到 0.04，因此新东西是**共享低秩因子 + 融合 kernel**，不是低秩量化本身 |
| MoEQuant | MoE 专用 PTQ 基线，2-bit 下直接崩（Qwen1.5 PPL 5e5） |
| MiLo、MxMoE（附录 D.2） | 其他 MoE 专用方法，只在附录对比，主文没有数字 |
| [AlphaQ (2606.04980)](../AlphaQ-2606.04980/README.md) / [VSRAQ (2606.05688)](../VSRAQ-2606.05688/README.md) | 正交：它们分别管位宽分配 / 校准目标；TileQ 管权重表示（低秩共享），论文限制部分也提到可与剪枝等叠加 |
| [CodeQuant (2604.10496)](https://arxiv.org/abs/2604.10496)、[MoBiE (2604.06798)](https://arxiv.org/abs/2604.06798) | 同一时间段的 MoE 低比特工作（据标题/摘要，未读全文） |

判断：**核心是"同层专家权重可共享低秩结构"这个假设 + 把它做成能融合的 kernel**；量化器本身并无新意（论文自己也这么说："核心创新不在量化器，而在 2D-Tiling"）。
消融支持这一点：去掉 2D-Tiling 精度不变但额外位宽 ×8；去掉低秩则 Qwen1.5 2-bit PPL 12.5 / 准确率 38.5（即退化成 GPTQ）。

## 实验与工程可落地性

- **精度（Qwen1.5-MoE，FP16 PPL 6.79 / MMLU 59.3 段）**：2-bit：GPTQ PPL 12.5、GPTVQ 8.12、LoPRo 7.52、TileQ-s 7.56、TileQ-v 7.35；3-bit：GPTQ 7.98、GPTVQ 7.1、LoPRo 6.94、TileQ-s 6.94、TileQ-v 6.93。
  Mixtral-8×7B：GPTQ 2-bit PPL 15.3，TileQ-v 4.78（FP16 3.87）；3-bit TileQ-s 4.10。Qwen3-30B-A3B 2-bit：GPTQ 14.6，TileQ-v 10.1（FP16 8.07）。
  **2-bit 仍有明显差距**（Qwen3-30B 10.1 vs 8.07），3-bit 才基本追平。
- **额外开销**：低秩附加位宽 0.04 bit/权重（TileQ）vs 0.32（逐专家），即 8× 降低；摘要说"最多 10×"（来源于某个模型/配置，未逐项核对）。
- **延迟**：A800 / H800 / 5090 上测 MoE MLP 块。prefill 下 TileQ 的额外延迟"< 5%"；decode 下逐专家低秩与 MiLo-1D 仍慢（Qwen3-MoE 约 75%）。**摘要的"延迟降到约 5%"是相对逐专家低秩基线的开销口径，不是相对 FP16 / 纯量化的加速**，措辞容易误读。MoE MLP 约占端到端 50%，论文据此推断端到端开销更低（无端到端实测数字，未核实）。
- **量化时间**：瓶颈在后端 GPTQ / GPTVQ，专家多且小的模型（Qwen3-30B-A3B）尤其慢（论文 Limitations 自述）；分解与低秩阶段很轻。
- **与 AMD / 生产推理**：需要自定义融合 kernel（一次 GEMM 做所有专家的低秩分支）和向量码本（TileQ-v 要 VQ 反量化）；论文的 kernel 在 NVIDIA 三种卡上测，AMD 上需要自己移植。标量版 TileQ-s 更易落地（更适合"硬件友好、低延迟"，论文自己的建议），TileQ-v 在 2-bit 精度更好但依赖 VQ 路径。
- **精读发现的问题（据精读代笔 agent 对全文的核对，我没有逐项复算）**：
  - 延迟收益主要来自融合 kernel，不是 2D 布局：表 7 里 1×60 布局（等价 1D 共享，用 TileQ 自己的 fused kernel）decode 开销 13.2%，对比 MiLo 式 1D 约 75%。
  - 七种网格形状的 PPL 都在 7.52–7.65，去掉 2D-Tiling 是 7.49 vs 7.35；论文没有"随机分组"对照，所以"把相似专家聚类放一起"是否真有用，没被证明。
  - 定理 A.1 是同义反复（额外项被定义为 2D 低秩误差减逐专家低秩误差）。
  - 表 9：Mixtral-8×7B 3-bit，TileQ PPL 4.12，MiLo 4.03，与正文"全面领先"的说法不符。
  - 附录 A.3 的压缩率算例不能精确复算（逐专家项得 0.458 而非 0.427，TileQ 项对应 12×12 而非所述 16×8），数量级仍成立。
- **待确认**：无正式开源代码（匿名仓库）；没有生成式 / 推理类任务（主要是 PPL + ARC / PIQA / WinoGrande / MMLU）；Qwen3-Next-80B 的结果在哪张表（本文主表里只看到 Mixtral / Qwen1.5 / Qwen3-30B，其余在 Appendix 或主表其他段，需到 PDF 核）；没有同时测路由是否被量化扰动（与 VSRAQ 的关注点相反）。

## 趋势定位

MoE 低比特（2–3 bit）这一端：AlphaQ / Q-Strata 等偏"分配位宽"，TileQ 偏"改权重表示 + 融合 kernel"，是目前少见的**系统 + 算法联合**的 MoE PTQ 工作，且 composite 在 radar 里是这批里最高。
对关心 2-bit MoE 部署的团队值得精读；对 W4 / FP4 主流部署场景收益较小（W4 下不需要低秩补偿）。

## Triage 结论

未 triage。若入 seed：理由是"2D-Tiling 共享低秩 + 融合 kernel，MoE 2-3 bit 低秩量化的最新一跳，实验覆盖 5 个 MoE 含 80B"；
注意口径提醒（5% 延迟 = 相对逐专家低秩的开销）和"匿名仓库、无官方开源"这两点。
