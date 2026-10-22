# QuantEase — arXiv:2309.01885

> 把 GPTQ 的逐层重构目标当成"整数网格上的最小二乘"，用**循环坐标下降**求解：
> 每次只动一个权重，闭式更新 = 先算无约束最优值再取整；一遍扫完还能接着扫，
> 目标值单调不增，且不需要矩阵求逆 / Cholesky。外加一个把少数异常权重留全精度的变体（块坐标下降 + 迭代硬阈值）。

本文件是这篇 paper 的**解读总入口**。radar-native 分析在下方，各角度详读见导航。

## 论文

- **标题**：QuantEase: Optimization-based Quantization for Language Models
- **链接**：[abs](https://arxiv.org/abs/2309.01885) · [PDF](https://arxiv.org/pdf/2309.01885v2) · code：原文称"将在 GitHub 发布"，未见官方实现（CDQuant 附录称其无公开代码、自行复现）
- **作者**：Kayhan Behdin, Ayan Acharya, Aman Gupta, Qingquan Song, Siyu Zhu, Sathiya Keerthi, Rahul Mazumder（LinkedIn / MIT）
- **发表**：2023-09-05（v2 2023-12-01）· stat.ML / cs.CL / cs.LG
- **许可**：arXiv 非独占许可（非 CC-BY）——中文翻译只做逐节复述，不是逐句翻译

## 解读导航

| 角度 | 文件 | 内容 |
|---|---|---|
| 原理故事 | [paper.org](paper.org) | 七拍故事版（以"两权重、GPTQ 不能悔棋"为锚点；该例为讲解自构、已数值验算） |
| 中文伴读 | [reading.org](reading.org) | 选择性精读（非交互，自问自答代笔） |
| 中文翻译 | [translation_zh.md](translation_zh.md) | 全文中文复述（逐节释义，不是逐句翻译；由另一并行任务生成，见文件是否存在） |
| 溯源倒读 | [../../paper-river/QuantEase-2309.01885.org](../../paper-river/QuantEase-2309.01885.org)（中）· [_en](../../paper-river/QuantEase-2309.01885_en.org)（英） | 倒读法脉络（由另一并行任务生成，见文件是否存在） |

数据来源：radar 无打分记录（见下）；解读基于 arXiv 全文（HTML v2）。

## Radar 记录

- **未收录**：2023-09 发表，早于 radar 归档起点（2025-11），`resolve_paper.py` 只能 live 抓 arXiv，没有 composite / bucket / 摘要。
- **未 triage**（既非 seed 也非 rejected）。
- 按内容归桶应落在 `ptq`。

## 同类对比 / novelty

radar 里没有同时代邻居，但 `ptq` 桶里有一批"GPTQ 之后怎么做得更准"的后续工作，可以反推本文位置：

| 论文 | 与本文的关系 |
|---|---|
| [CDQuant (2406.17542)](https://arxiv.org/abs/2406.17542)（radar 未收录） | **最近的邻居，且点名对比**：用**贪心**坐标下降（选误差下降最多的坐标），QuantEase 是**循环**坐标下降。CDQuant 自述两者在 per-channel 上"性能相当"（其附录 D，自行复现 QuantEase、按其推荐的 20 个 epoch），差异在于 CDQuant 还做了 group / sub-channel 量化与块坐标下降（BCD），并强调可替换 QuIP / FrameQuant 里的 GPTQ 组件 |
| [ADMM-Q (2605.11222)](https://arxiv.org/abs/2605.11222) | 同样把逐层重构当组合优化，用 ADMM 替换 GPTQ，属于"更强求解器"路线，可与 QuantEase 并列看 |
| [OJBKQ (2602.08376)](https://arxiv.org/abs/2602.08376) | 用格视角（Babai-Klein）做取整搜索；与"循环取整"思路不同，但都是对 GPTQ 一遍扫的改进 |
| [GPTQ-Babai (2507.18553)](https://arxiv.org/abs/2507.18553)（本仓库已有[解读](../GPTQ-Babai-2507.18553/README.md)） | 从格点几何解释"GPTQ 一遍扫"为何有最坏界；与本文"多遍改进"互补 |
| [Rethinking Residual Errors (2604.07955)](https://arxiv.org/abs/2604.07955)、[GPTQ-2D (2607.27042)](https://arxiv.org/abs/2607.27042) | 仍在"补偿 / 取整顺序"框架内改进 |
| [OPTQ 泛化行为 (2609.31560)](https://arxiv.org/abs/2609.31560) | 研究 OPTQ 在校准集之外的泛化与正则化——正好对应本文"Falcon-180B 只跑一次迭代以防过拟合"那条脚注（未核实内容细节，仅据标题与摘要） |

判断：**不是 re-skin，是较早把"循环 CD + 闭式取整"明确用在 LLM 逐层 PTQ 上的工作**，思路与 CDQuant 同源、几乎并行（相差约 9 个月）。
真正区别于 GPTQ 的只有两点：(1) 不丢掉常数项、不依赖 OBS，所以能多遍迭代；(2) 异常权重位置可在优化中增删。
没有新的理论保证（Lemma 2 只是收敛到坐标最小点），也没有说明为什么坐标最小点对应更好的困惑度。

## 实验与工程可落地性

- **精度（WikiText2 PPL，C4 校准 128 条 x 2048，per-channel、无 grouping）**：
  OPT-6.7b 3-bit GPTQ 15.16 vs QuantEase 12.95（FP16 10.86）；Falcon-7b 3-bit 9.62 vs 8.83（FP16 6.59）；
  逐层相对误差中位数比 GPTQ 低 12%，最多 30%（BLOOM-1b1）。4-bit 差距很小。反例：OPT-13b 3-bit GPTQ 11.90 优于 QuantEase 12.41。
- **规模 / 可跑性**：OPT-66b 能在单张 V100 上跑（GPTQ / AWQ 同机显存不够）；Falcon-40b/180b 上 GPTQ 的 Cholesky 多数种子数值失败，QuantEase 可跑。
  注意：摘要的"单张 A100 约 3 小时量化 Falcon-180B"是**单次迭代**口径（表 3 脚注：只跑一次以防过拟合）；主实验默认 25 次迭代，OPT-66b 总计 13.8 小时。
- **成本**：每次迭代 O(p²q)，与调优过的 GPTQ 相近；25 次迭代意味着总时长大约比 GPTQ 一遍扫高一个量级（我的估计，论文未并排给出 GPTQ 总时间）。
- **异常感知**：0.5% / 1% 非结构化异常 ≈ +0.15 / +0.3 bit；OPT-1.3b 3-bit 带 1% 异常 PPL 16.25 vs SpQR 18.17（不带异常 21.30）。
  2-bit + 2% 异常（≈2.6 bit）下 OPT-13b 19.3 vs SpQR 22.3（FP16 10.13）；小模型（350m：158 vs 323）仍不可用。
  非结构化异常需要稀疏通路；结构化（整列）版本精度收益被吃掉一半（OPT-1.3b 18.86 已不如 SpQR 1% 的 18.17）。
- **与 AMD / 生产推理**：权重仍是 per-channel 均匀 INT3/INT4，无需特殊 kernel；但 3-bit 的优势是在**无分组**下测的，业界常用 g=128，差距大概率更小（论文未测，列为开放问题）。
  非结构化异常的 kernel 成本论文不评估，落地应选结构化版本重测。
- **待确认**：与 GPTQ 的总时间并排对比；迭代次数 vs 留出集 PPL 曲线（只有 OPT-350m 的一张图）；group-wise 与 AWQ 叠加的实际收益（论文留作 future work）；官方代码状态。
- **与 MXFP4/NVFP4**：论文只处理等间距整数网格，Lemma 1 的"先求最优再取整"对非均匀的 E2M1 网格也成立（取整换成最近邻即可），但这是我从公式推的，论文没有验证（未核实）。

## 趋势定位

`ptq` 桶长期活跃，"GPTQ 之后怎么更准"是热点。本文属于**早期的求解器替换路线**（CD → 后来的贪心 CD、ADMM、格视角等），
2023 年后 GPTQ 变体越来越多，本文的直接价值更多是"给 GPTQ 输出做后处理"这个接口和 outlier 位置动态增删的想法；
作为独立方法，已被后续方法（CDQuant、ADMM-Q）覆盖大部分场景。值得读 Lemma 1 与 Algorithm 2，不必从头跑复现。

## Triage 结论

未 triage。若要入 seed：理由是"循环 CD 求解 GPTQ 目标的早期代表作 + 被 CDQuant 明确对比"，属 `ptq` 桶的参考文献；
不建议作为"可直接部署的方法"入 seed（无官方代码、无分组实验、总时长未并排）。
