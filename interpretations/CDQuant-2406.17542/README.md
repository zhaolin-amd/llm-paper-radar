# CDQuant — arXiv:2406.17542

> 把 GPTQ 的"固定次序、取整一次、不回头"换成**贪心坐标下降**：整数向量始终完整，
> 每步在所有（或随机配成的块内）取值里改"使层输出误差降得最多"的那一个，直到收敛；
> 起点用按输出误差选缩放的 OWC。2–4 比特下逐层误差和困惑度都低于 GPTQ，可直接替换 QuIP / FrameQuant 里的 GPTQ。

本文件是这篇 paper 的**解读总入口**。radar-native 分析在下方，各角度详读见导航。

## 论文

- **标题**：CDQuant: Greedy Coordinate Descent for Accurate LLM Quantization
- **链接**：[abs](https://arxiv.org/abs/2406.17542) · [PDF](https://arxiv.org/pdf/2406.17542v3) · code：论文未给链接
- **作者**：Pranav Ajit Nair, Arun Sai Suggala（Google DeepMind）
- **发表**：2024-06-25（v3 2024-10-22）· cs.LG / cs.AI / cs.CL
- **许可**：CC BY 4.0

## 解读导航

| 角度 | 文件 | 内容 |
|---|---|---|
| 原理故事 | [paper.org](paper.org) | 七拍故事版（含博导审稿、启发） |
| 中文伴读 | [reading.org](reading.org) | 选择性精读，非交互版 |
| 中文翻译 | [translation_zh.md](translation_zh.md) | 全文中文翻译（CC BY 4.0；公式/表格渲染） |
| 溯源倒读 | [../../paper-river/CDQuant-2406.17542.org](../../paper-river/CDQuant-2406.17542.org)（中）· [_en](../../paper-river/CDQuant-2406.17542_en.org)（英） | 倒读法脉络 |

数据来源：radar 无打分记录（见下）；解读基于 arXiv 全文（HTML v3）。

## Radar 记录

- **未收录**：该文 2024-06 发表，早于 radar 归档起点（2025-11），`resolve_paper.py` 只能 live 抓 arXiv，没有 composite / bucket / 摘要。
- **未 triage**（既非 seed 也非 rejected）。
- 按内容归桶应落在 `ptq`。

## 同类对比 / novelty

核心贡献是**求解器**：同一个逐层重构目标，用整数向量上的贪心坐标下降（Alg.1）+ 随机块（Alg.2，k=2）+ OWC 初始化，换掉 GPTQ 的一遍扫。
不改量化格式（仍是均匀整数 + scale/bias），所以输出可被现有 GPTQ 推理 kernel 直接用。

| 论文 | 与本文的关系 |
|---|---|
| [QuantEase (2309.01885)](https://arxiv.org/abs/2309.01885) | **同期、同思路**（论文自己说是 parallel work）：也是坐标下降，但**按序循环**（推荐 20 轮），本文是**贪心**；本文额外做了分组量化（OWC-CD）和块版本（BCD）。论文附录 D 自己复现 QuantEase 对比：PaLM2-Otter 3 比特 6.996（QuantEase）/ 7.002（CD）/ 6.979（BCD）持平；Gecko 上 QuantEase 10.731 反而优于 CD 10.920。贪心的好处主要是更新次数少（CD 只需约 d_in/8 步就近收敛，QuantEase 要 20×d_in） |
| [ADMM-Q (2605.11222)](https://arxiv.org/abs/2605.11222)（score 9） | 同样的"替换 GPTQ 取整器"路线，用 ADMM 求解，带收敛保证，在 SmoothQuant / SpinQuant 流程里替换；Qwen3-8B 上 W3A16 PPL 12.85→10.06。是否与 CDQuant 做过对比：未核实 |
| [Two-Stage Grid Optimization (2602.02126)](https://arxiv.org/abs/2602.02126)（score 9） | 只管分组量化的 scale：GPTQ 前用重构损失初始化、GPTQ 后用坐标下降闭式更新细化。与本文 OWC-CD 同类，但留着 GPTQ 取整不动 |
| [OJBKQ (2602.08376)](https://arxiv.org/abs/2602.08376)（score 7） | 把 PTQ 写成整数最小二乘，用 Babai-Klein 搜索：另一种"比 GPTQ 搜得更好"的路线 |
| [TurboBoA (2602.04929)](https://arxiv.org/abs/2602.04929)（score 9） | 改的是目标（考虑层间误差），不是求解器；与本文正交，可叠加 |
| [GPTQ-Babai (2507.18553)](https://arxiv.org/abs/2507.18553) | 给 GPTQ 提供几何解释和误差界，是"GPTQ 为什么有效"的理论；本文是"怎么搜得比 GPTQ 更好"的实证 |

判断：**不是 re-skin，但新意是工程层面的**。"用坐标下降解逐层整数最小二乘"这个思路与 QuantEase 同期出现（论文自己说明），
本文的独立贡献是贪心选择、随机块 BCD、OWC / OWC-CD 初始化，以及"替换 QuIP/FrameQuant 里的 GPTQ"这个定位。
没有理论保证（和 GPTQ-Babai 那种定理不同）：全部证据是困惑度和零样本表。

## 实验与工程可落地性

- **方法**：CD（Alg.1）、BCD k=2（Alg.2）、OWC（按 γ 网格选缩放；γ=1 即 MinMax）、OWC-CD（分组量化的 γ 选择）。
- **精度（困惑度，w2a16g128，GPTQ→BCD(k=2)）**：Gemma-1 7B 375.2→68.7；PaLM2-Otter 10.82→9.82；Gemma-2 9B 13.79→13.66（小）。
  3 比特 g128：Gemma-1 7B 15.56→13.00；4 比特差距只在小数点后第二位。收益集中在 ≤3 比特和小模型。
- **作零件**（INT2 逐通道，GPTQ→BCD）：QuIP 里 Gemma-1 7B 25.94→19.17；Gemma-2 27B 11.91→11.55（CD 11.48）；FrameQuant 里 Gemma-2 27B 10.96→10.61。
  引言写 ~5%，按表我算 3–4%。
- **例外**：并非处处优于 GPTQ，PaLM2-Otter w3a16g128 上 CD（6.655）略差于 GPTQ（6.635）；Otter/Bison 3 比特下游平均分基本持平（55.88→56.16，62.58→62.69）。
- **运行时间（量化阶段，Gemma-2 27B，8×H100）**：整模型 CD ≈ 2× GPTQ，BCD ≈ 一个数量级；单 H100 上差距更大（FFN 上 CD 一个数量级、BCD 两个数量级）。
  慢来自 gather/scatter；两条提速法：FFN2 只用 CD（Table 8，质量几乎不变）；CD 步数砍到 d_in/8（Table 22，质量几乎不变，时间可降到与 GPTQ 相当）。
- **推理侧**：输出格式与 GPTQ 相同（均匀整数 + scale/bias），不需要新 kernel。论文自己的时延实验显示 FFN 权重量化才有收益（Gemma-2 27B 解码步 FFN 16→8 位 26.18→17.41 ms；
  注意力量化几乎无收益），所以主实验只量化 FFN，注意力固定 INT8。
- **规模/覆盖**：只测 Gemma-1 7B、Gemma-2 9B/27B 和不对外开放的 PaLM2（Gecko/Otter/Bison）；没有 Llama/Qwen，没有 MoE；weight-only，没有激活量化主实验。
- **待确认**：论文无代码，CD/BCD 的实际速度取决于 gather/scatter 的实现（作者自己猜在 CPU 上做会更快，未验证）；校准峰值显存未给；在 ROCm / AMD GPU 上 argmin + 索引更新的性能未知。
  对 Quark 这类已有 GPTQ 的流程，有意义的验证是：同一层、同一校准集上比较 GPTQ 与 CD 的重构误差（论文给的参照：Gemma-2 9B 第一层 FFN INT3 逐通道，GPTQ 0.1449 / CD 0.1362 / BCD 0.1357，相对全 0 解）。

## 趋势定位

`ptq` 桶里"GPTQ 之后怎么更准"是长期热点：2026 年就有 ADMM-Q、Two-Stage Grid、OJBKQ、TurboBoA 等一批。
CDQuant 属于**较早的一批"替换取整器"工作**（2024），后来的工作大多沿同一目标函数，要么换更强的求解器（ADMM、Babai-Klein），要么改目标（层间误差）。
今天读它的价值是：一个简单、容易自己实现的基线，对照后来者（比如 ADMM-Q）的增益是否值得更复杂的实现。

## Triage 结论

未 triage。若要入 seed，建议理由：属于 `ptq` 桶的"取整求解器"谱系参考文献，而非可直接部署的工程成果（无代码、PaLM2 不可复现）。
