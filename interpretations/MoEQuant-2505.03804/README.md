# MoEQuant — arXiv:2505.03804

> 把 GPTQ / AWQ 直接搬到 MoE 上，会碰到两种"不均衡"：一是**专家之间**，WikiText2 这类单领域校准集把 token 集中喂给少数专家，冷门专家几乎没被校准；
> 二是**专家内部**，每个 token 进专家时带着不同的门控权重 `c_i`，但 GPTQ 算 Hessian 时把所有 token 一视同仁。
> MoEQuant 对应两招：**EBSS** 让模型自己生成校准文本，用"困惑度 + 专家负载标准差"做 beam 式剪枝，得到专家均衡、又贴合模型分布的校准集；
> **AGQ** 把门控权重乘进逐层重构误差，Hessian 变成按 `c_i` 加权。两招都是插件，可接在 AWQ / GPTQ 上。只做权重量化（W4 / W3）。

本文件是这篇 paper 的**解读总入口**。radar-native 分析在下方，各角度详读见导航。

## 论文

- **标题**：MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance
- **链接**：[abs](https://arxiv.org/abs/2505.03804) · [PDF](https://arxiv.org/pdf/2505.03804v1) · code：原文只说"将开源"，正文未给链接（是否已开源未核实）
- **作者**：Xing Hu, Zhixuan Chen, Dawei Yang, Zukang Xu, Chen Xu, Zhihang Yuan, Sifan Zhou, Jiangyong Yu
- **发表**：2025-05-02（v1）· cs.LG / cs.AI · **ICML 2025**（会议版一作为 Zhixuan Chen，所以后续论文常引作 "Chen et al. 2025"）
- **许可**：arXiv 非独占许可（非 CC-BY）——中文翻译只做逐节复述，不是逐句翻译

## 解读导航

| 角度 | 文件 | 内容 |
|---|---|---|
| 原理故事 | [paper.org](paper.org) | 七拍故事版 |
| 中文伴读 | [reading.org](reading.org) | 选择性精读 |
| 中文翻译 | [translation_zh.md](translation_zh.md) | 全文中文复述（逐节释义，含公式/表/图） |
| 溯源倒读 | [../../paper-river/MoEQuant-2505.03804.org](../../paper-river/MoEQuant-2505.03804.org)（中）· [_en](../../paper-river/MoEQuant-2505.03804_en.org)（英） | 本次新生成 |

数据来源：radar 无打分记录（见下）；解读基于 arXiv HTML v1 全文。

## Radar 记录

- **未收录**：2025-05 发表，早于 radar 归档起点（2025-11），`resolve_paper.py` 只能 live 抓 arXiv，没有 composite / bucket / 摘要。
- **未 triage**。按内容应落在 `ptq` 桶。
- 但它在 radar 里**作为基线反复出现**：EAQuant（Table 7）、TileQ（主表 MoEQuant 列）、VSRAQ（相关工作）都拿它对比，是 MoE PTQ 这一支的公共参照点。

## 同类对比 / novelty

| 论文 | 关系 |
|---|---|
| GPTQ / AWQ | 宿主方法。MoEQuant++ = GPTQ（带 QuaRot 式离线 Hadamard 去离群）+ EBSS + AGQ；MoEQuant+ = AWQ + 两招 |
| GPTQ 的 Hessian `XX^T` | AGQ 只改了这一处：按门控权重加权 token。思路上等价于"带权最小二乘"，实现成本极低 |
| LLM-QAT 的自生成数据（Liu et al. 2023） | EBSS 的"让模型自己写校准集"来自这里；新加的是专家负载标准差这一项和 beam 剪枝 |
| [EAQuant (2506.13329)](../EAQuant-2506.13329/README.md) | **一个月后的直接竞争者**。同样处理"冷门专家校准不足"，但改成从真实训练数据补抽（批评 EBSS 自生成数据让对比不公平），并额外加了激活平滑和路由对齐；做的是 W-A 量化 |
| [TileQ (2605.09281)](../TileQ-2605.09281/README.md) | 把 MoEQuant 作为 MoE 专用 PTQ 基线：2-bit 下在 DeepSeek / Qwen1.5 / Qwen3 上崩溃（Qwen1.5-MoE PPL 5e5），但 Mixtral 上没崩（13.4，好于 GPTQ 的 15.3） |
| KBVQ-MoE (2602.11184) | 同团队后续工作，以 MoEQuant 为基线（Qwen1.5 2-bit PPL 583542 vs 9.61） |
| GEMQ (2605.23078) | 以 MoEQuant 为基线，数字为抄录 |
| [AlphaQ (2606.04980)](../AlphaQ-2606.04980/README.md) / [VSRAQ (2606.05688)](../VSRAQ-2606.05688/README.md) | 后续的位宽分配线 / 路由对齐线；VSRAQ 把 MoEQuant 归为"校准集构造"一类 |

判断：**是 MoE PTQ 的早期奠基作之一**——"冷门专家校准不足"和"门控权重应进入重构目标"这两个观察被后续论文普遍接受。
两招本身都很轻：AGQ 是一行 Hessian 加权，EBSS 是带约束的自采样。

## 实验与工程可落地性

- **设置**：只做权重量化，per-channel 对称，**无分组**；A6000；模型 Qwen-MoE-14B、DeepSeek-MoE-16B、Mixtral-8x7B（+ 两个 chat 版）。基线的校准集是 WikiText2 128 段。
- **4-bit 精度（9 任务平均）**：Mixtral GPTQ 53.42 → MoEQuant++ 55.58（FP 56.80）；DeepSeek 39.01 → 40.01（FP 40.86）；Qwen 49.00 → 49.59（FP 51.20）。
  收益集中在 **HumanEval / GSM8K** 这类生成任务（如 Mixtral GSM8K 57.92 → 61.79，HumanEval 27.60 → 32.15）；多选任务差别很小。
- **3-bit**：Mixtral 平均 45.04 → 49.75，GSM8K 22.59 → 43.21（翻倍）；DeepSeek 35.85 → 36.47，收益小。AWQ 系在 DeepSeek 3-bit 上 PPL 上千，MoEQuant+ 救不回来（5100）。
- **WikiText2 PPL 一律变差**（如 Mixtral 4.03 → 4.12）：基线用 WikiText2 校准，作者解释为基线过拟合，C4 PPL 改善（7.67 → 7.34）支持这一解释。
- **效率**：速度 1.17–2.08×、显存 3.27–3.74×，这是 4-bit 权重量化本身的收益，与 MoEQuant 无关（MoEQuant 不改推理图）。
- **成本**：EBSS 需要在模型上做 beam 宽度 `w=4` 的自回归生成，论文**没报生成校准集的耗时**；AGQ 无额外成本。
- **与 AMD / Quark**：AGQ 只需要在收集 Hessian 时把 router 的门控权重乘进去，任何 GPTQ 实现都能加；EBSS 是独立的校准集生成步骤，可以离线跑一次复用。两者都不需要新 kernel。

**读全文发现的问题**（我自己核对表格，非 agent 转述）：
- **摘要说"DeepSeekMoE-16B 4-bit 下 HumanEval 提升超过 10 个点"，表格里找不到**：base 模型 22.56 → 25.00（+2.44），chat 模型 13.41 → 21.95（+8.54）。超过 10 点的只有 Mixtral 3-bit GSM8K（+20.6）这类别的组合。
- **表 2 与附录表 8 数字不一致**（同为 4-bit chat 模型）：Qwen-chat 的 MoEQuant+ HumanEval 表 2 为 18.92、表 8 为 13.60，GSM8K 22.34 vs 20.87，MMLU 53.22 vs 55.34；DeepSeek-chat 的 MoEQuant+ GSM8K 47.42 vs 45.41。
- **式 19 自相矛盾**：式 18 的损失按 `c_i` 线性加权，对应 Hessian 应为 `X diag(c) X^T`；式 19 写成 `(X·c)(X·c)^T`（即 `c_i^2` 加权）又等于 `(X·c)X^T`，两种写法不等价，实际实现用哪种不明。
- **消融（表 5）里 EBSS 是主力，AGQ 单独效果弱**：AGQ 单用 Qwen 49.00 → 49.02、DeepSeek 39.01 → 39.50、Mixtral 53.42 → 54.24；EBSS 单用 49.21 / 39.87 / 55.15。
  附录 A.1 的文字与表 5 对不上：文字说 DeepSeek 上 EBSS 约 +1.3%、AGQ 约 +2%，表里正好反过来（EBSS +0.86、AGQ +0.49 分）；文字说"AGQ 单用在 Mixtral 上不优于 GPTQ"，表里 AGQ 54.24 > GPTQ 53.42。
- **§5.2 说"MoEQuant 平均超过原始性能 1%"不成立**：表 1 三个模型 MoEQuant++ 平均 49.59 / 40.01 / 55.58，都低于 FP 的 51.20 / 40.86 / 56.80。
- **图 1 的相对增益与表格不符**（翻译 agent 核对，我抽查了 DeepSeek GSM8K 一项：图 1 标 +5.85%，表 1 是 19.18 → 19.18 没变）。
- 无分组、无激活量化，与现在业界常用的 W4 g128 / FP4 设置有距离。

## 趋势定位

2025 年中是 MoE PTQ 的起步期（MoEQuant、EAQuant、MoQa、MC-MoE 几乎同时出现）；到 2026 年 `ptq` 桶里 MoE 专用工作已经分化成位宽分配（AlphaQ、Q-Strata、DuoRate…）、路由对齐（VSRAQ）、权重表示（TileQ）几支，
MoEQuant 主要作为**基线和问题定义**被引用。现在读它的价值是理解"专家间 / 专家内不均衡"这两个概念，方法本身不必复现。

## Triage 结论

未 triage（radar 未收录）。适合作为 `ptq` 桶 MoE 方向的**参考文献**，不建议作为"可直接部署的方法"入 seed：无分组实验、代码状态不明、摘要与表格有不一致。
