# CrossKV — arXiv:2608.03893

> 线上服务换模型（级联、路由、对话中途升级）每次都要让新模型重新 prefill 一遍完全相同的上下文。
> 本文发现：**同一家族内**（KV head 数和 per-head 维度相同）的 source/target 模型之间，KV cache
> 存在很强的线性结构——单层 source 就能解释 target keys 56% 的方差（Qwen3 14B→32B），不需要训练
> 一个神经网络翻译器，一个 per-head 的 **closed-form ridge 回归**就够了。

本文件是这篇 paper 的**解读总入口**。radar-native 分析在下方，各角度详读见导航。

## 论文

- **标题**：Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse
- **链接**：[abs](https://arxiv.org/abs/2608.03893) · [PDF](https://arxiv.org/pdf/2608.03893v1)（CC-BY 4.0）
- **作者**：Taekyung Heo, Rasoul Shafipour, Ritchie Zhao, Maximilian Golub, Mohammad Mahdi Kamani, Ritika Borkar, Makesh Tarun Chandran, Pantea Zardoshti, Bita Darvish Rouhani
- **发表**：2026-08-04 · arXiv cs.LG

## 解读导航

| 角度 | 文件 | 内容 |
|---|---|---|
| 原理故事 | [paper.org](paper.org) | 七拍故事版——从"换模型就要重新还 prefill 的债"到"家族内 KV 关系简单到一个线性映射就够" |
| 中文伴读 | [reading.org](reading.org) | 选择性精读，关键段落信达雅翻译 + 碰撞提问（归档模式） |
| 中文翻译 | [translation_zh.md](translation_zh.md) | 全文中文复述（覆盖全部正文 + 附录，18 张表、9 张图均已嵌入；原文 CC-BY 4.0 授权，复述形式与仓库其他条目保持一致） |
| 溯源倒读 | [../../paper-river/CrossKV-2608.03893.org](../../paper-river/CrossKV-2608.03893.org)（中）· [_en](../../paper-river/CrossKV-2608.03893_en.org)（英） | 倒读法脉络：SGLang/Prompt Cache（同模型前缀缓存）→ DroidSpeak（跨模型但同架构）→ IAM（只搬注意力模式）→ C2C/LatentAlign/PrefillShare（跨架构但需训练）→ 本文（同家族内，训练自由） |

数据来源：[data/summarized/2026-08-04.json](../../data/summarized/2026-08-04.json)（radar 打分记录）。

## Radar 记录

- composite=**8**  ·  bucket=**kv_cache**  ·  topic_relevance=4  ·  practicality=4  ·  hard_gate=no
- format/method：跨模型 KV cache 线性映射，per-head ridge 回归
- largest model tested：Qwen3-32B
- accuracy：73-98% standalone-prefill 准确率保留（六对中四对），两对退化严重，MLP 补救后 HellaSwag retention 最多 +37 pp
- inference perf：prefill 速度提升 2.7-25x，支持多轮对话 handoff 场景下保持稳定
- calibration cost：500 条 FineWeb-Edu 序列（每条 1024 token），ridge 回归拟合，成本极低
- peak memory：unknown（论文未披露）

## Triage 状态

尚未 triage（既未 accept 也未 reject）。

## 同类对比 / novelty

`kv_cache` 桶内近邻大多做**同模型**内的 cache 压缩（量化、淘汰、摘要），本文是桶里少见的**跨模型** cache 复用方向：

| 论文 | 路线 |
|---|---|
| [1-Bit KV Cache VQ (2610.03027)](https://arxiv.org/abs/2610.03027) | query-guided channel weighting + covariance-aware grouping 的 1-bit KV 量化 |
| [QuantMLA (2609.36760)](https://arxiv.org/abs/2609.36760) | MLA 双路径的 INT4/INT2 KV 量化 |
| [LOCKS (2607.24555)](https://arxiv.org/abs/2607.24555) | page-local 谱摘要 + attention-mass top-page 选择，压缩同模型长上下文 KV |
| [RoPE-Aware Bit Allocation (2606.24033)](https://arxiv.org/abs/2606.24033) | RoPE 感知的逐块贪心比特分配 |

这些邻居全部默认"KV 只在同一个模型内部复用"，围绕**压缩率**做文章（量化到更低比特、更激进的淘汰）。
本文问的是一个桶里几乎没人问过的问题：**KV 能不能跨模型搬**？而且给出的答案出人意料地朴素——不需要
训练一个神经网络翻译器（C2C/LatentAlign 等并发工作的路线），同家族模型间一个 ridge 回归就够。
与桶内"压缩同一份 KV"的主流思路相比，这是维度完全不同的收益来源：省掉的不是存储/带宽，而是整段
prefill 计算本身。

## 趋势定位

`kv_cache` 桶近期活跃度持续偏高（9 月下旬到 10 月初稳定在每天 2-6 篇非 gated 论文），但几乎全部挤在
量化/淘汰/摘要这几条拥挤赛道上。跨模型 KV 复用目前还是一个边缘方向（本文自称是"首个问家族内线性映射
是否够用"的工作），值得现在深读——一旦 production 级多模型路由/级联场景铺开，这个方向大概率会从边缘
变成刚需。

## 工程可落地性（practicality=4）

- **方法简单**：per-head ridge 回归是 closed-form 解，拟合成本=500 条校准序列，没有梯度训练，没有额外
  推理时网络前向开销（映射矩阵在线下拟合好，线上只是一次矩阵乘法）。
- **直接对应真实场景**：级联（cheap→expensive 升级）、路由、对话中途换模型——这几个都是生产部署里
  真实存在、而非论文假想的场景。
- **限制也很明确**：只对**同一家族**（KV head 数、per-head 维度相同）的模型对有效，跨家族（比如
  Qwen 换 Llama）这个方法直接不适用——这是刻意限定的范围，不是尚未发现的短板。
- **两对退化严重**：六对里有两对线性映射效果差，需要额外训一个非线性 MLP 才能补救，意味着"要不要
  上这个方案"需要先跑一遍方差分析（R²）判断这对 source/target 是否落在线性结构好的区域，不能盲目
  all-in。
- **peak memory 未披露**：映射矩阵本身（per-head、per-layer）的存储开销论文没给数字，如果要在 AMD
  推理栈上落地，这是需要补的一项验证。
