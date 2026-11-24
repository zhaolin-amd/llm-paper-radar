# IGSP — arXiv:2606.03328

> 剪枝校准集"影响不大"的结论，是把能力**平均**掉的假象。拆成 General / Commonsense / Code / Math 四维后，
> 校准数据的困惑度（PPL）对 General 是正相关、对 Math / Code 是负相关（反号权衡），任何单一来源都顾此失彼。
> 解法是**多源混合**：四个能力池各取 1/4（mix_balanced），LLaMA-3.1-8B 在 SparseGPT 60% 下总保留率 58.8，
> 比 C4 默认高 18.8、比最好的单源 MetaMath 高 8.8。没有现成对齐语料时，用 IGSP（分维度提示自生成 + 每维 PPL 区间 + 贪心最小化 4-gram 重复）自动造，
> 比 Self-Cal / SGS 高 2–5 分，但仍比真实数据混合低 9.7 分（差距几乎全在 Code / Math）。

本文件是这篇 paper 的**解读总入口**。radar-native 分析在下方，各角度详读见导航。

## 论文

- **标题**：Calibration Data Trade-offs Across Capability Dimensions: Why Multi-Source Mixing Matters for High-Sparsity LLM Pruning
- **链接**：[abs](https://arxiv.org/abs/2606.03328) · [PDF](https://arxiv.org/pdf/2606.03328v1) · code：正文称"released code"，未给链接
- **作者**：Hu Xu, Zhaolong Xing, Congcong Liu, Jiaxing Wang, Zhida Jiang, Junshi Huang, Zhen Chen, Jianfeng Xu
- **发表**：2026-06-02 · cs.LG, cs.AI
- **许可**：CC BY 4.0——中文翻译为完整翻译

## 解读导航

| 角度 | 文件 | 内容 |
|---|---|---|
| 原理故事 | [paper.org](paper.org) | 故事版 |
| 中文伴读 | [reading.org](reading.org) | 选择性精读 + 疑点核对 |
| 中文翻译 | [translation_zh.md](translation_zh.md) | 全文翻译（含附录、公式、表、图） |
| 溯源倒读 | [../../paper-river/IGSP-2606.03328.org](../../paper-river/IGSP-2606.03328.org)（中）· [_en](../../paper-river/IGSP-2606.03328_en.org)（英） | 已有，沿用：SparseGPT → Wanda → Williams & Aletras → Self-Cal → SGS → IGSP |

数据来源：[`data/summarized/2026-06-02.json`](../../data/summarized/2026-06-02.json)（radar 摘要与打分）+ arXiv HTML / PDF 全文。

## Radar 记录

- composite 8，bucket `survey`，topic_relevance 4，practicality 4，未触发 hard gate。**未 triage**。
- radar 记的"最大模型 LLaMA-3.1-8B"偏保守：表 1 也列了 LLaMA-3.1-70B，但正文说 70B 在 60% 下只跑完了 Wanda + C4（见下方疑点 1）；另有 OPT-6.7B 的部分结果（只评了 General / Commonsense）。

## 15 个校准来源（附录 C.1，每种 128 条）

| 类别 | 数据集 |
|---|---|
| 通用文本 | C4、DCLM-10B、Wikipedia、WikiText-2、arXiv |
| 指令 / 故事 | Alpaca、Dolly、TinyStories |
| 常识 | HellaSwag、OpenBookQA、WinoGrande |
| **代码** | **MBPP、Code-Alpaca** |
| **数学** | **GSM8K、MetaMath** |

- 主表只跑了 7 个（C4、Wikipedia、GSM8K、MetaMath 等），另外 8 个只进了相关性分析（n=15，每个来源 8 个 seed）。
- 多源混合按"主要对应哪个评测维度"把语料分成四个池：mix_balanced = 1:1:1:1；mix_4_3_3_math / _code 正文只叫"targeted weights"，比例没定义；3 个数对 4 个池，可能是丢了一个池（mix_4_3_3_math 的 Code 23.45 接近"去掉 Code 池"的 19.87，见 [reading.org](reading.org) 的推断）。
- 评测：General = LAMBADA ×2、TriviaQA；Commonsense = PIQA、ARC-Easy；Code = HumanEval、MBPP；Math = GSM8K、Minerva Math。**校准用了 MBPP / GSM8K，评测也用 MBPP / GSM8K**，Code / Math 的大幅提升部分可能来自同分布数据（训练集 vs 测试集的划分正文没说清）。
- 没有 reasoning trace（长 CoT）、agent / tool-use、多轮对话数据。

## 关键数字（LLaMA-3.1-8B，SparseGPT 60%，总保留率 S_tot）

| 校准 | Gen | Com | Code | Math | Total |
|---|---|---|---|---|---|
| C4（默认） | 60.74 | 84.35 | **0.41** | 14.47 | 39.99 |
| MetaMath（最好的单源） | 46.20 | 81.83 | 19.87 | 52.16 | 50.02 |
| mix_balanced | 59.69 | 85.32 | 52.34 | 37.76 | **58.78** |
| IGSP（自生成） | 54.09 | 85.51 | 37.34 | 19.36 | 49.08 |

- C4 校准下 Code 几乎归零；去掉 mix_balanced 里的 Code 池，Code 从 52.34 掉到 19.87（附录表 8）。
- **剪枝器决定敏感度**：SparseGPT 下各校准方案的总分跨度 18.8，Wanda 下只有 3.8。作者的解释是 SparseGPT 用校准 Hessian 的逆去**更新剩余权重**，Wanda 只用激活范数选 mask。
- **稀疏度决定敏感度**：30% 时 mix_balanced 只比 MetaMath 高 0.07，60% 时高 8.76；Wanda 40% 时 SGS（83.50）反而略高于 mix_balanced（82.69）。

## 同类对比 / novelty

bucket 内的 siblings（NoiseFloor、Broken on Arrival 等）都是量化或 MoE 的测量型论文，跟本文问题不同。真正的同类是 radar 里"校准数据怎么选"这条线：

| 论文 | 关系 |
|---|---|
| Williams & Aletras 2024 | 本文要推翻的"影响不大"结论的来源；他们也发现剪枝比量化对校准更敏感 |
| Self-Cal（Williams 2025）、SGS（Ji 2025） | 自生成基线；IGSP 的区别是分维度提示 + 每维 PPL 区间，而不是单一提示 + 单一 PPL 过滤 |
| [ZipCal 2603.16105](https://arxiv.org/abs/2603.16105) | 也优化词汇多样性（Zipf），不需要跑模型；本文的 4-gram Aggregation 是类似的信号，但没跟 ZipCal 比 |
| [2610.15578](https://arxiv.org/abs/2610.15578)（MoE 专家剪枝） | 结论相反：多样的预训练类数据 ≥ 测试分布数据。可能原因是它看的是通用域，本文把 Code / Math 单独拆出来了 |
| [FragCal-DC 2611.11079](../FragCal-DC-2611.11079/README.md) | 量化侧的对应：加入任务相关数据只对 2-bit + 迭代量化器明显有用，对 3-bit GPTQ / AWQ 帮助小 |
| [AlphaQ 2606.04980](../AlphaQ-2606.04980/README.md) | MoE 量化侧的旁证：PMQ 用 C4 / MATH / GitHub-Code 校准，位宽分配和各维度表现都偏向校准域 |

novelty：核心是**测量结论**（分维度看 + 反号权衡），方法（均匀混合四个池）本身很简单；IGSP 是在没有对齐语料时的补救，且不如真实数据混合。

## 趋势 / timing

`survey` 桶近一个月日均约 1 篇，平稳。"校准数据怎么选"在 radar 里是慢变量（FAQ、DPQ、FragCal-DC、2610.15578、2611.07654…），适合当一个小专题持续跟踪，不急。

## 工程可落地性（对 Quark / 量化）

- **能直接拿走的**：校准集按 General / Commonsense / Code / Math 四池均匀混合，128 条，各 32 条，零额外成本。若要保代码 / 数学能力，C4 / WikiText 单源是最差选择之一。
- **能否迁移到量化未验证**：本文只做剪枝，局限里明说没测量化和剪枝 + 量化。GPTQ 和 SparseGPT 都用 `H = XᵀX` 做 OBS 式补偿，机制上应该同样敏感；但 Williams & Aletras 与 FragCal-DC 都发现量化（尤其 4-bit）对校准来源不如剪枝敏感。合理预期：2–3 bit 或激进的 W4A4 有收益，W4A16 收益小——需要自己测。
- **MoE 未覆盖**（局限里明说）。结合 AlphaQ 的 PMQ 实验和 MoEQuant 的专家均衡问题，MoE 上校准域对专家激活影响更大，分维度混合可能更重要，但没有证据。
- **评测要分维度**：只看 PPL 或 PIQA / ARC 平均，会完全看不到 Code 从 52 掉到 0.4 这种塌缩。

## 疑点（读全文时核对）

1. **表 1 的 70B 数字和正文矛盾**：表 1 给出了 70B 在 60% 下所有行的数字，但 §5.3 说 70B 只跑完 Wanda + C4，其余行"标为 −"；正文还说 Magnitude 60% 没跑完，表里也有数。
2. **IGSP 的分维度数字两处不一致**：表 1 是 54.09 / 85.51 / 37.34 / 19.36，附录表 12 是 58.4 / 84.6 / 31.5 / 21.8（总分都约 49.1）。
3. **PPL 区间两处不一致**：表 11 的 Commonsense 是 [4, 8]、Code 是 [2, 4.5]，紧接的正文写 [5, 13]、[2, 5]；表 11 说区间来自评测任务，正文说来自对齐语料。
4. **"不需要能力对齐语料"打折扣**：IGSP 的 PPL 区间是用 Wikipedia / MBPP / GSM8K / MetaMath 等对齐语料测出来的（附录 A.3、E.5）。
5. **标题数字混用剪枝器**：摘要的 ρ = +0.71（General）、−0.53（Math）是 Wanda 下的，ρ = −0.59（Code）是 SparseGPT 下的；Wanda 的 Code（−0.31，p=0.27）和 SparseGPT 的 Math（−0.44，p=0.10）都不显著，没有一个剪枝器在三个维度上同时显著。48 个检验也没做多重比较校正。
   "只有 Aggregation 和 Distortion 有预测力"也不成立：SparseGPT 下 Variety（+0.56）、Granularity（−0.58）同样过线。
6. **IGSP 低于最好的单源 MetaMath**（49.08 vs 50.02，且在每个稀疏度都低于 MetaMath），"SOTA"只限 SparseGPT 下的自生成组；Wanda 60% 下 IGSP（35.95）还低于 Self-Cal（36.60）。
   贪心 4-gram 去重这一步在 30 / 40 / 50% 都是去掉更好（–no-diversity 胜出），正文只承认了 50%。
7. **OPT-6.7B 上** mix_balanced ≈ C4，IGSP 低于 C4（只评了 General / Commonsense，且总分按四维平均，缺失维度算 0）。
