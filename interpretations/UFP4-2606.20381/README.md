# UFP4 — arXiv:2606.20381

> 当前 FP4 训练硬件（NVIDIA Blackwell/Rubin、AMD MI350）用的 E2M1 格点本身不均匀，
> 舍入误差系统性偏向负方向（Shrinkage Bias），跨层乘法累积；业界用来压 outlier 的
> RHT 非但没治好，反而把数值推进 E2M1 最不均匀的区段，放大了偏差。换成均匀格点
> E1M2/INT4，偏差从几何上直接消失，不用再在取整策略上打补丁。

本文件是这篇 paper 的**解读总入口**。radar-native 分析在下方，各角度详读见导航。

## 论文

- **标题**：Rethinking Shrinkage Bias in LLM FP4 Pretraining: Geometric Origin, Systemic Impact, and UFP4 Recipe
- **链接**：[abs](https://arxiv.org/abs/2606.20381) · [PDF](https://arxiv.org/pdf/2606.20381.pdf)（CC-BY 4.0）
- **作者**：Qian Zhao, Kunlong Chen, Changxin Tian, Zhonghui Jiang, Haitao Zhang, Chaofan Yu, Peijie Jiang, Mingliang Gong, Jia Liu, Ziqi Liu, Zhiqiang Zhang, Jun Zhou
- **发表**：2026-06-18 · arXiv cs.AI

## 解读导航

| 角度 | 文件 | 内容 |
|---|---|---|
| 原理故事 | [paper.org](paper.org) | 七拍故事版《刻度不匀，数字缩水》——量化误差从"对称噪声"到"有方向的系统偏移"的认知转折 |
| 中文伴读 | [reading.org](reading.org) | 选择性精读，骨架段三层翻译（含收缩偏差几何推导的信达雅翻译）+ 碰撞提问（归档模式，Agent 代答） |
| 中文翻译 | [translation_zh.md](translation_zh.md) | 全文中文复述（覆盖 §1-7 + 附录 A-C；版权原因采用 section-by-section 复述而非逐字翻译；论文全部 12 张图 + Table 1/3 均已嵌入） |
| 溯源倒读 | [../../paper-river/UFP4-2606.20381.org](../../paper-river/UFP4-2606.20381.org)（中）· [_en](../../paper-river/UFP4-2606.20381_en.org)（英） | 倒读法脉络：FP8 → MXFP4 → FP4 All The Way/Quartet I → NVFP4+RHT → Quartet II/MS-EDEN → 本文 → 并发的 MixFP4 |

数据来源：[data/summarized/2026-06-19.json](../../data/summarized/2026-06-19.json)（radar 打分记录）。

## Radar 记录

- composite=**7**  ·  bucket=**qat**  ·  topic_relevance=4  ·  practicality=3  ·  hard_gate=no
- format/method：FP4 QAT，E1M2/INT4 均匀格点 + Random Hadamard Transform + 随机取整（仅 dY）
- largest model tested：MoE 124B（另含 Dense 1.5B、MoE 7.9B）
- accuracy：UFP4 在 Dense 1.5B / MoE 7.9B / MoE 124B 预训练中 BF16 相对 loss degradation 持续低于 E2M1 强基线
- inference perf：无端到端推理 speedup 数据，面向未来支持 E1M2/INT4 的加速器
- calibration cost：全量预训练级验证，需要 pretraining 规模计算资源，复现成本高
- peak memory：unknown（论文未披露）

## 同类对比 / novelty

`qat` 桶内近邻大多在 E2M1/NVFP4 框架内做校准、蒸馏或知识恢复，UFP4 走的是"跳出格式框架、从几何根因下手"：

| 论文 | 路线 |
|---|---|
| [QATFactory (2609.39223)](https://arxiv.org/abs/2609.39223) | NVFP4/MXFP4/Q4_K QAT/QAD 统一框架 |
| [Quantization-Aware Distillation (2601.20088)](https://arxiv.org/abs/2601.20088) | NVFP4 QAD，蒸馏恢复精度 |
| [Quantization-Aware Healing (2608.20953)](https://arxiv.org/abs/2608.20953) | MXFP4 QAH，KD 恢复 |
| [QUASAR (2608.13966)](https://arxiv.org/abs/2608.13966) | W2/W3/W4 QAT + loss-aware reconstruction |
| [ReQAT (2606.15682)](https://arxiv.org/abs/2606.15682) | MXFP4 W4A4KV4 QAT |
| [WinQ (2605.17471)](https://arxiv.org/abs/2605.17471) | sub-4-bit QAT 加速，Hessian 谱分析 + 权重重置 |

差异清晰：邻居们默认 E2M1/NVFP4 的格点已经是既定事实，围绕它做蒸馏/校准/加速；
本文反过来问"E2M1 的误差是随机噪声还是系统偏移"，答案是**有方向的系统偏移**——
这是质变而非改良，因为有方向的误差才会跨层乘法累积（O(N) 而非 O(√N)）。和它同时期
独立发表的 [MixFP4 (2605.31035)](https://arxiv.org/abs/2605.31035) 发现了同一个几何真相，
但选择让 E2M1/E1M2 按块自适应共存，而不是像本文一样彻底换格点——两条队伍殊途同归，
说明 E2M1 的格点问题已经无法忽视。

## 工程可落地性（practicality=3，偏低）

- 方法本身不需要新硬件指令（E1M2/INT4 的 bit 布局并不新鲜），但当前 Blackwell/MI350
  的原生 FP4 路径是围绕 E2M1 设计的——真要落地意味着倒逼硬件生态重新支持 E1M2/INT4
  作为一等公民，这个转变不会比算法层面的工作快。
- 这是**全量预训练级**验证，不是 PTQ：复现成本是跑一次完整的 Dense 1.5B / MoE 7.9B /
  MoE 124B 预训练，calibration cost 远高于本仓库常见的 PTQ 校准工作。
- 论文只报告训练 loss / BF16-relative loss degradation，**没有下游 benchmark**
  （MMLU/GSM8K 等）数据——这是 topic_relevance 不满分的主因，复现前需要自己补下游评测
  才能判断对实际任务精度的影响。
- peak memory 论文未披露（radar 标 unknown），复现前需要翻 PDF 附录或补实验确认。
- E1M2 的动态范围比 E2M1 窄 1 位指数，对值域跨度极大的张量（如 FFN 里的激活 spike）
  有潜在精度代价，论文自己也承认这是待解决的 trade-off。

## 趋势定位

`qat` 桶近期（09-29~10-07）日均约 1 篇，不算爆发期。composite=7 在桶内处于中上，
但几何分析视角（而非经验调参）在同类投稿里少见，是这个方向里少数"挖地基"而非
"框架内优化"的工作，值得优先深读。

## Triage 建议

尚未 triage。这篇对 Quark（推理侧量化）**不能直接抄配方**——它是预训练级理论工作，
不是 PTQ 配方；但它的诊断方法（统计每层量化误差的符号分布，几行代码就能区分"问题出在
格点几何还是取整策略"）值得借鉴到 PTQ 场景：如果在调 MXFP4/NVFP4 PTQ 配方时遇到
反直觉的精度损失，这个视角可能是排查方向。建议按"方法论参考"而非"直接可部署方案"
的标准决定是否 accept。
