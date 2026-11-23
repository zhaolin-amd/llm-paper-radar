# QDA — Closed-Loop LLM Quantization Design with Agents

arXiv:2611.23388 · Li, Zhao, Han, Zhu · MIT / NVIDIA · cs.LG · 2026-11-17 · CC-BY 4.0
[abs](https://arxiv.org/abs/2611.23388) · [pdf](https://arxiv.org/pdf/2611.23388) · 代码/库/evaluator/harness 均为 "will release"，暂无链接

一句话：让 Claude Code（Opus 5.5）为每个「模型×精度」自动设计 PTQ 配方——用与原模型在推理轨迹上的输出 KL 做分钟级打分、再用长生成检查抓循环/截断，配一个 21 方法、可叠加、自动适配新架构的库；产物仍是标准 GPTQ/AWQ/NVFP4 checkpoint，无推理开销。

## 导航
- [中文翻译（全文，CC-BY 4.0）](translation_zh.md)
- [中文精读](reading.org) · [原理与故事](paper.org)
- paper-river 溯源：[zh](../../paper-river/QDA-2611.23388.org) · [en](../../paper-river/QDA-2611.23388_en.org)
- radar 数据：[data/summarized/2026-11-19.json](../../data/summarized/2026-11-19.json) · [digest](../../digests/2026-11-19.md)

## Radar-native 分析 (D)

**Triage 状态**：2026-11-19 digest 已收录（composite 8，bucket `ptq`，topic_relevance 5，practicality 3）；未被 accept/reject，尚未进 seeds。

**对 radar 摘要的核对（已对照正文）**
- 数字吻合：+0.9/+1.4/+2.2 pt（W4A16/W4A8/W3A16）、64 GPU·h、GLM-5.3-Flash NVFP4 98.3% vs 96.9%（NVIDIA）/ 96.3%（Red Hat AI）。
- radar 摘要写"最大 320B"——属实，但**该规模只做了 NVFP4 一次搜索**，且从 FP8 checkpoint 起步、没有公开配方可比；W4A16/W4A8/W3A16 的主表最大是 35B-A3B MoE 和 27B dense。
- radar 的"+0.9/+1.4/+2.2"是 4 模型均值；MoE（Qwen3.6-35B-A3B）上只领先 +1.3 / +0.1，W3A16 还比 GSQ 低 0.2 pt（Table 2）。收益主要来自 dense / 线性注意力混合小模型（R1-Llama W3A16 +6.1 pt 是单点最大值，对比的是 GPTQ 33.9；该模型上 GSQ 仅 27.3，比 GPTQ 还弱，QDA 40.0）。
- "每次搜索 64 GPU·h"不含最终 test 评测，另有 $19–75 agent token；GLM 一次 6.5 h × 16 GB300 + $69。

**需要警惕的点**
- **agent 本身的贡献证据偏薄**：只在 Qwen3.6-35B-A3B 做 agent 消融（每格 3 次搜索）。W4A16 上 greedy/ASHA 与 QDA 差 ≤0.1 pt（Table 5a），收益主要来自库 + KL 评估器；agent 在 W4A8/W3A16 才拉开 0.2–1.2 pt。
- **固定配方已吃掉大半 W4A16 收益**：5/12 搜索趋同于"自生成数据 GPTQ + 两轮 scale distillation"，固定后 W4A16 追平/超过最佳公开（如 Qwen3.6 78.6 vs 公开 77.7，QDA 79.0）。W4A8、W3A16 上固定配方在 Qwen3.5 上不迁移（落后 QDA 2.2 / 5.0 pt）。实用上可以先抄这个模式。
- **KL 代理只是 82% 排序一致**（44 对），长生成检查靠 agent 自己挑题与解读，依赖 Opus 5.5 + 一套 task statement；换 agent/不同随机种子的稳定性只在 Qwen3.6 上报了 σ（0.1–0.2 pt）。
- **公开基线的强度**：GPTQ/AWQ 用 llm-compressor 默认值；W3A16 的 GSQ 在三个模型上是作者自己跑官方代码（30–52 GPU·h），并非全部为其发布 checkpoint；ParoQuant 用发布 checkpoint。基线调参投入 vs QDA 的 64 GPU·h 并不对等。
- 不可复现：库、evaluator、harness 尚未发布；agent 是闭源模型。

**同类对比 / novelty**（同 bucket `ptq`）
- 与 `ptq` 桶里其它论文（SoloQ、Replay the Curvature、Integer-Exact Microscaling 等）正交：它们提出新方法/格式，QDA 是"方法之上的选择与组合层"，并把它们当成可叠加零件。最近邻是 SoScale（同样在 GPTQ 补偿流程里做更好的 scale 选择）——QDA 的库里 scale 搜索只是 21 个零件之一。
- 想法的新意在系统：agent 写配方 + agent 不可见的 KL 评估器 + 评估对齐到真实 serving kernel（FP8 scale）。KL 当 PTQ 比较指标、agent 当优化器各自都不新（论文自己引了 EvoPress、llama.cpp KL、AIRA2 等）。

**趋势 / timing**：`ptq` 桶 ~1–6 篇/天，11-14、11-16 有小高峰；"agent 自动化压缩"在桶里是新方向。因作者是 Song Han 组 + NVIDIA 背景，大概率会成为该类工作的参照；值得精读第 3、4.2、6.3、6.4 节，其余可略。

**工程可落地性**（radar practicality 3，同意）
- 优点：checkpoint 与 GPTQ/AWQ/NVFP4 同格式同 kernel，零额外推理成本；方法库对新架构只需 adapter（GLM 72 行）；专家批量 GPTQ（16 → 0.42 GPU·h）可直接借鉴，对 MoE 在 Quark 里做 GPTQ 很实用。
- 成本：每个新模型 × 精度 64 GPU·h + agent token，一次性可接受，但要 8 卡/16 卡专用并依赖 Claude Code。
- AMD 侧：论文全部在 GH200/GB300 + vLLM Marlin/CUTLASS/NVFP4 kernel 上；ROCm kernel 的 W4A8 FP8 scale 存储方式未必相同——"码选择对齐 kernel 存储的 scale"这一点需要按 ROCm kernel 重新核对。MXFP4/AMD 格式没覆盖。
- unknown：peak memory（radar 记 unknown，正文未报）、KV cache/激活量化以外的组合。

**待 PDF 核查**：附录 C（KL 评估细节、σ 的估计）、附录 D（GLM adapter）、附录 E（task statement 与 ModelOpt 对照设置）、Table 7（解码速度）、Table 8（21 个方法清单）。
