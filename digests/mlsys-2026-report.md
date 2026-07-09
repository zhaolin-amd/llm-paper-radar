# MLSys 2026 — LLM Inference Deployment Optimization Trend Report

## Subfield distribution

| Subfield | # Papers |
|---|---|
| compiler_kernel_fusion | 11 |
| scheduling_batching | 8 |
| long_context_pd_disaggregation | 7 |
| kv_cache | 7 |
| speculative_decoding | 5 |
| multi_gpu_heterogeneous | 5 |
| moe_inference | 4 |
| quantization | 2 |
| other: neuron sparse activation | 1 |
| other: tensor lossless compression | 1 |
| other: automated inference optimization | 1 |
| other: cold start latency | 1 |
| other: diffusion LM inference | 1 |
| other: RAG inference optimization | 1 |
| other: production serving optimization | 1 |
| other: mobile DVFS inference scheduling | 1 |

## compiler_kernel_fusion (11 papers)

**Core problems:** Papers in this subfield tackle the gap between rapidly evolving LLM model/hardware designs and the labor-intensive, hand-tuned nature of GPU kernel engineering. Recurring problems include: (1) writing and maintaining high-performance kernels (attention, GEMM, fused ops) is slow and requires deep hardware expertise, especially for new hardware (Blackwell, AMD CDNA) or new attention variants; (2) static kernel templates/compilers can't handle dynamic, data-dependent control flow (variable sequence lengths, sparsity patterns, MoE routing) needed for efficient LLM serving; (3) multi-GPU/distributed inference wastes time on serialized compute-communication phases and underutilized resources during operator execution; (4) numerical precision tricks (quantization, integer-only pipelines) introduce new bottlenecks (e.g., softmax) that need custom low-level solutions; (5) there's no standardized way to generate, benchmark, and safely deploy AI-generated or automatically-compiled kernels into production serving stacks.

**Representative papers:**
- Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel
- Wave: A Symbolic Python DSL And Compiler for High-Performance Machine Learning
- FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling
- FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems

**Method commonalities:** Commonalities: nearly all methods raise the level of abstraction above raw CUDA/HIP/assembly (DSLs, compiler IRs, tile-based primitive libraries) so that expert-level performance can be achieved without manual low-level tuning per hardware/model combination — e.g. Wave and Flashlight embed kernel authoring in Python/compiler passes, HipKittens and ParallelKittens extend the ThunderKittens tile-primitive philosophy to AMD GPUs and multi-GPU settings respectively, and Event Tensor introduces a new IR-level abstraction (dependency-encoded tiled tasks) specifically to support dynamic megakernels. Several papers focus on exploiting asynchrony/overlap as the core lever for speedups: FlashAttention-4 redesigns pipelining around asynchronous MMA on Blackwell, TokenWeave and DynaFlow overlap communication/computation or heterogeneous operators, and ParallelKittens generalizes overlap primitives to multi-GPU kernels. A second cluster targets attention-specific bottlenecks via algorithmic/numerical changes rather than new abstractions: BLASST (dynamic block sparsity via thresholding) and IntAttention (integer-only softmax pipeline) both modify the attention computation itself rather than the compiler/DSL layer, targeting decode/edge efficiency. FlashInfer-Bench diverges by not proposing a new kernel or DSL at all, instead building infrastructure (benchmarking + substitution) to close the loop between kernel generation (potentially by any of the above methods, or AI-generated) and production deployment in existing serving engines (vLLM/SGLang). Divergence in generality vs. specialization: general-purpose compiler/DSL papers (Wave, Flashlight, Event Tensor, HipKittens, ParallelKittens) aim to support arbitrary kernels/attention variants, whereas BLASST, IntAttention, and FlashAttention-4 are narrowly scoped to attention kernels but push deeper into algorithm-hardware co-design for that specific operator.

**All papers in this subfield:**
- [FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems](https://openreview.net/forum?id=IyryZno8Hh)
- [Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel](https://openreview.net/forum?id=PJqFhAbUHa)
- [HipKittens: Fast and Furious AMD Kernels](https://openreview.net/forum?id=xxSSrndQrI)
- [Wave: A Symbolic Python DSL And Compiler for High-Performance Machine Learning](https://openreview.net/forum?id=gcXV1E8HRH)
- [ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels](https://openreview.net/forum?id=Cv5e5uRXFb)
- [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://openreview.net/forum?id=mN5RtvuYl3)
- [Flashlight: PyTorch Compiler Extensions to Accelerate Attention Variants](https://openreview.net/forum?id=lboOMA8XWr)
- [BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding](https://openreview.net/forum?id=6INSBXTQ4x)
- [DynaFlow: Transparent and Flexible Intra-Device Parallelism via Programmable Operator Scheduling](https://openreview.net/forum?id=i0yqC9954S)
- [IntAttention: A Fully Integer Attention Pipeline for Efficient Edge Inference](https://openreview.net/forum?id=CPCRITwAaP)
- [TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference](https://openreview.net/forum?id=rh2Ylffkq6)

## scheduling_batching (8 papers)

**Core problems:** Papers in this subfield tackle the problem of making LLM inference serving efficient and SLO-compliant under real-world resource constraints, by improving how requests, tokens, and system resources (compute, memory, power) are scheduled and batched. Sub-themes include: (a) maximizing throughput for large-scale batch/offline inference via smarter batching and prefix reuse (BatchLLM); (b) navigating deployment configuration trade-offs (hardware, parallelism, batching vs. disaggregation) at massive production scale (Meta's deployment paper, AIRS); (c) energy/power-aware scheduling that exploits SLO slack (BEAM); (d) adaptive/dynamic resource allocation that reacts to workload fluctuations at runtime, via quantization swapping, KV cache resizing, early-exit model selection, or elastic scaling in serverless settings (MorphServe, FaaScale, HELIOS); (e) fine-grained request/KV-cache scheduling to avoid head-of-line blocking and meet latency SLOs on advanced hardware (SuperInfer). Common thread: reconciling heterogeneous, bursty, or resource-constrained workloads with strict latency/throughput/energy SLOs through smarter scheduling, batching, and resource-adaptation policies rather than purely algorithmic model changes.

**Representative papers:**
- BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching
- Optimizing Deployment Configurations for LLM Inference
- MorphServe: Efficient and Workload-Aware LLM Serving via Runtime Quantized Layer Swapping and KV Cache Resizing
- SuperInfer: SLO-Aware Rotary Scheduling and Memory Management for LLM Inference on Superchips

**Method commonalities:** Commonalities: nearly all papers treat scheduling/batching decisions as a resource-allocation optimization problem under SLO or throughput constraints, often introducing new scheduling policies or memory/resource managers rather than modifying model architecture; several exploit slack or heterogeneity (latency slack for energy in BEAM, prefix overlap for throughput in BatchLLM, workload fluctuation for adaptive compression in MorphServe, exit-point flexibility in HELIOS) to gain efficiency without accuracy loss; system-level focus on KV cache and memory management recurs (MorphServe, SuperInfer) as the key bottleneck resource. Divergences: the optimization target varies — pure throughput (BatchLLM), energy/power (BEAM), latency/SLO adherence (SuperInfer, HELIOS), deployment cost/config search (Meta paper), or elasticity/scaling speed (FaaScale, AIRS); some operate at request-scheduling granularity (SuperInfer, BEAM) while others operate at model/layer or config granularity (MorphServe, HELIOS, Meta paper) or infrastructure/network granularity (FaaScale's multicast weight transfer). Evaluation settings also diverge from offline batch workloads (BatchLLM) to live/production serving at massive scale (Meta paper, AIRS) to specialized hardware (SuperInfer's superchips).

**All papers in this subfield:**
- [AIRS: Scaling Live Inference in Resource Constrained Environments](https://openreview.net/forum?id=g1RWik4Gy1)
- [BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching](https://openreview.net/forum?id=IuVHde07l6)
- [Optimizing Deployment Configurations for LLM Inference](https://openreview.net/forum?id=gEbKQeIdxB)
- [BEAM: Joint Resource-Power Optimization for Energy-Efficient LLM Inference under SLO constraints](https://openreview.net/forum?id=BfNBXM8CCT)
- [MorphServe: Efficient and Workload-Aware LLM Serving via Runtime Quantized Layer Swapping and KV Cache Resizing](https://openreview.net/forum?id=PDu13oOl4G)
- [FaaScale: Unlocking Fast LLM Scaling for Serverless Inference](https://openreview.net/forum?id=jgL8LuOVyT)
- [HELIOS: Adaptive Model And Early-Exit Selection for Efficient LLM Inference Serving](https://openreview.net/forum?id=CV52m9NJFK)
- [SuperInfer: SLO-Aware Rotary Scheduling and Memory Management for LLM Inference on Superchips](https://openreview.net/forum?id=RuslSHdIHa)

## long_context_pd_disaggregation (7 papers)

**Core problems:** These papers target the latency and efficiency bottlenecks of long-context / multi-request LLM serving, primarily focused on Time-to-First-Token (TTFT) and throughput. The core problems are: (1) prefill is compute-heavy and its latency directly determines TTFT, especially when context is long, retrieved externally, or shared/overlapping across requests; (2) naive prefill-decode co-location or full-batch prefill wastes GPU cycles and causes head-of-line blocking, motivating disaggregation of prefill from decode (and in multimodal cases, encode) onto separate resources or scheduling paths; (3) heterogeneous request shapes (short vs. long prompts, streamed/incrementally-arriving context, multi-turn conversations, multi-agent shared prefixes) require request- or length-aware scheduling/batching rather than one-size-fits-all treatment; (4) redundant computation across requests that share context (RAG chunks, cached prefixes, agent conversation histories, common spans) is wasted work that can be reused or overlapped instead of recomputed.

**Representative papers:**
- Stream2LLM: Overlap Context Streaming and Prefill for Reduced Time-to-First-Token
- PLA-Serve: A Prefill-Length-Aware LLM Serving System
- Beyond the Buzz: A Pragmatic Take on Inference Disaggregation
- ContextPilot: Fast Long-Context Inference via Context Reuse

**Method commonalities:** Commonality: nearly all papers treat prefill as a distinct, separable phase that should be disaggregated, overlapped, or specialized away from decode (and from other phases like retrieval or encoding) — via dedicated instances/resources (PLA-Serve, TriInfer, "Beyond the Buzz"), pipelining/overlap in time (Stream2LLM streams context into prefill; FlashAgents streams tokens between agents), or by cutting redundant prefill work via reuse/caching (ContextPilot's context index; the span-query paper's generalized cache-locality abstraction). Most also introduce request-aware scheduling/batching that exploits structural knowledge of the workload (prompt length in PLA-Serve, retrieval arrival pattern in Stream2LLM, agent/task structure in FlashAgents and span queries, modality in TriInfer) rather than treating all requests uniformly.

Divergences: the granularity and target of disaggregation differs — PLA-Serve and "Beyond the Buzz" disaggregate at the level of prefill vs. decode instances/hardware, TriInfer adds a third encode stage for multimodal inputs, while Stream2LLM and FlashAgents disaggregate/overlap at a finer temporal/streaming granularity within a single phase rather than across separate hardware pools. ContextPilot and the span-query paper focus on cache/data reuse and locality optimization rather than instance placement or scheduling per se, making them more about algorithmic/data-structure innovation (context indexing, generalized query abstraction) than systems scheduling. "Beyond the Buzz" is distinct in being an empirical/measurement study across a huge design space rather than proposing a new system.

**All papers in this subfield:**
- [Stream2LLM: Overlap Context Streaming and Prefill for Reduced Time-to-First-Token](https://openreview.net/forum?id=FuRo7Ur5Ib)
- [PLA-Serve: A Prefill-Length-Aware LLM Serving System](https://openreview.net/forum?id=dzjCkSEDyG)
- [Beyond the Buzz: A Pragmatic Take on Inference Disaggregation](https://openreview.net/forum?id=NqC5tcBsa0)
- [TriInfer: Hybrid EPD Disaggregation for Efficient Multimodal Large Language Model Inference](https://openreview.net/forum?id=nNovi8fvGN)
- [ContextPilot: Fast Long-Context Inference via Context Reuse](https://openreview.net/forum?id=RnKvDy1jv2)
- [Using Span Queries to Optimize Cache and Attention Locality](https://openreview.net/forum?id=qcGGSXpFcM)
- [FlashAgents: Accelerating Multi-Agent LLM Systems via Streaming Prefill Overlap](https://openreview.net/forum?id=m14PPUfgEc)

## kv_cache (7 papers)

**Core problems:** These papers all target the memory/compute overhead of the KV cache in LLM inference. The core problems break into: (a) GPU memory capacity — KV cache grows linearly with sequence length/batch size and quickly exceeds GPU memory (FlexiCache, Kitty); (b) redundant/wasted compute and memory bandwidth during decode, especially from re-scanning or re-transferring KV for similar/overlapping queries or beams (MAC-Attention, Locality-Aware Beam Scheduling); (c) system-level throughput bottlenecks when combining sparsity/eviction with paged KV cache management (OPKV); (d) reliability — recovering KV state after hardware/device failures without expensive recomputation (GhostServe); and (e) reasoning-specific overhead where long chain-of-thought generation with large reasoning models produces oversized KV caches that need workload-aware compression (SkipKV). Collectively, the subfield is about making the KV cache smaller, faster to access, more reusable, and more robust, without hurting accuracy or throughput.

**Representative papers:**
- OPKV: A High-Throughput Plugin-Driven Framework for Recallable Sparsity in Paged KV Cache Systems
- Kitty: Accurate and Efficient 2-bit KV Cache Quantization with Dynamic Channel-wise Precision Boost
- SkipKV: Selective Skipping of KV Generation and Storage for Efficient Inference with Large Reasoning Models
- MAC-Attention: a Match-Amend-Complete scheme for fast and accurate attention computation

**Method commonalities:** Commonalities: nearly every method exploits some form of redundancy or non-uniform importance in the KV cache — across tokens (sparsity/eviction in OPKV, SkipKV), across heads/channels (FlexiCache's stable/unstable heads, Kitty's channel-wise precision), across queries/beams (MAC-Attention's query reuse, Locality-Aware Beam Scheduling's beam/token locality), or across time (GhostServe's incremental checkpointing). Most are training-free / plug-in style additions layered on top of existing serving systems (paged attention, standard decoding) rather than requiring model retraining, and most explicitly co-design algorithm choices with system/memory-hierarchy considerations (GPU vs. host memory offloading, precision boosting, erasure coding) to preserve accuracy while cutting memory or bandwidth. Divergences: the axis of compression differs — quantization/precision (Kitty) vs. token/page eviction (OPKV, SkipKV, FlexiCache) vs. computation reuse (MAC-Attention) vs. redundancy/fault-tolerance (GhostServe) vs. scheduling for a specific workload like beam search test-time compute (Locality-Aware Beam Scheduling). Some target general inference (OPKV, FlexiCache, Kitty, GhostServe) while others target specific workloads like long CoT reasoning (SkipKV) or test-time beam search on consumer GPUs (Locality-Aware Beam Scheduling), and their goals split between throughput/latency (OPKV, MAC-Attention, Locality-Aware Beam Scheduling), memory footprint (FlexiCache, Kitty), reliability (GhostServe), and accuracy-under-compression (SkipKV, Kitty).

**All papers in this subfield:**
- [OPKV: A High-Throughput Plugin-Driven Framework for Recallable Sparsity in Paged KV Cache Systems](https://openreview.net/forum?id=EB5bgzv4qA)
- [FlexiCache: Leveraging Temporal Stability of Attention Heads for Efficient KV Cache Management](https://openreview.net/forum?id=GgX6dPJx9M)
- [MAC-Attention: a Match-Amend-Complete scheme for fast and accurate attention computation](https://openreview.net/forum?id=b6HBRCejb7)
- [SkipKV: Selective Skipping of KV Generation and Storage for Efficient Inference with Large Reasoning Models](https://openreview.net/forum?id=0EsV9SIm8p)
- [Locality-Aware Beam Scheduling for Efficient Test-Time Compute with a Consumer-grade GPU](https://openreview.net/forum?id=dTo8jAXm9K)
- [GhostServe: A Lightweight Checkpointing System in the Shadow for Fault-Tolerant LLM Serving](https://openreview.net/forum?id=xKjYiUgeOK)
- [Kitty: Accurate and Efficient 2-bit KV Cache Quantization with Dynamic Channel-wise Precision Boost](https://openreview.net/forum?id=r3mQiuYKIN)

## speculative_decoding (5 papers)

**Core problems:** Papers in this cluster tackle how to make speculative decoding (SD) actually deliver its promised throughput/latency gains in real LLM serving. Two sub-threads emerge: (a) improving the drafter itself — better draft model architectures/capacity (PRISM disaggregates parameters per predictive step; SpecDiff-2 and TiDAR use non-autoregressive diffusion drafters to produce multiple tokens per forward pass; SpecGen reuses the target model itself with sparse attention as a cheap self-drafter, avoiding a separate draft model and its alignment/staleness issues) and (b) critically evaluating whether SD's theoretical speedups hold up under realistic, production-grade serving conditions (the vLLM systematic study exposes gaps between paper-reported and measured speedups due to batching, workload diversity, and engine overheads). Underlying all of these is the classic SD tension: draft quality/acceptance-rate vs. draft cost, and how architectural or systems choices shift that tradeoff.

**Representative papers:**
- PRISM: Parametrically Refactor Inference for Speculative Decoding Draft Models
- SpecDiff-2: Scaling Diffusion Drafter Alignment For Faster Speculative Decoding
- Speculative Decoding: Performance or Illusion?
- Accelerating Large-Scale Reasoning Model Inference with Sparse Self-Speculative Decoding

**Method commonalities:** Commonality: nearly all method papers keep the standard draft-then-verify SD protocol (propose k tokens cheaply, verify in parallel with the target model, accept/reject to preserve output distribution) and target the same bottleneck — reducing the cost or improving the quality of the draft step to raise effective speedup and tokens-per-second, evaluated via throughput/acceptance-rate on decoding benchmarks. Divergence is in where they get the cheap draft from: PRISM restructures/decouples the draft model's own parameters across steps (still autoregressive, still a separate small model); SpecDiff-2 and TiDAR both depart from autoregressive drafting entirely, using discrete diffusion to draft multiple tokens non-autoregressively in one shot, differing in how they reconcile diffusion drafts with AR verification (SpecDiff-2 calibrates a diffusion drafter to an AR verifier as two separate models, TiDAR fuses drafting and verification into a single hybrid forward pass); SpecGen avoids a separate draft model altogether via self-speculation with sparse attention (PillarAttn) as the "draft," trading model-diversity for zero staleness/alignment cost, targeting long-context reasoning workloads specifically. The "Performance or Illusion?" paper is methodologically distinct — it contributes no new drafting technique but instead systematically benchmarks existing SD variants in a production engine (vLLM) across batch sizes/workloads, serving as an empirical check on whether the gains claimed by architecture papers like the other four actually materialize at scale.

**All papers in this subfield:**
- [PRISM: Parametrically Refactor Inference for Speculative Decoding Draft Models](https://openreview.net/forum?id=cvU2HuuxEf)
- [Speculative Decoding: Performance or Illusion?](https://openreview.net/forum?id=fzkqtezFEi)
- [SpecDiff-2: Scaling Diffusion Drafter Alignment For Faster Speculative Decoding](https://openreview.net/forum?id=o42VU86ZsV)
- [TiDAR: Think in Diffusion, Talk in Autoregression](https://openreview.net/forum?id=onfxEjoE4L)
- [Accelerating Large-Scale Reasoning Model Inference with Sparse Self-Speculative Decoding](https://openreview.net/forum?id=yeqrwcWjPu)

## multi_gpu_heterogeneous (5 papers)

**Core problems:** These papers tackle efficient LLM/generative-model inference and serving across multiple or heterogeneous GPUs (and even client CPU/GPU systems) under real-world constraints: limited/heterogeneous hardware (VRAM-constrained clients, mixed GPU types), unreliable or irregular GPU availability (fault tolerance), heterogeneous workloads needing cost-aware routing, communication bottlenecks in disaggregated/distributed inference (RDMA portability), and load imbalance introduced by sparse/structured computation patterns in parallel execution (sequence parallelism for sparse attention). The unifying theme is squeezing more throughput/cost-efficiency/resilience out of imperfect, non-uniform multi-GPU (or GPU+CPU) environments rather than assuming a clean, homogeneous, fully-available cluster.

**Representative papers:**
- RaidServe: High-performance Resilient Serving
- BOute: Cost-Efficient LLM Serving with Heterogeneous LLMs and GPUs via Multi-Objective Bayesian Optimization
- fabric-lib: RDMA Point-to-Point Communication for LLM Systems
- db-SP: Accelerating Sparse Attention for Visual Generative Models with Dual-Balanced Sequence Parallelism

**Method commonalities:** Commonality: nearly all methods center on load/resource balancing across GPUs as the key mechanism — RaidServe balances compute and memory across GPUs to survive irregular GPU availability; BOute balances load via query routing and GPU utilization; db-SP balances sparse attention load across GPUs along head/block dimensions; the client-inference paper adapts computation placement to CPU/GPU conditions. Several also emphasize portability/generality across hardware (fabric-lib across NIC vendors, client work across CPU/GPU configs, BOute across heterogeneous GPU types). Divergence is in the layer of abstraction: fabric-lib and RaidServe operate at the systems/communication and fault-tolerance layer (low-level infra), db-SP and the client-inference paper focus on computation/parallelism scheduling within a model's forward pass (algorithm-system co-design), while BOute sits at a higher orchestration/scheduling layer using optimization (Bayesian) rather than static balancing heuristics. Target models also diverge: most target autoregressive LLMs/VLMs/MoE, while db-SP specifically targets diffusion transformers (visual generative models), showing the same heterogeneous/parallel-GPU problem generalizing beyond text LLMs.

**All papers in this subfield:**
- [Efficient, VRAM-Constrained xLM Inference on Clients](https://openreview.net/forum?id=VKqQYg6JPb)
- [RaidServe: High-performance Resilient Serving](https://openreview.net/forum?id=5pl9fdbEkq)
- [BOute: Cost-Efficient LLM Serving with Heterogeneous LLMs and GPUs via Multi-Objective Bayesian Optimization](https://openreview.net/forum?id=ZVQb92umqX)
- [fabric-lib: RDMA Point-to-Point Communication for LLM Systems](https://openreview.net/forum?id=SjVa05wEiY)
- [db-SP: Accelerating Sparse Attention for Visual Generative Models with Dual-Balanced Sequence Parallelism](https://openreview.net/forum?id=XgKteNxNe0)

## moe_inference (4 papers)

**Core problems:** All four papers target the inefficiency of serving large Mixture-of-Experts (MoE) LLMs at scale, where the sparse, expert-parallel architecture introduces overheads that dense-model serving systems don't face. The concrete pain points are: (1) blocking all-to-all communication between expert-parallel GPUs stalling computation; (2) a broader "MoE serving tax" — systematic throughput/latency degradation relative to FLOP-equivalent dense models, arising across prefill/decode phases and parallelism strategies; (3) load imbalance across experts caused by skewed, dynamic token-to-expert routing, which underutilizes some GPUs while overloading others; and (4) scheduling conflicts between prefill and decode requests (or across layers) that make it hard to simultaneously hit TTFT/TBT SLOs and maximize throughput given fixed compute/memory/interconnect budgets. Collectively, the subfield is diagnosing and fixing the specific structural costs (communication, imbalance, phase interference) that MoE sparsity and expert parallelism impose on real-world serving.

**Representative papers:**
- FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models
- CRAFT: Fine-Grained Cost-Aware Expert Replication For Efficient Mixture-of-Experts Serving
- From Tokens to Layers: Redefining Stall-Free Scheduling for MoE Serving with Layered Prefill
- Demystifying the Mixture of Experts Serving Tax

**Method commonalities:** Commonality: all four treat expert parallelism (EP) as the root cause of serving inefficiency and attack it from a systems/scheduling angle rather than purely algorithmic (quantization/pruning) angle; three of the four (FarSkip-Collective, CRAFT, Layered Prefill) propose concrete mechanisms while Demystifying is an empirical/measurement study that motivates and categorizes the problems the others solve. All share the goal of better overlapping or reorganizing computation and communication/scheduling to hit latency SLOs (TTFT/TBT) and raise throughput under fixed hardware budgets.

Divergences in method locus: FarSkip-Collective intervenes at the model architecture level (adding skip connections to decouple compute and communication dependencies, enabling communication-computation overlap). CRAFT intervenes at the expert-placement/replication level (dynamically replicating "hot" experts based on cost-aware load estimates to fix token-level imbalance under EP). Layered Prefill intervenes at the request/layer scheduling level (redefining scheduling granularity from token-batches to layers to avoid stalls and meet per-phase SLOs). Demystifying doesn't propose a system change but provides the taxonomy/measurement (2-3x tax, broken down by phase and parallelism strategy) that frames why architecture-, placement-, and scheduling-level fixes are all needed — effectively serving as the diagnostic backbone motivating the other three's differing intervention points.

**All papers in this subfield:**
- [FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models](https://openreview.net/forum?id=ruOpvLzsGV)
- [Demystifying the Mixture of Experts Serving Tax](https://openreview.net/forum?id=lELxqcgrsN)
- [CRAFT: Fine-Grained Cost-Aware Expert Replication For Efficient Mixture-of-Experts Serving](https://openreview.net/forum?id=zdRvzU9ZCe)
- [From Tokens to Layers: Redefining Stall-Free Scheduling for MoE Serving with Layered Prefill](https://openreview.net/forum?id=yyDbI3HXco)

## quantization (2 papers)

**Core problems:** Both papers target post-training quantization for efficient LLM inference, specifically how to reduce numerical precision (bit-width) of weights/activations while minimizing accuracy loss, under hardware-friendly formats (block-wise/microscaling). The shared core problem is that uniform, naively-chosen quantization parameters (precision assignment across features, or scale factors within a block) leave accuracy on the table, and the papers seek smarter, low-overhead ways to choose these parameters without sacrificing the system efficiency/throughput gains that motivate quantization in the first place.

**Representative papers:**
- MixLLM: LLM Quantization with Global Mixed-precision between Output-features and Highly-efficient System Design
- Search Your Block Floating Point Scales!

**Method commonalities:** Commonality: both start from the observation that a "one-size-fits-all" quantization decision (same precision for all output features, or a fixed max-magnitude scale per block) is suboptimal, and both introduce a search/selection step over a discrete design space (which features get higher precision; which scale value per block) guided by an error/importance metric, while keeping compatibility with efficient GPU kernels/hardware-native formats (mixed-precision GEMM, microscaling/BFP). Divergence: MixLLM operates at the granularity of output-feature-wise precision allocation (a global, cross-layer importance-based mixed-precision assignment) plus co-designed system/kernel support, whereas ScaleSearch operates within a fixed block/format (BFP/microscaling) and searches only the scale factor per block, leaving precision assignment uniform. MixLLM's contribution is more about "where to spend more bits" (allocation across features/layers) and system co-design; ScaleSearch's contribution is more about "how to best summarize a block's dynamic range" (scale selection) purely at the numerical level.

**All papers in this subfield:**
- [MixLLM: LLM Quantization with Global Mixed-precision between Output-features and Highly-efficient System Design](https://openreview.net/forum?id=VBbMRQ4VOc)
- [Search Your Block Floating Point Scales!](https://openreview.net/forum?id=innqECyZPK)

## other: neuron sparse activation (1 papers)

**Core problems:** Papers in this subfield tackle the problem of LLM inference inefficiency caused by dense activation of all neurons/parameters for every input, despite evidence that only a subset of neurons is actually relevant to any given task or token. The core challenge is how to identify and exploit this activation sparsity dynamically (at inference time, without retraining) to reduce compute/memory cost while preserving task-specific accuracy, since sparsity patterns are highly input- and task-dependent rather than static or hardware-friendly by default. This connects to broader inference deployment optimization goals (latency, throughput, cost reduction) but focuses on the "which neurons can be skipped for this input" attribution/decision problem rather than architectural changes (e.g., MoE) or quantization.

**Representative papers:**
- Attribution-based Sparse Activation in Large Language Models

**Method commonalities:** Only one paper is provided for this subfield, so cross-paper comparison is limited. Its approach: use attribution methods (estimating each neuron's contribution to the output for a specific input/task) to decide which neurons to deactivate per-input, rather than relying on a fixed/global sparsity mask or a trained gating network. This is training-free/retraining-free and adapts dynamically to downstream task and runtime input distribution, distinguishing it from static magnitude-based pruning or learned sparse-activation predictors (e.g., ReLU-based or auxiliary router networks) that other work in this space typically uses. Without additional papers, broader commonalities/divergences cannot be established from the given data.

**All papers in this subfield:**
- [Attribution-based Sparse Activation in Large Language Models](https://openreview.net/forum?id=gJFigZeb5D)

## other: tensor lossless compression (1 papers)

**Core problems:** This subfield addresses the growing storage/bandwidth/memory-transfer cost of ML tensors (weights, activations, KV-cache, gradients, checkpoints) in LLM training and inference pipelines. Unlike lossy quantization, the goal here is lossless compression: shrinking tensor representations without any loss of numerical fidelity, while keeping compression/decompression fast enough (ideally GPU-friendly, low-latency, high-throughput) to not bottleneck inference serving or training I/O. Key sub-problems include: (a) exploiting the non-uniform, often near-Gaussian or otherwise skewed statistical distribution of tensor values (especially floating-point exponent/mantissa patterns) to approach the Shannon entropy limit of the data; (b) doing so with practical encoding schemes (e.g., ANS, Huffman, arithmetic coding) that are efficient in both compression ratio and computational overhead; (c) partitioning or modeling the value space so a single entropy coder can be applied effectively despite heterogeneous tensor statistics across layers/channels; (d) integrating the compressor into deployment-relevant contexts such as checkpoint storage, cross-device tensor transfer, or memory-bound inference stages (e.g., KV-cache) where reducing bytes moved directly improves latency/throughput.

**Representative papers:**
- Shannonic: Efficient Entropy-Optimal Compression for ML Workloads

**Method commonalities:** Based on the single paper provided (Shannonic), the methodological approach centers on: (1) treating compression as an entropy-coding problem grounded in Shannon's source coding theorem, aiming for near-entropy-optimal rates rather than heuristic or ad-hoc compression; (2) an offline/calibration phase that analyzes tensor value distributions and partitions the value space into optimally chosen subranges; (3) using Asymmetric Numeral Systems (ANS) as the entropy encoder for high throughput and near-optimal coding efficiency. Only one paper is available, so no cross-method divergence can be characterized.

**All papers in this subfield:**
- [Shannonic: Efficient Entropy-Optimal Compression for ML Workloads](https://openreview.net/forum?id=NhMxI0GbB8)

## other: automated inference optimization (1 papers)

**Core problems:** Deploying LLMs efficiently in production requires navigating a large, complex search space of infrastructure choices (hardware type, parallelism strategy, batching, quantization, serving framework configuration, etc.) while meeting service-level objectives (SLOs) like latency and throughput. Most enterprises lack the specialized systems/ML expertise needed to manually tune these configurations, especially when workloads are heterogeneous and infrastructure is diverse. This subfield tackles the automation of this optimization process itself—building tools/frameworks that can automatically discover near-optimal deployment configurations.

**Representative papers:**
- Meeting SLOs, Slashing Hours: Automated Enterprise LLM Optimization with OptiKIT

**Method commonalities:** Only one paper was provided (OptiKIT), so cross-paper comparison cannot be substantiated. The method emphasizes an automated, toolkit-based approach that treats deployment optimization as a search/decision problem over heterogeneous infrastructure configurations, targeting explicit SLO satisfaction as the objective.

**All papers in this subfield:**
- [Meeting SLOs, Slashing Hours: Automated Enterprise LLM Optimization with OptiKIT](https://openreview.net/forum?id=om4H7AI2hc)

## other: cold start latency (1 papers)

**Core problems:** Papers in this "cold start latency" subfield focus on the engine/system startup overhead of LLM inference frameworks (e.g., vLLM), i.e., the time from process launch/model load to when the serving engine is ready to handle the first request. Core problems include: (a) understanding where startup time is spent (model loading, CUDA graph capture, kernel compilation via torch.compile, memory profiling, weight initialization, distributed/worker setup), (b) how newer architectural features trade off steady-state throughput/latency gains against increased cold-start cost, and (c) how this latency affects deployment scenarios such as autoscaling, serverless inference, and elastic GPU provisioning.

**Representative papers:**
- Breaking the Ice: Analyzing Cold Start Latency in vLLM

**Method commonalities:** Only one paper is provided. Its methodological approach is empirical/measurement-based: systematically profiling and benchmarking vLLM to decompose startup latency into its architectural components, rather than proposing a new algorithm.

**All papers in this subfield:**
- [Breaking the Ice: Analyzing Cold Start Latency in vLLM](https://openreview.net/forum?id=eoEobeKTNZ)

## other: diffusion LM inference (1 papers)

**Core problems:** Diffusion language models (DLMs) offer parallel, any-order text generation as an alternative to autoregressive LLMs, but suffer from two core inference bottlenecks: (1) they require many denoising/sampling steps to reach acceptable quality, and (2) their non-causal, bidirectional attention structure is incompatible with standard KV caching, so naive implementations recompute the full sequence's attention at every step, negating potential speed gains.

**Representative papers:**
- CDLM: Consistency Diffusion Language Models for Faster Sampling

**Method commonalities:** Only one paper (CDLM) is available. Its approach applies consistency-model-style distillation so that a single or few forward passes can finalize multiple tokens at once, cutting the number of denoising steps, while also adapting the model so KV-cache-like reuse becomes feasible despite the bidirectional diffusion process.

**All papers in this subfield:**
- [CDLM: Consistency Diffusion Language Models for Faster Sampling](https://openreview.net/forum?id=eB8yjR6alL)

## other: RAG inference optimization (1 papers)

**Core problems:** Papers in this "RAG inference optimization" bucket target the systems-level latency/throughput/memory bottlenecks that arise when serving retrieval-augmented generation pipelines at scale: retrieval latency from large datastores stalling decoding, poor overlap between retrieval and generation, GPU memory pressure from keeping indexes resident, and scheduling inefficiencies under concurrent RAG requests.

**Representative papers:**
- TeleRAG: Efficient Retrieval-Augmented Generation Inference with Lookahead Retrieval

**Method commonalities:** Only one paper (TeleRAG) is available. Its approach predicts/prefetches future retrieval needs ("lookahead retrieval") based on partial generation state so datastore lookups overlap with ongoing decoding rather than executing as a blocking serial step, with minimal additional GPU memory footprint.

**All papers in this subfield:**
- [TeleRAG: Efficient Retrieval-Augmented Generation Inference with Lookahead Retrieval](https://openreview.net/forum?id=YsOyCpMUYD)

## other: production serving optimization (1 papers)

**Core problems:** Papers in this "production serving optimization" bucket focus on deploying LLMs/SLMs in real industrial systems under strict, business-driven latency and throughput SLAs: compressing/adapting a large model into a small one that retains task quality, designing the serving system around that model to meet throughput at production traffic volumes, and reconciling model-side and systems-side optimization as a joint engineering problem.

**Representative papers:**
- Scaling Up Large Language Models Serving Systems for Semantic Job Search

**Method commonalities:** Only one paper is available. Its methodology combines model-side compression (a small decoder-only LM) with systems-side serving optimizations tailored to a real production task (semantic search at LinkedIn), reporting applied, deployment-driven lessons rather than a generic algorithm.

**All papers in this subfield:**
- [Scaling Up Large Language Models Serving Systems for Semantic Job Search](https://openreview.net/forum?id=re82zZczHj)

## other: mobile DVFS inference scheduling (1 papers)

**Core problems:** This subfield addresses the inefficiency of running LLM inference on mobile/edge devices when hardware-level power management (DVFS) is controlled independently and heuristically by OS-level governors, without awareness of LLM inference workload characteristics (e.g., prefill vs. decode phases). Uncoordinated per-component DVFS decisions cause either wasted energy or latency spikes — the reported 23-40% longer latency or 5-17% higher energy consumption.

**Representative papers:**
- Rethinking DVFS for Mobile LLMs: Unified Energy-Aware Scheduling with CORE

**Method commonalities:** Only one paper is provided. Its approach profiles existing mobile LLM inference frameworks' independent per-component DVFS behavior, diagnoses the lack of coordination and workload-phase awareness as the root cause, and proposes a unified scheduling layer that jointly makes energy-aware frequency/voltage decisions across CPU/GPU/memory.

**All papers in this subfield:**
- [Rethinking DVFS for Mobile LLMs: Unified Energy-Aware Scheduling with CORE](https://openreview.net/forum?id=PSyHQ8kVUT)
