# W4. How labs post-train reasoning/agentic models (2025-2026), and what transfers to a 15-day, one-node effort

Researcher W4, 2026-10-18. Scope: concrete post-training pipelines (stage order, data, RL environment scale, distillation, merging, QAT, evaluation, regression control) at Qwen, DeepSeek, Moonshot, Zhipu, MiniMax, NVIDIA, Ai2, Prime Intellect, Mistral and ByteDance Seed; then small-team case studies. Builds on `research/reports/finetune-plan-2026-10-17.md` (the "prior plan") and `research/data/finetune-research-2026-10-17/training.md`; it does not repeat MiniCPM/WSD, ReST-EM, GiGPO, RAGEN, Router Replay or SWE-Gym, which are covered there.

Legend: [E] = read in the cited primary source in this session; [S] = via a search-result summary or secondary page only; [I] = my inference; UNVERIFIED = I could not confirm it.

## 1. Qwen: Qwen3 → Qwen3-Next → Qwen3.5/3.6 → Qwen3.8-Flash-Next

### Takeaway
Qwen has published no post-training recipe for Qwen3.8-Flash-Next. Its tech report covers architecture and pre-training only. An internal "Qwen3.8-Flash-Next-Base" exists, but only the post-trained checkpoint and its FP8 copy are public. So our job is continued post-training of an RL-tuned model. The most transferable Qwen evidence is from Qwen3: on-policy distillation beat RL at about 1/10 of the GPU-hours, and the later general-purpose stages cost about 2-3 points on hard reasoning.

### Cited Findings
- **Qwen3 pipeline (4 stages)** [E]: (1) long-CoT cold start (verified math/code/logic/STEM, QwQ-32B candidates); (2) reasoning RL with GRPO on **3,995 query-verifier pairs**, which took Qwen3-235B-A22B on AIME'24 from **70.1 to 85.1 in 170 RL steps**; (3) thinking-mode fusion, a continual SFT on rejection-sampled stage-2 outputs plus non-thinking data, with `/think` and `/no_think` flags; (4) general RL over **20+ tasks** with rule-based, reference-judge and preference-RM rewards. Source: [Qwen3 Technical Report, arXiv 2505.09388 §4.1-4.4](https://arxiv.org/html/2505.09388).
- **Strong-to-weak distillation** [E]: off-policy distillation, then on-policy distillation that minimizes KL to a Qwen3-32B or 235B teacher on the student's own samples. **Table 21 (Qwen3-8B):** off-policy distillation 55.0 AIME'24, + RL 67.6 (**17,920 GPU-h**), + on-policy distillation **74.4 (1,800 GPU-h)**. LiveCodeBench v5: 42.0 / 52.9 / 60.3. RL left pass@64 unchanged; on-policy distillation raised it. Source: [arXiv 2505.09388 §4.5, §4.7, Table 21](https://arxiv.org/html/2505.09388).
- **The general stages cost specialist skill** [E]: thinking-mode AIME'24 fell **83.8 (stage 2) → 81.4 (stage 4)** and LiveCodeBench v5 **68.4 → 65.7**. Qwen "accepted this trade-off for overall versatility." Source: [arXiv 2505.09388 Table 22](https://arxiv.org/html/2505.09388).
- **Qwen3-Next** used GSPO to handle RL stability and efficiency for hybrid attention combined with high-sparsity MoE [S]. Source: [Qwen3-Next-80B-A3B-Thinking model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking), read through a search summary. GSPO "defines the importance ratio based on sequence likelihood" and "notably stabilizes Mixture-of-Experts (MoE) RL training" [E]. Source: [GSPO, arXiv 2507.18071 abstract](https://arxiv.org/abs/2507.18071), 2025-07-24.
- **Qwen3.5** (2026-02-17): RL "scaled across ... million-agent environments with progressively complex task distributions" [S, secondary]. Source: [HF blog by mlabonne](https://huggingface.co/blog/mlabonne/qwen35). Qwen's own blog wording is UNVERIFIED, because [qwen.ai/blog?id=qwen3.5](https://qwen.ai/blog?id=qwen3.5) renders only through JavaScript.
- **Qwen3.8-Flash-Next tech report** [E]: "On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability", dated 2026-08-26. Sources: [GitHub tech_report.pdf](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf), [arXiv 2608.30320](https://arxiv.org/abs/2608.30320).
  - The report covers architecture, Muon, hyperparameter scaling and stability. **§4 evaluates only "Qwen3.8-Flash-Next-Base"** (Table 11: MMLU-Pro 73.23, MATH 72.78, SWEBench-Pretrain 50.99). It has no post-training or RL section.
  - Two relevant points. First, sparse residual writes left pre-training loss and benchmarks "almost unaffected, but the quality degraded clearly after post-training" (§2.2). Second, the conclusion says "the tightest bottleneck is evaluation throughput: a cheaper mid-scale probe that reliably predicts post-training ordering" is what they lack (§5).
- **No public base checkpoint** [E]: the HF API for author=Qwen, queried 2026-10-18, lists only `Qwen3.8-Flash-Next` and `Qwen3.8-Flash-Next-FP8`, plus `Qwen3.8-27B`, `Qwen3.8-2.4T-A95B` and their FP8 copies. Source: `https://huggingface.co/api/models?author=Qwen&search=3.8`. The model card says "Training Stage: Pre-training & Post-training" and lists "MTP: 1 layer, trained with multi-steps" [E]. Source: [HF model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
- **Optimizers:** the card says "Muon and AdamW optimizers are applied to specific weight categories" [E] ([HF card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)). The README recommends Unsloth, Swift and LLaMA-Factory for "SFT, DPO, GRPO, etc." [E] ([GitHub README](https://github.com/QwenLM/Qwen3.8-Flash-Next)).

### Inferences
- [I] Qwen's own numbers say the cheapest strong signal is a teacher's per-token distribution on the student's own samples, not RL. Our only same-family teacher that fits one node is Flash-Next itself. The 2.4T-A95B model is far too large for 8×H200 (about 2.4 TB even at FP8 against 1.1 TB of HBM). So any on-policy distillation we do must be self-distillation with privileged context (§5).
- [I] The tech report's warning that changes invisible in pre-training "degrade after post-training" is the same failure we would see: a LoRA can look fine on loss and short evals and still hurt long-horizon play. Gate on game play, not on loss.
- [I] Qwen took a 2-3-point loss on hard reasoning in exchange for generality. We can make the opposite trade, toward one benchmark. Use general benchmarks as a canary, not as a hard "<2-point drop" gate as the prior plan does. Held-out games are the gate.

### Gaps
- No Qwen3.8 post-training details: RL algorithm, environments, agentic RL scale, distillation from 2.4T-A95B, and whether QAT was used are all unknown.
- I did not check whether Qwen3.8-2.4T-A95B is served by any API with logprobs, which would allow an off-policy text teacher.

## 2. DeepSeek: R1 → V3.2 → V4

### Takeaway
DeepSeek's line moved from "one pipeline with mixed RL" (R1, V3.2) to "train specialists with SFT+GRPO, then merge them into one student by **full-vocabulary on-policy distillation**" (V4, 2026-04). V4 also added **FP4 quantization-aware training of the MoE expert weights during post-training**. RL compute is at least 10% of pre-training cost. For us, only the stabilisation tricks and the QAT idea transfer.

### Cited Findings
- **R1** [E]: cold start ("thousands" of examples) → RL → rejection-sampling SFT (reasoning + non-reasoning) → second RL of **1,700 steps**, with preference rewards only in the last 400; GRPO with 16 outputs per question. No neural ORM/PRM, as they are "susceptible to reward hacking". Source: [arXiv 2501.12948 §2.2, §3.2](https://arxiv.org/html/2501.12948).
- **V3.2 specialists** [E]: specialists in maths, programming, logic, general agentic tasks, agentic coding and agentic search, all fine-tuned from the same base with large-scale RL, generate data for the final model. The distilled model is "only marginally below" the specialists, and the gap is "effectively eliminated through subsequent RL." Source: [DeepSeek-V3.2, arXiv 2512.02556 §3](https://arxiv.org/html/2512.02556v1).
- **V3.2 RL budget** [E]: "exceeding 10% of the pre-training cost" (§1, §4.1). Reasoning, agent and alignment RL were merged into **one RL stage** to avoid "catastrophic forgetting issues commonly associated with multi-stage training" (§3). Source: [arXiv 2512.02556](https://arxiv.org/html/2512.02556v1).
- **V3.2 GRPO stabilisers** [E], all from §3.1 of [arXiv 2512.02556](https://arxiv.org/html/2512.02556v1):
  - an unbiased K3 KL estimator;
  - **off-policy sequence masking**, which drops negative-advantage sequences whose log-ratio drift exceeds δ;
  - **Keep Routing**, which replays the sampled expert routes in training ("crucial for RL training stability of MoE models");
  - **Keep Sampling Mask**, which reuses the top-p/top-k mask in training.
- **V3.2 agentic task synthesis** [E]: **1,827** synthesized general-agent environments (4,417 tasks, kept only if pass@100 > 0), 24,667 code-agent tasks, 50,275 search tasks and 5,908 code-interpreter tasks, about 85k prompts in total. RL on the synthetic general-agent data alone improved Tau2Bench, MCP-Mark and MCP-Universe; RL on code and search alone did not. Source: [arXiv 2512.02556 §3.2.3 Table 1, §4.3](https://arxiv.org/html/2512.02556v1).
- **V4** (arXiv v1, 2026-04-26) [E], all from [DeepSeek-V4, arXiv 2606.19348](https://arxiv.org/html/2606.19348):
  - "the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD)" (§5.1).
  - "more than ten teacher models" are distilled into one student with reverse KL computed over the **full vocabulary**. Sampled-token estimates give "high variance ... and often cause training instability." OPD avoids "the performance degradation often encountered in traditional weight-merging or mixed RL techniques" (§5.1.2).
  - **FP4 (MXFP4) QAT** in post-training covers the MoE expert weights and the indexer QK path, with 1×32 sub-blocks inside 128×128 FP8 blocks and a straight-through estimator. Rollouts use native FP4. QAT also covers the teacher and reference models (§5.2.1).

### Inferences
- [I] V4 confirms a 2026 lab consensus: consolidation happens through on-policy KL to a teacher, not through weight averaging or a mixed SFT.

### Gaps
- I did not read V4's number of specialists or its OPD compute. I did not read the R1 distillation-vs-RL table; it is in the supplement, which was not in the fetched text.

## 3. Moonshot (Kimi): k1.5 long2short → K2 → K2-Thinking INT4 QAT

### Takeaway
Kimi published the most direct precedent for our R3 lever, "shortest rejection sampling," and for our deployment format: K2-Thinking was trained with **INT4 weight-only QAT on the MoE experts during post-training**, and every benchmark is reported at INT4.

### Cited Findings
- **k1.5 long2short** [E]:
  - **model merging** averages the weights of a long-CoT and a short-CoT model;
  - **shortest rejection sampling** samples n = 8 and fine-tunes on the shortest correct answer;
  - **DPO** uses the shortest correct answer as the positive, and wrong answers or correct answers at least 1.5× longer as negatives;
  - **long2short RL** adds a length penalty and a smaller maximum rollout length.
  - Results: k1.5-short with RL reaches 60.8 AIME24 at about 3,272 tokens, and k1.5-shortest reaches 88.2 on MATH500. Long2short RL was "the highest token efficiency" against DPO and merging.
  - Source: [Kimi k1.5, arXiv 2501.12599 §2.4, §3.4, Fig. 7](https://arxiv.org/html/2501.12599v1).
- **K2 RL** [E], §3.2.2-3.2.3 of [arXiv 2507.20534](https://arxiv.org/html/2507.20534):
  - a **self-critique rubric reward**, with a critic that is itself refined on rollouts from verifiable tasks;
  - a **per-task maximum token budget**, with over-budget responses truncated and penalised, which "significantly improves token efficiency";
  - a **PTX loss** on hand-picked high-quality data "to reduce forgetting and overfitting to the RL task set";
  - temperature decay from exploration to exploitation.
- **K2 optimiser** [E]: Moonshot uses Muon in post-training and recommends it for fine-tuning, because "a Muon-pretrained checkpoint performs best with Muon fine-tuning" (§3.1). Source: [arXiv 2507.20534](https://arxiv.org/html/2507.20534).
- **K2-Thinking** [E]: "Quantization-Aware Training (QAT) is employed in post-training stage", with "INT4 weight-only quantization to the MoE components", for a "roughly 2x generation speed" gain. "All benchmark results are reported under INT4 precision." The model keeps "stable tool-use across 200–300 sequential calls." Source: [HF model card](https://huggingface.co/moonshotai/Kimi-K2-Thinking).
- **Open reproduction of INT4 QAT** (LMSYS/SGLang RL team, 2026-01-26) [E]:
  - In slime/Megatron, BF16 master weights are fake-quantized (QDQ) in the forward pass with 1×32 groups, a [-7, 7] range and a straight-through estimator. Rollouts run real W4A16 on SGLang's Marlin INT4 kernels.
  - It was tested on Qwen3-235B-A22B, Qwen3-30B-A3B and Kimi-K2-Thinking. The train-inference logprob gap "almost overlaps with the BF16 baseline" and is "significantly lower than FP8". The paper gives no exact numbers.
  - Source: [lmsys.org/blog/2026-01-26-int4-qat](https://www.lmsys.org/blog/2026-01-26-int4-qat).

### Inferences
- [I] The **Intel W4A16 checkpoint we serve keeps every non-expert tensor in BF16; only the routed experts are INT4** (`arc-agi-v4/arms.md:634`). The ms-swift recipe's LoRA targets (in_proj/out_proj/linear_proj/linear_qkv, `arms.md:635`) are attention and GDN projections, which are BF16. Such a LoRA merges into BF16 tensors with **no re-quantization**. This contradicts the prior plan's statement that W4A16 "must be redone" (`finetune-plan-2026-10-17.md:38`), unless experts are trained too.
- [I] The Kimi/DeepSeek/LMSYS lesson is to train against the weights you will serve. For us that costs nothing: build the training base from the Intel checkpoint's **dequantized INT4 experts** instead of Qwen's BF16 experts. The LoRA then learns on the exact deployed forward pass, which also absorbs part of the quantization error. Rollouts collected on Kaggle come from the same function the trainer optimizes, so the data is truly on-policy for ReST-EM. It is static-expert QAT with no STE needed, because the experts are frozen.

### Gaps
- K2-Thinking's QAT group size and ablation are not published on the card. K2.5 and K3 post-training were not checked.

## 4. Zhipu (GLM-4.5, GLM-5), MiniMax, Prime Intellect, NVIDIA, Mistral, ByteDance Seed

### Takeaway
Every lab runs some version of: SFT cold start → RL with verifiable rewards → (agentic RL) → general RL, with a consolidation step to stop forgetting. The consolidation is self-distillation in GLM-4.5, on-policy cross-stage distillation in GLM-5, a single mixed RL in DeepSeek V3.2 and Nemotron, and a PTX loss in Kimi K2. The RL runs are 10²-10³ GPUs for weeks. One small exception: RL on an already post-trained model, done by DeepSWE and Magistral, worked and kept the model's other skills.

### Cited Findings
- **GLM-4.5** [E], from [arXiv 2508.06471 §3](https://arxiv.org/html/2508.06471):
  - Pipeline: expert models (reasoning, agent, general chat), each built by cold-start SFT then RL; then a unified SFT on "millions" of samples distilled from the experts at 128K context; then general RL (§3.1-3.4).
  - Reasoning RL used a **single stage at 64K**, because shorter multi-stage RL makes the model "unlearn" long context "irreversibly".
  - **Dynamic temperature** was raised on plateau to the highest value within 1% of the optimum on a held-out set.
  - Agentic RL: K traces per prompt with a mean baseline; the loss covers model tokens only; wrongly formatted tool calls get zero reward. **Iterative distillation** replaces the cold-start data with RL outputs, re-SFTs, then continues RL (§3.3.2).
  - slime runs agentic RL **fully asynchronously** with FP8 rollouts and BF16 training (§3.5).
- **GLM-5** [E], from [arXiv 2602.15763 §1, §3.3, §3.5, §4.1.2, §4.2.1](https://arxiv.org/html/2602.15763):
  - Order: multi-task SFT, then Reasoning RL, then Agentic RL, then General RL.
  - **On-Policy Cross-Stage Distillation** comes last. "The final checkpoints from the preceding training stages serve as teacher models", and the advantage is the stop-gradient teacher/student log-ratio, with group size 1 and batch 1024.
  - Async RL drops stale samples, masks tokens outside [1-ε_l, 1+ε_h], and uses a **TITO gateway** so training sees exactly the token IDs the inference engine produced.
  - It has **10k+ verifiable SWE environments**.
- **MiniMax-M1** [E]: CISPO "clips importance sampling weights rather than token updates." Full RL took **512 H800 for three weeks, about $534,700**, including sandboxed SWE environments. Source: [arXiv 2506.13585 abstract](https://arxiv.org/abs/2506.13585), 2025-06-16.
- **INTELLECT-3 (Prime Intellect, 2025-12)** [E], from [arXiv 2512.16144 §2-4](https://arxiv.org/html/2512.16144v1):
  - From GLM-4.5-Air-**Base**: SFT of about 200B tokens (Table 1 sums) at 65K, then agentic SFT at 98K; then RL with masked token-level importance sampling (GSPO showed reward collapse in ablations; CISPO kept), 256 prompts × 16 rollouts, ≤8 off-policy steps, pass-rate-1 prompts dropped, 16 trainer : 44 inference H200 nodes, ~1,500 s/step.
  - About **two months on 512 H200** in total. AIME25 82.0 → 88.0, LCB v6 61.5 → 69.3 over GLM-4.5-Air (Table 2).
- **Nemotron 3 Nano (NVIDIA)** [E], from [arXiv 2512.20848 §3-4](https://arxiv.org/html/2512.20848):
  - SFT on **18M samples**, then RLVR over 9 environments trained **all at once** (single-environment training "often results in un-recoverable degradation of other benchmarks"), then RLHF, then a second RLVR.
  - Group-relative length bonuses cut "verbosity ... 30% ... without sacrificing accuracy". FP8 PTQ on 1K SFT calibration samples, 12 layers kept BF16: ~99% median recovery. NeMo-Gym/NeMo-RL are Apache-2.0.
- **Magistral (Mistral, 2025-06)** [E]: Medium was trained "on top of Mistral Medium 3 with RL alone". RL on text "maintains most of the initial checkpoint's capabilities" and maintains or improves multimodal, instruction-following and function-calling skills. Small adds "cold-start data from Magistral Medium". Source: [arXiv 2506.10910 abstract](https://arxiv.org/abs/2506.10910).
- **UI-TARS-2 (ByteDance Seed)**, the closest lab analogue to ARC-AGI-3 because it trains on browser games [E]. Source: [arXiv 2509.02544 §2.3, §2.5, §2.6, §3](https://arxiv.org/html/2509.02544).
  - **Data flywheel:** each round, the latest RL model's trajectories are scored by a validator. V = 1 goes to SFT and V = 0 goes to continual pre-training, and CT, SFT and RL are retrained in sequence.
  - **Game RL:** PPO with decoupled and length-adaptive GAE, value pretraining and Clip-Higher, with JavaScript verifiers reading score and level.
  - **Results:** on the 15-game suite, **SFT 44.27 → RL 59.8** mean normalised score (human = 100), with gains claimed on out-of-domain LMGame-Bench and "from-zero" gains on near-unplayable games.
  - **Weaknesses:** Sokoban 0.3 and Tetris 16.0, blamed on long-horizon planning.
  - **Specialist merging:** per-domain RL specialists merged by **parameter interpolation** perform "almost comparably" to the best specialist. No numbers are given.

### Inferences
- [I] The RL budgets (512 H800 × 3 weeks; about 480 H200 × weeks; ≥10% of pre-training) confirm the prior plan's "no RL before 11-02". Two RL-adjacent things fit one node: (a) DeepSWE-scale RL from a post-trained checkpoint (§6), and (b) on-policy distillation.
- [I] UI-TARS-2 is the only lab result on game RL. RL added 15.5 points over SFT on in-domain games, but SFT is what got it to 44. Long-horizon puzzle games stayed near zero. ARC-AGI-3 levels are closer to Sokoban than to 2048.
- [I] The flywheel's V = 0 → CT route suggests failed trajectories still carry true game transitions. Our version would be next-board prediction or world-model loss on stuck-level transitions. It is unproven for decision quality (`arms.md` Model prong: better transition replay need not give better decisions), so it is low priority.

### Gaps
- Not researched: MiniMax-M2 interleaved-thinking training, ByteDance Seed1.5-Thinking/VAPO, and Kimi K2.5/K3. For OLMo 3 I confirmed only the stage order SFT → DPO → RLVR ([interconnects.ai](https://interconnects.ai/p/olmo-3-americas-truly-open-reasoning)); its "delta learning" details are UNVERIFIED. No lab published GPU-hours for an agentic SFT stage on a model of our size.

## 5. Cross-cutting patterns: on-policy distillation, merging, QAT, regression control, checkpoint selection

### Takeaway
In 2026 labs consolidate and protect skills with **on-policy distillation** (reverse KL on the student's samples), not with plain SFT or weight averaging. They **train against the deployed low-bit weights**. Of the cheap tricks, **checkpoint interpolation** has the best measured effect on token length, and it costs only evaluation.

### Cited Findings
- **On-policy distillation (OPD)** [E], from [Thinking Machines, "On-Policy Distillation", 2025-10-27](https://thinkingmachines.ai/blog/on-policy-distillation/):
  - Per-token reverse KL to the teacher on student samples. Qwen3-8B-Base reached 70% AIME'24 from a 60% SFT checkpoint in about 150 steps, a **9-30× compute cut** against extrapolated SFT.
  - It recovers an RL-trained policy in 7-10× fewer gradient steps.
  - **Continual-learning experiment:** mid-training Qwen3-8B on internal documents dropped IF-eval **85% → 45%** at 100% domain data and **→ 79%** at a 70/30 mix. OPD with the *original model as teacher* restored **83%** while keeping the knowledge (41% vs 36%).
  - The post was updated in June 2026: the recipes now use Qwen3.5-9B.
- **Self-distillation fine-tuning (SDFT)** [E]: "a demonstration-conditioned model as its own teacher", trained on-policy with reverse KL. It reports "higher new-task accuracy" than SFT, "substantially reducing catastrophic forgetting", and sequential skill accumulation "without performance regression". The abstract gives no numbers. Source: [Shenfeld et al., arXiv 2601.19897](https://arxiv.org/abs/2601.19897), ICML 2026, v1 2026-01-27.
- **Framework support** [S]: ms-swift **Megatron GKD** supports LoRA with TP/PP/CP/EP. On-policy generation goes through vLLM (colocate or server), and `--lmbda 1 --beta 1` is the Thinking-Machines OPD setting. Without vLLM it silently falls back to off-policy. The teacher can be offloaded to CPU, or be a vLLM server with `--gkd_logits_topk` (v4.4). Teachers with different parallelism are "slated for a future release." Sources: [Megatron-SWIFT GKD v3.12](https://swift.readthedocs.io/en/v3.12/Megatron-SWIFT/GKD.html), [v4.4](https://swift.readthedocs.io/en/v4.4/Megatron-SWIFT/GKD.html).
- **Merging and interpolation:**
  - **AIMO-2 winner** [E]: a linear merge **CoT×0.3 + TIR×0.7** raised maj@16 66.8 → **69.1** (TIR alone) and cut average length **15,834 → 12,489 tokens** and code executions **2.73 → 0.85** (Table 10). Source: [arXiv 2504.16891 §6.1-6.2](https://arxiv.org/html/2504.16891).
  - **Kimi k1.5:** long/short weight averaging (above).
  - **UI-TARS-2:** specialist interpolation (above).
  - Evidence on merging RLVR checkpoints conflicts [S]: "Sparsity Curse" calls RLVR updates fragile to aggregate ([arXiv 2606.18521](https://papers.cool/arxiv/2606.18521)), while another paper finds RL reduces task conflicts in merging ([arXiv 2607.22039](https://www.alphaxiv.org/abs/2607.22039)).
  - DeepSeek-V4 explicitly prefers OPD over weight merging (§2).
- **QAT for low-bit deployment** [E]: Kimi K2-Thinking (INT4 experts), DeepSeek-V4 (FP4 experts, rollouts native FP4) and the LMSYS slime INT4 QAT reproduction (§2, §3).
  - The AIMO-2 Kaggle table shows how much PTQ to 4 bits can cost on a dense 14B: **W4A16 AIME24 72.7 vs BF16 82.7**, while FP8 kept 83.3 (Table 9, 4×L4). Source: [arXiv 2504.16891 §6.3.1](https://arxiv.org/html/2504.16891).
  - The same team trained a ReDrafter draft head on the model's own solutions (about 65% of steps accept 3 tokens; §6.3.2).
- **Regression control** (sources above): one mixed RL stage (DeepSeek V3.2, Nemotron 3); PTX loss (Kimi K2); on-policy cross-stage distillation (GLM-5); a single long-context RL stage (GLM-4.5); RL-only on a post-trained model (Magistral).
- **Evaluation and checkpoint selection:**
  - INTELLECT-3 ran online evaluation every 15 steps ([arXiv 2512.16144](https://arxiv.org/html/2512.16144v1)).
  - GLM-4.5 tuned temperature against a held-out set within 1% ([arXiv 2508.06471 §3.2](https://arxiv.org/html/2508.06471)).
  - Nemotron re-profiles task difficulty with the best checkpoint ([arXiv 2512.20848 §3.2.2](https://arxiv.org/html/2512.20848)).
  - Qwen names evaluation throughput as its tightest bottleneck ([tech_report.pdf §5](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)).
  - None of the labs publish a checkpoint-selection rule beyond "evaluate often on held-out data".

### Inferences
- [I] **OPD as our forgetting guard.** This is the strongest new tool for us; the prior plan had replay only. Run the game SFT, then a short OPD on general and tool-use prompts with the **original Flash-Next as teacher**. The Thinking Machines result shows a replay mix alone kept 79/85 IF-eval and OPD recovered 83. GLM-5 does the same between stages.
- [I] **SDFT-style self-distillation as a better R4/R5.** The teacher is Flash-Next given privileged information: the hindsight-correct action, the eventual level rule, or the full memory card. The student is the same model with the normal harness prompt, trained on its own samples at frozen decision states. This is on-policy, so it keeps the thinking style better than SFT on rewritten rationales. The state-lab resampler already produces student samples.
- [I] OPD is test-first. It needs the vLLM-from-source path that the GRPO recipe also needs (`finetune-plan-2026-10-17.md` §1, "8*135G with a vLLM built from source"), and a teacher with the same parallelism layout. Plain SFT with replay remains the floor.
- [I] **Merging costs us one evaluation per α.** Scale the merged LoRA by α ∈ {0.5, 0.75, 1.0}, i.e. interpolate between base and fine-tuned weights, and pick on held-out games. The AIMO-2 table shows interpolation can shorten outputs and raise accuracy together.

### Gaps
- No public OPD or SDFT result on a model of our class (125B MoE, hybrid GDN) or on multi-turn agents.
- Whether ms-swift GKD supports Qwen3.8-Flash-Next (vLLM model support, PLE offload, hyper-connections) is UNVERIFIED.
- No evidence on interpolating a LoRA on an RL-tuned MoE.

## 6. Small-team case studies: what was done, how fast, and what it gained

### Takeaway
Narrow-benchmark adaptations that worked fall into two groups. One used about 0.5-5k verified trajectories for SFT or a few hundred RL steps on an already post-trained model (SWE-Gym, SWE-smith, DeepSWE). The other was the AIMO pattern: heavy SFT, then a cheap merge, then a deployment-specific quantization and speculative-decoding pass. None used a 125B MoE, and none had our noise-limited evaluation.

### Cited Findings
- **AIMO-2 (NVIDIA NemoSkills, 1st place)** [E], from [arXiv 2504.16891 §6](https://arxiv.org/html/2504.16891):
  - Qwen2.5-14B-Base was trained on 2.2M R1 CoT solutions, then a light 15k-sample TIR fine-tune, then a linear merge.
  - Deployment used TensorRT-LLM FP8 plus ReDrafter, with a shared 350 s per question and a 210 s buffer.
  - Result: **34/50 private**. GPU-hours are not reported.
- **DeepSWE (Agentica / Together, 2025)** [E], from the [HF model card](https://huggingface.co/agentica-org/DeepSWE-Preview):
  - RL only from Qwen3-32B on **4.5K R2E-Gym problems**, filtered against SWE-bench-Verified repositories, for **200 RL steps**; the score rose 23% → 42%. The card labels this curve both "SWE-Bench-Hard" and "Verified", so it is ambiguous.
  - Final Pass@1 is **42.2%** (16 runs), or 59% with hybrid test-time scaling (Best@16).
  - The modified GRPO uses Clip-High, no KL, no reward std, length normalisation, leave-one-out advantages, **compact filtering** (loss masked on trajectories that hit max context, time out or reach max steps) and no entropy loss.
  - Compute is not stated on the card; a figure of about 64 H100 × 6 days is UNVERIFIED and from memory.
- **SWE-smith** [E]: 50k task instances from 128 repositories; SWE-agent-LM-32B reached **40.2%** SWE-bench Verified. Source: [arXiv 2504.21798](https://arxiv.org/abs/2504.21798). The number of trajectories is not in the abstract.
- **SWE-Gym** (prior notes): 491 trajectories gave +12-17 points on Qwen2.5-32B ([arXiv 2412.21139](https://arxiv.org/abs/2412.21139)).

### Inferences
- [I] The feasible small-team template on one 8×H200 node in about 12 days:
  1. Verified-trajectory SFT, done as ReST-EM / expert iteration.
  2. Optionally a short DeepSWE-style RL or OPD phase, only if the infrastructure is ready by about day 6.
  3. A merge / α sweep.
  4. Deployment engineering: quantization unchanged if non-expert LoRA, and an MTP check.

  The gains in these cases came where the base was weak on the task format. Flash-Next is already a strong agent here, so expect smaller gains.
- [I] DeepSWE's compact filtering transfers directly to our SFT/RFT data selection. Exclude or mask decision windows cut by the 2,048-token yield or the context trim, so the model is not trained to run out of budget.

### Gaps
- I did not verify the compute numbers for DeepSWE or SWE-smith. I found no write-up of a team fine-tuning a ≥100B MoE for a Kaggle competition. I also found no fine-tuning write-up for the Konwinski Prize round-1 winner (7.5%).

## 7. Implications for our plan (do / don't / test-first)

1. **DO: train against the deployed weights.**
   - Build the LoRA training base from the Intel W4A16 checkpoint's **dequantized INT4 routed experts**, with all other tensors BF16 as shipped.
   - LoRA only non-expert modules, as the ms-swift targets already are.
   - Result: no AutoRound re-quantization, and the trained function equals the served one.
   - This removes a deployment step from the prior plan (`finetune-plan-2026-10-17.md:38`, stage 4).
   - Lab precedent: Kimi K2-Thinking, DeepSeek-V4, LMSYS INT4 QAT.
   - Test first that ms-swift loads such a BF16 checkpoint: a forward-pass logprob diff against SGLang on 10 transcripts.
2. **DO: R3 is shortest rejection sampling (Kimi k1.5).**
   - Sample k ≈ 8 at frozen decision states and keep the shortest reply whose action matches the verified good action.
   - Add length-contrast DPO pairs (shortest correct versus correct but ≥1.5× longer) only after plain SFT is measured.
3. **DO: interpolation sweep as free checkpoint selection.**
   - Save checkpoints before and after the decay, and evaluate LoRA scale α ∈ {0.5, 0.75, 1.0} on the frozen-state decision benchmark (right-move rate + tokens/decision) before any 20-min panel.
   - AIMO-2 got higher accuracy and shorter outputs from interpolation.
4. **DO: mask truncated windows.** Use DeepSWE-style compact filtering: drop or mask decisions cut by the yield or the trim, so training never rewards running out of budget.
5. **TEST-FIRST (days 1-5): OPD forgetting guard.** ms-swift Megatron GKD, `--lmbda 1 --beta 1`, with teacher = original Flash-Next on general and tool prompts after the game SFT. Do it only if the vLLM-from-source path boots this model by about day 5; otherwise use 30-50% self-generated replay plus a Kimi-style PTX share of curated samples.
6. **TEST-FIRST, after stage 1: SDFT-style hindsight self-distillation for stuck levels (R4).** The teacher is the same model prompted with the hindsight-correct action or rule; the student uses the normal prompt. The core risk is that the teacher's distribution mostly encodes a different prompt.
7. **DON'T:**
   - specialist-then-merge (we have one domain);
   - assume short-window training is harmless. GLM-4.5 found that RL stages at shorter maximum lengths made the model "unlearn" long context "irreversibly" (§3.2). Training on 8-16k windows for 55k play is the SFT analogue [I]: keep windows as long as memory allows, and check behaviour at 55k;
   - any GRPO/GSPO run before 11-02 (labs use 10²-10³ GPUs for weeks, and INTELLECT-3 saw GSPO reward collapse);
   - Muon fine-tuning experiments. Kimi recommends Muon for Muon-pretrained models, but the evidence is mixed and LoRA narrows the gap ([arXiv 2605.10468](https://arxiv.org/pdf/2605.10468)).
8. **Evaluation is the binding constraint, as Qwen says of itself.**
   - Gate in this order: frozen-state benchmark (cheap, on the Kaggle GPU), then held-out-game 20-min panels, then full runs, then LB draws.
   - Treat general benchmarks as a canary, not a ±2-point gate. Qwen3 itself traded 2-3 hard-reasoning points for breadth, and we trade the other way.

## Fact-check (independent verifier, 2026-10-18)

How this was checked: every source below was fetched again in this session (HF API and raw model cards, the GitHub tech-report PDF, arXiv HTML and API, the Thinking Machines post, ms-swift docs, and the ms-swift source at tags v3.12.6 and v4.4.3). The Intel checkpoint's `config.json` was downloaded from `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`. No browser was used. Verdicts: **confirmed**, **partially confirmed** (the gist holds but a detail is off), **refuted**, **unverifiable**.

### The eight claims the plan rests on

1. **There is no public Flash-Next base checkpoint, and the tech report has no post-training section: confirmed.**
   - The HF API (`author=Qwen&search=3.8`) returns exactly six models: Qwen3.8-27B, Qwen3.8-Flash-Next and Qwen3.8-2.4T-A95B, each with an FP8 copy.
   - No HF search finds a "Flash-Next-Base" from any author.
   - The PDF (dated 2026-08-26) evaluates only "Qwen3.8-Flash-Next-Base" in §4 (Table 11: MMLU-Pro 73.23, MATH 72.78, SWEBench-Pretrain 50.99).
   - The §1 "Organization" paragraph promises an "evaluation of the resulting base and post-trained models", but no post-trained evaluation and no post-training or RL section appear.
   - Both quotes are accurate: the §2.2 sparse-writes remark and the §5 "evaluation throughput" sentence.
   - The word "internal" is our inference; the report does not use it.

2. **Qwen3-8B: on-policy distillation 74.4 AIME'24 in 1,800 GPU-h, against RL 67.6 in 17,920 GPU-h, both from 55.0: confirmed** (§4.7, Table 21).
   - Caveat: the comparison used math and code queries only, and the teacher was a larger Qwen3 (32B or 235B), not the student itself.
   - Pass@64 stayed flat under RL (90.0 to 90.0) and rose under OPD (to 93.3).

3. **Thinking Machines continual-learning table: confirmed.**
   - The numbers match: IF-eval 85, then 45 (100% documents) and 79 (70/30 mix), then 83 after OPD. Internal QA was 36 after the mix and 41 after OPD.
   - The teacher was the original Qwen3-8B, run on Tulu3 prompts. The post is dated 2025-10-27, and a June-2026 update moved the recipes to Qwen3.5-9B.
   - Caveat on the "why it matters" line: OPD recovered *nearly* full IF-eval (83 of 85), not all of it. Its knowledge score (41) is also below the 100%-document run (43).
   - The "replay" baseline was chat data resampled from Qwen3-8B itself, so it was on-policy data. The post adds that no mix, even 100% chat, kept 85.

4. **Kimi K2-Thinking INT4 QAT, and DeepSeek-V4 FP4 QAT on experts: confirmed.**
   - The K2-Thinking card says: "we adopt Quantization-Aware Training (QAT) during the post-training phase, applying INT4 weight-only quantization to the MoE components ... All benchmark results are reported under INT4 precision."
   - The card says "MoE components", not "routed experts" specifically.
   - DeepSeek-V4 §1 and §5.2.1 apply MXFP4 QAT to (1) the MoE expert weights and (2) the indexer QK path, using an STE.
   - V4's QAT also covers the teacher and reference models, and its rollouts use native FP4.
   - arXiv lists V4's v1 as 2026-04-26, as the notes say.

5. **The Intel W4A16 checkpoint has INT4 routed experts only, and the ms-swift LoRA targets are non-expert: confirmed** against the HF `config.json`.
   - `quantization_config`: auto-round 0.15.0, 4-bit symmetric int, group_size 128, `block_name_to_quantize = model.language_model.layers`.
   - `extra_config` (2,340 entries) keeps these at 16 bit: self_attn (q/k/v/o and the indexer), linear_attn (in_proj_qkv/z/a/b and out_proj), shared_expert and shared_expert_gate, mlp.gate, every hyper_connection, mtp (including its 512 experts), ple, embed_tokens and visual.
   - The ms-swift best-practice page uses `--target_modules in_proj out_proj linear_proj linear_qkv` twice, at lines 67 and 131 of the md source.
   - `arms.md:634-635` and `finetune-plan-2026-10-17.md:38` say what the notes quote.
   - Caveats:
     - I checked Intel's HF repo, not a byte-level copy of the Kaggle mirror `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround`.
     - The expert INT4 uses group size 128, not the 1×32 groups in the LMSYS recipe. This matters for §7.1's "dequantized INT4 experts" base, though not for the claim.

6. **AIMO-2 merge: confirmed** (§6.2, Table 10).
   - maj@16: CoT 62.9, TIR 66.8, merge (CoT×0.3 + TIR×0.7) 69.1.
   - Length: CoT 11,203, TIR 15,834, merge 12,489. Code executions: TIR 2.73, merge 0.85.
   - Caveats:
     - The benchmark is Comp-Math-24-25, not AIME.
     - The merge is shorter than TIR but still *longer* than the CoT start (11,203). "Shorter and more accurate" holds only against the fine-tuned endpoint.
     - TIR was fine-tuned from the CoT checkpoint, so the merge equals CoT + 0.7·(TIR − CoT). That is an α = 0.7 interpolation, which does support §7.3's α sweep.

7. **Kimi k1.5 shortest rejection sampling and its DPO variant: confirmed** (§2.4).
   - "samples the same question n times (in our experiments, n=8) and selects the shortest correct response for supervised fine-tuning."
   - DPO negatives are "both wrong longer responses and correct longer responses (1.5 times longer than the chosen positive sample)".
   - Nuances:
     - "At least 1.5×" is our reading of "1.5 times longer".
     - The negatives also include wrong *longer* answers, not all wrong answers, so §3 of these notes slightly misstates this.
     - The paper used it to move long-CoT priors into a short-CoT model ("merge + rs").

8. **ms-swift Megatron GKD: partially confirmed.**
   - What holds:
     - The v3.12 Megatron GKD page lists LoRA, TP/PP/CP/EP, CPU offload of the teacher and on-policy generation through vLLM only.
     - It says that when `lmbda > 0` without vLLM it "will automatically fall back to off-policy mode".
     - The v4.4 docs define `--lmbda 1` as pure on-policy and `--beta 1` as reverse KL.
     - The v4.4.3 code `gkd_loss.py` computes KL(student‖teacher) when beta = 1.
   - Corrections:
     - (a) The fallback is **not silent**. Both v3.12.6 and v4.4.3 `swift/megatron/trainers/gkd_trainer.py` call `logger.warning_once('On-policy mode triggered but use_vllm=False. Falling back to dataset responses...')`.
     - (b) The cited `v4.4/_sources/Instruction/GKD.md.txt` returns **404**. In v4.4 that content lives in `Instruction/Distillation.md`. The v4.4 Megatron GKD page no longer lists the vLLM-only or teacher-parallelism limitations.
     - (c) The docs never say "Thinking Machines". `--lmbda 1 --beta 1` is on-policy reverse KL computed over the full vocabulary (or top-k), which is closer to DeepSeek-V4's full-vocabulary OPD. The exact Thinking Machines estimator (the sampled-token log-ratio used as a per-token advantage) is ms-swift's separate **OPD-RL** path, `--rlhf_type grpo` with a teacher and `--teacher_kl_coef`, which also needs vLLM rollouts.
     - (d) This is new and favourable to us. In v4.4.3 Megatron, LoRA with `--teacher_model` equal to `--model` uses the base through `disable_adapter()` as a fixed teacher, with **no second model load** (`megatron_args.py`, `rollout_mixin.py`). v4.4 also supports **OPSD**: self-distillation where the teacher gets a privileged `teacher_prompt` (Megatron GRPO path). It also supports **multi-turn GKD** (`--multi_turn_scheduler`).

### Other errors found in the notes

- **§6 SWE-Gym:** "+12-17 points on Qwen2.5-32B" is off. With 491 trajectories, Qwen2.5-**Coder**-32B gained **+12.3** on Lite (3.0 to 15.3) and **+13.6** on Verified (7.0 to 20.6) (arXiv 2412.21139 §4.2, Table 3). The abstract's "up to 19%" is a different, overall figure.
- **§4 INTELLECT-3:** "CISPO kept" is wrong. Fig. 10 compares GSPO with "CISPO (our algorithm at the time)" and reports the GSPO collapse. The final RL used **masked token-level importance sampling** (IcePop-style, Zhao et al. 2025), not CISPO. The other numbers check out: 256×16 rollouts, max_off_policy_steps 8, 16:44 nodes, ~1,500 s per step, two months on 512 H200, AIME25 82.0 to 88.0, LCB v6 61.5 to 69.3, and evaluation every 15 steps.
- **§7.7 Muon:** arXiv 2605.10468 ("Can Muon Fine-tune Adam-Pretrained Models?") is about *Adam*-pretrained models: fine-tuning them with Muon causes an optimizer mismatch. Flash-Next was pretrained with Muon plus AdamW on some weight categories (HF card). By that paper's mismatch logic, matched (Muon) fine-tuning is favoured, consistent with Kimi. So it is not evidence against Muon. Skipping Muon may still be right for time reasons, but the cited reason does not support it.
- **§5 Inferences, OPD bullet:** the precondition "a teacher with the same parallelism layout" is overstated. See claim 8(d): with LoRA and teacher = base, ms-swift v4.4.3 needs no separate teacher. A separately loaded Megatron teacher is built with the student's own parallel args. The real blocker is still vLLM support for Flash-Next. The SDFT bullet (and §7.6) also has a built-in ms-swift v4.4 counterpart, OPSD via `teacher_prompt`, which the notes do not mention.
- **§4 UI-TARS-2:** "Sokoban 0.3 and Tetris 16.0" come from the out-of-domain **LMGame-Bench** (Table 3), not the 15-game in-house suite. In that suite SFT scored 44.27 and RL 59.77 (rounded to 59.8). The weakest in-suite games were Free-the-key (0.70 vs human 5.54) and Laser-maze (5.6 vs 17.83).
- **§1 Qwen3, Table 22:** these numbers are for **Qwen3-32B**, which the notes do not say. They are correct as given: AIME'24 83.8 to 81.4, LCB v5 68.4 to 65.7.
- **Paraphrases written as direct quotes** (the gist is fine, the wording is not the source's):
  - Kimi K2: "significantly improves token efficiency". The paper says "significantly enhances the model's token efficiency".
  - Kimi K2: "to reduce forgetting and overfitting to the RL task set". The paper says "To prevent the potential forgetting of valuable, high-quality data during joint RL training ... mitigates the risk of overfitting to the limited set of tasks".
  - Kimi K2: "a Muon-pretrained checkpoint performs best with Muon fine-tuning". The paper says "produces the best performance with Muon fine-tuning".
  - Qwen3: "accepted this trade-off for overall versatility". The paper says "we choose to accept this performance trade-off to enhance the model's overall versatility".
- **Spot-checked and correct:**
  - DeepSeek-V3.2: 1,827 environments and 4,417 tasks; 24,667, 50,275 and 5,908 tasks; >10% of pre-training cost; the single mixed RL stage; Keep Routing; and the generalisation result for synthetic agent RL.
  - DeepSeek-V4: OPD replaced mixed RL; more than ten teachers; full-vocabulary OPD.
  - GLM-4.5: the 64K "unlearn"/irreversible result and the 1% temperature rule.
  - GLM-5: cross-stage OPD (group size 1, batch 1024), TITO, and 10k+ SWE environments.
  - MiniMax-M1: 512 H800 × 3 weeks, $534,700.
  - Nemotron 3 Nano: 18M SFT samples, the "un-recoverable" degradation, the 30% verbosity cut, FP8 PTQ calibrated on 1K samples with 6 attention + 6 Mamba layers in BF16, ~99% median recovery.
  - DeepSWE: 4.5K problems, 200 steps, 23 to 42%, 42.2% / 59%, compact filtering.
  - SWE-smith: 50k instances, 128 repositories, 40.2%.
  - AIMO-2: Table 9 (W4A16 72.7 vs BF16 82.7, FP8 83.3, maj@12 on 4×L4) and 34/50.
  - LMSYS INT4 QAT (2026-01-26): 1×32 groups, [-7, 7], STE, Marlin W4A16, the logprob gap "almost overlaps" BF16.
  - SDFT: arXiv v1 2026-01-27; ICML 2026 confirmed through OpenReview and PMLR.
  - Magistral and GSPO: abstract quotes correct.
