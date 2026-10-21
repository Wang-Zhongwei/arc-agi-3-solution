# W1: Fine-tuning pedagogy: how to SFT an already-RL-tuned reasoning MoE without breaking it (2026-10-18)

Scope: SFT, rejection-sampling FT and light preference tuning of a strong RL-tuned thinking model, applied to Qwen3.8-Flash-Next playing ARC-AGI-3 in our harness. This builds on `research/reports/finetune-plan-2026-10-17.md` and `research/data/finetune-research-2026-10-17/training.md`. It challenges two of their choices: "30-50% self-generated replay" as the forgetting guard, and the recipe's attention-only LoRA.

Legend: [E] is evidence from the cited source. [I] is my inference. UNVERIFIED means not confirmed in a primary source. Paper dates are arXiv v1 dates.

## Q1. Data quality vs quantity; on-policy vs off-policy SFT data

### Takeaway
For an already strong model, curation and on-policyness matter far more than volume. Hundreds to a few thousand verified samples move behaviour. Uncurated or off-policy text can actively damage the model. The safest data for an RL-tuned reasoner is its own outputs, filtered by a verifier and **regenerated each round**. Data sampled once from the initial model still causes measurable forgetting.

### Cited Findings
- **LIMO (2025-02).** 817 curated samples. Full FT of Qwen2.5-32B-Instruct at LR 5e-6, cosine schedule, no warmup, 15 epochs, batch 64. Results:
  - AIME24 63.3%, MATH500 95.6%.
  - 400 samples alone took AIME24 from 16.5 to 57.5.
  - Uncurated NuminaMath-100k *degraded* the benchmark average from 49.9 to 32.3; LIMO's set reached 78.1.

  [E] — [arXiv 2502.03387](https://arxiv.org/html/2502.03387)
- **s1 (2025-01).** 1,000 samples chosen jointly for quality, difficulty and diversity. Random, longest-trace or diversity-only selection were "significantly worse". Training took 26 min on 16 H100s. [E] — [arXiv 2501.19393 §2, §5](https://arxiv.org/html/2501.19393). Its LR and epochs were not extracted; 1e-5 and 5 epochs are from memory (UNVERIFIED).
- **Qwen's own practice.** Qwen3 Stage 3 ("Thinking Mode Fusion") ran continual SFT on the Reasoning-RL model: "To ensure that the performance of the Stage 2 model is not compromised by the additional SFT, the 'thinking' data is generated via rejection sampling on Stage 1 queries using the Stage 2 model itself." [E] — [Qwen3 Technical Report arXiv 2505.09388 §4.3](https://arxiv.org/html/2505.09388). This is the closest lab precedent for our case.
- **Retaining by Doing (Chen, Razin, Narasimhan, Chen; 2025-10, ICML 2026).** Compares teacher SFT (Llama-3.3-70B outputs), "Self-SFT" (the initial model's own outputs, filtered to correct, 5 samples per prompt) and GRPO, on Llama 3 / Qwen 2.5 models of 1–8B.
  - Self-SFT still forgets. Table 2, Qwen-2.5-1.5B, non-target drop when trained on IFEval / MMLU / Countdown:
    - Self-SFT: 3.0 / 14.0 / 19.5
    - teacher SFT: 6.2 / 11.9 / 29.5
    - GRPO: 0.6 / 0.5 / 0.9
  - **Iterative-SFT**, which regenerates data from the current model each epoch, reaches target accuracy "higher than or comparable to" SFT "while only exhibiting mild to no forgetting" (§4.2, Fig. 7).
  - Ablations credit on-policy data, not the KL penalty or the advantage estimator (§4.1, Table 1).
  - Setup: full FT; LR 1e-4 (1–1.5B) or 5e-6 (7–8B); 2 epochs (App. A.3).

  [E] — [arXiv 2510.18874v2](https://arxiv.org/pdf/2510.18874v2)
- **RL's Razor (Shenfeld, Pari, Agrawal; 2025-09).** Forgetting tracks the KL between fine-tuned and base policy, measured on the new task. On-policy RL is biased toward KL-minimal solutions, while "SFT can converge to distributions arbitrarily far from the base model". [E] — [arXiv 2509.04259](https://arxiv.org/abs/2509.04259). Retaining by Doing finds this correlation only moderate (Pearson 0.52, App. A.5).
- **Thinking Machines, "On-Policy Distillation" (K. Lu, 2025-10-27).**
  - SFT of Qwen3-32B on its *own* samples (T = 1, zero KL in expectation) "degrades performance". Any practical LR above zero hurt IF-eval, attributed to finite-batch noise driving the model off-policy.
  - Document mid-training cut Qwen3-8B IF-eval from 85% to 79% (70% docs) or 45% (100% docs). On-policy distillation from the original model on unrelated Tulu3 prompts restored it to 83%, and internal QA rose from 36% to 41%.
  - LoRA rank 32 trailed full FT by 13% after SFT and by 6% after on-policy distillation.

  [E] — [thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- **SDFT 2024 (Yang et al., ACL 2024).** The model rewrites targets "in its own words" before fine-tuning, which mitigates forgetting at comparable task performance. [E] — [arXiv 2402.13669](https://arxiv.org/abs/2402.13669)
- **SDFT 2026 (Shenfeld, Damani, Hübotter, Agrawal; ICML 2026).** The model conditioned on a demonstration in context is the teacher. The student is trained on-policy with reverse KL. On Qwen2.5-7B-Instruct it is "the only approach to improve performance on the new task without significant degradation in prior capabilities", while SFT forgets on all benchmarks (§4.3). [E] — [arXiv 2601.19897](https://arxiv.org/html/2601.19897). A secondary review reports cost at about 2.5× SFT's FLOPs (UNVERIFIED).
- **External concise data vs self-generated.** Fine-tuning on GPT-4o's concise reasoning "causes a severe drop in relative accuracy". Self-generated shortest-correct data preserves accuracy (§5, Table 2). [E] — [Munkhbat et al., arXiv 2502.20122](https://arxiv.org/html/2502.20122)

### Inferences
- [I] **Data, safest to riskiest:**
  1. on-policy, filtered, regenerated each round (Iterative-SFT / ReST-EM);
  2. own samples filtered once;
  3. own-words rewrites of external solutions;
  4. teacher or frontier text, or scripted-solver rationales.

  The 10-17 plan's R2/R3 are type 1–2. Its "scripted solvers + hindsight rationale" is type 4: keep it small, or route it through on-policy distillation.
- [I] **Why on-policy data forgets less.** Relative to the model, the information it carries is mostly the *selection* (about log2 k bits per kept sample), not the text. This mirrors LoRA Without Regret's ≈1 bit per episode for RL. So shortest-correct selection (R3) is low-risk and low-capacity, but its gains per round are modest (Q6).
- [I] **Self-replay is not a free regulariser.** Thinking Machines saw unfiltered self-sample SFT drift, and Retaining by Doing saw filtered one-shot Self-SFT forget. A replay share only dilutes the task gradient.

### Gaps
- No published SFT or RFT on a Qwen3-Next-class or Flash-Next model, nor on any agent with ~55k-token contexts. All forgetting studies use ≤32B models on single-turn tasks.

## Q2. Catastrophic forgetting and mitigations

### Takeaway
Ranked levers:
1. on-policyness (Q1);
2. LoRA, which learns less and forgets less if α and LR are sane;
3. weight interpolation after training (WiSE-FT, ExPO, merging; free with LoRA);
4. a general-data mix.

KL regularisation matters less than folklore says. For an exploring agent, SFT also brings diversity/entropy collapse.

### Cited Findings
- **LoRA Learns Less and Forgets Less (Biderman et al., 2024-05, TMLR).** In standard low-rank settings, LoRA "substantially underperforms" full FT on code and math, but better preserves out-of-domain ability and output diversity. It forgets less than weight decay or dropout do. Full FT learns perturbations of 10–100× higher rank. [E] — [arXiv 2405.09673](https://arxiv.org/abs/2405.09673)
- **Illusion of Equivalence (Shuttleworth et al., 2024-10).**
  - LoRA's "intruder dimensions" cause forgetting.
  - "different selections of LoRA alpha and learning rate lead to starkly different generalization behaviors, even leading to LoRA forgetting more than full fine-tuning."
  - α = 2r gives fewer intruders than α = 8 at high rank.
  - Lower ranks forget more in sequential training.

  [E] — [arXiv 2410.21228 §4, App.](https://arxiv.org/html/2410.21228)
- **General-data mix (AgentTuning).** The agent-data ratio η was scanned from 0 to 1 and **η = 0.2** chosen: 20% agent trajectories, 80% general chat. "training solely on agent tasks performs worse on unseen tasks compared to mixed training." [E] — [AgentTuning arXiv 2310.12823 §2.2](https://arxiv.org/pdf/2310.12823)
- **KL penalty is not the protection.** GRPO without KL had a similar gain/forgetting trade-off to GRPO with it (§4.1, Fig. 6). [E] — [Retaining by Doing](https://arxiv.org/pdf/2510.18874v2)
- **The LR trade-off.** "a high learning rate is typically required to reach high target performance for SFT, often at the cost of severe forgetting; a smaller learning rate reduces forgetting but fails to reach the same target performance even with more epochs." [E] — [Retaining by Doing §2.3, Fig. 3](https://arxiv.org/pdf/2510.18874v2)
- **Weight ensembling.**
  - Interpolating the final SFT checkpoint with an early one (WiSE-FT) "nearly fully" restores the Pass@k lost to diversity collapse in reasoning SFT, improves Pass@1, and makes a better RL start. [E] — [Dang et al., arXiv 2504.10478](https://arxiv.org/abs/2504.10478)
  - ExPO extrapolates along the SFT→aligned direction: a DPO model trained 20% of the steps, plus ExPO, beat full DPO (1.8B–70B). [E] — [arXiv 2404.16792](https://arxiv.org/abs/2404.16792)
  - Kimi k1.5 averaged long- and short-CoT model weights for token efficiency. [E] — [arXiv 2501.12599 §2.4](https://arxiv.org/html/2501.12599)
- **Entropy collapse.**
  - Proximal SFT (ICLR 2026) adds PPO-style clipping to SFT: it matches SFT in-domain, does better out of domain, and is stable under long training. [E, abstract] — [arXiv 2508.17784](https://arxiv.org/html/2508.17784v1)
  - CurioSFT reports that token entropy "rapidly collapses" during SFT. [E, abstract] — [arXiv 2602.02244](https://arxiv.org/html/2602.02244v1)
- **Asymmetric cost.** In long-CoT SFT, "reasoning improves while safety degrades". [E] — [arXiv 2604.06628](https://arxiv.org/html/2604.06628)

### Inferences
- [I] What we must not forget is not MMLU. It is:
  - (a) valid tool calls over long multi-turn contexts;
  - (b) diverse proposals on stuck levels (96% of stuck decisions had an untried new-board action);
  - (c) near-optimal cleared-level play.

  Gate on: tool-call validity; distinct actions across k samples at frozen stuck states; hindsight-route agreement; plus a small math/logic probe.
- [I] With LoRA, WiSE-FT and ExPO cost nothing. Merge W + λ·(α/r)BA with λ ∈ {0.5, 0.75, 1.0, 1.25}, so one training run gives 3–4 panel candidates. Test-first: this has not been tested on agent behaviour.

### Gaps
- No study of forgetting or diversity collapse in multi-turn agent SFT on a ≥100B MoE, and none on router drift under SFT.

## Q3. LoRA best practice 2025–2026, and our merge constraint

### Takeaway
LoRA matches full FT for small-to-medium SFT and for RL *if it covers the MLP/MoE layers*. Attention-only LoRA underperforms even at equal parameter count. LR should be about 10× the full-FT LR, with a small batch. The ms-swift Flash-Next recipe is attention-only at rank 8. On our checkpoint, the only BF16 MLP is the small shared expert.

### Cited Findings
- **LoRA Without Regret (J. Schulman et al., Thinking Machines, 2025-09-29).**
  - "Attention-only LoRA underperforms MLP-only LoRA", and adding attention to MLP-only gains nothing. This holds for Llama-3.1-8B, for MoE Qwen3-30B-A3B-Base, and in MATH RL.
  - At similar parameter counts, attention-only rank 256 (0.25B) still lost to MLP-only rank 128 (0.24B).
  - MoE: one LoRA per expert, with rank = total rank / active experts.
  - Optimal LR ≈ 10× full FT (fit 9.8; ~15× for runs under ~100 steps), nearly rank-independent (<2× change from rank 4 to 512, with α = 32 and 1/r scaling). LR ∝ (2000/hidden)^pow, with a family-specific exponent (Qwen3 ≠ Llama).
  - LoRA is "less tolerant of large batch sizes", and the gap does not depend on rank.
  - Rank 1 matches full FT in RL. LoRA lags only when data exceeds capacity (~2 bits per parameter).
  - Constant LR, no warmup. Standard PEFT init was not beaten.

  [E] — [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/)
- **ms-swift Flash-Next recipe.**
  - SFT: `--lora_rank 8 --lora_alpha 32 --target_modules in_proj out_proj linear_proj linear_qkv --num_train_epochs 1 --global_batch_size 8 --padding_free true --max_length 8192 --lr 1e-4`.
  - GRPO: `--lr 5e-5 --beta 0.0 --moe_aux_loss_coeff 1e-3`.
  - No expert modules are targeted, and no reason is given.

  [E] — [ms-swift best practice](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html)
- **Flash-Next config (`text_config`).** `hidden_size` 2560, 48 layers, `num_experts` 512, `num_experts_per_tok` 10, `moe_intermediate_size` 640, `shared_expert_intermediate_size` 640, `full_attention_interval` 4, arch `Qwen4ExpForConditionalGeneration`. [E] — [HF config.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json) (fetched 2026-10-18). [I] So the shared expert is 1 of 11 active MLP blocks: about 4.9M parameters per layer (3×2560×640), against about 2.5B for the routed experts in each layer.
- **What stays BF16.** In our W4A16 checkpoint only the routed experts are int4. Attention, GDN, shared expert, router, hyper-connections, MTP and PLE stay BF16, so a LoRA on those merges without requantisation. [E] — `arc-agi-v4/arms.md:631-636`

### Inferences
- [I] **Do not ship the recipe's attention-only list as the only arm.** Test:
  - (A) attention + GDN in/out + **shared-expert fc1/fc2**, at rank 32–64 (all BF16, mergeable);
  - (B) A + routed-expert LoRA, which forces a BF16 merge, AutoRound requant and an MTP re-check.

  Adopt B only if it clearly beats A.
- [I] **Train against what we serve.** With experts frozen, use the *dequantised AutoRound int4 experts* as the frozen base, so LoRA learns the served function and the merge into the W4A16 shards is exact. Test-first: ms-swift loading and SGLang logit parity.
- [I] **Freeze the router at first.** MoE routing is fragile (Router Replay, in the 10-17 notes).
- [I] **LR.** Full FT at 7–32B uses 5e-6 (LIMO, Retaining by Doing), which suggests LoRA at about 5e-5. The recipe's 1e-4 sits inside the band, and hidden 2560 is near the 2000 reference. Sweep {3e-5, 1e-4, 2e-4} with a global batch of 8–32.

### Gaps
- MoE LoRA under expert parallelism is an open question (Thinking Machines). There is no evidence on shared-expert-only MLP LoRA. On rsLoRA, DoRA and PiSSA I found no 2025–2026 evidence of gains over tuned standard LoRA, so treat them as unnecessary.

## Q4. Multi-turn agent SFT mechanics

### Takeaway
Render training data byte-for-byte as served: preserved thinking, the same effort sentence and the same tools schema. Put loss only on chosen assistant turns. Because thinking is preserved, one trim segment can carry loss on many turns.

### Cited Findings
- **Flash-Next model card.** "By default, Qwen3.8-Flash-Next retains thinking blocks from all historical messages … especially beneficial for agent scenarios." The template renders `<think>` for every past assistant turn when `preserve_thinking` is undefined or true; otherwise only after the last user query. [E] — [HF README](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/README.md); `chat_template.jinja` ~line 116
- **Effort sentence.** The template's default effort `xhigh` injects into the system turn "Reasoning effort is set to xhigh. Please think carefully …". `low` says "Keep your thinking brief"; `medium` adds nothing (template ~lines 45–56). The model card warns: "lower reasoning effort does not always reduce overall task completion time … more failures, and repeated retries." [E] — same sources
- **Our harness.**
  - It sends `enable_thinking` and `preserve_thinking` true (`kaggle/bench.json:15-19`; `ARC3-Inference/inference/agent/tool_agent.py:2335-2347`).
  - `prefix_reuse.render_assistant_turn` reproduces the rendering (`prefix_reuse.py:30-80`).
  - "dropping thinking from history was a large regression for Franzen" (`arms.md:390`).
  - `reasoning_effort=medium` didn't shorten thinking (`arms.md:231`).

  [E]
- **ms-swift masking.** A per-assistant-message `"loss"` field overrides `--loss_scale`, and ms-swift ≥4.2.0 accepts a per-message `loss_scale` weight. Tool responses never take loss. `preserve_thinking` can be passed per sample. [E] — [ms-swift Custom dataset](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html)
- **Loss on prompts.** It helps mainly with short outputs and little data. [E] — [arXiv 2405.14394](https://arxiv.org/html/2405.14394)
- **Long context.** "for SFT, using only short instruction datasets yields strong performance on long-context tasks". This concerns retaining long-context *ability*, not agent behaviour. [E] — [ProLong arXiv 2410.02660](https://arxiv.org/abs/2410.02660)
- **Memory.** Peak per GPU is 72.5 GiB at 8k with PLE offload and padding_free. Context parallelism is undocumented for this model. [E] — [ms-swift best practice](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html)

### Inferences
- [I] **Segment training.** Turn t renders the same whether or not later turns follow it. So one sequence from one trim point to the next can carry loss on every kept assistant turn, with `loss: false` on rejected turns, which still serve as context. Loss-token density rises from about 2–3% (one reply after a 55k prompt) to tens of percent.
  - A *replaced* turn (an R3 resample) makes every later turn off-policy context. Put resamples only as the final turn.
- [I] **Render-diff check.** Re-render logged requests with the training template (preserve_thinking, the served effort sentence, mem-unified tools and system prompt, `|trim`) and diff against the request logs before training.
- [I] **Length mismatch.** Train mostly on left-truncated post-trim contexts (system prompt, memory card, recent turns), plus some 32–64k sequences if memory allows. Gate at real 55k contexts. Packing with GDN needs the state reset at boundaries. Test-first: packed-vs-unpacked loss parity.
- [I] **Effort as a teacher, not a runtime setting.** Generate under `reasoning_effort=low`, keep the replies whose action matches the verified-good action, and train them under the served xhigh prompt. This is Munkhbat's conditioned BoN (Q6), and it avoids the model card's runtime risk.

### Gaps
- No published comparison of preserved- vs stripped-thinking *training* for Qwen3.6/3.8 agents. GDN varlen correctness under padding_free and context parallelism for GDN in Megatron-SWIFT are UNVERIFIED.

## Q5. Hyperparameters and the small-model wind tunnel

### Takeaway
For a strong model on curated data:
- 1–3 epochs;
- a small batch;
- a low LR (full FT ≈ 5e-6 at 7–32B; LoRA ≈ 10×);
- constant LR with a short decay;
- several evaluated checkpoints.

LR *values* transfer only through a family-specific hidden-size rule. *Rankings* transfer better (weak evidence).

### Cited Findings
- **LRs in use:**
  - full FT 5e-6 (LIMO 32B, 15 epochs on 817 samples; Retaining by Doing 7–8B, 2 epochs);
  - 5e-5 with batch 256 and 8 epochs from base models (2604.06628);
  - LoRA 1e-4 at r = 8, α = 32 (ms-swift);
  - DPO 1e-6 at β = 0.1 (ETO).

  [E] — as cited above
- **Short runs mislead.** Short-training checkpoints "can systematically underestimate" generalisation (a dip, then recovery). Response length rises early, then falls; the longest responses coincide with the weakest performance. Setting: base models distilled from Qwen3-32B. [E] — [arXiv 2604.06628 §3](https://arxiv.org/html/2604.06628)
- **No μP for fine-tuning.** MiniCPM's wind tunnel relied on μP, which does not apply to fine-tuning a non-μP checkpoint. [E/I] — `finetune-research-2026-10-17/training.md` §1

### Inferences
- [I] **One epoch per round, then regenerate** (Iterative-SFT). Multi-epoch training on a fixed set drifts off-policy. Save checkpoints at 25/50/75/100% and combine them with the λ-merges.
- [I] **Wind tunnel.** Use an MoE sibling (Qwen3-30B-A3B, which Thinking Machines used, or Qwen3.6-35B-A3B), not the dense 27B. Only an MoE can rank target-module choices and router freezing. Transfer rankings, not LR values. Whether 3.6-35B-A3B has GDN like Flash-Next is UNVERIFIED.
- [I] **Don't judge by training loss.** On-policy data starts at low NLL, so loss barely moves while behaviour changes.

### Gaps
- No evidence that fine-tuning hyperparameters transfer from a ~30B MoE to a 125B MoE with a different attention mix.

## Q6. What to train on for agent behaviour

### Takeaway
Best supported:
1. rejection-sampled own trajectories (RFT/ReST-EM), keeping the shortest correct reply for efficiency;
2. "early experience": the agent's own alternative actions and the states they produced, which needs no reward.

Agent DPO adds a few points but is fragile. Goal relabelling (HER) fits poorly, because the ARC-AGI-3 goal is fixed and unknown.

### Cited Findings
- **Shortest-correct self-training (Munkhbat et al., Findings ACL 2025).** Naive best-of-N with N = 16 preserves accuracy, but its length reduction "is limited to 12% on average". Reduction grows log-linearly in N, so going further costs exponentially more samples. Few-shot-conditioned BoN reaches about 30% (GSM8K/MATH, five model families). [E] — [arXiv 2502.20122 §3, §5](https://arxiv.org/html/2502.20122)
- **Kimi k1.5 long2short.** Shortest rejection sampling, DPO and weight averaging. DPO chosen = shortest correct; rejected = wrong longer replies *and* correct replies 1.5× longer than the chosen one. [E] — [arXiv 2501.12599 §2.4](https://arxiv.org/html/2501.12599)
- **ETO (Song et al., ACL 2024).** Trajectory-level DPO over success/failure pairs, iterated.
  - WebShop: 63.1 → 67.4 (Llama-2-7B), 66.3 → 70.7 (13B), 60.1 → 66.2 (Mistral-7B). ScienceWorld-Unseen: 57.6 → 68.6 (13B).
  - **Step-level** pairs collapsed to 8.3 (LR 1e-6, β 0.1) and reached 62.8 at 1e-7, β 0.5, below SFT's 63.1.
  - Without behaviour cloning first, ETO scored 12.5, while RFT scored 48.4.

  [E] — [arXiv 2403.02502 Tables 3–5](https://arxiv.org/html/2403.02502)
- **Early Experience (Zhang et al., Meta; ICML 2026).**
  - *Implicit world modelling*: predict next states from the agent's own alternative actions.
  - *Self-reflection*: explain why the chosen action beats sampled alternatives, given their outcomes.
  - Over imitation learning, IWM gave +2.3 to +5.5 on ALFWorld/ScienceWorld and +11.3 to +18.4 on WebShop. SR gave +12.8 to +15.0 on TravelPlanner.
  - Better OOD and a better RL warm start. Works with LoRA, up to 70B.

  [E] — [arXiv 2510.08558 §5](https://arxiv.org/html/2510.08558)
- **Hindsight Supervised Learning (ICLR 2026).** Relabels rollouts with the goals they achieved. On ALFWorld it beat full-data baselines using ¼ of the demonstrations. [E, secondary summary] — [OpenReview PDF](https://openreview.net/pdf/da69e9267374809a5be480e5c619b4a68725576f.pdf)
- **SFT and generalisation.** Chu et al.: SFT "tends to memorize", but "stabilizes the model's output format, enabling subsequent RL". [E] — [arXiv 2501.17161](https://arxiv.org/abs/2501.17161) "Debunk the Myth" attributes much of that failure to one frozen prompt template; prompt diversity plus CoT matches RL on Sokoban. [E, abstract] — [arXiv 2510.00237](https://arxiv.org/html/2510.00237v1)

### Inferences
- [I] **R3 is well supported but bounded.** Naive selection at k = 8–16 should give about 10–15% fewer tokens per round. Conditioned generation (low effort or concise exemplars, then trained without the condition) plus two rounds makes a 30% cut in tokens per action plausible, not given.
- [I] **"Correct" must be engine-verified.** The replayed action must go where the successful trajectory went. On stuck levels, keep samples that reached a new board (R4), not the shortest ones: short thinking may be the failure there, and the model card's warning applies.
- [I] **Early-experience data is nearly free.** Logs hold every next frame, and the state-lab can replay alternative actions. Two test-first mix components at 10–20% each:
  - (a) next-board prediction;
  - (b) a self-reflection turn contrasting the chosen action with no-op alternatives.

  The arms.md M3 caveat stands (better dynamics need not give better decisions). This is the first evidence that dynamics data helps *as an auxiliary in an imitation mix*.
- [I] **DPO.** If used, only Kimi-style same-state pairs, after SFT, with an NLL term and a low-β sweep. Never trajectory- or step-level DPO over whole games.

### Gaps
- No published SFT or DPO on hidden-rule interactive games. No study of shortest-correct selection in multi-turn agents, where shorter turns might cost extra turns.

## Implications for our plan

1. **Data.**
   - **Keep:** R2 and R3 as the core.
   - **Change:** make them iterative. Each round, regenerate from the current model and train 1 epoch (Iterative-SFT / ReST-EM).
   - **Precedent:** Qwen3 Stage 3 did the same on its RL-tuned reasoner (2505.09388 §4.3).
   - **Limit:** teacher and scripted rationales at ≤10%, or have the model rewrite them.
2. **Forgetting guard.** Replace "30–50% self-replay" with:
   - on-policy task data;
   - LoRA at α = 32 and rank ≥16;
   - λ-scaled merges (0.5–1.25), compared on the panel.

   Keep a 10–30% replay arm as test-first only.
3. **LoRA targets.**
   - Test attention + GDN + shared-expert LoRA against the recipe's attention-only list.
   - Freeze the router.
   - Use dequantised int4 experts as the frozen base.
   - Add routed-expert LoRA only if the MoE wind tunnel shows a clear gain.
4. **Hyperparameters.**
   - LR 3e-5 to 2e-4 (recipe 1e-4), global batch 8–32, constant LR with a short decay.
   - Checkpoints at 25/50/75/100%.
   - MoE wind tunnel; transfer rankings only.
5. **Mechanics.**
   - Run the render-diff check against the request logs.
   - Use segment sequences with per-turn `loss`, and put R3 resamples only as the final turn.
   - Check packed-loss parity before using padding_free.
6. **Test-first additions:**
   - effort-conditioned generation, distilled into the xhigh prompt;
   - Early-Experience auxiliaries;
   - same-state DPO after SFT.
7. **Gates:**
   - tool-call validity;
   - stuck-state action diversity at k samples;
   - cleared-level route agreement;
   - tokens per action;
   - levels per wall-clock at 55k contexts.

   Judge 2–4 checkpoints, not only the last.
8. **Don't:**
   - attention-only rank-8 LoRA as the sole arm;
   - multi-epoch training on a fixed self-generated set;
   - agent DPO without an SFT anchor;
   - judging by training loss.

## Fact-check (independent verifier, 2026-10-18)

Each claim was checked against the primary source: arXiv HTML/PDF, the Thinking Machines posts, the HF `config.json` / `chat_template.jinja` / README of `Qwen/Qwen3.8-Flash-Next`, the `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` `config.json`, the ms-swift docs, and the local files on `main`. No browser was used.

| # | Claim | Verdict | Correction / note |
|---|---|---|---|
| 1 | Qwen3 Stage 3 ran continual SFT on the Reasoning-RL model, with thinking data rejection-sampled from that model so its performance would not be compromised (2505.09388 §4.3, 2nd paragraph) | **confirmed** | The quote is exact. Caveat: the paper's stage table for Qwen3-32B still shows thinking-mode drops after Stage 3: AIME'24 83.8 → 81.9, LiveCodeBench v5 68.4 → 67.2, BFCL 69.0 → 68.4. The Stage 3 mix also held curated non-thinking data. The precedent shows that self-generated data limits regressions; it does not remove them. |
| 2 | Retaining by Doing: Self-SFT on Qwen-2.5-1.5B drops 3.0/14.0/19.5 vs GRPO 0.6/0.5/0.9; Iterative-SFT shows mild to no forgetting; setup of full FT, 2 epochs, LR 1e-4 / 5e-6 | **confirmed** | Table 2 (App. A.5), the §4.2 quote and the A.3 LRs and epochs are exact (teacher SFT 6.2/11.9/29.5 is also right). Scope: Fig. 7 tests Iterative-SFT only on Qwen-2.5-1.5B/7B-Instruct, on IFEval (10 rounds) and MMLU (3 rounds). A.3 never says "full FT"; it is implied because no PEFT is mentioned. The Fig. 3 caption calls 1e-4 the "default" LR, which contradicts A.3's 5e-6 for the 7-8B models. Venue confirmed: ICML 2026 (arXiv journal-ref). |
| 3 | SFT on Qwen3-32B's own unfiltered T=1 samples degraded IF-eval at any practical LR > 0 (TM On-Policy Distillation) | **confirmed** | The quote is exact: "any practical learning rate that's greater than zero leads to a degradation of performance on the instruction-following evaluation". The prompts were Tulu3 chat prompts. The samples were neither filtered nor selected, which matters when applying this to verifier-filtered data. Post date Oct 27, 2025 confirmed; the LoRA r=32 13% / 6% figures and the 85/79/45/83 and 36→41 numbers are also confirmed. |
| 4 | LoRA Without Regret: attention-only < MLP-only even at matched params (attn r256 0.25B vs MLP r128 0.24B), including Qwen3-30B-A3B and RL; LoRA LR ≈ 9.8× FullFT; LoRA less tolerant of large batches | **partially_confirmed** | The parameter-matched comparison and its counts are for **Llama-3.1-8B** only ("Parameter counts for LoRA on Llama-3.1-8B"). For Qwen3-30B-A3B-**Base** the post shows attention-only < MLP-only, but not at matched parameter count. The RL comparison is on MATH, in the Llama-3.1-8B-base RL setup. The 9.8 multiplier is a fit to Tulu3 SFT LR sweeps over 14 Llama/Qwen models (~15× for runs under ~100 steps). The large-batch penalty holds "in some settings" (shown on a 10k OpenThoughts3 subset). The post adds that both methods reach their best loss at small batch, "so this gap may not matter as much in practice". |
| 5 | Flash-Next shared expert = 640 = one routed expert (hidden 2560, 512 experts, top-10); it is the only BF16 MLP in the Intel W4A16 checkpoint, where only routed experts are int4 | **partially_confirmed** | All the `text_config` numbers are exact (also 48 layers, `full_attention_interval` 4, arch `Qwen4ExpForConditionalGeneration`). Wrong as written: the Intel config quantizes only `block_name_to_quantize = model.language_model.layers`. The **MTP layer's whole MoE (512 routed experts + shared expert) and the vision tower's MLPs are also BF16** (`extra_config`: `mtp.layers.N.mlp.experts.N.*` bits 16). `arms.md:634`, the cited source, itself lists `mtp` and `visual` as BF16. The claim holds only for the 48 main decoder layers. The same wording appears in the Q3 Takeaway ("the only BF16 MLP is the small shared expert"). |
| 6 | Template keeps all past thinking unless preserve_thinking is false; harness sends preserve_thinking=true; default xhigh adds a system sentence | **confirmed** | `chat_template.jinja` line 116 renders `<think>` for every assistant turn when `preserve_thinking is undefined or true`. Lines 45-56 set the xhigh sentence, which lines 59-61 and 80-85 put in the system turn. The sentence is added only when `enable_thinking` is undefined or true. `bench.json:15-19` and `tool_agent.py:2335-2347` (`ARC3_PRESERVE_THINKING` defaults to "1", merged at about line 7511) match; `reasoning_effort` is sent only while the effort ladder is active. The leader notebook `v4-mem-repair-10x-v1.ipynb` sets it on the server instead (`--default-chat-template-kwargs '{"preserve_thinking":true}'`). See the ms-swift caveat below on whether *training* renders the same. |
| 7 | Munkhbat: naive BoN (N=16) keeps accuracy but cuts length only ~12%; FS-BoN ~30%; FT on external GPT-4o concise traces severely drops accuracy | **partially_confirmed** | 12% (§5.1), N=16 (§4) and 30% (abstract, §5) are confirmed. The log-linear / exponential-cost statement (Fig. 3) is in **§3.2**, not §3.1. FT on **GPT-4o CoT does not drop accuracy "severely"**: Table 2 gives relative accuracy 97.65% (GSM8K) and 90.52% (MATH). The "severe drop" sentence describes the FT-External Data group as a whole, driven by Direct Answer (24.9/35.2) and Human CoT (83.8/75.6). For FT-GPT4o the paper says only that it "lies below the Pareto-curve" of self-training. Setting: small non-reasoning models (Llama-3.2-3B, Gemma-2-2B, Qwen2.5-3B, Qwen2.5-Math-1.5B, DeepSeekMath-7B) with ~200-480-token outputs. |
| 8 | ETO: step-level DPO collapsed to 8.3 (lr 1e-6, β 0.1) vs SFT 63.1 and trajectory ETO 67.4; ETO w/o BC 12.5 vs RFT 48.4 | **confirmed** | Table 4 and Table 5 are exact (WebShop, Llama-2-7B-Chat). Context: the untuned Llama-2-7B-Chat scores 17.9, so ETO without BC lands *below the untuned model*; RFT+ETO without BC reaches 51.2. |

### Other errors found in the notes
- **LIMO (Q1):** "817 curated samples" is v1, which reported AIME 57.1% / MATH 94.8%. The 63.3% / 95.6% figures (and 16.5→57.5, 49.9→32.3, 78.1) come from the current version, which trains on **800** samples ("LIMO ours(800)"). The LR 5e-6, cosine, no warmup, 15 epochs and batch 64 are confirmed.
- **s1 (Q1):** the LR and epochs marked UNVERIFIED are correct. App. D: LR 1e-5, 5 epochs, batch 16, 315 steps, 5% linear warmup, then cosine to 0.
- **SDFT 2026 (Q1):** the cost marked UNVERIFIED is in the paper itself: "approximately 2.5× the computational cost in FLOPs and roughly 4× the wall-clock training time compared to SFT". The ICML 2026 venue is not shown on the arXiv page, so it stays unverified.
- **ms-swift masking (Q4):** "`preserve_thinking` can be passed per sample" overstates the docs. Per-sample `chat_template_kwargs` needs ms-swift ≥ 4.3.0, and the docs say `enable_thinking`/`preserve_thinking` "take effect during inference". Training-time rendering of historical thinking is therefore undocumented, and the render-diff check should run on ms-swift's *training* encode output. The per-message `"loss"` field also overrides only the basic `--loss_scale` strategies (default/last_round/all); components such as `ignore_empty_think` still apply.
- **ExPO (Q2):** "DPO trained 20% of the steps + ExPO beat full DPO" comes from the controlled experiments. The "1.8B-70B" range refers to improving existing open-source models on AlpacaEval 2.0 / MT-Bench, not to the 20%-steps result.
- **Munkhbat (Q6):** "five model families" should be five main models from four families (Qwen2.5 and Qwen2.5-Math share a family).
- Spot-checked and correct: Illusion of Equivalence quotes (α=2r vs α=8, lower ranks forget more in sequential training), RL's Razor quote and authors, AgentTuning η=0.2, Early Experience gains (+2.3 to +5.5, +11.3 to +18.4, +12.8 to +15.0, LoRA at 70B), 2604.06628 defaults (5e-5, batch 256, 8 epochs) and quotes, WiSE-FT abstract, the ms-swift recipe flags and 72.5 GiB row, the HF README quotes, `prefix_reuse.render_assistant_turn` (about line 74), and `arms.md:231` and `:390`.
