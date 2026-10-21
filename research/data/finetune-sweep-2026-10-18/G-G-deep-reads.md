# G-G: deep reads of the key cautions, and the five round-1 contradictions

Researcher G-G, 2026-10-18. I read seven papers in full text (arXiv PDFs converted with `pdftotext`; tables read from the layout text), not their abstracts. The papers are 2607.05184, 2502.04463v4, 2510.18874v2, 2604.06268v1, 2605.26494v2 (the MiniMax-M2 report) plus the MiniMax blog, 2602.10090v3 and 2601.20802v2. I then settled contradictions C2, C4, C8, C5 and C6 from `_critic.md:211-217`. All quoted numbers come from these papers' tables. Lines marked [I] are my inferences.

## Q1. The seven deep reads: setup, numbers, and what each means for us

### Takeaway
Every caution holds up in full text, and most are stronger than their abstracts suggest. The closest analogue to R3 collapsed. It did worse than simple truncation, at a learning rate picked because it shortened replies. Privileged self-distillation hurts specifically when training rollouts include thinking. SDPO's positive results were all run with thinking off. Labs that train with retained thinking train each call on exactly the context it was served, not on packed segments.

### Cited Findings

**"Rethinking On-Policy Self-Distillation for Thinking Models"** (Kaur, Ri, He, Fowl, Arora; Princeton; [arXiv 2607.05184v1](https://arxiv.org/abs/2607.05184), 2026-07-06).
- **Setup.**
  - The teacher is a copy of the initial student, given privileged context. Loss is JSD (α=0.5), 1 epoch, effective batch 64, 1 completion per prompt at T=1.0, full FT (FSDP, bf16) on 8×H100.
  - LR is 5e-6 for models up to 4B and OLMo-7B, and 2e-6 for Qwen3-8B, with a cosine schedule and 0.1 warmup.
  - Training completions are capped at 4,096 tokens; evaluation uses 16 samples × 38,912 tokens (App. B, Tables 9-11).
  - Data: an OpenThoughts 15k subset (privileged context = full solution) and Countdown 15k.
- **Table 1 (avg@16 over AIME24/25 and HMMT25).** All five thinking models drop:

  | Model | Base | +OPSD |
  |---|---|---|
  | Qwen3-1.7B | 0.372 | 0.308 |
  | Qwen3-4B | 0.591 | 0.534 |
  | Qwen3-8B | 0.635 | 0.578 |
  | Qwen3-4B-Think-2507 | 0.720 | 0.683 |
  | OLMo-7B-Think | 0.612 | 0.604 |

  Bootstrap CIs exclude zero for 4 of the 5 (App. C.1).
- **Table 3: thinking during training is the trigger.** Qwen3-4B trained with thinking on falls 0.591→0.532. Trained with thinking off (evaluated with thinking on) it holds at 0.590.
- **Table 4: the privilege is the cause, not distillation.** On Qwen3-1.7B, vanilla OPD from a Qwen3-8B teacher raises the score 0.372→0.392. OPD with a gold demo gives 0.350, and OPSD 0.308.
- **Budget dependence (Fig. 1, Fig. 3).** OPSD helps at 4-8k-token budgets and matches or falls below base at 32-38k. A dense demo shrinks mean length to about 0.8× base at 38k; final-answer-only context does milder damage.
- **Mechanism (diagnostics; the authors say they are not causal, §4.3).**
  - Qwen3-4B thinking: median fork rate 0.083→0.035 and lock rate 0.69→0.77 from base to dense privilege. The instruct model stays flat (about 0.06 / 0.78).
  - Epistemic-token density: 1.080% at base, 1.074% after OPD, 0.850% after gold-demo OPD. "wait" falls 3.85→2.54 per 1k tokens (Table 5).
  - Over 7,200 paired rollouts, per 1k tokens: verification 1.63→1.35, backtracking 6.45→6.29, hedging 3.79→3.62 (Table 6).
- **A conciseness-prompted self-teacher with no gold context (CRISP-style; Qwen3-8B, Fig. 9)** shortens rollouts but stays below base at long budgets: "making the student concise alone is not enough".

**Arora & Zanette, "Training Language Models to Reason Efficiently"** ([arXiv 2502.04463v4](https://arxiv.org/abs/2502.04463), NeurIPS 2025): the collapsed RS+SFT baseline.
- **Exact setup (§5.2).**
  - Sample 8 solutions per prompt from R1-Distill-Qwen-1.5B or 7B and keep the shortest correct one.
  - 3,200 prompts gave about 2,200 (1.5B) and 2,500 (7B) examples.
  - Full FT (OpenRLHF, ZeRO; App. A). LRs tried: 1e-5, 5e-6 and 2e-6. "We find that 5e-6 effectively reduces response length in a meaningful way."
  - Epochs are not stated (UNVERIFIED). Evaluation is at T=0.6 with a 32k cap.
- **7B results (Tables 8-12), base at 32k → SFT:**

  | Benchmark | Base: accuracy @ tokens | SFT: accuracy @ tokens |
  |---|---|---|
  | AIME24 | 0.563 @ 12,837 | **0.183** @ 5,806 |
  | MATH500 | 0.932 @ 4,086 | 0.782 @ 1,407 |
  | GSM8K | 0.920 @ 1,547 | 0.881 @ 427 |
  | CommonsenseQA | 0.670 | 0.657 |
  | Logical Deduction | 0.826 | 0.830 |

  - The **base model simply truncated at 8k scores 0.390 at 6,814 tokens on AIME**. So SFT is far worse than cutting generation off at a similar length.
  - Normalised over benchmarks (Table 13): SFT reaches 0.822 accuracy at 0.597 of base tokens. DPO reaches 0.960 at 0.737, and RL α=0.2 reaches 0.959 at 0.516.
- **1.5B AIME:** 0.320 @ 15,476 → 0.243 @ 13,922. Accuracy fell 7.7 points for only a 10% length cut (Table 3).
- **The paper does not diagnose the collapse.** It does report two related facts:
  - its RL compresses easy problems more than hard ones (α=0.2: GSM8K −83% vs AIME −27%, §5.3.1);
  - compression lowers verification counts (4.596 for R1 → 1.176 at α=0.4, Table 2).
- **The DPO baseline** used longest-vs-shortest correct pairs at LR 1e-5. 7B: AIME 0.523 @ 13,147, MATH500 0.908 @ 3,277, GSM8K 0.859 @ 506.

**"Retaining by Doing"** (Chen, Razin, Narasimhan, Chen; [arXiv 2510.18874v2](https://arxiv.org/abs/2510.18874)).
- **Setup (App. A.3).**
  - Models: Llama-3.2-1B, Llama-3.1-8B, Qwen-2.5-1.5B and Qwen-2.5-7B (all Instruct). Target tasks: IFEval (13k), MMLU (12k), Countdown (10k).
  - **Self-SFT:** sample 5 times from the *initial* model, filter by reward, and keep *all* correct replies per prompt.
  - Full FT with AdamW. LR 1e-4 (1-1.5B) or 5e-6 (7-8B), cosine with 0.03 warmup, batch 128 (64 for Countdown), 2 epochs, max length 4,096.
  - GRPO baseline: group 5, KL 0.05.
- **Table 2 (Qwen-2.5-1.5B), non-target drop / KL, trained on IFEval, MMLU and Countdown:**

  | Method | IFEval | MMLU | Countdown |
  |---|---|---|---|
  | Self-SFT | 3.0 / 25.0 | 14.0 / 3.0 | 19.5 / 896.1 |
  | Teacher SFT | 6.2 / 47.8 | 11.9 / 9.2 | 29.5 / 846.6 |
  | GRPO | 0.6 / 1.5 | 0.5 / 0.4 | 0.9 / 34.9 |

  Llama-1B Self-SFT drops 6.9 / 34.6 / 25.3.
- **Learning rate (Fig. 3, §2.3).** At LR 1e-5 Self-SFT forgets less, but it "fails to reach the same target performance even with more epochs" (10 epochs tested).
- **On-policyness is what protects (§4.1).** GRPO with β=0 behaves like β=0.05, and REINFORCE forgets as little as GRPO. The authors credit on-policy data, not the KL term or the advantage estimator.
- **Iterative-SFT** regenerates data at the start of each epoch. It reaches target accuracy "higher than or comparable to" SFT "while only exhibiting mild to no forgetting" (Fig. 7; Qwen 1.5B and 7B on IFEval and MMLU). SFT on GRPO's own rollouts also forgets less (App. A.4.1). KL vs forgetting: Pearson 0.52 (App. A.5).

**RAGEN-2** ([arXiv 2604.06268v1](https://arxiv.org/abs/2604.06268), 2026-04-07): the result tables.
- **Table 4: SNR-Aware Filtering** (keep prompts with the top reward variance, top-p ρ=0.9), shown as baseline peak (gain):
  - PPO Qwen2.5-3B average 67.6 (+6.9); its Sokoban 12.9 (+16.0);
  - GRPO 67.5 (+3.7); Dr. GRPO 55.8 (+0.8);
  - Qwen2.5-7B 77.3 (+4.1); Qwen2.5-VL-3B with image input 42.3 (+35.8).
- **Table 6: quartile ablation (Sokoban, 3B).** Training only on the highest-variance quartile gives 21.1% (MI 0.95); the lowest gives 11.0% (MI 0.73).
- **Table 7: prompt-level vs trajectory-level filtering.** Prompt-level 23.6% (MI 1.80), trajectory-level 16.8% (MI 0.20), no filter 12.9% (MI 0.83).
- **Table 9: filters that collapse.** A *length-based* filter scored 0.16 vs 0.17 unfiltered and collapsed (✗), even though its MI was 0.91. Reward variance scored 0.38 and stayed stable.
- **Diagnostics.**
  - Spearman correlation with success: +0.39 for MI-ZScore vs −0.11 to −0.14 for the entropy metrics (Fig. 8).
  - Format validity is "largely decoupled from collapse" (Fig. 12).
  - Reasoning length "declines monotonically across all eight environments ... a behavioral signature of template collapse" (§4.2, Fig. 7).
  - The MI proxy Retrieval-Acc scores each trace against every prompt in the batch; under collapse it falls to chance, 1/P (Table 1).

**MiniMax-M2 series report** ([arXiv 2605.26494v2](https://arxiv.org/abs/2605.26494), 2026-07-30; 229.9B total, 9.8B active).
- **SFT (§5).** "Our SFT data interleaves thinking traces with intermediate actions and observations", built by "large-scale rejection sampling against domain-specific reward signals followed by multistage data cleaning".
- **History (§7.1, Eq. 10).** Under "reasoning state persistence" the full output, including thinking, is appended: H_{t+1} = H_t ⊕ assistant(r_t, a_t) ⊕ tool(o_t).
- **Training unit (§6.1.3).** "Training operates on individual (s_t, a_t) pairs as atomic units ... the model need not be aware of whether s_t resulted from a simple message append, an aggressive context truncation, or a complete history rewrite." For black-box agents, s_t is "the context as presented to the LLM at each completion request" (§6.2.3).
- **Prefix-tree merging (§6.2.5)** computes shared prefixes once. It is "mathematically equivalent" to independent samples and gives "up to 40× training speedup".
- **MTP heads** are "continuously co-trained with the RL policy via top-K KL divergence loss" (§6.2.6).
- **Retained vs discarded prior thinking** (MiniMax blog [minimax.io](https://www.minimax.io/news/why-is-interleaved-thinking-important-for-m2), 2025-11-03):

  | Benchmark | Retained | Discarded |
  |---|---|---|
  | SWE-Bench Verified | 69.4 | 67.2 |
  | τ² | 87 | 64 |
  | BrowseComp | 44.0 | 31.4 |
  | GAIA | 75.7 | 67.9 |
  | xBench | 72.0 | 66.0 |

**Agent World Model** ([arXiv 2602.10090v3](https://arxiv.org/abs/2602.10090)): the Fig. 4 points.
- **Setup.**
  - GRPO with LR 7e-7, 64 tasks × G=16 = 1,024 instances per step, 96 steps, KL 0.001.
  - Max response 2,048 tokens, 32k context, 20 turns, history window 3 (Table 12).
- **Fig. 4 is Qwen3-4B only.** The bar labels are rendered in a shifted font; I decoded them by character offset (script below). Values are BFCLv3 / τ² Pass@1 / MCP:

  | Training set | BFCLv3 | τ² Pass@1 | MCP |
  |---|---|---|---|
  | Base (Table 4) | 54.92 | 15.83 | 6.15 |
  | 10 environments | 48.0 | 14.2 | 3.9 |
  | 100 environments | 57.0 | 16.6 | 6.1 |
  | 191 environments (AWM) | 57.6 | 16.4 | 6.1 |
  | EnvScaler, 191 environments | 54.1 | 29.0 | 4.5 |
  | 526 environments | 64.5 | 22.6 | 6.7 |

  - The 526 and EnvScaler bars match Table 4 exactly (64.50/22.57/6.70 and 54.06/28.96/4.47). That validates the decode. The 100 and 191 AWM bars are decoded only (medium confidence).
- **Table 11: context mismatch.** The history-limited model evaluated with full history falls on τ² Pass@1 from 22.57 to 9.35 (Pass@4 43.89→15.11). The full-history model is "relatively insensitive" (§6.3). AWM's fix, "sample splitting", makes "each training sample mirror the truncated context the agent will see at inference" (App.).

**SDPO, "Reinforcement Learning via Self-Distillation"** ([arXiv 2601.20802v2](https://arxiv.org/abs/2601.20802)).
- **Setup (Table 12, E.1).**
  - Models: Qwen3-8B and Olmo3-7B-Instruct. **Thinking: False in every setting.** Max prompt 2,048, max response 8,192.
  - 32 questions × 8 rollouts per step. Top-K distillation with K=100 (JSD) or K=20 (reverse KL); teacher EMA rate 0.05 or 0.01.
  - Full FT, LR 1e-5 or 1e-6 constant; verl with vLLM on one node of 4×GH200, about 6 h per run.
- **Cost.** Time per step is +5.8% or +17.1% over GRPO (Fig. 5). An unregularised teacher eventually diverges; trust-region and EMA teachers score 50.6 and 49.3 (Table 4).
- **Results.**
  - Qwen3-8B at 5 h (Table 3): Chemistry 80.9 vs GRPO 74.5; Physics 75.6 vs 72.7; **Biology 56.8 vs 59.9 (GRPO wins)**.
  - Mean response length 255.8 vs 820.8 tokens for GRPO, 3.2× shorter (Table 8).
  - LCBv6 (Table 5): base 27.9, SFT on self-teacher 42.7, GRPO 41.2, SDPO 48.8. Holdout average: 43.5 / 41.4 / 41.8 / 42.4.
  - Putting the student's own attempt in the teacher's context collapses diversity: 30.0% same output vs 10.1% (Table 6).
  - Gains grow with model size from 0.6B to 8B (Fig. 8).

### Inferences
- [I] **Our setting is the harmful one.**
  - Flash-Next is a thinking model and the leader runs with thinking on, so 2607.05184's Table 3 trigger applies.
  - W10's teacher option (b), a successful sibling placed in context, is the "dense gold demo" condition, the worst case. Option (a), outcome text such as "no change" or "new board", is closer to the milder sparse condition.
  - Per-reply budgets are effectively uncapped: 4 of 700 replies hit `length`, max 12,288 (`W7-efficient-reasoning.md:190`). That is the long-budget regime where OPSD loses.
- [I] **One mechanism probably sits under four papers.** Selecting or distilling toward short output strips verification and fork tokens (Arora Table 2, 2607.05184 Tables 5-6). RAGEN-2 sees falling length as the signature of collapse. Arora's SFT, trained at one LR on all prompts at once, compressed AIME by 55% when RL compressed it by 8-27%.

### Gaps
- Arora's SFT epochs and validation procedure are not stated.
- MiniMax does not say whether historical thinking takes loss.
- No paper here tests a model over 14B, a context over 38k, or ARC-like games.

## Q2. Settling contradictions (a)-(e)

### Takeaway
Here are the five verdicts:
- **(a)** R3 is **medium** risk. It is safe only with per-state, ratio-capped selection, a median-length anchor, regeneration each round, a LoRA LR of about 1e-5 to 3e-5, and fork-marker gates.
- **(b)** Use **0%** generic self-replay.
- **(c)** Packing does not apply to R3 data. Train per call, on the deployed context.
- **(d)** Take LoRA LR as about 10× the full-FT LR.
- **(e)** Use same-state sibling pairs only, after SFT, as a test-first step. No segment-level or whole-trajectory pairs.

### Cited Findings
- **Local facts for (a).**
  - At hard states, right replies are not shorter than wrong ones: P(success shorter) = 0.4447 over 1,048 pairs (`W7-efficient-reasoning.md:260`).
  - The shortest right reply is 0.677× the median right reply (`W7-efficient-reasoning.md:262`).
- **Counter-precedent for (a).** Qwen3's Stage 3 builds its "thinking" data by rejection sampling from the model itself (`W1-sft-pedagogy.md:20`).
- **Self-replay does not anchor (b).**
  - Thinking Machines: even 100% chat resampled from Qwen3-8B itself did not keep IF-eval at 85 (`W4-lab-pipelines.md:252`).
  - PAPRIKA: WildChat SFT degraded every task group (`W8-exploration-worldmodel.md:252`).
- **Normalised history (c).** The normaliser strips blank lines and think tags, so history "differs from the tokens the model generated" (`ARC3-Inference/CONFIGURATION.md:878`).
- **The only full run with verbatim history (c).** `tput-safe` scored 42.09 vs control 50.62. It was confounded with the ledger, gate11 and stub-dead levers (`arms.md:538,544`).
- **Window facts (c).** Append-only windows hold a median of 15 requests. The final prompt has a median of 79.9k tokens, and post-trim windows start at about 42k (`L1-data-inventory.md:150-155`).
- **Compute (c).** Training throughput is an estimate: 2-7 node-hours per 100M tokens, at an UNVERIFIED MFU (`W6-toolchain-infra.md:130`). An ND H200 v5 node costs $84.80/h (task brief).
- **LoRA LR (d), from "LoRA Without Regret"** (Schulman / Thinking Machines, [blog](https://thinkingmachines.ai/blog/lora/), 2025-09-29):
  - "the optimal LR for LoRA is consistently 10x the one used for FullFT"; the fitted multiplier is 9.8.
  - "around 15x" for runs under about 100 steps.
  - The optimum moves by less than 2× between rank 4 and rank 512.
  - LoRA is "less tolerant of large batch sizes".
  - Attention-only LoRA underperforms MLP LoRA even at matched parameter count.
- **Full-FT LR anchors (d).**
  - Arora collapse: 5e-6 at 7B.
  - Retaining by Doing forgetting runs: 5e-6 at 7-8B.
  - OPSD: 2e-6 at 8B.
  - SDPO: 1e-5 or 1e-6.
  - AWM GRPO: 7e-7.
  - PAPRIKA: SFT 1e-6, RPO 2e-7 (`W8-exploration-worldmodel.md:289`).
- **Preference-pair evidence (e).**
  - ETO (WebShop, Llama-2-7B): trajectory-level DPO 67.4 vs SFT 63.1; step-level DPO collapsed to 8.3; ETO without behaviour cloning scored 12.5 (`W1-sft-pedagogy.md:297`).
  - PAPRIKA uses **one pair per task**, because more caused "unintentional unalignment". Its RPO beat SFT-only on 6 of 6 groups (`W8-exploration-worldmodel.md:286-289`).
  - Kimi long2short DPO rejects wrong replies and correct replies 1.5× or more longer (`W1-sft-pedagogy.md:210`).
  - Arora's DPO (above) was far gentler than its SFT.

### Inferences
**(a) R3 risk: MEDIUM. A recipe that makes it safe** (thresholds marked P are my proposals, not from any source):
1. **Select per state, with a ratio cap.**
   - Keep the shortest verified sibling only if it is at least about 0.7× the state's median verified length (P; the local ratio is 0.677).
   - Otherwise keep the next-shortest that clears the cap. Skip states with fewer than 2 verified siblings.
   - This avoids Arora's extreme-tail, pooled selection.
2. **Keep a length anchor.** 30-50% of examples should be median-length verified siblings at the same states (P), so hard states keep "think longer when needed". Kimi's 1.5× rule bounds what counts as long.
3. **Never** use a concise-prompted, `reasoning_effort=low` or privileged self-teacher as the source of R3 targets (2607.05184 Fig. 9).
4. **One epoch per round, then regenerate** from the current checkpoint (Iterative-SFT), with at most 2-3 rounds before 11-02.
5. **LoRA LR 1e-5 to 3e-5** (full-FT equivalent 1e-6 to 3e-6, below Arora's 5e-6). Save checkpoints at 25/50/75/100% and sweep λ ∈ {0.25, 0.5, 0.75, 1}.
6. **Gates, judged separately on hard and easy frozen states:**
   - right-move rate;
   - verification, backtracking and hedging markers per 1k tokens (2607.05184 Table 6 families), with a relative fall over 10% as the back-off trigger (P);
   - distinct actions across k samples;
   - no-tool-call rate;
   - a Retrieval-Acc spot check (RAGEN-2): about 100 replies × 8 candidate states, as prefill only;
   - tokens per decision.

**(b) Self-replay share: 0% generic or unfiltered self-replay.**
- The failures were not only unfiltered T=1 samples. Filtered, one-shot Self-SFT also forgot (Table 2), and pure self-chat still lost IF-eval.
- What protects is on-policyness of the task data, low LR, LoRA's small capacity, and λ-merging.
- The anchor should be the within-task median-length siblings from (a). Use OPD to the base model only if vLLM boots.
- Skip the critic's 0% vs 20% pair. It costs GPU-hours, and both arms of it are predicted not to anchor.

**(c) Segment packing with normalised history: not needed for R3.**
- R3 targets are state-lab siblings. They never appear in the logged window, so later turns can't supervise them. Each example is (s_t as served, sibling reply), which is MiniMax's (s_t, a_t) unit and AWM's sample splitting.
- **Cost:** 3,000 examples × 57k median context ≈ 171M tokens ≈ 3.4-12 node-hours ≈ $290-1,020. Packing is not needed at that scale.
- **For behaviour cloning of logged replies only:** either validate `ARC3_VERBATIM_HISTORY=1` alone in a full run first (its only full run is confounded), or use prefix-tree branching. Branching keeps the context normalised and the target raw. It needs GDN state forking at the branch points; I found no ms-swift support for it (UNVERIFIED).
- **Don't train on short rebuilt contexts if the harness serves 42-85k windows.** AWM Table 11 shows a context-format mismatch costing 59% relative on τ².
- State-lab must save the raw reply and token ids. The logs keep only the normalised reply (`L1-data-inventory.md:87`).

**(d) LoRA learning rates: state each LR as a full-FT equivalent and multiply by about 10 for LoRA (about 15 for runs under 100 steps).**
- W3's 1e-6 to 2e-6 (full FT) becomes LoRA 1e-5 to 3e-5. That is R3's band.
- W1's 1e-4 is about 1e-5 full-FT equivalent, twice Arora's collapse LR. Use it at most as the sweep ceiling, not as the default.
- RPO: PAPRIKA's 2e-7 becomes about 2e-6 for LoRA.
- Keep the global batch at 32 or below.
- Caveat: the 10× rule was measured with LoRA on all layers. Ours leaves out the routed experts, so treat 10× as a starting point.

**(e) Preference pairs after SFT: same-state sibling pairs, as a test-first step.**
- One pair per state. Chosen: a verified sibling that clears the ratio cap. Rejected: a wrong-action sibling, or a verified sibling 1.5× or more longer.
- Train RPO with an NLL term, low β and LoRA LR about 2e-6.
- Per-level segment pairs and whole-trajectory pairs are **not** valid DPO pairs from logs, because two runs reach a level with different histories, i.e. different prompts. Building them from a frozen level start costs full rollouts (12 game-runs per GPU-hour, `_critic.md:222`). Skip them before 11-02.
- Length pairs penalise verification tokens directly, so DPO gets the same marker gate as (a).

### Gaps
- No direct test of selection at a 0.7× ratio cap; this is extrapolated from Arora and W7.
- Prefix-tree support for GDN in any trainer is unconfirmed.
- The 10× rule has not been checked on a partial-module MoE LoRA.

## Implications for our plan
1. **Change R3 from "low risk" to medium risk** and adopt recipe (a). Make the first R3 run a frozen-state screen, judged by the gates in (a)6, before any full run.
2. **Demote SDPO and OPSD.**
   - All of SDPO's evidence is thinking-off, at 8k responses or less and 8B or smaller. With thinking on, privileged OPSD loses 3-10% relative.
   - If one is tried, give the teacher sparse outcome text only. Never put sibling solutions or the student's attempt in the teacher's context. Gate on fork markers.
3. **Generic self-replay: 0%.** Use the median-length anchor instead, with λ-merge as the forgetting back-off and OPD only if vLLM boots by about day 5.
4. **Data format: per call, on the deployed shape.**
   - Each example is the served normalised post-trim context plus the raw sibling reply. Historical turns get loss=false; past thinking is retained in context, as MiniMax and our `preserve_thinking` do.
   - Budget 3-12 node-hours per 3,000 examples, and log raw replies and token ids in state-lab now.
5. **Learning rates.** LoRA SFT 1e-5 to 3e-5, LoRA RPO about 2e-6, batch 32 or below, one epoch per round.
6. **Select states by outcome variance** (RAGEN-2 Tables 6-7), but not by length (Table 9 collapse).
7. **Wait on R4 and any content-teaching SFT until external games exist.** 25 public game families sit in AWM's degrade regime, where 10 environments scored below base on all three benchmarks. AWM gains were small at 100 environments and flat to 191.
8. **MiniMax co-trains its MTP heads during RL.** After any trunk LoRA, retrain or check the MTP drafter (R1) before measuring tokens per second.

AWM Fig. 4 decode: `pdftotext -layout -f 9 -l 9 2602.10090.pdf`, then a +29 code-point shift on the control characters (scratch: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/ce155f4d-514a-4385-ac65-a7e7e5a23e1c/scratchpad/gg-scripts/decode_fig.py`).

## Fact-check (independent verifier, 2026-10-18)

I checked all 10 key results and skipped none. I downloaded the seven arXiv PDFs again at the cited versions into a new folder, converted them with `pdftotext` (layout and raw), and read each cited table and passage. I fetched both blogs again, rendered AWM page 9 to an image, and re-read the cited local files. Scratch files: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/ce155f4d-514a-4385-ac65-a7e7e5a23e1c/scratchpad/fcgg-dl/` (PDFs), `.../fcgg-txt/` (text), `.../fcgg-web/` (blogs) and `.../fcgg-work/awm_fig4-09.png` (rendered Fig. 4).

**Summary:** 7 confirmed and 3 partially confirmed (5, 10, and the scope of 3 and 4 noted). None refuted. The table numbers are exact everywhere. The corrections are about what the numbers mean.

1. **Arora RS+SFT (2502.04463v4): confirmed.**
   - §5.2 matches word for word: 8 samples, shortest correct kept, 3,200 prompts giving about 2,200 (1.5B) and 2,500 (7B) examples, LRs 1e-5 / 5e-6 / 2e-6, and "5 × 10−6 effectively reduces response length in a meaningful way".
   - Tables 8-10 match: AIME24 0.563 @ 12,837 → 0.183 @ 5,806; MATH500 0.932 → 0.782; GSM8K 0.920 → 0.881. The 8k-cutoff row is 0.390 @ 6,814. Table 13: SFT 0.8216 / 0.5968 and DPO 0.9602 / 0.7369.
   - Epochs are indeed not stated anywhere.
   - Note: "full FT" is implied, not stated. App. A (OpenRLHF, ZeRO 2/3) describes the RL runs, and no PEFT is mentioned anywhere. The results at the other two SFT LRs are not reported.
2. **OPSD (2607.05184v1): confirmed.**
   - Table 1 matches for all five models (Qwen3-8B 0.635 → 0.578; Qwen3-4B 0.591 → 0.534). Table 3 (thinking off) gives 0.590.
   - Table 4: OPD 0.392, OPD gold demo 0.350, OPSD 0.308.
   - Tables 9-10: JSD α=0.5, 1 epoch, effective batch 64, LR 5e-6 (2e-6 for Qwen3-8B), 4,096-token completion cap. The evaluation cap is 38,912 (§2, B.5).
   - Note: Table 3 trains on OpenThoughts **30k**, not the 15k used in Table 1, so its thinking-on row is 0.532, not 0.534.
3. **Deliberation diagnostics (2607.05184v1): confirmed, with scope notes.** All numbers match, but they come from three different setups:
   - The fork rate 0.083 → 0.035 is "roughly", read from Fig. 4 (§4.1). It is measured on the **teacher's** distribution along sampled rollouts, base vs dense privileged context. It shows what the supervision signal looks like, not a trained student's output.
   - The epistemic density 1.080% → 0.850% is **Qwen3-1.7B** under OPD with a gold demo from a **Qwen3-8B teacher** (Table 5, §4.2), not OPSD and not Qwen3-4B.
   - Verification 1.63 → 1.35 per 1k tokens (Table 6, §4.3) is pooled over the five thinking models, from 7,200 paired rollouts.
   - Fig. 9 (Qwen3-8B, conciseness teacher with no gold context) "remains below the base and gold-solution-only curves" at long budgets.
   - The authors say outright that these diagnostics are not causal.
4. **Retaining by Doing (2510.18874v2): confirmed.**
   - App. A.3 matches: 5 samples from the initial model, all correct ones kept, LR 1e-4 (1-1.5B) or 5e-6 (7-8B), 2 epochs, max length 4,096.
   - Table 2 (in App. A.5): Self-SFT 3.0 / 14.0 / 19.5 vs GRPO 0.6 / 0.5 / 0.9. Pearson 0.52 (App. A.5). The "mild to no forgetting" quote is §4.2, Fig. 7.
   - Caveat 1: Fig. 3's caption calls 1e-4 the "default" LR for all four models, which contradicts A.3's 5e-6 at 7-8B. So "1e-5 forgets less but misses the target" is unambiguous only at 1-1.5B.
   - Caveat 2: Iterative-SFT was tested only on Qwen-2.5-1.5B and 7B, on IFEval and MMLU.
5. **RAGEN-2 (2604.06268v1): partially confirmed.**
   - Confirmed: Table 4 (PPO Qwen2.5-3B 67.6 (+6.9)), Table 6 (Q1 21.1% vs Q4 11.0%), Table 7 (23.6% vs 16.8%), and the §4.2 / Fig. 7 quote.
   - **Correction (Table 9):** the *unfiltered* baseline (0.17) is also marked collapsed (✗). So are the reward-sum, entropy and entropy-variance filters. Only reward-variance filtering (and the sampling variants) stayed stable. The length filter **failed to prevent** collapse; it did not cause it. The paper says "Entropy- and length-based filters either suppress MI or fail to prevent collapse".
   - Minor: Table 6 is not strictly monotonic (Q3 10.7% < Q4 11.0%). The Fig. 7 length decline comes from eight environments reproduced from other papers' setups.
6. **AWM (2602.10090v3): confirmed, and stronger than stated.**
   - I rendered page 9 with `pdftoppm` and read the bars directly: 10 envs 48.0 / 14.2 / 3.9; 100 envs 57.0 / 16.6 / 6.1; 191 envs 57.6 / 16.4 / 6.1; EnvScaler 54.1 / 29.0 / 4.5; 526 envs 64.5 / 22.6 / 6.7. So the 100 and 191 values are now **verified from the image, not only from the decode**, and medium confidence can be raised to high.
   - The base row is from Table 4 (54.92 / 15.83 / 6.15) and is not in Fig. 4.
   - Table 11: misaligned w/ HL τ² P@1 22.57 → 9.35 (P@4 43.89 → 15.11). Table 12: LR 7e-7, G=16, 1,024 instances, 96 steps, KL 0.001.
   - Note: 100 environments gave MCP 6.1, equal to base (6.15). The 100- and 191-environment gains over base are about 2-3 BFCL points, about 0.6-0.8 τ², and zero on MCP.
7. **SDPO (2601.20802v2): confirmed.**
   - Table 12 / E.1: Thinking False in all three settings, max response 8,192, 4×GH200, LR 1e-5 or 1e-6 constant, top-K 100 or 20, teacher EMA 0.05 or 0.01.
   - Table 3 caption: about 6 h per run. Fig. 5: +5.8% / +17.1%. Table 8: 255.8 vs 820.8 (3.2×). Table 5: LCBv6 48.8 / 42.7 / 41.2. Table 3: Biology at 5 h, 56.8 vs GRPO 59.9.
   - Note: Olmo3-7B appears only in the §3 setting. The LCBv6 result (§4) is Qwen3-8B only, with LR 1e-6 and reverse KL. Olmo also loses on Biology (52.8 vs 55.8).
8. **MiniMax-M2 (2605.26494v2): confirmed.**
   - The §6.1.3 and §6.2.3 quotes match. Eq. 10 is in §7.1. Prefix-tree merging (§6.2.5) is "mathematically equivalent", with "zero approximation error" and "up to 40×".
   - MTP co-training via top-K KL is in **§6.2.6** (the claim's source list omits it).
   - The blog's numbers are in its HTML text: τ² 87 vs 64, BrowseComp 44.0 vs 31.4, SWE-Bench Verified 69.4 vs 67.2, GAIA 75.7 vs 67.9, xBench 72.0 vs 66.0.
   - Caveat: the (s_t, a_t) unit is the **RL policy-gradient** sample. The report does not say SFT is built that way, nor whether historical thinking takes loss (as Gaps already says). I could not confirm the blog's date from its HTML.
9. **LoRA Without Regret (blog dated Sep 29, 2025): confirmed.**
   - Quotes match: "consistently 10x"; fitted multiplier 9.8; "around 15x" for runs under "~100 steps"; less than 2× change between rank 4 and rank 512; "less tolerant of large batch sizes"; attention-only LoRA underperforms even at matched parameters. `W1-sft-pedagogy.md:133` does give the recipe's 1e-4.
   - The arithmetic holds: 1e-4 / 10 = 1e-5, which is 2× Arora's 5e-6. 1e-6 to 2e-6 full FT times 10-15 gives 1e-5 to 3e-5.
   - Caveat 1: the blog itself calls 15x "preliminary" and "anecdotal".
   - Caveat 2: the 10x rule maps *loss-optimal* LRs between methods, so it does not say a LoRA LR reproduces full FT's collapse or forgetting behaviour.
   - Caveat 3: Arora's own sweep included 1e-5, and its outcome is unreported.
10. **Self-replay and the verbatim-history run: partially confirmed.**
    - Confirmed: the Thinking Machines on-policy distillation post says "there is no weighting which maintains the original performance on IF-eval. This is true even if the SFT dataset contains 100% chat data". The replay was Tulu3 prompts resampled by Qwen3-8B.
    - **Correction:** "does not anchor" overstates it. The same post says a mix with at least 30% chat "helps preserve most instruction-following ability": IF-eval was 79 with replay vs 45 without, out of 85. Replay **anchors partially, not fully**. The setting was knowledge mid-training on documents, not task SFT.
    - Local half confirmed: `arms.md:538,544` show `tput-safe` 42.09 vs `control` 50.62. Across `kaggle/arms.json` on main and `runs/index.csv`, it is the only full run with `ARC3_VERBATIM_HISTORY`. `fz-tput-all` v7 was a 25-min, 10-game demo run with no ARC3 flags in its log.
    - The confound list is incomplete. tput-safe also set `ARC3_PREFIX_REUSE_DIAG` and CUDA graphs 3/5/6. `arms.md:544` attributes the loss mainly to gate11 queueing (66.6 vs 7.1 min of queue wait).

**Other errors found in the note (outside the 10 results):**
- Q1 Inferences (line 147) says RL compressed Arora's 7B AIME length by "8-27%". From Table 8 the range is 8% (α=0) to **31.5%** (α=0.4: 8,798 vs 12,837). 27% is α=0.2. SFT's 54.8% is right.
- Arora Table 2's verification counts (4.596 → 1.176) are raw counts per response on MATH500, not length-normalised. They partly reflect the shorter outputs, so they are weaker evidence of "stripping verification" than 2607.05184 Table 6, which is normalised per 1k tokens.
- Q1 Setup (line 31) says "OPSD helps at 4-8k-token budgets". The paper says OPSD models "perform comparably to or above their base" there.
