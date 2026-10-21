# W7: Training for shorter thinking at equal decision quality (efficient reasoning), especially in agents

Researcher W7, 2026-10-18. Scope: cutting reasoning length at equal decision quality, and how it applies to R3 on Flash-Next W4A16. Siblings already cover Munkhbat and Kimi k1.5 at headline level (W1 §Q6, W4 §Kimi), ShorterBetter, Fatemi and the correct-only length term (W2), and RAGEN/TTI/GiGPO (W3). This file adds paper-level numbers, the 2026 agentic papers (RACE, CogRouter, Agent-Omit, TAB, EAPO) and **one new local measurement** on our own Kaggle samples (§5).

Local baseline facts used below: 872 generated tokens per action for Flash-Next vs 1,793 for the 27B, which played half as far (`research/reports/finetune-plan-2026-10-17.md:40-41`). Mean output is 1,990 tokens per request (median 1,182, p90 4,644). The 31% of requests at or above 2,048 tokens make 70% of generated tokens (`arc-agi-v4/research-sweep.md:9`). The harness keeps the model's thinking in history, and removing it "leads to a large regression" (`WRITEUP.md:69`). `reasoning_effort=medium` doesn't shorten thinking (`arms.md:231`); `low` was never run (`arms.md:387-390`).

---

## 1. SFT-based methods: what they buy and what they cost

### Takeaway
SFT on the model's **own** shortest correct samples keeps accuracy but cuts length only modestly: about 10–13% with naive best-of-16. Conditioning the sampler (a few-shot "be concise" exemplar, or a low-effort setting), then training without that condition, about doubles the cut. SFT on **foreign** concise text costs accuracy: GPT-4o or human CoT, or LLMLingua-pruned CoT on small models. Truncation-style SFT (Self-Braking Tuning) reaches 23–63% cuts at −2.9 to +0.6 points on math.

### Cited Findings
- **Munkhbat et al., "Self-Training Elicits Concise Reasoning"** (KAIST; arXiv v3 10 Jun 2025; the page names no venue, so the "Findings ACL 2025" in W1 is UNVERIFIED by me). Table 2, averaged over 5 small models (Llama-3.2-3B, Gemma-2-2B, Qwen2.5-3B, Qwen2.5-Math-1.5B, DeepSeekMath-7B per Table 1). Each row gives relative accuracy and length reduction. [arXiv 2502.20122 Table 2](https://arxiv.org/html/2502.20122)
  - Naive BoN, N = 16: GSM8K 98.79% / −12.83%; MATH 101.74% / −10.11%.
  - FS-GPT4o-BoN (few-shot-conditioned sampling, then shortest correct): GSM8K 97.00% / −35.75%; MATH 102.56% / −23.70%.
  - SFT on external text: GPT-4o CoT (FT) 97.65% / −32.40% on GSM8K but **90.52%** on MATH; Human CoT (FT) 83.82% / 75.61%.
  - Length reduction grows log-linearly with N (§3.2, Fig. 3), so each extra halving costs exponentially more samples. Reduction is about 20–40% at MATH levels 1–2 and shrinks with difficulty (§5.2, Fig. 5). Reduction grows with model size, 1B→8B (§5.2, Fig. 6).
- **Kimi k1.5 long2short** (§2.4): shortest correct of n = 8 for SFT; DPO with that sample as the positive and wrong or ≥1.5× longer correct replies as negatives; long+short weight merging; and long2short RL (a length penalty and a smaller maximum rollout). Long2short RL had "the highest token efficiency" (§3.4; per-method counts only in Fig. 7). k1.5-short with RL: 60.8 AIME24 at about 3,272 tokens. [arXiv 2501.12599](https://arxiv.org/html/2501.12599)
- **TokenSkip** (LLMLingua-2 prunes CoT tokens by importance, then LoRA SFT with a ratio γ in the prompt; v3 Sep 2025). [arXiv 2502.12067 Tables 1–2, §4.2](https://arxiv.org/html/2502.12067)
  - GSM8K, Qwen2.5-14B: γ = 0.6 gives 313→181 tokens at −0.4 points.
  - γ = 0.5 costs −1.7 points on 14B, −5.4 on 7B, −9.3 on 3B and −8.0 on LLaMA-3.1-8B.
  - Larger models degrade less. Long-CoT models were not tested.
- **C3oT** (a GPT-4 compressor, then training on long and short CoT with a condition token): more than 50% shorter CoT at roughly unchanged accuracy. Abstract and summary level only. [arXiv 2412.11664](https://arxiv.org/abs/2412.11664), AAAI 2025. The id 2312.11562 is an unrelated survey.
- **Self-Braking Tuning (SBT)** (arXiv v4 30 Oct 2025). It detects overthinking with a ratio of steps-to-first-correct-answer and reflection-word density, then truncates or masks the redundant tail and appends a natural-language stop cue ("Wait, I've gotten the same answer multiple times, time to end the thinking"). 92k examples. Table 1 averages, as computed by the fetch tool from the table: [arXiv 2505.14604 §3, Table 1](https://arxiv.org/html/2505.14604)
  - Llama-3.1-8B, SBT-E: −62.8% tokens / −2.86 points. SBT-D: −50.0% / +0.58.
  - Qwen2.5-Math-7B, SBT-E: −30.7% / −2.65. SBT-D: −23.0% / −1.95.
  - Limitation: "premature termination may omit critical steps" on deep multi-step tasks (App. A).
- **Distilling System 2 into System 1** (Yu, Xu, Weston, Kulikov; Llama-2-70B-chat; self-consistency-filtered targets). [arXiv 2407.06023 §4.5, Tables 1–4](https://arxiv.org/html/2407.06023)
  - It works where System 2 clarifies input or judges (BSM agreement 58.4% at 4 tokens vs 49.1% at 2,118).
  - It **fails for CoT math**. GSM8K distilled accuracy is 7.35%, against 59.44% with CoT (Table 4). The authors say multi-step reasoning "requires a very different kind of reasoning."
- **Long+short model merging**: up to 55% shorter responses while "preserving or even improving" accuracy. Efficacy correlates with model scale (abstract; the direction of the correlation is not in the abstract, UNVERIFIED). [arXiv 2503.20641](https://arxiv.org/abs/2503.20641)
- **L1 appendix (negative SFT result)**: SFT on length-relabelled data **did not** teach length control (about 21K tokens whatever was requested; A.6, Table 6). Length requests in the prompt were ignored (A.5). [arXiv 2503.04697](https://arxiv.org/html/2503.04697)

### Inferences
- For an RL-tuned model, the order from safest to riskiest is: own text selected or truncated (modest cuts), then conditioned own text (larger cuts), then foreign compressed text. The risk is off-policy style, not shortness (W1's "~log2 k bits" argument).
- Thinking is not removable for multi-step search (System2→1 on GSM8K). We should compress thinking, not train a no-think policy.
- SFT teaches "be shorter" via selection, but not "obey a number" (L1 A.6). So a budget-token interface needs RL, not SFT.

### Gaps
- No SFT length-compression result on any open MoE above 30B (Kimi's weights are closed). The C3oT and merging tables were not read.

---

## 2. RL-based methods and lab length penalties

### Takeaway
RL methods cut 40–80% of tokens at roughly −3 to +3 points on math, mostly on 1.5–7B R1-distills, which are very verbose to start with. Saturated RL-tuned models lose more: QwQ-32B lost 3.2 points on average and 6.3 on AIME. Three design rules recur: penalize length only on correct samples; normalize length per prompt; never use the raw shortest sample as the target.

### Cited Findings
- **Arora & Zanette, "Training Language Models to Reason Efficiently"** (v4 Nov 2025). [arXiv 2502.04463 Eq. 4–5, Tables 8–12](https://arxiv.org/html/2502.04463)
  - Reward 1{correct}·(1 − α·σ((len − mean)/std)), with mean and std over the **correct** responses to the same prompt (PPO with RLOO).
  - R1-Distill-7B at α = 0.1: MATH500 −37% tokens / −2.2 points; AIME24 −16% / −3.3; GSM8K −66% / −1.7. At α = 0.2: GSM8K −83% / −4.8.
  - Out of domain at α = 0.2: CommonSenseQA −36% / −0.6; Logical Deduction −48% / −2.3.
  - Cost: about 100 RL steps, 3.2k prompts, 8 GH200, about 20 h. Easy problems shrink most (§5.3.1). An RLOO length bias shortened outputs even at α = 0 (App. G).
- **ShorterBetter.** [arXiv 2504.21370 §3.2, §5.1–5.3, A.5](https://arxiv.org/html/2504.21370)
  - Reward α·1{correct} − β·|len − SOL|, where SOL is the group's shortest correct length (the group mean if no sample is correct).
  - 7B, α = 2: −62.1% / +7.1 in domain; −62.3% / −0.48 out of domain. Cost: 8 A100, under 12 h.
  - The pure shortest-response target **collapses within 100 steps** (§5.3, Fig. 5).
  - What is removed: verification and self-correction fall 14.1%; output after the first correct answer falls 15–20%.
- **Fatemi et al., "Concise Reasoning via RL"** (Wand AI; v3 Nov 2025). [arXiv 2504.05185 §4–5, §7, Table 2](https://arxiv.org/html/2504.05185)
  - Under PPO and GRPO, negative rewards or advantages favour longer responses (Theorems 4.1, 5.2): verbosity is "a side effect of minimizing loss."
  - A second RL phase on only **8 occasionally-solvable MATH problems**: R1-1.5B 6,848→3,119 tokens (−54%), accuracy 57.0→58.4; R1-7B 5,249→3,149 (−40%), 71.6→72.1.
  - Correct answers are shorter than wrong ones on average (Table 1).
- **L1 / LCPO** (DeepScaleR-1.5B, 700 steps): reward 1{correct} − α·|n_gold − n_y| with α = 0.0003. It beats s1 budget forcing by 100–150% relative at 512–1,024 tokens. Out-of-domain length error is 21–41% (Table 2), and it doesn't extrapolate past the trained lengths. [arXiv 2503.04697 §3–5](https://arxiv.org/html/2503.04697)
- **ThinkPrune** (2 Apr 2025): GRPO where over-length output gets reward 0, on a 4k→3k→2k schedule. [arXiv 2504.01296 Table 1, §4.2–4.3](https://arxiv.org/html/2504.01296)
  - R1-Distill-1.5B: 10,355→3,574 tokens (−65%), +0.6 average. DeepScaleR-1.5B (already RL-tuned): −43%, −2.5.
  - **QwQ-32B: 8,763→4,494 (−49%), −3.2 average, −6.3 AIME24**, with a sharp drop past a critical length. Harder problems get smaller cuts.
  - It mostly removes hesitation and self-correction, and some self-verification.
- **AdaptThink** (19 May 2025): RL toward NoThinking under an accuracy constraint. In domain: −53.0% / +2.4 at 1.5B; −40.1% / +2.3 at 7B. The 7B picks NoThinking on 99.6% of GSM8K but 6.3% of AIME. MMLU, 7B: −31.9% / +0.2. [arXiv 2505.13417 Tables 1, 4](https://arxiv.org/html/2505.13417)
- **DAST** (EMNLP 2025 Industry): a difficulty-dependent token budget plus preference optimization gives more than 30% fewer tokens with hard-problem accuracy "preserved." Abstract level only. [arXiv 2503.04472](https://arxiv.org/abs/2503.04472)
- **O1-Pruner**: per-problem pre-sampling baseline plus RL under an accuracy constraint. The abstract gives no numbers (UNVERIFIED magnitude). [arXiv 2501.12570](https://arxiv.org/abs/2501.12570)
- **Lab length penalties**
  - Kimi k1.5: λ = 0.5 − (len − min)/(max − min) within the group, λ for correct samples and min(0, λ) for wrong ones. It is switched on **after a warm-up** because it slowed early training (§2.3.3). [arXiv 2501.12599](https://arxiv.org/html/2501.12599)
  - Qwen3's thinking-mode-fusion SFT yields budget control (answering from incomplete thinking). The budget curves were not read (UNVERIFIED). [arXiv 2505.09388](https://arxiv.org/pdf/2505.09388)
  - Flash-Next's card exposes `reasoning_effort` (xhigh, medium, low) and `preserve_thinking`, and says nothing on how post-training set its length. [HF card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

### Inferences
- Bigger cuts with less accuracy loss come from per-prompt-relative, correct-only rewards (Arora, ShorterBetter, Kimi) than from hard caps (ThinkPrune). A hard cap hurts most where the model needs length, and our stuck states are exactly that.
- Fatemi's mechanism predicts the R2/RL risk: training on states that mostly fail, as our stuck levels do (~11% right-move rate, §5), pushes length **up** unless a correct-only length term is present.
- The big numbers start from R1-distills (10k-token, highly redundant outputs). Flash-Next is already RL-tuned at about 2k tokens per request, so for RL expect ThinkPrune's "saturated" regime (about −40% at −2.5 to −3.2 points), not −80%.

### Gaps
- No length-RL result on a hybrid-attention MoE (Qwen3-Next class). No result on 50k+ context prompts.

---

## 3. Agentic specifics: overthinking, adaptive effort per turn, games

### Takeaway
In agents, "overthinking" means reasoning in place of interacting. Cutting thinking by fiat (low effort, NoThinking everywhere) hurts. The wins come from **selective** shortening: skipping reasoning on turns that earlier reasoning already covers (RACE, Agent-Omit, CogRouter), picking the less-overthinking sample (Cuadron), or a turn-level length term in game RL (MARSHAL). At equal compute, acting more beats thinking more (TTI).

### Cited Findings
- **Cuadron et al., "The Danger of Overthinking"** (12 Feb 2025; 19 models, SWE-bench Verified, OpenHands). [arXiv 2502.08235 §3.2, §5.1–5.6, §6.1](https://arxiv.org/html/2502.08235)
  - Patterns: analysis paralysis, rogue actions (interdependent actions issued without waiting for feedback), premature disengagement. Overthinking vs resolution slope −7.894 (R² 0.892) for reasoning models.
  - o1 high effort 29.1% at $1,400; **low** effort 21.0% at $400. Lowest-overthinking of 2 low-effort samples: 27.3% at $800; of 3: 30.3%, about $200 cheaper than high effort.
  - Native function calling: 29.1%→47.7%. Some text claims ("~3×", "35%") don't match the tables; use the table values.
- **RACE, "When Should Agents Think?"** (Renmin + Alibaba; Oct 2026; Qwen3.5-9B). [arXiv 2610.12061 §3, §4.3, Tables 1–3](https://arxiv.org/html/2610.12061)
  - LoGiC drops a turn's reasoning only if the log-likelihood of the current and later reference actions falls by at most ε = 0.001. Then SFT, then GRPO with skipped-turn reasoning masked and a "closure" loss.
  - Against SFT+GRPO: reasoning tokens −32.3% (ScienceWorld), −80.2% (WebShop), −46.4% (AppWorld), −62.0% (DeepSearch), with success within 1.4 points (§4.3).
  - Table 1, success / tokens: RACE ScienceWorld 73.28 / 1,403, WebShop 53.40 / 535, AppWorld 65.13 / 1,900. AdaptThink 64.49 / 1,252, 44.60 / 748, 68.72 / 2,381, so AdaptThink wins AppWorld. Qwen3.8-27B is a baseline row (ScienceWorld 38.37 / 2,011).
  - After RL, reasoning appears on 30–48% of turns in 3 of 4 benchmarks (Table 3). Without LoGiC, AppWorld tokens go 1,900→8,574 (Table 2): RL alone **inflates** length.
- **CogRouter** (13 Feb 2026; Qwen2.5-7B, Llama3.1-8B; ALFWorld, ScienceWorld). Four depth levels per step; SFT, then RL that reweights each level by the log-probability of the fixed successful action. [arXiv 2602.12662 Table 1, §4.2, §5.2](https://arxiv.org/html/2602.12662v1)
  - Qwen2.5-7B average, success / tokens: CogRouter 82.3 / 1,641; GRPO 68.3 / 4,367; GiGPO 67.5 / 3,779; AdaptThink 63.0 / 1,343; ReAct no-think 26.5 / 1,156.
  - GRPO collapses to 76.8% deepest-level thinking on ALFWorld. Reweighting failed trajectories too lowers ALFWorld success (83.0 vs 92.5).
- **Agent-Omit** (4 Feb 2026; Qwen3-4B/8B, AgentGym). Cold-start SFT (2–4k samples), then GRPO with reward 0.8·task + 0.2·tokens-saved, the saving zeroed on failure. The 8B uses 43–65% fewer tokens with Pass@1 up from a weak base (BabyAI 60.8→84.4). Omitting thoughts in the **first or final** turns hurts (§3.2); heuristic thought dropping loses accuracy (§6.2). [arXiv 2602.04284 Table 3](https://arxiv.org/html/2602.04284v1)
- **TAB** (Apr/Aug 2026): a GRPO-trained per-turn budget allocator saves up to 35% of tokens and 30% of latency at maintained accuracy. Abstract only. [arXiv 2604.05164](https://arxiv.org/abs/2604.05164)
- **TTI** (Gemma-3-12B web agents): budget forcing plateaus near 0.26 and changes the action only **15%** of the time. At equal compute, budget forcing or best-of-n gains under 3%. TTI training cuts per-step reasoning while interaction grows. [arXiv 2506.07976 §4.2, §6.1](https://arxiv.org/html/2506.07976)
- **RAGEN** (Qwen2.5-0.5B): Sokoban `<think>` length falls 307→90 tokens over 200 RL steps. NoThink is about equal on multi-turn tasks (Sokoban 20.73 vs 21.48). The "Echo Trap" collapses reasoning into templates. [arXiv 2504.20073 §4.4, Tables 3–4](https://arxiv.org/html/2504.20073)
- **"Demystifying RL in Agentic Reasoning"** (Oct 2025): "deliberative" agents (more thinking, fewer tool calls) beat "reactive" ones. RL from Qwen3-4B-Thinking drove tool calls toward **zero**. [arXiv 2510.11701 §5.1–5.2](https://arxiv.org/html/2510.11701v1)
- **MARSHAL** (Qwen3-4B games). Turn-level length reward 0.5·max(0, 1 − (l − 11)/(2048 − 11)). Removing it: length 1,700→1,954; Mini Hanabi 50.48→38.18, Simple Hanabi 29.75→20.10, Leduc 43.00→28.61; overlong-reply losses 9.6%→20.4%. Tic-Tac-Toe, Kuhn and Connect-4 were slightly **better** without it, so the gain is mostly fewer over-length forfeits. [arXiv 2510.15414 §3.4, App. H Table 13](https://arxiv.org/html/2510.15414)
- **EAPO** (Jun 2026): difficulty-aware tool-call penalties give 18–25% fewer calls at +7–10% over GRPO. Uniform penalties "may suppress useful tool-assisted exploration." [arXiv 2606.02132](https://arxiv.org/abs/2606.02132)
- **Flash-Next card**: lower effort "can also lead to insufficient analysis, more failures, and repeated retries, which may increase total latency and token consumption." `preserve_thinking` keeps all historical thinking. [HF card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

### Inferences
- Our harness keeps thinking in history (`WRITEUP.md:69`). Shorter thinking therefore also slows context growth toward the 118k trim and the prefix-cache break it causes. That is a second-order gain on top of decode time [I].
- The same retained thinking is working memory. RACE's skip criterion ("earlier reasoning still supports the action") is the principled version. Agent-Omit's first/last-turn result warns not to compress level-opening turns [I].
- TTI's 15% and our "96% of stuck decisions had an untried new-board action" point the same way. On stuck levels, extra thinking rarely changes the move, so fewer tokens per action means more probes per minute. This is Cuadron's "analysis paralysis" in our setting [I]. **Test-first:** not proven on our games.
- RL alone lengthens agent reasoning (RACE Table 2; CogRouter's GRPO). Later RL needs an explicit length or skip term.

### Gaps
- No paper trains efficient reasoning on hidden-rule interactive games at 50k+ token contexts or on a model above about 9B in agent settings. RACE (9B) and MARSHAL (4–8B) are the largest.
- No agent paper reports whether shorter thinking changes actions per episode (RACE gives no turn counts).

---

## 4. Measured trade-offs, transfer and risks

### Takeaway
Typical cuts by method, measured on models from 1.5B to 32B:
- self-selection SFT: −10–36%, roughly neutral accuracy;
- truncation SFT: −23–63% at about −3 to +0.6 points;
- per-prompt-relative RL: −16–83%, from −4.8 to +7 points;
- hard-cap RL on saturated models: −43–49% at −2.5 to −3.2 average and up to −6.3 on the hardest set;
- agent turn-skipping: −32–80% at about −1.4 points.

Out-of-domain transfer usually keeps accuracy within about 2.3 points. The main risks are lost verification and exploration on hard instances, collapse to no-think or templates, and length inflation under RL.

### Cited Findings (summary table; all values from the sources in §1–3)

| Method (paper) | Model | Token cut | Accuracy change | Source |
|---|---|---|---|---|
| Naive BoN-16 SFT | 5 models, 1.5–7B | −10 to −13% | −1.2 to +1.7 relative | 2502.20122 Table 2 |
| Conditioned BoN SFT | same | −24 to −36% | −3.0 to +2.6 relative | same |
| SBT-E / SBT-D | Llama-3.1-8B | −63% / −50% | −2.9 / +0.6 points | 2505.14604 Table 1 |
| TokenSkip γ = 0.6 | Qwen2.5-14B | −42% | −0.4 points | 2502.12067 Table 2 |
| Arora α = 0.1 | R1-7B | −16% (AIME) to −66% (GSM8K) | −1.7 to −3.3 points | 2502.04463 Tables 8–10 |
| ShorterBetter α = 2 | R1-7B | −62% | +7.1 in domain / −0.5 out of domain | 2504.21370 Table 1 |
| Fatemi phase-2 | R1-7B | −40% | +0.5 | 2504.05185 Table 2 |
| ThinkPrune | QwQ-32B | −49% | −3.2 average / −6.3 AIME | 2504.01296 Table 1 |
| AdaptThink | R1-7B | −40% | +2.3 | 2505.13417 Table 1 |
| RACE vs SFT+GRPO | Qwen3.5-9B agent | −32 to −80% | within 1.4 points | 2610.12061 §4.3 |
| Agent-Omit RL | Qwen3-8B agent | −43 to −65% | up (weak base) | 2602.04284 Table 3 |

- **Out of domain:** AdaptThink MMLU (7B) −32% / +0.2 [2505.13417 Table 4]; Arora Logical Deduction −48% / −2.3 [2502.04463 Tables 11–12]; ShorterBetter −62% / −0.5 [2504.21370]. L1's length control transfers poorly (21–41% error).
- **Hard instances gain less and pay more:** Munkhbat Fig. 5; ThinkPrune §4.2 (smaller cuts on AIME, QwQ cliff); Arora (AIME −16% vs GSM8K −66% at the same α).
- **Verification is what gets cut:** fewer self-correction and self-verification phrases (ThinkPrune §4.3); verification and self-correction down 14.1% (ShorterBetter §5.2).
- **Collapse modes:**
  - The shortest-sample target collapses [2504.21370 §5.3], and the additive L1-Max reward collapses to short, poor CoT (L1 A.7).
  - RAGEN's Echo Trap.
  - RL from a thinking model drove tool calls to zero [2510.11701 §5.2].
  - A loss-masking failure on Kaggle gave no-action loops (`W5b-kaggle-intel.md:45`).
- **Redundancy on math:** R1-distill-32B settles on its final answer after about 60% of its reasoning steps ([arXiv 2506.02536](https://arxiv.org/abs/2506.02536), EMNLP 2025, snippet level). For R1-Preview, over 80% of questions had a correct shortest sample ([arXiv 2505.00127](https://arxiv.org/pdf/2505.00127), abstract level).

### Inferences
- For Flash-Next (RL-tuned, about 2k tokens per request), a realistic first-round target is **−15 to −30% tokens per action**, not −60% [I].

### Gaps
- No study has measured whether length savings in one domain cut tokens in a different interactive domain; held-out-game transfer is untested.

---

## 5. Designing our R3: how papers handled the pitfalls, plus a local measurement

### Takeaway
Our own Kaggle samples at hard states show that "shortest = best" does **not** hold there. Over 30 frozen-state cells (700 samples), right replies are no shorter than wrong ones, and the shortest right reply is about 0.68× the median right reply. So R3 should:
- take the shortest verified reply relative to each state's own distribution;
- lean on easy and cleared-level states for compression;
- leave stuck states to R4.

### Cited Findings: how papers handled each pitfall
- **"Shortest ≠ best":** ShorterBetter uses the shortest correct length as an α-weighted soft target, never a hard label; the pure target collapsed [2504.21370 §3.2, §5.3]. Kimi's DPO takes negatives only from wrong or ≥1.5× longer correct replies [2501.12599 §2.4]. Arora normalizes length within each prompt's correct samples [2502.04463 Eq. 5].
- **Selection bias toward easy states:** Munkhbat's cut shrinks with difficulty (Fig. 5). DAST sets a budget per difficulty [2503.04472]. O1-Pruner pre-samples a baseline per problem [2501.12570]. Fatemi's phase 2 uses problems with 0 < p < 1 [2504.05185 §6–7]. AdaptThink constrains accuracy to the reference [2505.13417 §4.1].
- **Shortened thinking on hard states:** RACE keeps reasoning wherever removing it lowers the log-likelihood of later actions by more than 0.001 [2610.12061 §3]. CogRouter scores each depth by the log-probability of the fixed successful action, on successes only [2602.12662 Eq. 2–5]. SBT masks the redundant tail rather than deleting it, and adds a stop cue [2505.14604 §3.4].
- **Equivalence of the decision:** papers check the final answer. RACE and CogRouter check **action likelihood** under the shortened reasoning, a prefill-only check that is cheaper than resampling.

### Local measurement (new, 2026-10-18)
Data: state-lab samples from Flash-Next W4A16 on our Kaggle server. These are 700 non-fake samples over 30 (state, variant) cells, N = 20–30 each, at deciding or stuck states; 17 cells have at least one success. Files: `arc-agi-v4/kaggle-outputs/replays/state-lab-kaggle/{round2_*,round3_*}/{samples,grades}.jsonl`. Script: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/ce155f4d-514a-4385-ac65-a7e7e5a23e1c/scratchpad/w7/lablen.py` (run under `nice -n 15`).
- Lengths: mean 1,962, median 1,226, p90 4,613, max 12,288; 4 of 700 hit `length`. This matches the run-wide 1,990 / 1,182 / 4,644 (`research-sweep.md:9`), so the samples look representative of in-game replies.
- Success: 78/700 = 11.1% (median 1,577 tokens when successful, 1,221 when failed).
- Within the 17 mixed cells, P(a successful sample is shorter than a failed one) = **0.445** over 1,048 pairs. Success by within-cell length tercile (all 30 cells): shortest 10.4%, middle 10.0%, longest 13.2%. The successful median is shorter in only 6 of 17 cells. So there is no "short = correct" at these states, unlike math (Fatemi Table 1; Su et al.).
- Headroom: in the 12 cells with at least 2 successes, the median ratio of shortest to median successful length is **0.68** (0.63 against the cell mean). At about 1 success in 9 samples, k ≈ 20 is needed to find any verified reply.
- Caveats: hard, one-step states; most cells are prompt-fact variants, not control; some criteria are regex-based (`arms.md:313-320`). Easy, cleared-level states were **not** measured.
- Toy RSI test (Qwen3-4B; `finetune-plan-2026-10-17.md:261-263`): answering within budget rose (Connect-4 0.24→0.84, tic-tac-toe 0.57→0.88). But "correct when answered" **fell** on Connect-4 (0.43→0.29→0.25) while rising on tic-tac-toe (0.17→0.25→0.28). Shortest-verified training traded per-answer quality on the harder task.

### Inferences: a recommended R3 recipe
1. **Unit:** a frozen logged decision state, rebuilt as the state lab does. Sample k = 8 at the serving temperature on **cleared-level** states. Use k ≥ 16–20 only on hard states.
2. **Verify by effect, not by string:** the same next frame hash as the logged successful step, or the same level-progress event, with no death. Multi-action replies must pass every intermediate frame, which guards against Cuadron's "rogue actions."
3. **Select relatively:**
   - p(verified) ≥ 0.5: take the shortest verified reply (Kimi).
   - 0 < p < 0.5: take the shortest only if it is at most the median verified length; otherwise take the median (Arora/DAST-style protection).
   - p = 0: send the state to R4, not R3.
   - Optional DPO pairs: the chosen reply against a correct reply at least 1.5× longer.
4. **Stretch the cut cheaply:**
   - (a) Sample half under `reasoning_effort=low` and train under the served xhigh prompt (conditioned BoN; W1).
   - (b) **Truncation probe:** cut a verified reply's thinking at 40/60/80%, close the think block, decode only the tool call, and keep the earliest engine-equivalent cut. Basis: answer convergence near 60%; ShorterBetter's 15–20% after the first correct answer.
   - (c) Or score the logged action's log-probability under the truncated thinking, prefill only (RACE/CogRouter). Test whether abrupt cuts need SBT's stop cue.
5. **Don't compress** a level's first turn or the turn after a death (Agent-Omit's first/final-turn result). Mask windows cut by the yield or a trim (W4).
6. **Gate (pre-register):**
   - The state-lab right-move rate holds against 78/700 in a paired rerun. About 30 min of sampling (`arms.md:313-320`).
   - Tokens per verified decision fall ≥15% on held-out easy states.
   - The rate of replies with no tool call does not rise (W5b).
   - Only then a 20-min panel, then full runs at matched wall-clock.

### Gaps
- No local measurement yet of length spread at **easy** states, which is where R3's yield lives (about 1 GPU-h, test-first).
- Unknown whether ≤16k-window training changes length at 55k-token prompts.
- Unknown whether shorter thinking changes MTP acceptance (2.68).

---

## Implications for our plan

- **DO: R3 first, as SFT on our own engine-verified samples, using the §5 recipe.** Own-text selection is the one family that keeps accuracy level in every paper (Munkhbat, Kimi), and it needs no RL tooling.
  - Expect −10 to −15% tokens per action from naive shortest-of-8, and −20 to −30% with conditioned sampling plus truncation probes [I].
  - The only conversion rate to score we have is the 27B (2× tokens per action, half the levels), so the score gain stays test-first.
- **DO: select relative to each state and stratify by p(verified).** At stuck states length doesn't predict correctness (local P = 0.445). Uniform shortest-selection would teach underthinking where it costs most (ThinkPrune's QwQ: −6.3 AIME).
- **DO: verify cheaply.** Engine replay plus prefill-only action-likelihood checks (RACE/CogRouter) fit the free Kaggle quota.
- **If any RL happens (not before 11-02):** use a correct-only, per-state-normalized length term with a warm-up (Arora; Kimi §2.3.3), or MARSHAL's turn-level term. Without one, RL on mostly-failing stuck states lengthens replies (Fatemi Thm 5.2; RACE Table 2: 1,900→8,574). Fatemi's 8-problem phase is the cheapest RL precedent.
- **DON'T:**
  - Train on teacher-written, GPT-compressed or LLMLingua-pruned rationales (GPT-4o CoT: 90.5% relative accuracy on MATH; TokenSkip: −5 to −9 points on 3–8B).
  - Use the raw shortest length as an RL target.
  - Turn on global `reasoning_effort=low` or NoThinking at play time (Cuadron: 29.1→21.0%; the card's warning; AdaptThink 63.0 vs CogRouter 82.3).
  - Train no-think on multi-step decisions (System2→1 on GSM8K: 7%).
- **TEST-FIRST, in order, on Kaggle quota:**
  1. The easy-state shortest/median ratio on about 200 cleared-level states at k = 8 (about 1 GPU-h).
  2. The verified-equivalence rate under `reasoning_effort=low`.
  3. Truncation-probe yield.
  4. After the first LoRA, the §5 gate, then a panel, then full runs, reporting wins and time-to-win first.
- **Watch:** retained thinking is working memory (`WRITEUP.md:69`). If shorter replies drop facts that later turns need, expect more repeated probes on cleared levels. Shorter thinking also slows context growth toward the 118k trim [I].

---

## Fact-check (independent verifier, 2026-10-18)

How checked: each paper's arXiv HTML page (the version named in the claim) and its arXiv abstract page were fetched with curl and read locally (no browser). The local claim was recomputed with a separate script, `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/ce155f4d-514a-4385-ac65-a7e7e5a23e1c/scratchpad/fc7/scripts/verify_lablen.py`, over the same seven state-lab folders.

### Claims

1. **Munkhbat: naive BoN-16 gives −12.83% / −10.11% at 98.79% / 101.74%; FS-GPT4o-BoN gives −35.75% / −23.70%.** **Confirmed.** In Table 2 (averaged over the five main models), relative lengths are 87.17 / 89.89 and 64.25 / 76.30, and naive BoN generates 16 paths per question. Caveat: FS-GPT4o-BoN is the augmented setting (16 few-shot paths plus 16 default paths per question), so it uses twice naive BoN's sampling budget. The budget-matched row (8 + 8 paths) gives −32.85% / −19.57% at 97.44% / 101.58%.
2. **GPT-4o CoT (FT) reaches 90.52% on MATH; Human CoT reaches 75.61% (MATH) and 83.82% (GSM8K); self-generated selection stays at about 99–102%.** **Partially confirmed.** The three external-data numbers are exact. The self-generated rows span 97.00–103.02% (FS-GPT4o-BoN GSM8K 97.00, RM GSM8K 97.21, RM MATH 103.02), not 99–102%. On GSM8K, GPT-4o CoT reaches 97.65% at −32.40% length, as good as the self-trained rows, so in this paper the accuracy cost of teacher text shows only on MATH. All five models are short-CoT models of 1.5–7B, and none is RL-tuned for long thinking, so "rules out teacher text for an RL-tuned model" is an extrapolation.
3. **ShorterBetter: a pure shortest target collapses within 100 steps; −62.1% length at +7.1 points in domain on R1-Distill-7B.** **Partially confirmed.** The collapse (§5.3, Fig. 5) and −62.1% / +7.1 on DeepSeek-R1-Distill-Qwen-7B (Table 1, §5.1) are right. The +7.1 is in points: mean accuracy over AIME, Olympiad, AMC and Minerva goes 48.95 → 56.0. The −62.1% is the mean of the per-benchmark cuts. One detail is wrong: the reward is α·1{correct} − β·|ℓ − SOL| (Eq. 1) with β = 0.001, so α (2 for this model) weights correctness, not the distance. Also, the runs that collapsed targeted the shortest response *regardless of correctness*: either with no correctness term, with a correctness term but an always-shortest target, or with the shortest response used when no sample is correct. So the lesson is to target the shortest *correct* length and fall back to the group mean when no sample is correct, not "soft vs hard target". The same fix applies to §5 ("an α-weighted soft target").
4. **ThinkPrune: QwQ-32B 8,763 → 4,494 tokens, 85.6 → 82.4 average, 78.8 → 72.5 AIME24; R1-Distill-1.5B 10,355 → 3,574 tokens at +0.6.** **Confirmed** (Table 1, rows 4k→3k→2k; R1-Distill-1.5B 56.8 → 57.4). The paper's own text calls the QwQ drop "2%" (§1) and "2.7%" (§4.2). The table gives −3.2 points, which is the figure the notes use.
5. **RACE: −32.3% / −80.2% / −46.4% / −62.0% reasoning tokens against SFT+GRPO, with success within 1.4 points; without LoGiC, AppWorld goes 1,900 → 8,574.** **Confirmed.** In Table 2, SFT+GRPO uses 2,073 / 2,702 / 3,545 / 2,163 tokens against RACE's 1,403 / 535 / 1,900 / 822, and the success gaps are −1.27, +1.40, −0.17 and −0.75. v1 is dated 8 Oct 2026, and ε = 0.001 applies to the mean per-token log-likelihood. Caveat on "RL alone inflates length": the 8,574 row is "w/o LoGiC in RL", i.e. plain GRPO started from RACE-SFT. From that start, length rises on all four benchmarks (1,966→2,573, 1,232→2,670, 3,330→8,574, 1,856→3,507). For standard SFT+GRPO against the untrained base the picture is mixed: 2,073 vs 2,210, 2,702 vs 2,713, 3,545 vs 3,134 and 2,163 vs 1,281. SFT+GRPO also has the higher success rate on 3 of the 4 benchmarks.
6. **Cuadron: o1 low effort 21.0% ($400) vs high 29.1% ($1,400); the lowest-overthinking of 3 low-effort samples gives 30.3%.** **Confirmed** (§5.6). With 2 samples the result is 27.3% at $800; with 3 it is 30.3% while "still saving $200". Context: this o1 setup has no native function calling (with it, high effort reaches 47.7%), and the selection uses an LLM-judge overthinking score. The k = 3 figure appears only in the §5.6 text; Fig. 3's caption gives only k = 2.
7. **Local state-lab measurement (P = 0.445 over 1,048 pairs; terciles 10.4 / 10.0 / 13.2%; ratio 0.68 over 12 cells).** **Confirmed**, recomputed independently:
   - 700 non-fake samples, all with `completion_tokens`; 78 successes.
   - 30 cells (20 with N = 20 and 10 with N = 30); 17 have a success, and all 17 also have failures.
   - 1,048 success–fail pairs; P(success shorter) = 0.4447, with no ties.
   - Terciles: 25/240 = 10.4%, 24/240 = 10.0%, 29/220 = 13.2%.
   - 12 cells have at least 2 successes; the median ratio of shortest to median successful length is 0.677 (0.626 against the cell mean).
   - Also reproduced: mean 1,962, median 1,226.5, p90 4,613, max 12,288, 4 `length` finishes; median length 1,577 for successes vs 1,221 for failures; the success median is shorter in 6 of 17 cells.

   Caveat: the 30 cells cover only 14 distinct states (control and fact variants share a state), so they are not independent, and 16 of the 30 cells are fact variants. `arms.md:313-320` describes the criteria (pre-registered, some regex-based) in prose; the pre-registered state files are named at `arms.md:320`.
8. **MARSHAL: removing the length term raised length 1,700 → 1,954; Mini Hanabi fell 50.48 → 38.18; overlong losses rose 9.6% → 20.4%; Tic-Tac-Toe 54.65 vs 53.70.** **Partially confirmed.** The Table 13 (App. H) numbers for α = 0.5 → 0 are exact, but the 9.6% → 20.4% overlong-loss figure is for **Mini Hanabi only** (Simple Hanabi goes 14.3% → 27.3%). The ablation also trains **Tic-Tac-Toe specialists**, so every game other than Tic-Tac-Toe is a held-out evaluation, not a training game. The formula in §3 of these notes, 0.5·max(0, 1 − (l − 11)/(2048 − 11)), is right.

### Other errors in these notes

- **Munkhbat's venue (line 15).** The comments on the arXiv abstract page say "Accepted to Findings of ACL 2025". W1's venue is correct, and "the page names no venue" is wrong.
- **"Own-text selection ... keeps accuracy level in every paper" (Implications, line 225) is contradicted by a paper these notes already cite.** Arora & Zanette's "Rejection Sampling + SFT" baseline (§5.2) is R3's recipe: take the shortest correct of 8 own samples per prompt, then SFT.
  - R1-Distill-Qwen-7B: AIME24 56.3 → 18.3, MATH500 93.2 → 78.2, GSM8K 92.0 → 88.1 (Tables 8–10, "SFT" rows).
  - 1.5B: AIME24 32.0 → 24.3, MATH500 83.3 → 77.5, GSM8K 79.7 → 75.7 (Tables 3–5).
  - The authors chose the learning rate that "effectively reduces response length", so this is an aggressively tuned SFT. Even so, it is the only result in these notes for SFT on a long-thinking reasoning model's own shortest samples, which is the case closest to Flash-Next. Munkhbat (short-CoT small models) and Kimi (no accuracy table per method) don't cover that case. R3's risk estimate should include it.
- **2505.00127 (line 164).** Neither the abstract nor the body supports "over 80% for R1-Preview". The body ("Shortest Correct Response", Fig. 7) says "for over 60% of questions—across all model and dataset combinations—the shortest sampled response is already correct", and both models are 1.5B (R1-Distill-1.5B and DeepScaleR-1.5B-Preview).
- **2506.02536 (line 164).** The arXiv page lists no venue, so "EMNLP 2025" is unverified. The abstract states the 60% convergence for models in general; R1-Qwen-32B is the main model in that analysis, so the gist holds.
- **Minor.**
  - Cuadron's text claim of "35%" higher overthinking at low effort (line 98) doesn't match Table 4 (2.774 vs 2.426, about 14% higher), so the notes' advice to use the table values is right.
  - L1's 21–41% out-of-domain length error (line 66) is for L1-Exact. After 500 more RL steps, L1-Exact+ brings it down to 11–24% (App. A.1).
