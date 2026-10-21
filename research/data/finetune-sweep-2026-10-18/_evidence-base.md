# Evidence base: training Qwen3.8-Flash-Next for ARC-AGI-3 (2026-10-18)

This consolidates the 23 notes in this folder (L1–L3, W1–W10, W5b, `_critic.md`, G-A to G-H). Where a note's fact-check corrected or refuted a claim, the corrected value is used and refuted claims are dropped.

**Sources.** Each fact names its note (file prefix, e.g. G-A = `G-A-token-tail-stalls.md`) and the original source (URL or repo file:line). "FC" is the note's fact-check. Repo paths are relative to `daniel-franzen-sol/`. [I] marks my synthesis across notes.

**Five facts that frame the plan.**
- A consistent LB 60 means a draw mean of about 64.5 (draw sd 3.50); ours is about 31 (L3 FC 7).
- No public evidence shows fine-tuning Flash-Next helps on ARC-AGI-3 (L2 FC 6; W5b; G-D).
- The measured lever, fewer tokens per action, is worth about +2 to +8 LB at realistic cuts. Efficiency alone would need about 2.7× effective compute to reach 60 (G-H Q4).
- Deployment is cheap: a LoRA on BF16 non-expert modules merges into our W4A16 checkpoint with no re-quantization (W6 FC 1; G-D).
- Diversity, not volume, binds: 151 distinct cleared levels, all on public games (L1).

## 1. Our situation in numbers

### 1.1 The logged corpus

| Item | Value | Source |
|---|---|---|
| Usable runs | 85 Franzen + Flash-Next runs with request logs (65 `v4-*`, 20 `fz-*`); 1,642 game-runs; all on the 25 public games | L1 Q1 |
| Requests / responses | 142,572 / 141,483; finish tool_calls 139,883, length 1,514, stop 86 (1.13% no tool call) | L1 Q1; G-B Q6 |
| Tokens | 269.6M generated (87% reasoning); 8.04B prompt (90.5% cached, 12% image) | L1 Q1 (`reqstats.json`) |
| Prompt length per call | Median 57,313, p90 84,800, max 120,756; ≤8k 2.2%, ≤16k 6.8%, ≤32k 16.5%, ≤64k 62.8% | L1 Q3, FC 3 |
| Actions, clears, wins | 289,520 actions; 6,034 clears (events; `benchmark.json` 5,997); 354 wins (bench; events 357) from 14 games | L1 Q1, FC 5 |
| Distinct cleared levels | 151 of 183, about 40 clears each | L1 Q1 |
| Special categories | 191 clears costing ≥100k tokens; 240 stall-then-clear breakthroughs on 191 levels; 874 deaths; 123 levels with both a stuck and a cleared attempt | L1 Q4 |

**What the logs lack or guarantee.**
- **No token ids or logprobs** anywhere in `ARC3-Inference/inference`. Response rows omit the reply (`tool_agent.py:9673-9687`); it is recoverable from the next request in 98.5% of transitions, the rest from event transcripts (L1 Q2, FC 1).
- **Leader history is normalised** (`ARC3_VERBATIM_HISTORY` off): 0 of 468 history reasoning messages have a blank line, against 486 of 498 in verbatim `v4-tput-safe`. No raw reply is stored anywhere (L1 FC 2; G-C FC 8; `tool_agent.py:4856-4875`).
- **Render parity is exact.** Emulating SGLang preprocessing reproduced `prompt_tokens` and `image_tokens` on 200 of 200 requests (150 more in FC). The only mismatch source is the tool-JSON dump (−11 tokens). A 640×640 frame is 400 tokens (G-C Q1, FC 1–3; `serving_chat.py:1156`).

**External game pools**

| Pool | Content | Licence | Status | Source |
|---|---|---|---|---|
| DreamTeam generated | 25 games × 8 levels; all embed `solve_level`, 20 of 25 match their baselines. Baselines are solver lengths: median 145 actions per game vs 638 public | Apache-2.0 | Never run in our harness | L1 FC 8; W9 FC 6 (`al7306.py:819`) |
| arc-interactive | 249 games in 150 families; 192 list 5 levels; about 8 have ≥6 levels and a grid ≥16; placeholder baselines; ships other versions of ft09, vc33 and ls20 | MIT | Training pool after an id-prefix contamination filter | W9 Q1, FC 7; G-B Q3 |
| sonpham / Training-Data | ~62 Claude-written games / 429 games incl. augmented public ones | None | Validation only | W9 Q1; L2 §3 |
| DreamTeam `game_creator` | Generator, 30-min timeout per game; only ever run on GPT-5.5, Gemini or Opus | Apache-2.0 | Yield unknown | L2 FC 7; W9 (`config.yaml:7`) |

### 1.2 Token budget and throughput
- **A full run**: 25 games in 121 min, 5.03–5.26M generated tokens, about 690–720 tok/s, about 2.3 Kaggle GPU-h including load (L1 Q6; G-H FC 11). A leader game-run uses a median 224k tokens (L1 Q5).
- **Kaggle rates:** 2.5–2.9M tokens per GPU-h; about 11–12 game-runs per GPU-h (G-B FC: 10.9); 730–960 state-lab frozen-decision samples per GPU-h all-in (1.2–1.6k counting only sampling minutes); server boot 487–523 s (L1 Q6, FC 6; `runs/kaggle/v4-lab-round{2,3}/v1/lab_summary.json`).
- **Leader serving:** 146.0M input tokens (89.1% cached), 5.29M output; MTP accept 2.684; harness 688 tok/s (G-E Q3, FC 4; `v4-mem-repair-10x/v1/serve.log.xz`).
- **Prefill stalls** take about 20.6% of wall; the system prompt is 4,134 tokens (W10 FC 7).

### 1.3 Where the tokens go
- **Actions and turns.** 80% of actions carry 0 tokens (action batches). A deciding turn has mean 4,411 tokens, median 2,234 (L3 Q2). Turns ≥5k tokens hold 69.3% of all tokens (leader 71.9%). Clipping at 4k / 6k / 8k would remove 48 / 35 / 26% (G-A Q1, FC 2; L3 FC 8).
- **By level class** (all 85 runs; G-A Q1 table, `scratchpad/ga/ana.json`):

  | Class | Token share |
  |---|---|
  | Cleared, never stalled, on-route | 39.4% |
  | Cleared, never stalled, off-route | 21.8% |
  | Stalled, then cleared | 4.4% |
  | Not cleared here, cleared in another run (Nx) | 32.4% |
  | Never cleared in any run (N0) | 1.9% |

- **What R3 can reach.** Cleared levels hold 65.7% (leader 72.3%); on-route, verifiable states about 41% (leader about 45%); cheap (<50k-token) cleared levels 25.4%. Meanwhile 29.4% of all tokens are ≥5k turns on stalled or never-cleared levels, and never-cleared levels hold 44.5% of 12k+ turn tokens (G-A Q1, FC 1–2).
- **Long turns are call loops, not long thoughts.** A 12k+ turn averages 7.67 calls, with 85.5% of its tokens in calls that didn't act. `finish=length` calls are 6.8% of all tokens (G-A Q1, FC 3).
- **Never-cleared levels take 22–28% of tokens in leader runs** (32% overall). The older "50–60%" is stalled levels (≥50k tokens plus never-cleared ≈ 60–66%), traced to `research/reports/rsi-proposals-2026-10-15.md:30`. Every never-cleared attempt is the level the clock ran out on: censored, not walls (L1 correction 1; G-A FC 9).
- **Sandbox errors** are 8.9% of calls and 9.2% of tokens; the call after an error costs 1.3%. About 25–45% of errors are made by the harness: StaleStateActionError 14.8%, ImportError 8.1% (`time`, `inspect`, `sys`…), TerminalStateActionError 1.5%, part of NameError's 23.3%. That is about 2–4% of tokens fixable without training (issues #18, #176) (G-A Q4, FC 7–8).
- [I] **A realistic R3 cut is worth less than its headline.** A −25% cut on on-route cleared states is about −10% of total tokens. At the measured elasticity (§1.6) that is about +1.3 to +2.4 offline, +0.7 to +1.3 LB under P1 and +1.3 to +2.5 under P3, unless the model also shortens stalled-level turns (unmeasured spill-over).

### 1.4 What stalls are made of
- **Novelty stalls are rare:** ≥15 actions with no new board occur in 302 of 7,204 level attempts and cost 0.6% of tokens (G-A Q2, FC 4).
- **Exploration is not the deficit.** New (state, action) pairs are 96.0% of actions on cleared levels vs 89.1% on Nx, nearly equal at matched action index; Nx's new pairs reach new boards 5–13 points less often (G-A Q2, FC). The 21,901-decision calibration gives 91% new pairs, 87% of them new boards, and on stuck levels 96% of decisions had an untried new-board action (`arms.md:183-185`; W8 FC 1).
- **Breakthrough actions** are a never-used action type only 7.5% of the time and on the hindsight route about 68% (G-A Q2, FC 5).
- **A 30-stall audit** (one coder, counts ±3, labels not reproducible): 8 were planned revisits (reset-and-replay, deliberate bar burns). Of 22 real stalls, 12 were mechanic errors, 5 goal inference, 4 execution and 1 perception, via the model's own parser (G-A Q3, FC 6; `scratchpad/ga/audit.txt`).
- **The forum agrees:** in 83% of Makarov's forks the stuck run had tried the decisive action and drew the wrong conclusion ([743723](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743723); G-F FC 6).

### 1.5 Score mechanics and headroom
- **Scoring.** Level score = min(115, 100·(human/AI actions)²); game score = the level-index-weighted mean, capped at the weighted share cleared; total = mean over games. RESET counts as an action, and agents are cut off at 5× human actions (L3 Q1, FC 1; `arc_agi/scorecard.py:170-206, 701-704`; [docs](https://docs.arcprize.org/methodology); W5 Q2).
- **Depth dominates.** Clearing k of n levels at human efficiency scores 100·k(k+1)/(n(n+1)). Our cleared levels already take 0.64× human actions (median), and 82.5% score ≥100; human efficiency everywhere would lift only 51.17 → 53.06 (L3 Q1, FC 2).
- **Stuck levels are mostly soft.** Clearing the final stuck level adds +3.2 (at L1) to +24.7 (at L6) game points, +10.4 per non-won game on average. Of 451 final stuck levels across 26 runs, only 45 (10%, 11 walls) were never cleared anywhere. P(later clear | ≥60k / ≥85k / ≥125k tokens spent) = 58 / 47 / 27%, lower bounds censored by the deadline (L3 Q2, FC).
- **At the deadline** 15.6 of a run's 17.3 live games sit on a level some other run cleared (G-H Q2, FC 10).
- **Counterfactuals** (26 runs, mean 51.17): +1 level per non-won game → 61.6; every game at its best-ever depth → 75.4. LB 31 ≈ −1.4 levels per game, so +1 level per hidden game ≈ +13–15 LB (L3 Q2).

### 1.6 Compute elasticity
- **Downward is measured.** Eight uniform slowdowns (fewer experts, a slower engine, lost compute) land on the reference curve at their own token count: residual −0.01 ± 1.68. All 10 downward runs, including stream and gating changes, give −1.60 ± 3.67. So the curve holds for uniform slowdowns only (G-H Q1, FC 4).
- **Upward is untested.** No run has exceeded 121 min. All 4 runs that bought tokens did it by changing streams, window or sampling, and all fell below the curve (−4.3 to −23.2) (G-H Q1, FC 5; `_critic.md` spot check 2).
- **The reference curve** (23 runs): S(60 / 80 / 100 / 121 min) = 30.5 / 39.7 / 45.7 / 51.6; late slope 0.25–0.29 points per minute; S(121) − S(110) = 2.77 (se 0.43) (G-H Q2, FC 1–2).
- **Elasticity: +2.7 offline per +10% effective compute** (95% CI 1.9–3.5), about 28 per ln unit (G-H Q4, FC 8).
- **The shape above 121 min is unsettled.** A finer-grid saturating exponential fits better than the log (SSE 2.90 vs 4.02); at 161 min they predict 58.6 vs 60.2, at 242 min 65.3 vs 72.4 (G-H FC 3).
- **sim2 is too steep:** 15–20% on the time axis, 25–28% on the token axis. It over-predicted every upward lever and the 27B (24.3 vs a real 14.51) (G-H Q3, FC 6; L3 FC 6).
- **Value of a tokens-per-action cut at equal decisions** (G-H Q4, FC 9; L3's sim2 figures run 17–28% high):

  | Cut | Offline [range] | LB, P1 | LB, P3 |
  |---|---|---|---|
  | −15% | +3.6 [1.9–4.7] | +2.0 | +3.7 |
  | −25% | +6.3 [3.3–7.6] | +3.5 | +6.5 |
  | −30% | +7.8 [4.0–8.8] | +4.3 | +8.0 |

- **Throughput loss is costed exactly:** −10% effective throughput costs about −2.7 offline and −1.5 to −2.7 LB (G-H Q4).
- **The cheapest upward test:** the leader at `demo_minutes` 242 (about 4.3 GPU-h) or 161 (about 3.0). R3's compute premise is falsified if S(151) − S(121) < +2.5 or S(161) − S(121) < +3.5; the false-falsification rate is about 13% (G-H Q5, FC 11).

### 1.7 Noise, LB mapping and the hidden set
- **Offline noise.** Run sd 3.15 (within replicates, df 4) to 3.28 (across configs). A 10-game subset has sd 5.8, so a paired difference has sd 8.2 (L3 Q4; G-B Q1, FC 1).
- **LB noise.** Per-draw sd is 3.50 (10x-family repeats, df 7); the earlier 3.04 pooled in taaf draws. The 10x family's 19 draws average 29.12; mem-repair-10x drew 38.22 / 28.73 / 29.54 / 26.90. +5 LB needs about 6 draws per arm, +8 about 1.5–2.4, +10 about 1 (L3 FC 7; G-B Q4, FC 4; `runs/submissions.csv`).
- **LB/offline = 0.548** (sd 0.051, n = 12); the old taaf harness ran at 0.30. Across harness generations r = 0.98, but within the 10x family r = 0.44 (L3 Q2, FC 3).
- **Two bracketing models.** P1, the floor: LB gain = 0.548 × offline gain. P3, the ceiling: hidden games ≈ public games at about 0.6× compute; it fits the 0.30 → 0.55 rise better (L3 Q2). One LB draw at a halved private deadline would separate them: P1 predicts −11.4, P3 −17.6 to −18.8, about 1.6–1.9 draw-sd apart (G-H Q5, FC 12).
- **Screens can't rank arms.** Within the 10x family a single run has reliability 0.08; across 48 runs no log metric predicted score in the direction a token-cut screen assumes; minute-25 panels correlate 0.21 with final rank. Never pick finals by public score: its split-half correlation is −0.45 to −0.48 (G-B Q4–5, FC 8–9; L3 Q4; `research/archive/reports/next-moves.md:195`).
- **The hidden set is 110 private games,** half on the public LB and half on the private LB ([Kaggle data page](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/data); G-F G18, FC 1).
- **Best-of-N inflation.** At sd 3.5 the best of 166 draws averages about mean + 9.4, so Tufa's typical draw could be about 59.5 if its noise matches ours. That is an upper bound, since the submissions are different systems (L2 §5; `_critic.md` C10).

### 1.8 What LB 60 requires
- **Efficiency alone** needs about 2.7× effective compute under P3, if the log-linear curve holds that far (sim2 saturates first, near 63), and about 6× under P1 (G-H Q4; L3 Q3).
- **Realistic 15-day levers:** R3 at −15 to −30% gives +2 to +8 LB (§1.6), less if it only touches cleared states (§1.3). R1 (MTP) gives about +1 to +2 LB (§4.3).
- **So LB 60 needs new capability:** clearing levels no current run clears, on out-of-distribution mechanics. The ~63 "ceiling" is our agent's under efficiency levers, not the benchmark's; Tufa and Chen already exceed it (L3 Q3; `_critic.md` C19).

## 2. Training evidence that applies to us

Strength: **Strong** = replicated in several primary sources or read in full text; **Moderate** = one solid primary source; **Weak** = abstract-level, single run or self-reported. **Model-size caveat:** apart from lab reports (Qwen3, ScaleRL), everything below is on models ≤35B total (most ≤14B), contexts ≤38k, and no ARC-AGI-3-like games. None is on a 125B hybrid GDN/QSA MoE.

### 2.1 What to train on

| Finding | Evidence | Strength |
|---|---|---|
| On-policy data forgets least; filtered one-shot self-SFT still forgets; Iterative-SFT (regenerate each round) shows mild or no forgetting | Retaining by Doing [2510.18874](https://arxiv.org/abs/2510.18874) Table 2: Qwen2.5-1.5B non-target drop, Self-SFT 3.0 / 14.0 / 19.5 vs GRPO 0.6 / 0.5 / 0.9; credit goes to on-policyness, not KL (W1; G-G FC 4) | Strong; 1–8B, single-turn |
| Lab precedent: Qwen3 Stage 3 did continual SFT on its own rejection samples, and still lost AIME'24 83.8 → 81.9 | [2505.09388 §4.3](https://arxiv.org/html/2505.09388) (W1 FC 1) | Strong |
| Off-policy expert SFT transfers weakly or harms; RL or self-filtered data transfers | Orak (held-out games barely move); SPIRAL SFT flat at 2× data; ViGaL Snake SFT 49.1 → 47.2 vs RL 62.3; Chu V-IRL OOD 80.8 → 1.3 under SFT (W3 §1, §5) | Strong pattern. Chu's "SFT first" holds only for base models (W3 FC 2) |
| Curation beats volume | LIMO's 800 samples, while uncurated 100k degraded; Kaggle 739047: "more trajectories … made it worse" (W1 FC; W5b) | Moderate |
| 10 training environments degrade; 100 give small gains | AWM [2602.10090](https://arxiv.org/abs/2602.10090) Fig. 4, Qwen3-4B only: about +2–3 BFCL and 0 MCP at 100–191 environments (G-G FC 6; W9 FC 1) | Moderate |
| Coverage-selected diversity beats count | [2608.03571](https://arxiv.org/html/2608.03571v1) Table 3 (4B): 30 selected environments +40.3% OOD vs all 170 at +7.5%; +2.8% without conflict control (W3 FC 6) | Moderate; small absolutes |
| Our frontier is tiny | 38 cells with a 0.2–0.8 clear rate (deduplicated Flash-Next runs), about 23 after a 10-game holdout (W3 FC 8) | Measured |
| Early-experience auxiliaries beat imitation by +2.3 to +18.4 | [2510.08558](https://arxiv.org/html/2510.08558) (W1; W8) | Moderate |
| Standalone world-model SFT doesn't improve decisions | CWM [2510.02387](https://arxiv.org/html/2510.02387v1) Table 4: SWE-bench 18.6 → 18.4 with traces vs 22.1 with agentic data. RWML [2602.05842](https://arxiv.org/html/2602.05842v2): world-model SFT alone collapsed ALFWorld to 2.8; WM-SFT + RL ≈ RL on ALFWorld, +2.3 on τ² (W8 FC 2–3) | Moderate; off-format targets |
| Hindsight relabelling beats success-only SFT by +7.6–11.4% | AgentHER [2603.21357](https://arxiv.org/html/2603.21357v4) (W8 Q4) | Moderate; explicit goals |
| Cross-episode reward transfers | ORBIT [2602.04089](https://arxiv.org/html/2602.04089v1), Qwen3-8B, 100 GRPO steps: Mastermind 0.27 → 0.59, where single-episode RL fell to 0.21 (W8 FC 4) | Weak; one run |
| Self-generated multi-task SFT + RPO transfers to 9 of 10 held-out groups with no general regression | PAPRIKA [2502.17543](https://arxiv.org/html/2502.17543v4): SFT LR 1e-6, RPO 2e-7, one pair per task (W8 FC 5) | Moderate; 8–12B non-reasoning |

[I] For us, train on the model's own engine-verified decisions at frozen states (R3) and on mechanic-discovery breakthroughs from external games (R4). Not on teacher text, scripted-solver rationales, world-model SFT or novelty rewards.

### 2.2 SFT mechanics and hyperparameters
- **LoRA targets.** Attention-only LoRA underperforms MLP LoRA: shown at matched parameters on Llama-3.1-8B, and also on Qwen3-30B-A3B-Base and in MATH RL ([LoRA Without Regret](https://thinkingmachines.ai/blog/lora/); W1 FC 4). Moderate. Our trunk's only BF16 MLP is the shared expert, 1 of 11 active FFNs per token (about 4.9M params per layer). Shared-expert-only LoRA is untested anywhere (W1 Q3; W2 FC).
- **Learning rate.** The best LoRA LR is about 10× the full-FT LR (fit 9.8; about 15× under 100 steps, "preliminary"), nearly rank-independent; LoRA tolerates large batches less well (same source; G-G FC 9). Full-FT anchors: Arora's collapse 5e-6 (7B); OPSD 2e-6 (8B); PAPRIKA SFT 1e-6 and RPO 2e-7; AWM 7e-7. That gives **LoRA SFT 1e-5 to 3e-5, RPO about 2e-6, global batch ≤32**; the ms-swift recipe's 1e-4 is a sweep ceiling (G-G (d)).
- **Weight interpolation.** WiSE-FT restores pass@k lost to SFT diversity collapse ([2504.10478](https://arxiv.org/abs/2504.10478)). The AIMO-2 merge at α = 0.7 scored 69.1 vs 66.8 and was shorter than the fine-tuned end, though longer than the start ([2504.16891](https://arxiv.org/html/2504.16891) Table 10; W4 FC 6). A λ-sweep of a LoRA costs only evaluation. Moderate.
- **Forgetting control.** KL isn't what protects (GRPO at β = 0 ≈ β = 0.05). Self-replay anchors only partly: a 70/30 mix kept IF-eval at 79 of 85 (45 without), no mix kept 85, and on-policy distillation from the original model restored 83 ([TM on-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/); W4 FC 3; G-G FC 10). Moderate.
- **Game training costs general skill.** SIMA 2 lost AIME −15 to −25% and GPQA −16 to −20% relative despite a pretraining-data mix ([2512.04797](https://arxiv.org/html/2512.04797v1); W3 FC 4). Moderate.
- **Context shape.** AWM's history-limited model fell on τ² from 22.57 to 9.35 when evaluated with full history (G-G Q1, FC 6). GLM-4.5 found shorter multi-stage RL made the model "unlearn" long context irreversibly (W4 §4). Tufa trains on shorter sequences on purpose (L2 FC 7). [I] Train on the served post-trim context (31–85k) where memory allows, and gate at real ~57k contexts.

### 2.3 Shorter thinking (R3): the main lever, rated medium risk
- **Self-selection cuts length modestly.** Shortest-of-16 own samples: −10 to −13% length at about equal accuracy. Few-shot-conditioned sampling: −24 to −36% (−33 / −20% when matched for sampling budget) ([2502.20122](https://arxiv.org/html/2502.20122) Table 2; W7 FC 1). Moderate; 1.5–7B short-CoT models.
- **The closest analogue collapsed.** Arora's RS+SFT (shortest correct of 8, R1-Distill-7B, full FT at 5e-6) took AIME24 from 0.563 to 0.183 at 5.8k tokens, worse than truncating the base at 8k (0.390). Normalised over benchmarks it kept 0.822 accuracy at 0.597 of the tokens; DPO kept 0.960 at 0.737 and RL 0.959 at 0.516 ([2502.04463v4](https://arxiv.org/abs/2502.04463) Tables 8–13; G-G Q1, FC 1). Strong (full text).
- **Local measurement at hard frozen states** (700 samples, 30 cells, 14 states): right replies are not shorter than wrong ones (P(success shorter) = 0.445; success by length tercile 10.4 / 10.0 / 13.2%). The shortest verified reply is 0.68× the median verified one. Success is 11.1%, so k ≈ 20 samples per state. Easy states are unmeasured (W7 §5, FC 7; `kaggle-outputs/replays/state-lab-kaggle/`).
- **RL length methods** cut 40–80% on verbose 1.5–7B distills. On a saturated RL-tuned model they cost more: ThinkPrune took QwQ-32B −49% tokens at −3.2 average and −6.3 AIME. Targeting the shortest response regardless of correctness collapses (ShorterBetter), and negative advantages push replies longer (Fatemi) (W7 §2, FC 3–4). Moderate.
- **Agent turn-skipping.** RACE (Qwen3.5-9B) cut reasoning tokens 32–80% within 1.4 points. Plain GRPO from RACE-SFT lengthened replies; against the untrained base the picture is mixed ([2610.12061](https://arxiv.org/html/2610.12061); W7 FC 5). Moderate.
- **General "think less" doesn't obviously transfer to agents.** Swift 1.5's mean output rose 11.9% on Terminal-Bench; o1 at low effort scored 21.0 vs 29.1 on SWE-bench (L2 FC 5; W7 FC 6). Moderate.
- **A collapse to watch.** Sentence-level down-weighting of reasoning (92% of it at weight 0.1) gave 1,474 of 4,767 turns with no tool call, against 132 of 3,668. Single run, 7-h budget only, base model unnamed ([discussion 743319](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743319); W5b FC 3). Weak.
- **The synthesised recipe** (G-G (a); its thresholds are proposals):
  - whole-turn targets: every call up to the acting call, scored on total turn tokens and calls (G-A Implication 2);
  - per-state selection capped at about 0.7× the median verified length;
  - 30–50% median-length verified siblings as a length anchor;
  - one epoch per round, then regenerate (multi-epoch on a fixed self set drifts off-policy; W1 Q5); 2–3 rounds; LoRA LR 1e-5 to 3e-5; a λ-sweep;
  - no compression of a level's first turn or the turn after a death (Agent-Omit, W7 §3).

  Gates: right-move rate; verification and backtracking markers per 1k tokens; distinct actions across k; no-tool-call rate; tokens and calls per decision.

### 2.4 RL fundamentals for this MoE (not before 11-02)
- **Routing replay is required whichever loss is used.** On Qwen3-30B-A3B, GRPO crashed in all 3 settings, while all 6 runs with rollout routing replay stayed stable. GSPO without replay also collapsed in three sources (R3 paper App. B, SAPO §5.1, ScaleRL A.17) ([2510.11370](https://arxiv.org/html/2510.11370v2); W2 FC 1–2). Strong.
- **Set the aux load-balancing loss to 0 in RL** (NeMo AutoModel note; the R3 paper uses none). The ms-swift Flash-Next recipe still sets 1e-3 (W2 FC 4). Moderate.
- **Train–inference mismatch.** INT4 PTQ rollouts with a BF16 trainer drift; NVFP4 rollouts collapse after about 150 steps (QUADS); LMSYS INT4 QAT "almost overlaps" BF16. An FP32 LM head helps (MiniMax correlation 0.9 → 0.99; ScaleRL asymptote 0.52 → 0.61). A loose bar is NeMo's FP8 35B gate, KL ≤ about 0.004 (W2 §3, FC 5). Moderate.
- [I] **Our variant is untested anywhere:** LoRA on BF16 modules over frozen dequantised int4 experts makes the trainer's weights equal the served weights; kernels and the FP8 KV cache remain (W2 §3; W4 §3).
- **Cheaper alternatives.** RAFT is competitive early, but vanilla RAFT reached 52.3 vs GRPO's 56.3 later, and it was online and iterative. Online or semi-online DPO ≈ GRPO; offline methods fall short. On-policy distillation is about 10× cheaper than RL: Qwen3-8B reached 74.4 AIME in 1,800 GPU-h vs RL's 67.6 in 17,920 (W2 FC 8; W4 FC 2). Moderate.
- **Budget mismatch.** Published MoE RL costs 10k–100k+ GPU-h; our 450–950 GPU-h buys a few hundred steps at most (W2 §5, FC).
- **The usable unit is a frozen decision state with 0 < p < 1;** full games have near-zero group variance. Select states by reward variance: in RAGEN-2 the highest-variance quartile gave 21.1% vs 11.0% for the lowest, and length filters failed to prevent collapse (W2 §5; G-G FC 5). Moderate.
- **Rewards.** "Frame changed" holds for 94.5% of actions, so it is useless; use level clears, verified routes or progress over earlier attempts, with length terms on correct samples only (W2 §6; W8 Q5).

### 2.5 Agentic and game RL: what transfers
- **Near transfer is large, far transfer small.** VHD-Play trained Qwen3.6-35B-A3B for 34 GRPO steps (34,816 rollouts on 2,176 environments): held-out +0.56, near-OOD +0.63, far-OOD +0.16 to +0.24, policy shift about 0.09 nats per token ([2609.27321](https://arxiv.org/html/2609.27321v1); W3 FC 1). Moderate; the closest hidden-dynamics analogue.
- **Gains shrink on models that already reason:** SPIRAL +10.5 on a base model vs +1.4 on R1-Distill (W3 §1).
- **Lab game RL:** UI-TARS-2 went SFT 44.27 → RL 59.77 on its 15-game suite, but scored Sokoban 0.3 on the out-of-domain LMGame-Bench (W4 FC). Successful runs used 2–5×10⁴ short episodes at 2–32k contexts (W3 §5).
- **On ARC-AGI-3 itself,** no paper or entry shows a fine-tuned LLM gaining on held-out games; non-LLM cross-game pretraining gave no consistent benefit; every near-100 public-set result is a training-free frontier harness (W3 §4; W5 Q4, FC 5).

### 2.6 Distillation
- **No usable teacher.** Anthropic (Commercial Terms D.4), Gemini and OpenAI terms bar training competing models on outputs, and the reasoning is hidden. Qwen3.8-2.4T-A95B shares our tokenizer and has an MIT-style licence ($2 / $6 per M on OpenRouter), but it is **text-only** and can't read our image contexts (W10 Q1–Q2, FC). Strong.
- **Privileged self-distillation hurts thinking models.** OPSD lowered all 5 thinking models tested (Qwen3-8B 0.635 → 0.578 avg@16). Trained with thinking off, the model held (0.590 vs 0.532). Vanilla on-policy distillation from a larger teacher helped (0.372 → 0.392) ([2607.05184](https://arxiv.org/abs/2607.05184); G-G Q1, FC 2). Strong (full text); 4k training cap, ≤8B.
- **SDPO's wins were all thinking-off**, at ≤8B and ≤8k responses; its 3.2× shorter outputs are relative to GRPO, not the starting model ([2601.20802](https://arxiv.org/abs/2601.20802); G-G FC 7; W10 FC 3). Moderate.
- [I] **Demote SDPO and OPSD.** If one is tried, give the teacher sparse outcome text only (never sibling solutions or the student's attempt), and gate on fork markers.
- **Skip two ideas.** Distilling the system prompt saves at most about 4k cached tokens, under 2% of wall (W10 Q3). Test-time training is infeasible on the card: the weights take 69.85 GB of 96 GB, and aTTT's "up to +5" ran on dedicated training GPUs (W5 Q5, FC 7; W10 Q5).

## 3. What labs and competitors do

### 3.1 Lab pipelines
- **Qwen3:** cold start → reasoning RL (3,995 queries; AIME 70.1 → 85.1 in 170 steps) → fusion SFT on its own rejection samples → general RL. The general stages cost Qwen3-32B AIME'24 83.8 → 81.4 ([2505.09388](https://arxiv.org/html/2505.09388); W4 §1, FC).
- **Flash-Next:** the tech report covers architecture and pre-training only, with no post-training section. The Base model is evaluated but not released. The card says "MTP: 1 layer, trained with multi-steps"; training used Muon plus AdamW. So our job is continued post-training of an RL-tuned model (W4 FC 1; [tech report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)).
- **DeepSeek:** V3.2 used one mixed RL stage with Keep Routing and off-policy sequence masking, on 1,827 synthetic agent environments kept only if pass@100 > 0. V4 replaced RL with on-policy distillation from more than ten teachers and applied FP4 QAT to the experts (W4 §2; [2512.02556](https://arxiv.org/html/2512.02556v1), [2606.19348](https://arxiv.org/html/2606.19348)).
- **Kimi:** k1.5 long2short combines shortest-of-8 SFT, length-contrast DPO, a long/short weight merge and length-penalty RL; K2-Thinking used INT4 QAT on its MoE components (W4 FC 4, 7).
- **GLM-4.5 and GLM-5:** single-stage 64K RL to avoid long-context "unlearning", then cross-stage on-policy distillation, with a TITO gateway so training sees the exact inference token ids (W4 §4).
- **MiniMax:** M1's CISPO RL cost $534,700. M2 trains on (s_t, a_t) pairs as served and retains thinking (τ² 87 retained vs 64 discarded); its MTP heads are co-trained via top-K KL ([2605.26494](https://arxiv.org/abs/2605.26494); G-G Q1, FC 8).
- **Small-team wins came on weak bases:** DeepSWE (RL only, 200 steps, 23 → 42%); SWE-Gym (491 trajectories, +12–14 points) (W4 §6, FC).
- [I] **What transfers to us:** train against the deployed weights; on-policy rejection sampling; checkpoint interpolation; treat evaluation as the bottleneck, as Qwen says of itself.

### 3.2 Tufa Labs (#1, 68.91)
- **What they said** (MLST, 2026-07-01, [transcript](https://app.rescript.info/api/public/sessions/23f8cebdc5ce8f5f/pdf)): the per-level ARC score at level transitions is "the most naive implementation"; their rewards included level rewards, ARC-AGI scores, whether code executes and reasoning-step length; "we can for example train on" self-generated games; and because training at 100–200k tokens is "extremely difficult", they train on shorter sequences and generalise to longer ones. This describes a research pipeline, not a submitted RL model (L2 FC 1, 7).
- **Compute and model.** 48 B200, with 144 B300 still "expected" ([tufalabs.ai](https://tufalabs.ai/); L2 FC 2). The model is unknown: M1 was stock Qwen3.6-27B FP8. driessmit1 uploaded a Qwen3.8-27B-FP8 snapshot on 08-18, and the M1 code holds a dead vLLM LoRA guard. They jumped 56.52 → 68.91 on 10-13 and have 166 submissions (L2 §1, §5, FC).
- **Release only after the competition ends** ("best of 4000 copies"). Forum claims that they fine-tune are speculation ([discussion 742801](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742801); G-F G9).

### 3.3 Other teams and public fine-tunes
- **Yi-Chia Chen (#2, 67.49):** solo, 34 submissions, steady climbs 48.59 → 67.49 between 10-08 and 10-12. She has GRPO and on-policy-distillation history, but her OPD-32B and DFlash models are AIMO artifacts, not ARC ones (L2 §5, FC 4).
- **Below them, #3 mtg is at 50.37.** The 17-point gap from #2 exceeds best-of-N alone, which is about 2.7 sd (L2 §6, FC).
- **Public weights.** Every public notebook scoring ≥20 runs stock Flash-Next, and no team-best score is shown to come from a fine-tune (L2 §1, FC 6).
- **Forum reports.** Two H200 users (SFT; RL on miles) and two others got nothing from Flash-Next. On 27B bases, a STaR LoRA moved LB 1.25 → 1.94, and behaviour cloning's +2 public points "didn't translate" to the hidden set (W5b FC 1–6; discussions 742835, 748593, 739047).
- **Harness saturation.** Makarov found nearly every harness layer at or below baseline, and Le Grand is now pursuing fine-tuning (G-F G9, FC 6–7).
- **Franzen** won M2 (27.89) with no training and is now at 38.85; NVARC3 (ARC-AGI-2 winners plus the DreamTeam authors) is at 43.94 (W5 Q3, FC 3; L2 §2).

### 3.4 Post-trained Flash-Next variants (547 HF derivatives; about 8 are real post-trains)

| Variant | Tensors changed | Overlay on Intel W4A16? | Evidence |
|---|---|---|---|
| Swift 1.5 (UkisAI) | 300 tensors, 5.82 GB: GDN qkv/z/out, QSA q/k/v/o, shared expert | Yes, no requant | Efficient-thinking tune (−31 to −57% thinking on short benchmarks). Felix's Kaggle A/B was a 10-game, 8–11-min screen: decode 4–8% *faster*, tokens per action −4% by solver totals, 7 vs 13 levels |
| Tinfield-1 | 518 tensors, 6.64 GB: GDN incl. a/b, QSA, hyper-connections; tiny deltas (0.03–0.12% RMS) | Yes | Self-reported Terminal-Bench 33.0 vs 29.0; NVFP4 ARC probe +5.6 pp, NO_GO, p ≈ 0.54 |
| AZ-Focus, Agnes-3.0, CYBER-FROST, abliterations | Routed experts | No: needs int4 requant | — |
| Tong Hui Kang's LoRA (32.48) | Merged into NVFP4; scope unknown | Unknown | No score claim |

Sources: G-D §1–3, FC 2–10 (HF `?blobs=true`, safetensors headers, Felix's `benchmark.json` and `serve.log`). [I] Swift and Tinfield show the deployable LoRA shape exists; their overlays are the cheapest in-harness test of whether post-training Flash-Next helps here.

## 4. Toolchain, deployment, compute, rules

### 4.1 Trainers
- **ms-swift Megatron (mcore-bridge) is the only documented one-node LoRA path.** Recipe: LoRA r8 on in_proj / out_proj / linear_proj / linear_qkv, TP2/EP4/PP2, 72.5 GiB per GPU at 8k with PLE offload and padding_free (93.3 without), measured only to 8k on synthetic text. A one-node GRPO-LoRA recipe exists too: vLLM colocated, 1k prompt + 8k completion, "8*135G" (L1 FC 4; W6 FC 2, 4; [best-practice page](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html)).
- **Pin a working user's versions:** ms_swift 4.6.0, mcore_bridge 1.6.5, megatron-core 0.16.1, transformers 5.19.0 ([ms-swift #10376](https://github.com/modelscope/ms-swift/issues/10376); W6 §1).
- **In code but undocumented in practice:** context parallelism (CP > 1 needs `cp_comm_type all_gather`), a multimodal loader (`Qwen4ExpMMLoader`), and MTP training (`--mtp_num_layers 1 --mtp_loss_scaling_factor 0.1`) (W6 §1, FC 3; `_critic.md` spot check 1).
- **Memory by length** (extrapolated): 16k fits at CP1 on H200, 32k probably CP2, 55–64k CP4. The fp32 logits are 60.6 GiB at 64k, so `--cross_entropy_loss_fusion` is needed. Throughput is unpublished; the estimate is 2–7 node-hours per 100M processed tokens at an unverified 3–10% MFU (W6 §3).
- **Other stacks:** NeMo AutoModel (SFT, 64 H100, 4k, no MTP); NeMo RL (GRPO, 64 GPUs, 4k); miles (GRPO, 32 GPUs, SGLang rollouts, MTP unmapped); prime-rl (16 B300, 4k); Axolotl QLoRA (1×B300, about 120 GiB, 4k, sdpa only, no H200 figure). verl, LlamaFactory, TRL, Unsloth and Megatron-Bridge don't support the model (W6 §1, FC 4–5).
- **Kernels.** GDN and QSA backward both exist (fla; mcore-bridge, AutoModel, HF sdpa), but the QSA indexer has no gradient anywhere, so freeze it. H200 (sm90) is the best-trodden target (W6 §2, FC).

### 4.2 Data rendering for ms-swift (closed)
- **Four exporter transforms make ms-swift 4.6.0 `qwen3_8` match exactly:** 25 of 25 full logged windows id-for-id in inference and training mode, re-verified on 25 new rows.
  1. `--enable_thinking true` (the default False drops the xhigh effort sentence).
  2. Embed the tools block in the system text (null-key stripping otherwise drops `defer_loading`).
  3. Join consecutive user messages with a literal turn break.
  4. Render each tool→user group as literal `<tool_response>` user text (otherwise an AssertionError).

  Use the default `qwen3_5` agent template, not `qwen3_coder` (+15 to +19 tokens), and not the jinja backend, which trains on every token. Historical thinking is kept in training as at serving (G-C Q1, FC 4–7; `swift/template/base.py:133, 344, 1398, 1449`).
- **Packing.** For behaviour cloning of logged replies, packing append windows costs 11.5M vs 145M tokens per run (12.6× less) and is self-consistent, since logged targets and contexts are both normalised. For R3 siblings, train per call on the served context: 3,000 examples × 57k ≈ 171M tokens ≈ 3.4–12 node-h ≈ $290–1,020 (G-C Q2; G-G (c)).
- **Log raw data from now on.** Collection runs should set `ARC3_VERBATIM_HISTORY=1` and log raw replies and token ids. The flag's only full run (tput-safe, 42.09 vs control 50.62) was confounded by gate11 queueing; the mechanism saves about 2M prefill tokens per run (G-C Q3, FC 10–11).

### 4.3 Deployment path
- **Intel W4A16 quantizes only the routed experts of the 48 trunk layers** (int4 g128 sym, AutoRound 0.15.0). Attention, GDN (incl. in_proj_a/b), the shared expert, the router, hyper-connections, the QSA indexer, the MTP (incl. its 512 experts), PLE, embeddings and vision are BF16 (W2 FC 6; W6 FC 1; [config.json](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/blob/main/config.json)).
- **So a non-expert LoRA merges with no AutoRound re-run,** deleting the 10-17 plan's re-quant step (`finetune-plan-2026-10-17.md:38`; W4 §3; W6 §5).
- **Overlay recipe.** Patch copies of only the shards that hold changed tensors, keyed by tensor name (our Kaggle copy is resharded), and symlink the rest. Leave `config.json` and the tokenizer untouched: every transformers release with `qwen4_exp` rewrites `full_attention` on save. Copying whole Intel shards would be about 72 GB; new-shard overlays are about 5.8–6.7 GB plus an index (G-D §3; W6 FC 8).
- **Merge rather than serve an adapter:** SGLang LoRA serving is undocumented for MoE, quantized bases and speculation (W6 §4; W10 Q5).
- **Sizes.** BF16 checkpoint 360 GB (180.0B params incl. the 51.2B PLE table); routed experts 120.8B, GDN 2.09B, QSA 0.60B, shared experts 0.24B. The served weights take 69.85 GB of GPU, with the 95.4 GiB PLE table in host RAM (W6 §3, FC; W5 Q5; G-B Q7).
- **The MTP drafter (albucino)** has 90.6M BF16 params of its own plus int4 g32 experts and serves at accept 2.68. Clean decode tok/s is proportional to accept length; harness tok/s rises +2.25% per +0.1 accept. QSA caps a verify block at 4 tokens (G-E Q2–3, FC 1–2, 9).
- **R1, retraining the MTP's dense tensors on our own transcripts.** Central gain +0.15 to +0.25 accept (Qwen3-Next analogue +0.21) → +2.0 offline, +1.1 to +1.9 LB. The AngelSpec analogue, +0.36 to +0.41, gives up to +3.9 offline and +2.2 to +3.9 LB. Cost is under 15 GPU-h. The main risk is that no SGLang hidden-state capture hook exists for qwen4_exp; botched head-only retrains have regressed (Holodeck: hard labels, LR 2e-4) (G-E Q1, Q4, FC 8, 10).
- **The G-MTP gate after any trunk change:** SHA-check the drafter files; check spec-on vs spec-off greedy identity against the base; on a panel, require accept ≥ base − 0.05, clean decode ≥ 0.98× base and step time within ±1 ms; in the full run, accept ≥ 2.63 (G-E Q5).
- **Logprobs** (read from source, not run). The chat endpoint returns output logprobs with top-k and `return_prompt_token_ids`. Prompt logprobs come only from native `/generate` with `max_new_tokens 0`. Output logprobs under speculation are temperature-scaled unless `SGLANG_RETURN_ORIGINAL_LOGPROB=1`, and never top-p/top-k truncated. A one-session smoke spec exists (G-C Q4, FC 12–13).

### 4.4 Compute options and prices

| Option | Price | Notes | Source |
|---|---|---|---|
| Azure ND96isr H200 v5 (8×H200 141 GB, 1,850 GiB RAM, 28 TiB NVMe) | $84.80/node-h ($10.60/GPU-h) in eastus2, westus2, westus3; spot meter the same | 450–950 GPU-h ≈ $4.8–10.1k; quota timeline unknown (vendor: 1–4 weeks, UNVERIFIED) | W6 §7, FC 6 |
| Azure NC144lds RTX PRO 6000 BSE (1 GPU, 264 GiB) | $5.50/h; spot about $1.02 | Exact serving GPU; a full run ≈ $13 on-demand, about $2.5 spot; Kaggle-card equivalence unverified | W6 FC 7; G-B Q7 |
| Azure MI300X / GB200 | $48 / $108.16 per node-h | No Flash-Next training on ROCm; GB200 is aarch64 | W6 §7 |
| Lambda 8×B200 | $53.52/node-h ($6.69/GPU-h) | Fallback; fla had sm100 bugs | W6 §8 ([pricing](https://lambda.ai/pricing)) |
| Kaggle | Free; 6 GPU-h/day on mellamoa | Full run ≈ 2.3 GPU-h; scoring reruns use no quota | CLAUDE.md; G-H FC 11 |

An RTX PRO 6000 can't train a ~100B model (W5b §1). The ~9 h AutoRound pass was measured for a default, not experts-only, quantization (W6 FC).

### 4.5 Rules and licences
- **Submissions:** 1 per day and 2 final selections. Entry deadline 2026-10-26; final deadline 2026-11-02 23:59 UTC; about 16 mellamoa slots remain from 10-18. The scoring box is an RTX 6000 (g4-standard-48), ≤9 h, internet off (W5b FC 5; G-B Q4).
- **One account per person.** "You will be disqualified if you make Submissions through more than one Kaggle account" (rule 3.5.a); teams are ≤8 (G-F G17, FC 2). G-B's fact-check found snowiers draws in `runs/submissions.csv` (28.73, 29.54, 34.04, 28.06), with snowiers as a separate leaderboard entry: a live disqualification risk. Compute-only use of a second account is not addressed (G-B FC 11).
- **Data, weights and code.** External data and models are allowed if "reasonably accessible to all" at "minimal cost" (2.6), and fine-tuning on rented compute is not barred. Winners must open-source the system, model and weights under CC-BY 4.0, with a carve-out for pretrained models under incompatible licences, and deliver training code (2.5, 2.8). Staff have not answered whether the Qwen licence is prize-eligible (G-F G17, FC 3; W5b §4).
- **ARC's taxonomy** calls training on lookalike games, *or a harness with ARC-specific strategies*, "domain-specific overfitting". It is defined for ARC's official leaderboard, not Kaggle (W5 FC 4; W9 FC 3).
- **Licences.** DreamTeam Apache-2.0; arc-interactive MIT; sonpham and Training-Data none; Qwen3.8-2.4T MIT-style; Swift adds a US$1M revenue threshold. Anthropic (AUP, Commercial Terms D.4), Gemini and OpenAI bar training competing models on outputs. GPQA must not be revealed online (W9 FC 8; W10 FC 6; G-D §2; G-B FC 12).

## 5. Settled contradictions and open questions

### 5.1 Settled (claim → verdict; deciding evidence)
1. "50–60% of tokens on never-cleared levels" → 22–28% in leader runs (32% overall); 50–60% is stalled levels (L1; G-A level-cost table).
2. "DreamTeam has no solutions" → all 25 embed `solve_level` (20 of 25 match baselines); use as verifiers, not SFT text (L1 FC 8; W9 FC 6; `_critic.md` C17).
3. "A merged LoRA needs AutoRound requant" → not for a non-expert LoRA (Intel `extra_config`; Swift and Tinfield overlays) (W6 FC 1; G-D).
4. "The overlay is about 5.9 GB" → about 72 GB if whole Intel shards are copied; small only with new shards or name-keyed patches (W6 FC).
5. "Only transformers ≥5.18 renames attention" → every qwen4_exp release does; keep the original `config.json` (W6 FC 8).
6. "GSPO removes the need for routing replay" → no; it collapsed without replay in three sources (W2 FC 2).
7. "The 2.4T can teach" → it is text-only (W10 FC).
8. "LB sd 5 or 3.04" → 3.50; consistent 60 ≈ mean 64.5 (L3 FC 7).
9. "State-lab does 1.2–1.6k samples per GPU-h" → 730–960 all-in (L1 FC 6).
10. "The hidden set is 55 games" → 110, about 55 per LB (Kaggle data page; G-F FC 1).
11. "Swift decodes 16% slower" → 4–8% faster per Felix's `serve.log`; it was a 10-game screen (G-D FC 7).
12. "AngelSpec 52.8 → 66.4% is unverified" → confirmed in the full text (G-E FC 8).
13. Novelty rewards / "explores too little" (C1) → drop them: novelty stalls are 0.6% of tokens, and new-pair rates are similar on failed levels (G-A Q2).
14. "R3 is low risk" (C2) → medium: Arora's RS+SFT collapse and the local P = 0.445 (G-G (a)).
15. "30–50% self-replay" (C4) → 0% generic replay; a median-length within-task anchor plus λ-merges; replay anchors only partly (G-G (b), FC 10).
16. "LR 1e-6 vs 1e-4" (C5) → full-FT equivalent × 10 for LoRA: 1e-5 to 3e-5 (G-G (d)).
17. Trajectory- vs step-level DPO (C6) → same-state sibling pairs, one per state, after SFT; cross-run segment pairs have different prompts (G-G (e)).
18. "Short-window training is harmless" (C7) → train on the served shape; AWM Table 11 shows a 59% relative loss from context mismatch (G-G Q1).
19. Packing vs normalised history (C8) → moot for logs, since both sides are normalised; R3 trains per call (G-C Q2; G-G (c)).
20. Training on public games (C9) → a 15/10 split with held-out H10 (lp85, tr87, cn04, sc25, vc33, dc22, ka59, lf52, re86, wa30). Evaluate trained checkpoints only on U25 (H10 + 15 DreamTeam games) and the LB, never on H10 inside a run with trained games, where freed compute adds up to +3–8 (G-B Q1–2, FC 5).
21. Which forgetting gate (C16) → game metrics plus tool-call validity, with a GPQA-100 / IFEval-150 canary as the back-off trigger (G-B Q6).
22. Forum failures vs Tinfield (C18) → not decisive: tiny deltas, NO_GO probe (G-D).
23. "~63 is this setup's physics" (C19) → our agent's efficiency ceiling (§1.8).
24. "Tufa trains at 100–200k tokens" → Tufa trains shorter to avoid that (L2 FC 7).
25. "R1 reaches 3.0–3.2 accept" → that is the upside; the central case is +0.15 to +0.25 (G-E Q1, FC 8).

### 5.2 Still open (cheapest test where known)
1. **Upward elasticity (R3's premise) and the curve's shape above 121 min.** The leader at `demo_minutes` 242 or 161 (about 3–4.3 GPU-h), with the pre-registered falsification rule (G-H Q5, FC 3).
2. **P1 vs P3 on the hidden set.** One halved-deadline LB draw, or the first draws of a token-cut checkpoint (G-H Q5).
3. **R3 yield at easy, on-route states, and the source of samples.** Can whole-turn selection reach −15%? W1 and W7 recommend sampling at `reasoning_effort=low` and training under xhigh (Munkhbat's conditioned best-of-N). G-G says never to use a low-effort or concise teacher, but its evidence (OPSD Fig. 9) is a distillation result, not a selection one. Unresolved; a state-lab test costs about 1–2 GPU-h (`_critic.md` G4).
4. **Spill-over and memory.** Does R3 on cleared states shorten stalled-level turns? Does compressed thinking cost the working memory later turns read (`_critic.md` G14)?
5. **Training at the deployed shape:** a 64k multimodal step with CP, GDN packing parity, training tokens/s. Needs a day-1 node smoke (`_critic.md` G5).
6. **Node quota, start date and the training prong's kill dates** (`_critic.md` G1).
7. **Logprobs on Pennyroyal.** The smoke spec is ready (G-C Q4).
8. **Flash-Next on external games:** pass rates on DreamTeam and arc-interactive, U25 noise, and whether external games track the LB (~31) or public (~51). One panel is about 2.3 GPU-h (`_critic.md` G7; G-B Q3).
9. **`game_creator` yield with an open model:** a 20-seed pilot (W9; `_critic.md` G16).
10. **Whether frozen-state screens predict full-run score;** existing data can't back-test it (G-B Q5).
11. **The Swift and Tinfield overlay smokes** (G-D §4).
12. **LoRA scope and LR:** shared-expert vs attention-only LoRA; the 10× rule on a partial-module MoE LoRA (W1; G-G).
13. **MTP after a trunk LoRA:** acceptance change, E[max p] ceiling, capture-hook feasibility (G-E).
14. **Whether verbatim history is score-neutral;** no isolated run exists (G-C Q3).
15. **Tufa's and Chen's methods;** Tufa releases after the deadline (G-F).
16. **Rules exposure:** past snowiers submissions, compute-only use of a second account, Qwen-licence prize eligibility, and whether the best of the two final picks counts (G-F; G-B FC 11, Q4 gaps; W5b).
17. **The wind tunnel (C12):** Axolotl QLoRA on H200 without expert LoRA, or short ms-swift runs.
18. **Deployment and stall mechanics:** Kaggle upload limits; whether Pennyroyal loads a symlink-overlay directory; goal vs mechanic errors beyond the 30-stall audit (W6 §7; G-D §3; G-A Q3).
