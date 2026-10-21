# Training disciplines for fine-tuning an open-weight agent on ARC-AGI-3 (2026-10-17)

Legend: [E] = evidence from a cited source; [I] = my inference. Search-tool summaries were used for many sources; items marked (unverified) were not read in full. The MiniCPM-2B paper details were read via ar5iv (first ~100k of 117k chars).

## 1. MiniCPM training scheme

### What it is [E]
- Source: MiniCPM-2B paper https://arxiv.org/abs/2404.06395 (ar5iv: https://ar5iv.labs.arxiv.org/html/2404.06395); MiniCPM4 https://arxiv.org/abs/2506.07900.
- **WSD (Warmup-Stable-Decay)**: warmup, a long constant-LR stable phase, then a short decay phase (exponential, f = 0.5^((s-S)/T), T = 5,000 steps ~ 20B tokens in the 2B model, ~2% of ~1.1T tokens... the 0.036B ablations found ~10% decay beat 2.5%). A stable-phase checkpoint can be annealed at any time, so one run yields many "finished" models and cheap scaling-law fits (6 sizes x 6 decay checkpoints).
- WSD vs cosine: the paper reports cosine with endpoint set exactly equal to total steps gives the lowest loss at a given fixed length; WSD's advantage is continuability and cheap checkpoint-anywhere, not lower loss at fixed length.
- **Decay-phase data mixing ("two-stage pretraining")**: at the start of decay, mix pretraining data with high-quality/SFT-style data (UltraChat, SlimOrca, OssInstruct, EvolInstruct, proprietary). Then a short true SFT (~6B tokens, LR continuing from the annealed end, also WSD-decayed). Ablation at 1.2B: "decay mix + 6B SFT" beat "pretrain-only decay + 12B SFT" on C-Eval (49.1 vs 41.2), CMMLU (46.8 vs 42.0), MMLU (49.6 vs 47.9), MATH (10.5 vs 7.3), but not GSM8K (31.8 vs 34.4). The authors argue it is not just SFT-token count.
- **Model wind tunnel**: hyperparameters tuned on a 9M-param model (~0.09B tokens) and transferred with muP (scale_depth 1.4, scale_emb 12, init_std 0.1, lr 0.01); batch-size scaling law bs ~ 1.21e9 / L^6.24; LR stayed ~0.01 from 0.04B to 0.5B (checked at 2.1B). Compute-optimal data:model ratio estimated ~192:1 (vs ~20 Chinchilla).
- MiniCPM-DPO: one epoch took MT-Bench 6.89 -> 7.25 with tiny benchmark dips.
- MiniCPM4 adds ModelTunnel v2 (strategy search), UltraClean/UltraChat v2, chunk-wise rollout for load-balanced RL, 8T tokens for 8B. I could not retrieve its decay-stage details (unverified).
- Related annealing evidence: OLMo 2 spends the last ~5-10% FLOPs on "Dolmino" mid-training (high-quality + instruction/math data); "micro-annealing" cheaply ablates individual data sources; reported dev-average 50.6 -> 61.2 at 7B (secondary summary) https://arxiv.org/abs/2501.00656. Llama 3 annealing: no source retrieved.

### Does it transfer to continued training of an existing model? [I]
- WSD itself is a from-scratch pretraining device; our model is already decayed. Transferable parts:
  1. **Late-phase "data-quality injection with a retained general-data share"**: the real MiniCPM lesson is that mixing task/SFT-style data with pretrain-style data during a still-warm LR phase beats a pure SFT afterwards. For us: mix ARC-like trajectories with replayed general/reasoning data in one stage rather than pure-SFT-on-trajectories. This is also our forgetting guard.
  2. **Re-warm then re-decay**: a short re-warmup to a modest LR (well below original peak), a plateau, then a decay of ~10% of steps to ~1-10% of peak. Cheap and gives an intermediate checkpoint-anywhere property; useful so one training run produces several evaluable checkpoints.
  3. **Wind tunnel**: do all hyperparameter search (LR, mixture ratio, epochs, LoRA rank) on a small sibling model (e.g. a ~4-8B dense or small Qwen3.x of the same family) using the same data, and transfer only the rankings, not LR values (muP LR transfer does not apply to a fine-tune of a non-muP checkpoint).
  4. **Do not** expect MiniCPM's gains to come from the WSD curve: the decay-phase mixture matters, not the shape.
- Cost: MiniCPM's decay mix (20B tokens) + SFT (6B) is ~26B tokens; for us a trajectory corpus is 20-100M tokens, 2-3 orders smaller, so the "mixture" is mostly replay + our data at roughly 1:1 to 1:3 token ratio [I].

## 2. Agentic / multi-turn RL (2025-2026)

- **RAGEN / StarPO** (arXiv 2504.20073; https://arxiv.org/abs/2504.20073): multi-turn RL on Sokoban etc. Documented the "Echo Trap" collapse (reward-variance cliffs, entropy fall, gradient spikes); StarPO-S stabilizers = trajectory filtering by reward variance, critic, decoupled clipping. Reasoning does not emerge without reasoning-aware reward; diverse initial states and moderate interaction granularity help. RAGEN-2 reports "template collapse" invisible to entropy. [E]
- **GiGPO / verl-agent** (https://arxiv.org/abs/2505.10978, https://github.com/langfengq/verl-agent): critic-free, step-level groups from repeated states; +12% ALFWorld, +9% WebShop over GRPO with no extra rollout cost (Qwen2.5-1.5B/7B). Relevant because grid-game states repeat across rollouts (anchor-state grouping is natural for ARC-AGI-3). [E; numbers from abstract]
- **SPIRAL** (https://arxiv.org/abs/2506.24119): self-play on zero-sum games improves general reasoning ~10%; beat SFT on 25k expert trajectories. Games are not hidden-rule discovery, so transfer to our case is weak. [E]
- **MARSHAL** (https://arxiv.org/abs/2510.15414): per-response length penalty in game RL; removing it increased length 1700->1954 tokens and hurt Mini Hanabi 50.5->38.2. So length penalties can coexist with, even help, game performance. **T2PO** (https://arxiv.org/abs/2605.02178) reduces tokens on WebShop/ALFWorld; **TAB** (https://arxiv.org/abs/2604.05164) learns per-turn budgets (math, up to 35% saving). **Explore More, Drift Less** (https://arxiv.org/abs/2609.01245): outcome-only RL on AppWorld with moderate train turn budget and larger test budget. (All abstract-level, unverified.)
- **MoE RL stability**: routing mismatch between inference engine and trainer causes collapse; Router Replay R2/R3 (https://arxiv.org/abs/2510.11370; ms-swift docs https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/router_replay.html), PR2 predictive replay (https://arxiv.org/abs/2606.00395). Tested on Qwen3-30B-A3B. [E]
- Frameworks seen: verl/verl-agent, slime (Megatron+SGLang), NeMo-RL (Megatron backend; Qwen3-30B-A3B 2-stage SWE RL https://docs.nvidia.com/nemo/rl/latest/guides/swe-rl-qwen3.html), siiRL, KRL tree-training (~6x claimed, https://arxiv.org/abs/2603.27703). I found nothing on prime-rl/ROLL/SkyRL or any Qwen3-Next-class RL run. [E / gap]
- No published multi-turn RL result on ARC-AGI-3, PuzzleScript, or hidden-rule exploration found. Closest: ARC-AGI-3 preview winner StochasticGoose (CNN + RL predicting which actions change the frame, 12.58% on 3 private games, then ~0.25% on the full benchmark per a DataCamp analysis) https://arcprize.org/competitions/arc-agi-3-preview-agents. [E]
- Credit assignment [I]: with sparse level-clear reward, per-level reward (+ shaped "novel board reached" bonus) plus GiGPO-style anchor-state grouping is the most natural fit. Novelty shaping directly targets our measured failure (untried actions that reach new boards).

## 3. SFT-style methods

- **ReST-EM** (https://arxiv.org/abs/2312.06585): generate -> binary-filter -> finetune from base each round; scales with model size, beats human-data SFT on MATH/APPS (PaLM 2). [E]
- **SWE-Gym**: 491 sampled successful trajectories took Qwen2.5-32B from ~3% to 15.3% (Lite) and 20.6% (Verified), still scaling with count (https://arxiv.org/abs/2412.21139). SWE-smith does rejection-sampled teacher trajectories (https://arxiv.org/abs/2504.21798). JetBrains step-level rejection FT (https://blog.jetbrains.com/research/2026/06/step-rejection-fine-tuning/). So hundreds to low-thousands of verified trajectories gave double-digit gains in agent tasks, but those had a large untrained-format gap; ours is a strong already-agentic model, so expect smaller gains [I].
- **DPO**: MiniCPM-DPO gives a small chat gain; for agents, preference pairs from same-state branches (cleared vs. stuck) are plausible but unproven here [I].
- **Naestro-AGI3-27B** (https://friendli.ai/models/star-ga/naestro-agi3-27b): QLoRA r=16 on Qwen3.6-27B using procedurally generated ARC-AGI-3-style sims, frame -> win-condition + short plan. Page reports no scores, no compute; vendor claim, treat as unverified. The page itself calls first-exposure generalization "the unsolved core". [E, weak]
- Think-less distillation: no strong ARC-specific source; MARSHAL/T2PO above are the nearest. [gap]

## 4. ARC-specific lessons

- ARC Prize 2025 (https://arcprize.org/blog/arc-prize-2025-results-analysis, https://arxiv.org/abs/2601.10904): winners NVARC (24.03% ARC-AGI-2 private; synthetic puzzle generation, 4B model, offline reasoning moved into data), ARChitects (2D masked-diffusion LM, recursive self-refinement), MindsAI (test-time fine-tuning, augmentation ensembles). Theme: task adaptation in weights at test time, heavy synthetic data, refinement loops. Poetiq harness lifted Gemini 3 Pro 31 -> 54% on ARC-AGI-2: harness matters as much as weights. [E]
- ARC-AGI-3: preview winners were non-LLM (CNN-RL, state-graph search); 2026 milestone first prize went to tufalabs "Duck" harness on open Qwen 3.6 27B at ~1.2% (secondary repost; unverified). Commentary says the same model scores 7.8% vs 38.3% depending on harness. [E, secondary]
- Implication [I]: ARC-AGI-3 hidden-mechanic inference cannot be memorized from public games alone (110 private games); test-time-in-context learning (within the 9-h run) is the analog of TTT. Weights should teach *how to explore and record hypotheses*, the harness should carry *what was learned this game*.

## 5. Practicalities

- **Hybrid linear-attention MoE**: Qwen3-Next = 75% Gated DeltaNet + gated attention, ~512 experts (10 active + shared). NVIDIA Megatron-Bridge docs state LoRA/DoRA is NOT available for Qwen3-Next; only pretrain/full SFT recipes (https://docs.nvidia.com/nemo/megatron-bridge/latest/models/qwen/qwen3-next.html). I could not confirm Unsloth/ms-swift support or target-module names. Check our exact "3.8-Flash-Next" checkpoint's module names. [E + gap]
- **Forgetting**: evidence conflicts on LoRA ("learns less, forgets less" vs. intruder dimensions, forgets more); replay of ~10% recovers a large share of forgetting in a toy study, diminishing past 25% (https://dev.to/javieraguilarai/the-forgetting-you-dont-measure-3h11, weak). Replay with self-generated data (model's own thinking traces on general prompts) is the cheapest to produce and preserves thinking style [I].
- **Re-quantization**: W4A16 AutoRound after FT is just PTQ and should be redone on the fine-tuned weights with calibration data drawn from our agent transcripts [I]. NVIDIA reports QAD (500 iterations against a BF16 teacher) recovers NVFP4 W4A4 loss on Qwen3.6-35B-A3B (https://nvidia.github.io/Model-Optimizer/pr-preview/pr-2517/announcements/qwen36-w4a4-qad.html); same page says W4A16 was slower than BF16 in 10 of 12 Blackwell shapes under vLLM, which may matter for the serving choice [E, vendor]. Our SGLang path may differ.
- **MTP draft head**: Self-distilled head-only retrains regressed acceptance (49.75% vs 58.55%) in one community experiment (https://huggingface.co/Wiself/Holodeck-Lounge-Regressed-MTP); AngelSpec reports training the drafter on target-model rollouts lifts average acceptance 52.8% -> 66.4% (https://arxiv.org/abs/2607.25852); Nemotron 3 Super uses shared-parameter heads for gentler degradation (https://arxiv.org/abs/2604.12374). Narrow-distribution heads lose acceptance off-distribution. [E, weak/mixed] So: after FT, either keep MTP loss in the training objective (frozen-trunk MTP loss on the new data) or retrain head on rollouts of the *fine-tuned, quantized* model, measure acceptance first, and keep the old head as fallback.
- **GPU-hour estimates** [I, back-of-envelope; no source gives these for an 80B MoE]:
  - Price: ND H100 v5 ~$98/hr per 8 GPUs on-demand (~$12/GPU-hr; third-party, https://spheron.network/blog/azure-h100-pricing); no GB200 price found.
  - SFT of 20-100M tokens, ~3B-active MoE, full-parameter with Megatron, 1-3 epochs: ~100-400 H100-hours (MoE all-to-all and sequence length 32k+ dominate, not FLOPs). Smaller-sibling wind tunnel: <50 GPU-h.
  - RL: one iteration of 32 games x 8 rollouts x ~200k tokens/episode ~ 50M generated tokens; at ~20-40k tok/s on 8 H100s that is ~0.5 h per iteration generation-side plus train step. 100 iterations ~ 400-800 H100-hours, plus game-sim CPU. Sensitive to episode length: cap thinking per action.
  - Data harvesting from existing logs is free; re-sampling with the fine-tuned model is the main cost.

## 6. Recommended curriculum (for our case)

Our measured facts [from the brief]: loss is goal/mechanic inference on stuck levels; ~half of tokens go to stuck levels; clear levels are near-optimal; shorter thinking buys more actions.

**Stage 0 - Eval harness and wind tunnel (~50 GPU-h).** Freeze held-out games (hold out ~5 of 25 public games entirely for never-trained-on generalization; the private set is unseen, so public-game wins are contaminated otherwise). Metric: levels cleared + actions-to-clear at matched wall-clock, 3+ seeds per arm (user's alpha rule). Tune LoRA rank / LR / replay ratio on a small same-family model [MiniCPM wind-tunnel analog].

**Stage 1 - Mixed "decay-style" SFT / expert iteration (~150-400 H100-h).**
- Data: (a) successful-level segments from the ~50 logged runs (only steps whose resulting board/level progressed; hindsight-relabeled rationales at stall points per R4, written as short facts rather than long thinking); (b) shorter-thinking variants of the same decisions (R3: select the shortest successful sample); (c) 30-50% replay of general reasoning/tool-use data and the model's own thinking traces on general prompts; (d) optionally teacher trajectories from a frontier API model on held-in games (distillation).
- Schedule: short re-warmup to low LR, plateau, ~10% exponential decay (WSD-style), loss on assistant/tool-call tokens only, MTP loss kept on.
- Eval: held-out-game clears and tokens/action vs. base; general benchmarks (math/code/tool-use) for regression; stop if general drops > ~2 points.

**Stage 2 - Rejection-sampling round(s) with the new model (R2/R6, ~200-400 GPU-h per round).** Re-sample on all games including synthetic variants (procedural mechanic sims; Naestro/NVARC-style), keep cleared-level and novel-board-reaching segments, retrain from the base (ReST-EM style) to avoid drift. 2 rounds max; watch diversity collapse.

**Stage 3 - Optional RL (400-800 H100-h; only if Stage 2 plateaus).** GRPO/GiGPO with per-level clear reward + novelty shaping + per-action token-length penalty (MARSHAL-style), Router Replay for MoE, trajectory filtering by reward variance (StarPO-S), reduced-length curriculum. Framework: verl-agent or slime/NeMo-RL (Megatron backends). High risk: no public precedent on a Qwen3-Next-class model for this task.

**Stage 4 - Deployment engineering (~50-100 GPU-h).** Re-run AutoRound W4A16 with agent-transcript calibration; verify the quantized model against the BF16 teacher on held-out games (QAD if gap is real); retrain/verify MTP head, measure acceptance and tokens/s on real streams; re-tune harness prompts last (do not touch prompt and weights in the same A/B).

**Order of expected value [I]:** R3-style short thinking + hindsight stall rationales (Stage 1) first, since they directly target the measured failure and cost the least; RL last.

## Top risks
1. Overfitting to the 25 public games (contamination), giving a public score that does not carry to the 110 private games.
2. Forgetting / loss of thinking quality, and LoRA unsupported for Qwen3-Next in Megatron-Bridge, forcing full FT.
3. Quantization and MTP misalignment erasing gains; unverified outside vendor claims.
4. Noisy per-game scores: need many seeds to detect real gains.
5. RL instabilities on MoE (routing mismatch, Echo Trap) and cost explosion from long episodes.
