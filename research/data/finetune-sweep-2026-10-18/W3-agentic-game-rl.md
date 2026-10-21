# W3: Training LLM agents for multi-turn games and interactive environments, and what generalizes to unseen ones (2026-10-18)

Scope: 2024-2026 work that trains weights (SFT, RS, RL) for games/interactive environments, with emphasis on transfer to unseen environments. Builds on `research/data/finetune-research-2026-10-17/training.md` (which had RAGEN, GiGPO, SPIRAL, MARSHAL at abstract level); this file adds result tables, scale, the 2026 work, and an in-repo measurement. Numbers come from the primary paper pages fetched on 2026-10-18 (arXiv HTML unless stated). "UNVERIFIED" = abstract-only or secondary.

## 1. What the main game/agent-RL works trained, at what scale, and what transferred

### Takeaway
Every transfer result to *unseen* environments in this literature comes from RL or on-policy filtered self-training on small models (0.5B-12B, plus one 35B-A3B MoE). Episodes are short, there are tens of thousands of them, and the context is small. Near transfer is real (new instances or nearby families get +10 to +60 points). Far transfer is small (+2 to +24). SFT on another model's expert trajectories transfers weakly or negatively. The one 2026 hidden-dynamics result (VHD-Play, Qwen team) is the closest analogue to ARC-AGI-3 and needed only 34 GRPO steps on an already post-trained Qwen MoE.

### Cited Findings
- **RAGEN / StarPO (2504.20073).** Setup:
  - Qwen2.5-0.5B-Instruct on Bandit, Sokoban and FrozenLake; 3B on WebShop.
  - P=8 prompts x N=16 rollouts per batch, 100-200 iterations, at most 5 turns x 5 actions (10 actions per episode), H100/A100. No GPU-hours are given.

  Results:
  - The trained 0.5B reaches Sokoban 20.70 / FrozenLake 21.48, against GPT-4o 27.73 / 26.56 zero-shot (Table 5).
  - Generalization from Sokoban training is best with 4 responses per prompt: SokobanNewVocab 25.39 vs 22.27 with 1 response, FrozenLake 21.48 vs 12.50 (Table 1).
  - 5-6 actions per turn is best (Sokoban 28.13 / 33.59); 7 actions drops to 22.27 (Table 2).
  - Fresh rollouts every update generalize better (Fig. 7).

  Source: [RAGEN](https://arxiv.org/html/2504.20073).
- **RAGEN, the "Echo Trap".** The policy overfits to locally rewarded reasoning templates.
  - Symptoms: reward-std collapse, entropy drop and gradient-norm spikes. Reward std falls before reward does (FrozenLake-PPO: std at step 40, reward at step 90). Gradient spikes mark irreversible collapse (step 170 Bandit, 110 Sokoban, 90 FrozenLake).
  - StarPO-S fix: keep the top p% of prompts by reward std (default 25%). Keeping 75% extends stability from 100 to 140 steps; 50% avoids collapse. Also no KL and clip-higher (0.28/0.2).
  - Reasoning shrinks under sparse multi-turn reward (Sokoban 307 -> 89.5 tokens by step 200, Table 4). In multi-turn tasks "think" barely beats "no-think" (Sokoban 21.48 vs 20.73; SokobanNewVocab 18.75 vs 26.17, Table 3).

  Source: [RAGEN](https://arxiv.org/html/2504.20073).
- **RAGEN-2 (2604.06268, ICML 2026 oral).** "Template collapse": entropy stays stable while reasoning becomes input-agnostic.
  - Mutual information between input and reasoning tracks final performance far better than entropy does.
  - The cause is low reward variance: weak task gradients let regularizers erase differences between inputs. The fix, SNR-aware filtering, picks prompts by reward variance.
  - Only abstract-level numbers were found. Source: [RAGEN-2 abs](https://arxiv.org/abs/2604.06268).
- **GiGPO / verl-agent (2505.10978, NeurIPS 2025).** Setup:
  - Qwen2.5-1.5B/7B, group size 8 (16 groups x 8 = 128 envs), 150 iterations.
  - 2xH100 (1.5B) or 4xH100 (7B).
  - Prompt 2,048 / response 512 tokens on ALFWorld (up to 50 steps), with a **history window of only 2 steps**.

  Results (1.5B):
  - ALFWorld: GRPO 72.8 -> GiGPO 86.7. WebShop success: 56.8 -> 67.4.
  - At 7B: ALFWorld 77.6 -> 90.2, WebShop 66.1 -> 75.2.
  - VLM Sokoban (Qwen2.5-VL-3B): 67.1 -> 81.0.

  Why the grouping works: under 35% of anchor groups have size 1, so more than 65% of states recur across rollouts. Source: [GiGPO](https://arxiv.org/html/2505.10978).
- **verl-agent repo.** Supports ALFWorld, WebShop, Search, Sokoban, Gym Cards and AppWorld. Models: Qwen3, Qwen3-VL and LoRA (2025-06), with vLLM plus a `requirements_sglang.txt`.
  - Its memory module keeps "context length nearly constant"; there is "no requirement to concatenate the full history".
  - MoE models and Megatron are not mentioned.
  - Newer methods from the same group: HGPO (ICLR 2026) and GraphGPO (ICML 2026). Their numbers are UNVERIFIED.

  Source: [verl-agent README](https://github.com/langfengq/verl-agent).
- **SPIRAL (2506.24119).** Setup:
  - Self-play on TicTacToe, Kuhn Poker and Simple Negotiation.
  - 400 steps x 128 samples, 8xH100, about 25 h for Qwen3-4B-Base.

  Transfer to 8 reasoning benchmarks:
  - Qwen3-4B-Base 34.0 -> 44.5; Qwen3-8B-Base 39.5 -> 49.6.
  - Small on already-reasoning models: DeepSeek-R1-Distill-Qwen-7B +1.4, Llama-3.1-8B-Instruct +2.0.
  - SFT on 25k Qwen3-32B expert trajectories reaches 39.7; doubling the SFT data to 52k is still 39.7.

  Diversity and failure modes:
  - Multi-game beats single-game (44.5 vs 40.0-43.4).
  - Without role-conditioned advantage estimation (RAE), "thinking collapse" sets in: response length goes to ~0 and math falls from 35% to 12%.
  - Fixed opponents push the model below base (29.6 / 33.4 vs 34.0).
  - OOD games (Snake, Pig Dice, Truth & Deception): multi-game 59.5% win rate vs 52.9% for the best specialist; in the harder setting SPIRAL 36.1% vs SFT 18.4%.

  Source: [SPIRAL](https://arxiv.org/html/2506.24119).
- **ViGaL "Play to Generalize" (2506.08011, ICLR 2026).**
  - Qwen2.5-VL-7B, RLOO without KL, on Snake + Rotation (36K samples per game, 6xA100).
  - Math average 47.7 -> 50.6; multi-discipline average 50.3 -> 64.7; general perception unchanged (72.4 -> 72.2).
  - Unseen Atari games: 1253 -> 2251 cumulative reward.
  - In the Snake ablation, SFT on the same game data **drops** the average 49.1 -> 47.2 while RL raises it to 62.3. Random rewards give no gain.

  Source: [ViGaL](https://arxiv.org/html/2506.08011).
- **lmgame-Bench (2505.15146).**
  - Qwen2.5-7B-Instruct, RAGEN StarPO-S/PPO, trained on 6x6 Sokoban or 4x4 Tetris.
  - Sokoban training: Blocksworld-2D 9.0 -> 29.5, WebShop 7.0 -> 19.1. Tetris training: WebShop 13.4. No transfer to GSM8K or BIRD (Table 4).
  - Harness: without perception and memory modules, about 40% of runs fail to beat random; with them, 86.7% beat random.

  Source: [lmgame-Bench](https://arxiv.org/html/2505.15146).
- **Game-RL / Code2Logic (2505.13886).** GameQA covers 30 games and 158 tasks. RL on it improved several VLMs across 7 benchmarks. Effect sizes are UNVERIFIED (abstract only). Source: [abs](https://arxiv.org/abs/2505.13886).
- **Orak (2506.03610, ICLR 2026).** SFT of Llama-3.2-1B/3B on about 11k GPT-4o / o3-mini trajectory samples from 12 games.
  - Held-out games barely move: Super Mario 31.8 -> 34.4 (3B, within the reported sd); 2048 0.1 -> 3.1.
  - Unseen scenarios in seen games jump: Street Fighter 12 -> 40.
  - Baba Is You stays at 20 -> 20 (Table 7).

  Source: [Orak](https://arxiv.org/html/2506.03610v2).
- **BALROG (2411.13543) and VideoGameBench (2505.18134)** are evaluation only.
  - BALROG: vision often *hurts*, and NetHack is far from solved.
  - VideoGameBench: 3 of its 10 games are secret; the best model completes 0.48% (Lite: 1.6%).

  Sources: [BALROG](https://arxiv.org/abs/2411.13543), [VideoGameBench](https://arxiv.org/abs/2505.18134).
- **AgentGym-RL (2509.08755).** Qwen2.5-3B/7B on verl. Environments: WebArena, search, TextCraft, BabyAI and SciWorld (27 tasks).
  - ScalingInter-RL grows the turn cap over training; the repo config shows `rounds=[10,20,30]`, `steps_scaling_inter=100`. It is claimed to prevent long-horizon collapse.
  - The claimed GPT-4o parity is UNVERIFIED (shown only in a figure; the HTML returned 404).

  Sources: [abs](https://arxiv.org/abs/2509.08755), [repo](https://github.com/WooooDyy/AgentGym-RL).
- **PAPRIKA (2502.17543).** The best precedent for our planned SFT-on-own-trajectories approach.
  - Setup: Llama-3.1-8B, 10 exploration task groups (Twenty Questions, Wordle, Mastermind, Battleship, Minesweeper, Cellular automata, bandits, ...).
  - Data and training: 17,181 filtered self-generated trajectories for SFT, then RPO (multi-turn DPO + SFT) on 5,260 pairs. LR 1e-6 / 2e-7, min-p sampling at temperature 1.5, 20 samples per task.
  - Leave-one-group-out: improves 9 of 10 held-out groups (bandit 42.25 -> 62.25). Mastermind is minimal and Wordle shows no or negative transfer.
  - No general regression: MT-Bench 7.88 -> 8.14, GPQA 33.5 -> 32.8, MMLU-Pro 46.7 -> 46.2.
  - A learning-potential curriculum (coefficient of variation σ/R of the turns-to-solve reward) beats uniform sampling by 1.4 points average / 3.3 points pass@4.

  Source: [PAPRIKA](https://arxiv.org/html/2502.17543).
- **LaMer, cross-episode meta-RL (2512.16848, ICLR 2026).** Qwen3-4B, 3 sequential episodes per trial, in-context reflection between episodes, GiGPO optimizer.
  - pass@3 vs best RL: Sokoban 55.9 vs 44.1, MineSweeper 74.4 vs 55.1, WebShop 89.1 vs 75.2.
  - **pass@1 is lower** on MineSweeper (44.1 vs 52.0) and WebShop (67.8 vs 73.4).
  - Unseen ALFWorld categories: Cool 81.0 vs 58.1, Pick2 50.2 vs 36.0.
  - Reflection-only memory beats raw trajectories (Sokoban 56.4 vs 34.8).
  - RL lowers trajectory diversity; meta-RL keeps more of it. Training is about 2x slower.

  Source: [LaMer](https://arxiv.org/html/2512.16848v2).
- **VHD-Play, "Verifiable Hidden Dynamics Play" (2609.27321, 2026-09-23, Qwen authors).** It samples and solves a mathematical model first, then renders it as stateful tools, at $0.01-0.03 per environment. 3,300 environments in 11 families.

  Training: GRPO on **Qwen3.6-35B-A3B**, no SFT warm start.
  - 64 prompts x 16 rollouts per step (oversampled to 18), LR 2e-6, no KL, no std normalization.
  - **34 optimizer steps = 34,816 rollouts on 2,176 environments.**
  - Episodes average 59.7 assistant turns.

  Results:
  - Agentic macro score 0.204 -> 0.815, while the written-out (fully specified) score only goes 0.962 -> 0.992. "Most of the learnable gap ... lies in stateful interaction, not underlying problem solving" (Table 2).
  - Transfer: held-out instances +0.56, near-OOD families +0.63, far-OOD +0.24 / +0.16. BFCL-V4 interaction cells +2.84 points. TravelBench 0.700 -> 0.794.
  - Policy shift is only about 0.089 nats per token.

  Hardware is not stated. Source: [VHD-Play](https://arxiv.org/html/2609.27321v1).
- **Kimi-Researcher (frontier, inspiration only).** End-to-end REINFORCE takes HLE 8.6% -> 26.9%, "almost entirely through end-to-end RL".
  - Gamma-decayed outcome reward favours shorter solutions.
  - Some negative samples are discarded to avoid entropy collapse.
  - Context management let rollouts run past 50 iterations; an ablation found the managed model used 30% more iterations and performed better.

  Source: [Kimi-Researcher](https://moonshotai.github.io/Kimi-Researcher/).
- **Frameworks.** SkyRL-Agent (2511.16108): Qwen3-32B reaches 39.4% on SWE-Bench Verified, on top of SkyRL-train, verl or Tinker. Agent-R1 (2511.14460) uses a step-level MDP abstraction. Megatron or Qwen3-Next support was not confirmed for SkyRL, rLLM or Agent-R1.
  - Sources: [SkyRL-Agent](https://arxiv.org/html/2511.16108v1), [Agent-R1](https://alphaxiv.org/abs/2511.14460).

### Inferences
- **The pattern across RAGEN, SPIRAL, ViGaL, PAPRIKA, LaMer and VHD-Play:** on-policy data (RL, or SFT on the model's own filtered rollouts) transfers, while off-policy expert SFT (Orak, SPIRAL's SFT, ViGaL's SFT) transfers little or harms. This supports the prior plan's choice of self-generated data over teacher trajectories.
- **Gains are smallest on models that already reason well** (SPIRAL: +1.4 on R1-Distill vs +10.5 on base). Flash-Next is already heavily RL-tuned, so expect gains in the low single digits on general skill. The bigger lever is the format-specific "stateful interaction" gap, as in VHD-Play, which in our case means the harness's tool loop on our games.
- **Scale mismatch:** every positive result uses short contexts (GiGPO 2-step history and 2-4k prompts; RAGEN 10 actions). Our decisions run at 55k-token prompts. Nobody has shown RL or RS gains at that context length on games.

### Gaps
- AgentGym-RL, Game-RL and RAGEN-2 numbers beyond the abstracts (HTML 404 or not read).
- No hardware or GPU-hours for VHD-Play. Voyager (prompt-only skill library) was not fetched.
- No result on an RL-tuned model at more than 35B total parameters in games.

## 2. Long-horizon credit assignment and context management during agentic RL

### Takeaway
Two proven ways to train through context resets:
1. Split each rollout at every summary or trim point and **broadcast the episode advantage to every segment** (ReSum-GRPO).
2. Train the model to write a rolling state and mask attention accordingly (MEM1).

Step-level grouping on repeated states (GiGPO) is the cheapest turn-level credit assignment, and deterministic grid games repeat states heavily. Context management *must be inside training*: SFT on the compressed format fails to generalize past the training horizon, while RL does.

### Cited Findings
- **ReSum-GRPO (2509.13313 v3).** How it trains:
  - A rollout with K summaries becomes K+1 segments. Each segment's input is the previous summary and its output runs up to the next summary. Tool observations are masked.
  - The trajectory-level GRPO advantage (group 8) is copied to every segment.
  - 1K training samples, 4 epochs, LR 2e-6, rLLM. 8x144 GB GPUs (3B/7B) or 16x (30B-A3B). Steps take 1.05-1.44 h vs 0.62-0.96 h for plain GRPO.
  - The summarizer is a separately fine-tuned Qwen3-30B-A3B (about 9K summary pairs).

  Results (Pass@1):
  - WebSailor-30B on BrowseComp-zh: ReAct 23.9, ReSum 29.3, ReSum-GRPO 33.3.
  - WebSailor-7B on GAIA: 31.7 / 37.2 / 42.4.
  - Plain GRPO on ReAct adds less (30B BC-zh 23.3).

  Source: [ReSum](https://arxiv.org/html/2509.13313v3).
- **MEM1 (2506.15841).** Qwen2.5-7B, PPO on 4 H100/H200. Each turn the model writes an internal state, and older turns are pruned using a 2D attention mask.
  - On 16-objective QA it gets 3.5x the performance of Qwen2.5-14B at 3.7x lower peak memory.
  - **SFT on the same format: EM 0.433 vs 0.709 for RL at 2 objectives, and 0.000 at 16 objectives (collapse beyond about 6).**
  - Failure modes: a format reward raises convergence speed but lowers final EM (0.466 vs 0.709), and the agent learned a "fewer searches" shortcut.

  Source: [MEM1](https://arxiv.org/html/2506.15841).
- **Context-Folding / FoldGRPO (2510.11967).** Seed-OSS-36B with 32K branches.
  - BrowseComp-Plus 0.620 vs ReAct-327K 0.478; SWE-Bench Verified 0.580 vs 0.552.
  - Against plain GRPO with the same fold tools: 0.567 -> 0.620; finish rate 0.738 -> 0.935; main trajectory 22,285 -> 7,752 tokens.
  - Process penalties: -1 on main-thread tokens past half the context, -0.2 on off-scope branch tokens, -1 on failed tool calls.
  - Source: [abs](https://arxiv.org/abs/2510.11967), numbers via [alphaXiv overview](https://www.alphaxiv.org/overview/2510.11967) (secondary rendering of the paper).
- **TTI, "Thinking vs Doing" (2506.07976).** Gemma-3-12B with online filtered BC (binary-reward REINFORCE, no negatives).
  - The horizon curriculum h = clip(10*i, 30) reaches WebVoyager 64.8 vs fixed h=10 59.1 vs fixed h=30 **45.2**; WebArena 26.1 vs 23.8 / 19.0.
  - At a fixed token budget, "acting more" scales better than "thinking more": budget forcing plateaus near 0.26, and forced extra thinking changed the action only about 15% of the time.
  - **Per-step reasoning shrank as interaction length grew** under training.

  Source: [TTI](https://arxiv.org/html/2506.07976v2).
- **GiGPO's anchor-state statistics** (section 1). Groups of size 10 or more make up over 20% at iteration 10 (early loops) and shrink later. **Seen elsewhere:** MARSHAL (length penalty helps game RL), Explore-More-Drift-Less and T2PO are in `finetune-research-2026-10-17/training.md` and were not re-fetched.

### Inferences
- **Trims map to segments.** Our harness trims at about 118k -> 59k tokens and gives each level a natural boundary, so ReSum-GRPO's segment-plus-broadcast scheme maps directly. Train each post-trim window as its own sample, carrying the level's (or the game's) advantage. This makes the prior plan's "train on trimmed windows" option principled rather than a hack. It also lets training samples stay at 8-16k tokens if we cut extra segment boundaries, but then the training context differs from the 55k served context (test-first).
- **Shorter replies come with the outcome reward.** TTI and RAGEN both show outcome-reward training shrinks per-step thinking. That is our R3 lever ("same decision, fewer tokens") arriving as a side effect of RS or RL, so R3 is the safest target. RAGEN-2's warning: shrinking can hide template collapse, so monitor input dependence and not only length or entropy.
- **No horizon curriculum for now.** A horizon curriculum (TTI, AgentGym-RL) matters if we ever run online RL: start with short level budgets. Not needed for RS/SFT.

### Gaps
- No paper trains through *mid-episode trims of raw transcript*. All use model-written summaries or states. Our trims drop turns and keep a memory card, which is closer to ReSum than to MEM1, but this is untested.

## 3. Generalization vs number and diversity of training environments

### Takeaway
Classic deep RL needs thousands of distinct levels to generalize (Procgen: up to 10,000). Pretrained LLMs transfer with far fewer environments (VHD-Play: 2,176 environments in 3 families). But **count alone doesn't help at fixed compute**: curated diversity and a difficulty curriculum beat "train on everything". Large models overfit small task pools (AdA). Fine-tuning a frontier-class model on 10 games cost it double-digit relative losses on AIME and GPQA (SIMA 2).

### Cited Findings
- **Procgen (1912.01588).** PPO trained on 100 to 100,000 levels for 200M steps "strongly overfit[s] to small training sets in almost all cases".
  - Closing the gap needs "as many as 10,000 levels". The 500-level protocol still shows "a high amount of overfitting".
  - Larger networks improve both sample efficiency and generalization (Sec. 4, Fig. 5).

  Source: [Procgen](https://ar5iv.labs.arxiv.org/html/1912.01588).
- **AdA (2301.07608).** XLand 2.0 has more than 10^40 tasks; the main runs use a 25B-task pool.
  - A 25B pool beats a 200M pool, "most pronounced for larger models", which are "especially prone to overfitting to a smaller task pool" (Sec. 3.5).
  - PLR curriculum beats no-op filtering, which beats uniform sampling.
  - Distillation matters: without kickstarting, a 265M model is worse than a 23M one; with 4B frames of distillation from the 23M teacher it is clearly better (Fig. 15).
  - Scale of training: 169M Transformer-XL (353M total), 100B steps. It adapts within trials on more than 80% of 1,000 held-out tasks.

  Source: [AdA](https://ar5iv.labs.arxiv.org/html/2301.07608).
- **SIMA 2 (2512.04797).** Gemini Flash-Lite fine-tuned with SFT then RLVR on **10 training environments**, with pretraining data mixed in.
  - Held-out MineDojo: task completion in 26 of 50 categories vs 2 for SIMA 1. On ASKA it is more than 10% above SIMA 1.
  - Gemini-set tasks and rewards enable self-improvement in new games.
  - **Capability loss (SFT / SFT+RL vs base): LiveCodeBench -4.0% / -8.4%, AIME -25.5% / -15.4%, GPQA -16.3% / -19.5%.**

  Source: [SIMA 2](https://arxiv.org/html/2512.04797v1).
- **"Beyond Simply Environment Scaling" (2608.03571).** Qwen3-VL-4B/8B, a pool of 200 multimodal environments.
  - At fixed compute, more environment types "does not monotonically improve" results and can decrease them.
  - Mixed multimodal training averages -10.7% vs single-environment training.
  - OOD relative gain: all 200 environments +7.5%, random 30 +2.2%, ability-coverage-selected 30 (AES) +40.3%, AES + difficulty curriculum +68.5% (Table 3).
  - Failure analysis names visual state extraction and world modelling as the bottlenecks.

  Source: [paper](https://arxiv.org/html/2608.03571v1).
- **Environment counts in other work:**

  | Work | Count | Notes | Source |
  |---|---|---|---|
  | Agent World Model (2602.10090, ICML 2026) | 1,000 synthetic code+DB environments | "Strong OOD generalization"; numbers not read | [abs](https://arxiv.org/abs/2602.10090) |
  | Agent-World (2604.18292) | 1,978 environments | Secondary summary, UNVERIFIED | — |
  | Kimi K2 | 3,000+ real MCP tools, 20,000+ synthetic tools, "thousands" of agents | Diversity is not measured against generalization | [Kimi K2 Sec. 3.1.1](https://arxiv.org/html/2507.20534) |
  | AgentScaler (2509.13311) | Not in abstract | Two-phase: general agentic skills, then domain specialization | [abs](https://arxiv.org/abs/2509.13311) |
  | Qwen3.8-Flash-Next | **No source found** | Its own RL environment counts are unknown | — |

### Inferences
- **Our game pool is tiny.** It is about 25 public + 25 DreamTeam + about 250 arc-interactive (many tiny) + about 60 sonpham (validation only). By Procgen/AdA standards that is very small, and a 125B model is the "large model that overfits small pools" case.
  - The defence is light, on-policy updates: VHD-Play's KL shift was about 0.09 nats per token; PAPRIKA used LR 1e-6.
  - Pick environments by mechanic coverage and difficulty, not by count (AES/HDC).
- **SIMA 2's losses are the forgetting warning for Flash-Next.** They came despite pretraining-data mixing. Our harness depends on general reasoning (sandbox code, rule inference). PAPRIKA, with *self-generated* data at a tiny LR, showed no regression. This argues again for self-generated replay and a general-reasoning regression gate.

### Gaps
- No LLM-agent paper gives a Procgen-style curve of held-out score against the number of training *games* for grid puzzles.
- AWM and Agent-World tables were not read.

## 4. Training on ARC-AGI-3-like games specifically

### Takeaway
I found no paper or public entry that fine-tunes an LLM on ARC-AGI-3-style games and reports held-out-game results. What exists:
- Trained **non-LLM** agents either learn per level from scratch (StochasticGoose resets its CNN each level) or fail (DreamerV3 and TRM near 0; cross-game pretraining no consistent benefit).
- The strong public-set results (61-99 RHAE) are all **frozen frontier models plus harness**.

### Cited Findings
- **StochasticGoose (Tufa Labs, preview winner, 12.58%, 18 levels).** A 4-layer CNN (32->256 channels) predicts *whether an action changes the frame*, trained with BCE plus entropy on a buffer of 200K deduplicated (state, action) pairs.
  - It does a **"Dynamic Reset: clears buffer and resets model when reaching new levels"**, i.e. test-time learning with no cross-game transfer.
  - Sources: [preview results](https://arcprize.org/competitions/arc-agi-3-preview-agents); `competitors/ARC3-solution/README.md:10-17,69-79`.
  - The prior notes cite DataCamp for a drop to about 0.25% on the full benchmark (secondary).
- **"Competence without Performance" (NeurIPS 2026 workshop).** DreamerV3 and a 7M TRM score 0.005 / 0.018 against human parity 1.0 across six games x 2 seeds. "Cross-game pretraining gave no consistent benefit."
  - On 21 of 25 games reward is identically zero over 500k steps.
  - A factored action interface triggers reward discovery on cd82 in 6 of 6 seeds vs 0 of 6. A cloned human prior is the other intervention that helps.
  - Source: [NeurIPS page](https://neurips.cc/virtual/2026/159692).
- **Stanford CS224R project.** A learned click-priority function on top of the Rudakov graph explorer, on vc33 only.
  - First-order MAML 4.42 vs baseline 0.59 (7.5x); PPO 0.54; MLP 0.04.
  - The authors conclude "level-to-level distribution shift ... is the dominant challenge", so fast adaptation beats globally optimized policies.
  - Source: [PDF](https://cs224r.stanford.edu/projects/pdfs/Kyle%20Avery%20Feinstein%20Steve%20Roy%20Mendeleev%20Ryan%20Bookman%20submission_416280185/CS_224R_Project_Paper-3.pdf), extended abstract.
- **Naestro-AGI3-27B** (QLoRA r16 on Qwen3.6-27B, procedural ARC-like simulations) has no scores. It is in the prior notes and remains UNVERIFIED.
- **AERA (2605.25931).** Qwen2.5-0.5B, 0.2116 RHAE on public games (4/25), 0.30 claimed on a 55-game private evaluation. The abstract doesn't say whether weights are trained. It also claims trivial reachability exploits, which are UNVERIFIED. Source: [abs](https://arxiv.org/abs/2605.25931).
- **Frozen-model harnesses:**
  - Rodionov (2607.15439): verification harness 74.78 RHAE with gpt-5.5 xhigh; model and effort mattered more than architecture (34.36 -> 72.52). Held-out games untested. [alphaXiv](https://www.alphaxiv.org/overview/2607.15439)
  - Twin (2608.14490): 93.3%, 23 of 25 games. The same base model scores 7.8% raw and 61.1% with an off-the-shelf harness. [abs](https://arxiv.org/abs/2608.14490)
  - Agentick (2605.06869): the reasoning harness multiplies LLM performance 3-10x; ASCII beats natural language. [abs](https://arxiv.org/abs/2605.06869)
- **Connect the Dots (2606.20002, v2 2026-09-20).** End-to-end RL on alternating solve / update-context rollouts, with Qwen3-8B and Qwen3.6-27B (v2 mentions Qwen3.6/3.8), code in Trinity-RFT. It claims cross-domain OOD generalization; the numbers are UNVERIFIED. Source: [abs](https://arxiv.org/abs/2606.20002).

### Inferences
- **ARC-AGI-3 is won in-context.** Cross-game weight learning has not been shown to help on ARC-AGI-3. Weights should learn the *meta-skill*: cheap probing, writing reflections that carry across levels (LaMer's reflection-only memory won), and short decisions. Game content can't be trained in.
- **Tufa's preview agent learned online per level.** Its LB-1 Kaggle method is undisclosed, so nothing here says Tufa fine-tunes.

### Gaps
- Tufa's and Yi-Chia Chen's methods (68.91 / 67.49).
- Any LLM fine-tune with held-out ARC-AGI-3 results.

## 5. Distilled practical recipe from the literature

### Takeaway
**SFT is a format warm start; RL or on-policy filtering does the generalizing.** Typical successful runs use 2-5x10^4 short episodes with group size 8-16, variance-filtered prompts, no KL, clip-higher, and multi-environment mixtures. The main failure modes have cheap monitors.

### Cited Findings
- **SFT vs RL ("SFT Memorizes, RL Generalizes", 2501.17161).** Llama-3.2-Vision-11B.
  - OOD rule variants: RL V-IRL-L 80.8 -> 91.8 vs SFT 80.8 -> **1.3**; GeneralPoints RL +3.5 vs SFT -8.1.
  - SFT was *needed* as a format warm start: RL from scratch failed.
  - "RL cannot recover OOD performance from an overfitted SFT checkpoint."
  - Source: [paper](https://arxiv.org/html/2501.17161).
- **Episode budgets in successful runs:**

  | Run | Episodes / samples | Model | Hardware |
  |---|---|---|---|
  | RAGEN | ~25.6k episodes (8x16x200) | 0.5B | — |
  | GiGPO | 19.2k episodes (128x150) | 1.5-7B | 2-4 H100 |
  | SPIRAL | 51.2k games (400x128) | 4B | 8 H100 x 25 h |
  | VHD-Play | 34.8k rollouts | 35B-A3B | not stated |
  | ReSum-GRPO | 32k rollouts (1K x 4 epochs x 8) | — | 8-16 x 144 GB |
  | PAPRIKA | 17k SFT trajectories + 5.3k pairs | 8B | — |

  Sources: sections 1-2 above.
- **Failure modes and fixes:**
  - Echo trap: reward-std collapse precedes reward collapse; keep high-variance prompts.
  - Template collapse: monitor input-output mutual information, not entropy; use SNR filtering.
  - Thinking collapse in self-play: per-role baselines (RAE).
  - Fixed-opponent degradation: SPIRAL.
  - Diversity loss under RL: LaMer keeps more of it.
  - Format-reward shortcuts: MEM1.
  - Long fixed horizons collapse (TTI h=30: 45.2); use a horizon curriculum.
  - Capability loss: SIMA 2.

  Sources: sections 1-3.

### Inferences
- **Our rejection-sampling round is TTI's "online filtered BC".** The prior plan's ReST-EM round is that algorithm, which TTI used successfully at 12B. It inherits the echo-trap risk, because it reinforces whatever template succeeded. Filter by level-state variance and mix data across games.

## Implications for our plan (Flash-Next, 15 days, one 8xH200 node, ~11 Kaggle GPU-h/day)

1. **Do: keep the data on-policy, as the prior plan says.** This is backed by the SFT-vs-RL evidence above (Chu, SPIRAL, ViGaL, Orak, MEM1). Use expert or teacher trajectories, if at all, only as a small format warm start. Never multi-epoch SFT on public-game logs: "RL can't recover an overfitted SFT checkpoint", and V-IRL OOD fell 80.8 -> 1.3 under SFT.
2. **Do: select training level-states by outcome variance (RAGEN/RAGEN-2, PAPRIKA, AdA PLR).** In-repo measurement (scripts `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/ce155f4d-514a-4385-ac65-a7e7e5a23e1c/scratchpad/w3/lvl2.py` over `runs/kaggle/*/*/score_timeline.jsonl`):
   - Across 34 full 25-game runs, 153 (game, level) cells were reached by 3 or more runs: 56 always cleared, 5 never, **92 intermediate**, and 36 had a conditional clear rate of 0.2-0.8.
   - Caveat: a "miss" can be the deadline, not inability.
   - After holding out 10 public games, only about 20 such frontier cells remain. That is two orders of magnitude below VHD-Play's 2,176 environments, so **external games are mandatory for diversity**. Measure the same per-level variance on DreamTeam and arc-interactive during collection.
3. **Do: train on trims as ReSum-style segments, each carrying the level's advantage or verification label.** Test first: whether training at ≤16k windows changes behaviour at 55k (no precedent at our context length).
4. **Do: make the primary target "same decision, fewer tokens" (R3).** RAGEN, TTI, Kimi-Researcher and the RSI test all show outcome-filtered training shortens thinking. Gate it with an input-dependence check (RAGEN-2): e.g. fraction of replies whose action or plan differs across different boards, not only reply length.
5. **Do not expect cross-game "content" transfer.**
   - Non-LLM cross-game pretraining did nothing on ARC-AGI-3.
   - Far-OOD gains in VHD-Play were only +0.16 to +0.24, against +0.63 near-OOD.
   - Train the interaction meta-skill: probe, verify, write the level recap. LaMer's reflection-only memory beat trajectories (56.4 vs 34.8 on Sokoban), which supports training the recap lines our harness already makes the model write.
6. **Do not run VHD-scale online RL before 11-02.**
   - Kaggle: 690 tok/s gives about 2.5M tokens/h, so about 27M tokens/day.
   - Our run spends about 43k tokens per cleared level (5.03M / 116 in mem-repair).
   - 34.8k level-rollouts at 20-40k tokens is 0.7-1.4B tokens, about 25-50 Kaggle-days, before H200 generation is counted.
   - One RS round of about 1-2k level attempts/day on Kaggle is feasible. This fits the prior plan.
7. **Test first: a general-reasoning regression gate.** SIMA 2 lost 15-25% relative on AIME and GPQA after game fine-tuning even with pretraining data mixed in. Use LR ≤1e-6-2e-6, one epoch, and self-generated replay (PAPRIKA: no regression at LR 1e-6).
8. **Curate the game mix by mechanic coverage plus a difficulty ramp.** "Beyond Simply Env Scaling": the curated 30 environments beat all 200 (OOD +40.3% vs +7.5%) at fixed compute. Mixing many visual environments naively averaged -10.7%.

## Fact-check (independent verifier, 2026-10-18)

Each claim was checked against the primary source: the arXiv HTML pages, fetched again on 2026-10-18, the local files, and the script rerun on `runs/kaggle/`. Verdicts: confirmed, partially confirmed (the gist holds but a detail is off), refuted, or unverifiable.

1. **VHD-Play training and transfer: confirmed.**
   - The training-implementation section (App. C.5) and Tab. 9 give GRPO with 64 prompts per step and 16 retained rollouts per prompt (oversampled to 18). LR is 2e-6, with no KL and no std normalization. The checkpoint is "34 optimizer steps, consumes 2,176 environments ... 34,816 scored rollouts".
   - Table 2 family macro (agentic) is 0.204 -> 0.815; written-out is 0.962 -> 0.992.
   - Sec. 5.1 gives held-out +0.56, near-OOD +0.63 and far-OOD +0.24 / +0.16.
   - Note: the paper never says "no SFT" in words. It applies GRPO directly to the starting checkpoint (labelled "Base") and describes no SFT stage.
2. **SFT vs RL (Chu et al., 2501.17161): partially confirmed.**
   - The numbers are exact (Sec. 5.2): V-IRL-L per-step accuracy goes 80.8 -> 91.8 under RL and 80.8 -> 1.3 under SFT. Both start from the same SFT-initialized checkpoint.
   - The "SFT is needed" finding is conditional. Sec. 5.4 shows RL without SFT fails only for the *base* Llama-3.2-Vision-11B, on GP-L, because it "does not follow instructions". The authors say this does not contradict DeepSeek-R1's finding that SFT is unnecessary. For an instruction-tuned model like Flash-Next, this paper does not show an SFT warm start is needed.
   - The "RL cannot recover" result (App., Fig. 19) is a single *extremely* overfitted checkpoint on V-IRL-VL, with under 1% initial OOD per-step accuracy. It is not shown for moderately overfit SFT.
3. **ReSum-GRPO: confirmed.**
   - Sec. 3.2: K summaries give K+1 segments, observations are masked, and the trajectory-level, group-normalized advantage is broadcast to every segment.
   - App. D.2: G=8, batch 64, LR 2e-6, 4 epochs, 1K SailorFog-QA samples.
   - Table 2, WebSailor-30B BrowseComp-zh Pass@1: ReAct 23.9, GRPO 23.3, training-free ReSum 29.3, ReSum-GRPO 33.3.
   - Note: the model is WebSailor-30B-**A3B**, an MoE. About 5.4 of the 9.4 points come from the training-free ReSum paradigm; training adds 29.3 -> 33.3.
4. **SIMA 2 capability loss: confirmed.**
   - Table 1, as relative change vs base Gemini (SFT / SFT+RL): AIME -25.5% / -15.4%, GPQA-Diamond -16.3% / -19.5%, LCB -4.0% / -8.4%.
   - Sec. 3.3: the base is Gemini Flash-Lite, and Gemini pretraining data is mixed into SFT.
   - Clarifications: "10 games" means 10 training *environments*, 3 research environments plus 7 commercial games. The paper calls the loss "minor" or "modest", and says RL added "no significant additional regression" over SFT. Most of the SFT data is human gameplay, so it is off-policy.
5. **PAPRIKA: confirmed.**
   - Setup (v4, 2025-10-31): 17,181 SFT trajectories and 5,260 RPO pairs after filtering, LR 1e-6 / 2e-7.
   - Leave-one-out improves 9 of 10 groups over the initial model. Mastermind gains little; Wordle shows no or negative transfer.
   - Table 2: MT-Bench 7.88 -> 8.14, GPQA 33.5 -> 32.8 (SE 1.5), MMLU-Pro 46.7 -> 46.2.
   - Clarifications:
     - The model is Llama-3.1-8B-**Instruct**, and v4 adds Gemma-3-12B-IT.
     - The 17,181 / 5,260 counts are for the full 10-group run. Each leave-one-out model trains on 9 groups.
     - The benchmark table evaluates the full PAPRIKA model, not the leave-one-out models.
     - The curriculum gain of 1.4 / 3.3 points (notes line 100) was measured on twenty questions only (Fig. 4).
6. **"Beyond Simply Environment Scaling": partially confirmed.**
   - Sec. 2.1 / Fig. 3 (Qwen3-VL-4B, nested environment sets, fixed compute) shows non-monotonic gains.
   - Table 3 (4B) gives OOD relative gain: AES-30 +40.3%, All Envs +7.5%, Random-30 +2.2%, AES+HDC +68.5%.
   - **Correction:** "All Envs" trains on **170** environments, the 200-environment pool minus the 30 held-out OOD environments (App. B), not on all 200. The same error is on notes lines 219 and 336.
   - Fixed compute means 7,680 training samples with GRPO group size 16 on 4 A100s.
   - The relative gains sit on small absolute numbers: OOD single-turn 13.8 -> 21.0 (AES) vs 17.6 (All).
   - At 8B the gap is +47.1% vs +15.9%.
7. **StochasticGoose: confirmed.**
   - `competitors/ARC3-solution/README.md:10-17` and `:69-79` match the claim: a 4-layer CNN predicts whether an action changes the frame, uses BCE plus entropy and a 200K deduplicated buffer, and does a "Dynamic Reset: Clears buffer and resets model when reaching new levels".
   - The code agrees. `custom_agents/action.py:340-354` clears the buffer and resets the model and optimizer when the score rises.
   - The arcprize.org preview page lists "1st Place, StochasticGoose @ Tufa Labs, Score: 12.58%, Levels Completed: 18".
8. **In-repo level-variance count: partially confirmed.**
   - Rerunning `lvl2.py` reproduces the claim exactly: 34 runs, 153 cells, 56 always cleared, 92 intermediate, 5 never cleared, 36 in 0.2-0.8.
   - The 34 runs are not 34 independent Flash-Next runs:
     - `v4-mem-recap-10x/v2` and `v4-mem-recap-10x-r2/v1` have byte-identical `score_timeline.jsonl` files (same md5; `runs/index.csv` shows the same 1433 / 49.55). It is one run archived twice.
     - `v4-q27-dflash2-10x/v1` served `qwen-qwen3-8-27b-fp8`, not Flash-Next.
   - Deduplicated (33 runs): 153 cells, 56 / 92 / 5, and **38** in 0.2-0.8, across 18 games.
   - Deduplicated and Flash-Next only (32 runs): 153 cells, 62 always / **86** intermediate / 5 never, and 38 in 0.2-0.8.
   - The "about 20 frontier cells after a 10-game holdout" estimate still holds (about 38 x 15/25 ≈ 23).

**Other errors and imprecisions found in the notes:**
- Line 13 (RAGEN): the paper's setup is "up to 5 turns and 10 actions". It sets no per-turn cap of 5 actions in the main configuration.
- Lines 195 and 213 (SIMA 2): calling it a "frontier-class model on 10 games" overstates both points. Flash-Lite is the small, latency-driven Gemini tier, and the 10 training environments include 3 research environments.
- Lines 219 and 336: "all 200 environments" should read "all 170 non-held-out environments" (see claim 6).
- Lines 287 and 319: "SFT was needed as a format warm start: RL from scratch failed" holds only for a base, non-instruction-following model (see claim 2). The quoted sentence "RL cannot recover OOD performance from an overfitted SFT checkpoint" paraphrases the paper. The paper's words are "RL cannot save overfitted checkpoints", and the test checkpoint was extremely overfitted.
- Line 99 (PAPRIKA): the model is Llama-3.1-8B-Instruct.
- Line 321: "34 full 25-game runs" includes one duplicated run and one Qwen3.8-27B run (see claim 8).
- Line 35 (GiGPO): the GiGPO paper text does not state the "history window of only 2 steps". It is `history_length: 2` in verl-agent's default trainer config, and the paper's prompts only say "most recent {history_length} observations".
- Line 43 (verl-agent): Qwen3-VL support came in 2025-12. The 2025-06 update added Qwen3 and LoRA.
- Lines 8 and 111: "Qwen authors / Qwen team" is loose. The paper lists Alibaba Token Foundry (Alibaba Group) and one Georgia Tech author.
- Spot checks found no errors in RAGEN Tables 1-5, the GiGPO results, SPIRAL Table 9, the ViGaL figures, LaMer, MEM1, TTI, the lmgame-Bench harness numbers or Orak Table 7, beyond the items above.
