# W8: Training exploration, hypothesis testing and world-model induction (learning to learn new games)

Researcher notes, 2026-10-18. Full texts were read from the arXiv HTML and checked against arXiv API metadata. Numbers rendered as math in the HTML were fetched separately or left out. "In-dist" means trained and tested on the same environment family; "unseen" means a held-out task family. This builds on `research/archive/reports/lit-2026-10-12/world-models.md` (harness-side world-model agents) and W3 (PAPRIKA basics, LaMer, VHD-Play) without repeating them.

**The in-repo fact that frames everything below.** The calibration on 21,901 decisions gives these numbers ([arms.md:185](../../../../arms.md)):
- Flash-Next already tries a new (state, action) pair on **91% of actions**, and **87%** of those reach a new board.
- On cleared levels, 81% of decisions lie on the hindsight-shortest route.
- On never-cleared levels "the shortfall is volume, not choice": 96% of decisions had an untried action that reached a new board, at ~37 s per action.

Our model is therefore not "greedy" in the sense of the bandit literature below. Novelty is not scarce. What is scarce is the *right* hypothesis about goal and mechanics, plus actions per hour.

## Q1. PAPRIKA ("Training a Generally Curious Agent") and its follow-ups

### Takeaway
PAPRIKA is still the cleanest evidence that fine-tuning on self-generated, filtered multi-task interaction data teaches transferable information gathering. Its recipe was SFT on successes, then preference pairs (fewest turns vs failed or long), at a tiny learning rate. The gains are moderate (+47% relative), uneven across held-out groups, and shown only on 8-12B non-reasoning instruct models. The strongest 2026 follow-up, ORBIT, shows that *cross-episode* meta-RL transfers to unseen tasks where ordinary single-episode RL does not.

### Cited Findings
**PAPRIKA** ([arXiv 2502.17543](https://arxiv.org/html/2502.17543v4); v1 2025-02-24, v4 2025-10-31).
- **Setup:**
  - Models: Llama-3.1-8B-Instruct and Gemma-3-12B-IT (§4).
  - 10 task groups; 6 have hard-coded program feedback (Wordle, Cellular automata, Mastermind, Battleship, Minesweeper, bandit) and 4 use an LLM as the environment (20Q, Guess-my-city, Customer service, Murder mystery) (Table 1).
  - Data was sampled with min-p 0.3 at temperature 1.5, 20 trajectories per task (§4).
- **Data scale:** 17,181 SFT trajectories and 5,260 DPO/RPO pairs (Appendix C, Table 4). Preferred = the success with the fewest turns; dispreferred = a random failure or a much longer success. Trajectories over 12,000 tokens were dropped (App. C).
- **Gain:** averaged over all 10 groups, +47% of the original success rate for Llama-3.1-8B "with only about 22,500 trajectories" (§4).
- **Transfer (leave-one-group-out):**
  - Improves 9 of 10 held-out groups.
  - Mastermind gain is "minimal"; Wordle shows "no transfer at all or negative transfer" (§4, Fig. 3).
  - Bandit: 42.25% → 62.25% without any bandit training (§4).
  - Full multi-group training beats single-group training on 7 of 10 groups (§4).
- **Efficiency and safety:** fewer turns on all groups (§4.1, Fig. 7); no general regression (MT-Bench 7.88 → 8.14, GPQA 33.5 → 32.8; Table 2).
- **Curriculum:** sampling tasks by "learning potential", the coefficient of variation σ/μ of the turns-to-solve reward, beat uniform sampling by +1.4 (average) and +3.3 (pass@4) after 3 rounds of 250 tasks (§3.4, §4).
- **Preference stage matters:** RPO with negatives improved on SFT-only on all 6 groups tested (App. I.9).
- **Generic multi-turn chat data hurts:** fine-tuning on 100k WildChat (GPT-4) trajectories "show[s] significant performance degradation on all task groups" (App. I.10).
- **Reasoning models are untested:** "DeepSeek-R1 distilled Llama-8B and Qwen-7B … we could not obtain a final answer from them … even after generating 10,000 tokens" (App. I.11).
- **Other:** the authors note the gains depend on base-model quality and diversity (§6). Training ran on 8× L40S (Llama) and 8× H100 (Gemma). Code is at github.com/tajwarfahim/paprika and datasets are on HF (App.).

**ORBIT: cross-episode meta-RL** ([arXiv 2602.04089](https://arxiv.org/html/2602.04089v1), 2026-02-03). Found through Semantic Scholar's list of papers citing PAPRIKA.
- **Environments:** train on RPS, Minesweeper, Hangman, Wordle, Blackjack; test on **unseen** Maze and Mastermind, 3 episodes per instance (Table 1).
- **Method:** GRPO with a binary completion reward over the multi-episode trajectory.
- **Model and budget:** Qwen3-8B with thinking on, 32k context; 100 optimization steps, group size 4, batch 64 trajectories, LR 1e-6, no KL or entropy terms, 8× H100 (§4.1, App.).
- **Results (episode-3 success, Table 2):**

  | Task | Base | Single-episode RL | ORBIT |
  |---|---|---|---|
  | Maze | 0.22 | 0.45 | 0.55 |
  | Mastermind | 0.27 | 0.21 (below base) | 0.59 |

- **Behaviour:** the single-episode RL baseline "quickly saturates or even degrades as episodes progress" (§4.3). The paper claims Qwen3-14B ORBIT "match[es] the performance of GPT-5.2"; that is shown only in a figure, with no numbers in the text.
- **Scaling:** the 4B → 8B → 14B gain is largest in later episodes (§4.5).

**LaMer** ([arXiv 2512.16848](https://arxiv.org/html/2512.16848v2)) is the same idea with GiGPO on Qwen3-4B. pass@3 improves, but pass@1 is lower on MineSweeper and WebShop (numbers in W3). **MAGE** ([arXiv 2603.03680](https://arxiv.org/abs/2603.03680)) applies multi-episode meta-RL to multi-agent games; I extracted no numbers.

### Inferences
- **Two signals transfer.** The ingredients with transfer evidence are (a) many task *families*, (b) self-generated data, (c) a contrastive signal (preference pairs or group RL), and (d) a reward on *later-attempt* performance (ORBIT). Our games are already multi-level, and levels share mechanics, so "level k+1 is easier because of level k" is exactly ORBIT's cross-episode signal. The plan's R2/R4 data filters can use it offline: favour segments where a level was cleared after an earlier failed or long attempt on the same level, or where later levels took fewer actions than the first.
- **Expect smaller gains on Flash-Next.** PAPRIKA's +47% relative was measured on non-reasoning 8B models whose baseline lacked strategy. Flash-Next is an RL-tuned reasoning model that already explores (91% new pairs).

### Gaps
- No PAPRIKA-style result on a reasoning model, on a >30B MoE, or with contexts beyond ~12k tokens (PAPRIKA) or 32k (ORBIT).
- ORBIT does not report hardware hours.
- No follow-up applies PAPRIKA or ORBIT to 2D grid games with hidden goals.

## Q2. In-context RL and exploration: what's wrong with LLM explorers and what training fixes

### Takeaway
On bandits and small games, LLMs fail by greediness and frequency bias even when they "know" the right algorithm. RL fine-tuning, a +1 untried-action bonus, or plain "try every action first" closes much of the gap. These failure modes are *coverage* failures, and our logs show Flash-Next does not have a coverage problem. The transferable lessons are about the *knowing-doing gap* and about harness-summarised history, not about novelty bonuses.

### Cited Findings
**"Can large language models explore in-context?"** ([arXiv 2403.15371](https://arxiv.org/html/2403.15371v3), 2024-03-22).
- Tested GPT-3.5, GPT-4 and Llama2 on bandits; "only a single configuration … results in satisfactory exploratory behavior" (§1).
- That configuration was GPT-4 plus a suggestive hint to explore, history externally summarised into per-arm averages, and zero-shot CoT (§1).
- All other configurations failed through "suffix failures" (never choosing the best arm after a few rounds) or uniform-like behaviour (§3).

**"LLMs are Greedy Agents"** ([arXiv 2504.16078](https://arxiv.org/html/2504.16078), 2025-04-22, Google DeepMind; Gemma2 2B/9B/27B).
- **Greediness:** coverage stagnates beyond ~10 steps, with 9B/27B covering about 6.5 of 10 arms (§4.2).
- **Frequency bias:** the 2B model repeats the most frequent action in context (§4.2).
- **Knowing-doing gap (27B):** 87% of rationales computed UCB correctly, yet on correct rationales the model picked the greedy action 58% of the time vs the optimal one 21% (§4.2).
- **RL fine-tuning on its own CoT** (30K updates, batch 128, 256-token budget, 1,792-token context; App. B): +12% action coverage on 2B; Tic-tac-toe return vs random 0.15 → 0.75 (§4.3).
- **Exploration mechanisms (§4.4):** "try-all" (every untried action first) gave "the biggest performance improvements", the 27B "almost clos[ing] the gap" to UCB. A +1 reward for untried actions "significantly increases exploration … and lowers regret".
- **Other ablations (§4.5):** a 512-token budget lifts 2B to 9B-RLFT level; SFT on 32K UCB rollouts matched UCB regret.

**EVOLvE** ([arXiv 2410.06238](https://arxiv.org/html/2410.06238v2), 2024-10-08, rev. 2025-07; Gemma-2B/9B, Gemini-1.5 Flash/Pro; pairwise win-rate, Tables 1-2).
- Fine-tuning on UCB trajectories lifted Flash on MAB from 27.7 to 65.6, above Pro's 45.5. Hand-written few-shot demonstrations *hurt* Pro (45.5 → 39.1).
- Algorithm-guided summary statistics helped large models (contextual bandits, Flash 0.0 → 46.4) but not Gemma-2B/9B (7.6 → 4.9, 10.5 → 4.1).

**"When Greedy Wins"** ([arXiv 2509.24923](https://arxiv.org/abs/2509.24923), 2025-09-29; Qwen2.5-3B/7B).
- SFT on UCB and RL with shaped rewards both reach UCB/Thompson-level regret and generalise to 6× longer horizons.
- RL generalises better out of distribution (§1).
- But "gains often stem from more sophisticated but greedier exploitation: RL/SFT agents are more prone to early catastrophic failure than pre-trained models, prematurely abandoning exploration" (abstract).

**Algorithm Distillation** ([arXiv 2210.14215](https://arxiv.org/abs/2210.14215), 2022): a transformer trained on cross-episode *learning histories* of an RL algorithm improves in context, and is more data-efficient than its source (abstract). It is the ancestor of ORBIT and LaMer, and needs many full histories per task family.

**Information-gain training.**
- BED-LLM ([arXiv 2508.21184](https://arxiv.org/abs/2508.21184), 2025-08-28) picks questions by expected information gain, estimated from the LLM's own predictive distribution.
- ASIG ([arXiv 2607.03426](https://arxiv.org/abs/2607.03426), 2026-07-03) trains that reward into a 7B policy with multi-turn GRPO: it "more than doubles" 20 Questions success and transfers to unseen MediQ (abstract). [arXiv 2610.20832](https://arxiv.org/abs/2610.20832) (2026-10-14) reports similar transfer.
- IGPO ([arXiv 2510.14967](https://arxiv.org/abs/2510.14967)) rewards each turn by the rise in the probability of a known answer.
- All of these need an explicit hypothesis space or answer string.

### Inferences
- **Novelty bonuses fix a problem we don't have.** The bandit literature's failure is low coverage. Flash-Next's 91% new-pair rate and 87% new-board rate ([arms.md:185](../../../../arms.md)) mean a novelty or untried-action bonus would be near-constant across rollouts and give almost no gradient. The prior plan's "novel-board bonus" for RL ([finetune-plan-2026-10-17.md:98](../../reports/finetune-plan-2026-10-17.md)) should be demoted unless a log check shows stuck levels have *lower* novelty than cleared ones.
- **The knowing-doing gap is our failure shape.** The rsi-proposals R4 note says "the evidence was on screen and the model held a wrong hypothesis". That is the knowing-doing gap. Schmied et al. fixed it with on-policy RL on the model's own CoT, not with expert SFT.
- **Keep harness summaries.** Krishnamurthy (summaries necessary) and EVOLvE (summaries help large models) match our finding that the model uses computed facts. Train with those facts in the prompt; R5 is an addition, not a replacement.
- **Watch for premature exploitation.** Shortest-route SFT (R3) and fewest-action preference pairs reward exploitation, and "When Greedy Wins" shows the failure appears after training.

### Gaps
- No in-context-RL study uses 64×64 grid observations, 55k-token prompts, or a reasoning model with 1-2k-token thoughts.
- Information-gain rewards have no hypothesis space in ARC-AGI-3 unless the model writes explicit hypotheses, and Flash-Next ignores optional structure ([arms.md:626](../../../../arms.md)).

## Q3. World-model learning: does training transition prediction improve play?

### Takeaway
Mostly no, when it is done as a separate prediction objective. Training to predict next states makes the model better at predicting next states; decisions improve only when the world-model signal is woven into policy learning on the model's own rollouts, and even then by a few points, in-dist. The best controlled ablation (Meta's CWM) shows execution-trace data leaving agentic SWE-bench unchanged while agentic interaction trajectories raised it.

### Cited Findings
**CWM, Meta FAIR** ([arXiv 2510.02387](https://arxiv.org/html/2510.02387v1), 2025-09-30; 32B dense, open weights).
- Mid-trained on Python execution traces and ForagerAgent agentic Docker trajectories (§2).
- In the small-scale mid-training ablation (§7.1, Table 4), SWE-bench Verified (SBV) pass@1 after SFT went:

  | Mid-training data | CruxEval-O | SBV pass@1 |
  |---|---|---|
  | none | 45.4 | 14.6 |
  | + PR data | 44.6 | 18.6 |
  | + execution traces | 73.9 | 18.4 |
  | + ForagerAgent trajectories | 74.5 | 22.1 |

- The authors' reading: trace data "significantly improves CruxEval-input and -output prediction but leaves all SBV-related metrics unaffected"; "only the addition of ForagerAgent data improves agentic SBV NLLs" (§7.1).

**RWML** ([arXiv 2602.05842](https://arxiv.org/html/2602.05842v2), 2026-02-05, Microsoft; Qwen2.5-7B on ALFWorld, Qwen3-8B on τ²-Bench). Results (Tables 1-2):

| Method | ALFWorld avg | τ²-Bench avg |
|---|---|---|
| ReAct | 13.0 | — |
| World-model SFT alone | **2.8** (collapse) | — |
| Policy RL | 81.0 | 38.0 |
| World-model SFT + Policy RL | 80.4 | 40.3 |
| RWML (embedding-similarity reward on predicted vs real next state) + Policy RL | 87.9 | 43.7 |
| Imitation of expert data | 82.5 | — |

- Next-state SFT "prioritizes token-level fidelity … and can lead to model collapse" (abstract).
- RWML forgets less on MMLU-Redux, MATH-500, GPQA and LiveCodeBench than world-model SFT (§3.3, Table 3).

**Early Experience, Meta** ([arXiv 2510.08558](https://arxiv.org/html/2510.08558v3), 2025-10-09; Llama-3.2-3B, Qwen-2.5-7B, Llama-3.1-8B, up to 70B).
- "Implicit world modelling" (IWM) is next-state prediction on the agent's own alternative actions (8 non-expert actions per expert state, App. C.1), run as stage 1 *before* expert SFT (§4.2).
- IWM vs imitation: ALFWorld/ScienceWorld +2.3 to +5.5, WebShop +11.3 to +18.4. Self-reflection: TravelPlanner +12.8 to +15.0 (§5.2).
- Out-of-domain gains sometimes meet or exceed in-domain gains (§5.3), and post-RL ceilings are higher (§5.4).
- Caveat: every result is relative to imitation of *expert* demonstrations.

**PaW** ([arXiv 2606.02388](https://arxiv.org/html/2606.02388v2), 2026-06-01).
- Method: an auxiliary next-observation loss on the policy's own RL rollouts, selected by action entropy and balanced against the RL loss.
- ALFWorld GRPO: 1.5B 70.0 → 77.9; 7B 77.6 → 80.6.
- GiGPO 7B: 90.8 → 91.8.
- WebShop success: 1.5B 60.6 → 68.6 (Table 1).
- Other models: Qwen2.5-14B GRPO 74.7 → 77.1; Llama3.2-3B 4.0 → 62.2 (a GRPO run that failed without it) (§4.3 table).
- All in-dist.

**VAGEN** ([arXiv 2510.16907](https://arxiv.org/html/2510.16907v1), 2025-10-19; Qwen2.5-VL-3B).
- Method: RL with structured StateEstimation and TransitionModeling reasoning plus a turn-level world-modelling reward.
- Overall score over 5 tasks: 0.21 base → 0.76 (structured reasoning) → 0.82 (with world-modelling reward and Bi-Level GAE).
- Sokoban: 0.14 → 0.61 → 0.79 (Table 5).
- Trained per task (in-dist).
- Natural-language state descriptions beat structured or symbolic ones on Sokoban and FrozenLake (Table 4).

**MAP, Map-then-Act** ([arXiv 2605.13037](https://arxiv.org/html/2605.13037v1), 2026-05-13).
- Fine-tuned Qwen3-4B-Thinking with ms-swift on 8× H800 on MAP-2K, which is 2K exploration-first trajectories distilled from GPT-4.1/Claude 4.5 (§4.1, App. B.2).
- It beat the same model trained on 2K expert *execution* traces: ALFWorld 94.1 vs 84.3, TextCraft 95.6 vs 79.4, ScienceWorld 40.5 vs 23.6 (MAP column, Table 1).
- Its ARC-AGI-3 result uses Claude 4.6 Opus prompting only: ReAct ≈ 0 vs MAP 2-4 levels on 6 shown games, with improvement on "22 out of 25 games" (§4.2, Table 2). No fine-tuned model was run on ARC-AGI-3.

**AutumnBench / WorldTest** ([arXiv 2510.19788](https://arxiv.org/html/2510.19788v3), v1 2025-10-22, v4 2026-05-07, ICML 2026).
- 43 grid-world environments, 517 humans vs Claude 4 Sonnet, Gemini 2.5 Pro and o3. "Humans outperformed all reasoning models across all environments" (§5.2); in 18 of 43 environments more compute gave no benefit (§5.2.1).
- Models "do not treat resets as special actions, unlike humans", who use resets as experiments, and show "a narrow view of what counts as informative actions" (§5.2.2, §5.3).

**ARC-AGI-3 world-model agents.**
- New since the 10-12 notes: Kepler, 100.00 RHAE with Opus 5 at 858M tokens / US$778 ([arXiv 2610.00834](https://arxiv.org/abs/2610.00834)); Schema, RHAE "58.7% to 99.2% with the same base model" ([arXiv 2609.39140](https://arxiv.org/abs/2609.39140)); VISTA, Opus 5 40.68 → 100 ([arXiv 2610.02200](https://arxiv.org/abs/2610.02200)).
- All are frontier and train no weights. So does DreamTeam's 2605.09650 ([arms.md:657](../../../../arms.md)).
- Rodionov ([arXiv 2610.22289](https://arxiv.org/abs/2610.22289), 2026-10-15): moving *reset decisions* to a scripted controller recovered part of a specialised agent's advantage.

**In-repo evidence.** Model-written simulators go unused when optional (0 of 10 games defined `simulate`), world-model notes appear in ~1% of turns, and the Tycho port scored 3.53 vs 29.22 ([arms.md:626](../../../../arms.md)). A forced restart doesn't beat continuing ([finetune-plan-2026-10-17.md:98](../../reports/finetune-plan-2026-10-17.md)).

### Inferences
- **Don't make transition prediction a standalone SFT target.** CWM (trace data, no agentic gain) and RWML (world-model SFT + RL ≈ RL; world-model SFT alone collapses) are the two controlled tests, and both say it doesn't improve decisions. This confirms the prior "M3 not before 11-02" call ([rsi-proposals-2026-10-15.md:224](../../reports/rsi-proposals-2026-10-15.md)) with new evidence.
- **Any world-model signal should be a small auxiliary on our own rollouts** (Early Experience/PaW style, high-entropy decisions only). Its target should be a compact board diff (objects moved or recoloured, level change), not the ~4k-cell grid; that is a token-fidelity target of the kind RWML ties to collapse. Expected size in the literature: +2 to +8 points in-dist, never shown on a held-out game family.
- **Train on interaction, not answers.** MAP-2K vs ACT and CWM's Forager result agree: trajectories that *include exploration and the agent's own interaction* beat expert execution traces. Scripted-solver trajectories should carry a probing prefix, as `data.md:64` already says. Better still, use the model's own successful explorations.

### Gaps
- No paper trains an open model's world model on ARC-AGI-3 or PuzzleScript-like games and measures held-out play.
- MAP-4B was never run on ARC-AGI-3.
- No result covers predicted-diff targets on grid frames.

## Q4. Goal inference: inverse RL, hindsight relabelling, goal-conditioned tuning

### Takeaway
Hindsight relabelling reliably turns failed trajectories into extra supervised or preference data, with +7-11% over success-only SFT on task-disjoint held-out sets. But every method relabels to an *explicit natural-language goal*. In ARC-AGI-3 the goal is hidden, so the useful analogue is a rationalisation step: using the level-clear transition to write, at the stuck state, the reasoning that infers the win condition (R4). No paper tests "train to infer a hidden goal from observation" for LLM agents on games.

### Cited Findings
- **AgentHER** ([arXiv 2603.21357](https://arxiv.org/html/2603.21357v4), 2026-03-22, rev. 05-10) relabels failed trajectories to goals they *did* achieve, with confidence gating.
  - On task-disjoint held-out WebArena/ToolBench: +7.6 to +11.4% over success-only SFT across GPT-4o, Qwen2.5-72B/7B and Llama-3.1-8B; gains compound over 4 rounds (+10.4%).
  - Cross-model judging cuts label noise from 5.9% to 2.9%. Cost: US$2.98 and 26 min per 3,000 trajectories (abstract, §1).
- **HSL, "Spinning Straw into Gold"** ([arXiv 2607.04235](https://arxiv.org/html/2607.04235v1), 2026-07-05; ICLR 2026 poster):
  - Setup: Llama-3.2-1B agent, Llama-3.3-70B relabeller (§5).
  - Method: relabels each rollout with *all* goals actually achieved, adding irrelevant-action masking and sample reweighting.
  - Improves SFT and DPO, more on long-horizon, many-goal ALFWorld.
  - Beats full-data baselines using one quarter of the demonstrations (abstract).
- **ECHO** ([arXiv 2510.10304](https://arxiv.org/html/2510.10304v2), 2025-10-11; prompting only): the LM rewrites failed trajectories for achievable sub-goals and stores them. On a stateful XMiniGrid: +80% over ReAct, +42% over the next-best baseline (§1).
- **GRACE** ([arXiv 2510.02180](https://arxiv.org/abs/2510.02180), 2025-10-02): an LLM evolves *code* reward functions from expert trajectories (BabyAI, MuJoCo, AndroidWorld). This is goal inference as program synthesis (abstract).
- **ARC-AGI-3 harness evidence (frontier):** Twin infers the goal before reward on 87.2% of levels (`research/archive/reports/lit-2026-10-12/world-models.md`, A5). The ARC-AGI-3 work does not separate goal inference from mechanics.

### Inferences
- **For R4, filter by replay, not by judge.** Treat the engine-verified level-clear frame as the "achieved goal"; write a hinted rationale at the earlier stuck state and keep it only if its actions replay toward the clear. AgentHER and HSL suggest the filtering and masking matter more than the generator. Relabelling is cheap; the cost is rollouts.
- **Hindsight sub-goals as an auxiliary (test-first).** An uncleared level still "achieved" something, such as an object moved onto a marker. "What did this sequence achieve?", with achievements defined as harness-computed board diffs, is a game-agnostic goal-conditioned task. It is untested on grids.

### Gaps
- No ARC-AGI-3 or grid-game study of training goal inference.
- No measurement of how often our stalls are *goal* errors vs *mechanic* errors. The su15 example in [rsi-proposals-2026-10-15.md:176](../../reports/rsi-proposals-2026-10-15.md) is a mechanic error.

## Q5. Curiosity and novelty bonuses in LLM-agent RL, and their effect on unseen games

### Takeaway
Novelty, count and curiosity bonuses help when the policy under-covers the state space: bandits, text adventures, ScienceWorld. The benefits are mostly in-dist. Our model over-covers rather than under-covers, so a bonus is a weak lever for us. The better-supported exploration signals are task-progress-relative-to-prior-attempts (RetroAgent), memory-driven exploration (EMPO2) and explicit explore phases (Look Before You Leap, MAP).

### Cited Findings
- **Schmied et al.:** a +1 bonus for untried actions narrows the gap to UCB on bandits (§4.4, App. C) (see Q2).
- **Look Before You Leap** ([arXiv 2605.16143](https://arxiv.org/abs/2605.16143), 2026-05-15): task-only RL gives "narrow and repetitive behaviors"; interleaved exploration rollouts rewarded by "Exploration Checkpoint Coverage" fix it (abstract). +2.2 to +4.1 points on 4-8B models (from `lit-2026-10-12/world-models.md` §C; not re-verified today).
- **EMPO2** ([arXiv 2602.23008](https://arxiv.org/abs/2602.23008), 2026-02-26):
  - Memory-augmented exploration with mixed on- and off-policy updates.
  - "+128.6% and 11.3% improvements over GRPO" on ScienceWorld and WebShop; adapts to new tasks with only a few trials (abstract).
- **RetroAgent** ([arXiv 2603.08561](https://arxiv.org/abs/2603.08561), v6 2026-06-09):
  - An intrinsic reward for "incremental subtask progress relative to prior attempts" plus distilled lessons.
  - Over GRPO: ALFWorld +18.3%, WebShop +15.4%, Sokoban +27.1%, MineSweeper +8.9% (abstract; base models not given there).
- **XTX** ([arXiv 2201.01251](https://arxiv.org/abs/2201.01251), 2022, pre-LLM): imitate promising past trajectories to the frontier, then explore. +27% / +11% normalised score on 12 Jericho games (abstract).
- **CDE** ([arXiv 2509.09675](https://arxiv.org/abs/2509.09675)) and **MERCI** ([arXiv 2510.16614](https://arxiv.org/abs/2510.16614)) are curiosity or pseudo-count bonuses for *math* RLVR (CDE about +3 AIME per search snippet; not read). Neither has game results.
- **AutumnBench:** reasoning models' exploration is narrow and ignores resets (§5.3) (see Q3).

### Inferences
- **Reward progress over the model's own earlier attempt, not novelty.** If RL ever runs on our games, use a RetroAgent-style term: per-level progress relative to the model's earlier attempt (fewer actions to the same achieved diff, or a level reached sooner). This is game-agnostic.
- **XTX is the harness half.** "Return to the frontier, then explore" matches level-path and solved-route memory. Those are harness features we already have, not a training target.

### Gaps
- No curiosity-bonus result on unseen game families with an open model above 14B.
- Not checked: whether our stuck levels differ from cleared levels in novelty rate. This is the test that decides whether any novelty term is worth adding.

## Implications for our plan

Constraints: 15 days, one ~8×H200 node, ~11 Kaggle GPU-h/day, game-agnostic code only.

1. **Don't:**
   - **A world-model training stage.** That covers transition-prediction SFT, world-model GRPO (M3), and a separate code-world-model head. Controlled evidence says it improves prediction, not decisions: CWM Table 4 (SBV 18.6 → 18.4 with traces), RWML Table 1 (world-model SFT + RL 80.4 vs RL 81.0; world-model SFT alone 2.8). Flash-Next also ignores optional simulators ([arms.md:626](../../../../arms.md)).
   - **A novelty/curiosity bonus as the core exploration fix.** Our model already takes new (state, action) pairs on 91% of actions ([arms.md:185](../../../../arms.md)). Demote the "novel-board bonus" in the prior plan's stage 3.
2. **Do (data selection, no new infrastructure):**
   - **Make R2/R4 filters cross-attempt (the ORBIT/LaMer signal in offline form).** Weight segments where a level was cleared after an earlier failed or long attempt, and where later levels of a game cost fewer actions than earlier ones. ORBIT shows that cross-episode reward transfers to unseen tasks (Mastermind 0.27 → 0.59) where single-episode RL went *below* base (0.21).
   - **Use PAPRIKA's selection rules:** same-level preference pairs (fewest-actions clear vs a failure or a ≥2× longer clear; App. C), levels picked by σ/μ across our ~110 runs of each game, a preference stage after SFT (RPO beat SFT-only on 6/6 groups; App. I.9), and LR ~1e-6 (PAPRIKA, ORBIT).
3. **Do (cheap replay that preserves exploration):** fill part of the 30-50% replay with Flash-Next's own rollouts on the 6 programmatic PAPRIKA groups (Wordle, Mastermind, Battleship, Minesweeper, Cellular automata, bandit; code is public).
   - The rollouts are short, verifiable and cheap on Kaggle quota.
   - Do *not* use PAPRIKA's released Llama data (off-policy) or generic chat; WildChat SFT degraded every group (App. I.10).
   - It is a guard, not the main signal: Wordle showed no or negative transfer, and transfer to grids is unproven.
4. **Test first: R4 hindsight rationales** with AgentHER/HSL-style controls.
   - Use the engine-verified clear as the "achieved goal", keep only rationales whose actions replay toward it, mask irrelevant actions, and hold out whole games.
   - Metric: right-move rate on frozen stuck states of held-out games, plus tokens per decision. Expect single-digit gains (AgentHER: +7.6-11.4% over success-only SFT, with explicit goals).
5. **Test first: an auxiliary board-diff prediction loss** (Early Experience/PaW style) at a small mixture weight, only on high-entropy decisions, predicting object-level diffs.
   Only if the SFT pipeline works by about day 7; measure on held-out stuck states before any Kaggle run (literature: +2-8 in-dist, nothing on unseen families).
6. **Guard against greedier exploitation after training.** Shortest-route SFT (R3) and fewest-action preferences reward exploitation, and "When Greedy Wins" shows trained agents abandon exploration early.
   - Add an eval metric: the share of stuck levels where the fine-tuned model stops trying new (state, action) pairs, or repeats the last failing plan, compared with base.
   - Reject a checkpoint whose new-pair rate on stuck levels falls well below base's rate (91% over all actions; measure base on stuck levels first).
7. **Scale check:** the transfer results used ~6.4k (ORBIT) to ~22.5k (PAPRIKA) trajectories of 12k-32k tokens on 4-14B models. At ~80 GPU-h per 1k game attempts (prior plan §4) and ~11 GPU-h/day, Kaggle yields at most ~1.9k new attempts before 11-02 (11 GPU-h × 14 days ÷ 80 GPU-h per 1k), with nothing left for evaluation. So exploration data must come from the ~180k logged calls plus targeted resamples at frozen stuck states. Online meta-RL on Flash-Next stays out of scope before 11-02.

## Fact-check (independent verifier, 2026-10-18)

Each claim was checked against the arXiv HTML full text (same versions as cited) or the local file at the cited line. No browser was used. Verdicts: **confirmed** (exact), **partially confirmed** (gist holds, detail off), **refuted**, **unverifiable**.

1. **Flash-Next exploration rates (91% new pairs, 87% new board; 96% untried new-board action on never-cleared levels, ~37 s/action).** **Confirmed** against `arms.md`. Two small points:
   - The 91% and 87% figures are on `arms.md:183`. Only the 96% and ~37 s figures are on `:185`.
   - The calibration covers 21,901 decisions from 4 user-combo-10x-family runs on 24 games, pooled over cleared and stuck levels. The new-pair rate on stuck levels alone is not given (the notes already list this gap).
2. **CWM Table 4 (CruxEval-O 44.6 → 73.9 with traces; SBV 18.6 → 18.4; ForagerAgent → 22.1).** **Confirmed**: every number matches Table 4, §7.1. Caveats for how the plan uses it:
   - The ablation used **8B models** (6T pre-training plus 1T mid-training tokens, SFT, no RL), not the released 32B CWM.
   - It is a single run per data mix, and the rows are cumulative (PRs, then + traces, then + Forager).
   - CWM itself counts ForagerAgent as world-model data: the model "predict[s] both agent and environment turns" (observation loss is masked on 50% of turns, §2). So Table 4 shows that execution-trace prediction in a different format did not transfer to agentic SBV, while agentic-interaction data that includes observation prediction did help. That is evidence against a *standalone, off-format* transition target. It is not evidence that world-model data never helps decisions.
3. **RWML ALFWorld (WM SFT 2.8, ReAct 13.0, WM SFT + RL 80.4, RL 81.0, RWML + RL 87.9).** **Confirmed**: all match Table 1 (Qwen2.5-7B-Instruct, mean of 3 runs). The caveats limit the "no benefit as RL pretraining" reading to the ALFWorld average:
   - On τ²-Bench (Qwen3-8B), WM SFT + Policy RL scored **40.3 vs 38.0** for Policy RL alone. WM SFT alone scored 27.9 vs ReAct 31.9, a drop rather than a collapse.
   - On ALFWorld OOD, WM SFT + RL was **82.3 vs 79.2**. Only the in-distribution split (76.2 vs 82.1) drags the average down.
   - The notes' table leaves blanks that the paper fills: τ² ReAct (Qwen3-8B) is 31.9, and τ² imitation is 43.7, which ties RWML + RL.
   - arXiv lists v2 as 09 Feb 2026.
4. **ORBIT (Qwen3-8B, 100 GRPO steps, batch 64, group 4, LR 1e-6, 8×H100, 32k; Maze 0.55 vs base 0.22 vs RL 0.45; Mastermind 0.59 vs 0.27 vs 0.21).** **Confirmed**. Table 2 prints only the RL and ORBIT values plus Δ vs base. The base values 0.22 and 0.27 are implied by those deltas.
   - The training setup is in §4.2, not §4.1 (§4.1 covers environments), and 8×H100 is in the appendix.
   - Each test set has 256 instances, from one training run. No seeds or CIs are reported.
   - The reward is the *number* of successful completions across the 3 episodes, not a binary reward (the Q1 "binary completion reward" wording is slightly off).
5. **PAPRIKA (17,181 SFT and 5,260 RPO pairs; LOO improves 9/10; Wordle no or negative transfer; RPO > SFT-only on 6/6; WildChat 100k degraded all).** **Confirmed** (§4, App. C Table 4, App. I.9, I.10). Details that matter for the plan:
   - The counts are for Llama-3.1-8B. Gemma-3-12B's statistics differ slightly.
   - Mastermind was sampled at 100 trajectories per task, not 20.
   - App. C uses **one** preference pair per task, because multiple pairs per task caused "unintentional unalignment". The dispreferred trajectory is a random lower-performing one, a failure or a success that took "a lot more turns". There is no 2× threshold; the "≥2× longer clear" in Implications §2 is our rule, not App. C's.
   - DPO trajectories were capped at 8,192 tokens (SFT at 12,000).
   - The RPO learning rate was **2e-7** (SFT 1e-6), so "LR ~1e-6" in Implications §2 applies to SFT only.
   - The curriculum result (+1.4 / +3.3) is from twenty questions only.
   - App. J (omitted in the notes): on the standard Krishnamurthy bandit, PAPRIKA's gain vanishes as the number of arms grows.
6. **Schmied et al. (Gemma2-27B: 87% correct UCB rationales, greedy 58% vs optimal 21%; +1 untried-action bonus increases exploration and lowers regret).** **Confirmed** (§4.2, Fig. 5; §4.4).
   - The paper's own introduction says 64% greedy. The 58% in §4.2 and Fig. 5 is the figure the notes use.
   - The bonus raised action coverage from about 50% to 70% on the MAB setup.
   - The bonus is detailed in App. B.4, not App. C.
   - The Tic-tac-toe 0.15 → 0.75 result is in §4.5, not §4.3.
   - The paper does **not** show that novelty bonuses help "only where coverage is low": it tested only low-coverage bandits. That conclusion is our inference.
7. **MAP-4B vs ACT-4B (ALFWorld 94.1 vs 84.3, TextCraft 95.6 vs 79.4, ScienceWorld 40.5 vs 23.6, MAP columns).** **Confirmed** (Table 1; §4.1: ms-swift, 8×H800, LR 1e-5, 3 epochs). Two caveats for the plan's use:
   - The advantage holds in every inference column. In the ReAct columns it is 87.1 vs 78.2, 91.2 vs 70.1 and 35.0 vs 16.9.
   - MAP-2K is in-distribution: 667 trajectories from each of the three benchmarks' training splits. It was distilled from GPT-4.1 and Claude 4.5 under a goal-free exploration prompt, then corrected and filtered against the engine's ground-truth state (App. B.2). The source of ACT-4B's 2K expert traces is not described. The result therefore supports teacher-distilled exploration traces over execution traces in-distribution. It does not by itself show transfer to unseen games.

**Other checks.** All 30 other arXiv IDs cited in the notes resolve to papers with matching titles and dates, including Kepler, Schema, VISTA, Rodionov 2610.22289, ASIG and 2610.20832. Spot-checked numbers match their sources:
- Kepler: 100.00 RHAE, 858.0M tokens, $777.72.
- Schema: 58.7% → 99.2%.
- VISTA: 40.68 → 100.
- EVOLvE: Tables 1-2.
- Early Experience: §5.2 ranges.
- PaW: Table 1 and §4.3.
- VAGEN: 0.21 / 0.76 / 0.82 and Sokoban 0.79.
- AutumnBench: 517 humans, 18 of 43 environments with no compute benefit, the reset findings.
- Krishnamurthy: the "single configuration" quote.
- The local references: `arms.md:626`, `arms.md:657`, `finetune-plan-2026-10-17.md:98` and its stage 3 and §4 (80 GPU-h per 1k attempts), `rsi-proposals-2026-10-15.md:224` and `:176`, `data.md:64`, and Twin 87.2% and Look Before You Leap +2.2 to +4.1 in `lit-2026-10-12/world-models.md`.

Minor issues:
- AutumnBench v3's body evaluates five models (it adds Gemini 2.5 Flash and Qwen3-235B), not three.
- In Q3, "CWM (trace data, no agentic gain) … both say it doesn't improve decisions" overstates CWM; see claim 2.
