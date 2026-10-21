# W9: Synthetic environment generation, curricula and generalization scaling for training on ARC-AGI-3-like games (2026-10-18)

Researcher W9. Scope: how to get many ARC-AGI-3-like training games beyond the 25 public ones quickly; how many and how diverse they must be; curricula; acceptance checks; cost and licences.

This builds on:
- `research/data/finetune-research-2026-10-17/data.md` (source inventory);
- `L1-data-inventory.md`;
- W3 §3 (Procgen, AdA, SIMA 2, "Beyond Simply Environment Scaling");
- W2 §5 (pass-rate filtering).

It does not repeat their numbers except where it corrects or extends them. Local stats come from static reads of the competitor folders; no game code was run. The scripts were one-off Python calls on `metadata.json` and `GAMES.md`.

## Q1. Where can ARC-AGI-3-like environments come from, and how good is LLM generation today?

### Takeaway
LLM-written executable environments work at scale for tool-use agents: 1,000 environments at about $0.57 each. Multi-level *puzzle games* with verified solvability are much harder. A one-shot LLM yields 5-20% fully solvable games; an agentic generate-verify loop keeps 60-80%. DreamTeam's `game_creator` is the only ARC-AGI-3-native generator: Apache-2.0, an agentic loop that requires a replayed winning policy for every level. Its yield and cost are not published, and it has only been run with proprietary models.

### Cited Findings
**General environment synthesis**
- **Agent World Model (AWM, 2602.10090, ICML 2026)** made 1,000 executable environments and 10,000 tasks with GPT-5.
  - Cost: **$57.09 per 100 environments** in total. Per-stage success: env code 86.8%, database 88.3% ([HTML Table 1](https://arxiv.org/html/2602.10090v3)).
  - With other generators (Table 7, 100 environments): Claude-4.5-Sonnet reaches 99.0% env-code success. **The open-weights Qwen3.5-122B-A10B reaches 77.0%**, "usable but bug-heavier".
  - Even GPT-5 environments had bugs in 74% of cases (Table 7). Pipeline stages allowed up to 10% errors to cut cost (§3.3).
- **AutoEnv (2511.19304)** used Claude-4-Sonnet with coding agents, a YAML DSL and a 40-step self-repair loop. Of 100 themes, 90.0% passed execution, 96.7% level generation and 74.7% a reliability check, so **65% survived overall** (60% fully automated, 80% human-reviewed).
  - Cost: **$4.12 per environment**.
  - Its reliability check is differential testing: if a weaker model consistently outscores a stronger one, the reward is treated as broken.
  - Up to 15 levels are tried per environment and only validated ones are kept: 358 levels across 36 environments.
  - Inverting the meaning of on-screen semantics alone cut performance 68.8%. Source: [HTML](https://arxiv.org/html/2511.19304v2), validation and benchmark sections.
- **ScriptDoctor (2506.06524)** has an LLM write PuzzleScript games and BFS-playtests them (up to 1M nodes), counting a level as solved only if its solution is longer than 10 moves.
  - **All levels solvable: 7% (GPT-4o), 7% (o1), 20% (o3-mini).** Compile rate: 67-93% (Table II).
  - Without few-shot examples, 0% of games were solvable (Table I).
  - No cost or diversity metric is reported. Source: [HTML](https://arxiv.org/html/2506.06524v1).
- **OMNI-EPIC (2405.15568, ICLR 2025).**
  - An LLM proposes tasks; another writes PyBullet code, with up to 5 compile-repair rounds.
  - An LLM-written `get_success` function judges success. It agreed with humans 72.7% of the time.
  - A model of interestingness filters out tasks too close to the archive.
  - It trains one DreamerV3 specialist per task. The authors "do not provide evidence of their generalization across tasks."
  - Source: [HTML](https://arxiv.org/html/2405.15568).
- **InternBootcamp (2508.08636).**
  - DeepSeek-R1 evolved 100 hand-written task classes ("bootcamps") into more than 1,000 candidates; 704 were kept after filtering.
  - Filter: drop tasks where a reference model (R1-Distill-Qwen-32B) scores above 0.85 (too simple) or below 0.03 (likely wrong).
  - In iteration 1 the generator "simplified" tasks in 97.93% of its thinking traces. Three refinement rounds brought this to 32.46% (Tables 1-2, §3.3).
  - Source: [HTML](https://arxiv.org/html/2508.08636v1).
- **RLVE (2511.07317, ICML 2026)** built 400 environments **by hand**. The authors tried "employing frontier LMs to perform automatic environment engineering" but found "maintaining high environment quality without human intervention proved nontrivial" (Discussion, "Model-based automatic environment engineering"). Source: [HTML v2](https://arxiv.org/html/2511.07317v2).
- **Agent-generated environments in self-play.**
  - SPADE (2608.19197, Aug 2026, work in progress): one LLM is both a regret-driven environment designer and the agent. It reports +5.3 average on 8 held-out benchmarks over the best fixed-environment baseline ([abs](https://arxiv.org/abs/2608.19197)).
  - Envs-FORGE (2608.14312): synthesizes 100 environments aimed at a pass-rate frontier. Qwen3.5-35B on tb-core goes 40.0 → 49.2, which is +2.4 over the best fixed-recipe generator ([abs](https://arxiv.org/abs/2608.14312)).
  - Absolute Zero (2505.03335), R-Zero (2508.05004, +6.49 math and +7.54 general for Qwen3-4B-Base) and SPICE (2510.24684, +8.9% math and +9.8% general) all self-propose single-turn questions, not interactive games ([AZR](https://arxiv.org/abs/2505.03335), [R-Zero](https://arxiv.org/html/2508.05004v4), [SPICE](https://arxiv.org/abs/2510.24684)).
- **Puzzle/game suites (single-turn or text, not ARC-like):**
  - [Enigmata](https://arxiv.org/html/2505.19914v2): 36 tasks, 400 samples per task (App. B.2).
  - [SynLogic](https://arxiv.org/abs/2505.19641): 35 tasks.
  - [Reasoning Gym](https://arxiv.org/abs/2505.24760): more than 100 generators.
  - [KORGym](https://arxiv.org/abs/2505.14552): more than 50 games, evaluation only.
  - Prime Intellect Environments Hub: "2,500+" environments, from a secondary source only (UNVERIFIED) ([rl-list](https://www.rl-list.com/vendors/prime-intellect)).
  - Kimi K2: see W3 §3.

**ARC-AGI-3-native sources (local, static reads)**
- **DreamTeam `game_creator`.**
  - Loop: 5 phases (design, schema, implement, verify with a winning policy, consistency check). "Every level must print `solved=True`" (`competitors/dream-team/arc_agi_3/game_creator/skill.jinja:545`).
  - Budget: depth 75 rounds and a 30-minute timeout per game (`.../game_creator/config.yaml:5,7`).
  - Diversity is sampled over 15 genres × 12 cognitive skills × feature tags × 15 palettes (`.../game_creator/README.md:46-124`). An "Action Variety Rule" rejects monotone solutions (`skill.jinja:72-78`).
  - Generator models offered: GPT-5.5 (default), Opus, and "flash" = Gemini 3 Flash Preview (`README.md:27`, `arc_agi_3/llm_configs.py:78-81`).
  - Its paper (2605.09650) does not mention the game creator; I grepped the HTML for "game creator" and "generated games" with 0 hits. **No yield or cost data exists.**
- **The 25 shipped DreamTeam games carry their solver in code.**
  - L1 is right that there is no `solution.json`. But **25 of 25** `.py` files define `solve_level`, e.g. `environment_files_generated/al7306/000afcc7/al7306.py:819`.
  - `baseline_actions` equal the solver's lengths: al7306 level 0 is 9 actions in both `solve_level(0)` and `metadata.json:8-9`.
  - So the solver's action counts serve as the RHAE baseline, not a human's.
- **The DreamTeam games are much shorter than the public games (from metadata):**

  | | Median baseline actions per game | Range | Median per level | Levels |
  |---|---|---|---|---|
  | DreamTeam (solver baseline) | 145 | 53-345 | 18.1 | 8 |
  | Public (human second-best baseline) | 638 | 171-1,843 | 79.8 | 6-10 |

  Action mix: DreamTeam is 14 click, 8 keyboard, 3 mixed; the public games are 13 mixed, 7 click, 4 keyboard.
- **arc-interactive (MIT).** `GAMES.md` lists 249 games in 150 two-letter families.
  - Grids: 203 of 249 have a grid of 16 or less.
  - Levels: 192 list 5 levels, 52 list 7, 2 list 6 and none 8 or more.
  - Its metadata disagrees: the median baseline is 3.5 actions per level, and only 27 of 252 `baseline_actions` arrays have 6 or more entries.
  - **It ships other versions of three public games:** `environment_files/ft09/9ab2447a`, `vc33/9851e02b`, `ls20/cb3b57cc`. Their SHA-256s differ from our public engines (ft09 `37493d87…` vs `aa5b54f4…`). `wm01.py:260,270` copies "ls20 pattern" code.
- **sonpham `arc-agi-3`.** Its 62 games were written by Claude Opus 4.6/4.7: there are 70 "Author: Claude" entries, e.g. `competitors/arc-agi-3/CHANGELOG.md:9`. There is no licence.

### Inferences
- **Our fastest proven path to new ARC-native games is `game_creator`.** It already has the right acceptance test (a replayed per-level winning policy). The unknown is yield with a model we can legally and cheaply run.
  - ScriptDoctor (one-shot, 7-20% fully solvable) and AutoEnv (agentic, 65%) bracket what to expect. Games are harder than tool environments: AWM's 77% for Qwen3.5 is an upper bound.
- **The generator simplifies its own tasks.** InternBootcamp saw this in 98% of first-pass traces, and DreamTeam's games are about 4x shorter than the public ones. Expect generated games to skew easy, and filter on difficulty for our model, not on the generator's opinion.
- **Of the existing pool, only DreamTeam's 25 and roughly 25-55 arc-interactive games are both multi-level (6+) and not toy-sized.** That is about 50-80 usable families before generation.

### Gaps
- No yield, token or cost figures for `game_creator` with any model; never run here.
- No paper generates ARC-AGI-3-style games with an LLM and trains on them (W3 §4 found none either).
- AWM Fig. 4 and RLVE Fig. 5 values were only described in text; exact points were not extracted.

## Q2. How many and how diverse must training environments be for transfer to unseen games?

### Takeaway
For LLM agents trained with RL, the published threshold is clear:
- About 10 environments overfit and can *degrade* held-out performance.
- About 100 give large gains.
- Several hundred keep improving.

At fixed compute, **coverage-selected diversity beats raw count**: 30 well-chosen environments beat all 200. Conflicting environments erase the gain. ARC says the private set shares few mechanics with the public set, so diversity must be across mechanics, not levels.

### Cited Findings
- **AWM §6.4, Fig. 4** (Qwen3 4B/8B/14B, multi-turn RL, 3 OOD benchmarks): "Training on 10 environments leads to severe performance degradation, likely due to overfitting to the limited distribution. Scaling to 100 environments yields substantial gains … Further scaling to 526 environments continues to improve agent performance."
  - Training: 526 environments, 3,315 tasks (about 6 per environment), 96 steps, LR 7e-7, batch 64 × 16 rollouts (§5, Implementation Details).
  - Source: [HTML](https://arxiv.org/html/2602.10090v3).
- **RLVE §4.2, Fig. 5.** Nested training collections of 1, 4, 16 and 256 environments, tested on **50 held-out disjoint environments** (2,500 problems). Expanding the collection "consistently leads to better performance on held-out environments" for all four models.
  - On a saturated 1.5B model: about 1,100 H100-h of RLVE gives +3.37 average vs +0.49 for 3,600 H100-h of continued RLVR (§5.1).
  - Source: [HTML v2](https://arxiv.org/html/2511.07317v2).
- **InternBootcamp §4.2, Fig. 5** (Qwen2.5-7B, DAPO-like RL): near-linear gains from 8 to 512 tasks. In the 8-task setting "the roll-out process collapses after 300 steps".
  - Some tasks unlearnable alone were learned inside a 512-task mix (§4.2.2).
  - **Caveat:** the training tasks may share categories with Bootcamp-Eval; only instances were deduplicated (§4.1). So this is not a held-out-task test.
  - Source: [HTML](https://arxiv.org/html/2508.08636v1).
- **"Beyond Simply Environment Scaling" (2608.03571), Tables 3-4** (Qwen3-VL-4B, OOD relative gain):

  | Training set | OOD relative gain |
  |---|---|
  | All 200 environments | +7.5% |
  | Random 30 | +2.2% |
  | Ability-coverage-selected 30 (AES) | +40.3% |
  | AES without redundancy control | +25.3% |
  | AES without conflict control | **+2.8%** |
  | AES + difficulty curriculum (HDC) | +68.5% |

  - Its curriculum has an outer axis of harness weakening: from full harness (H0) to bare (H4) (§4.1).
  - Source: [HTML](https://arxiv.org/html/2608.03571v1).
- **AutoEnv:** the best single fixed learning method gained +3.0 on 36 environments (Gemini-2.5-Flash), vs about +8 on a 6-environment subset (DeepSeek-V3.1). The models differ, and these are in-context learning methods, not weight training ([HTML](https://arxiv.org/html/2511.19304v2) §5).
- **Classic RL, for scale contrast:**
  - Procgen needs about 10,000 levels per game, and AdA's large models overfit small pools (W3 lines 198-213).
  - [Kinetix](https://arxiv.org/abs/2410.23208): an agent trained on "tens of millions" of generated physics tasks zero-shot solves unseen human-designed ones (numbers not extracted).
- **ARC-AGI-3 §3.6:** the private set is "intentionally out-of-distribution relative to the public set … with limited overlap with the mechanics found in the public environments". The public set "does not comprehensively represent the mechanics found in the private set". Source: [HTML](https://arxiv.org/html/2603.24621v1).

### Inferences
- **Target at least 100 distinct mechanic families in training,** each with several levels. Ten to thirty games is the overfit or degrade regime (AWM 10, InternBootcamp 8).
  - A 125B model is the case AdA flags as most prone to overfitting a small pool.
  - With about 50-80 usable external families today, **we need roughly 50-150 newly generated, verified games**. Thousands are not needed: LLM priors replace Procgen-scale counts.
- **Select for coverage and against conflict.** "Conflict control" was worth most of AES's gain.
  - Two conflicts are ARC-specific: games that reuse a colour or icon with a fixed meaning, which ARC forbids as "cultural conventions", and near-duplicate mechanics.
  - Diversify the action mode toward mixed keyboard+click, which is 13 of 25 public games but only 3 of 25 DreamTeam games.
- **Don't expect mechanic content to transfer** (§3.6). Transfer has to come from meta-skills: probing, hypothesis testing, shorter correct thinking. Diversity's job is to stop the model learning game-specific shortcuts while it learns those.

### Gaps
- No curve of held-out score against the number of training *games* exists for grid games or ARC-AGI-3 (also W3 §3).
- All the LLM-agent scaling evidence is at 4-14B dense or 35B-A3B. None is on a 125B-total/6B-active hybrid like Flash-Next.

## Q3. Curricula: what to sample, and when

### Takeaway
Every working LLM-RL recipe filters to prompts or levels where the model sometimes succeeds. Theory and experiment put the sweet spot near a 50% pass rate. Regret- or learning-progress-based level selection (PLR/ACCEL/SEC) beats uniform sampling. Adversarial environment design (PAIRED/ACCEL/SPADE) is the strongest but needs an online generator in the loop, which we can't afford in 15 days.

### Cited Findings
- **DAPO dynamic sampling:** "over-sample and filter out prompts with the accuracy equal to 1 and 0", keeping a fixed number of effective prompts (§3.2) ([HTML](https://arxiv.org/html/2503.14476)).
- **Online difficulty filtering (2504.03380, EACL 2026):**
  - Keeps prompts with 0.25 ≤ pass rate ≤ 0.75.
  - Shows the KL lower bound to the optimal policy is maximised near 0.5.
  - Gains over plain GRPO: +10% AIME and +4.2 average at 3B; +12% AMC and +4.5 at 7B.
  - Source: [HTML v2](https://arxiv.org/html/2504.03380v2).
- **RLVE adaptive difficulty** (per-environment difficulty that tracks the policy) outperformed static [0,20] difficulty on 256 environments (Fig. 4) ([HTML v2](https://arxiv.org/html/2511.07317v2) §4.1).
- **Self-Evolving Curriculum (2505.14970, COLM 2026)** treats each problem category as a bandit arm, with absolute advantage as the learning-gain reward and a TD(0) update. It improves OOD generalization and the balance across domains ([HTML](https://arxiv.org/html/2505.14970v3)).
- **PLR (2010.03934, ICML 2021)** replays levels by TD-error learning potential. Combined with the prior SOTA it reaches "over 76% improvement in test return" on Procgen ([abs](https://arxiv.org/abs/2010.03934v4)).
- **ACCEL (2203.01302)** edits high-regret levels. On held-out mazes its IQM "approaches a perfect solved rate compared to below 80% for the next best method, PLR", with an 80.2% probability of improvement (§5, Fig. 4) ([HTML](https://arxiv.org/html/2203.01302v3)).
- **Generation-side difficulty bands:**
  - InternBootcamp: 0.03-0.85 reference accuracy.
  - Enigmata: Easy/Medium/Hard by GPT-4o pass@1/10/100, with balanced 1:1:1 mixing best (Table 6).
  - AES+HDC: a curriculum on top of selection took the gain from +40.3% to +68.5% (above).
- **Our own data (W2 §5, W3 implication 2):** full-game episodes have near-zero group variance. The usable unit is a level or a frozen decision state with 0 < p < 1, and only about 20 such public cells remain after a 10-game holdout.

### Inferences
- **Do: PLR-lite over (game, level) cells.** Weight sampling by p(1−p), from 2-4 attempts per game by our model. Re-estimate after each ReST round. Drop cells that are always or never cleared.
  - This is what DAPO, online filtering and SEC do. It needs no online generator.
- **Optional: ACCEL-style level edits.** Mutate a verified game's `LEVELS` and re-verify with its solver. This adds levels, not mechanics, so it is only worth doing for games with fewer than 6 levels.
- **Don't: PAIRED/SPADE/OMNI-EPIC-style co-evolving generation before 11-02.** Each designer step costs a generation run, a verification run and rollouts.
- **Consider harness weakening as the outer curriculum** (rich hints, then the bare prompt), using our default-off memory and hint flags.

### Gaps
- No LLM paper applies PLR to multi-level interactive puzzle games specifically.

## Q4. What makes a generated game ARC-AGI-3-like, and how to check it

### Takeaway
The ARC-AGI-3 paper gives an explicit, mostly automatable acceptance spec:
- core-knowledge-only visuals with no numbers, letters, clip-art or colour conventions;
- six or more levels, with an easy tutorial first level;
- multiple mechanics that compose;
- novelty by a program-compression test;
- random-play gates: 50k steps beat no level, and 1M steps beat no non-tutorial level, with a random solve probability of 1 in 10,000 or less per level;
- graph-based state-space exploration;
- 2 of 10 first-time humans solving it.

The docs site has no overfitting guidance. The paper calls training on synthetic lookalikes "domain-specific overfitting".

### Cited Findings
All from [arXiv 2603.24621 HTML](https://arxiv.org/html/2603.24621v1); I grepped the quotes in the fetched text.
- **§3.4 design constraints:**
  - Core Knowledge priors only (objectness, geometry/topology, basic physics, agentness).
  - "Environments never use numbers, letters, recognizable real-world clip-art … or cultural conventions (like green meaning 'go')".
  - "Tutorial level: … intentionally easy". Later levels "require the accumulation and integration of concepts learned earlier".
  - "Environments centered on a single mechanic that scaled in size or difficulty are treated as an anti-pattern". "At least six levels per environment".
  - Novelty test: a single program solving two environments "at least 50% shorter than the concatenation of two independent solution programs" means they are "insufficiently distinct".
- **§3.5.1 random-play regimes:** "up to 50,000 steps and asserts that no level can be beaten by accident". Then 1,000,000 steps, where "non-tutorial levels must remain unbeaten". Then a third 1M-step fuzzing sweep.
- **§3.5.2:** an explicit hashed directed graph of reachable states. "a random policy should not successfully solve a level more often than 1 in 10,000 times".
- **§5:** 10 testers per candidate; kept if 2 or more solve all levels on first sight; 414 candidates, 486 participants.
- **§4.3:** agents are cut off at 5x human actions per level.
- **§4.3.1:**
  - "Task-specific overfitting" covers agents "directly trained on" public environments. Public-set evaluation is "emphatically not a valid measure of progress towards AGI".
  - "Domain-specific overfitting" covers agents "trained on many synthetically generated ARC-AGI-3 lookalike environments".
  - Kaggle rules don't forbid either (W5b §4; W5 implication 9).
- **docs.arcprize.org:** the `llms.txt` index (69 lines, fetched 2026-10-18) has game-creation pages (`add_game.md`, `edit_games.md`) and scoring (`methodology.md`), but no page on overfitting or the public/private split ([index](https://docs.arcprize.org/llms.txt)).
- **Ready-made solvability tooling:**
  - DreamTeam's phase-4 `solve_level` replay (above).
  - ScriptDoctor's BFS to 1M nodes.
  - AutoEnv's differential test: a weaker model consistently beating a stronger one flags a broken reward.

### Inferences
**An acceptance pipeline for our generated games**, cheapest check first:
1. **Static:**
   - 6 or more levels;
   - 2 or more `active_tags` introduced across levels;
   - no glyph or digit sprites (check that render code draws no text-like bitmaps);
   - no public-game ids, names or code in the prompt or output.
2. **Solver replay:** WIN on every level, as in DreamTeam phase 4. Record solver length as the optimal-length proxy.
3. **Random-play gate:** at least 50k random steps with no non-tutorial level won.
   - ARC's engine targets 1,000 fps (§3.3), so 50k steps is about 1 CPU-min. ARC's full 1M is about 17 CPU-min per game (estimate, unmeasured on DreamTeam's adapter).
4. **Small-state BFS where feasible:** check the solver length is within about 1.5x of BFS optimal, and that level 0 is short.
5. **Novelty:**
   - reject a game whose (genre, skill, tag-set, action-mode) tuple and code-embedding nearest neighbour both match an accepted game;
   - always reject the nearest neighbour of any public game, by code similarity or tag tuple.

   ARC's compression test is the ideal but isn't automatable cheaply.
6. **Difficulty for us:** 2 attempts by Flash-Next in our harness. Keep games with at least one level at 0 < p < 1, or with partial progress.
7. **Human spot-check:** the user plays 5-10 games.

**Contamination guards:**
- exclude by id prefix, not only SHA-256 (arc-interactive's ft09, vc33 and ls20 are other versions with different hashes);
- drop Training-Data's ARC-native set and sonpham's public-id versions;
- keep the public games entirely out of the generator's context.

### Gaps
- Not measured: how often `game_creator` games violate the no-numbers or no-letters rule, or are single-mechanic.
- Not measured: random-play throughput of DreamTeam's `arcengine_adapter`.
- Human solvability of generated games is unknown.

## Q5. Cost, generator licences, and trajectories per game

### Takeaway
At published rates, 200-1,000 verified environments cost about $100-6,000 with a frontier API. But the obvious API generators (OpenAI, Anthropic) have terms against using outputs to train models. Generating with an open-weights model on our own node avoids that. Flash-Next itself is the simplest choice, and AWM shows an open Qwen3.5-122B works at 77% code success on easier tool environments. Per-game trajectory needs are modest: game count, not samples per game, is the binding constraint.

### Cited Findings
- **Published costs:**
  - AWM: about $0.57 per environment (GPT-5).
  - AutoEnv: $4.12 per environment at 65% survival, so about $6.3 per kept environment (Claude-4-Sonnet).
  - DreamTeam: up to 30 min and 75 rounds per game, no cost figure.
  - Sources: above, and `game_creator/config.yaml:5,7`.
- **Anthropic.**
  - Its Usage Policy prohibits: "Use outputs to train an AI model (e.g., 'model scraping' or 'model distillation') without prior authorization from Anthropic". The fetched page reads "Effective November 12, 2026", an odd future date, so recheck it ([AUP](https://www.anthropic.com/legal/aup)).
  - Commercial Terms D.4 bar using the Services "to build a competing product or service, including to train competing AI models" (effective June 17, 2025) ([terms](https://www.anthropic.com/legal/commercial-terms)).
- **OpenAI** forbids using "Output to develop models that compete with OpenAI". The primary page returned 403, so this is from a secondary source and UNVERIFIED ([secondary](https://ilikekillnerds.com/2023/11/15/openais-api-december-2023-terms-of-use-update/)). Google and xAI reportedly have similar clauses (secondary, UNVERIFIED: [bioethics.com/Atlantic](https://bioethics.com/archives/101968)).
- **Licences of existing sources:**
  - DreamTeam games: Apache-2.0 from NVIDIA, generated with proprietary LLMs (README).
  - arc-interactive: MIT.
  - sonpham: no licence, Claude-authored.
  - Prize rules require released weights and open code (W5b §4; prior plan §5).
- **Trajectory budgets in comparable work:**
  - AWM: about 6 tasks per environment over 526 environments, 96 RL steps.
  - Enigmata: 400 samples per task.
  - Episode totals in successful agent-RL runs: 2-5×10^4 (W3 §5).
  - Our rollout cost: about 140 frozen decision states per Kaggle GPU-h at k=8 (W2 §5). About 80 GPU-h per 1k full public-game attempts, with external games probably 2-5x cheaper (`finetune-research-2026-10-17/data.md` §3).

### Inferences
- **Generator choice:**
  - Use Flash-Next (licence allows derivatives, per the project brief) or another Apache/MIT open-weights coder, served on the Azure node while it would otherwise sit idle in days 1-3.
  - Don't use Claude or GPT to author *training* games without written authorization. This isn't legal advice; flag it to the user.
  - Keep sonpham's Claude-written games for validation only.
- **Back-of-envelope generation cost (UNVERIFIED estimate):**
  - Assume about 0.2-0.4M generated tokens per `game_creator` attempt (75 rounds of code and thinking).
  - At our measured ~690 tok/s aggregate on one RTX PRO 6000, that is 5-10 GPU-min per attempt. At 30-60% yield, that is about 2-7 kept games per GPU-h.
  - So 150 kept games is about 20-75 RTX-PRO-6000-hours, or fewer on 8×H200.
- **Difficulty-screening cost:** 2 attempts × 150 games is about 300 game attempts, roughly 5-24 Kaggle GPU-h (at 2-5x cheaper than public games). That is 1-2 days of quota.
- **Per-game sample cap:** limit each game to about 2% of training states, so roughly 50-200 verified decision states per game across 100-150 games, or 10-20k states. That matches the published 2-5×10^4 episode scale. Beyond that, extra samples per game buy less than extra games.

### Gaps
- Flash-Next's ability as a game generator is untested.
- Primary-source OpenAI and Gemini terms were not read.
- Real token counts per `game_creator` run are unknown.

## Implications for our plan (15 days, one 8×H200 node, ~11 Kaggle GPU-h/day, game/time-agnostic code)

1. **Test first (days 1-2): `game_creator` yield with Flash-Next on 20 seeds.**
   - Measure: the fraction passing solver replay on all 8 levels plus the random-play gate; generated tokens per attempt; and 5 games hand-checked by the user.
   - Kill criteria: yield under 25%, or more than 1 GPU-h per kept game. Then fall back to existing games plus ACCEL-style level edits.
   - Precedents: ScriptDoctor (7-20% one-shot), AutoEnv (65% agentic), AWM-Qwen3.5 (77% on easier tool environments).
2. **Do: aim for at least 100 training mechanic families,** not thousands of levels and not 25 games.
   - Pool: DreamTeam 25, about 25-55 qualifying arc-interactive games, and 50-150 new generated games.
   - Select for coverage: vary genre, skill and action mode, and push mixed keyboard+click toward the public mix.
   - Remove conflicting or near-duplicate games (AES: conflict control was worth +40.3 → +2.8).
   - Fewer than about 30 games is the documented overfit or degrade regime (AWM, InternBootcamp).
3. **Do: implement the ARC §3.4/§3.5 acceptance spec as a script** (static checks, solver replay, a 50k-step random gate, small-state BFS, novelty, difficulty). It is cheap CPU work and keeps junk out.
   - **Execute generated code only in a sandbox.** DreamTeam's README warns that it runs generated Python.
4. **Do: run a PLR-lite curriculum.** Sample (game, level) cells by p(1−p) from 2-4 Flash-Next attempts per game, keep 0.25 ≤ p ≤ 0.75 where possible, and re-score after each ReST round. **Don't** build an online adversarial generator (PAIRED/SPADE/OMNI-EPIC) before 11-02.
5. **Do: fix contamination now.**
   - Exclude arc-interactive `ft09/9ab2447a`, `vc33/9851e02b` and `ls20/cb3b57cc` (different SHA-256s from our public engines, so a SHA-only filter misses them).
   - Exclude Training-Data's ARC-native set and sonpham's public-id versions. Never put public games in generator prompts.
   - Hold out 15-20 generated or external games by family as the dev set, alongside the prior plan's 10 held-out public games.
6. **Watch out: DreamTeam RHAE baselines are solver lengths,** about 18 actions per level vs about 80 for the public human baselines. Scores there will look harsh. Compare checkpoints paired on the same games, and report levels cleared and wins first.
7. **Don't: train on sonpham** (no licence, Claude-authored), **and don't use Claude or GPT outputs as training environments** without authorization. Keep unlicensed sets in validation.
8. **Expect meta-skill transfer only.** ARC §3.6 says the private mechanics barely overlap the public ones, so generated games are a diversity regulariser for R3 (shorter correct thinking) and R4 (stall-breaking), not a source of private-set content. Gate on held-out-family gains before spending node time (consistent with W3/W5).

## Fact-check (independent verifier, 2026-10-18)

Method: I re-downloaded each arXiv HTML page, plus the v2 pages of 2608.03571 and 2603.24621, which give the same numbers. I also fetched the live Anthropic AUP, its linked "Previous Version" and the Commercial Terms, and repeated the local reads. For the DreamTeam solvers I ran each `solve_level` with a stub for `arcengine_adapter` (no game engine) and compared the lengths with `metadata.json`. No browser was used.

**Claims**
1. **AWM: 10 environments degrade, 100 give substantial gains, 526 keep improving (Qwen3 4B/8B/14B).** *Partially confirmed.*
   - The §6.4 quote is verbatim.
   - The model sizes are wrong. Fig. 4's caption reads "Scaling curve of AWM over environment quantity with the 4B model", so the 10/100/526 curve is Qwen3-4B only. 8B and 14B appear in the main results, not in the scaling curve.
   - The text gives no numeric points.
2. **AES 30 = +40.3% OOD vs all 200 = +7.5%; without conflict control +2.8%; AES+HDC +68.5%.** *Confirmed (4B).*
   - All the numbers match Table 3 (Qwen3-VL-4B) and Table 4.
   - "OOD" means held-out environments (OOD-ST/MT). Gains on general benchmarks (MathVision/MMMU/MMStar) are only 0.1-1.5%.
   - The ablation sets are not 30 environments: the variant without conflict control used 34, the one without redundancy control 41.
   - At 8B the figures are: all 200 +15.9%, AES +47.1%, AES+HDC +73.7%.
   - AES was selected using 4B trajectories and gradients.
3. **ARC §3.6 private set OOD with "limited overlap"; §4.3.1 "domain-specific overfitting".** *Confirmed.*
   - The quotes are verbatim.
   - The overfitting categories are defined for the official leaderboard (API models, no harness), which "will seek to discount" such gains. They are not Kaggle rules.
4. **ARC acceptance constraints (six levels, single-mechanic anti-pattern, no numbers/letters/clip-art, 50% novelty test, 50k/1M random gates, ≤1 in 10,000).** *Confirmed.*
   - Every item is in §3.4 (headed "Game design constraints" in v1 and "Environment design constraints" in v2), §3.5.1 or §3.5.2.
   - Nuance: §3.4 accepts that random agents "occasionally stumble into success" on the tutorial level, yet §3.5.1's 50k regime says "no level can be beaten by accident". Our pipeline step 3 (no *non-tutorial* level at 50k) is looser than the paper's 50k gate.
   - The 1-in-10,000 threshold sits under §3.5.2, "Exploring graph-based state space construction …".
5. **ScriptDoctor "one-shot" 7%/7%/20% all-solvable; AutoEnv 65% at $4.12.** *Partially confirmed.*
   - The numbers are right. Table II gives all-solvable rates of 7/7/20% and compile rates of 67/87/93%. Table 2 gives AutoEnv 65.0% overall at $4.12.
   - ScriptDoctor is **not one-shot**. §III gives the LLM 10 attempts per game, with compiler errors, grammar (CFG) parse errors and BFS solver feedback, few-shot human games and a 10k-token context.
   - "All solvable" means every level has a BFS solution longer than 10 moves within 1M nodes.
   - The samples are tiny. Table II's caption says 10 games per model; the percentages fit 15 (1/15 = 7%).
   - AutoEnv's 65% includes 25 human-reviewed themes (80%, $3.48). The 75 fully automated themes kept **60.0% at $4.33**.
   - Table 2 only says "per environment"; it doesn't say whether that means per attempt or per kept environment, so the derived "$6.3 per kept environment" is an assumption.
   - So the bracket is a weak 2025 PuzzleScript repair loop (7-20%) against an agentic Claude-4-Sonnet loop (60% automated), not one-shot against agentic.
6. **All 25 DreamTeam games define `solve_level`; `baseline_actions` = solver counts; medians 145 vs 638.** *Partially confirmed.*
   - Confirmed: 25 of 25 define `solve_level`; al7306 level 0 is 9 actions (`al7306.py:819`, `metadata.json:8-9`); the medians 145 (53-345) and 638 (171-1,843) reproduce.
   - But `baseline_actions` equal the solver's lengths in only **20 of 25** games:
     - dl4827: level 7 is 30 vs 29.
     - ml2048: levels 2 and 4 are one action longer than the baseline.
     - sf2048: 7 of 8 levels are 1-2 actions longer.
     - ll4821 and sl4821 have a state-dependent `solve_level(level, h_t, z_t)`, so they can't be compared statically.
   - The baselines are solver-derived (possibly replay counts) but don't always equal the code's action list.
7. **arc-interactive ships other versions of ft09/vc33/ls20 with different SHA-256s.** *Confirmed.*
   - SHA-256 (arc-interactive vs public engine): ft09 `37493d87…` vs `aa5b54f4…`; vc33 `0af7a3c0…` vs `8afa9f55…`; ls20 `20a503d4…` vs `298c810d…`.
   - The ft09 copy differs by one line (it lacks `available_actions=[6]`).
   - The vc33 and ls20 copies differ in about 3,000 diff lines, carry their own baselines and were downloaded on 2026-03-17/18.
   - `wm01`, `wm02` and `wm03` also reference ls20.
8. **Anthropic AUP: "Use outputs to train an AI model … without prior authorization".** *Confirmed.*
   - The quote is verbatim under "Do Not Abuse Our Platform".
   - The odd date has an explanation: the live page is a version published ahead of its 2026-11-12 effective date.
   - Its "Previous Version" link (effective 2025-09-15, in force today) reads "Utilization of inputs and outputs to train an AI model (e.g., 'model scraping' or 'model distillation') without prior authorization from Anthropic". So the restriction applies under both versions.
   - Commercial Terms D.4 (effective June 17, 2025) is also verbatim.

**Other errors in these notes**
- Line 23: "Even GPT-5 environments had bugs in 74% of cases (Table 7)" is wrong. Table 7 gives 83% for GPT-5, 89% for Claude-4.5 and 100% for Qwen3.5. The 74% comes from Table 6 and the Limitations section (AWM environments judged by GPT-5.1).
- Lines 22, 85 and 292: Qwen3.5-122B's 77.0% is the env-code stage only. Its database stage was 79.0% and its sample-data stage 97.0%, and 100% of its environments had bugs. End to end that is about 59%, so 77% overstates it as a yield precedent.
- Lines 16, 85 and 292: ScriptDoctor is called "one-shot" (see claim 5).
- Line 31: "Without few-shot examples, 0% of games were solvable (Table I)" holds only with neither few-shot examples nor CoT. With CoT but no few-shot examples, 10% had a solvable level and 5% had all levels solvable.
- Line 16: `game_creator` is not the only ARC-native generation path. arc-interactive ships coding-agent skills `skills/create-arc-game/SKILL.md` and `skills/check-arc-game-solvable/SKILL.md` (with `AGENTS.md` and `opencode.jsonc`). That is a human-driven agent workflow, not an automated loop.
- Lines 66-67 and 305: the solver equals the baseline in 20 of 25 games, not all (claim 6).
- Line 75: the public action mix of 13/7/4 sums to 24, because ft09's metadata has empty `tags`.
- Line 81: the 70 "Author: Claude" CHANGELOG entries are 42 Opus 4.6, 12 Opus 4.7, 14 Sonnet 4.6, 1 Sonnet 4.6 + Opus 4.6 and 1 Sonnet 4. Many of them cover infrastructure, not games. "Claude-authored" holds; "Opus 4.6/4.7" is incomplete.
- Line 105, the Q2 takeaway and implication 2: the 10 → 100 → 526 threshold comes from a single 4B model (claim 1). It shouldn't be read as general.
- Related to the static filter: `game_creator/README.md` itself suggests "clue numbers" for `maze + deduction`, which breaks ARC's no-numbers rule. Expect some generated games to fail the glyph/digit check.
