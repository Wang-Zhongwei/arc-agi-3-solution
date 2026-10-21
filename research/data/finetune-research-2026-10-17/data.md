# Training and testing data for fine-tuning an ARC-AGI-3 agent (2026-10-17)

Script used for Q1: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/research-ft/inventory.py` (output `inventory.json` next to it; run under `nice -n 15`, reads only `benchmark.json(.xz)`; the call counts come from one fully decompressed run, extrapolated).

## 1. Our own trajectory data

Source: `daniel-franzen-sol/runs/kaggle/*/v*/` (index `runs/index.csv`, 223 rows). Counted from `benchmark.json`.

| Item | Count |
|---|---|
| Kernel versions with a parseable benchmark | 150 (110 are full 25-game runs) |
| Date range | 2026-09-07 to 2026-10-17 |
| Game-level runs (game x run) | about 3,100 overall; 1,705 in the 89 runs that kept full request logs |
| Levels cleared, all runs | 7,785 (7,172 in full 25-game runs) |
| Games fully won, all runs | 369 (354 in full runs) |
| Completion tokens, all runs | 378M (356M in full runs) |
| Actions taken | 455k |
| Runs with `*_requests.jsonl.xz` (full chat request and response) | 89 runs, 1,710 per-game files |
| Levels cleared inside logged runs | 6,083; 354 game wins |
| Of 1,705 logged game-runs | 164 cleared 0 levels (8,062 actions, only 2.8% of 292k actions), 1,187 partial, 354 won |

Families (benchmark counts): `v4-*` 73 runs / 4,887 levels / 284 wins (Franzen + our harness); `fz-*` 20 / 1,253 / 73 (earlier Franzen arms); `taaf-*` 26 / 829 / 10 and `duck-*` 27 / 765 / 2 and `wang-*` 3 (older Wang/Duck harness, weak, no request logs except 1); `v4-gguf-*` and `v4-q27-dflash2-10x` use other models/quantisations. Almost everything is Qwen3.8-Flash-Next W4A16 on one RTX Pro 6000 (per `kaggle/template-metadata.json`); 3 runs are different models.

Per-call numbers from one full run (`v4-mem-repair-10x/v1`): 2,624 model calls, 5.25M completion tokens (about 2.0k per call), 145M prompt tokens (about 55k per call, mostly prefix-cached), 99% of calls end in `tool_calls`, 27 in `length`. So all logged runs are roughly 180k calls, about 360M completion tokens, and several billion prompt tokens if every call is expanded to full context. The benchmark `tokens=` field is completion tokens only.

Important limits:
- 100% of this data is on the 25 public games. Any SFT on it contaminates the only exact-format eval. It can only go into a final-phase model (see split), or into leave-game-out experiments.
- Data is skewed toward easy levels: early levels are cleared in nearly every run, late levels rarely, and only the 25 games' mechanics are covered (about 110 runs of each game).
- Request logs contain a full 128k-context prompt per call. Segmenting into SFT examples means picking, for each cleared level, the calls between level-start and level-clear; at about 23 calls per cleared level (2,624 calls / 116 levels in the reference run) that is roughly 140k calls for 6,083 levels, but with heavy near-duplicate prompts. Use final-turn-only plus a trimmed context.
- Usable splits: SFT = calls in level-clear segments of won/cleared games (about 6k level segments). Preference or RL: same level, a stuck attempt (partial or 0-level games, 1,351 game-runs) vs a cleared attempt (pairs exist for games played in several runs, which is every public game). The `solution replay` (action list that cleared a level) is also reusable as a verified solver for that exact level, but only for public games.

## 2. Environment sources

Counts verified locally are marked (local); others are from the cited pages.

| Source | Count | License | Format | Closeness to ARC-AGI-3 | Unlimited variants? | Solver / verifier |
|---|---|---|---|---|---|---|
| ARC-AGI-3 public demo (25) [paper](https://arxiv.org/html/2603.24621v1) | 25 public + 55 semi-private + 55 private = 135 total (private pair = our 110) | ARC-AGI-3-Agents is MIT; game code in `arc-agi-3/data/environment_files` (local) | native arcengine `.py` | exact | no | no solver; replay lists exist only from our runs |
| ARC-AGI-3-Training-Data, SimonOuellette35 (local `competitors/ARC-AGI-3-Training-Data`) | 434 entries in `games_db.json`: 221 PuzzleScript, 44 MiniGrid + 44 partial-obs, 29 Gym-Gridworlds, 3 Procgen, about 63 homemade, plus 25 "ARC-native" augmented versions of the PUBLIC games (all 25 present) | none in repo (all rights reserved by default); underlying MiniGrid, Procgen, PuzzleScript are separately licensed | own `game_envs.py`, 64x64 int frames, ARC action subset | good; homemade and PuzzleScript are discrete grid with levels | yes: seeds and augmentations built in | yes: 390 `solvers/generate_*_training.py`; `master_data_generator.py --episodes 1000` writes JSON trajectories (frames plus actions plus `optimal`) |
| DreamTeam game_creator + 25 generated games (local `competitors/dream-team`, [arXiv 2605.09650](https://arxiv.org/abs/2605.09650)) | 25 shipped, 8 levels each; generator can make more | Apache-2.0 (NVIDIA) | arcengine adapter `.py` + `metadata.json`; each has a `solution.json` that replays to WIN | very good (same engine and scoring, genres x 12 skills x feature tags) | yes, but generation used GPT-5.5 or Flash; needs a strong coder LLM and Python execution of generated code (untrusted) | yes: exported solution per level |
| sonpham-org/arc-agi-3 (local `competitors/arc-agi-3`, last push 2026-05-07, no license) | 62 AI-written games documented in `GAMES_PROGRESS.md` (26 action, 36 strategy, 10 levels each); local `environment_files` has 86 version dirs, 92 `.py`, about 60 non-public custom games plus other versions of public games (e.g. `sk48-41055498`) | none declared | arcengine `.py` | good; written by Claude with smoke-tested scripted wins | no (fixed), but easy to extend | scripted solutions tested in CHANGELOG; no packaged solver |
| sonpham-org/arc-3 (found by web search only; repo now 404 to `gh`/WebFetch, probably private or renamed) | search snippets: catalog of about 294 games (25 official, 17 in-house, 252 community), 571 unreviewed AI-generated games, and PR #76 "Held-out copycats (8 games), TEST ONLY, NEVER TRAIN" | unknown | unknown | likely same as above | maybe | unknown |
| arc-interactive (local, MIT) | 252 version dirs, GAMES.md lists about 250 rows (tutorials ez01-04, sokoban sk01/pb01-03, memory, flood fill, bridge builder, rule switcher...) | MIT | arcengine, 64x64 or smaller grids | good, community-authored | no | not packaged; many tiny and easy |
| PuzzleScript corpora: [script-doctor/PuzzleJAX](https://github.com/smearle/script-doctor) (MIT code) | 952 scraped games under `data/scraped_games`; paper says over 500 validated ([arXiv 2508.16821](https://arxiv.org/pdf/2508.16821)); ScriptDoctor dataset 610 | MIT for the tool; each game is the author's own work (license varies) | PuzzleScript text; JAX engine | grid puzzles with unknown rules and multiple levels, no goal text | no (but LLM generation of new games exists) | tree-search solutions in repo; BFS easy |
| MiniGrid / BabyAI [Farama](https://github.com/Farama-Foundation/Minigrid) | dozens of env ids, unlimited seeds | Apache-2.0 per project (GitHub reports NOASSERTION) | gymnasium | navigation, key/door; goals are stated in missions, small grids | yes | BabyAI bot gives optimal |
| Boxoban [google-deepmind/boxoban-levels](https://github.com/google-deepmind/boxoban-levels), gym-sokoban | about 900k levels | Apache-2.0 / MIT | text grids | push puzzles, matches several public games' mechanics | yes | BFS/A* |
| Procgen (MIT), Crafter (MIT), BALROG (MIT), TextArena (MIT) | Procgen 16 games, Crafter 1, BALROG 6 suites, TextArena dozens | MIT | pixel or text | poorer match (continuous or text, long horizon) | yes | mostly none |
| game-reasoning (our own, worktree `.../wt/rsi/game-reasoning`) | chess mate-N, tic-tac-toe, connect four, lane, hidden-rule `grid`/`grid2`, sokoban, `grid-next`, `grid-code` | ours | text | `grid` hidden-rule env mimics ARC-AGI-3 scoring; others are auxiliary skills | yes, per seed | exact verifiers, BFS-optimal counts |
| HF human trajectories [magic-sword/arc_agi_3_public_demo_human_testing](https://huggingface.co/datasets/magic-sword/arc_agi_3_public_demo_human_testing) | 340 human sessions, 475 MB, parquet; visible rows are all `sk48` | "fully open-source", licence not named | frame trajectories | exact (human imitation) | no | n/a; PUBLIC game, eval contamination |
| HF frontier-agent trajectories [AgentNativeResearchLab/arc-agi3-*](https://huggingface.co/datasets/AgentNativeResearchLab/arc-agi3-wa30-agent-trajectories) | many repos (one per harness x model x game: g50t, ar25, wa30, ft09, tr87, r11l, s5i5, su15, ls20; models Claude, Kimi k2.7, GPT-5.6, Gemini 3.1, GLM 5.2) | CC BY 4.0 | JSONL traces, reasoning, per-level solution scripts, frames | exact; teacher reasoning | no | per-level replay scripts; PUBLIC games |
| STARGA v3 corpus ([naestro-agi3-27b card](https://friendli.ai/models/star-ga/naestro-agi3-27b)) | unknown size, "procedurally-generated ARC-AGI-3-style simulations" | adapter Apache-2.0; corpus availability unverified | rendered frames, goal, action plan | claims; no scores | yes | unverified; treat as an idea, not a source |
| ARC-AGI-1/2, RE-ARC (MIT), BARC (HF barc0) | ARC-AGI-2 1,000 train tasks; RE-ARC unlimited per task; BARC about 400k (license MIT on a 100k sibling, 400k card not checked) | Apache-2.0 / MIT | static input-output grids | core-knowledge priors only; no interaction | RE-ARC yes | exact outputs |

Take-aways:
- The best sources by closeness: DreamTeam generated games, sonpham-org/arc-agi-3 custom games, arc-interactive (all share the arcengine API, so our harness and scoring run unchanged), then Training-Data's homemade and PuzzleScript games (own wrapper, needs an adapter to arcengine or a dual-format prompt).
- Combined distinct non-public games in arcengine format: 25 + about 60 + about 250 = about 335, though many are trivial. Quality filter: keep games where our current model clears some but not all levels over 4 attempts.
- Training-Data contains augmented versions of ALL 25 public games and sonpham contains other versions of the public ids. Remove them (by game-id prefix and by mechanic family) from any training set that is also evaluated on the public games.
- Audit for mechanic overlap beyond ids: sokoban, key-door, flood fill and memory-match games in the external sets resemble public games (ARC-AGI-3 paper stresses private games are out of distribution). Use a held-out family list rather than only id exclusion.

## 3. Generating trajectories at scale

| Method | What you get | Rough cost per 1k trajectories | Notes |
|---|---|---|---|
| Scripted solvers/BFS on engines (Training-Data `solvers/`, Boxoban, MiniGrid bot, DreamTeam `solution.json`) | optimal action sequences with exact frames; no reasoning text | CPU minutes to hours; effectively free | Teaches action competence but not exploring unknown rules. Must be paired with an exploration prefix (random or scripted probe actions) and a rationale. Training-Data already inserts RESET, exploration, `expert` phases. |
| Hindsight relabel scripted trajectories into agent-style transcripts | text reasoning stating the discovered rule, then the action | one local LLM call per level (about 2-5k tokens), so about 5M tokens per 1k trajectories; minutes on our GPU | Explicitly state "probe, infer rule, plan" form. Risk: rationalised reasoning that no model could infer (leakage of solver knowledge); filter by checking the model can reproduce the plan. |
| Our model, rejection sampling (expert iteration) | genuine on-policy reasoning that cleared a level | measured: one 2-h run on 1 RTX Pro 6000 = 5.25M completion tokens, 2.6k calls, about 116 cleared levels on public games, so about 17 GPU-h per 1k cleared levels (harder than most external games) and about 80 GPU-h per 1k full-game attempts; easier external games probably 2-5x cheaper | Already have about 6k public-game segments of this type. Needs external games for new data. |
| Local larger open teacher on Azure | stronger reasoning traces | about 0.2B completion tokens per 1k game attempts (at about 105 calls x 2k tokens); at an assumed 5k tok/s aggregate that is 11 node-hours; real throughput depends on the model, unverified | Check teacher license allows distillation. The only direct measurement of frontier behaviour is in the HF datasets above (public games). |
| Frontier API | best traces | not funded | Use free public HF trajectories instead, but only in the final phase (public games). |
| LLM-generated new games (DreamTeam game_creator, sonpham style) | new environments with `solution.json` | one big-model run per game (hours); validate by replaying solutions through the engine | Execute generated code only in a sandbox (not done here; static inspection only). Best lever for diversity. |

Filters worth applying to all sources: reject trajectories where the winning action count is more than 5x the optimal (matches the scoring cutoff); keep per-level `(optimal / actions)^2` as a weight; keep stuck-vs-cleared pairs for DPO/GRPO.

## 4. Testing and evaluation protocol

Facts: public 25 = only exact-format eval; ARC says official scores are never reported on public games and training on them is overfitting ([paper](https://arxiv.org/html/2603.24621v1)); private 110 = 55 semi-private + 55 private. Our noise: one full 2-h run moves the public mean by about 2.7 (sd), same-notebook leaderboard draws 38.22/28.73/29.54/26.90 (sd about 5).

Proposed protocol:
1. Never train on the 25 public games in the main line. They are the held-out test set for every model checkpoint. Phase B (final submission only) may add public-game data, justified by leave-game-out evidence from Phase A (see split).
2. Leave-family-out validation: build a validation set of 15-20 external arcengine games (e.g. 8 DreamTeam, 6 sonpham, 6 arc-interactive), excluded from training by mechanic family as well as id. This is the dev set for every decision; it is a proxy for the 110, cheaper than public runs, and does not burn the public set.
3. Metrics, in this order: games fully won and time-to-win (per user's target), levels cleared at a fixed token budget, then ARC mean score. Report per-game paired differences with `make significance`.
4. Number of runs needed (normal approximation, 80% power, alpha 0.05, n = 15.7 sd^2 / delta^2 per arm): with sd 2.7 and a +3 mean-score target, about 13 full runs per arm; for +5, about 5 per arm. With sd 5 (leaderboard) it is about 50 and 16. That is infeasible at 2 GPU-h per run, so full runs are for confirmation only, and the screen has to be paired by game: same 25 games, same seeds, levels cleared and tokens per cleared level as the paired metric (the per-game test removes game-difficulty variance).
5. Cheap screen: (a) 20-minute panel on 6-8 games of the validation set, 3 seeds, measuring levels cleared and fraction of games with >=1 level; (b) a no-GPU offline check, teacher-forced action accuracy and next-frame prediction on held-out level-clear segments; (c) only the top candidate gets 2-3 full public runs. Alpha rule remains the user's: repeated runs must consistently beat baseline.
6. Contamination guards: hash every training game file, exclude by id and by mechanic family list; scan training text for public game ids and strings from public game source; keep a SHA list of the 25 engines (`experiments/arc-agi-3/data/environment_files_manifest.json` already stores sha256s); for HF/competitor data, drop any game whose id or name matches a public id (e.g. Training-Data's ARC-native subset).

## 5. Proposed train / validation / test split

| Split | Content | Use |
|---|---|---|
| Train | Training-Data PuzzleScript/MiniGrid/homemade (minus ARC-native and mechanic-overlap), arc-interactive minus held-out, sonpham-custom minus held-out, 17-20 of the DreamTeam games, game-reasoning procedural games, Boxoban/BabyAI for rule-inference warm-up. Trajectory types: scripted + hindsight-relabelled, own rejection-sampled successes, stuck/cleared pairs for DPO/GRPO | SFT then expert iteration |
| Validation (dev) | 15-20 external games, held out by family: about 8 DreamTeam (the ones whose solution.json replays), 6 sonpham, 6 arc-interactive, never trained on | all hyper-parameter and checkpoint choices, fast screens |
| Test | 25 public games, untouched | final report only; full 2-h runs, 2-3 repeats |
| Phase B (optional, last) | add our 6k public-game level segments, the HF human sk48 and AgentNativeResearchLab traces | only after Phase A shows how much same-game data helps via a 5-game leave-out check on the public set (train on 20 public + external, test on 5; run 2 folds) |
| Extra | the sonpham "held-out copycats (TEST ONLY)" if the repo is ever obtained | additional unseen test |

## 6. Rules (what is and is not verified)

Verified from [arcprize.org/competitions/2026](https://arcprize.org/competitions/2026) and [the ARC-AGI-3 track page](https://arcprize.org/competitions/2026/arc-agi-3):
- No internet in Kaggle evaluation; no API systems (GPT/Claude).
- To win prizes, code and methods the submitter wrote must be open under a permissive public-domain licence (CC0 or MIT-0); third-party code under a licence that permits public sharing (Apache-2.0, GPLv3). Participants must open source before receiving official private evaluation scores. Milestone prizes require open-sourcing by 2026-06-30 and 2026-09-30. Submissions due 2026-11-02, results 2026-12-04.
- "Anyone is welcome to use our public data and leaderboards independently."

Search snippets only (the Kaggle rules page is JavaScript-rendered and WebFetch returned only its title, so these are unverified): the Kaggle ARC-AGI-3 rules mention an "External Data and Tools" reasonableness standard decided by the Host (cost, accessibility) and a CC-BY 4.0 licence for winning submissions with an exemption for pretrained models/input data with incompatible licences. For ARC-AGI-2 a third-party summary said external data and pretrained models are allowed if freely and publicly available.

Implications: (1) a fine-tuned model's weights are a "method" and must be released with the code to be prize-eligible; Qwen3.8-Flash-Next's licence terms for derivatives were not checked (the related Qwen3.6-27B is Apache-2.0); (2) training data we use should be publicly available and licensed for redistribution at the time of release: Training-Data repo and sonpham repos have no licence, so generating trajectories from them and releasing the result is legally unclear. Prefer re-implementing or using MIT/Apache sources (DreamTeam, arc-interactive, MiniGrid, Boxoban, script-doctor) for the released training set; (3) training on public games is not prohibited in anything verified, but ARC regards it as overfitting.

## 7. Gaps

- Kaggle rules text on external data and weights: not read. Have a Sonnet subagent with Chrome open https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules and report "External Data and Tools", licence of winning submissions, and whether weights must be public.
- `sonpham-org/arc-3` (the "294 games / 571 AI-generated games" repo) is not reachable by `gh` (404); it may be private. The reachable one is `sonpham-org/arc-agi-3` (cloned, 62 documented games). Ask the user for the exact URL.
- No quality audit of arc-interactive / Training-Data games (many are 8x8 and trivial); run our model for 4 attempts per game to rank difficulty. Games were inspected statically; none were executed.
- Call counts (about 180k) are extrapolated from one run. Not measured: how many level-clear segments have a clean prompt-to-reasoning trace usable for SFT, and prompt-level near-duplication.
- Teacher-model cost is formula-based; no throughput measured on the Azure nodes.
