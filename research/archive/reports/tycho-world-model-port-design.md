# Porting Tycho's executable world model into Franzen's harness: design

2026-10-08. Design only; nothing here is implemented. It answers the owner's request of the same day: "somethings in Tycho like the world model and stuff work very well (so use them to modify Dan Fran's world models.)"

Sources:
- Tycho at `arc-agi-v4/tycho-port` (NIMI-research, Apache-2.0), and our Kaggle run of it, `runs/kaggle/v4-tycho-port/v1`.
- Franzen's harness on `main` (`ARC3-Inference/`), `WRITEUP.md`, `arms.md`, and the request logs of `fz-ledger-on-trim-r2` (49.14).
- Read-only online sources (§1.7): Vladimir Yakunin's public notebook DF-WM, pulled with `kaggle kernels pull` and diffed against our `kaggle/template.ipynb`; Sergei Makarov's Kaggle thread 743723; and papers on code world models.

## Recommendation in one paragraph

**The idea has already been tried, on our model, and did not beat baseline.**
- DF-WM is Franzen plus almost exactly the helpers drafted here: a code simulator, a replay check, BFS search and a divergence-guarded batch executor. Its page score is 29.3, against 27.89–31.5 for plain Franzen.
- Makarov's controlled study found an executable world model with replay validation and search planning "at or below the baseline". The model "largely ignored" it. Its simulators that looked exact on past transitions were no better than "nothing changes" on new ones.
- Tycho's own paper reaches the same conclusion: transition accuracy is not the bottleneck; inferring the goal is.

**So the first step is a cheap measurement, not a build.**
- Run DF-WM's existing patch as an arm on the 10-game panel and count how often the model calls its helpers (§5.0). If adoption is near zero, stop this line.
- Only if the model uses the helpers, build the narrower version below. It differs from DF-WM in four ways:
  1. It checks the model's own **abstract** state (`wm_observe`/`wm_step`), which is what our model already writes, instead of a full-grid `simulate`.
  2. It reports **prospective** accuracy (predictions made before the transition happened) next to a **"nothing changes" baseline**. That answers Makarov's overfitting finding.
  3. It adds a **passive falsification line** (`ARC3_WM_AUTOCHECK`) that needs no new calling habit.
  4. Its system-prompt text is facts and contract only, with no "instead of manual guessing" advice.

**Arm configuration.** Two flags, both off by default:
- `ARC3_WM_TOOLS` adds `wm_verify()`, `wm_plan()` and `wm_run()`, which act on two functions the model may define: `wm_observe(frame)` and `wm_step(state, action)`.
- `ARC3_WM_AUTOCHECK` adds one line to a tool result, only when the simulator mispredicted an action that call just executed.

**Why there is a gap to fill.** In r2 our model wrote simulator- or search-shaped functions in 16 of 25 games (110 distinct names). It compared a prediction with the observed result in about 7 of 2,183 python calls.

**Don't port Tycho's builder subagent.** It never fired in our 25-minute run, and at our wall-clock each pass costs minutes of a slot.

**Expectation.** Roughly a 1-in-7 chance of a repeatable gain of +3 or more at matched wall-clock. Cost when unused is about 400 cached tokens. Treat this as a screen-then-repeat arm, not a likely win.

---

## 1. Tycho's world-model machinery

### 1.1 The contract (`tycho/workspace/templates/seed_world_model.py.tmpl`)

The model edits a file, `world_model.py`, that describes the game as a deterministic Moore machine:

| Function | Role |
|---|---|
| `State` (dataclass) | Fields the model chooses: board, avatar position, counters, palette, hidden phase. |
| `init_state(grid0, level)` | s0 from the level's first frame. |
| `transition(state, action)` | Next state. `action` is `{"action","row","col"}`. It clones before mutating (`TRANSITION_RETURNS_NEW_STATE = True`). |
| `render(state)` | The full 64×64 grid the state shows. `UNKNOWN = -1` is allowed only for cells it cannot yet know (for example an unseen viewport). |
| `outcome(state)` | Exactly `"ongoing"`, `"level_complete"` or `"game_over"`. |
| `actions(state)` | Candidate actions for the planner. Click games must return a focused set, not all 4,096 cells. |
| `subgoals`, `heuristic`, `planner_key`, `observation_variants` | Optional: planning aids, and up to 5 tolerated HUD renderings. |

### 1.2 The verifier (`tycho/workspace/wmlib_template.py`)

- **`verify(model)`** (line 567) runs per level. It renders `init_state` against the level's first frame, then **threads** `transition()` forward from s0 over every observed action. That catches hidden state such as counters, because the model never gets to re-read the true board. It scores each **graded** transition, meaning one where the observed grid changed **or** the model predicted a change. That second condition catches "walks through a wall" on a real no-op. It returns `simulation_accuracy` (exact, with variants allowed), `strict_simulation_accuracy`, `known_cell_accuracy`, `prediction_coverage` (so an all-UNKNOWN model can't pass), `n_false_change_on_noop`, a per-level breakdown, `first_divergence` (level, turn, action, cell diff) and `errors`.
- **`verify_outcome(model)`** (line 1071) checks `outcome()` against terminal evidence. It reports `level_complete` too loose (a phantom win the planner would steer toward) or too tight, `game_over` missed or too loose, and terminal-render mismatch.
- **Evidence helpers:** `current_state(model)` (line 339) threads the model to "now". `death_events()` and `terminal_events()` keep GAME_OVER and winning frames outside ordinary transitions.

### 1.3 The planner

- `plan_bfs` / `plan_astar` / `plan_subgoals` (lines 1543–1770) search through `transition()` to `outcome == "level_complete"`. They prune `game_over` states, cap at 20,000 nodes, and report `found` / `no_plan_exhausted` / `no_plan_within_budget`.
- `validate_plan` (line 1643) replays a candidate through the canonical model. `game_over_action_report` (line 1449) lists the one-step actions the model predicts are fatal.
- `seed_plan.py.tmpl` writes `notes/validated_plan.json` only after canonical replay reaches level_complete.
- `Workspace.validated_plan_hint` (`workspace.py` ~line 597) feeds the actor the plan's next action each turn, but only while the observed frame hashes match the plan's predicted renders. It pauses on the first divergence or on a model edit.

### 1.4 Who writes the model, and the triggers (`tycho/agent/modes.py` lines 50–56)

| Mode | Who writes `world_model.py` | Trigger |
|---|---|---|
| `no_world_model` | nobody | none |
| `single` | the actor itself (`wm_single.j2`) | the actor decides |
| `orchestrator` (the paper's best) | the **builder subagent** only | the actor calls the `invoke_builder` tool (`agent_tools.py` line 148, `wm_orch.j2`: "a few times per level") |
| `trigger` | the builder subagent | the harness fires it (`agent.py` line 721 `_run_trigger_builder`) on a `wm_signal` misprediction, at level start, and after a GAME_OVER reset |

- **The builder** (`agent/builder.py` line 286, `WorldModelBuilder.build`): a separate conversation with its own system prompt (`builder.system.j2`). The kickoff auto-attaches the current verify state plus two PNGs of the first divergent transition. It runs up to `TYCHO_BUILDER_MAX_STEPS` tool calls (40 in our run), and returns a fixed-format report: confidence, dynamics accuracy, outcome rule and open hypotheses, plan validated or not, recommended action.
- **The divergence probe** (`agent/wm_signal.py`): one sandboxed subprocess per committed turn prints `DYN:ACCURATE|INACCURATE|NOMODEL` and `OUTCOME:...`. It treats coverage below 0.75 as inaccurate.

### 1.5 Feedback to the actor

- **Auto-verify on every semantic save** of `world_model.py` (`agent_tools.py` line 324 `_wm_feedback`, using `wm_feedback_probe.py.tmpl`). The verifier output is appended to that edit's tool result. Comment-only edits are skipped. This is free, since no extra LLM call is made.
- The builder report arrives as a user message ("[auto world-model builder fired — advisory, you decide the action]").
- The plan hint is placed in the per-turn user message (`actor.user.j2`, `plan_hint`).

### 1.6 What the paper measured (`artifacts/appendix_metrics.json`, `figure_data.json`)

There is one public-25 run per mode, with Claude Opus 4.8 and a $750 per-game soft cap:

| Mode | RHAE | Δ vs none, paired (95% CI) | Games better / worse / tied | Exact-transition rate | Builder calls | Model calls | Output tokens | List cost |
|---|---|---|---|---|---|---|---|---|
| no world model | 79.07 | — | — | — | 0 | 22,856 | 32.8M | $5,658 |
| single | 85.36 | +6.28 (−0.67, +13.15) | 11 / 5 / 9 | **22.2%** | 0 | 25,640 | 34.3M | $7,267 |
| orchestrator | **88.49** | +9.42 (+3.97, +15.76) | 11 / 3 / 11 | 45.6% | 147 | 24,054 | 35.8M | $5,781 |
| trigger | 83.07 | +4.00 (−6.35, +15.68) | 8 / 8 / 9 | 99.97% | 1,192 | 44,391 | 63.1M | $8,025 |

The "selected" orchestrator runs with stronger models scored 100.0 (GPT-5.6 Sol, exact rate 99.5%, 660 builder calls) and 100.0 (Opus 5, 70.3%, 130 builder calls). On levels both runs cleared, orchestrator used 18.9% fewer actions than no-model; single used 8.5% fewer.

**Caveats that matter for us:**

- **One run per mode.** Only the orchestrator's CI excludes zero. Single's CI includes zero, and trigger lost 8 games. By our own rule (repeated runs must agree) none of this is a verified alpha.
- **Budget dependence** (`figure_data.json` `budget_sensitivity`, the same runs re-scored at lower caps):

  | Cap per game | none | single | orchestrator | trigger |
  |---|---|---|---|---|
  | $100 | 65.6 | **62.1** | 69.4 | **42.3** |
  | $250 | 77.9 | 78.0 | 86.6 | 71.1 |
  | $750 | 79.1 | 85.4 | 88.5 | 83.1 |

  At the lowest cap, the actor writing its own model (single) is 3.5 below no-model, and harness-triggered building is 23 below. Only the orchestrator stays ahead (+3.8).
- **Our budget is far below even that cap.** A full 121-min, 25-game run of ours generates about 4.5M tokens (about 180k per game) from a 30B-class model. Opus 4.8's no-model run generated 32.8M, about 1.3M per game, roughly 7× ours, from a frontier model.
- **Even Opus 4.8 wrote poor models.** In single mode only 22% of graded transitions were reproduced exactly, and single mode planned zero levels (`plan_levels = 0`; 25 auto-planner actions followed in total). Its gain over no-model came while the model was mostly wrong. That points to the discipline of writing mechanics as checkable code and seeing them falsified, more than to search.
- **The paper names its own limit** (arXiv 2607.28287, abstract): "Automatic repair after verification failures produces models that reproduce observed transitions much more accurately, yet reaches only 83.07 RHAE. Transition match indicates whether a simulator reproduces observed dynamics, not whether it has identified the objective or improves the next action."

### 1.7 Outside evidence: the same idea elsewhere, including on our exact model

**DF-WM** (Vladimir Yakunin, `kaggle.com/code/vladimiryakunin/df-wm`, last run 2026-10-04; page score 29.3 as reported by the coordinator, which I could not fetch myself).
- Pulled read-only to the session scratchpad (`nb/df-wm/`). Its harness patch differs from our `kaggle/template.ipynb` patch only by:
  - a new `inference/utils/world_model.py` (`SimFrame`, `compare_grids`, `check_sim`, `search`);
  - sandbox globals `check_sim`, `search` and `act_checked`, the last a per-action guarded executor that raises on divergence;
  - a system addendum, `SIMULATOR_WORLD_MODEL_ADDENDUM`, which is partly advice ("plan multi-step routes with `search(...)` instead of manual guessing");
  - a level-start reminder to run `check_sim()`;
  - persistent functions defaulting to on, scope `game`;
  - `ARC3_WORLD_MODEL_SIM=1`.
- Everything else (`run.py`, docs, viewer) is packaging. The setup is the same model, the same serving and the same flags.
- Its contract is a **full-grid** `simulate(frame, action)` returning a grid or a `{(r, c): color}` dict, compared on the interior with a 4-cell border ignored. `check_sim` looks at only the last 25 transitions. `search` hashes whole grids, caps at 1,500 states and excludes MOUSE by default. It never reports a "nothing changes" baseline or prospective accuracy.
- That score is within leaderboard noise of plain Franzen (27.89 on his board, 31.47 on his page; identical code moves about ±4). **On our model and harness, a near-copy of Design 1 shows no detectable gain.** The notebook has no saved outputs, so we can't tell whether the model used the helpers.

**Sergei Makarov, Kaggle thread 743723** ("Harness or model? Notes from a week of controlled harness experiments", 2026-09-26). Read through Kaggle's read-only topic API; nothing was posted.
- Each layer was added alone on a Duck-based harness and compared with repeated baselines.
- "Almost every added layer scored at or below the baseline: … and an executable world model with replay validation and search-based planning."
- "The world-model scaffold is the clearest example: **the model largely ignored it**, so that run measured the presence of an unused instruction rather than the method."
- "Offline, the model rarely produced a simulator that predicted unseen transitions better than a trivial 'nothing changes' baseline. … **Several simulators that looked exact on the observed transitions were no better than it on new ones.**"
- Failure analysis: state-by-state alignment of stuck vs successful runs of the same game. "In 83% of the forks the stuck run had already tried the action type that later turned out to be decisive. … It had the evidence; it drew the wrong conclusion about the goal and kept defending it."
- Noise: two identical 30-minute runs scored 2.60 and 1.45.
- His model is not named, and his scores (1.45–9.4) sit on a lower scale than ours. The direction still matches ours.

**Papers** (arXiv abstracts read 2026-10-08):

| Work | Setting | What made the world model help, or not |
|---|---|---|
| Tycho (2607.28287) | Frontier models, ARC-AGI-3 | Helps at large budgets; trigger-repair raises transition match but not score. The objective is the open problem. |
| Twin (2608.14490) | Frontier coding agent, ARC-AGI-3 | 93.3 RHAE. The harness refuses any action until the program reproduces every past transition. "Building a usable world model is simpler than anticipated, whereas the harder problem is inferring the right goal." |
| Kepler (2610.00834) | Claude Opus 5, ARC-AGI-3 | 100 RHAE. Retrospective transition checks plus conditional prediction checks. 858M tokens, 97% cache reads, about $778 per board. A scale we cannot approach. |
| Executable WMs for ARC-AGI-3 (2605.05138) and its ablation (2607.15439) | Codex / GPT-5.x | 58.1 (GPT-5.5) and 41.3 (GPT-5.4). In the ablation, **"requiring an executable deliverable is not universally beneficial: textual outperforms flexible-interface executable in both gpt-5.5 conditions"**. Exact replay verification ranks first "but uses substantially more resources". Gains from model capability exceed gains from variants. |
| OPINE-World (2607.01531) | Opus 4.8 | Keeps provisional objects and causal hypotheses in a Bayesian, exploration-centric representation. The representation is what it adds over WorldCoder. |
| WorldCoder (2402.12275) | Gridworlds, task planning | The program must explain the data **and** be optimistic that reward is reachable, a constraint between program and planner. That optimism, which targets the goal, drives exploration. Compute-efficient vs ReAct. |
| Code World Models for General Game Playing (2510.04542) | Board and card games | Works when the **rules are given in natural language** plus trajectories. Then the code model feeds MCTS. |
| PoE-World (2505.10819) | Atari Pong, Montezuma | A product of many small programmatic experts generalises from few observations. Small composable rules beat one monolithic program. |
| EMPA / theory-based RL (2107.12544) | 90 Atari-style games | Human-level sample efficiency, but from a strong built-in game-engine DSL prior with Bayesian program inference. The prior does much of the work. |
| AutumnSynth (Das et al., POPL 2023; from memory, not re-read) | Autumn grid worlds | Synthesises reactive programs with latent-state automata from traces. Latent state is the hard part, and the method relies on a DSL and search, not an LLM. |
| AutumnBench / WorldTest (2510.19788) | 43 grid worlds | Frontier models trail humans on environment-level queries; the gap is attributed to exploration and belief updating. |

**Across these sources, code world models help when one of these holds:**
- the rules are given, or a strong DSL prior supplies them (CWM-GGP, EMPA, AutumnSynth);
- a frontier coding model has a large budget and replay verification is a hard gate (Twin, Kepler, Tycho at $750);
- the model is optimistic about, or explicitly searches over, the goal (WorldCoder, Twin's goal search).

**They fail to help when:**
- the scaffold is opt-in and the model ignores it (Makarov, plausibly DF-WM);
- the budget is low (Tycho at $100: single −3.5, trigger −23);
- fit is judged on past transitions only (Makarov's overfit simulators);
- dynamics accuracy is optimised while the goal stays wrong (Tycho trigger mode, Twin, Makarov's 83% of forks).

Our setting has none of the enabling conditions: a 30B-class model, a budget at least 7× smaller, and no rules given.

---

## 2. Why Franzen's text world model hurt, and what our Tycho run shows

### 2.1 Franzen's structured world model (`ARC3_MEMORY_SECTIONS`, `tool_agent.py` lines 1491–1525)

The original Duck mechanism had the model write labelled memory sections (World model, Plan, Action model, Open questions, …). The harness parsed them and re-inserted the carried block into later prompts. WRITEUP §6: "The original parsing often missed model-written updates when their formatting differed from the expected labels. That could leave stale information repeatedly reinserted into context. … The largest improvement in this area came from disabling the structured world-model mechanism entirely. With a larger rolling context, the conversation itself already retained much of the useful evidence."

The code's own measurements (docstring of `_memory_section_labels`): over 25 games, World model was written 351 times and Plan 255; the other sections 0–25 times. "A section written once and never revised keeps being re-injected as current, which was observed holding an answered question open for a whole game."

**What made it costly:**

1. **Context and tokens.** The block, up to 8,000 characters per section, rode along in the prompt. Every word was the model's belief, not a fact.
2. **Wrong beliefs persisted.** Nothing checked a section against the game, so a wrong rule stayed "current" until the model happened to rewrite it, and parse misses meant a rewrite might not land.
3. **Generation spent on bookkeeping.** The model wrote prose about the game instead of acting or computing.
4. **Prefix churn.** A block that changes inside the prompt invalidates the cached tail behind it. This is the same mechanism that sank Tycho's cache hit rate on our run (64% vs Franzen's ~93%; corrected from 57%, see research/tycho-port/README.md).

**Related user-side evidence** (`arms.md`):

- Opt-in channels the model must choose to use go unused: scratchpad tags (0 written), the `act` tool (0 calls), a thinking guide (0 of 28 `Predict:` lines), memory tags (0.9–2.1% of calls).
- Prediction text without a mechanical check was inconclusive: hypothesis-loop had 93% compliance and no measured gain. Franzen's stronger step-verification hint gave no gain.
- Extra prompts on stuck levels lost: audit60k scored 44.10.

**What that implies for a port:**

- Never re-insert model-written beliefs into the prompt.
- The harness should only state **facts it can check by executing code**: transitions it recorded, and predictions it computed.
- Keep the system prompt static.
- Put any per-event text at the tail of the current tool result, where it costs no cached prefix.
- Piggyback on behaviour the model already shows instead of asking for a new habit.

### 2.2 Our Tycho run at matched wall-clock

Both arms below ran the same balanced 10-game panel, all 10 games at once on one RTX Pro 6000 with the W4A16 Qwen3.8-Flash-Next on Pennyroyal SGLang, with **25 minutes of wall-clock per game**. That is the owner's matched-wall-clock comparison, not a matched-token one.

| | Tycho, orchestrator (`v4-tycho-port` v1) | Our harness, Fable layer off (`v4-fable-f001-off`) |
|---|---|---|
| Mean score | **3.53** | 29.22 |
| Levels cleared | 8 | 27 |
| Actions (all games) | 280 (6–47 per game) | about 820 |
| Model calls | 633 completed (mean 22.5 s, ~1,140 output tokens each) | — |
| Generated tok/s per wall-clock second (engine decode median) | 478 (734) | 603 (731) |
| Generated tokens | 720k | ~0.9M (603 tok/s × 25 min; estimate) |
| Fresh prompt tokens | 6.53M; prefix-cache hit 0.64 | Franzen ~93% reuse (WRITEUP) |
| Builder invocations | **0** | n/a |

**Did the actor write `world_model.py`? No, and in orchestrator mode it cannot:** `actor_writes_wm=False`, and the builder owns the file. What the logs show (`tycho-results/game_*.json.xz`, `trace[*].reasoning`):

- The actor had `invoke_builder` in its tool list in all 10 games and called it **0 times**. Tool calls overall: `take_action` 280, `run_python` 221, `write_file` 9, `read_file` 12, `set_verbosity` 13, `ls` 6, `edit_file` 1.
- It **read** the seed `world_model.py` in 3 games (bp35 turn 11, vc33 turns 5, 16 and 30, lf52 turn 8). In bp35 it noted "world_model.py is a stub. Let me record my beliefs in notes/actor_beliefs.md and then invoke the builder", then kept probing. In sp80 and tn36 it also said it would invoke the builder and chose one more experiment first.
- It wrote `notes/actor_beliefs.md` in 6 games, which is the builder's input. No builder ran to read it.
- Of the two workspaces that downloaded, `ws/bp35/world_model.py` differs from the seed only in the pre-filled `actions()` line. `cd82` has none.

**Why it never fired:**

- At about 20 s per call, a game got 23–73 calls in 25 minutes.
- `wm_orch.j2` tells the actor to invoke the builder once it "has a mechanics hypothesis", "a few times per level". With 6–47 actions per game, the actor always preferred one more probe.
- A builder pass is up to 40 calls; at our latency that is ~13 minutes, more than half a game's budget.

So the run says nothing about whether executable models help our model. It does say that **the builder as Tycho runs it does not fit our wall-clock**, and that Tycho's history rewriting costs about 40% of the prefix cache.

### 2.3 What our model already does inside Franzen's harness

This is a scan of every python tool call in `fz-ledger-on-trim-r2` (49.14), deduplicated across request logs. Script: `scan_defs.py` in the session scratchpad. Counts are regex-based.

- **2,183 python calls; 1,077 `def` statements; 542 functions retained** by persistent functions (scope `game`).
- **110 distinct functions with simulator- or search-like names** (`step`, `move`, `simulate`, `bfs`, `solve`, `search`, `apply…`, `plan…`) in **16 of 25 games**, and 73 of them retained. Examples:
  - m0r0's `step(st, d)` over a pair of positions, followed by a BFS over `seen`.
  - tu93's `bfs` over an extracted edge graph.
  - su15's merge simulator.
  - sp80's `simulate(paddles)`.
- These simulators work on **abstract states the model extracted itself** (tuples of positions, cup sets), not on full 64×64 renders.
- Only about **7 acting snippets compare a predicted state with the observed one**. The model builds simulators and searches through them, but never checks them against the transitions the harness already records (`transitions`, `history`).

**This is the gap Tycho's verifier fills.** It is also why the port should verify the model's **abstraction** (`step(observe(before), a) == observe(after)`) rather than demand Tycho's full-grid `render()`, or DF-WM's full-grid `simulate()`. Opus 4.8 reached only 22% exact renders, and our model already thinks in abstractions.

**A caution from Makarov.** A gap is not demand. That the model *could* use a verifier does not mean it *will*: his scaffold was ignored. Retrospective fit is also misleading, because his exact-on-history simulators were no better than "nothing changes" on new transitions. So the verifier must:
- report accuracy on transitions that happened **after** the current `wm_step` was defined (prospective accuracy), separately from accuracy on the transitions it may have been fitted to;
- always print the "nothing changes" baseline next to both.

### 2.4 Consequences for the port

| Tycho piece | Port? | Why |
|---|---|---|
| Executable model contract | **Yes, adapted:** `wm_observe` + `wm_step` (+ optional `wm_outcome`, `wm_actions`) on abstract state | Matches what the model writes today. A full-grid render is optional in v2. |
| Verifier, including false-change-on-no-op and outcome checks | **Yes:** `wm_verify()` | Pure facts from recorded transitions. Zero LLM tokens. |
| BFS planner with game_over pruning | **Yes:** `wm_plan()` | The model already writes BFS; the helper adds pruning, budgets and a verified start state. |
| Plan replay that stops at divergence (`validated_plan_hint`) | **Yes, as an action helper:** `wm_run(plan)` | Mechanical "stop when the prediction fails"; Franzen's prompt version failed. |
| Auto-verify feedback in the tool result | **Yes:** `ARC3_WM_AUTOCHECK`, on a miss only | Same placement as Tycho's `_wm_feedback`; tail-only, so cache-safe. |
| Builder subagent, `invoke_builder` | **Last; probably never** | Never fired in 25 min; costs minutes of a slot per pass; audit60k-style side prompts lost. |
| Workspace files, `notes/*.md`, scribe | No | That is the text-memory mechanism Franzen removed. |
| History rewriting, image eviction | No | Breaks the prefix cache (64% vs 93%). |

---

## 3. Port designs, ranked

All flags default to off. With every flag off, request payloads must be byte-identical to `main` (see §4).

**Common contract.** These are model-defined, retained by `ARC3_PERSISTENT_FUNCTIONS`, scope `game`:

- `wm_observe(frame) -> state`: a frame view (`current_frame`, `history[i].frame`, `t.before_frame`) to the model's compact state. Lists, dicts and sets are frozen by the helpers for hashing and comparison.
- `wm_step(state, action) -> state`: `action` takes the same form `action(...)` accepts, `'UP'` or `{'action': 'MOUSE', 'row': r, 'col': c}`.
- Optional `wm_outcome(state) -> 'ongoing' | 'level_complete' | 'game_over'`.
- Optional `wm_actions(state) -> list`: candidate actions; required for MOUSE games, defaulting to `valid_actions` minus RESET and UNDO for keyboard games.

All helpers also accept the functions as arguments, so the model can test what it already wrote: `wm_verify(observe=parse, step=step)`.

### Design 1 (rank 1): sandbox verifier, planner and guarded runner, no extra LLM calls (`ARC3_WM_TOOLS`)

**Helpers** are provided globals, built in the sandbox bootstrap like `frame_diff`.

- **`wm_verify(level=None, mode='step', last=None, observe=None, step=None, outcome=None, budget=8)`**
  - Walks `transitions` for the current level (or `level='all'`). It skips `automatic` transitions (the auto RESET) and RESET/UNDO, and treats a `result.level_completed` or `result.game_over` transition as an **outcome check**, not a state check.
  - `mode='step'` (the default) is teacher-forced: compare `wm_step(wm_observe(before), a)` with `wm_observe(after)`.
  - `mode='rollout'` threads the state forward from each attempt's first frame, Tycho's hidden-state check.
  - Grading follows Tycho: a transition is graded if the observed abstract state changed **or** the model predicted a change. It also counts **"blind" transitions**, where gameplay cells changed (`result.gameplay_changed`) but `wm_observe` did not. That is the abstraction analogue of Tycho's `prediction_coverage`: a constant observer cannot score 100% silently.
  - **Prospective vs retrospective, plus a trivial baseline** (from Makarov's finding):
    - The host keeps, per game, the source hash of `wm_step`/`wm_observe` and the action count at which that version was first retained. It passes this to the sandbox in `initial`.
    - `wm_verify` splits graded transitions into **prospective** (executed after this version existed) and **retrospective** (before it existed, so possibly fitted to).
    - It always reports the **"nothing changes" baseline** on the same transitions: how many satisfy `wm_observe(after) == wm_observe(before)`. Many ARC actions are no-ops on the abstract state, so a simulator that never beats this baseline prospectively has learned nothing.
  - Returns a dict whose first key is `summary`, a single line of 300 characters or fewer, e.g. `level 3: since this wm_step: 6/7 matched (nothing-changes baseline 2/7); before it: 8/8 (baseline 3/8); 2 blind; outcome 1/1; first miss step 214 RIGHT: predicted ((12, 30), 5) observed ((12, 31), 5)`. It also carries `n_graded`, `n_matched`, `n_blind`, `n_false_change`, `outcome_checked/ok`, `first_miss` (with a compact element-wise diff for tuples and dicts), `errors` (the first 3 exceptions, sanitized), `checked/total` and `truncated`.
  - `wm_observe` results are memoised by frame object, so each frame is observed once.
- **`wm_plan(goal=None, max_depth=30, max_nodes=50000, start=None, budget=10)`**
  - Runs a BFS from `wm_observe(current_frame)` (or `start`) through `wm_step` and `wm_actions` to `wm_outcome(s) == 'level_complete'` or to `goal(s)`. It prunes states that `wm_outcome` calls `'game_over'`.
  - Returns `{'status': 'found'|'exhausted'|'budget'|'error', 'plan': [...], 'expected': [state after each step], 'expanded', 'seen', 'note'}`. `plan` can be passed straight to `action(...)`. A no-plan result says "not reachable in this model", as Tycho's does.
- **`wm_run(plan, expected=None)`**
  - Executes the plan one `action([a])` at a time. After each action it compares `wm_observe(current_frame)` with the expected state, from `wm_plan` or by stepping the model.
  - It stops at the first mismatch, terminal flag, guard refusal or time budget, and returns `{'executed': k, 'stopped': reason, 'at': {...}}`. This is Tycho's `validated_plan_hint` done in code.
  - It spends real actions. The existing guards (batch no-op, stale-state, terminal latch, death and no-op refusals) apply unchanged, because each step goes through the normal `action()`.

**Touch points:**

- **`inference/agent/python_tool_sandbox.py`:**
  - In `_SANDBOX_BOOTSTRAP` `main()`, define the helpers as closures over `runtime_globals`, the same way `frame_diff` is defined.
  - Record `_t0 = time.monotonic()` at child start, with `time` imported in bootstrap scope; the snippet still cannot import it.
  - Add the helpers to `runtime_globals` **after** `restore_functions(...)`, so a retained or shadowing definition cannot replace them for the next call. Add their names to `_available_names` **before** the restore, so retained functions may call them.
  - Exclude these reserved names from `collect_functions`.
  - Gate all of this on a new `initial["wm_tools"]` key. `run_sandboxed_python(..., wm_tools=False)` passes it through.
- **`inference/utils/`:** a new module, `wm_helpers.py`, holds freeze, diff, BFS and the verify loop. It is spliced into the bootstrap like `segmentation` and `frame_diff` (`inspect.getsource` + `.replace`), so the child stays self-contained.
- **`inference/agent/tool_agent.py`:**
  - `_wm_tools_enabled()` env reader.
  - `_run_python_tool` passes `wm_tools=...`.
  - `_build_system_prompt` appends `WM_TOOLS_ADDENDUM` after `FRAME_DIFF_HINT_ADDENDUM`.
  - `_PYTHON_TOOL_DESCRIPTION` gets one extra sentence under the flag (module-level constant chosen at import, so it stays static).
  - The opener's `Only tool:` line (line ~5103) gets the three names appended under the flag.
  - Pop a host-only `wm_stats` field from `sandbox_result` before `_render_tool_payload`, and log it to the transcript and events so it never reaches the model.
- **`inference/agent/prompts.py`:** a new `WM_TOOLS_ADDENDUM`.
- **`ARC3-Inference/CONFIGURATION.md`, `README.md`, `kaggle/arms.json`:** arms `wm-tools`, `panel-wm-tools`.

**Prompt text** (static system addendum, about 1,450 characters or 340 tokens, cached; facts and contract only, no advice):

```
Simulator checks:
- If you write a simulator for this game, two retained functions named `wm_observe(frame)` and `wm_step(state, action)` can be tested by the harness. `wm_observe` maps a frame view (such as `current_frame` or `t.before_frame`) to your own compact state; `wm_step` returns the state you expect after `action`, given in the form `action(...)` accepts (`'UP'` or `{'action': 'MOUSE', 'row': r, 'col': c}`). Optional: `wm_outcome(state)` returns 'ongoing', 'level_complete' or 'game_over'; `wm_actions(state)` lists candidate actions (needed for MOUSE games).
- `wm_verify()` replays this level's recorded transitions: it reports how often `wm_step(wm_observe(before), action) == wm_observe(after)`, separately for transitions after and before the current `wm_step` was defined, next to how often "nothing changes" would have been right; the first mismatch with both states; transitions where cells changed but `wm_observe` did not; and `wm_outcome` against recorded level completions and game overs. It spends no game action. `wm_verify(observe=f, step=g)` checks other functions.
- `wm_plan(max_depth=30)` searches breadth-first through `wm_step` from `wm_observe(current_frame)` to a state where `wm_outcome` is 'level_complete' (or `goal(state)` is true when `goal=` is passed), skipping states predicted as 'game_over'. Its `plan` can be passed to `action(...)`; it is a prediction of your simulator, not of the game.
- `wm_run(plan)` executes a plan one action at a time and stops at the first state that differs from the simulator's prediction.
- Each stops within about 10 seconds and reports what it did not finish.
```

Tool-description sentence: "`wm_verify`, `wm_plan` and `wm_run` test and use your `wm_observe`/`wm_step` simulator (see the system prompt)."

**Cost:**

| Item | Tokens | When |
|---|---|---|
| System addendum + tool sentence | ~400 | Once per prefix, cached (re-prefilled only when the cache is lost) |
| `Only tool:` line suffix | ~12 | Each new turn message (uncached tail) |
| A `wm_verify()` call | ~20 code + ≤100 result | Only when the model calls it |
| Writing `wm_observe`/`wm_step` | 300–1,500 once per game or level | Largely replaces simulators it writes today |
| CPU | `wm_verify` 0.05–8 s; `wm_plan` ≤10 s; per-call setup unchanged | Inside the existing 30 s python deadline |

Measured on the laptop: `segment_layer` costs **~58 ms per 64×64 frame**. An observer built on `.segmentation` therefore pays about 6 s for 100 transitions. Memoisation, the `last=` argument and the 8 s budget, which returns partial counts, keep this inside the deadline. An `.ascii`-parsing observer is about 100× cheaper.

**Interactions:**

- **Persistent functions** (`ARC3_PERSISTENT_FUNCTIONS=1`, scope `game`, imports on): `wm_*` functions are retained across levels. Tycho's "one engine across levels" is the right default, and `wm_verify()` at a new level's first transitions shows whether it transfers. The retention rules still apply: no decorators, so no `@dataclass` State (use tuples or dicts), and literal defaults. On `main`, **a `wm_step` that reads a snippet constant is refused** (r2 had 412 such refusals), so the contract steers constants into `wm_observe`'s output. Retention notes already tell the model when a function was refused.
- **sandbox-robust** (PR #76, `ARC3_SANDBOX_PERSIST_VARS`): carried plain-data constants (a wall set, a palette) become legal dependencies, which makes simulators far easier to retain. They clear on a level change, so a `wm_step` reading a cleared constant raises `NameError` at the next level. `wm_verify`/`wm_check` report that as an error, which is honest but can confuse. Test both with and without #76.
- **The 30 s tool deadline:**
  - Helpers budget against `30 − (now − _t0) − 2 s`, measured from the child's start. That is conservative under `main` (host deadline from spawn, including action waits) and under #52 (the snippet's own time only).
  - **On `main` a timeout wipes every retained function (#5)**, which here would delete the simulator. The arm must carry #51 (`arm/fix-5-retained-functions`, a 9-line change) or sit on #76, which includes it.
  - `wm_run` pays one host round trip per action. On `main` that is ~635 ms at 310 history frames, so `wm_run` stops on budget after roughly 30–40 actions late in a game. #43 (history-state perf) brings it to ~11 ms.
- **Level changes:** `wm_verify` defaults to `current_frame.level`. The winning transition (`before.level != after.level`, `result.level_completed`) is graded only as an outcome. `wm_plan` starts from the current frame, so a plan never crosses a level.
- **Deaths, RESET and UNDO:** `automatic` RESET transitions are skipped. The fatal transition is graded as an outcome (`game_over`), and its after-frame is the death frame (history semantics in `STRUCTURED_RUNTIME_STATE_ADDENDUM`). UNDO is skipped by default (`include_undo=True` grades it). Rollout mode restarts at each attempt's first frame.
- **Animations:** the after-frame is the settled frame, which is what `action()` returns.
- **Ledger** (ledger-on-trim and its stacks): no interaction by default. Optional sub-flag `ARC3_WM_LEDGER` adds one harness-written fact line to the trimmed-history ledger, e.g. `wm: last check L3 step 212, 14/15 matched, first miss step 214 RIGHT`. It is shown only at the ledger's existing post-trim position, so it is cache-safe. Off in the first arm.
- **Yields:** writing a simulator is a long generation, so a 2,048-token yield may land mid-design. The retained functions survive and resumption is unchanged.

### Design 2 (rank 2, an add-on to Design 1): falsification line in the tool result (`ARC3_WM_AUTOCHECK`, requires `ARC3_WM_TOOLS`)

**Behaviour:**

- At the end of any python call that **executed at least one action**, the bootstrap checks only the transitions this call created, in both the success and error paths. It does this only if `wm_observe` and `wm_step` are callable in the snippet's globals, so retained functions and same-snippet redefinitions both count.
- If every prediction matched, it emits **nothing**.
- On the first mismatch, outcome miss or exception, the tool result gets one `wm_check` field of 240 characters or fewer:
  - `wm_check: wm_step mispredicted 1 of 3 actions in this call; first: step 214 RIGHT predicted ((12, 30), 5), observed ((12, 31), 5).`
  - `wm_check: wm_outcome gave 'ongoing' after step 220 MOUSE(30,12); that action completed the level.`
  - `wm_check: wm_step raised NameError: name 'WALLS' is not defined at step 231.`
- It is silent when no simulator is defined. This is the deliberate difference from Tycho's trigger mode, which bootstraps a model; ours never nags.

**Touch points:**

- `python_tool_sandbox.py` bootstrap: compute it after `exec`, before `_send({"type": "final"| "error", ...})`, with a 1.5 s budget and an exception-proof wrapper. Add `"wm_check"` to the payload.
- `tool_agent.py` `_run_python_tool`: copy `sandbox_result["wm_check"]` into `payload["wm_check"]` after `stdout`/`error` and before `function_retention`. Exclude it from middle truncation. Count it in host-only `wm_stats`.
- No system-prompt text beyond Design 1's: the field explains itself, so it adds no static cost.

**Cost:** 0 tokens when silent; ~40–70 tokens per miss, at most one per tool result. CPU is ~1–2 `wm_observe` calls per executed action, about 0.1 ms with ASCII parsing or ~60–120 ms with segmentation.

**Why rank 2 and not 1:** it adds no capability, only automation. Its value depends on the model defining `wm_*` functions. It has a nag risk if the simulator is coarse: a line on most acting calls is ~50 tokens per call and plausibly distracting. Measure lines per acting call; if more than 30% of acting calls carry a line in the panel, add a "silent after 3 consecutive misses on this level" rule before full runs.

### Design 3 (rank 3): a cheap builder call (`ARC3_WM_BUILDER`, not recommended for the first arm)

**Behaviour:** at most once per level, when (a) the level has at least 12 non-automatic transitions and no `wm_step` is defined, or (b) autocheck has reported 3 misses on this level, the harness issues a **side request** on the same game stream:

- a static builder system prompt, a compact Design-1-style contract plus "write or repair wm_observe/wm_step; do not act";
- the python tool with `action()` disabled;
- at most 3 tool steps and 4,096 output tokens per step.

Functions it defines are merged into `self._kept_functions`. A report of 400 characters or fewer is prepended to the actor's **next** opener (new content, so the actor's cached prefix is untouched).

**Touch points:** `ToolAgent.analyze()` (a new pre-turn hook next to the yield and resume logic), `_request_payload` (separate messages and system), `_retained_sources` / `_record_retained_functions`, the sandbox `action_handler=None` path, transcript and request logging, and the scheduler's per-game token accounting (builder tokens must count as the game's spend).

**Cost:**

- Uncached prefill of the evidence prompt, ~10–20k tokens.
- 2–3 decode rounds of 1–4k tokens at the per-stream ~60–80 tok/s of a 10-stream batch: **about 40–150 s of the game's slot per build**. Median cleared-level cost is 14.5–22.8k tokens, so one build costs on the order of a fifth to a half of a typical level.
- A different prefix on the same Mamba-hybrid stream can evict the actor's Mamba checkpoint under slot pressure (Franzen's prefix-cache patch keeps only the final-prefill checkpoint).

**Why rank 3:**

- Tycho's builder never fired in our wall-clock.
- Side prompts on stalled levels lost (audit60k −2.2). Franzen's summaries showed no gain.
- The paper's trigger mode, the harness-fired builder, was the worst world-model mode and collapsed at low budget (42.3 at $100).
- Build it only if Design 1+2 runs show models are being **defined but stay inaccurate** (median `wm_verify` match rate under 50%) while defining games clear more levels than non-defining ones.

### Summary of costs

| | Static (cached) | Per turn | Per use | Extra LLM calls | Cache risk |
|---|---|---|---|---|---|
| D1 helpers | ~400 tokens | ~12 tokens | CPU ≤10 s inside the call; ~120 tokens | 0 | none (static system prompt, tail results) |
| D2 autocheck | 0 | 0 | 40–70 tokens per miss | 0 | none (tail of the current tool result) |
| D3 builder | builder prompt | 0 | ~10–20k prefill + 3–10k decode per build | 1–3 per level | side prefix; Mamba-checkpoint pressure |

### Which design is most likely to beat baseline at matched wall-clock

The question is score at the same elapsed minutes, so a design wins only if its benefit beats the wall-clock it costs.

- **D3 (builder): least likely.** Each build takes about 40–150 s of a game's slot. That is wall-clock taken directly from acting, and nothing in Tycho's runs, our run or the audit60k loss suggests a 30B builder repays it. It also targets dynamics, which three independent sources (Tycho, Twin, Makarov) say is not where runs fail.
- **D1 alone (helpers the model must call): unlikely.**
  - DF-WM is essentially D1 with a full-grid contract and some advice text, on our model. It shows no detectable gain.
  - Makarov's equivalent was ignored.
  - D1 costs almost nothing when unused, so its downside is small. That is also why it most likely lands within noise: a feature the model doesn't use can't move the score.
- **D1+D2 (helpers plus the passive falsification line): the best of the three, still more likely to tie than to win.**
  - D2 is the only piece that reaches the model without a new calling habit, provided the model names its simulator `wm_step`.
  - Its wall-clock cost is milliseconds of CPU and about 50 tokens per miss.
  - What it can plausibly buy is fewer actions spent on batches built from a wrong simulator. Those are part of the 14.5% of actions in died attempts and the 5.6% no-ops (failure taxonomy).
  - It does nothing for the dominant failure, a wrong goal defended early, which costs 32.8% of tokens on levels never cleared.

**Implication.** If the owner wants a world-model arm, run the DF-WM adoption screen first (§5.0) and then D1+D2 as one arm. Keep the bar at the owner's rule: a repeated, paired gain. A goal-side world-model feature might attack the actual bottleneck: `wm_outcome` checked against recorded level completions, or a WorldCoder-style "is the goal reachable in my model?" check. That belongs in a separate design, not in this arm.

---

## 4. Test plan: "make sure absolutely everything works as intended and integrates with the loop correctly harness wise"

Base the branch `arm/wm-tools` on `main`, with `arm/fix-5-retained-functions` (#51) merged first so a timeout cannot delete the simulator. Repeat the suite on a scratch merge with `arm/sandbox-robust` (#76). Run heavy tests one at a time under `nice`.

### 4.1 Unit tests on the real sandbox child (`tests/test_wm_tools.py`)

These run through `run_sandboxed_python` with the real bootstrap and a scripted `action_handler`, as in sandbox-robust's tests.

1. **Correct model.** Use a recorded toy history: a keyboard mover with a HUD step bar, one wall death, one level change and one UNDO. The result must be `n_matched == n_graded`; outcome 2/2 (level_complete, game_over); automatic RESET and UNDO skipped; the winning transition graded only as an outcome.
2. **Wrong model.** An off-by-one `wm_step` gives `first_miss` at exactly the right step, with both states and the diff. Wall-walk on a real no-op counts in `n_false_change`. A constant `wm_observe` gives `n_blind == gameplay-changing transitions` and `n_graded == 0`, never a silent 100%.
3. **Rollout vs step.** A hidden counter that only rollout catches is caught: step mode passes and rollout fails at the right step.
3b. **Prospective vs retrospective and the baseline.**
   - A `wm_step` defined at action k is graded prospective only on transitions after k.
   - Redefining it resets the split, while a byte-identical redefinition does not.
   - A simulator that memorises past transitions as a lookup table scores 100% retrospective and only the "nothing changes" rate prospective. The summary must show both.
   - The baseline counts equal a direct recount on the toy history.
4. **Errors.** A `wm_step` that raises gives the sanitized message in `errors`, and the snippet continues. A missing `wm_observe` gives a clear `summary`. A list or dict state is frozen correctly. An unhashable custom object gives an error message, not a crash.
5. **Budgets.** A deliberately slow `wm_step` makes `wm_verify` return `truncated=True` with partial counts. `wm_plan` returns `status='budget'` before the deadline. The **python call never times out** at `timeout_seconds=30`, nor at 5 with a scaled budget.
6. **Planner.** On the toy game: shortest plan found; `game_over` states pruned; `exhausted` vs `budget` statuses; MOUSE game with `wm_actions`; missing `wm_actions` on a MOUSE game gives a clear error; `plan` accepted verbatim by `action(...)`.
7. **`wm_run`.**
   - Stops at the first mismatch with `executed == k`.
   - Stops on `level_completed` or `game_over` (the terminal latch interplay).
   - Surfaces `KnownNoOpActionError` and `StaleStateActionError` as `stopped` reasons.
   - Never sends an action after the budget is spent.
8. **Retention.**
   - `wm_observe`/`wm_step` defined in call 1 are retained (scope `game`) and used by `wm_verify` in call 2.
   - A `wm_step` reading a snippet constant is refused on `main` semantics and accepted with `ARC3_SANDBOX_PERSIST_VARS=1` (scratch merge).
   - A model-defined `wm_verify` is neither retained nor able to shadow the helper in the next call.
   - A timeout after defining the model keeps it (requires #51).
9. **Autocheck (D2).**
   - Silent on matches.
   - Exactly one line on a miss, in both the `final` and the `error` branch (an exception after `action()`).
   - Silent when no simulator is defined, or when no action was executed.
   - 240 characters or fewer.
   - Budget-exceeded is silent and counted in `wm_stats`.
10. **Flags off.** `sorted(dir())` inside a snippet equals `main`'s, so no new globals appear. Tool results are byte-identical to `main`'s for a fixed script.

### 4.2 Real-frame replay (`tests/test_wm_tools_real_history.py`)

Build a sandbox state payload from a recorded game in `runs/kaggle/fz-ledger-on-trim-r2/v1/artifacts/*_events.jsonl.xz`: board, `action_display`, level, state and automatic RESETs. Use a game with deaths and level changes, and one MOUSE game.

- Run `wm_verify` with hand-written partial simulators, e.g. m0r0's own `step` and `move` lifted from its request log.
- This pins the level, death and animation semantics on real ARC frames and real HUD bars.
- It also times verify over 300+ frames: under 2 s with an ASCII observer, and truncated but within budget with a segmentation observer.

### 4.3 Loop-level test through the real `ToolAgent.analyze()` (`tests/wm_loop_sim.py` + `tests/test_wm_tools_loop.py`)

This follows `arm/reasoning-scaffold`'s `scaffold_loop_sim.py`: it runs under the Kaggle notebook's own `setup_env` (`notebook_env()` read from `kaggle/template.ipynb`) and renders every request through the model's chat-template fixture. It uses `arm/sandbox-robust`'s `sandbox_loop_sim.py` approach for the tools: the **real** `_run_python_tool`, `run_sandboxed_python` and retention, with only HTTP faked and the toy game behind `step_env`.

The script, one or more requests per turn:

| Turn | What the scripted model does | What is checked |
|---|---|---|
| T1 | Defines `wm_observe`, `wm_step`, `wm_outcome`; acts RIGHT | The functions are retained |
| T2 | `print(wm_verify())` | 1/1 matched |
| T3 | Redefines `wm_step` wrongly; acts | A D2 `wm_check` line appears in that tool result, once |
| T4 | Yield turn | Resumption is unchanged |
| T5 | `p = wm_plan(); action(p['plan'])` | Clears level 1 |
| T6 | Level 2: `wm_verify()` | Defaults to level 2; the model is still retained |
| T7 | `wm_run` into a hazard the model doesn't know | Stops at the first mismatch (the death frame); outcome miss is reported |
| T8 | A snippet that loops forever after defining a helper | Times out; retained functions survive |
| T9 | Wins | — |

Assertions:

- The game finishes under every config.
- The system prompt and `tools` are **identical across every request** in a run, so prompts are static.
- Each rendered prompt **extends the previous one** except at real trims, so the prefix cache holds. Run this check in all three resume modes, as cutoff-signal did.
- `wm_*` text appears only in the system prompt, the tool description, the `Only tool:` line and tool results.
- `wm_stats` never appears in any request but does appear in the transcript.
- Token counts per request with the flag on minus off equal the addendum plus the per-turn suffix plus the tool-result text, and nothing else.

### 4.4 Byte-identical flags-off vs `main`

Run the loop sim with every new flag off on this branch, and on a `main` checkout via `--inference-root`, the same `SCAFFOLD_SIM_BASELINE_ROOT` pattern. Assert the request payloads (messages, tools, kwargs) are equal byte for byte. Also pin the system prompt string and the tool schema to `main`'s for the default, Kaggle and stack-v2 envs. The parent re-runs this independently, as for #72–#76.

### 4.5 Mutation checks (each must make at least one test fail)

1. Grade only observed changes (drop the false-change condition).
2. Pair `before` with the wrong frame (`history[i].frame` for both).
3. Drop the level filter.
4. Grade the winning or fatal transition as a state transition.
5. Remove the time budget.
6. Emit autocheck on matches too.
7. Put autocheck in the turn opener instead of the tool result (the prefix test must fail).
8. Inject helpers when the flag is off.
9. Allow a model-defined `wm_verify` to be retained.
10. `wm_run` continuing after a mismatch.
11. Forget `automatic` RESET filtering.

### 4.6 Build check

`python kaggle/build_notebook.py --arm wm-tools` and `--arm panel-wm-tools` must build locally. The embedded harness patch must equal the branch diff, the arm env must set `ARC3_WM_TOOLS=1` and `ARC3_WM_AUTOCHECK=1`, and the `kaggle/tests` suite must pass.

---

## 5. Evaluation plan

### 5.0 Before building anything: a DF-WM adoption screen (about 40 GPU-minutes, almost no code)

DF-WM's patch is Franzen plus a world-model layer, on our model, built for the same flags. Kaggle public notebooks default to Apache-2.0; confirm on its page before reusing code. Port its four additions as one arm, `panel-dfwm`, behind its own flag `ARC3_WORLD_MODEL_SIM`, with its prompt text verbatim:
- `world_model.py`;
- the sandbox globals;
- `SIMULATOR_WORLD_MODEL_ADDENDUM`;
- the level-start line.

Run it on the 10-game panel in the same batch as `panel-control`, with request logs saved. Then count, from the request logs:
- games that define `simulate`;
- `check_sim`, `search` and `act_checked` calls;
- `act_checked` divergence stops;
- tokens in snippets that define `simulate`.

**Decision:**
- **Fewer than 2 of 10 games define a simulator and call `check_sim`:** stop the world-model line. Makarov's "ignored scaffold" holds on our model too. Log it in `arms.md` under Didn't work.
- **The model defines simulators and checks them:** build D1+D2 (§3) as designed. The DF-WM logs then give a free calibration of how accurate this model's simulators are, prospectively and against "nothing changes". Those are the numbers that justify D2's line rate and wording.

This screen measures adoption, not score. A 10-game panel resolves only large score effects.

### 5.1 Arms

| Arm | Env | Compare with |
|---|---|---|
| `panel-dfwm` (step 0) | `ARC3_WORLD_MODEL_SIM=1` (DF-WM layer, verbatim) | `panel-control`, same batch; adoption gate only |
| `panel-wm-tools` | `ARC3_WM_TOOLS=1`, `ARC3_WM_AUTOCHECK=1` | `panel-control`, same batch |
| `wm-tools` | same | `control`, same batch, same branch (flags off) |
| `wm-tools-noauto` | `ARC3_WM_TOOLS=1` only | Ablation, only if `wm-tools` wins |
| `tput-safe-wm` | tput-safe env + both flags | `tput-safe`, only after both are validated |

### 5.2 Screen (panel smoke)

Run the balanced 10-game panel in the same batch as `panel-control`. This is a screen, not a verdict. It passes when:

- There are no errors, and tok/s is within ±3% of panel-control.
- The python error rate per call rises by no more than 1 point. Python timeouts are not above control, and none come from helpers.
- **Usage:**
  - At least 3 of 10 games define `wm_observe` + `wm_step`.
  - Those games call `wm_verify` at least once each.
  - The median helper time is under 2 s.
  - Autocheck lines appear on 30% or fewer of acting calls.
- Generated tokens per action rise by no more than 10%.

**If usage is 0**, do not spend full runs. The arm has hit the opt-in failure seen with scratchpad tags and the `act` tool. Allow at most one wording revision and one more panel.

### 5.3 Verdict (full runs)

- Run the full 25-game, 121-minute runs, `wm-tools` paired with `control` in the same batch, **twice** (owner's rule: the effect must repeat).
- **Judge at matched wall-clock.** Compare the score at the same elapsed wall-clock from each run's start: the 121-minute final score, plus the score curve at 30, 60 and 90 minutes from per-action timestamps (`run_elapsed_seconds` in the action results or events; failing that, transcript header times, as `analysis/failure_taxonomy.py` does).
  - Never compare at matched tokens or on any runtime clock that excludes startup, queueing or slot waits.
  - Report startup time separately, so a slow-start notebook is not read as an effect.
  - The curve also shows whether simulator-writing delays early levels.
- **Run `make significance BASELINE_SCORE=<control> CANDIDATE_SCORE=<wm-tools>` per pair and pooled.**
- **Keep** if the 2-run mean Δ is +3 or more (gaps under ~3 are noise) and P(Δ>0) is 0.9 or more pooled, with neither run below control.
- **Drop** if the mean Δ is −1 or less, or fewer than 20% of games use the helpers.

### 5.4 Evidence that the world model is actually used

These come from host-only `wm_stats` plus the request logs; the parser is a new `analysis/wm_usage.py`.

**Adoption**

- Games defining `wm_observe`/`wm_step`, and the level at which they first appear.
- Retention refusals involving `wm_*` names.
- Generated tokens spent in snippets that define `wm_*` functions.

**Verification**

- `wm_verify` calls per game and per level.
- The distribution of match rate (`n_matched/n_graded`), `n_blind` and `n_false_change`.
- How many verify calls are followed by a `wm_step` redefinition, i.e. falsification leading to repair.

**Planning**

- `wm_plan` calls by status.
- Plan length vs the level's eventual winning route (`level_path`, if stacked).
- **Plan-followed actions:** actions executed by `wm_run`, or by an `action(...)` whose list equals a plan from the same snippet.
- **Plan-followed clears:** levels whose winning action was plan-followed.
- Deaths on plan-followed actions.

**Autocheck**

- Lines per acting call, and the share followed by a redefinition within 2 calls.

**Outcome**

- Actions per cleared level vs human baseline on levels both arms cleared (Tycho's matched-level efficiency).
- Tokens per cleared level.
- Share of tokens on never-cleared levels (the failure taxonomy's biggest sink, 32.8%).

### 5.5 Compute

A panel costs about 35–40 GPU-minutes. Each full pair costs about 2 × 2.2 GPU-hours, so the two pairs need about 9 GPU-hours. Queue behind the running `tput-safe` batch.

---

## 6. Risks, stated honestly

1. **The model may not use it.** This is the most likely failure, and now the best-documented one.
   - Our record on opt-in channels is poor: scratchpad 0 tags, `act` tool 0 calls, Predict 0/28, memory tags at most 2.1%.
   - Makarov's world-model scaffold was "largely ignored".
   - DF-WM, the same idea on our model, scored within noise of plain Franzen.
   - Counterweights: the model already writes `step`/`bfs` in 16 of 25 games, and the helpers accept its existing functions as arguments. D2 needs no new habit beyond the naming.
   - The §5.0 DF-WM screen tests this risk before any build.
1b. **Retrospective fit is not prediction.** Makarov's simulators were exact on history and no better than "nothing changes" on new transitions.
   - A model may reach 100% on `wm_verify` by encoding what it saw, then plan confidently through it.
   - Prospective accuracy and the trivial baseline (§3, D1) make this visible, but only to a model that reads them. `wm_plan` does not refuse to plan with an unproven simulator; a gate like Twin's ("no action until every past transition reproduces") costs far more than our budget allows.
1c. **Dynamics is not the bottleneck.** Tycho, Twin and Makarov independently place the remaining failure in **goal inference**: Makarov's stuck runs had tried the decisive action at 83% of forks. A perfect dynamics simulator does not fix a wrong goal; `wm_plan` would then search efficiently toward the wrong target.
   - This caps the upside of every design here.
   - The goal-side pieces (`wm_outcome` checked against recorded level completions and game overs) are the only parts that touch the bottleneck, and only weakly.
1d. **Executable is not automatically better than text.** In the controlled ablation (2607.15439), a flexible executable world model lost to a textual one in both GPT-5.5 conditions. Only the full exact-verification variant ranked first, and it used "substantially more resources". We are on the cheap end of that trade.
2. **Over-investment.** In the paper, single mode (closest to D1) lost to no-model at the $100 cap (62.1 vs 65.6), and our budget is well below that.
   - A model that keeps perfecting a simulator spends tokens it would have spent acting.
   - The prompt deliberately does not ask the model to build one. Measure tokens in `wm_*` snippets; if they exceed 15% of a game's tokens without plan-followed clears, that is the signal.
3. **A weak model writes weak simulators.** Opus 4.8 reached 22% exact. Ours will be lower on rich games, and a wrong `wm_plan` can lead a batch into a hazard.
   - `wm_run` and D2 bound the damage to one mispredicted action per run; a plain `action(plan)` is unguarded beyond the existing guards.
   - Watch deaths on plan-followed actions.
4. **Vacuous abstractions.** An observer that drops the relevant variable matches trivially. `n_blind` exposes it, but only if the model reads it.
5. **Deadline and payload on `main`.**
   - Without #51, a single helper timeout deletes all retained functions.
   - Without #43, `wm_run` is slow late in long games (~635 ms per step).
   - Segmentation-based observers are ~58 ms per frame; budgets truncate rather than time out, but truncated verifies are weaker evidence.
6. **Constants and retention friction.** On `main`, simulators that read module constants are refused (the 412-refusal pattern in r2). With #76 they carry, but clear at level changes, so a `NameError` follows.
7. **Merge conflicts.** `PYTHON_ADDENDUM_*`, `_PYTHON_TOOL_DESCRIPTION`, the `Only tool:` line and the sandbox bootstrap are touched by budget-truth (#74), sandbox-robust (#76), memory-xml (#65) and level-path (#66). Expect add/add conflicts when stacking; keep each hunk self-contained.
8. **The external evidence points the wrong way for us.**
   - Every positive result is from frontier models with large budgets: Tycho, Twin, Kepler (858M tokens per board), and 2605.05138.
   - Tycho has one run per mode, and only the builder-pull mode has a CI excluding zero, which is the mode we cannot afford.
   - The two results on our class of setup (DF-WM on our exact model; Makarov's controlled layer) are null or negative.
   - Prior after the online search: roughly a 1-in-7 chance of a repeatable gain of +3 or more, down from about 1-in-4 before it.
   - The design is chosen so the downside is small (~400 cached tokens and no extra calls when unused), and so the usage metrics separate "didn't use it" from "used it and it didn't help".
9. **Advice-text risk.** rellik13's runs saw advice blocks hurt (6.42 → 4.16). The addendum is written as a contract plus facts, with no imperative "you should". Keep it that way in review.

## 7. Build order (one logical change per commit)

0. First, on its own branch `arm/dfwm-screen`: port DF-WM's layer verbatim behind `ARC3_WORLD_MODEL_SIM`, add `panel-dfwm`, and run §5.0. Continue below only if the adoption gate passes.
1. Merge `arm/fix-5-retained-functions` (#51).
2. Add `inference/utils/wm_helpers.py` (pure functions: freeze, diff, verify loop, BFS), with unit tests outside the sandbox.
3. Splice the helpers into the sandbox bootstrap behind `initial["wm_tools"]`: reserved names, injected after restore, time budget.
4. Add `ARC3_WM_TOOLS` to `tool_agent.py` (flag, pass-through, `wm_stats` pop and logging), and `WM_TOOLS_ADDENDUM` plus the tool and opener text.
5. Add `wm_run`.
6. Add `ARC3_WM_AUTOCHECK` (bootstrap end-of-snippet check and payload field).
7. Add the real-sandbox tests, the real-frame replay test, the loop sim with prefix and identity checks, and the mutation runs.
8. Update docs (`CONFIGURATION.md`, fork `README.md`) and add arms to `kaggle/arms.json`.
9. Log the arm in `arms.md` under "In progress", and move this design's "World Model Extension" and "REPL improvements" backlog items to point at it.
