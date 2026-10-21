# Failure taxonomy: where tokens, wall-clock and actions go

2026-10-08. Tool: [`analysis/failure_taxonomy.py`](../../../analysis/failure_taxonomy.py) (method in [`analysis/README.md`](../../../analysis/README.md)). Data: [`research/data/failure-taxonomy/`](../data/failure-taxonomy/) (`tables.md` is the tool's full output, `runs.json.xz` the per-game buckets, `responses.csv.xz` one row per model response).

This is the research-sweep Tier 2 item "Failure-taxonomy miner over existing transcripts". It ranks where the compute of a full run goes, says which arm already targets each cause, and proposes arms for the causes nothing targets.

## Summary

- **The largest sink is levels that are never cleared: 32.8% of generated tokens, 22.6% of actions and 37.7% of game wall-clock** (pooled over 7 full runs). The two worst runs have the most: stack-v1 40.7% (43.32) and audit60k 38.2% (44.10). Arms have targeted this (D′, audit60k, UCB, fresh-eyes reset); all lost or were refuted. Only ledger-on-trim has worked, and it works by clearing more levels, not by spending less on stuck ones.
- **The second-largest sink is long thinking with no action.** 40.7% of tokens go to python calls that ran fine but did not act. Steps that took 12k or more tokens are 6.8% of executed steps but hold 34.1% of tokens. This is mostly real exploration, not failure.
- **The failures proper, in tokens:**
  - length cut-offs: 6.3%, and 56% of those tokens sit on never-cleared levels
  - python errors that ended a response before any action: 5.4%
  - responses whose every action was a no-op: 5.0%
  - the tail after a game's last action: 2.7%
  - timeouts and guard refusals: under 0.2% combined
- **The failures proper, in actions:**
  - attempts that ended in a game over: 14.5% (13.7% died-attempt actions plus 0.8% for the fatal action and its automatic reset). 79% of the died-attempt actions are in attempts of 40 or more effectful actions, which fits running out the step bar.
  - actions above the human baseline on cleared levels: 6.9%
  - no-op actions: 5.6% (2.0% changed nothing; 3.7% changed only the 4-cell HUD border)
- **Causes with no arm** (proposals below):
  1. **Yield resumes.** 3,519 yields in 7 runs. Each resume re-sends the opener, about 1,000 tokens (≈600 text and a 400-token image): 0.47–0.68M uncached prompt tokens per run, **4.6–5.2% of prefill** (4.9% pooled; PR #129's tokenizer count gives 4.3–5.0%). Yields do not release the gate slot, so they buy no scheduling. An earlier version said the resumed requests carry 31.7% of prefill, but almost all of that is the previous reply and its tool result, which the next request prefills whether or not the turn yields (issue #130).
  2. **Deadline tail.** 3.2% of tokens produce no action.
  3. **Code errors that sandbox-robust does not touch** (Type/Index/Value/Key/Syntax): 2.7% of tokens.
- **Corrections to research-sweep:**
  - "'clamped from'" in transcripts is the text-token calibration note (36–51 per run), not a coordinate clamp. There are 0 coordinate clamps and 0 invalid-action results in any run.
  - The "CUT OFF" nudge is shown 0 times in 169 cut-offs: every cut-off ends in an immediate 2,048-token yield. This confirms the cutoff-signal premise.

## Runs

| Label | Run | Score | What it is |
|---|---|---|---|
| control | `fz-arms/v4` | 46.25 | Franzen control |
| temp06 | `fz-arms/v5` | 45.92 | temperature 0.6 (control-like) |
| audit60k | `fz-arms/v6` | 44.10 | strategy audit every 60k tokens on a stalled level |
| streams14 | `fz-streams14/v1` | 46.38 | 14 streams (control-like) |
| ledger-r1 | `fz-ledger-on-trim/v1` | 52.56 | ledger-on-trim |
| ledger-r2 | `fz-ledger-on-trim-r2/v1` | 49.14 | ledger-on-trim repeat |
| stack-v1 | `v4-stack-v1/v1` | 43.32 | D′ + ledger-on-trim + #2 fix |
| F001-panel | `v4-fable-f001/v2` | 29.93 | 10-game, 25-min panel, Fable layer on (benchmark only) |
| F001-off-panel | `v4-fable-f001-off/v1` | 29.22 | same panel, layer off (benchmark only) |

The full runs are 25 games, 121 minutes, 10 slots (14 for streams14). All seven have transcripts, request logs and per-action events. The two panels saved only `benchmark.json`, `summary.txt` and logs, so they get the level split only (see below).

## Method in one paragraph

Generated tokens come from the request log (`usage.completion_tokens` per response). They are matched in order to the transcript's `[MODEL RESPONSE META]` blocks; step and finish reason match in all 175 games.

Each response gets one cause, in this order:
1. length cut-off
2. no tool call
3. if its step executed actions: level clear, death, all no-op, or effect
4. otherwise: python timeout, guard refusal, uncaught python error (by class), or "probe" (ran fine, no action)

The step's actions go to the step's last response, since a turn ends when an action executes. Actions get one cause each from the per-action events. No-ops are recomputed from consecutive boards with the harness's 4-cell border. Wall-clock is the time between transcript headers, split over an invocation's responses by tokens.

Levels come from `actions_per_level` (the same convention as `level_dataset.py` on arm/sim-v2). Tokens after a game's last action go to the level it ended on.

## Ranked list: where tokens, actions and wall-clock go

Pooled over the 7 full runs: 32.7M generated tokens, 37,763 actions, 324 game-hours. The rows are exclusive cuts (E) or overlays on them (O), so the columns of different rows overlap where marked O.

| # | Cause | Tokens | Actions | Wall | Kind | Arms that address it |
|---|---|---|---|---|---|---|
| 1 | Spent on levels never cleared | **32.8%** | **22.6%** | **37.7%** | O | ledger-on-trim (worked: 52.56/49.14); D′/stack-v1, audit60k, UCB (lost); easy-first (smoke only); fresh-eyes reset (refuted in research-sweep) |
| 2 | Python calls that ran fine but did not act ("probe") | **40.7%** | – | 36.3% | E | reasoning-effort (#73), reasoning-scaffold (#72); no-action circuit breaker rejected (research-sweep, Kinetic v8 evidence) |
| 3 | Steps of 12k+ tokens (6.8% of executed steps) | 34.1% | – | – | O | reasoning-effort (#73); yield-6k (below) for the resume overhead, which is small (about 3% of prefill) |
| 4 | Attempts ending in a game over (died attempt + fatal action + reset) | 1.1% | **14.5%** | 0.7% | E | budget-alert (#60), budget-truth (#74), level-path (#66) for re-walks; known-death guard is on |
| 5 | Waiting for a first gate slot | – | – | 11.9% | E | gate11 (#38), scheduler arms; D′ doubled it (stack-v1 20.5%) |
| 6 | Actions above the human baseline on cleared levels | – | 6.9% | – | O | level-path (#66), memory-v2 |
| 7 | Length cut-offs (169 replies, 1.0% of replies) | **6.3%** | – | 3.4% | E | cutoff-signal (#75), stub-dead-reasoning (#39), reasoning-effort (#73) |
| 8 | No-op actions (strict 2.0%, HUD-only 3.7%) | 5.0%* | 5.6% | 3.7%* | E | Franzen guards (on), memory-v2 action-effect notes, indicator-notes (#59) |
| 9 | Uncaught python error before any action | 5.4% | – | 6.5% | E | sandbox-robust (#76) for Name/Import/Attribute (2.7%); **nothing for Type/Index/Value/Key/Syntax (2.7%)** |
| 10 | Tail after the last action + stop-requested turns | 3.2% | – | – | O | **none (endgame-commit, below)** |
| 11 | Re-sent opener on yield resumes | 4.9% of uncached prefill (0.53M per run; first reported as 31.7%, issue #130) | – | – | O | yield-6k (below) and `ARC3_OPENER_DEDUP=resume_image`, both built in prefill-v2 (#129); `ARC3_RESUME_PROMPT_IMAGES=0` is a research-sweep idea for the images only |
| 12 | Request timeouts (68) | 0 counted | – | 0.2% | E | none needed: they hit at game deadlines |
| 13 | Python timeouts (43) | 0.1% | – | 0.1% | E | fix-5 (#51), sandbox-robust (#76) |
| 14 | Guard refusals (396 tool results) | 0.0% | – | 0.0% | E | none needed: they stop a batch after a no-op and cost almost nothing |
| 15 | Coordinate clamps | 0 | 0 | 0 | – | none needed: none happen |

\* Tokens and wall of responses whose every action was a no-op. 32.8% of those tokens are on never-cleared levels.

The token split of the exclusive response causes, per run:

| cause | control | temp06 | audit60k | streams14 | ledger-r1 | ledger-r2 | stack-v1 | pooled | of which on never-cleared levels |
|---|---|---|---|---|---|---|---|---|---|
| probe_no_action | 38.8% | 39.8% | 41.7% | 43.6% | 40.6% | 38.0% | 42.3% | 40.7% | 13.6% |
| action_effect | 40.9% | 41.2% | 38.7% | 37.4% | 39.1% | 39.6% | 37.8% | 39.2% | 12.0% |
| length_cutoff | 6.7% | 6.2% | 7.4% | 4.7% | 6.5% | 7.6% | 5.6% | 6.3% | 3.6% |
| python_error | 5.1% | 4.2% | 4.5% | 6.5% | 5.9% | 5.0% | 6.1% | 5.4% | 1.6% |
| action_noop_only | 5.1% | 4.8% | 5.0% | 4.2% | 4.8% | 5.9% | 5.1% | 5.0% | 1.6% |
| action_level_clear | 2.3% | 2.3% | 1.6% | 2.1% | 2.2% | 2.5% | 2.0% | 2.1% | 0.0% |
| action_death | 1.0% | 1.3% | 1.1% | 1.2% | 0.7% | 1.2% | 1.0% | 1.1% | 0.4% |
| python_timeout, no_tool_call, guard_refusal | 0.2% | 0.3% | 0.0% | 0.3% | 0.2% | 0.2% | 0.1% | 0.2% | 0.0% |

## Per-run comparison

| run | score | tokens | on never-cleared levels | length cut-off | python error | probe, no action | steps ≥12k tokens (share of tokens) | no-op actions | died-attempt + death actions | actions over human on cleared levels | wait before first turn (wall) | tail after last action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control | 46.25 | 4.57M | 29.4% | 6.7% | 5.1% | 38.8% | 78 (34.7%) | 5.3% | 12.0% | 369 (6.6%) | 11.8% | 1.8% |
| temp06 | 45.92 | 4.58M | 32.7% | 6.2% | 4.2% | 39.8% | 71 (31.5%) | 5.8% | 11.3% | 290 (5.6%) | 10.8% | 3.0% |
| audit60k | 44.10 | 4.66M | 38.2% | 7.4% | 4.5% | 41.7% | 95 (41.0%) | 5.6% | 12.3% | 274 (5.5%) | 11.3% | 2.7% |
| streams14 | 46.38 | 5.25M | 27.1% | 4.7% | 6.5% | 43.6% | 85 (35.3%) | 5.9% | 14.1% | 341 (6.4%) | 5.3% | 2.1% |
| ledger-r1 | 52.56 | 4.51M | 31.4% | 6.5% | 5.9% | 40.6% | 71 (31.0%) | 5.0% | 11.8% | 295 (5.6%) | 11.8% | 4.0% |
| ledger-r2 | 49.14 | 4.53M | 31.0% | 7.6% | 5.0% | 38.0% | 70 (31.2%) | 5.9% | 17.4% | 576 (10.2%) | 11.8% | 2.9% |
| stack-v1 | 43.32 | 4.62M | 40.7% | 5.6% | 6.1% | 42.3% | 67 (33.9%) | 5.8% | 21.3% | 456 (7.9%) | 20.5% | 2.6% |
| F001-panel | 29.93 | 0.94M | 21.3% | – | – | – | – | – | – | 79 | 2.5% (to first action) | 3.7% |
| F001-off-panel | 29.22 | 0.94M | 35.1% | – | – | – | – | – | – | 42 | 2.5% (to first action) | 2.9% |

What stands out:
- **stack-v1 loses on the level split, not on failures.** It has the most tokens on never-cleared levels (40.7%), the most actions in died attempts (21.3%: 12 deaths after 40+ actions, 1,126 actions) and twice the wait for a first slot (20.5%, from D′'s start order). Its cut-off and error shares are ordinary. This matches the sim-v2 diagnosis that D′ lost in level outcomes.
- **audit60k fired its audit and still spent the most on stuck levels**: 38.2% of tokens, and 95 steps over 12k tokens, the highest count.
- **The ledger runs do not have fewer failures.** Their cut-off, error, no-op and probe shares are at control level. What differs is the level costs: cleared levels cost a median of 18.1k / 21.1k tokens, against 23.9k for control.
- **Per-cause shares are stable across runs** (cut-off 4.7–7.6%, errors 4.2–6.5%, no-op actions 5.0–5.9%). So the failure causes are a property of the harness and model, not run noise.

Cleared levels cost a median of 18–25k tokens (p90 52–95k). Never-cleared final levels cost a median of 75–108k. In each run, 4–11 games sink 100k or more into a level they never clear: 0.48M (control) to 1.36M (audit60k) tokens.

## Python errors by class

Counts are uncaught errors in tool results across all responses. "Tokens" counts the responses where the error was the outcome, before any action ran.

| class | control | temp06 | audit60k | streams14 | ledger-r1 | ledger-r2 | stack-v1 | total | tokens | covered by |
|---|---|---|---|---|---|---|---|---|---|---|
| NameError | 50 | 42 | 42 | 65 | 55 | 43 | 42 | 339 | 550k | sandbox-robust (persisted variables, globals/builtins) |
| StaleStateActionError | 38 | 32 | 40 | 40 | 32 | 45 | 42 | 269 | 0 | guard working as designed (follows an executed no-op) |
| TypeError | 29 | 26 | 30 | 44 | 21 | 20 | 24 | 194 | 322k | – |
| IndexError | 23 | 15 | 14 | 19 | 30 | 10 | 33 | 144 | 202k | – |
| KeyError | 26 | 18 | 17 | 17 | 13 | 12 | 32 | 135 | 139k | partly: about a third are harness dict keys |
| ImportError | 17 | 14 | 11 | 16 | 17 | 16 | 17 | 108 | 207k | sandbox-robust (time 26, sys 17, inspect 9, builtins 7, pickle 5 in four runs) |
| ValueError | 12 | 12 | 9 | 29 | 21 | 9 | 12 | 104 | 144k | – |
| AttributeError | 6 | 10 | 12 | 27 | 12 | 6 | 8 | 81 | 112k | sandbox-robust (`HistoryEntryView.step`, `_LazyAnimation.items`, `TransitionView.step`) |
| SyntaxError | 6 | 4 | 5 | 2 | 10 | 14 | 9 | 50 | 67k | – |
| PythonTimeout (30 s) | 7 | 7 | 2 | 8 | 6 | 11 | 2 | 43 | 0* | fix-5 (#51): a timeout wipes retained functions |
| TerminalStateActionError | 1 | 3 | 6 | 4 | 0 | 5 | 10 | 29 | 0 | guard |

\* The responses that timed out are the separate `python_timeout` cause (35k tokens).

Error chains are short. In control and ledger-r2 there are 128 / 113 single erroring responses, 15 / 8 pairs and 3 / 1 triples. The model fixes an error on the next call, so an error costs about one response (median 1.2k tokens over 959 such responses), not a spiral. In four runs, 17 NameErrors are `open` and 7 are `globals`: the model tries to persist state through files and globals.

## Causes with no arm: proposed arms

### 1. yield-6k: stop re-opening turns every 2,048 tokens (env-only)

**Evidence.**
- With `LOCAL_ANALYZER_YIELD_TOKENS=2048`, a turn that has generated 2,048 tokens without acting ends and is re-invoked at once.
  - The solver just `continue`s (`solver.py` ~1097–1106 on `franzen/base`, the harness these runs used; ~719–727 on main). The gate slot is not released; it is released only at a trim handover.
  - The stop and deadline checks run before every request anyway (`control_yield_reason` at the loop top).
  - So the only effect of a yield is a new invocation that re-sends the opener with the current-grid image (`ARC3_RESUME_PROMPT_IMAGES` on). Measured, that is about 1,000 tokens, not the "2–3k tokens restating state" first assumed.
- **Yields per run:** 449–635 (3,519 in 7 runs). The tokens of yielded invocations are 46.6% of all tokens.
- **Prefill cost:** a yield's own cost is its resumption opener. The corrected overlay of `analysis/failure_taxonomy.py` (PR #77) sizes it at about 1,070 tokens (the text at 3.9 characters per token and one 400-token image); `analysis/prefill_levers.py` in PR #129 counts 976–1,040 with the tokenizer. That is 0.47–0.68M uncached prompt tokens per run, **4.6–5.2% of uncached prefill (4.9% pooled)**. PR #129 gets 0.45M per run, 4.3–5.0%.
  - The first request of a resumed invocation (3,496 of them) averages 6.95k uncached prompt tokens: 24.3M in all, 31.7% of prefill. **That gross figure is not a yield cost.** It also holds the previous reply and its tool result, which the next request prefills whether or not the turn yields. An earlier version of this report charged all of it to the yield (issue #130).
  - Example, control bp35 step 9: a 12,288-token cut-off, then a resume whose first request prefilled 13,282 uncached tokens. That is about the cut-off reply plus a 1,000-token opener, and only the opener is the yield's.
- **Context cost:** each resume opener also stays in history until a trim, so trims come sooner.
- **Effect of a larger budget**, by replaying the observed response sequence (this model reproduces the observed count: 469 against 475 at 2,048):
  - 4,096 tokens: about 45% fewer resumes (3,544 → 1,944).
  - 6,144 tokens: about 65% fewer (control 469 → 154; 3,544 → 1,242 over the 7 runs).
  - At 6,144 that removes 2,302 resumes over the 7 runs (about 329 per run), and with them about 0.35M opener tokens per run. PR #129's replay with the tokenizer also charges the no-tool-call nudge where a removed resume followed a bare reply. It gives **0.28–0.35M uncached prefill tokens per run, 2.8–3.2% of prefill.**
  - An earlier version of this section said about 2.2M per run (about 21% of all prefill). That counted the resumed requests' whole uncached prompt, including the previous reply and its tool result (issue #130).

**Arm.** `yield-6k`: `LOCAL_ANALYZER_YIELD_TOKENS=6144`, everything else control. A variant also sets `ARC3_YIELD_RESUME_PROMPT=state_only`. PR #129 (`arm/prefill-v2`) built the arm and screened it in sim2 (below). A ≤0.3-point effect cannot be confirmed by two full runs, so run it inside a stack, or once as a behaviour check (yield rate, actions per token) against control.

**Risks.**
- The resume opener may act as a decisiveness nudge. Against that, resumed invocations act only 50% of the time (fresh turns 77%), and their first reply is longer (median 1,587 tokens against 996).
- Franzen found that resume-prompt wording variants made no consistent difference (WRITEUP).

**Expected gain.** Close to zero from prefill. 0.28–0.35M tokens at the calibrated 11.6k tok/s is 24–30 s of GPU prefill in a 7,260 s run (about 0.4%), plus slightly later trims. At matched wall-clock, sim2 puts yield 6,144 at −0.06 points (no credit for freed prefill time) to +0.09 (full credit) on no-ledger level costs, and +0.03 to +0.26 on ledger costs (PR #129, `research/reports/prefill-v2.md` §2). A 25-game score's run-to-run sd is about 2.4, so that is far inside the noise. If the arm is worth anything, it is through behaviour (the risks above), not prefill. An earlier version of this section said +0.5 to +1.5 points, from the overstated prefill (issue #130).

### 2. endgame-commit: act before the clock runs out

**Evidence.**
- Tokens generated after a game's last action, when the deadline cut it off: 83k–182k per run, **2.7% pooled (0.89M)**.
- Turns cut by the stop request: another 0.5% (0.16M).
- So 3.2% of tokens produce no action at all.
- A step takes a median of 33–47 s of wall-clock, p75 74–112 s and p90 150–226 s. A game that starts a long step in its last 1–3 minutes loses it.

**Arm.** When a game's remaining clock is below about 90 s (p75 of step wall-clock), add one opener line ("about N s left in this game: execute your best candidate in this reply") and send `reasoning_effort=low` (reuse the PR #73 plumbing). The research-sweep Tier 3 "endgame bank-it line" is the prompt half of this; it was never built, and the earlier spend-actions prompt (c5cfbb0) was negative, so keep it to the last 90 s.

**Expected gain.** It converts part of 3.2% of tokens into actions on mostly stuck levels: +0.2 to +0.5. A ride-along, not a standalone arm.

### 3. api-schema-errors: the code errors sandbox-robust does not touch

**Evidence.**
- TypeError, IndexError, ValueError, KeyError and SyntaxError: 627 uncaught errors in 7 runs, **0.88M tokens (2.7%)** as the outcome of the response.
- About a third of the KeyErrors (28 of 83 in the four runs read by hand) are harness dict keys: `'steps'` 12, `'action'` 5, `'frames'` 3, `'stop_reason'`, `'noop'`/`'no_op'`, `'gameplay_changed'`, `'executed_actions'`.
- 96 of the 244 TypeError, IndexError and ValueError tracebacks in those four runs are "list index out of range" or "min()/max() of an empty sequence": a search found nothing.

**Arm.** Add the exact key list of `action(...)`'s return value, `last_action_call_result` and the animation record to the python tool description (the research-sweep "tool-doc" item). Then retry the KeyError rate. Ride along with sandbox-robust (#76).

**Expected gain.** At most about 1% of tokens. The rest are ordinary model bugs that a harness cannot prevent cheaply.

### Not proposed (already tried or refuted)

- **Stuck levels** (cause 1) already have arms: a scheduler that takes slots from stalled games (D′, UCB) and an audit prompt (audit60k). Both lost.
  - The fresh-eyes reset was refuted in research-sweep: 14 of 25 control levels that ran past 60k tokens were still cleared.
  - This report's numbers agree. Cleared levels reach a p90 of 52–95k tokens, so no fixed token trigger separates stuck levels from slow ones.
- **The no-action circuit breaker** for cause 2 was rejected in research-sweep on Kinetic v8's evidence.

## Fable panels (benchmark only)

- **Missing files:** `v4-fable-f001/v2` and `v4-fable-f001-off/v1` saved no transcripts, request logs or events. Their notebook logs show `save_request_logs=True`, but the files were not downloaded into `runs/`. So no cause split is possible.
- **Level split:** with the layer on, 21.3% of tokens and 7.7% of actions were on the level each game was still playing at the 25-minute limit. With it off: 35.1% and 33.7%.
- **Request timeouts** in the logs: 16 (on) against 10 (off).
- **Caveat:** with one run each and every game stopped by the clock (no per-game token cap: `max_generated_tokens_per_game=None`; about 94k tokens per game), this is not evidence for the layer.

## Verification

**Spot checks by hand** (transcript, request log and benchmark read directly; tool rows from `responses.csv.xz`):

1. **ledger-r2 sb26, step 1** (21:18:23). The request log has two responses for step 1: 95 tokens (a `print(current_frame.shape, ...)` probe) and 523 tokens (`action([{'action':'MOUSE','row':58,'col':19}])` → `[action] MOUSE(row=58, col=19) -> effect`).
   - The tool rows are `probe_no_action` 95 and `action_effect` 523.
   - Their sum, 618, equals `benchmark.json` `history[0].generated_tokens` for the first action (ACTION6 x=19 y=58).
   - Step 2 has 359 tokens in both the request log and `history[1]`.
2. **control bp35, step 9** (01:36:10). The second response's META reads `finish_reason: length`, `tool_call_count: 0`, `reasoning_chars: 37143`. The request log for step 9, request 2 has `completion_tokens` 12,288 (prompt 75,416, cached 73,536).
   - The invocation ends `message: Yielded control to solver: turn_token_budget.`, with no follow-up USER PROMPT.
   - "CUT OFF" appears nowhere in the file.
   - The resumed invocation at 01:39:01 prefilled 13,282 uncached tokens and acted with a 561-token reply.
   - The tool row is `length_cutoff`, 12,288 tokens, 166 s of the 171 s invocation, level 1 (cleared).
3. **ledger-r2 sk48, step 24** (22:41:33). The second response's TOOL RESULT is `Traceback ... NameError: name 'D' is not defined`; the request log has step 24, request 2 at 1,957 tokens. The tool row is `python_error`/`NameError`, 1,957 tokens.
   - Step 24 took 10 responses and about 15k tokens over several yields before it acted: an instance of cause 3.
4. **No-ops against the harness's own verdicts** (ledger-r2): where a step ran one action, the transcript echo (`-> NO-OP` / `-> effect`) and the tool's board-based no-op agree in 377 of 381 cases. In the other 4, the tool says HUD-only and the harness said effect.
   - The echoed action count equals the event count for 1,079 of 1,110 executed steps. The 31 others are steps whose echo lines did not reach the printed output, plus automatic RESETs.

**Cross-check against `summary.txt` and `benchmark.json`** (all 9 runs):

| run | request-log tokens | `solver_note` tokens | tool-attributed tokens | `summary.txt` tokens = benchmark history sum | `summary.txt` actions = benchmark = events |
|---|---|---|---|---|---|
| control | 4,561,854 | 4,565,244 | 4,565,348 | 4,482,155 | 5,632 |
| temp06 | 4,576,777 | 4,579,302 | 4,579,267 | 4,442,199 | 5,136 |
| audit60k | 4,647,133 | 4,657,402 | 4,656,576 | 4,533,757 | 4,991 |
| streams14 | 5,241,649 | 5,245,465 | 5,245,976 | 5,136,753 | 5,318 |
| ledger-r1 | 4,504,101 | 4,512,572 | 4,511,212 | 4,330,950 | 5,278 |
| ledger-r2 | 4,523,498 | 4,532,994 | 4,533,898 | 4,401,932 | 5,654 |
| stack-v1 | 4,617,193 | 4,621,469 | 4,620,139 | 4,499,736 | 5,754 |
| F001-panel | – | 941,330 | 941,330 | 906,946 | 810 |
| F001-off-panel | – | 935,749 | 935,749 | 908,225 | 822 |

Reconciling the columns:
- **`summary.txt` tokens equal the benchmark history sum in every run.** They count only tokens charged to an action, so they leave out the tail after a game's last action. `solver_note` minus `summary.txt` equals the tail exactly (control: 4,565,244 − 4,482,155 = 83,089).
- **The request log lacks exactly one response per run**: the in-flight reply when the run ended. The transcript has its META, and the harness counted it in `solver_note`. The tool estimates it from its characters, which leaves the attributed total within ±0.03% of `solver_note` (control +104 tokens, audit60k −826).
- **Actions match exactly.** The benchmark history, `summary.txt` and the events agree, and the tool's action causes sum to the total.
- **Wall-clock buckets** sum to the games' `final_wallclock_seconds`, within a few seconds of header rounding.
- **Unanswered requests:** 9–13 per run, which is the request timeouts plus the final in-flight request.
- **Request timeouts happen at the deadline.** Their read timeout is the game's remaining clock (1–102 s in ledger-r2). The 16 that followed a logged response all started within 125 s of their game's end.
- **Other mismatches:**
  - 1 action per run has no response: tn36's initial RESET (action 1 at `analysis_step` 0).
  - The `levels.csv` convention on arm/sim-v2 (`level_dataset.py`) charges tail tokens to the final level, as this tool does.

## Could not verify

- **The panels' causes.** Transcripts, request logs and events were not saved into `runs/`.
- **Wall-clock per response is a split, not a measurement.** Transcripts timestamp only invocations (to the second), so each invocation's time is shared out by generated tokens. Gate waits at trims are inside these times, and prefill time is not separated.
- **Why a no-op happened.** The tool cannot tell a deliberate probe of a new key from a mistake; the 3.7% HUD-only no-ops include the first press of every key.
- **The death cause.** "Bar exhaustion" is inferred from attempt length (40+ effectful actions), not read from the frames. budget-alert's replay matched 17 of 17 bar deaths and is the better source.
- **The yield-6k and endgame-commit gains** are estimates from replaying the recorded responses. The model's behaviour would change with the budget, so they need runs.

## Reproduce

```sh
R=runs/kaggle
nice -n 15 python3 analysis/failure_taxonomy.py \
  $R/fz-arms/v4 $R/fz-arms/v5 $R/fz-arms/v6 $R/fz-streams14/v1 $R/fz-ledger-on-trim/v1 \
  $R/fz-ledger-on-trim-r2/v1 $R/v4-stack-v1/v1 $R/v4-fable-f001/v2 $R/v4-fable-f001-off/v1 \
  --label fz-arms/v4=control --label fz-arms/v5=temp06 --label fz-arms/v6=audit60k \
  --label fz-streams14/v1=streams14 --label fz-ledger-on-trim/v1=ledger-r1 \
  --label fz-ledger-on-trim-r2/v1=ledger-r2 --label v4-stack-v1/v1=stack-v1 \
  --label v4-fable-f001/v2=F001-panel --label v4-fable-f001-off/v1=F001-off-panel \
  --md tables.md --json runs.json --responses-csv responses.csv
```

About 27 s for all nine runs on the laptop, one run at a time, peak memory about 100 MB.
