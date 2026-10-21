# G-A: Where the long-turn token tail sits, and what stalls are made of (critic G3, G13, G21)

Local log analysis, 2026-10-18. No web sources were needed: every number below comes from the 85 usable request-log runs (L1's set: 65 `v4-*` and 20 `fz-*` Franzen + Flash-Next runs, 1,642 game-runs).

**Scripts and outputs.** They live in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/ce155f4d-514a-4385-ac65-a7e7e5a23e1c/scratchpad/ga/` (written `ga/` below) and ran under `nice -n 15`, one at a time.
- `scripts/reqcalls.py` → `reqcalls.json`: the exact `usage.completion_tokens` of every response row, in file order. It found 1,642 game-runs and 141,483 responses, the same as L1.
- `scripts/ev.py` → `ev.json`: action rows plus each call parsed from the events transcripts (tool result, error, acted), paired with request-log usage by file order.
  - 1,556 game-runs paired one-to-one with 0 step mismatches; the other 86 split each step's exact tokens across its calls by characters. 20,286 tokens could not be assigned (`ev.log`).
  - Total 269,595,809 tokens, matching L1's 269.6M.
- `scripts/ana.py` → `ana.json`, `ana_steps.json`, `ana_bt.json`; `scripts/ana2.py` → `ana2.json`; `scripts/turnmix.py`, `sens.py`, `errtypes.py` (printed output); `scripts/extract.py` → `audit.txt`.

**Definitions** (they reuse L1's `decisions2.py` conventions):
- **Turn:** one analysis step, i.e. every model call up to and including the call that acts. In the ft09 sample, the step totals from `benchmark.json` equal the sum of per-call `completion_tokens` (126,527 = 126,527). This is L3's "deciding turn".
- **Level of an action:** the level before it.
- **New board:** an interior hash (2-cell edge crop) not yet seen in this level attempt.
- **Stall:** 15 or more consecutive actions without a new board.
- **Level classes:**
  - **C0:** cleared, never stalled.
  - **C1:** stalled, later cleared in the same run.
  - **Nx:** not cleared in this run, but cleared by another of the 85 runs.
  - **N0:** never cleared by any run.
- **On-route:** the turn's first action is an edge of the shortest path, found by BFS over the observed board→board edges, from the level's first board to the board the clearing action was taken from.
- **Leader:** `v4-mem-repair-10x/v1`, `v4-mem-unified-10x/v1`, `v4-stall-fix-10x/v1-snowiers` and `v4-user-combo-10x/v1` (100 game-runs, 21.4M tokens).

## Q1 (G3). Token distribution by turn length and level class; what R3 can act on

### Takeaway
- An R3 recipe that touches only states on cleared levels acts on **65.7%** of generated tokens (leader **72.0%**).
- Restricted to on-route states (the ones with a verifiable target), it acts on **40.9%** (leader **44.8%**). Restricted further to levels cleared in under 50k tokens, on **25.4%**.
- Turns of 5k tokens or more hold 69.3% of tokens. **29.4 points of that sit on stalled or never-cleared levels**, which is 42% of the long tail. In turns of 12k or more, never-cleared levels hold 44.5% of the tokens.
- The long turns are mostly long **loops of calls**, not single long thoughts. A 12k+ turn averages 7.7 calls, and 85.5% of its tokens are in calls that did not act.

### Cited Findings
- **Share of all 269.6M generated tokens, all 85 runs** (`ga/ana.json` `all_turn_table_pct`):

  | Level class | <1k | 1–5k | 5–12k | 12k+ | All |
  |---|---|---|---|---|---|
  | C0 on-route | 1.82 | 11.46 | 12.87 | 13.29 | **39.44** |
  | C0 off-route | 0.80 | 7.29 | 7.28 | 6.47 | 21.83 |
  | C1 on-route | 0.10 | 0.34 | 0.49 | 0.53 | 1.46 |
  | C1 off-route | 0.10 | 0.98 | 1.06 | 0.83 | 2.97 |
  | Nx | 0.69 | 6.58 | 8.87 | **16.26** | 32.40 |
  | N0 | 0.07 | 0.51 | 0.63 | 0.68 | 1.89 |
  | All | 3.59 | 27.15 | 31.20 | 38.06 | 100 |

- **The same table for the 4 leader runs** (`lead_turn_table_pct`):

  | Level class | <1k | 1–5k | 5–12k | 12k+ | All |
  |---|---|---|---|---|---|
  | C0 on-route | 2.06 | 12.01 | 14.42 | 14.87 | 43.35 |
  | C0 off-route | 0.79 | 6.92 | 7.78 | 8.06 | 23.55 |
  | C1 (on / off) | — | — | — | — | 1.48 / 3.93 |
  | Nx | 0.45 | 4.03 | 5.87 | 15.16 | 25.51 |
  | N0 | 0.04 | 0.41 | 0.57 | 1.16 | 2.18 |
  | All | 3.54 | 24.60 | 30.84 | 41.02 | 100 |

- **Tail shares** (`ga/ana2.json` `all_tail` / `lead_tail`):
  - Turns of 5k or more: 69.26% of tokens (leader 71.86%).
  - Of that, on C1 + Nx + N0: 29.36% of all tokens (leader 26.74%), or 42.4% of the ≥5k tokens (leader 37.2%). On never-cleared levels alone: 26.45%.
  - ≥5k turns on cleared on-route states: 27.18% of all tokens (leader 30.33%).
  - Never-cleared levels hold 44.5% of the tokens in 12k+ turns (leader 39.8%).
  - Median turn: 1,934 tokens on C0 vs 2,923 on Nx (on-route C0 1,778; off-route C0 2,194).
- **The same split by level token cost** (`ga/ana2.json` `all_levelcost`; level attempts in brackets):

  | Level | Share of tokens | On-route |
  |---|---|---|
  | Cleared in <50k tokens (5,015) | 36.01% | 25.44% |
  | Cleared in 50–100k (828) | 20.82% | 11.30% |
  | Cleared in ≥100k (191) | 8.88% | 4.16% |
  | Not cleared, <100k (844) | 13.51% | — |
  | Not cleared, ≥100k (396) | 20.78% | — |

  - Leader: 34.41 (24.72) / 25.12 (13.82) / 12.79 (6.29) / 6.10 / 21.59.
  - So levels of 50k tokens or more plus every never-cleared level make up about 60–66% of tokens. This is probably the origin of the "50–60% on stalled levels" figure (`rsi-proposals-2026-10-15.md:30`). That reconciliation is my inference.
- **What the long turns are made of** (`scripts/turnmix.py`):

  | Turn length | Turns | Calls per turn | Tokens in non-acting calls | Largest call's share of turn | Tokens in `finish=length` calls |
  |---|---|---|---|---|---|
  | <1k | 19,385 | 1.06 | 4.3% | 97.3% | 0.0% |
  | 1–5k | 28,819 | 1.72 | 26.6% | 81.8% | 0.0% |
  | 5–12k | 11,093 | 3.08 | 49.4% | 66.8% | 0.2% |
  | 12k+ | 4,876 | **7.67** | **85.5%** | **39.9%** | **17.6%** |

- **Per call instead of per turn** (`all_call_table_pct`): <1k 12.2%, 1–5k 54.5%, 5–12k 26.4%, **12k+ only 6.85%**.
- **Route-definition sensitivity:** counting a turn as on-route if *any* of its actions is a route edge raises C0 on-route from 39.44% to 43.72% of tokens (`ana.json` `C0-anyroute`).
- **Turns with no action:** 850 turns (3.56% of tokens) executed nothing. All of them fall on Nx/N0 levels: they are the turns the clock ran out on.

### Inferences
- **Arithmetic:**
  - A −25% cut on on-route cleared states (40.9%) is about **−10% of total tokens**.
  - The same cut on cheap cleared levels (25.4%) is about −6%.
  - Both assume the trained model changes nothing on other states. Its spill-over onto stalled and never-cleared states, positive or negative, is the unmeasured term.
- **The tail is mostly multi-call inspection loops.** A per-call clip or a "shorter thinking" target leaves untouched the 85% of 12k+ turn tokens that sit in non-acting calls. So R3 targets must be whole turns (state → acting call, all intermediate calls included), scored on total turn tokens. Resampling single calls in state-lab would miss most of the tail.
- `finish=length` calls (12,288-token caps) are 17.6% of 12k+ turn tokens and 6.8% of all tokens (Q4). This is a harness-addressable slice.

### Gaps
- **"Never-cleared" is censored by the clock.** Levels only advance on a clear, so each of the 1,170 uncleared attempts is the final level of its game-run. Some would clear with more time; L3 makes the same point.
- **The crops are approximate.** The HUD crop is L1's 2 cells. One game (tr87) shows 0.0% no-ops, which suggests a counter leaks through there (`scripts/pergame.py`).
- **Small mismatch with L1.** Cleared-level share here is 65.7% vs L1's 68%. My per-call attribution puts final no-action turns on the last level, and L1 noted 56 misaligned `benchmark.json` histories.

## Q2 (G13). Novelty per action: stalled vs cleared, and around breakthroughs

### Takeaway
- Novelty-defined stalls are rare and cheap:
  - 302 of 7,204 level attempts;
  - **0.6% of tokens** are spent while 15 or more actions have passed without a new board.
- During a stall, the model mostly **repeats** (state, action) pairs: 18% new pairs, vs 94% elsewhere. After a breakthrough it returns to 87% new pairs and 74% new boards.
- Whole levels differ only modestly. At the same action index, cleared and never-cleared levels try new pairs at almost the same rate; the never-cleared new pairs reach new boards 8–13 points less often.

### Cited Findings (`ga/ana.json` `*_q2_actions`, `*_q2_breakthrough`; `ga/ana2.json` `novelty_by_index`; `scripts/sens.py`)
- **Level counts.** Cleared without a stall 5,843; cleared after a stall (C1) 191, which matches L1's 191; not cleared with a stall 111; not cleared without one 1,059.
- **Per-action rates, all 85 runs:**

  | Class | Actions | New pair | New pair (clicks bucketed 4×4) | New board | No-op | Death |
  |---|---|---|---|---|---|---|
  | C0 | 202,353 | 96.0% | 95.9% | 86.8% | 5.4% | 0.23% |
  | C1 | 22,286 | 71.4% | 71.3% | 59.2% | 6.1% | 0.31% |
  | C1 in stall | 2,595 | 16.4% | 16.1% | 9.2% | 5.7% | 0.50% |
  | Nx | 60,331 | 89.1% | 88.9% | 73.6% | 8.9% | 0.55% |
  | N0 | 4,550 | 90.2% | 90.2% | 73.4% | 14.2% | 0.11% |
  | Any class, in stall | 4,210 | 19.7% | 19.0% | 8.4% | 10.7% | 0.69% |
  | Any class, not in stall | 285,310 | 93.7% | 93.5% | 82.8% | 6.3% | 0.30% |

  The leader runs are similar: C0 96.7 / 88.7, Nx 89.5 / 75.9, in stall 12.7 / 9.7.
- **At matched action index within the level** (new pair / new board):

  | Actions into the level | Cleared | Never cleared |
  |---|---|---|
  | 0–19 | 96.1 / 83.9 | 94.8 / 76.0 |
  | 20–49 | 93.6 / 85.9 | 90.3 / 73.4 |
  | 50–99 | 90.2 / 83.8 | 86.1 / 71.8 |
  | 100–199 | 82.0 / 77.3 | 81.7 / 72.5 |

- **Breakthroughs** (a new board after a stall; 354 events: 240 on cleared levels, 114 on uncleared):

  | | Cleared levels | Uncleared levels |
  |---|---|---|
  | 15 actions before: new pair | 15.8% | 22.2% |
  | 15 actions before: no-op | 9.6% | 16.8% |
  | 15 actions after: new pair | 88.1% | 83.9% |
  | 15 actions after: new board | 78.5% | 63.6% |
  | Breakthrough action is a new pair | 95.8% | — |
  | ...the first use of that action type on the level | 7.5% | — |
  | ...the first click in a new 4×4 bucket | 12.9% | — |
  | ...on the hindsight route | 67.9% | — |

  Median stall length 20 actions; median 31 actions from breakthrough to clear.
- **Threshold sensitivity.** Token share of turns decided after at least k actions without a new board: k ≥ 1: 35.2%; ≥ 3: 7.3%; ≥ 5: 3.8%; ≥ 10: 1.2%; ≥ 15: 0.6%.
- **In-stall turns** batch more actions and cost less per action: 7.46 actions per turn at 448 tokens per action, vs 4.55 and 904 outside stalls (`all_turn_phase`).

### Inferences
- **The difference is measurable but is not an exploration deficit.** Never-cleared levels try new pairs at about the cleared rate; their new actions just pay off less. This supports the critic's C1 resolution: drop novelty rewards and the "explores too little" side-model rationale.
- **The breakthrough is rarely a never-tried action type** (7.5%). It is usually a new pair inside a plan already running (68% on-route). Cheap batched repeats look like planned replays, which Q3 confirms.

### Gaps
- The interior hash counts any pixel change as a new board, so "new board" is a weak proxy for progress. The logs give no counterfactuals.

## Q3 (G13). Audit of 30 stalls against their clears

### Takeaway
- 8 of the 30 "stalls" are not reasoning stalls at all. They are **planned revisits**: a RESET and replay of a known route, a rewind mechanic, a deliberate bar-burn to force a reset, or a search over a cycle.
- Of the 22 real stalls:

  | Cause | Count | Share |
  |---|---|---|
  | **Mechanic error** | 12 | 55% |
  | **Goal inference** | 5 | 23% |
  | Execution / planning | 4 | 18% |
  | Perception | 1 | 5% |
  | Budget / time | 0 (as primary cause) | — |

- Counting the underlying cause behind the artifacts too: mechanic 17, goal 5, execution 4, perception 1, none identifiable 3.

### Cited Findings (`ga/audit.txt`, from `scripts/extract.py`, seed 7)
- **Sample** (at most 2 per game, 20 games):
  - 20 C1 stalls with a later clear in the same run;
  - 5 Nx stalls compared against another run's clear of the same level (same harness family where available);
  - 5 expensive clears (≥100k tokens on the level, no novelty stall), to cover the token-heavy levels that Q2 shows novelty misses.

  For each, I read the first, middle and last stall turns, the breakthrough turn and the clear (first 200–500 and last 700–1,500 characters of the acting call's thinking).
- **Counts by subset:**

  | Subset | Artifact | Mechanic | Goal | Execution | Perception |
  |---|---|---|---|---|---|
  | Same-run stalls (20) | 5 | 7 | 3 | 4 | 1 |
  | Cross-run (5) | 3 (underlying mechanic) | 1 | 1 | 0 | 0 |
  | Expensive clears (5) | 0 | 4 | 1 | 0 | 0 |

  Perception was also a secondary factor in 2 cases (#15 cn04, #29 re86); budget was mentioned in 4 (#3, #11, #19, #24).
- **Five quoted examples:**
  1. **Goal inference.** `v4-mem-repair-10x/v1` lf52 L2, step 25 (`audit.txt:10`): "So the level goal: reduce each region to 1 peg → key appears. Now what? Maybe the key needs to be moved to the orange cell (lock)." The level actually cleared by jumping into the target to leave 1 peg (`audit.txt:43`).
  2. **Mechanic error.** `v4-uc-11x/v1-snowiers` wa30 L4, 45 actions without a new board: "I'm going to monitor and burn actions while the bots work" (`audit.txt:98`). Then: "The bots are completely stuck… maybe the bot needs to be carrying something" (`audit.txt:103`).
  3. **Mechanic error.** `v4-control/v1` dc22 L1, a 70-action click sweep: "Confirmed: only the two hat clicks do anything… maybe the game's rule is inverted: the SOLID blue is a 'hole'/'passage'" (`audit.txt:709,719`).
  4. **Perception, through its own parser.** `v4-uc-18s-64k/v1` tr87 L5: "the solver's ok() check was broken (read empty), so it never detected a match even if present!" (`audit.txt:596`).
  5. **Artifact (deliberate burn).** `fz-streams14/v1` sc25 L6: "I'll burn it: batch LEFT/RIGHT oscillation ~20 times" (`audit.txt:1100`).
- **Execution errors were loops in the model's own helpers.**
  - ar25: the plan was "move_active(9,9) then SPACE then 7 UP" (`audit.txt:383`), but the 17 stall actions were DOWN/UP alternations (`:378`).
  - tu93: "my strategy's first move must not be the oscillating one" (`:750`).
- **The expensive clears were spent mapping rules:**
  - "BREAKTHROUGH! … word 33 = b0 + s2 = UP 1" (tn36, `:1512`);
  - push physics (ka59);
  - growth-blocking rules (s5i5);
  - shape connectivity (re86);
  - target-pattern inference from clue lenses (ft09, goal).

### Inferences
- **R4 rationales should target mechanics plus experiment design** ("what does this control do, and which probe tells us"): 12–17 of 30 cases. Win-condition inference (5 of 30) is secondary.
- **Perception SFT** (frame → object list) would address about 1–3 of 30 directly. The perception failures seen were the model's own parsing code misreading cells, not the image.
- **A "no new board for N actions" stall detector fires on planned replays** (8 of 30), so harness interventions keyed on it would interrupt deliberate plans.

### Gaps
- One coder, no second rater, excerpts rather than full transcripts. Categories overlap, so each case got one primary cause; treat the counts as ±3. 30 cases is a screen, not a census.

## Q4 (G21). Sandbox errors and calls that did not act

### Takeaway
- Calls whose sandbox code raised an error hold **9.2% of tokens** (8.9% of calls; leader 10.5%).
- Turns that contain at least one error call hold **32.1%** (leader 37.3%). But that counts whole turns; the follow-up call right after an error costs only 1.26% of tokens.
- Non-acting calls hold 55.3% of tokens. Clean python that ran without acting is **42.5%**, and most of it is deliberate inspection.
- Calls that returned no tool call hold 6.5% of tokens on 1.1% of calls, almost all of them `finish=length`.
- About half of the errors come from the harness, not from model logic.

### Cited Findings (`ga/ana.json` `*_q4`; `ga/ana2.json` `repair`; `scripts/errtypes.py`)
- **All 85 runs, 141,515 calls:**

  | Measure | Share of calls | Share of tokens |
  |---|---|---|
  | Error calls | 8.93% | 9.17% |
  | Turns with an error | 13.9% of turns | 32.11% |
  | Non-acting calls | 55.25% | 55.32% |
  | Non-acting clean python | 47.9% | 42.48% |
  | Non-acting with an error | — | 6.38% |
  | Acting with an error | — | 2.79% |
  | `finish=length` | — | 6.79% |
  | Acting calls | — | 44.68% |

  - Median tokens: error call 1,242, clean call 1,131, acting call 1,191.
  - Error streaks: 9,397 single errors, 1,101 runs of 2, 194 of 3, 70 of 4 or more.
- **Error share by class** (error tokens): C0 9.07%, C1 9.50%, Nx 9.36%, N0 8.76%. In stall 11.88% vs 9.16% outside; in the leader runs 23.1% vs 10.4%.
- **Exception classes** (12,634 error results):

  | Exception | Share | Note |
  |---|---|---|
  | NameError | 23.3% | — |
  | **StaleStateActionError** (no-op guard) | 14.8% | harness |
  | TypeError | 14.1% | — |
  | IndexError | 10.1% | — |
  | KeyError | 9.3% | — |
  | **ImportError** (sandbox whitelist) | 8.1% | harness |
  | ValueError | 6.5% | — |
  | SyntaxError | 4.7% | — |
  | AttributeError | 4.6% | — |
  | TerminalStateActionError | 1.5% | harness |
  | Unparsed | 1.9% | — |

  - The ImportErrors are blocked modules: `time` 380, `inspect` 242, `sys` 145, `builtins` 79, `pickle` 77.
  - Some NameErrors are definitely names lost between calls: "NameError (mv not defined — variables reset)" (`audit.txt`, #12 g50t). The full split is UNVERIFIED.
  - Related open issues: #176 (sandbox gaps) and #18 (the false stale-state message), per `CLAUDE.md`.

### Inferences
- **The harness-made errors can be fixed without training:** the guard (14.8% + 1.5% of errors), imports (8.1%) and lost state (part of the 23.3% NameErrors). That is roughly 25–45% of errors, about 2–4% of tokens, and cheaper and safer than RFT.
- **RFT that keeps only error-free code** targets at most about 9% of tokens plus any downstream confusion inside the 32% of error-turn tokens. The direct repair cost is small (1.3%).
- **"Ran without acting" is mostly the designed observe-then-act loop.** Treat it as the call-count lever from Q1. The clearly wasted slice is `finish=length` (6.8%).

### Gaps
- Error detection is a regex over transcript tool results; "acted" is the turn's `step_executed` flag. Both were checked by hand on the ft09 sample only (8 of 8 errors and 20 of 20 acting calls confirmed).

## Implications for our plan

1. **Restate R3's payoff with the measured share.**
   - R3 on on-route cleared states touches about 41–45% of tokens, so a −25% cut there is about −10–11% of total tokens, not −25%.
   - Only 27–30% of all tokens are long turns (≥5k) on such states.
   - Feed −10% into L3/sim2, not −25%, unless spill-over onto the never-cleared tail is shown (Nx holds 16% of all tokens in 12k+ turns).
2. **Make R3 targets whole turns, and score them on calls per decision.** 12k+ turns are 7.7 calls with 85% non-acting tokens. Concretely:
   - state-lab samples should run the full multi-call turn to the acting call;
   - selection should keep the sample that reaches the same action or board with the fewest total turn tokens;
   - the screen should report calls per turn next to tokens per turn.

   A single-call shortest-of-k recipe (W7's) misses most of the tail.
3. **Drop novelty rewards and stall-count triggers.**
   - Novelty-stalls hold 0.6% of tokens.
   - The model explores at the same rate on levels it fails.
   - 8 of 30 detected stalls were deliberate replays.

   This settles C1 against the novelty rewards (W2, prior plan stage 3) and the side-model rationale (W5).
4. **Aim R4 and any rationale data at mechanic and rule discovery** (12–17 of 30 stalls): hypotheses about what a control does and probe design. Win-condition inference comes second (5 of 30). Perception SFT is low priority (about 1 of 30, and that one a code-parser failure).
5. **Take the cheap harness wins before training:**
   - the StaleState/Terminal guard errors (16% of errors) — issue #18;
   - the sandbox import whitelist (8%) — issue #176;
   - lost cross-call state (part of the NameErrors at 23%);
   - `finish=length` (6.8% of tokens).

   Together they are about 5–10% of tokens, behind flags as usual, and need no node.
6. **Report never-cleared tokens as clock-censored.** Every uncleared attempt is the level the clock ran out on. G2 (a longer budget) is the test of how much of that 32% turns into clears.

## Fact-check (independent verifier, 2026-10-18)

**Method.** I re-derived the numbers from `ga/ev.json` with my own script (`gacheck/recheck.py`, `gacheck/nov.py` in the same scratchpad, all under `nice -n 15`). It has its own level segmentation, route BFS and stall logic. I also re-ran `ana.py` under `PYTHONHASHSEED=1` and `2`, and re-ran `sens.py` and `errtypes.py` (the latter reads the raw event logs). I read `audit.txt` at the cited lines and `L1-data-inventory.md`. All 9 key results were checked; none were skipped. The audit cause counts are one coder's judgement and could not be re-derived.

**Cross-cutting caveat: on-route numbers are not deterministic.** `ana.py`'s BFS walks a `set` of board hashes, so when two shortest routes tie, which one counts as "the route" depends on Python's per-process string-hash seed. Across seeds 1 and 2, the original run and my list-ordered BFS:
- C0-on ranges 39.44–39.52, and all on-route 40.9–41.0;
- leader on-route ranges 44.8–45.2;
- ≥5k cleared on-route ranges 27.18–27.23 (leader 30.33–30.60);
- breakthroughs on the hindsight route range 67.5–68.8%.

None of this changes a conclusion. Quote these as about 41%, about 45% and about 68%, or fix the BFS order.

| # | Result | Verdict | Correction / note |
|---|---|---|---|
| 1 | R3 shares (65.7%, 40.9%, 25.44%) | **Partially confirmed** | All-run numbers reproduce: cleared 65.71 (C0 61.27 + C1 4.44), on-route 40.9–41.0, cheap-level on-route 25.44. **Leader cleared share is 72.31%, not 72.0%** (C0 66.90 + C1 5.41 = 100 − Nx 25.51 − N0 2.18). Leader on-route is 44.8–45.2 depending on tie-breaks. |
| 2 | Tail shares (69.26, 29.36, 42.4, 26.45, 27.18, 44.5) | **Confirmed** | Exact match from my own turn records: 69.26 / 29.36 / 42.39 / 26.45 / 44.51 (leader 71.86 / 37.21 / 39.78). The ≥5k cleared on-route figure is 27.18–27.23 (leader 30.33–30.60). "26.45%" means never-cleared tokens *in ≥5k turns*; all never-cleared tokens are 34.3%. |
| 3 | Long turns are multi-call loops | **Confirmed** | Recomputed exactly: 4,876 turns of 12k+, 7.67 calls, 85.5% non-acting, largest call 39.9%, `finish=length` 17.6%. 5–12k turns: 3.08 calls, 49.4%. Per-call 12k+ share is 6.85%. |
| 4 | Novelty stalls are rare (302 / 7,204; 0.60%) | **Confirmed** | `sens.py` re-run, and my own segmentation, give: 5,843 C0, 191 C1, 111 stalled and uncleared (105 Nx + 6 N0), 1,059 uncleared without a stall. Thresholds: ≥1 35.19%, ≥3 7.33, ≥5 3.81, ≥10 1.22, ≥15 0.60. C1 is 4.43–4.44%. |
| 5 | Novelty per action and breakthroughs | **Confirmed** | In a stall, new pairs are 19.7% and new boards 8.4% (n = 4,210); outside, 93.7% and 82.8%. The index table reproduces exactly. Breakthroughs: 354 in all, 240 cleared; new pairs 15.8% before and 88.1% after, new boards 78.5% after; 7.5% are the first use of an action type. Hindsight-route share is 67.5–68.8% (tie-break), not exactly 67.9. |
| 6 | Stall audit, 30 cases | **Partially confirmed** | The sample (20 A + 5 B + 5 C, seed 7) and quotes at :10, :98, :103, :596, :709, :719 and :1100 match. Case metadata matches too (wa30 had 45 actions, dc22 70, tr87 is case #14). **"At most 2 per game" holds only within each subset**: lf52 (#1, #5, #22), g50t (#6, #12, #24) and sc25 (#10, #11, #23) each give 3 of the 30 cases. `audit.txt` records no cause labels, so the counts 8 / 12 / 5 / 4 / 1 / 0 and "mechanic 17" are unverifiable. The tn36 quote spans :1512–1513 and is paraphrased; the source reads "Col2's word = 33 = b0 + s2 = row-1 bar + row-3 stem = **UP 1**". |
| 7 | Sandbox errors (8.93% of calls, 9.17% of tokens, …) | **Confirmed** | From `ev.json`: 141,515 calls; errors 8.93% of calls and 9.17% of tokens (leader 10.49); error turns 13.90% of turns and 32.11% of tokens. The next call after an error is 1.26% of tokens (n = 12,634 errors). Non-acting calls 55.32%, clean non-acting python 42.48%, `finish=length` 6.79%, no tool call 6.46% of tokens on 1.06% of calls. |
| 8 | Exception classes | **Confirmed** | `errtypes.py` re-run gives 12,634 error results, with the class shares exactly as listed. ImportError breakdown: time 380, inspect 242, sys 145, builtins 79, pickle 77. This is a re-run of the same regex, not an independent parser. |
| 9 | Never-cleared levels are clock-censored; 850 no-action turns | **Confirmed** | There are 0 decreasing level transitions and 0 revisited levels. 0 of the 1,170 uncleared attempts is anything but the last segment of its game-run. All 850 no-action turns (3.56%) come after the game-run's last action, and 0 fall mid-run. 73 of them follow a final clear, so they sit on about 70 levels with no actions; that is why the level-cost table counts 1,240 uncleared levels (844 + 396) against 1,170 elsewhere. The L1 reconciliation holds: cleared tokens are 177.2M here vs 176.7M in L1, and 65.71 / (100 − 3.56) = 68.1%, which matches L1's denominator of 258.8M. The L1 line is **66**, not 67. |

**Other small errors in the body:**
- The Q2 Takeaway says "18% new pairs" during a stall. The per-action in-stall rate is **19.7%**; 17.9% is the 15-action window before all 354 breakthroughs.
- "Reach new boards 8–13 points less often": the gaps are 7.9 / 12.5 / 12.0 / 4.8 points for the bins 0–19 / 20–49 / 50–99 / 100–199, so about 5–13 points.
- In Q1 Gaps, "each of the 1,170 uncleared attempts" is right for attempts with actions. The censored tail also includes the roughly 70 action-less final levels.
