# mem-repair-10x: 10 games won, and whether the code is game- and time-agnostic (2026-10-14)

**What the user asked:**
- "if we can solve 9 or 10 games in that same time period, it would help." This led to the build of `arm/mem-repair`.
- After the run: "it finished (now with 10 games actually solved.) make sure no hard coding to these games is
  done and it is actual game/time agnostic."

**Answer:**
- The run won **10 of 25 games** with a mean score of **54.69**. That ties the most wins of any run, and it
  used fewer tokens than stall-fix or mem-unified.
- No code is hard-wired to a public game or to the dev run's clock.
- Some thresholds were calibrated on the public games. Two lines misled the model at the margin on su15.
- One inherited deadline setting is a risk for the real submission.

It is one run, so it is not yet an alpha.

**Leaderboard (added 2026-10-15):** submitted as-is, it scored **38.22** on the public leaderboard (submission
57200258), our best so far (previous 34.04). The first attempt (57198575) failed after ~37 min with a generic "Kaggle
Error" (a system error, not a notebook exception) and the identical resubmission completed.

## 1. Result

`mellamoa/v4-mem-repair-10x` v1, 18:36–20:38 UTC. Run record: `runs/kaggle/v4-mem-repair-10x/v1`.
Notebook: `kaggle/notebooks/v4-mem-repair-10x-v1.ipynb`.

**Wins and time to win:**

| Game | Minutes to win |
|---|---|
| ft09 | 46 |
| tr87 | 49 |
| ar25 | 55 |
| sb26 | 77 |
| cd82 | 83 |
| vc33 | 94 |
| sc25 | 96 |
| lp85 | 104 |
| r11l | 105 |
| cn04 | 110 |

- Three wins came by 60 min and five by 90 min.
- cn04 had been won only once in the 13 earlier runs, and vc33 and sc25 six times.

**Comparison with earlier runs** (mean score at matched wall-clock minutes):

| Run | 30 | 60 | 90 | 120 | Final | Wins | Tokens |
|---|---|---|---|---|---|---|---|
| **mem-repair-10x** | 9.3 | 32.0 | 46.6 | 54.7 | 54.69 | **10** | **5.03M** |
| stall-fix-10x | 10.8 | 27.3 | 43.4 | 51.8 | 51.85 | 8 | 5.30M |
| mem-unified-10x | 10.5 | 30.6 | 42.1 | 55.1 | 56.09 | 8 | 5.20M |
| pf-keep-10x | 10.7 | 28.2 | 47.4 | 55.3 | 55.79 | 8 | 4.96M |
| user-combo-10x v1 | 13.9 | 41.3 | 50.6 | 56.9 | 56.94 | 10 | 5.10M |

**Paired per-game tests** (`make significance`):

| vs | Difference | P(better) |
|---|---|---|
| stall-fix | +2.85 | 0.72 |
| mem-unified-core | +3.54 | 0.73 |
| mem-unified | −1.40 | 0.38 |
| pf-keep | −1.10 | 0.40 |
| user-combo v1 | −2.25 | 0.33 |

The 90% interval against stall-fix is [−5.3, +11.0].

**Stall load:**
- 28 stalls (9 open), down from 31 in stall-fix.
- 2.65M stall tokens, down from 3.01M.
- 1,571 stall-minutes, down from 1,653.

**Levels the repairs targeted** (minutes on the level):

| Level | stall-fix | mem-repair | Note |
|---|---|---|---|
| sc25 L6 | 63, open | 16 | The new "Hidden since … / Appeared at …" line carried the teleport-marker fact across the marker's disappearance, which is the case it was built for |
| vc33 L4 | 87 | 41 | |
| tr87 L2 | 82 | 3 | |
| cn04 | L2 35, L4 40 | L2 29, L4 4 | won |
| tn36 L2 | 118 | 98 | reached L4 |
| m0r0 L3 | 46 | 86, open | |
| su15 L2 | 27 | 118, open | worse; see §4 |

**New lines that fired** (count in new messages; content checked by hand):

| Line | Count |
|---|---|
| Not touched yet | 1,017 |
| Rules tagged with their action | 1,900 |
| Retained functions unchanged | 1,177 |
| cleared_boards | 199 |
| Also clicked | 85 |
| Hit background | 17 |
| Hidden since | 21 |

## 2. Is it game-agnostic?

A read-only audit of every file the run executes (`arm/mem-repair` with its flags) found **no game-specific
code**:
- Nothing branches on a game id, title, grid size, palette index, board hash, level count or action set.
- No colour is special-cased. The background is always the most common colour.
- Public game ids appear only in comments and docstrings, and in Franzen's local game list
  (`framework/kaggle.py`). That list chooses which games a local CLI run plays, and the notebook does not use it.
- The arm's flags are global; nothing is set per game.
- Nothing reads public-game engines, routes, recordings or test fixtures at run time. The Kaggle patch ships
  only `ARC3-Inference/inference` and `configs`.

What remains is **heuristics calibrated on the 25 public games**. Ranked by risk on unseen games:

1. **Budget-bar verdict** (`death_facts.py`). The thresholds are fitted to public bars; with no bar found, it
   stays silent. The risk is in the wording. It says "not a budget death" whenever the bar's colour is still
   present in the death frame. A game whose empty bar keeps an outline would get a false verdict.
2. **"Not touched yet" row** (`object_facts.py`). It skips objects under 4 cells or touching the edge, but its
   header doesn't say so. It also replaced the colours row, which covered every colour. su15's 1-cell pieces
   were missing from it (§4).
3. **Marker frames** (`board_facts.py`, sides 4–8). These were fitted to sc25's teleport marker. On other games
   they simply don't fire.
4. **Rules filter word lists** (`confirmed_rules.py`). These are generic hedge, plan and echo English, plus a
   few verbs lifted from public rules (teleport, absorb, frozen, cycle). This only affects which of the model's
   own sentences are quoted; it never writes new text.
5. **Smaller thresholds:**
   - The cleared board is taken as the second-last animation frame. That assumes the next level arrives in one
     frame.
   - Move slack is 2 cells.
   - Click-run and edge bands are 4 cells.
   - Click-miss reach is 3 cells.
   - Prompt text adds no game nouns or mechanics.

## 3. Is it time-agnostic?

- The harness doesn't depend on time or game count.
- **No minutes and no game counts reach the model.** `ARC3_BUDGET_TRUTH` strips the run clock from every
  action result. The prompt describes the bar as an action budget, not a clock.
- Scheduler constants are in tokens, actions or levels.
- The tail fade is a share (0.2) of each run's own deadline: 24 min in the dev run, 106 min in the private run.
- The submission path (`TRUE_SUBMISSION`) changes only these settings:
  - concurrency 120;
  - the 532-min deadline;
  - no soft end and no request logs;
  - the live gateway.
  No harness code reads it, so every flag behaves the same in both paths.
- The 121-min dev run gives each game the same compute as the private run: 14 × 121 / 25 = 67.8
  stream-minutes per game, against 14 × 532 / 110 = 67.7.

**Risks:**
- **Deadline (inherited from Franzen, not new).** `max_runtime_s_per_game = 532*60` counts from benchmark
  start, which comes about 10 min after the notebook starts.
  - This run's nominal end was 32,513 s, past the 32,400 s (9 h) the notebook asserts.
  - Earlier submissions completed with the same setting, including today's 34.04, so Kaggle has tolerated
    it. Nothing guarantees that.
  - Fix: derive the deadline from the notebook's start time minus a margin.
- **Scheduler ratio.** Dev runs 25 games on 14 slots; the private run has 110 on 14. The priority gate
  chooses among about 97 waiting games instead of 12, so its constants are untested at that ratio. They
  degrade smoothly; they don't break.

## 4. The su15 regression (L2 open for 118 min; never stalled in 13 earlier runs)

Mechanics:
- A click pulls every piece within about 9 cells to the click point.
- Same-size pieces merge up a size ladder (1×1 → 2×2 → 3×3 → 4×4).
- When pieces of different sizes collide, the board blinks and the move is rejected.

What went wrong:
- The model built a false "knight-like jumper with r+c parity" model. No earlier run had this model; it
  appears 56 times ("leaper") and 115 times ("parity") in this run.
- Of the 118 minutes, about 65 were spent parked after trims and 53 generating.

Two new lines misled it at the margin:
- **"Hit background … it did not touch it"** fired on four failed clicks.
  - Each click failed because a 1-cell dot of a different size was pulled in, not because the click missed.
    The line can't name a 1-cell object.
  - At request 15 the model read it as "too far / no valid move". That explained away the click that should
    have broken the jump model.
- **"Rules you confirmed"** quoted the model's own wrong deductions from request 35 to 49: "merging two pieces
  requires same parity" and "Confirmed: the goal is not 'fill a 4x4'". The model cited the first at
  request 48.

The verdict is mainly the model's own exploration plus scheduling. These lines added to it; they were not the
single cause.

## 5. Fixes for the next arm (not built yet; each goes behind its own flag)

- **Hit-background line:**
  - Emit it only when the click's whole animation moved nothing.
  - Have it name nearby objects of any size, 1-cell ones included.
- **Rules filter:** don't quote negated goal claims ("the goal is not …") or deductions ("so …") that no
  observed change backs.
- **Budget bar:** state only the counts ("k of n cells left"). Drop the "not a budget death" verdict.
- **Contact row:**
  - Say "objects of 4+ cells" in its header.
  - Keep a short colours row for anything smaller.
- **Submission deadline:** count it from notebook start minus a margin, instead of 532 min from benchmark start.

Sources:
- Run outputs: `kaggle-outputs/v4-mem-repair-10x/v1`, outside the repo.
- Audits: three read-only agent reads (game-specific code, time and run-setup, su15 L2), whose findings are
  summarised above.
