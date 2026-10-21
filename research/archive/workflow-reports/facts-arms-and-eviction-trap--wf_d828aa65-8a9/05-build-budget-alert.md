# facts-arms-and-eviction-trap: build:budget-alert

Workflow run `wf_d828aa65-8a9`, phase Build. Revise attention-prompt from advice to harness-computed steering facts, build indicator-notes and budget-alert arms, investigate and file the memory pin/eviction trap issue; review, fix, integrate, PRs

---

The step-bar detector is built and committed on `arm/budget-alert` (9 commits on top of stack-v2 `d55df75`, not pushed). Replayed on 10 recorded games it never mistook another line for the bar, and its count of actions until the bar empties was off by 1–2 on average. It hasn't been scored on Kaggle yet. Only my three test files were run (36 pass); the full suite is left for the integrate step.

**Commits** (worktree `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/budget-alert`):
- `f775d8f` Add a step-budget bar tracker that reads recorded frames
- `9e545b1` Add the step-bar fact lines and when to show them
- `350010b` Show step-bar facts in the turn message behind ARC3_BUDGET_ALERT
- `346f766` Test that requests match stack-v2 with the new flags off
- `91aff8d` Read the step bar only after ten actions
- `76d04b4` State actions per full bar, not per attempt
- `3fae16d` Document ARC3_BUDGET_ALERT and its replay results
- `ccb1e03` Add the budget-alert and stack-v2-budget Kaggle arms
- `13b82f3` Separate the step-bar text helpers with two blank lines

**What it does**
- **Flags:** `ARC3_BUDGET_ALERT=early|low|both|off`, default off. The low thresholds are `ARC3_BUDGET_ALERT_LOW_FRACTION` (0.25), `_LOW_ACTIONS` (12) and `_REPEAT` (10 actions between low lines).
- **Placement:** at most one line per turn, right after the "Valid actions" line of the new user message. Nothing earlier in the history changes.
- **Flag off:** the tracker is never built. A test checks that the system prompt and five turn messages match a fixture built by stack-v2's own code, under both default and competition settings. With the flag on, removing the "Step bar" lines gives the fixture back.
- **Changes from Fable's version:**
  - It finds growing bars as well as shrinking ones.
  - UNDO doesn't count as an action, and if UNDO restores a cell that isn't counted as a refill.
  - RESET, the automatic reset after a death, and a level change each start a new attempt.
  - A refill inside an attempt is followed rather than ending the bar.
  - Parallel lines that move together are one thick bar.
  - It looks at lines inside the existing edge border (`ARC3_NOOP_GUARD_BORDER`). It also accepts a full-width line in a non-background colour, which is how sk48 and sb26 draw their bar on row 53. A strict edge band would have missed those.
  - The rate is measured from the first tick to the last, which is exact for a bar that ticks every k actions. It is pooled over earlier attempts at the same level.
  - It counts deaths that happened with the bar empty.
  - Any error gives no line.

**Example lines** (from the replay):
- `Step bar (read from the frames by the harness): bottom edge, row 63, 64 cells, ~1 cell per 3 actions => ~192 actions per full bar; 60 cells left => empty in ~180 actions.`
- `Step bar: 9 of 64 cells left, ~1 cell per 1.6 actions => empty in ~14 actions. An attempt in this game ended in GAME OVER as this bar ran out (2 times).`

The lines say "per full bar", not "per attempt" as in your example. In ls20 an empty bar costs one of three lives and refills, so "per attempt" would be false there.

**Validation** (one niced process of at most 5 games per run, under 1 s CPU each):

| | wa30, sc25, ka59, tr87, sk48 (fz-ledger-on-trim-r2) | same 5 games (v4-stack-v1, held out) | sb26, cn04, lp85, ls20, tu93 (r2) |
|---|---|---|---|
| Bar read in attempts of 10+ actions | 30/30 | 25/25 | 33/33 |
| Readings on a line that isn't the bar | 0/194 | 0/166 | 0/73 |
| First-reading rate vs final rate (median / max) | 1.5% / 5.9% | 0.5% / 5.9% | 4.0% / 7.7% |
| Deaths with the bar empty | 8 | 6 | 0 |
| Error in actions until empty | average 1.4 (median 1) | average 2.1 (median 1) | n/a |
| Error at the first low line | 0, 0, 0, −3, −1, −2, 0, 0 | 0, −1, −2, 0, −1 | n/a |
| Lines per game | about 6 (about 250 tokens) | about 7 | about 5 |

I raised the actions needed before the first reading from 6 to 10 after the replay. At 6, the worst first-reading rate was 19–26% off (ka59 ticks after 2, 2, then 3 actions); at 10 it was 6%. The line now first appears at a median of 10 actions into an attempt.

**Limitations**
- A bar that changes colour as it runs low, or is drawn in segments, is not read.
- When the cost per action varies (sc25 takes 2–6 cells per action), early numbers can be off by up to about 10–16 actions, mostly too low.
- A progress bar that ticks regularly could be read as a step bar. None was in the 15 games.
- Truth for "is this the real bar" and "was this a budget death" came from my own frame inspection.
- Fable's design was never scored, so whether these lines raise the score is still untested.

**Kaggle arms:** `budget-alert` = `{ARC3_BUDGET_ALERT: both}`, and `stack-v2-budget` = `{ARC3_LEDGER_ON_TRIM: 1, ARC3_BUDGET_ALERT: both}`. `build_notebook.py --arm stack-v2-budget` builds a 367 KB notebook locally.

**Not done (for the integrate step):**
- The fork README's arm table lives on `docs/fork-readme`, so it has no `budget-alert` row yet.
- `arms.md` isn't updated.
- One possible follow-up: on the turn after a death, a line stating the bar's cells in the death frame would answer the existing "was the bar depleted?" death-diagnosis text directly. I didn't build it because it's outside the two line types asked for.

Main new file: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/budget-alert/ARC3-Inference/inference/agent/budget_bar.py`

Replay scripts and outputs are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/budget/`:
- `validate.py`
- `summarize.py`
- `validate_fz-ledger-on-trim-r2.json`
- `validate_v4-stack-v1.json`
- `validate_fz-ledger-on-trim-r2_b.json`
