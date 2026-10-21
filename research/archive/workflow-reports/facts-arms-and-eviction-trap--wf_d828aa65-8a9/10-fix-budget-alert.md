# facts-arms-and-eviction-trap: fix:budget-alert

Workflow run `wf_d828aa65-8a9`, phase Fix. Revise attention-prompt from advice to harness-computed steering facts, build indicator-notes and budget-alert arms, investigate and file the memory pin/eviction trap issue; review, fix, integrate, PRs

---

Every review finding was real: I fixed the three defects and both nits in 6 new commits on `arm/budget-alert`, which is not pushed. The three targeted test files pass (40 tests).

**Per finding**
1. **Fixed (medium): ls20 bar dropped after its third refill** (`17e4ec0`). On ls20 level 2 the bar was dropped at action 57 with 3 refills, gave no reading from entry 97 on, and the final GAME OVER was not counted. A bar that goes from empty (within one action's cells) back to full no longer counts towards `MAX_REFILLS`. Any other refill still counts, so a sprite leaving an edge line is still dropped.
   - New test `test_an_empty_bar_that_comes_back_full_is_not_a_pickup`: one pickup, two lives lost, then GAME OVER. It fails without the fix.
   - After the fix, the ls20 replay has readings up to the end and counts the death, `{row61: 1}`.
   - The validator now judges a death by the bar's own cells (`lo..hi`), not the whole 64-cell line. It also leaves out readings taken before a refill in the same attempt. sc25 has pickups (50→40 and 26→18 cells), so its earlier "-10" errors were readings made before a pickup. The CONFIGURATION.md evidence row is corrected (`c39675a`).
2. **Fixed (low–medium): bar deaths undercounted when the rate isn't a whole number** (`4594b26`). su15 had 2 cells left at 1.302 cells per action and the death was not counted. Of the review's two options I used the death frame:
   - **Bar visible:** the GAME OVER frame decides, and it counts only if the bar is empty there.
   - **Bar hidden by a game-over screen, or shown full again:** the frame before decides. At most one tick's cells may be left, and on a bar slower than one cell per action the next tick must have been due.
   - **Over-count case:** a hazard death at 1 cell per 3 actions with 1 cell left is no longer counted.
   - **Tests:** `test_death_frame_decides_a_bar_death` and `test_bar_death_behind_a_game_over_screen` (5 cases) both fail without the fix.
   - **Replay (25 recorded runs of 20 games, `deathcheck.py`):** every GAME OVER frame still showed the bar, so the fallback never ran. The only change is that the su15 death is now counted.
3. **Fixed (low): tracker left corrupted after an error part-way through an update** (`265c284`). I confirmed it: after a one-off error, the next update showed 22 actions at 0.95 cells per action where the truth was 15 actions at 1.0. The tracker now marks itself `failed`, and both `update` and `reading()` do nothing for the rest of the game. Test: `test_a_failed_update_reads_nothing_more`.
4. **Nit fixed: "empty in ~0 actions"** (`ba40656`). `actions_left` is now at least 1 while any cell is left, and 0 only when the bar is empty.
5. **Nit fixed: border wording** (`72e3532`). The tracker keeps using the outer 4 lines when `ARC3_NOOP_GUARD_BORDER=0`, because the bar sits on edge lines. The two docstrings and CONFIGURATION.md now say that, instead of claiming the border is shared with the solver.
6. **Eviction item: not applicable.** The review confirmed this branch has no eviction or handover change, so I posted no GitHub issue comment.

**git log arm/stack-v2..HEAD**
```
c39675a Correct the step-bar replay evidence after the death fixes
72e3532 Say the step bar uses the outer 4 lines when the guard border is 0
ba40656 Never state "empty in ~0 actions" while cells are left
265c284 Stop reading the step bar after an update fails part-way
4594b26 Judge a bar death from the GAME OVER frame
17e4ec0 Keep following a bar that empties and comes back full
13b82f3 Separate the step-bar text helpers with two blank lines
ccb1e03 Add the budget-alert and stack-v2-budget Kaggle arms
3fae16d Document ARC3_BUDGET_ALERT and its replay results
76d04b4 State actions per full bar, not per attempt
91aff8d Read the step bar only after ten actions
346f766 Test that requests match stack-v2 with the new flags off
350010b Show step-bar facts in the turn message behind ARC3_BUDGET_ALERT
9e545b1 Add the step-bar fact lines and when to show them
f775d8f Add a step-budget bar tracker that reads recorded frames
```

**Targeted pytest** (niced; imports resolve to the worktree): `tests/test_budget_bar.py tests/test_budget_alert.py tests/test_prompt_identity.py` → `40 passed in 0.27s`

**PR summary**
- **Hypothesis:** many games end in GAME OVER when the step bar runs out, and the model notices the bar late or misreads it. Stating the bar's numbers, computed by the harness from recorded frames, should cut those deaths. These are facts, not advice: no imperatives and no "you". The design is adapted from leoprovorov's Fable layer (Apache 2.0), which was never scored.
- **Change:** a new `inference/agent/budget_bar.py` tracker reads only new history entries and needs no model call. Behind `ARC3_BUDGET_ALERT=early|low|both`, default off, it adds at most one "Step bar" line per turn after the valid-actions line of the new user message, so nothing earlier in the cached history changes. It finds shrinking and growing bars, treats parallel lines as one thick bar, and doesn't count UNDO as an action. It follows pickups and lives lost, judges a bar death from the GAME OVER frame, and any error means no line.
- **Tests:**
  - 40 targeted tests.
  - With the flag off, the system prompt and five turn messages match a fixture built by `arm/stack-v2`'s own code, under both the default and competition settings.
  - With the flag on, the only difference is the added "Step bar" lines.
- **Replay** of 30 recorded game runs, at most 5 per niced process, about 0.5 s CPU each:
  - The bar was read in 140 of 140 attempts of 10+ actions.
  - 0 of 677 readings were on a line that isn't the bar.
  - The rate at the first reading was within 7.7% of the attempt's final rate.
  - All 17 bar deaths were counted, and none of the 11 other deaths was.
  - The error in predicted actions until that death averaged 0.55–1.35 actions.
  - That comes to about 5–7 lines (180–240 tokens) per game.
- **Arms:**
  - `budget-alert` = `{ARC3_BUDGET_ALERT: both}`.
  - `stack-v2-budget` adds `ARC3_LEDGER_ON_TRIM=1`.
  - Neither has been scored on Kaggle, so the effect on score is untested.
- **Not done:**
  - The fork README's arm table, which lives on `docs/fork-readme`.
  - `arms.md`.

The replay validator and the death check now live in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/budget/`:
- `validate.py` (judges deaths on the bar's own cells and leaves out readings before a refill)
- `deathcheck.py`
- `validate_*_fix.json` for all five game sets

Main file: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/budget-alert/ARC3-Inference/inference/agent/budget_bar.py`
