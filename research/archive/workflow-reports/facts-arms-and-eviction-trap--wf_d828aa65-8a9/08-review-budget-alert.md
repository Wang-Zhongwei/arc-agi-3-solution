# facts-arms-and-eviction-trap: review:budget-alert

Workflow run `wf_d828aa65-8a9`, phase Review. Revise attention-prompt from advice to harness-computed steering facts, build indicator-notes and budget-alert arms, investigate and file the memory pin/eviction trap issue; review, fix, integrate, PRs

---

**Review of arm/budget-alert (9 commits on d55df75; no edits made)**

I found three defects. The two that matter are in death counting and refill handling, and both show up on real recorded games. The rest of the branch holds up.

**What I checked, and it holds**
- **Targeted tests:** I ran the three test files niced and imports resolved to the worktree. All 36 pass.
- **Flags-off identity:** the test exists and is meaningful. I regenerated `tests/fixtures/stack_v2_prompts.json` by running `tests/prompt_scenarios.py` against the stack-v2 worktree (PYTHONPATH=. at d55df75), and the output matches the committed fixture exactly. The scenario covers the system prompt and five turns: start, mid-game, the first readable turn, game over plus auto-RESET, and a level change. It runs under both the default and competition environments. The flag-on test shows that only lines starting with "Step bar" are added.
- **Cache safety:**
  - The line is appended only inside `_build_user_prompt`, after the valid-actions and death-ledger lines (tool_agent.py:5211-5215).
  - There is one call site per turn (tool_agent.py:6722).
  - The system prompt is unchanged, and `memory_ledger` never reads user prompts.
  - There is one ToolAgent per game (solver.py:1826-1834), and the tracker is rebuilt when the session changes (tool_agent.py:4893).
- **Board semantics:**
  - Each action gets its own HistoryEntry (solver.py:1339), including every action in a batch and the auto-reset after a death, which is recorded as "RESET".
  - UNDO is displayed as "UNDO" under EXPOSE_UNDO=on, and the code also accepts "ACTION7".
  - A level change is read from `frame.level`.
  - `result["game_over"]` is the key the solver writes.
  - The history is never capped, so `_seen` works.
  - Rows and columns are 0-based, the same as `.ascii` and MOUSE(row, col).
- **Real boards I checked:**
  - tn36: row 1, cells 1–61 in colour 9, losing one cell per action from the right. The reported "top edge, row 1, 61 cells, ~1 cell per action" is correct.
  - ls20: rows 61–62, cols 13–54, losing 2 cells per action from the left. The ticks are correct (but see defect 1).
  - su15 level 4: row 63, losing 1/1/2 cells per action. A rate of 1.30 is correct (but see defect 2).
- **Held-out replay:** I ran 10 games the implementer did not use, from the fz-ledger-on-trim-r2 run (su15, g50t, vc33, bp35, tn36, m0r0, sp80, ar25, lf52, cd82). Readings on a line that isn't the bar: 0 of 241. Every attempt of 10 or more actions was read at frame level. CPU was 0.3–0.4 s per 5 games.
- **Facts, not advice:** both model-visible strings (`early_line` and `low_line`, budget_bar.py:400-416) contain only numbers. They have no imperatives and no "you".
- **Caps and cost:**
  - At most one line per turn.
  - The early line shows once per level.
  - Low lines repeat at most every 10 actions on the same attempt.
  - Reading is incremental, at about 0.01–0.02 s CPU per game.
- **Docs and arms:** CONFIGURATION.md section 2 has the new subsection. `kaggle/arms.json` has `budget-alert` and `stack-v2-budget`. Fable is credited in the module docstring. Every commit carries both trailer lines.

**Defects, most severe first**

1. **Medium: a bar that refills when a life is lost (ls20) is dropped after its third refill, so the last life gets no reading and the final GAME OVER is not counted.**
   - **Where:** budget_bar.py:63 (`MAX_REFILLS = 2`) and :150-153.
   - **Real case:** ls20 in fz-ledger-on-trim-r2, level 2, attempt starting at entry 40. The bar refills at entries 62 (bar empty, life lost), 74 (from 20 cells left) and 97 (bar empty, life lost). At entry 97, which is action 57 of the attempt, rows 61 and 62 are killed with `refills=3`. The bar empties again at entry 118, and GAME OVER follows at 119. `reading()` returns None from 97 onwards, and `bar_deaths` stays empty. In this particular run the last life fell inside one turn (turn 19), so no line was lost here. On any run where the last life spans several turns, it would be.
   - **Validation numbers are wrong too:** the "Deaths with the bar empty … 0" entry for the sb26/cn04/lp85/ls20/tu93 column (CONFIGURATION.md:93) is incorrect. validate.py:152 counts a budget death only when at least 95% of the whole 64-cell line changed. A 42-cell bar can never reach that; this one scored 0.75. The ls20 death was a death with the bar empty. "33/33 read" also hides that readings stop after the third refill.
   - **Fix:**
     - Don't count a refill toward `MAX_REFILLS` when the bar was empty just before it (`self.used >= self.cells - self.max_step`, the life-loss cycle). Or count only refills that do not restore the bar to full.
     - Add a test: `Sim(thick_bar, cells=2)` with one pickup `refill()` and two empty-then-full cycles, then GAME OVER. Assert a reading exists in the third life and `bar_deaths == 1`.
     - Change the validator to measure the bar's own `lo..hi` span instead of the full line, and correct the evidence row in the docs.

2. **Low–medium: bar deaths are undercounted when the bar takes more than one cell per action at a non-integer rate.**
   - **Where:** budget_bar.py:271, `before.left <= max(1, int(round(before.rate)))`.
   - **Real case:** su15 in fz-ledger-on-trim-r2, level 4, entry 139. Before the fatal action the reading was left=2 at rate 1.302, so the threshold is 1. The fatal action took 2 cells, and the death frame shows `used=64`, an empty bar. The death was not counted, so the "ended in GAME OVER as this bar ran out" note is missing from later low lines. A synthetic 1/1/2-cell run reproduces it.
   - **The same rule also over-counts:** at rate 1/3 the threshold is 1 cell, which covers up to 3 actions. A hazard death with 1 cell left therefore gets labelled a bar run-out.
   - **Fix:** decide from the death frame itself. After the head candidate observes the GAME OVER frame, count a bar death if it is still alive and `used >= cells`. If that frame killed the candidate, fall back to `before.left <= max(cells for _, cells in head.ticks)`. Add the su15-like test. This also makes the note's wording ("ran out") literally true.

3. **Low: an exception in the middle of an update leaves the tracker corrupted instead of doing nothing.**
   - **Where:** budget_bar.py:241-243. `_seen` is assigned only after the loop finishes. If `_feed` raises on entry k, entries before k have already been fed, and the next turn feeds them again. That double-counts `attempt.actions`, or starts a new attempt if one of them is a RESET. The except branch at tool_agent.py:4909-4911 suppresses only that one turn's line, and later lines can carry wrong numbers.
   - The trigger is unlikely: it would take a ragged grid causing an IndexError in `_line`.
   - **Fix:** in the except branch, turn off alerts for the session (for example, record `_budget_failed_session = self._session_runtime_dir` and return [] while it matches). Alternatively, advance `_seen` per entry and skip the bad one.

**Nits (optional)**
- tool_agent.py:4894-4895 replaces `ARC3_NOOP_GUARD_BORDER=0` with 4, while the solver uses 0 (solver.py:119). The docstring at tool_agent.py:4888 and CONFIGURATION.md:77 say the border is shared. Either say "4 when the guard border is 0" or follow the solver.
- `low_line`/`early_line` can print "empty in ~0 actions" when left/rate rounds to 0, for example 1 cell left at 3 cells per action. "~1 action" would be better as a floor.
- The "eviction item" / handover question does not apply to this branch, which has no eviction or handover changes.

Replay copies and debug scripts are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/review-budget/`. The two held-out replay outputs there are `validate_fz-ledger-on-trim-r2.json` and `validate_fz-ledger-on-trim-r2_b.json`. The scripts are `dbg_death.py`, `dbg_ls20.py`, `dbg_trace.py` and `repro.py`, and the regenerated fixture is at `.../scratchpad/review_budget_regen.json`.
