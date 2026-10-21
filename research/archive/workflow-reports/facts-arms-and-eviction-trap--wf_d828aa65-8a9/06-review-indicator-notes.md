# facts-arms-and-eviction-trap: review:indicator-notes

Workflow run `wf_d828aa65-8a9`, phase Review. Revise attention-prompt from advice to harness-computed steering facts, build indicator-notes and budget-alert arms, investigate and file the memory pin/eviction trap issue; review, fix, integrate, PRs

---

**Review of `arm/indicator-notes`: 5 commits on `arm/stack-v2` (d55df75). The off path is clean, but one model-visible fact is false in real replays.**

The branch's 24 tests pass (niced, imports resolve to the worktree). I re-ran `bytes_check.py` in the stack-v2 worktree and in this branch with the flag off: system prompt and both turn messages (image parts included) are byte-identical. I checked three real boards from `fz-ledger-on-trim-r2`:
- **sk48 steps 16 and 21:** the 2x2 centre of the tray icon goes R→W→R. These are true facts and a real HUD mark.
- **dc22 steps 202 and 203:** the clicked button at (28,56) toggles the end caps of the bar on rows 40-41. True as a far effect, but the LOST baseline is wrong (defect 1).
- **wa30 steps 244 and 252:** a 4x4 block's border goes W→G, later c→B. True, but it is an object state, not an indicator.

## Defects, most severe first

**1. Medium-high: the LOST lines state a false fact on real boards.**
- **Where:** `indicator_notes.py:518` sets `original=_snapshot(before, box)`, the content just before the first change the tracker detected, not the content at level start. `:550` compares against it, and `:393-396` describes it as "its colour when first seen on this level".
- **dc22, step 203 (shown in replay):** the line says rows 40-41 cols 16-17 and 30-31 "went back to light gray (w), its colour when first seen on this level". At the start of level 4 those cells were orange (`OOOO`); they were `wwww` only just before step 202.
- **wa30, step 259 (shown):** the line says "went back to white (W)…". At the start of level 3 rows 20-23 cols 24-27 were light-gray background with no block.
- **Scale:** a level-start check over the replay found 18 of the 30 "first" marks whose baseline is not the level-start content.
- **It can also invert the meaning** (`scratchpad/indicator-review/probe2.py`). The mark starts red, is earned (R→W) while the player is within 8 cells (not detected), is lost far away (W→R, reported as the first change), and is re-earned (R→W). The re-earn is then reported as "Indicator LOST: … went back to white (W), its colour when first seen on this level".
- **Fix:**
  - Keep the level-start grid: `cf.grid` in `_new_level`, or `entries[0].frame.grid` for the first level. A RESET returns the board to this state.
  - Set `_Mark.original` from that grid.
  - Seed `mark.states` with `(discovery_step - 1, _snapshot(before, box))`. A return to the pre-discovery content can then be worded "back to its content before step N", which is true by construction.
  - Add a test with a dc22-shaped case, where the level-start content differs from the content at discovery, and with the probe2 sequence.

**2. Medium: the labels "Indicator" and "LOST"/"AGAIN" claim a meaning the harness cannot check.**
- **Where:** `indicator_notes.py:393-413` and `:590`.
- **What the replay shows:** the strings contain no imperatives. But by the implementer's own replay only sk48's hits are HUD indicators. dc22's bar end caps (toggled by a button) and wa30's block borders reach the model as "Indicator LOST" and "Indicator AGAIN".
- **Same for UNDO:** an UNDO revert is shown as "Indicator LOST … (UNDO)" (`probe.py`).
- **Why it matters:** this nudges the model with a judgement ("an earned mark was lost") that is false on most hits.
- **Fix:** use a label that is true for every hit, which still points the model at the change:
  - "Far change: after step 21 (LEFT), rows 58-59 cols 28-29 changed white->red (W->R), back to its level-start content; it had changed to white after step 16."
  - "… same change as after step 16."
  - Write "the largest change in that step was at …" instead of "the main change". See also defect 4.
  - The docs at `CONFIGURATION.md:64` use the same words and need the same change.

**3. Low-medium: HUD counters are never reported, although the docs say they are.**
- **Where:** `_is_move` (`indicator_notes.py:244-257`) treats a glyph change as a move. A digit going 1→2 or 2→3 changes some cells W→B and others B→W in similar numbers.
- **Evidence:** both digit changes, 5 cells away from nothing and far from the player, gave no line (`scratchpad/indicator-review/probe.py`).
- **Docs to correct:** the docstring at `indicator_notes.py:5` and `CONFIGURATION.md:64` ("tray mark, legend icon or counter").
- **Related:** `_bar_like` (`:211-214`) counts the board edge as a run stop. A 2x2 HUD pip flush with col 63, on plain background, is classed as a timer-bar tick and dropped (probe "edge pip"). So fix 9 is only partial.
- **Fix:**
  - Replace the colour-count test with a translation test: does `before` shifted by some (dr, dc) in [-2..2] reproduce `after` on the patch?
  - In `_bar_like`, ignore runs of the board's most common (background) colour.
  - If neither is done, at least drop "counter" from the docs.

**4. Low: the "main change" and the reported patch can swap roles.**
- **Where:** `analyze_transition`, `:321-322`, picks the main change by size alone.
- **Evidence:** when the action's own change is no bigger than the mark, the mark becomes "the main change" and the other patch is reported. A 2-cell change was reported with "the main change was at rows 58-59 cols 27-28" (probe "mark bigger than main").
- **sk48 step 16** worked only because of a tie-break: 4 cells in 2 pieces against 4 cells in 1 piece.
- **Fix:** the "largest change" wording from defect 2 keeps the line truthful. Optionally prefer the cluster that contains the player or the click as the main change.

**5. Low: the committed byte-identity test is weaker than the scratch check.**
- **Where:** `tests/test_indicator_notes.py:286-303`.
- **Gap:**
  - The test compares flag-off with flag-on inside the branch and covers only `_build_user_prompt` text, not `_build_user_message` (image parts).
  - It is not pinned to stack-v2's output.
- **Why it is low:** the off path is a single guarded `if` (`tool_agent.py:5203-5204`), and the rollback hunk (`:7317-7320`) does nothing when no tracker exists. I confirmed identity independently.
- **Fix (optional):** add a `_build_user_message` comparison, or a golden file built from stack-v2.

**6. Nit: a rollback can show events that the closed gate dropped.**
- **Where:** `take_lines` (`:576-579`) assigns `self._taken = pending` before the gate check returns `[]`.
- **Effect:** `rollback()` then queues those never-shown events again. This goes against the documented "changes found before that are dropped". It hardly matters in practice, because a failed request restores `_session_generated_tokens`.
- **Fix:** set `self._taken = []` on the early-return path.

**7. Nit: setting `ARC3_INDICATOR_MAX_LINES=0` silently gives 1 line.**
- **Where:** `max(1, …)` at `:580`.
- **Effect:** a turn can also carry `max_lines` plus 1 lines, because of the overflow line. Neither point is in `CONFIGURATION.md`.
- **Fix:** document both, or treat 0 as off.

## Checks that passed
- **Placement and cache safety:** lines go only into the new opener, after the valid-actions line, and nothing in older history is rewritten. Resumed short or state-only openers carry no lost events, because no actions run between a yield and its resume.
- **Rollback:** it is wired in the `preserve_history=False` branch of the request-loop `finally`.
- **Errors:** an exception in `_indicator_note_lines` returns `[]`.
- **Token gate:** it uses `_tokens_at_level_start`, which is updated after each action in `_update_summarized_knowledge_from_step_summary`, before the next opener is built.
- **Batch actions:** each action is its own history entry.
- **RESET, death and level-up:** RESET (including the auto-reset), game over and level-up transitions are skipped. The tracker is per game (`ToolAgent` per game, session check).
- **CPU cost:** from the replay, about 0.05–0.1 ms per action.
- **Docs and arms:** the `CONFIGURATION.md` section and source-map row are present. The `indicator-notes` and `stack-v2-indicator` arms are in `kaggle/arms.json`. The Apache-2.0 attribution to leoprovorov is in the module docstring.

Probe scripts are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/indicator-review/`:
- `peek_steps.py`
- `probe.py`
- `probe2.py`
- `levelstart.py`
