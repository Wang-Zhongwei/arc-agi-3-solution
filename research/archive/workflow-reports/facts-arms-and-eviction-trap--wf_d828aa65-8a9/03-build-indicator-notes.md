# facts-arms-and-eviction-trap: build:indicator-notes

Workflow run `wf_d828aa65-8a9`, phase Build. Revise attention-prompt from advice to harness-computed steering facts, build indicator-notes and budget-alert arms, investigate and file the memory pin/eviction trap issue; review, fix, integrate, PRs

---

I built the indicator-notes arm on branch `arm/indicator-notes`, off `arm/stack-v2`, in 5 commits. My 24 tests pass; I did not run the full suite. With the flag off, the requests are byte-identical to `arm/stack-v2`. In replay, sk48's real tray mark is reported; most notes on other games are true facts about objects changing away from the player, not HUD indicators. Nothing is pushed and no PR is open.

**Commits**
- `03a5b0a` Add an indicator detector ported from the Fable layer (`ARC3-Inference/inference/agent/indicator_notes.py`)
- `848430f` Test the indicator detector on synthetic frames
- `f032b9c` Show indicator notes in the turn message behind ARC3_INDICATOR_NOTES (`tool_agent.py`, +45 lines, plus tests that go through ToolAgent)
- `9460aab` Document the indicator-notes flags (`CONFIGURATION.md`, section 2 "Indicator notes", and the source map)
- `59cfc12` Add the indicator-notes and stack-v2-indicator Kaggle arms (`kaggle/arms.json`). A build-only run of `stack-v2-indicator` succeeded, and the notebook patch includes the new module.

**How it works**
- **Flags:** `ARC3_INDICATOR_NOTES` turns it on. `ARC3_INDICATOR_AFTER_TOKENS` (default 15000, as in Fable; 0 = from the first turn, negative = never) gates it on generated tokens spent on the current level. `ARC3_INDICATOR_MAX_LINES` (default 3) caps lines per turn; any extra changes get one line with their count.
- **Token gate:** it uses the harness's own per-level counter, `_session_generated_tokens - _tokens_at_level_start`.
- **Input:** each executed action's before/after frames from the recorded history, including every action inside a batch. It only processes entries added since the last turn.
- **Placement:** lines go right after the `Valid actions right now:` line of the new turn message only. The system prompt is unchanged and no earlier history is rewritten.
- **Errors and dropped turns:** any error shows nothing. If a turn is dropped from the history (e.g. a failed request), its lines carry over to the next turn.
- **Byte-identity:** a script built the system prompt and two full turn messages (image parts included) under the template's flags with `arm/stack-v2`'s code and with this branch, flag off; the outputs match byte for byte. With the flag on, the only difference is the inserted line. The tests check the same off/on comparison.

**Example lines the model would see** (replayed from run `fz-ledger-on-trim-r2`)
```
Indicator LOST: after step 21 (LEFT), rows 58-59 cols 28-29 went back to red (R), its colour when first seen on this level; it had changed to white (W) after step 16.
Indicator AGAIN: after step 27 (RIGHT), rows 58-59 cols 28-29 changed red->white (R->W), as after step 16.
Indicator: after step 202 (MOUSE(row=28, col=56)), rows 40-41 cols 16-17 and rows 40-41 cols 30-31 changed light gray->black (w->B); the main change was at rows 20-21 cols 8-23.
```
There is no "you should" or "compare what you did" wording, which Fable's lines had. Step numbers match `Current state: step N`.

**Validation** (5 games of `fz-ledger-on-trim-r2`, one niced process, about 0.6 s CPU)

| Game | Actions | Turns | Changes found | Lines shown | Turns with a note |
|---|---|---|---|---|---|
| sk48 | 102 | 46 | 6 | 5 | 8.7% |
| dc22 | 245 | 77 | 16 | 8 | 9.1% |
| wa30 | 575 | 85 | 19 | 16 | 11.8% |
| re86 | 372 | 52 | 5 | 3 | 1.9% |
| sb26 | 153 | 37 | 0 | 0 | 0% |
| **Total** | **1447** | **297** | **46** | **32** | **7.4%** |

That is 3.2 changes found per 100 actions and 2.2 lines shown per 100 actions. A turn that has notes carries about 210 characters of them.
- **Busy boards:** sb26 is fully silent. wa30's levels 2 and 3 went silent under the more-than-6-places rule. None of these games had a frame with more than 60 changed pieces, so that rule is only covered by unit tests.
- **Fable's own detector, same games:** 4 / 13 / 17 / 2 / 0 lines (39 in total), against 32 here.

**5 example notes, checked against crops of the board**
1. sk48 step 21 LOST, rows 58-59 cols 28-29. **Real indicator:** the centre of the red icon in the bottom tray goes WW→RR, the game's only progress mark.
2. sk48 step 27 AGAIN at the same spot. **Real.**
3. sk48 step 84, rows 14-15 cols 35-40, charcoal→light gray. **Not a HUD indicator:** a dashed chain link appearing on the board.
4. dc22 step 202/203, two 2x2 bridge segments flip light gray↔black on each click at (28,56). **A real far effect** of a button, not a HUD indicator.
5. wa30 step 150, a block's border goes white→dark gray away from the player. **A real object-state change**, not a HUD indicator. re86 step 206 (gray→red where a red line crosses) is **ambiguous**.

**Fixes to Fable's code** (each is listed in the module docstring)
1. **Moved objects:** a patch counts as a move when a colour both leaves and enters it, or when the reverse colour change happens elsewhere in the same frame. Fable reported a one-piece move, dropped two marks lighting up together, and reported the far ends of re86's moving lines.
2. **Mark identity:** a mark is any later patch that overlaps it, and its state is the exact cell content. Fable matched exact boxes and compared colour-letter strings.
3. **Watching known marks:** every known mark is checked on every action, so a mark that goes out on its own or in a busy frame is still reported LOST. A change only counts if nothing else changed within 8 cells and nothing moved inside the mark (dc22's player walked onto one).
4. **Clicks:** the patch a MOUSE action clicked is ignored. Fable reported dc22's clicked tiles.
5. **Flicker:** a mark that changes on 4 of its last 5 actions, or on half of 8 or more, is muted for the level.
6. **RESET:** marks are kept across a reset. The reset itself is never reported as LOST, and re-earning the mark reads as AGAIN.
7. **Death and level-up:** those transitions are skipped.
8. **Wording:** facts only, no advice.
9. **Timer bars:** Fable's bar test dropped any patch near the edge that appeared on plain background, such as a HUD counter. A bar tick now has to sit at one end of its run.

**Limitations**
- **sk48 is the only game where the reported patch is a HUD indicator.** Elsewhere the lines are true far-away state changes, and dc22/wa30 produce several per turn. Whether they help or distract needs the Kaggle runs. rellik13 found extra prompt text hurt, though that was at 37k context.
- **The gate drops early changes:** a change found before the token gate opens is never shown. A later LOST line does point back to it ("it had changed to white after step 16").
- **A change on its own is missed:** an action whose only change is a new mark is not reported, because "far from" needs a main change. Once a mark is known, it is watched on every action.
- **One run only:** the thresholds come from Fable plus checks on these 5 games of one run. Token gating in the replay uses sim-v2's per-step token calibration, which only `fz-ledger-on-trim-r2` has.
- **Rollback path untested end to end:** the carry-over of lines from a dropped turn is tested on the tracker, not through the request loop.

**Left for the integrate step**
- Run the full test suite.
- Add a row to the fork README on `docs/fork-readme`: `indicator-notes` | "Fact lines when a small patch far from the action's change flips (LOST/AGAIN); `stack-v2-indicator` stacks it on stack-v2" | "Replay: 7.4% of turns, sk48 tray caught" | PR #?
- Move this arm in `arms.md` from Not yet implemented to "built, to run".
- Push and open the PR.

The user's other asks in this request (budget alerts, the memory-v2 issue) were not in my task.

Files are in /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad:
- `wt/indicator-notes/ARC3-Inference/inference/agent/indicator_notes.py`
- `wt/indicator-notes/ARC3-Inference/tests/test_indicator_notes.py`
- `indicator-val/replay_indicator.py`
- `indicator-val/replay_fz-ledger-on-trim-r2_15000.json`
- `indicator-val/bytes_check.py`
