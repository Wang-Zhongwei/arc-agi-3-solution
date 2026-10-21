# memory-v2-arm: review

Workflow run `wf_df663b0e-1c8`, phase Review. Measure memory failures in real transcripts, then design, implement, review and fix harness-extracted memory (solved levels, world-model card, action effects, seen-state notes, hypothesis deltas) on stack-v2

---

I found no high-severity defects and no cache-safety defects. The worktree is still clean.

**What I ran**
- The full suite passes: 97 passed, no skips with `-rs`. The flags-off identity golden test and the replay tests actually ran, since they were not skipped.
- `tests/test_memory_eviction.py` ran all 10 configs (`mem-trim`, `mem-delta`, `mem-both` included) and all passed. The three `neg-*` traps are still caught, so the checks can see them.
- I fuzzed the recorder, ledger and `_mem_apply_events` over 12 seeds of 400 turns, with up to 30 solved levels and oversized notes. The card never exceeded 3900 chars, deltas stayed within the 2×480 limit, and nothing raised.

**Spec items checked**
- Placements: nothing earlier in history is edited between trims. Deltas and the Seen line are written once into the new opener. Resume openers get none. Rollback restores deltas and the card flag.
- Level clears, batch clears, RESET, game over and empty history are handled. The `(row, col)` order in click signatures is right. `FactRecorder` hashes with border 4, as `gameplay_changed` does.
- Fable-comparison edits:
  - E1: hold of 15000 tokens per level, with solved lines, first deaths, clears and the trim card bypassing it. The "first death" bypass is per key, not per game.
  - E3: none of the banned words appear.
  - E4, E5 and E6 are honoured and tested.
  - E7: the kill rule is documented.
  - E8: only the placement is done, with no shared cap and no dedup against Salient, as the report says.

**Defects, most severe first**

1. Medium: hazard and fatal wording blames the last action (`memory_facts.py:878` "killed attempt", `:789` "FATAL").
   - The harness cannot tell a budget death from a hazard death. The Fable comparison found 17 of 44 deaths had the step bar empty or nearly empty. In those cases "RIGHT from the board first reached at action 14 killed attempt 2" is misleading.
   - It cuts against the spec's own "never a cause" rule and the "no verdict words" check.
   - "action 14" is also ambiguous between a global step and an index within the attempt.
   - Fix: say "attempt 2 ended in GAME OVER after 31 actions; last action RIGHT from the board first reached at action 14 of that attempt". Use "GAME OVER (attempt n)" instead of "FATAL".

2. Medium: the `how:` clause is low-value text in a never-evicted line (`memory_facts.py:243-290`).
   - The implementer's own README examples show "Just try UP now (1 action) — the dial matches now!" and "The 4th DOWN takes us to (35,14); the 5th presses into the box." These are plans for the final move, not how the level was solved, and an imperative like "Just try UP" may mislead on the next level.
   - Fix: require a score of at least 3. Reject sentences starting with try, just, let, I'll, should or next. Otherwise drop `how` and keep the note only. The `how` clause is the arm's least certain part.

3. Medium-low: the `end` diff is dead weight (`tool_agent.py:202-250`, called from `FactRecorder.step`). It appears on 1 of 112 clears and costs about 114 ms per clear. Remove it or give it a larger budget.

4. Low: stale per-level events survive a level change.
   - `remove_section("hazard")` (`memory_ledger.py:517`, called at `tool_agent.py:4408`) leaves queued `hazard:` and `fx:` events in `_queue`.
   - I reproduced it: after a clear of L1, the first non-held delta on L2 contains "+ L1: RIGHT from the board … killed attempt 2" and "+ LEFT: first moved".
   - Fix: drop queued `hazard:` events in `remove_section`, or filter `take_delta` with `_event_alive`.

5. Low: the Seen line's outcome words go stale (`memory_facts.py:793`, rendered at `:855-858`).
   - The word is fixed when the action is tried. After going UP to A, UP to B, DOWN to A, the line at A reads "UP -> new board" although B has been visited (reproduced). Compute the word at render time from whether the next board has visits.

6. Low: the `<actions>` section carries no level label (`memory_ledger.py`, card build).
   - A head card kept across a level change still says "this level" with the old level's counts, until the next trim. Use `<actions level="3">` or accept the staleness.

7. Low: README links and arm rows point to files and arms that are not on this branch.
   - `serving/CUSTOM.md`, `analysis/README.md`, `serving/kaggle/` and `serving/runtime_patches/` do not exist here, and the arm table lists arms from other branches.
   - This conflicts with the "keep READMEs current" rule.

8. Low: a config trap in `_note_history_evicted` (`tool_agent.py:6624`).
   - `ARC3_MEMORY_PLACEMENT=delta` with only `ARC3_LEDGER_ON_TRIM` silently turns the ledger card off, while `LEDGER_ON_TRIM_ADDENDUM` still promises a head card. No arm does this.

9. Low: `ARC3_MEM_STATE_BORDER` is hard-coded to 4 (`tool_agent.py:215` and `:4364`). The spec says it equals the no-op guard border so the two agree. Default it to `NOOP_GUARD_BORDER_WIDTH` instead.

**Test gaps**
- The 15k hold is never exercised end to end, because the sim sets `ARC3_MEM_LINES_AFTER_TOKENS=0`.
- The spec's level-change matrix is one test, and it missed defect 4.
- The "head card plus later deltas" property test uses `card_chars=100000`, so the cap path is not exercised.
- The replay anchor test uses the cached `ih` hashes, so it does not validate the recorder's own border-4 hashing from frames.

**Accepted risk worth noting:** `loop_free` merges boards that differ only in the edge band. A hidden-state change there, such as a collected-key icon, can be cut out of the path. The system prompt still calls it "the winning attempt's path with its loops cut out".

**Spec deviations that are fine:**
- The hard cap can drop refuted entries and hazards as a last resort, where the spec says never.
- Arm names differ from the spec's (no B+C+D `mem-facts` arm).
- `anim` is checked before `noop` and `hud`.

Files are under `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/memory-v2/ARC3-Inference/`:
- `inference/agent/memory_facts.py`
- `inference/agent/memory_ledger.py`
- `inference/agent/tool_agent.py`
- `inference/agent/prompts.py`

My probe scripts are in `.../scratchpad/review-memv2/` (`fuzz1.py`, `sem1.py`, `sem2.py`, `sem3.py`, `sp.py`).
