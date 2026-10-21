# memory-v2-arm: fix

Workflow run `wf_df663b0e-1c8`, phase Fix. Measure memory failures in real transcripts, then design, implement, review and fix harness-extracted memory (solved levels, world-model card, action effects, seen-state notes, hypothesis deltas) on stack-v2

---

All 9 findings are fixed in 12 new commits on `arm/memory-v2`; none rejected. Two fixes go beyond the reviewer's exact proposal (findings 2 and 5) and say why. The full suite passes when run niced, and nothing was pushed.

**Per finding**

1. Hazard wording blamed the last action — **fixed** (`bc27548`).
   - Lines now read "L2: attempt 2 ended in GAME OVER after 31 actions; last action RIGHT from the board first reached at action 14 of the attempt". The action index is now clearly within the attempt.
   - The Seen line's `FATAL (attempt n)` became `GAME OVER (attempt n)`.
   - A test now forbids killed/kills/fatal/cause/because, and fixes a vacuous assertion in the hold test.
2. Low-value `how:` clause — **fixed, stricter than proposed** (`1110333`).
   - It needs a score of at least 3. It skips sentences that open with a plan or command word (the reviewer's list plus execute, press, check, need, handle and others), also after a short "label:". It also skips hedges (should, would, might, maybe, I need, we need) and sentences ending in "!".
   - Measured on r2's 112 clears: the reviewer's rule alone still kept 53 sentences, many of them imperative ("Execute: …", "batch: …", "be careful …").
   - Now 34 kept (was 84), none imperative. `how` appears on 27 of 112 solved lines (was 65). Both README examples are gone.
   - Precision is still only about a third useful; the rest is harmless description. It stays the arm's least certain part.
3. `end` diff dead weight — **fixed by removal** (`a90dff3`).
   - A larger budget can't help: 100 of 112 level ends change 6 or more objects, and the solved line is capped at 165 chars.
   - Removed `_mem_end_diff`, `FactRecorder(end_diff=)`, the start grid kept per level, and `end` from solved lines and the card stages (one fewer stage).
4. Stale per-level events — **confirmed with `sem2.py`, fixed** (`3a3ba8c`, test in `2819e3b`).
   - New `MemoryLedger.clear_level_facts()` sets a version floor and drops queued `hazard:` and `fx:` events below it. `remove_section` also prunes dead events.
   - `restore_display` now filters its snapshot too, so a rolled-back turn can't bring them back.
   - The new end-to-end test fails on the old code: L2's "UP: first edge only" leaked to L3 in delta placement.
5. Stale Seen-line outcome words — **confirmed with `sem3.py`, fixed differently** (`6622c4f`).
   - The reviewer's rule doesn't work as written: the destination board always has a visit, because the action itself recorded one.
   - The line now stores the outcome kind and the board reached, and words it when shown: `UP -> the board after action #2` or `the start board`, otherwise `no board change` or `GAME OVER`. The revisit stats are still counted when the action is taken.
6. `<actions>` has no level label — **fixed** (`1c65ee2`). It now opens as `<actions level="N">`, and the system-prompt line says the counts are for level N, then the whole game after `| game`.
7. README dead links and other branches' arms — **fixed** (`d252fa3`). The section now lists only what this branch contains (#34, #45, #51, #52, #56, #61, memory-v2) and points to `docs/fork-readme` for the fork-wide list. Every link now resolves on this branch.
8. Config trap with `ARC3_MEMORY_PLACEMENT=delta` — **fixed** (`6a33203`).
   - The placement now applies only when a memory-v2 flag is on, so the plain ledger always shows after a trim.
   - `memory_addendum` drops `LEDGER_ON_TRIM_ADDENDUM`, which promises a head card, when no head card will come. No arm used either combination.
9. Hard-coded border — **fixed** (`a6b4746`). `_mem_state_border()` defaults to `ARC3_NOOP_GUARD_BORDER`, which is itself 4 by default.

**Test gaps**
- **15k hold:** partly right. The existing hold test did run with the default 15000, but the sim reported no generated tokens, so the hold never released. `memory_sim` now reports `MEMSIM_COMPLETION_TOKENS` when set (unset keeps the flags-off golden test unchanged). The new test checks the hold per level, the release on L1 and no stale lines on L3 (`2819e3b`).
- **Level-change matrix:** covered by the same end-to-end test plus a ledger unit test.
- **Capped card:** a new property test caps the card at 400–1200 chars. Every card stays under the cap, and the only lines the model never sees are ones the cap cut (`7d7bda0`).
- **Border-4 hashing from frames:** a new test feeds r2's real frames through `FactRecorder.feed` and gets 14.27% revisits and 8.95% exact repeats, inside the 14.2/9.0 anchor. It takes 2.7 s and skips when the run data isn't on disk (`5965a6f`).
- **Accepted risk, left unchanged:** `loop_free` still merges boards that differ only in the edge band.
- The reviewer's fuzz (12 seeds × 400 turns) still passes with `end_diff` taken out: card at most 3900 chars, nothing raised.

**`git log --oneline arm/stack-v2..HEAD` (new commits on top)**
```
d252fa3 README: list what this branch contains, without dead links
5965a6f Test the replay anchor from the run's frames, not cached hashes
7d7bda0 Test the head card plus deltas under the card's hard cap
2819e3b Test the 15k hold releasing and restarting per level end to end
6a33203 Keep ARC3_MEMORY_PLACEMENT=delta from hiding the plain ledger
a6b4746 Default ARC3_MEM_STATE_BORDER to the no-op guard's border
1c65ee2 Name the level the card's action counts are for
3a3ba8c Drop the last level's queued hazard and first-effect lines on a new level
a90dff3 Drop the solved line's end diff
1110333 Keep plans and commands out of the solved line's how sentence
6622c4f Word the Seen line's tried actions when the line is shown
bc27548 Say how an attempt ended, not what killed it, in hazard and Seen lines
c6731e7 README: add the memory-v2 arm
ae9cf6c README: carry the fork's arm list from docs/fork-readme
f6a1e88 Document the memory-v2 flags, what the model sees and the read-outs
598b77b Add the memory-v2 Kaggle arms
c1d5dcf Check every placement keeps requests extending between trims
2578765 Reuse the previous action's board hash in FactRecorder
671ba0c List keys never tried this game on one actions row
a1ce132 Skip code and move traces in the solved line's how sentence
95643a9 Test memory-v2 placements end to end through analyze()
2233132 Bring the memory card back after a rolled-back turn
b3f8d24 Run the memory-v2 placements in the eviction sim with every line shown
0f26a21 Wire memory-v2 into ToolAgent behind ARC3_MEM_* flags
300310c Add the memory-v2 system-prompt lines
982c4b4 Give MemoryLedger versioned harness items, a card renderer and deltas
b197476 Hash boards for the repeat guards through memory_facts.state_hash
fdeba21 Add memory_facts: board hashes, effect classes, attempts and seen boards
1e21ccd Document the slot-handover rule for memory features
8fb1ec6 Test that only real history drops hand the GPU slot over (#58)
b89c93e Add eviction_sim: does anything but a real drop hand the slot over?
f18f8f1 Add setup and before_turn hooks to memory_sim.run()
d01f845 Snapshot what stack-v2 sends the model, for a flags-off identity test
```

**pytest (`nice -n 10`, `-rs`):** `106 passed in 30.32s`, no skips.

**PR summary (memory-v2, review follow-up)**
- **Hypothesis (unchanged):** the model wastes evidence it already has (9.0% exact (board, action) repeats, 27% of actions on net-zero loops, 15% of tokens in attempts that die). So the harness records facts from every action and shows them as never-evicted solved lines, action-effect counts, hazards and Seen lines, behind `ARC3_MEM_*` flags.
- **Changes from the review:**
  - Death lines state how an attempt ended, never the cause: GAME OVER after n actions, last action, and the board it was taken from.
  - Seen lines name the board each tried action reached, as it stands now.
  - The solved line's `how` skips plans, commands and hedges; `end` is removed.
  - A new level drops the previous level's queued hazard and first-effect lines.
  - `<actions level="N">` says which level the counts are for.
  - The board border follows the no-op guard's.
  - Placement `delta` can no longer hide the plain ledger or leave a system prompt that promises a missing card.
  - The README lists only this branch's contents.
- **Testing:**
  - New tests cover the hold releasing and restarting per level (this one catches the old leak), the capped card plus deltas, the frame-hash replay anchor, plan filtering, the hazard wording, the level label, the border default and the placement trap.
  - The flags-off identity golden test and the #58 eviction guard (all 10 configs) still pass.
- **Results and cost (r2 replay, 25 games, no hold):**
  - Revisits 14.27% and exact repeats 8.95% from the recorder's own frame hashes.
  - Seen lines on 13.8% of turn messages (mean 194 chars) and deltas on 39% (mean 91 chars), about 900 tokens per game together.
  - Final card: median 978 chars, max 1394. No more ~114 ms `end` diff per clear.
  - No Kaggle score yet; the keep/drop rule in `CONFIGURATION.md` still applies.

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/memory-v2/`:
- `ARC3-Inference/inference/agent/memory_facts.py`
- `ARC3-Inference/inference/agent/memory_ledger.py`
- `ARC3-Inference/inference/agent/tool_agent.py`
- `ARC3-Inference/inference/agent/prompts.py`
- `ARC3-Inference/tests/memory_sim.py`
- `ARC3-Inference/tests/test_memory_facts.py`
- `ARC3-Inference/tests/test_ledger_on_trim.py`
- `ARC3-Inference/tests/test_memory_placement.py`
- `ARC3-Inference/CONFIGURATION.md`
- `README.md`

Measurement scripts are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/memv2-fix/`.
