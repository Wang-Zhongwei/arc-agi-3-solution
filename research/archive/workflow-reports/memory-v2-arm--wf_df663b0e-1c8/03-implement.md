# memory-v2-arm: implement

Workflow run `wf_df663b0e-1c8`, phase Implement. Measure memory failures in real transcripts, then design, implement, review and fix harness-extracted memory (solved levels, world-model card, action effects, seen-state notes, hypothesis deltas) on stack-v2

---

Branch: `arm/memory-v2`, in the worktree `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/memory-v2`. I fast-forwarded it to `arm/eviction-guard` first, as asked. Nothing was pushed, and the main working tree was not touched.

**Tests:** the full suite ran once, niced, and gave `97 passed in 27.46s`. That includes the 3 memory-v2 eviction configs that used to be skipped (`mem-trim`, `mem-delta`, `mem-both`); all three now pass. The 3 negative controls still fail as designed, and the flags-off golden is byte-identical to stack-v2. Imports resolve to the worktree.

## git log arm/stack-v2..HEAD
```
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

## What the model sees, per component
Examples marked "real" come from replaying the r2 ledger run's frames and transcripts. Examples marked "sim" come from `tests/memory_sim.py`.

**System prompt (static, all components on, placement both)** — "Memory kept by the harness:" followed by one bullet each for:
- the head card,
- `<memory_update>`,
- `<solved>` (explains U D L R S and M(r,c)),
- `<actions>`/`<hazard>`, defining the counted words: moved = cells moved, colour counts unchanged; changed = colour counts changed; edge only; no change; anim only,
- `Seen before`,
- the note tags, including tags written in python comments,
- the line "Paths, action effects and deaths are already recorded by the harness, so a note holds what the frames do not show".

With only the ledger on, it is `LEDGER_ON_TRIM_ADDENDUM`, unchanged.

**A, solved (real, ls20)** — on the first turn message of level 2:
```
<memory_update>
+ solved L1: 13 acts [L3 U4 R3 U3], spent 40, 0 deaths | how: Just try UP now (1 action) — the dial matches now!
</memory_update>
```
A sim example with a model note attached: `L1: 7 acts [U3 R4], spent 22, 2 deaths | how: Moving RIGHT 3 times from (5,9) reaches the yellow goal at (5,12). | note: the goal is the yellow cell`

**C, actions (real)** — rows in the card:
```
UP: moved 5, changed 7, no change 1 | game clear 1, moved 89, changed 31, edge only 2, no change 2
LEFT: moved 12, changed 7, edge only 1 | game clear 1, death 1 (1 board), moved 65, changed 17, edge only 6
MOUSE@b 6 targets: death 1, moved 7, changed 68, edge only 6; median 63 cells; e.g. (61,9) changed, (61,39) changed
colours on the board not clicked this level: W c B
<hazard>L4: M(61,15) from the board first reached at action 48 killed attempt 1 (50 actions)</hazard>
```
Delta lines look like `+ RIGHT: first moved; UP: first edge only`. Counts appear only in the card. A key that never changed anything reads `SPACE: no interior change on 3 presses across 3 boards`.

**D, seen (real, ls20 level 3):**
`Seen before (board interior matches): after action #8 and #44 of this attempt. From it you tried: RIGHT -> no board change, LEFT -> new board. Untried: UP, DOWN, SPACE.`
The spec's worked example reproduces exactly as a unit test: `after action #6 and #8 ... DOWN -> back to a seen board (loop). Untried: UP, LEFT, RIGHT.`

**B, card (sim)** — `<solved>` lines, then `<actions>`, `<hazard>`, `<world_model>`, then confirmed and refuted hypotheses. In card mode, "Correction:", word-overlap dedup and confirmed/refuted flips are applied.

**E, placement:**
- `trim`: the card only at the head after a real trim.
- `delta`: only `<memory_update>` blocks in new turn messages.
- `both`: both of the above.
- The per-turn block sits after the "Valid actions" line, same slot as facts-stack (E8).
- Under the default 15k hold (sim), the only lines are first-death hazards and `+ solved ...`.

## Limitations
- **`end` diff is nearly dead at the spec's 30-token budget.** It appears on 1 of 112 real r2 level clears, because 100 of 112 level ends change 6 or more objects. It still costs about 114 ms per clear (median; max 0.8 s). Drop or redesign it.
- **`how` is present on 65 of 112 clears, and quality is mixed.** I extended `distill_think` beyond the spec. It now skips code and arrow traces, bullets, and sentences under 20 characters. It also no longer counts the noun "block" as a finding word.
- **Deviations from the spec:**
  - `anim` is checked before `noop` and `hud`; in the spec's order it can never be reached.
  - The 165-char solved-line cap admits parts in priority order: note, then how, then end. Paths are cut to 64 characters with `...+N`.
  - The per-turn block follows E8's placement, not "after the last image".
- **Rollback fix.** I found an existing gap: when a persist trim is followed by a failed turn, the card is lost until the next trim. I fixed it by restoring `_ledger_due`, but only under memory-v2 flags, so the stack-v2 ledger arm still has the gap. A related existing issue is untouched: after that rollback, the retried request breaks the prefix without a handover.
- **More handovers in the eviction sim: 8 → 9-10.** Every one follows a real drop. The extra memory text causes more trims in the sim's small 14k-token window.
- **Hold workaround in tests.** The eviction and placement tests set `ARC3_MEM_LINES_AFTER_TOKENS=0`, because the scripted replies report no tokens. The hold's real effect is therefore unmeasured.
- **Volume without the hold (25 real games):**
  - Seen lines appear on 13.8% of turns, mean 183 characters (the spec estimated about 19%).
  - Deltas appear on 39% of turns, mean 95 characters.
  - Together that is about 900 tokens per game.
  - The final card is a median 1.0k characters, max 1.5k.
  - Recorders cost 0.39 ms per action.
- **Board identity.** The border-4 crop merges boards that differ only in hidden state. In ls20, moves often count as "changed" because a bar inside the interior changes colour counts.
- **Not done:**
  - E8's dedup against Salient's lines (needs the facts-stack merge).
  - E2 (skipped, as instructed).
  - No no-hold arm variant (the knob is documented).
  - `arms.md` is not updated; it is outside the worktree.
  - Nothing has run on Kaggle.
- **README.** The README commit brings in docs/fork-readme's README (35993e2), because this branch had no arm list to add to.

## Files
- `.../wt/memory-v2/ARC3-Inference/inference/agent/memory_facts.py` (new)
- `.../wt/memory-v2/ARC3-Inference/inference/agent/memory_ledger.py`
- `.../wt/memory-v2/ARC3-Inference/inference/agent/tool_agent.py`
- `.../wt/memory-v2/ARC3-Inference/inference/agent/prompts.py`
- `.../wt/memory-v2/ARC3-Inference/tests/test_memory_facts.py` (new)
- `.../wt/memory-v2/ARC3-Inference/tests/test_memory_placement.py` (new)
- `.../wt/memory-v2/ARC3-Inference/tests/test_ledger_on_trim.py`
- `.../wt/memory-v2/ARC3-Inference/tests/memory_sim.py`
- `.../wt/memory-v2/ARC3-Inference/tests/eviction_sim.py`
- `.../wt/memory-v2/ARC3-Inference/CONFIGURATION.md`
- `.../wt/memory-v2/kaggle/arms.json` (adds `memory-v2-trim`, `memory-v2-delta`, `memory-v2-both`, `solved-levels`, `action-effects`, `seen-state`; `memory-v2-both` builds locally)
- `.../wt/memory-v2/README.md`

`...` = `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad`

Replay scripts: `.../memv2/replay_real.py`, `.../memv2/summary.py`, `.../memv2/show.py`
