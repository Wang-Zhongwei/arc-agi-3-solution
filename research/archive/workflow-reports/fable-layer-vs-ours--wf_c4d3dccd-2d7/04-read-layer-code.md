# fable-layer-vs-ours: read:layer-code

Workflow run `wf_c4d3dccd-2d7`, phase Read. Compare leoprovorov Fable & Astra notebook (handbook + Fable layer on Franzen) with our stack: read, measure claims on our runs, compare, skeptic-verify

---

# Fable layer (leoprovorov F001) compared with our stack-v2

Scratch files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/fable-compare/`. Abbreviations used below:
- **TA**: Franzen's `tool_agent.py` at tag `franzen-m2`. Applying F001 cell 2 to `bundle-base` produces this file byte for byte, so it is exactly what F001 patches. [verified]
- **P**: `fable_layer_patch.py` (analysis notebook cell 42).
- **C**: `fable_layer_core.py` (cell 41).

The copies embedded in F001 cell 6 are identical to cells 41 and 42 after whitespace strip. [verified]

## Short verdict: what their notebook has that ours doesn't

- **Harness-computed facts from frames that need nothing from the model.** Our ledger depends on the model writing tags, which are rare (2.7–3.9 tags per 100 calls, memory-v2-design.md:3). memory-v2 heads the same way but isn't built yet.
- **Step-budget bar detector.** It works on real frames.
  - On ft09 in our `fz-ledger-on-trim-r2`, row 63 loses 2 cells per action.
  - Level 2 hit GAME_OVER at action 37.
  - At step 7 the Fable layer would have said: "the bar along row 63 has 20 of 64 cells left after 23 actions… about 10 more actions empty it". [verified by replay]
  - Nothing in our stack or in memory-v2 does this. memory-v2 explicitly never names budget as a cause (design §3).
- **Indicator / tray-mark tracking, including "lost" marks.** On sk48 in our r2 run it finds the same tray patch the handbook describes (rows 58-59, cols 28-29, red↔white). [verified]
- **Gated by tokens spent on the level.** Levels solved quickly get nothing.
- **Pinned digest of cleared levels.** This is the idea behind rellik13's +2.8.
- **Notebook robustness:** a fallback to Python 3.12 for the server venv.

What theirs lacks:
- **No score has ever been measured for it.** Handbook p.84 lists "the pinning variants" as "not yet run". [verified] Their only measured change, a journal that replaced history, cut levels from 46 to 23 (p.84). Our ledger-on-trim measured 52.56 and 49.14 against control 46.25 (arms.md:53-54).
- The bugs in section 4. Two of them break when combined with our ledger-on-trim.
- It adds advice, against their own "facts, not advice" rule.

## 1. F001 vs our `kaggle/template.ipynb`, every difference except cell 0b

Cells were diffed one by one: F001 0–4 against ours 0–4, F001 7–24 against ours 5–22. [verified]

1. **Cell 0 (markdown):** F001 adds one line, "F001 (Fable layer). Based on P001 (pipeline check)…".
2. **Cells 5–6 (new):** the "0b. Fable layer" markdown and code (switch panel at cell 6 lines 862–867; `apply()` at 869–880).
3. **Serving cell (F001 cell 14 vs ours cell 12, line 186):** F001 builds the venv from `/usr/bin/python3.12` when the kernel isn't 3.12 ("server wheels are cp312… 3.13 image"). Ours calls `uv venv --python sys.executable`.
   - Our `template-metadata.json` pins Franzen's byod docker image, so we likely don't need this. [inferred]
   - F001's docker image is unknown; the fix suggests they ran on a 3.13 image. [inferred]
4. **Notebook metadata:** the same 5 dataSources, with the two modelInstanceVersions (1007514/1007009) in swapped order. Ours also carries papermill metadata (start 2026-09-29). [verified]
5. **Everything else is identical** [verified]:
   - The harness patch (cell 2).
   - The env cell (cell 4): ACTION_INFO, ANIMATION, DIFF_IMAGE, GAMEOVER_DIFF_IMAGE, EXPLAIN_GAMEPLAY_CHANGED, NEW_CHANGED_PROMPTS, FRAME_DIFF_HINT, LEVEL_TRANSFER_GUIDANCE, MEMORY_SECTIONS=off, PERSISTENT_FUNCTIONS(game, imports), EXPOSE_UNDO=on.
   - Context 128k, max output 12k, drain 58k, 150 turns, drain 30, temperature 0.7.
   - Priority scheduler with 10 streams, refresh queue, tail fade 0.2, score normalization.
   - Model: W4A16 AutoRound plus the albucino drafter, served by Pennyroyal v2.5.3.
   - Every serving arg except the venv line.
   - Time limits (cell 18 vs our 16): submission 532 min per game at concurrency 120; demo 25 min on Franzen's 10 games.
6. **What our `build_notebook.py` adds on top of the template, so these also separate our submission from F001:**
   - The patch is `git diff bundle-base..HEAD`, gzip+base64. For stack-v2 that includes ledger-on-trim and the #4/#5 fixes (:264-277).
   - Arm env overrides: stack-v2 sets `ARC3_LEDGER_ON_TRIM=1` (:287-292, arms.json).
   - Input-path resolver (:78-104).
   - Fail-fast if the server dies (:145-151).
   - serve.log streaming (:158-213).
   - Hooks for draft KV, attention backend and adaptive speculation, all inert by default (:105-144).
   - Demo panel of all 25 games at 121 min instead of Franzen's 10 games at 25 min (:214, :327-339). The submission path is the same.

## 2. Each mechanism: exact semantics

### Gates

- **Layer switch:** on only when `ARC3_FABLE_LAYER=1`. F001 sets it whenever `apply()` returns applied (cell 6:873).
- **Tokens on the level:** `_session_generated_tokens - _tokens_at_level_start` (P:53-54). The level start is reset on `level_transition` mid-turn, inside `_update_summarized_knowledge_from_step_summary` (TA:3863; called from tool dispatch at TA:5945). [verified]
- **Deaths:** count toward the same level. [verified]

### M1: Indicator notes (P:123-154, C:151-189, C:328-417)

**Detector.** Runs on each consecutive pair of frames (`HistoryEntry`, one per action, TA solver:1339).
- Fewer than 3 changed cells, or more than 60 components: nothing.
- Drops bar-like components: within 6 cells of the edge and part of a one-colour run of 12 or more (C:103-122).
- Clusters components within 8 cells of each other; the largest cluster is the "main" activity.
- Reports clusters with 2–16 cells, a box of at most 6×6, at least 8 cells from main, and not an equal-size vacated/occupied pair.
- Returns `[]` if more than 2 qualify.

**Ledger state.**
- Marks are keyed by the exact box. Events are classified first / again / lost (lost means back to the first colour).
- More than 6 distinct boxes makes the level permanently noisy: events are cleared and no further notes appear.
- RESET clears marks but not the noisy flag (C:331-334, C:383-388).

**Display.**
- Only after at least 15k tokens on the level and only when the level isn't noisy (P:134-135).
- Shows the last 2 events found since the previous sync (P:137-139).

**Text the model would have seen** (replayed on our r2 frames; their own example is narrative cell 23) [verified]:
- sk48 step 9: "Indicator note: after action 20 on this level (LEFT), the small patch at rows 58-59, cols 28-29 went back to its original colour, red (it had changed to white after action 15). A mark you earned may have been lost: compare what you did between those two actions."
- ar25: "…a small patch away from the moved objects changed: rows 15-17, cols 0-2, sky blue to charcoal. Legends, trays and counters often report progress; note which action did this."

**Placement.** A line inside the turn opener (user message), after "Valid actions right now" and before `_death_ledger_observe` and the closing tool block (TA:5084-5092). It is new tail text, so it costs nothing in cache.
- Default yield-resume mode is "full" (TA:776-791), so resumed openers recompute the lines; fresh events are already consumed by then. [verified]
- Each note is about 75 tokens.

### M2: Pinned LEVEL LOG (P:171-237, C:419-540)

**Trigger.**
- At least 3 actions and at least 30k tokens on the level (P:185).
- Rewritten every further 40k tokens (P:187-189).
- Skipped if the last history message is a user message (P:190-192).
- Called in `analyze()` after the opener is built and after the rolling summary, before the opener is appended (TA:6631-6635).

**Content.**
- A header with level, actions here and tokens.
- Per key: times pressed, times changed, times did nothing, and a "typical effect", which is the centroid shift of the colour that moved most (C:272-301, 442-472).
- Clicked coordinates.
- "Pressed fewer than twice…" and "Valid but not pressed…".
- UNDO count.
- The last 5 indicator events, or the sentence "No indicator patch … has changed on this level so far; if the screen has one, no action of yours has moved it."
- The budget line, shown even when the bar isn't low.
- A digest of the last 2 cleared levels: actions, effects, last 10 key names.
- 5 PROTOCOL lines headed "How strong runs work on a level like this (use it now):" (C:509-516). These are advice.

It is followed by an assistant message the model never wrote: "Noted: these are harness-computed facts about this level, and I will check my plan against them." (C:507).

**Real example** (r2 bp35 level 1, 1,649 chars):
> LEVEL LOG - facts the harness computed … level 1, 15 actions here, about 42982 generated tokens so far.
> - LEFT: pressed 4, changed the board 3, did nothing 1 (first at action 9); typical effect: blue cells shifted left 6 (2 of 3 presses checked)
> - MOUSE: pressed 4, … clicked at (15,33), (50,30), (33,20), (15,45)
> No indicator patch … Step budget …: the bar along row 63 has 49 of 64 cells left after 15 actions…
> [5 protocol lines] [verified]

**Placement and cache.**
- The first pin is appended at the tail as a user/assistant pair, so it costs no cache.
- `_fl_restore_pinned` runs at the end of `_trim_messages_for_context` (edit 5). If the pin's user content is missing from history, it puts the pair back right after the system prompt (P:219-231).
- The restore comes after the `len(history) != len(messages)-1 → _note_history_evicted()` check (TA:6544-6545).
- A normal trim drops blocks, which already raises the flag and invalidates the prefix from the head, so the restore is free.
- A refresh does two things:
  - It removes the old pair wherever it sits (P:156-169, 206-207). After a trim that is the head, which forces a full re-prefill.
  - It appends a new pair at the tail.
- A new level only forgets the pin (P:233-237). The old log stays in history and ages out.

**Measured size and cost.**
- Size: mean 2,211 chars per pin (about 670 tokens at 3.3 chars/token), which matches their "0.6–1k tokens".
- 25-game replay of r2: 53 pins, 18 refreshes, 41% of the 129 levels reached got a pin.
- Their own counts: of baseline levels that passed 30k tokens, 73% were cleared anyway (handbook p.86). [their claim, unverified]

### M3: Budget alert (C:192-269, P:112-121, 141-150)

**Detection.**
- Uses only frames from the current attempt (since the last RESET or level start).
- Looks for a row or column with a one-colour run of 12 or more in the first frame. Changed cells must grow monotonically, stay contiguous at one end, end in a single colour, change by at most 6 per step, change on at least 30% of steps, and drain at 0.15–4 cells per action.
- `remaining = left / rate`; the rate counts UNDO and no-op actions too.

**Display.**
- An opener line only when `left/cap ≤ 25%` or `remaining ≤ 12`, and only if the level has at least 6 actions. There is no token gate.
- Example: g50t, "the bar along row 63 has 13 of 64 cells left after 103 actions … About 26 more actions empty it, and the level then resets (actions already spent still count)."
- r2 replay: 19 alerts across the 25 games.

### Totals, failure handling and hot-path cost

**Totals, r2 replay** (counterfactual text on the real trajectories, about 4.52M generated tokens) [verified]:
- 128 note lines, 19 budget alerts, 71 pins or refreshes.
- About 33k chars added to openers and about 157k chars of pins.

**Failure handling.**
- `apply()` requires each anchor exactly once and a compile check; otherwise it returns "skipped" and leaves the layer off (P:281-313).
- Each hook is wrapped in try/except that logs at `log.debug` and returns `[]` or `None`. Failures are silent at the default log level.
- Off means no-op on every path (P:31, 125, 173, 222). [verified]

**Hot-path cost** (their real code on r2 frames, niced laptop) [verified]:
- Mean 0.3–2.6 ms per analyze call per game; maximum 24 ms; 1.96 s CPU over 1,565 calls.
- Sync is incremental. The budget detector is cached for up to 5 new frames.
- Memory:
  - Measured: Ledger keeps up to 40 before/after grids per key per level, for the whole game. In r2 that reached 457 samples in one game.
  - Estimate: each sample may hold about 2 distinct grids because of per-turn reloads, so about 33 MB per game at 121 min. Across 110 games that could be several GB of host RAM; the fz-ledger-on-trim log showed 106.6 of 176.9 GB used. [inferred]
- A refresh costs one unflagged prefill. A median ~55k-uncached request took 4.7 s to prefill in r2 (`requests.csv`). [verified]

## 3. The 5 anchored edits on stack-v2

All 5 anchors occur exactly once in stack-v2 and in attention-prompt, there are no `_fl_` name clashes, and `apply()` succeeds on a scratch copy of stack-v2. [verified]

| Edit | TA line | stack-v2 line | attention-prompt line | Conflict |
|---|---|---|---|---|
| 1. Methods before `_build_user_prompt` | 4806 | 4867 | 4880 | None |
| 2. Opener lines after valid-actions | 5087 | 5148 | 5161 | Textually none. memory-v2 puts `_mem_observe` after `_death_ledger_observe` (~5153) and deltas at the opener's end. That makes two independent frame ledgers with different semantics: Fable counts the clearing action as "did nothing" (P:88-92), memory-v2 classes it as clear. Fable's key table duplicates memory-v2 (C) and its digest duplicates (A). |
| 3. `_fl_forget_pin` on level change | 3863 | 3910 | 3923 | Sits just before stack-v2's `self._ledger.on_level_change` (3912-3916). Fine. |
| 4. `_fl_maybe_pin` in `analyze` | 6635 | 6717 | 6730 | Still after the rollback snapshot (stack-v2:6697), so bug B4 below carries over. |
| 5. Restore after the eviction check | 6544 | 6626 | 6639 | **Breaks with ledger-on-trim** (next paragraph). |

**Edit 5 and ledger-on-trim.**
- The stack-v2 request loop runs trim (Fable restores the pin at the head), then `_apply_due_ledger` (stack-v2:6825-6826).
- `_apply_due_ledger` merges `<memory>` into the first user message, which is now the pin.
- The pin's content no longer equals `_fl_pinned`, so the next request re-inserts a plain copy at the head with no trim flag.
- Simulated on the real stack-v2 functions: after every trim the next request has a changed head and no flag, giving 2 copies of the LEVEL LOG (6 of 12 requests in the sim, script `sim_stackv2_conflict.py`). [verified]
- memory-v2's "trim" placement uses the same `_with_ledger_head` slot, so it hits the same collision.
- A fix would match the pin by a marker key, not by content, or render the LEVEL LOG inside the single head card.

**attention-prompt** changes only the static system prompt, so there is no code conflict. In content:
- Its rule "set aside changes confined to a timer or remaining-steps bar" works against Fable's protocol line 2 ("look at … the step bar") and the budget alert.
- rellik13 deleted a similar "ignore the bar" line because the bar is the action budget (WRITEUP.md:84-85).
- Adding both would stack two advice blocks, and rellik13 measured advice hurting: 6.42→4.16 and 13.40→9.98 (WRITEUP.md:59-60).
- Its "colors covering only a few cells" rule overlaps the indicator notes and the arms.md:197 "harness-flagged new elements" idea. [inferred]

## 4. Bugs and edge cases in their code

- **B1. The budget alert fires about once per level, not per attempt** as the comment at P:146 says. The rate limit `remaining ≤ last-10` is keyed by level and only reset on level change. A synthetic 3-attempt test with real code: attempt 1 alerts at 16 left; attempts 2 and 3 never alert (`test_budget_attempts.py`). [verified]
- **B2. The budget cache can be up to 5 frames stale** (P:117). In the trace it showed 16 left when 12 were true, so "about N more actions" overstates. [verified]
- **B3. A refresh removes the old pin with no `_note_history_evicted`.** The prefix (often the full head) is invalidated, but `_maybe_handover` only fires on `_context_was_trimmed` (TA:6380). The game keeps its slot and warm band through a full prefill: 18 times in the r2 replay, about 5 s each. [inferred from code, with measured frequency and prefill time]
- **B4. A failed request leaves `_fl_pinned` and `_fl_pin_state` set.**
  - On a request error `analyze` restores the snapshot taken at TA:6615, before the pin; tokens roll back as well (TA:7145, 7184-7186).
  - The pin isn't re-added; the next trim restores it at the head with no flag, an unflagged full re-prefill.
  - If the failed turn was a refresh, the old log also stays, so there are two. [inferred]
- **B5. The ledger-on-trim and memory-v2 head-slot collision** from section 3. [verified by sim]
- **B6. Possible loop when the server rejects the request for length.** If only the pin and the current turn remain and the server still rejects, the force-drain trim drops and restores the pin, so `trimmed == messages`. `_force_reduce_messages` then drops the pin and the next iteration restores it. With `LOCAL_ANALYZER_TOOL_STEPS=0` the loop is unbounded (TA:189, 3489, 6811-6826) until the game stops. Without Fable this raises and ends the turn. Rare. [inferred]
- **B7. Detector false positives in click games.** In r2, 99 of 283 patches (35%) on MOUSE actions contain the click point within 2 cells, meaning the clicked object rather than a far indicator; sb26 is 65 of 65. Only 62 of 725 patches (9%) lie within 6 cells of the edge (`patch_audit.py`). The text still says "Legends, trays and counters…" and "A mark you earned may have been lost" (for example dc22's 4×4 tiles). [verified]
- **B8. Heuristics are stated as facts.** Real outputs include "SPACE: white cells shifted down 9 and left 22 (1 of 2 presses checked)" and "DOWN: charcoal cells shifted up 4". The negative sentence "no action of yours has moved it" comes from a detector that rejects busy frames, more than 2 patches, anything within 8 cells of the action, and anything over 16 cells. This is the same "conditional result written as general" failure they diagnosed in their journal (p.84). [verified examples; risk inferred]
- **B9. The digest's "last actions before clearing it" drops MOUSE coordinates** (key_of, P:96-98). Real output: "MOUSE MOUSE LEFT LEFT LEFT MOUSE RIGHT MOUSE LEFT LEFT". This is nearly useless for click games. The clearing action is also logged as a no-op (P:91). [verified]
- **B10. Note action numbers don't match the step numbers in the opener.** Notes use "action N on this level", counted across attempts without RESET, while the opener shows global step numbers. [inferred]
- **B11. Design inconsistencies, not crashes.**
  - The 5 protocol lines and the ACK are advice plus a fake assistant turn, against their own rule (narrative cell 16).
  - "Restoring can never count as an eviction" (cell 30) is only true inside `_trim_messages_for_context`: `_persistent_history_messages` flags any change (TA:6348). It is also wrong in B4/B5, where the head changes with nothing dropped. [verified code]
- **B12. Cosmetic issues.**
  - `_fl_log` writes `/kaggle/working/fable_layer_log.jsonl` on by default, including the competition rerun.
  - A pin made on a resumed call lands mid-turn, between a tool result and the resume opener. [verified code]

## Caveats

- The replay is counterfactual. The trajectories come from a run without the layer, so it shows what text would have appeared, not its effect on play.
- Token gates are approximate. requests.csv aggregates tokens per analysis step, so the replay credits a step's tokens to its first analyze call and gates can fire up to one step early.

## Files

All under `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/fable-compare/`:
- `fable_layer_core.py`, `fable_layer_patch.py`: the extracted source (cells 41 and 42).
- `replay.py`, `replay_fz-ledger-on-trim-r2.json`: replay of the layer on r2 frames, with sample texts.
- `patch_audit.py`, `test_budget_attempts.py`, `sim_stackv2_conflict.py`
- `tool_agent_franzen_m2.py`: the pristine Franzen file (TA).
- `stackv2_fable/`: stack-v2 with the patch applied, scratch only.
- `cells/`: extracted notebook cells.
- `handbook.txt`: text of the handbook PDF.
