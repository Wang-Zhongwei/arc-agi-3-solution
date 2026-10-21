# memory-xml-and-level-path: review:level-path

Workflow run `wf_f7a7e5db-bad`, phase Review. Research, design, build, review and integrate two arms on memory-v2: memory writes in the Qwen3 XML tool-call format with a memory addendum and revision log, and a shortest-known-path tracker shown after every action

---

**Verdict:** arm/level-path matches the spec and nothing blocks it. There is one medium-low robustness defect and several low ones. I changed nothing in the worktree.

**Checks I re-ran myself (niced, one at a time; imports load from the level-path worktree)**
- **Tests:** test_level_path 35 passed (the r2 replay ran, not skipped); test_memory_flags_off_identity 8 passed; test_memory_eviction 12 passed; test_memory_wording 7 passed.
- **Golden fixture:** I exported d252fa3 with `git archive` into `scratchpad/review-lp/d252` and ran this branch's `memory_sim.py` against it. For kaggle-memv2-both and kaggle-memv2-trim, the system prompt, all 31 requests, trims, ledger render and `tools_sha` all match `tests/fixtures/memory_v2_on_golden.json`.
- **Flags off:** I read each change with the flags off and each one is either guarded or produces the same result, so requests are byte-identical. The `_only_tool_line()` output and the clear-event dict key order are unchanged. The sandbox and file-export paths with the flag off are covered by `test_flag_off_*`.
- **#58:** model-visible text appears only in the four allowed places: the system prompt and tool description (static per run), the echo inside the tool result being built (`tool_agent.py:6356-6362`), and the new turn message (`_mem_turn_lines`, `:4573`). The head card is unchanged. Resumed turns get no line. The rollback snapshot carries the cursor and chars used (`:7104`, `:7722`). The `mem-path` and `path-only` simulations do emit route text: 29 turn lines and 19–25 echo lines.
- **BFS on real levels:** I rebuilt the move links and ran my own BFS over r2 (137 levels), v4-stack-v1 (128) and fz-experts8 (25). Distances matched `LevelGraph.dist` on every level, every `route()` follows recorded links back to the start, and no level had two start boards. Steps with more than one outcome show up where expected (dc22 L4, g50t L1/L4, sk48 L1 and others).
- **CPU:**
  - Real runs, fed one entry at a time with the suffix and `route_state` computed each time: at most 0.5 ms per action.
  - Synthetic 1024-board grid walk, 6,000 actions: 76 µs mean, 0.38 ms worst, no cap hit.
  - Synthetic graph filled to the caps (1024 boards × 16 clicks × 4 outcomes, about 54k links): 8.6 ms worst per new link, about 60 s in total. A game can't reach this, because links never outnumber executed actions.
- **Edge cases checked by hand:** UNDO, RESET, death, automatic reset, MOUSE(4,7), batches (including one starting with RESET), level change, nondeterminism, and 300-character long lines all behave as specified. Batches stop at a level clear (`solver.py:1124`), so one echo can't span two levels.

**Defects, most severe first**

1. **Medium-low: an error in the route code wipes out memory-v2's facts.** This breaks the spec's rule that every new hook is wrapped in try/except.
   - **Where:** `memory_facts.py:1051` calls `self._route_step(...)` and `:1070` calls `self.graph.route(pre_h)`, both inside `FactRecorder.step()` and before the death and clear events are appended. `:1009` calls `_route_reset` the same way. None of them is guarded.
   - **What happens:** I made `_route_step` raise on the clearing action. The route-off recorder emitted `['first','clear','level']` and moved to level 2. The route-on recorder emitted `['first']` and stayed on level 1 with the current board stale. `feed()` has already moved its cursor past the entry, so the clear is never re-processed and the `<solved>` line is never written.
   - **Fix:** wrap all three calls in `try/except Exception: self.stats["route_errors"] += 1`, optionally dropping the graph until the next level. Then the memory-v2 path runs unchanged. Add a test that monkeypatches `_route_step` to raise and asserts the events equal those of a `route=False` recorder.

2. **Low: the per-action list misstates its scope after a manual RESET.** In a turn with several calls, an R/R/U snippet, then a RESET and D in a later snippet, gives `...since the last message: 0 1.` The three actions before the RESET are silently left out and nothing mentions attempt 1. The cause is the `shown = [... row.attempt == attempt ...]` filter in `route_line`. Fix: label it `after each action of this attempt since the last message`, or list every non-automatic row (the RESET already shows as 0). The death case can keep its spec wording.

3. **Low: the system prompt promises a line the harness may stop sending.** `prompts.py:87` says "a turn message after actions carries a Shortest route on record line". That stops being true once `ARC3_MEM_PATH_GAME_CHARS` (16k) runs out, which happened in 3 of 25 r2 games, or while the line is held. Fix: add "(up to 16000 chars per game)" to the prompt line, or send one factual "route lines end here" line in the turn message where the cap first blocks.

4. **Low, cosmetic: hard-to-read batch head when the starting board has no route.** `memory_facts.py:1362` renders ` | route -->- (- -)`. Fix: put spaces around the arrow when either side is `-`, or use `none->-`.

5. **Low, wording versus spec:** the spec defines "known board" as `d == d_before`. A known board that gets its first route (`before` is None, `d` is now a number, for example a board first reached by UNDO and later linked by a move) is also printed as `(known board)`. That is true, but the spec's change kinds don't cover it.

6. **Low, docs:** the `LevelGraph` docstring (`memory_facts.py:253`) and CONFIGURATION.md say "a distance never grows". That is false when a RESET lands on a different board and `set_start` moves the BFS source. The data has no such level, so this is only a wording fix.

7. **Low, tests:**
   - The replay's start-board detector (`test_level_path.py:739`) records a start only when a BFS runs, so it would miss a start that changes before any link exists. My independent scan confirms the claim anyway.
   - `test_only_real_drops_hand_the_slot_over` never asserts that route text appeared, so the `mem-path` and `path-only` configs could pass vacuously. Add `assert` counts of `Shortest route` and `| route` lines.

**Cost note (not a code defect):** the turn line is about 60 tokens on 89% of turns, plus about 7 tokens per `action()` call. The per-action list repeats the echo values the model already saw in that turn's tool results. Executing turns will probably exceed the spec's +60-token budget. Dropping the list would recover roughly 20–40 characters per turn, but it is in the spec.

**Merge notes for integrate:** `tests/test_memory_wording.py` imports `Walk` from `test_level_path.py` and will conflict with memory-xml's copy of that file. `memory_sim.MEMORY_V3_FLAGS` needs memory-xml's flags added.

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/`:
- `wt/level-path/ARC3-Inference/inference/agent/memory_facts.py`
- `wt/level-path/ARC3-Inference/inference/agent/tool_agent.py`
- `wt/level-path/ARC3-Inference/inference/agent/prompts.py`
- `wt/level-path/ARC3-Inference/tests/test_level_path.py`
- `review-lp/probe_real.py` (my independent BFS check)
- `review-lp/probe_edge.py` (edge cases and CPU)
