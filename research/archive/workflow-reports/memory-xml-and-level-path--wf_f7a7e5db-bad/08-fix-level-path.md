# memory-xml-and-level-path: fix:level-path

Workflow run `wf_f7a7e5db-bad`, phase Fix. Research, design, build, review and integrate two arms on memory-v2: memory writes in the Qwen3 XML tool-call format with a memory addendum and revision log, and a shortest-known-path tracker shown after every action

---

All seven findings have been dealt with in 8 new commits on `arm/level-path`: six fixed, and finding 5 kept as it was with its docs clarified. The targeted tests pass, run one file at a time, niced, with imports loading from the worktree. Nothing is pushed and the working tree is clean.

**Findings**
1. **Fixed** (`4d355e3`). Reproduced first: an error in `_route_step` on the clearing action left the clear event unwritten, and the clear was lost. All three route calls now go through `_route_guarded`, which counts the error in `stats["route_errors"]` and drops that level's graph until the next level. While the graph is dropped, `route_state()` returns `{"path": None, "len": None, "varied": []}`, so the sandbox's `level_path` does not keep an out-of-date value. A new test (3 cases: `_route_step`, `_route_reset`, the clear's route lookup) checks that events and state match a recorder with routes off. It fails on the old code.
2. **Fixed** (`0965999`). Reproduced: after R R U, a manual RESET and D, the line read "since the last message: 0 1". When an earlier attempt also acted since the last message, the label now reads "of attempt N"; otherwise the line is byte-identical. In the death-then-move case the longer line goes over 300 characters, so the existing cap drops the list (tested).
3. **Fixed** (`3b878d2`). The system-prompt line now says "after actions on the current level", gives the per-game cap, and names the hold when `ARC3_MEM_PATH_AFTER_TOKENS` is above 0. Both knobs are read when the prompt is built, so it stays the same for the run. `memory_addendum` gains `path_chars` and `path_after`.
4. **Fixed** (`54bdeed`). The arrow gets spaces when either side is `-`. Other heads such as `0->3` are unchanged.
5. **Not changed in code; docs clarified** (`a813f50`). "known board" is true: the board was already on record. The case never happens in the recorded runs: 0 of 5,654 r2 rows and 0 of 484 fz-experts8 rows. A comment and CONFIGURATION.md now say that "known board" covers it.
6. **Fixed** (`a813f50`). The docstring and CONFIGURATION.md now say that "a distance never grows" only holds while the start board stays the same.
7. **Fixed.**
   - 7a (`b69ce76`): a `StartsRecorder` notes the start board when each level begins and after every RESET. A test shows it counts a start that moved before any link. The replay also asserts `route_errors == 0`.
   - 7b (`9a5a151`): `eviction_sim` now counts route turn lines and route echoes, which appear escaped inside the tool-result JSON. The test requires at least 10 of each with the flag on (mem-path 29/19, path-only 28/23) and 0 with it off.

`a438802` updates the README paragraph to match.

The r2 replay gives the same costs as before: echo suffix mean 24.2 characters (max 56), turn line on 88.6% of turn messages (mean 219.4, max 299), 3 games at the cap. No r2 line used the new "of attempt N" label or the spaced arrow. One niced process, about 3 s.

**git log arm/memory-v2..HEAD**
```
a438802 Bring the level-path README paragraph in line with the prompt
9a5a151 Assert route text in the path eviction configs
b69ce76 Record every start board in the r2 replay
a813f50 Correct the route docs on growing distances and known boards
54bdeed Space the batch route arrow when a side has no route
3b878d2 State the route line's per-game cap in the system prompt
0965999 Name the attempt when the route list leaves earlier rows out
4d355e3 Guard the route hooks inside FactRecorder
4162a36 Resolve the route log path inside a guard
e1dee63 Document ARC3_MEM_PATH and add the level-path arm
997249a Anchor the route graph to the r2 run's frames
954025b Cover ARC3_MEM_PATH in the identity and eviction tests
a1c4a77 Describe route facts in the system prompt
2806ef3 Write per-action route rows to level_path.jsonl
f10eeb4 Use the shortest route on record in the solved line
8e2bd80 Expose level_path in the sandbox and name it in the tool text
905285a Add the Shortest route line to the turn message
7221086 Add the route suffix to the action echo
1f9c4ac Feed the recorder after each action call and buffer its events
4f28f1d Feed LevelGraph from FactRecorder steps and resets
8aba531 Add LevelGraph with move links and BFS routes
```

**Targeted pytest**
```
test_level_path 42 passed (r2 replay ran, not skipped); test_memory_wording 7 passed; test_memory_flags_off_identity 8 passed; test_memory_eviction 12 passed; test_memory_placement 20 passed; test_ledger_on_trim 26 passed; test_memory_facts 18 passed; sandbox_deadline + retained_functions + tool_timeout_result 26 passed
```

**Model-visible text after the fixes**
- System prompt line (defaults):
  `- Route facts come from links the harness records on each level: a board-changing action links one board to the next (boards compared with the outer 4 cells cropped; UNDO, RESET, GAME OVER, clears and no-change actions add no link; a click is its exact cell). The route is the fewest links from this level's start board to the current board. Each action(...) call prints route N and what changed; a turn message after actions on the current level carries a Shortest route on record line, up to 16000 characters of these lines per game. In python, level_path is that route as a list action(...) accepts and level_path_len its length.`
- With a hold set, the same line reads: `... carries a Shortest route on record line once the level has used 5000 generated tokens, up to 8000 characters of these lines per game. ...`
- Turn line after a manual RESET in the same turn:
  `Shortest route on record from this level's start board to the current board: 1 action [D]; this attempt so far: 1 action. Route length after each action of attempt 2 since the last message: 0 1.`
- Turn line after a death and then a move, with a 400-character limit:
  `... 1 action [D]; this attempt so far: 1 action. Attempt 1 ended in GAME OVER after 2 actions; its last board before GAME OVER is 1 action from the start on record [R]. Route length after each action of attempt 2 since the last message: 1.`
- At the default 300-character limit, that line ends after `[R].`
- Batch echo heads: ` | route - -> - (- -)`, ` | route - -> 1 (0 1)`, and ` | route 1->X (- X)` (unchanged).
- Unchanged and still tested: the spec's sequence line ending `since the last message: 0 1. A shorter route ... (3 -> 1).`, ` | route 0->3 (1 2 1 2 3)`, the death line, and the solved line.

**Merge notes for integrate**
- `memory_addendum` now takes `path_chars` and `path_after`. The memory-v3 merge must pass them through from `_mem_path_game_chars()` and `_mem_path_after_tokens()`.
- `tests/test_memory_wording.py` changed by one line (it adds `memory_path_line(4, 8000, 5000)`). It still conflicts with memory-xml's copy of that file.

**PR summary (draft)**
- **Hypothesis:** the model re-walks boards it has already reached (27% of r2's actions are on net-zero loops, 9.0% exact repeats). Showing the shortest route on record after every action should cut that waste.
- **Change:** `ARC3_MEM_PATH` (off by default) keeps a per-level `LevelGraph` of the board-to-board links that observed actions made, and searches it (BFS) from the attempt's start board.
- **What the model sees:**
  - a route suffix on every `action(...)` echo;
  - a `Shortest route on record` line in turn messages, up to 16000 characters per game, stated in the system prompt;
  - `level_path` and `level_path_len` in python;
  - a shorter solved route when one is on record.
- **Safety:** route hooks are guarded. A route error drops only that level's graph and leaves the memory-v2 facts unchanged (tested).
- **#58:** text appears only in the system prompt or tool schema, the tool result being built, and the new turn message. The `mem-path` and `path-only` eviction configs pass all four checks and must now emit route text.
- **Identity:** with the flag off, requests are byte-identical to `arm/memory-v2` at d252fa3 (golden fixture: system prompt, 31 requests, `tools_sha`).
- **Offline (r2, 25 games):** the route beats the winning attempt at 2 of 112 clears, and is shorter than the attempt's own loop-free path after 2.2% of actions.
- **Cost:** echo suffix about 7 tokens per call; turn line about 60 tokens on 89% of turn messages. This probably exceeds the +60-token budget per executing turn.
- **Status:** not yet run on Kaggle. Arm `level-path` = `memory-v2-both` + `ARC3_MEM_PATH=1`. The keep/drop rule was fixed before any run (CONFIGURATION.md §10).

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/`:
- `wt/level-path/ARC3-Inference/inference/agent/memory_facts.py`
- `wt/level-path/ARC3-Inference/inference/agent/prompts.py`
- `wt/level-path/ARC3-Inference/inference/agent/tool_agent.py`
- `wt/level-path/ARC3-Inference/tests/test_level_path.py`
- `wt/level-path/ARC3-Inference/tests/eviction_sim.py`
- `wt/level-path/ARC3-Inference/tests/test_memory_eviction.py`
- `lp-fix/replay_cost2.py` (the r2 cost replay)
