# memory-xml-and-level-path: build:level-path

Workflow run `wf_f7a7e5db-bad`, phase Build. Research, design, build, review and integrate two arms on memory-v2: memory writes in the Qwen3 XML tool-call format with a memory addendum and revision log, and a shortest-known-path tracker shown after every action

---

All 11 planned commits are on `arm/level-path`, plus two small extras (a replay test on recorded run data and a try/except fix). The targeted tests pass, and a replay of the recorded r2 run (`fz-ledger-on-trim-r2`) gives the numbers the spec expects. Nothing is pushed and no PR is open (integrate does that). It has not run on Kaggle.

Worktree: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/level-path`. The branch starts from `arm/memory-v2` at cf40ac1, which is d252fa3 plus one `arms.json`-only commit. The new golden fixture was generated on d252fa3, as the spec says.

**Commits (`git log arm/memory-v2..HEAD`, newest first)**
```
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

**Tests (each file run alone, niced; imports confirmed to load from the worktree)**
```
test_level_path: 35 passed in 4.54s
test_memory_wording: 7 passed in 0.39s
test_memory_flags_off_identity: 8 passed in 2.10s
test_memory_eviction: 12 passed in 3.89s
test_memory_placement: 20 passed in 2.81s
test_ledger_on_trim: 26 passed in 0.13s
test_memory_facts: 18 passed in 2.85s
test_sandbox_deadline + test_retained_functions_survive_failures + test_tool_timeout_result: 26 passed
```
- **Identity:** with the path flags off, `kaggle-memv2-both` and `kaggle-memv2-trim` match the new golden (`tests/fixtures/memory_v2_on_golden.json`, generated on d252fa3). That covers the system prompt, all 31 requests and a single `tools_sha`. With the flag on, every request differs, so the test is sensitive.
- **#58 guard:** the new configs `mem-path` and `path-only` pass all four checks, and the three trap configs still fail as they should.
- **Rollback:** I removed the cursor restore by hand and the rollback test failed, so it does catch a missing restore.

**What the model sees**
- **System prompt**, placed after the `Seen before` line, or on its own under "Memory kept by the harness:" when memory-v2 is off:
  `- Route facts come from links the harness records on each level: a board-changing action links one board to the next (boards compared with the outer 4 cells cropped; UNDO, RESET, GAME OVER, clears and no-change actions add no link; a click is its exact cell). The route is the fewest links from this level's start board to the current board. Each action(...) call prints route N and what changed; a turn message after actions carries a Shortest route on record line. In python, level_path is that route as a list action(...) accepts and level_path_len its length.`
- **Tool description and turn message:** the tool description gains the spec's `level_path` sentence. The `Only tool:` line now reads `` ... `frame_diff(before, after)`, `level_path`, `level_path_len`, and `action(actions)`. ``
- **Tool result (action echo)**, real lines from the r2 replay, plus the spec's worked example run through the real sandbox:
  ```
  [action] batch of 5 actions, 5 executed: RIGHT, RIGHT, LEFT, UP, UP | route 0->3 (1 2 1 2 3)
  [action] LEFT -> effect | route 14 (shorter than 23 on record)
  [action] LEFT -> GAME OVER (attempt ended immediately after this action) | route - (GAME OVER from a board at route 1)
  [action] UP -> LEVEL COMPLETED | route - (level cleared; 13 on record)
  [action] batch of 5 actions, 5 executed: DOWN, DOWN, DOWN, DOWN, DOWN | route 40->C (41 42 43 44 C)
  ```
  With the action echo off, the suffix prints as its own line: `[route] route 1 (new board)`.
- **Turn message line**, placed after `Seen before` and before `<memory_update>`:
  ```
  Shortest route on record from this level's start board to the current board: 1 action [D]; this attempt so far: 1 action. Route length after each action since the last message: 0 1. A shorter route to a known board was found at action #1 of this attempt (3 -> 1).
  Shortest route on record from this level's start board to the current board: 0 actions (this is the start board). Attempt 1 ended in GAME OVER after 2 actions; its last board before GAME OVER is 1 action from the start on record [R].
  Shortest route on record from this level's start board to the current board: 12 actions [M(19,52) D R M(28,46) M(28,56)x2 M(28,46)x4 M(28,56)x2]; this attempt so far: 31 actions. 2 steps on the route had more than one outcome on this level.
  ```
- **Card and delta lines:** only the solved line changes, and only when the recorded route is strictly shorter than the winning attempt: `<solved>L1: 2 acts [D R] (4 in the winning attempt), spent 6, 0 deaths</solved>`. Otherwise it is byte-identical to memory-v2.
- **Python:** `level_path == ["DOWN"]`, or `[{"action": "MOUSE", "row": 4, "col": 7}]` for a click. It is `[]` with length 0 on the start board, and `None` when no recorded link reaches the board. `level_path_varied == [1]` in the spec's varied case.
- **Export:** one line per action in `*_level_path.jsonl` with the fields `{act, level, attempt, i, sig, board, d, k, change, varied, path}`.

**Validation (r2 run, 25 games, 5,630 actions, one niced process)**
- **Route beats the winning attempt** at exactly 2 of 112 clears: g50t level 1 (16 vs 17 actions) and sc25 level 3 (19 vs 25).
- **Shorter than the attempt's loop-free path** after 120 of 5,495 actions with a route, which is 2.2%.
- **Graph caps and start boards:** no cap was reached and no level had two start boards.
- **CPU:** 3 s for the whole replay. The searches take 32 ms in total, and recorder time goes from 1.63 s to 1.73 s.
- **Echo cost:** the suffix appears on 4,534 action calls, mean 24 characters, max 56.
- **Turn line cost:** it appears on 89% of turn messages, mean 219 characters (about 60 tokens), max 299. The median game uses 6.7k characters. The second run with frames on disk, fz-experts8 (10 games), shows the turn line on 91% of turn messages, mean 211 characters.

**Limitations and choices where the spec was silent or reality differed**
- **Token budget:** the turn line alone uses the whole +60-token budget per executing turn. Each `action(...)` call adds about 7 tokens on top, so executing turns will probably go over budget.
- **Game cap:** 3 of the 25 r2 games hit the 16,000-character cap, so their late turns get no route line.
- **Repetition:** the line is repeated every turn even when the route has not changed.
- **Cursor:** when the line is held or over the cap, the cursor still advances, so the list keeps meaning "since the last message".
- **Strings I chose:** the GAME OVER sentence on the start board says `is the start board`, and with no route it says `has no route on record`. A suffix with no route reads `GAME OVER from a board with no route on record` or just `level cleared`. Batches too long for 80 characters fall back to showing 3 then 2 values from each end, then `N actions`. The GAME OVER sentence sits between the route sentence and the per-action list.
- **New board:** this means not yet seen on this level, including boards reached by UNDO. So a new board can have route `-`.
- **Search timing:** the search reruns only when a link or the start board changes. The results are the same as rerunning after every action.
- **Solved-line wording:** the prompt line says "the shortest route on record", but when the recorded route is not shorter the line uses the winning attempt's own path. That is usually the same length; it can differ when the attempt used UNDO.
- **Simulation only:** `memory_sim` now mimics the mid-snippet feed and the echo, and prints the route as the first line of stdout.
- **Wording test:** `test_memory_wording.py` reuses the `Walk` helper from `test_level_path.py`. The memory-xml arm creates the same file, so merging the two will conflict there.

Docs are updated in `CONFIGURATION.md` §10, the branch `README.md` and `kaggle/README.md`. The `level-path` arm in `kaggle/arms.json` is `memory-v2-both` plus `ARC3_MEM_PATH=1`, and its notebook builds locally. The `arms.md` entry is under "In progress"; the memory-xml agent's concurrent entry sits next to it and both survived.

Main files:
- `.../level-path/ARC3-Inference/inference/agent/memory_facts.py`
- `.../level-path/ARC3-Inference/inference/agent/tool_agent.py`
- `.../level-path/ARC3-Inference/tests/test_level_path.py`
- `.../level-path/ARC3-Inference/tests/fixtures/memory_v2_on_golden.json`
- `.../scratchpad/level-path-measure/replay_cost.py` (the cost replay script)
