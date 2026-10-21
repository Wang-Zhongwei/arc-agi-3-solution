# memory-xml-and-level-path: research:shortest-path

Workflow run `wf_f7a7e5db-bad`, phase Research. Research, design, build, review and integrate two arms on memory-v2: memory writes in the Qwen3 XML tool-call format with a memory addendum and revision log, and a shortest-known-path tracker shown after every action

---

Research B (shortest known path) — fz-ledger-on-trim-r2 replay (25 games, 5,630 actions), checked against a second 25-game run, v4-stack-v1 (5,732 actions)

**Bottom line**
- The path is cheap to compute (about 8 µs per action) and the graphs are small. SeenStates already holds the graph and never evicted an edge in either run.
- Measured headroom is small:
  - After a death, re-walks back to the frontier are already almost optimal: 2 extra actions in total in r2 and 1 in v4.
  - At level clears, the shortest-path route beats memory-v2's `loop_free` path in only 2 of 112 clears (r2) and 1 of 103 (v4).
  - The large gap is between the actions spent in an attempt and the shortest route to the current board (median 6, p90 39). That gap is loops and exploration inside an attempt. A display can make it visible but cannot give those actions back.
- Expect value from making loops visible, from the `shortest_path` replay variable and from the route line after a death, not from finding new shortcuts.

**1. What FactRecorder keeps (arm/memory-v2, `inference/agent/memory_facts.py`)**
- **Board identity:** sha1 of `repr(crop(grid, border))`. The border is `ARC3_MEM_STATE_BORDER`, which defaults to `ARC3_NOOP_GUARD_BORDER`=4. `_core` reuses the previous action's post-board hash.
- **Feeding:** `feed(entries)` walks the solver's full history with a cursor. Each batch member is its own HistoryEntry, so every executed action is fed. It runs only in `_mem_observe` (`tool_agent.py:5371`), once per new turn message, and only if `_mem_facts_on()` (`tool_agent.py:168`, which is SOLVED, EFFECTS or SEEN). Nothing is fed per `action()` call inside a snippet.
- **AttemptTracker:** holds only the current attempt: `steps` (`Step(sig, pre_h, post_h, cls)`), `start_h`, `spent`, `deaths` and `first_index`. Steps are cleared on restart, so earlier attempts' routes are not kept as steps.
- **SeenStates:** per level, `boards[h].tried[sig] = (kind, next_board, attempt)`. `kind` is move, noop, death or clear. It keeps only the latest outcome and is capped by LRU at 16 actions per board and 512 boards, and it is reset on a level change. This is already an observed-transition graph. Measured: at most 11 distinct actions per board and 219 boards per level, so 0 edges were evicted in both runs.
- **RESET:** `reset_step` closes the attempt. An automatic reset after a death has `result["automatic"]` set (`solver.py:1228`/`1331`) and is not counted in `spent`; a manual RESET adds 1 to `spent`. A level change starts a new level.
- **Death:** `current_h` is set to the death frame, and the next entry is the RESET.
- **Clear:** records `loop_free(steps)` and starts the next level from the post-board.
- **MOUSE:** `action_signature` gives `MOUSE(r,c)`, so a click is identified by its exact cell. The effect key `MOUSE@<colour>` is used only for counts.
- **UNDO:** not special-cased; it is stored as an ordinary move. It depends on history, so a BFS must skip it.
- **No-ops:** noop, hud and anim (STILL) are kind "noop" and are not edges.

**2. Where the model sees an action's result, and where a path line can go without touching sent history**
- **The `[action]` echo**, in the tool result being created now:
  - `solver.py:210 _build_action_echo` writes one line per `action()` call: `[action] UP -> effect` or `[action] batch of N actions, M executed: ... | stop_reason: ...`.
  - It is attached at `solver.py:1194`, passed through `_compact_action_result` (`tool_agent.py:5965`) and printed into the snippet's stdout by the sandbox's `action()` wrapper (`python_tool_sandbox.py:654`).
  - `ARC3_ACTION_ECHO=1` is on in the competition template.
  - Hook: in `_handle_action` (`tool_agent.py:6047`), after `compact_payload = ...` at 6146, append a suffix to `compact_payload["action_echo"]`.
  - Risk: long stdout is middle-truncated (`tool_agent.py:6228`), so echo lines in long loops can be lost.
- **The sandbox state:**
  - `_serialized_runtime_state` (`tool_agent.py:6000`) already reloads the full history after every call; feed the tracker from that same history.
  - Add a key; `_refresh_state` (`python_tool_sandbox.py:490`) sets globals. `death_ledger` at :517 is the existing pattern for a `shortest_path` global.
- **The new turn message:**
  - `_build_user_prompt` (5071) prints "The code executed N actions..." and "Executed actions: (first 10)". After the valid-actions line it calls `_mem_observe` and `_mem_turn_lines` (5371-5372), where the Seen-before line goes. Put the route line there.
  - `_mem_turn_lines` returns nothing on a resumed turn, which is correct for #58.
  - The 15k hold (`_mem_held`) should not apply to this line; give it its own knob, default 0.
- **Integration trap:**
  - Problem: if the echo or variable needs a path in the middle of a snippet, the recorder has to be fed in `_handle_action`.
  - Fix: gate that early feeding on the new flag, buffer the events, and apply them in `_mem_observe`.
  - Guard: add the new flag to `_mem_facts_on`, or the recorder stays empty (the death-ledger lesson).
- **The "Only tool: `python`. It receives ..." line (5375):** this is the only text that names the variables. Change it only when the flag is on.

**3. Replay method**
- **Data:** `artifacts/*_events.jsonl` has one board per executed action, batch members included.
- **Hashing:** the same as the harness (border 4).
- **Edges:** (pre, exact signature including MOUSE coordinates) → post, for moves only. Death, clear, noop, hud and anim are not edges, and UNDO edges are excluded (r2 has 1; v4 has 9).
- **RESET:** a RESET right after a GAME_OVER is treated as automatic. Levels with more than one start board after a reset: 0, so a BFS from the attempt start is the same as from the level start.
- **Distances at each action:** after adding this action's edge, `d` is the BFS distance from the start to the current board. `k` is the actions in the current attempt (including no-ops), `km` is the board-changing actions, and `lf` is `loop_free` (`memory_facts`).
- **Pairs with different outcomes:** the BFS uses every observed outcome; a BFS over consistent edges only is computed for comparison.

**Results (r2 / v4)**
- **Shorter route than the attempt's actions** (`d < k`): 67.5% / 69.2% of actions. Actions saved when shorter: median 6 / 9, p90 39 / 57, max 89 / 118.
  - Attempt 1: 71% / 73%. Attempt 2 and later: 46% / 44%.
- **Shorter than board-changing actions** (`d < km`): 48.8% / 55.3% (median 8 / 12, p90 42 / 49).
- **Shorter than the loop-free walk** (`d < lf`, needing another attempt's edges or a better edge): only 2.2% / 2.1% (median 2 / 4, p90 6 / 18, max 10 / 26).
- **How the shortest path changes after an action** (current-board rows):

| Change | r2 | v4 |
|---|---|---|
| Old path + this action (+1) | 87.5% | 88.7% |
| No board change | 6.0% | 5.9% |
| Back to a known board nearer the start | 6.4% | 5.0% |
| Other re-route | 0.1% | 0.4% |

  - Most of the "back" cases are immediate reversals (263 of 349 in r2 dropped exactly 1 below the previous length).
  - A shorter route to an already-known board was found by an action only 9 / 16 times (0.2-0.3% of moves).
  - `action()` calls containing any change other than +1: 330 of 4,533 (7.3%) / 285 of 4,903 (5.8%).
- **Path length at the current board:** median 15, p90 48 / 49, max 96 / 122.
  - Run-length path in chars: median 26 / 31, p90 89 / 116, p95 126 / 160, max 343 / 377. Cap it with `_fit_path`.
- **Graph sizes:** 137 / 128 levels. Nodes per level median 25, p90 69 / 70, max 219 (wa30 L3) / 212. Edges per level median 25, p90 75, max 222 / 212.
- **Outcomes that differ for the same board and action:**
  - Repeated (board, action) pairs: 504 / 570.
  - Same outcome as last time: 92.9% / 95.8%.
  - Different outcomes in r2: board → different board 28, board → death 6, death → board 2. v4: 20 board → board.
  - Concentrated in g50t (17) and dc22 (9) in r2, and dc22 and ls20 (9 each) in v4. Likely hidden state, or the bar inside the cropped border.
  - Distinct pairs with more than one outcome: 30 of 5,126 / 21 of 5,162.
  - A consistent-edges-only BFS gives a different `d` in 2.5% / 3.8% of rows, and at 4 / 3 clears.
- **CPU per BFS:** mean 8.1 / 7.0 µs, p50 5 µs, p99 52 / 34 µs, max 320 / 45 µs. The whole replay (parsing 144 MB of JSON plus 3 BFS per action) took 3.2 s of CPU.
- **Re-walks after a death** (graph frozen at the attempt start; the frontier is the last known board before the first new one):
  - r2: 23 deaths and 1 manual RESET. 14 of the 23 re-walked boards known before the attempt. 2 were longer than the known shortest route, by 1 action each (2 in total).
  - v4: 21 deaths, 16 re-walks, 1 longer by 1 action.
  - Counting any later return to a known board after exploring new boards: 6 / 5 attempts, excess median 4 / 33, max 19 / 67, sum 41 / 167. That is exploration returning, not clearly waste.
  - In r2 a re-walk takes 1 turn (max 2) and a median 3.5k reasoning chars, below the median executing turn of 4.1k. Re-walks are already cheap.
- **Level clears:**
  - The shortest known route to the winning board is shorter than the winning attempt in 72 of 112 / 64 of 103 clears: median 5, p90 27 / 23, max 71 / 66 actions saved.
  - But `loop_free` already captures that (lf < k in 71 / 64 cases). The BFS beats `loop_free` only for g50t L1 (16 vs 17, through a pair with differing outcomes; no consistent-only path), sc25 L3 (19 vs 25) and v4 dc22 L2 (42 vs 46).
  - Level spend minus shortest route: median 4 / 3, p90 39 / 23, max 200 / 133.

**4. Recommended display (facts only, capped, all behind one default-off flag)**
- **Graph:** reuse `SeenStates.tried` with move edges only; skip sig "UNDO" and kind death, noop and clear. Add a small `varied` set of (board, sig) pairs whose outcome changed, and BFS from `attempts.start_h`.
- **Turn message:** one line per executing turn, at most 240 chars, after the Seen-before line:
  `Route on record from this level's start to this board: 9 actions [R2 U4 L3]; this attempt has used 15. After each action this turn: 7 8 9 8 9.`
  - The path is cut by `_fit_path` at 64 chars with `...+N`.
  - The per-action list holds at most 12 values; beyond that, the first 5, "...", the last 5 and "(N actions)". Actions per executing turn: median 3, p90 12, p95 17, max 57 / 127.
  - If the path uses a pair with differing outcomes, add: "(1 step on it had more than one outcome on this level)".
  - First turn after a death: `Route ...: 0 actions (start board). Attempt 2 ended after 23 actions; its last board before GAME OVER is 21 actions from the start on record [..].`
- **Echo suffix** on the existing `[action]` line, at most 24 chars and no new lines: ` | route 9 (was 8)`, or for a batch ` | route 6->9`.
  - `action()` calls per executing turn: median 2, p90 11, max 57 / 127. Actions per call: median 1, p90 1, so this is effectively per action.
- **Python variable:** `shortest_path`, a list in `action()` format (`"RIGHT"`, `{"action":"MOUSE","row":r,"col":c}`), empty at the start board, refreshed after each `action()` call.
- **Revision store:** a per-level log of (action_num, board id, `d`, `k`), capped at about 512, for the memory addendum.
- **Cost:** about 50-60 tokens per turn × about 1,100 executing turns ≈ 60k prompt tokens per 25-game run (cache-friendly), plus about 25k for the echo suffixes. CPU is negligible.

Scripts and per-action rows are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/memv3-research/`:
- `sp_replay.py`
- `sp_analyze.py`
- `sp_more.py`
- `sp_fanout.py`
- `sp_rewalk_tokens.py`
- `sp_rows_fz-ledger-on-trim-r2.json`
- `sp_rows_v4-stack-v1.json` (rows for fz-experts8 and the three v4-panel runs are also saved)
