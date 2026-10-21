# memory-xml-and-level-path: design

Workflow run `wf_f7a7e5db-bad`, phase Design. Research, design, build, review and integrate two arms on memory-v2: memory writes in the Qwen3 XML tool-call format with a memory addendum and revision log, and a shortest-known-path tracker shown after every action

---

# SPEC memory-v3 = arm/memory-xml + arm/level-path, both on arm/memory-v2 @ d252fa3
The research confirms the interpretation:
- **qwen3_xml.** No Kaggle run used `qwen3_xml`; all 54 server logs say `qwen3_coder`, which reads the same `<parameter=K>` wire format. So "encoding memory for qwen3_xml" means a `memory` parameter on the `python` tool.
- **Shortest path.** "Shortest path to the current board" means a breadth-first search (BFS) over this level's observed move links, from the level's start board to the current board.

## 0. Common rules (both arms)
- **Branches.** `arm/memory-xml` and `arm/level-path` branch from `arm/memory-v2` @ d252fa3 (PR base `arm/memory-v2`). `arm/memory-v3` is `arm/memory-v2`, plus `git merge --no-ff` of both arms, plus the §3 commits. Paths are under `ARC3-Inference/`; `ta` = `inference/agent/tool_agent.py`.
- **Flags.** `ARC3_MEM_XML` and `ARC3_MEM_PATH` are off by default and read at call time. With a flag off, its code is never entered: no schema key, addendum text, tool-result key, echo suffix, sandbox global or artifact file.
  - `_mem_v2_on()` gains `_mem_xml()`, because XML needs the ledger and the card.
  - `_mem_path()` must not join `_mem_v2_on()`, or `_apply_due_ledger` would switch `render()` to `render_card()`.
  - Instead, add `_mem_recorders_on() = _mem_facts_on() or _mem_path()` and gate `_mem_observe` on it. The ledger may be None there; then only the recorder is fed and no events are applied.
- **Where model-visible text may go (#58), and nowhere else:**
  - (a) the system prompt and tool schema, built from flags only and static for the run;
  - (b) the tool result being built now: the `_dispatch_tool` return value and the `[action]` echo inside it;
  - (c) the new turn message, through `_mem_turn_lines`;
  - (d) the head card, through `_apply_due_ledger` after a real trim.
- **Safety rules for every hook:**
  - Stored assistant `tool_calls`, including their `memory` argument, are never edited.
  - Every new hook is wrapped in try/except and returns its input unchanged on any error.
  - Resumed turns get no route line.
  - The rollback snapshot (`ta` ~6898 save, ~7509 restore) also carries `_route_cursor` and `_route_chars_used`.
- **Wording.** Facts only. A shared test (`tests/test_memory_wording.py`) checks every harness-made string in §1 and §2 against `\b(inert|useless|impossible|pointless|waste|should|must|avoid|try|better|best|optimal|don't|instead)\b`, case-insensitive. The model's own TEXT is excluded. The harness never sets an item's status; its only own operation is a cap drop.
- **Laptop.** Run only the targeted test files, niced, one at a time (see the brief). The replay test runs in one process, under 60 s of CPU.

## 1. ARM arm/memory-xml (`ARC3_MEM_XML`)
### (i) Encoding: an optional `memory` parameter on `python`
- **Schema** (`ta._tools`, flag on): properties `code` (unchanged), then `"memory": {"type":"string","description": MEMORY_PARAM_DESCRIPTION}`. `required` stays `["code"]`.
  - It is the same object on every request. Test: `json.dumps(_tools())` is equal across two calls.
  - `MEMORY_PARAM_DESCRIPTION` (prompts.py), verbatim:
    `Optional memory changes, one per line: "+ wm TEXT", "+ hyp TEXT" or "+ note TEXT" adds an item; "~ ID TEXT" replaces its text; "ok ID" or "x ID" marks a hypothesis confirmed or refuted (text after the ID replaces its text); "- ID" drops it. The code runs whether or not this is given; the result's memory field lists the changes and the ids of new items.`
- **Parser:** `memory_items.parse_memory_param(value) -> (ops, notes)`. Lines are stripped and matched case-insensitively, but the text keeps its case. Blank lines are skipped. The first matching rule wins:
  1. `^\+\s*(wm|world_model|hyp|hypothesis|note|level_note)\b\s*:?\s*(.+)$` → add
  2. `^~\s*(m\d+)\b\s*:?\s*(.+)$` → revise
  3. `^(ok|confirmed?)\s+(m\d+)\b\s*:?\s*(.*)$` → confirm
  4. `^(x|refuted?)\s+(m\d+)\b\s*:?\s*(.*)$` → refute
  5. `^-\s*(m\d+)\b` → drop
  6. A line holding a memory tag → the tag mapper (iii), source `param`.
  7. Any other line → add a note. So `- some text` without an id is a note.
- **Caps.**
  - At most 8 lines are applied; more adds `N lines over the 8-line limit not recorded`.
  - The value is capped at 2000 chars.
  - Text is cut per kind with `cut_words`: wm 400, hyp 250, note 200.
  - A value of None (the parser's `null`) is ignored.
- **Aliases (flag on only).** The arguments `world_model`, `hypothesis` and `level_note` become `+ wm|hyp|note VALUE`, and `remember` becomes `+ note VALUE`. The model invented these names 39 times in old runs.
- **Truncation.** When the reply's `finish_reason == "length"`, drop the last line and add `last line not recorded (the reply hit the output limit)`. The flag is stored as `self._mem_reply_cut` before the dispatch loop (`ta` ~7376).
- **Dispatch** (`ta._dispatch_tool`, flag on):
  1. `_mem_absorb_param(arguments)` applies the memory and the aliases.
  2. If `code` is empty or missing, return `json.dumps({"error": "python requires a non-empty `code` string.", "memory": ACK}, indent=2)` without running anything.
  3. Otherwise run as now, `_mem_absorb_code` included (its tags get source `code`). Then parse `dispatch.content`, append the key `"memory": ACK` when changes are pending, and re-dump with `indent=2`.
  - On any error, return the original content.
- **ACK** (the text of the tool result's `memory` key).
  - It is at most 240 chars, then `; +N more`.
  - It lists all changes since the last tool result, from every source, so tags in this reply's thinking show up too.
  - Format: `act A: ` followed by the changes joined with `; `.
  - Change texts:
    - add: `+m7 world_model`, `+m4 hypothesis refuted`
    - text change: `m1 revised (rev 2)`
    - status change: `m2 open->confirmed (rev 2)`
    - drop: `m5 dropped`, or `m6 dropped (cap)`
    - problems: `m7 unknown id`, `m1 not a hypothesis`, `m1 unchanged`
  - Changes from sources other than the parameter add ` [reasoning tag]`, ` [reply tag]` or ` [code tag]`.
- **`act` and `turn`.**
  - `act` is `self._mem_act`. It is set from `action_num` in `analyze()` and updated from `compact_payload["action_num"]` after each executed action.
  - `turn` is `analysis_step`.

### (ii) MEMORY ADDENDUM
- `memory_addendum(..., xml=)` keeps the harness-fact lines exactly as on memory-v2: head card, delta, solved, effects, seen, plus the §2 path line.
- With `xml` on, the card branch's `MEMORY_NOTES_LINE + _MEMORY_TAG_EXAMPLES + "- One claim per tag.\n"` is replaced by `MEMORY_ITEMS_ADDENDUM` (verbatim below), so nothing is duplicated.
- `MEMORY_NOTES_SCOPE_LINE` follows when solved or effects is on, with "a note" changed to "an item".
```
- Your memory is a list of items. Each has an id (m1, m2, ... in creation order, never reused), a kind and a status: world_model (what the board is and what the actions do) is active or dropped; hypothesis is open, confirmed, refuted or dropped; level_note (about one level) is active or dropped.
- To change items, give the `python` tool a `memory` parameter after `code`, one change per line, for example:
    + wm the blue block is the piece you steer
    + hyp SPACE toggles the door
    ok m3 UP moves the blue block up one cell
    x m4
    ~ m2 walls stop the block; it does not wrap
    - m5
  `+ wm`, `+ hyp` and `+ note` add an item; `~ ID` replaces its text; `ok ID` and `x ID` mark a hypothesis confirmed or refuted, and text after the id replaces its text; `- ID` drops it. A line in none of these forms is added as a level_note. The code runs either way; the tool result's `memory` field lists what changed and the ids of new items.
- Tags in your reply, your reasoning or a python comment change the same items: `<world_model>`, `<hypothesis status="confirmed">` or `status="refuted"`, `<level_note>`, with `id="m3"` to name an item. A tag without an id whose words mostly match an item of its kind revises that item; `Correction:` at its start revises the closest world_model or hypothesis item; any other tag adds an item.
- Every change is a revision. The harness keeps each one with the game action count, its source (memory parameter, reply, reasoning or code) and the text and status before it. On the card and in `<memory_update>` an item reads `<hypothesis id="m3" status="confirmed" rev="2" act="41" was="open">UP moves the blue block up one cell</hypothesis>`: rev = revisions so far, act = game actions run at its last change, was = the status, or the start of the text, before that change. In python, `memory_items` holds the items and `memory_log` the newest 50 revisions.
- At most 6 world_model, 6 open, 8 confirmed, 8 refuted and 6 level_note items are kept; one more drops the oldest of its kind, and that drop is a revision too.
```
With the flag on, two other static texts change:
- The tool description (`_PYTHON_TOOL_DESCRIPTION`) appends ` \`memory_items\` and \`memory_log\` are read-only copies of the memory items and the newest 50 revisions.`
- The "Only tool:" line (`ta` 5375) lists `memory_items` and `memory_log` before `and \`action(actions)\``.

### (iii) Item store with a revision log: new `inference/agent/memory_items.py` (pure)
- **Item:** `Item(id, kind ∈ world_model|hypothesis|level_note, status, text, level, revs: int, act, was: str|None, version)`.
- **Revision:** `Revision(rev = the ledger's version counter, item, kind, op ∈ add|revise|confirm|refute|reopen|drop, source ∈ param|reply|reasoning|code|harness, act, turn, level, before={status,text}|None, after={status,text}, reason ∈ ""|"cap")`.
- **Store.** `ItemStore.apply_ops`, `apply_tag` and `drop` return `Change`s. The store keeps the newest 400 revisions in memory. Every revision also goes to the JSONL file (v).
- **Ledger wiring.** With the flag on, `_new_memory_ledger` sets `MemoryLedger(item_mode=True, card_mode=True)`.
  - `absorb(text, source=)` routes tags to `apply_tag` instead of `_absorb_card`.
  - Callers pass "reply" and "reasoning" (`ta` ~7221). `_mem_absorb_code` passes "code", and it also runs when the XML flag is on.
  - `level_note_for` reads the newest active note of that level.
  - Harness items (solved, hazard, fx) also log `Revision(source="harness", op=add|revise)` when bumped.
- **Tag matching** for tags without an `id` attribute, using jaccard and `_CORRECTION_RE` from memory_ledger:
  - `<hypothesis status=S>T`: the best non-dropped hypothesis at ≥ 0.6 is confirmed, refuted or revised to S and T. Otherwise a hypothesis is added with status S.
  - `<world_model>T`: the best active world_model at ≥ 0.6 is revised. With a `Correction:` prefix, the best world_model or hypothesis at ≥ 0.35 is revised. Otherwise a world_model is added.
  - `<level_note>T`: the best active note of this level at ≥ 0.6 is revised. Otherwise a note is added.
  - Identical text and status make no revision, so a tag repeated in the reply and the reasoning counts once.
- **Tags with `id="mN"`** act directly on mN. If the kind differs, the text is still revised, the status is left alone, and the ACK says `mN not a hypothesis`.
- **Parameter operations** act on the named id. `ok` or `x` on a non-hypothesis revises the text only if text is given; otherwise the ACK says `not a hypothesis`.
- **Caps** (active items per kind, as in the addendum). The oldest by `version` is dropped first; refuted items go last. The drop is logged as op `drop`, source `harness`, reason `cap`.

### (iv) How the card and deltas show change
- **Card** (`_belief_entries` in item mode).
  - Order: world_model, open, confirmed, refuted, level_note.
  - The memory-v2 drop order is unchanged; open items drop like any non-refuted item.
  - One line per item:
    `  <world_model id="m1" rev="2" act="41" was="the blue block is the piece you steer">the blue 2x2 block is the piece you steer</world_model>`
  - `rev` and `was` are left out at rev 1. A note adds `level="N"`.
  - `was` is:
    - `<old status>` if only the status changed;
    - `cut_words(old text, 40)` if only the text changed;
    - `<old status>: <cut text>` if both changed.
- **Delta line.** `+ ` followed by the same element.
  - It is queued under the event id `i:<id>`, so the newest state replaces a queued one. `_event_alive` returns True for `i:`.
  - It waits for the 15k hold like other deltas.
- **Drop line.** `+ <dropped id="m5" kind="world_model" act="57">walls stop the block...</dropped>`, with `by="cap"` added for cap drops.
- The card shows only the latest revision. The full history stays in `memory_log` and the JSONL file.

### (v) Export
- **File.** Each `Revision` is appended as one JSON line to `_resolve_named_run_artifact(state_path, default_name="memory_log.jsonl", per_game_suffix="_memory_log.jsonl")`, the same scheme as `requests.jsonl`. Each line has the `Revision` keys plus `"text_full"`. The write is in try/except.
- **Sandbox.** With the flag on, `_serialized_runtime_state` adds `"memory": {"items": [...], "log": newest 50}`. `python_tool_sandbox._refresh_state` sets the globals `memory_items` and `memory_log` only if that key is present.
- **Read-out tool.** New `inference/tools/memory_log.py RUN_DIR`. Per game and in total it reports:
  - writes per 100 calls by source (calls counted from `*_requests.jsonl`);
  - operations;
  - items with 2 or more revisions;
  - confirms and refutes;
  - ACK chars.
- **Viewer.** It can load the file later; the format is fixed here.

### (vi) Kaggle arm
`memory-xml` = the env of `memory-v2-both` + `"ARC3_MEM_XML": "1"`, in `kaggle/arms.json`.

### Commits (arm/memory-xml)
1. Add the memory item store with a revision log
2. Parse the memory parameter's line operations
3. Map memory tags onto item revisions
4. Route ledger notes through the item store under ARC3_MEM_XML
5. Add the optional memory parameter to the python tool
6. Apply the memory parameter and report changes in the tool result
7. Expose memory_items and memory_log in the sandbox
8. Write each game's memory revisions to memory_log.jsonl
9. Describe the memory items in the system prompt
10. Cover ARC3_MEM_XML in the identity and eviction tests
11. Add the memory log read-out tool
12. Document ARC3_MEM_XML and add the memory-xml arm

### Tests: `tests/test_memory_xml.py` (exact strings)
- **Parse.** `parse_memory_param("+ wm a b\n+ hyp c d\nok m1\n~ m2 e f\n- m9\nplain line")` gives `[("add","world_model","a b"),("add","hypothesis","c d"),("confirm","m1",""),("revise","m2","e f"),("drop","m9",""),("add","level_note","plain line")]`.
- **Store.** Level 2.
  - At act 3, the param `+ wm the blue block is the piece you steer\n+ hyp SPACE toggles the door` gives ACK `act 3: +m1 world_model; +m2 hypothesis`.
  - At act 41, the param `ok m2 SPACE opens and closes the door\n~ m1 the blue 2x2 block is the piece you steer\n+ note the key is behind the left wall\n- m7` gives ACK `act 41: m2 open->confirmed (rev 2); m1 revised (rev 2); +m3 level_note; m7 unknown id`.
  - The card then holds three lines: the m1 line shown in (iv), plus
    `  <hypothesis id="m2" status="confirmed" rev="2" act="41" was="open: SPACE toggles the door">SPACE opens and closes the door</hypothesis>`
    `  <level_note id="m3" level="2" act="41">the key is behind the left wall</level_note>`
  - The m2 log record equals `{"rev":3,"item":"m2","kind":"hypothesis","op":"confirm","source":"param","act":41,"turn":T,"level":2,"before":{"status":"open","text":"SPACE toggles the door"},"after":{"status":"confirmed","text":"SPACE opens and closes the door"},"reason":""}`.
- **Tags.**
  - In the reasoning, `<hypothesis status="confirmed">SPACE toggles the door open</hypothesis>` on an open m2 "SPACE toggles the door" (jaccard 0.75) confirms m2, with source "reasoning". The same tag in the content adds no revision.
  - `<world_model id="m1">walls stop the block</world_model>` revises m1.
  - An unmatched refuted tag gives `+m4 hypothesis refuted [reply tag]`.
- **Memory only.** `_dispatch_tool(sp,"python",{"memory":"+ wm walls stop the block"})` returns exactly `json.dumps({"error":"python requires a non-empty `code` string.","memory":"act 0: +m1 world_model"},indent=2)`, and `_run_python_tool` is not called.
- **Edge cases.**
  - Truncation: `+ wm a\n+ hyp b c` gives `act 0: +m1 world_model; last line not recorded (the reply hit the output limit)`.
  - 9 lines give the over-limit note.
  - Aliases.
  - Caps: the 7th world_model drops m1 with `m1 dropped (cap)`, and a refuted item survives longer than a confirmed one.
- **Wiring.** Schema keys are `["code","memory"]` with `required` `["code"]`. There is one JSONL line per revision. The sandbox globals are set.

## 2. ARM arm/level-path (`ARC3_MEM_PATH`)
### (i) Graph: `LevelGraph` in memory_facts.py
`LevelGraph` is owned by `FactRecorder` and fed from `FactRecorder.step` and `reset_step`.
- **Nodes** are board hashes (the interior, with border `ARC3_MEM_STATE_BORDER`). `links[pre][sig] = [post, ...]` keeps at most 4 outcomes, in arrival order.
- **Links.**
  - Only an action of kind `move` adds a link (`post_h != pre_h`, class shift or change).
  - These add no link: noop, hud, anim, death, clear, RESET, and `UNDO`/`ACTION7` (UNDO depends on history).
  - A MOUSE link keeps its exact signature `MOUSE(r,c)`.
- **`varied`** is the set of `(pre, sig)` pairs seen with more than one outcome: two different post boards, or a move plus a death or noop. The BFS uses every move outcome and records which steps are varied.
- **Resets.**
  - A RESET (automatic or manual) sets the current board to `attempts.start_h`, with route 0.
  - A level change starts a new graph.
  - The BFS source is the current attempt's `start_h`. In the r2 and v4 runs, no level had two start boards.
- **Caps.** 1024 boards, 16 signatures per board, 4 outcomes. Over a cap the link is not added (no LRU, so distances never grow) and `stats["graph_full"]` is incremented.
- **BFS** runs in full after every executed action. The previous distance map supplies the "before" value.
- **Change kind** per action:
  - `new board`: the post board was not on record;
  - `known board`: `d == d_before(post)`;
  - `shorter than M on record`: `d < d_before(post)`;
  - `no board change`;
  - `start board`: a RESET;
  - GAME OVER and clear are special cases (see (ii)).
- **Row log.** Each action appends `RouteRow(act, attempt, i, sig, d, k, change)`. `k` is the actions so far in the attempt; spent counts a manual RESET, but `k` does not.

### (ii) What the model sees after every action
- **Echo suffix**, built in `_handle_action`:
  1. Build `state = _serialized_runtime_state(...)` first. With the flag on, this feeds the recorder from the history it just loaded. The events are buffered in `self._mem_pending_events`, and `_mem_observe` applies them ahead of that turn's new events.
  2. Append the suffix to `compact_payload["action_echo"]`. With the echo off, set it to `"[route] " + suffix[3:]` instead.
  3. Return `{"action_result": compact_payload, "state": state}`.
- **Suffix format.**
  - At most 80 chars. Rows for automatic RESETs are not shown.
  - Single action, one of:
    - ` | route 9 (new board)`
    - ` | route 7 (known board)`
    - ` | route 1 (shorter than 3 on record)`
    - ` | route 8 (no board change)`
    - ` | route 0 (start board)`
    - ` | route - (GAME OVER from a board at route 8)`
    - ` | route - (level cleared; 19 on record)`
  - Batch: ` | route A->B (v1 v2 ...)`.
    - Values are numbers, `X` for GAME OVER and `C` for a clear.
    - More than 12 values show the first 5, `...` and the last 5.
    - `; shorter route found` is added if any step was shorter.
  - Worked example: batch `[RIGHT, RIGHT, LEFT, UP, UP]` from the start board. S-R→A is new, A-R→B new, B-L→A known, A-U→F new, F-U→G new. The echo is `[action] batch of 5 actions, 5 executed: RIGHT, RIGHT, LEFT, UP, UP | route 0->3 (1 2 1 2 3)`.
- **Turn line.**
  - `_mem_turn_lines` builds it before its `ledger is None or not _mem_v2_on()` early return. It goes after the `Seen before` line and before `<memory_update>`.
  - It appears on every turn message after actions ran on the same level.
  - It is not held by default (`ARC3_MEM_PATH_AFTER_TOKENS`, default 0).
  - Caps: 300 chars per line, and 16000 chars per game (`ARC3_MEM_PATH_GAME_CHARS`).
  - The worked example gives: `Shortest route on record from this level's start board to the current board: 3 actions [R U2]; this attempt so far: 5 actions. Route length after each action since the last message: 1 2 1 2 3.`
  - The path is `_fit_path(path, 64)`.
  - At d = 0 the route reads `0 actions (this is the start board)`. When no links reach the board, it reads `none on record`.
  - Optional additions:
    - ` 1 step on the route had more than one outcome on this level.`
    - ` A shorter route to a known board was found at action #4 of this attempt (3 -> 1).`
  - The first message after a death reads: `Shortest route on record from this level's start board to the current board: 0 actions (this is the start board). Attempt 1 ended in GAME OVER after 2 actions; its last board before GAME OVER is 1 action from the start on record [R].`
  - Over the cap, the per-action list is dropped first, then the additions.
  - `_route_cursor` marks the last row shown.

### (iii) Python
- **Sandbox state.** With the flag on, `_serialized_runtime_state` adds `"level_path": {"path": [...], "len": n|None, "varied": [indexes]}`. `_refresh_state` sets `level_path`, `level_path_len` and `level_path_varied` only when that key is present.
  - Items are in `action()` form: `"RIGHT"`, or `{"action":"MOUSE","row":r,"col":c}`.
  - At the start board the path is `[]` and the length is 0. Both are None when no links reach the current board.
  - The values refresh at snippet start and in the state returned by every `action()` call.
- **Tool description** (flag on, static) appends: ` \`level_path\` is the shortest route on record from this level's start board to the current board, as a list action(...) accepts, and \`level_path_len\` its length; both refresh after every action(...) call.`
- **"Only tool:" line** lists `level_path` and `level_path_len` before `and \`action(actions)\``.
- **Addendum line** `MEMORY_PATH_LINE`, with `{b}` filled at build time. It goes after `MEMORY_SEEN_LINE` inside `memory_addendum`. When `_mem_v2_on()` is false, it is appended after the ledger text at `ta` ~2818 instead.
  `- Route facts come from links the harness records on each level: a board-changing action links one board to the next (boards compared with the outer {b} cells cropped; UNDO, RESET, GAME OVER, clears and no-change actions add no link; a click is its exact cell). The route is the fewest links from this level's start board to the current board. Each action(...) call prints route N and what changed; a turn message after actions carries a Shortest route on record line. In python, level_path is that route as a list action(...) accepts and level_path_len its length.`

### (iv) Solved line (with ARC3_MEM_SOLVED)
- **Path.** At a clear, compute `bfs(start→pre_h of the clearing action) + [clear sig]`.
  - If it is strictly shorter than `loop_free`, it becomes the solved path, and `solved_head` adds ` (K in the winning attempt)` after the brackets.
  - Otherwise the output is byte-identical to memory-v2.
- **Prompt text.** `MEMORY_SOLVED_LINE` gets a static flag-on variant: `... the shortest route on record to the winning board, then the winning action (...legend...); "(K in the winning attempt)" follows when that attempt, with its loops cut out, was longer; then the actions spent on the level and its deaths.`
- **Test.** Attempt 1 is S-D→C, then a manual RESET. Attempt 2 is S-R→A-R→B-D→C-R→clear. Result: `L1: 2 acts [D R] (4 in the winning attempt), spent 6, 0 deaths`.

### (v) Caps, CPU, export, knobs
- **CPU.** The BFS costs 8 µs on average and 320 µs at most (r2). The graph caps bound it to 16k links. Feeding the recorder mid-snippet only moves the existing hashing earlier; it adds none.
- **Export.** `*_level_path.jsonl`, one row per action: `{act, level, attempt, i, sig, board: hash[:10], d, k, change, varied, path: rle≤200}`.
- **Knobs:** `ARC3_MEM_PATH`, `ARC3_MEM_PATH_AFTER_TOKENS=0`, `ARC3_MEM_PATH_GAME_CHARS=16000`.

### (vi) Kaggle arm
`level-path` = the env of `memory-v2-both` + `"ARC3_MEM_PATH": "1"`.

### Commits (arm/level-path)
1. Add LevelGraph with move links and BFS routes
2. Feed LevelGraph from FactRecorder steps and resets
3. Feed the recorder after each action call and buffer its events
4. Add the route suffix to the action echo
5. Add the Shortest route line to the turn message
6. Expose level_path in the sandbox and name it in the tool text
7. Use the shortest route on record in the solved line
8. Write per-action route rows to level_path.jsonl
9. Describe route facts in the system prompt
10. Cover ARC3_MEM_PATH in the identity and eviction tests
11. Document ARC3_MEM_PATH and add the level-path arm

### Tests: `tests/test_level_path.py` (hash-only `FactRecorder.step` replays, exact strings)
- **Sequence.** S-R→A, A-R→B, B-D→C, C-U→B, a manual RESET, then S-D→C.
  - The suffixes are `route 1 (new board)`, `2 (new board)`, `3 (new board)`, `2 (known board)`, `0 (start board)`, `1 (shorter than 3 on record)`.
  - Then `level_path == ["DOWN"]`.
  - The turn line after the last two actions is `Shortest route on record from this level's start board to the current board: 1 action [D]; this attempt so far: 1 action. Route length after each action since the last message: 0 1. A shorter route to a known board was found at action #1 of this attempt (3 -> 1).`
- **Other cases.**
  - The worked batch example.
  - The death line.
  - Varied: A-R reaches B, later D, so `level_path_varied == [1]` and the note appears.
  - UNDO adds no link (`none on record`).
  - MOUSE(4,7) gives `{"action":"MOUSE","row":4,"col":7}` and `[M(4,7)]`.
  - The cap counter.
  - The solved line.
- **Replay.** Skipped when `ARC3_MEM_REPLAY_RUNS` lacks fz-ledger-on-trim-r2. Over `artifacts/*_events.jsonl`:
  - the BFS beats `loop_free` at exactly 2 of 112 clears (g50t L1, sc25 L3);
  - `d < loop_free` in 2.2% ± 0.3 of actions;
  - `graph_full == 0`;
  - all of it under 60 s of CPU.

## 3. arm/memory-v3 (both arms, on memory-v2)
- **Merge.** Merge memory-xml first, then level-path. Resolve conflicts in:
  - `memory_addendum`: fact lines, then the path line, then `MEMORY_ITEMS_ADDENDUM`;
  - `_tools`: one description string, built from flags;
  - the "Only tool" line: `level_path, level_path_len, memory_items, memory_log`;
  - `_serialized_runtime_state` keys;
  - `memory_sim.MEMORY_V2_FLAGS`;
  - the eviction configs;
  - CONFIGURATION.md and arms.json.
- **Combine commits:**
  1. `Compose the item and route addendum lines`: tests the exact combined system text with both flags on.
  2. `Log solved routes as harness revisions`: the solved item's revision `after.text` carries the chosen path.
  3. `Add the memory-v3 arm`: `memory-v3` = the env of `memory-v2-both` + `ARC3_MEM_XML=1` + `ARC3_MEM_PATH=1`.

## 4. Identity, #58 and docs (each arm, and again on v3)
- **Identity.**
  - Add every new flag and knob to `memory_sim.MEMORY_V2_FLAGS`, so the stack-v2 golden still has to pass.
  - New fixture `tests/fixtures/memory_v2_on_golden.json`:
    - Scenarios `kaggle-memv2-both` and `kaggle-memv2-trim`: KAGGLE_ENV plus the arm env, with `MEMSIM_COMPLETION_TOKENS=4000`.
    - Plus `tools_sha`, the sha of `json.dumps(_tools())`. `fake_chat` records `kwargs["tools"]` for this.
    - Generate it by running this branch's `tests/memory_sim.py` against a checkout of `arm/memory-v2` @ d252fa3, and store that commit in the fixture.
  - New test: with every new flag off, the system prompt, every request and `tools_sha` equal that golden.
- **#58.**
  - New `eviction_sim` configs `mem-xml`, `mem-path` and `mem-v3` (MEMORY_V2_ON, placement both).
  - With `MEMSIM_MEMORY_PARAM=1`, the scripted replies carry `memory` on turns 4, 9 and 18, and a memory-only probe on turn 11.
  - `test_only_real_drops_hand_the_slot_over` must pass all four assertions, and the existing traps must still fail.
- **Docs** (same PR):
  - `CONFIGURATION.md` §10: a row for each new flag, what the model sees (the strings above), placement and cache safety, cost, read-outs and keep/drop.
  - The branch `README.md`: the arm table and a paragraph.
  - `kaggle/README.md`: the arm names.
  - `arms.md`: new entries under In progress, moved once results are in.

## 5. Read-outs and keep/drop, fixed before any run
- **Batch.** Each batch runs `memory-v2-both` (the paired control), `memory-xml`, `level-path` and `memory-v3`; two runs each. Compare with `make significance` (paired per game) and report tokens per run.
- **memory-xml**
  - Read-outs:
    - model memory writes per 100 calls (baseline 2.6);
    - the share written through the parameter;
    - items with 2 or more revisions;
    - confirms and refutes per game;
    - empty-code errors;
    - the `finish_reason=length` rate;
    - tokens per call.
  - **Keep:** a 2-run mean of at least control + 2, at least 10 writes per 100 calls, and tokens per call up by at most 3%.
  - **Drop:** under 5 writes per 100 calls, or a score at most control − 1.
- **level-path**
  - Read-outs:
    - actions on net-zero loops (baseline 27%);
    - exact repeats (baseline 9.0%);
    - median `k − d` at the end of each attempt;
    - level spend minus the route (baseline median 4);
    - excess actions on re-walks after deaths;
    - the share of executing snippets that reference `level_path`;
    - tokens per executing turn (budget +60).
  - **Keep:** a mean of at least control + 2, and the loop share down by at least 3 points.
  - **Drop:** a score at most control − 1, or loops and repeats unchanged with `level_path` used in under 1% of snippets.
- **memory-v3**
  - **Keep** if it beats the better single arm by at least 1 and control by at least 3.
  - Otherwise ship the better single arm.

A copy of this spec is saved at `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/memv3-research/memv3-spec.txt`. All code references are to the worktree `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/memory-v2`.
