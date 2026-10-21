# memory-xml-and-level-path: build:memory-xml

Workflow run `wf_f7a7e5db-bad`, phase Build. Research, design, build, review and integrate two arms on memory-v2: memory writes in the Qwen3 XML tool-call format with a memory addendum and revision log, and a shortest-known-path tracker shown after every action

---

Arm 1 (`arm/memory-xml`) is built and committed but not pushed and not run on Kaggle. The targeted test files pass, the memory-v2 arms are byte-identical with the new flag off, and the #58 eviction guard passes with the memory parameter in use.

Worktree: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/memory-xml`, branch `arm/memory-xml`. The branch starts from `arm/memory-v2` at cf40ac1, as your worktree command said. cf40ac1 differs from d252fa3 only in `kaggle/arms.json`, so the new golden was generated from a `git archive` of d252fa3, and the fixture records that commit.

## Commits (`git log arm/memory-v2..HEAD`)
```
e34ec43 Keep a hypothesis's status when a tag names none
4ddd337 Show the memory field in transcripts and prompt logs
42dbc35 Document ARC3_MEM_XML and add the memory-xml arm
f9b8453 Add the memory log read-out tool
6e1fb4f Cover ARC3_MEM_XML in the identity and eviction tests
a8c93f5 Describe the memory items in the system prompt
27cd572 Write each game's memory revisions to memory_log.jsonl
365802b Expose memory_items and memory_log in the sandbox
5768d28 Apply the memory parameter and report changes in the tool result
8e96b7b Add the optional memory parameter to the python tool
c072741 Route ledger notes through the item store under ARC3_MEM_XML
b7b5fc7 Map memory tags onto item revisions
3b24d90 Parse the memory parameter's line operations
253b45c Add the memory item store with a revision log
```
These are the spec's 12 commits plus two (`4ddd337`, `e34ec43`). The tree is clean.

## Tests (niced, one at a time, imports checked to resolve to the worktree)
```
test_memory_xml: 44 passed
test_memory_wording: 54 passed
test_memory_log_tool: 2 passed
test_memory_flags_off_identity: 11 passed   (stack-v2 golden + new memory_v2_on_golden.json + xml-changes-it check)
test_memory_eviction: 11 passed             (new mem-xml config: violations all empty, 10 handovers, 8 tool results with a memory field, 1 memory-only call; traps still fail)
test_memory_placement: 20 passed
test_memory_facts: 18 passed
test_ledger_on_trim: 26 passed
test_sandbox_deadline: 16 passed
```
`python kaggle/build_notebook.py --arm memory-xml` builds the notebook (no push).

## What the model sees
- **Tool schema:** properties are `["code","memory"]` and `required` stays `["code"]`. The `memory` description is the spec's `MEMORY_PARAM_DESCRIPTION`, verbatim.
- **Tool description** gets this appended: `` `memory_items` and `memory_log` are read-only copies of the memory items and the newest 50 revisions.``
- **"Only tool:" line:** `` ...`frame_diff(before, after)`, `memory_items`, `memory_log`, and `action(actions)`. ``
- **System text:** `MEMORY_ITEMS_ADDENDUM` is the spec's text, verbatim and tested. It replaces the tag examples and "One claim per tag". The scope line now reads: "...so an item holds what the frames do not show: the goal, the rules, and why."
- **Tool-result `memory` field**, from `memory_sim` with `MEMSIM_MEMORY_PARAM=1`:
  ```
  "act 1: +m1 world_model [reasoning tag]"
  "act 4: +m2 hypothesis confirmed [reasoning tag]"
  "act 8: +m3 world_model; +m4 hypothesis"
  "act 9: +m5 hypothesis refuted [reply tag]"
  "act 10: +m6 world_model [code tag]"
  "act 17: m4 open->refuted (rev 2); m3 revised (rev 2)"
  "act 38: +m9 level_note; m6 dropped; m99 unknown id"
  ```
- **Memory-only call** returns exactly `json.dumps({"error":"python requires a non-empty `code` string.","memory":"act 0: +m1 world_model"},indent=2)`, and the code does not run.
- **Other field forms:** `m1 dropped (cap)`, `m1 not a hypothesis`, `m1 unchanged`, `1 line over the 8-line limit not recorded`, `last line not recorded (the reply hit the output limit)`, `; +N more`.
- **Card lines** (exact strings, tested):
  ```
    <world_model id="m1" rev="2" act="41" was="the blue block is the piece you steer">the blue 2x2 block is the piece you steer</world_model>
    <hypothesis id="m2" status="confirmed" rev="2" act="41" was="open: SPACE toggles the door">SPACE opens and closes the door</hypothesis>
    <level_note id="m3" level="2" act="41">the key is behind the left wall</level_note>
  ```
- **Delta lines:** `+ ` followed by the same element, plus drop lines like `+ <dropped id="m4" kind="hypothesis" act="57" by="cap">red tiles slide</dropped>`.
- **Revision record** (exact, tested): `{"rev":3,"item":"m2","kind":"hypothesis","op":"confirm","source":"param","act":41,"turn":9,"level":2,"before":{...},"after":{...},"reason":""}`. Each line of `memory_log.jsonl` adds `text_full`.

## Validation on recorded runs
One niced process replayed the r2 and r1 transcripts through the item store, in under 1 s of CPU.

| | r2 | r1 |
|---|---|---|
| Tags → items | 104 → 101 | 92 → 85 |
| Items revised by a later tag | 3 | 6 |
| Cap drops | 29 (tu93's 40 tags) | 3 |
| Tool results with a `memory` field | 57, mean 71 chars, max 215 | 31, mean 100 chars, max 235 |
| Largest card per game, median | 447 chars (memory-v2: 430) | 348 chars (memory-v2: 330) |
| Largest card, max | 3,555 chars | 3,889 chars (cap 3,900) |
| Item deltas per game, no hold | 1,251 chars (memory-v2: 1,154) | 982 chars (memory-v2: 854) |

- **Speed:** absorbing every tag takes 11–15 ms per run.
- **Prefix cost** with the memory-v2-both flags: the schema grows by 519 chars (about 150 tokens) and the system text by 1,421 chars (about 400 tokens). Both are static, so they stay in the cached prefix.
- **Not replayed:** the 1.4 GB xz corpus of old-run invented argument names was too heavy; the aliases are covered by unit tests only.

## Where I departed from the spec (safer option, each documented)
1. **Attribute name:** the ledger's store is `item_store`, because `MemoryLedger.items()` is already a method.
2. **Absorb order:** with the flag on, the reasoning is absorbed before the reply, in generation order. The spec's "tag in the reasoning confirms; the reply repeat is a no-op" test needs this. The flag-off order is unchanged.
3. **Hypothesis status:** a `<hypothesis>` tag without a `status` keeps the matched or named item's status. Defaulting to open would have reopened confirmed claims whenever the model restated one. `status="open"` still reopens.
4. **Cut replies:** only the last tool call of a reply cut at the output limit loses its last memory line.
5. **Dropped items:** `~`, `ok`, `x` or a tag with an id on a dropped item brings it back (`m5 dropped->active (rev 3)`). `- ID` on a dropped item reports `unchanged`.
6. **Note lines:** a leading bullet is stripped, so `- some text` becomes the note "some text".
7. **Grammar:** a single extra line reads `1 line over...` rather than `1 lines`.
8. **Harness facts:** solved, hazard and first-effect facts are logged as harness revisions but are not in the tool-result field. The actions table is not logged.
9. **Sandbox size:** `memory_items` is capped at 200 items.
10. **Log path:** `memory_log.jsonl` is resolved at each write, as `requests.jsonl` is, so a game that starts before the others still gets its own file.
11. **Extra commit `4ddd337`:** the transcript and prompt-log display now shows the `memory` field. It was invisible there before; this is display only.

## Limitations
- **Rolled-back turns:** a turn rolled back after a request failure keeps its memory changes, but its `memory` field is lost with the dropped tool result. The items still reach the model through the card and deltas.
- **Malformed cut replies:** the harness drops the tool call before dispatch, so its memory is never applied.
- **Solved levels:** once a level's note rides on its solved line, that level's note items leave the card and the delta queue, as in memory-v2.

## Notes for the integrate step and v3
- **Eviction configs:** only `mem-xml` is added here; `mem-path` and `mem-v3` belong to the other branches.
- **Flag list:** `memory_sim.MEMORY_V3_FLAGS = ("ARC3_MEM_XML",)` feeds `MEMORY_V2_FLAGS`. level-path has to add its own flags there.
- **Merge conflicts to expect:**
  - `memory_addendum`: the `xml` parameter.
  - `_tools`: the description is now a local variable plus `MEMORY_TOOL_GLOBALS`.
  - The "Only tool:" line: now built with `.replace`.
  - `_serialized_runtime_state`: `**self._mem_sandbox_state()`.
  - `memory_sim`: the `MEMORY_PARAM` dict and `tools_sha`.
  - `eviction_sim`: `probe_reply` and `CONFIGS`.
  - `CONFIGURATION.md` section 10, `arms.json`, `README.md`, `kaggle/README.md`.
- **`arms.md`:** I added a memory-xml entry under In progress, just above the existing level-path entry.

## Files (under the worktree's `ARC3-Inference/`)
New:
- `inference/agent/memory_items.py`
- `inference/tools/memory_log.py`
- `tests/test_memory_xml.py`
- `tests/test_memory_wording.py`
- `tests/test_memory_log_tool.py`
- `tests/fixtures/memory_v2_on_golden.json`

Changed:
- `inference/agent/tool_agent.py`, `memory_ledger.py`, `prompts.py`, `python_tool_sandbox.py`
- `tests/memory_sim.py`, `tests/eviction_sim.py`, `tests/test_memory_eviction.py`, `tests/test_memory_flags_off_identity.py`
- `CONFIGURATION.md`, plus `../README.md`, `../kaggle/README.md` and `../kaggle/arms.json`

Scratch files (in the scratchpad):
- Replay script: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/memxml_replay.py`
- Replay output: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/memxml_replay.json`
- Golden generator: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/gen_golden.py`, run against the extracted copy in `.../scratchpad/memv2-d252fa3/`

Arms log entry: `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/arms.md`.
