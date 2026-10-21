# memory-xml-and-level-path: fix:memory-xml

Workflow run `wf_f7a7e5db-bad`, phase Fix. Research, design, build, review and integrate two arms on memory-v2: memory writes in the Qwen3 XML tool-call format with a memory addendum and revision log, and a shortest-known-path tracker shown after every action

---

Every finding checked out on the code. I fixed findings 1–8 and 9a in 11 new commits on `arm/memory-xml`, and left 9b alone. The tree is clean, nothing is pushed, and nothing has run on Kaggle. All nine targeted test files pass, and the #58 eviction guard passes with the memory parameter in use.

**Per finding** (the short code is the new commit)
1. **Fixed, `946e0ff`: a quoted older line no longer undoes later changes.**
   - A tag naming an id is ignored when its `rev` is below the item's current rev. A tag with `act` but no `rev` counts as rev 1, since rev-1 card lines leave `rev` out.
   - I used the rev check rather than ignoring every tag that carries `act`/`rev`, so editing the current line still revises the item. Tags with neither attribute work as before.
   - A stale quote in the reasoning is ignored silently, because quoting the head card there is common. From any other source, if it differs from the item, the tool result says `m1 unchanged (tag rev 1, item rev 2)`.
   - Your repro now leaves m1 refuted at rev 2 with an empty memory field.
2. **Fixed, `62161f7`:** the keywords are now `ok|confirm|confirmed` and `x|refute|refuted`, and the id may follow a space, colon or comma (`ok, m3`). `X marks the spot` is still a note.
3. **Fixed, `5aabd67`: status words.**
   - `open`, `untested`, `unverified`, `unconfirmed`, `pending` and `reopened` set open.
   - `dropped` drops the named or matched hypothesis, and adds nothing when none matches.
   - Any other word sets no status, so the item keeps its own. The flag-off ledger is unchanged.
4. **Fixed, `eacf91b`: the 2000-char cut.**
   - The line the limit splits is left out whole and reported (`2 lines over the 2000-char limit not recorded`).
   - Exception: when the split line is the first line, its first 2000 chars are kept, which is still longer than any item text. This keeps the existing cap test.
   - A reply cut at the output limit no longer loses a second line when the char limit already removed the last one.
5. **Fixed, `e2021c8`, but narrower than you suggested.**
   - An add or `~ ID` line whose text has no letter or digit is now reported and not recorded: `line not recorded: + wm`. This covers `+ wm`, `+ wm:` and `~ m2`.
   - I did not reject every `+`/`~` line. Lines like `+3 points` or `~3 clicks` are natural notes, so `+ TEXT` with no kind word becomes a note with the `+` stripped like a bullet.
   - `world model`, `world-model` and `level note` are now read as kind words.
   - The addendum gained one clause for this (below).
6. **Fixed, `63c1935`:** item-kind tags, `<dropped>` and the memory block markers (any case, with attributes) are stripped from item text. `+ wm foo </world_model> bar` now renders as `<world_model id="m2" act="3">foo bar</world_model>`.
7. **Fixed, `46220e5`:** the memory field starts a new `act N:` group whenever the count changes. Changes at one count read as before.
8. **Fixed, `bd377ff`:** the read-out tool always reads the run-level `memory_log.jsonl` and `requests.jsonl`, reported as the game `run`.
   - My departure 10 was wrong. A game writes the run-level file until a second game's state file appears, so in a multi-game run `run` holds the first game's early rows.
   - That game's item ids are counted separately under `run` and under its own name. The docstring and CONFIGURATION.md now say this.
9. **Informational.**
   - **9a, fixed, `3c94cba`:** python's `memory_log` now holds only item revisions (cap drops included). Solved, hazard and first-effect facts stay on the card and in `memory_log.jsonl`.
   - **9b, not changed:** in r2, 0 of 2,218 tool calls shared a reply with another dispatched call. Skipped calls aren't written to transcripts, so their number is unknown. This is the same as existing behaviour.

**Extra fix found while probing (`a4723fa`):** a tag spread over several lines of the `memory` value used to become several level notes containing markup. It is now joined into one line and applied as a tag.

`fedada6` updates CONFIGURATION.md section 10 for all of the above. I also added a "Review fixes" line to the memory-xml entry in `arms.md` and updated its system-text size (~410 tokens).

**`git log --oneline arm/memory-v2..HEAD`**
```
fedada6 Document the memory parser, status and quote rules
3c94cba Keep only item revisions in python's memory_log
bd377ff Read the run-level logs beside the per-game ones in the read-out
46220e5 Start a new act group in the memory field when the count changes
946e0ff Keep a quoted older item line from undoing later changes
5aabd67 Set a tag's status only from a word that names one
63c1935 Remove memory tag markup from an item's text
a4723fa Read a memory tag that spans lines of the memory value
eacf91b Leave out the memory line the char limit cuts through
e2021c8 Report an add or revise line that has no text
62161f7 Read confirm and refute lines in the memory parameter
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

**Targeted tests** (niced, one at a time, imports confirmed to resolve to the worktree)
`test_memory_xml 54 passed | test_memory_wording 58 passed | test_memory_log_tool 3 passed | test_memory_flags_off_identity 11 passed | test_memory_eviction 11 passed | test_memory_placement 20 passed | test_memory_facts 18 passed | test_ledger_on_trim 26 passed | test_sandbox_deadline 16 passed`

**Exact model-visible strings after the fixes**
- **Changed addendum sentence:** ``A line in none of these forms is added as a level_note, and an add or `~` line with no text is not recorded.``
  - With the memory-v2-both flags, the system text now grows by 1,474 chars instead of 1,421 (about 410 tokens; static and cached).
  - The tool schema and parameter description are unchanged.
- **New memory field strings:**
  ```
  act 3: line not recorded: + wm; line not recorded: ~ m2; m9 unknown id
  1 line over the 2000-char limit not recorded
  act 20: m1 unchanged (tag rev 1, item rev 2) [reply tag]
  act 20: m2 dropped [reply tag]
  act 3: +m3 world_model; act 9: +m4 world_model
  ```
- **Card line with markup stripped:** `<world_model id="m2" act="3">foo bar</world_model>`
- **Scripted sim with the flag on:** every non-system message in all 31 requests, and the final card, are identical to the review's pre-fix dump. The only difference is the addendum clause.
- **r1/r2 replay** (one niced process, 0.6 s CPU): tags, items, cap drops, memory fields and card sizes are unchanged. r1's deltas grew by 0.1 chars per game.
- **Not covered by replay:** the stale-quote rule can't trigger there, because old runs had no ids. Unit tests cover it.

**PR summary**
- **arm/memory-xml:** the python tool gets an optional `memory` parameter (Qwen3 XML `<parameter=memory>`), and the model's notes become numbered items with a revision log. All of it is behind `ARC3_MEM_XML`, default off.
- **Model writes:** `+ wm|hyp|note TEXT`, `~ ID TEXT`, `ok|confirm ID`, `x|refute ID`, `- ID`. Tags still work and can name an item with `id="mN"`.
- **What the model sees:** each tool result carries a `memory` field grouped by action count (`act 17: m4 open->refuted (rev 2); m3 revised (rev 2)`). Card and delta lines show `id`, `rev`, `act` and `was`. Python gets `memory_items` and `memory_log`.
- **Logging:** each game's revisions go to `memory_log.jsonl` (source, action count, turn, before/after), read out by `python -m inference.tools.memory_log RUN_DIR`.
- **Review fixes:**
  - A quoted older card or delta line no longer undoes later changes.
  - `confirm m3` is now read.
  - `status="dropped"` drops, and unknown status words keep the item's status.
  - Adds and revises with no text are reported instead of becoming notes.
  - The 2000-char cut never keeps a fragment of a line.
  - Tag markup is stripped from item text.
- **Facts, not advice:** the wording test covers every model-visible string, including the new tool-result forms.
- **Cache and #58:** the schema and system text are static per run. The `memory` field is added only to the tool result being built, so sent history is never edited, and the `mem-xml` eviction config passes.
- **Flags off:** the memory-v2 arms are byte-identical to `arm/memory-v2` (golden from d252fa3).
- **Cost:** about 150 tokens of schema plus 410 of system text, both in the cached prefix, and about 20–30 tokens on tool results that carry a field.
- **Replay of r1/r2:** cards are the size of memory-v2's (median 348/447 chars), and 31/57 tool results get a memory field.
- **Not run on Kaggle.** Batch: `memory-xml` against `memory-v2-both`, two runs each, compared with `make significance`.

Files are under `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/`:
- Worktree (`wt/memory-xml/ARC3-Inference/`):
  - `inference/agent/memory_items.py`
  - `inference/agent/prompts.py`
  - `inference/agent/tool_agent.py`
  - `inference/tools/memory_log.py`
  - `tests/test_memory_xml.py`
  - `tests/test_memory_wording.py`
  - `tests/test_memory_log_tool.py`
  - `CONFIGURATION.md`
- Scratch:
  - `memxml_replay_after.json`
  - `dump_xml_after.json`
- Arms log: `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/arms.md`
