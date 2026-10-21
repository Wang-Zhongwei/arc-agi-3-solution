# memory-xml-and-level-path: review:memory-xml

Workflow run `wf_f7a7e5db-bad`, phase Review. Research, design, build, review and integrate two arms on memory-v2: memory writes in the Qwen3 XML tool-call format with a memory addendum and revision log, and a shortest-known-path tracker shown after every action

---

**arm/memory-xml review: no blocking defects, one medium-severity risk, seven low-severity defects**

The branch meets the hard rules, so it can go to integrate after fix 1. Fixes 2–4 are cheap and worth doing in the same pass.

**Hard rules, checked against the code and by running it**
- **Flags-off byte identity:** I rebuilt the golden independently by running `git archive d252fa3` with this branch's `tests/memory_sim.py`. It matches `tests/fixtures/memory_v2_on_golden.json` exactly: system prompt, tools_sha, all 31 requests and the trim counts, in both scenarios. With `ARC3_MEM_XML=1` all 31 requests differ, so the identity test is not passing vacuously.
- **#58 (no edits to sent history):**
  - The `memory` field is only added to the tool result being built now (`_mem_with_ack` in `tool_agent.py`, about line 6365). That function returns a new `_ToolDispatchResult` and keeps all three of its fields.
  - The stored tool calls are never edited.
  - Item deltas go through the existing `_queue_event` and `take_delta` path; `restore_display` keeps the `i:` lines.
  - A flag-on dump shows item lines only in new turn messages, in the head card after real trims, and in tool results. The `mem-xml` eviction config passes.
- **Facts, not advice:** the wording test covers the addendum, the parameter description, the tool text, the "Only tool" line, the card, deltas and every tool-result message form. It passes.
- **Caps:** ACK 240 chars, 8 lines, 2000 chars, text per kind 400/250/200, card 3900, 64 pending changes, 400 revisions in memory, 200 items and 50 revisions in the sandbox.
- **Cache safety:** the schema is the same object on every request (`tools_static`), and with the flag off the `memory` argument is ignored.
- **Tests:** I ran 9 files, niced and one at a time, with imports confirmed to resolve to the worktree: xml 44, wording 54, log tool 2, identity 11, eviction 11, placement 20, ledger_on_trim 26, facts 18. All pass.
- **Real data:** I replayed three r2 games (g50t, tu93, ls20) with no errors.
  - g50t: three replies reworded tags from the reasoning, and each correctly revised items m2–m4.
  - tu93: 40 world_model adds, 29 cap drops.
  - ls20: 6 tags became 6 items.
- **Level-path checks:** the UNDO, MOUSE, batch, BFS and CPU checks belong to arm/level-path and do not apply to this branch.

**Defects, most severe first**

1. **Medium: quoting an older line with an id undoes a later change.**
   - Where: `memory_items.py:584-595`, the `apply_tag` path for tags that name an id.
   - Why it happens: the head card stays at the top of the context until the next trim, so it is often older than the deltas below it. If the model restates the card line in its reasoning, the item reverts.
   - Reproduced: m1 was refuted at act 17. At act 20 the reasoning quoted the head card's `<hypothesis id="m1" status="open" act="8">SPACE changes nothing on this board</hypothesis>`. Result: `m1 refuted->open (rev 3) [reasoning tag]`, with the old text restored and a new delta queued.
   - Fix: treat any tag that carries an `act=` or `rev=` attribute as a copy of a harness-rendered line and make it a no-op. At minimum, ignore it when its `rev` is lower than `item.revs` or its `act` is lower than `item.act`. Add a test that quoting a stale head-card line changes nothing.

2. **Low–medium: `confirm m3` is not parsed.**
   - Where: `_CONFIRM_RE` at `memory_items.py:87`. The regex `(ok|confirmed?)` matches "confirme" or "confirmed", not "confirm". The spec has the same regex, but the docstring at line 300 promises `ok|confirm|confirmed`.
   - Effect: `confirm m3` becomes a junk level_note (`+m9 level_note`) and m3 stays open. `refute m3` does work.
   - Fix: use `^(ok|confirm(?:ed)?)\s+…` and `^(x|refute(?:d)?)\s+…`, and add parse tests for both.

3. **Low: an unknown status value, or `status="dropped"`, reopens a confirmed hypothesis.**
   - Where: `memory_items.py:572-574`, where `_STATUS_ALIASES.get(…, OPEN)` maps any unknown value to open.
   - Reproduced: `<hypothesis id="m2" status="dropped">` gives `m2 confirmed->open`. A matched restatement with `status="likely"` also reopens m2. The fix in departure 3 only covers tags with no status at all.
   - Fix: map `open` to open, make `dropped` a drop, and treat any other unrecognised value as None so the item keeps its status.

4. **Low: the 2000-char cut splits a line, and the fragment is recorded.**
   - Where: `parse_memory_param`, `memory_items.py:315`.
   - Reproduced: `"+ wm " + 1990×"a" + "\n+ hyp second"` adds a level_note with the text `+ hy`.
   - Fix: when the value is over 2000 chars and the cut is not at a newline, drop the partial last line and report it, for example `N chars over the 2000-char limit not recorded`.

5. **Low: malformed operation lines become junk notes.**
   - Where: `_parse_line`, `memory_items.py:268-289`.
   - Reproduced: `+ wm` gives the note "+ wm", `~ m2` gives the note "~ m2", and `+ wm:` adds a world_model with the text ":".
   - Fix: a line that starts with `+`, `~`, `ok`, `x` or `-`+id but fails its rule should be reported as `line not read: <first 30 chars>`. The text of an add should also have to contain a letter or digit.

6. **Low: tag markup inside parameter text breaks the card line.**
   - Where: `clean_text` and `element`, `memory_items.py:97-102` and `185-200`.
   - Reproduced: `+ wm foo </world_model> bar` renders as `<world_model id="m1" act="0">foo </world_model> bar …</world_model>`. If the model quotes that line, m1 is cut to "foo".
   - Fix: in `clean_text`, neutralise `<`/`>` around `world_model`, `hypothesis`, `level_note` and `dropped` (or escape the element body).

7. **Low: the ACK puts every pending change under the first change's act.**
   - Where: `ack_text`, `memory_items.py:650`.
   - Reproduced: changes made at act 3 and act 9 report as `act 3: +m3 world_model; +m4 world_model`. This happens when a reply with tags but no tool call carries its changes over to a later tool result.
   - Fix: start a new `act N:` group whenever the act changes.

8. **Low: the read-out tool drops the run-level files when per-game files exist.**
   - Where: `inference/tools/memory_log.py:56-61`, which uses `memory_log.jsonl` and `requests.jsonl` only when no per-game file exists.
   - Effect: the first game's revisions and requests written before a second state file appears are skipped. Departure 10's claim that "a game that starts before the others still gets its own file" is therefore inaccurate.
   - Fix: always include the run-level file (report it under "run", or attribute it from the request rows), and correct the departure note.

9. **Informational:**
   - Python's `memory_log` (the newest 50 revisions) includes harness revisions for solved, hazard and first-effect (fx) facts, which can crowd out the model's own. Consider filtering `source == "harness"` (except cap drops) out of the sandbox view.
   - When a reply has several tool calls and an earlier one executes an action, the later calls are skipped and their `memory` changes are silently lost. This matches existing behaviour; noting it only.

My probe scripts are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/review-memxml/`:
- `probe1.py`, `probe2.py`
- `cmp_golden.py`
- `spot3b.py`
- `dumpxml.py`, with its output in `dump_xml.json`
