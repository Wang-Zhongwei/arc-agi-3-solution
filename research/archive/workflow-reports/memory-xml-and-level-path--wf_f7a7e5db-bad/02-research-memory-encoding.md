# memory-xml-and-level-path: research:memory-encoding

Workflow run `wf_f7a7e5db-bad`, phase Research. Research, design, build, review and integrate two arms on memory-v2: memory writes in the Qwen3 XML tool-call format with a memory addendum and revision log, and a shortest-known-path tracker shown after every action

---

Research A report: how the model should write memory, measured on old runs and our ledger runs, plus a recommended encoding and item model.

**Headline**
- No recorded Kaggle run ever used the `qwen3_xml` parser. Every vLLM server log in the 54 old runs says `tool_call_parser': 'qwen3_coder'`.
- `qwen3_xml` appears only in the local HPC vLLM configs, for example `wang-duck-harness/ARC3-Inference/configs/inference.flash-next-intel-w4a16.json:68`. The Kaggle bundle overrides it to `qwen3_coder` at line 146, and Wang's own Kaggle default is `SERVER_TOOL_CALL_PARSER="qwen3_coder"` (`inference/framework/kaggle.py:257`).
- Both parsers read the same wire format: `<tool_call><function=python><parameter=K>…</parameter></function></tool_call>`. vLLM's `qwen3_xml` uses expat, escapes special characters and auto-closes parameters. `qwen3_coder` uses a regex. SGLang only has `qwen3_coder`.
- So "encoding for qwen3_xml" can only mean writing memory in this XML wire format, which is what the interpretation assumed.
- The data does not show the tags working well. The model wrote 0.38 tags per 100 calls in Wang's harness and 2.6 per 100 in ours.
- What it does show is that the model already puts memory into the tool call by itself: it invents memory parameters, and in Wang it wrote Python `store` on 8% of calls.

## (1) How the old Wang harness let the model write memory
1. **XML tags anywhere in the reply.** Defined in `memory_ledger.py:37-58` and taught in `prompts.py:200-211` (`MEMORY_LEDGER_ADDENDUM`). The tags are `<world_model>`, `<hypothesis status=confirmed|refuted|open>`, `<level_note>`, and an optional `<scratchpad>` (`prompts.py:213-218`, on only in `taaf-g1`).
   - The harness absorbs tags from both the reply text and the thinking, and also folds a think-fragment (`tool_agent.py:2207-2223`).
   - It renders a `<memory>` block at the end of the current user turn (`tool_agent.py:2461-2468`).
   - It removes old copies from history with regex (`tool_agent.py:961-1010`, `_STALE_PROMPT_SECTION_RE`). That is an edit to history already sent, which #58 forbids. memory-v2 does not do this.
2. **`remember(k, v)`, `recall`, `forget` and a persistent `store` dict.** Defined in `python_tool_sandbox.py:1101-1112` and `1216-1218`, and advertised in the tool description (`tool_agent.py:355`). They keep Python state only; the model never sees `store` in its prompt.
3. **`update_memory(world_model=…, plan=…, …)` was dead code.**
   - It is defined at `python_tool_sandbox.py:1055-1093` and sends `{"type":"memory"}`. The host side is at `python_tool_sandbox.py:1565-1586`.
   - But `tool_agent.py:2962-2967` calls `self._sandbox.run(...)` without passing `memory_handler`, so every call would return "update_memory is not available in this session."
   - It was never in any prompt or the tool description. Zero calls appear in any run.
4. **Tool schema** (`tool_agent.py:2471-2516`): `python(code: string, required; timeout_seconds: integer, optional, 1..max)`. There is no memory parameter.

## (2) How often each channel was used, per 100 model calls
Scripts are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/memv3-research/` (`r0_flags.py`, `r2_channels.py`, `r4_extra_params.py`, `memA_trunc.py`, `memA_regex.py`). Results are in `memA_channels.json` and `r2_all.txt`. The prompt's own example tags are excluded from the counts.

| Group | Calls | Tags per 100 calls (think / reply / code) | Calls with a tag | Game-runs with no tags | Python `store` writes | "Will record" intent per 100 |
|---|---|---|---|---|---|---|
| Wang, ledger on (21 taaf runs) | 28,339 | 0.38 (0.06 / 0.31 / 0.01) | 0.15% | 427/457 | 8.3% of calls (`remember` 8.1/100, `store[]=` 5.6/100) | 0.89 |
| Wang, ledger off (0922-v2, w4a16-explore) | 2,509 | 0 | 0 | 50/50 | 8.45% | 0.08 |
| Ours, ledger on (5 fz-ledger runs + v4-stack-v1) | 13,207 | 2.60 (0.30 / 1.65 / 0.65) | 1.15% | 92/150 | about 0 (no API for it) | 4.31 |
| Ours, control (fz-arms v4/v5, fz-ucb, fz-streams14) | 9,685 | 0 | 0 | 100/100 | 0.01% | 0.10 |
| Duck (23 runs, prose "World model:" notes instead) | 41,117 | 0 | n/a | n/a | 0.03% | 0.19 |

- The rows above reproduce the memory card's r1 and r2 numbers: 92 and 104 tags, matching 66+26 and 88+16.
- **Reply text is almost never present in our harness.** Assistant text appears in 1.16% of our calls against 31.5% in Wang. Wang had the text and still wrote 0.31 tags per 100.
- A tool call is present in about 99% of calls in both harnesses (no-tool-call rate 0.1–1.3%). So the tool call is the one place the model writes on nearly every turn.
- **The model invents memory parameters on its own.**
  - Old runs: 39 tool calls carried a memory-like extra parameter, out of 77,875 calls. Names: `remember` 24, `world_model` 7, `store_update` 3, `memory` 2, plus `key`/`value`, `hypothesis`, `plan`, `action_model`, `def_note`, `remember_me`.
  - 25 of those 39 had no `code`, so the whole call came back as "python requires a non-empty `code` string" and the memory was lost.
  - Examples (from `r3_params.py`): taaf-b4 ar25 `<parameter=world_model>`, with a stray `</world_model>` inside the value, followed by code; final-c1 ar25 `<parameter=remember>`, `<parameter=key>mech`, `<parameter=value>…` with no code.
  - Ours: 1 such call in 22,892 (`world_model` plus code, fz-ledger-streams12 ls20).
  - Unrelated junk parameters (`python`, `command`, `=python`) appeared 38 times in old runs and 28 in ours, mostly as empty calls.
- **The model handles an optional second parameter.** `timeout_seconds` was used 0.36 times per 100 Wang calls, and those two-parameter calls parsed normally.
- **About 25% of our tags are written inside Python code** (0.65 of 2.60 per 100). memory-v2 already removes them from the code before it runs (`tool_agent.py:4422-4443`).
- **Truncation at the 12k output cap is rare and usually inside thinking.** `finish_reason=length` happens about 1 time per 100 calls. In 104 of 110 such replies (r1, r2, v4-stack-v1, fz-ucb) there was no tool call at all; mean reasoning was about 35k chars.

## (3) Our harness: schema, argument parsing and what the parser does
**Our harness (memory-v2 worktree)**
- **Schema:** `tool_agent.py:5446-5467` defines `python(code: string, required)`, with the description at `tool_agent.py:409-420`.
- **Argument parsing:**
  - The server returns arguments as a JSON string. `tool_agent.py:7166-7200` drops every call in the reply if any arguments string is invalid JSON.
  - `tool_agent.py:7376-7388` runs `json.loads` on the arguments.
  - Dispatch goes through `_dispatch_tool` (`tool_agent.py:6233-6236`), then `_mem_absorb_code`, then `_run_python_tool`, which reads only `arguments["code"]` (`tool_agent.py:5989`).
  - Empty code returns the error at `tool_agent.py:5991`.
  - The calls also run when the reply was cut off (`finish_reason=length`); the harness only lowers reasoning effort (`tool_agent.py:7141-7156`).
- **Unknown parameters are silently ignored.** Nothing checks them against the schema, but they stay in `assistant_message["tool_calls"]` in history.
- **Absorption:** tags are absorbed from both content and reasoning (`tool_agent.py:7217-7223`). The tag prompt text is at `prompts.py:23-45`.
- **Requests are non-streaming** for the local server; streaming is only on for openrouter (`utils/openai_compat.py:57-80`). So SGLang uses its one-shot `detect_and_parse` path.

**SGLang `qwen3_coder` parser**
- Source: `competitors/sglang-rtxpro6000/python/sglang/srt/function_call/qwen3_coder_detector.py`. It is byte-identical to the Pennyroyal pin d00d88ef (git diff of `function_call/` is empty).
- **Parameter split:** the regex `<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)` (lines 35-38, 252-266). Two `<parameter>` blocks parse in either order.
- **Whitespace:** one leading and one trailing newline are stripped from each value (lines 260-263).
- **Undeclared parameter names** are passed through as raw strings, with a server-log warning (lines 135-140).
- **Undeclared function names** go into normal text instead of a call (lines 241-247).
- **Types:** string values come back raw, and any value equal to `null` becomes None. Object and array types go through `json.loads`, then `literal_eval`, then fall back to a string (lines 127-202).
- **Wrapper span:** `<tool_call>` nesting is tracked by depth (lines 302-321). An earlier Markdown fence owns any tool call inside it (lines 346-380).
- **Incomplete calls are parsed.** If there is no complete `<tool_call>` wrapper, the incomplete one is still parsed (lines 211-216).
- **`serving_chat.py:2092-2150`:** the parser runs whatever the finish reason was, and `length` is kept as the finish reason.
- **`serving_chat.py:145-160`:** before rendering the template, SGLang turns history argument strings back into dicts, so earlier calls are re-rendered as `<parameter=k>` blocks in insertion order.

**Replay of that regex on memory cases** (`memA_regex.py`)

| Case | Result |
|---|---|
| Code then memory | Both parameters parse |
| Memory then code | Both parameters parse |
| Memory not closed, then code | Both parameters parse |
| Memory text containing a literal `</parameter>` | Memory is cut at that point |
| Truncated inside memory | Code is complete, memory is partial |
| `<parameter name="code">` (attribute syntax) | Swallowed into the previous parameter, so code is lost (happened once, final-c1 ka59) |
| XML tags or `<` inside the memory value | Kept as text |

**Chat template**
- The competition passes `MODEL_DIR/chat_template.jinja` (`kaggle/template.ipynb` cell 12). That file is not stored anywhere locally, so I could not check its exact wording.
- The reference Qwen3.8 template I did find (`competitors/sglang-rtxpro6000/configs/pennyroyal/templates/froggeric-v22.5.jinja`, a fixed copy of Qwen's):
  - At lines 167-209 it renders each tool as one `tool | tojson` line inside `<tools>`, in the system turn, followed by an XML call example with two parameters and the rule "NO conversational text before tool_call".
  - At lines 376-379 it re-renders past calls from `arguments.items()`.
- Consequences:
  - The parameter schema lives in the cached system prefix, so it must stay the same on every request; with the flag on it is present from request 1.
  - A `memory` parameter the model wrote is re-rendered from its own `tool_calls` on every request, so it stays cache-stable as long as the harness never edits stored calls.

## (4) Recommended encoding
**Primary: an optional `memory` parameter (type string) on the `python` tool, behind a flag, with the schema fixed for the whole run.** The value is a few lines, each one operation:

```
+ wm: TEXT      add a world-model item
+ hyp: TEXT     add an open hypothesis
+ note: TEXT    add a level note
~ ID: TEXT      replace an item's text
ok ID: TEXT     mark an item confirmed
x ID: TEXT      mark an item refuted
- ID            drop an item
```

- A line with no operation is stored as a note.
- An existing tag inside the value (`<world_model>…`) is accepted and mapped the same way as the tag channel.

Why this one:
- It is the only channel present in about 99% of calls; reply text is in 1.2%.
- It matches what the model already does: it invented memory parameters 39 times, wrote 25% of our tags inside code, and wrote Python `store` on 8.3% of Wang calls versus 0.15% of calls with a tag.
- It cannot break the code. 42 tags were lost inside code in r1+r2, one of them as a SyntaxError.
- XML parameter values need no escaping, unlike JSON or Python strings.
- The value arrives as a parsed field, so `arguments["memory"]` is deterministic.

Risks and how to handle them:
- **Memory without code.** 25 of the 39 spontaneous cases had no code. With the flag on, store the memory and return a tool result that says so and says no code ran, instead of the empty-code error.
- **Literal `</parameter>`, `<parameter=` or `</function>` in the text.** The value is cut at that point. Store what parsed and state the cut in the tool result.
- **Attribute-style `<parameter name=…>`.** Existing failure, rare (1 case).
- **Truncation.** Rare (about 1% of calls, 95% of those without any call). Put memory after code in the example, and when `finish_reason=length`, drop the last memory line.
- **Model ignores it.** Tags with an explicit prompt reached only 2.6 per 100 calls. Score the arm on write rate per 100 calls.
- **Prefix cost.** Roughly 80–120 extra prompt tokens, paid once and cached.
- **Strategy wording.** The parameter description must only say how the field works. Example: "Optional. Each line changes one memory item: … The code runs whether or not this is present; the result lists the item ids and changes."

Channels to keep or drop:
- Keep the tags in reply and thinking, and the tags-in-code stripper, as secondary sources feeding the same operations.
- Keep the think-fragment fold off. 60% of ledger entries were fragments, and 61% of those start with filler.
- Don't build `update_memory()` (it was dead code) or a separate memory tool (an extra call and an extra tool in the schema).
- Expose read-only `memory_items` and `memory_log` in the sandbox. They cost no prompt tokens.

## (5) Item model for the revision log
- **Item:** `id` (model items `m1, m2, …`, numbered per game and never reused; harness items keep stable ids such as `fx:UP`, `solved:L2`, `hazard:k`, `path`), `kind` (wm, hyp, note; harness kinds effect, solved, hazard, seen, path), `status`, `text` (capped per kind: wm 400, hyp 250, note 200, as in `CARD_ENTRY_CHARS`), `level`, `created_rev`, `updated_rev`, `n_revs`, `last_source`, `last_action`.
- **Status by kind:** wm is active or dropped; hyp is open, confirmed, refuted or dropped; note is active or dropped and belongs to one level.
- **Revision record:** `rev` (the existing ledger version counter), `item_id`, `op` (add, revise, confirm, refute, drop, plus harness set/update/evict), `source` (param, tag_reply, tag_think, tag_code, harness), `call_idx`, `action_idx`, `level`, `before` (status and first 60 chars), `after`, `reason` (up to 80 chars).
  - Keep the full log for the game, capped at about 400 revisions, in the ledger journal. Expose it as `memory_log`.
- **Mapping the tag channel onto operations:**
  - A confirmed or refuted hypothesis tag whose word overlap with an open item is at least 0.6 becomes a confirm or refute of that id.
  - A `Correction:` prefix becomes a revise of the best match at 0.35 or more.
  - Anything else becomes an add.
  - Cap evictions are logged as `drop` with source `harness`.
- **Showing history compactly (#58 compliant):**
  - **This turn's tool result:** one line of at most 200 chars, for example `memory rev 14 a57 L3: +m7 wm | m3 hyp open→confirmed | ~m5 wm (rev 2) | cut: 1 line`. This is the only place new ids are shown when they are created.
  - **Card after a real trim, and deltas in the new turn message:** `m3 hyp confirmed r2 a41: TEXT`. Older text is not repeated; full history stays in `memory_log`.
  - **Revised or refuted items in deltas:** `m5 wm r2 a57 (was: first 40 chars…)`.
