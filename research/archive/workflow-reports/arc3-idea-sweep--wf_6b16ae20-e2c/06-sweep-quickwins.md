# arc3-idea-sweep: sweep:quickwins

Workflow run `wf_6b16ae20-e2c`, phase Sweep. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "ideas": [
  {
   "title": "Make the per-action state refresh scale with new history entries instead of the full game history",
   "category": "bugfix-quickwin",
   "description": "Every action currently does O(full game history) JSON work, and it all holds the GIL. The fix keeps behavior identical. (a) Drop indent=2 in write_runtime_state (runtime_state.py:281); the file is 12.7MB at 200 entries. (b) solver.py:1876 calls write_runtime_state() inside _execute_action, so it runs once per action even inside a single batched action([...]) call. Write once at the end of step_env instead (solver.py:1045 already writes before each analyzer turn). (c) Delete the unused full reload at tool_agent.py:7936 (its own Note says the values are unused). (d) In _serialized_runtime_state (tool_agent.py:7939-7982), stop re-reading the file and re-serializing every history frame after each action. Keep the solver's in-memory history, cache the serialized dict per HistoryEntry (entries are immutable), and send the child only the entries appended since its last refresh. The child's _refresh_state (python_tool_sandbox.py:419-447) would append rather than rebuild. (e) _ascii_frame_view_payload (tool_agent.py:4038-4057) sends both 'ascii' and 'grid' for every frame; send only grid and build ascii lazily in the child FrameView. Check by replaying a recorded snippet and diffing sandbox globals.",
   "mechanism": "I benchmarked one action() round trip (write, read, rebuild, dump, child parse) on realistic 64x64 frames. It cost 0.10s at 50 history entries, 0.40s at 200, 0.99s at 500 and 2.0s at 1000, plus 0.16-0.40s per action inside a batch for the redundant write alone. User runs have a median of about 190 actions per game (p75 about 300, max about 545). Late in a game, each action call therefore burns 0.5-1s+ of host CPU. During that time the game holds one of only 10 admission slots and its GPU stream is idle, and all 10 game threads contend for one GIL. Removing this raises GPU utilization (more turns in the fixed runtime). It also removes the main cause of 30s timeouts in per-action loops: tr87 executed only 12 actions in 30s at step 279.",
   "evidence": [
    "ARC3-Inference/inference/agent/runtime_state.py:281: json.dumps(payload, indent=2) of the full history on every write",
    "ARC3-Inference/inference/framework/solver.py:1876: write_runtime_state() runs per executed action, including each action of a batch",
    "ARC3-Inference/inference/agent/tool_agent.py:7935-7936: Note says the full load_runtime_state result is unused",
    "ARC3-Inference/inference/agent/tool_agent.py:7957: every action() reloads and re-serializes the whole history for the sandbox",
    "ARC3-Inference/inference/framework/solver.py:1889: each game runs on a worker thread in one process (GIL contention)",
    "Local benchmark (scratchpad copy): N=200 0.40s, N=500 0.99s, N=1000 2.0s per round trip; file 31.6MB, pipe 8.4MB at N=500",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs/fz-streams14/transcripts/tr87-cd924810_p0.txt:16302: loop of single UP actions timed out after 30s with only 12 actions executed",
    "benchmark.json in fz-arms-v4-control/fz-streams14: median about 188 actions per game, p75 about 300-320, max about 545"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Persistent JSON 'memo' store across python calls (data, not just functions)",
   "category": "tools-repl",
   "description": "Add a `memo` dict global to the sandbox (python_tool_sandbox.py runtime_globals at ~395). The host restores it at each call start and receives it back in both the 'final' and 'error' messages (python_tool_sandbox.py:772-800), JSON-safe and capped at about 200KB. Store it host-side next to _kept_functions (tool_agent.py:6497-6571, 8161-8179), scoped to the game like retained functions, and clear or flag it at level change. Document it in PERSISTENT_FUNCTIONS_LINE (prompts.py:355-365), e.g. 'store computed maps, plans and search results in memo[...]'. Optional: auto-restore small JSON-serializable top-level variables and list them the way retained functions are listed. Then retained functions that read memo stop being rejected for unavailable globals.",
   "mechanism": "The model keeps trying to persist computed state across calls. User runs show 302 NameErrors on the model's own earlier variables or functions (e.g. 'sols', 'plan', 'CELLS'), 30 'open' NameErrors, and blocked imports of pickle (10) and a nonexistent 'store' module (4). Each failure wastes a full LLM round trip. Without persistence, the model also regenerates code that recomputes expensive maps and searches on every call. Franzen's retained functions saved generation; persistent data is the missing half.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs/fz-*/transcripts: 376 NameErrors total, of which 302 reference model-defined names, 45 blocked builtins (open 30, globals 10), and 27 stdlib names",
    "fz-* transcripts: 'Module 'pickle' is not allowed' x10, 'Module 'store' is not allowed' x4",
    "ARC3-Inference/inference/agent/prompts.py:355-365: only functions are retained; 'Python variables reset between tool calls'",
    "ARC3-Inference/CONFIGURATION.md section 7: function retention only, explicitly not a persistent kernel"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Send the static per-turn instruction block once per context window",
   "category": "memory-context",
   "description": "_build_user_prompt appends about 10 fixed lines to every opener and resume (tool_agent.py:7096-7157): 'Only tool: python...', 'Only letter-coded...', 'Keep tool output compact...', 'For the most recent change...', 'Use Python to inspect...', 'Focus on what changed...', 'When ready, call action...', 'You may call action more than once...', TOOL_CALL_FORMAT_GUIDANCE, and 'If you use MOUSE...'. Most of this restates the system prompt. _retained_function_context (tool_agent.py:6461) also appends the full retained-signature list every turn. Emit the static block only in the first opener after a context drain or level change (track with a flag set in _note_history_evicted, tool_agent.py ~8700) and otherwise use a one-line pointer. List retained functions only when the set changed or after a drain. Gate behind an env flag for A/B.",
   "mechanism": "In competition-config transcripts, the static block has a median of 1,725 chars and makes up 62% of all user-prompt characters. The retained list adds 13% (p90 787 chars, max 4,375). User prompts are about 22% of all transcript text. Cutting most of that slows context growth by roughly 10-15%, so the 116k-to-58k drain fires less often: fewer full re-prefills and more history kept per window. Franzen found more retained context helped hard games. It is prefix-safe because only new messages change.",
   "evidence": [
    "ARC3-Inference/inference/agent/tool_agent.py:7096-7157: static lines appended to every opener",
    "ARC3-Inference/inference/agent/tool_agent.py:6461: retained-function list appended to every user message",
    "Measured on fz-arms-v4-control, fz-ucb and fz-streams14 transcripts: 4,932 user prompts, avg 2,761 chars; static block median 1,725 chars = 61.9% of user-prompt chars; retained list = 13.0%",
    "WRITEUP.md section 2: larger context helped difficult games; section 6: shorter *resume* prompts showed no consistent gain (opener boilerplate not reported)"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "Don't wipe all retained functions when a tool call times out or the sandbox crashes",
   "category": "bugfix-quickwin",
   "description": "The sandbox's timeout, crash and invalid-response returns (python_tool_sandbox.py:1001-1011, 1020-1040) contain no 'keepable_functions'. _record_retained_functions then iterates `sandbox_result.get('keepable_functions') or []` (tool_agent.py:6532), builds kept={}, replaces the pool (tool_agent.py:6566) and emits 'Your previous function X is no longer retained' for every helper. Fix: if 'keepable_functions' is not in sandbox_result, return early and leave self._kept_functions unchanged, optionally adding a note that retained functions were kept because the call did not finish. About a 5-line change.",
   "mechanism": "Game-scope function retention is one of Franzen's token-saving wins, and helper libraries are largest in late, high-weight levels. One timeout currently drops the whole library. In tr87 (level 6, step 279), about 40 retained helpers (read_state, codebook, go_to, ...) disappeared after one 30s timeout, and the next reply started redefining them (4.8k reasoning chars plus code). Timeouts occur in about 0.3% of calls (2-12 per 25-game run), concentrated where regeneration is most expensive.",
   "evidence": [
    "ARC3-Inference/inference/agent/python_tool_sandbox.py:1001-1011: timeout return has only error/stdout/action_results",
    "ARC3-Inference/inference/agent/tool_agent.py:6532,6563-6566: missing key is treated as 'keep nothing' and the pool is replaced",
    "Verified locally: a timed-out call returns keys ['action_results','error','stdout'] with keepable_functions=None",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs/fz-streams14/transcripts/tr87-cd924810_p0.txt: before the timeout the prompt lists about 40 retained functions; after it the list is gone and the model redefines glset/boxes6...",
    "fz-* transcripts: 51 'Tool timed out after 30s' in about 17.4k tool calls"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Make tool timeouts soft and non-destructive (keep stdout/actions, don't charge host time, allow >30s)",
   "category": "tools-repl",
   "description": "(a) In the child bootstrap, arm signal.setitimer(ITIMER_REAL, timeout-2) with a handler that raises a new ToolTimeout(BaseException), and add it to the except tuple at python_tool_sandbox.py:778-800. The child then sends a normal 'error' message carrying partial stdout, action_results and the retention payload. The host deadline (python_tool_sandbox.py:1001) stays as a hard backstop. (b) Extend the host deadline by the time spent inside action_handler/animation_handler (python_tool_sandbox.py:1063-1095), so the game's own step latency isn't charged to model code (Note at :906). (c) Remove the hardcoded min(30, ...) at tool_agent.py:5131, make prompts.py:378 format the real value, and try 45-60s. (d) Expose a `time` shim (perf_counter/monotonic) so searches can stop at a deadline and return their best plan so far.",
   "mechanism": "About 23 of the roughly 50 transcript timeouts happened in snippets that executed actions. Those actions were spent, but the model received only 'Tool timed out after 30s' with empty stdout (python_tool_sandbox.py:1008-1010). It lost the observations its code printed and didn't know where its loop stopped, which risks more wasted actions under a quadratic action penalty. The other timeouts were pure-compute searches that a deadline-aware search could have completed. 'import time' was blocked 47 times, which shows the model wants exactly this. Each timeout costs at least one extra round trip.",
   "evidence": [
    "ARC3-Inference/inference/agent/python_tool_sandbox.py:906-908: Note says host action time counts against the snippet's wall-clock limit",
    "ARC3-Inference/inference/agent/python_tool_sandbox.py:1001-1011: on deadline the child is killed and stdout is ''",
    "ARC3-Inference/inference/agent/tool_agent.py:5131: self._python_timeout = min(30, ...), so LOCAL_ANALYZER_TOOL_TIMEOUT cannot raise it",
    "ARC3-Inference/inference/agent/prompts.py:378: hardcoded 'hard time limit of 30 seconds'",
    "fz-* transcripts: of timed-out calls, 23 had step_executed True (19 contained action( calls); about 26 were compute-only",
    "fz-* transcripts: 'Module 'time' is not allowed in the sandbox' x47"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Fix the dead 'reply was CUT OFF' signal and the bogus resume text after truncation",
   "category": "bugfix-quickwin",
   "description": "A length-truncated reply (12,288 tokens) always exceeds LOCAL_ANALYZER_YIELD_TOKENS=2048. So control_yield_reason() at tool_agent.py:9566 yields before the purpose-built nudge at :9569-9578 ('Your previous reply was CUT OFF... keep your reasoning much shorter') is ever sent. The next resume prompt from _stale_summary_lines (tool_agent.py:536-563) instead says 'Your last python snippet executed NO actions (it only computed or inspected state - that is fine)' or 'produced no tool call'. Fix: set self._resume_reason='truncated' (or a flag) when finish_reason=='length', and lead the resume prompt with the CUT-OFF sentence. Also, when ARC3_MEMORY_SECTIONS=off, strip the memory-section sentences from the follow-up nudge (tool_agent.py:9605-9622). With sections off, _advertised_section_labels() still returns all seven, so the nudge tells the model to write `World model:`...`Cross-level notes:` sections that are never stored, and to 'restate what is still true'. Franzen's own comment at :7106-7116 measured exactly this kind of broken WM instruction hurting replies, but only fixed it in the opener.",
   "mechanism": "184 of 17,406 replies (1.06%) hit the output cap, and they hold 7.3% of all generated reasoning characters. 10 of the 184 were immediately followed by another truncation. Across every user run, the CUT-OFF text appears 0 times and the generic nudge once, so the model never learns that length, not formatting, killed its call. Telling it should shorten the next deliberation. The memory-section text would otherwise invite long section dumps exactly when the model should be brief.",
   "evidence": [
    "ARC3-Inference/inference/agent/tool_agent.py:9566: yield check runs before the follow-up nudge is built",
    "ARC3-Inference/inference/agent/tool_agent.py:9569-9578: CUT-OFF text (unreachable under yield_tokens=2048)",
    "ARC3-Inference/inference/agent/tool_agent.py:536-563: resume text claims a snippet ran or that there was just no tool call",
    "ARC3-Inference/inference/agent/tool_agent.py:9605-9622 and 1870-1889: follow-up nudge advertises seven memory sections even with ARC3_MEMORY_SECTIONS=off (verified: _advertised_section_labels() returns all 7 under the competition env)",
    "fz-arms-v4-control transcript: after 'finish_reason: length' -> 'Yielded control to solver: turn_token_budget' -> 'Your last python snippet executed NO actions...'",
    "fz-* transcripts: 0 occurrences of 'Your previous reply was CUT OFF'; 184 'finish_reason: length' replies"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Cap overlong deliberations earlier (max_output 12k to about 8k, or a harness-side thinking budget)",
   "category": "model-config",
   "description": "A/B LOCAL_ANALYZER_MAX_OUTPUT=8192 against 12288. Truncated reasoning is kept in context (tool_agent.py:9500-9566) and the turn resumes, and reply_reserve equals max_output, so this also frees about 4k tokens of context budget. A better variant (about 1 day): stream the request, count reasoning tokens client-side, and at a budget of about 7k abort the stream. Then re-request with the partial reasoning kept plus a one-line 'stop deliberating and emit the python call now'. Do this client-side because SGLang's built-in thinking_budget / Qwen3ThinkingBudgetLogitProcessor hardcodes </think> token ids and was reported not enforcing on Qwen3.6-family tokenizers. Combine with the truncation-aware resume prompt idea.",
   "mechanism": "Truncated replies are not degenerate repetition: zlib ratio median is about 0.33, the same as normal reasoning. They are a distinct long-deliberation mode. Successful replies thin out about 2x per 5k chars (79 at 30-35k, 15 at 35-40k, 2 at 40-45k), yet 184 pile up at the 12,288-token cap, and 173 of 184 are followed by a successful tool call. Cutting at about 8k saves about 4k generated tokens per truncation, roughly 3% of all generation. The cost is that about 100 replies that would have finished between 8k and 12k now need one continuation request, with their reasoning still in context.",
   "evidence": [
    "fz-* transcripts: 184/17,406 finish_reason=length; truncated replies = 7.34% of reasoning chars (6.41M / 87.3M)",
    "fz-* transcripts: successful replies by reasoning chars: 25-30k: 158, 30-35k: 79, 35-40k: 15, 40-45k: 2",
    "ARC3-Inference/inference/agent/tool_agent.py:9500-9566: reasoning-only truncated replies are appended to context",
    "https://github.com/sgl-project/sglang/issues/25536: thinking_budget not enforced for Qwen3.6 in SGLang 0.5.11-0.5.12; Qwen3ThinkingBudgetLogitProcessor hardcodes token ids 151667/151668 vs the model's 248068/248069",
    "https://docs.nvidia.com/nim/large-language-models/1.15.0/thinking-budget-control.html: budget-forcing mechanism (count thinking tokens, then force the early-exit string)"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Stop re-sending duplicate context on yield-resumes (image off, short resume, higher yield threshold)",
   "category": "memory-context",
   "description": "Config sweep with no code changes: ARC3_RESUME_PROMPT_IMAGES=0, ARC3_YIELD_RESUME_PROMPT=short, and LOCAL_ANALYZER_YIELD_TOKENS at 2048 vs 6144 vs 0. A resumption only happens when nothing executed (tool_agent.py:9830), and the knob's own docstring says the re-attached board 'is identical to the one the retained opener already showed' (tool_agent.py:1249-1258). The 'full' resume mode re-sends the whole opener text (tool_agent.py:9213-9245). Optionally also try ARC3_PRUNE_RESUME_PROMPTS, but note that pruning changes the cached prefix.",
   "mechanism": "31% of analyzer turns end in a turn_token_budget yield: 3,690 yields vs 8,033 executed turns in the user runs. Each yield appends roughly 750 text tokens plus a duplicate grid image (about 402 estimated tokens), around 4M context tokens over 8 runs. That pushes the 116k-to-58k drain sooner, costing history and a full re-prefill each time.",
   "evidence": [
    "ARC3-Inference/inference/agent/tool_agent.py:1249-1260: ARC3_RESUME_PROMPT_IMAGES default True, though the docstring says the board is identical",
    "ARC3-Inference/inference/agent/tool_agent.py:9213-9221: resume_frame attaches the grid image unless the knob is off",
    "fz-* transcripts: 'Yielded control to solver: turn_token_budget' x3,690 vs 'Step executed.' x8,033",
    "fz-arms-v4-control and fz-ucb: 4,927 images across 3,094 user prompts (1.6 per prompt)",
    "WRITEUP.md section 6: shorter resume prompts, tones and 'removing some reinsertion' were not consistently better; image-off and yield threshold values are not reported"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "Close sandbox allowlist/builtin gaps (class, globals, time, recursion, pre-imports, retained_source)",
   "category": "tools-repl",
   "description": "SAFE_BUILTINS (python_tool_sandbox.py:107-176) lacks __build_class__, so every `class` statement fails with 'NameError: __build_class__ not found'. It also lacks object, super, globals, id, setattr, vars, property, staticmethod and classmethod. SAFE_MODULES (:85-103) lacks time, dataclasses, typing, array and enum, and the default recursion limit of 1000 kills recursive DFS (I verified depth 3000 fails, and sys is blocked). Fix: add those builtins (the child is already isolated with -I -S and rlimits); add a `time` shim; call sys.setrecursionlimit(20000) in the child before exec; pre-bind deque, defaultdict, Counter, heapq, itertools, json, math, re, functools and collections in runtime_globals; add retained_source(name), which returns the stored source of a retained function. Then update the module list in prompts.py:253, which also omits difflib (Note at python_tool_sandbox.py:33).",
   "mechanism": "Each blocked import or NameError kills the whole snippet and costs a round trip. The user runs show about 114 ImportErrors (time 47, inspect 32, sys 16, pickle 10, builtins 7, ...) and about 72 NameErrors on blocked builtins or un-imported stdlib names (open 30, globals 10, deque 7, json 7, itertools 5, ...). The 32 `import inspect` attempts show the model wants to read its retained helpers' code after it has scrolled out of context.",
   "evidence": [
    "ARC3-Inference/inference/agent/python_tool_sandbox.py:85-176: SAFE_MODULES / SAFE_BUILTINS",
    "ARC3-Inference/inference/agent/python_tool_sandbox.py:313-317: _safe_import raises ImportError for anything else",
    "Local test: class -> 'NameError: __build_class__ not found'; globals/object NameError; 'import time' ImportError; recursion depth 3000 -> RecursionError",
    "fz-* transcripts: blocked imports time 47, inspect 32, sys 16, pickle 10; NameError open 30, globals 10, deque 7, json 7",
    "ARC3-Inference/inference/agent/prompts.py:253: module list shown to the model"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "Fix system-prompt text bugs (merged bullet, unbulleted paragraph, glued sentences, grammar)",
   "category": "prompting",
   "description": "I rendered _build_system_prompt with the exact competition env. (1) prompts.py:132 lacks a trailing '\\n', so the competition prompt reads '...Sometime they might also be 1x1 tokens.- Some games are logic or layout puzzles with no explicit player avatar...'. Add the newline and fix the grammar ('are usually be rendered', 'Sometime'). (2) prompts.py:136 'A common failure mode...' has no '- ' bullet. (3) tool_agent.py:3822-3828 appends the diff-image explanation with a leading space after the Multimodal bullets, so it renders as an unbulleted ' Alongside the current grid image...' line; the same applies to the auto-diff sentence at :3814. Make both proper bullets.",
   "mechanism": "The 'no explicit player avatar' rule matters for logic/layout games, and it is currently buried mid-line after a typo'd bullet. Clean bullets improve instruction salience at zero runtime cost. The system prompt is constant for the whole run, so prefix caching is unaffected.",
   "evidence": [
    "ARC3-Inference/inference/agent/prompts.py:127-133: Note says the 'Game entities' line lacks a newline and merges into the next bullet",
    "ARC3-Inference/inference/agent/prompts.py:136: paragraph without bullet marker",
    "ARC3-Inference/inference/agent/tool_agent.py:3812-3828: sentences appended with a leading space, not as bullets",
    "Rendered competition system prompt (env from /Users/advaith/Documents/studying-stuff/arc-agi-3-v3/external/dfranzen/nb.txt:10175-10260) shows '...1x1 tokens.- Some games are logic...'"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Remove contradictions between the tool schema/prompts and the competition config",
   "category": "prompting",
   "description": "(a) With ARC3_PERSISTENT_FUNCTIONS=1, the tool description still says 'Run one ephemeral Python snippet' (tool_agent.py:420-431) and the `code` parameter says 'The snippet is ephemeral and is not saved across tool calls' (tool_agent.py:7199-7201). The system prompt says functions are retained for the whole game. When retention is on, say 'Variables reset; top-level functions you define are retained', and add frame_diff and last_animation_frames to the description. (b) With ARC3_MEMORY_SECTIONS=off, the prompts still say 'update your working world model in Python' (prompts.py:119), 'Record the diagnosed cause in your world model' (tool_agent.py:6983) and 'record the correction in your world model' (tool_agent.py:7034). Reword these to 'state it in your reasoning'. (c) Keep prompts.py:378 in sync with the real timeout.",
   "mechanism": "Tool-schema text sits right next to the call, so telling the model the code 'is not saved' pushes it to redefine helpers, which wastes generation that function retention was meant to save. Franzen himself measured that a broken world-model instruction with memory off dropped replies carrying assistant text from 76% to 43% (comment at tool_agent.py:7106-7116). These are leftover instances of the same problem.",
   "evidence": [
    "ARC3-Inference/inference/agent/tool_agent.py:420-431 and 7199-7201: 'ephemeral', 'not saved across tool calls'",
    "ARC3-Inference/inference/agent/prompts.py:355-365: game-scope retention rule in the system prompt",
    "ARC3-Inference/inference/agent/prompts.py:119; tool_agent.py:6983, 7034: world-model references while ARC3_MEMORY_SECTIONS=off",
    "ARC3-Inference/inference/agent/tool_agent.py:7106-7116: Franzen's note on the 76%->43% drop from a broken WM instruction"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Show 'level L of N' and actions spent on this level in the per-turn state line",
   "category": "prompting",
   "description": "The state line (tool_agent.py:7073-7077) shows only the global step and the level. Its 'out of observed max level' branch is dead: observed_max_level comes from history frames, whose levels can never exceed current_level (tool_agent.py:6812-6815). Replace it with 'level L of N, A actions spent on this level'. N is the public game.number_of_levels, already read by _priority_total_levels (tool_agent.py:8773). A comes from run.actions_per_level or the step at level start. Optionally add one system line saying that score per level falls with the square of actions used.",
   "mechanism": "Per-level score is min(115, (baseline/actions)^2 x 100), quadratic in actions, and later levels are weighted more. The model currently can't see its per-level spend or how many levels remain, so it can't balance probing against committing. It's cheap and prefix-safe, since only the newest message changes.",
   "evidence": [
    "ARC3-Inference/inference/agent/tool_agent.py:7073-7077: state line content",
    "ARC3-Inference/inference/agent/tool_agent.py:6812-6815: observed_max_level cannot exceed current_level (dead branch)",
    "tufa-arc-agi-framework/src/taaf/game.py:380-410: score formula min(115,(baseline/actions)^2*100), level weight = index+1",
    "ARC3-Inference/inference/agent/tool_agent.py:8773-8795: harness already reads number_of_levels for the scheduler"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Strip old images (optionally old reasoning) only when a drain fires",
   "category": "memory-context",
   "description": "ARC3_HISTORY_IMAGE_KEEP (tool_agent.py:1240-1246) is applied in _persistent_history_messages on every commit (tool_agent.py:8737-8741). That changes the cached prefix every turn, which is why it is cache-hostile and off. Instead, apply _apply_history_image_window (tool_agent.py:1309), keeping the last K\u22486 image-bearing messages, only in the commit where token trimming actually fires (token_trim_dropped, about tool_agent.py:8746). The prefix is invalidated then anyway. Then re-run the trim so the freed budget keeps older text. Riskier variant: also drop reasoning_content from assistant messages older than the last ~20 turns at drain time.",
   "mechanism": "Each user prompt carries about 1.6 images (about 402 estimated tokens each at 10x upscale), so in the 58k post-drain window the images of dozens of old turns take a big slice. Stripping them only at drain keeps Franzen's 93% prefix reuse intact while each window holds more text history. Franzen found larger retained context helped hard games.",
   "evidence": [
    "ARC3-Inference/inference/agent/tool_agent.py:8722-8770: image window applied on every commit when enabled",
    "ARC3-Inference/CONFIGURATION.md section 4: 'Filtering historical images changes request prefixes and can affect prefix reuse'",
    "fz-arms-v4-control and fz-ucb: 4,927 images in 3,094 user prompts; calibration logs show 20 images = 8,040 estimated tokens in a 59,650-token request",
    "WRITEUP.md section 2: removing thinking entirely caused a large regression, so the reasoning variant needs a careful A/B"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "Refine the stale-state guard for MOUSE/SPACE 'select' actions",
   "category": "bugfix-quickwin",
   "description": "The sandbox arms the stale flag after any executed action whose interior (beyond the 4-cell border) didn't change (python_tool_sandbox.py:581-602). The solver then refuses the snippet's next action (solver.py:1530-1590). Add a flag-gated refinement and A/B it: don't arm after a MOUSE or SPACE action when board_changed (any cell) was true or its animation had transient pixels; or arm only after directional no-ops, which are wall bumps and the guard's real target.",
   "mechanism": "In the user runs, 145 of about 310 stale-guard firings followed a MOUSE no-op, versus about 150 for all four directions combined. In select-then-target click games, the first click often changes only HUD/edge cells or nothing visible. Refusing the follow-up click forces a new LLM round trip (about 1-3k tokens) and a re-plan, while directional loops stay protected. The effect is uncertain, so it needs the paired significance test.",
   "evidence": [
    "ARC3-Inference/inference/agent/python_tool_sandbox.py:581-602: stale flag armed on any interior no-op",
    "ARC3-Inference/inference/framework/solver.py:1530-1590: stale_state refusal before execution",
    "fz-* transcripts: 'The action immediately before it, MOUSE' x145 vs RIGHT 43, UP 36, LEFT 35, DOWN 33, SPACE 11",
    "WRITEUP.md section 5: the stale-state guard is enabled from level 2 in the submission"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "Generalize the scheduler's level-count handling beyond the [6,10] clamp",
   "category": "scheduling",
   "description": "parse_total_levels (priority_scheduler.py:63-84) clips N to [6,10]. For a 3-5 level game, TAIL_LOOKUP_REMAINING (priority_scheduler.py:58-60) then gives B=8 on the game's last level instead of 0, and A is normalized by 55/21 instead of 55/(N(N+1)/2). For more than 10 levels, B is wrong in the other direction. Fix: compute the 'remaining' tail directly from N-level (\u22653 gives 8, 2 gives 7, 1 gives 5, 0 gives 0) and normalize A with the true N. Keep the clip only for the empirical 80k table.",
   "mechanism": "With the clamp, the scheduler would over-invest in the final level of short games and mis-rank games with more than 10 levels. That costs nothing on the public set, where every observed game has 6-10 levels, but it is a cheap safety fix if the 110-game private set has other level counts.",
   "evidence": [
    "ARC3-Inference/inference/agent/priority_scheduler.py:63-84: clip to [6,10]",
    "ARC3-Inference/inference/agent/priority_scheduler.py:238-241: tail index uses the clipped N",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs + external: number_of_levels values seen are only 6, 7, 8, 9, 10"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "Skip list: Notes that do not cost score under the competition config",
   "category": "other",
   "description": "I checked these against the actual notebook env (nb.txt:10175-10260) and judged none worth fixing for score. _build_user_message dropping diff/animation/game-over images when there is no grid image (tool_agent.py:6606): MULTIMODAL_CONTEXT=current_grid is always on, so image_part is None only without a frame. _summary_turn_context_lines branches reading the wrong keys (tool_agent.py:6736-6760): used only by rolling summaries, which are off. OpenRouter streaming breaking response.json() (tools/chat.py:147, utils/openai_compat.py:113): offline vllm provider. Unused DEATH_GUARD/NOOP_GUARD addenda (prompts.py:29, 319-334): those guards are off. Exact-string prompt rewrites silently no-oping (tool_agent.py:3787, prompts.py:61-107): verified that all five level-transfer replacements, the MOUSE-line removal and the PYTHON_ADDENDUM_HEAD swap fire with the competition env (re-verify after any prompt edit). Multi-tool-call responses discarding turn history (tool_agent.py:9682): only 14 of about 17.4k replies had 2 calls. Board-dump guard off (tool_agent.py:2165): only 0.1% of tool results contain full-board dumps. Segmentation cost Note (segmentation.py:236): 20-200ms on real boards. Also negligible: retry-backoff, 80k-table truthiness, frame_diff rotation modulus and viewer Notes.",
   "mechanism": "This lets the team spend its few remaining days on the items above instead of on Notes that are inert in this configuration.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/external/dfranzen/nb.txt:10175-10260: competition env (no summaries, memory off, current_grid images on, vllm provider)",
    "fz-* transcripts: tool_call_count 2 appears 14 times vs 17,217 single-call replies",
    "fz-arms-v4-control and fz-ucb: 3 of 4,521 tool results contain at least 20 full-width board rows",
    "Local benchmark on Franzen example-run boards: segment_layer median 35ms, max 195ms (bp35, 169 nodes)"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  }
 ]
}
```
