# arc3-bottleneck-audit: audit:agent-request-path

Workflow run `wf_4971e9bd-207`, phase Audit. Audit the whole Franzen harness (inference, serving, TAAF, notebook, configs) for bottlenecks; verify each; classify DEFINITELY vs MAYBE for GitHub issues

---

```json
{
 "findings": [
  {
   "title": "Redundant O(history) disk reload + reserialization on every analyze()/python call/action()",
   "location": "ARC3-Inference/inference/agent/tool_agent.py::analyze (L6587 load_runtime_state), ::_run_python_tool (L5727 dead load; L5730-5760 _serialized_runtime_state; re-called L5807/5833/5887 per action() and L5899 for initial_state); inference/agent/runtime_state.py::load_runtime_state/normalize_grid/Frame.ascii (L113/L48/L28), write_runtime_state (L139, indent=2)",
   "resource": "wall-clock-cpu-io",
   "problem": "Each analyze() re-reads and re-parses the whole runtime-state JSON from disk (file is ~64 KB per history frame because of indent=2). _run_python_tool then loads it again into variables that are never used (dead load). _serialized_runtime_state loads it a third time, rebuilds ascii (uncached Frame.ascii property) and grid lists for EVERY history frame, and pushes the full history through the sandbox pipe. After every action() inside a snippet the same full reload and reserialization runs again, and the sandbox re-parses it. All of this happens while the game holds its gate slot, so the GPU slot is idle. Cost grows linearly with actions taken in the game, and the long games are the ones making progress.",
   "evidence": "Competition config runs this path on every turn (no gating env var). 3x load_runtime_state per inspect-only analyze, confirmed by call-count profile. I measured analyze() against an instant stub server on an M-series Mac with 30 history turns: 0.30 s at 100 history frames, 0.62 s at 300 frames, and about 0.9 s extrapolated at 450. Component timings at N=300: load_runtime_state 92 ms each (x3), payload build 96 ms, json.dumps 38 ms, sandbox parse 43 ms. A python call with 3 action() calls at N=400 costs 0.9 s in the sandbox path alone, plus about 170 ms of disk reload per call. write_runtime_state runs after every executed action in solver.py L1342 and takes 62 ms for a 19 MB file at N=300. Server ReqTimeStats from fz-arms-v4-control give a client gap between requests of median 1.8 s for inspect follow-ups and 4.1 s for the next opener, mean about 3.3 s per request (10.5% of slot time; batch mean 8.9/10). Other per-request tool_agent CPU is negligible: token estimate 1-2 ms, trim 0.7-20 ms, json deepcopy 1-2 ms, prompt-log write 2 ms, PNG encode 2.4 ms.",
   "estimated_cost": "About 0.2-0.4 s per request at mean N~110 on this Mac, 0.6-0.9 s late in 300-450-action games, possibly 1.5-2x on Kaggle CPUs. Roughly 1-3% of slot time (about 1-3% tok/s). Larger in the long games that earn the later, higher-weight levels.",
   "fix": "Delete the dead load at L5727. Read current_frame/history once per analyze, or take them from the solver's in-memory history_entries via step_env.__self__ instead of disk. Cache per-frame ascii and payload (frames are immutable and history is append-only), and send the sandbox an incremental or lazily-fetched history. Write runtime state as compact JSON or append-only.",
   "proposed_class": "DEFINITELY"
  },
  {
   "title": "Gate slot held across client-side gap: decode batch averages 8.9/10, about 11% of slot time idle",
   "location": "ARC3-Inference/inference/agent/tool_agent.py::_PriorityGate.acquire/release/handover (L2059-2118), ::_maybe_handover (L6372-6417), ::_max_active_streams (L2119); notebook ARC3_MAX_ACTIVE_STREAMS=10 with server max_running_requests=10",
   "resource": "gpu-throughput",
   "problem": "A game acquires a slot once, at game start, and gives it up only on a post-trim handover or when the game ends. It therefore holds the slot through the python sandbox, engine step, prompt build and state IO between requests. The slot count equals the server's max_running_requests, so every client-side gap is an idle decode slot and the batch can never sit at 10. Any structural gap shows up as lost occupancy: this run lost 11%, and the 12- and 14-stream runs lost 12-9%.",
   "evidence": "fz-arms-v4-control serve.log: 2308 requests, mean forward_duration 28.3 s, 10 slots x 7334 s = 73.3K slot-seconds, 65.3K spent in forward, about 8K slot-seconds (11%) outside it, i.e. about 3.3 s per request. #running-req histogram: 10 for 42% of decode logs, 9 for 30%, 8 for 14%, 7 or fewer for 14%; mean 8.9. Atlas serving.csv: decode median 741 tok/s vs harness 611 tok/s, and decode at full batch 793 tok/s. 14 streams gave batch mean 12.8/14 and 703 tok/s (+15%); 12 streams gave 10.5/12 and +9%. Both of those also shrank context, so the score effect is confounded.",
   "estimated_cost": "Recovering a full batch of 10 is worth about +7% decode throughput (741 to 793 tok/s) and, if tokens map to levels roughly linearly, about +3 offline-mean points. The saving is partly offset by KV pool and mamba-cache pressure.",
   "fix": "Release or lend the slot while the game is in client-side work (sandbox, engine) and re-acquire just before the next request, or oversubscribe the gate by 1-2 streams (11-12 active on max_running_requests=10) without shrinking the 128K window. Watch the KV pool (about 1.01M tokens; usage peaked at 0.99) and the 60 mamba slots. Untested in the user's runs; needs an A/B.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Trim-induced re-prefill: about half of all prefill tokens are prefix rebuilds (about 1:1 re-prefill per token of context growth)",
   "location": "ARC3-Inference/inference/agent/tool_agent.py::_trim_messages_for_context (L6454-6546), ::_persistent_history_messages (L6313-6350), ::_context_drain_tokens (L227); notebook ARC3_CONTEXT_DRAIN_TOKENS=58*1024, LOCAL_ANALYZER_CONTEXT_WINDOW=128K",
   "resource": "prefix-cache",
   "problem": "When the estimate passes about 118K tokens, the oldest blocks are dropped until about 59K remain. Removing anything at the front invalidates the whole server prefix, so the next request re-prefills about 59K tokens, and the next trim comes after about 59K more tokens of growth. Re-prefill therefore costs about one token per token of context growth, and each trim also leaves the retained history unchanged apart from being shorter. 54% of context is old reasoning_content and about 10-12% is old base64 images. Compacting those at trim time would be free in cache terms, because the prefix is already lost. ARC3_HISTORY_IMAGE_KEEP cannot do this today: it slides every turn and would bust the prefix every turn.",
   "evidence": "fz-arms-v4-control ReqTimeStats: 2308 requests, 166.4M input tokens, 93.6% cached, 10.6M new tokens. 92 requests with more than 20K new tokens carry 5.09M tokens (48% of all new tokens, 458 s of summed prefill latency, about 5 s each). Prefill speed is at least 13.6K tok/s (6x8192-token chunks in 3 s), so about 375 s, or 5%, of a 7334 s run is full-context rebuilds during which the other streams do not decode. Total prefill is about 10% of GPU time. Calibration lines show images are 10-12% of prompt tokens (e.g. 6834 of 66316). WRITEUP says retaining thinking is deliberate and that removing it regressed the score, so any change must be validated.",
   "estimated_cost": "Up to about 5% of GPU time sits in trim rebuilds. Trim-time compaction could cut the post-trim context by 10-40% and lengthen the interval between trims by a similar amount, so a gain of about 1-3% GPU time, with uncertain score effect.",
   "fix": "Apply history compaction only inside the trim, where the prefix is already invalidated: strip images from all but the last K image-bearing messages, and optionally truncate older reasoning_content. Then lower the post-trim target or lengthen the interval. A/B against the 46.25 control.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Replies cut off at the 12K output cap burn tokens with no action and stay in context",
   "location": "ARC3-Inference/inference/agent/tool_agent.py::analyze (L6828-6845 length handling; L6979 and L7053 append assistant message incl. reasoning_content; L6988 CUT OFF nudge), ::_chat_completion (L5218, non-streaming, max_tokens L5231); notebook LOCAL_ANALYZER_MAX_OUTPUT=12*1024",
   "resource": "tokens",
   "problem": "Generation is non-streaming with max_tokens 12288, so a runaway chain of thought runs to the cap. The reply has no tool call, the 12K-token truncated reasoning is committed to history, and a long CUT OFF nudge user message is appended. ARC3_REASONING_EFFORT_LADDER and ARC3_STUB_DEAD_REASONING are both unset in the competition config, so nothing mitigates this.",
   "evidence": "fz-arms-v4-control transcripts: 25 of 2299 responses end with finish_reason=length (server log: 25 requests at output_len>=12288). 873K reasoning chars, about 300K tokens, is 6.7% of the 4.48M generated tokens. 105 responses (4.6%) exceed 20K reasoning chars and carry 24.5% of all reasoning characters. Each blob is about 3.5x a normal turn's context growth. In 22 of 25 cases the next reply is a short tool_call (<=1.6K chars, 3 of those <=100), so part of the truncated thinking is reused.",
   "estimated_cost": "Up to 6.7% of generated tokens in a run, but part of it is reused by the next short reply, so the real waste is smaller. Plus about 3% extra context growth from the blobs.",
   "fix": "Stream and abort at a soft cap (about 6-8K tokens) and inject the nudge early, or enable the effort ladder after a truncation. Optionally stub or trim the truncated reasoning in history once the next reply has acted. Needs an A/B because the long thoughts are partly useful.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Every opener repeats about 1.7-2.1K chars (about 500 tokens) of static boilerplate",
   "location": "ARC3-Inference/inference/agent/tool_agent.py::_build_user_prompt (L5100-5165: 'Only tool: python...', 'Keep tool output compact...', 'Use Python to inspect...', 'When ready, call action...', TOOL_CALL_FORMAT_GUIDANCE L5160, MOUSE line L5164); the same tool-call guidance already sits in the system prompt via COMPACT_TOOL_SESSION_ADDENDUM",
   "resource": "tokens",
   "problem": "Every user opener, including resume openers, ends with the same block of instructions that restates the system prompt (Python guidance, tool-call format, 'compare previous_frame to current_frame'). TOOL_CALL_FORMAT_GUIDANCE is a verbatim duplicate of a system-prompt line. The block is identical every turn and is retained in history.",
   "evidence": "fz-arms-v4-control transcripts: 1560 openers average 2578 chars, of which about 2138 chars (83%) are the static tail from 'Only tool:' onward. At about 4 chars/token that is about 535 tokens x 1560 = 0.83M tokens, 7.8% of the 10.6M new prefill tokens and about 8.4% of the 9.9M tokens of context growth. Roughly 8.5% of the average 72K-token context is this repeated text. The system prompt (18.6K chars) already carries the same guidance. The author tested shorter resume prompts only (WRITEUP: no consistent gain).",
   "estimated_cost": "About 1.3% of GPU time (0.9% direct prefill plus about 8% fewer trim rebuilds), and about 8% of context. Behavioural effect of removing the reminders is untested.",
   "fix": "Remove the verbatim duplicate (TOOL_CALL_FORMAT_GUIDANCE) first, then move the rest of the static lines into the system prompt, keeping only dynamic state in the opener. A/B for score.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Token-budget yields (2048) re-open turns with a full duplicate opener and image, and the duplicate stays in context",
   "location": "ARC3-Inference/inference/agent/tool_agent.py::analyze (L6648-6700 resume_mode/resume_frame/_mark_control_message; L6718-6732 control_yield_reason), ::_yield_resume_prompt_mode (L776, default 'full'), ::_resume_prompt_images_enabled (L908, default True), ::_prune_control_context_enabled (L1186, default False); notebook LOCAL_ANALYZER_YIELD_TOKENS=2048",
   "resource": "tokens",
   "problem": "After any inspect-only python call with at least 2048 cumulative tokens in the turn, analyze() returns. The next analyze() rebuilds the full opener (including the static block above), re-attaches the same 640x640 board image (402 tokens), and does not prune it from history because ARC3_PRUNE_CONTROL_CONTEXT is off. It also pays the full per-analyze state-reload overhead again. The yield itself does not stop or resume generation; should_stop is already checked inside the loop.",
   "evidence": "fz-arms-v4-control: 466 of 1560 analyze invocations (30%) are resumes after turn_token_budget, which is 20% of the 2308 requests. Resume openers average 2518 chars plus one image. That is about 0.45-0.5M tokens (4.5% of new prefill; overlaps the boilerplate finding for the static text) and about 0.2-0.3 s of analyze overhead per resume at moderate N. WRITEUP says shorter and different-tone resume prompts gave no consistent gain.",
   "estimated_cost": "About 0.5-1% of GPU time plus about 5% of context growth and 466 extra analyze() round trips per run.",
   "fix": "Keep the opener behaviour (the author tested alternatives) but avoid the duplicate cost: drop the re-attached image (ARC3_RESUME_PROMPT_IMAGES=0), prune the resume message after the turn commits (ARC3_PRUNE_RESUME_PROMPTS=1), or continue the loop inside the same analyze() when should_stop is false. A/B first.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "analyze() converts any non-HTTP exception into a permanently crashed game",
   "location": "ARC3-Inference/inference/agent/tool_agent.py::analyze (L7161-7179 'except Exception ... return None'; unguarded pre-try section L6587-6741: load_runtime_state, _build_user_prompt, _build_user_message PNG encode, _trim_messages_for_context); inference/framework/solver.py::_play_inner (L689-690 raises RuntimeError on None; L731-736 sets state='crashed')",
   "resource": "reliability",
   "problem": "Only requests.RequestException is turned into a retryable failure (up to 10 consecutive, with a 10 s backoff). Any other exception inside the try returns None, and anything raised before the try (prompt building, state load, image encode, trim) propagates straight out. In both cases the solver marks the game crashed and stops playing it for the rest of the run, forfeiting all remaining time and levels, which are the highest-weighted ones. Examples are a disk write error in transcript or prompt-log writes, a malformed server body, or a sandbox or step_env surprise.",
   "evidence": "0 crashes observed in 250 completed games across the user's 12 finished fz-* runs, so the rate is low (95% bound about 1.2%). The path is live in the competition config (no gating). With 110 games and a 32K-second budget a single event costs a whole game's remaining levels. Disk writes happen on every request (prompt log, transcript) and every action (runtime state, viewer payload).",
   "estimated_cost": "Low probability, high cost per event. Expected loss is under about 1 game in 110 at the observed rate, and each loss is up to a full game's remaining levels.",
   "fix": "Treat non-fatal exceptions as retryable: return retryable_failure=True with history rollback instead of None, and wrap the pre-try section the same way. Reserve a hard crash for the same exception repeating more than N times.",
   "proposed_class": "MAYBE"
  }
 ]
}
```
