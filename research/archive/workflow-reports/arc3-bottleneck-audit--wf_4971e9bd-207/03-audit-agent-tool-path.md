# arc3-bottleneck-audit: audit:agent-tool-path

Workflow run `wf_4971e9bd-207`, phase Audit. Audit the whole Franzen harness (inference, serving, TAAF, notebook, configs) for bottlenecks; verify each; classify DEFINITELY vs MAYBE for GitHub issues

---

```json
{
 "findings": [
  {
   "title": "Whole game history re-written, re-read and re-shipped to the sandbox on every action",
   "location": "ARC3-Inference/inference/agent/tool_agent.py::_run_python_tool._serialized_runtime_state (main 5730-5760; dead load 5727; returned at 5807/5833/5887; initial 5899) + ::_ascii_history_view_payload (2843-2858); inference/agent/runtime_state.py::write_runtime_state (127-140, indent=2 at 139) and ::load_runtime_state/normalize_grid (113-124, 48-62); inference/framework/solver.py::_execute_action (1234-1342, write_runtime_state at 1342); inference/agent/python_tool_sandbox.py::main()._refresh_state (384-412) and host reply (947-954)",
   "resource": "wall-clock-cpu-io",
   "problem": "Every executed action (even inside one batched action() call) makes the solver rewrite tool_runtime_state.json with ALL N history frames as indent=2 JSON (one cell per line, ~64 KB/frame). Then _handle_action calls load_runtime_state again (re-parse with int() per cell), recomputes frame.ascii for all N frames (a property, not cached), json.dumps ~17 KB per frame (grid AND ascii) through the stdin pipe, and the child re-parses it and rebuilds N FrameView objects plus the transitions list. Cost per action is O(N), so a game is O(N^2). The same full build runs once per python call (initial_state), and line 5727 does a further full load_runtime_state whose result is never used. Config check: always on (ARC3_ANIMATION/persistence irrelevant); history is never truncated.",
   "evidence": "Micro-benchmark of the exact main code path (fast laptop core, synthetic 64x64 frames): per action() inside a snippet 63 ms at N=50, 196 ms at N=150, 409 ms at N=300, 574 ms at N=500 (about 70% of it host-side, GIL-held). Per-call initial state build 26/95/177/461 ms. State file 3.2/9.6/19.2/32 MB; pipe payload 0.9/2.6/5.2/8.7 MB per action. Control run fz-arms-v4-control: 5632 actions, 2275 python calls, mean game length 225 actions, max 545. Model: sum(F^2)=1.7M gives about 1,400 s of serial stream-time per 25-game run (about 920 s of it GIL-held host CPU), about 55 GB of JSON written and 20 GB piped. Real snippet that hit the 30 s cap in sk48 at action 75: 37 UNDO iterations in 30 s, about 0.8 s per iteration at N of only 75-112.",
   "estimated_cost": "About 1.9% of stream-time at laptop core speed (1,400 s of 73,000 stream-s), about 2.5-4% on Kaggle-class cores. Host part is GIL-serialized across 10 game threads. Direct token loss is small (<=0.3%, because decode throughput is flat from 9 to 10 streams, 611 to 657 tok/s) but this is the main reason action loops hit the 30 s tool timeout (see finding 3).",
   "fix": "Stop round-tripping through the file. Have _serialized_runtime_state read the solver's in-memory history_entries (the analyzer already reaches the solver via step_env.__self__). Compute and cache ascii once per Frame. Send the child only ascii per history frame (rebuild the grid lazily from ascii and COLOR_CHARS when .segmentation is requested), full history only in the initial message and a 1-entry delta after each action. Write the state file once per analyzer turn, compact (no indent), or drop it. Delete the dead load at 5727. Expected per-action overhead about 10 ms at N=300.",
   "proposed_class": "DEFINITELY"
  },
  {
   "title": "Any sandbox timeout/crash wipes the whole retained-function library for the game",
   "location": "ARC3-Inference/inference/agent/python_tool_sandbox.py::run_sandboxed_python (timeout return 866-876, EOF/crash return 882-889, invalid-response 891-901, unknown-type 969-974: none carry keepable_functions); inference/agent/tool_agent.py::_record_retained_functions (4614-4658: 4621 reads keepable_functions or [], 4652 'no longer retained', 4653 self._kept_functions = kept)",
   "resource": "correctness",
   "problem": "On every non-final exit the host result has no 'keepable_functions'. _record_retained_functions treats that as 'the snippet kept nothing': kept becomes {}, every previous function is announced as 'no longer retained; redefine it if needed' (about 70 chars each, injected into the tool result), and self._kept_functions is overwritten with {}. With ARC3_PERSISTENT_FUNCTIONS=1 and SCOPE=game the model's accumulated toolkit (often 20-50 helpers built over 100+ turns) is destroyed by a single 30 s timeout. It also covers 'Sandbox process exited unexpectedly' and 'could not start'.",
   "evidence": "In 3 control-like runs (fz-arms-v4-control, fz-ledger-on-trim, fz-ledger-streams12) all 25 timeouts show the retained list collapsing at the next prompts: cn04 39 to 7, m0r0 54 to 1 (and 47 to 1), g50t 27 to 2, tr87 35 to 3, ls20 24 to 6, sk48 39 to 2, lp85 29 to 2, vc33 34 to none (mean about 24 functions lost per event). Models then re-emit a few helpers (0-15 defs, up to 4.3 KB code) or hit NameError. 41 timeouts in 6 runs, about 7 per 25-game run.",
   "estimated_cost": "About 7 events per run x 1-2 extra turns plus about 2-4k re-emitted tokens each: roughly 20-30k tokens (0.5%) per run, plus lost accumulated search code (cn04, m0r0, tr87 are the heavy users) which is exactly what costs the most to rebuild.",
   "fix": "Return keepable_functions=None (or omit retention fields) on timeout/crash paths and make _record_retained_functions return early (keep the previous pool, no notes) when the key is absent. Optionally collect retention state from the child before killing it, but the early return alone fixes the wipe.",
   "proposed_class": "DEFINITELY"
  },
  {
   "title": "30 s tool deadline bills host action time and discards all stdout of already-spent actions",
   "location": "ARC3-Inference/inference/agent/python_tool_sandbox.py::run_sandboxed_python (deadline 866, timeout return 872-876 with stdout ''), child main() (stdout = io.StringIO() at 354; only sent in 'final'/'error' at 699-731); inference/agent/tool_agent.py::_python_timeout (3490), ::_run_python_tool (5896-5899)",
   "resource": "reliability",
   "problem": "The deadline is wall-clock from just after the initial write and also runs while the host executes action() (env step + the O(N) state rebuild of finding 1). A loop of ~50-70 single-action calls at N=300 therefore dies at 30 s even though the child used a few seconds of CPU. The child buffers all print output in a StringIO until the end, so on kill the model gets only 'Tool timed out after 30s' with stdout='' while the executed actions were already charged to the game. The model must spend a turn re-deriving what happened.",
   "evidence": "41 timeouts over 6 Franzen-config runs (14,529 python calls, 0.28%); 7 in fz-arms-v4-control. At least 4-5 of the 7 control timeouts are action loops: g50t scripted route of ~36 moves, sc25 goto/make_plus loop, tr87 dial enumeration (action(['UP']) x12 per dial, twice), m0r0 plan executor, s5i5 22 clicks, vc33 click stages. sk48: 'for t in transitions: state(...)' (150 segmentations) then 'action([UNDO])' x45 -> 37 UNDOs executed (step 75 to 112), output lost, next prompt just says 'The code executed 37 actions'. Each timeout idles the stream 30 s.",
   "estimated_cost": "About 7 x (30 s idle + 1 re-observe turn, about 2k tokens) per run: about 15k tokens and about 450 stream-s (0.6%) directly, plus lost information and the retention wipe of finding 2. Mostly a score-risk item on games that need long scripted sequences.",
   "fix": "Accumulate only child-side time (time from sending a reply to the next child message, excluding action_handler) against the 30 s budget. Have the child flush stdout to the host at each action() boundary (it already round-trips) so partial output survives a kill. On timeout return the buffered stdout plus a note of how many actions executed.",
   "proposed_class": "DEFINITELY"
  },
  {
   "title": "Stale-state block halts multi-probe snippets after the first no-op (extra turn each)",
   "location": "ARC3-Inference/inference/agent/python_tool_sandbox.py::main().action (515-518 raise, 546-567 arm/clear); inference/framework/solver.py::step_env (1027 gate, 1036-1045 and 1085-1090 stale_state refusal), ::_stale_detail (415-429); inference/agent/tool_agent.py guard construction (3555-3570) and hook install (6579-6582)",
   "resource": "tokens",
   "problem": "With ARC3_STALE_STATE_BLOCK=1 (levels >=2), after any executed action with gameplay_changed False the child arms _stale_state and the host refuses every later action() in that snippet. The refusal does not clear the flag (it only clears on a gameplay-changing executed action, which cannot happen while every action is refused), so the message's claim 'StaleStateActionError can be caught if a snippet has good reason to keep acting through no-ops' is false. Independent probes (summ('UP'); summ('RIGHT'); summ('LEFT')) or plan loops (click, SPACE, ...) are cut at the first no-op and the rest costs a new LLM turn. Context: the stronger guards (known_noop, known_death, repeated_action) are inert in this config because ARC3_NOOP_REPEAT_GUARD/ARC3_DEATH_REPEAT_GUARD are unset (tool_agent.py:3558 gives _noop_repeat_guard=None, hook not installed), so this is the only snippet-halting guard besides the batch no-op stop.",
   "evidence": "StaleStateActionError appears as a tool error 38 times in fz-arms-v4-control (1.7% of 2,275 calls), 32-48 in the other 5 arms. Examples: ar25 probe of UP/RIGHT/LEFT stopped after UP; cd82 plan stopped at SPACE after a no-op click; cn04 click then SPACE. No model ever wrote except StaleStateActionError (0 of 14.5k snippets). Measured repeated NO-OP displays are at most 110 of 5,632 actions (<2%), so the saved-action upside of a strict block is small.",
   "estimated_cost": "About 40 forced extra turns per run: about 80k generated tokens (about 1.7%) and about 1,250 stream-s, in exchange for an unmeasured action saving.",
   "fix": "Clear _stale_state when the refusal is raised so catch-and-retry works as documented. Better: arm the block only for a repeat of the same action or a same-batch continuation, not for a different single-action probe. Consider turning it off for snippets whose actions are independent probes.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Segmentation is O(components x 4096) per frame and recomputed per FrameView/call",
   "location": "ARC3-Inference/inference/utils/segmentation.py::segment_layer (containment loop 192-226, per-component full-grid flood fill at 194-218); inference/agent/python_tool_sandbox.py child FrameView.segmentation (169-173), frame_diff (365-382), _refresh_state (384-412, builds fresh FrameViews each action); fresh child per call (Popen 820-830, bootstrap 65.7 KB)",
   "resource": "wall-clock-cpu-io",
   "problem": "To find which component encloses which, segment_layer floods the complement of EVERY component over the full 64x64 grid. This is the dominant cost (about 90% of self time in a profile). FrameView caches only per object, and every action() creates new FrameViews, so previous_frame (the same board that was current_frame one iteration earlier) is re-segmented; every snippet starts in a new interpreter, so historical frames are re-segmented on every call. Fixed per-call extras: 40 ms interpreter start with a 65 KB -c compile, plus 5-80 ms of retained-function restore/collect (ast.parse/compile/dis several times per function).",
   "evidence": "Real boards from the example runs, fast core: 15-16 nodes 25 ms, 40 nodes 50 ms, 89 nodes (tn36) 100-160 ms, 227 nodes (bp35) 160-235 ms; frame_diff() = 2 segmentations (195 ms tn36, 475 ms bp35). 451 of 2,270 snippets in the control run touch .segmentation, 140 call frame_diff, 216 loop over history/transitions (N or 2N segmentations: the sk48 sweep above). Retention restore+collect: 13 ms for 10 funcs, 79 ms for 31.",
   "estimated_cost": "About 0.3-0.5% of stream-time overall (a few hundred stream-s per run) but it is what turns history sweeps and per-iteration state() loops into the 30 s timeouts (finding 3). Expect 2x on Kaggle cores.",
   "fix": "Compute containment on the component adjacency graph (BFS over n nodes/edges excluding b) instead of pixel floods, about 10x faster with identical output. Memoize segment_layer by hash of the grid bytes in the child so previous_frame/revisited boards are free. Optionally ship precomputed segmentation for the last frame.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Sandbox API gaps cause avoidable error turns (time import, history .step, animation .items)",
   "location": "ARC3-Inference/inference/agent/python_tool_sandbox.py::_SANDBOX_BOOTSTRAP: SAFE_MODULES (50-71), SAFE_BUILTINS (72-145), HistoryEntryView (182-191), TransitionView (194-210), _LazyAnimation (599-635)",
   "resource": "tokens",
   "problem": "Whole snippets abort at the first missing name, costing one full LLM turn. (a) 'time' is not importable although the prompt imposes a hard 30 s limit; models import it to time-box searches (while time.time()-t0<22 in r11l, cn04) and then get ImportError and lose the time-box, which also raises timeout risk. 'inspect' and 'sys' are the next most requested. (b) HistoryEntryView/TransitionView expose no .step/.level though FrameView does. (c) last_animation_timeline is dict-like in the prompt but _LazyAnimation lacks items()/values(). (d) open and globals NameErrors.",
   "evidence": "Over 6 Franzen-config runs (14,529 calls): ImportError 'not allowed' 92 (time 35, inspect 23, sys 14, pickle 6), NameError open 24 / globals 10, AttributeError 'HistoryEntryView has no attribute step' 22 (+5 other history/transition attrs), '_LazyAnimation has no attribute items' 18 (+6 steps/action). About 179 error turns total, about 30 per 25-game run (1.3% of calls).",
   "estimated_cost": "About 30 wasted turns per run: about 60k generated tokens (1.3%) and about 950 stream-s, plus extra timeouts from lost time-boxing.",
   "fix": "Allow a time shim (time/perf_counter/monotonic only, sleep removed). Add .step/.level on HistoryEntryView/TransitionView delegating to frame, and keys/items/values/__getitem__ parity on _LazyAnimation. Optionally a tiny 'sys' shim exposing setrecursionlimit.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Non-retained snippet variables cause NameError turns (2% of turns)",
   "location": "ARC3-Inference/inference/utils/retained_functions.py (module doc 1-4: 'no snippet state is serialized'; _function_record 44-74 rejects functions reading non-retained globals); inference/agent/python_tool_sandbox.py main() persistence block (644-663), inference/agent/tool_agent.py::_run_python_tool (5895-5905)",
   "resource": "tokens",
   "problem": "Only function source is retained; top-level constants and tables (CELLS = {...}, plan = [...], A = ...) vanish between calls, and any retained helper that reads such a global is dropped by prepare_functions (line 129-136). Models keep assuming they exist and the next snippet fails at its first use.",
   "evidence": "NameError for an undefined variable in 308 of 14,529 results across 6 runs (about 51 per run, 2.1%; 42-64 per arm); top names: a, A, P, G, CELLS, sols, SH, CAN, plan, W4, G0. Part (a, sols, plan, start) are computed values; part (CELLS, CAN, SH) are literal tables that would survive literal retention. The prompt already says 'Python variables reset between tool calls'.",
   "estimated_cost": "About 45 turns per run: about 90k generated tokens (about 2%) and 1,400 stream-s; realistically recoverable share perhaps a third (about 0.7%).",
   "fix": "Retain top-level assignments whose RHS is ast.literal_eval-able and size-capped (same source-only mechanism, same scope), or at minimum append 'name X was assigned in call k but variables do not persist' to the NameError. Needs an A/B for stale-constant side effects.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Sandbox child has no memory cap or CPU deprioritisation next to the SGLang server",
   "location": "ARC3-Inference/inference/agent/python_tool_sandbox.py::_set_limits (285-299: only RLIMIT_CPU, FSIZE, NOFILE), run_sandboxed_python Popen (820-830), _sandbox_env (765-773)",
   "resource": "reliability",
   "problem": "Model-written search code (BFS/DFS/enumeration) can allocate unbounded memory inside the 30 s window and runs at normal priority beside the SGLang scheduler/tokenizer/detokenizer threads, whose latency gates decode steps. Up to 10 children can run concurrently. An OOM kill of the server (or memory pressure on the page-cache-heavy 190 GB model host) would end every game, not one.",
   "evidence": "A plain tuple-set BFS grows to about 1.36 GB RSS in 10 s (10M states), so about 4 GB per child in 30 s, 40 GB if 10 overlap. Runaway compute is routine: 41 timeouts in 6 runs, including cn04 DFS/enumerations (4+ times), re86 four-nested range loops, tr87 permutation sweeps. No OOM or server crash observed in about 16 runs and Kaggle RAM size is not recorded, so this is a latent rather than demonstrated loss.",
   "estimated_cost": "Unmeasured; low probability but catastrophic (whole-run loss) and a possible small decode-throughput tax from CPU contention.",
   "fix": "Add RLIMIT_AS/RLIMIT_DATA (about 2-3 GB) in _set_limits and os.nice(10) (or sched_setaffinity away from SGLang cores) in the child bootstrap. Zero cost on the hot path.",
   "proposed_class": "MAYBE"
  }
 ]
}
```
