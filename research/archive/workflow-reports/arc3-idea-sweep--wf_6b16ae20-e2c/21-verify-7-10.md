# arc3-idea-sweep: verify:7/10

Workflow run `wf_6b16ae20-e2c`, phase Verify. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "verdicts": [
  {
   "id": "sandbox-allowlist-gaps",
   "keep": true,
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 0.25,
   "confidence": "high",
   "reason": "Verified real but small. Franzen already added difflib and the narrower exception types (python_tool_sandbox.py:33, 85-106). The rest is still missing: time/sys/inspect/pickle not in SAFE_MODULES, no globals/id/__build_class__ in SAFE_BUILTINS, HistoryEntryView/TransitionView have no .step/.level, _LazyAnimation has no .items, prompts.py:253 omits difflib. I counted fz-* transcripts (7 runs, 17,242 python calls): ImportError time 43, inspect 28, sys 16, pickle 10; NameError globals 10, deque 7, json 7, itertools 5, id 3; AttributeError HistoryEntryView.step 24, _LazyAnimation.items 18, plus 51 'Tool timed out'. That is about 200 events, about 1.2% of calls and about 1% of tokens. The idea's 114 ImportErrors matches, but its '330 of 4.7k' mixes denominators: v4-control is 119 errors in 2,270 calls, and 38 of those are the intentional StaleStateActionError guard. Observed 0 RecursionErrors and 1 class-definition error, so drop the recursion, class and dataclasses items. A time shim is the most useful piece, because the model reaches for it to bound search loops (43 ImportErrors, 51 30s timeouts that discard all stdout). Two gotchas: class bodies also need runtime_globals['__name__'] (I reproduced the NameError without it), and any pre-bound names (deque, math, json...) must be added before _available_names is computed (~line 645). Otherwise retained_functions.prepare_functions rejects helpers with 'unavailable dependencies', or flags alias conflicts. Gain is about +0.1 pt, but risk is near zero.",
   "first_step": "python_tool_sandbox.py: add a time shim (time/perf_counter/monotonic), a sys shim (setrecursionlimit capped at 20000), and globals/id/vars to SAFE_BUILTINS. Pre-bind deque/defaultdict/Counter/json/math/itertools/collections into runtime_globals right after runtime_globals is built, before _available_names. Add .step/.level to HistoryEntryView/TransitionView and .items/.values to _LazyAnimation. Add difflib/time to the module list at prompts.py:253."
  },
  {
   "id": "inspect-source",
   "keep": true,
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 0.15,
   "confidence": "medium",
   "reason": "Real gap: after hysteresis trims history to about 57k, the opener lists only signatures (tool_agent.py:6479-6489) and the model cannot re-read its helper bodies. fz transcripts show about 35-67 inspect.getsource attempts (67 raw matches, some duplicated in the rendered log) and 28 'ImportError: Module inspect' across 17k calls, so about 0.3% of calls. The 'misuses/redefines helpers' mechanism is not evidenced by the cited data. Simplification: the child already receives the sources (kept_functions=self._retained_sources() at tool_agent.py ~8137 and _kept_pool in the bootstrap), so no host plumbing is needed. Allowing the inspect module alone would not work, because exec'd code has no file and getsource would raise OSError; it must be a shim. Do NOT extend the signature list with doc lines: it is repeated in every opener and persists in history. Tiny upside, near-zero risk. Ship together with sandbox-allowlist-gaps.",
   "first_step": "python_tool_sandbox.py, right after restore_functions (~line 640): runtime_globals['source'] = lambda f: _kept_pool.get(getattr(f,'__name__',f), 'not a retained function'), defined after _available_names so it is not reserved. Add one sentence about source(name) to the retained-functions text in tool_agent.py _retained_function_context."
  },
  {
   "id": "numpy-grid",
   "keep": false,
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 0.3,
   "confidence": "low",
   "reason": "Evidence does not support the claim. The GB10 note (gb10-wt/numpy-perception/notes/gb10-arms/numpy-perception.md) says the ar25 L2 gain traces to the <perception> text block, not numpy. In v2 the model read .grid in only 2 of 17 ar25 calls. The numpy-only arm was never run, and Fisher p is 0.17 on 2 games at 7x slower scale. ar25 is already 100 in the Franzen control (summary.txt). There is no model demand either: 0 'import numpy' or np.array across 17,242 fz calls, and 0 uses of the already-reachable current_frame._grid. Tufa/Franzen hide the grid on purpose (prompts.py:193, tool_agent.py:429 and :7100). The saving is a few code tokens per call against about 2k reasoning tokens per call. It adds subprocess risk: numpy import per fresh sandbox process, BLAS threads under 10 concurrent sandboxes, numpy scalars stringified by _json_safe, and retained-function import bundles. Drop it. If wanted, take the perception text block (see scene-card / auto-frame-diff) instead.",
   "first_step": "n/a (dropped)."
  },
  {
   "id": "plain-text-tool-output",
   "keep": true,
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 0.1,
   "confidence": "low",
   "reason": "Code claim verified: _render_tool_payload ends in json.dumps(result, indent=2) (tool_agent.py:7508) and the model receives dispatch.content raw (:9673-9690). Only the transcript viewer decodes it via _render_tool_result_display, so the viewer hides what the model actually sees. But the benefit is unproven and probably tiny. Mean tool result is 621 chars (median 405) over 16.8k results, so an 11% char change is about 25 tokens per call; JSON '\\n' vs a newline is about 1 token either way. LLMs do not see column alignment, so 'destroys alignment' is the weak part of the mechanism. SWE-agent (arxiv 2405.15793, fetched) supports 'interface design matters' in general, not JSON vs plain text. Only 3 of 16.8k results contain non-ASCII, so \\u-escaping is not an issue. No downstream code parses the JSON except the viewer. Cannot be A/B'd (noise about 3 pts). Cheap and plausibly slightly positive, so ride along in a bundle of free tweaks. Do not spend a dedicated Kaggle arm on it.",
   "first_step": "tool_agent.py _render_tool_payload (~7508): behind ARC3_PLAIN_TOOL_OUTPUT, return stdout verbatim plus compact 'error:' / 'result:' / 'function_retention:' / 'truncated:' lines; keep _trim_tool_text and _suppress_board_dumps."
  },
  {
   "id": "effects-track-history",
   "keep": false,
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 0.5,
   "confidence": "low",
   "reason": "Mostly redundant. frame_diff(before, after) already accepts any history frames (python_tool_sandbox.py ~375-390, documented at prompts.py FRAME_DIFF_HINT). Every transition already stores .result with gameplay_changed/board_changed/no_op/animation (solver.py:1862-1875). A one-line comprehension over transitions gives the same effect table. Model-written helpers persist across the whole game under ARC3_PERSISTENT_FUNCTIONS_SCOPE=game, which is on, so the model writes track()-like code once. The cited sk48 example (sk48-d8078629_p0.txt:8380-8470) is a game-specific HUD-icon parse that a generic track() would not replace, and sk48 scores 0 for deeper reasons. Stats checked: history loops/indexing 241 of 2,270 calls (10.6% by my regex), frame_diff in 140 (6.2%). Any useful part (stored per-transition effect) belongs in action-effect-echo. A per-transition object diff also costs about 160 ms (measured), so it would be slow. Drop.",
   "first_step": "n/a (dropped; fold the stored per-transition effect into action-effect-echo if built)."
  },
  {
   "id": "auto-frame-diff",
   "keep": true,
   "status_in_franzen": "present-but-off",
   "impact": 3,
   "effort_days": 1,
   "confidence": "low",
   "reason": "Present-but-off confirmed: CONFIGURATION.md:76-77 default Off, absent from the competition setup_env (franzen/kaggle/template.ipynb). arms.md backlog #4 already lists it as a worked-on-GB10, not-yet-transferred item. The user has not run it on the Franzen stack (no fz-* arm). Skeptic findings. (1) Evidence is weaker than stated: gb10-wt/frame-diff notes give 2/2 vs 2/8 on one game (Fisher p=0.13, pooled 0.03 is post hoc); control b1 found the same ar25 insight at the same turn without it; ar25 is already 100 in the Franzen control; it was a different (cell-level, last-action) implementation on a slower harness with no diff image. (2) Likely tried by Franzen: the code is kept as an optional 'unsuccessful or inconclusive' experiment (CONFIGURATION.md:461); WRITEUP \u00a76 says auto-inserted animation timeline had 'no clear gain' and composite images hurt. That is adjacent, not the same, but not encouraging. (3) Franzen's matcher produces many fake moves. I replayed 640 consecutive-board diffs from 5 recorded games (example-run events.jsonl) through compute_frame_diff(max_group_match=40): 52% of 'moved' entries are <=2px; ar25 has 124 of 595 tiny 'moves' jumping >6 cells; dc22 115 of 115; bp35 identical 21px tiles 'move' 6-18 cells. Idea item (e) (1x1 only) does not cover tile games, and the system prompt calls the block 'ground truth' (tool_agent.py:3812-3819). Mean cost is 162 ms per diff (p95 460 ms, laptop py3.14; the idea's 47 ms is optimistic), but it runs once per opener, so it is fine. Rendered size: mean ~140 tokens, 5% over 300. (4) Upside: the diff spans the whole analyzer turn, net of batches (tool_agent.py:5836-5845), which fixes the GB10 'last action only' gap. Model-side usage of frame_diff() is only 6% of calls. Realistic effect is +0 to +3 offline with sign uncertainty, not 'high'. Needs >=2 runs per arm because noise is about 3 pts. Worth one arm after the matcher fix.",
   "first_step": "Replay recorded boards (arc-agi-3-v3/franzen/example-run/artifacts/*_events.jsonl, or the newer fz run boards) through inference/utils/frame_diff.py compute_frame_diff. Fix many-to-many pairing of identical-hash objects (report them as 'gone at / new at', not 'moved', when displacement is large or the group is >3). Merge touching same-delta movers and fold border changes into one 'HUD: bar a->b' line. Then set ARC3_AUTO_FRAME_DIFF=1 (budget 300) in the arm env. Keep ARC3_DIFF_IMAGE on."
  },
  {
   "id": "action-effect-echo",
   "keep": true,
   "status_in_franzen": "partial",
   "impact": 2,
   "effort_days": 0.5,
   "confidence": "medium",
   "reason": "Keep a narrowed version. Already present and on: ARC3_ACTION_ECHO=1 prints '[action] X -> effect/NO-OP' per call (solver.py:248-330, 1688-1696), and per-action gameplay_changed/board_changed/animation are stored in transition results (solver.py:1862-1875). Verified stats from v4-control: 157 of 1,036 MOUSE actions echo NO-OP; 281 of about 3,700 actions overall. The animation mislabel is real but narrower than claimed. The next opener already says 'animated over N frames ... was not a no-op' (45 unique cases in v4-control, mostly sc25/su15/s5i5/cd82), so the mislabel window is only inside one snippet. The same blind spot also drives guards: BATCH_NOOP_BLOCK and the stale-state arming key on gameplay_changed only (solver.py:1629; sandbox action() stale arming), so animation-only actions cut batches. A cheap, real fix: make _action_verdict (solver.py ~248) animation-aware. Do NOT call compute_frame_diff per action: measured about 160 ms mean and 460 ms p95 per diff, so a 100-action loop adds 16-46 s against the 30 s tool timeout, which counts host time per the run_sandboxed_python docstring. Use cheap O(4096) stats instead (changed cells, bbox, top colour transition, about 15 tokens) and cap about 25 echo lines per snippet. Drop the Scott Le Grand ledger / UNTESTED line: the idea itself says evidence is mixed, and Siriki found no gain. lordhansolo's changed_pixels fields exist (kaggle-datasets/lordhansolo-taaf-kaggle-source prompts.py:74-167), but that harness scored 23.84 vs Franzen 27.89 with no isolation. Honest gain is +0 to +1.5 pt.",
   "first_step": "solver.py _action_verdict (~line 248): when gameplay_changed is False and item.get('animation') is set, return 'no board-area change but ANIMATED (N frames, T transient px) - read last_animation_timeline'. Then consider exempting animated actions from the no-op guards at solver.py:1629 (an arm of its own, since it changes behavior). In _execute_action add changed_cells and bbox from previous_grid/new_grid (cheap) to the echo line, gated by ARC3_ACTION_EFFECTS."
  },
  {
   "id": "scene-card",
   "keep": true,
   "status_in_franzen": "partial",
   "impact": 2,
   "effort_days": 0.75,
   "confidence": "low",
   "reason": "Partial overlap: ARC3_LEVEL_INVENTORY (tool_agent.py:1743-1756, 6073-6146) already gives a START-vs-START 'new types appeared' block for level >= 2 and is off in the competition config and untested by the user. The turn-1 / all-levels card is new. Measured demand (v4-control, 25 games): the model runs a mean of 3.2 (median 2) non-acting python calls at each of 71 level starts and 2.3 before its first action in a game. At about 2k tokens per call that is about 6k tokens per level start, about 15% of the 42.7k tokens/level in versions.md. A card could realistically remove about 1 call per level, so about 1-3% of compute (+0.5 to +1 pt), not a big win. Evidence quality is thin. The GB10 'model copied bboxes' observation comes from the every-turn numpy-perception block (p=0.17, confounded). versions.md:62-63 (ASCII only 4.63, ASCII+image 9.86 vs 17.74) is a warning against full-board text, which this card avoids, but it is not support. arXiv 2305.18354 (fetched) is real: object-based representations nearly doubled GPT-4 on static ARC, a different setting. The Kaggle discussion 743060 was not verified. Risks: wrong HUD/background guesses can anchor the model, and the model still re-inspects anyway. Cheap first step: flip ARC3_LEVEL_INVENTORY=1 (one env var, no code) before building the card. Note it reuses compute_frame_diff, so the fake-move issue (see auto-frame-diff) applies; prefer a type comparison keyed on shape_hash.",
   "first_step": "One-line test: set ARC3_LEVEL_INVENTORY=1 in the arm env (level >= 2 only). Then build the card in tool_agent.py _prepare_auto_diff / _build_user_prompt: segment_layer(current_frame.grid) -> background colour, edge-flush thin HUD candidates, <=12 objects grouped by shape_hash (count, colour, hxw, bbox, rarest first), once at game start and at each level start; reuse the inventory diff to tag new types."
  }
 ]
}
```
