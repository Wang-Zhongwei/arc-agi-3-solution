# arc3-idea-sweep: sweep:tools

Workflow run `wf_6b16ae20-e2c`, phase Sweep. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "ideas": [
  {
   "title": "Return a short object-level effect for each action from action() and the action echo",
   "category": "tools-repl",
   "description": "In ARC3-Inference/inference/framework/solver.py `_execute_action` (lines 1776-1880), after the engine step, compute `compute_frame_diff(previous_grid, new_grid, seg_prev, seg_new)`. Reuse the segmentation of the after-frame as the next before-frame so each action segments only one frame. Render it with the existing `_render_auto_frame_diff` / `_diff_entry_line` logic (tool_agent.py:3646-3767), with a budget of about 40-60 tokens per action, dropping resized entries that cover most of the grid and folding HUD-border changes into a short 'bar -1' note. Store the result as `individual_result['effect']`, so it also lands in `transitions[i].result['effect']`. Then extend `_action_verdict` / `_build_action_echo` / `_build_action_trace` (solver.py:248-375) so that `[action] RIGHT -> effect` becomes `[action] RIGHT -> effect: B 16px (15,18)->(15,21); c 16px (15,36)->(15,33); 109 cells`. Cap the total effect lines per snippet (e.g. 25) to protect the 3,072-token tool budget. Gate it behind a new flag `ARC3_ACTION_EFFECTS=1` so it can be A/B tested against the current submission config, which has `ARC3_ACTION_ECHO=1` and `ARC3_AUTO_FRAME_DIFF` off.",
   "mechanism": "The model spends about 90% of generated characters on thinking. A large share of that is re-deriving 'what did my action just do' in coordinates, or writing code to find out. Giving the exact harness-computed change for every action, inside the loop where it acts, removes that work. The model can then branch on effects mid-snippet without extra requests. That should mean fewer requests and fewer thinking tokens per cleared level (currently about 42k generated tokens per level), and so more levels within the fixed token budget.",
   "evidence": [
    "ARC3-Inference/inference/framework/solver.py:248-330: the echo verdict is only 'effect' / 'NO-OP' / 'GAME OVER'; it carries no object-level content",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs/fz-arms-v4-control/transcripts/ar25-0c556536_p0.txt (analysis_step 3): the model wrote its own `summ(tag)` wrapper to print an object diff after every action, i.e. it builds this feature by hand",
    "Mining 2,273 tool calls in fz-arms-v4-control: 6.2% call frame_diff(), 7.9% loop over history/transitions, 2.0% hand-write cell diffs (my regex scan)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/GB10-OVERNIGHT-RESULTS.md: 'Exact change information is the one lever that showed up across arms'. All 5 arm runs that gave a harness-computed 'what moved' line cleared ar25 level 2, against 2 of 8 controls (pooled Fisher p~0.03, chosen post hoc)",
    "ARC3-Inference/CONFIGURATION.md section 3: ARC3_AUTO_FRAME_DIFF (per-turn prompt diff) exists but is off by default and was not in the submission env; WRITEUP.md section 6 reports no result for it"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "Fix bug: a tool timeout or crash wipes every retained function and drops the snippet's output",
   "category": "bugfix-quickwin",
   "description": "Two changes. (1) In tool_agent.py `_record_retained_functions` (6512-6571): if `sandbox_result` has no 'keepable_functions' key (the timeout, crash and invalid-response paths in python_tool_sandbox.py:1001-1038, 1107-1112 omit it), keep `self._kept_functions` unchanged instead of replacing it with {}. (2) Add a soft deadline in the sandbox child. In `main()`, set `signal.setitimer(signal.ITIMER_REAL, timeout-3)` with a handler that raises a new `ToolTimeoutError(BaseException)`. Add it to the except tuple at line 743 so the child still sends its 'error' message with partial stdout, action_results and retention payload. Keep the host hard-kill as a backstop. Also put a one-line 'executed before timeout: UP, UP, UNDO x12' summary into the error payload from `host_action_results`.",
   "mechanism": "Each timeout today has three costs. The model loses all of its game-scoped helpers and must spend requests redefining them. It gets an empty stdout, so it cannot see what its loop measured. And the board is in an unknown, partly executed state. Each costs several thousand generated tokens of recovery. The fix makes all three disappear for nearly zero effort.",
   "evidence": [
    "ARC3-Inference/inference/agent/python_tool_sandbox.py:1001-1011: on timeout it returns {'error','stdout':'', 'action_results'} with no keepable_functions",
    "ARC3-Inference/inference/agent/tool_agent.py:6529-6566: an empty keepable list replaces the pool and emits 'Your previous function X is no longer retained'",
    "fz-arms-v4-control transcripts: cn04-2fe56bfb_p0.txt:28608, g50t-5849a774_p0.txt:25629, re86-8af5384d_p0.txt:18271, tr87-cd924810_p0.txt:8092 all show the model noticing that every retained function was cleared right after 'Tool timed out after 30s'",
    "Timeouts per run across 7 full fz runs: 7, 7, 2, 6, 11 (12 streams), 8 (14 streams), 7 (from grepping '^Tool timed out after' in /Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs/fz-*/transcripts)",
    "Same-name function redefinitions: 331 in one 25-game run (fz-arms-v4-control, my scan). Some are edits; timeout wipes add to them"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Stop counting host-side action time against the 30 s tool limit (and remove the hidden 30 s clamp)",
   "category": "bugfix-quickwin",
   "description": "In python_tool_sandbox.py `run_sandboxed_python`, time each `action_handler(...)` / `animation_handler(...)` call and push `deadline` back by that elapsed time. Runaway model compute stays bounded by the child's RLIMIT_CPU (timeout+1 s of CPU, line 323), which does not count time spent blocked waiting on the host. Also remove the clamp in tool_agent.py:5131 (`self._python_timeout = min(30, ...)`). That clamp silently ignores LOCAL_ANALYZER_TOOL_TIMEOUT above 30. Keep the prompt line 'hard time limit of 30 seconds' (prompts.py:378) accurate by rewording it to '30 seconds of computation; time spent executing actions does not count'.",
   "mechanism": "Long action loops (rewinds, enumerations, executing a found plan step by step) are killed mid-plan when host overhead per action is about 0.7-1.5 s under 10-14 concurrent games. That wastes the plan, leaves the state ambiguous and triggers the bug in the previous idea. More streams make it worse, which matters if the user raises the stream count.",
   "evidence": [
    "ARC3-Inference/inference/agent/python_tool_sandbox.py:905-908 docstring Note: 'time the host spends inside action_handler/animation_handler counts against it'",
    "ARC3-Inference/inference/agent/tool_agent.py:5131: `min(30, max(1, _LOCAL_ANALYZER_TOOL_TIMEOUT))` means the env var cannot raise the limit",
    "sk48-d8078629_p0.txt:8548-8732 (fz-arms-v4-control): timed out after 37 UNDOs (~0.8 s/action); g50t:25629 after 20 actions (~1.5 s/action); tr87:8683 after 41 actions (~0.73 s/action)",
    "Timeouts rise with streams: 11 at 12 streams (fz-ledger-streams12) vs 6-7 at 10 streams"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Persistent per-game Python kernel (true REPL) instead of a fresh process per call",
   "category": "tools-repl",
   "description": "Turn `_SANDBOX_BOOTSTRAP` into a long-lived child per game. It loops on `{'type':'exec', code, state_delta}` messages and keeps `runtime_globals` between calls: variables, classes, imports and numpy arrays all persist. Provided globals (current_frame, history, ...) are refreshed before each exec, and the history is sent incrementally (see the incremental-state idea). Timeouts become a soft SIGALRM-raised BaseException, so the kernel survives. On a hard kill or crash, restart the kernel and rehydrate from the existing retained-function pool (retained_functions.py), so worst case equals today's behavior. Add RLIMIT_AS (e.g. 2-4 GiB) and cap live kernels. Kill kernels for games parked at the priority gate and rehydrate on re-admission, because only 10 streams are active. Prompt change: replace 'Python variables reset between tool calls' (tool_agent.py:3841-3843, prompts.py:356) with 'variables persist for this game; provided globals are refreshed each call; don't bind provided globals as default arguments'. The last clause is the pitfall the GB10 arm found. Flag: `ARC3_PERSISTENT_KERNEL`.",
   "mechanism": "This works with the model's habits instead of against them. The model already assumes variables persist and tries to force persistence with pickle, open, globals and builtins. Persistence removes those errors and the recomputation (re-parsing boards, re-running searches, re-building tables). It also cuts the 50-100 ms process spawn and the full-state JSON on every call. In the GB10 test it gave shorter requests: more model turns for the same decode budget, which means more levels per token.",
   "evidence": [
    "ARC3-Inference/CONFIGURATION.md section 7: 'This is source-based function restoration into fresh runtime globals, not a persistent Python kernel'",
    "Across 7 fz runs (17,242 tool calls): 377 NameErrors (2.2%); top names in fz-arms-v4-control are earlier-call variables such as 'a', 'CELLS', 'A', 'W4', 'Ar', 'sols', 'data'",
    "ImportErrors from persistence attempts across the fz runs: pickle 10, builtins 4; 'open' and 'globals' NameErrors appear in fz-arms-v4-control",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/gb10-wt/sandbox/notes/gb10-arms/sandbox.md: persistent interpreter plus helpers gave 0 python errors in 47 calls; on sk48, 666 vs 923-1304 completion tokens per request and 26 vs 13-18 requests in the same 45 min (n=2 games, inconclusive on score)",
    "https://arxiv.org/abs/2402.01030: CodeAct (ICML 2024) runs code actions in a persistent Jupyter-kernel-backed interpreter, with up to 20% higher success than JSON/text actions"
   ],
   "effort": "1 day",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Observed state-transition graph helpers: untried(), known_path(), goto(), plus segment-based click candidates",
   "category": "tools-repl",
   "description": "Add a sandbox-side helper (pure Python in the bootstrap, built from `history`) that hashes each frame, by default with the same crop as `_interior_state_hash` (tool_agent.py:617). It builds a level-scoped graph: node = board hash; edge = (action signature, using noop_repeat_guard.action_signature conventions) -> next hash; no-op and fatal flags come from transitions[i].result. Expose four functions. `untried(frame=current_frame)`: valid key actions plus click candidates not yet tried from this exact state. Click candidates are one per non-background, non-HUD segmentation node (a center cell), ranked small and rare-color first. `known_path(target_frame_or_hash)`: shortest observed action sequence from the current state, including UNDO edges. `goto(target)`: executes that path, checks the hash after each step and stops on mismatch. `graph_stats()`: states, edges, frontier size. One short line in FRAME_DIFF_HINT-style prompt text.",
   "mechanism": "This turns exploration bookkeeping, which the model now does in its head across trimmed context, into one call. It stops repeated clicks on the same inert segments from the same state, and replays a known-good prefix after a death or a long rewind in one call instead of re-deriving it. Clicks are about 28% of executed actions and occur in about 17 of 25 games, so cutting the click search space from 4,096 cells to a few segments saves both actions and thinking.",
   "evidence": [
    "https://github.com/wd13ca/ARC-AGI-3-Agents: Blind Squirrel (2nd place, ARC-AGI-3 preview) keeps a state graph of 'every previously encountered game state, the actions taken from that state, and the resulting next states' and uses segments as click targets, cutting the action space by about 2 orders of magnitude",
    "https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore: graph exploration with frontier tracking and segment-priority clicking solved 12 (later 14-19) private preview levels with no learning",
    "fz-arms-v4-control executed-action lines: MOUSE is about half of the actions in s5i5, vc33, lp85, r11l and tn36, and present in 17/25 games (my count)",
    "sk48-d8078629_p0.txt:8380-8548: the model hand-writes `snap(state(...))` state comparison and a 45-iteration UNDO rewind loop to return to a remembered state",
    "ARC3-Inference/inference/agent/tool_agent.py:617-642 and noop_repeat_guard.py: state hashing already exists, but only to refuse actions (guards were off in the submission); nothing model-facing"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "World-model checker plus anytime search: check_model(parse, step) and search(start, succ, goal, budget_s)",
   "category": "tools-repl",
   "description": "Add two generic helpers to the bootstrap. (a) `check_model(parse, step, level=None, limit=5)`: for every transition on the current level (or a given one), compute `step(parse(before_frame), action)` and compare it to `parse(after_frame)`. Report 'k/n transitions explained' and the first `limit` mismatches as compact diffs of the parsed state (dict/tuple). (b) `search(start, successors, is_goal, key=None, budget_s=12, max_nodes=300000)`: BFS, or best-first when a heuristic is passed. It checks a time budget and returns `{'path':..., 'expanded':..., 'status':'found'|'budget'|'exhausted', 'best':...}` instead of hitting the 30 s kill. Advertise both in two prompt lines next to the existing BFS advice (prompts.py:263-264): 'before searching over your simulator, run check_model to confirm it reproduces observed transitions'.",
   "mechanism": "The model already writes simulators and searches, but runs them unverified. A wrong simulator produces confident multi-action plans that die or no-op, which costs actions, deaths and long re-analysis. Checking the model against all recorded transitions in one call catches rule errors before actions are spent (the WorldCoder / code-world-model loop). A search that respects its budget turns search timeouts into usable partial results.",
   "evidence": [
    "https://proceedings.neurips.cc/paper_files/paper/2024/hash/820c61a0cd419163ccbd2c33b268816e-Abstract.html: WorldCoder (NeurIPS 2024) writes the world model as Python checked against its interactions; reported as more sample-efficient than deep RL and more compute-efficient than ReAct-style agents",
    "https://arxiv.org/abs/2405.15383: GIF-MCTS code world models are refined against environment trajectories and unit tests, then used for planning",
    "fz-arms-v4-control: 136 model-defined functions named like sim/step/solve/search/bfs/path/plan (bfs 18, step 15, solve 10, sim 8), i.e. the model already builds simulators",
    "3 of 7 control timeouts were search or enumeration blowups (cn04 DFS at transcript line 28608, re86 at 18271, tr87 at 8092); 'time.time()' appears 135 times in fz-run code (manual time budgets)",
    "Caveat: /Users/advaith/Documents/studying-stuff/arc-agi-3-v3/gb10-wt/sandbox/notes/gb10-arms/sandbox.md shows a provided path() helper used 0 times, so adoption needs a prompt nudge and must be measured"
   ],
   "effort": "1 day",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Let the model read its retained functions' source (inspect.getsource shim / source())",
   "category": "tools-repl",
   "description": "The host already holds every retained function's source (`self._kept_functions`). Pass the dict into the child and provide (a) `source(name_or_fn)` returning the stored text and (b) a tiny `inspect` shim module (allow 'inspect' in `_safe_import`, python_tool_sandbox.py:313) whose `getsource(fn)` reads the same dict and whose `signature` wraps the real one. Also, in `_retained_function_context` (tool_agent.py:6479-6489), append the first docstring or comment line of each function, truncated to about 60 chars, to the signature list in the opener.",
   "mechanism": "After context trimming, the code that defined a game-scoped helper is gone from the model's context and only its signature remains. The model then misuses it, redefines it, or tries `inspect.getsource` and gets an ImportError, which wastes a request. Cheap access to the source keeps helpers reusable across trims and levels, so fewer tokens go to rewriting code.",
   "evidence": [
    "`inspect.getsource(` appears 70 times in fz-run code; 28 'ImportError: Module 'inspect' is not allowed' across the 7 full fz runs (17,242 calls)",
    "ARC3-Inference/inference/agent/tool_agent.py:6479-6489: the opener lists only `function_signature(src)`",
    "ARC3-Inference/inference/agent/python_tool_sandbox.py:85-106: 'inspect' is not in SAFE_MODULES",
    "WRITEUP.md section 2: history trims down to about 57k retained tokens from a 128k window, so the defining snippets regularly fall out of context"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Make the REPL tolerate the model's common guesses: time/sys shims and missing attributes",
   "category": "bugfix-quickwin",
   "description": "In the sandbox bootstrap: allow a `time` shim exposing only time(), perf_counter() and monotonic(), and a `sys` shim exposing setrecursionlimit (capped, e.g. at 20000) and getrecursionlimit. Add `.step` and `.level` properties to HistoryEntryView and TransitionView (proxy to the frame). Add `FrameView.__getitem__(r)` returning the ascii row string, so `current_frame[r][c]` works. Add `.items()/.values()` to `_LazyAnimation`. Make `build_animation_view` (utils/animation.py:427-435) return `{'error':..., 'steps': [], 'frames': 0}` instead of a dict without those keys. Update the module list in prompts.py:253, which also omits the already-allowed difflib (sandbox docstring Note).",
   "mechanism": "Every Python error costs a full extra request (about 1.5k reasoning tokens plus code), and some fire after actions already ran, leaving a half-executed plan. These are the model's most frequent unforced errors. Making them succeed is free.",
   "evidence": [
    "ImportErrors across the 7 full fz runs: time 43, sys 16 (sys.setrecursionlimit 12x in code), plus 'time.time()' used 135x for search budgets",
    "Error messages in fz-arms-v4-control + fz-ledger-on-trim: 'HistoryEntryView' has no attribute 'step' (6), KeyError 'steps' (5), KeyError 'frames' (3), 'FrameView' object is not subscriptable (3), '_LazyAnimation' has no attribute 'items' (2), 'TransitionView' has no attribute 'step' (2)",
    "Of about 4.7k tool calls in those two runs, about 330 ended in a Python error (excluding guard refusals); KeyError, IndexError and ValueError often came after actions had executed (18, 12 and 12 cases)",
    "ARC3-Inference/inference/agent/python_tool_sandbox.py:195-226: FrameView and HistoryEntryView lack those members"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "Render tool results as plain text instead of JSON with escaped newlines",
   "category": "tools-repl",
   "description": "In tool_agent.py `_render_tool_payload` (7476-7508), stop returning `json.dumps(result, indent=2)`. Emit plain text instead: stdout verbatim with real newlines, then compact `error:` / `result:` / `function_retention:` / `truncated:` lines. Only the `result` value (when not a string) stays JSON. Keep truncation and the board-dump guard as they are. Flag: `ARC3_PLAIN_TOOL_OUTPUT`.",
   "mechanism": "Today any multi-row board crop the model prints reaches it as one long line, like \"WWbb\\nWbbb\\n...\". That destroys the 2D column alignment the model relies on to read local geometry, and adds about 11% characters. Plain text keeps crops spatially aligned, which should mean fewer misreads, fewer verification calls and a slightly smaller context.",
   "evidence": [
    "ARC3-Inference/inference/agent/tool_agent.py:7508 (`return json.dumps(result, indent=2)`) and 9673-9680 (that string is appended verbatim as the role=tool message)",
    "fz-arms-v4-control: 329 of 2,275 tool results (14.5%) contain board crops of 3 or more rows; re-encoding the displayed results as the harness does inflates them by 11.3% (my measurement)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/gb10-wt/sandbox/notes/gb10-arms/sandbox.md: the plain-text result arm averaged 408 vs 657-783 chars per result with a higher cache hit (confounded with other changes)",
    "https://arxiv.org/abs/2405.15793: SWE-agent ablations credit gains to agent-facing interface formatting (concise, structured feedback)"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Incremental runtime-state transfer and per-frame segmentation cache (make action() O(1))",
   "category": "tools-repl",
   "description": "Today every `action()` call goes through the full runtime-state JSON. The solver rewrites it with indent=2 after every action, including each action inside a batch (solver.py:1876, runtime_state.py:281). The agent reloads and re-serializes the whole history plus 64x64 grids (tool_agent.py:7957-7982). The child re-parses everything and rebuilds every FrameView, which also drops cached segmentations (python_tool_sandbox.py:419-447). Replace this with three changes. (1) Keep the HistoryEntry list in memory and pass only entries appended since the child's last refresh. The child appends them and keeps its FrameView objects, so segmentation caches survive. (2) Write the on-disk state once per tool call, without indent. (3) Cache segmentation by grid hash in the child (or the host), so history scans don't recompute the pure-Python segmentation (~47 ms per frame).",
   "mechanism": "Per-action host overhead grows linearly with game length and runs under one GIL shared by all 10 game threads. That causes action-loop timeouts late in games and adds latency to every tool round trip, during which the game's GPU slot sits idle. Constant-time refresh keeps late-game turns as fast as early ones.",
   "evidence": [
    "My benchmark of the repo's runtime_state.py (/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/bench/b.py, one M-series core): history 50 gives a 3.2 MB file and ~0.05 s per action round trip; 200 gives 12.7 MB and ~0.22 s; 500 gives 31.7 MB and ~0.56 s (write + load + serialize + child parse)",
    "My benchmark (scratchpad/bench/s.py): segment_layer ~47 ms per 64x64 frame; frame_diff ~0.1 ms once segmentations exist",
    "fz-arms-v4-control: games run 34-545 actions (summary.txt), and history_entries is never truncated across levels (solver.py:1873-1876)",
    "Observed 0.73-1.5 s per action in timed-out loops (sk48:8548-8732, g50t:25629, tr87:8683)"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Expose numpy and the numeric grid in the REPL",
   "category": "tools-repl",
   "description": "The child runs `python -I -S`, so numpy cannot be imported. Pass the host's numpy site path explicitly, appending only that directory to sys.path in the bootstrap, and add 'numpy' to SAFE_MODULES (python_tool_sandbox.py:85). Give FrameView a `.grid` property: a cached tuple-of-tuples, or an np.ndarray when numpy is imported. Change the 'raw numeric grid is intentionally not exposed' lines (prompts.py:193, tool_agent.py:429) to document `.grid` and `import numpy as np`. Pairs best with the persistent kernel, which pays numpy's ~0.1 s import once per game instead of per call.",
   "mechanism": "Qwen models are very fluent in numpy: np.argwhere(G==c), bounding boxes, (G!=P) diffs, slicing. That replaces the hand-written ascii parsing and color-search loops the model writes in about 30-40% of calls, which means fewer code tokens and fewer indexing bugs. This is an A/B, because Tufa and Franzen deliberately hid the numeric grid to steer the model toward segmentation.",
   "evidence": [
    "ARC3-Inference/inference/agent/prompts.py:193: 'The raw numeric grid is intentionally not exposed'; python_tool_sandbox.py:947 launches with -I -S",
    "fz-arms-v4-control: 38.6% of tool calls split `.ascii` into rows by hand (395 identical `a=current_frame.ascii.split('\\n')` lines) and 28.8% hand-search cells by color letter (my regex scan)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/GB10-OVERNIGHT-RESULTS.md: the numpy-perception arm was 'promising -> Kaggle' (cleared ar25 L2 in 2/2 runs vs 2/8 controls; small sample)",
    "gb10-wt/sandbox/notes/gb10-arms/sandbox.md: the numpy-backed `find()` helper was invoked 82 times in one sk48 game, so the model adopts numpy-style helpers readily"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Small starter helper set in the model's own style, delivered through the retained-function channel",
   "category": "tools-repl",
   "description": "Seed `self._kept_functions` at game start with 5-6 short, readable helpers matching what the model rewrites most: `crop(r0,r1,c0,c1,f=None)` (with row/col rulers), `find(color,f=None)` returning cells, bbox and count, `bbox(cells)`, `objs(f=None, min_px=1, ignore=('B',))` (compact segmentation table), `delta(a=None,b=None)` (one-line frame_diff), and `bfs(start, nbrs, goal)`. Because they live in the existing retention pool, they appear in every opener's 'retained functions' list. Label them honestly ('starter helpers; editable; use source(name)'), and the model can redefine them freely. Leave out structural helpers the model ignored in testing. A/B it with a flag.",
   "mechanism": "Code is about 10% of generated characters, and the model re-writes the same utilities at the start of games and levels (1,092 definitions per 25-game run). Pre-seeding the ones it already writes saves code tokens and avoids the small bugs in each rewrite, without adding new concepts. That follows Tufa's 'lean into the model's natural biases' advice rather than a hand-crafted toolkit.",
   "evidence": [
    "fz-arms-v4-control most-defined function names: crop 31, bfs 18, snap 11, show 8, objs 7, comps 5, bbox 4; 1,092 defs, 605 unique names, 331 same-name redefinitions (my scan)",
    "gb10-wt/sandbox/notes/gb10-arms/sandbox.md: habit-matching helpers were used (find 82, changes 19, delta 11); structural ones were not (path 0, L 0, acts 0)",
    "https://arcprize.org/blog/arc-prize-2026-milestone-1: Tufa reports that 'hand-crafted tools actually hurt the model; letting it improvise worked better', so the set must stay tiny, optional and A/B tested",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/final-c1/talk_by_tufa_lab.md: 'Lean into the model's natural biases instead of fighting them'"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "History effect table and object tracks: effects() and track()",
   "category": "tools-repl",
   "description": "Building on the stored per-action `effect` from the first idea (or a cached frame_diff per transition), add `effects(level=None, action=None, last=30)`. It returns one line per past transition: index, action, effect summary, NO-OP / GAME OVER / LEVEL flags, and bar delta. Also add `track(hash=None, color=None, px=None)`, which returns the (step, bbox) positions of matching segmentation nodes across this level's frames, using cached segmentations. Both are read-only and print compactly.",
   "mechanism": "Rule induction, e.g. 'what does SPACE do in different contexts' or 'when does the red block become marked', currently means writing and debugging a history-scanning parser each time. These calls are slow (segmentation on every frame), sometimes time out, and burn requests. One call that answers 'what did each action do so far' should shorten hypothesis cycles, which are the long-thinking replies that make up about 25% of reasoning.",
   "evidence": [
    "sk48-d8078629_p0.txt:8380-8470 (fz-arms-v4-control): the model hand-writes `state()` + `hudicon()` over all 70+ history frames to test a hypothesis about which actions mark the R icon",
    "fz-arms-v4-control: 7.9% of tool calls loop over history/transitions; replies over 20k reasoning chars hold 24.5% of all reasoning characters (my scan of reasoning_chars)",
    "My benchmark: segmentation ~47 ms/frame in pure Python and recomputed for each fresh FrameView, so a 300-frame scan approaches the 30 s limit"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  }
 ]
}
```
