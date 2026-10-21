# Sandbox v3: what to take from the Wang-era sandbox work, built on sandbox-robust (#76)

Date: 2026-10-08. Status: design only. Nothing here is implemented or run.

Owner's request (2026-10-08): "i made many sandbox improvements on the wang notebook; perhaps you can smartly mix and combine to make an even better sandbox?"

## TL;DR

- **Base.** Build on `arm/sandbox-robust` (#76: carried plain-data variables and imports, plus shims). #76 already covers the largest measured error class: 271 of 374 NameErrors were names an earlier call assigned. Every feature below is a separate default-off flag on top of it. With all flags off, requests must be byte-identical to `main`.
- **What the Wang work offers that Franzen lacks.** Four things are worth taking, plus one hardening item:
  1. **`ARC3_SANDBOX_ERRORS`: errors that teach the API.** The traceback names the failing line's source, including inside retained functions, which `main` hides today. KeyErrors on harness dicts list the keys that do exist. Short hints are added for the recurring mistakes. No prompt text. It targets the 2.7% of tokens lost to Type/Index/Value/Key/Syntax errors that #76 does not touch.
  2. **`ARC3_SANDBOX_ECHO_DELTA`: an exact change line on every `action()` echo.** `[action] RIGHT -> effect` becomes `[action] RIGHT -> effect: 30 cells [15,18,23,29] B>b x15, b>B x15 | B moved (+0,+3)`. This is the "exact change information" lever. It is the best-supported idea from the GB10 night (5 of 5 arm runs cleared ar25 L2 vs 2 of 8 controls, pooled post hoc). It is delivered inside the tool result of the call that acted, so the model's own filtering code cannot drop it.
  3. **`ARC3_SANDBOX_HELPERS`: cell-level helpers in letter space, pure Python.** `delta`, `where`, `bbox_of`, `color_counts` and `show_crop`. These are Wang v2's `delta/find/bbox/count/show` without numpy, renamed to avoid names the model already uses. The tool description grows by about 130 tokens.
  4. **`ARC3_SANDBOX_STORE`: a game-scoped `store` dict.** It survives level changes and deaths, using #76's literal-only carry. This is Wang's `store`/`remember`, which the model used in 5-7% of Kaggle calls. It replaces the 32 `open('/tmp/...')` persistence attempts seen in two Franzen-era runs.
  5. **`ARC3_SANDBOX_HARDEN`: hardening only, no score claim.** An address-space cap, `nice`, and a stdout cap in the child (#33 and Wang v2).
- **Lower priority.** `ARC3_SANDBOX_SOFT_TIMEOUT` (partial stdout plus the line where a timeout hit) and `ARC3_SANDBOX_RESULT_TEXT` (plain-text tool results instead of escaped JSON) both have weak or confounded evidence.
- **Not taken.**
  - The persistent per-game interpreter: #76 gets most of its benefit and keeps Franzen's per-snippet guard closures intact.
  - numpy `.grid`: the GB10 gain traced to the perception text block, not numpy (`.grid` was read in 2 of 17 calls); it also adds about 65-80 ms per call.
  - The learned predictor and `mechanics()`: 0 `register_predictor` calls in 2,762 Kaggle calls.
  - `track_objects`/`detect_*` (0-6 uses), `path()` BFS (0 uses), `timeout_seconds` >30 s, ruff format/lint echo, and the `L`/`acts` aliases (0 uses).
  - Lord Han Solo-style `save_module` with regression tests is deferred. It has no measured effect, his write-up says he "cannot confirm that any of these changes improved the score", and Makarov's controlled study found the model ignored an executable world-model scaffold.
- **Eval.** Compare at matched wall-clock (121 min, 25 games), with ≥2 repeats per arm, paired per game with `make significance`. Report score-by-wall-minute curves, not per-token or per-runtime figures. A 20-min panel is only a crash and timing screen.

## 1. Sources and how much they can carry

| Source | What it is | Weight |
|---|---|---|
| `wang-duck-harness` `my-main` `c2084ed`, `194f2cb`, `804d450`, `8da10ee`, `05fdabb`, `9fceefd` | Persistent interpreter, store, predictor, adaptive timeout, perception primitives, ruff, raw grid, numpy perception | Shipped in every Wang-era Kaggle run (0922 baseline → c1 → final-c1), but never A/B'd in isolation |
| `gb10-wt/sandbox` (`gb10/sandbox`, `b0f606a`, `969f4e6`, note `notes/gb10-arms/sandbox.md`) | SANDBOX_V2: helpers, error text, stdout cap, plain-text results, incremental history, RLIMIT_AS | One GB10 batch, 2 games, 47 calls: process evidence only |
| `gb10-wt/numpy-perception` (`59dd590`, `1bdb928`) | numpy `.grid` + `<perception>` block | 2 GB10 runs; the gain is attributed to the block |
| `gb10-wt/frame-diff`, `action-tools`, `state-manager`, `token-efficiency` | Auto diff text, `act` tool, state record, prompt shrink | Context only (not sandbox) |
| `GB10-OVERNIGHT-RESULTS.md` | Cross-arm summary | "Exact change information is the one lever that showed up across arms" (5/5 vs 2/8, pooled Fisher p ≈ 0.03, chosen post hoc) |
| Wang-era Kaggle transcripts (`kaggle-outputs/b1-thinking-real`, `final-c1-ledgerfix`) | 2,762 python calls | Measured here for helper adoption (§2.2) |
| Franzen `main` | `python_tool_sandbox.py`, `tool_agent.py`, `prompts.py`, `retained_functions.py`, `frame_diff.py` | Code read |
| `arm/sandbox-robust` (#76) | Persisted variables and shims | PR body: error breakdown over 17,490 calls in 9 runs |
| Issues #16, #18, #20, #23, #33 | Bottleneck audit | Labeled `MAYBE` |
| `research/reports/failure-taxonomy.md` (#77) | Token shares by failure cause | 7 runs, 32.7M tokens |
| `research-sweep.md` | Earlier verdicts on numpy, persistent kernel, soft timeouts, perception helpers | Consistent with this doc; cited where reused |
| Web (2026-10-08, read-only) | Lord Han Solo write-up, rellik13 repo (Wang-Ludvig port), Makarov controlled study (743723), gedouluhui (743060), Tufa Duck, CodeAct, Self-Debug, Voyager, Tool Documentation | Mostly claims; see §2.4 |

Scale caveat: Wang-era scores (10-19 offline) and Franzen-era scores (46-53) are not comparable. The Wang-era error rates also come from different prompts, so they can only be compared within an era.

## 2. Inventory

### 2.1 Every sandbox feature in the owner's Wang-era work

| # | Feature | Where | Behaviour |
|---|---|---|---|
| W1 | Persistent per-game interpreter | `c2084ed` | One subprocess per game. `exec` messages go into the same globals. Respawn after a crash restores a JSON-safe `store` snapshot. Per-exec SIGALRM budget (env round-trips excluded); a timeout keeps state. |
| W2 | `store` dict + `remember/recall/forget` | `c2084ed` | Scratch that survives between calls (JSON snapshot across respawns). |
| W3 | Learned predictor | `c2084ed` | `register_predictor(fn)` records `predict(state, action) -> state`; every real transition is auto-scored; `predictor_stats()` gives exact-match rate and cell accuracy; each `action()` returns a `prediction` field. |
| W4 | `mechanics()` | `c2084ed` | Table of observed `<s, a> -> s'` transitions. |
| W5 | Adaptive timeout | `194f2cb` | Optional `timeout_seconds` tool argument, clamped to [30, 120] by `LOCAL_ANALYZER_TOOL_TIMEOUT_MAX`. |
| W6 | Perception primitives | `804d450` | `track_objects`, `detect_transform`, `detect_symmetry`, `detect_events` (stdlib). |
| W7 | Ruff format + pyflakes lint | `8da10ee` | Host-side `ruff format` + `ruff check --select F`. Results gain `lint` and `formatted_code`. A no-op on Kaggle (ruff is a dev dependency). |
| W8 | Raw grid | `05fdabb` | `current_frame.numeric_grid` / `raw_grid()` behind `EXPOSE_RAW_GRID`. The same commit adds labeled-grid and diff images (vision, not sandbox). |
| W9 | numpy perception | `9fceefd`, `59dd590`, `1bdb928` | Read-only int ndarray `.grid`, `np` preloaded, `show/components/changes/animation_steps`. Fixes: numpy import failure no longer kills the sandbox, numpy scalars JSON-safe, one BLAS thread. |
| W10 | SANDBOX_V2 preloads | `b0f606a` | `G`, `P`, `acts`, `level`, `L` (dict emptied on level change, with a notice). |
| W11 | SANDBOX_V2 helpers | `b0f606a` | `find(color)`, `bbox`, `count`, `delta(P, G)` (one line: count, bbox, transitions, rigid moves per color, appear/vanish; colors covering >1/4 of the board are treated as background), `path()` (BFS for an h×w object, `step` cells per action). |
| W12 | Builtins/modules | `b0f606a` | `class` (`__build_class__`), `object/super/property/...`, common exceptions, `Counter/defaultdict/deque/heapq/math/itertools/collections` preloaded; `dataclasses/typing/enum` importable. |
| W13 | Error text | `b0f606a` | `line 7: <source>`; `in f() (defined in call 4) line 2: <source>` for functions from earlier calls; hints: name is a `store`/`L` key, difflib close match, "a name exists only if a call reached its assignment", FrameView attributes, unhashable list/ndarray/dict, tuple index on list, array truth value. Hint code can never kill the interpreter. |
| W14 | Timeout as `BaseException` | `b0f606a` | A model's `except Exception` cannot swallow it. The message names the line and says names and actions before the timeout are kept. |
| W15 | Stdout cap in the child | `b0f606a` | Head 8,000 + tail 4,000 chars, cut at line boundaries. |
| W16 | Plain-text tool result | `b0f606a` | Restart/level notices, then stdout, `result:`, `error:`, `lint:` lines. Head+tail trim. No `formatted_code`; REPL-noise lint codes dropped. A silent snippet gets an `actions: ... -> level, state` line. |
| W17 | Incremental history | `b0f606a`, `969f4e6` | The host sends only new history entries plus `history_offset`; the interpreter re-syncs on mismatch (`history_len`). |
| W18 | Resource limits | `b0f606a` | `RLIMIT_AS` 4 GiB (set after numpy import), one BLAS thread, broken pipe becomes a tool error, restart announced. |
| W19 | Transcript trace | `b0f606a` | `[SANDBOX V2] helpers: find=3 ...`, error kind, notices (never sent to the model). |
| W20 | Single reference card | `b0f606a` | The tool description lists every name with its signature (~500 tokens). The ~700-token numpy section is dropped: +326 tokens vs control. |

### 2.2 Measured effect of each

**Kaggle, Wang era.** Two runs, `b1-thinking-real` and `final-c1-ledgerfix`, with 1,323 + 1,439 = 2,762 python calls. Both had W1-W6 on; W7 was a no-op. I counted with a parser over `[TOOL CALL: python]` and `[TOOL RESULT: python]` sections.

| Feature | Calls whose code uses it |
|---|---|
| `register_predictor` | **0** |
| `predictor_stats` | 2 |
| `mechanics(` | 8 |
| `track_objects` | 3 |
| `detect_events` | 6 |
| `detect_transform`, `detect_symmetry` | **0** |
| `remember(` | 189 (6.8%) |
| `recall(` | 103 (3.7%) |
| `store[` | 152 (5.5%) |
| `timeout_seconds` (adaptive) | ≤15 (≤0.5%) |
| `.segmentation` / `.ascii` | 920 / 1,132 |

Errors in those runs: 82 of 2,762 results (3.0%). NameError was 13 (0.47%), against 374 of 17,490 (2.1%) in Franzen-era runs. That gap is consistent with true persistence removing most NameErrors, but the prompts and harnesses differ.

**GB10 SANDBOX_V2.** One batch (b4), ar25 + sk48, 45 GB10-min each:

- 0 python errors in 47 calls, against 2 in 142 calls across 8 controls.
- Helpers appeared in 34 of 47 calls: `find` 82 invocations (sk48), `changes` 19, `delta` 11, `show` 9, `bbox` 1. `path`, `L`, `acts` and `components` were never used.
- sk48: 26 requests and 15 turns, against 13-18 and 8-11 for controls. Completion tokens per request were 666, against 923-1,304.
- Plain-text results were shorter: 408 chars per result on sk48, against 657-783.
- Cache hit 0.745/0.755, against at most 0.717/0.687.
- ar25 reached L2, which is within control variance.
- Two new failure modes:
  - `def f(g=G)` binds a stale board at definition time.
  - `delta()` printed `moved (+0,-3): color 0; moved (+0,+3): color 4` (correct), but the model read it as "Confusing" and re-derived it.

**GB10 numpy-perception.** Two runs, both reaching ar25 L2 (2/2 vs 2/7 controls, p = 0.17). In both, the transcripts trace the key fact to the `<perception>` text block. In v2 the model read `.grid` in 2 of 17 ar25 calls. A numpy-only arm was never run.

**Kaggle numpy, raw grid, ruff and labeled images.** numpy and the raw grid never ran on Kaggle. Ruff was inert there. b3 labeled-diff image: 12.34 vs b1 17.74.

### 2.3 Franzen `main` and #76

**What `main` has.** On `main`, `python_tool_sandbox.py` (974 lines) is a fresh `python -I -S` child per call. It has:
- an import allowlist of 15 modules (including `difflib`) and a restricted builtins set (narrow exception classes added);
- `RLIMIT_CPU` timeout+1, `FSIZE` 1 MB, `NOFILE` 32; no `RLIMIT_AS`, no nice (#33);
- a host deadline of 30 s, with env round-trips counted against it.

The REPL-like part: the last expression is auto-printed and becomes `result`. The provided state is:
- `current_frame`/`previous_frame`/`history`/`transitions`/`last_transition` views (`.ascii`, `.segmentation`, `.step`, `.level`, `.shape`; no numeric grid by design);
- `valid_actions`, `last_action_call_result`;
- `frame_diff()`: object-level, Hungarian matching over segmentation, which costs 2 segmentations per call (25-475 ms on real boards, #23);
- `last_animation_frames` / `last_animation_timeline` (lazy, fetched on demand);
- `death_ledger`.

`action()` sends `stale_after` and raises the guard errors. Of these, `StaleStateActionError` is catchable but cannot really be caught and retried: the flag is never cleared (#18). The action echo `[action] X -> effect|NO-OP|...` is printed into stdout.

**Function retention.** `ARC3_PERSISTENT_FUNCTIONS=1`, `SCOPE=game`, `IMPORTS=1` in the competition config. Plain top-level `def`s are retained and recompiled from source as `<retained>` in the next call. Names that conflict with provided globals are rejected.

**Errors and output.** `_sanitize_exception` keeps only `<python_tool>` frames, as `File "<python_tool>", line N, in name`, with no source text. Frames inside retained functions (`<retained>`) are dropped, so an error inside a retained helper points only at the calling line. About 20% of tracebacks are raised inside a snippet-defined function: 35 of 169 in ledger-r2 and 39 of 202 in control. A syntax error returns `Python syntax error: invalid syntax (<python_tool>, line 10)` without the line. The tool result is `json.dumps(payload, indent=2)` with stdout as an escaped string, middle-truncated to 3,072 tokens (`ARC3_MIDDLE_TRUNCATION=1`).

**#76 adds:**
- `ARC3_SANDBOX_PERSIST_VARS`. Plain-data top-level variables round-trip through a strict literal reader (never evaluated), along with imports. Views, functions and loop targets are not kept. Caps are 100k chars per variable and 400k total. Variables clear on a level change and survive deaths; imports last the game. A timeout or crash keeps state (#51's rule, under the flag). Retained helpers may read carried names. The NameError hint says why a name was dropped.
- `ARC3_SANDBOX_SHIMS`. Stand-in `time`/`sys`/`inspect`, `globals`/`id`, `.step`/`.level` on history entries and transitions, and `.items()`/`.values()` on the timeline.
- Validation: 56 real-sandbox tests, a loop test through `analyze()` under the Kaggle `setup_env`, flags-off byte-identical to `main`, and 4 mutations caught. **Not run on Kaggle.**

**Errors #76 does not touch.** Measured over 7 runs (failure taxonomy):
- TypeError 194, IndexError 144, KeyError 135, ValueError 104, SyntaxError 50: 0.88M tokens (2.7%) as the outcome of a response.
- StaleStateActionError 269.
- 96 of 244 Type/Index/Value tracebacks in four runs are "list index out of range" or "min()/max() of an empty sequence" (a search found nothing).
- KeyErrors over five runs, counted here: `'steps'` 12, `'action'` 4, `'frames'` 3, `'stop_reason'` 2. These are mostly the animation timeline, which is `{}` when the last action did not animate, while the prompt describes it as "a dict with `action`, `frames` and `steps`".
- An error costs about one response (median 1.2k tokens). Chains are short: 15/8 pairs and 3/1 triples in control/ledger-r2.

**How the model uses Franzen's sandbox.** Counted in ledger-r2 (2,217 calls) and control (2,275):

| Pattern | ledger-r2 | control |
|---|---|---|
| `.ascii.split`/`splitlines` (hand-built char grid) | 36.7% | 38.6% |
| `def` | 30% | 30% |
| `.segmentation` | 23.3% | 19.8% |
| `frame_diff(` | 11.0% | 6.2% |
| `last_animation_*` | 7.3% | 5.3% |
| BFS (`deque`/`popleft`/`heappush`) | 5.8% | 4.7% |
| `prev.ascii` vs `cur.ascii` hand diffs | 1.4% | 0.9% |

Model-defined function names to avoid when naming helpers: `crop` 52, `bfs` 42, `show` 15, `cells_of` 13, `find` 6, `bbox` 5. Zero occurrences: `delta`, `where`, `store`, `bbox_of`, `color_counts`, `show_crop`.

### 2.4 What other harnesses and research add (web, 2026-10-08, read-only)

- **Lord Han Solo** (Milestone 2, 2nd open source; https://www.kaggle.com/writeups/lordhansolo/arc-agi-3-milestone-2).
  - `save_module` "keeps tested code across turns, levels and context truncation". Tests replay against the observations captured when they were added, and an update that breaks an earlier test is rejected.
  - After each action the model gets "the changed regions and the per-step changes in a batch". Animation frames are classified blink/tween/path/other.
  - **No measured effect:** "I cannot confirm that any of these changes improved the score." His stated main driver was throughput.
- **rellik13 / sirikilohit** (LB 22.53; https://github.com/LohitSiriki/arc-agi-3-milestone2-solution).
  - Ports Wang's harness (Wang-Ludvig): ephemeral `-I -S` child, `frame_diff` returning colour-transition groups.
  - Teaching errors, e.g. `DiffGroup.__missing__`: "diff group has no field 'from'; use from_color ... Diff-group fields: from_color, to_color, count, bbox, cells". No isolated measurement.
  - **Measured:** `action()` calls killed by the 30 s timeout rose from 0/328 to 36/150 as history grew (O(history) JSON rewrites, ~0.8 s each at 800 actions). That is the same failure family as our #2/#43.
  - Wang's harness design: 9.96 → 13.40 (single runs).
- **Makarov controlled study** (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743723).
  - An executable world model with replay validation and search planning scored at or below baseline. "The model largely ignored the world-model scaffold". Its simulators were mostly no better than a "nothing changes" predictor.
  - In 83% of forks between stuck and successful runs, the stuck run had already tried the decisive action: a goal-judgement failure, not coverage.
- **gedouluhui** (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743060). Connected-component click candidates were never isolated (one run, no matched control). The "+28%" is a two-line prompt patch (scoring formula + time line), 1.12 → 1.43. Our own scoring-guide arm scored 11.12 vs 17.74.
- **Tufa Duck** (https://github.com/Tufalabs/duck-harness): the origin of Franzen's design. Ephemeral child, same allowlist, user-frame tracebacks, no `frame_diff`, no persistence, raw grid deliberately hidden.
- **Research:**
  - CodeAct (arXiv 2402.01030): code actions beat JSON/text actions by up to 20% success. Supports the python-tool design, not a specific helper.
  - Self-Debug (2304.05128): execution feedback plus explanation gives +2-12%. Better error text is the cheap version of this.
  - Voyager (2305.16291): a verified skill library. Our retained functions are an unverified version; Lord Han Solo's tests are the verified version.
  - Tool Documentation (2308.00675): documentation alone matches few-shot demos. Supports documenting at the point of failure (KeyError lists keys) and one-line signatures in the tool description.

**Net effect on the ranking.** The external evidence has no measured A/B for any sandbox helper. It supports:
- teaching errors (Self-Debug, Tool Documentation, the Wang-Ludvig `__missing__` pattern);
- automatic per-action change regions (Lord Han Solo claims it; our GB10 night measured it on a different harness).

It argues against passive simulation and verification scaffolds for this model class (Makarov; our 0 `register_predictor` calls). That is why `save_module`/predictor ideas are deferred and change text is ranked high.

## 3. Feature-by-feature comparison

Verdict key: **take** = in sandbox-v3; **low** = flagged but lower priority; **have** = Franzen or #76 already covers it; **reject**; **defer**.

| Feature | Wang implementation + evidence | Franzen `main` / #76 status | Verdict |
|---|---|---|---|
| Persistent per-game interpreter (W1) | `c2084ed`. NameError 0.47% of Kaggle calls (vs 2.1% Franzen-era; different prompts). GB10 v2: 0 restarts in 47 calls. New failure: stale `g=G` defaults. | Fresh child per call + function retention; #76 carries plain data and imports (covers 271/374 NameErrors). Guards rely on per-snippet closures (`_stale_state`, `_terminal_state`, repeat map). | **have/reject.** #76 gives most of it. A long-lived child would need per-exec resets of guard state, a respawn path, and memory growth control for 10 concurrent games. research-sweep: "saving under ~2% of calls, <1% of tokens, risks high". |
| `store` + remember/recall (W2) | Kaggle: `remember` 6.8%, `store[` 5.5%, `recall` 3.7% of calls | #76 carries variables but clears them at level change; 32 `open('/tmp/...')` attempts in 2 runs; `pickle`/`builtins` imports refused | **take (low cost) → `ARC3_SANDBOX_STORE`**: an explicit game-scoped dict, literal-only |
| Learned predictor (W3) | 0 `register_predictor` in 2,762 calls; `predictor_stats` 2 | none | **reject** (Makarov: world-model scaffolds ignored) |
| `mechanics()` table (W4) | 8 uses / 2,762 | transition `.result` per entry; memory-v2 action-effect table (#64) is the harness-side version | **reject** |
| Adaptive timeout >30 s (W5) | ≤15 uses / 2,762 | hard 30 s (`min(30, …)`), 43 timeouts in 7 runs (0.1% tokens) | **reject.** Holds a stream slot longer; #76 `time` shim lets the model time-box. |
| Perception primitives (W6) | `track_objects` 3, `detect_events` 6, `detect_transform`/`detect_symmetry` 0 | `frame_diff` covers moved/rotated/appeared/disappeared/changed_color/resized | **have/reject** |
| Ruff format/lint (W7) | Inert on Kaggle; v2 found `formatted_code` echo and F821 noise harmful and removed them | none | **reject** |
| Raw grid / numpy `.grid` (W8, W9) | GB10: gain traced to the text block; `.grid` 2/17 calls in v2; never on Kaggle. numpy import costs +65-80 ms per fresh `-I -S` child (measured here: 16-20 ms bare vs 80-100 ms with numpy 2.2.6) | Deliberately hidden ("raw numeric grid is intentionally not exposed"); letters everywhere (ascii, segmentation, images) | **reject for v3.** Ints would add a second colour vocabulary. Helpers in letter space give the useful part without numpy. |
| `G`/`P`/`acts`/`level`/`L` (W10) | `L`, `acts` used 0 times; `G`/`P` tied to numpy | `current_frame`, `previous_frame`, `last_action`; #76 makes variables level-scoped (= `L`) | **reject** |
| `delta()` one-liner (W11) | Used 11 times in 47 calls; one misread; part of the GB10 change-information pattern | `frame_diff` (object-level, ambiguous matching, 2 segmentations, 25-475 ms); no cell-level one-liner; model hand-diffs ascii in ~1% of calls and builds char grids in ~37% | **take → `ARC3_SANDBOX_HELPERS` `delta()`** and the automatic form **`ARC3_SANDBOX_ECHO_DELTA`** |
| `find`/`bbox`/`count`/`show` (W11) | `find` 82 invocations on sk48; `show` 9; `bbox` 1 | none (model writes `crop` 52×, `cells_of` 13×) | **take → `where`, `bbox_of`, `color_counts`, `show_crop`** (renamed: `find`/`bbox`/`show`/`crop` collide with model functions, which retention would then reject) |
| `path()` BFS (W11) | 0 uses (GB10) | prompt pushes BFS; model writes it (5% of calls) and retention keeps it for the game | **reject** |
| Builtins/modules (W12) | class/exceptions/collections preloaded | `main` already has narrow exceptions; #76 shims `time`/`sys`/`inspect`, `globals`, `id`; pre-binding collections not supported by data (1 NameError) | **have.** `class` raises `__build_class__ not found` on `main` (checked), but 0 occurrences in two runs (§6) |
| Error text with source line + retained frames + hints (W13) | 0 errors in 47 calls (confounded); rellik13 `__missing__` messages; Self-Debug | Line numbers only; retained frames dropped; #76 adds dropped-name reasons | **take → `ARC3_SANDBOX_ERRORS`** |
| Timeout as BaseException (W14) | No timeouts in the GB10 run | Host kill (cannot be swallowed) but returns empty stdout and no line | **low → `ARC3_SANDBOX_SOFT_TIMEOUT`** |
| Stdout cap in child (W15) | — | unbounded `StringIO` in child; host middle-truncates | **take → `ARC3_SANDBOX_HARDEN`** |
| Plain-text tool result (W16) | 408 vs 657-783 chars/result (confounded with helper use) | JSON indent=2; Franzen tuned "faithful action results" | **low → `ARC3_SANDBOX_RESULT_TEXT`**, not in the first v3 arm |
| Incremental history (W17) | ~1 s/call at 1,000 actions removed | needs a persistent process; #43 (#2 fix) is the spawn-per-call answer (635 → 10.7 ms per round trip) | **defer to #43** |
| RLIMIT_AS, BLAS 1 thread (W18) | 4 GiB, no MemoryError seen | none (#33) | **take (AS + nice) → `ARC3_SANDBOX_HARDEN`**; BLAS moot without numpy |
| Transcript trace (W19) | used to prove the arm fired | none | **take**: a `[SANDBOX V3]` transcript line (never sent to the model) under any v3 flag |
| Single reference card (W20) | +326 tokens net | `_PYTHON_TOOL_DESCRIPTION` + system prompt sections | **take the style**: one line per helper in the tool description, only under the flag |
| Labeled grid / diff images (`05fdabb`) | b3 12.34 vs 17.74 | Franzen has diff images (on) | **not sandbox; have** |
| `save_module` with regression tests (Lord Han Solo) | not Wang; no measured effect | retained functions without tests | **defer** |
| Click candidates (gedouluhui) | not isolated | segmentation nodes give components | **defer** (not a sandbox feature; would be a prompt/perception arm) |
| Stale-state escape hatch (#18) | — | message says "can be caught"; flag never clears; 0 of 14.5k snippets catch it | **not v3**; its own guard arm per #18 |
| Segmentation cost (#23) | — | 25-475 ms per `frame_diff` | **not v3**; helpers avoid segmentation entirely (0.23 ms per `delta`) |

## 4. sandbox-v3 design (ranked)

All of this goes on a branch off `arm/sandbox-robust` (`arm/sandbox-v3`), one commit per flag. Every flag is default-off and read once per process, the same pattern as `_sandbox_persist_vars()`. Each child-side behavior is enabled by a key in the child's init message, which is absent when the flag is off, so the child's message is unchanged. The prompt and tools are static within a run.

Line numbers below are on `main` and move under #76, so anchor on the function names.

### Rank 1: `ARC3_SANDBOX_ERRORS`, errors that teach the API

**Why first.**
- No prompt text, so no cache or context cost on every request.
- It touches every error class #76 leaves alone: 2.7% of tokens, plus the KeyErrors and the hidden retained-function frames.
- The external pattern (Self-Debug, Tool Documentation, Wang-Ludvig `__missing__`) is the one consistent recommendation.
- Expected effect: small. Errors already cost about one response, so the gain is fewer repeat errors and fewer reasoning tokens spent locating a bug. That is probably ≤0.5-1% of tokens. It is a quality fix, not a score lever.

**What changes in the error text.**
1. **Source line in every user frame.** `line 7: x = cells[r][c]`. Frames from retained functions are kept (today they are dropped) as `in route() (retained) line 3: <source>`, with the source taken from `_kept_pool`. At most the last 2 frames are shown; each line is clipped at 160 chars.
2. **Teaching KeyErrors on harness dicts.** Provided dicts become a `dict` subclass, `_HarnessDict`, whose `__missing__` raises a `KeyError` subclass with a readable `__str__`. Examples:
   - `KeyError: 'steps' (last_animation_timeline is empty: the last executed action did not animate)`
   - `KeyError: 'stop_reason' (this action result has: executed, action_num, level, score, state, board_changed, gameplay_changed, ...)`, with the list capped at 12 keys.

   Which dicts are wrapped:
   - the `action()` return value and `last_action_call_result`;
   - `history[i].result` / `transition.result`;
   - the timeline dict returned through `_LazyAnimation` (including its empty `{}`);
   - `frame_diff()` output and its entries;
   - segmentation dicts and nodes.

   Properties that must not change: `isinstance(x, dict)`, `repr`, `==`, `json.dumps`, `_json_safe`, iteration order. Values are not copied.
3. **One short `hint:` line for the measured recurrent cases.**

   | Error | Hint |
   |---|---|
   | NameError | difflib close match among namespace names; #76's dropped-reason hint takes precedence |
   | `min()/max() arg is an empty sequence`, `list index out of range` on an empty list | "the sequence was empty: the search/filter found nothing; check it before indexing" |
   | `unhashable type: 'list'/'set'/'dict'` | "use a tuple/frozenset as the key" |
   | AttributeError on a view | lists that view's real attributes |
   | `'FrameView' object is not subscriptable` | "use .ascii rows or .segmentation" |
   | `too many values to unpack` when the failing line iterates a dict | "iterate .items()" (only when the line text contains `for` and the value is a dict) |
4. **SyntaxError text.** `Python syntax error at line 10: <source line>\n<msg>`, instead of the current text without the line.

**Touch points.**
- `python_tool_sandbox.py`:
  - `_sanitize_exception` (L268-275): add a source-line lookup over `<python_tool>` (`_code_text`) and `<retained>` (`_kept_pool`), plus a hint function.
  - `action()` (L489-568): wrap `action_result` before returning; `_refresh_state` wraps `last_action_call_result`, history and transition results.
  - `_LazyAnimation.__getitem__/get` (L599-635); `frame_diff` (L365-382); the `FrameView.segmentation` property (L169-173).
  - A new `_HarnessDict` class in the bootstrap.
  - #76's `_plain_data_problem`/`_PLAIN_CONTAINERS` must treat `_HarnessDict` as `dict`, or a carried copy of an action result is wrongly dropped as "type". Its repr is a plain dict literal, so `_plain_from_node` restores a plain dict.
- `tool_agent.py` `_run_python_tool` (L5717-5723): the SyntaxError branch gains the line text under the flag.
- Host: pass `"errors": True` in the init message (`run_sandboxed_python`, L850-864).

**Tool-description text.** None.

**Safety.**
- Hint and source lookup run inside `try/except BaseException`; on any failure they fall back to today's text (Wang v2's `_v2_describe` rule). The richer message must never be what kills the child or replaces the original exception class.
- Source lines are the model's own code, so no host paths or internals leak.
- The `_HarnessDict` subclass adds no attribute surface beyond `dict`. The sandbox's real boundary is process isolation (`-I -S`, temp cwd, rlimits, kill), not Python-level restriction, and `().__class__.__base__.__subclasses__()` is already reachable. Exposure does not change.
- The KeyError subclass still matches `except KeyError` and `except LookupError`.

**Cost.**
- About +1-2 lines on the ~8% of calls that error (≤ ~60 tokens each).
- CPU is microseconds (wrapping is O(keys) on dicts the child already built).
- Bootstrap source +~150 lines, about +1-2 ms compile per call against the ~40 ms spawn.

### Rank 2: `ARC3_SANDBOX_ECHO_DELTA`, exact change line on every action echo

**Why.**
- The best-supported idea from the owner's GB10 night: exact, harness-computed change text gave ar25 L2 in 5 of 5 arm runs vs 2 of 8 controls, across three arms. In every case the transcripts show the key fact (the mirror stripe moving +3) arriving through harness text while the model's own python filtered it out.
- Lord Han Solo also feeds "changed regions and per-step changes in a batch" back, though unmeasured.
- On Franzen today the echo says only `-> effect`, so the specifics come only if the model calls `frame_diff()` (6-11% of calls) or diffs by hand.
- Unlike `ARC3_AUTO_FRAME_DIFF` (in the turn opener, object-level, never run on Franzen, matcher noise per research-sweep), this goes into the stdout of the very call that acted. It needs no segmentation and makes no object-identity claim.

**What.** In the child's `action()`, after the reply:
- Compute `delta(before, after)` over the interior. `before` is the board this call started from; for batches, use the net delta of the batch, like the frame-diff v2 "net diff over the whole batch".
- Border cells (`ARC3_NOOP_GUARD_BORDER`, 4, passed in the init message) are folded into one `| border: N cells` clause, so the step bar never dominates.
- Append the result to the echo line, for example `[action] RIGHT -> effect: 30 cells [15,18,23,29] B>b x15, b>B x15 | B moved (+0,+3) | border: 2 cells`.
- Notation matches the timeline's `old>new` letters. "Moved" is claimed only for a letter whose changed-box cells shift rigidly by one offset with equal counts (Wang's rule, measured correct on ar25). Letters covering >25% of the board are treated as background and never "move".
- No delta is shown when the action is NO-OP, refused, `level_completed` (new board: `| new level board`), or `game_over` (the fatal frame is already shown by Franzen's game-over image and diff).

**Caps.**
- 140 chars per echo.
- After 8 echoes in one snippet, only the count and box are shown, and a total of 1,200 echo-delta chars per snippet. This protects the 3,072-token budget for loop-heavy snippets. Without it, a 30-step loop could add ~1.1k tokens and push the snippet's own prints into the middle cut.

**Touch points.**
- `python_tool_sandbox.py` `action()` (L541-545, where `_echo` is printed): grab `runtime_globals["current_frame"]` before `_send`, compute after `_refresh_state`, and append.
- The new stdlib-only module `inference/utils/sandbox_helpers.py` (shared with Rank 3; injected like `frame_diff.py` via a `__SANDBOX_HELPERS_SOURCE__` placeholder; the host can import it for tests). No solver change: `_build_action_echo` stays the same.

**Tool-description text.** None; the line describes itself. Optionally, one clause in `STRUCTURED_RUNTIME_STATE_ADDENDUM`, A/B'd separately only if transcripts show the model misreading it:

> "Each `[action]` line ends with the exact cells that changed inside the board area, as `old>new` letter counts and letters that moved."

**Safety.** The whole computation sits in `try/except Exception`. On failure the plain echo is printed: a diff must never fail an action (Wang frame-diff `86a4910` rule). Pure functions over the two grids; no I/O.

**Cost.**
- About 0.23 ms per delta (pure Python, 64×64, measured here).
- About 35-45 tokens per acting call. Franzen-era runs have ~950 action calls per run, so ~40k tool-output tokens per run, appended once and cached afterwards.

**Risk.** It adds text to every acting turn. rellik13 measured "adding text to every prompt" as harmful (6.42 → 4.16), and Franzen found auto-inserted timelines gave no clear gain. This is the arm most likely to move score in either direction. Hence the separate flag and the ablation in §8.

### Rank 3: `ARC3_SANDBOX_HELPERS`, cell-level helpers in letter space

**Why.**
- 37-39% of Franzen-era calls rebuild a char grid from `.ascii`. `crop` (52 definitions) and `cells_of` (13) are the model's own versions.
- Wang v2's `find` was the most-used helper (82 invocations on sk48), and its probe-measure loops shortened requests (666 vs 923-1,304 completion tokens).
- Retention already amortises helpers the model writes itself, so the gain is smaller than on the Wang harness. research-sweep gave a similar item impact 2/5, low confidence.
- `delta()` is the model-invoked form of Rank 2, for arbitrary history pairs.

**What.** All in letter space (the `.ascii` letters, so they agree with segmentation, images and `frame_diff`). A frame argument accepts a view, an ascii string, or a list of row strings, and defaults to `current_frame`. Boxes are inclusive `[r0, c0, r1, c1]` (segmentation and MOUSE order).
- `delta(before=None, after=None)` → one line (defaults `previous_frame` → `current_frame`). Same text as Rank 2, without the caps.
- `where(letters, frame=None)` → sorted list of `(row, col)`. `letters` is a str of one or more letters.
- `bbox_of(cells)` → `[r0, c0, r1, c1]` or `None`. Accepts a list of `(row, col)` or a segmentation node.
- `color_counts(frame=None)` → `{letter: cells}`, most common first.
- `show_crop(frame=None, r0, c0, r1, c1)` prints rows with row and column rulers; clipped to the board, at most 40×64 cells.

**Tool-description text.** Appended to `_PYTHON_TOOL_DESCRIPTION` under the flag, about 130 tokens:

> "Helpers (letters as in `.ascii`; a frame defaults to `current_frame`; boxes are inclusive `[r0, c0, r1, c1]`): `delta(before=None, after=None)` one line on what changed (default `previous_frame` to `current_frame`): cell count, box, `old>new` letter counts, letters that moved rigidly; `where(letters, frame=None)` list of `(row, col)` cells with those letters; `bbox_of(cells)`; `color_counts(frame=None)` `{letter: cells}`; `show_crop(frame, r0, c0, r1, c1)` prints that region with row/col rulers."

The "Only tool:" line (`tool_agent.py` L5105) also lists the five names.

**Touch points.**
- `sandbox_helpers.py` (shared).
- Bootstrap: bind the helpers into `runtime_globals` before `_available_names` is computed (L651), only when the init message has `"helpers": True`. That makes them reserved names only under the flag, so flags-off retention behaviour is unchanged. #76 never carries provided names.
- `tool_agent.py`: the tool schema (L5183), the `_PYTHON_TOOL_DESCRIPTION` variant, and the init-message key.

**Safety.** Pure functions; inputs validated (shape, letters); errors are raised as `ValueError` with the expected usage, which shows as the caller's line under Rank 1. `show_crop` is clipped so it cannot print a whole board twice.

**Cost.** About 130 cached prompt tokens per request; about +2 ms bootstrap compile per call.

**Name choice.** `find`, `bbox`, `show`, `crop`, `cells` and `grid` are all model-defined names. A provided global with that name would make retention reject the model's own function ("name conflicts with a provided global"). The chosen names had 0 hits in 4,492 calls.

### Rank 4: `ARC3_SANDBOX_STORE`, a game-scoped `store` dict

**Why.**
- The Wang-era model used `remember`/`store` in 5-7% of Kaggle calls.
- Franzen-era models try `open('/tmp/…')` (32 in two runs), `globals` and `pickle` to keep state.
- #76 clears variables at a level change, which is right for coordinates but loses learned mechanics (action → effect tables, a colour key). An explicit, model-chosen game-scoped dict keeps cross-level knowledge without the stale-variable risk #76 is careful about.
- Complements ledger-on-trim and memory-v2, which are text memory: this is executable memory.

**What.**
- `store` is a provided dict, restored at the start of each call and snapshotted at the end with #76's machinery: `_plain_data_problem`, the literal reader, caps of 100k chars per key and 200k total.
- Lifetime: survives level changes, deaths and timeouts (keep the last good snapshot), and is cleared on a new game.
- If the model rebinds `store` to a non-dict, the previous store is kept and a one-line notice is shown.
- Non-plain values are dropped per key, with the #76-style reason given on the next `KeyError` (Rank 1 machinery).

**Tool-description text** (~45 tokens):

> "`store` is a dict kept for the whole game, across levels and deaths; put plain data in it (numbers, strings, lists, dicts) to keep it. Other top-level variables are cleared when the level changes."

**Touch points.**
- Bootstrap: `runtime_globals["store"]`, restore and snapshot next to `_persist_restore`/`_persist_snapshot`.
- `tool_agent.py`: `self._sandbox_store` beside `self._sandbox_carry`, cleared where #76 clears carry on a new game (`_ensure_session`, ~L3610), and NOT cleared on a level change.

**Requirement.** It needs `ARC3_SANDBOX_PERSIST_VARS`; if it is off, the store flag is ignored and a warning is logged.

**Safety.** Same as #76: literal-only, never evaluated, size-capped, no code objects.

**Cost.** About 45 prompt tokens per request; one extra `repr`/parse per call (≤ a few ms at the cap).

### Rank 5: `ARC3_SANDBOX_HARDEN`, child resource hardening (rides along; no score claim)

**What.**
- `RLIMIT_AS` = 3 GiB, set in `_set_limits` after bootstrap imports. A runaway BFS (1.36 GB in 10 s, #33) becomes a `MemoryError` error turn, not host OOM risk with 10 concurrent children.
- `os.nice(10)` called inside the child. Not `preexec_fn`: Python documents that as unsafe in a multithreaded host, and the harness runs one thread per game.
- Stdout capped inside the child (head 200k + tail 50k chars, line-bounded, with an omission marker), so a print loop cannot build an unbounded `StringIO` and JSON line.

**Touch points.** `_set_limits` (L285-299), `main()` (stdout object, L354), and the init-message key.

**Safety.** It only tightens limits. Check that `RLIMIT_AS` leaves headroom for `segment_layer` on 64×64 boards and the snapshot (both tiny).

**Cost.** None measurable. Watch serve.log tok/s for the claimed SGLang-contention effect, which research-sweep found no data for.

### Rank 6 (low): `ARC3_SANDBOX_SOFT_TIMEOUT`

**What.**
- The child arms `signal.setitimer` at timeout − 2 s.
- The handler raises `_SoftTimeout(BaseException)`, so `except Exception` cannot swallow it. If it fires during an `action()`/animation round trip, it is deferred until the reply has been read, so the protocol never desyncs.
- The reply carries partial stdout, action results and `Timed out after 28s at line N: <source>`.
- #76's rule holds: no variable snapshot on a timeout. The host hard kill at 30 s stays as the backstop for C-level hangs (catastrophic regex, huge allocations).

**Evidence.** 43 timeouts in 7 runs (0.1% of tokens); research-sweep rated this low value. Keep it out of the first arm.

### Rank 7 (low, not in the first arm): `ARC3_SANDBOX_RESULT_TEXT`

Plain-text tool result (Wang W16): stdout unescaped first, then `result:`, `error:`, `function_retention:` and `truncated` lines.

- **Evidence:** confounded with helper use on one GB10 game. Escaping costs only ~15-25 tokens per result.
- **Risk:** changes a format Franzen tuned and that every arm's tests compare against.
- **Touch points:** `_render_tool_payload` (L5390-5408) and `_trim_tool_text`.
- Run it only as a single-flag arm after v3 is judged.

### Not in v3

| Item | Why not |
|---|---|
| Persistent interpreter | §3 |
| numpy | §3 |
| Predictor / `mechanics` / `track_objects` / `detect_*` / `path` | 0-8 uses on Kaggle |
| Adaptive timeout | Holds slots; ≤0.5% use |
| Ruff | Inert and harmful echo |
| `L` / `acts` | 0 uses |
| `save_module` with regression tests | Unmeasured anywhere; the model class ignores verification scaffolds (Makarov). Revisit only if v3 shows retained helpers being re-written after level changes. |
| Click candidates | A perception/prompt arm, not sandbox |
| #18 escape hatch | A guard arm; 0 snippets catch the error |
| #23 segmentation memo | Helpers avoid segmentation; <0.5% gain |

### Interactions and conflicts

- **Prompt length and prefix cache.**
  - v3 adds ≤ ~175 tokens (HELPERS + STORE) to the static tools/system prompt; ERRORS, ECHO_DELTA and HARDEN add none.
  - All text is fixed per run, so every request still extends the previous one. ECHO_DELTA and ERRORS change only new tool results.
- **3,072-token tool output.** The echo-delta caps (140 chars, 8 full echoes, 1,200 chars per snippet) and hint caps (1 line) keep v3 additions under ~10% of the budget in the worst loop. The error text sits at the end, where middle truncation keeps it (tail = 1/3 of the budget).
- **30 s deadline.**
  - v3 adds ~2-4 ms per call (bootstrap) and ~0.2 ms per action.
  - The deadline also counts host env time, and history serialization grows with game length: #2 measured 635 ms per round trip at 310 frames (fixed by #43, not on `main`), and rellik13 measured 36/150 `action()` timeouts late in games.
  - Run v3 with #43 if #43 is in the base by then, and log tool wall time (§8).
- **Persistent functions.** Helper names are reserved only under the flag and chosen to collide with nothing observed. `_HarnessDict` must be carried as a `dict` by #76 (Rank 1 touch point). Errors inside retained functions become visible (Rank 1), which should help repair retained helpers.
- **#51.** #76 already applies the timeout-keeps-functions rule under its flag; v3 changes nothing there.
- **memory-xml / level-path.** They add python globals `memory_items`, `memory_log`, `level_path`, `level_path_len`; no name overlap. If stacked, the "Only tool:" line and tool description each get both sets of names (same conflict pattern as mem-v3).
- **Guards.** ECHO_DELTA prints after the guard decisions and never changes them. A NO-OP echo is unchanged, so the batch/stale guard text the model knows is preserved.

## 5. Safety model (model-written code)

**Threat model.** Buggy or runaway code from a cooperative model, not an adversary. The real boundary stays as it is:
- `python -I -S` child per call, temp cwd, restricted builtins and import allowlist;
- `RLIMIT_CPU`/`FSIZE`/`NOFILE`, host hard kill at 30 s;
- host errors scrubbed to "Sandbox process exited unexpectedly".

**v3 rules.**
1. No new importable modules. The helpers are bootstrap source, and the shims are #76's.
2. Nothing is evaluated from model data. `store` and carried variables go through #76's literal reader.
3. Every harness-side addition (hints, source lookup, echo delta, store snapshot) is wrapped. On failure it falls back to today's behavior and never changes the snippet's original error class or its executed actions.
4. Messages contain only the model's own code, harness key names and counts; never host paths.
5. The limits only get stricter (AS cap, nice, stdout cap).
6. Retention conflict rules are unchanged, apart from the flag-gated reserved names.

## 6. Open questions (check during implementation)

1. `class A:` fails on `main`. Checked against the real sandbox: `NameError: __build_class__ not found`.
   - The error appears 0 times in the ledger-r2 and control transcripts, so the model has learned not to write classes. It is not worth a flag.
   - If it is ever wanted, adding `__build_class__` + `object`/`super`/`property` is a one-line shim candidate for #76. Security impact: none beyond today's object access.
2. Is `_HarnessDict` wrapping of segmentation nodes worth it? KeyError `'color'` appeared only twice. Start with action results, the timeline and `frame_diff`; add nodes only if the tests show no cost.
3. Echo delta when the board shape changes (some games resize): print `| board resized RxC -> R'xC'` and nothing else.

## 7. Test plan

All tests run against the **real child process**, one suite at a time under `nice` (laptop rule).

**1. Unit tests, `tests/test_sandbox_v3.py`.**
- Flags off: the child init message and result keys equal `main`'s and `arm/sandbox-robust`'s for the same snippet. Each flag on adds exactly its init key.
- ERRORS:
  - the source line is shown;
  - a retained-function frame shows `(retained)` and its source;
  - NameError close match, and #76's hint wins when both apply;
  - a KeyError on the action result lists keys; the empty-timeline KeyError explains itself;
  - `except KeyError` still catches it;
  - `isinstance(dict)`, `repr`, `==`, `json.dumps` and `_json_safe` are unchanged;
  - #76 carries a wrapped dict as a plain dict;
  - fault injection: a hint function that raises gives exactly today's message;
  - SyntaxError text includes the line.
- ECHO_DELTA:
  - effect, NO-OP, refused, level change and game over cases;
  - border folding with a bar that ticks every action;
  - a rigid move found, and no false move for identical copies that vanish in one place and appear in another;
  - batch net delta;
  - caps (140 chars, 8 echoes, 1,200 chars);
  - an exception inside delta still prints the plain echo, and the action still counts.
- HELPERS:
  - property tests against brute force on 500 random 64×64 boards (`where`, `bbox_of`, `color_counts`, `delta` counts and transitions);
  - defaults;
  - `show_crop` clipping;
  - a retained model function named `where` is rejected only with the flag on;
  - helpers are never carried by #76.
- STORE:
  - survives a level change (inside a call and between calls), a death and a timeout;
  - cleared on a new game;
  - caps;
  - a non-plain value is dropped with a reason;
  - rebinding to a non-dict keeps the old store;
  - the flag is ignored without `PERSIST_VARS`.
- HARDEN:
  - `bytearray(4 * 1024**3)` gives a clean `MemoryError` error turn, and the next call works;
  - the child's nice value is 10;
  - a 10 MB print is capped with a marker.
- SOFT_TIMEOUT:
  - `try: while True: pass` inside `except Exception` is reported at its line with partial stdout;
  - an alarm during an action round trip is deferred;
  - a C-level hang (catastrophic `re`) still hits the host kill;
  - no variable snapshot on a timeout.

**2. Loop test through `ToolAgent.analyze()`.** File: `tests/sandbox_v3_loop_sim.py` plus `tests/test_sandbox_v3_loop.py`, extending #76's `tests/sandbox_loop_sim.py`.
- Setup: the real request loop, `_run_python_tool`, `run_sandboxed_python`, retention, persistence and shims, under the Kaggle notebook's own `setup_env` (persistent functions on, scope game). Only two things are faked: the HTTP post (a fake SGLang rendering through `tests/fixtures/qwen38_flash_next_chat_template.jinja`, as in `arm/reasoning-scaffold`'s `tests/scaffold_loop_sim.py`) and the two-level toy game as `step_env`.
- The script exercises:
  - an acting call whose echo carries a delta, which must appear in the NEXT request's tool message;
  - a KeyError on `last_action_call_result`;
  - an error inside a retained helper;
  - `store` written on level 1 and read on level 2;
  - a helper call;
  - a timeout.
- Asserts:
  - same request count in every config;
  - system prompt and tools static within a run;
  - every prompt extends the previous one (prefix);
  - **flags-off requests and tool results are byte-identical to `main`** (`SANDBOX_SIM_BASELINE_ROOT` = a `main` checkout), and #76-flags-only identical to `arm/sandbox-robust`;
  - the tool-description token delta, measured through the chat template, equals the budget above (±10 tokens).

**3. Mutation checks** (each must fail at least one test, then be restored):
- the hint wrapper's `try/except` removed (fault-injection test);
- the KeyError subclass replaced by a plain `KeyError` (message test);
- echo delta computed after → after (it must not print "no change");
- helpers bound with the flag off (byte-identity);
- `store` cleared on a level change;
- `RLIMIT_AS` not applied (memory test);
- soft-timeout alarm not deferred (round-trip test);
- `_HarnessDict` not accepted by #76 carry (carry test).

**4. Replay check (offline, no GPU).** Run `delta()` over the recorded boards in two runs' `requests.jsonl`/transcripts and report:
- the echo length distribution;
- how often a rigid move is claimed;
- spot checks of 30 claims against the boards (the frame-diff matcher audit in research-sweep found 52% of "moved" entries ≤2 px and fake moves in tile games; the cell-level rule should avoid both, and this checks it).

**5. Docs in the same PR.** `CONFIGURATION.md` (six flags), the `README.md` fork overview, `kaggle/README.md` arm list, and `kaggle/arms.json`.

## 8. Eval plan (judged at matched wall-clock)

Owner's rules: compare at wall-clock, not runtime clock; an alpha counts only if repeated runs score consistently higher; gaps under ~3 offline points are noise.

**Arms** (`kaggle/arms.json`), all on the competition config with the W4A16 Qwen3.8-Flash-Next + MTP drafter:

| Arm | Flags |
|---|---|
| `control` | `main` |
| `sandbox-robust` (#76) | `PERSIST_VARS`, `SHIMS` |
| `sandbox-v3` | #76 + `ERRORS` + `ECHO_DELTA` + `HELPERS` + `STORE` + `HARDEN` |
| `sandbox-v3-quiet` | #76 + `ERRORS` + `HELPERS` + `STORE` + `HARDEN` (no echo delta) |

`sandbox-v3-quiet` isolates the one feature that adds text to every acting turn, the likeliest source of a sign flip.

**Stage 0: screen (not a verdict).** `panel-sandbox-v3`: balanced 10-game panel, 20 min, `ARC3_MAX_ACTIVE_STREAMS=4`, paired with `panel-control`. Check:
- no crash, `[SANDBOX V3]` traces present;
- error mix shifted;
- python tool wall time p50/p95 not worse;
- prefix-cache hit within ±2 points of control;
- serve.log tok/s within noise.

**Stage 1.** Full 25 games × 121 min. `sandbox-v3` ×2 and `control` ×2, same day where possible; `sandbox-robust` ×2 when #76's own runs happen (share them).

**Stage 2** (only if stage 1 is positive). `sandbox-v3-quiet` ×2, to decide whether the echo delta earns its tokens.

**Primary metric.** Offline mean over the 25 games at the same 121-min wall budget, compared with `make significance BASELINE_SCORE=<control scores> CANDIDATE_SCORE=<arm scores>` (paired per-game). Keep the arm only if both repeats beat control (owner's rule) and the pooled paired test agrees.

**Wall-clock view.** Score-vs-wall-minute curves from per-level completion timestamps (levels cleared and weighted score at 30/60/90/121 min, per arm, mean over repeats). This shows whether v3 buys levels earlier for the same wall time, which is the throughput-equivalent claim behind fewer error turns.

**Secondary metrics** (per run, from transcripts and request logs; via `make traces`/`make eval` and `analysis/failure_taxonomy.py`):
- python errors per 100 calls by class: expect NameError/ImportError/AttributeError ↓ from #76; KeyError/TypeError/Index/Value repeat-pairs ↓ from ERRORS;
- tokens on error outcomes (failure taxonomy row 9);
- requests and acting turns per game-hour (Wang's metric);
- generated tokens per cleared level;
- share of calls using each helper, `store`, and the echo delta. The echo is always present, so for it measure how often the next reasoning cites a moved letter or box;
- tool wall time p50/p95 per call (a new transcript-only `tool_wall_ms` trace under any v3 flag);
- prefix-cache hit rate and tok/s;
- timeouts and MemoryErrors.

**Stop rules.**
- Any flags-off byte-identity failure blocks the run.
- A stage-0 tool-time p95 regression over 20% blocks stage 1.
- If `sandbox-v3` is negative in both repeats while `sandbox-v3-quiet` is not, drop ECHO_DELTA and log it under Didn't work.

**Logging.** Record each run's result in `arms.md` under `sandbox-v3` (In progress → Worked / Didn't work), with score, compute (tokens, tok/s, tool wall share) and the significance output.

## Appendix: measurements made for this doc (2026-10-08)

**Wang-era Kaggle usage and errors.** Scripts in the session scratchpad: `wang_usage.py` and `wang_errs2.py` over `arc-agi-3-v3/kaggle-outputs/{b1-thinking-real,final-c1-ledgerfix}/transcripts`.
- 2,762 calls.
- Errors: b1 51 (TypeError 17, Index 7, Name 7, Key 5, Attribute 5, Value 5, Import 3, Runtime 1, timeout 1); final-c1 31 (Type 9, Name 6, Attribute 5, Key 4, Value 3, Index 3, Assertion 1).

**Franzen-era usage.** `fz_usage.py` over `kaggle-outputs/{fz-ledger-on-trim-r2,fz-arms-v4-control}/transcripts` (§2.3 table).
- Import refusals: `time` 5/7, `sys` 4/5, `builtins` 4, `inspect` 2/2, `pickle` 3.
- `open('/tmp…` 32 in the two runs.
- Traceback frames inside snippet functions: 35/169 and 39/202.

**Error messages over five Franzen-era runs** (ledger-r2, control, ledger-on-trim, ucb, streams14):
- NameError 266; KeyError 85 (`'steps'` 12, `'action'` 4, `'frames'` 3, `'stop_reason'` 2).
- `list index out of range` 58, `min() iterable argument is empty` 33, `too many values to unpack (expected 2)` 26, `string index out of range` 26.
- `HistoryEntryView.step` 17, `_LazyAnimation.items` 14.

**Model-defined names** (collision check): `crop` 52, `bfs` 42, `show` 15, `cells_of` 13, `find` 6, `bbox` 5; 0 for `delta`, `where`, `store`, `bbox_of`, `color_counts`, `show_crop`.

**Timings** (local Mac, under `nice`):
- fresh `-I -S` child 16-20 ms; with `import numpy` (2.2.6) 80-100 ms;
- pure-Python `delta` on a 64×64 board 0.23 ms.
