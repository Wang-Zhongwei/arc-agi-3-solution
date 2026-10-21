# Kaggle arms

Every arm runs as its own private kernel, `mellamoa/v4-<arm>`. It is built from
`template.ipynb` (Franzen's Milestone 2 notebook). The notebook's harness patch is
replaced with `git diff bundle-base..HEAD` over the harness's runtime paths for the
checked-out commit, so a run always executes exactly the committed code.

- **Runtime paths** (`RUNTIME_PATHS` in `build_notebook.py`): `ARC3-Inference/inference`
  without `inference/tools`, plus `configs/`, `pyproject.toml` and `uv.lock`. The patch,
  the code hash behind the batch rule and the uncommitted-changes check all cover exactly
  these.
- Docs (`README.md`, `CONFIGURATION.md`), `Makefile`, `tests/`, `viewer/`, `scripts/`,
  `bench/` and the local analysis tools in `inference/tools` never run on Kaggle. A commit
  that touches only them leaves the notebook byte-identical, so it neither breaks an open
  batch nor makes a replica differ from run 1. Uncommitted edits there don't block a build.
- Records from before this rule (2026-10-08) hold the hash of the whole `ARC3-Inference`
  diff. The builder recomputes their runtime hash from their recorded commit when that
  commit is in the clone, and otherwise compares with the old hash.

    python kaggle/build_notebook.py --arm panel-control              # build only, into kaggle/out/
    python kaggle/build_notebook.py --arm panel-control --push --new-batch
    python kaggle/build_notebook.py --arm panel-control --push --replica 2   # repeat a run
    kaggle/push_queue.sh panel-control panel-<arm> panel-<arm>@2     # staggered pushes

Setup on a fresh clone: `bundle-base` is a tag on the commit holding Tufa's Kaggle source
bundle as shipped (`git fetch origin tag bundle-base`).

An arm in `arms.json` can set:
- `env` (harness), `serve` (launcher `CFG`), `serve_env` (SGLang process env), `demo_games` and
  `demo_minutes`.
- `datasets`: extra Kaggle datasets (`"owner/slug"`) appended to the kernel's `dataset_sources`,
  e.g. a published JIT cache (`serving/CUSTOM.md`, "Cold start").
- `runtime_patches`: files of `serving/bench_patches/` to apply to the installed SGLang, after
  the shared `serving/runtime_patches/`, e.g. `["m64.py", "nosplitk.py"]`. This lets a bench
  config's patches reach a game run without becoming the default for every arm. The builder
  refuses a missing file, a name a shared patch already uses, and a name that would sort
  before a shared patch (the launcher applies them in name order).
- `gate`: what must hold before the arm runs, such as a bench result. The builder builds a gated
  arm freely but pushes it only with `--gate-passed` (in `push_queue.sh`,
  `GATE_PASSED="<arm> ..."`), which confirms the gate holds.

Tests for the builder and the serving helpers run without a GPU, from `ARC3-Inference`:
`python -m pytest -q ../kaggle/tests`.

## Rules the builder enforces
`runs.json` is committed per branch, so the builder reads it from every local branch and every
fetched remote-tracking branch as well as the checked-out file. Only `mellamoa/v4-*` records
count (an older project's remote can carry its own `runs.json`).
- **No duplicates.** It refuses a notebook identical to a recorded run, under any name and on
  any branch.
  - Arms with the same settings count as one run: `stack-v2` and `ledger-on-trim` are the
    same notebook, so pushing one blocks the other.
  - A deliberate repeat, for a second paired sample, passes `--replica N` (N ≥ 2; in
    `push_queue.sh`, `<arm>@N`). That adds the inert harness env var `TAAF_REPLICA=N`, so the
    notebook differs, and `runs.json` records `"replica": N`. Run 1 builds exactly as without
    the option. It works for `--bench` too.
- **One code version per batch.** It refuses to build at different runtime code from this
  branch's open batch, because the arms would then not be comparable with the batch's
  control. A branch with no open batch of its own is also refused while an open batch on
  another branch ran different code: the error names that batch's arms, commits and branches.
  Start a new batch with `--new-batch`, running its control first.
- **Gated arms wait for their gate.** It refuses `--push` for an arm with a `gate` unless
  `--gate-passed` is given.
- **Record every push.** `--push` records the version, commit and fingerprint in `runs.json`.
  Commit that file after each push (`push_queue.sh` does), and push the branch so other
  clones see the record after a `git fetch`. A record that exists only in an uncommitted file
  is invisible to builds on other branches.
- **Submissions are not experiments.** `--submission` pushes a new version of
  `mellamoa/v4-submission`. It skips both checks above, since submitting the notebook an arm
  already ran is the point, and says which recorded run had the same notebook. Its record is
  marked `submission` and never opens or closes a batch, and it can't be combined with
  `--new-batch`.

Arm branches stack, and every arm is behind an env flag that is off by default. One commit can
therefore run the control and several arms side by side.

## Panels
- **`all`**: the 25 offline games for 121 min. That is Franzen's 532 min × 25/110, the
  per-game compute of the competition rerun.
- **`balanced`**: 10 games.
  - The 5 hardest by fraction of levels cleared: sk48, bp35, tn36, sp80, lf52.
  - 5 easy games that still vary between runs: ft09, vc33, tu93, cd82, r11l.
- **The `panel-*` arms** run `balanced` for 20 min with `ARC3_MAX_ACTIVE_STREAMS=4`.
  - With 10 games on 10 slots nothing ever waits at the gate, so a scheduling change would
    be inert.
  - 4 slots give 2.5 games per slot, the same ratio as the 25-game offline run (the
    competition is 11 per slot).

## Score timeline log
Every notebook the builder makes has a cell **8b** right after the system monitor. It starts a
daemon thread that logs the score as the run goes (`kaggle/score_logger.py`, pasted into the cell).
- **Every 60 s** it appends one JSON line per game to `/kaggle/working/score_timeline.jsonl`, plus
  one `"kind": "run"` line.
  - Clocks, in seconds: `t_notebook_s` (since the notebook started), `t_ready_s` (since the model
    server answered `/health`) and `t_bench_s` (since the benchmark started), plus `wall`.
  - Per game: `game`, `pass`, `levels`, `n_levels`, `actions`, `score`, `tokens`, `state`
    (`playing`, `finished` or `crashed`) and `run_state` (the framework state: won, gave_up, ...).
  - The run line: `mean_score`, total `levels`, `actions` and `tokens`, `games_finished` and
    `n_games`.
  - `score` is the official formula: `min(115, (baseline/actions)^2 * 100)` per cleared level,
    level *i* weighted by *i*, capped as the framework does.
- **Every 5 min** it prints one line to the log:
  `[score] t=37m nb=48m mean=9.75 lv=40 act=1605 tok=1.40M fin=0/25 | ar25 4L 27.8 | bp35 1L 2.2 | ...`
  - `t` is minutes since the benchmark started and `nb` minutes since the notebook started.
  - Each game shows its levels and score. `*` marks a finished game and `!` a crashed one.
- **It only reads.** It reads `bm.game_runs` (the framework's live `GameRun` objects) and the
  `SERVER_READY_TIME` that the serving cell's wait loop now stamps. It never calls the solver, the
  engine or the server, so no game decision, prompt or request changes.
  - Every read is wrapped. A game that cannot be read logs an `error` line, and a failed tick prints
    at most 3 messages.
  - A tick takes well under a millisecond for 26 games and is tested at under 50 ms for 110.
- **It is off in a competition rerun**, which hides the baselines and whose output is never seen.
- Download it with the other outputs and read it next to `analysis/score_timeline.py`, which
  rebuilds the same curves offline from `benchmark.json` (see `analysis/README.md`).
- Adding the cell changed every newly built notebook's fingerprint. Recorded runs keep theirs.

Tests: `kaggle/tests/test_score_logger.py`. It runs the logger against real framework `GameRun`
objects and stubs, including broken ones, and executes the built cell against a stub benchmark.
Run it with `PYTHONPATH=tufa-arc-agi-framework/src ARC3-Inference/.venv/bin/python -m pytest kaggle/tests -q`.

## Memory arms
Their env is in `arms.json`; what each flag shows the model is in `ARC3-Inference/CONFIGURATION.md` §10.
- `stack-v2`: the control for memory arms (ledger-on-trim + the #4 and #5 fixes).
- `memory-v2-trim`, `memory-v2-delta`, `memory-v2-both`: all memory-v2 facts, in each placement.
  `solved-levels`, `action-effects`, `seen-state`: one component each. `panel-memory-v2-both`: a
  20-minute balanced-panel smoke run of `memory-v2-both`.
- `level-path`: `memory-v2-both` + `ARC3_MEM_PATH` (the shortest route on record after every action).
  Its paired control is `memory-v2-both` in the same batch.
- `mem-solved-10x`, `mem-v2-10x`, `level-path-10x`: the same memory stacks (solved lines only,
  `memory-v2-both`, `memory-v2-both` + `ARC3_MEM_PATH`) on `user-combo-10x`, each with
  `ARC3_MEM_REPLY_WINS=1`; 121 min, all games. Their paired control is `user-combo-10x` (and
  `mem-v2-10x` for `level-path-10x`). Build with `--new-batch`; judge only on full runs and
  repeats (`research/archive/reports/memory-main.md`). Results: mem-v2-10x 43.81, level-path-10x 48.31
  against user-combo-10x 56.94/52.70/49.09; the audit of both runs is in the report.
- `mem-v2-fixed-10x`: `mem-v2-10x` as the audit corrected it: `ARC3_MEM_DEBRIEF=1` (the
  model's own recap on each solved line instead of `how`; model text first on the card and cut
  last), `ARC3_MEM_KEEP_EDGE=1` (solved paths keep edge-only actions), the card only at trims
  (`ARC3_MEMORY_PLACEMENT=trim`) and no route lines. Paired control: `user-combo-10x`. Build
  with `--new-batch`.
- `mem-layout-10x`: `mem-v2-fixed-10x` plus the user's memory layout (2026-10-10):
  `ARC3_MEM_EVERY_CALL=1` (a `<previous_levels>` message right after the system prompt, one
  entry per cleared level, redrawn only at a level commit or a trim; a `<this_level>` block at
  the end of every turn message), `ARC3_MEM_LEGIBLE=1` (every memory line in plain words) and
  `ARC3_LEVEL_COMPACT=1` (a commit drops the cleared level's turns except the last 2, in the
  same re-prefill). Replayed on user-combo-10x v1's logged requests: +3.2% prefill, 8.3 prefix
  breaks per game against 5.7. Paired control: `user-combo-10x`. Build with `--new-batch`
  (`research/archive/reports/memory-layout.md`).
- `mem-recap-10x`: `mem-layout-10x` plus `ARC3_MEM_RECAP_ASK=1` (the level-start message asks
  for `Level N recap:`; `extract_recap` reads it) and `ARC3_LEVEL_COMPACT_NEEDS_RECAP=1` (a
  cleared level's turns are dropped only once it has a recap). mem-layout-10x (49.56) carried a
  recap on 24 of 106 cleared levels and dropped the other levels' turns anyway. Paired controls:
  `user-combo-10x`, `mem-layout-10x`. Build with `--new-batch`.
- `mem-recap-keep-10x`: `mem-recap-10x` with level-clear compaction off (`ARC3_LEVEL_COMPACT` and
  `ARC3_LEVEL_COMPACT_NEEDS_RECAP` unset): the recap is asked for and saved, and a cleared level's
  turns stay until the window trims them, as in `user-combo-10x`. mem-recap-10x run 1 (45.39)
  saved a recap for 100 of 102 cleared levels yet cleared fewer levels; this arm tests whether
  dropping the history is the cost. Paired controls: `user-combo-10x`, `mem-recap-10x`. Build
  with `--new-batch`.
- `pf-solved-10x`, `pf-keep-10x` (`arm/pf-mem`): `mem-solved-10x` and `mem-recap-keep-10x` plus
  the four `prompt-facts-10x` flags (`ARC3_BUDGET_TRUTH`, `ARC3_PROMPT_HYGIENE`,
  `ARC3_IMAGE_GEOMETRY`, `ARC3_PREFER_TOOL_CALLS`): do the two best single arms stack? Under
  `ARC3_PROMPT_HYGIENE` the whole memory section moves after the Python guidance, unchanged.
- `pf-solved-cap8k-10x` (`arm/pf-mem`): `pf-solved-10x` with each reply capped at 8,192 output
  tokens (12,288 before). Replies over 8k are ~3.5% of replies but ~19% of generated tokens and
  cleared no level in user-combo-10x; the cap spends that time on actions. Reasoning effort is untouched.
  Paired controls: the memory parent and `prompt-facts-10x`. Build with `--new-batch`.
- `pf-keep-append-10x`, `pf-keep-recap-10x`, `pf-keep-memv2-10x`, `pf-keep-stuck-10x`
  (`arm/mem-v2`): `pf-keep-10x` plus one memory-v2 factor each (flags in
  `ARC3-Inference/CONFIGURATION.md`, "memory v2 on pf-keep-10x"). `append`: a level clear is
  staged in the newest message and folded into `<previous_levels>` at the next trim instead
  of redrawing message 1, which re-prefilled the context (31% of B's uncached prefill); the
  slot handover is kept. `recap`: one 800-char cap for the recap's capture and display, tidied
  and marked when cut. `memv2`: the recap change plus readable routes, head words, the
  goal-first recap ask and the `<this_level>` words. `stuck`: one harness count line per level
  once its tokens pass max(60k, 2x the game's cleared-level median). Paired control:
  `pf-keep-10x` (55.79), at matched wall-clock and, for `append`, by cache category. Build
  with `--new-batch`.
- `pf-keep-legible-10x` (`arm/legibility`): `pf-keep-10x` plus `ARC3_LEGIBILITY=1`, six harness
  texts the model misread restated as facts (`ARC3-Inference/CONFIGURATION.md`, "Legibility").
- `pf-keep-v2-all-10x` (`arm/pf-keep-v2`): the four memory-v2 arms' flags and `ARC3_LEGIBILITY=1`
  on `pf-keep-10x` in one run, as the user chose instead of one run per factor. Each factor
  keeps its own flag, so the single-factor arms above bisect a result. Paired control:
  `pf-keep-10x` (55.79), at matched wall-clock and by cache category. Build with `--new-batch`.

## Starting runs and reading results
- **Start one run at a time.** Kaggle allows 2 GPU sessions, but two cold starts reading the
  ~190 GB model at once hung both servers. `push_queue.sh` staggers pushes by 20 min.
- **Score over time:** `score_timeline.jsonl` in the outputs (above), or
  `python3 analysis/score_timeline.py --runs <kernel>/v<N> ... --at 25,60,121` on a saved run.
- **Download outputs** with `python kaggle/fetch_version_output.py mellamoa v4-<arm> <version> <dir> <regex>`.
  `serve_report.py <run_dir>...` summarizes `serve.log` (tok/s, batch, accept length, cache hits).
  It prints the harness's wall-clock tok/s first, then decode rows weighted by the wall time each
  SGLang decode line covers, then the plain per-line statistics as engine diagnostics (SGLang
  writes one line per 40 decode iterations, so small, fast batches write more lines).
- **Measure the #4 fix against an earlier batch.** It is a deliberate exception to the flag rule:
  a bug fix that changes what the model sees and has no off switch. The python tool's deadline
  stops counting the harness's action time, and a killed call returns its partial output and
  executed-action count. Every control built at or after it includes the fix, and
  `ARC3_TOOL_WALL_CAP_FACTOR=1` restores only the old deadline. Push it with `--new-batch` and
  compare with the previous batch's control.
- **Treat small gaps as noise.**
  - Per-game swings of 50 points between identical runs are normal, and a 25-game mean gap
    under ~3 is noise.
  - A 20-minute 10-game panel is a smoke test and early signal, not a verdict.

## Throughput stack arms (`arm/throughput-stack`)
The branch is stack-v2 merged with the lossless throughput levers. Every lever stays behind a flag
or a `CFG` key that is off by default, so `control` and `stack-v2` launch and send exactly what
they did on `arm/stack-v2` (`kaggle/tests/test_throughput_arms.py`,
`ARC3-Inference/tests/test_flags_off_requests.py`).

| Arm | What it adds to `stack-v2` | Run it when |
|---|---|---|
| `stack-v2` | Ledger-on-trim alone | In the same batch as a `tput-*` run, to credit its gap to the throughput levers. Not run yet |
| `tput-safe` | `ARC3_MAX_ACTIVE_STREAMS=11` (gate11), `ARC3_STUB_DEAD_REASONING=1`, `ARC3_VERBATIM_HISTORY=1` with `MAMBA_CACHE` kept at 60, `ARC3_PREFIX_REUSE_DIAG=1` (logging only), `CUDAGRAPH_BS_EXTRA=[3, 5, 6]` | Ran once against `control` (below) |
| `tput-safe-lowm` | `tput-safe` plus `runtime_patches` `m64.py` and `nosplitk.py` (low-M BF16 GEMM up to M=64, no split-K) | Don't: m64+nosplitk measured −0.6% decode on the `tput-levers` panels (PR #138). Its `gate` (bench C3 ≥1% faster) still makes the push need `--gate-passed` |
| `tput-accept05` | `tput-safe` plus `SPEC_ACCEPT_SINGLE`/`SPEC_ACCEPT_ACC` 0.5 | Don't: 0.5 acceptance was faster (+15% on the `tput-levers` panels) but lost on score as `user-combo-10x-accept05` (46.77 vs 56.94 / 52.70; below). Keep acceptance at 1.0 |
| `panel-tput-safe` | `tput-safe` on the balanced 20-min panel with 4 streams | Smoke test: the merged stack runs clean and `PREFIXDIAG totals` shows the reuse. 4 streams cannot test gate11 |
| `tput-coldstart-save` | `coldstart-save` with `tput-safe`'s `CFG` | Saves the JIT cache that `tput-safe`'s config can restore |

**JIT cache.** A cache restores only under the `CFG` and server env it was saved with, less the keys
that compile nothing (`CFG_IGNORED` in `serving/kaggle/jit_cache.py`: the cache and loading flags and
the acceptance thresholds). So the `coldstart-save` cache does not fit `tput-safe`
(`CUDAGRAPH_BS_EXTRA` differs), while `tput-coldstart-save`'s fits every `tput-*` arm. Publish
`tput-coldstart-save`'s cache as in `serving/CUSTOM.md` ("Publishing the cache"). Then add a consumer arm
with `"datasets"` and `serve` `JIT_CACHE_DIR` next to `tput-safe`'s keys. In the offline runs the
clock starts after startup, so the cache adds no play time there; it only counts in the submission.

**Batch run (2026-10-08, code `519c4f5`).** `control` v1 and `tput-safe` v1 ran in one batch
(full 25 games, 121 min). `control` has no `ARC3_LEDGER_ON_TRIM`, so the gap measures ledger-on-trim
and the throughput levers together, not the levers alone. The sim-v2 prior for `tput-safe` (about
+2 offline points) was computed against ledger costs, that is against `stack-v2`, so it is not the
expected delta for this pairing.

| | `control` v1 | `tput-safe` v1 |
|---|---|---|
| Mean score (wins) | **50.62** (7) | **42.09** (6) |
| Harness tok/s (wall clock) | 602 | 591 |
| Decode tok/s, time-weighted | 600 | 629 |
| Batch mean, time-weighted (full share) | 8.93 (41%) | 9.72 (81%) |
| Requests queued (share of decode time) / queue wait total | 4% / 7.1 min | 60% / 66.6 min |
| Prefix reuse / spec accept len | 93.9% / 2.69 | 90.6% / 2.67 |
| Requests / mean output tokens | 2,369 / 1,903 | 2,054 / 2,273 |

- `make significance`: P(`tput-safe` better) = 0.03 over one pair. One pair is a screen, not a verdict.
- The levers did not raise the harness's wall-clock throughput (591 against 602 tok/s). The batch
  was fuller (full for 81% of decode time, against 41%) and requests queued far more.
- `tput-safe@2` was cancelled and `control@2` dropped. Nothing else in this section has run: bench
  `tput-gates`, `tput-coldstart-save`, `panel-tput-safe`, the gated arms and `stack-v2`.
- Logs: `runs/kaggle/v4-control/v1` and `runs/kaggle/v4-tput-safe/v1`.

**If the arm is pursued.** The gated variants are settled without the bench: m64+nosplitk measured
−0.6% and 0.5 acceptance lost on score (above), so neither `tput-safe-lowm` nor `tput-accept05` is
worth a push. `research/archive/notes/arms.md` puts the queue wait above down to gate11: keep the gate
equal to the server's max running requests, as `user-combo-10x` does (14 and 14).
1. To split the ledger from the levers: `stack-v2` and `tput-safe` in one new batch from the same
   code (`kaggle/push_queue.sh stack-v2 tput-safe`), then `stack-v2@2 tput-safe@2` (`--replica 2`).
2. `tput-coldstart-save` and `panel-tput-safe` as needed.

Judge first by the harness's wall-clock tok/s (the first `serve_report.py` row, from `summary.txt`),
then by its time-weighted decode, batch and queue rows and the accept length. The per-decode-log
rows are engine diagnostics. Judge score only over at least two paired runs with `make significance`.

## User-combo arms (`arm/user-combo`, PR #135)
`arm/throughput-stack` plus MoE routing (#48), prefill-v2 (#129), the animation layers (#40) and
the RESET guard (#134). Full runs: 25 games, 121 min. Results and verdicts are in
`research/archive/notes/arms.md`.

| Arm | Settings | Result |
|---|---|---|
| **`user-combo-10x`** (recommended full run) | `ARC3_MAX_ACTIVE_STREAMS=14`, `ARC3_LEDGER_ON_TRIM=1`, `ARC3_ANIM_TRUTH=1`, `ARC3_ANIM_RING=8`, `ARC3_ANIM_DIGEST=1`, `EXPOSE_RESET=on`, `LOCAL_ANALYZER_TEMPERATURE=0.65`, `LOCAL_ANALYZER_CONTEXT_WINDOW=98304`, `ARC3_CONTEXT_DRAIN_TOKENS=43008`, `ARC3_OPENER_DEDUP=1`; serve `MAXREQ` 14, `CUDAGRAPH_MAXBS` 14, `CUDAGRAPH_BS_EXTRA` [3, 5, 6, 11, 12, 13], `MAMBA_CACHE` 84, `CTX` 106496, `MEMFRAC` 0.96; stock 10 experts | 56.94, 52.70, 49.09: mean 52.91 vs control 50.62, not proven better in score; harness tok/s +17% (701–706 vs 602). Fingerprint `399abe34aa243a07`; the exact notebook is in `kaggle/notebooks/` |
| `user-combo` | `user-combo-10x` plus 13 experts per token (`TEXT_CONFIG_OVERRIDE` `num_experts_per_tok` 13, `serve_env` `SGLANG_ARC_MOE_ROUTING=base_norm`, `SGLANG_ARC_MOE_BASE_K=10`) | 49.61; the 13 experts cost ~12% decode |
| `user-combo-10x-accept05` | `user-combo-10x` plus `SPEC_ACCEPT_SINGLE`/`SPEC_ACCEPT_ACC` 0.5 | 46.77: faster decode, worse score; keep acceptance 1.0 |
| `user-combo-16x` | `user-combo-10x` at 16 streams (window 81920, drain 36864, `CTX` 90112, `MAMBA_CACHE` 96, `MEMFRAC` 0.97) | Server CUDA OOM at ~90 min (39.65, invalid); never use `MEMFRAC` 0.97 |
| `prompt-facts-10x` | `user-combo-10x` plus `ARC3_BUDGET_TRUTH=1`, `ARC3_PROMPT_HYGIENE=1`, `ARC3_IMAGE_GEOMETRY=1`, `ARC3_PREFER_TOOL_CALLS=1` (prompt facts and wording, `ARC3-Inference/CONFIGURATION.md`); serving identical | 54.56 (one run). Pair with `user-combo-10x` (56.94, 52.70, 49.09) at matched wall-clock, at least two runs |
| `pf-solved-10x` (`arm/pf-mem`) | `mem-solved-10x` plus the four `prompt-facts-10x` flags; serving identical | Not run. Pair with `mem-solved-10x` (53.20, 53.77) and `prompt-facts-10x` (54.56); build with `--new-batch` |
| `pf-keep-10x` (`arm/pf-mem`) | `mem-recap-keep-10x` plus the four `prompt-facts-10x` flags; serving identical | Not run. Pair with `mem-recap-keep-10x` (53.79) and `prompt-facts-10x` (54.56); build with `--new-batch` |
| `pf-solved-cap8k-10x` (`arm/pf-mem`) | `pf-solved-10x` plus `LOCAL_ANALYZER_MAX_OUTPUT=8192` (one reply capped at 8k output tokens instead of 12k); serving identical | Not run. Pair with `pf-solved-10x`; build with `--new-batch` |
| `pf-keep-append-10x` (`arm/mem-v2`) | `pf-keep-10x` plus `ARC3_MEM_APPEND_COMMIT=1`; serving identical | Not run. Pair with `pf-keep-10x` (55.79); expect about 4.4M fewer uncached prefill tokens; build with `--new-batch` |
| `pf-keep-recap-10x` (`arm/mem-v2`) | `pf-keep-10x` plus `ARC3_MEM_RECAP_CHARS=800`, `ARC3_MEM_COMPRESS_ROUTES_FIRST=1`; serving identical | Not run. Pair with `pf-keep-10x`; build with `--new-batch` |
| `pf-keep-memv2-10x` (`arm/mem-v2`) | `pf-keep-recap-10x`'s flags plus `ARC3_MEM_ROUTE_FORMAT=readable`, `ARC3_MEM_HEAD_WORDS=1`, `ARC3_MEM_RECAP_ASK_V2=1`, `ARC3_MEM_ATTEMPT_COUNTS=1`, `ARC3_MEM_SEEN_GAME_ACTION=1`, `ARC3_MEM_EFFECT_WORDS=1`; serving identical | Not run. Pair with `pf-keep-10x` and `pf-keep-recap-10x`; build with `--new-batch` |
| `pf-keep-stuck-10x` (`arm/mem-v2`) | `pf-keep-10x` plus `ARC3_MEM_STUCK_FACT=1`; serving identical | Not run. Pair with `pf-keep-10x`; needs three consistent sightings; build with `--new-batch` |
| `pf-keep-legible-10x` (`arm/legibility`) | `pf-keep-10x` plus `ARC3_LEGIBILITY=1`: six harness texts the model misread, restated as facts (`ARC3_SEG_BBOX`, `ARC3_ANIM_SCHEMA`, `ARC3_RETAINED_SOURCE`, `ARC3_STALE_TRUTH`, `ARC3_RESET_BAR_FACT`, `ARC3_TURN_BASED_FACT`; `ARC3-Inference/CONFIGURATION.md` §2). System prompt static per run; every other new text is tool output or the newest turn message; serving identical | Not run. Pair with `pf-keep-10x` (55.79); build with `--new-batch`. Measure `boundary` IndexErrors, animation-helper KeyErrors, `inspect` ImportErrors, `action()` calls after a caught `StaleStateActionError`, RESET-bar doubts and real-time bar readings against run B |
| `pf-keep-v2-all-10x` (`arm/pf-keep-v2`) | `pf-keep-10x` plus every memory-v2 and legibility factor in one run, by the user's choice: `ARC3_MEM_APPEND_COMMIT=1` (`pf-keep-append-10x`), `pf-keep-memv2-10x`'s eight flags (`ARC3_MEM_RECAP_CHARS=800`, `ARC3_MEM_COMPRESS_ROUTES_FIRST=1`, `ARC3_MEM_ROUTE_FORMAT=readable`, `ARC3_MEM_HEAD_WORDS=1`, `ARC3_MEM_RECAP_ASK_V2=1`, `ARC3_MEM_ATTEMPT_COUNTS=1`, `ARC3_MEM_SEEN_GAME_ACTION=1`, `ARC3_MEM_EFFECT_WORDS=1`), `ARC3_MEM_STUCK_FACT=1` (`pf-keep-stuck-10x`) and `ARC3_LEGIBILITY=1` (`pf-keep-legible-10x`). System prompt static per run (+271 tokens); every factor keeps its own flag for bisection (`ARC3-Inference/CONFIGURATION.md` §10); serving identical | Not run. Pair with `pf-keep-10x` (55.79) at matched wall-clock and by cache category; a gain or loss is the bundle's, and the single-factor arms above bisect it; build with `--new-batch` |
| `expose-reset` | `EXPOSE_RESET=on` | Not run alone |
| `prefill-v2`, `opener-dedup`, `panel-prefill-v2` | `ARC3_OPENER_DEDUP=1` (+ `LOCAL_ANALYZER_YIELD_TOKENS=6144` in `prefill-v2`) | Not run alone |
| `experts12-basenorm`, `experts13-basenorm`, `experts16-basenorm` | That many experts per token with `base_norm` routing (`serving/CUSTOM.md`) | Not run alone |

The builder refuses to build a fingerprint that already ran, with or without `--push`, so a plain
`python kaggle/build_notebook.py --arm user-combo-10x` stops with "not building a duplicate" and
writes nothing. `--arm user-combo-10x --submission` builds the recorded notebook (fingerprint
`399abe34aa243a07`) as the submission kernel, and `--replica N` builds repeat N; add `--push` to
push either. Build under Python 3.12 (the repo venv's version): the fingerprint covers the gzipped
blobs in the notebook, and Python 3.14 writes a different gzip header byte (OS field), so the same
notebook gets another fingerprint (`11d2c46673d51188`) and slips past the duplicate check.

## Serving benchmarks
A bench measures serving speed and plays no games. `kaggle/bench.json` defines each one: its
configs, its phases and a session budget. Each bench builds its own kernel, `mellamoa/v4-bench-<name>`.

    python kaggle/build_notebook.py --bench smoke             # build only: ~15 min machinery check
    python kaggle/build_notebook.py --bench levers --push     # the throughput plan's session (~150 min)

**What the notebook does.**
- It keeps every setup cell. The serving cell only loads the bench, and the game cell runs it.
- For each config, it runs the same launcher text the arms use, with that config's overrides:
  - `cfg`: keys of `CFG`
  - `env`: the server process env
  - `args_set` / `args_add` / `args_remove`: edits to the launch argv
  - `runtime_patches`: files in `serving/bench_patches/`, applied to a fresh copy of the installed SGLang
- It waits for the server to be healthy, sends one preflight request, runs the config's phases, and
  kills the server.
- A config with `compose` takes the overrides of the fastest earlier config in each group (by
  `select_by`, default clean bs10 median).
- The bench drops configs in `drop_order` when the budget runs short.

**The load.**
- Closed-loop async clients replay real harness conversations from `kaggle/bench_corpus.json.gz`
  (rebuild with `kaggle/make_bench_corpus.py <transcripts dir>`).
- Stream *s* plays game *s*. Every stream shares the 5k-token system prompt, and each turn
  extends the previous one, so the prefix cache serves all but the new exchange.
- The sampling and tools match what the harness sends: T 0.7, top_p 0.95, top_k 20, thinking on,
  `preserve_thinking`, and the `python` tool.
- Phases set the concurrency, duration, `max_tokens` / `min_tokens`, streaming, and the share of
  cold requests (a nonce first).
- The requests are deterministic in (seed, phase, stream, turn), so every config gets the same
  load.

**The output** goes to `/kaggle/working`:
- `bench_results.md`: one row per config and per measured phase.
- `bench_results.json`: adds per batch size, prefill and request numbers.
- `bench/<config>/serve.log`, and `bench/<config>/profile/` for a profile phase.

Both results files are rewritten after every phase.

**Metrics.**
- *Clean* windows are SGLang decode-log windows (40 steps) at a full batch with no prefill since
  the previous line. Compare configs on their median, p90 and share at or above 900, and on step
  ms (1000 x bs x accept / tok/s) at a fixed batch size.
- *Client* tok/s counts the tokens generated in the phase.
- TTFT and per-stream decode tok/s come from streamed requests.
- *Stall* is the decode time lost against the clean speed at the same batch size.
- Lossy configs, such as relaxed MTP acceptance, need a score A/B before adoption.

**Rules.**
- The fingerprint check applies: a bench identical to a recorded run is refused.
- `--push` records the run in `runs.json` as `"kind": "bench"` with `batch_open: false`. A bench
  never joins or blocks an arm batch, because it runs no harness code.
- Download it like an arm: `python kaggle/fetch_version_output.py mellamoa v4-bench-<name> <version> <dir> 'bench'`.
