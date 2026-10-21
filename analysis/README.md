# Analysis tools

These are offline tools that read saved Kaggle runs (`runs/kaggle/<kernel>/v<N>/`). None of them needs a GPU. Other tools live on their own branches until those merge: the scheduler simulators on `arm/replay-sim` and `arm/sched-policies`, and the failure-taxonomy miner on `arm/failure-taxonomy`. Their README sections will go here too. Simulator v2 (from `arm/sim-v2`) and the prefill lever tools came in with prefill-v2 and are described below.

Tests: `ARC3-Inference/.venv/bin/python -m pytest analysis/tests -q`, run from the repo root. Set `PYTHONPATH=tufa-arc-agi-framework/src` so the framework cross-checks run. Without it they are skipped.

## Score timeline (`score_timeline.py`)

This tool rebuilds each game's official score, levels, actions and generated tokens at every moment of a saved run, on the wall-clock axis, plus the run's mean over all games. Use it to compare arms at matched wall-clock minutes instead of only at the end. Findings: [`research/archive/reports/score-timeline.md`](../research/archive/reports/score-timeline.md).

It uses the standard library only, so plain `python3` works. Run it from the repo root. A full 25-game run takes about 1 s.

```sh
nice -n 15 python3 analysis/score_timeline.py \
    --runs fz-arms/v4 fz-ledger-on-trim/v1 fz-ledger-on-trim-r2/v1 \
    --at 25,60,121 --out /tmp/st --html /tmp/st/overlay.html
```

- `--runs`: run directories, as `<kernel>/v<N>` under `runs/kaggle/` or as any path.
- `--at 25,60,121`: prints each run's mean score at those minutes. A `*` marks a minute past the run's end, where the run's final score is shown. The table ends with the final score and `summary.txt`'s mean.
- `--axis bench|notebook|ready`: the clock that `--at` uses.
  - `bench` (the default) counts from the benchmark start. On this clock the 121-minute game budget ends at minute 121.
  - `notebook` counts from the notebook start, so it includes the 9–12 minutes of server startup.
  - `ready` counts from the moment the model server answered `/health`.
- `--out DIR`: writes, for each run, `<kernel>_v<N>_games.csv`, `_mean.csv` and `_levels.csv`. With `--at` it also writes `at_minutes.csv`, which holds every run at every minute on all three clocks.
- `--html FILE`: a self-contained overlay chart. You can switch the clock and the measure (mean score, levels, games finished, actions, tokens). Up to 8 runs are shown at once, picked with checkboxes. The page also has a matched-minute table and per-game small multiples.
- `--step 60`: the grid step in seconds for the CSVs and the chart.
- `--check-events`: also checks the level boundaries against `artifacts/<game>_p0_events.jsonl`.
- `--strict`: exits with status 1 on any mismatch.

The tool prints a note when the runs play different game sets, because their means then cannot be compared.

### Outputs

| File | One row per | Columns |
|---|---|---|
| `_games.csv` | game and grid minute | `t_bench_min`, `t_notebook_min`, `t_ready_min`, `game`, `levels`, `n_levels`, `score`, `tokens`, `actions`, `state` (`waiting`, `playing`, or the final state once the game finished) |
| `_mean.csv` | grid minute | the three clocks, `mean_score`, total `levels`, `tokens` and `actions`, `games_finished`, `n_games` |
| `_levels.csv` | cleared level, at the exact second of the clearing action | `game`, `level`, `t_bench_min`, `t_notebook_min`, `action_num`, `level_actions`, `baseline`, `level_score`, `game_score_after`, `run_mean_after` |
| `at_minutes.csv` | run, clock and `--at` minute | `mean_score`, `levels`, `tokens`, `actions`, `past_end` |

### How it is computed

- **Score.** The score function is a copy of `GameRun._compute_final_score` (`tufa-arc-agi-framework/src/taaf/game.py`, which mirrors `arc_agi.scorecard`), and a test checks the copy against it.
  - A cleared level scores `min(115, (baseline / actions)^2 * 100)`. Uncleared levels score 0.
  - Level *i* has weight *i*.
  - The game score is the weighted mean, capped at `100 × (weight of the levels that scored) / (total weight)`. This cap is why a game won at or under baseline scores exactly 100.
  - `inference.tools.eval` reads the same `final_score`.
- **Level times.** `benchmark.json` stores every action's `wallclock_seconds` since its game started, and each game's `started_at`. The framework counts every action into `actions_per_level[level at that moment]`. Level *i* was therefore cleared by action number `sum(actions_per_level[:i+1])`, and a cleared level's action count is final from then on.
  - This matches the per-action `events.jsonl` for every game of every run that saved it.
- **Final values.** At the end of the run, every game's score, levels, actions and tokens are compared with `summary.txt` and the framework's `final_score`, and any mismatch is printed.
  - They match for all 1,294 games of the 64 saved runs that have a `summary.txt`.
  - The September `duck-*` runs count the last turn's no-move tokens in their summary. Either token convention is accepted.
- **Clocks.** The notebook log (Kaggle's JSON list, where `time` is seconds since the notebook started) provides the other two clocks.
  - **Server ready:** the `READY after Ns` line. Failing that, the tool takes it from `score_timeline.jsonl` (written by notebooks built from now on).
  - **Benchmark start:** the median over games of the `[finished] <game>` line's time minus that game's finish second. In all 52 runs with two or more such lines they agree to within 1 s. Without them, the tool uses `Starting benchmark...`.
- **Passes.** Runs with several passes show each pass as `<game>#p<k>`. The run mean averages the passes per game first, as `summary.txt` does.

The live counterpart, which runs in every notebook built from now on and logs the same quantities while the run is going, is described in [`kaggle/README.md`](../kaggle/README.md#score-timeline-log).

## Simulator v2 (`sim2.py`)

Standard-library Python 3.10+; run it and the prefill tools below from `analysis/` under `nice`, one at a time.

An event-driven model of the harness at Kaggle scale: the priority gate, decode and prefill
on the shared server, client gaps, context growth and trims, and level costs drawn from the
real runs. It is calibrated on `calibration/` (`calibrate.py` rebuilds it; the data card is
`calibration/README.md`).

`sim2.py`, `test_sim2.py`, `calibrate.py` and `calibration/` come unchanged from
`arm/sim-v2` (PR #63), whose README holds the design, the hold-out validation and the
phase-4 scheduler results. In short: **trust it for throughput, slot and context mechanics;
do not use it to rank policies that move compute between games or reorder starts.** The
D′ and easy-first tests need scheduler code that only the stacked arms carry, so on `main`
they skip.

    nice -n 15 python3 sim2.py --policy franzen --ledger off --reps 200
    nice -n 15 python3 test_sim2.py          # or: python -m pytest test_sim2.py

Changes made here (prefill-v2, PR #129):
- **Trim level.** After a trim the prompt is scaled by the retained target
  `budget - min(drain, budget // 2)`, as the harness drains. The sim used to scale by the
  drain itself, which agrees only where drain = budget / 2 (every calibration run); a drain to
  80k was simulated as a drain to 36k.
- **`--lever <name>`** applies a column of `calibration/prefill_levers.csv` to the request
  shapes: each request's prompt increment changes by the tokens the lever adds or removes,
  and a yield lever also moves the invocation boundaries the client-gap model reads.
- **`--resume-gaps`** fits the client gap with separate terms for a fresh turn and a yield
  resumption. The default pools them, and a fresh turn's gap holds the action's env step
  (r2 means: 5.05 s fresh, 2.62 s resumption, 2.08 s inside an invocation), so pooled, a
  removed resumption looks about four times too valuable. Off, every simulation is identical
  to the phase-3 calibration.
- Each result reports `uncached_tokens`, the prompt tokens prefilled.

## Prefill levers (`prefill_levers.py`, `prefill_sizing.py`)

`prefill_levers.py` reads a run's per-game request logs (`*_p0_requests.jsonl.xz`), works
out which messages each request adds, and counts the tokens each prefill-v2 lever removes:
the opener's static block (`ARC3_OPENER_DEDUP=static`), the duplicate board image on a yield
resumption (`resume_image`), and the openers a larger token or time yield budget never sends
(replayed from the observed completions; 2,048 reproduces every real resumption). With
`--table` it writes the per-request table the simulator reads. Token counts use the
Flash-Next tokenizer when `--tokenizer` and the `tokenizers` package are available.

    nice -n 15 python3 prefill_levers.py ../runs/kaggle/fz-ledger-on-trim-r2/v1 \
        --tokenizer "$HOME/.cache/huggingface/hub/models--Intel--Qwen3.8-Flash-Next-W4A16-AutoRound/snapshots/*/tokenizer.json" \
        --requests-csv calibration/requests.csv --table calibration/prefill_levers.csv

`prefill_sizing.py` runs every lever, drain setting and combination against the control
with paired replicates (25 games, 10 slots, 121 min, Franzen), with and without crediting
the decode time that less prefill frees (the calibrated decode curve has today's prefill
stalls baked in).

    nice -n 15 python3 prefill_sizing.py --reps 1000 --ledger off

Results and the recommendation are in `research/archive/reports/prefill-v2.md`.
