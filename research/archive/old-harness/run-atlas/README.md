# Run atlas: every ARC-AGI-3 harness run, compared

This folder aggregates the outputs of every run we have data for, across the three
harness lineages we have used, and turns them into comparable numbers, charts and docs.

| doc | what it is |
|---|---|
| [`findings.md`](findings.md) | What we learned, with the numbers and the chart behind each claim |
| [`versions.md`](versions.md) | Every version: what it changed, its parent, how it scored, verdict. Generated |
| [`data/`](data/) | Tidy CSVs: `runs.csv`, `per_game.csv`, `serving.csv`, `leaderboard.csv`. These are also the table view of every chart |
| [`charts/`](charts/) | The charts listed below |
| [`build_atlas.py`](build_atlas.py) | Regenerates `data/`, `charts/` and `versions.md` from raw run outputs |

## Lineages

| era | dates | harness | serving | context | scheduler | best offline mean | best public LB |
|---|---|---|---|---|---|---|---|
| Tufa Duck | Sep 7-21 | Tufa's Milestone-1 Duck harness, our early patches | vLLM, Qwen 3.6/3.8 27B | ~64k | fixed per game | 13.80 (v2-adaptive; budgets vary, not comparable) | 4.37 |
| **Wang fork** | Sep 20-29 | Wang's fork of Duck + our memory ledger, UCB, perception | vLLM, Intel Qwen3.8-Flash-Next W4A16 | 49k (trim at 37k) | UCB + exploration floor | 19.43 (c1) | 7.07 (b1) |
| **Franzen base** | Oct 1- | Franzen's Milestone-2 patch + our flags (ledger-on-trim, schedulers, ...) | SGLang Pennyroyal, same model + MTP drafter, FP8 KV | 128k (trim to ~58k) | priority gate, 10 slots | 52.56 (ledger on trim) | 20.87 (control) |

"Offline" = the 25 public games played in a Kaggle notebook. Full-budget runs are 25 games ×
121-132 min, which gives each game about the compute it gets in the real 110-game rerun.

## Charts

| # | chart | reads as |
|---|---|---|
| 1 | [`01_scores_all_runs.png`](charts/01_scores_all_runs.png) | every comparable full run, mean and trimmed mean, by lineage |
| 2 | [`02_leaderboard_history.png`](charts/02_leaderboard_history.png) | every public submission over time |
| 3 | [`03_franzen_arms.png`](charts/03_franzen_arms.png) | Franzen-era arms against the control |
| 4 | [`04_per_game_heatmap.png`](charts/04_per_game_heatmap.png) | per-game scores: how much single games swing between runs |
| 5 | [`05_throughput.png`](charts/05_throughput.png) | serving speed per Franzen-era run (P90 decode, harness tok/s) |
| 6 | [`06_tokens_vs_levels.png`](charts/06_tokens_vs_levels.png) | Wang vs Franzen: tokens per game vs levels cleared |
| 7 | [`07_completion_curve.png`](charts/07_completion_curve.png) | chance of clearing a level soon vs tokens already spent, against Franzen's C |
| 8 | [`08_tokens_per_level.png`](charts/08_tokens_per_level.png) | generated tokens per cleared level, every comparable run |

## Regenerating

From `arc-agi-3-v3/`, with any Python that has matplotlib (e.g. `final-c1/ARC3-Inference/.venv/bin/python`):

    python reports/run-atlas/build_atlas.py

Inputs: `kaggle-outputs/fz-*` and `kaggle-outputs/all/<kernel>/` (summary.txt, benchmark.json,
serve.log), Franzen's reference run in `external/dfranzen/run-output`, and the per-level dataset
`franzen/analysis/levels.csv`. To pull a specific Kaggle version's output use
`franzen/kaggle/fetch_version_output.py` (the kaggle CLI ignores `/N`).

Scoring note: RHAE caps a game's score at its weighted fraction of levels cleared, so a level
cleared faster than humans earns nothing extra once the cap binds. All scores here use the
harness's own per-game score, which applies the cap.
