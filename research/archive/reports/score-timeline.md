# Score timeline: how the score grows over wall-clock

2026-10-08. Tool: [`analysis/score_timeline.py`](../../../analysis/score_timeline.py) (method in [`analysis/README.md`](../../../analysis/README.md)). Data: [`research/data/score-timeline/`](../data/score-timeline/). Charts: `full-runs.html`, `panels-balanced.html` and `panels-franzen10.html` there. Open one locally: it has a clock switch (benchmark start, notebook start, server ready), a measure switch (mean score, levels, games finished, actions, tokens), a matched-minute table and per-game small multiples.

The owner asked for extra logging of how the score changes through time (2026-10-08), and the standing rule is to compare runs on wall-clock time. Each game's official score is rebuilt at every minute of a saved run from `benchmark.json`. The final values equal `summary.txt` for **every game of every saved run that has one: 1,294 games in 64 runs, 491 of them in the 29 Franzen-based `fz-*`/`v4-*` runs**. They also equal the framework's `final_score`. The level boundaries also match the per-action `events.jsonl` wherever a run saved it.

From now on every notebook that `kaggle/build_notebook.py` builds also logs this live, to `score_timeline.jsonl` and a `[score]` line every 5 minutes (see [`kaggle/README.md`](../../../kaggle/README.md)).

## Summary

- **Half the final score arrives after minute 50.** Over 11 full runs the mean reaches half its final value at minute 49–62 and 90% at minute 94–115. The first 20 minutes give about 10% of the final score, even though 253 of the 1,156 level clears (22%) happen then. Early clears are mostly level 1 and 2, which carry the smallest weights.
- **The score climbs steadily from minute 20 to 80, then slows.** It gains 0.37–0.60 points per minute from minute 20 to 80 and 0.16–0.40 from minute 80 to 121. Throughput does not fall: the control decodes 36–38k tokens per minute in every phase. The slowdown comes from what is left: the easy games are won (median win at minute 76) and the remaining levels are harder.
- **The 25-minute mark says almost nothing about the final ranking.** Across the 11 full runs the correlation between the mean at minute 25 and the final mean is r = 0.21 (Spearman 0.18). It is 0.63 at minute 60, 0.77 at minute 90 and 0.90 at minute 110. The control was last of the 11 at minute 25 (4.88) and finished fifth (46.25).
- **ledger-on-trim's gain appears only after minute 50.** Its two runs match the four control-like runs up to minute 50 (gap −0.5 to +0.2). From minute 60 both ledger runs sit above all four: the gap is +3.9 at minute 60, peaks at +8.1 to +8.2 between minutes 80 and 100, and closes to +5.2 by minute 121 as the controls catch up late. That timing fits a mechanism that starts working only once a game's context first fills and is trimmed. This is consistent with the arm's design but was not measured here.
- **Startup costs about 3 points on the notebook clock.** The benchmark starts 9.2–12.2 minutes after the notebook does. At notebook minute 121 the full runs are 0.8–5.8 points below their final scores (median 2.7). The server was ready before the games started in every full run but one: fz-ledger-stallswap's benchmark was released at the 12-minute deadline and the server came up 1.0 minute later.
- **Panels flip late too.** On the balanced panel at 4 streams, `v4-panel-memory-v2-both` led at minute 10 (4.67 vs 1.78 for D′) and finished last but one (5.95 vs 10.77). The F001 pair swapped the lead 5 times in 25 minutes (minute by minute) and ended 0.7 apart.

## Full runs: mean score at matched wall-clock minutes

25 games, 121-minute game budget, minutes since the benchmark started. The final value equals `summary.txt` for every run. `at_minutes.csv` has the same table on the notebook and server-ready clocks.

| run | arm | 25 min | 60 min | 90 min | 121 min | final |
|---|---|---:|---:|---:|---:|---:|
| fz-arms/v4 | control | 4.88 | 24.29 | 34.01 | 46.25 | 46.25 |
| fz-arms/v5 | temp06 | 7.80 | 22.65 | 38.60 | 45.92 | 45.92 |
| fz-arms/v6 | audit60k | 8.39 | 24.69 | 35.50 | 44.10 | 44.10 |
| fz-streams14/v1 | streams14 | 9.32 | 25.35 | 33.62 | 46.38 | 46.38 |
| fz-ledger-on-trim/v1 | ledger-on-trim | 8.48 | 28.71 | 42.23 | 52.56 | 52.56 |
| fz-ledger-on-trim-r2/v1 | ledger-on-trim repeat | 7.19 | 27.57 | 43.92 | 49.14 | 49.14 |
| fz-ledger-streams12/v1 | ledger + 12 streams | 9.78 | 29.13 | 39.49 | 48.08 | 48.08 |
| fz-ledger-stallswap/v1 | ledger + stall swap | 9.09 | 27.16 | 39.74 | 45.22 | 45.22 |
| v4-stack-v1/v1 | D′ + ledger + #2 fix | 9.18 | 27.71 | 34.37 | 43.32 | 43.32 |
| fz-ucb/v1 | UCB admission | 7.82 | 28.01 | 38.94 | 43.84 | 43.84 |
| fz-experts16-full/v1 | 16 experts | 6.52 | 20.82 | 30.39 | 38.69 | 38.69 |

On the **notebook clock** (minutes since the notebook started, which includes the 9–12 minutes of server startup), minute 121 gives 43.59 for control, 49.78 for ledger-on-trim, 47.97 for its repeat and 42.38 for stack-v1.

Other patterns in the table:
- stack-v1 and UCB were among the leaders at minute 60 (27.71, 28.01) and among the weakest at the end. UCB gained the least of any run after minute 80 (0.16 points per minute). stack-v1 gained only 6.7 points from minute 60 to 90, against 13.5 for ledger-on-trim.
- streams14 ran 15% more tokens and led the field at minute 40 (21.1). It finished level with control, and 3.7 of its points came in the last 5 minutes.

## Where the gains happen (11 full runs pooled)

| minutes | level clears | share of the final mean score gained |
|---|---:|---:|
| 0–10 | 116 | 4.0% |
| 10–20 | 137 | 6.4% |
| 20–30 | 186 | 12.7% |
| 30–40 | 118 | 9.5% |
| 40–50 | 118 | 11.6% |
| 50–60 | 99 | 12.7% |
| 60–70 | 87 | 9.9% |
| 70–80 | 77 | 8.9% |
| 80–90 | 56 | 6.0% |
| 90–100 | 57 | 5.9% |
| 100–110 | 52 | 5.7% |
| 110–121+ | 53 | 6.8% |

The game score weights level *i* by *i* and caps it at the weight of the levels that scored. A late level-6 clear therefore moves the mean far more than an early level-1 clear. Clears fall off after minute 30 while each clear is worth more, so the score keeps rising steadily until about minute 80.

## Panels: mean score at matched minutes

Balanced 10-game panel. The `v4-panel-*` arms run 4 streams for 20 minutes, the F001 pair 10 streams for 25 minutes, and the `fz-*` arms 10 streams for 20 minutes. Compare only within a setup.

| run | 5 min | 10 min | 15 min | 20 min | 25 min | final |
|---|---:|---:|---:|---:|---:|---:|
| v4-panel-control | 0.58 | 1.08 | 3.47 | 5.13 | – | 5.13 |
| v4-panel-dprime | 0.58 | 1.78 | 5.12 | 10.77 | – | 10.77 |
| v4-panel-easy-first | 0.54 | 1.82 | 3.80 | 7.51 | – | 7.51 |
| v4-panel-memory-v2-both | 1.44 | 4.67 | 5.81 | 5.95 | – | 5.95 |
| v4-fable-f001 v2 | 7.00 | 8.28 | 17.47 | 24.03 | 29.93 | 29.93 |
| v4-fable-f001-off | 3.20 | 9.75 | 11.40 | 18.86 | 29.22 | 29.22 |
| fz-tput-base | 5.10 | 11.96 | 16.48 | 17.45 | – | 17.45 |
| fz-experts8 | 4.39 | 6.55 | 10.12 | 13.93 | – | 13.93 |
| fz-spec-adaptive | 0.79 | 8.90 | 13.22 | 16.08 | – | 16.08 |
| v4-gguf-smoke (llama.cpp, 15 min) | 0.48 | 2.09 | 2.62 | – | – | 2.62 |

On Franzen's own 10-game demo set (`panels-franzen10.html`), experts16 led tput-all v7 at every 5-minute mark (6.70 vs 3.32 at minute 5, 45.97 vs 43.20 at the end). The full-run comparison of the same change went the other way (38.69 vs 46.25).

## What this means for the arms

- Judge arms at minute 121 of full runs, or on a full-run curve. Do not judge them at minute 25. A screen that cuts a full run short ranks arms close to chance (r = 0.21 at minute 25).
- An arm that acts only after contexts fill, such as ledger, trim or memory changes, cannot show up in a 20–25-minute panel by design. Panels suit startup, throughput and early-scheduling changes.
- When two full runs are compared, look at the gap on the overlay from minute 50 to 121, not only at the end. ledger-on-trim's +5 at the end follows a +8 lead between minutes 80 and 100. An arm that keeps the lead it builds there would be worth about 3 more points.

## Unverified

- Server-ready time is missing for runs whose log has no `READY after` line (all `duck-*`/`taaf-*` runs and v4-gguf-smoke, which ran llama.cpp). Builds from now on stamp it, and the tool also reads it from `score_timeline.jsonl`.
- The benchmark-to-notebook clock offset is fitted from the `[finished]` lines (median of log time minus the game's finish time). In all 52 runs with two or more such lines they agree to within 1 s. Runs without `[finished]` lines fall back to the `Starting benchmark...` line.
- Old Duck-harness runs (`duck-*`, Sept 2026) count the last turn's no-move tokens in `summary.txt`. The cross-check accepts either convention, and their scores match.
