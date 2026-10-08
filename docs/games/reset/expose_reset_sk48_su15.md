# Exposing RESET on undo games (sk48, su15)

**Conclusion:** exposing RESET did not help. The model swaps undo for RESET, but the score does not improve: 8.08 with RESET against 8.89 without. sk48 got worse, and su15 is unchanged once one outlier run is set aside. No difference is significant. Keep `EXPOSE_RESET` off.

## Why test it

Six of the 25 public games list ACTION7 (UNDO): ar25, bp35, lf52, sb26, sk48 and su15. RESET (action 0) is always legal, but the harness only shows it to the model with `EXPOSE_RESET=on`, which defaults to off.

Most undo use is concentrated in two games. Across 18 earlier Kaggle runs that used undo at least once:

| Game | Undos / actions | Share | Undos per run | Undos inside chains of 2+ | Longest chain |
|---|---|---|---|---|---|
| sk48 | 288 / 1,842 | 15.6% | 22.2 | 98% | 76 |
| su15 | 280 / 3,029 | 9.2% | 21.5 | 90% | 24 |
| lf52 | 37 / 2,531 | 1.5% | 2.8 | 38% | 3 |
| bp35 | 37 / 4,176 | 0.9% | 0.9 | 68% | 8 |
| sb26 | 4 / 2,438 | 0.2% | 0.3 | 0% | 1 |
| ar25 | 2 / 4,415 | 0.0% | 0.1 | 0% | 1 |

In sk48 and su15 the model chains undos to rewind a level. In the 76-undo chain (sk48, `jonathanwang2022/bp35-lordhansolo` v08), the model decided the level was unsolvable from its current state and weighed restarting it. It then found that UNDO does not shrink the budget bar, and undid back toward the level start. RESET would do that rewind in one action, so the question was whether giving it to the model saves turns and raises the score.

None of the top-3 Milestone 2 write-ups reports a result for exposing RESET. Daniel Franzen implemented `EXPOSE_RESET`, but his submission runs with it off.

## Setup

Both arms use the Lord Hansolo baseline (`configs/lordhansolo/v11.json`, the v12 runtime): vLLM serving, 14 active streams, an 8192-token thinking budget and `EXPOSE_UNDO=on`. The only difference is `EXPOSE_RESET`.

| | RESET on | RESET off |
|---|---|---|
| Config | [on.json](../../../ARC3-Inference/configs/reset-ab-sk48-su15/on.json) | [off.json](../../../ARC3-Inference/configs/reset-ab-sk48-su15/off.json) |
| Kaggle notebook | `jonathanwang2022/reset-ab-sk48-su15-on` v1 | `jonathanwang2022/reset-ab-sk48-su15-off` v1 |
| Games × passes | sk48, su15 × 7 | sk48, su15 × 7 |
| Play time | 59.5 min | 58.2 min |
| Source | `c4e61a8` | `c4e61a8` |

Pushed with `make kaggle-duck CONFIG_PATH=configs/reset-ab-sk48-su15/<arm>.json KAGGLE_USERNAME=jonathanwang2022 N_PASSES=7`. All 14 trials of an arm run at once, so every trial generates for the whole play window. The run folders are `ARC3-Inference/runs/kaggle/jonathanwang2022/reset-ab-sk48-su15-{on,off}/v01` on Drive.

## Results

| Game | Arm | Mean score | Median | Mean levels | Levels per pass | Actions/run | RESETs/run | Undos/run | Longest undo chain |
|---|---|---|---|---|---|---|---|---|---|
| sk48 | on | 2.91 | 1.66 | 1.00 | 0,0,1,1,1,2,2 | 189 | 3.0 | 0.4 | 1 |
| sk48 | off | 7.53 | 8.28 | 1.57 | 0,0,1,2,2,3,3 | 187 | 0.1 | 11.6 | 23 |
| su15 | on | 13.26 | 10.87 | 2.86 | 1,2,2,3,3,3,6 | 143 | 5.3 | 0.1 | 1 |
| su15 | off | 10.25 | 13.33 | 2.71 | 2,2,3,3,3,3,3 | 308 | 6.0 | 16.3 | 13 |

The RESET counts include the automatic resets that follow a game over. The off arm shows that these are rare in sk48 but common in su15.

Per-pass scores (on / off):

| Pass | sk48 | su15 | Mean D (on − off) |
|---|---|---|---|
| 0 | 2.05 / 0.00 | 3.99 / 3.03 | +1.51 |
| 1 | 8.33 / 2.78 | 2.22 / 13.33 | −2.78 |
| 2 | 1.20 / 8.33 | 46.67 / 13.33 | +13.10 |
| 3 | 1.66 / 16.67 | 12.81 / 3.01 | −2.60 |
| 4 | 0.00 / 0.00 | 2.89 / 12.40 | −4.75 |
| 5 | 0.00 / 8.28 | 13.33 / 13.33 | −4.14 |
| 6 | 7.13 / 16.67 | 10.87 / 13.33 | −6.00 |

**t-tests** (following the [t-test guide](../../rules/t-test.md)):

- **Across passes:** d̄ = −0.81, s_d = 6.58, t = −0.33, df = 6, p = 0.76 two-sided, and p = 0.62 one-sided for on > off. RESET loses in 5 of 7 passes.
- **Across games:** D = (−4.62, +3.00), N = 2, t = −0.21, df = 1, p = 0.87. Two games are too few for this test to say anything.
- **Per game, unpaired:** sk48 score p = 0.16 (Welch) and 0.24 (Mann–Whitney); su15 score p = 0.64 and 0.51.

## What happened

- **The model stops using undo.** With RESET available, undos per run fall from 11.6 to 0.4 in sk48 and from 16.3 to 0.1 in su15, and chains of 5 or more disappear.
- **sk48 loses progress.** Undo rewinds only as far as needed and does not use up the budget bar. RESET throws away the whole level and counts as an action. sk48 levels fall from 1.57 to 1.00 per run, and the score from 7.53 to 2.91.
- **su15 trades actions for restarts.** Runs with RESET use half as many actions (143 against 308), because one RESET replaces a long rewind. The level count is unchanged (2.86 against 2.71). The higher mean rests on a single 46.7-point run with 6 levels; the median is lower with RESET.

## Limits

- **Small sample.** It is 7 passes of 2 games, and per-pass scores vary from 0 to 47.
- **Only the two heaviest undo users.** The four other undo games use undo rarely, so this test says little about them. It also says nothing about games without undo, where RESET would be the only way to rewind.
- **A first attempt was lost.** The same A/B on RunPod (pods `r9wea83tvxqdz0` and `z8u44a1tfcu2w5`) played to completion, but its outputs never reached Kaggle. The runner deletes the pod even when the output upload fails.
