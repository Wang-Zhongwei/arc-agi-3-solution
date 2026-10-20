# Dataset `rft_v1`

Built by [rft_data.py](../rft_data.py): levels cleared in < 80% of attempts; per level, clears within the best 35% by actions whose generated tokens are at most the level's 50% quantile (thresholds over all runs' clears). Token columns are exact ([check_rft_data.py](../check_rft_data.py)).

Train: 354 samples from 213 trajectories (one clear of one level in one run), 44 levels, 17 games, 47 runs; 33.6M tokens, 7.30M trained.

## Per game

| Game | Levels | Trajectories | Samples |
|---|---|---|---|
| ls20 | 2 | 19 | 41 |
| wa30 | 3 | 23 | 39 |
| g50t | 5 | 21 | 33 |
| sp80 | 3 | 18 | 27 |
| sk48 | 4 | 18 | 25 |
| tn36 | 5 | 19 | 24 |
| s5i5 | 3 | 13 | 22 |
| m0r0 | 2 | 16 | 20 |
| re86 | 2 | 8 | 20 |
| vc33 | 1 | 9 | 18 |
| bp35 | 2 | 6 | 15 |
| lf52 | 3 | 12 | 15 |
| cn04 | 2 | 10 | 14 |
| su15 | 3 | 7 | 14 |
| ka59 | 2 | 7 | 13 |
| sc25 | 1 | 4 | 7 |
| tu93 | 1 | 3 | 7 |

No level selected in 8 games (ar25, cd82, dc22, ft09, lp85, r11l, sb26, tr87): all their levels were cleared in ≥ 80% of attempts, or no qualifying clear was in a usable run.

## Per level

Clears: all runs / usable runs. Caps: the action and generated-token limits a clear had to meet.

| Level | Pass rate | Attempts | Clears | Caps (actions / tokens) | Trajectories | Samples | Tokens (trained) |
|---|---|---|---|---|---|---|---|
| bp35 L2 | 0.30 | 84 | 25/19 | 49 / 96K | 5 | 14 | 1.55M (0.32M) |
| bp35 L3 | 0.28 | 25 | 7/5 | 42 / 61K | 1 | 1 | 0.12M (0.04M) |
| cn04 L2 | 0.66 | 62 | 41/25 | 41 / 48K | 7 | 8 | 0.64M (0.26M) |
| cn04 L5 | 0.52 | 31 | 16/9 | 60 / 43K | 3 | 6 | 0.64M (0.08M) |
| g50t L1 | 0.76 | 58 | 44/32 | 55 / 62K | 7 | 7 | 0.56M (0.34M) |
| g50t L3 | 0.69 | 39 | 27/17 | 87 / 46K | 6 | 9 | 0.90M (0.17M) |
| g50t L4 | 0.63 | 27 | 17/9 | 106 / 51K | 5 | 10 | 0.95M (0.18M) |
| g50t L5 | 0.41 | 17 | 7/3 | 101 / 67K | 2 | 4 | 0.42M (0.06M) |
| g50t L6 | 0.29 | 7 | 2/1 | 73 / 50K | 1 | 3 | 0.31M (0.05M) |
| ka59 L4 | 0.73 | 48 | 35/26 | 69 / 46K | 6 | 11 | 1.04M (0.16M) |
| ka59 L6 | 0.33 | 30 | 10/5 | 120 / 58K | 1 | 2 | 0.21M (0.05M) |
| lf52 L2 | 0.54 | 61 | 33/24 | 76 / 63K | 8 | 8 | 0.70M (0.18M) |
| lf52 L3 | 0.46 | 33 | 15/11 | 60 / 82K | 2 | 4 | 0.39M (0.06M) |
| lf52 L4 | 0.20 | 15 | 3/3 | 68 / 88K | 2 | 3 | 0.32M (0.06M) |
| ls20 L4 | 0.76 | 79 | 60/47 | 70 / 47K | 15 | 31 | 3.14M (0.55M) |
| ls20 L5 | 0.23 | 60 | 14/9 | 101 / 56K | 4 | 10 | 1.01M (0.16M) |
| m0r0 L1 | 0.79 | 68 | 54/38 | 22 / 35K | 12 | 12 | 0.50M (0.27M) |
| m0r0 L6 | 0.45 | 38 | 17/10 | 74 / 51K | 4 | 8 | 0.82M (0.14M) |
| re86 L6 | 0.71 | 51 | 36/20 | 113 / 75K | 7 | 17 | 1.66M (0.39M) |
| re86 L7 | 0.28 | 36 | 10/6 | 233 / 99K | 1 | 3 | 0.34M (0.09M) |
| s5i5 L3 | 0.57 | 49 | 28/17 | 53 / 89K | 7 | 12 | 1.20M (0.23M) |
| s5i5 L4 | 0.46 | 28 | 13/9 | 42 / 38K | 4 | 8 | 0.78M (0.14M) |
| s5i5 L5 | 0.54 | 13 | 7/5 | 34 / 27K | 2 | 2 | 0.20M (0.04M) |
| sc25 L6 | 0.69 | 36 | 25/15 | 97 / 49K | 4 | 7 | 0.68M (0.13M) |
| sk48 L1 | 0.51 | 79 | 40/31 | 36 / 55K | 10 | 10 | 0.55M (0.26M) |
| sk48 L2 | 0.50 | 40 | 20/18 | 44 / 51K | 5 | 10 | 0.99M (0.17M) |
| sk48 L3 | 0.25 | 20 | 5/5 | 47 / 45K | 2 | 3 | 0.29M (0.05M) |
| sk48 L4 | 0.40 | 5 | 2/2 | 41 / 32K | 1 | 2 | 0.20M (0.03M) |
| sp80 L2 | 0.70 | 60 | 42/29 | 16 / 49K | 11 | 11 | 0.85M (0.34M) |
| sp80 L4 | 0.71 | 38 | 27/20 | 47 / 64K | 6 | 12 | 1.11M (0.29M) |
| sp80 L5 | 0.04 | 27 | 1/1 | 149 / 133K | 1 | 4 | 0.43M (0.14M) |
| su15 L4 | 0.39 | 59 | 23/14 | 53 / 67K | 5 | 12 | 1.26M (0.24M) |
| su15 L5 | 0.30 | 23 | 7/4 | 22 / 55K | 1 | 1 | 0.12M (0.01M) |
| su15 L7 | 0.50 | 6 | 3/1 | 26 / 14K | 1 | 1 | 0.10M (0.02M) |
| tn36 L2 | 0.44 | 61 | 27/19 | 40 / 88K | 6 | 7 | 0.68M (0.18M) |
| tn36 L3 | 0.78 | 27 | 21/14 | 12 / 17K | 4 | 4 | 0.37M (0.02M) |
| tn36 L5 | 0.67 | 18 | 12/7 | 39 / 21K | 4 | 5 | 0.51M (0.06M) |
| tn36 L6 | 0.67 | 12 | 8/6 | 43 / 24K | 3 | 5 | 0.51M (0.06M) |
| tn36 L7 | 0.50 | 8 | 4/4 | 58 / 25K | 2 | 3 | 0.30M (0.06M) |
| tu93 L9 | 0.28 | 40 | 11/4 | 34 / 83K | 3 | 7 | 0.71M (0.18M) |
| vc33 L4 | 0.78 | 59 | 46/28 | 47 / 69K | 9 | 18 | 1.70M (0.32M) |
| wa30 L2 | 0.62 | 61 | 38/26 | 115 / 69K | 10 | 16 | 1.50M (0.46M) |
| wa30 L3 | 0.66 | 38 | 25/19 | 99 / 46K | 8 | 15 | 1.55M (0.19M) |
| wa30 L4 | 0.48 | 25 | 12/9 | 83 / 21K | 5 | 8 | 0.82M (0.10M) |

Hard levels with nothing selected: bp35 L4 (pass 0.29, 0 clears in usable runs); cn04 L6 (pass 0.31, 2 clears in usable runs); ls20 L6 (pass 0.07, 0 clears in usable runs); su15 L8 (pass 0.67, 1 clears in usable runs). With usable clears, every one used more generated tokens than the cap.

## Per run

| Run | Trajectories | Samples |
|---|---|---|
| kaggle/ticbudja/animation-min-frames/v01 | 12 | 18 |
| kaggle/ticbudja/context-64-128k/v01 | 11 | 20 |
| kaggle/jonathanwang2022/bp35-lordhansolo/v07 | 10 | 14 |
| kaggle/jonathanwang2022/reset-context-100k/v01 | 10 | 17 |
| kaggle/ticbudja/last-animation/v01 | 9 | 16 |
| kaggle/ticbudja/reset-baseline/v01 | 9 | 14 |
| kaggle/jonathanwang2022/main-lorddraft/v01 | 8 | 13 |
| kaggle/jonathanwang2022/early-level-recovery-baseline/v01 | 7 | 15 |
| kaggle/jonathanwang2022/persistent-animations/v01 | 7 | 9 |
| kaggle/jonathanwang2022/simple-event-prompts/v01 | 7 | 9 |
| kaggle/jonathanwang2022/bp35-lordhansolo/v02 | 6 | 8 |
| kaggle/jonathanwang2022/bp35-lordhansolo/v12 | 6 | 11 |
| kaggle/jonathanwang2022/context-adaptive-16/v01 | 6 | 14 |
| kaggle/jonathanwang2022/early-level-recovery/v01 | 6 | 11 |
| kaggle/jonathanwang2022/animation-frames/v02 | 5 | 7 |
| kaggle/jonathanwang2022/compressed-trace/v01 | 5 | 8 |
| kaggle/jonathanwang2022/intermediate-frames/v01 | 5 | 9 |
| kaggle/jonathanwang2022/lorddraft-baseline-no-io/v01 | 5 | 8 |
| kaggle/ticbudja/context-60-110k/v01 | 5 | 9 |
| kaggle/ticbudja/context-60-110k/v02 | 5 | 12 |
| kaggle/ticbudja/reset-daniel-context/v01 | 5 | 10 |
| kaggle/jonathanwang2022/api-signatures-and-frame-diff/v03 | 4 | 8 |
| kaggle/jonathanwang2022/early-level-recovery/v02 | 4 | 9 |
| kaggle/jonathanwang2022/new-baseline/v03 | 4 | 6 |
| kaggle/jonathanwang2022/reset-ab-sk48-su15-off/v01 | 4 | 4 |
| kaggle/ticbudja/animation-guards/v01 | 4 | 6 |
| kaggle/ticbudja/thinking-budget-and-user-prompt-trim/v03 | 4 | 6 |
| kaggle/jonathanwang2022/animations-in-context/v01 | 3 | 4 |
| kaggle/jonathanwang2022/api-signatures-and-frame-diff/v05 | 3 | 6 |
| kaggle/jonathanwang2022/bp35-lordhansolo/v08 | 3 | 4 |
| kaggle/ticbudja/yield-ucb-scheduling-1001/v01 | 3 | 4 |
| kaggle/jonathanwang2022/bp25-experiment/v03 | 2 | 5 |
| kaggle/jonathanwang2022/bp35-lordhansolo/v03 | 2 | 3 |
| kaggle/jonathanwang2022/bp35-lordhansolo/v09 | 2 | 2 |
| kaggle/jonathanwang2022/bp35-lordhansolo/v11 | 2 | 3 |
| kaggle/jonathanwang2022/early-level-recovery-baseline/v02 | 2 | 2 |
| kaggle/jonathanwang2022/intermediate-animation-frames/v01 | 2 | 2 |
| kaggle/jonathanwang2022/yield-tokens/v01 | 2 | 3 |
| kaggle/ticbudja/api-signatures-and-frame-diff/v01 | 2 | 4 |
| kaggle/ticbudja/api-signatures-and-frame-diff/v02 | 2 | 5 |
| kaggle/ticbudja/priority-one-wave-lord-vllm/v02 | 2 | 2 |
| kaggle/ticbudja/thinking-budget-and-user-prompt-trim/v02 | 2 | 3 |
| kaggle/ticbudja/yield-ucb-scheduling/v01 | 2 | 5 |
| kaggle/jonathanwang2022/history-revisit/v01 | 1 | 2 |
| kaggle/jonathanwang2022/reset-ab-sk48-su15-on/v01 | 1 | 1 |
| kaggle/ticbudja/priority-one-wave-lord-vllm/v01 | 1 | 1 |
| kaggle/ticbudja/reset-baseline/v03 | 1 | 2 |

Usable runs with nothing selected: kaggle/jonathanwang2022/priority-falloff/v01, kaggle/jonathanwang2022/priority-falloff/v02, kaggle/jonathanwang2022/startup-read-threads/v01, kaggle/jonathanwang2022/startup-read-threads/v02, kaggle/jonathanwang2022/startup-read-threads/v05, kaggle/jonathanwang2022/thinking-budget-priority-four-waves/v02, kaggle/jonathanwang2022/thinking-budget-priority-one-wave/v01, kaggle/ticbudja/reset-baseline/v02.

Runs left out (29 of 84): no source snapshot (17); no harness settings (5); trimmed-history note (not in the transcript) (2); no saved frames (2); ARC3_GAME_PRIORS: per-game hints the model will not have (1); view tool (messages the transcript does not log) (1); another team's fork of the harness; its rebuild was never validated (1).

Requests dropped while building samples: image not drawable: Last level-complete image: (88); image not drawable: last_animation_frames[N]: (67); request over max_tokens (21); group without certain requests (9).

## Validation

12 trajectories, 34 samples: for each level, the next-best clear by actions that was not selected: bp35 L2, g50t L3, g50t L4, g50t L5, lf52 L2, lf52 L3, ls20 L4, re86 L6, s5i5 L3, s5i5 L4, sc25 L6, vc33 L4.
