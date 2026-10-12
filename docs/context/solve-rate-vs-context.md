# Level-up rate versus context size

Across five saved Kaggle runs, larger retained context did not show a clear association with a higher level-up rate after controlling for progress within a level or restricting the analysis to requests after the first context drain. This is an observational analysis of level completion, not full-game solve rate, and does not establish that reducing context is harmless.

## Runs and method

The analysis uses server-reported prompt tokens and action-event sidecars from these exact versions:

- `matijaludvig/arc3-vcintel-25x1/v01`
- `matijaludvig/arc3-vcintel-lorddraft-25x1/v01`
- `ticbudja/thinking-budget-and-user-prompt-trim/v02`
- `jonathanwang2022/lorddraft-15-games-14-seqs/v01`
- `jonathanwang2022/lorddraft-retain-decode-checkpoints/v01`

Each level completion is attributed to the response whose tool call executed the completing action. A request is successful if it contains at least one level completion. The steady-state subset begins after a prompt-token drop exceeding 25,000 tokens, excluding the initial context build-up. Conditional logistic regressions group requests by run and game. The high-versus-low context comparison bootstraps those groups 5,000 times with seed 0.

Reproduce from `ARC3-Inference/` with `python ../docs/harness/scripts/context_vs_levelup.py`; the script requires pandas, NumPy, and statsmodels. [Analysis script](../harness/scripts/context_vs_levelup.py). The figures below were reproduced on 2026-10-11.

## Findings

There are 10,772 requests with 474 successful requests; the steady-state subset contains 6,003 requests with 238 successful requests.

| Comparison | Estimate | 95% confidence interval | p-value |
| --- | ---: | ---: | ---: |
| All requests: odds ratio per 10k additional input tokens | 1.038 | 1.005–1.072 | 0.0253 |
| All requests, controlling for log(requests since last level completion) | 0.983 | 0.947–1.019 | 0.347 |
| Steady-state requests: odds ratio per 10k additional tokens | 1.023 | 0.952–1.099 | 0.543 |
| Steady-state assistant turns: odds ratio per 10k additional tokens | 1.014 | 0.938–1.097 | 0.722 |

The steady-state per-request level-up rate at 85k or more input tokens was 1.17 times the rate at 53–85k tokens, with a group-bootstrap 95% confidence interval of 0.92–1.49. This interval includes no difference.

| Timed assistant turns | Turns | Level-up rate | Mean requests per turn | Median generated tokens | Median elapsed seconds | Mean elapsed seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Other turns | 4,837 | 8.5% | 1.924 | 2,012 | 37 | 67.438 |
| Immediately after a drain | 391 | 8.2% | 2.903 | 3,445 | 101 | 373.414 |

Post-drain turns accounted for 31% of measured turn wall time. Elapsed time is measured between transcript turn starts and includes scheduling waits and tool execution; it is not a pure inference-latency measurement. The results suggest that trimming and scheduling overhead deserve attention, while the data do not establish a causal context-size effect. Fixed effects do not remove all differences in level difficulty or game state, and successful-request regressions exclude groups without outcome variation.

The transition-reminder feature preserves the most recent level-up and fatal-action evidence when their original exchanges are evicted. Its [validation and exact Kaggle versions](validation.md) are recorded separately; these observational findings are not an A/B result for that feature.
