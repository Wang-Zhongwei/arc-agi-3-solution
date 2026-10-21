## Tokens by response cause (exclusive; % of generated tokens)

| cause | control | temp06 | audit60k | streams14 | ledger-r1 | ledger-r2 | stack-v1 | pooled | pooled on uncleared levels | pooled uncached prompt tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| probe_no_action | 38.8% | 39.8% | 41.7% | 43.6% | 40.6% | 38.0% | 42.3% | 40.7% (13327k) | 13.6% | 44.6% |
| action_effect | 40.9% | 41.2% | 38.7% | 37.4% | 39.1% | 39.6% | 37.8% | 39.2% (12818k) | 12.0% | 37.3% |
| length_cutoff | 6.7% | 6.2% | 7.4% | 4.7% | 6.5% | 7.6% | 5.6% | 6.3% (2077k) | 3.6% | 1.5% |
| python_error | 5.1% | 4.2% | 4.5% | 6.5% | 5.9% | 5.0% | 6.1% | 5.4% (1753k) | 1.6% | 7.1% |
| action_noop_only | 5.1% | 4.8% | 5.0% | 4.2% | 4.8% | 5.9% | 5.1% | 5.0% (1626k) | 1.6% | 4.7% |
| action_level_clear | 2.3% | 2.3% | 1.6% | 2.1% | 2.2% | 2.5% | 2.0% | 2.1% (698k) | 0.0% | 3.6% |
| action_death | 1.0% | 1.3% | 1.1% | 1.2% | 0.7% | 1.2% | 1.0% | 1.1% (355k) | 0.4% | 0.9% |
| python_timeout | 0.1% | 0.3% | 0.0% | 0.2% | 0.1% | 0.1% | 0.1% | 0.1% (35k) | 0.0% | 0.1% |
| no_tool_call | 0.0% | 0.0% | 0.0% | 0.0% | 0.1% | 0.1% | 0.0% | 0.0% (15k) | 0.0% | 0.1% |
| guard_refusal | 0.1% | 0.0% | 0.0% | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% (9k) | 0.0% | 0.1% |

## Python errors (uncaught, all responses) and tokens of responses they ended

| class | control | temp06 | audit60k | streams14 | ledger-r1 | ledger-r2 | stack-v1 | pooled count | tokens where it was the outcome |
|---|---|---|---|---|---|---|---|---|---|
| NameError | 50 | 42 | 42 | 65 | 55 | 43 | 42 | 339 | 550k |
| StaleStateActionError | 38 | 32 | 40 | 40 | 32 | 45 | 42 | 269 | 0k |
| TypeError | 29 | 26 | 30 | 44 | 21 | 20 | 24 | 194 | 322k |
| IndexError | 23 | 15 | 14 | 19 | 30 | 10 | 33 | 144 | 202k |
| KeyError | 26 | 18 | 17 | 17 | 13 | 12 | 32 | 135 | 139k |
| ImportError | 17 | 14 | 11 | 16 | 17 | 16 | 17 | 108 | 207k |
| ValueError | 12 | 12 | 9 | 29 | 21 | 9 | 12 | 104 | 144k |
| AttributeError | 6 | 10 | 12 | 27 | 12 | 6 | 8 | 81 | 112k |
| SyntaxError | 6 | 4 | 5 | 2 | 10 | 14 | 9 | 50 | 67k |
| PythonTimeout(30s) | 7 | 7 | 2 | 8 | 6 | 11 | 2 | 43 | 0k |
| TerminalStateActionError | 1 | 3 | 6 | 4 | 0 | 5 | 10 | 29 | 0k |
| UnboundLocalError | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 3 | 3k |
| AssertionError | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 2 | 2k |
| RuntimeError | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 5k |
| UnparsedError | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1k |

## Actions by cause (exclusive; % of actions)

| cause | control | temp06 | audit60k | streams14 | ledger-r1 | ledger-r2 | stack-v1 | pooled |
|---|---|---|---|---|---|---|---|---|
| noop_strict | 2.3% | 2.0% | 2.3% | 1.7% | 1.9% | 1.7% | 1.9% | 2.0% (741) |
| noop_hud_only | 3.1% | 3.8% | 3.3% | 4.2% | 3.1% | 4.1% | 3.9% | 3.7% (1380) |
| death_action | 0.7% | 0.8% | 0.6% | 0.8% | 0.8% | 0.8% | 0.7% | 0.8% (286) |
| died_attempt | 11.4% | 10.5% | 11.7% | 13.3% | 11.0% | 16.6% | 20.6% | 13.7% (5171) |
| effect_uncleared_level | 12.3% | 12.9% | 19.2% | 11.0% | 10.4% | 14.8% | 15.1% | 13.6% (5154) |
| effect_cleared_level | 70.3% | 69.9% | 62.9% | 69.0% | 72.8% | 62.0% | 57.8% | 66.3% (25031) |

## Wall-clock by cause (exclusive; % of summed game wall-clock)

| cause | control | temp06 | audit60k | streams14 | ledger-r1 | ledger-r2 | stack-v1 | pooled |
|---|---|---|---|---|---|---|---|---|
| probe_no_action | 32.7% | 37.9% | 41.3% | 42.3% | 30.2% | 33.6% | 35.6% | 36.3% (117.5 h) |
| action_effect | 38.6% | 40.8% | 34.1% | 32.6% | 34.9% | 31.2% | 28.9% | 34.4% (111.5 h) |
| wait_before_first_turn | 11.8% | 10.8% | 11.3% | 5.3% | 11.8% | 11.8% | 20.5% | 11.9% (38.6 h) |
| python_error | 5.4% | 2.0% | 4.4% | 7.2% | 9.0% | 8.7% | 9.0% | 6.5% (21.1 h) |
| action_noop_only | 3.3% | 2.3% | 3.5% | 4.9% | 3.8% | 6.1% | 2.3% | 3.7% (12.1 h) |
| length_cutoff | 3.8% | 3.6% | 3.1% | 3.5% | 2.6% | 4.9% | 2.3% | 3.4% (11.0 h) |
| action_level_clear | 3.0% | 1.6% | 1.6% | 2.7% | 6.0% | 2.1% | 0.9% | 2.5% (8.2 h) |
| action_death | 1.0% | 0.6% | 0.5% | 0.9% | 0.3% | 1.3% | 0.5% | 0.7% (2.3 h) |
| request_timeout | 0.2% | 0.3% | 0.2% | 0.2% | 0.2% | 0.2% | 0.1% | 0.2% (0.6 h) |
| no_tool_call | 0.0% | 0.0% | 0.0% | 0.0% | 1.1% | 0.1% | 0.0% | 0.2% (0.5 h) |
| python_timeout | 0.1% | 0.2% | 0.0% | 0.2% | 0.1% | 0.0% | 0.0% | 0.1% (0.3 h) |
| guard_refusal | 0.1% | 0.0% | 0.0% | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% (0.1 h) |
| invocation_without_response | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% (0.0 h) |

## Per-run counts

| count | control | temp06 | audit60k | streams14 | ledger-r1 | ledger-r2 | stack-v1 |
|---|---|---|---|---|---|---|---|
| responses | 2,299 | 2,369 | 2,230 | 2,735 | 2,414 | 2,246 | 2,467 |
| python_calls | 2,275 | 2,348 | 2,209 | 2,717 | 2,391 | 2,218 | 2,445 |
| length_cutoffs | 25 | 23 | 28 | 20 | 24 | 28 | 21 |
| length_cutoffs_with_tool_call | 0 | 2 | 3 | 2 | 2 | 2 | 0 |
| length_cutoffs_nudged_cut_off | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| request_timeouts | 9 | 8 | 10 | 12 | 10 | 9 | 10 |
| request_timeout_s | 357 | 455 | 301 | 400 | 255 | 282 | 254 |
| end:yield_token_budget | 466 | 455 | 481 | 635 | 506 | 449 | 527 |
| yields_without_action | 475 | 466 | 490 | 641 | 513 | 457 | 535 |
| yield_invocation_tokens | 2,034,719 | 2,030,195 | 2,239,306 | 2,577,776 | 2,133,267 | 2,022,693 | 2,221,164 |
| end:yield_stop_requested | 9 | 11 | 9 | 6 | 7 | 8 | 8 |
| end:request_error | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| end:analyzer_error | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| end:no_action_captured | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| deaths | 19 | 21 | 16 | 22 | 21 | 23 | 21 |
| noops_on_cleared | 192 | 212 | 208 | 250 | 182 | 267 | 205 |
| noops_on_uncleared | 109 | 87 | 70 | 62 | 84 | 65 | 128 |
| tool_results_with_guard_refusal | 51 | 52 | 57 | 52 | 39 | 73 | 72 |
| tool_results_truncated | 2 | 5 | 4 | 5 | 5 | 3 | 3 |
| tool_markup_in_text | 0 | 1 | 0 | 0 | 0 | 1 | 1 |
| status:context_overflow_recovered | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| status:degenerate_ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| actions_over_human_on_cleared | 369 | 290 | 274 | 341 | 295 | 576 | 456 |
| tail_tokens_after_last_action | 83,089 | 137,103 | 123,645 | 108,712 | 181,622 | 131,062 | 121,733 |
| stop_requested_tokens | 15,255 | 19,508 | 38,559 | 13,615 | 15,873 | 39,745 | 15,701 |
| uncached_prompt_tokens | 10,523,658 | 10,552,478 | 10,426,561 | 13,462,767 | 10,752,061 | 10,124,373 | 10,926,803 |
| resume_first_requests | 462 | 452 | 477 | 631 | 502 | 449 | 523 |
| resume_uncached_prompt_tokens | 3,433,935 | 2,740,419 | 3,606,762 | 4,396,042 | 3,579,047 | 3,092,965 | 3,458,793 |
| executed_steps | 1,076 | 1,174 | 1,061 | 1,184 | 1,127 | 1,099 | 1,149 |
| executed_steps_ge_12k_tokens | 78 | 71 | 95 | 85 | 71 | 70 | 67 |
| tokens_in_steps_ge_12k | 1,555,280 | 1,399,553 | 1,859,070 | 1,813,526 | 1,342,467 | 1,372,584 | 1,523,978 |
| tokens_in_executed_steps | 4,482,155 | 4,442,199 | 4,533,757 | 5,136,753 | 4,330,950 | 4,401,932 | 4,499,736 |
| deaths_after_40plus_actions | 7 | 7 | 8 | 8 | 6 | 9 | 12 |
| died_attempt_actions_40plus | 461 | 379 | 487 | 603 | 319 | 713 | 1,126 |
| auto_resets | 19 | 21 | 16 | 22 | 21 | 23 | 21 |
| invalid_action_results | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| clamp_lines_in_tool_results | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration_clamped_notes | 40 | 41 | 38 | 51 | 41 | 40 | 36 |

## Levels: tokens, actions and wall-clock on cleared vs never-cleared levels

| run | score | tokens | on uncleared levels | actions | on uncleared | wall on uncleared | final level's tail after last action |
|---|---|---|---|---|---|---|---|
| control | 46.25 | 4.57M | 29.4% | 5632 | 22.9% | 36.2% | 1.8% |
| temp06 | 45.92 | 4.58M | 32.7% | 5136 | 20.0% | 38.5% | 3.0% |
| audit60k | 44.1 | 4.66M | 38.2% | 4991 | 27.9% | 43.4% | 2.7% |
| streams14 | 46.38 | 5.25M | 27.1% | 5318 | 18.5% | 26.0% | 2.1% |
| ledger-r1 | 52.56 | 4.51M | 31.4% | 5278 | 19.5% | 36.8% | 4.0% |
| ledger-r2 | 49.14 | 4.53M | 31.0% | 5654 | 18.1% | 34.6% | 2.9% |
| stack-v1 | 43.32 | 4.62M | 40.7% | 5754 | 31.0% | 49.8% | 2.6% |
| F001-panel | 29.93 | 0.94M | 21.3% | 810 | 7.7% | 22.6% | 3.7% |
| F001-off-panel | 29.22 | 0.94M | 35.1% | 822 | 33.7% | 36.9% | 2.9% |

## Cross-check against summary.txt and benchmark.json

| run | request-log tokens | solver_note tokens | benchmark history tokens | summary.txt tokens | summary.txt actions | benchmark actions | games misaligned | unlogged responses | unanswered requests |
|---|---|---|---|---|---|---|---|---|---|
| control | 4,561,854 | 4,565,244 | 4,482,155 | 4,482,155 | 5,632 | 5,632 | 0 | 1 | 10 |
| temp06 | 4,576,777 | 4,579,302 | 4,442,199 | 4,442,199 | 5,136 | 5,136 | 0 | 1 | 9 |
| audit60k | 4,647,133 | 4,657,402 | 4,533,757 | 4,533,757 | 4,991 | 4,991 | 0 | 1 | 11 |
| streams14 | 5,241,649 | 5,245,465 | 5,136,753 | 5,136,753 | 5,318 | 5,318 | 0 | 1 | 13 |
| ledger-r1 | 4,504,101 | 4,512,572 | 4,330,950 | 4,330,950 | 5,278 | 5,278 | 0 | 1 | 11 |
| ledger-r2 | 4,523,498 | 4,532,994 | 4,401,932 | 4,401,932 | 5,654 | 5,654 | 0 | 1 | 10 |
| stack-v1 | 4,617,193 | 4,621,469 | 4,499,736 | 4,499,736 | 5,754 | 5,754 | 0 | 1 | 11 |
| F001-panel | 0 | 941,330 | 906,946 | 906,946 | 810 | 810 | 0 | 0 | 0 |
| F001-off-panel | 0 | 935,749 | 908,225 | 908,225 | 822 | 822 | 0 | 0 | 0 |
