# 1009 error analysis: the 2,048-token analyzer yield

The [intermediate-animation-frames Kaggle v1 run](https://www.kaggle.com/code/jonathanwang2022/intermediate-animation-frames) enabled `LOCAL_ANALYZER_YIELD_TOKENS=2048` in its [config](../../ARC3-Inference/configs/intermediate-animation-frames/v1.json#L88). This was an optional **per-turn yield threshold**, not the model's output limit. It interrupted turns before an action even when the model had been writing useful Python code. The completed run scored 49.4942 across 25 games, but that score alone cannot establish the effect of yielding.

The downloaded v1 transcripts in `ARC3-Inference/runs/kaggle/jonathanwang2022/intermediate-animation-frames/v01/transcripts/` contain **442** `turn_token_budget` yields among **1,523** analyzer invocations (29.0%). Every game had at least two. These counts come from the 25 `*_p0.txt` files: one `--- analysis_step=` header per invocation and one `message: Yielded control to solver: turn_token_budget.` status per yield.

## Earlier runs

This was not isolated to the combined-frame notebook. On 2026-10-09, an audit of all **47 downloaded exact-version Kaggle folders** under `ARC3-Inference/runs/kaggle/<owner>/<notebook>/vNN/` found **34 versions across 21 notebooks** with **12,868 observed token-budget yields**. Every affected transcript that prints the effective setting says `yield_tokens: 2048`. The counts below are yield-status messages, not scores or estimates of lost actions. They include two reference notebooks owned by other accounts.

| Notebook | Exact version: observed yields |
|---|---|
| `dangthuygiang/arc3-v207-danibase-base-dfranzen` | v05: 537 |
| `dfranzen/arc-agi-3-milestone-2-solution` | v01: 91 |
| `jonathanwang2022/animation-frames` | v02: 406 |
| `jonathanwang2022/api-signatures-and-frame-diff` | v03: 379; v05: 385 |
| `jonathanwang2022/bp25-experiment` | v02: 211; v03: 230 |
| `jonathanwang2022/bp35-lordhansolo` | v02: 426; v03: 445; v07: 545; v08: 564; v09: 570; v11: 578; v12: 510 |
| `jonathanwang2022/context-adaptive-16` | v01: 497 |
| `jonathanwang2022/history-revisit` | v01: 219 |
| `jonathanwang2022/intermediate-animation-frames` | v01: 442 |
| `jonathanwang2022/intermediate-frames` | v01: 431 |
| `jonathanwang2022/lord-nvfp4-on-pennyroyal` | v01: 191 |
| `jonathanwang2022/new-baseline` | v01: 980; v02: 95; v03: 414 |
| `jonathanwang2022/pennyroyal-w4a16-baseline` | v01: 296 |
| `jonathanwang2022/persistent-animations` | v01: 515 |
| `jonathanwang2022/reset-ab-sk48-su15-off` | v01: 123 |
| `jonathanwang2022/reset-ab-sk48-su15-on` | v01: 149 |
| `jonathanwang2022/thinking-budget-priority-four-waves` | v02: 35 |
| `ticbudja/api-signatures-and-frame-diff` | v01: 355; v02: 370 |
| `ticbudja/thinking-budget-and-user-prompt-trim` | v01: 363; v02: 471; v03: 501 |
| `ticbudja/yield-ucb-scheduling` | v01: 202 |
| `ticbudja/yield-ucb-scheduling-1001` | v01: 342 |

Four more Kaggle versions had `yield_tokens: 2048` in their transcripts but **no observed token-budget yield**: `jonathanwang2022/concurrency-14-streams/v01` and `jonathanwang2022/startup-read-threads/v01`, `/v02`, and `/v05`. Eight other saved Kaggle version folders have no game transcripts, so their runtime behavior cannot be checked: `jonathanwang2022/animation-frames/v01`, `bp25-experiment/v01`, `bp35-lordhansolo/v10` and `/v13`, `startup-read-threads/v03`, `thinking-budget-priority-four-waves/v01`, and `ticbudja/thinking-budget-and-user-prompt-trim-submit/v01` and `/v02`. The remaining reference folder, `lordhansolo/built-on-tufa-labs-duck-harness-milestone-2/v293`, has transcripts but no observed token-budget yield. A separate RunPod smoke output, `ARC3-Inference/runs/runpod/jonathanwang2022/runpod-smoke/c7gq8re8ot95cg/`, shows 11 more yields across two games with the same effective setting. This audit does not cover remote notebook versions whose outputs have not been saved locally.

One clear example is `cn04-2fe56bfb_p0.txt`, `analysis_step=41`, action 132, at 06:44:36. The two model replies show only **434 + 78 = 512 characters of visible thinking**, but their rendered Python calls are about **3,431 + 3,401 characters**. The corresponding Python results are only about **143 + 184 characters**. The turn ends with `step_executed: False`, `yield_tokens: 2048`, and `message: Yielded control to solver: turn_token_budget.` This shows why a short-looking thought can still hit the threshold. The transcript's `raw_tool_calls` JSON is a diagnostic rendering, not a second generated copy. Character counts are illustrative; the transcript does not record the exact completion-token split between thinking and tool-call arguments.

The harness [adds each model reply's reported `completion_tokens`](../../ARC3-Inference/inference/agent/tool_agent.py#L3765-L3793) to `_turn_generated_tokens`, then [checks the threshold](../../ARC3-Inference/inference/agent/tool_agent.py#L7010-L7024). Generated thinking, text, and tool-call arguments are part of those replies; **returned Python output is not added**. The check runs between replies, so it does not cut off a tool call mid-generation. The [code comment](../../ARC3-Inference/inference/agent/tool_agent.py#L202-L209) says token-based yielding was introduced to avoid a wall-clock yield changing with server throughput; it does not make this threshold mandatory. The default is `0` (disabled), and the solver still has its normal stop check.

[V2](../../ARC3-Inference/configs/intermediate-animation-frames/v2.json#L88) changes only `LOCAL_ANALYZER_YIELD_TOKENS` from `2048` to `0`. `yield_seconds=0` and `tool_steps=0` were already disabled. The separate per-response limits remain `max_output=12288` and `thinking_token_budget=8192`; the public run retains its 120-minute runtime limit. V2 is prepared and JSON-validated, **not submitted or scored**, so no gameplay improvement is claimed.
