# Tool output size

Measured on `jonathanwang2022/bp35-lordhansolo` **v12**: 25 games, one pass,
3,006 Python tool results. The run used `LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS=3072`.

Source: [`v12/transcripts/`](../../ARC3-Inference/runs/kaggle/jonathanwang2022/bp35-lordhansolo/v12/transcripts/).

## Statistics

Sizes are the displayed tool-result text's character count divided by four,
after stripping trailing whitespace. These are approximate tokens, not
tokenizer counts. Images and the JSON wrapper sent to the model are excluded.

| Statistic | Approximate tokens |
|---|---:|
| Mean | 131 |
| Median | 81 |
| P90 | 274 |
| P95 | 395 |
| P99 | 808 |
| Maximum | 3,084 |

Percentiles use linear interpolation between sorted observations. Each result
is weighted equally across the 25 transcript files.

| Observation | Results | Share |
|---|---:|---:|
| Displayed size above 1,024 | 21 | 0.70% |
| Displayed size above 2,048 | 4 | 0.13% |
| Truncation marker present | 2 | 0.07% |

These are sizes after truncation; the original lengths of the truncated
outputs were not measured. The threshold counts do not directly predict how
many fields would be truncated under a smaller budget.

## What the setting limits

`analyzer.tool_output_tokens` maps to `LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS`.
The limiter uses a fixed four-characters-per-token approximation, so `3072`
allows **12,288 characters per string field** (`stdout`, `error`, or `result`),
rather than limiting the entire tool response. Structured result values,
metadata, images, and JSON formatting are outside this text-field limit.

With `ARC3_MIDDLE_TRUNCATION=1`, it retains roughly two thirds from the start
and one third from the end, inserting a truncation marker. That marker can
push displayed size slightly above the nominal budget.

Implementation: [`_trim_tool_text` and `_render_tool_payload`](../../ARC3-Inference/inference/agent/tool_agent.py).

At 3,072, the budget leaves room for almost every displayed output in this
run. This is an output-size measurement, not a score comparison between
different budgets.
