# Main lorddraft

`main`'s harness on the 2026-10-10 engine and allocation.

- Harness: `main`'s agent code and [new-baseline/v4.json](../new-baseline/v4.json) settings
  (board upscale 8, sampled animation frames for ordinary and fatal actions at upscale 2,
  `tool_output_tokens = 1024`).
- Engine: Daniel's Intel W4A16 weights on Lord Hansolo's vLLM with Lord's NVFP4 MTP draft
  head and the bounded parallel startup prefetch, ported from `allocation-vc-one-wave`.
- Allocation: priority scheduler with tail fade, `ARC3_MAX_ACTIVE_STREAMS = 15` game slots
  over `max_num_seqs = 14`, `ARC3_GAME_SOFT_STOP_SECONDS = 300` (cherry-picked `f85e0fb`),
  `ARC3_MAX_ANALYZER_FAILURES = 150`.
- Public run: all 25 games, 120-minute soft end. Private rerun: one wave, 120 lanes,
  532 minutes.

Compare with [lorddraft-baseline](../lorddraft-baseline/README.md) (same engine and
allocation, `92dece7` agent) to measure `main`'s harness changes.
