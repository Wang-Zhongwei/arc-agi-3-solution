# Adaptive context on Daniel's W4A16 baseline

2026-10-08 · branch `adaptive-context`

Keep Daniel's Intel W4A16 target, INT4 MTP draft, patched Pennyroyal engine,
BF16 PLE, BF16 ordinary input embeddings, and existing FP8 KV cache. The
quality difference observed with Lord's checkpoint is sufficient reason to
retain this baseline; its cause has not been isolated. No FP8 PLE or embedding
quantization is planned for Kaggle.

Follow-up agreement: set the thinking budget to **10,240 tokens** in all three
comparison configs, inside the existing 12,288-token total output limit.
Pennyroyal's existing strict-thinking mode enforces this budget.

## Agreed context policy

All context thresholds below count retained request input tokens; `k` means
1,024 tokens. Output reservation and the safety margin are additional.

- Start with a 64k ceiling and a 32k trim target.
- At that ceiling, expand to 128k if no level has been completed within the
  latest 64k of newly appended conversation. Keep the existing history when
  expanding. Count conversation growth, not repeated request input totals.
- While expanded and without a level-up, trim from 128k toward 96k.
- Any level-up while expanded schedules a return to the small range: at the
  **next context trim**, drop toward 32k, restore the 64k ceiling, and yield.
  This decision persists even if the game stalls again before that trim.
- Preserve Daniel's trim-boundary scheduling handover. Expansion skips a trim
  and therefore grants more uninterrupted work; it is also a scheduling change.
- Prevent the assistant-turn limit from trimming ahead of these token
  boundaries. Server-overflow recovery must still work.

## Serving work

1. Target 16 active streams, with matching harness admission, server request
   capacity, and CUDA graph coverage. Keep a 10-stream control so the context
   policy and concurrency can be compared separately. Measure whether 16
   remains sustainable when several games expand.
2. Keep Daniel's five engine patches: sparse Mamba checkpoint retention and
   recency refresh, speculative-state accounting, Marlin scale compatibility,
   low-M BF16 GEMM, and bounded checkpoint prefetch.
3. Add an optional BF16 `embed_tokens` offload to pinned host memory, sharing
   the table with the draft. This aims to free about 1.18 GiB of GPU memory;
   confirm identical lookup values and measure PCIe overhead. PLE stays BF16.
4. Instrument BF16 GEMM shapes and prepare optional coverage for M=40–64,
   including 56 and 64. Target verification has four positions per sequence
   with three draft tokens; use actual graph-padded shapes, not mean batch
   size, to select kernels. Benchmark before claiming a faster dispatch.
5. Keep Mamba slot count, memory fraction, speculative depth, and prefill
   settings fixed initially. Tune them only if measurements identify a limit.

## Validation and sequence

Implement and check the context transitions first, then prepare 10- and
16-stream configurations. Keep embedding offload and wider GEMM dispatch as
independent switches. Verify serialization/resumption, trimming and handover,
output headroom, and the unchanged default behavior locally.

GPU validation must cover startup, exact embedding lookups, graph capture,
generation and tool calls, long-context continuation, and mixed short/long
streams. Compare one change at a time on identical games and budgets, using
score per wall-clock budget, early-level completion, aggregate throughput,
prefix reuse, prefill time, cache eviction/retraction, and queueing. Follow the
[A/B guide](../rules/t-test.md). Local checks do not establish GPU speed or
quality. Package configurations locally; launching runs is a separate step.

References: [Milestone 2 comparison](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792),
[measured memory budget](MEMORY_BUDGET.md), and
[Daniel's engine patches](../../serving/build_bundle_pennyroyal.sh).

## Implementation status

Implemented locally on `adaptive-context` after recording this plan:

- Per-game context state, newly appended token accounting, progress latch,
  token-boundary trims, overflow recovery and existing scheduler handover.
- Three configs under `ARC3-Inference/configs/adaptive-context/`: fixed-context
  `control-10`, `adaptive-10`, and `adaptive-16`. All use this branch's BP35-v2
  prompt/tool settings, the same 25 public games, one pass and 120-minute public
  budget. The BP35-only token cap is removed from all three. Private play stays
  at 532 minutes with 120 game lanes; only 10 or 16 streams generate at once.
- Adaptive configs reserve 144k total harness context and 152k server context
  for the 128k input ceiling plus 12k output and safety margin. The control uses
  the same server ceiling with its original fixed harness policy.
- Independent BF16 input-table offload and wider GEMM hooks, both **off** in
  every prepared config. Candidate kernels cover selected M=36–64 shapes; 36
  covers B=9 verification. The five existing wheelhouse patches stay intact.
- Configuration documentation and a GPU validation/benchmark command in
  [the serving README](../../serving/README.md#optional-adaptive-context-experiments).

Local validation: **211 tests passed**, including context transitions, real
agent serialization/resumption, whole tool exchanges, overflow recovery,
handover, Make config export, server capacity/headroom and optional patch
fallbacks. New files pass Ruff; `tool_agent.py` and `kaggle.py` pass too. The
launcher retains the same 36 pre-existing lint findings as HEAD. Python syntax
checks and `git diff --check` pass. Optional hooks applied twice successfully
to copies of the pinned Qwen4 source and Daniel's original low-M patch.

`make kaggle-duck` dry-run packaging succeeded for `adaptive-16` with the
explicit local destination `jonathanwang2022/context-adaptive-16` and source
dataset `jonathanwang2022/context-adaptive-16-src`. The prepared bundle is
`/tmp/arc3-adaptive-package/kaggle/jonathanwang2022/context-adaptive-16/dry-runs/20261008_002535_context-adaptive-16/`.
Its seven changed/new agent and serving runtime files match this branch byte
for byte, and the handoff contains 16 streams, 16 server requests, graph coverage
through 16, the adaptive thresholds, and both optional patches disabled.
This dry-run bundle predates the 10,240-token thinking-budget update; regenerate
it from the current config before launching.

No notebook was pushed and no GPU run was launched. GPU startup, numerical
checks, performance, memory capacity at 16 streams and ARC quality remain
unmeasured. The shape logger includes warmup/capture shapes; it does not measure
their runtime frequency. Enable and measure each optional patch separately
after the context-only comparison.
