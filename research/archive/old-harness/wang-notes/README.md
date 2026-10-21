# Design notes and experiments

This is the repository's shared home for design discussions, observations,
experiment plans, and historical run records. These notes can describe older
configurations or untested proposals; they are not the current parameter specification.
For definitions, defaults, and policy applicability, use the
[reference documentation](../ARC3-Inference/docs/README.md).

## Allocation

- [Rolling extension](allocation/rolling-extension-policy.md): scheduling tradeoffs,
  historical experiments, and the original serving feasibility investigation.
- [Stall allocation](allocation/stall-allocation-policy.md): integration provenance,
  reservation decisions, and a short test recipe.

Exact behavior and settings: [allocation reference](../ARC3-Inference/docs/allocation.md).

## Serving

- [UCB retry root-cause fix](serving/ucb-retry-root-cause-fix.md): cache-drift
  prevention, persistent context recovery, and preserving games after failures.

- [Flash-Next on four A100s](serving/flash-next-a100-smoke.md): runtime pins, smoke
  results, and throughput measurements.
- [Flash-Next on two H100 nodes](serving/flash-next-h100-2node.md): failed attempts,
  successful upstream offload, benchmarks, and the game-run record.
- [NVFP4 vendor comparison](serving/nvfp4-vendor-ab-campaign.md): experiment plan,
  validation criteria, and implementation findings.

- [SPN startup latency](serving/spn-startup-latency.md): checkpoint hashing volume,
  NFS loading differences, and the adapter fix.
- [UCB v5 disk and retry loop](serving/ucb-v05-disk-and-retry-loop.md): why the 2026-09-17
  submission ended at about 6 h, and the vLLM 400 → 500 loop that stalls games.

## Agent behavior

- [Qwen3.8 API ideas](agent/api-change-qwen38.md): draft interface observations.
- [Observed gaps](agent/gaps.md): qualitative failure cases and comparison ideas.
- [Insights](agent/insights.md): brief observations for follow-up.
- [Python tool-call analysis](agent/tool_call_analysis_20260717_142339.md): measured
  call patterns from the July 17 run.

## Adding or updating notes

Use the topic folders above and retain dates, run identifiers, provenance, and
limitations when editing experiment records. Link to canonical definitions in
`ARC3-Inference/docs/` rather than copying parameter tables into notes. Add new
notes to this index. Commands keep their stated working directory; moving a note
does not change the working directory of its historical commands or artifact paths.
