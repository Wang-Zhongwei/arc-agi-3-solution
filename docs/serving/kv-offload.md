# KV offload to host memory

`kaggle.vllm.kv_offloading_gib` adds a CPU tier under the int4 pool: vLLM's `SimpleCPUOffloadConnector`, eager mode. Config: `configs/int4-kv/v5.json`. Base: the int4-kv v3 code, `85a273fa`.

Preemption does not use this tier. vLLM's preemption frees the blocks and recomputes, and the v2 storm had no preemptions at all: LRU evicted the cached prefixes of waiting requests. The tier keeps copies of those prefixes, so a returning request loads them instead of prefilling again.

Every request still has to fit in GPU KV while it runs, so the tier does not add lanes the way row-level host KV would. Its job is to make a brief overshoot survivable.

## Why eager SimpleCPU

With retention 0, Mamba state exists only at prefill checkpoints.

| Option | Hit depth | Decision |
|---|---|---|
| SimpleCPU, eager | P − 128 to 255, the same as a GPU hit | Chosen: it stores the Mamba hand-off at the previous prefill's end |
| SimpleCPU, lazy | median P − 3,048, with 18 of 48 turns near zero | Stores no hand-offs |
| OffloadingConnector | Multiples of 1,664 only | No partial tail with an eagle group |

A reload copies whole blocks: 57-59 blocks, about 720 MB, for an 85-90K prefix.

## Runtime patches

`inference/framework/lordhansolo/kv_offload/runtime_patches.json`, applied after the int4 and token-exact patches. Each anchor must occur once, or setup stops before the load.

| File | Edit | Without it |
|---|---|---|
| `config/vllm.py` | RecoverSSM's "no KV connector" check accepts this connector | `ValueError` at startup |
| `models/qwen4_exp/nvidia/qsa.py` | `supports_kv_connector` returns True | QSA backend rejected |
| `v1/core/kv_cache_utils.py` | The early return skips this connector | 104 blocks per request, not 102 (about −1.9% pool) |
| `v1/simple_kv_offload/manager.py` | A block found in the tier moves to the evict-last end | The CPU pool runs as FIFO |
| `v1/simple_kv_offload/manager.py` | Hooks for `arc3_offload_stats.py` | No `KV_OFFLOAD_STATS` line |

The first two gates are stated by the code with no reason given, so passing them is unproven until the reload probe and acceptance say otherwise. The kv_cache_utils early return clears a PLE replication marker; the only readers at TP=1 are outside the connector. It does not change the block size, as was first assumed.

## Checks before the games

1. `kv_offload_copy_available`: `cuMemcpyBatchAsync` resolves before the model load. Missing means fallback.
2. `kv_offload/probe.py`, test runs only, with `VLLM_SERVER_DEV_MODE=1`: one prompt of about 12,000 tokens with an image is served twice from GPU hits, `/reset_prefix_cache?reset_external=false` clears the GPU cache, and it is served again. The next-token distributions must agree: mean total variation on the top-5 probabilities, over the positions where the replies still match, within max(0.02, 3 × the same distance between the two GPU hits). Cached tokens must reach P − 300. Greedy ids are only logged, because two GPU hits differ from each other on this runtime.
3. A wrong reload, a shallow tier hit, or a server crash, hang or 5xx at any point in the probe logs `KV_OFFLOAD_PROBE_FAILED` and relaunches. A refused request or a GPU hit too shallow to judge logs `KV_OFFLOAD_PROBE_INCONCLUSIVE`, keeps the tier, and means no tier speed from that run counts. A failed smoke test also relaunches.

Fallback: 25 sequences, no tier, top graph 100, the same as v3. It logs `KV_OFFLOAD_FALLBACK`, and the first server's log is kept as `vllm-openai-server.offload-failed.log`. A crash during the games restarts on this command, but the harness keeps its 27 streams.

## int4-kv v4 settings

| Setting | v3 | v4 | Reason |
|---|---|---|---|
| `max_num_seqs`, `ARC3_MAX_ACTIVE_STREAMS` | 25 | 27 | First peak needs 1,906 of about 1,844 blocks, so the tier must work. Steady state needs 91% of the pool. 30 overfills it even in steady state |
| `kv_offloading_gib` | 0 | 40 | 3,446 CPU blocks, 1.87× the GPU pool. In the replay 40 removes the first-window dip that 32 keeps |
| `cudagraph_capture_sizes` | up to 100 | adds 104, 108 | Exact graphs for 26 and 27 running |
| `sm120_gemm_candidate_rows` | none | 108, 104, 100, 96 | Rows for 24-27 running. The plans were tuned up to M=64, so the preflight decides |
| `kv_offload_fallback_seqs` | none | 25 | The healthy v3 shape |
| `token_exact_history`, `ARC3_TOKEN_EXACT_HISTORY` | off | on | Requested in the same version |
| `N_PASSES` (make line) | 1 | 2 | 25 games in one pass cannot fill 27 lanes |

Replay on the real KV manager and connector, with an emulated scheduler and a step-time model, generated tok/s in minutes 5-45 / 45-120. It is model output, about 6% low before the peak, and it stormed at C=25 with no tier where real v3 did not.

| Setup | 5-45 | 45-120 |
|---|---|---|
| C=25, 32 GiB | 1,112-1,114 | 1,104-1,105 |
| C=27, 32 GiB | 1,080-1,094 | 1,133-1,135 |
| C=27, 40 GiB | 1,108-1,124 | 1,135-1,139 |
| C=28, 40 GiB | 1,089-1,103 | 1,121-1,131 |
| C=30, 40 GiB | 1,070 | 1,036 |

Host memory: v3's lowest `host_available_bytes` was 57.40 GiB and v2's 50.89, so a 40 GiB pin leaves 10.9-17.4 GiB. `cgroup_available_bytes` reads about 4.6 GiB in healthy runs because it does not count reclaimable page cache. Below 12 GiB here, the 532-minute rerun should use 32 GiB.

## Results

**v4 (commit `17715c62`) did not test the tier.** The reload worked: 12,160 of 12,402 tokens from the tier, as deep as the GPU hit, in 0.61 s. The probe then failed on its own bug: the 5th-ranked first token swapped (`User` −7.89 vs `I` −8.14), and it treated any change in the top-5 set as an infinite gap. The launcher fell back to 25 streams, no tier. Boot took 11.7 min, then 10.6 min for the relaunch, leaving 94.5 min of play.

That makes v4 the C=25 control below: 25 streams, no tier, 2 passes, token-exact on, one extra GEMM row. It has about 12 minutes less play than a run without the relaunch.

- Token-exact: probe passed, no `the prompt lacks` errors, hit rate 92.8% at minute 14 (v3 92.3%).
- GEMM: M=96 accepted at time ratio 0.979. M=100, 104 and 108 rejected at 1.06. Control M=64 0.943. Output error about 3.5e-3 at every row. Low-prefill generated tok/s at 24+ running is 1,310 in both v3 and v4, so the row shows no visible gain.
- Throughput, generated tok/s: minutes 5-45 1,233 at 24.6 running (v3 1,183 at 24.0). Minutes 45-90 1,150 at 24.6 running with prefill 3,424 tok/s, the first sustained reading with all lanes full. The FP8 baseline at 14 streams gives 917 in the same window, so +25%. Job level 1,125.
- KV reached 99.0% at minute 15-20 with no storm: prefill stayed near 2,300 tok/s, and the lowest host available memory was 55.34 GiB.

**v5:** the same settings with the probe fixed. Compare with v4, not v3.

## What to read from v5

Two passes make the score incomparable with 58.32 or 65.68, and they raise tok/s by themselves because lanes stay full. v4 is the control for whether C=27 beats C=25.

Boot: `SimpleCPUOffloadWorker [CPU]: 1 tensors, 3446 CPU blocks`, `Setting attention block size to 1664`, and at least 2,640,000 GPU KV tokens (fewer means the grouping patch failed); `KV_OFFLOAD_PROBE_PASSED`, `Token-exact probe passed`, `SM120_GEMM_ACCEPTED`.

| Check | Bar | v3 / v2 reference |
|---|---|---|
| Worst 5-min bin, minutes 14-115 | Generated ≥ 900, prefill ≤ 6,000 tok/s | v3 1,025 and 4,277; v2 211-278 and about 10,000 |
| Minutes 5-14 | Generated ≥ 1,295, prefill ≤ 2,700 tok/s | v3 1,268 at 24.8 running; v2 1,398 at 29.76 |
| Minutes 45-115 | Generated ≥ 1,100 tok/s, running ≥ 25.5 | Replay 1,135-1,139 |
| Acceptance, minutes 15-25 / 45-115 | ≥ 2.80 / 2.82 | v3 2.88 / 2.91; a drop points to bad reloaded state |
| `KV_OFFLOAD_STATS` | ≥ 50 loads, p95 load-to-ready ≤ 2 s, ≥ 3 GB/s | Fewer loads means the run says nothing about the tier |
| Preemptions | Warn above 600 per 120 min | Replay 279-506; real v2 and v3 had none |
| Host | Lowest `host_available_bytes` ≥ 8 GiB | |
| Token-exact | No `token_exact_history: the prompt lacks`; hit rate at minute 14 ≥ 91% | v3 92.2% |

To adopt C=27 with the tier for the 110-game rerun, against the control: minutes 45-90 at least 1.5% higher, minutes 15-45 no more than 2% lower. Otherwise stay at C=25, keeping the tier only if storm-proofing proves necessary.
