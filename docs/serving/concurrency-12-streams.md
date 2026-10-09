# 12 streams on Daniel's engine (2026-10-09)

**Question:** can Daniel's SGLang engine run more than 10 game streams at once, with the same context (118K history drain) and no new quantization, and does throughput go up?

**Answer:** 12 streams, yes: **727 vs 666 tok/s (+9%)** over the first 45 minutes against new-baseline v3, with no KV squeeze. 14 streams decode faster still, but they overflow the KV pool at the synchronized history peak and collapse for ~20 minutes. Score is not tested yet; these are 20–48-minute speed runs.

Configs: [`configs/concurrency-14-streams/`](../../ARC3-Inference/configs/concurrency-14-streams/) (the [README](../../ARC3-Inference/configs/concurrency-14-streams/README.md) has every run's tables). KV options considered: [kv-cache-room-1009](kv-cache-room-1009.md).

## Runs

All are new-baseline v3 ([config](../../ARC3-Inference/configs/new-baseline/v3.json): Daniel's W4A16 model, input embedding offloaded to host, 25 public games, one pass) with only the serving settings below changed.

| Config | Kaggle notebook version | Streams | `MEMFRAC` | Prefill chunk | `MAMBA_CACHE` | KV pool (tokens) | Result |
|---|---|---|---|---|---|---|---|
| [new-baseline v3](../../ARC3-Inference/configs/new-baseline/v3.json) | `jonathanwang2022/new-baseline` v3 | 10 | 0.96 | 8192 | 60 | 1,190,528 | baseline, 666 tok/s over 0–45 min |
| [v1](../../ARC3-Inference/configs/concurrency-14-streams/v1.json) | `jonathanwang2022/concurrency-14-streams` v1 | 14 | 0.968 | 8192 | 76 | 1,181,312 | OOM on the first prefill |
| [v2](../../ARC3-Inference/configs/concurrency-14-streams/v2.json) | `ticbudja/concurrency-14-streams` v1 | 14 | 0.968 | 4096 | 76 | 1,181,312 | +21% for 15 min, then squeeze |
| [v3](../../ARC3-Inference/configs/concurrency-14-streams/v3.json) | `ticbudja/concurrency-14-streams` v2 | 14 | 0.974 | 2048 | 70 | 1,249,216 | +20–25% when healthy, squeeze at 15–35 min, −17% overall |
| [v4](../../ARC3-Inference/configs/concurrency-14-streams/v4.json) | `ticbudja/concurrency-14-streams` v3 | **12** | 0.974 | 2048 | 62 | 1,282,496 | **+9%, no squeeze** |

All candidates set `WIDE_BF16_GEMM` (BF16 projections at M 36–64) and capture a CUDA graph for every batch size from 7 to the stream count (`CUDAGRAPH_BS`). Run folders: `ARC3-Inference/runs/kaggle/<owner>/concurrency-14-streams/vNN/` on Drive. jonathanwang2022 ran out of weekly GPU quota after v1, so v2–v4 ran on ticbudja.

## v4 against new-baseline v3

Tokens per second from `serve.log` (decode throughput integrated over time), in 5-minute bins from benchmark start:

| Minutes | 0–5 | 5–10 | 10–15 | 15–20 | 20–25 | 25–30 | 30–35 | 35–40 | 40–45 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| new-baseline v3, 10 streams | 661 | 690 | 651 | 662 | 686 | 696 | 667 | 645 | 638 | 666 |
| v4, 12 streams | 731 | 763 | 757 | 722 | 745 | 758 | 733 | 688 | 643 | **727** |

Every bin is higher (mean +60.4 tok/s, paired t = 6.77, df = 8, one-sided p ≈ 7e-5). Treat that p-value loosely: bins of one run are correlated, and each side is one run on one Kaggle host. The harness's own count agrees in direction: 651 tok/s over v4's 48-minute job against 607 over v3's 111 minutes.

| | new-baseline v3 | v4 |
|---|---|---|
| Mean running requests | 9.42 | 11.17 |
| KV usage p50 / p95 / max | 0.48 / 0.78 / 0.86 | 0.57 / 0.80 / 0.87 |
| Prefix-cache hit | 95.4% | 94.7% |
| Intervals with queued requests | 1.1% | 4.7% |
| Draft accept length | 2.62 | 2.64 |
| Free GPU memory after graphs / lowest while serving | 3.14 GB / 0.04 GiB | 1.73 GB / 0.33 GiB |
| Decode-only tok/s at 9 / 10 running | 767 / 794 | 804 / 847 |
| Decode-only tok/s at 11 / 12 running | — | 876 / 895 |

Part of the gain is at equal batch size (847 vs 794 at 10 running). That is either the wide BF16 GEMM, which the baseline lacks at 9–10 running (verify M = 36–40 goes to cuBLAS there), or a faster host; these runs cannot separate the two.

## What we learned

1. **Prefill needs ~0.8 GB + 0.28 MB per chunk token of free GPU memory.** v1 OOMed with 2.20 GB free after graph capture at an 8192 chunk. The lowest free memory during the first prefill: Daniel's v05 4.25 → 0.94 GiB, new-baseline v3 3.14 → **0.04 GiB**, v2 (4096) 2.20 → 0.25 GiB, v3 (2048) 1.63 → 0.37 GiB. **new-baseline v3 survived by ~40 MB**, so its config is one unlucky prefill away from the same OOM. A smaller chunk buys headroom for `MEMFRAC`; prefill is only ~15–20% of wall time, so 2048 costs little.

2. **SGLang caps running requests at `MAMBA_CACHE // 5`.** The pinned wheel (`kv_cache_configurator._calculate_mamba_ratio`) reserves 3 + 2 ping-pong slots per request under `extra_buffer` with the overlap scheduler, though only 4 stay locked during decode. The remaining slots hold prefix-cache checkpoints for games between turns; 14 spare kept the hit rate at 93–97%. Floor: 60 for 12 streams, 70 for 14 (56 would silently cap the server at 11).

3. **The KV pool, not decode, limits concurrency.** All games start together, so their histories grow in lockstep toward the 118K drain. At 14 streams that peak overshoots the pool by ~18%; once full, each admitted turn evicts a paused game's prefix, that game re-prefills its whole history (3.86M tokens in v2), and the squeeze feeds itself until enough games drain (v3: minutes 15–35 at 171–430 tok/s, hit rate 8–59%). At 12 streams the pool peaked at 0.87 (~93K per stream: games do not all peak at once), the same headroom as 10 streams.

4. **The draft's dead LM head is not free memory.** albucino's draft ships an `lm_head.weight` that is never loaded, but SGLang frees it (`set_embed_and_head` + `empty_cache`) right after sizing the pools, so it already serves as prefill headroom. Putting it on the meta device equals raising `MEMFRAC` by ~0.009 ([details](kv-cache-room-1009.md#option-1-checked-not-a-gain-correction-same-day)).

5. **The wide BF16 GEMM is correct and faster.** The pre-server check (`pennyroyal-gemm-bench.jsonl`) passed in every run: 42 shapes at M 36–56 match torch within 0.03 and run 1.01–1.49× faster than cuBLAS under CUDA graphs.

## Not known yet

- **Score.** v4 needs a full 120-minute run (or two passes) against new-baseline v3's 40.33 before it replaces the baseline.
- **Wide GEMM alone.** new-baseline v3 + `WIDE_BF16_GEMM` at 10 streams would separate the kernel from the stream count.
- **13 streams.** At v4's per-stream peak, 13 streams would reach ~0.94 of the pool: possible, close to where 14 tipped over.
- **The last 10 minutes.** v4's lead narrowed to +1–7% at 35–45 minutes as uncached prefill rose (0.55M tokens in 40–45).

## Code

- `pennyroyal_launch.py`: `CUDAGRAPH_BS` (graph sizes) and `WIDE_BF16_GEMM_BENCH` (GPU check of the wide kernels before the server starts; serves without them if the check fails). Defaults keep the existing behavior. See [CONFIGURATION.md](../../ARC3-Inference/CONFIGURATION.md).
- The runs used the `inference-tuneup-on-new-baseline` branch, whose launcher differs from this one only in the Lord-model tuning switches, all off in these configs.
