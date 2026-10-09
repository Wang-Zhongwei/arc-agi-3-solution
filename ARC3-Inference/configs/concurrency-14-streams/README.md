# Concurrency 14 streams

[v1.json](v1.json) is [new-baseline v3](../new-baseline/v3.json) (Daniel's W4A16 model, input embedding offloaded to host) at 14 streams instead of 10, as a short speed run: `public_runtime_minutes` 32, about 20 minutes of games after the server is up. Numerics are unchanged: no weight, KV, or activation quantization beyond v3's.

| `kaggle.sglang` | v3 | v1 | Why |
|---|---|---|---|
| `MAXREQ`, `CUDAGRAPH_MAXBS` | 10 | 14 | 14 streams; `ARC3_MAX_ACTIVE_STREAMS` 14 in `harness_env` |
| `CUDAGRAPH_BS` | 1, 2, 4, 7, 8, 9, 10 | adds 11, 12, 13, 14 | 10 streams averaged ~1 idle slot, so 11–13 running is common; uncaptured sizes would pad to 14 |
| `MAMBA_CACHE` | 60 | 76 | 4 slots per running request (56 at 14) plus the same 20 spare for prefix-cache checkpoints |
| `MEMFRAC` | 0.96 | 0.968 | v3 left 3.14 GB free after graph capture; keep ≥ 1.2 GB (0.972 OOMed on the tuneup branch) |
| `WIDE_BF16_GEMM` | off | on | Verify forwards carry 4 rows per request (44–56 at 11–14), above the M ≤ 32 low-M path; `WIDE_BF16_GEMM_BENCH` checks and times the kernels first (`pennyroyal-gemm-bench.jsonl`) |

Expected KV pool ≈ 1.18M tokens (v3: 1,190,528): +0.008 memory fraction ≈ +57K, 16 more Mamba slots ≈ −68K. At v3's per-request footprint (median ~65–76K, p95 ~88K at 10 running), 14 streams fit at the median but may queue at the p95.

Compare against v3 (Kaggle `jonathanwang2022/new-baseline` v3): decode tok/s by running requests from `serve.log` (v3: 718 / 753 / 773 at 8 / 9 / 10), KV usage and queueing, and the harness tok/s over the same game window. Scores from a 20-minute run are not comparable to 120-minute runs.

## v1 result (Kaggle v1, 2026-10-09): out of memory on the first prefill

The server came up (KV pool 1,181,312 tokens, 2.20 GB free after graph capture vs v3's 3.14) and crashed on the first batch of game prompts: `torch.OutOfMemoryError` in `_short_conv`, 105 MiB free. No game got a response. An 8192-token prefill forward needs about 3.1–3.3 GB on top of the pools: Daniel's v05 fell from 4.25 GB free to 0.94 GiB during its first prefill, and new-baseline v3 from 3.14 GB to **0.04 GiB** (it survived by ~40 MB). So 0.968 does not fit with an 8192 chunk, and 0.96 at 14 streams (~2.96 GB free) would not either.

The wide BF16 GEMM check passed (`pennyroyal-gemm-bench.jsonl`): all 42 shapes (M 36–56) match torch within 0.03 absolute and run 1.01–1.49× faster than cuBLAS under CUDA graphs.

## v2: 4,096-token prefill chunk

[v2.json](v2.json) is v1 with `CHUNK` and `MAX_PREFILL` 4096 (from 8192 and 16384); everything else, including `MEMFRAC` 0.968, is unchanged. Scaling the ~3.1–3.3 GB peak of an 8192-token forward, 4096 needs roughly 1.6–1.9 GB against v1's 2.20 GB free. (The tuneup runs held 0.967 at 14 streams with a 2048 chunk and 1.26 GB free; 6144 would need ~2.4–2.5 GB and likely OOM at 0.968.)

Prefill is ~15–20% of wall time (decode-only intervals 53–54%; the rest mostly decodes alongside prefill), and each forward pays a fixed ~40–60 ms (mostly weight reads) on ~0.5 s of compute at ~8K tok/s, so halving the chunk should cost ~5% prefill time, ~1% tok/s (estimate). Chunk size does not change decode batches: compare decode-only intervals by running requests (concurrency effect) separately from prefill intervals (chunk effect).

## v2 result (Kaggle `ticbudja/concurrency-14-streams` v1, 2026-10-09)

v2.json ran on the teammate account (jonathanwang2022's weekly GPU quota was used up), so its Kaggle version is v1. No OOM: lowest free memory during serving 0.25 GiB (v3: 0.04). 23 minutes of games, 25 games, 753K tokens. Compared with new-baseline v3 over the same windows from benchmark start (`serve.log`):

| | v3, 10 streams | v2, 14 streams |
|---|---|---|
| **First 15 min** (before the KV pool filled) | | |
| Effective tok/s | 667 | **808** (+21%) |
| Decode-only tok/s | 778 | 972 (+25%) |
| Mean running requests | 9.47 | 13.48 |
| Prefix hit | 95.8% | 95.3% |
| **All 23 min** | | |
| Effective tok/s | 668 | 602 (−10%) |
| Uncached prefill tokens | 1.36M | 4.83M |
| Prefix hit | 95.4% | 82% |
| KV usage p50 / p95 / max | 0.41 / 0.79 / 0.86 | 0.54 / 0.96 / 0.99 |
| Intervals with queued requests | 1.4% | 12.6% |

Decode-only tok/s by running requests: 793 at 10 (v3); 921 / 953 / 985 at 12 / 13 / 14 (v2). Draft acceptance is unchanged (2.6).

**At ~15 minutes the KV pool filled and the prefix cache thrashed.** All 14 games started together, so their histories grew in lockstep and reached ~85K tokens each before the first 118K history drain (14 × 85K ≈ 1.19M against 1,181,312 tokens). With locked usage at 0.94–0.99, requests queued 80–100% of the time, and admitting a queued request evicted the cached prefix of a game between turns; that game then re-prefilled its whole conversation. 954 prefill chunks restarted from the system prompt alone, 3.86M tokens (80% of all prefill); the hit rate fell to 5–38% per 3 minutes. v3 peaked at 0.86 and then drained to ~0.3.

Over a full drain cycle (59K → 118K, ~88K average) 14 streams need ~1.23M tokens, more than the pool; 12 need ~1.06M (0.9). Without new quantization, 14 streams need either fewer tokens per stream (earlier drain, a context change) or admission that backs off when the pool is nearly full. A fixed 12 streams (decode-only ~921 tok/s) should fit as is.

## v3: the remaining KV room, 45 minutes

[v3.json](v3.json) is v2 plus the options in [kv-cache-room-1009](../../../docs/serving/kv-cache-room-1009.md) that keep numerics:

| `kaggle.sglang` | v2 | v3 | Why |
|---|---|---|---|
| `MAMBA_CACHE` | 76 | 70 | SGLang caps running requests at `MAMBA_CACHE // 5`; 70 is the floor for 14 (+0.33 GiB, ~27K tokens) |
| `CHUNK`, `MAX_PREFILL` | 4096 | 2048 | Peak prefill headroom ≈ 0.80 GB + 0.28 MB/token (fit to ~3.10 GB at 8192 and ~1.95 GB at 4096), so ~1.37 GB at 2048 |
| `MEMFRAC` | 0.968 | 0.974 | Spends ~0.57 GiB of that headroom (+~46K tokens); ~1.63 GB stays free after graph capture (≥ 1.2 GB) |
| `public_runtime_minutes` | 32 | 55 | ~45 minutes of games, past the first history drain |

CUDA graphs stay at 1–14: dropping 11 and 13 frees ~0.05 GB and would pad the common 13-request batch to 14. Expected KV pool ≈ 1.25M tokens (v2: 1,181,312), still below the ~1.47M a synchronized start reaches before the first drain, so expect a milder, later squeeze rather than none. Watch: lowest free memory, prefix hit rate and queueing around minutes 15–20, accept length (v2: 2.63), and effective tok/s against new-baseline v3 over the same window.

## v3 result (Kaggle `ticbudja/concurrency-14-streams` v2, 2026-10-09)

The memory settings landed as predicted: KV pool 1,249,216 tokens (+5.7% over v2), 1.63 GB free after graph capture, lowest 0.37 GiB during serving, no OOM, no Mamba cap (70 // 5 = 14). The cache hit rate held at 93–97% outside the squeeze, so 70 Mamba slots are enough; accept length 2.63. 46.6 minutes of games.

Per 5 minutes from benchmark start, against new-baseline v3 (10 streams) over the same window:

| Minutes | v3 baseline tok/s | v3 run tok/s | Running | KV mean / max | Queued | Uncached prefill | Prefix hit |
|---|---|---|---|---|---|---|---|
| 0–5 | 661 | 765 | 13.6 | 0.19 / 0.35 | 11% | 0.42M | 90% |
| 5–10 | 690 | 823 | 13.5 | 0.48 / 0.64 | 5% | 0.31M | 95% |
| 10–15 | 651 | 825 | 13.2 | 0.75 / 0.94 | 1% | 0.24M | 97% |
| 15–20 | 662 | **299** | 13.1 | 0.92 / 0.99 | 45% | 1.82M | 55% |
| 20–25 | 686 | **187** | 12.2 | 0.95 / 0.99 | 100% | 2.37M | 8% |
| 25–30 | 696 | **171** | 11.5 | 0.95 / 0.99 | 100% | 2.36M | 18% |
| 30–35 | 667 | 430 | 12.7 | 0.67 / 0.95 | 31% | 1.55M | 59% |
| 35–40 | 645 | 772 | 13.4 | 0.37 / 0.51 | 5% | 0.44M | 93% |
| 40–45 | 638 | 821 | 13.2 | 0.62 / 0.79 | 1% | 0.27M | 97% |

Over 46.6 minutes: 552 vs 666 tok/s (−17%); decode-only 922 vs 775; 10.3M vs 2.8M uncached prefill tokens; queueing in 16.4% vs 1.2% of intervals. Decode-only by running requests: 879 / 938 / 959 / 981 at 11 / 12 / 13 / 14.

**The squeeze is self-sustaining.** Once the pool is full, every admitted turn evicts a paused game's prefix; that game re-prefills its whole history on its next turn, which needs more KV and more time, so games advance slowly and reach the 118K drain point late. It lasted ~20 minutes (15–35) and ended only when enough games drained. Games then restart near 59K together and grow at similar rates, so the next squeeze is likely one drain cycle later (new-baseline v3's own KV peaks recur every ~15–20 minutes at 10 streams). +6% KV did not move it: a fresh 14-stream run needs ~1.47M tokens at its peak.

Healthy phases (0–15 and 35–45 minutes) ran 765–825 tok/s against 638–696: 14 streams are worth +20–25% whenever the pool is not full. Keeping that gain needs load control rather than more KV: admit fewer turns or drain history earlier when the server's KV usage nears full, or a lower fixed concurrency (11 fits the peak).

## v4: 12 streams

v3 showed that 14 streams overshoot the KV pool at the synchronized peak by ~18% (14 × ~109K tokens per game before its 118K drain against 1.25M), which turned into a self-sustaining squeeze. At 12 streams the peak is ~1.31M against ~1.29M, a 1–5% overshoot: Daniel's v05 ran at 0.98 of its pool with 1.9% queueing and no collapse, so a small overshoot should mean brief queueing, not eviction thrash.

[v4.json](v4.json) is v3 with everything that scales with concurrency resized for 12:

| | v3 | v4 | Rule |
|---|---|---|---|
| `MAXREQ`, `CUDAGRAPH_MAXBS`, `ARC3_MAX_ACTIVE_STREAMS` | 14 | 12 | streams |
| `CUDAGRAPH_BS` | 1, 2, 4, 7–14 | 1, 2, 4, 7–12 | every size from 7 up, so common batches are not padded |
| `MAMBA_CACHE` | 70 | 62 | ≥ 5 × 12 = 60 (SGLang's cap); 62 − 4 × 12 = 14 unlocked, as in v3 (hit rate 93–97%) |
| Wide BF16 GEMM rows | 36–56 | 36–48 | verify M = 4 × batch for batches 9–12 (the launcher benchmarks exactly these) |
| `CHUNK`/`MAX_PREFILL`, `MEMFRAC` | 2048, 0.974 | unchanged | v3: 1.63 GB free after graphs, lowest 0.37 GiB |

Expected KV pool ≈ 1.29M tokens (v3 + 8 Mamba slots ≈ +36K, plus slightly smaller graphs). Expected ~740–790 tok/s effective if no squeeze forms (decode-only 879–938 at 11–12 running, ~11 running on average). Pass: minutes 15–35 stay healthy (prefix hit ≥ 90%, little queueing); otherwise fall back to 11 streams.

## v4 result (Kaggle `ticbudja/concurrency-14-streams` v3, 2026-10-09): no squeeze, +9%

KV pool 1,282,496 tokens (predicted ~1.29M), 1.73 GB free after graph capture, lowest 0.33 GiB, no OOM, `max_running_requests` 12 (no Mamba cap). 48 minutes of games; accept length 2.64.

| Minutes | new-baseline v3, 10 streams | v4, 12 streams | Running | KV mean / max | Queued | Prefix hit |
|---|---|---|---|---|---|---|
| 0–5 | 661 | 731 | 11.7 | 0.17 / 0.31 | 9% | 90% |
| 5–10 | 690 | 763 | 11.5 | 0.43 / 0.58 | 3% | 96% |
| 10–15 | 651 | 757 | 11.2 | 0.66 / 0.82 | 6% | 97% |
| 15–20 | 662 | 722 | 11.0 | 0.74 / 0.87 | 5% | 97% |
| 20–25 | 686 | 745 | 11.6 | 0.35 / 0.75 | 4% | 92% |
| 25–30 | 696 | 758 | 11.5 | 0.41 / 0.56 | 2% | 96% |
| 30–35 | 667 | 733 | 11.2 | 0.60 / 0.70 | 9% | 97% |
| 35–40 | 645 | 688 | 11.0 | 0.72 / 0.85 | 3% | 97% |
| 40–45 | 638 | 643 | 10.8 | 0.68 / 0.82 | 1% | 93% |
| **0–45 mean** | **666** | **727 (+9%)** | | | | |

(45–50 is v4's benchmark ending at 48 minutes.) Whole 48-minute window: 704 vs 666 tok/s, decode-only 858 vs 776, KV p50/p95/max 0.57/0.80/0.87 (v3: 0.48/0.78/0.86), prefix hit 94.7% vs 95.4%, queueing 4.7% vs 1.1% of intervals. Decode-only by running requests: 804 / 847 / 876 / 895 at 9 / 10 / 11 / 12.

The pool peaked at 0.87 (~1.12M tokens, ~93K per stream: the games do not all peak at once), so 12 streams stay clear of the eviction squeeze that 14 hit. The gain is smaller than the healthy phases of 14 streams (+20–25%) and below the +12–18% estimate: 11.2 requests run on average, decode-only at 12 running was 895 here against 921–938 in v2/v3 (longer contexts throughout, or a different host), and the last 10 minutes narrowed to +1–7% as uncached prefill rose (0.55M in 40–45).

13 streams would need ~1.21M at this run's per-stream peak (0.94 of the pool): possible but at the edge.
