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
