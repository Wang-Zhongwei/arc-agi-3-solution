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

## v2: 2,048-token prefill chunk

[v2.json](v2.json) is v1 with `CHUNK` and `MAX_PREFILL` 2048 (from 8192 and 16384), the setting the tuneup runs used to hold 0.967 at 14 streams with 1.26 GB free. Everything else, including `MEMFRAC` 0.968, is unchanged. Smaller prefill forwards may cost prefill speed; compare decode-only intervals by running requests (concurrency effect) separately from prefill intervals (chunk effect).
