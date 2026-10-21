# Config sweep `config-sweep`

Launch configs on one real mid-game request in one 25-minute session, most informative first; the budget cuts the tail (config-sweep-b runs it).

Started 2026-10-08T18:35:42+00:00; last written 2026-10-08T18:58:24+00:00; budget 25 min, 1376 s used. Ran 2 of 7 configs.

Request: kaggle/fz-ledger-on-trim-r2/v1/m0r0-492f87ba_p0_requests.jsonl.xz line 66 (m0r0), recorded 75539 prompt tokens. Every request generates 1024 tokens (min_tokens 1024), sampling {"temperature": 0.7, "top_p": 0.95, "top_k": 20}.

Batch prefix: shared. The bsN copies are identical and reuse the one prefix cached by bs1, so the KV pool holds about one context plus N x the generated tokens: these rounds show no KV-pool pressure (evictions, retractions), which real streams with distinct contexts meet. Use batch_prefix "distinct" (bench config-sweep-kv) for that.

| config | status | change | launch | startup s | load s | verify graph s | KV pool tokens | prefill tok/s | TTFT s (first / cached) | bs1 tok/s (median / peak) | bs10 tok/s (median / peak) | bs10 window (median / max) | accept (bs1 / bs10) | bs10 KV use max / retracted | other batches | note |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| S0 | ok | baseline (no overrides) | 0 cold | 613 | 364 | 145 | 1011264 | 5578 | 1.12 / 1.22 | 202 / 209 | 542 / 556 | 698 / 1278 | 3.13 / 3.08 | 0.08 / 0 | - |  |
| S1 | failed | CFG SPEC_ACCEPT_ACC=0.5, SPEC_ACCEPT_SINGLE=0.5 | 1 | - | - | - | - | - | - / - | - / - | - / - | - / - | - / - | - / - | - | RuntimeError: server not healthy within 667 s |
| S2 | skipped | patches m64.py, nosplitk.py | - | - | - | - | - | - | - / - | - / - | - / - | - / - | - / - | - / - | - | budget: needs ~489 s, 124 s left |
| S3 | skipped | CFG CUDAGRAPH_MAXBS=14, MAMBA_CACHE=84, MAXREQ=14, MEMFRAC=0.98 | - | - | - | - | - | - | - / - | - / - | - / - | - / - | - / - | - / - | - | budget: needs ~540 s, 124 s left |
| S4 | skipped | CFG SPEC_STEPS=2 | - | - | - | - | - | - | - / - | - / - | - / - | - / - | - / - | - / - | - | budget: needs ~579 s, 124 s left |
| S5 | skipped | CFG CHUNK=16384, MAX_PREFILL=32768 | - | - | - | - | - | - | - / - | - / - | - / - | - / - | - / - | - / - | - | budget: needs ~459 s, 124 s left |
| S6 | skipped | CFG FRSPEC=False | - | - | - | - | - | - | - / - | - / - | - / - | - / - | - / - | - / - | - | budget: needs ~459 s, 124 s left |

startup s: launch to healthy (launch 0 includes the one-time install and the cold JIT compile). load s / verify graph s: SGLang's own startup timings. prefill tok/s: cold prefix of the request, client-timed. TTFT first: the newest turn prefilled on top of the cached prefix. bs10 tok/s: total tokens of the round over its wall time; window: SGLang decode-log gen throughput. accept: mean accept length of the decode windows. KV use max: peak 'full token usage' of the decode windows; retracted: requests SGLang retracted because the KV pool was full. Details in sweep_results.json.
