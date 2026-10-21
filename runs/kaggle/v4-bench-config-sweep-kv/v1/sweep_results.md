# Config sweep `config-sweep-kv`

S0 against S3 with one prefix per concurrent copy (batch_prefix distinct), so bs10 and bs14 hold 10 and 14 separate 75.5k contexts in the KV pool, as real streams do, and report peak KV use, KV-full retractions and round-2 cache hits. config-sweep's identical copies share one cached prefix and cannot show this. Every copy prefills cold, so it needs a longer session.

Started 2026-10-08T18:56:03+00:00; last written 2026-10-08T19:26:55+00:00; budget 50 min, 1865 s used. Ran 2 of 2 configs.

Request: kaggle/fz-ledger-on-trim-r2/v1/m0r0-492f87ba_p0_requests.jsonl.xz line 66 (m0r0), recorded 75539 prompt tokens. Every request generates 1024 tokens (min_tokens 1024), sampling {"temperature": 0.7, "top_p": 0.95, "top_k": 20}.

Batch prefix: distinct. Each bsN copy starts its system prompt with its own tag, so every stream holds its own context in the KV pool, as real streams do. Round 1 prefills every copy cold, so its tok/s includes that prefill; round 2's cache hit shows how much of each stream's prefix the pool kept.

| config | status | change | launch | startup s | load s | verify graph s | KV pool tokens | prefill tok/s | TTFT s (first / cached) | bs1 tok/s (median / peak) | bs10 tok/s (median / peak) | bs10 window (median / max) | accept (bs1 / bs10) | bs10 KV use max / retracted | other batches | note |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| S0 | ok | baseline (no overrides) | 0 cold | 575 | 323 | 145 | 1011264 | 5605 | 1.11 / 1.22 | 190 / 199 | 329 / 530 | 505 / 1127 | 2.98 / 2.86 | 0.75 / 0 | - |  |
| S3 | failed | CFG CUDAGRAPH_MAXBS=14, MAMBA_CACHE=84, MAXREQ=14, MEMFRAC=0.98 | 1 | - | - | - | - | - | - / - | - / - | - / - | - / - | - / - | - / - | - | RuntimeError: server not healthy within 1133 s |

startup s: launch to healthy (launch 0 includes the one-time install and the cold JIT compile). load s / verify graph s: SGLang's own startup timings. prefill tok/s: cold prefix of the request, client-timed. TTFT first: the newest turn prefilled on top of the cached prefix. bs10 tok/s: total tokens of the round over its wall time; window: SGLang decode-log gen throughput. accept: mean accept length of the decode windows. KV use max: peak 'full token usage' of the decode windows; retracted: requests SGLang retracted because the KV pool was full. Details in sweep_results.json.
