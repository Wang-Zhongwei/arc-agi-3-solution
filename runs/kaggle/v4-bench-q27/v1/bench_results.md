# Serving bench `q27`

Qwen3.8-27B FP8 throughput at the arms' 14 streams (arm/q27): DFlash2 with 8 and 4 draft tokens and the built-in MTP layer, on the q27 launcher. Fits mellamoa's last 0.69 GPU-h of the week.

Started 2026-10-15T20:35:49+00:00; last written 2026-10-15T20:51:45+00:00. Sampling: {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": true, "preserve_thinking": true}, "tool_choice": "auto", "send_seed": false}.

## Configs

| config | status | change | startup s | note |
|---|---|---|---:|---|
| Q8 | ok | CFG CTX=106496, CUDAGRAPH_BS_EXTRA=[3, 5, 6, 11, 12, 13], CUDAGRAPH_MAXBS=14, DFLASH_TOKENS=8, MAMBA_CACHE=84, MAXREQ=14, MEMFRAC=0.92, SPEC_ALGO=DFLASH | 240 |  |
| Q4 | dropped | CFG CTX=106496, CUDAGRAPH_BS_EXTRA=[3, 5, 6, 11, 12, 13], CUDAGRAPH_MAXBS=14, DFLASH_TOKENS=4, MAMBA_CACHE=84, MAXREQ=14, MEMFRAC=0.92, SPEC_ALGO=DFLASH | - | session budget |
| QM | ok | CFG CTX=106496, CUDAGRAPH_BS_EXTRA=[3, 5, 6, 11, 12, 13], CUDAGRAPH_MAXBS=14, MAMBA_CACHE=84, MAXREQ=14, MEMFRAC=0.92, SPEC_ALGO=NEXTN, SPEC_STEPS=3 | 78 |  |

## Decode by phase (gen throughput windows from serve.log; clean = full batch, no prefill)

| config | phase | conc | bs | clean/all win | clean median | clean p90 | max (any) | clean >=900 | accept | step ms | client tok/s | TTFT p50 s | stream tok/s p50 | stall | length |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q8 | q-clean14 | 14 | 14 | 14/73 | 1636 | 2350 | 2377 | 100% | 5.79 | 43.0 | 1360 | 1.51 | 112 | 18% | 100% |
| Q8 | q-mixed14 | 14 | 14 | 0/40 | - | - | 1227 | - | - | - | 670 | - | - | 39% | 5% |
| QM | q-clean14 | 14 | 14 | 58/85 | 1209 | 1433 | 1481 | 100% | 3.44 | 39.0 | 1072 | 0.36 | 80 | 14% | 100% |
| QM | q-mixed14 | 14 | 14 | 9/43 | 951 | 994 | 998 | 100% | 2.90 | 42.3 | 594 | - | - | 37% | 2% |

Per batch size, prefill and request numbers are in bench_results.json.
