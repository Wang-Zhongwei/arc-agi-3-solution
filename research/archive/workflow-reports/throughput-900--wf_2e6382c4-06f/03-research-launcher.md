# throughput-900: research:launcher

Workflow run `wf_2e6382c4-06f`, phase Research. Find serving levers to reach 900+ tok/s peak with W4A16 Flash-Next, rank them, and build a Kaggle serving benchmark arm

---

# Launcher review: decode-throughput levers for W4A16 AutoRound Flash-Next

Nothing was edited and no local jobs were run. Sources:
- **L** = `.../wt/custom-serving/serving/kaggle/launch_sglang.py`
- **B** = `daniel-franzen-sol/serving/build_bundle_pennyroyal.sh`
- **SA/environ/…** = Pennyroyal d00d88ef source fetched with `gh api` into `scratchpad/r/python/sglang/srt/`
- **logs** = `arc-agi-3-v3/kaggle-outputs/*/serve.log`

## Facts from the logs and source

1. **Peak 900 is already reached in single windows.** There were 3,939 bs10 decode windows across 3 control logs: p50 796, p90 861, p99 898.6, max 943.9 tok/s.
   - Windows at or above 880 tok/s averaged accept len 2.89, against 2.68 overall.
   - Median bs10 tok/s by accept len: 2.4 → 755.6, 2.6 → 795.9, 2.8 → 831.1, 3.0 → 859.7.
   - Sustained 900 therefore needs accept len of about 3.1 or higher, or about 12% less step time (33.7 → 29.5 ms at accept 2.66).
   - Context length has no effect (correlation of tok/s with context 0.02).
2. **Derived step time** is bs×accept/tps, using medians over the non-experiment logs.
   - bs1 15.2 ms, bs2 18.1, bs4 22.8, bs5-7 28.1-28.6 (graph padded to 7), bs8 29.9, bs9 31.7, bs10 33.7.
   - That is roughly 13 ms fixed plus about 1.8 ms per added stream.
   - There is no jump at bs8→9, where M goes from 32 to 36. The bucket increments are +1.7, +1.8, +2.0 ms.
3. **MoE expert-union growth plausibly explains the per-stream cost.** This is my arithmetic, not a measurement.
   - With uncorrelated top-10-of-512 routing there are about 39 unique experts per layer at M=4 and about 280 at M=40.
   - At roughly 2.5 MB per INT4 expert-layer and 1.79 TB/s that is about +16 ms, against +18.5 ms observed from bs1 to bs10.
   - Supporting evidence: experts16 gave 625 vs 793 tok/s (`research-sweep.md:10`).
   - The log shows `GPTQMarlinMoEMethod`, so Marlin MoE is the biggest unmeasured share of the step.
4. **Intel's checkpoint quantizes only the routed experts** (config.json `extra_config`).
   - `linear_attn.*`, `self_attn.*`, `shared_expert.*`, `hyper_connection.*`, `mlp.gate`, `indexer`, `mtp` and `ple` are all bits=16.
   - `lm_head` is also unquantized (`lm_head_quantized: False` in the log), so the whole dense stack is BF16.
5. **Job-level throughput is lower than the window median.** `fz-arms-v4-control/summary.txt` shows 611.12 tok/s against 793.
   - The control log shows 10.68M new prefill tokens. At an assumed 12.5k tok/s that is about 854 s, or 11.6% of the 7,341 s span.
   - Per-bucket medians of `initial_prefill_elapsed` are 125 ms for under 512 new tokens, 216 ms for 512-2k, and 4,768 ms for over 30k.
6. **The harness does not stream.** `build_chat_payload` defaults to stream=False and `tool_agent.py:5339` does not pass it. Streaming is forced only for openrouter (`openai_compat.py:76-79`), so `--stream-interval` does nothing for us.
7. **Calibration against the other numbers.**
   - mratsim's 1170 tok/s at 8 streams (README) is NVFP4 + MXFP8 on a power-limited GPU with a +3000 MT/s memory overclock.
   - Gabriel's relaxed-acceptance C8 is 758 (`PERF_CEILING.md:40`). Our lossless bs8 median is 725.6, within about 4%.

## Options table

| Knob | Now | Proposed | Expected effect | Risk | Layer |
|---|---|---|---|---|---|
| Accept thresholds | 1.0/1.0 (L:32-33, 337-338) | 0.5/0.5, then 0.3/0.3 | Gabriel measured 179/203/231 tok/s at 1.0/0.5/0.3 (C1, T0.6; `scripts/serve.sh:73-75`). Expect accept +8-16%, so +5-10% tok/s (840-880 median) | Lossy at T>0 (SA:2241-2248). Harness T=0.7 (WRITEUP:60). Watch finish_reason=length | CFG |
| FP8 weight-only dense | `SGLANG_SM120_LOWM_FP8_WEIGHT=0`, `SM120_LM_HEAD_FP8=0` (L:275-276) | Apply Gabriel patch 0005, then set =1 | Neither env var is declared in d00d88ef `environ.py`, so they are silent no-ops today. GDN 4.17 + QSA 1.34 GB (`PERF_CEILING.md:62-64`) halved saves about 2.8 GB, about 1.5 ms of 33.7, so +4-5%. Gabriel saw C4 428→477 | About 3 GB extra VRAM against an 11.58 GB KV pool (usage p90 0.86, max 0.99). Needs the M cap raised | runtime patch |
| M>32 caps | `_MAX_M=32` (B:102), `_MAX_KERNEL_ROWS=32` (`sm120_online_fp8.py:20`) | Raise both to 64 | Verify and draft-extend run at M=bs×4, which is 36/40 at bs9/10. BF16 low-M alone shows no cliff (fact 2), so at most about 1%. In the online-FP8 lm_head, rows above 32 take a Python-loop dequant plus matmul (`sm120_online_fp8.py:331-361`), which confounds the −27% ONLINE_MXFP8 result. Pennyroyal reports +14.96% at C4 (`RESULTS.md:69`) | Untested at BLOCK_M=64 | runtime patch |
| Low-M tuning table | `_SM120_TUNED_TACTICS={}`, generic (split_k,16,128,4,4) with split-K when n_ctas×split_k<128 | Drop split-K as in 0005 ("zero+atomic+cast epilogue measured worse than split-free on every real shape"). Log shapes with `SGLANG_BF16_GEMM_LOG_SHAPES=1` (environ:1030), then fill M in {28,32,36,40} | Unknown, about 1-3% | Tuning noise | runtime patch + env |
| Marlin MoE | `block_size_m` loop picks 8 at M=40, since 40×10/512=0.78<0.9 (`fused_marlin_moe.py:240`). Small-batch configs {128,128,256},{64,128,128} (`moe_wna16_marlin.cuh:140-145`) | Profile bs10 verify first, then try thread configs and the blocks_per_sm cap of 4 (cuh:561) | Largest share of step (fact 3), unmeasured | Needs GPU time. JIT recompiles at launch | runtime patch (JIT .cuh) |
| Rejection sampling | off | `--speculative-use-rejection-sampling` (SA:2251) | Lossless Leviathan draft sampling, q=softmax(l/T) (`spec_utils.py:176-186`). May raise accept at T>0 | Raises ValueError unless the draft vocab equals the 248,320 target vocab, so FR-Spec must go (`eagle_worker_v2.py:228-240`). FR-Spec is worth +9% at C4 (541→592, `serve.sh:78-80`). Net unknown | args |
| FR-Spec | on, 64K map | Keep. Rebuild map from ARC transcripts for about +1% accept (`research-sweep.md:34`) | Small | None | model artifact |
| Spec steps/topk/draft tokens | 3 / 1 / 4 | Keep | QSA caps draft tokens at 4 (`PERF_CEILING.md:111`). Adaptive spec was −36% | n/a | n/a |
| Overlap scheduler | On: `disable_overlap_schedule False` in the args dump; `scheduler.py:437`; EAGLE always uses the V2 worker (`spec_info.py:287-299`); extra_buffer keeps overlap (`overrides.py:1909-1917`) | No change. Optional A/B `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` (environ:1181, `spec_utils.py:1125-1133`, default off) | At most 1-2% (guess) | Untested | env |
| CUDA graph bs list/max | [1,2,4,7,8,9,10], max 10 (L:301) | Add 3, 5, 6 | bs5-7 all cost 28.1-28.6 ms. bs5/6 are about 4% of windows, so about +0.2% overall | About 70 MB per bucket | args |
| Chunk / max prefill | 8192 / 16384 | Keep | No prefill batch exceeded 8,192 new tokens (2,941 batches), so MAX_PREFILL is inert. Mixed-chunk is rejected with spec (SA:10106) | n/a | n/a |
| `--num-continuous-decode-steps` | 1 | Skip | Declared at SA:1023. No consumer in `scheduler.py`, `schedule_policy.py`, `schedule_batch.py` or `batch_result_processor.py` at d00d88ef, so likely a no-op | n/a | n/a |
| Conservativeness / policy | 1.0 / lpm | Keep | No retraction events in the control log. Queue usually 0 | n/a | n/a |
| Page size / track interval | 64 / 64 | Keep | Page 64 is forced by compressed QSA (`overrides.py:1655`). The interval only coarsens prefix checkpoints (issue #3) | n/a | n/a |
| FlashInfer autotune | on | `AUTOTUNE=False` optional | No autotune line in serve.log besides the args dump. MoE runs Marlin, so this affects startup only | None | CFG |
| torch compile | off | Do not enable | Gabriel: fused ops raise NotImplementedError under dynamo (`PERF_CEILING.md:29`) | Capture fails | n/a |
| Stream interval / workers | 1 / 1 / 1 | No change | See fact 6. Non-stream flush every `SGLANG_FORCE_STREAM_INTERVAL=50` tokens (environ:587, `output_streamer.py:430-435`) is harmless | n/a | n/a |
| Logging / metrics | metrics, cache-report, request-time-stats; log interval 40 | Keep | Per-40-iterations or per-request only. Estimate under 0.5%. Accept len and gen throughput come from these logs | n/a | n/a |
| MM preprocessing | pil / cpu / transport cpu | No change | Affects TTFT/CPU, not decode. Knob: `--mm-preprocess-cache-size-mb` (SA:2947) | n/a | n/a |
| `OMP_NUM_THREADS` | 8 (L:272) | Try 1 (mratsim radixark script:92) | At most 1-2%, only if CPU is oversubscribed | Low | env |

## Patchable at runtime without a wheel rebuild
Paths are under `site-packages/sglang/`. Line numbers marked "base" are d00d88ef lines before Franzen's patch 01 shifts them.
- **`kernels/ops/gemm/sm120_lowm_bf16_gemm.py`** (created by patch 01, B:102).
  - Tunables: `_MAX_M`, `_MIN_WEIGHT_BYTES` (512 KiB), `_MAX_WEIGHT_BYTES` (128 MiB, which excludes the 1.27 GB lm_head), `_SM120_TUNED_TACTICS`, `_generic_tactic`, `use_sm120_lowm_bf16_gemm`.
  - Gabriel's 0005 (`competitors/sglang-flashnext-sm120/patches/0005-sm120-fp8-weight-only.patch`) is pure Python/Triton and targets this file. Its environ.py hunk needs a hand edit (`research-sweep.md:46`).
- **`srt/layers/quantization/unquant.py`.**
  - `bf16_gemm_dispatch` (base L273) is where the low-M branch is inserted.
  - Also `UnquantizedLinearMethod.apply` (L376), `initialize_bf16_gemm_config` (L159) and `_log_bf16_gemm_shape` (L152).
- **`srt/environ.py`** for new switches, and **`kernels/ops/gemm/sm120_online_fp8.py`** (L20, L331).
- **JIT CUDA with the bundled nvcc** (cache in `SGLANG_JIT_CACHE_DIR`; a cold start recompiles): `kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`, `kernels/ops/moe/moe_wna16_marlin.py`, `srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`.
- **Speculative decoding:** `srt/speculative/{eagle_worker_v2,spec_utils,eagle_info}.py`. **Scheduler output:** `srt/managers/scheduler_components/output_streamer.py`.
- **Mechanism:** the `SGLANG_RUNTIME_PATCHES` dict (L:199, empty today) takes `*.patch` applied with `patch -p1`, or `*.py` run as `python <file> <site-packages>` (L:194-220).
- **Needs a rebuild:** the compiled sgl-kernel, FlashInfer 0.6.17 cubins and jit-cache, and torch 2.13.0+cu130 (B:1074, 1545-1547).

## Suggested order
1. **Thresholds and env arms (20 min each).** Run 0.5/0.5 and 0.3/0.3 (CFG only). Run one combined arm for `OMP_NUM_THREADS=1`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` and graph bs 3/5/6. Run a separate arm for rejection sampling with `FRSPEC=False`.
2. **Low-M patch arm.** Set the M caps to 64 and drop split-K, with FP8 off. Compare the derived step times at bs9/10 from fact 2.
3. **FP8 weight-only arm.** If it passes, run the variant that frees the BF16 copies.
4. **One bs10 profile** (nsys or torch profiler) to size Marlin MoE before any `.cuh` work.
5. **Realistic target.** Thresholds (+5-10%) plus FP8 dense (+4-5%) would put the median around 880-930. 900+ sustained is plausible only with both, and the thresholds are lossy. Single-window peaks already reach 944.
