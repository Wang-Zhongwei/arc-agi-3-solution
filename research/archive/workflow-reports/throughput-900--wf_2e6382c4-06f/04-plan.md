# throughput-900: plan

Workflow run `wf_2e6382c4-06f`, phase Plan. Find serving levers to reach 900+ tok/s peak with W4A16 Flash-Next, rank them, and build a Kaggle serving benchmark arm

---

PLAN: 900+ tok/s peak aggregate decode, W4A16 AutoRound, 1x RTX Pro 6000. All numbers are from the notes; "derived" means arithmetic on those notes, not measured.

0. REFRAMING
- Nobody shows sustained 900 at 10 streams. 946 is dfranzen's own single-batch peak, with p90 833 and median 722 (TONG/COMPARISON.md:204-211). Our p90 837 matches it.
- Other SGLang W4A16 results used more slots. sirikilohit reached peak 1,159 / p90 1,039 / median 899 at 16 slots, and lordhansolo (vLLM) 1,135 / 1,056 / 976 at 14 slots. That fits our own 14-stream +15%.
- We already touch 900 in single windows. Across 3,939 bs10 windows in 3 control logs: p50 796, p90 861, p99 898.6, max 943.9.
- Only 16 of 1,161 clean bs10 windows (1.4%) in fz-arms-v4-control reach 900. The real goal is to make 900+ windows common.
- Calibration: fz-experts8 cut the step 8% (32.2 to 29.6 ms) at accept 2.60. It got a clean median of 872 and 17% of windows at 900+.
- Today at clean bs10, accept is 2.66 and step is 32.2 ms, so 10*accept/step = 826 clean. The 789-793 medians include prefill stalls.
- Sustained 900 clean needs (accept, step ms) of (2.66, 29.6), (2.80, 31.1), (2.90, 32.2) or (3.00, 33.3). Clean accept p90 is 2.84 and max is 3.04, so sustained 3.0 needs a lever that changes acceptance.
- Derived, uniform-routing upper bound: expert bytes set the slope (~34 GB/step at M=40, ~19 ms, ~1.4-1.8 ms/stream) and the BF16 dense stack is the intercept (8.71 GB, ~4.9 ms). Intel's config.json quantizes only the routed experts.
- Dense levers therefore cap at about 4 ms of 32.3 (~12%).
- Corrections: the low-M GEMM issue is #12 (#11 is trim re-prefill). `SGLANG_SM120_LOWM_FP8_WEIGHT` and `SGLANG_SM120_LM_HEAD_FP8` (launch_sglang.py:275-276) are inert. They exist only in Gabriel's 0005/0006.
- OPEN CONFLICT: WRITEUP.md:60 says harness T=0.7. serve.log:111 shows server default T=1.0 and requests carry no temperature field. Resolve this before the threshold test. Relaxed accept is a no-op at T=0.

1. RANKED LEVERS

L1. Relaxed MTP accept, 0.5/0.5 then 0.3/0.3
- Gain: Gabriel C1 at T0.6 is 179 / 203 / 231 tok/s at thresholds 1.0 / 0.5 / 0.3 (GAB/scripts/serve.sh:75; docs/STATUS.md:52). C4 is 549 greedy vs 620 relaxed (results_8h_final.json). Accept len is 2.08-2.22 greedy vs 2.4-3.0 relaxed.
- Our base accept is already 2.66, so expect less: +5-10% (launcher estimate, 840-880 median), or +13% if Gabriel's gain transfers (forks, accept to ~2.9). Each +0.1 accept is about +31 tok/s clean, about +24 incl. stalls.
- It is the only lever here that can reach sustained 900 alone.
- Transfer is unproven. Gabriel used NVFP4 plus a BF16 draft, one C++ prompt and short context. Ours is an INT4-g32 draft.
- Quality: CHANGES OUTPUTS at T>0 (lossy, "sharpens sampling", serve.sh:73-74). The kernel accepts if `coin <= p/threshold_acc || p >= threshold_single` (speculative_sampling.cuh:80). It is an exact no-op at T=0. It needs a score A/B at the harness temperature. Watch finish_reason=length.
- Effort: minutes. Layer: CFG.
- Change: `SPEC_ACCEPT_SINGLE=0.5, SPEC_ACCEPT_ACC=0.5` (launch_sglang.py:32-33, passed at :337-338), then 0.3/0.3. Log accept len.

L2. Raise `_MAX_M` 32 to 64 (issue #12)
- Why: verify and draft-extend run at M=4*bs, which is 36/40 at bs 9/10. That is above `_MAX_M = 32` (GAB/patches/0004-sm120-lowm-triton-gemm.patch:60; `_SM120_TUNED_TACTICS = {}` at :55; both embedded at CS/serving/build_bundle_pennyroyal.sh:102). Those shapes fall to cuBLAS, which runs at 60-75% of DRAM BW in-graph (0004 patch:7-8).
- Gain: unmeasured at M=36/40.
  - 0004 at M=4 took the verify graph from 14.43 to 13.79 ms and C1 greedy from 142.8 to 147.9.
  - Logs show no step jump at M 32->36 (29.3, 30.7, 32.2 ms), so the gain is probably small. Issue #12 guesses 0-5%.
  - Saving 0.5-1.0 ms would be +13 to +26 tok/s clean (+1.6-3.2%).
  - lordhansolo's vLLM overlay extended its SM120 GEMM to M=40..64 (PATCH_README.md:21-27). Its source tar is not in the local copy.
- Quality: no model change. It is the same BF16 math with a different kernel and accumulation order, so not bit-identical. Effort: one line. Layer: runtime patch.
- Change: a `.py` entry in `SGLANG_RUNTIME_PATCHES` (launch_sglang.py:199) setting `_MAX_M=64` in site-packages/sglang/kernels/ops/gemm/sm120_lowm_bf16_gemm.py.
  - BLOCK_M becomes 64 (0004 patch:149). The x-tile re-read may go L2-bound. If slower, widen BLOCK_N.
  - Optionally drop split-K (0005 measured it worse than split-free on every real shape) and fill `_SM120_TUNED_TACTICS` for M in {28,32,36,40}.
  - Confirm shapes with `SGLANG_BF16_GEMM_LOG_SHAPES=1` (environ.py:1030).

L3. Port Gabriel 0005 + 0006 (W8A16 FP8 dense stack, FP8 lm_head); requires L2
- Gain: 0005 takes C1 greedy from 147.9 to 163.5 (+10.5%) and C4 from 428 to 477 (0005 patch:13-15). 0006 takes C1 from 163.5 to 188.8 (best 193.3) and C4 from 477 to 529 (0006 patch:6-13).
- Derived at bs10: byte-bound, so about -3.9 ms of 32.3 (-12%), taking 789 to about 895. The honest range is +5-14% (forks). The launcher estimates +4-5% for 0005 alone.
- Caveats:
  - The HC fused kernel exists only for rows<=16 (hc_mix_triton.py:33,208), so the HC gain is absent at bs>=5.
  - The fp8 lm_head returns None above `_MAX_M` (0006 patch:68). The lm_head part also halves every MTP draft step's logits.
  - It costs about 3.3-3.6 GB VRAM (PERF_CEILING.md:27; STATUS.md:45-47). The KV pool is 11.58 GB and peaks at 0.99 (1,000,768 of 1,011,264 tokens). dfranzen had 4.25 GB free (TONG/SUMMARY.md:18).
- Quality: BF16 weights Intel left in 16-bit become per-row e4m3, which is a small logit change. The notes contain no quality eval, so run a score A/B. Gabriel's numbers are throughput only.
- Effort: medium. The environ.py and unquant.py hunks need re-anchoring, since JPE added an online-FP8 block at unquant.py:221. Layer: runtime patch (pure Python/Triton, same file as L2).
- Change: add the (1,32,128) fp8 Triton kernel (89% of fp8 floor) and cache to sm120_lowm_bf16_gemm.py. Declare the env vars. Then set `SGLANG_SM120_LOWM_FP8_WEIGHT=1` and `SGLANG_SM120_LM_HEAD_FP8=1`.
- Cheap first step: lm_head only (+0.64 GB copy). Once it passes, try freeing the BF16 copies at load.

L4. Profile bs10, then tune Marlin MoE (largest unmeasured share)
- Gain: unknown. Experts set the slope, and experts16 gave 625 vs 793 tok/s (research-sweep.md:10). No fork tunes Marlin-INT4 MoE, since all are NVFP4.
- Knobs: `block_size_m` loop picks 8 at M=40 (40*10/512=0.78<0.9, fused_marlin_moe.py:240). Small-batch thread configs are {128,128,256},{64,128,128} (moe_wna16_marlin.cuh:140-145). The blocks_per_sm cap is 4 (cuh:561).
- Quality: none. Effort: high, and it needs GPU time (JIT recompile at launch). Layer: runtime patch of the JIT .cuh (bundled nvcc, no wheel rebuild).
- Change: take an nsys or SGLang-profile trace at bs10 in the baseline config. The draft vs verify split is not recoverable from logs. Then try block_size_m, thread-config and blocks_per_sm variants.

L5. Repair online-FP8 fallbacks, then re-test jpezzulli online FP8 (only if L3 stalls)
- Gain: JPE C1 161.465 to 207.124 (+28.28%) and 128K C4 367.127 to 422.037 (+14.96%) (JPE/RESULTS.md:53-54,67-70). Ours was -27%.
- Hypothesis, unproven: MRA designed it for "bs <= 8, verify <= 16" (0003 patch:38). Above the cap, HC dequantizes per call at rows>16 (hyperconnection.py:296-312, ~125 MB x ~100 calls). lm_head dequantizes in chunks at rows>32 (sm120_online_fp8.py:20,331,352, ~13 GB). That is about 25 GB of temp traffic, about 14 ms, the same size as -27%.
- Quality: FP8 weights change outputs. Effort: high. Layer: runtime patch.
- Change: `_MAX_KERNEL_ROWS` 32 to 64. `_FUSED_MIX_MAX_ROWS` 16 to 64 with `rows_pad=next_pow2(rows)`; ROWS is constexpr, so run a numeric test. Skip `convert_eligible_linears_to_mxfp8` (qwen4_exp.py:1443) to get W8A16 only. First reproduce +15-28% at bs<=4.

L6. FR-Spec map from our own transcripts (data only)
- Gain: about +1% accept (research-sweep.md:34), roughly +8 tok/s. lordhansolo's map is "23853 ids generated in 125 Flash-Next Kaggle transcripts", with median accepted draft 2.9 vs 2.7 and 2.6 for the others (COMPARISON.md:211). That is confounded by vLLM and the NVFP4 head.
- The draft head is about 0.98 GB/step, so shrinking saves at most 0.3 ms. The value is coverage.
- Quality: none under exact acceptance. Effort: low-medium. Layer: model artifact.
- Change: measure coverage of archived outputs under the current map (sha becfa41d..., launch_sglang.py:39). Rebuild with GAB/scripts/make_hot_tokens.py and update the sha asserts (launch_sglang.py:39,342).

L7. Sampling temperature 1.0 to 0.6 (only if the harness really runs T=1.0)
- Gain: fz-arms-v5-temp06 gave accept 2.698 vs 2.661 and a clean median of 837 vs 826 (+1.4%). The 0.6 is inferred from the arm name.
- Quality: changes outputs. Layer: harness request parameter. Low priority.

L8. Free env/args bundle (each at most 1-2%, not resolvable in a 4-minute window)
- `OMP_NUM_THREADS` 8 to 1 (launch_sglang.py:272; mratsim radixark script:92).
- `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` (environ.py:1181, default off; at most 1-2%, a guess).
- CUDA graphs at bs 1..10 (launch_sglang.py:301). bs5/6 pad to 7 at 28.1-28.3 ms, so this is at most +0.3% overall and about 70 MB per bucket.
- Prior is low: decode is GPU-bound at about 99% busy (PERF_CEILING.md:9). Quality: none.

L9. Harness wall-clock levers (these do not move peak decode)
- The run-level rate was 613 tok/s against 826 clean. Prefill stall is 21.4% of wall (1,604 of 7,496 s).
- Cost model: fixed cost 0.279 s x 2,941 batches is about 820 s (10.9%). Token cost is 68 us/token (14.6K tok/s), about 730 s (9.7%).
- Prefill is serial, 97.8% of batches are single-sequence, none uses a CUDA graph, and decode is frozen between chunks.
- Headroom (log table): halve the fixed cost for about +36 tok/s. Remove cache-miss prefill (4.89M tokens, 45.8% of prefill tokens; issues #3/#10/#11/#22) for about +29. Prefill 1.3x faster for about +14. Perfect bs10 occupancy for +46 clean, mostly client-side.
- Even a 900 clean median is only about 706-710 incl. stalls.
- Untested: `--prefill-decode-interval`, and a larger `--chunked-prefill-size` with NEXTN and extra_buffer Mamba (issue #10). Mixed chunk is rejected with spec (server_args.py:10103-10106).
- The extra 150-200 ms per small prefill batch needs a profiler.
- MRA 0005 ghost-abort fix: grep serve.log for aborts before porting.

Projection (derived)
- L2+L3 (lossless-ish) gives +5-14%, or 828-900 on the 789 base. That is about the experts8 calibration (clean 872, 17% of windows at 900+).
- L1 on top adds +5-13%, so about 900-1,000 (multiplicative, optimistic).
- Not worth time: GDN recovery and overlap scheduling are already on. Lordhansolo's graphs 44/52 are not relevant (we capture 1,2,4,7,8,9,10). MRA's 800@4 / 1170@8 is NVFP4 + MXFP8 with a +3000 MT/s memory overclock. JPE commits after d00d88ef are docs and scripts only. torch compile fails (PERF_CEILING.md:29). `--num-continuous-decode-steps` is likely a no-op. `--stream-interval` is inert because the harness uses stream=False.
- Rejection sampling needs FRSPEC off, and FR-Spec is worth +9% at C4 (541 to 592), so net is unknown and likely negative. Not scheduled.

2. CONFIGS FOR ONE KAGGLE SESSION (overrides relative to Franzen's; ~10 min startup + 8 min bench each)
- C0 baseline `{}`. Run bs10, plus bs1/bs4 (2 min) and an nsys/profile trace at bs10 (5 min). About 25 min.
- C1 `{CFG: SPEC_ACCEPT_SINGLE=0.5, SPEC_ACCEPT_ACC=0.5}`. 18 min.
- C2 `{CFG: SPEC_ACCEPT_SINGLE=0.3, SPEC_ACCEPT_ACC=0.3}`. 18 min.
- C3 `{SGLANG_RUNTIME_PATCHES: {m64.py: _MAX_M=64 (+ drop split-K)}, env: SGLANG_BF16_GEMM_LOG_SHAPES=1}`. 20 min.
  - Expect bs<=8 step unchanged vs C0, since M<=32 already uses the kernel. This also checks that shape logging is free. Any change shows at bs9/10.
- C4 `{runtime patches: m64 + FP8 port, env: SGLANG_SM120_LOWM_FP8_WEIGHT=1, SGLANG_SM120_LM_HEAD_FP8=1}`. 20 min.
  - Port must be ready beforehand. Compare relative deltas at bs1/bs4 against Gabriel's (+10.5% for 0005), watch KV usage max 0.99, and check for retractions (control had none).
- C5 best lossless of C3/C4, plus the best of C1/C2 (whichever has an acceptable accept len). 18 min.
- C6 C5 plus `{env: OMP_NUM_THREADS=1, SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1; args: CUDA graph bs 1..10}`. Effect is C6 vs C5. 18 min.
- Total is about 137 min, leaving about 13 min slack. If behind, drop C6, then C2. Scoring happens outside the session, as a separate harness A/B for C1/C2, C4 and L7.

3. BENCHMARK LOAD (identical for every config)
- Closed-loop clients, 10 slots, non-streaming (harness stream=False). 77% of wall is at bs>=9 and mean bs is 9.03. Also run short bs 1/4/8 sweeps on C0/C3/C4.
- Preferred: replay archived request jsonl (fz-arms-v4-control, 2311 requests) at concurrency 10, keeping per-session turn order. The keys are messages, tools, tool_choice and chat_template_kwargs. Use the same order every config.
  - This gives the true prefix reuse (93.6% cache hit) and the real prompt/output mix.
  - Set temperature explicitly to the harness's real value (see the open conflict above).
  - Take output lengths from the archived ReqTimeStats. The notes do not give them.
- If synthetic: each turn adds a mostly small suffix (p50 initial prefill 125 ms under 512 new tokens, 216 ms for 512-2k). About 25% of prefill batches are cold (728 of 2,941, `#cached-token: 0`), with some prompts over 30K (4,768 ms). Chunks are at most 8,192 and prompts run up to ~100K per stream (1,011,264-token pool). Context length does not affect tok/s (corr 0.02).
- Phases per config:
  - 1 min warmup (primes prefix cache and graphs).
  - 4 min "decode-clean" at bs10 with prompts pre-cached and outputs of at least 1,000 tokens (my choice). That gives about 185 clean 40-step windows (1.3 s each) and a median SE of about 0.5-1%.
  - 3 min harness replay for stall share and whole-phase tok/s.
- Metrics, parsed from `Decode batch` and `Prefill batch` lines (scratchpad/an/parse.py):
  - Clean-window gen throughput median / p90 / max, and the fraction of windows at 900+ (baseline 1.4%).
  - Accept len, and step ms = 40*bs*accept/tps at bs 7-10.
  - Whole-phase tok/s, prefill stall share, finish_reason=length rate, KV usage max, `#queue-req`.
- Decision rules: keep a lossless lever if clean bs10 median is up 2% or more. Adopt L1 only if the score A/B at the harness temperature is non-inferior. Effects under 2% (L2, L6, L8) will not resolve in-session, so use step ms at fixed bs.
