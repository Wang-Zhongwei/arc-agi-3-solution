# throughput-900: research:forks

Workflow run `wf_2e6382c4-06f`, phase Research. Find serving levers to reach 900+ tok/s peak with W4A16 Flash-Next, rank them, and build a Kaggle serving benchmark arm

---

# Fork-mined throughput levers for W4A16 on SGLang 2.5.3 (read-only)

Path shorthand, all under `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/`:
- GAB = `sglang-flashnext-sm120`
- MRA = `sglang-qwen38fn-sm120-turbo`
- JPE = `sglang-rtxpro6000`
- TONG = `tong-m2-comparison/kaggle`
- CS = `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/custom-serving`
- `launch_sglang.py` = CS/serving/kaggle/launch_sglang.py
- `WRITEUP.md` = CS/WRITEUP.md

## Corrections to the brief (all verified)
1. **946 tok/s is dfranzen's own peak, not lordhansolo's.** TONG/COMPARISON.md:204-211 lists peak/p90/median:
   - dfranzen 946/833/722 (SGLang, 10 slots, single-batch readings).
   - lordhansolo 1,135/1,056/976 (vLLM, 14 slots, 10 s averages).
   - sirikilohit 1,159/1,039/899 (SGLang v2.5.0 W4A16 AutoRound, 16 slots, 69,632 ctx).
   - Nobody shows sustained 900+ at 10 streams, and our P90 837 is about Franzen's p90 833.
   - "Peak 900+" is a tail reading. A sustained 900 median is +14% over 789 (about 90 vs 79 tok/s per stream).
2. **The low-M GEMM issue is #12, not #11** (adsdemaybe/daniel-franzen-sol; #11 is trim re-prefill). Its facts hold:
   - `_MAX_M = 32` (GAB/patches/0004-sm120-lowm-triton-gemm.patch:60) and `_SM120_TUNED_TACTICS = {}` (:55).
   - Both are embedded at CS/serving/build_bundle_pennyroyal.sh:102.
   - Verify M = 4 x bs = 36/40 at bs 9/10.
3. **`SGLANG_SM120_LOWM_FP8_WEIGHT` and `SGLANG_SM120_LM_HEAD_FP8` (set "0" at launch_sglang.py:276) are inert.**
   - They exist only in Gabriel's 0005/0006.
   - grep finds them in neither JPE nor Franzen's embedded patch (0004 only).
   - Setting them to 1 does nothing, so W8A16 must be ported.

## Why Gabriel's dense-stack levers carry over, and their ceiling
- **Which weights are INT4.** Intel's config.json (https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/raw/main/config.json; copy at CS/../intel_cfg.json) has bits 4, group_size 128 and `block_name_to_quantize` = `model.language_model.layers`.
  - `extra_config` keeps these 16-bit: linear_attn.*, self_attn.*, shared_expert.*, hyper_connection, mlp.gate, indexer, ple, mtp, embed_tokens.
  - Only routed experts are INT4, so the BF16 dense stack equals the NVFP4 build's 8.71 GB/step (GAB/docs/PERF_CEILING.md:64-68).
- **Derived (mine, uniform routing, upper bound).**
  - 512 experts, top-10, 2560x640x3 params per expert, INT4 g128 at about 2.53 MB/expert.
  - M=40 touches about 279 experts/layer x 48 layers, about 34 GB/step (about 19 ms at 1.79 TB/s), vs 8.7 GB dense (about 4.9 ms).
  - Check: bs9->10 adds about 2.3 GB, about 1.3 ms. Issue #7 measured step 27.7/29.2/30.7/32.3 ms at bs 7-10, about 1.5 ms per stream.
  - Experts set the slope and the dense stack sets the intercept, so dense levers are worth at most about 4 ms of 32.3 (about 12%).
  - No fork patch I read tunes Marlin-INT4 MoE. All are NVFP4, and MRA 0023 is a vision/Marlin-bias fix.
- **What 900 needs.** 900 = 10 x accept / step. At step 32.3 ms, accept 2.55 (= 79 x 0.0323, derived) must reach about 2.9, or the step must drop to about 28.3 ms at accept 2.55.

## Levers, ranked by value/effort (all quant-independent unless noted)
**1. Relaxed MTP accept (thresholds 1.0 -> 0.5 -> 0.3).**
- The kernel accepts a draft token if `coin <= p/threshold_acc || p >= threshold_single` (JPE/python/sglang/kernels/aot/csrc/speculative/speculative_sampling.cuh:80). It is an exact-match no-op at temp 0 (GAB/results/results_8h_final.json).
- Gain: Gabriel's C1 at temp 0.6 gives 1.0=179, 0.5=203, 0.3=231 tok/s (GAB/scripts/serve.sh:75; docs/STATUS.md:52).
  - Accept len is 2.08-2.22 greedy vs 2.4-3.0 relaxed. C4 is 549 greedy vs 620 (results_8h_final.json).
  - Conditions: NVFP4 + BF16 draft, one C++ coding prompt, top_p .95/top_k 20 (scripts/bench_sglang.py:14,29), short ctx. Our INT4-g32 draft may differ.
- Apply: CFG `SPEC_ACCEPT_SINGLE/ACC` (launch_sglang.py:32-33, passed :337-338), no rebuild.
  - It is lossy at temp>0 and "sharpens sampling" (serve.sh:73-74). Our sampling is temp 0.7/top-p .95/top-k 20 (WRITEUP.md:60), so a score A/B is required.
  - Sweep 0.5/0.5 then 0.3/0.3 and log accept len.

**2. Raise `_MAX_M` 32 -> 48/64** so bs 9-10 verify and draft-extend (M=36/40) use the Triton kernel, not cuBLAS.
- Gain: 0004 at M=4 took the verify graph from 14.43 to 13.79 ms and C1 greedy from 142.8 to 147.9 (0004 patch:7-12).
  - cuBLAS-in-graph runs at 60-75% of DRAM BW (0004 patch:7-8). PERF_CEILING.md:19-20 says 30-50% on mid shapes.
  - M=36/40 is unmeasured. Issue #12 guesses 0-5% and notes no step jump across M=32->36.
  - lordhansolo's vLLM overlay extended its custom SM120 GEMM to M=40/44/48/52/56/64 (TONG/lordhansolo/dataset-vllm-main-e975732-arc3/PATCH_README.md:21-27, VALIDATION.md). The source tar is omitted from the local copy.
- Apply: runtime .py patch editing the constant in site-packages/sglang/kernels/ops/gemm/sm120_lowm_bf16_gemm.py. Confirm shapes with `SGLANG_BF16_GEMM_LOG_SHAPES=1` (JPE/python/sglang/srt/environ.py:1030).
  - BLOCK_M becomes 64 (0004 patch:149). The x-tile re-read per 16-wide N tile may go L2-bound. If slower, widen BLOCK_N.

**3. Port 0005: W8A16 FP8 weight-only dense stack** (per-row e4m3 copies of every bf16 weight >=4 MB; (1,32,128) Triton kernel at 89% of fp8 floor).
- Gain: C1 greedy 147.9 -> 163.5 (+10.5%), C4 428 -> 477 (0005 patch:13-15).
  - It costs about 3.3-3.6 GB VRAM (PERF_CEILING.md:27; STATUS.md:45-47).
  - dfranzen has 4.25 GB free (TONG/SUMMARY.md:18) and KV peaks at 0.99 of the pool (issue #7), so it trades KV.
- Apply: runtime patch adding the fp8 kernel and cache to the embedded sm120_lowm_bf16_gemm.py (same base file as 0004).
  - The environ.py/unquant.py hunks need re-anchoring (JPE added an online-FP8 block at unquant.py:221).
  - Useless without #2, because it is gated by `_MAX_M`.

**4. Port 0006: FP8 HC mix + lm_head** (also halves every MTP draft step's logits).
- Gain: C1 greedy 163.5 -> 188.8 (best 193.3), C4 477 -> 529; HC 14.5 -> 10.8 us/mix (0006 patch:6-13).
- Caveats:
  - The fp8 lm_head returns None above `_MAX_M` (0006 patch:68).
  - The HC fused kernel exists only for rows<=16 (JPE/python/sglang/srt/layers/hc_mix_triton.py:33,208). Above that a 5-kernel chain runs (:3-6), so the HC part is absent at bs>=5.
- Apply: as #3. lm_head-only is the cheap variant (+0.64 GB copy).

**5. Fix the M>16 / M>32 dequant fallbacks, then re-test jpezzulli online FP8 (our -27%).**
- Gain: JPE C1 161.465 -> 207.124 (+28.28%); 128K C4 367.127 -> 422.037 (+14.96%) (JPE/RESULTS.md:53-54,67-70).
  - MRA designed it for "bs <= 8, verify <= 16" (MRA/patches/0003-sm120-online-fp8.patch:38).
  - Above the cap, HC dequantizes both mix matrices per call at rows>16 (JPE/python/sglang/srt/layers/hyperconnection.py:296-312; about 125 MB temp x about 100 calls).
  - Above the cap, lm_head dequantizes 248K x 2560 in chunks at rows>32 (JPE/python/sglang/kernels/ops/gemm/sm120_online_fp8.py:20,331,352; about 13 GB).
  - My arithmetic: about 25 GB of temp traffic, about 14 ms (L2 reuse ignored), the same size as the -27%. This is a hypothesis, unproven.
- Apply: runtime patch.
  - `_MAX_KERNEL_ROWS` 32->64, and `_FUSED_MIX_MAX_ROWS` 16->64 with `rows_pad=next_pow2(rows)`. ROWS is constexpr, so this needs a numeric test.
  - Skip `convert_eligible_linears_to_mxfp8` (qwen4_exp.py:1443) to get W8A16 only.
  - First reproduce +15-28% at bs<=4. Load-time replace frees about 1 GB instead of costing it.

**6. Own-transcript FR-Spec map** (ours is the Pennyroyal-bundle map, sha becfa41d..., launch_sglang.py:39; Gabriel's is f03551e4...).
- Gain: lordhansolo's configs/draft_vocab_32k.json is "23853 ids generated in 125 Flash-Next Kaggle transcripts".
  - Median accepted draft length is 2.9 vs dfranzen 2.7 and sirikilohit 2.6 (COMPARISON.md:211). This is confounded by vLLM and the NVFP4 head.
  - Gabriel's FR-Spec 64K gave C1 200.8 -> 219.4 and C4 541 -> 592 (serve.sh:80-81), which we already have.
  - The draft head is 3 x 64K x 2560 x 2 B, about 0.98 GB/step (about 0.5 ms). Shrinking saves at most 0.3 ms, so the value is coverage -> accept.
- Apply: data only. Measure coverage of our archived outputs under the current map, rebuild with GAB/scripts/make_hot_tokens.py, and update the sha asserts (launch_sglang.py:39,342).

**7. MRA 0005 abort/"ghost" fix** (cancelled client requests keep decoding and hold a running slot).
- Not quantified. It matters only if the harness cancels in flight, so check serve.log for aborts.
- Apply: runtime patch of scheduler.py + tokenizer_manager.py (165 lines, base 4ccff141, needs re-anchoring). Low priority.

## Already present, dead ends, not applicable
- **GDN recovery and scheduling.**
  - RecoverSSM with side-stream recovery (MRA 0002, GAB 0001b) is already in our wheel (JPE/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:993-1007; `GDN_MTP_CACHE_MODE=none`).
  - Overlap scheduling is always on with spec (JPE/python/sglang/srt/arg_groups/speculative_hook.py:119-125).
  - Gabriel found decode "GPU-bound at ~99% busy" (PERF_CEILING.md:9), so host overhead is not the lever.
- **vLLM mixed prefill+decode** (`--max-num-batched-tokens 2048`): SGLang asserts no mixed chunk with spec (JPE/python/sglang/srt/server_args.py:10103-10106).
- **lordhansolo's exact graphs 44/52** (PATCH_README.md:8-11) fix a vLLM capture-ladder gap (lordhansolo/dataset-taaf-kaggle-source/src/ARC3-Inference/inference/framework/kaggle.py:1306-1308). We capture 1,2,4,7,8,9,10 (launch_sglang.py:301), so only bs 3/5/6 pad. Negligible.
- **MRA's 800 @4 / 1170 @8** (MRA/README.md:39,46) is NVFP4 + online MXFP8 with default thresholds (launcher :112) and memory +3000 MT/s at 360 W (README.md:29-30).
  - It is not reproducible on Kaggle, and the QAD checkpoint is NVFP4-only.
  - Its 0006 sampler all-reduce skip only matters with grammars.
- **JPE after d00d88ef** has 4 commits (92f57171, 468e63a9, 12846e83, f9c4a89c): docs, setup and traffic scripts only (`gh api compare`). Nothing to adopt.

## Suggested A/B order and projection (derived, not measured)
1. #2 alone, with the shape log (one-line patch).
2. #1 at 0.5/0.5, then 0.3/0.3 (CFG only; accept len + score).
3. #3+#4 port, lm_head first, watching KV.
4. #5 only if step 3 stalls.

- **Dense-stack savings.** At C1, step = 2.2/tok/s (accept about 2.2 per patch text). That gives 14.87 -> 13.46 ms (0005, -1.42) -> 11.65 ms (0006, -1.80), plus 0004's -0.64 (measured).
  - These are byte-bound, so roughly M-independent: about -3.9 ms of 32.3 (-12%), taking 789 to about 895 at bs10.
  - This assumes cuBLAS at M=40 is as slow as at M=4, which is unmeasured. The honest range is +5-14%.
- **Thresholds.** #1 alone reaches 900 if Gabriel's +13% (0.5) transfers (accept 2.55 -> about 2.9). Both together give about 1,000 (multiplicative, optimistic).
- **Not verified.** lordhansolo's `arc3_sm120_gemm.py` is not in the local copy. Nothing was run on GPU. The effect of relaxed accept on ARC score and on our INT4-g32 draft is unknown.
