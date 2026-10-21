# Research
### forks
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

### logs
## Decode-time decomposition (serve logs, control + 3 comparisons)

**Method.** I parsed every `Decode batch`, `Prefill batch` and `ReqTimeStats` line with python. Source: `.../kaggle-outputs/fz-arms-v4-control/serve.log` (2h02m, 4803 decode logs, 2941 prefill batches, 2311 requests). I repeated it on fz-ledger-on-trim, fz-tput-base and fz-experts8; results are near-identical except where noted.
- **Log interval.** Each decode log is 40 decode steps (`decode_log_interval: 40`, server_args line 12). Implied interval `40*bs*accept_len/tp` sums to 7313 s against 7211 s of timestamp gaps, so it is within 1.4%.
- **Prefill stall.** Logged `gen throughput` includes any prefill that ran inside the interval. I call an interval "clean" if no `Prefill batch` line falls inside it. Stall = interval time minus 40 x the median clean step time at that bs.
- **Issue number.** The low-M GEMM issue is **#12**, not #11. #11 is trim re-prefill. #10 (prefill ~15% of wall), #7 (batch mean ~8.9/10) and #12 already cover parts of this; I cross-checked and the numbers agree.

### Per-bs table (control)

| bs (M=4*bs) | wall % | clean step ms | clean tok/s | accept len | tok/s incl. stalls |
|---|---|---|---|---|---|
| 5 / 6 | 1.1 / 2.9 | 28.1 / 28.3 | 492 / 591 | 2.71 / 2.73 | 469 / 520 |
| 7 | 5.9 | 27.5 | 690 | 2.73 | 614 |
| 8 (M=32) | 12.7 | 29.3 | 746 | 2.73 | 616 |
| 9 (M=36) | 32.5 | 30.7 | 788 | 2.70 | 582 |
| 10 (M=40) | 44.4 | 32.2 | 827 | 2.66 | 648 |

- All 4803 decode logs say `cuda graph: True`.
- Mean bs is 9.03, and bs>=9 is 77% of wall. Whole run is 613 tok/s, matching the summary's 611.
- Logged bs10 percentiles: median 793, p90 855, max 927. The clean-only median is 826 and p95 is 881.
- **900+ at bs10 is rare.** Only 16 of 1161 clean bs10 windows (1.4%) reach 900 in control; ledger-on-trim and tput-base are 1.2-1.3%.
- **Step cost per stream.** Step time rises 1.4-1.7 ms per added stream with no jump at M 32->36 (29.3 -> 30.7 -> 32.2). The same smooth slope is in issue #12's evidence.

### Where the time goes (control, run-level 613 tok/s)

| Step | tok/s | Note |
|---|---|---|
| Ideal: all time at clean bs10 | 826 | |
| After bs-mix loss | 780 | -46 tok/s (-5.6%) |
| Actual | 613 | -167 tok/s |

Prefill stall is 1604 s of 7496 s, i.e. 21.4% of wall (bs10 alone: 21.4%, bs9: 26.4%).
- **bs-mix loss.** In-flight mean is 8.9: 65,338 s of summed `forward_duration` over 7334 s. That matches issue #7's reading that the idle is client-side. Server queue is ~0.6% of slot-seconds (see Queueing below).
- **Stall by prefill tokens in the interval:**

| Tokens in interval | Intervals | Mean stall | Total stall | % of wall |
|---|---|---|---|---|
| <=8.2K | -- | -- | -- | 10.6% |
| 8-20K | 143 | 1.5 s | 216 s | 2.9% |
| >20K | 82 | 7.2 s | 593 s | 7.9% |

- **Stall cost model.**
  - Intervals with exactly one prefill batch: stall = 0.279 s + 68 us/token (14.6K tok/s).
  - At equal token counts, a second batch in the same interval adds ~0.39 s (pc=2: 883 ms vs pc=1: 496 ms).
  - Fit over all intervals: fixed cost = 0.279 s x 2941 batches ≈ 820 s (10.9% of wall), token cost ≈ 730 s (9.7%).
  - Small prefills (<=1.2K tokens) stall ~300-325 ms, while the request's own `initial_prefill_elapsed` is p50 124-150 ms. The extra 150-200 ms per batch is unexplained; CPU/scheduler/draft-extend work is my guess, unverified. It needs a profiler.
- **Prefill is serial and unbatched.** 97.8% of batches carry one sequence, none uses a CUDA graph (0/2941 True; serve.log:5 and :77, multimodal disables prefill graphs), and decode does not run between chunks. Example at serve.log ~line 7736: 6 back-to-back 8192-token chunks at 12.3-13.7K tok/s with 8 decoding requests frozen for about 3 s, then the next decode log shows 141.9 tok/s.
- **Cache-miss prefill is the big token source.** Batches with `#cached-token: 0` are 728 batches and 4.89M tokens, 45.8% of prefill tokens (cache hit overall 93.6%). That is ~334 s of token cost (4.5% of wall).
  - It is not KV-pool eviction: misses are 7-29% of batches at every usage level below 0.9.
  - KV usage peaks at 0.99 (1,000,768 of 1,011,264 tokens, serve.log:61) but is >=0.95 in only 1.4% of logs. Mamba peaks at 0.67 of 60 slots.
  - Zero retractions: `attempts=0` on every request and no retract lines.

### Queueing
- **Decode side.** `#queue-req` is >0 in only 102 of 4803 decode logs, max 1.
- **Prefill side.** It is >0 in 356 of 2941 prefill logs, max 7, mostly behind chunked re-prefills.
- **Request wait.** `queue_duration` is p50 1.48 ms, p90 3.3 ms, p99 6.2 s; 90 requests (3.9%) waited >1 s, ~470 s in total.
- **Over the run.** Mean bs drifts from 9.68 (first 10 min) to 8.31 (last 10 min).

### Headroom at bs10 (clean base 826; base incl. stalls 648)

| Lever | Estimated gain | How |
|---|---|---|
| Accept len 2.66 -> 2.80 / 3.0 | +44 / +106 tok/s clean | 10*al/0.0322 s; +0.1 al = +31 clean, ~+24 incl. stalls |
| Step time -1 ms | +26 tok/s clean (+3.2%) | |
| Remove all prefill stall | 648 -> 826 (+177) | Unreachable: prefill work must run; this is the ceiling |
| Halve per-prefill-batch fixed cost (~0.28 s) | ~+36 tok/s | Whole-run basis |
| Eliminate cache-miss prefill tokens | ~+29 tok/s | Issues #3/#10/#11/#22; 4.89M tokens |
| Prefill 1.3x faster | ~+14 tok/s | |
| Perfect bs10 occupancy | 780 -> 826 (+46) | Mostly client-side |
| Full decode CUDA-graph coverage | 0 | Already 100% |
| Add graphs for bs 3/5/6 | <=+0.3% overall | Graph list is `[1,2,4,7,8,9,10]` (serve.log:78, launch_sglang.py:301) |
| Lift low-M GEMM cap (issue #12) | unknown | Per-stream slope has no kink, so likely small; if it saved 0.5-1.0 ms at bs9-10, +13 to +26 clean (+1.6-3.2%) |

**Graph padding.** bs 5/6 cost 28.1/28.3 ms against 27.5 at bs7, and bs3 costs 22.4 against 22.7 at bs4. That is consistent with padding up to the next captured size.

**Accept len.** Clean bs10 windows are p10 2.49, median 2.66, p90 2.84, max 3.04 (3.13 in ledger-on-trim), so a sustained 3.0 is above anything observed.
- **Sampling.** Server default is T=1.0, top_k 20, top_p 0.95 (serve.log:111). Requests carry no temperature field (fz-experts8 request jsonl keys are messages, tools, tool_choice, chat_template_kwargs).
- **Temperature arm.** fz-arms-v5-temp06 (temperature 0.6 inferred from the arm name; I did not verify it) gives accept len 2.698 vs 2.661 and clean median 837 vs 826, only ~+1.4%.
- **Thresholds.** Accept thresholds 1.0/1.0 (launch_sglang.py:32-33, :337) are the only untested lever on accept length.

**Calibration from fz-experts8 (only the step time moves).**
- Clean bs10 step is 29.6 ms (-8% vs 32.2) at accept len 2.60. Clean bs10 median is 872 and 17% of windows reach 900+; control is 1.4%.
- I read this as a rough proxy for what a ~3 ms step-time cut would do: roughly the same 826 -> ~880 range, not a promise.

### Paths to a 900 clean bs10 median (10*al/step = 900)

| Accept len | Step ms |
|---|---|
| 2.66 (now) | 29.6 |
| 2.80 | 31.1 |
| 2.90 | 32.2 |
| 3.00 | 33.3 |

- Combining accept len ~2.8 with a step of ~31 ms reaches 900.
- Even 900 clean at bs10 gives only ~710 incl. stalls (900 x 648/826 ≈ 706), so the prefill stall (~21% of wall) caps whole-run throughput. Peak decode and wall-clock gains are largely independent.

### Caveats
- Timestamps are 1 s resolution; the per-bs "clean" classification uses the bs logged at the end of the interval.
- The draft vs verify split of the 32 ms step is not recoverable from the logs. It needs a profiler (nsys or the SGLang profile endpoint).
- Whether `--enable-mixed-chunk` (False), `--prefill-decode-interval` (0) or a larger `--chunked-prefill-size` helps or even works with NEXTN + extra_buffer Mamba is untested (issue #10).

Scripts: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/an/` (parse.py, a1-a7.py)

### launcher
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

# Plan
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

# Bench
The bench is built and tested on branch `arm/serving-bench`, but nothing has run on a GPU yet, so there are no speed numbers. All 33 unit tests pass in 8.5s, and dry builds of both benches and the control arm succeed. I did not push or run Kaggle.

For your 900+ goal, the most direct config is C7: Franzen's launcher with 14 slots instead of 10, and no lossy changes. Our earlier 14-stream run (fz-streams14) already logged a median of 849 tok/s, p90 1026 and max 1122 at 14 running requests. That's the slot count behind the 900+ peaks others report. C7 runs second so the time budget can't drop it. Every config also reports its peak window and the share of windows at 900 or above.

**Branch:** `arm/serving-bench` (from `arm/custom-serving`), worktree `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/serving-bench`.

**git log** (each commit ends with the two required trailer lines):
```
a89a949 Document serving benchmarks
241f044 Test the bench plan, requests, log parsing, client and build
d134ea5 Build serving benchmark notebooks with --bench
b4ea6fc Define the throughput plan's benchmark configs
bee456d Add bench patches that widen the SM120 low-M GEMM to M=64
537e6bc Add the serving benchmark driver
170471f Add a prompt corpus of real harness conversations for serving benchmarks
408db4b (base) Document every layer of the serving stack we can now modify
```

**Tests:** run from `<worktree>/ARC3-Inference` with `.../.venv/bin/python -m pytest -q ../kaggle/tests/test_bench.py`. The client and the bench driver run end to end against a small fake server. Normal arm notebooks are unchanged: control keeps fingerprint `1fa948b10cf43854`.

**How to launch** (from the worktree root):
- `python kaggle/build_notebook.py --bench smoke` builds `kaggle/out/v4-bench-smoke/`. It runs C0 with short phases, about 15 minutes, to check the machinery.
- `python kaggle/build_notebook.py --bench levers` builds the full session (about 582 KB). Add `--push` to push `mellamoa/v4-bench-<name>` and record it in `runs.json`. Benches are recorded as `"kind": "bench"` with `batch_open: false`, so they never join or block an arm batch. The duplicate-fingerprint check still applies.
- Results go to `/kaggle/working/bench_results.json` and `bench_results.md`, rewritten after every phase. Each config's serve.log is in `bench/<config>/`, and C0's profile trace is in `bench/C0/profile/`.
- I'd push `smoke` first, then `levers`.

**What each config does:** it launches SGLang through the same launcher text the arms use, with that config's overrides. Patches go onto a fresh copy of the installed SGLang. It waits until healthy, sends one test request, runs its timed load phases, then kills the server.

**The load:**
- Clients replay real harness conversations from `kaggle/bench_corpus.json.gz`: 16 games from fz-arms-v4-control, built by `kaggle/make_bench_corpus.py`. Speculative acceptance depends on what the model writes, so filler text would measure a different accept length.
- All streams share the 5k-token system prompt, and each turn extends the previous one, so the prefix cache is exercised.
- Requests match the harness: T 0.7, top_p 0.95, top_k 20, thinking on, the `python` tool. They are identical across configs.
- On the plan's temperature conflict: the harness code does send T 0.7. The T 1.0 in serve.log is only the server's default for requests that omit it.
- Phases:
  - `warmup10`: 60s at 10 streams, not reported.
  - `clean10`: 240s at 10 streams, outputs of 1,536–2,048 tokens.
  - `mixed10`: 180s, non-streamed like the harness, a quarter of requests cold.
  - Plus sweeps at 1, 4, 7 and 8 streams, `over12` (12 clients on 10 slots), `clean14`/`over16`, and a profile phase.

**Configs (`kaggle/bench.json`, bench `levers`, run in this order):**

| Config | Change | Note |
|---|---|---|
| C0 | None (Franzen's launcher) | Plus sweeps, 12-client phase and profile trace |
| C7 | 14 slots (`MAXREQ`/`CUDAGRAPH_MAXBS` 14, `MAMBA_CACHE` 84) | Lossless |
| C1 | Relaxed MTP acceptance 0.5/0.5 | Changes outputs, needs a score check |
| C2 | Relaxed MTP acceptance 0.3/0.3 | Changes outputs, needs a score check |
| C3 | GEMM patches `m64.py` + `nosplitk.py`, plus shape logging | Lets bs 9/10 use the faster kernel |
| C4 | FP8 port + `m64.py` | **Disabled**: the FP8 port isn't written yet |
| C5 | Fastest of C0/C3/C4 plus fastest of C0/C1/C2 | Picked during the run by clean bs10 median |
| C6 | C5 plus `OMP_NUM_THREADS=1`, overlap plan stream, CUDA graphs at every bs 1–10 | |

- **Budget:** 150 minutes from notebook start. The estimate is about 142 minutes plus setup. If it runs behind, C6 is dropped, then C2.
- **C4:** to enable it, add `serving/bench_patches/fp8-port.py` and set `enabled: true`.
- **`min_tokens`:** if this SGLang fork rejects it, the bench runs that config with natural output lengths and records a warning instead of failing.

**Main files:**
- `serving/kaggle/bench.py`
- `kaggle/bench.json`
- `kaggle/build_notebook.py` (adds `--bench`)
- `kaggle/make_bench_corpus.py`
- `kaggle/bench_corpus.json.gz`
- `serving/bench_patches/` (`m64.py`, `nosplitk.py`, `README.md`)
- `kaggle/tests/test_bench.py`
- `kaggle/README.md` (new "Serving benchmarks" section)