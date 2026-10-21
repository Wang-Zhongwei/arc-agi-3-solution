# throughput-900: research:logs

Workflow run `wf_2e6382c4-06f`, phase Research. Find serving levers to reach 900+ tok/s peak with W4A16 Flash-Next, rank them, and build a Kaggle serving benchmark arm

---

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
