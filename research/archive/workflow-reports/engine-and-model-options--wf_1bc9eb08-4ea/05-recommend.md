# engine-and-model-options: recommend

Workflow run `wf_1bc9eb08-4ea`, phase Recommend. Research Pennyroyal build optimization, alternative (pruned) models like Inkling Small, and alternative inference engines; verify and recommend

---

# Recommendation: Pennyroyal build, alternative model, alternative engine (read-only, 2026-10-04)

**Premise.** Throughput gains have not turned into score so far, so every option here is an enabler judged by cost to test.
- streams14 gave +15% throughput (703 tok/s) but scored 46.38 vs 46.25 (arms.md:83).
- lordhansolo's vLLM ran median 976 tok/s at 14 slots and scored LB 23.84. Franzen ran 588 tok/s and scored LB 27.89 (tong-m2-comparison/kaggle/COMPARISON.md:12,200-215).
- Offline gaps under about 3 points are noise (daniel-franzen-sol/kaggle/README.md).
- Prefill stall is 21.4% of wall. Removing all of it would lift 648 to 826 tok/s (throughput_research.md).

**What I re-checked locally.**
- build_bundle_pennyroyal.sh:29 pins PENNY_SHA d00d88ef.
- The launcher (scratchpad `wt/custom-serving/serving/kaggle/launch_sglang.py:275`) sets `SGLANG_SM120_ONLINE_MXFP8` to "0". It has no hicache, `FORWARD_UNKNOWN_TOOLS` or `reasoning_effort` setting.
- arms.md:92 records MXFP8 decode at -27% (555 vs 765 tok/s).
- Pennyroyal's `qwen3_coder_detector.py:119-124` drops undeclared tool names unless the flag is set.
- Every web, HF, Kaggle and GitHub figure below comes from the fact-checked notes. I did not re-fetch them.

## 1. Ranked options
Each item gives gain, risk, effort in days, and test cost. GPU-hour figures are my estimates, assuming about 8 min cold start (the notes give 478 s) plus run time.

**R1. Pennyroyal no-rebuild env arm: `SGLANG_FORWARD_UNKNOWN_TOOLS=true`.**
- Gain: correctness only. Unknown-tool calls are currently dropped silently, and the harness already answers them with "Unknown tool" (tool_agent.py:6278).
- How often it happens is unknown. Control transcripts hold 0 "Unknown tool" strings, but with the flag off the server drops the call first, so that count says nothing.
- Risk: very low. Effort: 0.25 d. Cost: it rides along in another 20-min panel arm.

**R2. Pennyroyal online MXFP8 plus a runtime patch raising the M caps.**
- Gain, as Pennyroyal writes it: C1 161.465 to 207.124 (+28.28%) and 128K C4 367.127 to 422.037 (+14.96%), with free VRAM after graphs 3.66 to 7.52 GiB (RESULTS.md:53-70).
- Against that, we measured decode -27% at full batch, and issue #4 shows -22.9% on native NEXTN at 4 streams.
- The M>32 dequant fallback (`sm120_online_fp8.py:20,331`) is the cause hypothesised in throughput_research.md. It is unproven.
- Risk: activation-quantization quality, constexpr `ROWS` needs numeric tests, and no one has run it above C4. Effort: 1-2 d.
- Cost: about 1 GPU-hour (two 20-min panel arms).

**R3. Model swap: Swift W4A16 AutoRound on the unchanged stack and harness.**
- Checkpoint: HF `ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AutoRound` (177,298,109,574 B), mirrored on Kaggle as `phuongncn/arc3-qwen38-swift-w4a16-autoround`.
- Gain: unproven. The 27B Swift check showed 43,599 to 38,205 generated tokens (-12%, SWIFT-27B.md:47-49), one run per checkpoint on an older runtime.
- rellik13 found "No better per token and about 35% slower to decode" (WRITEUP.md:147).
- Issue #21: it loads with the BF16 PLE, with PP/TG within +/-3% of Radix.
- Risk: our drafter, patch 04 and score on it are all unverified. Prefix caching and vision are the same engine path as today.
- Effort: 0.5-1 d. Cost: a 20-min smoke arm, then one 121-min run, about 3 GPU-hours.

**R4. Pennyroyal HiCache host write-through (no NIXL).**
- Gain: Pennyroyal's median revisit time falls 9.04 s to 1.05 s (NVFP4, one request at a time). Our ceiling is small.
- Our 4.89M cache-miss tokens (about 4.5% of wall, about +29 tok/s if all removed) are not eviction-driven, so a host tier does not target them.
- Risk: untested with Marlin W4A16, the INT4 g32 draft and patch 04. Host RAM would need 32 GB cache plus our 95.37 GiB BF16 PLE, and Kaggle host RAM is unverified.
- sirikilohit ran a 48 GB host tier on Kaggle v2.5.0 (COMPARISON.md:100-110).
- Effort: 0.5-1.5 d. Cost: needs a full 25-game run, about 2.2 GPU-hours, because the effect only shows at 10+ streams and long context.

**R5. Engine swap: lordhansolo's vLLM overlay, harness unchanged.**
- Datasets: `lordhansolo/vllm-main-e975732-arc3` (7,934,814,402 B) plus `lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8` (180,166,090,115 B).
- Gain: bounded. The fixed per-prefill-batch cost is about 0.28 s x 2,941 batches, about 820 s or 10.9% of wall, and whether vLLM avoids it is unmeasured.
- Per stream it is not clearly faster at 10 streams: about 70 tok/s (976/14) vs our clean bs10 of about 83 (826/10).
- Risk: it changes the quant to NVFP4 mixed, so it must be framed as the arm under CLAUDE.md. `VALIDATION.md:77` says no RTX Pro 6000 server benchmark was run for the newest bundle. Align-mode prefix reuse is unproven against our 93.6% hit rate. The vLLM image unpacks to about 19.5 GB and takes about 114 s.
- Effort: 2-3 d (harness already branches for vLLM, kaggle.py:81-83). Cost: about 4-5 GPU-hours with a control rerun.

**R6. REAP-448 on vLLM 0.30, or Karmic Kraken.**
- Gain: unknown. REAP-448 (`boristown`, 124,217,429,278 B) has no downstream eval and no score. Karmic Kraken is unrun on Kaggle, its numbers are Max-Q at 325 W, and its vLLM version and driver needs are unstated.
- Evidence against pruning: experts16 scored 38.69 vs 46.25, experts8 scored 13.93 vs 17.45, and rellik13's 25% prune played worse.
- Effort: 3+ d each. Defer.

**Rebuilding at the d00d88ef pin: zero gain.** The four later commits touch no `python/` (docs, setup, traffic scripts only). Every v2.4.0-v2.5.3 change is already in our wheel (CHANGES.md:23-266). Cost would be 2-3 h of human time plus 1-2 h of compute.

## 2. What to pursue first (about 5 days)

**First: R1 + R2 as one "serve flags" batch (days 1-2), with the already-queued throughput levers as free riders.**
- Those riders are relaxed accept 0.5/0.5, `_MAX_M` 64 and graph bs 5/6 (throughput_research.md; research-sweep.md:57). I do not repeat that research here.
- Build:
  - A runtime `.py` patch in `SGLANG_RUNTIME_PATCHES` (launch_sglang.py:199-220, currently `{}`). It raises `_MAX_KERNEL_ROWS` 32 to 64 and `_FUSED_MIX_MAX_ROWS` 16 to 64 with `next_pow2` row padding, and skips `convert_eligible_linears_to_mxfp8`.
  - Env arms in `kaggle/arms.json`: `ONLINE_MXFP8=1` and `FORWARD_UNKNOWN_TOOLS=true`. Each arm is an env flag, off by default, in one commit.
- Benchmark:
  - Notebook type: the existing private RTX Pro 6000 kernel, internet off, from `template.ipynb` (`mellamoa/v4-<arm>`, started with `build_notebook.py --push`).
  - A `balanced` 20-min panel at `ARC3_MAX_ACTIVE_STREAMS=4` to reproduce the +15% gate at bs<=4.
  - A second 20-min panel at 10 streams for the bs10 number.
  - Read tok/s and accept length from `serve.log` via `serve_report.py`, which is low-noise per research-sweep.md:55. Compare with fz-tput-base (770 tok/s, accept 2.58) and control (793, 2.66).
- Gate: kill R2 if the bs<=4 gain does not reproduce, or if bs10 does not beat the control clean median of 826 tok/s. If it passes, run one `all` 121-min arm, then `make significance`. Count "Unknown tool" strings in the R1 arm's transcripts.

**Second: R3, Swift W4A16 (days 2-4).**
- Build: a `swift-w4a16` arm that swaps only the model dataset. Keep the albucino drafter, patches and launcher unchanged.
- Zero-GPU pre-check: which of the two Kaggle instances (124.5 GB and 52.8 GB) holds the weights (unverified), and that the layout works with patch 04.
- Benchmark: a `balanced` 20-min 10-stream smoke arm on the same private kernel, checking that it loads, tok/s and accept length against control. Only if decode is within about -10% of control (my threshold), run one `all` 121-min arm. Compare tokens per request and the paired per-game score.
- Start one GPU session at a time. Two simultaneous cold starts reading about 190 GB hung both servers (kaggle/README.md).
- Why this second: it is the cheapest change that alters the model, and it reuses the engine, drafter and prefix cache unchanged.

**Day 5 and later.** R5 only if R2 and R3 are both flat and at least one GPU-day remains. Do the controlled A/B described in R5 and leave Karmic Kraken for later.

## 3. What NOT to do
- **Do not rebuild the wheelhouse at d00d88ef.** Only a changed patch set, NIXL or the NVMe reader justifies a new bundle.
  - The build needs internet, and whether the RTX Pro 6000 or competition-linked notebooks allow it is unverified.
  - The target image is private (`gcr.io/kaggle-private-byod/python@sha256:57e612b4...`).
  - Keep `dfranzen/pennyroyal-v253` (6,640,272,567 B) as the fallback.
- **Do not use Inkling Small, pruned or not.** The size, MTP and sm120 points are from the notes' Inkling section; the ARC-AGI-3 point is from the notes' arcprize.org source.
  - It is 276B total / 12B active (README), and the official NVFP4 is 170.8 GB. The vLLM recipe says "Requires at least 180 GB aggregated VRAM".
  - Its MTP head and DSpark speculator exist, so MTP is not the reason to skip it.
  - Pruned builds are MLX or llama.cpp GGUF. miweru's REAP-137B drops vision, and there is no CUDA 4-bit pruned build with vision under about 85 GB.
  - A 50% prune is about 1.7x Flash-Next bytes per step (derived, not measured), and the BF16 calibration pass over about 532 GB is not feasible locally.
  - The SGLang cookbook lists no RTX Pro 6000. arcprize.org says it "has not yet been evaluated on ARC-AGI-3".
- **Do not switch to Qwen3.8-27B or Gemma 4 (31B, 26B-A4B).**
  - 27B gives 374.98 tok/s at 4 requests (README.md:175-181) and has no 10-stream number.
  - Gemma has no ARC-AGI-3 score evidence. All three Milestone 2 winners served Flash-Next (SUMMARY.md:3-9).
- **Do not use TensorRT-LLM, llama.cpp, ExLlamaV3 or NInfer.**
  - TRT-LLM: issue #11799 (closed) says no plan for SM120/121, and PR #18585's CI has no sm120 entry.
  - llama.cpp MTP has open bugs (#28286, #29148).
  - ExLlamaV3 and NInfer are unmeasured for 10-stream serving, and NInfer needs an RTX 5090.
- **Do not enable online MXFP8 as a bare toggle** (-27%), and do not A/B `SGLANG_ENABLE_OVERLAP_PLAN_STREAM`. The overlap flag broke native MTP in a community TP2/EP2 setup (COMMUNITY-RESULTS.md:318).
- **Do not prune Flash-Next experts.** See R6 for the three pieces of evidence. The lovedheart benchmark table also looks inherited from the unpruned card, since 294,912 = 48 x 512 x 12.
- **Do not repeat the reasoning-effort `medium` arm.**
  - It was already tried on Flash-Next (GB10) and "doesn't shorten thinking" (arms.md:106; research-sweep.md:81).
  - The 80.4% token cut in the notes is 27B/DFlash2 only (RESULTS.md:583-591).
  - Static `low` is a separate backlog item. The harness already has `ARC3_REASONING_EFFORT_LADDER` (tool_agent.py:1306; CONFIGURATION.md:408), and the model card warns low effort can cause retries.
- **Do not enable NVMe PLE** unless we are RAM-bound. C4 median fell 428.90 to 369.73 (-13.8%, FP8 table, NVME-PLE.md:138-141), and BF16 and the albucino drafter are untested.
- **Do not treat a 20-min 10-game panel as a score verdict.** Use it for tok/s and accept length only.

**Unverified.**
- Kaggle host RAM, internet, and GPU clock control on this shape.
- Whether HiCache works with Marlin W4A16, the INT4 g32 draft and patch 04.
- The score and speed of the Swift W4A16 checkpoint under our harness.
- The vLLM version inside Karmic Kraken.
- Tufa's LB 55.89 (local notes only).
