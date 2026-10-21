# Recommendation
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

# Verified notes
Note: claude-sonnet-5[1m] (the safety classifier) was unavailable (timed out) when reviewing this subagent's work. Please carefully verify the subagent's actions and output before acting on them.

# Fact-checked notes (read-only, checked 2026-10-04)

Inkling Small exists. It is Thinking Machines Lab's open-weights multimodal MoE, announced 2026-07-30. Sources: https://huggingface.co/thinkingmachines/Inkling-Small and https://thinkingmachines.ai/news/inkling-small/. The notes' verdict stands: no for this competition, because of size and the absence of any ARC-AGI-3 evidence.

## pennyroyal-build (corrected)

**Bottom line.** No newer Pennyroyal runtime exists.
- The latest release is `pennyroyal-v2.5.3`, published 2026-09-28T02:00:51Z (`gh api repos/jpezzulli/sglang-rtxpro6000/releases`). The tag points at d00d88ef.
- Branch `pennyroyal-main-sm120-final` HEAD is `f9c4a89c` (2026-10-03).
- The four commits after d00d88ef are `92f57171` (WSL2 host-memory setting), `468e63a9` (install guides), `12846e83` and `f9c4a89c` (traffic-report scripts). The compare shows only docs, config, `penny_setup.py`/`penny_config.py` and `.github` scripts, with no `python/` changes. There is also a tag `pennyroyal-v2.5.3-setup1` at 468e63a9.
- Our bundle pins d00d88ef (`build_bundle_pennyroyal.sh:29`), so every v2.4.0–v2.5.3 change (CHANGES.md:23-266) is already in the wheel. A rebuild of the same pin gains nothing.
- The remaining levers are env/flag toggles or runtime patches. Only a changed patch set, NIXL, or the NVMe reader forces a new wheelhouse.

## Optimizations (citations under `competitors/sglang-rtxpro6000/` unless noted)

| Item | Measured gain, as written | W4A16 (Intel AutoRound), bs~10 | NVFP4 (RadixArk) at ~10 streams | Rebuild? |
|---|---|---|---|---|
| **Online MXFP8** `SGLANG_SM120_ONLINE_MXFP8=true` (v2.5.0; FP8.md:8-13, RESULTS.md:53-70) | C1 short 161.465→207.124 (+28.28%). 128K C1 154.703→195.634 (+26.46%). 490K C1 149.135→172.644 (+15.76%). 128K C4 367.127→422.037 (+14.96%). 490K C4 +39.01% (unequal token totals). Free VRAM after graphs 3.66→7.52 GiB. | We forced it off (`launch_sglang.py:275`, bundle `runtime_env.sh:1231`) after measuring decode −27% at full batch, 555 vs 765 tok/s (arms.md:92). | They never measured above C4. (C6 runs were regression-only, CHANGES.md:119-122.) [Issue #4](https://github.com/jpezzulli/sglang-rtxpro6000/issues/4) (v2.5.0, public RadixArk NVFP4, 4 streams) has its own deviations: RunAI streamer from S3, no NIXL, driver 580.x. Native-NEXTN arms: 291.0→224.5 tok/s (−22.9%). FR-Spec arms: 319.9→235.6, but confounded by accept len 1.00/rate 0.00 at batch 1 (batch 1 and 2 in another run). The maintainer could not reproduce it. | No for the toggle. The M>32 cap (`sm120_online_fp8.py:20,331`) would be a runtime `.py` patch. It is the hypothesis from the prior research, unproven. |
| **HiCache host tier, no NIXL** (v2.5.1 fix for issue #3; CHANGES.md:107,113-116; RUN.md:262-270) | "0/9 to 9/9" prefix restores; median revisit 9.04 s→1.05 s (NVFP4, one request at a time, ~105K-token chats). Issue #3's own numbers: 7.3 s→0.5-0.7 s with `--hicache-size 24`. Write-through remains the policy. | Our launcher has no `--enable-hierarchical-cache` (`launch_sglang.py:305-345`). `extra_buffer` was part of issue #3's recipe, so it is not the untested part. Untested: Marlin W4A16, the INT4 g32 draft, and Franzen's sparse-checkpoint patch 04. **Kaggle precedent:** sirikilohit ran a 48 GB host cache tier on Pennyroyal v2.5.0 (COMPARISON.md:100-110). Our 4.89M cache-miss tokens are not eviction-driven (prior research), so the ceiling is small. **Host RAM:** 32 GB default (RUN.md:262-270) plus our **95.37 GiB BF16 PLE** (COMPARISON.md:106), not the ~47.68 GiB FP8 PLE of README.md:33-36. | Their qualified path for decode and restore at C4; not measured at 10. | No, flags only. The bundle omits NIXL (`build_bundle:1354`), and the disk tier is not needed. |
| **NVMe PLE** (v2.5.0; NVME-PLE.md:138-141) | C4 median RAM 428.90 vs NVMe 369.73 (−13.8%), measured with the **FP8** table and online FP8 off. Community: C2 349→323, C4 411→403, 128K C1 176→164 (COMMUNITY-RESULTS.md:236-250). Saves 47.68 GiB (NVME-PLE.md:8). | Only worth it if RAM-bound. BF16 NVMe PLE staging is **supported in v2.5.3** (CHANGES.md:35-39; PROVENANCE.md), so our table would save ~95.37 GiB, not 47.68. The −13.8% was not measured on BF16. Compatibility with the albucino drafter is unverified. | Same speed cost. | Separate Rust reader build (`tools/ple_nvme/install.sh`) plus an overlay. Not an SGLang rebuild, but a new bundle component. |
| **FR-Spec map** (v2.3; RESULTS.md:277-280) | +9.7–16.8% C1, +4.5–7.0% C4. | Already used (`hot_tokens_64k.pt`). untcoder2 found full-vocab drafting beat the map on RU/EN prompts (COMMUNITY-RESULTS.md:393-394), in a TP2 GPU-resident-PLE setup. A/B only. | Same. | No (data only). |
| **v2.4.0 prefill, v2.4.1/2/3 fixes** (RESULTS.md:223-226; CHANGES.md:200,248) | 64K 13,506→14,842 tok/s (+9.90%). 490K 8,464→8,773 (+3.65%). v2.4.1 PLE kernel: "No net serving-speed gain". | Already in d00d88ef and quant-independent. No action. | Same. | n/a |
| **`SGLANG_FORWARD_UNKNOWN_TOOLS=true`** (v2.5.2, CHANGES.md:92-93; their launcher `serve-flash-next-frspec.sh:9`, and RUN.md:140-145 says launchers default true; code default False at `environ.py:686`, used at `qwen3_coder_detector.py:120`) | Correctness, not speed: unknown-tool calls reach the client for error recovery. | Not set in our launcher (grep), so off for us. Relevant to python-tool recovery. | Same. | No (env). |
| **Reasoning effort default "medium"** (PR #18, closed; adapted into v2.5.2; `reasoning-effort.sh`; template `froggeric-v22.5.jinja:17-27`) | **27B/DFlash2 only** (RESULTS.md:583-591): medium cut completion tokens 80.4% and summed request time 84.74% vs xhigh; reasoning 95.807 vs 98.26. | **Verified:** our checkpoint's `chat_template.jinja` (loaded at `launch_sglang.py:327-328`) honors `reasoning_effort`. It accepts only xhigh (default), medium and low, and raises an exception for anything else. We are therefore at xhigh today, and `medium` is a request-level or template-kwarg change. Quality trade-off; no Flash-Next number exists. | Same. | No (config). |
| **Swift (ukisai) fine-tune** (SWIFT-27B.md:47-49; issue #21) | 27B: generated tokens 43,599→38,205 (−12%), suite elapsed 169→158 s (−6.5%), one run per checkpoint on an older runtime. | **A Swift W4A16 AutoRound Flash-Next checkpoint exists:** HF `ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AutoRound` (177,298,109,574 B, created 2026-09-22), mirrored on Kaggle as `phuongncn/arc3-qwen38-swift-w4a16-autoround` (two instances, 124,500,425,336 + 52,790,776,295 B). Not verified: our drafter, patches and score on it. rellik13: Swift 1.5 "No better per token and about 35% slower to decode on my setup" (rellik13 WRITEUP.md:147). | Issue #21: loads with BF16 PLE; PP/TG "+/-3%" vs Radix. The NVFP4 Kaggle mirror `lordhansolo/swift-1-5-qwen3-8-flash-next-nvfp4` (186,458,042,122 B) says quantized quality benchmarking is pending and native Blackwell validation has not been run. | No (model swap). |
| **Larger KV cap / C6 recipe** (CHANGES.md:110,121-122; COMMUNITY-RESULTS.md:150,184-188) | Capacity only: 1,034,176 fitted; community 1,114,304 (a 1,310,720 request clamped by the server). | We already have 1,011,264. Skip. | n/a | No |

Not applicable:
- **GPU clocks:** the RESULTS.md:363-371 profile (power cap 450, core/mem offsets, `lactd`) needs driver control. I did not verify whether Kaggle allows it.
- **NIXL disk persistence and 27B/DFlash2:** C1 108.31 tok/s, C4 374.98 tok/s (README.md:175-181). The later dated release shows 108.75 and 390.23 near-empty context (RESULTS.md:565-575). Not useful for us.
- **Overlap plan stream:** Pennyroyal warns against `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`. It broke native MTP in a community **TP2/EP2** setup until removed (COMMUNITY-RESULTS.md:318, RESULTS.md:436-442). That was not TP1. The earlier launcher review proposed it as an A/B, so treat that arm as risky.

## Practical wheelhouse build without the laptop

**What I found**
- **Target image is private.** It is `gcr.io/kaggle-private-byod/python@sha256:57e612b4…`, machine shape `NvidiaRtxPro6000`, `enable_internet: false` (`daniel-franzen-sol/kaggle/template-metadata.json`).
- **The run log shows Python 3.12 and nvcc 13.0.88.** The log is `/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs/fz-arms-v4-control/fz-arms.log`. Python 3.12 is at line 142, nvcc 13.0.88 (pip overlay `nvidia/cu13/bin/nvcc`) at 147-151, and the venv plus wheelhouse install took ~3.4 s (lines 144-146).
- **The builder needs no Docker and no GPU.** The README says "No GPU is needed for the build" (`serving/README.md:21`). It needs python, git, curl, gcc, g++, make and pkg-config (`build_bundle_pennyroyal.sh:46-48`), plus internet.
  - It fetches rustup (`:1484`), PyPI, pytorch cu130, `docs.sglang.ai/whl/cu130` and `pypi.nvidia.com` (`:1521-1524`), and `flashinfer.ai` (`:1545-1547`).
  - It compiles only SGLang's Rust extensions (`:1488-1494`). torch, sglang-kernel and FlashInfer are prebuilt wheels.
- **The only compatibility gate is `check_platform`** (`:1049-1060`). CPython major.minor, SOABI and arch must be equal, and runtime glibc must be ≥ build glibc.
- **Existing bundle size is known.** `kaggle datasets list` shows `dfranzen/pennyroyal-v253` = 6,640,272,567 B (6.6 GB).

**Steps**
1. Open a private Kaggle notebook with Internet ON (CPU or any GPU). Verify `python3 --version` is 3.12 and note `ldd --version`.
2. Put the script in the notebook and run `OUT=/kaggle/working/out WORK=/tmp/penny BUILD_JOBS=4 bash build_bundle_pennyroyal.sh` in one `%%bash` cell. On failure, resume with `RESUME_STAGE=` (`:24`).
3. Publish `bundle-pennyroyal` as a dataset, either with "Save Version → New Dataset from output" or `kaggle datasets create` using Kaggle Secrets credentials.
4. Prove ABI match in an offline notebook on the real image. Attach the dataset and run `install_offline.sh` (~4 s). It fails fast on a Python or glibc mismatch.
5. **Fallback if Kaggle's internet image does not match:** run the same script on a GitHub Actions runner.
   - Use an ubuntu-22.04 container or runner with CPython 3.12, which has older glibc, so it is safe under the ≥ rule.
   - Upload with `kaggle datasets create`.
   - Pennyroyal's own CI builds the full image on ubuntu-24.04 with `timeout-minutes: 180` (`.github/workflows/pennyroyal-container.yml:31-32`) and frees disk first (`:72-77`).

**Blockers**
- **Image tag:** the private digest cannot be pulled. The embedded README says "Prefer a fixed image digest to a moving latest tag" (`build_bundle_pennyroyal.sh:1296`). The header example says to use the same image/tag as the target notebook (`:13`).
- **Internet:** Kaggle needs a phone-verified account to enable internet (secondary sources only: fast.ai forum threads, course notes). I did not verify whether internet can be enabled on the RTX Pro 6000 shape or competition-linked notebooks.
- **Disk and size:** the output bundle itself is ~6.6 GB. lordhansolo's source states `/kaggle/working` is a 20 GB volume and the root overlay on the RTX PRO 6000 host showed ~1.1 TB free (`tong-m2-comparison/kaggle/lordhansolo/dataset-taaf-kaggle-source/src/ARC3-Inference/inference/framework/kaggle.py:838-841`). The 25–40 GB scratch need (wheels, resolver venv, offline-check venv) is still my guess. The internet-notebook disk is unverified.
- **nvcc and arch:** no GPU is needed. The sm_120 compile check runs at the end of the script. The JIT recompiles at first launch, as today. Pennyroyal qualified CUDA 13.3.73, GCC 15.3.1 and Python 3.12.13 (BUILD.md:222-234); we already deviate (13.0.88, image GCC).
- **Hash guards:** patch helpers are pinned to d00d88ef with exact BEFORE/AFTER sha256, and there is `len(result) == 13` at `:1029`. Any new patch or revision needs the hash chain, README text and manifest edited.
- **Index drift:** a `docs.sglang.ai` or `flashinfer.ai` change could break the build. Keep `dfranzen/pennyroyal-v253` as the fallback.

**Effort estimate (mine, not measured for our script)**

| Task | Estimate |
|---|---|
| Compute | Pennyroyal's CI built the full image in 25–30 min. Run 36368166760 took 27m28s (02:00:53→02:28:21), 36080605586 took 30m25s (01:07:08→01:37:33), 35156312504 took 25m28s (22:10:25→22:35:53). Our script also downloads and wheels every dependency and runs an offline reinstall, so budget 1–2 h. |
| Identical rebuild, human time | 2–3 h |
| Add one embedded patch to the builder | +1–2 h |
| Runtime patch, no rebuild | ~30 min via `SGLANG_RUNTIME_PATCHES` (`launch_sglang.py:199-220`, currently `{}`) |

**Recommendation:** do not rebuild. Test without a rebuild, in this order:
1. `FORWARD_UNKNOWN_TOOLS=true`.
2. A host-HiCache write-through arm.
3. Reasoning effort `medium` (the template supports it).

Revisit MXFP8 only with the cap-raise runtime patch, after the prior research's reproduce-at-bs≤4 step.

## Unverified
- Kaggle host RAM for the RTX Pro 6000 shape. Lower bound: Franzen retains the 95.37 GiB BF16 PLE because "the competition machine has enough system memory" (daniel-franzen-sol/WRITEUP.md:33), and the other two winners also offload ~95-97 GB (COMPARISON.md:106).
- Internet availability on the RTX Pro 6000 and competition-linked notebooks, and the internet-notebook disk size.
- The glibc and Python patch version of the private image beyond Python 3.12.
- Whether HiCache works with the INT4 g32 draft, Marlin and Franzen's patch 04.
- BF16 NVMe PLE with our albucino drafter.
- Quality, speed and score of the Swift W4A16 checkpoint on our harness.

## models (corrected)

Evidence tags: [web] fetched this session, [local] file:line in the repo, [derived] my arithmetic, [unverified].

### 1. "Inkling Small" exists and is NOT a fit for one 96 GB card
- **Who/when:** Thinking Machines Lab, announced 2026-07-30 ("July 30, 2026" on the news page). The HF repo was created 2026-07-27 and last modified 2026-07-31. Apache-2.0. [web] https://huggingface.co/thinkingmachines/Inkling-Small , https://thinkingmachines.ai/news/inkling-small/
  - The larger sibling `thinkingmachines/Inkling` (HF, created 2026-07-14) is a different model. Repos named `Inkling-MLX-REAP12/25/50-4bit` belong to that big model, not to Small.
- **Architecture** (HF config.json): sparse MoE, "276B total, 12B active" per the README. Safetensors metadata counts 265,956,439,090 params (~266B). 42 layers, hidden 4096, 256 routed experts top-6 plus 2 shared, expert inner dim 2048, 1 dense-MLP layer (`dense_mlp_idx` 2).
  - 35 of 42 layers are sliding-window (512), 7 global. `model_max_length` 1,048,576 (news page: "up to 1M tokens").
  - **MTP:** `mtp_config.num_nextn_predict_layers` = 8, `mtp.safetensors` ships in the repo, and the SGLang cookbook documents MTP (needs `--enable-multi-layer-eagle`).
  - Separate drafter `RadixArk/Inkling-Small-DSpark`: 2,789,706,894 B, 1.39B params. Card: mean accept 3.8226 at T0 and 3.6864 at T1.0/top-p 0.95 across 12 datasets.
- **Capabilities:**
  - Input is text, image (40-4096 px, 40 px patches) and audio. Output is text.
  - Thinking via `reasoning_effort` none..max, default high (SGLang cookbook `competitors/sglang-rtxpro6000/docs/cookbook/autoregressive/ThinkingMachines/Inkling-Small.mdx:95`).
  - Tool calling through SGLang parsers. BFCL 76.54% on B200 NVFP4 (`docs/src/snippets/configs/thinkingmachines/inkling-small-benchmarks.jsx:7`).
- **Benchmarks** (README table): SWE-bench Verified 80.2, Terminal Bench 2.1 64.7, Toolathlon 54.4, MCP Atlas 79.6/79.2, tau3-Banking 15.5, MMMU Pro 74.0, CharXiv RQ 77.4 (81.3 with python), GPQA 89.5.
  - ARC-AGI-1 84.0% at $0.11/task and ARC-AGI-2 40.1% at $0.23/task, both at xhigh. ARC-AGI-1/2 at high: 78.0/33.1; medium 67.0/13.6; low 52.5/4.9 (https://arcprize.org/results/thinky-inkling-small). The same page lists minimal 48.5/4.0 and none 5.5/0.0.
  - **ARC-AGI-3: not evaluated.** The page says "It has not yet been evaluated on ARC-AGI-3."
  - No Flash-Next vs Inkling-Small comparison found. Strong static-ARC results are not evidence for the interactive games.
- **Size:** official NVFP4 `thinkingmachines/Inkling-Small-NVFP4` is 170,764,923,366 B (170.8 GB, 21 files). BF16 is 531.9 GB.
  - Kaggle has it: `mirzamilanfarabi/inkling`, instance `small-nvfp4` (170,764,923,366 B) and `2-bit-unsloth` GGUF (88,494,649,967 B). [web, `kaggle models get`]
  - NVFP4 needs ~2 big GPUs. The vLLM recipe says "Requires at least 180 GB aggregated VRAM" (https://recipes.vllm.ai/thinkingmachines/Inkling-Small). The 88.5 GB GGUF leaves ~7 GB for KV and graphs and runs on llama.cpp only.
- **Engine support:** the SGLang cookbook lists h200/b200/b300/gb200/gb300/dgx-spark/mi350x/mi355x only (`docs/src/snippets/configs/thinkingmachines/inkling-small.jsx:10-12`). RTX Pro 6000 / sm120 is not listed.
  - The NVFP4 CI test uses tp=4 on a 4-gpu-b200 runner and says "the checkpoint does not fit on fewer cards" (`test/registered/models_e2e/test_inkling_small_nvfp4.py:10,51`).
  - Pennyroyal d00d88ef has Inkling code (`server_args.py:2093,3933,4834-4850`, including `InklingForConditionalGenerationMTP`) but no README/RESULTS/CHANGES claim for Inkling on sm120. The vLLM recipe lists B300/GB300 TP1, B200/GB200/H200 TP2, or H100 TP4, with no RTX Pro 6000.
- **Pruned/compressed versions** (HF API byte sums; none usable):

| Repo | Size | Notes |
|---|---|---|
| miweru/Inkling-Small-REAP-137B-A12B-de-GGUF | IQ4_XS 72.0 GB, Q4_K_M 81.5 GB | 128 of 256 routed experts removed per layer; ~4M-token calibration, 56% German; **vision/audio dropped**; needs a llama.cpp build with Inkling support (PR branch); probe loss +0.72 nats English prose, +0.68 Python |
| pipenetwork / bielquant Inkling-Small-MLX-REAP25-4bit | 112.2 GB (both) | MLX (Apple) |
| studioburnside REAP25 2bE / 3bE | 63.9 / 88.0 GB | MLX (Apple) |
| 0xSero EXL3 2.0bpw | 81.4 GB | Quant of the full model, not a prune. turboderp-org/exllamav3's architecture directory has no inkling file, so this likely needs a fork; engine/vision support unverified |
| sabrewing colibri-int4 / cyankiwi AWQ-INT4 | 141.2 / 163.2 GB | too big |

  - Other non-fitting quants seen: RedHatAI FP8-dynamic 270.1 GB, wstewart3 W4A8-G128 141.7 GB, INCModel2 MXFP4-CT-AutoRound 153.5 GB.
  - **No CUDA checkpoint** (SGLang/vLLM-loadable) that is pruned, 4-bit, has vision and fits under ~85 GB exists, and no pruned Inkling is on Kaggle.
- **Decode cost** [derived, uniform routing, M=40]: unpruned Inkling-Small touches ~157 experts per layer, about 91.0 GB/step at 4.5 bit or 83.5 GB at 4.125 bit. Flash-Next (279 experts per layer) is 37.1 GB at 4.5 bit or 34.0 GB at 4.125 bit. The ~34 GB in `throughput_research.md` corresponds to 4.125 bit.
  - A 50% REAP would be 109 experts per layer, about 63.4 GB/step at 4.5 bit or 58.1 GB at 4.125 bit. That is **~1.7x** Flash-Next at consistent bit widths, not 1.9x.
  - Its routed-expert weights would be ~74 GB at 4.5 bit. Non-expert BF16 weights are extra, so "fits" is marginal and unverified.
  - Pruning needs a BF16 calibration pass over the ~532 GB of weights, which we cannot do locally.
- **Verdict:** drop for this competition. It needs a new SGLang sm120 bring-up and a custom pruned and quantized build, and it has no ARC-AGI-3 evidence.

### 2. Compressed Flash-Next variants
Flash-Next is 125B total / 6B active plus 51B n-gram (PLE) params, 512 experts, 10 routed + 1 shared. [web] https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- **boristown/qwen3-8-flash-next-nvfp4-reap-448e** is on Kaggle (instance `nvfp4-reap-k448` v2, 124,217,429,278 B). [web]
  - It is byte-identical in size to HF `lee-chang-93/Qwen3.8-Flash-Next-NVFP4-REAP-k448` (124,217,429,278 B), a REAP prune of `nvidia/Qwen3.8-Flash-Next-NVFP4`. 64 of 512 routed experts are removed per layer (12.5%), 123.6 to 115.6 GiB (the card says −7.9 GiB), with 92.65% of REAP mass retained.
  - Calibration used 16.49M tokens from 681 LLM calls of an agentic trace (48% coverage of 34.2M available tokens), including 626 unique images. The card has **no downstream eval** and recommends "QA diff vs the unpruned NVFP4 baseline on held-out tasks".
  - MTP needs vLLM >= 0.30.0, a `sitecustomize.py` and `VLLM_MTP_NUM_EXPERTS=512`.
  - The only user I found is cihanatak's notebook (`competitors/kaggle/arc3-b32-reap-v030-kv14-c12-ctx48-r1/*.txt:383-438`): vLLM 0.30.0 (`vllm/vllm-openai:v0.30.0` runtime layers), `modelopt_mixed`, ctx 49,152, `--kv-cache-memory-bytes 15032385536`, MTP3, prefix caching off. **No score found.** [unverified] on Pennyroyal SGLang.
  - Counter-evidence: rellik13 pruned 25% of experts, "More room, worse play" (`rellik13-milestone2/WRITEUP.md:146`). That is 25%, not 12.5%, so it is not a direct comparison.
  - Our experts-per-token tests also moved the wrong way ([local] `arms.md:88-90`): experts16 38.69 vs 46.25, experts8 13.93 vs 17.45 on the panel, no speedup.
  - Our own streams14 test gave +15% throughput (703 tok/s) for 46.38 vs 46.25 (noise, arms.md:83), so freeing KV for streams is a weak lever.
- **lovedheart/...-NVFP4-FP8-Pruned-RTXPRO-6000** (HF only, 123,433,391,711 B across 229 files; the card text says ~135 GB): AIMER calibration-free prune, 512 to 448, needs a patched SGLang or "GARBLED OUTPUT".
  - Its GSM8K 97.27 / AIME26 98.75 table is labelled "this checkpoint", but it is most likely inherited from the unpruned card. The audit count of 294,912 routed tensors equals 48 layers x 512 experts x 12, and does not divide evenly by 448 experts. [derived]
- **Davd-b01/qwen3.8-flash-next-40b-prune-research** (HF only, 296.4 GB): a **75% prune (512 to 128 experts)** to 40.22B params, "research-only". GPQA 27.78%, MATH-500 17.8%, IFEval 37.34%, LCB v6 15.55%.
  - The card calls it "selective degradation rather than uniform collapse" and attributes part of the low scores to loading bugs and measurement artifacts. It is not clean evidence about moderate pruning.
- **lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8** (Kaggle, 180,166,090,115 B): a mirror of primitive-ai/...-mixed-NVFP4-FP8 (FP8 GDN/attention projections; the HF card says 183.7 GB, the API sums to 191.0 GB).
  - Card: 84.4 tok/s single-stream and 526.0 tok/s at concurrency 32 on an RTX PRO 6000, vs 74.4 and 483.8 for plain NVFP4 (+13% and +8.7%). Whether MTP was on is not stated. Accuracy "statistically identical" (tie band +/-1.0).
  - vLLM only, requires `VLLM_GDN_DECODE_KERNEL=triton`.
  - Decode speed did not buy score: lordhansolo's vLLM NVFP4 ran median 976 tok/s at 14 slots (peak 1,135, p90 1,056), job-wide 933 tok/s, yet scored LB 23.84 vs Franzen 27.89 at 588 tok/s (public LB values from 2026-09-30, [local] `tong-m2-comparison/kaggle/COMPARISON.md:12,200-215`).
- **local-inference-lab QAD NVFP4:**
  - `Qwen3.8-Flash-Next-NVFP4` is 106,390,811,196 B on HF and "approximately 98 GiB" per the card. It has NVFP4 experts and PLE, MXFP8 shared experts and attention projections, and QAD on 200,004,844 tokens. GPQA 89.9, AA-LCR 79.4, Tool Eval Bench 91. On GPQA it "scored higher than NVFP4 PTQ while using 9% fewer tokens on average" (no PTQ number given).
  - The `-MXFP8-CSF-QAD` repo (102,931,790,657 B, created 2026-10-04) needs vLLM `--quantization nvfp4_csf --load-format nvfp4_csf`, carries a Local Inference Lab License that prohibits re-uploads and mirrors, and has no benchmarks in its card. A sibling `-NVFP4-CSF` (102.4 GB, 2026-10-01) also exists. I did not see either on Kaggle.
- **nvidia/qwen3-8-flash-next-nvfp4** (Kaggle, official, 132,734,506,847 B) vs FP8: GPQA 92.0 to 91.5, HLE 34.7 to 35.4, tau2-Telecom 90.8 to 90.1, MMMU Pro 77.1 to 78.3, Terminal-Bench 2.1 83.3 to 82.9. Essentially lossless. RadixArk NVFP4 is also on Kaggle (keithtyser).
- **Swift 1.5 fine-tune of Flash-Next:** rellik13 found "No better per token and about 35% slower to decode" (`WRITEUP.md:147`). Pennyroyal's Swift 27B check shows 12% fewer tokens (43,599 to 38,205), not better quality ([local] `sglang-rtxpro6000/SWIFT-27B.md:47-49`).
  - **A W4A16 AutoRound of Swift Flash-Next exists** (ukisai HF repo, 177,298,109,574 B; Kaggle mirror `phuongncn/arc3-qwen38-swift-w4a16-autoround`).
  - Not verified: its quality or speed under our harness.
- **Our current W4A16:** Franzen says Intel W4A16 gave "similar quality and throughput" to RadixArk NVFP4 with more VRAM left for KV ([local] `daniel-franzen-sol/WRITEUP.md:31`).

### 3. Other candidates (one 96 GB GPU, vision + tools)
- **Qwen3.8-27B (dense, 27B):**
  - Same-card quality (HF cards): SWE-bench Pro 61.7 vs Flash-Next 62.5; GPQA Diamond 89.2 vs 91.7; ClawEval-MM avg 56.9 vs 60.4; OSWorld 2.0 partial 48.0 vs 52.3.
  - Pennyroyal 27B FP8 + DFlash2: **108.31 tok/s single, 374.98 tok/s aggregate at 4 requests** ([local] `sglang-rtxpro6000/README.md:175-181`). The later dated release is 108.75 / 390.23 (RESULTS.md:565-575). I found no 10-stream number, versus our ~789 on Flash-Next.
  - Franzen: each switch (3.6-27B to 3.8-27B to Flash-Next) "gave a substantial improvement" (WRITEUP.md:15,29). rellik13 calls Flash-Next "the strongest model that fits on one RTX PRO 6000" and "the first clear jump" (WRITEUP.md:47).
  - On Kaggle: foysalemonshanto/qwen3-8-27b-fp8-repacked-v1 (30,890,069,847 B), mikedan7/qwen3-8-27b-fp8-official, dangkhoa2016/z-lab-qwen3-8-27b-dflash2. [web]
- **Gemma 4 31B** (dense 30.7B, text+image, thinking, 256K ctx, Apache-2.0, per the HF card): Google QAT w4a16 (`google/gemma-4-31B-it-qat-w4a16-ct`) and `nvidia/Gemma-4-31B-IT-NVFP4` exist. Kaggle `google/gemma-4` has `gemma-4-31b-it-qat-w4a16-ct` (23,297,590,856 B).
  - MS1 #2 Reki and #3 Murad ("forge") both ran Gemma-4-31B locally on the official GPT-OSS-120B template. The blog gives no scores. [web] https://arcprize.org/blog/arc-prize-2026-milestone-1
  - All three MS2 winners (LB 27.89 / 23.84 / 22.53 on 2026-09-30) served Flash-Next ([local] `tong-m2-comparison/kaggle/SUMMARY.md:3-9`). I found no ARC-AGI-3 evidence that Gemma beats Flash-Next.
- **Gemma-4-26B-A4B** (MoE, 4B active, vision): no ARC-AGI-3 evidence. **Kaggle `google/gemma-4` has it** as Transformers (`gemma-4-26b-a4b-it`, 51,644,324,361 B), QAT-unquantized, GGUF q4_0 (15,634,217,687 B) and Flax. It is not GGUF-only.
- **Whittle-Next-27B-A3B** (Kaggle, davidaylward): its card says "not a finished, general-purpose model"; llama.cpp GGUF only (Q4_K_M 16.8 GB, Q8_0 29.3 GB). Skip.
- **Tufa (LB 55.89):** the score comes from local notes only (`research-sweep.md:64`, `arms.md:14`) and a web search did not confirm it. Model not disclosed. Web summaries saying "Qwen 3.6 27B" describe Milestone 1 (the Duck harness), not this run. [unverified]

### Corrections to the brief
- Tufa's MS1 winner used Qwen **3.6**-27B FP8, not 3.8 (https://arcprize.org/blog/arc-prize-2026-milestone-1).
- Franzen's BF16 PLE offload is **95.37 GiB** (the COMPARISON.md:106 cell says "95.37 GB" but the value is GiB; it is 102.4e9 B). Franzen keeps it deliberately (WRITEUP.md:33). The ~48 GB figure is the FP8 PLE: 47.68 GiB = 51,200,245,760 B (NVME-PLE.md:8), used by sirikilohit together with a 48 GB host cache tier (COMPARISON.md:106).

### Bottom line
1. Inkling-Small: no.
2. A model swap is unlikely to be the cheapest lever. Flash-Next stays ahead of the alternatives on the evidence found.
3. REAP-448 is the only pruned Flash-Next variant on Kaggle and has no quality or score evidence. A 25% prune hurt play (rellik13), and extra KV/streams did not move our score.
4. If we try one thing, use a paired 25-game A/B of one of these, otherwise keep W4A16:
   - the lordhansolo mixed NVFP4+FP8 (+13% single-stream, vLLM only);
   - REAP-448 on vLLM 0.30;
   - the Swift W4A16 AutoRound on the unchanged Pennyroyal stack, which is now the cheapest option because it reuses our engine (quality unproven).

## engines (corrected)

**Bottom line**
- **vLLM is the only credible alternative to the current SGLang stack.** Two variants matter:
  - lordhansolo's overlay, already shown to run offline on Kaggle.
  - local-inference-lab's "Karmic Kraken" image, with sm120 `b12x` kernels. It has the best same-GPU numbers but has not been run on Kaggle (my inference: nothing local or online shows it).
- **TensorRT-LLM, llama.cpp, ExLlamaV3, a custom engine and a pruned Inkling Small are not worth pursuing now.** They are unmeasured for our workload or have open blockers, not unsupported.
- **The score gap (23.84 vs 27.89) says little about engines.** The harnesses differ too much (list below). I could not find "28.32" anywhere in the local files.

**Same-GPU vLLM vs SGLang evidence** (the only matched comparison I found)
Source: https://github.com/local-inference-lab/rtx6kpro/blob/master/models/qwen38-flash-next-community-r35.md ("Historical stock-clock engine and Sieve comparisons"). Setup: one RTX Pro 6000 Workstation at 600 W, stock clocks, the same NVFP4 checkpoint, MTP3, context 0, T=1/top-p 0.95/top-k 20.

| Case | vLLM R28.1 | SGLang FlashInfer / NEXTN3 | vLLM vs SGLang |
|---|---|---|---|
| C1, tok/s | 172.8 | 152.5 | +13.3% |
| C4, tok/s | 485.4 | 446.8 | +8.6% |
| C8 / C16, tok/s | 632.4 / 944.5 | not measured (SGLang capped at 4 requests) | n/a |
| 32K prefill, tok/s | 14,813 | 15,583 | SGLang +5.2% |

- The source itself says "The table compares those deployments, not isolated engine implementations". Recurrent-state precision, draft-head precision and token budgets differ. vLLM ran on physical GPU0 and SGLang on GPU1, without swapping cards.
- The SGLang here is a community build on NVFP4, not Pennyroyal v2.5.3 on W4A16. A later R29 repeat gave vLLM C1 173.66→177.27 (+2.08%).
- Rough parity check (derived, not measured): Franzen's bs1 step is 15.2 ms (`throughput_research.md`). With an assumed accept length of about 2.7, that gives about 178 tok/s at bs1. It is also in line with vLLM NVFP4 at C4: 4x2.7/22.8 ms is about 474, against vLLM's 485.4.

**Engine matrix** (sm120, one GPU)

| Engine | Hybrid GDN+MoE (Flash-Next) | MTP | Prefix cache on hybrid | Images / tools | Offline Kaggle | Measured on RTX Pro 6000 |
|---|---|---|---|---|---|---|
| Pennyroyal SGLang (baseline) | yes | yes, NEXTN 3 | yes (radix, `extra_buffer`) | yes / `qwen3_coder` | yes, in use | W4A16: 789 tok/s at 10 streams; clean bs10 median 826 |
| lordhansolo vLLM 0.29.1rc1 nightly e975732 + overlay | yes (`qwen4_exp`) | yes, MTP3 | yes, `mamba_cache_mode` align plus his patches | yes / `qwen3_coder` | yes, shipped as image-layer datasets (`lordhansolo/vllm-main-e975732-arc3`, 7,934,814,402 B) | Kaggle logs: peak/p90/median 1,135/1,056/976 at 14 slots (10 s averages) |
| local-inference-lab vLLM "Karmic Kraken" (`karmic-kraken-beta`; CUDA 13.4.1, PyTorch 2.14, `b12x`; **vLLM version not stated in the docs I read**) | yes | yes, MTP3 | yes (GPU-only prefix cache; align) | yes / `qwen3_xml` | possible via layer unpack, untested (unverified); needs a driver reporting CUDA 13.4+ vs Kaggle driver unknown | **Max-Q at 325 W**, VRAM +6000: C1 173.2, C8 698.4 (713.7 on restart), 32K prefill 12,073. Workstation 600 W +6000 (R33 packaged image, r35.md): C1 199.69, C8 732.85 |
| TensorRT-LLM 1.3.0rc29 (NGC image) | yes, official guide | yes, 3 drafts, exact rejection sampling for T>0 | yes, but needs `mamba_state_config` snapshots (periodic 256, or offset 0 from prompt end) | yes / `--tool_parser qwen3` | NGC container; size and Kaggle driver unverified (guide: CUDA driver 575+) | none for Flash-Next. Only a Qwen3-Next-80B-A3B NVFP4 on TRT-LLM 1.3.0rc10, RTX PRO 6000 Server Edition, CUTLASS: 97 tok/s |
| llama.cpp (arch merged in #27742, in b10660) | yes | **exists but not production-ready**: draft-MTP for qwen4exp (PRs #27956 and #28243, both closed with merge status unverified), with open bugs (#28286: `--parallel > 1` cross-slot content contamination, seen on a ROCm fork; #29148: MTP fails to load) | n/a | vision unverified | build from source | CPU / Apple only |
| ExLlamaV3 | **yes**: README lists Qwen 3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`, vision column ticked); `architecture/qwen4_exp.py` and `qwen4_exp_mtp.py` exist; many EXL3 Flash-Next quants on HF (e.g. `turboderp/Qwen3.8-Flash-Next-exl3`, 2026-08-31) | yes (mtp module exists; behavior unmeasured) | unverified | unverified | unverified | none found for 10-stream serving |
| Custom single-model engine (NInfer, for comparison) | Qwen3.6/3.8-27B (and Qwen3.6-35B-A3B), requires an RTX 5090; Flash-Next not documented | yes | yes | yes | n/a | 35B-A3B MTP3 at C=8: 1,380.7 tok/s (68.0% accept) on a 5090 |

Sources:
- lordhansolo: `tong-m2-comparison/kaggle/COMPARISON.md:206-215`.
- Karmic Kraken: https://github.com/local-inference-lab/rtx6kpro/blob/master/models/qwen38-flash-next.md and `benchmarks/karmic-kraken-serving.md` ("Conditions": Max-Q, VRAM +6000, 325 W power limit).
- TRT-LLM guide: https://nvidia.github.io/TensorRT-LLM/latest/deployment-guide/deployment-guide-for-qwen3.8-flash-next-on-trtllm.html
- TRT-LLM bs=1 number: https://www.morphllm.com/blog/morph-decode.md
- llama.cpp: https://newreleases.io/project/github/ggml-org/llama.cpp/release/b10660 ("model: add Qwen3.8-Flash-Next (qwen4exp) (#27742)"), https://github.com/ggml-org/llama.cpp/issues/28286, https://huggingface.co/RockTalk/Qwen3.8-Flash-Next-MLX-4bit. The RockTalk card only says llama.cpp cannot run its gmlx MTP drafter, and says nothing about `-np 1`.
- ExLlamaV3: https://github.com/turboderp-org/exllamav3 (README model table).
- NInfer: https://github.com/Neroued/ninfer

**TensorRT-LLM blockers on sm120**
- **Memory.** The guide lists NVFP4 at 1 GPU with 100 GB GPU memory and 96 GiB host RAM (block-FP8: 145 GB). A 96 GB card is under that figure, though it is unclear whether it is a hard floor.
- **Kernels.** The guide says to use the `TRTLLM` MoE backend for NVFP4.
  - TRT-LLM issue #11799 is about trtllm-gen **FMHA (attention)** cubins. It was closed with "TRTLLM-Gen is designed mainly considering support of sm100/103 ... no plan to SM120/121". It does not directly cover the MoE backend.
  - Pennyroyal `BACKENDS.md:96-103` hit `CUDA_ERROR_NO_BINARY_FOR_GPU` on the QSA TRTLLM-Gen attention cubin in SGLang/FlashInfer.
  - Whether the TRTLLM or CUTLASS MoE backend runs this mixed checkpoint on sm120 is unverified.
- **CI and bugs.**
  - PR #18585 (Flash-Next support, merged 2026-09-04) adds B300 unit tests and a GB300 FP8 1-GPU MTP3 TRTLLM PLE-offload accuracy test. There are no sm120 entries and no NVFP4 test.
  - Issue #19623 (open, 2026-09-24) reports a ~6 s per-request TTFT floor with `nvidia/Qwen3.8-27B-NVFP4` (a Qwen3.5-architecture hybrid-GDN 27B) on 4x RTX 5060 Ti, 1.3.0rc29, against 0.26-0.30 s on SGLang. It is not our exact case.
  - PR #19801 (open), a multimodal scheduling stall on the KV cache V2 scheduler, is also relevant.
- **Defaults and docs.** The model defaults to eager execution, so CUDA graphs must be configured. The guide gives no throughput numbers.
- **Upside.** Exact rejection sampling for T>0, and a prompt-end snapshot that fits multi-turn reuse. Neither outweighs the blockers.

**lordhansolo's vLLM vs Pennyroyal SGLang for our workload**

Likely better. Everything here is inferred from config, and none of it is isolated:
1. **Prefill does not freeze decode.** He launches with `--async-scheduling --enable-chunked-prefill --max-num-batched-tokens` (`kaggle.py:1356-1362`, with 2,048 per COMPARISON.md:45; the file default is 8192, `kaggle.py:35`). SGLang with spec decoding rejects mixed chunks (JPE `server_args.py:10103-10106`), and our control run loses 21.4% of wall time to prefill stalls.
   - Only the fixed part, about 0.28 s per prefill batch (~820 s, 10.9% of wall), is plausibly removable. The token part (~730 s, 9.7%) is GPU work either way.
   - Whether vLLM avoids that fixed cost is unmeasured.
2. **Bigger KV pool.** 1,417,100 tokens vs 1,011,264 (`COMPARISON.md:212`).
   - 19.69 GB for KV + Mamba, vs 15.94 GB.
   - `embed_tokens` (1.18 GB BF16) pinned in host RAM.
   - 0.46 GB of CUDA graphs vs 1.14 GB.
   - Mamba states share the KV pool instead of a fixed 60-slot pool.
3. **Higher reported draft acceptance.** Median accepted draft length is 2.9 vs 2.7 (COMPARISON.md:211). This is confounded by his 32k draft vocab built from 125 transcripts, T=0.6 vs 0.7, and a different MTP head.
4. **Kernel work for batch 10 to 16.**
   - sm120 BF16 GEMMs for M=40..64.
   - Exact CUDA graphs at 44 and 52 tokens.
   - GDN RecoverSSM on (`kaggle.py:1084`).
   - Draft-vocab argmax reduction.
   - Nine vLLM-main PRs ported.
   - Not a gain for us: Pennyroyal already ships RecoverSSM, and our graph list `[1,2,4,7,8,9,10]` only pads bs 3/5/6.
5. **Scales past 10 streams.** Median 976 at 14 slots.
   - His per-stream rate is about 70 tok/s (976/14), against our clean bs10 of about 83 (826/10).
   - So SGLang W4A16 at 10 streams is not clearly slower per stream; vLLM's edge is capacity.

Likely worse or risky:
1. **Unbenchmarked overlay.** `VALIDATION.md:77` says no RTX Pro 6000 server benchmark or competition wave had been run for the newest bundle. The README says the newest runtime has not beaten the older scored one.
2. **Prefix reuse is weaker or unproven.** Align-mode checkpoints (match unit 128) share the pool with live requests, and his median hit rate is 91%. That is a different harness from our ~93.6% cumulative hit rate.
   - For comparison, lil reports "effective hybrid pages are 3,008 tokens". How that affects checkpoint granularity is unverified.
3. **Overlay source is available, just not in the local copy.** `OMITTED.md` omits the tar locally, but the public Kaggle dataset `lordhansolo/vllm-main-e975732-arc3` lists `arc3_vllm_main_e975732_arc3_overlay.tar.blob` (1,771,520 B) plus `PATCH_README.md`, `VALIDATION.md` and `apply_vllm_main_e975732_arc3.py`. I did not download it.
4. **Checkpoint.** NVFP4 mixed (primitive-ai) with MoE via `flashinfer_cutlass`. NVFP4 MoE on sm120 was broken in older stacks (CUTLASS TMA-WS issue, https://discuss.vllm.ai/t/sm120-rtx-pro-6000-nvfp4-moe-performance-report-qwen3-5-397b/2536). It works in his run.
5. **Startup.** The vLLM image unpacks to ~19.5 GB under `/tmp` (`kaggle.py:838-842`), about 114 s. Server launch to ready is 430 s, against our 478 s.

Harness confounders in the 23.84 vs 27.89 gap (`COMPARISON.md:41-48,73-74,128`):
- 14 vs 10 slots.
- 256 px (~66 tokens per board) vs 640 px (~402).
- Prefill chunk 2,048 vs 8,192.
- T=0.6 vs 0.7.
- 3,918 s per-game runtime with 14 games live, vs a 31,920 s shared pool with a priority gate over all 110.
- Context target 81.5k (81,536) vs ~59k trim.

Franzen's harness talks plain OpenAI-compatible endpoints and already branches for vLLM (`kaggle.py:81-83`, `openai_compat.py:~92-94`), so swapping the engine under it is cheap.

**Inkling Small, pruned** (model-switch idea)
- It exists. Thinking Machines Lab, announced 2026-07-30, 276B total (README) / 12B active, 42 layers, 256 experts (6 routed + 2 shared), vision and audio input, BF16 and NVFP4 checkpoints. **It ships an MTP head** (config.json `mtp_config`, 8 next-token layers; `mtp.safetensors`) and has a DSpark speculator (`RadixArk/Inkling-Small-DSpark`, 1.39B params, mean accept ~3.7-3.8). The SGLang cookbook documents both MTP and DSpark (HF model card; SGLang cookbook `Inkling-Small.mdx:93,308-312`).
- Pruned builds found:
  - MLX 4-bit for Apple (`pipenetwork/` and `bielquant/Inkling-Small-MLX-REAP25-4bit`, 192 of 256 experts) and MLX 2/3-bit REAP25 from studioburnside.
  - miweru's GGUF REAP-137B (128 of 256 experts removed, text-only, IQ4_XS 72.0 GB), which is llama.cpp-only.
  - I found no CUDA NVFP4 or INT4 pruned build.
- 266B params at NVFP4 is far over 96 GB. Fitting needs about 50% pruning plus host offload (my arithmetic, unverified).
- It has 2x Flash-Next's active parameters. Per-stream decode would be well below Flash-Next even with MTP: about 1.7x the expert bytes per step at 50% prune (derived, not measured).
- Pennyroyal carries Inkling kernels, but its Inkling NVFP4 test is under `8-gpu-models` (runners 8-gpu-h200 and 8-gpu-b200).
- Prior pruning evidence on Flash-Next is negative: 25% pruning played worse (rellik13), and sirikilohit's write-up lists expert pruning as a failure (`COMPARISON.md:~192-196`).
- Verdict: no.

**Suggested next steps**
1. **Controlled A/B.** Run Franzen's harness unchanged (10 streams, 640 px) on lordhansolo's public datasets: `lordhansolo/vllm-main-e975732-arc3` plus model `lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8`. Compare against the SGLang control. This isolates the engine effect, and the upside is bounded by the roughly 11% fixed prefill cost plus kernel gains.
2. **Cheaper alternative.** Port the ideas into Pennyroyal instead: the `_MAX_M` 32 to 64 lever from `throughput_research.md`, and the unexplained ~0.28 s per prefill batch.
3. **Probe Karmic Kraken** only if step 1 is positive. It is the best sm120 stack (`b12x`), but Kaggle driver compatibility and image size are unknown.

**Unverified:** the 28.32 score; Kaggle host RAM and GPU clocks; the vLLM version inside Karmic Kraken; the TRT-LLM TRTLLM/CUTLASS MoE backend on sm120 for this checkpoint; any Kaggle run of Karmic Kraken or TRT-LLM; ExLlamaV3 and llama.cpp MTP performance with concurrent slots on CUDA; the cause of the unexplained SGLang per-batch fixed cost.

Local scratch copies of fetched docs: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/eng/` and `.../scratchpad/fc/`.

## Corrections

**Wrong or unsupported claims, now fixed:**
1. **Inkling-Small release date.** The engines section said 2026-07-31. The news page says July 30, 2026 (HF repo created 2026-07-27, last modified 2026-07-31).
2. **"MTP is not documented" for Inkling-Small.** Wrong. `config.json` has `mtp_config` (8 next-token layers), `mtp.safetensors` exists, the SGLang cookbook documents MTP and DSpark, and Pennyroyal has `InklingForConditionalGenerationMTP`. The "no" verdict stands on size and the missing ARC-AGI-3 evidence, not on MTP.
3. **"ExLlamaV3: no Flash-Next or GDN support found."** Wrong. The README lists Qwen 3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`, vision ticked), `qwen4_exp.py` and `qwen4_exp_mtp.py` exist, and many Flash-Next EXL3 quants exist on HF.
4. **llama.cpp "MTP: no".** The RockTalk card only says llama.cpp cannot run its gmlx drafter. Qwen4Exp MTP PRs exist (#27956, #28243) and a `draft-mtp` spec type is in use, with open bugs (#28286, #29148). The "`-np 1` advised" claim is not in the RockTalk card (removed).
5. **"No W4A16 AutoRound of Swift Flash-Next is known."** Wrong. HF `ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AutoRound` (177,298,109,574 B) is mirrored on Kaggle as `phuongncn/arc3-qwen38-swift-w4a16-autoround`.
6. **Gemma-4-26B-A4B "Kaggle has only a GGUF."** Wrong. `google/gemma-4` on Kaggle has Transformers (51,644,324,361 B), QAT-unquantized, GGUF and Flax variants.
7. **"Karmic Kraken vLLM 0.29+."** Unsupported: the docs name CUDA 13.4.1 and PyTorch 2.14 but no vLLM version. 0.29.1rc1 is lordhansolo's nightly. The Karmic Kraken numbers are Max-Q at 325 W. The 732.85 C8 is the R33 packaged image at 600 W, not labelled Karmic Kraken.
8. **Pruned Inkling "only MLX 4-bit" (engines)** contradicted the models section. There is also miweru's GGUF REAP-137B and MLX 2/3-bit REAP25 builds. 0xSero EXL3 is a quant, not a prune.
9. **Decode-cost arithmetic.** The ~33 GB Flash-Next step needs 4.125 bit while the Inkling figures used 4.5 bit. At consistent bits a 50% REAP is ~1.7x Flash-Next, not 1.9x (Flash-Next 37.1 GB vs 63.4 GB at 4.5 bit; 34.0 vs 58.1 at 4.125 bit). "Fits" for ~74 GB of expert weights is marginal once non-expert BF16 weights are added.
10. **Host RAM cost of HiCache.** The ~48 GiB PLE is the FP8 table. Our BF16 PLE is 95.37 GiB (COMPARISON.md:106; the cell says GB but the value is GiB), so the host cost is ~32 GB + 95.37 GiB.
11. **HiCache "untested with extra_buffer."** `extra_buffer` was in issue #3's recipe. Untested are Marlin W4A16, the INT4 g32 draft and Franzen's patch 04. Kaggle precedent was missing: sirikilohit ran a 48 GB host tier (COMPARISON.md:100-110).
12. **NVMe PLE.** BF16 staging is supported in v2.5.3 (CHANGES.md:35-39). The saving for us would be ~95.37 GiB, and the −13.8% was measured on the FP8 table.
13. **"Our template honoring `reasoning_effort` (unverified)."** Verified: the Intel/Qwen `chat_template.jinja` accepts xhigh (default), medium and low only and raises on anything else. We run at xhigh today. The 80.4% and 84.74% numbers are 27B-only.
14. **Issue #4 numbers.** −22.9% is the native-NEXTN arm pair (291.0→224.5). The FR-Spec arms were 319.9→235.6 but confounded by accept 1.00. The reporter's deviations (RunAI/S3, no NIXL, driver 580.x) were added.
15. **Overlap-plan-stream failure** was a community TP2/EP2 report, not TP1.
16. **Davd-b01 prune** is 75% (512→128 experts, research-only, card cites loading and measurement artifacts), so it is weak evidence about moderate pruning.
17. **lovedheart benchmark table.** The suspicion that it was inherited from the unpruned card is supported by tensor-count arithmetic (294,912 = 48 x 512 x 12).
18. **TRT-LLM issue #11799** is about FMHA cubins and was closed, not about the MoE backend. PR #18585 CI covers B300/GB300 (an FP8 test), not an NVFP4 test or sm120.
19. **Line-number fixes:**
    - `serve-flash-next-frspec.sh:9`, not :8.
    - "Prefer a fixed image digest" is at `build_bundle_pennyroyal.sh:1296`, not 1286-1290.
    - Sirikilohit's pruning-failure line is COMPARISON.md:~192-196, not `research-sweep.md:75`.
    - Pennyroyal CI cites are `pennyroyal-container.yml:31-32,72-77`.

**Added evidence that resolves earlier "unverified" items:**
- `dfranzen/pennyroyal-v253` is 6,640,272,567 B.
- `/kaggle/working` is a 20 GB volume and the RTX PRO 6000 host showed ~1.1 TB free on the root overlay (lordhansolo `kaggle.py:838-841`).
- The lordhansolo overlay source tar is on Kaggle.
- Franzen's WRITEUP.md:33 gives a lower bound on host RAM.

**Verified unchanged:**
- Pennyroyal release and commit facts.
- All Pennyroyal RESULTS/CHANGES/NVME-PLE/COMMUNITY-RESULTS numbers, including MXFP8 +28.28/26.46/15.76/14.96/39.01%.
- CI run durations (27m28s, 30m25s, 25m28s).
- fz-arms.log Python 3.12 and nvcc 13.0.88.
- Inkling-Small config, benchmarks, ARC-AGI-1/2 scores, "not yet evaluated" on ARC-AGI-3, Kaggle byte sizes.
- The pruned-repo sizes.
- The REAP-448, local-inference-lab, primitive-ai and nvidia numbers.
- The Qwen3.8 card numbers.
- The vLLM-vs-SGLang same-GPU table.
- The lordhansolo harness confounders.
- Milestone 1 winners and models.

**Not verifiable:** Tufa LB 55.89 (local notes only), "28.32" (absent locally), Kaggle internet and RAM limits, Kaggle phone-verification (secondary sources only).