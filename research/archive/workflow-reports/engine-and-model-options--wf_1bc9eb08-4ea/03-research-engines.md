# engine-and-model-options: research:engines

Workflow run `wf_1bc9eb08-4ea`, phase Research. Research Pennyroyal build optimization, alternative (pruned) models like Inkling Small, and alternative inference engines; verify and recommend

---

## Engines for ARC-AGI-3 on 1x RTX Pro 6000 (sm120): read-only research

**Bottom line**
- **vLLM is the only credible alternative.** Two variants matter:
  - lordhansolo's overlay, already shown to run offline on Kaggle.
  - local-inference-lab's "Karmic Kraken" image, with sm120 `b12x` kernels. It has the best same-GPU numbers but has not been run on Kaggle (my inference: nothing local or online shows it).
- **TensorRT-LLM, llama.cpp, ExLlama, a custom engine and a pruned Inkling Small are not worth pursuing now.** Reasons are in the table and last section.
- **The score gap (23.84 vs 27.89) says little about engines.** The harnesses differ too much (list below). I could not find "28.32" anywhere in the local files (unverified).

**Same-GPU vLLM vs SGLang evidence** (the only matched comparison I found)
Source: https://github.com/local-inference-lab/rtx6kpro/blob/master/models/qwen38-flash-next-community-r35.md, the "Historical stock-clock" section. Setup: one RTX Pro 6000 Workstation at 600 W, stock clocks, the same NVFP4 checkpoint, MTP3, context 0, T=1/top-k 20.

| Case | vLLM R28.1 | Community SGLang | vLLM vs SGLang |
|---|---|---|---|
| C1, tok/s | 172.8 | 152.5 | +13.3% |
| C4, tok/s | 485.4 | 446.8 | +8.6% |
| C8 / C16, tok/s | 632.4 / 944.5 | not measured (SGLang capped at 4 requests) | n/a |
| 32K prefill, tok/s | 14,813 | 15,583 | SGLang +5.2% |

- The source itself says it "compares those deployments, not isolated engine implementations". Recurrent-state precision, draft-head precision and token budgets differ.
- The SGLang here is a community build on NVFP4, not Pennyroyal v2.5.3 on W4A16.
- Rough parity check (derived, not measured): Franzen's bs1 step is 15.2 ms (`throughput_research.md`). With an assumed accept length of about 2.7, that gives about 178 tok/s at bs1. It is also in line with vLLM NVFP4 at C4: 4x2.7/22.8 ms is about 474, against vLLM's 485.4.

**Engine matrix** (sm120, one GPU)

| Engine | Hybrid GDN+MoE (Flash-Next) | MTP | Prefix cache on hybrid | Images / tools | Offline Kaggle | Measured on RTX Pro 6000 |
|---|---|---|---|---|---|---|
| Pennyroyal SGLang (baseline) | yes | yes, NEXTN 3 | yes (radix, `extra_buffer`) | yes / `qwen3_coder` | yes, in use | W4A16: 789 tok/s at 10 streams; clean bs10 median 826 |
| lordhansolo vLLM e975732 + 32-file overlay | yes (`qwen4_exp`) | yes, MTP3 | yes, `mamba_cache_mode` align plus his patches | yes / `qwen3_coder` | yes, he shipped it as image-layer datasets | Kaggle logs: peak/p90/median 1,135/1,056/976 at 14 slots (10 s averages) |
| local-inference-lab vLLM "Karmic Kraken" (`b12x` MoE, dense and GDN kernels for sm120) | yes, vLLM 0.29+ | yes, MTP3 | yes (GPU-only prefix cache; align) | yes / `qwen3_xml` | possible via layer unpack, untested (unverified); CUDA 13.4 image vs Kaggle driver unknown | Max-Q, VRAM +6000: C1 173.2, C8 698.4 (713.7 on restart), 32K prefill 12,073. Workstation +6000: C8 732.85 |
| TensorRT-LLM 1.3.0rc29 (NGC image) | yes, official guide | yes, 3 drafts, exact rejection sampling | yes, but needs `mamba_state_config` snapshots (periodic 256, or offset 0 from prompt end) | yes / `--tool_parser qwen3` | NGC container; size and Kaggle driver unverified | none for Flash-Next. Only a Qwen3-Next-80B NVFP4 on TRT-LLM 1.3.0rc10 at bs=1: 97 tok/s on CUTLASS |
| llama.cpp b10660 | arch merged (#27742) | no: MTP drafter "cannot" run (RockTalk model card) | n/a | vision "unverified"; `-np 1` advised | build from source | CPU / Apple only |
| ExLlamaV3 | no Flash-Next or GDN support found | n/a | n/a | n/a | n/a | none |
| Custom single-model engine (NInfer, for comparison) | Qwen3.6/3.8-27B and 35B-A3B only, RTX 5090 | yes | yes | yes | n/a | 35B-A3B MTP3 at C=8: 1,380.7 tok/s on a 5090 |

Sources:
- lordhansolo: `tong-m2-comparison/kaggle/COMPARISON.md:206-215`.
- Karmic Kraken: https://github.com/local-inference-lab/rtx6kpro/blob/master/models/qwen38-flash-next.md ("Measured performance") and `benchmarks/karmic-kraken-serving.md`.
- TRT-LLM guide: https://nvidia.github.io/TensorRT-LLM/latest/deployment-guide/deployment-guide-for-qwen3.8-flash-next-on-trtllm.html
- TRT-LLM bs=1 number: https://www.morphllm.com/blog/morph-decode.md
- llama.cpp: https://newreleases.io/project/github/ggml-org/llama.cpp/release/b10660 and https://huggingface.co/RockTalk/Qwen3.8-Flash-Next-MLX-4bit
- NInfer: https://github.com/Neroued/ninfer

**TensorRT-LLM blockers on sm120**
- **Memory.** The guide lists NVFP4 at 1 GPU with 100 GB GPU memory and 96 GiB host RAM. A 96 GB card is under that figure, though it is unclear whether it is a hard floor.
- **Kernels.** The guide says to use the `TRTLLM` MoE backend for NVFP4. trtllm-gen cubins exist only for SM100/SM103 (https://github.com/NVIDIA/TensorRT-LLM/issues/11799; Pennyroyal `BACKENDS.md:96-103` hit `CUDA_ERROR_NO_BINARY_FOR_GPU`). Whether the CUTLASS backend runs this mixed checkpoint on sm120 is unverified.
- **CI and bugs.** CI entries for this model are GB300 (PR #18585). Issue #19623 (open, 2026-09-24) reports a ~6 s per-request TTFT floor on sm120 with a Qwen3.5 hybrid-GDN 27B on 1.3.0rc29, against 0.26-0.30 s on SGLang. That is a 27B dense model on 4x 5060 Ti, so it is not our exact case. PR #19801, a multimodal scheduling stall on the KV cache V2 scheduler, is also open.
- **Defaults and docs.** The model defaults to eager execution, so CUDA graphs must be configured. The guide gives no throughput numbers.
- **Upside.** Exact rejection sampling for T>0, and a prompt-end snapshot that fits multi-turn reuse. Neither outweighs the blockers.

**lordhansolo's vLLM vs Pennyroyal SGLang for our workload**

Likely better. Everything here is inferred from config, and none of it is isolated:
1. **Prefill does not freeze decode.** He launches with `--async-scheduling --enable-chunked-prefill --max-num-batched-tokens 2048` (`kaggle.py:1356-1362`). SGLang with spec decoding rejects mixed chunks (JPE `server_args.py:10103-10106`), and our control run loses 21.4% of wall time to prefill stalls.
   - Only the fixed part, about 0.28 s per prefill batch (~820 s, 10.9% of wall), is plausibly removable. The token part (~730 s, 9.7%) is GPU work either way.
   - Whether vLLM avoids that fixed cost is unmeasured.
2. **Bigger KV pool.** 1,417,100 tokens vs 1,011,264 (`COMPARISON.md:212`).
   - 19.69 GB for KV + Mamba, vs 15.94 GB.
   - `embed_tokens` pinned in host RAM.
   - 0.46 GB of CUDA graphs vs 1.14 GB.
   - Mamba states share the KV pool instead of a fixed 60-slot pool.
3. **Higher reported draft acceptance.** Median accepted draft length is 2.9 vs 2.7. This is confounded by his 32k draft vocab built from 125 transcripts, T=0.6 vs 0.7, and a different MTP head.
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
3. **Source not local.** The overlay tar is omitted (`OMITTED.md`), so I could not diff it or confirm the M=40..64 kernel.
4. **Checkpoint.** NVFP4 mixed (primitive-ai) with MoE via `flashinfer_cutlass`. NVFP4 MoE on sm120 was broken in older stacks (CUTLASS TMA-WS issue, https://discuss.vllm.ai/t/sm120-rtx-pro-6000-nvfp4-moe-performance-report-qwen3-5-397b/2536). It works in his run.
5. **Startup.** The vLLM image unpacks to ~19.5 GB under `/tmp` (`kaggle.py:840`), about 114 s. Server launch to ready is 430 s, against our 478 s.

Harness confounders in the 23.84 vs 27.89 gap (`COMPARISON.md` sections 1, 2, 4):
- 14 vs 10 slots.
- 256 px (~66 tokens per board) vs 640 px (~402).
- Prefill chunk 2,048 vs 8,192.
- T=0.6 vs 0.7.
- 3,918 s per-game cap with 14 games live, vs a priority gate over all 110.
- Context target 81.5k vs ~59k trim.

Franzen's harness talks plain OpenAI-compatible endpoints and already branches for vLLM (`kaggle.py:81`, `openai_compat.py:94`), so swapping the engine under it is cheap.

**Inkling Small, pruned** (model-switch idea)
- It exists. Thinking Machines Lab, released 2026-07-31, 276B total / 12B active, 42 layers, 256 experts (6 routed + 2 shared), vision and audio input, BF16 and NVFP4 checkpoints. MTP is not documented (https://huggingface.co/thinkingmachines/Inkling-Small; https://openrouter.ai/models/thinkingmachines/inkling-small).
- Pruned builds I found are only MLX 4-bit for Apple (`bielquant/Inkling-Small-MLX-REAP25-4bit`, 192 of 256 experts). I found no CUDA NVFP4 or INT4 pruned build.
- 266B params at NVFP4 is far over 96 GB. Fitting needs about 50% pruning plus host offload (my arithmetic, unverified).
- It has 2x Flash-Next's active parameters and no documented MTP, so per-stream decode would be well below Flash-Next (derived, not measured).
- Pennyroyal carries Inkling kernels, but its Inkling NVFP4 test is under `8-gpu-models`.
- Prior pruning evidence on Flash-Next is negative: 25% pruning played worse (rellik13), and sirikilohit lists expert pruning as a failure (`research-sweep.md:75`, `COMPARISON.md`).
- Verdict: no.

**Suggested next steps**
1. **Controlled A/B.** Run Franzen's harness unchanged (10 streams, 640 px) on lordhansolo's public datasets: `lordhansolo/vllm-main-e975732-arc3` plus model `lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8`. Compare against the SGLang control. This isolates the engine effect, and the upside is bounded by the roughly 11% fixed prefill cost plus kernel gains.
2. **Cheaper alternative.** Port the ideas into Pennyroyal instead: the `_MAX_M` 32 to 64 lever from `throughput_research.md`, and the unexplained ~0.28 s per prefill batch.
3. **Probe Karmic Kraken** only if step 1 is positive. It is the best sm120 stack (`b12x`), but Kaggle driver compatibility and image size are unknown.

**Unverified:** the 28.32 score; Kaggle host RAM and GPU clocks; the TRT-LLM CUTLASS backend on sm120 for this checkpoint; any Kaggle run of Karmic Kraken or TRT-LLM; the cause of the unexplained SGLang per-batch fixed cost.

Local scratch copies of fetched docs: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/eng/`
