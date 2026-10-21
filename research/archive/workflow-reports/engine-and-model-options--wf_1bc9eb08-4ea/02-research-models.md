# engine-and-model-options: research:models

Workflow run `wf_1bc9eb08-4ea`, phase Research. Research Pennyroyal build optimization, alternative (pruned) models like Inkling Small, and alternative inference engines; verify and recommend

---

## Alternative models for ARC-AGI-3 (read-only research, 2026-10-04)

Evidence tags: [web] fetched this session, [local] file:line in the repo, [derived] my arithmetic, [unverified].

### 1. "Inkling Small" exists and is NOT a fit for one 96 GB card
- **Who/when:** Thinking Machines Lab, released 2026-07-30. Apache-2.0. [web] https://huggingface.co/thinkingmachines/Inkling-Small , https://thinkingmachines.ai/news/inkling-small/
- **Architecture** (HF config.json): sparse MoE, 276B total / 12B active (README). 42 layers, hidden 4096, 256 routed experts top-6 plus 2 shared, expert inner dim 2048, 1 dense-MLP layer.
  - 35 of 42 layers are sliding-window (512), 7 global. `model_max_length` 1,048,576. MTP head with 8 next-token layers. Drafter RadixArk/Inkling-Small-DSpark is 2.8 GB.
- **Capabilities:**
  - Input is text, image (40-4096 px, 40 px patches) and audio. Output is text.
  - Thinking via `reasoning_effort` none..max, default high (SGLang cookbook in `competitors/sglang-rtxpro6000/docs/cookbook/autoregressive/ThinkingMachines/Inkling-Small.mdx`).
  - Tool calling through SGLang parsers. BFCL 76.54% (B200 NVFP4, `docs/src/snippets/configs/thinkingmachines/inkling-small-benchmarks.jsx`).
- **Benchmarks** (README table): SWE-bench Verified 80.2, Terminal Bench 2.1 64.7, Toolathlon 54.4, MCP Atlas 79.6/79.2, tau3-Banking 15.5, MMMU Pro 74.0, CharXiv RQ 77.4 (81.3 with python), GPQA 89.5.
  - ARC-AGI-1 84.0% at $0.11/task and ARC-AGI-2 40.1% at $0.23/task, both at xhigh. ARC-AGI-1/2 at high: 78.0/33.1; medium 67.0/13.6; low 52.5/4.9.
  - **ARC-AGI-3: "Not yet evaluated"**. [web] https://arcprize.org/results/thinky-inkling-small
  - No Flash-Next vs Inkling-Small comparison exists anywhere I found. Strong results on static ARC tasks are not evidence for the interactive games.
- **Size:** official NVFP4 is 170.8 GB (sum of 21 HF files); BF16 is ~266B params.
  - Kaggle has it: `mirzamilanfarabi/inkling`, instance `small-nvfp4` (170,764,923,366 B) and a 2-bit Unsloth GGUF (88,494,649,967 B). [web]
  - NVFP4 needs ~2 big GPUs. The 88.5 GB GGUF leaves ~7 GB for KV and graphs and runs on llama.cpp only.
- **Engine support:** the SGLang cookbook lists h200/b200/b300/gb200/gb300/dgx-spark/AMD only (`inkling-small.jsx:10-12`). RTX Pro 6000 / sm120 is not listed.
  - The NVFP4 CI test uses tp=4 on B200 and says "the checkpoint does not fit on fewer cards" (`test/registered/models_e2e/test_inkling_small_nvfp4.py`).
  - Pennyroyal d00d88ef has Inkling code (`server_args.py:2093,3933,4834`) but no README/RESULTS claim for Inkling on sm120. vLLM recipe exists (recipes.vllm.ai), not checked on sm120.
- **Pruned/compressed versions** (HF API sizes, none usable):

| Repo | Size | Notes |
|---|---|---|
| miweru/Inkling-Small-REAP-137B-A12B-de-GGUF | IQ4_XS 72.0 GB, Q4_K_M 81.5 GB | 128/256 experts removed; ~4M-token calibration, 56% German; **vision/audio dropped**; llama.cpp; probe loss +0.72 nats English prose, +0.68 Python |
| studioburnside REAP25 2bE / 3bE | 63.9 / 88.0 GB | MLX (Apple) |
| pipenetwork REAP25 4-bit | 112.2 GB | MLX (Apple) |
| 0xSero EXL3 2.0bpw | 81.4 GB | ExLlamaV3; engine/vision support unverified |
| sabrewing colibri-int4 / cyankiwi AWQ-INT4 | 141 / 163 GB | too big |

- **No CUDA checkpoint** (SGLang/vLLM-loadable) that is pruned, 4-bit, has vision and fits under ~85 GB exists.
- **Decode cost** [derived, uniform routing, ~4.5 bit]: unpruned Inkling-Small touches ~157 experts per layer at M=40, about 91 GB/step. The same formula gives ~33 GB/step for Flash-Next, matching the ~34 GB in `throughput_research.md`.
  - A 50% REAP would be ~74 GB of expert weights (fits) but ~63 GB/step, still ~1.9x Flash-Next.
  - Pruning needs a BF16 calibration pass over ~532 GB of weights, which we cannot do locally.
- **Verdict:** drop for this competition. It needs a new SGLang sm120 bring-up, a custom pruned and quantized build, and has no ARC-AGI-3 evidence.

### 2. Compressed Flash-Next variants
Flash-Next is 125B total / 6B active plus 51B n-gram (PLE) params, 512 experts, 10 routed + 1 shared. [web] Qwen card
- **boristown/qwen3-8-flash-next-nvfp4-reap-448e** is on Kaggle (instance `nvfp4-reap-k448` v2, 124,217,429,278 B). [web]
  - It is a mirror of HF lee-chang-93/...-NVFP4-REAP-k448, a REAP prune of the nvidia NVFP4 checkpoint. 64 of 512 experts are removed per layer, 123.6 to 115.6 GiB (-7.9 GiB), with 92.65% of REAP mass retained.
  - Calibration used 16.49M tokens of real agentic traces including 626 images (48% coverage). The card has **no downstream eval** and says "QA diff ... recommended".
  - MTP needs vLLM >= 0.30, a `sitecustomize.py` and `VLLM_MTP_NUM_EXPERTS=512`.
  - The only user I found is cihanatak's notebook ([local] `competitors/kaggle/arc3-b32-reap-v030-kv14-c12-ctx48-r1/*.txt:383-438`): vLLM 0.30.0, `modelopt_mixed`, 12 seqs, ctx 49,152, 15,032,385,536 B KV, MTP3, prefix caching off. **No score found.** [unverified] on Pennyroyal SGLang.
  - Counter-evidence: rellik13 pruned 25% of experts, "More room, worse play" ([local] `competitors/rellik13-milestone2/WRITEUP.md:146`).
  - Our experts-per-token tests also moved the wrong way ([local] `arms.md`): experts16 38.69 vs 46.25, experts8 13.93 vs 17.45 on the panel, no speedup.
  - Our own streams14 test gave +15% throughput for 46.38 vs 46.25 (noise), so freeing KV for streams is a weak lever.
- **lovedheart/...-NVFP4-FP8-Pruned-RTXPRO-6000** (HF only, 123.4 GB): AIMER calibration-free prune, 512 to 448, needs a patched SGLang or "GARBLED OUTPUT". Its GSM8K 97.27 / AIME26 98.75 table is labelled "this checkpoint" but is probably the unpruned line [unverified].
- **Davd-b01/qwen3.8-flash-next-40b-prune-research** (HF only): aggressive prune, GPQA 27.78%, MATH-500 17.8% in its model-index. That is a collapse, so heavy pruning is not viable.
- **lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8** (Kaggle, 180.2 GB): a mirror of primitive-ai/...-mixed-NVFP4-FP8 (FP8 GDN/attention projections).
  - Card: 84.4 tok/s single-stream and 526.0 tok/s at concurrency 32, +13% and +8.7% vs plain NVFP4. Accuracy "statistically identical" (tie band +/-1.0).
  - vLLM only, requires `VLLM_GDN_DECODE_KERNEL=triton`.
  - Decode speed did not buy score: lordhansolo's vLLM NVFP4 ran median 976 tok/s at 14 slots, job-wide 933 tok/s, yet scored LB 23.84 vs Franzen 27.89 at 588 tok/s ([local] `tong-m2-comparison/kaggle/COMPARISON.md:200-215`).
- **local-inference-lab QAD NVFP4:**
  - `Qwen3.8-Flash-Next-NVFP4` is HF 106.4 GB and "approximately 98 GiB". It has NVFP4 experts and MXFP8 attention/shared experts, QAD on 200M tokens. GPQA 89.9, AA-LCR 79.4, Tool Eval Bench 91; it beats NVFP4 PTQ on GPQA with 9% fewer tokens (no PTQ number given).
  - The `-MXFP8-CSF-QAD` repo (102.9 GB, uploaded 2026-10-04) needs vLLM `nvfp4_csf` and carries a "No reuploads" license. I did not see either on Kaggle.
- **nvidia/qwen3-8-flash-next-nvfp4** (Kaggle, official) vs FP8: GPQA 92.0 to 91.5, HLE 34.7 to 35.4, tau2-Telecom 90.8 to 90.1, MMMU Pro 77.1 to 78.3, Terminal-Bench 2.1 83.3 to 82.9. Essentially lossless. RadixArk NVFP4 is also on Kaggle (keithtyser).
- **Swift 1.5 fine-tune of Flash-Next:** rellik13 found "No better per token and about 35% slower to decode" (`WRITEUP.md:147`). Pennyroyal's Swift 27B check shows 12% fewer tokens (43,599 to 38,205), not better quality ([local] `sglang-rtxpro6000/SWIFT-27B.md`).
- **Our current W4A16:** Franzen says Intel W4A16 gave "similar quality and throughput" to RadixArk NVFP4 with more VRAM left for KV ([local] `daniel-franzen-sol/WRITEUP.md:31`).

### 3. Other candidates (one 96 GB GPU, vision + tools)
- **Qwen3.8-27B (dense, 27B active):**
  - Same-card quality: SWE-bench Pro 61.7 vs Flash-Next 62.5; GPQA 89.2 vs 91.7; ClawEval-MM avg 56.9 vs 60.4; OSWorld 2.0 partial 48.0 vs 52.3 (Qwen card).
  - Pennyroyal 27B FP8 + DFlash2: **108.31 tok/s single, 374.98 tok/s aggregate at 4 requests** ([local] `sglang-rtxpro6000/README.md`). I found no 10-stream number, versus our ~789 on Flash-Next.
  - Franzen: each switch (3.6-27B to 3.8-27B to Flash-Next) "gave a substantial improvement". rellik13 calls Flash-Next "the strongest model that fits on one RTX PRO 6000" and "the first clear jump".
  - On Kaggle: foysalemonshanto/qwen3-8-27b-fp8-repacked-v1, mikedan7/qwen3-8-27b-fp8-official, dangkhoa2016/z-lab-qwen3-8-27b-dflash2. [web]
- **Gemma 4 31B** (dense 30.7B, text+image, thinking, 256K ctx, Apache-2.0): Google QAT w4a16 and nvidia NVFP4 exist; Kaggle has google/gemma-4.
  - MS1 #2 Reki and #3 Murad ("forge") both ran it locally on the official GPT-OSS-120B template. The blog gives no scores. [web] https://arcprize.org/blog/arc-prize-2026-milestone-1
  - All three MS2 winners (LB 27.89 / 23.84 / 22.53 on 2026-09-30) served Flash-Next ([local] `tong-m2-comparison/kaggle/SUMMARY.md`). I found no ARC-AGI-3 evidence that Gemma beats Flash-Next.
- **Gemma-4-26B-A4B** (MoE, 4B active, vision): no ARC-AGI-3 evidence. Kaggle has only a GGUF.
- **Whittle-Next-27B-A3B** (Kaggle, davidaylward): its card says "not a finished, general-purpose model"; llama.cpp GGUF only. Skip.
- **Tufa (LB 55.89):** model not disclosed. Web summaries saying "Qwen 3.6 27B" describe Milestone 1 (the Duck harness), not this run. [unverified]

### Corrections to the brief
- Tufa's MS1 winner used Qwen **3.6**-27B FP8, not 3.8.
- Franzen's BF16 PLE offload is **95.37 GB** ([local] `COMPARISON.md:106`). The ~48 GB figure is the FP8 PLE (sirikilohit, 47.68 GB).

### Bottom line
1. Inkling-Small: no.
2. A model swap is unlikely to be the cheapest lever. Flash-Next stays ahead of the alternatives on the evidence found.
3. REAP-448 is the only pruned variant on Kaggle and has no quality or score evidence. A 25% prune hurt play (rellik13), and extra KV/streams did not move our score.
4. If we try one thing: the lordhansolo mixed NVFP4+FP8 (+13% single-stream, vLLM only) or REAP-448 on vLLM 0.30, each as a paired 25-game A/B. Otherwise keep W4A16.
