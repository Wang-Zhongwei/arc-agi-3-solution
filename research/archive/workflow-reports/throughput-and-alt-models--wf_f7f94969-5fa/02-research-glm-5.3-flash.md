# throughput-and-alt-models: research:glm-5.3-flash

Workflow run `wf_f7f94969-5fa`, phase Research. Win 2+ more games via throughput (sim-v2 sweep + a throughput stack of built-but-unrun fixes) and serve an alternative model (GLM-5.3-Flash EXL3, ISTA Qwen3.8-Flash-Next GSQ-RCO GGUF) by refactoring the RTX Pro 6000 SGLang launcher into per-model profiles

---

**Verdict: not feasible as given, and it would cost throughput rather than add it.** The Ablit checkpoint is 175.6 GB (163.6 GiB), so it cannot fit one 96 GB RTX Pro 6000. SGLang has no EXL3 loader. Our Pennyroyal 2.5.3 build cannot load GLM-5.3-Flash in any weight format. GLM-5.3-Flash is also not "similar size": it has 320B total and 18B active parameters, against 125B and 6B for Qwen3.8-Flash-Next. The best measured GLM setup on this GPU uses two cards and still reaches less aggregate throughput than our one-card Flash-Next. Nothing here was run on a GPU; it is all web and repo research.

## 1. What the checkpoint is
- **Base model:** Z.AI GLM-5.3-Flash, MIT license, "320B total parameters and just 18B active" (https://huggingface.co/zai-org/GLM-5.3-Flash). Safetensors metadata counts 321,323,031,390 parameters (https://huggingface.co/api/models/zai-org/GLM-5.3-Flash).
- **Files:** 83 safetensors shards, 175,642,267,944 B (https://huggingface.co/api/models/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit/tree/main).
  - The repo is gated with a Responsible Use form, so `config.json` returns "restricted". The card says it has the "same `config.json` as the parent".
  - The parent quant is Apache-2.0; the base license stays MIT.
- **EXL3 4bpw:** ExLlamaV3's format, "a streamlined variant of QTIP" (https://github.com/turboderp-org/exllamav3, README:264).
  - Only the routed experts in layers 3–44 and the MTP layer 45 are 4-bit EXL3.
  - "Everything else (attention, KDA, shared experts, dense layers, embeddings, head, vision tower)" stays BF16, about 9.67B BF16 parameters (https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold).
- **TensorFold:** a serving engine, not a quantization method (https://github.com/ashhart/TensorFold). It quantizes the dense layers itself at load time (`DENSE=q4`).
  - TensorFold 1.0 qualifies GLM only on Metal, on two M5 Ultras. On CUDA it qualifies only Nemotron, on GB10.
  - Its README says "Flash Next and GLM currently run one active reply per engine".
  - Mia's recipe instead uses TensorFold v0.6.0 plus 77 patches on two DGX Sparks (GB10, TP=2, 4 streams by default, 8 maximum). Each Spark needs about 110 GiB free (https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold).
- **Ablit (abliteration):** 30 `o_proj` tensors are replaced, in layers 15–43 and MTP layer 45, with a median relative change of 0.115.
  - The only metric reported is Refusal32 (32/32 bypass), measured on the NVFP4 parents with thinking off.
  - No capability benchmark exists for this checkpoint (Ablit card).
  - The recipe turns thinking off by default for the Ablit weights, "which give their best results that way" (recipe README:338). That is a bad sign for a reasoning harness.
  - A different abliteration of the same model by OrcaRouter moved MMLU −0.7, GSM8K −0.7 and MMLU-Pro +1.5 (https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-FP8). That is a separate edit, but it suggests refusal removal neither helps nor much hurts capability. For us it brings no upside.

## 2. Architecture compared with Flash-Next
GLM figures come from https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/config.json. Qwen figures come from https://huggingface.co/Qwen/Qwen3.8-Flash-Next and `research/reports/engine-model-research.md:250` on `origin/main`.

| | GLM-5.3-Flash | Qwen3.8-Flash-Next |
|---|---|---|
| Total / active params | 320B / 18B (about 304B in routed experts) | 125B / 6B, plus 51B n-gram (PLE) and 4B MTP |
| Layers | 45: 34 KDA linear-attention layers and 11 DSA sparse-MLA layers (every 4th layer) | 48 (GDN + QSA) |
| Hidden size | 4096 | — |
| MoE | 288 routed experts, top-8, plus 1 shared; expert inner dim 2048; first 3 layers dense | 512 experts, 10 routed + 1 shared |
| Attention cache | MLA latent `kv_lora_rank` 512, NoPE (`qk_rope_head_dim` 0, qk/v head dim 256, 64 heads). DSA indexer: 32 heads × 128, top-k 2048, `index_kpool` 4 | — |
| Linear attention | KDA: 64 heads × 128, conv kernel 4 | GDN |
| Other features | mHC (`hc_mult` 4), vision tower | — |
| MTP / NextN head | 1 layer (`num_nextn_predict_layers` 1); DFlash2 drafters also exist | yes |
| Context | 1,048,576 | 262,144 native |
| Vocab | 154,880 | — |

## 3. Benchmarks
| Benchmark | GLM-5.3-Flash | Qwen3.8-Flash-Next |
|---|---|---|
| Terminal-Bench 2.1 | 84.3 | 83.3 (FP8), 82.9 (NVFP4) |
| DeepSWE v1.1 | 63.4 | 58.7 |
| Agents' Last Exam | 26.3 | 24.3 (pass@1) |
| AutomationBench | 48.8 | — |
| HLE | 55.3 with tools | 35.9, GPT-4o judge (not comparable) |
| GDPval-AA v2 | 1773 | — |
| Toolathlon | — | 73.5 |
| GPQA | — | 91.7 |

- **Sources:** GLM, https://raw.githubusercontent.com/zai-org/GLM-5/refs/heads/main/resources/bench_53.png and `.eval_results/GLM-5.3-Flash.yaml`. Qwen, the Qwen card table; Terminal-Bench is from https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4.
- **Edge:** the gains are 1–5 points, for three times the active parameters, and the harnesses differ.
- **ARC Prize:** GLM-5.3-Flash (Max) scores 65.8% on ARC-AGI-2 semi-private and 91% on ARC-AGI-1. There is no ARC-AGI-3 entry: the 87 `v3_Semi_Private` rows include no GLM. Flash-Next is not listed at all (https://arcprize.org/media/data/evaluations.json, `models.json`). Prior research already found that static-ARC strength is not evidence for the interactive games (engine-model-research.md:224).

## 4. Serving
- **SGLang cannot load EXL3.** Neither the upstream nor the Pennyroyal d00d88ef file tree has any exl3 or exllama quantization (GitHub tree listings). No SGLang issue mentions EXL3 support.
- **Pennyroyal 2.5.3 (d00d88ef) does not support the architecture.**
  - It ships the GLM-5.3-Flash cookbook docs but no `models/glm5_next.py`, `glm5_next_nextn.py` or `configs/glm5_next.py`. The newer branches `rc/next` (529b57c6) and `rc/3.0rc2` (4432e9ec) lack them too.
  - Upstream added the model in #36507 (2026-09-06) and NVFP4 loading in #38621 (2026-09-09). The cookbook requires v0.5.20 or later (released 2026-09-18).
  - The parsers do exist in Pennyroyal: reasoning `glm45` (`reasoning_parser.py:2161`) and tool calls `glm47` (`function_call_parser.py:76`).
- **Upstream SGLang on sm120 is blocked or slow.**
  - #37105: the TRT-LLM DSA decode backend is sm100-only, and TileLang needs 202 KB of shared memory against sm120's 100 KB.
  - #42012: only the triton attention backend works. Triton prefill is about 140 tok/s on an 8K prompt, against 9,249 tok/s for vLLM.
  - #36830: FP8 KV is blocked by `index_kpool` > 1. The only sm120 fix is unmerged (#38430 plus FlashInfer #5022/#5075; tracker #37813).
  - The cookbook's supported hardware is gb300/h100/h200/b200/b300/gb200/mi355x, with no RTX Pro 6000 (`docs/src/snippets/configs/zai-org/glm-5.3-flash.jsx:4`).
- **The best sm120 SGLang build needs two to four GPUs.** ormandj/sglang-glm53-flash-sm120 v0.5.0 runs TP2 W4A16 at 181.6 / 287.6 / 381.5 tok/s aggregate at 1 / 2 / 4 concurrent requests, capped at 4 running requests. Its TP4 profile uses NVIDIA's NVFP4 (https://github.com/ormandj/sglang-glm53-flash-sm120).
- **No other 4-bit or 8-bit checkpoint fits one card.** SGLang-loadable full-model quants (from the HF API byte sums):

  | Checkpoint | Size |
  |---|---|
  | Intel W4A16 AutoRound | 181.5 GB |
  | canada-quant W4A16 | 190.8 GB |
  | wtdcode AWQ | 190.8 GB |
  | RedHat NVFP4 | 197.8 GB |
  | nvidia NVFP4 | 204.4 GB |
  | Official FP8 | 328.3 GB |

  The only checkpoints that fit are:
  - `Asilarkness/GLM-5.3-Flash-Uncensored-NVFP4-Pruning-62GB` (66.5 GB, 80 of 288 experts kept, smoke-tested only). On an RTX Pro 6000 it boots only with two extra NaN fixes, and the output is still "repetition-heavy" (fsadj's comment on #37813).
  - turboderp EXL3 2.05bpw (85.1 GB, ExLlamaV3 only). turboderp measured 83.9 GiB used with an FP16 256k cache on one GPU (exllamav3#348).

  The REAP50 NVFP4 prune is 93.6 GiB, which does not fit.
- **ExLlamaV3 / TabbyAPI:**
  - They support `Glm5NextForConditionalGeneration` (README:206) and run on sm120 on a single GPU (exllamav3#333).
  - The library has continuous batching, speculative decoding (MTP), an OpenAI-compatible API through TabbyAPI, and reasoning and tool parsing via its config.
  - Tensor parallelism is not implemented for this architecture (#333). The quantized cache arrived in v1.4.8 (#327).
  - There is no 10-stream throughput data. Prior research rejected ExLlamaV3 for this reason (engine-model-research.md:98-101).

## 5. KV and state memory at 10 × 136Ki tokens
These are my calculations, not measurements.

| Item | Size |
|---|---|
| Compact FP8 MLA rows: 528 B × 11 DSA layers | 8.0 GiB |
| Plus indexer keys | 8.0–9.4 GiB |
| Plus the MTP layer | 8.7–10.3 GiB |
| BF16 fallback instead (upstream sm120 without #38430) | 15.1–18.0 GiB |
| KDA recurrent state: 140.8 MiB per slot | 1.4 GiB at 10 slots; 8.3 GiB at our `MAMBA_CACHE`=60 |

So KV is not the obstacle; weights are. To leave room for cache, state and CUDA graphs, weights must stay under roughly 70–78 GiB of the 91.8 GiB pool. Only the 61.9 GiB prune clears that comfortably. The 79.3 GiB 2.05bpw EXL3 is borderline for 10 streams.

## 6. Throughput
This section is derived from published numbers, not measured.
- ormandj's two-GPU build reaches 381.5 tok/s at 4 streams. Our one-GPU Flash-Next does about 111 × 4 ≈ 444 tok/s, and 705–789 tok/s at 10 streams.
- Per GPU, GLM delivers well under half of Flash-Next's throughput.
- Mia's recipe on two Sparks reaches 108.8 tok/s aggregate at 4 streams (prose).
- A pruned single-GPU GLM keeps top-8 routing and about 9B non-expert parameters (BF16 in most quants), so it would decode slower per step than Flash-Next.
- That runs against "win at least 2 more games (so increase throughput)". Prior data also shows throughput losses convert to score losses (the experts16 arm lost 7.6 points).

## 7. Cheapest credible path
None is worth doing for this competition. If the user insists, these are the options:
- **(a) 62 GB pruned NVFP4 on SGLang.**
  - What it takes:
    - Leave Pennyroyal for upstream SGLang v0.5.21 or ormandj v0.5.0, plus #38430, FlashInfer #5022/#5075 and the two NaN fixes.
    - Build a new offline wheelhouse.
    - Rewrite the launcher's Qwen-specific settings:
      - The format asserts and AutoRound checks (`launch_sglang.py:49,306`).
      - `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder` and `preserve_thinking` (`launch_sglang.py:335-336`).
      - The NEXTN/albucino draft settings (`launch_sglang.py:347-356`).
    - Change the harness's thinking kwargs (`tool_agent.py:1317,5209-5215`). GLM uses `clear_thinking`/`thinking`, and `enable_thinking` does nothing on GLM.
  - Effort: about 1–2 weeks. Quality risk is high: 72% of experts are removed and the output is known to be repetition-heavy.
- **(b) EXL3 2.05bpw on ExLlamaV3 + TabbyAPI.** This needs a new engine bundle and client changes, and has unknown 10-stream speed and 2-bit quality. About 3–5 days to bundle.
- **Neither can be checked this week.** Only static checks are possible locally, there is no local RTX Pro 6000, and Kaggle quota is gone.

## 8. Unverified
- Whether the 2.05bpw build plus the cache for 10 × 136k fits.
- TensorFold v0.6.0 on sm120.
- How SGLang lays out the indexer with `kpool` compression; this is why the KV figure is a range.
- ExLlamaV3 and TabbyAPI throughput at 10 streams.

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/altmodel-research/`:
- `glm53-flash-notes.md`
- `glm_variant_sizes.tsv`
- `zai-org_GLM-5.3-Flash_config.json`
- `_sglang_issue_*.txt`
- `_ormandj_README.md`
- `_mia_recipe_README.md`
- `bench_53.png`
