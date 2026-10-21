# wf_57f4538b-24c: research:glm-runner

Workflow run `wf_57f4538b-24c`, phase Research. 

---

**Verdict: no GLM-5.3-Flash runner is worth building for a Fable test on one RTX Pro 6000. The user's 4bpw checkpoint cannot run at all, and the one GLM build that does fit gives fewer tokens and worse output.**

The baseline to beat is the Fable v2 run on Flash-Next with SGLang. It generated 906,946 tokens in 25m13s (599 tok/s across the whole job). The median decode rate in serve.log was 744 tok/s, and the 90th percentile was 831.

The only GLM build that fits one card is turboderp's 2.05bpw EXL3 on ExLlamaV3 + TabbyAPI. It has these problems:
- **Speed:** one stream decodes at 93.65 tok/s. Nobody has published a 10-stream figure. My estimate is 150–400 tok/s total, about 20–55% of Flash-Next.
- **Quality:** mean KL divergence from the full model is 0.19, against 0.045 for NVFP4. The model's top token matches the full model's 88.9% of the time.
- **User reports:** people running it describe endless loops and hangs on real tasks.

A 25-minute run would therefore mix three effects: the model, damage from 2-bit quantization, and a 2–5x token shortfall. It cannot tell us whether GLM thinks better than Flash-Next. Full-precision GLM is only 1–5 points ahead of Flash-Next on agentic benchmarks.

## (1) The user's 4bpw EXL3 Ablit checkpoint (175.6 GB) cannot run on one card, even with CPU offload
- **Size:** 175.6 GB is 163.6 GiB. The card has 95.6 GiB (from the Kaggle log census `GPU0 x/95.6`).
- **Format:** the parent config says `"quant_method":"exl3","bits":4,"codebook":"mcg","head_bits":16,"scope":"glm53_routed_experts_only"`, and everything except the experts is BF16 (Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold, README and config.json). The Ablit card says it uses the same config.
- **ExLlamaV3 offload is blocked by that format.** ExLlamaV3's CPU MoE offload only accepts experts in its `mul1` format. On anything else it prints `"experts are not mul1, CPU offload skipped"` and keeps those experts on the GPU, which runs out of memory (`exllamav3/modules/block_sparse_mlp_cpu.py:343-355`, and `:478-479` for the per-expert split; docs in `doc/env_vars.md:315-321`, ExLlamaV3 v1.6.0 dev tip).
- **TensorFold, the engine it was built for, does not cover this card.** It runs this checkpoint only as TP2 on two DGX Sparks. Mia's own figures there are 61.8 / 103.2 tok/s at 1 / 4 concurrent requests.
- **Even a checkpoint ExLlamaV3 can offload is too slow** for 10 streams:

  | Setup | Decode | Prefill / TTFT | Source |
  |---|---|---|---|
  | 4.05bpw, one RTX PRO 6000 + 16-core 9950X3D, CPU expert tail | 17.6 tok/s, one request | 505 tok/s, TTFT 45.6 s on a 22.8k prompt | github.com/anoane/02-glm53-flash-nvfp4-freetoken |
  | FreeToken NVFP4, same box | 22.2 tok/s | `--max-running-requests 1` | same |
  | 3.05bpw, 5090 + Threadripper 7960X, 80% offload | 30.6 tok/s at S=1 | — | ExLlamaV3 v1.5.0 release notes |
  | Same with an 8-core AVX2 CPU | 10.3 tok/s | — | same |

- **Kaggle host:** 176.9 GiB RAM (fz-arms-v4-control log). It is probably the GCP g4-standard-48 shape: 48 vCPU, 180 GB, EPYC Turin (cloud.google.com/compute/docs/accelerator-optimized-machines); that match is my inference.
- **Batching does not rescue it.** CPU MoE decode barely scales with batch size: 10 tokens touch about 71 of the 288 experts per layer, against 8 for one token. My estimate is 20–35 tok/s total, which is 20–35x below Flash-Next.
- **The only way to use the user's repo on one card:** the abliteration touches only 30 BF16 `o_proj` tensors, about 2.7 GiB. Those could be copied onto turboderp's 2.05bpw build. gitcommit90/glm-5.3-one-spark does this (`ablit/fetch_transplant.py`), but on vLLM with a different donor checkpoint. Doing it in ExLlamaV3 is unverified, needs an HF token that has accepted the repo's gate, and gives no benefit for the games.

## (2) turboderp EXL3 2.05bpw (85.23 GB = 79.38 GiB, mul1, rev 51058cd5) on ExLlamaV3 + TabbyAPI
- **Runs on sm120:**
  - The v1.6.0 release wheels cover cu128 (torch 2.8–2.11) and cu132 (torch 2.11–2.13), built with arch `12.0+PTX` (`.github/workflows/build.yml:56`).
  - TabbyAPI main (2fd6cc7) pins exllamav3 1.5.4 with torch 2.9.0+cu128 through its `[cu12]` extra (`pyproject.toml:82-86`).
  - A user confirmed it on one RTX PRO 6000 (issue #333): the answer was coherent, 20/20 on short arithmetic, 4/5 on program traces.
- **Speed:** measured on a Pro 6000 at one stream, 4k prefill runs at 3,938 tok/s and decode at 93.65 tok/s (v1.5.0 notes).
  - No multi-stream data exists for GLM on one GPU. The only ExLlamaV3 GLM concurrency ladder (4.05bpw on 4×CMP 170HX, v1.4.6) went from 25.2 tok/s at C1 to 44.6 at C8, a 1.77x gain (PixelML/club-170hx notebook).
  - The fast decode path (CUDA graphs and fused MoE) only covers batch sizes of 8 or fewer (`doc/env_vars.md:38,528`). At 10 streams it falls back to a slower path.
- **Batching:** continuous batching works. TabbyAPI defaults `max_batch_size` to 4 for recurrent models (`backends/exllamav3/model.py:488`), so it must be set explicitly. Prefix reuse works through recurrent-state checkpoints kept in system RAM (`recurrent_checkpoint_interval*`, `memory.sysmem_recurrent_cache`).
- **Context and streams:**
  - FP16 cache costs 1,600 B per DSA layer per token (`cache/mla.py`, `modules/mla_attn.py:168-171`). Over 11 layers that is about 17.6 KB per token.
  - turboderp measured 83.9 GiB total at a 256k FP16 cache (82.3 GiB with a 6-bit cache), in issue #348. That leaves about 9–10 GiB, so the pool is roughly 750–800k tokens. That fits 8–10 streams at about 80k each.
  - A quantized cache works since v1.4.8 (#327) but saves little, because the cache is already compressed.
  - Each stream's recurrent state is about 143 MB (my calculation).
- **API:** everything the harness needs is there.
  - Tool calls: the `glm4_5` format covers GLM5.3 (`docs/10.-Tool-Calling.md:38`).
  - `reasoning_content` comes back split out. `chat_template_kwargs` and `top_k` are accepted.
  - Images: `image_url` data URIs work (`common/image_util.py:16`). The 2.05bpw build includes the vision tower and the MTP head (1,007 `model.visual.*` tensors and `layers.45.*` in the index).
  - The harness's `priority` field should be ignored by pydantic, but that is unverified.
  - The model name `flashnext` is ignored when inline loading is off (`endpoints/OAI/utils/common_.py:160-170`).
- **Quality at 2.05bpw:**
  - turboderp's card chart: mean KL divergence 0.1935 (median 0.033), against 3.05bpw 0.068, 4.05bpw 0.0345, NVFP4 0.0452, AWQ 0.0565, UD-IQ1_M 0.232.
  - An independent run: mean KL divergence 0.1216, top-token match 88.92% (via gitcommit90/glm-5.3-one-spark).
  - Forum users: "not usable… endless loops and hangs in parallel processing" and "quality is really poor" (forums.developer.nvidia.com/t/382140). FP8 "feeling totally different – especially on longer tasks" (/t/382468).

## (3) Other single-card GLM paths — none is better

| Path | Status on one 96 GB sm120 card | Source |
|---|---|---|
| vLLM-EXL3 forks | Only ARM64 Spark (TP1) or x86 with two RTX cards. No single x86 sm120 build. Spark: C4 181.9 tok/s summed / 91.5 wall. Their DFlash2 drafter is CC BY-NC-ND. | gitcommit90/glm-5.3-one-spark; tpurtell/glm-5.3-flash-ext3-2x-rtx |
| NVFP4 80-expert prune (Asilarkness, 66.49 GB) | "Only smoke-tested". Needs NaN fixes on RTX Pro 6000, output repetition-heavy. Upstream SGLang on sm120 is triton-only, with prefill around 140 tok/s at 8k. | Card; prior report citing #37813 and #42012 |
| REAP50 GGUF (IQ3_M 67.18 GiB, IQ4_XS 82.0 GiB) | Needs a patched llama.cpp fork. The pruned parent alone matches the full model's top token only 84.25% of the time. | patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF |
| Unsloth UD-IQ1_S (86.69 GiB) | Barely fits. IQ1_M KL divergence 0.232, worse than EXL3 2.05bpw. | |
| UD-IQ3_XXS with experts in host RAM | One inference slot, 56.84 tok/s. | SSHdotCodes/glm-5.3-flash-pro6000 |
| Larger EXL3 quants | 2.32bpw is 91.75 GiB and 2.51bpw is 98.12 GiB; neither leaves room for a cache. | |
| Full 4-bit or 8-bit weights | 181–328 GB, so 2–4 GPUs. | |

llama.cpp b11476 does implement glm5-next with the DSA indexer (`src/models/glm5-next.cpp:973-984`).

## Fallback, only if the user insists on one GLM data point
This takes about a day to wire, plus one Kaggle run.
1. **Install** a cp312 venv: `git clone theroyallab/tabbyAPI@2fd6cc7 && pip install -e .[cu12]`.
2. **Download:** `hf download turboderp/GLM-5.3-Flash-exl3 --revision 51058cd551c7e570d87bd32a4adee720edce2349 --local-dir /tmp/models/glm53-2.05`. That is 85.2 GB. Raise `SERVER_STARTUP_TIMEOUT` from 12 to about 25 minutes to allow for it.
3. **config.yml:**
   - `network: {host: 127.0.0.1, port: 8001, disable_auth: true}`
   - `model: {model_dir: /tmp/models, model_name: glm53-2.05, backend: exllamav3, max_seq_len: 131072, cache_size: ~655360, cache_mode: FP16, chunk_size: 4096, max_batch_size: 8, vision: true, reasoning: true, tool_format: glm4_5, warmup: true}`. Raise `cache_size` after checking VRAM.
   - `memory: {sysmem_recurrent_cache: 32768}`
   - `draft_model: {draft_mode: disabled}`
4. **Harness:** set `ARC3_MAX_ACTIVE_STREAMS=8`, then run `python main.py --config config.yml` in place of notebook cell 14.
5. **Run length:** run to matched tokens, about 907k generated tokens. At the estimated speed that is roughly 60–120 minutes, not 25.

## Unverified
- 10-stream throughput on one card.
- Whether TabbyAPI accepts the harness's `priority` and its history messages that carry `reasoning_content`.
- The exact pool size on Kaggle.
- Copying the `o_proj` tensors in ExLlamaV3.
- That Kaggle's machine is g4-standard-48.

I wrote no summary file, on purpose. The raw evidence is saved under `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/runner-research/`:
- `glm/` — the READMEs and forum threads cited above, the Mia parent config, the size listings, `exl3_v160_assets.txt` and `exl3_issues_388_318_409.txt`
- `raw/` — the ExLlamaV3 and TabbyAPI clones, the KL divergence charts, and issues #327, #333 and #348
