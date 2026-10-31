# Flash-Next draft heads: quantization, size and acceptance

All four drafters we have served are the same MTP layer of Qwen3.8-Flash-Next (one decoder layer with QSA attention,
hyper-connections and a 512-expert MoE, plus `fc_embedding`, `fc_hidden`, two pre-norms and a final mixer). They differ
only in how the 512 routed experts are stored, and whether the checkpoint carries its own `embed_tokens` / `lm_head`.
The fine-tuned head is Lord's with retrained dense tensors ([its section](#fine-tuned-mtp-head-mtp-v1-50m)).
DFlash2, a separate block drafter, has [its own section](#dflash2-block-drafter-incoai) below.

| Drafter | Source | Routed experts | Everything else in the MTP layer | Own embed + lm_head | Bytes |
|---|---|---|---|---|---|
| Intel BF16 MTP (the target checkpoint's own layer) | Kaggle [`dfranzen/intel-qwen3.8-flash-next-w4a16-autoround`](https://www.kaggle.com/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround) = HF [`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/tree/4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897), `model_extra_tensors.safetensors` | BF16 (5.033 GB) | BF16 (0.181 GB) | no (the target's) | **5.214 GB** |
| Lord NVFP4 MTP | Kaggle [`lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8`](https://www.kaggle.com/models/lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8), variation `hf-mixed-mtp-nvfp4/1` = HF [`primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8@07915ee`](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8/tree/07915ee79ec217c117e8a57bf2557a4a1418c10f/mtp_nvfp4) `mtp_nvfp4/` | NVFP4 W4A4 (1.416 GB: U8 codes 1.258 + FP8 E4M3 scales 0.157) | BF16 (0.181 GB), **bit-identical to Intel's** (29 of 29 tensors) | no (the target's) | **1.597 GB** |
| Daniel's albucino INT4 | Kaggle [`dfranzen/albucino-qwen3-8-flash-next-drafter`](https://www.kaggle.com/models/dfranzen/albucino-qwen3-8-flash-next-drafter) `default/1` | INT4 (1.416 GB: I32-packed codes 1.258 + BF16 scales 0.157) | BF16 (0.181 GB) | yes, BF16 (2.542 GB) | **4.140 GB** |
| Fine-tuned MTP (mtp-v1-50m, step 1300) | Kaggle [`jonathanwang2022/qwen3-8-flash-next-mtp-tuned`](https://www.kaggle.com/models/jonathanwang2022/qwen3-8-flash-next-mtp-tuned) `PyTorch/default/1` (private) | Lord's NVFP4 file, unchanged (1.416 GB) | BF16 (0.181 GB), retrained: same 29 tensors, names, dtypes and shapes | a byte copy of the target's (2.542 GB), replaced by the target's at load | **1.597 GB** |

## Quantization of each layer

- **Intel**: `quantization_config.extra_config` of the checkpoint's `config.json` sets `.*mtp.*` to 16 bits, so the MTP
  layer is BF16 throughout. In the target, only the routed experts are quantized (AutoRound INT4, symmetric, group 128,
  `auto_round:auto_gptq` packing).
- **Lord** ([`mtp_nvfp4/config.json`](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8/blob/07915ee79ec217c117e8a57bf2557a4a1418c10f/mtp_nvfp4/config.json),
  compressed-tensors `mixed-precision`):
  - group_1 is `nvfp4-pack-quantized` on `mlp.experts.N.{gate,up,down}_proj`: FP4 E2M1 weights with an FP8 scale per
    16 inputs and an F32 global scale, `W = code x scale / weight_global_scale`. Activations are quantized the same way
    at runtime (`dynamic: local`).
  - The MTP layer's `self_attn` and `mtp.fc_*` are on the `ignore` list, so they stay BF16. group_0's FP8 attention
    applies to the target's own layers in that checkpoint, which we don't serve.
  - Also ignored, so BF16: norms, router gate, shared expert, hyper-connections, indexer.
  - Gate and up use the same global scale in all 512 experts. vLLM uses the gate's scale for both.
  - Dequantized experts match Intel's BF16 experts at cosine 0.995 (relative error 0.095, checked on experts 0, 7 and
    300). With 29 of 29 non-expert tensors identical, Lord's head is Intel's MTP layer with its experts rounded to
    NVFP4.
- **Albucino**
  ([`compact_sources.json`](https://www.kaggle.com/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1),
  `config.json`):
  - Source: `RadixArk/Qwen3.8-Flash-Next-NVFP4` @ `7b719225`.
  - "MTP routed experts locally RTN-quantized to INT4 symmetric group-32 with min/max scales; other MTP tensors
    unchanged".
  - `quantization_config` is compressed-tensors `pack-quantized`, 4 bits, group 32, targets `RoutedExperts`; weights
    only.
  - It carries 31 dense tensors, including its own `embed_tokens` and `lm_head`.

## Acceptance as served (3 draft tokens)

| Run | Drafter | Engine | Acceptance length | Per position | Mean gen tok/s | Log |
|---|---|---|---|---|---|---|
| [`matijaludvig/arc3-vcintel-25x1`](https://www.kaggle.com/code/matijaludvig/arc3-vcintel-25x1) v1 | Intel BF16 | vLLM (Lord's e975732 runtime), 13 streams | **2.752** | 0.767 / 0.572 / 0.418 | 737 | `runs/kaggle/matijaludvig/arc3-vcintel-25x1/v01/vllm-openai-server.log` |
| [`matijaludvig/arc3-vcintel-lorddraft-25x1`](https://www.kaggle.com/code/matijaludvig/arc3-vcintel-lorddraft-25x1) v1 | Lord NVFP4 | vLLM, 14 streams | **2.726** | 0.763 / 0.562 / 0.406 | 820 | `runs/kaggle/matijaludvig/arc3-vcintel-lorddraft-25x1/v01/vllm-openai-server.log` |
| [`jonathanwang2022/porting-advaith`](https://www.kaggle.com/code/jonathanwang2022/porting-advaith) v6 | Lord NVFP4, 32K draft vocabulary | vLLM, 14 streams | **2.743** | 0.767 / 0.568 / 0.412 | 894 | `runs/kaggle/jonathanwang2022/porting-advaith/v06/vllm-openai-server.log` |
| [`dangthuygiang/arc3-v207-danibase-base-dfranzen`](https://www.kaggle.com/code/dangthuygiang/arc3-v207-danibase-base-dfranzen) v5 | albucino INT4 (Pennyroyal draft view) | SGLang (EAGLE, 3 steps, top-1) | **2.688** | not logged | — | `runs/kaggle/dangthuygiang/arc3-v207-danibase-base-dfranzen/v05/serve.log` |

`runs/` is the Google Drive folder (see [AGENTS.md](../../AGENTS.md)). How each number is computed:
- **vLLM:** token-weighted from the "SpecDecoding metrics" lines: `1 + accepted / (drafted / 3)`. Per-position rates are
  the mean of the logged rates.
- **SGLang:** the mean of `accept len` over its decode lines, which is unweighted.
- **tok/s:** the mean of the 10 s "Avg generation throughput" lines.

The harnesses and runs differ, so the small gaps between rows are within run-to-run variation. What the table shows
is that NVFP4 or INT4 experts cost no measurable acceptance: Intel BF16 2.75, Lord NVFP4 2.73–2.74, albucino 2.69.
Lord's head pays off in memory (1.6 GB against 5.2 GB, which went to the KV pool) and speed, from smaller expert
reads in each draft pass, not acceptance. The vcintel-to-lorddraft switch raised mean throughput from 737 to 820 tok/s at the same score
(56.92 vs 56.89; see [lord-vllm-daniel-weights.md](lord-vllm-daniel-weights.md#lords-mtp-draft-head)).
[MEMORY_BUDGET.md](MEMORY_BUDGET.md) has the GPU memory of each setup, including the RadixArk NVFP4 MTP draft of
another team's notebook.

## Drafter KV precision

Conclusion: keeping the MTP drafter's own KV cache in FP8 costs acceptance. Real vLLM on v6's own requests measures
2.78 with an FP8 drafter KV and 2.91 with a BF16 one (+0.13, +4.6%); the offline simulation predicted 0.10–0.13. The
per-step cost does not change, so decode gets about 5% faster per token. The cost is 6.9% of the KV pool. The target's
KV stays FP8. A BF16 drafter KV needs a runtime patch on Lord's runtime (below); it has not been run on Kaggle yet.

### Measurement

The offline acceptance of Lord's NVFP4 head (step 0, untrained) on the 48 eval samples of `mtp_v1`, with the K/V and
QSA-indexer caches rounded to e4m3 where vLLM's `fp8_e4m3` / `fp8` caches would be:

| FP8 simulated in | Per position | Acceptance length | Frida job | Log |
|---|---|---|---|---|
| nothing (all BF16) | 0.813 / 0.642 / 0.489 | 2.943 | 209705 (step-0 eval of the training run) | `runs/local/frida/mtp-eval-fp8/none/train_log.jsonl` |
| target only | 0.813 / 0.642 / 0.490 | 2.945 | 209816 | `runs/local/frida/mtp-eval-fp8/target/train_log.jsonl` |
| drafter only | 0.804 / 0.598 / 0.407 | **2.809** | 209815 | `runs/local/frida/mtp-eval-fp8/mtp/train_log.jsonl` |
| both (as v6 serves) | 0.804 / 0.606 / 0.432 | 2.842 | 209817 | `runs/local/frida/mtp-eval-fp8/both/train_log.jsonl` |
| served, porting-advaith v6 | 0.767 / 0.568 / 0.412 | 2.743 | Kaggle v6 | `runs/kaggle/jonathanwang2022/porting-advaith/v06/vllm-openai-server.log` |

What the table shows:
- Rounding the target's KV changes nothing. The drafter's own FP8 KV costs 0.10–0.13, almost all of it at draft
  positions 2 and 3, where the head reads its own previous output.
- The drafter-only simulation reproduces the served third position (0.407 vs 0.412).
- "Both" scoring above "drafter only" is not explained.

How it was measured:
- The rounding is [`qsa.set_kv_fp8`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/97c3655d8db698c685140aae948eb1d1beb89c60/ARC3-Inference/inference/rft/qsa.py#L43-L49):
  K/V are rounded in [`qsa_project`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/97c3655d8db698c685140aae948eb1d1beb89c60/ARC3-Inference/inference/rft/qsa.py#L287),
  and the indexer queries and block keys [here](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/97c3655d8db698c685140aae948eb1d1beb89c60/ARC3-Inference/inference/rft/qsa.py#L96).
- It is switched on for the target and for the head separately by
  [`mtp_train.py --fp8-kv`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/97c3655d8db698c685140aae948eb1d1beb89c60/ARC3-Inference/inference/rft/mtp_train.py#L235-L237)
  ([flag](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/97c3655d8db698c685140aae948eb1d1beb89c60/ARC3-Inference/inference/rft/mtp_train.py#L441)),
  on branch `mtp-data` at
  [`97c3655d`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/commit/97c3655d8db698c685140aae948eb1d1beb89c60),
  as recorded in each job's `commit_<job>.txt`.
- All runs use the same 48 eval samples (seed 0) and the same numeric path (no compiled layers). The FP8 jobs ran with
  `--eval-only --no-compile-layers`.
- The full outputs are on Frida in `/shared/workspace/arc-agi-3/mtp_finetune/runs/eval-fp8-{mtp,target,both}/`.

This is not vLLM. The target is the transformers Qwen4Exp model with the RFT patches (QSA Triton kernel, grouped INT4
experts), and the head is vLLM's `Qwen4ExpMultiTokenPredictor` rebuilt from transformers modules. FP8 is a rounding of
the stored K/V with scale 1.0, not vLLM's FP8 cache kernels. Even with FP8, offline stays above served at positions 1–2.
The likely reasons:
- the eval set is 5 held-out games, not v6's 25;
- the offline count leaves out `uncertain` requests and requests over 124K tokens;
- draft steps 2–3 attend to keys from the same step, an approximation of the drafter.

The two implementations were not compared layer by layer.

### Cost in KV

- The MTP layer stores 2 KV heads × 256 dims for K and V
  ([config](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8/blob/07915ee79ec217c117e8a57bf2557a4a1418c10f/mtp_nvfp4/config.json)).
  That is 1,024 B per token in FP8 and 2,048 B in BF16.
- v6's pool is 21.73 GiB for 1,564,172 tokens, about 14.9 KB per token (`porting-advaith/v06/vllm-openai-server.log`).
- +1,024 B per token is +6.9%, so about 1.46M tokens instead of 1.56M. This is a geometry estimate; a run will give the
  real number.

### Serving it

- The setting exists: `speculative_config` `kv_cache_dtype`
  ([vLLM at the runtime's base commit](https://github.com/vllm-project/vllm/blob/e9757321527ca1ecd514c07c1418dd2c53da3d19/vllm/config/speculative.py#L429)).
  The draft loader applies it to the drafter's cache config only
  ([`_DRAFT_VLLM_CONFIG_OVERRIDES`](https://github.com/vllm-project/vllm/blob/e9757321527ca1ecd514c07c1418dd2c53da3d19/vllm/config/speculative.py#L383)).
- The launcher passes it as `kaggle.vllm.draft_kv_cache_dtype`, branch `draft-bf16-kv` at
  [`5831f199`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/commit/5831f1992f5c381c17c4aa40fb717bf8d9cf6c15)
  (`auto` = BF16; the runtime's docs call `bfloat16` invalid for models whose KV does not default to FP8).
- **Plain `auto` fails on Lord's runtime.**
  [`jonathanwang2022/draft-bf16-kv`](https://www.kaggle.com/code/jonathanwang2022/draft-bf16-kv) v1 (v6 + `auto`)
  died at startup: "To serve at least one request with the model's max seq len (147072), (211.88 GiB KV cache is needed,
  which is larger than the available KV cache memory (21.71 GiB)" (`runs/kaggle/jonathanwang2022/draft-bf16-kv/v01/vllm-openai-server.log`
  line 190).
- **Why** (traced on Frida, job 209822, with every KV spec and group logged before the memory check): the draft
  `kv_cache_dtype` gives the drafter a `replace()`d copy of the cache config. The platform then raises the attention
  block size to 1664 in place on the engine's config, after the model has loaded, so the copy keeps block size 16.
  The drafter's QSA indexer compressed-key cache reads its own copy, so it became a KV group of 16-token blocks.
  Lord's packed layout strides every block by the widest group's page (24.5 MB), so one 147,072-token request needed
  147,072 / 16 = 9,192 such blocks, the "211.88 GiB". The BF16 page size itself was not the problem.
- **The fix** ([`use_engine_block_size_for_draft_qsa`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/6c8bffd173c72b657e10052c755d749411614f6f/ARC3-Inference/inference/framework/lordhansolo/kaggle.py#L1464-L1501),
  branch `draft-bf16-kv` at
  [`6c8bffd1`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/commit/6c8bffd173c72b657e10052c755d749411614f6f)):
  - a setup-time patch with 3 anchored edits in `vllm/models/qwen4_exp/common/qsa_cache.py`, so the QSA caches read
    `vllm_config.cache_config.block_size` (the engine's) instead of their own copy;
  - applied only when a draft KV dtype is set, and the setup fails if an anchor does not match exactly once.

  With it (job 209823), the drafter's indexer cache joins the 1664-token group, and one full-length request needs
  2.19 GiB. v6 with the FP8 drafter needs about 2.04 GiB.

### Measured in vLLM on Frida

Lord's runtime rebuilt on one B200:
- the image `vllm/vllm-openai@sha256:e0eee5c5…` (the digest in his `runtime-manifest.json`; the tag is pruned) plus his
  32 overlay files, each checked against the manifest's sha256;
- the launcher's setup-time patches and Lord's MTP draft, then v6's exact `vllm serve` command;
- the harness is `runs/local/frida/vllm-replay-draft-kv/` (`run_vllm.sbatch`, `inside.sh`, `setup_runtime.py`).

The replay ([`replay.py`](../../ARC3-Inference/runs/local/frida/vllm-replay-draft-kv/replay.py)):
- porting-advaith v6's own 141 samples from `mtp_v1` (all 25 games), one request each: the context before the sample's
  last trained reply, mean 104K prompt tokens;
- the harness's sampling (T 0.7, top-p 0.95, top-k 20), a fixed seed per request, up to 2,048 generated tokens, 14 at a
  time;
- the two arms ran alone, one after the other, and got identical prefix-cache hits (3.15%).

Numbers are the differences of vLLM's `/metrics` counters before and after the replay
([`analyze.py`](../../ARC3-Inference/runs/local/frida/vllm-replay-draft-kv/analyze.py)).

| Drafter KV | Acceptance length | Per position | Time per output token | Decode tok/s per request | Decode time, all requests | Prefill time | KV pool (B200) | Job |
|---|---|---|---|---|---|---|---|---|
| FP8 (`fp8_e4m3`, as v6 serves) | 2.777 | 0.776 / 0.578 / 0.422 | 37.10 ms | 26.52 | 7,695 s | 971 s | 7,447,052 tokens | 209838 |
| BF16 (`auto` + the fix) | **2.910** | 0.790 / 0.624 / 0.496 | **35.23 ms** | **27.77** | 7,258 s | 974 s | 6,930,768 tokens | 209834 |
| change | +0.133 (+4.8%) | +0.014 / +0.046 / +0.074 | −5.0% | +4.7% | −5.7% | same | −6.9% | |

Reading the table:
- **Acceptance:** the served FP8 number reproduces Kaggle's v6 live 2.743 (0.767 / 0.568 / 0.412) to within 0.03.
  BF16 gains most at draft positions 2 and 3, as the simulation showed.
- **Speed:** a decode step costs the same in both: time per token × tokens per step is 103.0 ms (FP8) vs 102.5 ms
  (BF16). QSA reads only about 2,048 selected tokens per draft query, so doubling the drafter's KV bytes costs almost
  nothing per step. The whole gain is acceptance.
- **Wall time:** 946 s in both. That is end to end: the replay is prefill-bound (about 14.7M prompt tokens against
  0.2M generated), so it shows no difference.

What this does not show:
- the RTX PRO 6000 (Kaggle's GPU) kernels;
- KV pressure. The B200's pool is 7× Kaggle's, so the 6.9% smaller pool was never stressed. v6 ran its 21.7 GiB pool at
  89% (p99) and 93% (peak) with 14 streams. The first Kaggle run with BF16 drafter KV hit a prefill storm at minute 13
  (see [Served on Kaggle](#served-on-kaggle-with-bf16-drafter-kv-tuned-mtp-bf16-kv-v1)).

The files are in `runs/local/frida/vllm-replay-draft-kv/runs/`: `metrics_before.txt`, `metrics_after.txt`,
`replay.jsonl`, `server.log`, and the job output.

## Fine-tuned MTP head (mtp-v1-50m)

Lord's NVFP4 head with its 29 dense (non-expert) tensors retrained; the routed experts, `embed_tokens` and `lm_head`
are frozen.

### Training

- **Trainer:** [`inference/rft/mtp_train.py`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/e0d1f88e36af708602cc026a973bd70ccbf1529f/ARC3-Inference/inference/rft/mtp_train.py) (branch
  `mtp-data`, commit [`e0d1f88e`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/commit/e0d1f88e36af708602cc026a973bd70ccbf1529f), as recorded by the job), Slurm job 209705 on one Frida B200.
  The frozen Intel W4A16 target runs forward only. The head learns the target's sampling distribution (T 0.7, top-k 20,
  top-p 0.95) with a soft cross-entropy at each of the 3 draft positions. Embedding and `lm_head` are the target's.
- **Data:** `mtp_v1` from [`inference/rft/mtp_data.py`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/e0d1f88e36af708602cc026a973bd70ccbf1529f/ARC3-Inference/inference/rft/mtp_data.py),
  every request of 54 Intel-W4A16 runs. Train: 3,821 samples, 137M trained tokens. Eval: 842 samples from games ar25,
  cd82, cn04, g50t and sk48, none of which are in train.
- **Run:** 1,400 samples drawn round-robin over games (about 50M trained tokens), LR 3e-5, eval every 100 steps on 48
  eval samples, best checkpoint kept.
- **Offline result:** the acceptance length rises from 2.943 at step 0 (Lord's head as loaded) to **3.080** at step 1300
  (per position 0.839 / 0.686 / 0.555), the best of 15 evals. This offline eval runs in transformers with BF16 KV
  everywhere. Log: `/shared/workspace/arc-agi-3/mtp_finetune/runs/mtp-v1-50m/train_log.jsonl` on Frida.

### The served checkpoint

The trainer's own export (`draft/`) cannot be served as it is:
- It adds the retrained tensors in `mtp-finetuned-bf16.safetensors` and hard-links Lord's other files.
- Three of those files (`carry-model-bf16-00010/11/12`) are needed for `embed_tokens` and `lm_head`, but they also
  still hold Lord's original 29 dense tensors.
- The runtime's loader picks tensors by name (`mtp.*`), not by the file the index assigns them to
  (`_filter_safetensors_files_by_weight_name` in `vllm/model_executor/model_loader/default_loader.py`). Both copies
  would load, and the load order would decide which one is served.

[`inference/rft/frida/build_tuned_draft.py`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/68b1a5c63fc476652f69e82f077a1bcdfd603285/ARC3-Inference/inference/rft/frida/build_tuned_draft.py)
(standard library only) writes each draft tensor once:

| File | Tensors | Bytes |
|---|---|---|
| `mtp-experts-nvfp4.safetensors` | 6,144: Lord's NVFP4 experts, unchanged | 1,416,330,416 |
| `mtp-finetuned-bf16.safetensors` | 29: the retrained dense tensors | 181,140,352 |
| `draft-shared-bf16.safetensors` | `lm_head.weight` and `model.language_model.embed_tokens.weight`, byte copies | 2,542,797,048 |
| `config.json`, `model.safetensors.index.json` | Lord's config; an index of exactly these 6,175 tensors | |

- **Checks:** the build stops unless the retrained tensors have Lord's exact names, dtypes and shapes.
- **Embedding and `lm_head`:** these are byte-identical in Intel's target, Lord's checkpoint and this folder (SHA-256
  of the tensor bytes begins `40bddd25` for `lm_head` and `62288a3f` for `embed_tokens`).
- **What VRAM holds:** at load the runtime replaces the draft's embedding and `lm_head` with the target's
  (`maybe_share_target_embed` and the `lm_head` share in `vllm/v1/worker/gpu/spec_decode/eagle/utils.py`;
  `Qwen4ExpMTP` sets no `has_own_*` flag).
- **The 32K draft-vocabulary head:** it is cut from the draft's own `lm_head` before that swap (`build_draft_vocab_head`
  in `vllm/models/qwen4_exp/nvidia/mtp.py`), and that copy is identical to the target's.
- **Weight memory:** the drafter adds 1.597 GB, as Lord's head does. "Model loading took 69.54 GiB" in both v6 and the
  Kaggle run below.

### Measured in vLLM on Frida

The same harness, requests and sampling as [the drafter KV replay](#measured-in-vllm-on-frida), on one B200 (job
209913). The draft directory is the folder above, mounted in place of Lord's through the `LORD_DIR` override in
`run_vllm.sbatch`.

| Head | Drafter KV | Acceptance length | Per position | Time per output token | Decode tok/s per request | Job |
|---|---|---|---|---|---|---|
| Lord | FP8 | 2.777 | 0.776 / 0.578 / 0.422 | 37.10 ms | 26.52 | 209838 |
| Lord | BF16 | 2.910 | 0.790 / 0.624 / 0.496 | 35.23 ms | 27.77 | 209834 |
| fine-tuned | FP8 | 2.898 | 0.799 / 0.623 / 0.476 | 35.38 ms | 27.80 | 209926 |
| **fine-tuned** | BF16 | **3.020** | 0.820 / 0.664 / 0.537 | **34.14 ms** | **28.75** | 209913 |

- **Where the gain comes from:** the head and the drafter KV add up independently:
  - the fine-tuned head adds +0.122 with FP8 drafter KV and +0.110 with BF16;
  - BF16 drafter KV adds +0.133 with Lord's head and +0.122 with the fine-tuned one;
  - together that is +0.243 (+8.8%) over v6's served setup.
- **The head was trained with BF16 KV but keeps its gain on FP8:** with FP8 drafter KV it matches Lord's head with BF16
  (2.898 vs 2.910) at v6's pool size (7,447,052 tokens on the B200), so it pays no KV.
- **Speed:** time per token × tokens per step is 103.1 ms, the same step cost as both Lord arms, so the gain is all
  acceptance.
- **Wall time:** 946 s end to end, the same as the other arms (the replay is prefill-bound).
- **Files:** `runs/local/frida/vllm-replay-draft-kv/runs/replay3-tuned-{fp8,bf16}/` and
  `arc3_vllm_lord-2099{26,13}.txt`.

### Served on Kaggle with BF16 drafter KV (tuned-mtp-bf16-kv v1)

[`jonathanwang2022/tuned-mtp-bf16-kv`](https://www.kaggle.com/code/jonathanwang2022/tuned-mtp-bf16-kv) v1:
- **Config:** [`configs/tuned-mtp-bf16-kv/v1.json`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/d2532fe17c3e2ea709bd23126739a62b48752269/ARC3-Inference/configs/tuned-mtp-bf16-kv/v1.json)
  at [`d2532fe1`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/commit/d2532fe17c3e2ea709bd23126739a62b48752269) on branch `draft-bf16-kv`.
- **Against porting-advaith v6:** the fine-tuned draft, `draft_kv_cache_dtype: auto` with the
  [block-size fix](https://github.com/Wang-Zhongwei/arc-agi-3-solution/commit/6c8bffd173c72b657e10052c755d749411614f6f), and a 30-minute public run (17.7 minutes after the server came up). Everything
  else is identical.
- **KV pool:** 1,455,400 tokens against v6's 1,564,172 (−7.0%), from the same 21.73 GiB.

Minutes after the server's first metrics line, means of vLLM's 10 s lines, per-position rates as in the table above:

| Minutes | Run | Prompt tok/s | Gen tok/s | KV % (max) | Prefix hit % | Acceptance length | Per position |
|---|---|---|---|---|---|---|---|
| 0–5 | fine-tuned + BF16 | 2,090 | 916 | 28.7 (46.3) | 80.7 | 2.825 | 0.783 / 0.594 / 0.463 |
| | v6 | 2,018 | 862 | 26.9 (41.6) | 80.9 | 2.601 | 0.731 / 0.518 / 0.363 |
| 5–10 | fine-tuned + BF16 | 1,754 | 1,016 | 60.3 (75.8) | 90.0 | 2.888 | 0.790 / 0.617 / 0.484 |
| | v6 | 1,649 | 957 | 55.0 (67.0) | 90.4 | 2.671 | 0.745 / 0.541 / 0.385 |
| 10–15 | fine-tuned + BF16 | 4,677 | 750 | 86.9 (94.7) | 91.8 | 2.947 | 0.806 / 0.642 / 0.511 |
| | v6 | 1,644 | 959 | 77.9 (86.5) | 92.7 | 2.708 | 0.755 / 0.557 / 0.398 |
| 15–17.7 | fine-tuned + BF16 | 9,232 | 229 | 74.0 (91.9) | 85.1 | 3.007 | 0.818 / 0.664 / 0.541 |
| | v6 | 1,753 | 942 | 69.5 (87.5) | 93.6 | 2.700 | 0.753 / 0.551 / 0.396 |
| **0–17.7** | **fine-tuned + BF16** | | | | | **2.888** | 0.797 / 0.625 / 0.494 |
| | **v6** | | | | | **2.668** | 0.746 / 0.541 / 0.384 |

- **Acceptance:** +0.220 (+8.2%) over the same window, close to the Frida replay's +0.243.
- **Before the storm (minutes 0–12):** generation ran about 5–7% faster than v6 (1,000–1,030 against 944–989 tok/s).
- **The storm:**
  - The smaller pool, filled by faster generation, was 90.5% full at minute 12; v6 was at 79.8%.
  - From minute 13, at 93–95%, prompt throughput jumped from about 1.7K to 8–10K tok/s. Generation fell to 190–430 tok/s
    and the prefix-cache hit rate slid from 93% to 83%.
  - v6 never stormed at 14 streams; its maximum in the first hour was 88.5%.
  - The run ended during the storm, so it does not show whether it recovers.
- **Cause:** the storm comes from the BF16 drafter KV's smaller pool, not from the head, whose weights are the same size.
- **Ways to keep the gain without the storm:**
  - Serve the fine-tuned head with FP8 drafter KV. The pool is then v6's, and on Frida the head keeps +0.122 there
    (2.898). Measured on Kaggle [below](#served-on-kaggle-with-fp8-drafter-kv-tuned-mtp-fp8-kv-v1); this is the setup we
    chose.
  - Keep BF16 drafter KV at 13 streams: 1,455,400 / 13 ≈ 112.0K tokens per stream, against v6's 111.7K.
- **Logs:** `runs/kaggle/jonathanwang2022/tuned-mtp-bf16-kv/v01/vllm-openai-server.log` and
  `runs/kaggle/jonathanwang2022/porting-advaith/v06/vllm-openai-server.log`.

### Served on Kaggle with FP8 drafter KV (tuned-mtp-fp8-kv v1)

[`jonathanwang2022/tuned-mtp-fp8-kv`](https://www.kaggle.com/code/jonathanwang2022/tuned-mtp-fp8-kv) v1
([log](https://www.kaggle.com/code/jonathanwang2022/tuned-mtp-fp8-kv/log?scriptVersionId=364283059)):
- **Config:** [`configs/tuned-mtp-fp8-kv/v1.json`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/b35f2e9715d09f9bc80a1315c4a4d15bd774f059/ARC3-Inference/configs/tuned-mtp-fp8-kv/v1.json)
  at [`b35f2e97`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/commit/b35f2e9715d09f9bc80a1315c4a4d15bd774f059).
- **Against porting-advaith v6:** only the draft source (the fine-tuned checkpoint) and a 30-minute public run. The
  drafter KV stays FP8, so the block-size patch does not apply.
- **Server:** it came up about 15 minutes into the notebook and served for 15 minutes. "Model loading took 69.54 GiB"
  and the KV pool of 1,564,172 tokens (21.73 GiB) are both v6's.

The first 15 minutes after the server's first metrics line (90 lines each; minute 15 onwards of this run is
wind-down):

| | v6 | fine-tuned + FP8 | fine-tuned + BF16 |
|---|---|---|---|
| Acceptance length (`accept --until 15`) | 2.662 | **2.805** (+0.143, +5.4%) | 2.882 (+0.220) |
| Per position | 0.744 / 0.539 / 0.382 | 0.778 / 0.590 / 0.443 | 0.793 / 0.618 / 0.486 |
| KV pool | 1,564,172 tokens | 1,564,172 | 1,455,400 (−7.0%) |
| KV maximum, minutes 0–15 | 86.5% | **82.5%** | 94.7% |
| Prefill storm | no | **no** | from minute 13 |
| Mean generation tok/s, minutes 0–14 | 922 | **952** (+3.3%) | 941 (977 over minutes 0–12) |
| Mean prompt tok/s, minutes 0–14 | 1,787 | 1,887 | 2,335 |

- **Acceptance:** the gain matches Frida's (+0.122 there, +0.143 here), largest at position 3 (+0.06). Each 5-minute
  block is 0.15–0.21 above v6.
- **No storm:** prompt throughput stays at 1.5–2.1K tok/s and the prefix-cache hit rate at 92–94%, as in v6.
- **Throughput gains less than acceptance (+3.3% against +5.4%):** faster decoding brings more turns, so prefill rose
  5.6% (1,887 against 1,787 tok/s). The 10 s throughput lines are also noisy between runs. Frida's time per output
  token (−4.6%) is the cleaner measure.
- **Score:** 15 minutes of play (mean 6.84) says nothing about the score.
- **Log:** `runs/kaggle/jonathanwang2022/tuned-mtp-fp8-kv/v01/vllm-openai-server.log`.

### Decision: fine-tuned head with FP8 drafter KV

We serve the fine-tuned head with FP8 drafter KV, not BF16:
- **BF16's extra acceptance is small:** +0.08 on Kaggle (2.882 against 2.805) and +0.12 on Frida (3.020 against
  2.898).
- **BF16 costs 7% of the pool and storms at 14 streams.** Avoiding the storm takes 13 streams (1,455,400 / 13 ≈ 112.0K
  tokens each, v6's 111.7K). The lost stream, about −3.5% throughput by our earlier measurement, takes back the gain.
- **FP8 keeps v6's pool:** it gets most of the gain (+5.4% acceptance on Kaggle) at no KV cost, and leaves room to try
  a fourth draft token.

### Next: k = 4 (tuned-mtp-k4 v1)

[`jonathanwang2022/tuned-mtp-k4`](https://www.kaggle.com/code/jonathanwang2022/tuned-mtp-k4) v1:
- **Config:** [`configs/tuned-mtp-k4/v1.json`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/a4da31c0340096ef827985b007e441ad8214d7ac/ARC3-Inference/configs/tuned-mtp-k4/v1.json) at
  [`a4da31c0`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/commit/a4da31c0340096ef827985b007e441ad8214d7ac): `tuned-mtp-fp8-kv` v1 with `mtp_tokens: 4`, a 30-minute public run.
- **CUDA graphs:** the launcher sizes them from k: 14 × (1 + 4) = 70, rounded up to 72 (from 56).
- **The fourth draft position:** the head was trained on 3 positions, so position 4 reuses it untrained. A rough
  guess, if the drop from position 3 to 4 matches the one from 2 to 3 (×0.75): 0.443 × 0.75 ≈ 0.33, giving about
  2.805 + 0.33 ≈ 3.14 on Kaggle.
- **What decides it:** whether one more draft row per sequence costs less per decode step than the ~0.33 extra tokens
  it brings.
- **v1 cannot start.** k = 4 raises the attention block from 1664 to 1680 tokens, and 1680 is not divisible by v6's
  `prefix_match_unit` 128. Frida job 209959 stopped at engine start with "Invalid prefix_match_unit=128;
  prefix-cacheable KV cache group block sizes must be divisible by prefix_match_unit. Got group block sizes=[1680,
  1680, 1680, 8, 1680]" (`runs/local/frida/vllm-replay-draft-kv/runs/replay3-tuned-fp8-k4/server.log`). Kaggle v1
  has the same config.
- **v2:** [`configs/tuned-mtp-k4/v2.json`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/ece901afb5ef701260b1a6e4d12798a4251e0a5c/ARC3-Inference/configs/tuned-mtp-k4/v2.json) at
  [`ece901af`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/commit/ece901afb5ef701260b1a6e4d12798a4251e0a5c), v1 with `prefix_match_unit: 112`.
  - **Why 112:** the unit only sets where a hit may land; states are stored as before (`prefix_match_unit` in
    `vllm/config/cache.py`). It must divide every prefix-cacheable block size and be a multiple of each cache's
    per-state compression, which divides 16 (DFlash2 k = 7 booted with 16).
  - 112 = 1680 / 15 is the valid value closest to v6's 128, so matching stays nearly as before and the run tests
    k = 4 alone.
- **On Frida with 112:** job 209961 boots in 220 s with a KV pool of 7,452,680 tokens (k = 3: 7,447,052), and replays
  v6's 141 requests (the harness's `MTP_TOKENS` and `PREFIX_MATCH_UNIT` settings).

## DFlash2 block drafter (incoai)

Conclusion: once both prefix-cache patches are in, DFlash2 matches MTP-3's prefix-hit rate. It still does not beat
MTP-3 on throughput:
- **k = 3:** the same acceptance as MTP-3 at about the same step cost.
- **k = 7, B200 replay:** acceptance length 3.66 against 2.87 (+28% tokens per step), but each step takes 1.25× as
  long, prefill included. The decode part alone is 1.09–1.13× MTP-3's and 1.25× DFlash2 k = 3's, because the target
  verifies 8 rows per sequence instead of 4. Net +2%.
- **k = 7, Kaggle (RTX PRO 6000):** acceptance length 3.34 against v6's 2.74, but generation is 10% slower over
  the first 30 minutes. The decode part of a step costs 1.21× MTP-3's, and the prefix-hit rate stays at 88–89%
  against 92–93%, which adds prefill.

Drafting is cheaper than MTP's three sequential passes: at k = 3 the decode part of a step is 0.87–0.90× MTP-3's on
the B200. At k = 7 the extra verify rows outweigh that saving. [Step time](#step-time-drafting-vs-verification) has
the split.

### The checkpoint

| | |
|---|---|
| Source | HF [`incoai/Qwen3.8-Flash-Next-DFlash2@a22a1744`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/tree/a22a1744e85c17364dcc54a6a420af7135d4c7f6) ([`config.json`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/blob/a22a1744e85c17364dcc54a6a420af7135d4c7f6/config.json)) |
| Our Kaggle copy | `jonathanwang2022/qwen3-8-flash-next-dflash2/Transformers/default/1` (private) |
| File | `model.safetensors`, 1,206,140,008 B, BF16, sha256 `d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f` (HF `x-linked-etag`) |
| Layers | 5, all `sliding_attention`, window 2048, non-causal (`is_causal: false`); hidden 2560, 32 query / 8 KV heads × 128 |
| Draft block | `block_size` 8; `selector_top_k` 16, `selector_rank` 256 |
| Target taps | `target_layer_ids` [5, 15, 25, 35, 45] of the target's 48 layers |
| Embed / lm_head | not in the checkpoint; shared with the target |

Serving it on Lord's runtime takes setup-time patches,
[`DFLASH_RUNTIME_PATCHES`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/6fd992579012872e5f8926468d78f4c7fb53137f/ARC3-Inference/inference/framework/lordhansolo/kaggle.py#L1621-L1822)
in the launcher (branch `dflash2-drafter`). They are:
- admitting `dflash` for Qwen4Exp;
- guarding the MTP hidden-state buffer;
- capturing the five taps in the target;
- rounding the QSA key ring for k > 4;
- the two prefix-cache fixes below.

### KV pool

From each run's `vllm-openai-server.log` ("GPU KV cache size", "Available KV cache memory", "kv cache group sizes"):

| Run | Drafter | KV pool | Group block sizes |
|---|---|---|---|
| porting-advaith v6 | Lord MTP-3 | 1,564,172 tokens / 21.73 GiB | 1664 × 4, 8 |
| dflash2-k3 v2 | DFlash2, k = 3 | 1,616,364 tokens / 22.23 GiB | 1664 × 6, 8 |
| dflash2-k7 v3 | DFlash2, k = 7 | 1,580,652 tokens / 21.91 GiB | 1744 × 6, 16 |

- DFlash2 adds two sliding-window groups. The pool is still larger: "Model loading took" 69.11 GiB against v6's
  69.54 GiB (drafter 1.2 GB vs 1.6 GB).
- Every block of any group costs the same page in Lord's pool (22.23 GiB / 971 blocks ≈ 24.6 MB per block). So each
  sequence's sliding-window blocks (about 3 per group) cost full blocks, even though they hold only 2048 tokens.
- With k = 7 the GDN conv state grows by 4 rows, and the attention block grows from 1664 to 1744 to stay at least as
  large as the GDN page. 1744 is not a multiple of 128, so
  [`dflash2-k7`](https://www.kaggle.com/code/jonathanwang2022/dflash2-k7) v2 died at startup with "Invalid
  prefix_match_unit=128; prefix-cacheable KV cache group block sizes must be divisible by prefix_match_unit"
  (`runs/kaggle/jonathanwang2022/dflash2-k7/v02/vllm-openai-server.log` line 199).
- v3 uses `prefix_match_unit` 16
  ([`configs/dflash2-k7/v3.json`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/6fd992579012872e5f8926468d78f4c7fb53137f/ARC3-Inference/configs/dflash2-k7/v3.json)).

### Prefix cache

Without patches, DFlash2's prefix-hit rate collapses. Two runtime patches fix it, and both are needed (B200 table below):
- **Draft-group rule**
  ([L1722-L1733](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/6fd992579012872e5f8926468d78f4c7fb53137f/ARC3-Inference/inference/framework/lordhansolo/kaggle.py#L1722-L1733)).
  - vLLM did not identify DFlash2's KV group as the drafter's, so it applied the EAGLE drop to every group.
  - The target's GDN groups then fell back to a recurrent checkpoint they do not keep, and turns missed down to
    0 tokens.
  - Adding `dflash` to `_uses_trailing_mtp_layers` flags exactly the drafter's group.
- **Replay-tail hold**
  ([L1734-L1821](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/6fd992579012872e5f8926468d78f4c7fb53137f/ARC3-Inference/inference/framework/lordhansolo/kaggle.py#L1734-L1821)).
  - A hit at the replay boundary (the previous prompt's end) needs the sliding-window blocks just below it.
  - The window frees those blocks during decode, so they are evicted before the request's full-attention blocks.
  - The request now keeps them out of its block table until it finishes. The cost is up to 3 more blocks per
    sliding-window group per running request.

### Served on Kaggle, k = 3, no prefix-cache patches

[`jonathanwang2022/dflash2-k3`](https://www.kaggle.com/code/jonathanwang2022/dflash2-k3) v2
([`fc12dafd`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/commit/fc12dafdeaada5ab6fa1f1cf73ae08a15f51bc81),
[`configs/dflash2-k3/v2.json`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/fc12dafdeaada5ab6fa1f1cf73ae08a15f51bc81/ARC3-Inference/configs/dflash2-k3/v2.json),
45-minute public run) against [`porting-advaith`](https://www.kaggle.com/code/jonathanwang2022/porting-advaith) v6.
Per 10 minutes from the first stats line: mean "Avg generation throughput" / last "Prefix cache hit rate".

| Minutes | 0–10 | 10–20 | 20–30 | 30–40 |
|---|---|---|---|---|
| dflash2-k3 v2 | 764 / 74% | 382 / 58% | 283 / 47% | 359 / 43% |
| porting-advaith v6 | 909 / 92% | 930 / 93% | 916 / 93% | 869 / 93% |

- Acceptance length: 2.676 (0.757 / 0.554 / 0.397), against v6's 2.743 (0.767 / 0.568 / 0.412). Both are from
  `drafter_stats accept` on the runs' `vllm-openai-server.log`.
- The collapse is the prefix cache, not the drafter. The draft attention runs on `TRITON_ATTN` (v1 failed when
  FlashInfer JIT-built its sliding-window kernel against a missing `libcudart.so`).

### B200 replay, both patches

Frida `ixb2`, Lord's runtime rebuilt from his image and overlay, each arm's Kaggle setup with
`--kv-cache-memory-bytes` set to its Kaggle pool. The load is recorded GRPO episodes (porting-advaith v6 harness)
replayed at 14 streams from 40K-token contexts, with the harness's sampling, over a 900-s window
([`scripts/dflash2_b200/README.md`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/6fd992579012872e5f8926468d78f4c7fb53137f/ARC3-Inference/scripts/dflash2_b200/README.md),
[`bench.py`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/6fd992579012872e5f8926468d78f4c7fb53137f/ARC3-Inference/scripts/dflash2_b200/bench.py)).
Numbers are from each arm's `bench/summary.json` on Frida in
`/shared/workspace/arc-agi-3/dflash2_b200/out/<job>/<arm>/`. The acceptance length is 1 + the sum of the
per-position rates. The step time is acceptance length ÷ gen tok/s, relative to MTP-3, prefill included.

| Job | Drafter | Patches | Prefix hit | Gen tok/s | Acceptance per position | Acceptance length | Step time vs MTP-3 |
|---|---|---|---|---|---|---|---|
| 209821 | MTP-3 (v6) | n/a | 92.8% | 1,298 | .785 .610 .470 | 2.865 | 1 |
| 209821 | DFlash2 k = 3 | none | 47.4% | 545 | .791 .619 .485 | 2.894 | |
| 209829 | DFlash2 k = 3 | rule only | 62.1% | 681 | .786 .612 .478 | 2.876 | |
| 209829 | DFlash2 k = 3 | rule + hold | 90.1% | 1,315 | .780 .600 .463 | 2.843 | 0.98 |
| 209821 | DFlash2 k = 7 | none | 41.5% | 505 | .754 .559 .418 .324 .258 .211 .176 | 3.700 | |
| 209829 | DFlash2 k = 7 | rule + hold | 90.0% | **1,329** | .744 .552 .414 .318 .253 .207 .174 | **3.662** | **1.25** |

What the table shows:
- Both patches together bring DFlash2 to MTP-3's hit rate. The rule alone gets 62%, and the hold alone 54% (job
  209824, in the README).
- At positions 1–3, DFlash2 accepts slightly less than MTP-3 (k = 7: .744 / .552 / .414 against .785 / .610 / .470).
  Positions 4–7 add 0.95 tokens per step, at the price of 4 more verify rows per sequence.
- The step-time column is the ratio of two measured numbers, filled only for the patched rows; an unpatched run's gen
  tok/s is set by prefill. [Step time](#step-time-drafting-vs-verification) splits it into decode and prefill.
- B200 tokens/s are not Kaggle numbers.
- Unpatched hit rates here are from `summary.json` and differ from the README's table by up to 0.9 points
  (42.1% / 48.3%). The patched rows agree.

### Served on Kaggle, k = 7, both patches

[`jonathanwang2022/dflash2-k7`](https://www.kaggle.com/code/jonathanwang2022/dflash2-k7) v3
([`6fd99257`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/commit/6fd992579012872e5f8926468d78f4c7fb53137f),
[`configs/dflash2-k7/v3.json`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/6fd992579012872e5f8926468d78f4c7fb53137f/ARC3-Inference/configs/dflash2-k7/v3.json),
2026-10-31, 55-minute run), against [`porting-advaith`](https://www.kaggle.com/code/jonathanwang2022/porting-advaith)
v6. The games ran 14:42–15:13 UTC, when the run's time cap stopped them (KV use 80% → 16%, then 1–3 requests drained
until 15:15). So the comparison covers the first 30 minutes of stats lines in each `vllm-openai-server.log`
(180 lines each).

| First 30 min | dflash2-k7 v3 | porting-advaith v6 | change |
|---|---|---|---|
| Acceptance length (`drafter_stats accept`, whole run) | **3.336** | 2.743 | +21.6% |
| Per position | .740 .529 .376 .270 .197 .148 .111 | .767 .568 .412 | |
| Mean "Avg generation throughput" | **825** tok/s | 918 tok/s | −10.1% |
| Mean "Avg prompt throughput" (prefill) | 3,106 tok/s | 1,923 tok/s | +62% |
| Mean running requests | 13.73 | 13.91 | |
| Prefix hit rate at 2 / 10 / 20 / 30 min | 61% / 85% / 88% / 89% | 92% from 10 min | |

The harness's own 10-minute summaries in each `stdout.log` (the token difference ÷ the time between summaries, each time being "total tokens" ÷ "generated tokens/sec") show the
gap closing:

| Minutes of play | 0–10 | 10–20 | 20–30 |
|---|---|---|---|
| dflash2-k7 v3 | 702 tok/s | 744 | 847 |
| porting-advaith v6 | 806 | 956 | 852 |

What this shows:
- **Acceptance.** Kaggle accepts less than the B200 replay: 3.34 against 3.66, almost all of it at positions
  4–7 (.270 / .197 / .148 / .111 against .318 / .253 / .207 / .174). MTP-3 loses less in the same comparison
  (2.74 against 2.87). Why is not known: the replay starts at 40K-token contexts and skips early-game turns.
- **Prefix hits.** They now work on Kaggle, but they reach 85% only after 10 minutes and level off at 88–89%, below
  v6's 92–93%. The extra prefill is +62%.
- **Throughput.** With +21.6% tokens per step and −10.1% generation, a k = 7 step takes about 1.35× an MTP-3 step
  over these 30 minutes, prefill included. Over the whole run it takes 1.28× (57.4 vs 45.0 ms, below).
- **Over time.** Most of the shortfall is in the first 20 minutes, while the hit rate builds. In minutes 20–30
  the harness rates are equal (847 vs 852), and the server rate is −6% (864 vs 916).

This is one run, and 30 minutes of play is not a score comparison.

### Step time: drafting vs verification

From `drafter_stats steps` on each run's server log (10-s intervals with at least 12 running requests):

| Run | GPU | Mean step | Prefill tokens per step | Fit: fixed + per prefill token | R² |
|---|---|---|---|---|---|
| porting-advaith v6, MTP-3 | RTX PRO 6000 | 45.0 ms | 130 | 35.5 ms + 73.0 µs | 0.82 |
| dflash2-k7 v3 | RTX PRO 6000 | 57.4 ms | 199 | 44.9 ms + 62.8 µs | 0.80 |
| dflash2-k3 v2 (no patches) | RTX PRO 6000 | 101.5 ms | 905 | 48.5 ms + 58.6 µs | 0.75 |
| 209821 C, MTP-3 | B200 | 32.0 ms | 166 | 25.2 ms + 40.8 µs | 0.93 |
| 209829 B, DFlash2 k = 3 | B200 | 31.3 ms | 210 | 20.4 ms + 52.0 µs | 0.81 |
| 209829 A, DFlash2 k = 7 | B200 | 40.1 ms | 283 | 25.2 ms + 52.4 µs | 0.96 |

When a run's prefill varies little, the fixed term trades off against the slope: B200 k = 7 and MTP-3 both fit
25.2 ms. A steadier split subtracts prefill at one per-token cost per GPU, across the range of that GPU's fitted
slopes:

| Decode part of a step | MTP-3 | DFlash2 k = 3 | DFlash2 k = 7 |
|---|---|---|---|
| B200 (prefill at 41–52 µs per token) | 23.4–25.2 ms | 20.4–22.7 ms (0.87–0.90×) | 25.4–28.5 ms (1.09–1.13×) |
| RTX PRO 6000 (prefill at 63–73 µs per token) | 35.5–36.8 ms | not measured | 42.9–44.9 ms (1.21–1.22×) |

What the tables show:
- **Drafting.** At k = 3 both drafters' steps verify 4 rows per sequence, so the gap is the drafter.
  - DFlash2's single pass costs about 3 ms less per step on the B200 than MTP's three sequential passes, 10–13% of
    the step.
  - DFlash2 drafts the whole block in that one pass, so its draft cost barely grows with k.
- **Verification.** With the same drafter, k = 3 → 7 adds 25% to the decode part on the B200.
  - The target runs 14 × (k + 1) rows through its 48 layers: 56 rows at k = 3, 112 at k = 7.
  - Each token routes to 10 of 512 experts. With independent routing, a layer touches about 342 experts at 56 rows
    and 456 at 112: 33% more expert weights to read. Attention and GDN work also grow linearly with rows.
  - So verifying more is not free for this MoE, unlike a dense model at small batch. Even for one sequence, 8 rows
    touch about 75 experts against 39 for 4.
- **RTX PRO 6000.** The k = 7 decode part is 1.21× MTP-3's, against 1.09–1.13× on the B200. With no clean RTX
  k = 3 run, the drafting saving and the verify growth cannot be separated there.
- **Prefill.** k = 7 computes more prefill per second than MTP-3 on both GPUs. On the B200 that is
  283 / 40.1 ms ≈ 7.1K against 166 / 32.0 ms ≈ 5.2K tokens/s. This is the cost of the lower hit rate (90% vs
  92.8% on the B200, 88–89% vs 92–93% on Kaggle).
- **dflash2-k3 v2 is not used.** Its fixed term is inflated: during its prefill storm, most steps were mixed
  prefill and decode batches.

The step count assumes that every running request drafts in every step. A request in prefill does not draft, so
steps are slightly undercounted. That is at most about one request in 14 while a 2,048-token chunk prefills, and
more in prefill-heavy runs.

### k = 4 with a BF16 drafter KV (next)

Prediction for Kaggle, against v6's MTP-3:
- **Acceptance.** The first four positions of k7 v3 give 1 + .740 + .529 + .376 + .270 = 2.92. The same sum over
  three positions gives 2.65, and k3 v2 measured 2.68.
- **Decode part of a step.** RTX k = 3 is not measured. B200's ratios (k3 / MTP-3 0.87–0.90×, k7 / k3 1.25×),
  applied to the RTX numbers, give about 31–36 ms. k = 4 adds one row per sequence: 383 against 342 experts, 36% of
  the k = 3 → 7 step. That gives about 36–39 ms.
- **Throughput.** 2.92 / (36–39 ms + 9–13.5 ms of prefill) against 2.74 / 45.0 ms: −9% to +6%.
- **BF16 drafter KV.** For MTP it raised acceptance by 4.8% at the same step time
  ([above](#measured-in-vllm-on-frida)).
  - MTP lost most of that at positions 2–3, where it reads its own drafts back from the cache. DFlash2 drafts its
    block in one pass and never does that, so a smaller gain is expected. It is not measured.
  - The sliding-window group's pages double (2,048 → 4,096 B per token per layer). Whether it boots and what the KV
    pool becomes are also unknown.

The run measures those unknowns: the RTX decode cost at small k and DFlash2 with a BF16 drafter KV. The MTP
comparison that counts now is `tuned-mtp-fp8-kv` (fine-tuned MTP head with FP8 drafter KV, [the chosen
setup](#decision-fine-tuned-head-with-fp8-drafter-kv)), not v6.

## Reproduce

From `ARC3-Inference/` ([`inference/tools/drafter_stats.py`](../../ARC3-Inference/inference/tools/drafter_stats.py)).
It reads safetensors headers with HTTP range requests and downloads no weights. Kaggle files need
`~/.kaggle/access_token`.

```sh
P=https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8/resolve/07915ee79ec217c117e8a57bf2557a4a1418c10f
python -m inference.tools.drafter_stats bytes $P/mtp_nvfp4/mtp-experts-nvfp4.safetensors \
  $P/mtp_nvfp4/carry-model-bf16-00011.safetensors $P/carry-model-bf16-00012.safetensors \
  $P/carry-model-bf16-00010.safetensors --prefix mtp.                       # Lord: 1.597 GB
python -m inference.tools.drafter_stats bytes \
  https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/resolve/4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897/model_extra_tensors.safetensors \
  --prefix mtp.                                                             # Intel: 5.214 GB
K=https://www.kaggle.com/api/v1/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1/download
python -m inference.tools.drafter_stats bytes $K/mtp-dense.safetensors $K/mtp-routed-experts-int4.safetensors  # 4.140 GB
python -m inference.tools.drafter_stats accept runs/kaggle/jonathanwang2022/porting-advaith/v06/vllm-openai-server.log
python -m inference.tools.drafter_stats accept runs/kaggle/jonathanwang2022/dflash2-k3/v02/vllm-openai-server.log   # 2.676
python -m inference.tools.drafter_stats accept runs/kaggle/jonathanwang2022/dflash2-k7/v03/vllm-openai-server.log   # 3.336
for r in porting-advaith/v06 dflash2-k7/v03 dflash2-k3/v02; do                                    # step-time table
  python -m inference.tools.drafter_stats steps runs/kaggle/jonathanwang2022/$r/vllm-openai-server.log; done
# B200 rows: the same on /shared/workspace/arc-agi-3/dflash2_b200/out/{209821/C,209829/B,209829/A}/vllm-openai-server.log
make pull-kaggle NOTEBOOK=jonathanwang2022/dflash2-k7 VERSION=3                                             # the k = 7 run
make pull-kaggle NOTEBOOK=jonathanwang2022/tuned-mtp-bf16-kv VERSION=1                                      # fine-tuned + BF16
python -m inference.tools.drafter_stats accept runs/kaggle/jonathanwang2022/tuned-mtp-bf16-kv/v01/vllm-openai-server.log  # 2.888
python -m inference.tools.drafter_stats accept runs/kaggle/jonathanwang2022/porting-advaith/v06/vllm-openai-server.log --until 17.7  # 2.668
python -m inference.tools.drafter_stats timeline runs/kaggle/jonathanwang2022/tuned-mtp-bf16-kv/v01/vllm-openai-server.log --bucket 5
python -m inference.tools.drafter_stats timeline runs/kaggle/jonathanwang2022/porting-advaith/v06/vllm-openai-server.log --bucket 5
python3 runs/local/frida/vllm-replay-draft-kv/analyze.py runs/local/frida/vllm-replay-draft-kv/runs/replay3-tuned-bf16  # 3.020
python3 runs/local/frida/vllm-replay-draft-kv/analyze.py runs/local/frida/vllm-replay-draft-kv/runs/replay3-tuned-fp8   # 2.898
make pull-kaggle NOTEBOOK=jonathanwang2022/tuned-mtp-fp8-kv VERSION=1                                       # fine-tuned + FP8
for r in tuned-mtp-fp8-kv/v01 tuned-mtp-bf16-kv/v01 porting-advaith/v06; do
  python -m inference.tools.drafter_stats accept runs/kaggle/jonathanwang2022/$r/vllm-openai-server.log --until 15; done  # 2.805 / 2.882 / 2.662
curl -sIL https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/resolve/a22a1744e85c17364dcc54a6a420af7135d4c7f6/model.safetensors \
  | grep -i x-linked                                                       # DFlash2: 1,206,140,008 B, sha256
```

Two checks were done in a Python session on Frida against the shared copies of both checkpoints (2026-10-30), not with
a script in the repo:
- the 29-tensor bit-identity of Lord's and Intel's non-expert MTP tensors;
- the NVFP4 dequantization against Intel's BF16 experts. The same decode is `dequant_nvfp4` in
  `inference/rft/mtp_train.py` on branch `mtp-data`.
