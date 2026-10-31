# Flash-Next draft heads: quantization, size and acceptance

All three drafters we have served are the same MTP layer of Qwen3.8-Flash-Next (one decoder layer with QSA attention,
hyper-connections and a 512-expert MoE, plus `fc_embedding`, `fc_hidden`, two pre-norms and a final mixer). They differ
only in how the 512 routed experts are stored, and whether the checkpoint carries its own `embed_tokens` / `lm_head`.

| Drafter | Source | Routed experts | Everything else in the MTP layer | Own embed + lm_head | Bytes |
|---|---|---|---|---|---|
| Intel BF16 MTP (the target checkpoint's own layer) | Kaggle [`dfranzen/intel-qwen3.8-flash-next-w4a16-autoround`](https://www.kaggle.com/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround) = HF [`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/tree/4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897), `model_extra_tensors.safetensors` | BF16 (5.033 GB) | BF16 (0.181 GB) | no (the target's) | **5.214 GB** |
| Lord NVFP4 MTP | Kaggle [`lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8`](https://www.kaggle.com/models/lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8), variation `hf-mixed-mtp-nvfp4/1` = HF [`primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8@07915ee`](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8/tree/07915ee79ec217c117e8a57bf2557a4a1418c10f/mtp_nvfp4) `mtp_nvfp4/` | NVFP4 W4A4 (1.416 GB: U8 codes 1.258 + FP8 E4M3 scales 0.157) | BF16 (0.181 GB), **bit-identical to Intel's** (29 of 29 tensors) | no (the target's) | **1.597 GB** |
| Daniel's albucino INT4 | Kaggle [`dfranzen/albucino-qwen3-8-flash-next-drafter`](https://www.kaggle.com/models/dfranzen/albucino-qwen3-8-flash-next-drafter) `default/1` | INT4 (1.416 GB: I32-packed codes 1.258 + BF16 scales 0.157) | BF16 (0.181 GB) | yes, BF16 (2.542 GB) | **4.140 GB** |

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

Conclusion: keeping the MTP drafter's own KV cache in FP8 costs about 0.10–0.13 acceptance length. A BF16 drafter KV
is worth testing in serving. It costs an estimated 6–7% of the KV pool, which 14 streams should still fit (v6 peaked at
93%). The target's KV can stay FP8. Two caveats: the acceptance numbers are a simulation, and plain `auto` does not
start on Lord's runtime.

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
- **Why** (my reading of the code; not traced in a run): the BF16 draft page is twice the FP8 page of the target's
  layers, and
  [`unify_kv_cache_spec_page_size`](https://github.com/vllm-project/vllm/blob/e9757321527ca1ecd514c07c1418dd2c53da3d19/vllm/v1/core/kv_cache_utils.py#L1368)
  equalizes pages by raising the other layers' block size and padding the Mamba pages to the largest page. That breaks
  the 1664-token alignment Lord's runtime sets between attention and Mamba pages. Lord's overlay replaces this file, so
  the link is to the upstream base.
- **The fix to try:** a setup-time patch, like the DFlash ones, that halves the drafter group's block size (1664 → 832)
  before the pages are unified. Its BF16 page then equals the others, and nothing else changes size. The scheduler block
  stays LCM = 1664, and `prefix_match_unit` 128 divides 832. Not written yet.

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
```

Two checks were done in a Python session on Frida against the shared copies of both checkpoints (2026-10-30), not with
a script in the repo:
- the 29-tensor bit-identity of Lord's and Intel's non-expert MTP tensors;
- the NVFP4 dequantization against Intel's BF16 experts. The same decode is `dequant_nvfp4` in
  `inference/rft/mtp_train.py` on branch `mtp-data`.
