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
