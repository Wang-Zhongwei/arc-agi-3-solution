# Lord Hansolo's vLLM with Daniel's weights

[`matijaludvig/arc3-vcintel-25x1`](https://www.kaggle.com/code/matijaludvig/arc3-vcintel-25x1) (script version 356782280) runs the arc3-vc harness on Lord Hansolo's patched vLLM runtime (`lordhansolo/vllm-main-e975732-arc3`, vLLM e975732 plus the arc3 overlay) with Daniel's weights (`dfranzen/intel-qwen3.8-flash-next-w4a16-autoround`). That checkpoint is Intel AutoRound W4A16, and our Pennyroyal (SGLang) runs serve the same one. This page records what our harness needs to run the same serving.

## What already works

Since the merge of `main`, `kaggle.serving: "lordhansolo"` serves through Lord's setup script, with this config's harness settings applied last (`inference/framework/kaggle.py`, `lordhansolo_setup_command`). The vcintel notebook's serving changes are already covered by config:

| vcintel change | Here |
|---|---|
| Daniel's W4A16 AutoRound weights | `kaggle.model_source` |
| `--quantization` left to the checkpoint's `config.json` | `kaggle.vllm.quantization: ""` |
| MTP 3, 32k draft vocabulary, FP8 KV, GDN `triton`, `gpu_memory_utilization` 0.98, `max_model_len` 147072 | `kaggle.vllm.*`, as in the reset A/B configs |
| `max_pixels` 409,600 (a 64×64 grid at upscale 10) | Derived from `multimodal.upscale` |
| No forced `reasoning_effort` (template `{"preserve_thinking": true}`) | `kaggle.default_chat_template_kwargs`. The adapter also drops Lord's `LOCAL_ANALYZER_REASONING_EFFORT=xhigh` from the handoff. |
| `max_num_seqs` equal to the active streams | `kaggle.vllm.max_num_seqs` (10 here, as `ARC3_MAX_ACTIVE_STREAMS`) |
| `ARC3_MAX_ANALYZER_FAILURES=150`, to ride out a vLLM watchdog restart | `harness_env` |

## What was missing

Lord's runtime builds the PLE embedding only for the NVFP4, ModelOpt and FP8 configs of his checkpoints, and raises for INC (AutoRound), whose PLE table is unquantized BF16. The vcintel notebook patches the unpacked runtime's `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` to route `INCConfig` to the unquantized PLE method. The setup script now does the same, in `route_inc_ple_to_unquantized`, right after the runtime is unpacked. The extra branch matches only `INCConfig`, so Lord's own checkpoints take the same path as before. A runtime without the expected source fails at setup only when the checkpoint is AutoRound. `tests/test_lordhansolo_inc_ple.py` covers both cases on a stand-in runtime file.

## Config

[`configs/thinking-budget-priority-one-wave-vllm/v1.json`](../../ARC3-Inference/configs/thinking-budget-priority-one-wave-vllm/v1.json) is the one-wave config with serving swapped. It takes the harness, prompts and settings of `configs/thinking-budget-priority-one-wave/v3.json`, including sampled animation frames for the last action. It has no Pennyroyal drafter, uses Lord's runtime dataset and the vLLM block above, and uses its own notebook (`thinking-budget-priority-one-wave-vllm`). A `KAGGLE_DRY_RUN=true` build renders a setup script that parses, defines and calls the INC route, and hands over this config's harness env (`LOCAL_ANALYZER_REASONING_EFFORT` removed).

## GEMM M and Mamba cache: vLLM against SGLang

From the server logs of the vcintel run (`vllm-openai-server.log`, latest output of `matijaludvig/arc3-vcintel-25x1`) and of our Pennyroyal run `ticbudja/thinking-budget-priority-one-wave` v1 (`serve.log`). Both serve the same AutoRound weights to the 25 public games. M is the number of tokens one forward pass sends through the weight GEMMs.

| | vcintel (vLLM) | Ours (Pennyroyal, SGLang) |
|---|---|---|
| **Decode M**: sequences × (1 + 3 MTP drafts) | up to 13 × 4 = **52**. CUDA graphs are captured at `[1, 2, 4, 8, 16, 24, 32, 40, 48, 56]`, so 52 runs padded to 56. At the usual 12–13 running that's M = 48–52. | up to 10 × 4 = **40** (`speculative_num_draft_tokens` 4) |
| **Prefill M** per step | at most **2,048** tokens (`max_num_batched_tokens`), decode and prefill sharing one step | chunks of **8,192** (`chunked_prefill_size`), `max_prefill_tokens` 16,384 |
| **Mamba cache** | No separate cache. Mamba state pages live in the shared 18.33 GiB pool (`align` mode with prefix caching), padded by 0.48% to match the 1,664-token attention page. Each running sequence holds one page, plus prefix-cache checkpoints. Lord's RecoverSSM replaces per-draft state with 1.635 MiB of records per GDN layer. | A fixed pool of **60 states** (`max_mamba_cache_size`): 3.22 GB SSM state + 0.13 GB conv state + 0.05 GB conv window ≈ 3.4 GB. On average 57% was in use (about 34 states), and never more than 40. |

- vLLM's decode GEMMs run at a larger M (48–56 against at most 40), which suits the small-M, bandwidth-bound decode better. Its prefill chunks are also a quarter of SGLang's size. That is a plausible reason its decode steps stall less, but it has not been measured.
- On SGLang, the 60-state Mamba pool never had more than 40 in use. That is about 1.1 GB that could go to the KV cache instead.

## Not yet checked on a GPU

- **The INC route itself.** It follows the vcintel notebook's patch line for line, but the runtime file exists only on Kaggle.
- **The thinking budget.** On vLLM the harness sends `thinking_token_budget` (8,192 here). Whether Lord's build enforces it is untested; Pennyroyal enforces it in SGLang's strict mode. If it does not, thinking can run to `max_output` (12,288).
- **Images per request.** The launch caps each image at 409,600 pixels and disables video, but sets no image count. Sampled animation frames add up to eight 128×128 images per turn.
- **Startup.** Lord's setup blocks until vLLM answers, for up to 3,600 s, before any game starts. The vcintel notebook warns its competition rerun is not supported, because Daniel's flow releases the harness after 12 minutes. Ours runs no game until the server is ready, as Lord's own submissions do, but the competition-rerun timing has not been tried with this config.
- **Streams.** vcintel ran 13 streams, from the Intel checkpoint's KV pool (1,319,052 tokens in the vLLM log). This config keeps the harness's 10.
