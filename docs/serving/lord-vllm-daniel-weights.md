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

## Not yet checked on a GPU

- **The INC route itself.** It follows the vcintel notebook's patch line for line, but the runtime file exists only on Kaggle.
- **The thinking budget.** On vLLM the harness sends `thinking_token_budget` (8,192 here). Whether Lord's build enforces it is untested; Pennyroyal enforces it in SGLang's strict mode. If it does not, thinking can run to `max_output` (12,288).
- **Images per request.** The launch caps each image at 409,600 pixels and disables video, but sets no image count. Sampled animation frames add up to eight 128×128 images per turn.
- **Startup.** Lord's setup blocks until vLLM answers, for up to 3,600 s, before any game starts. The vcintel notebook warns its competition rerun is not supported, because Daniel's flow releases the harness after 12 minutes. Ours runs no game until the server is ready, as Lord's own submissions do, but the competition-rerun timing has not been tried with this config.
- **Streams.** vcintel ran 13 streams, from the Intel checkpoint's KV pool (1.316M tokens). This config keeps the harness's 10.
