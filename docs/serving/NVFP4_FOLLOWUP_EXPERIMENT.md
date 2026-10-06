# Follow-up experiment: Lord HanSolo's model on Daniel's SGLang

2026-10-05 — implemented; first Kaggle runs pushed, not yet evaluated.

**Status.** Configs [`lord-nvfp4-on-pennyroyal/v1.json`](../../ARC3-Inference/configs/lord-nvfp4-on-pennyroyal/v1.json) (candidate) and [`pennyroyal-w4a16-baseline/v1.json`](../../ARC3-Inference/configs/pennyroyal-w4a16-baseline/v1.json) (baseline), both `thinking-priority/v2` otherwise. Launcher changes: [`pennyroyal_launch.py`](../../ARC3-Inference/inference/framework/pennyroyal_launch.py), [`pennyroyal_views.py`](../../ARC3-Inference/inference/framework/pennyroyal_views.py). Deviations from the plan below, by request:

- **FP8 PLE, not BF16.** RadixArk's table (`keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1`, the one our duck-harness runs used) replaces Lord's BF16 shards in a symlink view. Lord's config already declares `ple_embedding_dtype: float8_e4m3fn`, and Pennyroyal loads FP8 shards into its pinned host table. GPU tensors are unchanged.
- **Lord's shard window prefetch** replaces Daniel's one-shard lookahead in the candidate only.
- **105 public minutes** instead of 85, for both runs. The 532-minute competition cap is unchanged.

Draft: the target's embedded NVFP4 MTP head, served from a view of its 4 MTP/embedding/LM-head shards. FR-Spec 64K map kept: primitive-ai's tokenizer has a different SHA (re-serialized merges, `\p{M}` in the pre-tokenizer split), but the vocab and added tokens are identical id for id. The chat template is byte-identical to Intel's.

## 1. Swap the model

Start with Daniel's patched Pennyroyal v2.5.3 and replace Intel W4A16 with Lord HanSolo's mixed NVFP4/FP8 model. Keep the harness, prompts, tools, scheduler, ten concurrent streams and 128K context policy unchanged.

- **Baseline:** Daniel's existing engine, W4A16 target and INT4 draft.
- **Candidate:** the same engine with `lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8/PyTorch/hf-mixed-mtp-nvfp4/1`. This packages [Primitive's mixed checkpoint][model] at revision `07915ee79ec217c117e8a57bf2557a4a1418c10f` with its NVFP4 MTP overlay already applied.

Make the adjustments needed to serve that target/draft combination correctly:

- Support its compressed-tensors mixed FP8/NVFP4 format and verify the selected kernels. Pennyroyal's standard NVFP4 recipe uses ModelOpt, so changing the model path alone may not suffice.
- Preserve **BF16 PLE** and its host offload. The checkpoint has a misleading FP8 PLE config hint; [Lord's safeguard][lord] prevents the [Pennyroyal loader][loader] from downcasting the BF16 shards.
- Adapt MTP loading and verify the draft vocabulary mapping. Keep Daniel's speculative settings where compatible and record necessary differences.

Check generation, images, tool calls and long-context continuation. Then compare the same games/passes and budgets: sustained output tok/s, prefix reuse, cache capacity/stalls and ARC score. Record actual prompt lengths and measure startup separately. Use the [A/B guide](../rules/t-test.md) for the game comparison.

This first experiment tests the model swap. A separate vLLM comparison can follow later. Background: [Milestone 2 discussion][discussion] and [memory budget](MEMORY_BUDGET.md).

## 2. Add improvements one at a time

After establishing the candidate above, test these additions independently. The table also records what to leave alone initially.

| Candidate | Why it may help / existing coverage | Action |
| --- | --- | --- |
| **Lord's BF16 GEMMs at M=40–64** | Daniel's added kernel stops at M=32. Ten streams × four verification positions can produce M=40. | First throughput candidate: profile SGLang's actual shapes and adapt matching kernels for projections that remain BF16. |
| **Lord's pinned-host `embed_tokens`** | Daniel offloads PLE but keeps the ordinary input embedding on GPU. Lord's implementation frees about 1.18 GiB. | Adapt lookup and target/draft sharing; measure added KV capacity and PCIe cost. |
| **Draft-only loading — Lord and Siriki** | Daniel already uses a separate draft, but the new full checkpoint may cause MTP loading to scan unrelated target shards. | Filter the new draft's actual tensors/shards before reading; measure startup savings. |
| **Siriki's parallel PLE row copy** | Daniel parallelizes shard prefetch, while the PLE copy itself remains serial. | Test disjoint CPU row copies in parallel; startup benefit only. |
| **Lord's align-state / prompt-tail caching** | Daniel already retains sparse Mamba checkpoints and refreshes recency on unlock; Lord's exact boundary logic is vLLM-specific. | Leave initially. Investigate a SGLang equivalent only if traces show avoidable tail recomputation. |
| **32K draft vocabulary** | Daniel already uses a 64K FR-Spec map. | Later tuning: compare acceptance and throughput. |
| **HiCache** | Already available in Pennyroyal; consumes host RAM alongside BF16 PLE. | Consider only if cache eviction becomes a measured bottleneck. |
| **Spec-state budget / Marlin dtype / prefetch-once** | Daniel covers the first two and has bounded prefetch lookahead. | Keep his implementation; do not stack Siriki's global prefetch-once flag onto it. |

Patch sources: [Daniel's builder][daniel], [Lord's manifest][lord] and [runtime bundle][runtime], [Siriki's startup patches][siriki]. The ordinary input-embedding offload above is verified in Lord's bundle; Siriki's published notebook offloads PLE.

[model]: https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8
[discussion]: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792
[daniel]: https://github.com/da-fr/arc-agi-3-solution/blob/10882e332d883af3e61616df8c8a17d03a328a75/serving/build_bundle_pennyroyal.sh
[lord]: https://github.com/tonghuikang/daniel-franzen-arc-agi-3/blob/f472820b24f85e4bfe7790c66bce7d6463b5d568/kaggle/lordhansolo/dataset-vllm-main-e975732-arc3/PATCH_IDENTITY.json
[loader]: https://github.com/jpezzulli/sglang-rtxpro6000/blob/d00d88efc8d6281b12be4f4073126aec95038c55/python/sglang/srt/models/qwen4_exp.py
[runtime]: https://www.kaggle.com/datasets/lordhansolo/vllm-main-e975732-arc3
[siriki]: https://github.com/LohitSiriki/arc-agi-3-milestone2-solution/blob/af105202327dcf35bd670a50b2558adab12e8dd3/cells/11_4_import_the_bundled_source_and_boot_sglang.py
