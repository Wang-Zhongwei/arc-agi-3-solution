# Serving memory budget: Milestone 2 top 3 vs duck-harness

Sources:

- The three Milestone 2 winners: Tong Hui Kang's comparison post, [Comparing the Milestone 2 winning solutions](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792) (2026-10-01).
- duck-harness: the full 25-game run `jonathanwang2022/adaptive-reasoning-ucb` v30 (config `configs/adaptive-reasoning-ucb/v30.json`): `vllm.log`, `kv-pool.json` and `flashnext-model-provenance.json`.

All sizes are GiB as the servers report them (SGLang prints GiB as "GB").

## Side by side


|                              | dfranzen                          | lordhansolo                                     | sirikilohit                       | ours                                                    |
| ---------------------------- | --------------------------------- | ----------------------------------------------- | --------------------------------- | ------------------------------------------------------- |
| Server                       | SGLang Pennyroyal v2.5.3          | vLLM 0.29.1rc1 nightly                          | SGLang Pennyroyal v2.5.0          | vLLM 0.28.1rc1 + NVIDIA Flash Next branch (spn runtime) |
| Target model                 | Intel W4A16 AutoRound             | primitive-ai NVFP4 + FP8 mixed                  | Intel W4A16 AutoRound             | Intel W4A16 AutoRound                                   |
| Speculative decoding         | 3 draft tokens, accept len 2.69 ³ | 3 draft tokens, built-in MTP, accept len 2.68 ⁶ | 3 draft tokens, accept len 2.64 ⁶ | none                                                    |
| **Target weights (GPU)**     | 69.85                             | 71.94 incl. MTP head                            | 69.78                             | 69.38                                                   |
| **Drafter weights (GPU)**    | 3.79 (Albucino INT4 MTP)          | — (built-in MTP head)                           | 3.79 (RadixArk NVFP4 MTP)         | — (no speculative decoding)                             |
| embed_tokens                 | on GPU                            | pinned host (1.18)                              | on GPU                            | on GPU                                                  |
| **PLE table (host RAM)**     | 95.37 BF16                        | 95.37 BF16                                      | 47.68 FP8                         | 47.68 FP8 ¹                                             |
| Other host memory            | —                                 | 1.18 embed_tokens                               | 48 hierarchical KV + Mamba tier   | —                                                       |
| **KV, target (attention)**   | 11.58                             | ≈16.22 ²                                        | 11.50                             | ≈11.95 ²                                                |
| **KV, draft**                | 0.96                              | (in target)                                     | 0.96                              | —                                                       |
| **Mamba state**              | 3.40                              | ≈3.47 ²                                         | 4.51                              | **≈8.30** ²                                             |
| KV + Mamba total             | 15.94                             | 19.69                                           | 16.97                             | 20.25                                                   |
| CUDA graphs                  | 1.14                              | 0.46                                            | 1.59                              | 0.39                                                    |
| GPU left free                | 4.25                              | 1.90                                            | 3.17                              | 1.90 ⁴                                                  |
| Pool sizing                  | mem fraction 0.96                 | gpu util 0.98, profiled                         | mem fraction 0.97                 | gpu util 0.98, profiled                                 |
| KV bytes per token (FP8)     | 12,295                            | 12,288 (assumed)                                | 12,295                            | 12,288                                                  |
| **KV pool (tokens)**         | 1,011,264                         | 1,417,100                                       | 1,004,288                         | 1,043,988                                               |
| Context (harness)            | 131,072                           | 127,488                                         | 69,632                            | 49,152                                                  |
| Streams                      | 10                                | 14                                              | 16                                | 25                                                      |
| KV pool used                 | 69% p50, 85% p90 ³                | 37% p50, 55% p90 ⁶                              | 47% p50, 75% p90 ⁶                | 62% p50, 68% p90                                        |
| Prefix cache hit             | 93.7% ³                           | 89.5% ⁶                                         | 82.2% ⁶                           | 58% p50, 66% p90                                        |
| Decode tok/s, p50            | 733 ³                             | 947 ⁶                                           | 900 ⁶                             | 606                                                     |
| Decode tok/s per stream, p50 | ~81 (9 running) ³                 | ~68 (14 running) ⁶                              | ~56 (16 running) ⁶                | ~25 (24 running)                                        |
| Decode tok/s, peak           | 946                               | 1,135                                           | 1,159                             | 874                                                     |
| Startup to ready             | 531 s                             | 544 s                                           | 615 s                             | ≥870 s ⁵                                                |


¹ RadixArk's FP8 PLE from `keithtyser/qwen3-8-flash-next-nvfp4`, the same table sirikilohit offloads. `flashnext-model-provenance.json` records 51,200,245,762 bytes in FP8 against 102,400,491,520 in BF16.

² vLLM reports one combined pool. The split uses **12,288 bytes per token** of FP8 attention KV, the harness's own figure in `kv-pool.json`, and treats the rest as Mamba state. Both SGLang setups agree: 11.58 GiB / 1,011,264 = 11.50 GiB / 1,004,288 ≈ 12,295 bytes per token.

³ From our copy of Franzen's setup (`dangthuygiang/arc3-v207-danibase-base-dfranzen` v05 `serve.log`).

⁵ vLLM's own start to ready: 712 s of model loading plus 95 s of engine init. The other columns count from notebook start, so ours is a lower bound.

⁴ vLLM's profile on the 94.97 GiB device: the 0.98 budget is 93.07 GiB. That covers 69.38 GiB of weights, 1.87 GiB of non-torch overhead, 1.57 GiB of peak activations, 0.39 GiB of CUDA graphs and the 20.25 GiB pool.

⁶ From each team's public notebook version, pulled into `duck-harness/ARC3-Inference/runs/kaggle/`. Kaggle doesn't publish logs of the scored submission reruns, so these are shorter interactive runs: lordhansolo [v293](https://www.kaggle.com/code/lordhansolo/built-on-tufa-labs-duck-harness-milestone-2), the version his notebook names as the Milestone 2 one, played 25 games for 9 minutes; sirikilohit [v1](https://www.kaggle.com/code/sirikilohit/arc-agi-3-duck-18-1gc-submit) ran about 29 minutes. Contexts never get long in runs this short, so KV usage and hit rate read low; sirikilohit reports 91–97% pool use in full runs. Decode speed and acceptance length are representative. Franzen's own public run (v1, 10 games, 25 minutes) agrees with our full copy: 723 tok/s p50, acceptance 2.67, 94.2% hit. SGLang hit rates are token-weighted over the run; vLLM's is its cumulative counter.

## Why lordhansolo has the biggest KV pool

His model isn't smaller. His NVFP4/FP8 target with its MTP head is 71.94 GiB plus 1.18 GiB of `embed_tokens` kept in host RAM, about 73.1 GiB. Franzen's target plus drafter is 73.64 GiB. The per-token KV cost is also the same: all four use FP8 KV, and his 1.417M tokens fit the 12,288 B/token rate with a normal ~3.5 GiB of Mamba left over.

The extra ~3.75 GiB, roughly 330k tokens, comes from how the GPU is budgeted:


| Source                                         | GiB gained vs dfranzen |
| ---------------------------------------------- | ---------------------- |
| No separate drafter (built-in MTP head)        | +1.70 weights          |
| Profiled 0.98 instead of a fixed 0.96 fraction | +2.35 free memory      |
| Smaller CUDA graphs (no separate draft graphs) | +0.68                  |


## Why ours has a large pool but few tokens

We give 20.25 GiB to the cache, more than anyone except lordhansolo, yet hold only 1.04M tokens. About 8.3 GiB goes to Mamba state, against 3.4–4.5 elsewhere. That's ~4.8 GiB, or ~420k tokens of attention KV.

vLLM's hybrid allocator in `align` mode, which it switches on with prefix caching, sizes attention and Mamba pages identically. The log shows it raised the attention block to 3,136 tokens and padded the Mamba page to match. With prefix caching, align mode also keeps Mamba states for cached blocks, not just for the up to 28 running sequences (`max_num_seqs=28`). This is the likely cause, not yet confirmed.

The pool isn't our bottleneck: with 24 streams and a 49k context cap it sits at 62–68%.

## The two "no-brainer" tests on Franzen's stack

**1. Memory fraction 0.96 → 0.98.** Each GiB is about 87k tokens, so 0.98 gives roughly +1.9 GiB, or about +165k tokens (+16%).

- **Not the same knob as lordhansolo's 0.98.** SGLang's `--mem-fraction-static` covers weights plus KV pool, and CUDA graphs (1.14 GiB) and prefill activations (chunk 8,192, max prefill 16,384) must fit in what's left. vLLM's `gpu-memory-utilization` is a profiled total that already includes those.
- **The headroom gets thin.** At 0.98 about 1.9 GiB is left, so roughly 0.8 GiB after graphs. Expect possible out-of-memory errors at prefill peaks or during CUDA graph capture.
- **Test 0.97 first.** sirikilohit ran Pennyroyal at 0.97 with larger graphs and 3.17 GiB still free.
- **At 10 streams and 128k context the pool is only 69–85% full**, so the extra tokens mostly matter if the context or the stream count goes up.

Set it with `"sglang": {"MEMFRAC": 0.98}` under `kaggle` in the config.

**2. FP8 PLE table.** The PLE table already lives in host RAM, so it **doesn't change GPU memory or the KV pool**. What it does:

- Halves host RAM, from 95.4 to 47.7 GiB. Kaggle reports about 132 GiB available, so BF16 PLE uses most of it. The freed ~48 GiB is what sirikilohit spent on a hierarchical KV + Mamba tier.
- Halves PCIe traffic for each per-token PLE lookup, which may help decode speed. Not measured yet.
- Changes numerics, so check the score and not just throughput



