# RL: rejection-sampling fine-tuning (RFT)

Step 1 of the [RL plan](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/RL-design/docs/RL/decisions.md): LoRA-tune Flash-Next on our own winning trajectories, then merge the adapter into the deployed checkpoint. GRPO with verl follows if RFT helps. Pitfalls and fixes: [AGENTS.md](AGENTS.md); what was tried on each GPU run and why: [Logs.md](Logs.md).

## Decisions

| Topic | Decision | Reason |
|---|---|---|
| Stack | transformers + PEFT + our own training loop | Fewest moving parts; full control of long packed samples, images and loss masks. verl only for GRPO. |
| Selection (2026-10-20) | Levels cleared in < 80% of attempts (an attempt = the level got ≥ 1 analyzer turn); per level, the clears within the best 35% by actions whose generated tokens (thinking, reply, tool code) are at most the level's median clear. Thresholds over all runs' clears; samples only from runs that rebuild faithfully | Score rewards few actions, and time goes with tokens: within a level, actions and generated tokens correlate (median Spearman 0.65), but 21% of the fewest-action quarter still out-think the median clear. 80% to start conservative. [rft_data.py](rft_data.py) |
| Sample | One model request: x = the context the model saw, y = its reply (thinking + tool calls) | An analyzer turn has 1.47 requests on average. |
| Context | The real 50–120K window, rebuilt from transcripts, with each run's own settings: image scales, tool description, window | Truncating would teach conclusions without the evidence for them, and break harness parity. Images are drawn again from the run's frames ([rft_images.py](rft_images.py)). |
| Packing | Requests between two trims share their prefix (`preserve_thinking`), so a segment is one sequence | reset-baseline v01: 209M input tokens as one sample per request, 14.4M packed by segment. |
| Loss | Only on assistant tokens of the selected (game, level); drop `uncertain` requests | Everything else is context. |
| LoRA targets | `self_attn.{q,k,v,o}_proj` and `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}`: 228 modules, 26.1M parameters at r=16. Frozen: routed experts, shared expert, QSA indexer, PLE, vision | All BF16 in the deployed checkpoint, so merging costs nothing. Shared-expert LoRA is out of the first round (2026-10-19): it slowed a 32K step by 12% (forward 21.0 → 38.2 s) and would need its own A/B; revisit if RFT gains little. The layers [ms-swift's recipe](https://swift.readthedocs.io/en/v4.5/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html) targets too (`linear_qkv linear_proj in_proj out_proj`). |
| Base | Intel W4A16 AutoRound checkpoint, the one we deploy | Routed experts identical to serving; 68.5 GiB on the GPU. |
| Hardware | 1×H200 (141 GB); layer checkpoints offloaded to host RAM | Weights take 68.5 GiB on the GPU; a 123K-token request peaks at 129 GiB (fits to ~133K). Its checkpoints take ~113 GiB of host RAM. |
| Merge | W′ = W + (α/r)BA on the BF16 attention/GDN tensors, swapped into a copy of the W4A16 checkpoint, config unchanged | Same kernels, so no serving slowdown. Check MTP draft acceptance after merging. Lord's mixed checkpoint keeps attention in FP8, where requantizing may round small updates away. |
| Sparse attention | Train through QSA with a frozen indexer, using our chunked implementation | Serving uses QSA; NeMo AutoModel also freezes the indexer. |

## Model: `qwen4_exp` (Qwen3.8-Flash-Next)

- 48 layers: 36 GatedDeltaNet and 12 full-attention layers (every 4th) with QSA. In a QSA layer an indexer scores 4-token blocks and each query attends to its top 512 blocks (2,048 tokens) plus the incomplete tail block ([paper](https://arxiv.org/html/2608.30320)).
- Hidden size 2,560 with 4 hyper-connection streams; 512 routed experts (top 10) plus a shared expert; a vision tower; an MTP head (unused in training).
- The W4A16 checkpoint (181 GB): PLE n-gram table 102.5 GB (one layer), routed experts INT4 62.8, MTP 5.2, GDN 4.2, hyper-connections 1.3, embeddings 1.3, lm_head 1.3, attention + indexer 1.2, vision 0.9, shared expert 0.5.

## Code

| File | Purpose |
|---|---|
| [context_rebuild.py](context_rebuild.py) | Rebuild every request's messages from a transcript |
| [validate_rebuild.py](validate_rebuild.py) | Compare rebuilds with `*_requests.jsonl` |
| [smoke_lora.py](smoke_lora.py) | Load W4A16 + LoRA, time forward/backward on real packed samples |
| [qsa.py](qsa.py) | QSA for training, replacing transformers' reference ([test](../../tests/test_qsa.py)) |
| [qsa_kernel.py](qsa_kernel.py) | Fused sparse-attention forward and backward in Triton ([kernel test](../../tests/test_qsa_kernel.py), [model test](../../tests/test_qsa_fused_model.py)) |
| [bench_layer.py](bench_layer.py) | Time one real-size GDN and QSA layer without the checkpoint ([pod/kernel_pod.sh](pod/kernel_pod.sh)) |
| [layer_patches.py](layer_patches.py) | Grouped INT4 experts and token-chunked decoder layers ([test](../../tests/test_layer_patches.py)) |
| [ple_mmap.py](ple_mmap.py) | Serve the PLE n-gram table from the checkpoint files ([test](../../tests/test_ple_mmap.py)) |
| [host_offload.py](host_offload.py) | Offload saved activations into host buffers reused across steps ([test](../../tests/test_host_offload.py)) |
| [rft_images.py](rft_images.py) | Draw a rebuilt request's images again from the run's events and frames ([validation](validate_images.py): 12,778 logged images identical) |
| [rft_data.py](rft_data.py) | Select level clears and write training samples (messages, image files, trained replies, tool schema) |
| [check_rft_data.py](check_rft_data.py) | Tokenize a dataset as training does: reply markers, loss spans, exact lengths |
| [train_rft.py](train_rft.py) | The training loop ([test](../../tests/test_train_rft.py)) |
| [pod/train_pod.sh](pod/train_pod.sh) | One-shot training run on a pod; [pod/watch_pod.sh](pod/watch_pod.sh) pulls the results and deletes it |
| [pod/smoke_pod.sh](pod/smoke_pod.sh) | The smoke test as one pod job |
| [data/python_tool.json](data/python_tool.json) | The harness's tool schema |

## Training-step patches

What a LoRA step on the W4A16 checkpoint needs beyond stock transformers + PEFT. Every row was hit or measured on a GPU; [Logs.md](Logs.md) has the history.

| Patch | Where | What it does | Evidence |
|---|---|---|---|
| Fused QSA kernel (default on CUDA) | [qsa_kernel.py](qsa_kernel.py) | Selection as block ids; forward and dQ one Triton program per query and KV head (online softmax, LSE); dK/dV per tile of 16 neighbouring queries over the union of their blocks, FP32 atomics. A checkpoint's recompute reuses the forward's selection. | Matches FP32 dense attention at the checkpoint's head layout and transformers' QSA through a tiny model. One QSA layer at 64K on an A100: 12.6 → 2.3 s. |
| Chunked QSA (CPU, `fused=False`) | [qsa.py](qsa.py) | Queries in chunks of 256. Each query's ~2,051 selected keys and values are gathered with `index_select`, the query heads that share a KV head are laid out as SDPA query positions, and each chunk is checkpointed. | transformers' reference OOMs at 32K. Matches it on a tiny model (selections up to ties, outputs, logits, gradients). One QSA layer at 64K: 0.57 s forward, 3.61 s forward + backward. |
| Grouped INT4 experts | [layer_patches.py](layer_patches.py) | Tokens sorted by expert; one layer's experts dequantized to BF16 once per pass (tiles of 128, `dequant248`) and cached (~5 GB), then one `torch._grouped_mm` per projection over all 512 experts with offsets on the GPU (no host sync). Replaces AutoRound's loop over 512 unfused experts. Each token's k outputs are summed in FP32 in a fixed order (deterministic); the sort and unsort gathers have gather backwards (no `index_put`). | Startup self-check against AutoRound's INT4 path: 0.3% max error. 64K backward 262 → 134 s. |
| Token-chunked decoder layers | [layer_patches.py](layer_patches.py) | Hyper-connections, router, experts and shared expert run in 8,192-token checkpointed chunks (`split`, so backward concatenates once); the PLE block is checkpointed whole. | Exact against transformers on a tiny model. 100K OOMed before; 123K now peaks at 129 GiB. |
| Triton grouped GEMM | [grouped_gemm.py](grouped_gemm.py) | On GPUs without `torch._grouped_mm` (A100, SM120): all 512 experts in one launch per projection, dX with the same kernel (frozen experts). Hopper keeps torch's. | Per-expert FP32 matmuls ([test](../../tests/test_grouped_gemm.py)); AutoRound self-check 0.53%. RTX PRO 6000 32K: 1,081 → 1,475 tok/s. |
| Fused LoRA update (`fast_lora`) | [layer_patches.py](layer_patches.py) | peft's FP32-adapter update as one compiled function (no FP32 intermediates the size of the output). | Bit-identical to peft on CPU; +3.4% at 32K. |
| PLE lookup once per step | [layer_patches.py](layer_patches.py) | The host-side n-gram gather is reused by the PLE checkpoint's recompute. | Bit-identical; one lookup instead of two. |
| Compiled hyper-connections and MoE combine (`compile_hc`, `compile_moe`) | [layer_patches.py](layer_patches.py) | The gated residuals and injections as pure functions under `torch.compile`, shared by all layers; chunks padded to power-of-two buckets so every chunk's recompute runs the same static graph as its forward. | As accurate as BF16 eager against FP32 ([test](../../tests/test_compile_hc.py)). GDN layer at 64K on an A100: 1.65 → 1.43 s. |
| Channels-last GDN convolution (`fast_conv`) | [layer_patches.py](layer_patches.py) | Four shifted multiply-adds in the projection's layout, compiled, instead of `F.conv1d` on a transposed copy. | Matches the `F.conv1d` fallback on CPU. |
| Memory-mapped PLE table | [ple_mmap.py](ple_mmap.py) | The 102.5 GB n-gram table is read from the checkpoint files instead of being loaded into RAM. | Bit-identical on a tiny model. The process holds 2.9 GB after loading (was ~105 GB); a lookup pass costs 0.46 s at 64K. |
| Checkpointing on | [smoke_lora.py](smoke_lora.py) `step` | `model.train()` before every step. | transformers checkpoints layers only in training mode; in eval mode every activation was kept. |
| Activation offload | `step(offload=True)`, [host_offload.py](host_offload.py) | Each layer's checkpointed input (N × 4 × 2,560) goes to pageable host memory, into 16 GiB segments kept on the model and reused by every later step (`reuse_host_buffers: false` in a job falls back to `save_on_cpu(pin_memory=False)`). | Needed above ~45K tokens; ~113 GiB of host RAM at 123K. Pinned memory got the container OOM-killed. Fresh pageable memory filled at 1.7 GiB/s on one host, a reused buffer at 16.5 GiB/s. Gradients exact on a tiny model; not yet timed on a GPU. |
| Chunked cross-entropy | `chunked_cross_entropy` | lm_head + loss on the trained positions only, 2,048 at a time, each checkpointed. | Never builds [K, 248,320] logits; exact. |
| Guards | `run_specs`, `host_memory_available_gb` | On OOM: free memory, halve the token chunk and retry (twice), else skip. Skip offload steps the container's memory (cgroup v1 or v2) can't hold. | Not triggered so far. |
| Environment | [pod/smoke_pod.sh](pod/smoke_pod.sh) | `tilelang` + `CUDA_HOME` for fla's GDN backward on Hopper, `expandable_segments`, an explicit GPU device map, no `use_kernels`. | See [AGENTS.md](AGENTS.md). |

## Dataset v1 (2026-10-20)

Built by [rft_data.py](rft_data.py) with the defaults (pass rate < 80%, best 35% by actions, generated tokens ≤ the level's median); `runs/rft-data/rft_v1.tgz` (31 MB; 128 MB unpacked: JSON lines plus 9,114 deduplicated PNGs).

| | Trajectories | Samples | Tokens (exact) | Trained | Prefix | Median / max length |
|---|---|---|---|---|---|---|
| Train | 213 (44 levels, 17 games, 47 runs) | 354 | 33.6M | 7.30M | 64% | 101K / 133K (2 over 131,072, skipped) |
| Validation (the next-best clear of 12 levels) | 12 | 34 | 3.7M | 0.93M | 59% | 115K / 128K |

- 85 runs scanned: 48 usable. Left out: no `src/` snapshot (17), another team's fork (1), no settings file (5), trimmed-history note (2), no frames (2), game priors (1), view tool (1).
- Requests dropped: 155 whose context holds a level-complete or animation image of bp35/lf52 (saved frames overwritten), 21 over the length cap, 9 windows with only `uncertain` requests.
- [check_rft_data.py](check_rft_data.py): every sample's reply markers and loss spans as intended; exact lengths 2.8% above the estimate (median).
- Expected training time on one H200 from the 10-20 measurements (16.1 s per 64.9K step at an 80% prefix): ~2.9 h per epoch.

## Latest results (RTX PRO 6000 96GB, 2026-10-19 evening)

Real model, second step, all of `qsa_fused`, `compile_hc`, `compile_moe`, `fast_conv`, `cache_experts`, `fast_lora`:

| Sample | Morning path | Now | Speed-up |
|---|---|---|---|
| 16K (14,173 tokens) | 572 tok/s | 1,314 tok/s | 2.3× |
| 32K (33,811 tokens) | 569 tok/s | 1,470 tok/s | 2.6× |

Checked layer by layer against transformers' layers, QSA attention and AutoRound's experts on real inputs (`layer_check` job): LoRA gradient cosine 0.9996–0.9997, input gradient 0.9999, as close as the morning's path. Whole-model gradient comparisons cannot validate this model (see [AGENTS.md](AGENTS.md#kernels-compile-and-benchmarks-10-19)). Not yet timed on an H200.

## Results (smoke test 4, 2026-10-19)

Secure 1×H200 in US-NC-1, 49 min, ~$4.30; logs and results in `runs/runpod/jonathanwang2022/rft-lora-smoke/93so3gmik3uttk/`. Same 64K request as smoke test 3 (64,893 tokens, 8,141 trained).

| Configuration | Forward | Backward | Tokens/s | Peak GPU |
|---|---|---|---|---|
| `save_on_cpu` (smoke test 3's path) | 59.0 s | 78.6 s | 472 | 99.3 GiB |
| Reused pageable buffers | 33.6 s | 79.9 s | 571 | 100.7 GiB |
| `recompute: "parts"`, synchronous copies | 29.5 s | 90.7 s | 540 | 96.2 GiB |
| **parts, pinned async copies, 35.6 GiB kept on the GPU** | **16.8 s** | **66.0 s** | **784** | 121.1 GiB |
| Same, 32K request (almost everything kept on the GPU) | 11.5 s | 30.7 s | 800 | 126.5 GiB |

- The forward is now GPU compute only; backward is ~80% of the step, mostly QSA.
- Two runs of the same step had LoRA gradients with cosine 0.87 until the MoE combine was made deterministic (0.99989 after). See [AGENTS.md](AGENTS.md#determinism).
- parts OOMs at 123K; such requests fall back to layer recompute. Plan and next steps: [Throughput.md](Throughput.md).

## Latest results (smoke test 3, 2026-10-18)

Secure 1×H200 in US-NC-1, 43 min, ~$3.80; logs and results in `runs/runpod/jonathanwang2022/rft-lora-smoke/i1snr1yuwi9qf9/`. LoRA r = 16 on attention + GDN (228 modules, 26.1M parameters). The jobs before the last row still gathered QSA keys with advanced indexing.

| Sample (all offloaded) | Tokens | Images | Trained | Forward | Backward | Tokens/s | Peak GPU | Loss |
|---|---|---|---|---|---|---|---|---|
| 32K | 33,811 | 15 | 5,664 | 38.6 s | 96.3 s | 251 | 87.3 GiB | 0.47 |
| 64K | 64,893 | 32 | 8,141 | 64.7 s | 133.5 s | 328 | 99.3 GiB | 0.28 |
| 106K, cd82 (first step: includes JIT) | 106,686 | 35 | 21,495 | 135.7 s | 252.7 s | 275 | 119.6 GiB | 0.30 |
| 123K, ft09: the longest request of the 128K-budget run | 122,943 | 29 | 11,700 | 123.8 s | 229.4 s | 348 | 129.0 GiB | 0.21 |
| **64K, QSA gather with `index_select`** | 64,893 | 32 | 8,141 | 58.4 s | 78.5 s | **474** | 99.3 GiB | 0.28 |

- Capacity: a quadratic fit through the first three peaks predicted the 123K peak to within 0.2 GiB, and reaches the ~139.5 GiB PyTorch can use at ~133K tokens. Every request of every run fits on one H200.
- Where the 64K step's 137 s go: in the 58 s forward, ~16 s is layer compute (36 GDN × 0.24 s, 12 QSA × 0.57 s, PLE 0.46 s) and ~35 s copies checkpoints into freshly allocated host memory at 1.7 GiB/s (first-touch page faults on this host; 16.5 GiB/s into a reused buffer). In the 78.5 s backward, QSA takes ~43 s (12 × 3.61 s), GDN layers ~25 s and host → GPU copies ~6 s.
- Training time: 160 trajectories (levels cleared < 90% of the time, the 2 fewest-action clears of each), 21.6M tokens to process and 4.5M trained: ~13 h on one H200 at 474 tokens/s.
- Inputs: the context rebuild was checked on the 10 runs with request logs (17,979 requests). On runs without non-main features, 98.3–100% of windows are exact among non-`uncertain` requests, and message content is ≥ 99.7% exact ([summary](../../../docs/RL/data/context_rebuild_validation.json)).

## Open issues and next steps

The plan for a 10× faster training step, step by step, is in [Throughput.md](Throughput.md).

1. Reuse host buffers for the offload: done in [host_offload.py](host_offload.py), exact on a tiny model and on MPS. Should save ~35 s per 64K step and stop depending on the host's page-fault speed; time it on the next pod (64K with `reuse_host_buffers` true and false).
2. QSA key/value gradients accumulate in BF16 in both gathers (1.7% error against FP32 on one real-size chunk; 0.33% with FP32 accumulation), and the same 64K step gave LoRA gradient norms 0.248 and 0.293 with the two gathers. Accumulate in FP32 and measure the spread on a repeated step.
3. Training loop and dataset: done (10-20, above). Next: the first training run, then the merge check against vLLM (QSA parity, MTP draft acceptance) and a Kaggle A/B of the adapter.

## Existing training paths for this model

- [ms-swift](https://swift.readthedocs.io/en/v4.5/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html): Megatron LoRA SFT and GRPO recipes, 8 GPUs, BF16 base, `PLE_CPU_OFFLOAD=1`, tested up to 8K tokens on text; its QSA kernel is not documented.
- [NeMo AutoModel](https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/qwen/Qwen3.8-Flash-Next): full-parameter SFT with a FlexAttention QSA and frozen indexer; text only, 64 H100s, validated at 4K tokens.
- Open PRs for Megatron-LM ([#7553](https://github.com/NVIDIA/Megatron-LM/pull/7553)), [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl/pull/3434), [miles](https://github.com/radixark/miles/pull/2777) and [NeMo RL](https://github.com/NVIDIA-NeMo/RL/blob/qwen3-8-flash-next-support/docs/guides/models/qwen/qwen3-8-flash-next.md).

None covers 100K contexts with images on a single GPU, which is why we keep our own loop.
