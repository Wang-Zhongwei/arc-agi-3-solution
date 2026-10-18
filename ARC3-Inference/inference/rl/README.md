# RL: rejection-sampling fine-tuning (RFT)

Step 1 of the [RL plan](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/RL-design/docs/RL/decisions.md): LoRA-tune Flash-Next on our own winning trajectories, then merge the adapter into the deployed checkpoint. GRPO with verl follows if RFT helps. Pitfalls and fixes: [AGENTS.md](AGENTS.md); what was tried on each GPU run and why: [Logs.md](Logs.md).

## Decisions

| Topic | Decision | Reason |
|---|---|---|
| Stack | transformers + PEFT + our own training loop | Fewest moving parts; full control of long packed samples, images and loss masks. verl only for GRPO. |
| Selection | Drop (game, level)s with no difficulty; keep the top k per level by fewest actions, then fewest tokens | Score rewards few actions; a cap per level stops easy levels dominating. |
| Sample | One model request: x = the context the model saw, y = its reply (thinking + tool calls) | An analyzer turn has 1.47 requests on average. |
| Context | The real 50–100K window, rebuilt from transcripts at training time | Truncating would teach conclusions without the evidence for them, and break harness parity. |
| Packing | Requests between two trims share their prefix (`preserve_thinking`), so a segment is one sequence | reset-baseline v01: 209M input tokens as one sample per request, 14.4M packed by segment. |
| Loss | Only on assistant tokens of the selected (game, level); drop `uncertain` requests | Everything else is context. |
| LoRA targets | `self_attn.{q,k,v,o}_proj` and `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}`: 228 modules, 26.1M parameters at r=16. Optional: shared expert MLP. Frozen: routed experts, QSA indexer, PLE, vision | All BF16 in the deployed checkpoint, so merging costs nothing. The layers [ms-swift's recipe](https://swift.readthedocs.io/en/v4.5/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html) targets too (`linear_qkv linear_proj in_proj out_proj`). |
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
| [qsa.py](qsa.py) | Chunked QSA for training, replacing transformers' reference ([test](../../tests/test_qsa.py)) |
| [layer_patches.py](layer_patches.py) | Grouped INT4 experts and token-chunked decoder layers ([test](../../tests/test_layer_patches.py)) |
| [ple_mmap.py](ple_mmap.py) | Serve the PLE n-gram table from the checkpoint files ([test](../../tests/test_ple_mmap.py)) |
| [pod/smoke_pod.sh](pod/smoke_pod.sh) | The smoke test as one pod job |
| [data/python_tool.json](data/python_tool.json) | The harness's tool schema |

## Training-step patches

What a LoRA step on the W4A16 checkpoint needs beyond stock transformers + PEFT. Every row was hit or measured on a GPU; [Logs.md](Logs.md) has the history.

| Patch | Where | What it does | Evidence |
|---|---|---|---|
| Chunked QSA | [qsa.py](qsa.py) | Queries in chunks of 256. Each query's ~2,051 selected keys and values are gathered with `index_select`, the query heads that share a KV head are laid out as SDPA query positions, and each chunk is checkpointed. | transformers' reference OOMs at 32K. Matches it on a tiny model (selections up to ties, outputs, logits, gradients). One QSA layer at 64K: 0.57 s forward, 3.61 s forward + backward. |
| Grouped INT4 experts | [layer_patches.py](layer_patches.py) | Tokens sorted by expert; per tile of 128 experts, one AutoRound `dequant248` call to BF16 and two `torch._grouped_mm` calls. Replaces AutoRound's loop over 512 unfused experts. | Startup self-check against AutoRound's INT4 path: 0.3% max error. 64K backward 262 → 134 s. |
| Token-chunked decoder layers | [layer_patches.py](layer_patches.py) | Hyper-connections, router, experts and shared expert run in 8,192-token checkpointed chunks; the PLE block is checkpointed whole. | Exact against transformers on a tiny model. 100K OOMed before; 123K now peaks at 129 GiB. |
| Memory-mapped PLE table | [ple_mmap.py](ple_mmap.py) | The 102.5 GB n-gram table is read from the checkpoint files instead of being loaded into RAM. | Bit-identical on a tiny model. The process holds 2.9 GB after loading (was ~105 GB); a lookup pass costs 0.46 s at 64K. |
| Checkpointing on | [smoke_lora.py](smoke_lora.py) `step` | `model.train()` before every step. | transformers checkpoints layers only in training mode; in eval mode every activation was kept. |
| Activation offload | `step(offload=True)` | Each layer's checkpointed input (N × 4 × 2,560) goes to pageable host memory (`save_on_cpu(pin_memory=False)`). | Needed above ~45K tokens; ~113 GiB of host RAM at 123K. Pinned memory got the container OOM-killed. |
| Chunked cross-entropy | `chunked_cross_entropy` | lm_head + loss on the trained positions only, 2,048 at a time, each checkpointed. | Never builds [K, 248,320] logits; exact. |
| Guards | `run_specs`, `host_memory_available_gb` | On OOM: free memory, halve the token chunk and retry (twice), else skip. Skip offload steps the container's memory (cgroup v1 or v2) can't hold. | Not triggered so far. |
| Environment | [pod/smoke_pod.sh](pod/smoke_pod.sh) | `tilelang` + `CUDA_HOME` for fla's GDN backward on Hopper, `expandable_segments`, an explicit GPU device map, no `use_kernels`. | See [AGENTS.md](AGENTS.md). |

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

1. Reuse host buffers for the offload: saves ~35 s per 64K step and stops depending on the host's page-fault speed.
2. QSA key/value gradients accumulate in BF16 in both gathers (1.7% error against FP32 on one real-size chunk; 0.33% with FP32 accumulation), and the same 64K step gave LoRA gradient norms 0.248 and 0.293 with the two gathers. Accumulate in FP32 and measure the spread on a repeated step.
3. Then the training loop (optimizer, LR, epochs), the dataset build (selection thresholds, harness-config filter, held-out games), a short LoRA A/B with vs without the shared expert on held-out loss, and the merge check against vLLM (QSA parity, MTP draft acceptance).

## Existing training paths for this model

- [ms-swift](https://swift.readthedocs.io/en/v4.5/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html): Megatron LoRA SFT and GRPO recipes, 8 GPUs, BF16 base, `PLE_CPU_OFFLOAD=1`, tested up to 8K tokens on text; its QSA kernel is not documented.
- [NeMo AutoModel](https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/qwen/Qwen3.8-Flash-Next): full-parameter SFT with a FlexAttention QSA and frozen indexer; text only, 64 H100s, validated at 4K tokens.
- Open PRs for Megatron-LM ([#7553](https://github.com/NVIDIA/Megatron-LM/pull/7553)), [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl/pull/3434), [miles](https://github.com/radixark/miles/pull/2777) and [NeMo RL](https://github.com/NVIDIA-NeMo/RL/blob/qwen3-8-flash-next-support/docs/guides/models/qwen/qwen3-8-flash-next.md).

None covers 100K contexts with images on a single GPU, which is why we keep our own loop.
