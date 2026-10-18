# RL: rejection-sampling fine-tuning (RFT)

Step 1 of the [RL plan](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/RL-design/docs/RL/decisions.md): LoRA-tune Flash-Next on our own winning trajectories, then merge the adapter into the deployed checkpoint. GRPO with verl follows if RFT helps. Pitfalls and fixes: [AGENTS.md](AGENTS.md).

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
| Hardware | 1×H200 (141 GB); PLE table and layer checkpoints in host RAM | Weights take 68.5 GiB; checkpoints are ~98 GB at 100K. |
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
| [pod/smoke_pod.sh](pod/smoke_pod.sh) | The smoke test as one pod job |
| [data/python_tool.json](data/python_tool.json) | The harness's tool schema |

## Status (2026-10-18)

- Rebuild checked on the 10 runs with request logs (17,979 requests): on runs without non-main features, 98.3–100% of windows exact among non-`uncertain` requests; message content ≥ 99.7% exact ([summary](../../../docs/RL/data/context_rebuild_validation.json)).
- Samples: 32K/64K/100K targets give 33,811 / 64,893 / 99,072 tokens with 15 / 32 / 55 images.
- Smoke test on 1×H200, first attempt: the model loads (68.5 GiB, PLE on CPU) and LoRA attaches; the first 32K step OOMs inside transformers' QSA reference.
- Since then: chunked QSA, activation offload and chunked cross-entropy, all checked against the reference on a tiny model on CPU.
- Smoke test on 1×H200 SXM (2026-10-18, two pods, ~2.3 h, ~$12; logs and results in `runs/runpod/jonathanwang2022/rft-lora-smoke/`):

  | Sample (o = checkpoints offloaded to host) | Tokens | Images | Trained | Forward | Backward | Tokens/s | Peak GPU |
  |---|---|---|---|---|---|---|---|
  | 32K | 33,811 | 15 | 5,664 | 19.7 s | 149.8 s | 200 | 123.4 GiB |
  | 32K o | 33,811 | 15 | 5,664 | 21.0 s | 150.8 s | 197 | 98.4 GiB |
  | 64K o | 64,893 | 32 | 8,141 | 63.8 s | 261.9 s | 199 | 127.6 GiB |
  | 100K o | 99,072 | 55 | 9,942 | OOM in the MoE forward | | | > 139.8 GiB |
  | 32K o, + shared expert LoRA (33.5M params) | 33,811 | 15 | 5,664 | 38.2 s | 156.5 s | 174 | 99.3 GiB |

  Loss on the model's own tokens is 0.47 (32K) and 0.28 (64K); with fresh LoRA it is the base model's loss. Peak memory grows ~0.9 MB per token above the 68.5 GiB of weights, mostly the AutoRound MoE dispatch (all token × top-10 rows at once: 4.7 GiB at 100K).
- GPU sampling during the steps: backward keeps kernels running 80–90% of the time but at ~350 W of 700 W and ~50% memory-bandwidth use; forward is CPU-bound (15–40% busy). Many small per-expert kernels, not a memory-bandwidth limit.
- Data: 79 local runs, 1,903 games, 153 distinct levels cleared. Dropping levels cleared ≥ 90% of the time and keeping the 2 fewest-action clears per level gives 160 trajectories, 21.6M tokens to process (4.5M trained): ~30 h on one H200 at 200 tokens/s.
- Next, moderate fixes only: chunk the MoE over tokens (fits 100K) and replace the per-expert loop with a grouped GEMM (speed); then a short LoRA A/B (with vs without shared expert) on held-out loss.

## Existing training paths for this model

- [ms-swift](https://swift.readthedocs.io/en/v4.5/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html): Megatron LoRA SFT and GRPO recipes, 8 GPUs, BF16 base, `PLE_CPU_OFFLOAD=1`, tested up to 8K tokens on text; its QSA kernel is not documented.
- [NeMo AutoModel](https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/qwen/Qwen3.8-Flash-Next): full-parameter SFT with a FlexAttention QSA and frozen indexer; text only, 64 H100s, validated at 4K tokens.
- Open PRs for Megatron-LM ([#7553](https://github.com/NVIDIA/Megatron-LM/pull/7553)), [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl/pull/3434), [miles](https://github.com/radixark/miles/pull/2777) and [NeMo RL](https://github.com/NVIDIA-NeMo/RL/blob/qwen3-8-flash-next-support/docs/guides/models/qwen/qwen3-8-flash-next.md).

None covers 100K contexts with images on a single GPU, which is why we keep our own loop.
