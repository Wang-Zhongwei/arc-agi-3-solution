# GRPO training throughput

How many H200s must train to keep up with one 2×H200 rollout pair, and what else GRPO needs. Measured 2026-10-26; the open choices are at the end.

## Rollout side

- One 2×H200 pair (TP=2, 64 concurrent rollouts) generates about 3.8–4.2K tokens/s (projection from the 40-rollout run in [h200-rollout-serving.md](h200-rollout-serving.md)).
- The trainer processes every token of a trajectory: under the 130K budget, a training-game trajectory averages 101K tokens, 51K of them generated. That is 1.96 tokens to train per generated token, so the pair produces 7.4–8.2K tokens/s of trajectories.

## Trainer side (H200, measured)

| Trajectory | Step | Tokens/s |
|---|---:|---:|
| 64K (64,893 tokens) | 20.3 s | 3,196 |
| 110K (106,078 tokens, the median trajectory) | 36.9 s | 2,878 |

- Every reply is trained, so the backward pass covers the whole sequence. RFT's 4,031 tokens/s came from a sample whose first 80% got a forward pass only.
- 110K needs layer recompute (parts mode's checkpoints do not fit the host arena) and costs only ~12% more per token than 64K.
- 2026-10-26: the QSA dK/dV backward now runs per query; the union kernel's atomics cost no more than plain stores, but it did 3.3× the useful work. The 64K step went from 22.6 s to 20.3 s (+11%), LoRA-gradient cosine against transformers' layer 0.99984. Details: [Throughput.md](../../ARC3-Inference/inference/rft/Throughput.md).

## Balance

Simulated on the 1,075 historical attempts (21 training games, 200 groups of 8 per game, reward 0 / ⅓ / 1 for levels cleared within the budget):

| Trained per group of 8 | Tokens to train | Training H200s per rollout pair |
|---|---:|---:|
| All 8 | 100% | 2.6–2.9 |
| Skip groups whose 8 rewards are equal (no advantage, no gradient) | 91% | 2.4–2.7 |
| **4 of 8 with the widest reward spread (top 2 + bottom 2)** | **45%** | **1.2–1.3** |

- The last row is PODS ([Xu et al., 2025, "Not All Rollouts are Useful"](https://arxiv.org/abs/2504.13818)): when rollouts are cheap and training is the bottleneck, train on the most informative subset of each group. With per-level action-efficiency rewards the top 2 are the most efficient clears, usually short, so the real share is likely below 45%.
- Equal-reward groups become rarer once rewards include action efficiency, so that row saves less than 9% in practice. It changes nothing in the gradient, so it is free.

## Options considered

| Option | Effect | Verdict |
|---|---|---|
| Train on 4 of 8 (PODS) | −55% training tokens | Recommended; an RL algorithm change, so the user decides |
| Skip equal-reward groups | up to −9%, same gradient | Do it |
| End a rollout when a level overruns 1.5× its share of the budget | −7% tokens; 13.5% of rollouts lose credit | No |
| Remaining kernel work (FP32 LoRA and hyper-connection matmuls, copies, gathers) | each ≤ 2–3% | Not on the critical path; the FP32 ones change numerics |
| Pipeline parallel over the two trainers | halves activation memory, so parts mode at 110K | Not needed: layer mode costs only ~12% |
| LoRA on the upper layers only | backward stops early, ~1.5× faster | Changes model capacity; only with an A/B |
| Async pipeline (Yi-Chia Chen's Pipeline RL) | trainers never wait for the slowest rollout | Plan on it |

## Recommendation

- 2 rollout H200 (TP=2, 64 concurrent) + 2 training H200 (LoRA data parallel), PODS 4 of 8, equal-reward groups skipped, asynchronous with a staleness limit. Training capacity 5.6K tokens/s against 3.3–3.7K needed.
- Without PODS: 3 training H200.

## Also needed for GRPO

- **Token in, token out.** Train on the exact token ids vLLM saw (`return_token_ids`; upstream vLLM also has a token-in-token-out server), not on a re-rendered transcript.
- **Off-policy correction.** Rollouts sample from the W4A16 merged model and may be a step stale; training runs BF16 attention with dequantized experts. Use the rollout's `processed_logprobs` with a DDIS/TIS mask, as Chen does; no old-policy or KL forward pass.
- **Routing replay.** Upstream vLLM returns each token's routed experts; replaying them in training removes the router as a source of train/rollout mismatch. Optional at first.
- **Prefix-cache salt per weight version**, so a new rollout never reuses KV computed with old weights.
- **Weight push.** Only the merged attention/GDN tensors change (~5.4 GB BF16).
- **CPU on the rollout pod.** 64 concurrent harness instances run model-written Python and a game each; check vCPU load in the first run.
- **Framework.** [decisions.md](decisions.md) picked verl (colocated, CUDA IPC weight sync, FSDP2) on the assumption that rollout dominates. Training dominates, and the trainer's optimizations (QSA kernel, checkpoint offload, memory-mapped PLE, token-chunked layers) live in our own loop ([train_rft.py](../../ARC3-Inference/inference/rft/train_rft.py)). An own asynchronous loop around it, like Chen's, may be simpler than porting them into verl's FSDP worker. Open.
