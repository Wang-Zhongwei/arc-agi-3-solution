# GRPO pipeline

Our own asynchronous loop around the RFT trainer, as proposed in [decisions.md](decisions.md): the evaluation harness plays the games against vLLM, the trainer learns from the exact tokens vLLM sampled, and the merged weights go back into the running server after every step. Built 2026-10-26; not run yet.

## Shape

One pod with 4 H200:

| GPUs | Process | Role |
|---|---|---|
| 0, 1 | `vllm serve`, TP=2, 64 sequences | Rollouts, with the H200 benchmark's flags plus processed logprobs and the weight-sync extension |
| — (CPU) | the harness, inline | Plays one wave: 8 games × 8 rollouts = 64 episodes |
| — (CPU) | `inference.grpo.loop` | Runs waves, builds samples, computes rewards, hands work to the trainers, syncs weights, saves, logs |
| 2, 3 | `inference.grpo.train_grpo` × 2 | One full model each (train_rft's load path), LoRA data parallel over NCCL |

## One step (one-step off-policy)

Wave 0 is played by the base model. Then, for every k ≥ 1, two things run at the same time:

- the server plays wave k with the weights θₖ₋₁;
- the trainers learn from wave k − 1 (played by θₖ₋₂) and produce θₖ.

When both are done, θₖ goes into the server (pause, load, reset the prefix cache, resume) and wave k + 1 starts. Every wave is trained exactly one step after the policy that played it; the importance ratio against the logprobs vLLM recorded corrects for that. A wave takes ~19 min and training ~11–16 min, so the rollouts set the pace (decisions.md, Hardware).

## Episodes: the harness in GRPO mode

The same harness, prompts, tools, observation rendering and sampling as evaluation. GRPO mode (`ARC3_RL_RECORD=1`) changes only what is invisible to the model plus where an episode ends:

- **Ends** when L2 is cleared, when the context reaches the game's budget min(1,750 × human actions of L1 + L2, 130K), or after a 60-minute safety cap.
- **Never trims.** History is append-only (no turn cap, no drains); turn yield is off. Every request then extends the previous one.
- **Budget per request:** `max_tokens` is lowered to what the budget leaves (anchored estimate of the prompt, 256-token margin), so no sequence passes the budget. The thinking budget stays on (it is skipped when `max_tokens` is passed as an argument, so the cap is applied to the payload instead).
- **Recorded per request** (`<run>/rl/<game>.jsonl`): the prompt length, the sampled token ids and their processed logprobs (`return_token_ids`, `logprobs`), the finish reason, the longest common prefix with the previous prompt + reply, and the levels completed and actions taken when it was sent.
- **Recorded at the end:** the last prompt's token ids, the images of the last request in order, TAAF's per-level action counts (`actions_per_level`, the official score's counter), the human counts, the levels cleared and why the episode ended.

## Training sequences: the tokens vLLM saw

- **Input ids:** vLLM's last `prompt_token_ids` plus the last reply's ids. Nothing is re-rendered or re-tokenized.
- **Images:** the last request's images, through the same processor (`Qwen2VLImageProcessor`); each image's placeholder run must match its grid, or the episode is dropped. `mm_token_type_ids` come from the ids (`create_mm_token_type_ids`).
- **Trained tokens:** every reply whose tokens sit unchanged at their place in the final sequence. The harness rebuilds earlier replies from parsed fields (`ARC3_NORMALIZE_HISTORY=0`; trimmed reasoning, content, tool calls re-rendered as XML), so a reply can differ from what was sampled at a whitespace or tokenization boundary. The longest-common-prefix chain finds such replies; their tokens stay as context but are not trained. The share is logged per wave.
- **Also not trained:** the `</think>` tokens vLLM forces when a reply reaches the 10,240-token thinking budget, and tokens outside the trainer's top-k / top-p set.
- **Behavior logprobs:** vLLM's processed logprobs (`--logprobs-mode processed_logprobs`): after temperature 0.7, top-k 20 and top-p 0.95, the distribution that was actually sampled.

## Reward, advantages, selection

- **Reward** (decided): R = ⅓ · Σ over levels cleared within the budget of wₗ · min(1, (hₗ / aₗ)²) · (1 + 0.2 · clip((1 − ρₗ) / 0.5, −1, 1)), w = 1, 2.
  - aₗ: TAAF's `actions_per_level` (RESETs included, as in the official score).
  - ρₗ: thinking tokens per action on the level ÷ τ̄, the base model's median for the game ([thinking_per_action.json](data/thinking_per_action.json)). Thinking tokens on a level are the reply tokens before `</think>` in requests sent while the game was on that level.
- **Advantage:** R minus the mean of its group of 8 (no division by the group's spread).
- **Skipped:** groups whose 8 rewards are equal (no gradient).
- **Selection:** 6 of each group's 8 (max-variance subset: the top i and bottom 6 − i for the i with the largest spread, as PODS does). 4 is the other option in decisions.md.
- **Balance:** trajectories go to the two trainers longest first, each to the one with fewer tokens.

## Loss

Per trained token t, with the processed logprob log πθ(t) (the trainer applies the same temperature, top-k and top-p, and renormalizes over the kept tokens):

- r = exp(log πθ(t) − log π_vLLM(t))
- objective = min(r · A, clip(r, 1 − 0.2, 1 + 0.28) · A), averaged over every trained token of the step (both trainers)
- tokens with r outside [0.5, 2] are dropped (train/rollout mismatch guard, as Chen's DDIS mask)
- no KL term, no entropy bonus, no old-policy forward pass
- the whole sequence runs with gradients: unlike RFT, no prefix stop-grad. The context before the first reply is ~5–8K tokens; its keys and values depend on the adapted projections, so stopping its gradient would bias the policy gradient.

One AdamW step per wave over all selected trajectories. The trajectories still run one at a time (one forward + backward each, up to 130K tokens); their LoRA gradients add up, and the step comes after the last one. GPU memory therefore depends on one trajectory's length, not on how many a step trains.

## Weight sync

- After the step, trainer rank 0 merges W = W_base + (α / r) · B · A for every adapted tensor (attention q/k/v/o, GDN in_proj_qkv/z/a/b and out_proj) in FP32 and rounds to BF16 **stochastically** into `sync/step_<k>.safetensors` (~5.4 GB, checkpoint names).
  - Why stochastic: a step moves a weight by ~1e-6 while BF16's spacing at a typical 0.02 weight is ~8e-5. Rounding to nearest would keep the base value for most elements for the first ~40 steps, and the rollouts would not follow the trainer. Stochastic rounding is unbiased from the first step.
- The loop calls `/pause`, then `/collective_rpc` `grpo_load_weights` (our worker extension: `model.load_weights` with only these tensors, in place, so CUDA graphs stay valid), `/reset_prefix_cache` and `/resume`.
- Partial updates avoid vLLM's layerwise reload: that path processes a half-loaded fused layer with uninitialized memory in the missing part. The adapted layers are plain BF16 linear layers with no post-processing.

## Settings

| Setting | Value | Why |
|---|---|---|
| Wave | 8 games × 8 rollouts, 64 at once | decisions.md: the measured best for TP=2 |
| Episode end | L2 cleared, the budget, or 60 min | 60 min is 3× the ~20 min a full-budget rollout takes at 64 concurrent; only against hangs |
| Request cap | min(14,336, budget − estimate − 256) | the estimate is anchored on the last real count (p99 +1.3%); 256 covers the rest |
| Sampling | T 0.7, top-p 0.95, top-k 20, thinking 10,240 | evaluation's settings |
| Selection | 6 of 8 per group | the most the trainers finish within a wave (~16 vs ~19 min) |
| Advantage | R − group mean | dividing by a near-tied group's spread turns its noise into full-size advantages |
| Clip | 0.2 / 0.28 | PPO's 0.2; the higher upper bound (DAPO) keeps rare good tokens from being clipped early |
| Ratio mask | [0.5, 2] | beyond 2× the rollout and trainer disagree, not learn |
| LoRA | r 16, α 32, attention + GDN | train_rft's tested setup |
| Optimizer | AdamW, lr 1e-5 constant, β (0.9, 0.999), no decay, clip 1.0 | ~10× a full-model RL rate (LoRA Without Regret); conservative per the 10-19 rule |
| Steps per wave | 1 (open: 2–4) | Every wave is exactly one update stale. The cost: Adam moves each weight by about the learning rate per update, whatever the batch size, so 30 waves are 30 updates. LoRA Without Regret's SFT runs lost more to full fine-tuning at larger batches, which argues for minibatches, not against them (the 10-26 version of this row said the opposite). |
| Sync | every step, stochastic BF16 | see Weight sync |
| Saves | adapter, optimizer, loop state every step | a resume loses at most one wave |

## Checks

**On CPU (passed 2026-10-26).** `tests/test_grpo_*.py`, 35 tests:
- reward, thinking factor, max-variance selection (against every subset), advantages, trainer balance;
- the recorder's prefix chain, budget, episode line and images; the episode builder's trained spans, forced `</think>` mask and image check;
- the processed logprob against vLLM's top-k/top-p algorithm on tokens sampled from it (BF16 ties included: a third of random rows tie at the 20th value; every sampled token is kept, values within 1e-5);
- the objective's gradient against a direct formula;
- a GRPO step through train_rft's machinery (chunks, recompute, checkpointed lm_head) against the objective computed directly on the tiny `qwen4_exp` (LoRA gradients within 1e-4);
- two gloo ranks' summed gradients against one rank;
- the merged export's checkpoint names and the stochastic rounding's mean;
- the probe with the adapter on and off;
- image tokens against the processor's grid (768 × 768 → 576).

The harness's GRPO mode and the loop also ran end to end on a laptop against [`fake_vllm.py`](../../ARC3-Inference/inference/grpo/fake_vllm.py), which renders prompts with the real chat template and tokenizer:
- 4 episodes, each ending on its budget;
- the prefix chain exact except after forced `</think>`s;
- 3 waves and 2 fake steps through the loop.

**On the pod, as gates in the loop.** A failed gate aborts the run before more is spent:

| Gate | When | Threshold |
|---|---|---|
| Replies trained whole (token-exact sequences) | after wave 0 | ≥ 50% |
| Episodes with errors | after wave 0 | ≤ 25% |
| Trainer vs vLLM on wave 0's tokens at identical weights: mean \|log ratio\| | step 1 | ≤ 0.1 |
| Trained tokens outside the trainer's kept set / ratio band | step 1 | ≤ 5% each |
| The first sync carries the update: correlation of vLLM's and the trainer's logprob changes on a 2K-token probe | after sync 1 | ≥ 0.5 |

Before the trainers start, the pod runs the trainer's CUDA tests (and the GRPO tests) on GPU 2 and switches off any feature whose test fails, as train_pod.sh does.

## Running it

- Pod: 4 × H200 SXM, image `vllm/vllm-openai@sha256:d428631bb917a33d22308265d1b8be91522f862bf9fd133bee62dde3313664ae`, a ≥ 400 GB volume at `/workspace`: checkpoint 169 GB, trainer venv, waves at ~1 GB each, the last two sync files.
- Upload `repo.tgz` (ARC3-Inference, tufa-arc-agi-framework, docs/RL/data) and `env.tgz` (environment_files), then run [`grpo_pod.sh`](../../ARC3-Inference/inference/grpo/pod/grpo_pod.sh). It works in phases:
  - download the checkpoint;
  - set up the harness venv and the trainer venv (torch 2.8 stack, separate from vLLM's torch 2.14);
  - start the server;
  - run the trainers' CUDA tests;
  - start both trainers;
  - start the loop.
- Settings: `STEPS` (30), `LR`, `KEEP`, `OPTIMIZER` (adamw or muon), `MAX_HOURS` (12), `POD_END` (stop), `WANDB_PROJECT`.
- Host memory: the trainers get the RAM left after the server is up, minus their shared PLE page cache (103 GB) and 60 GB, split in two, up to 200 GB each. 130K-token episodes in "layer" recompute need ~127 GB, so the script aborts below 120 GB per trainer.

## Open

- 4 or 6 of 8 per group (6 by default).
- Updates per wave: 1, or 2–4 minibatches that each mix all games. Weights still sync once per wave. With 4 minibatches the trained data is 4–7 updates behind the policy that played it instead of 1, which the clipping and ratio mask exist for. The runs make 4× as many updates for the same rollouts, and memory is unchanged.
- Held-out checks (decisions.md, When to stop): not in the loop yet. An evaluation wave every few steps costs ~19 min per check.
- Muon: built as an option (`--optimizer muon`), not used. Evidence on 2026-10-26: [Ruan et al., 2607.16169](https://arxiv.org/abs/2607.16169) on ALFWorld (0.5B, one seed) traces Muon's gain to update size, and no study pairs it with LoRA in RL. A candidate A/B if round 1 learns slowly.
