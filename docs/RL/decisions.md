Example from Yi-Chia Chen: [Agent-Factory-3 blog](https://huggingface.co/blog/ycchen/agent-factory-3-en), [code](https://github.com/ycchen-tw/agent-factory-3)
核心内容：在 4×H100 上对 GPT-OSS-120B 做多轮工具调用 RL

训练方法
- 只训练 attention 层的 LoRA（q/k/v/o）。MoE 专家和 router 冻结，专家权重保持 MXFP4。
- 反向传播时逐层临时把专家反量化成 bf16，峰值显存只多一层。
- FSDP2 改成能装 MXFP4 的版本。有个坑：打包后的字节可能刚好是 bf16 的 NaN 位模式，拷贝时被规范化就会损坏权重。
- 奖励是二元的（赢了记 1），用组内相对优势。全赢或全输的组没有信号，直接过滤。

系统部分
- 异步 pipeline RL：SGLang 负责生成，FSDP2 负责训练，然后把权重同步回 SGLang。不用 Ray，单机 multiprocessing 就够。
- DDIS 重要性采样（来自 GLM-5）：用生成时记下的 logprob 算比率，比率超出阈值的 token 被 mask 掉，过旧的样本丢弃。这样不需要再跑一遍 old-policy 前向。
- MoE routing replay：生成时记录每个 token 选了哪些专家，训练时沿用同一组专家。
- 前缀缓存按 rollout 或按组加 salt，避免新 rollout 复用旧权重算出的 KV，悄悄引入 off-policy 偏差。
- 生成阶段用 DFlash 投机解码（block-diffusion draft 模型），配合 SWA 和 attention sink，并在长 agent 轨迹上微调过。生成和整体训练都快了约 50%，胜率没有下降。
- Token-in-token-out：直接拿采样出来的 token 算 loss，不经过"解析 → 再序列化 → 重新分词"。
- 每个 episode 起一个独立的 MCP 环境，并设置轮数、token、上下文和工具耗时预算。作者特别强调超时、上下文溢出和解析失败都会污染奖励。
- 最近的 rollout 渲染成 HTML 传到 W&B。作者说看轨迹比看奖励曲线更快发现问题。

## Our decisions

- **Framework: [verl](https://github.com/volcengine/verl) (2026-10-12; reopened 2026-10-26).** Widely used. Supports LoRA, vLLM or SGLang rollout, and multi-turn tool use (`AgentLoop`). [slime](https://github.com/THUDM/slime) has no LoRA, so it would mean full-parameter RL on a multi-node cluster.
  - Reopened: verl was picked for rollout and training on the same GPUs, with rollout as the bottleneck. On separate rollout and training H200s (see Hardware), training is the bottleneck.
  - The trainer's optimizations (QSA kernel, checkpoint offload, memory-mapped PLE, token-chunked layers) live in our [train_rft.py](../../ARC3-Inference/inference/rft/train_rft.py).
  - The harness already plays the games against vLLM's OpenAI server.
  - Our own asynchronous loop around train_rft, like Chen's (multiprocessing, no Ray): built 2026-10-26 when the user asked to build the pipeline after this proposal. Design and checks: [grpo-pipeline.md](grpo-pipeline.md).
- **Updates per wave: 4 minibatches (decided 2026-10-26).** The kept rollouts of a wave are split into 4 minibatches, each mixing the games and each game's high and low advantages, with one optimizer step per minibatch. Weights still sync to vLLM once per wave.
  - Adam moves each weight by about the learning rate per update, whatever the batch size, so one step per wave would give only 30 updates in 30 waves. 4 per wave give 120.
  - The cost: the trained data is 4–7 updates behind the policy that played it instead of 1. PPO's clipping and the ratio mask are made for this; the loop logs the clipping and ratio statistics per minibatch.
  - Compute, memory and time per wave are unchanged (every rollout still runs one forward + backward). This is common practice: DAPO makes 16 updates per rollout step.
- **Optimizer: AdamW for round 1 (2026-10-26).** Muon is built as an option (`--optimizer muon`) and is the first A/B candidate if round 1 learns slowly.
  - The agentic-RL evidence for Muon ([Ruan et al., 2607.16169](https://arxiv.org/abs/2607.16169): ALFWorld, 0.5B, one seed) traces the gain to Muon's larger updates.
  - No study pairs Muon with LoRA in RL.
- **Rollout: vLLM.** Same engine as our Kaggle submission, so chat template, tool parsing and sampling match deployment. It is also more widely used ([vLLM](https://github.com/vllm-project/vllm) ~94k stars vs [SGLang](https://github.com/sgl-project/sglang) ~37k), and in our experience SGLang was slower and harder to configure.
  - Yi-Chia Chen's weight sync and routing replay call SGLang's HTTP endpoints directly ([`sglang_controller.py`](https://github.com/ycchen-tw/agent-factory-3/blob/main/src/agent_factory_3/trainer/sglang_controller.py)), which is why his code needs SGLang. verl does both for vLLM itself.
  - Weight sync: merge LoRA into the base weights before pushing (verl: `lora.merge=True`).
    - This avoids vLLM runtime LoRA, which may not support GDN's packed projections, and rollout runs the same merged model we deploy.
    - verl's CUDA IPC push needs training and rollout on the same GPUs. With separate GPUs, push the merged attention/GDN tensors (~5.4 GB BF16) through vLLM's NCCL weight transfer, or send only the LoRA factors and merge them in the vLLM workers.
    - Reset the prefix cache after every sync ([h200-rollout-serving.md](h200-rollout-serving.md)).
  - Routing replay (R3): vLLM returns each token's routed experts (`--enable-return-routed-experts`).
    - verl's training-side replay exists only for the Megatron and VeOmni engines, not FSDP ([router_replay](https://github.com/volcengine/verl/tree/main/examples/router_replay)). In our own loop it would be a top-k override in the trainer.
    - Start without it, since experts and router are frozen.
- **Model code: transformers.** Loads the HF checkpoint directly, no conversion.
- **Training: a whole model per H200, LoRA data parallel (2026-10-26; replaces FSDP2 from 2026-10-12).**
  - train_rft fits the whole model on one H200 (PLE table in host RAM) up to about 133K tokens, so nothing needs sharding.
  - Each trainer takes its own trajectories, and the LoRA gradients are averaged before every optimizer step.
  - train_rft is single-GPU today; the averaging is still to write.
  - FSDP2 and Megatron shard a model that doesn't fit one GPU. Training is the bottleneck here (see Hardware), but the remedy is fewer tokens to train, not sharding.
- **LoRA on attention + GDN, merged after training; routed experts frozen.**
  - GDN is most of the mixer layers (3 of 4 in Qwen3-Next) and controls what the recurrent state keeps or forgets.
  - Merge W′ = W + (α/r)·BA into the BF16 weights, then re-quantize. Start from BF16, not dequantized weights: a ΔW smaller than half a quantization step rounds back to the old value.
  - Expert LoRA is out. Merging needs the full BF16 experts in memory plus a rerun of the 4-bit quantization (NVFP4 / W4A16). Serving it unmerged slows vLLM inference.
  - Caveat: [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) finds attention-only LoRA significantly worse than MLP (incl. MoE) LoRA, in SFT and in RL on MATH, while attention on top of MLP adds nothing. We accept this for deployability. From the same post: LoRA LR ≈ 10× full fine-tuning LR, rank 1 suffices for policy-gradient RL, and LoRA prefers small batches.
- **Start GRPO from the base model, not RFT v1 (proposed 2026-10-26).** RFT v1 scored 8.6 points per game below the base model on Kaggle (p = 0.003), concentrated on the games and levels it trained on ([rft-v1-kaggle-ab.md](rft-v1-kaggle-ab.md)).
- **Harness, context and tools identical in training and evaluation.** RL learns behaviour conditioned on what the model sees: system prompt, tool names and schemas, action interface, observation rendering (frames, frame diff, level-up board) and context management (truncation, compressed trace). Any mismatch is a distribution shift, and the LoRA gain may not transfer. Freeze this model-facing interface before RL; a later prompt or observation change means retraining. Only invisible parts may differ: scheduling (streams, priority, allocation, turn yield), training-only branching, and serving precision (W4A16 with FP8 KV on the H200 rollouts vs the Kaggle runtime's quantization).

- **GRPO on levels 1–2 first, with a context budget per episode: min(1,750 × human actions of L1 + L2, 130K) context tokens (decided 2026-10-23; replaces min(1,000 × h, 100K) generated tokens from 2026-10-21).**
  - An episode starts at the game's reset and succeeds if it clears L2 within the budget. GRPO rollouts never trim the context, so one sequence holds the system prompt, observations, tool results, images and every generated token; the budget caps that whole sequence. One number is simpler than a generated-token budget plus a separate context limit.
  - 130K keeps every rollout trainable on one H200 (it fits about 133K). 1,750 gives the most games with mixed outcomes under that cap, and is the earlier rule translated: context is a median 1.83 × the generated tokens.
  - On 43 of our runs (25 public games, 1,075 attempts, context rebuilt untrimmed from transcripts), 51% of attempts clear L1 + L2 within this budget (82% with no limit; 53% before images were priced at each run's board size on 2026-10-25). 15 games have mixed outcomes (20–80%), 4 always succeed, and bp35 and sk48 (0% each) give little signal. Analysis: [grpo-l1-l2-token-budget.md](grpo-l1-l2-token-budget.md).
  - The cap is tighter than the earlier one (130K of context ≈ 70K generated). It binds on 14 games: ar25, cn04, dc22, g50t, ka59, lf52, ls20, m0r0, s5i5, sk48, sp80, tn36, tr87, wa30.
  - The bp35 exception (a flat 100K generated) is dropped: its 1,750 × h is 121K, close to the cap, and it stays at 0% either way.
  - Open: whether the model sees its budget.

- **Hardware: 4 H200 = 2 rollout + 2 training (decided 2026-10-26).**
  - Rollout: one TP=2 vLLM server with 64 concurrent rollouts, one group of 8 per game. It measured 3,920 generated tokens/s on the replay and 3,855 with the live harness ([h200-rollout-serving.md](h200-rollout-serving.md)).
  - A 64-rollout wave of the 8 games below holds about 7.6M trajectory tokens, 4.5M of them generated (historical attempts under the 130K budget). Generating it takes about 19 min.
  - Training all 64 on the two H200s (5.7K tokens/s) takes about 22 min, so the trainers fall ~15% behind. Training 4 of each group's 8 (PODS) takes about 11 min.
  - About 3 steps per hour, about $7 per step at secure prices ($5.29 per H200-hour).
  - Train on 6 of 8 per group (decided 2026-10-26): it uses the slack, about 16 min of training per ~19-min wave. The loop is built (see Framework); training runs 4 minibatches per wave (Updates per wave). [grpo-training-throughput.md](grpo-training-throughput.md)

- **Reward: per-level credit (decided 2026-10-21).** Reward = Σ over levels cleared within the budget of wₗ · min(1, (hₗ / aₗ)²), with w = 1 for L1 and 2 for L2, divided by 3. hₗ is the level's human action count, aₗ the agent's actions on it. This is the ARC score formula restricted to L1–L2. Clearing L1 alone earns credit, so games that rarely clear L2 still give signal.
  - **Thinking factor (decided 2026-10-26).** Each cleared level's credit is multiplied by 1 + β · clip((1 − ρₗ) / 0.5, −1, 1), with β = 0.2.
    - ρₗ is the thinking tokens per action on level l, divided by τ̄, the base model's median thinking tokens per action for that game. τ̄ comes from our historical runs and stays fixed during training, so the target doesn't move as the policy changes.
    - The factor runs from 1.2 at ≤ 0.5× the game's usual thinking per action, through 1.0 at 1×, to 0.8 at ≥ 1.5×.
    - On Kaggle each game has a time budget, so less thinking on L1–L2 leaves time for later levels. The 130K budget only cuts off a rollout at its edge; it doesn't reward a faster clear below it.
    - Thinking tokens only (no tool code or replies), per action, so the action term and this one don't both charge for extra actions. Each game is compared with its own average.
    - Failures stay at 0 and a slow clear keeps ≥ 80% of its credit, so clearing always beats failing. A fast clear can make up for at most a third of action efficiency: 0.67 × 1.2 ≈ 1 × 0.8.
    - To do before the run: compute τ̄ for the 8 games, and check how far ρ spreads within groups.

- **Games: 8 public games (decided 2026-10-26; replaces the 21 training games of 2026-10-21).** bp35, cd82, g50t, s5i5, sk48, sp80, tn36, wa30, one group of 8 rollouts each per wave. All have human action counts in `benchmark.json`.
  - cd82 replaced su15 the same day: su15 cleared L1 + L2 within budget in 70% of attempts, too easy. cd82 is harder on both levels and its rollouts disagree about twice as much on the level reward.
  - The other 17 public games stay untrained, including the 4 guards (ft09, lp85, re86, tr87). So they can measure transfer to games the policy never trained on (see When to stop).
  - Cleared within budget in our historical attempts, L1 / L1 + L2:

    | Game | L1 | L1 + L2 |
    |---|---:|---:|
    | bp35 | 93% | 0% |
    | cd82 | 58% | 35% |
    | g50t | 58% | 23% |
    | s5i5 | 100% | 51% |
    | sk48 | 40% | 0% |
    | sp80 | 100% | 37% |
    | tn36 | 91% | 14% |
    | wa30 | 100% | 21% |

  - Seven of the 8 run to the budget in most attempts (63–100%), so a wave lasts about as long as the budget allows.
  - Caveat (undecided): bp35 and sk48 never clear L2 within budget, so they train L1 only, and every one of their rollouts runs the full budget: 16 of the 64 slots. Swapping them for two mid-range games (e.g. dc22 and m0r0) raises the mean within-group variance of the level reward (0, ⅓ or 1) from 0.092 to 0.125, with the same wave time.

- **When to stop: level 3 as the held-out set (decided 2026-10-21).** Training rewards only L1–L2, so L3 of every game is never trained on, and memorized L1–L2 action sequences cannot help on it.
  - Every 5 steps, play each game from reset through L3 with the current policy (same harness and context as evaluation, so with its usual trimming: these rollouts are not trained on, so the 130K training limit does not apply), a few rollouts per game.
  - Held-out measure: P(clear L3 within 1,000 × h₃ generated tokens on L3 | reached L3), computed per game and averaged with fixed game weights, so that more hard-game attempts reaching L3 does not shift the mix. Count a game in a check only if ≥ 4 rollouts reached L3. Historically 76% (905 attempts reached L3), with mixed outcomes on 12 games.
  - Also tracked: L1–L2 within budget (the training measure), the guard games, thinking tokens per turn and MTP draft acceptance.
  - Stop when any of these happens: the L3 measure has not beaten its best for 2 checks in a row; a guard game drops below 95% within budget; thinking per turn falls more than 25% below the base model (10% until 2026-10-26; the reward now pays for less thinking); training reward keeps rising while L3 stays flat (memorization).
  - Caveat: L3 is the same game, so it measures transfer within a game, not to new games.
  - New with 8 training games (proposed 2026-10-26): the 17 untrained games measure transfer to new games, e.g. L1 + L2 within budget at every check. That costs about one extra wave per check.

## Data

For later rounds. Round 1 trains on 8 of the public games (see Games). Training on other games would let all 25 public games measure generalisation to the 110 hidden games. Local copies sit next to this repo in `../`.

| Source | Games | Format | Notes |
| --- | --- | --- | --- |
| [arc-interactive](https://github.com/theredbluepill/arc-interactive) (`../arc-interactive`) | 249 | arcengine | Community, mostly agent-built, many classic-game ports (Sokoban, Minesweeper). Quality varies; has a solvability checker. Skip its old copies of ft09, ls20, vc33. |
| [arc3-synthetic-games](https://github.com/Felix561/arc3-synthetic-games) (`../arc3-synthetic-games`) | 50 Studio + 25 [NVIDIA DreamTeam](https://github.com/NVIDIA/dream-team) | arcengine | 550 recorded level solves by GPT-6.1-sol, but source-informed (the agent read the game code). NVIDIA adapter never calls `lose()`, so a lost game sits on a dead board; fix before use. |
| [ARC3.Games](https://arc3.games/) ([code](https://github.com/Poolminer/ARC3.Games)) | public games only | arcengine | Database of human action sequences per level, with API (`getTopSequences`). No new games, but shortest human solutions, and replaying them reaches any level start as an RL branch point. |
| [PuzzleJAX](https://arxiv.org/abs/2508.16821) ([code](https://github.com/smearle/script-doctor), `../script-doctor`) | ~930 PuzzleScript | PuzzleScript DSL | Most diverse, but frames and actions need a converter to the ARC3 interface. Later. |
| Coding-agent-generated games | open | arcengine | New mechanics, each with a reference solution to prove it is winnable. |

- Filter by difficulty using our own agent's pass rate, and drop games whose mechanics duplicate the public 25.
- [LamDang PR #27](https://github.com/LamDang/arc-agi-3-solution-da-fr/pull/27) already plays the arcengine sets: gpt-6.1-sol won 22/25 NVIDIA games (179/200 levels, mean 76.7).
- "ARENA 50 games" from the shared plan: not found yet; ask for the link.
