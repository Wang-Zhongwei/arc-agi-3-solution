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

- **Framework: [verl](https://github.com/volcengine/verl).** Widely used. Supports LoRA, vLLM or SGLang rollout, and multi-turn tool use (`AgentLoop`). [slime](https://github.com/THUDM/slime) has no LoRA, so it would mean full-parameter RL on a multi-node cluster.
- **Rollout: vLLM.** Same engine as our Kaggle submission, so chat template, tool parsing and sampling match deployment. It is also more widely used ([vLLM](https://github.com/vllm-project/vllm) ~94k stars vs [SGLang](https://github.com/sgl-project/sglang) ~37k), and in our experience SGLang was slower and harder to configure.
  - Yi-Chia Chen's weight sync and routing replay call SGLang's HTTP endpoints directly ([`sglang_controller.py`](https://github.com/ycchen-tw/agent-factory-3/blob/main/src/agent_factory_3/trainer/sglang_controller.py)), which is why his code needs SGLang. verl does both for vLLM itself.
  - Weight sync: verl pushes weights into vLLM over CUDA IPC. Use `lora.merge=True`: the trainer merges LoRA into the base weights before pushing. This avoids vLLM runtime LoRA, which may not support GDN's packed projections, and rollout runs the same merged model we deploy.
  - Routing replay (R3): vLLM ≥ 0.22 returns routed experts, but verl's training-side replay exists only for the Megatron and VeOmni engines, not FSDP ([router_replay](https://github.com/volcengine/verl/tree/main/examples/router_replay)). Start without it, since experts and router are frozen. If training is unstable, switch to VeOmni (FSDP2 + expert parallelism; R3 example for Qwen3.5-35B-A3B).
- **Model code: transformers.** Loads the HF checkpoint directly, no conversion.
- **Training: FSDP2; Megatron later if MoE efficiency is too low.** FSDP2 is the simplest start. No DeepSpeed: verl has no DeepSpeed backend, and FSDP2 is about as fast as ZeRO-3. Megatron still helps with frozen experts, since the backward pass carries activation gradients through every MoE layer. But rollout usually dominates RL time, so switch only if profiling shows training is the bottleneck.
- **LoRA on attention + GDN, merged after training; routed experts frozen.**
  - GDN is most of the mixer layers (3 of 4 in Qwen3-Next) and controls what the recurrent state keeps or forgets.
  - Merge W′ = W + (α/r)·BA into the BF16 weights, then re-quantize. Start from BF16, not dequantized weights: a ΔW smaller than half a quantization step rounds back to the old value.
  - Expert LoRA is out. Merging needs the full BF16 experts in memory plus a rerun of the 4-bit quantization (NVFP4 / W4A16). Serving it unmerged slows vLLM inference.
  - Caveat: [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) finds attention-only LoRA significantly worse than MLP (incl. MoE) LoRA, in SFT and in RL on MATH, while attention on top of MLP adds nothing. We accept this for deployability. From the same post: LoRA LR ≈ 10× full fine-tuning LR, rank 1 suffices for policy-gradient RL, and LoRA prefers small batches. - **Harness, context and tools identical in training and evaluation.** RL learns behaviour conditioned on what the model sees: system prompt, tool names and schemas, action interface, observation rendering (frames, frame diff, level-up board) and context management (truncation, compressed trace). Any mismatch is a distribution shift, and the LoRA gain may not transfer. Freeze this model-facing interface before RL; a later prompt or observation change means retraining. Only invisible parts may differ: scheduling (streams, priority, allocation, turn yield), training-only branching, and serving precision (BF16 rollout vs NVFP4 on Kaggle).

- **GRPO on levels 1–2 first, with a token budget per episode: min(1,000 × human actions of L1 + L2, 100K) generated tokens (decided 2026-10-21).**
  - An episode starts at the game's reset and succeeds if it clears L2 within the budget. Generated tokens are thinking, replies and tool code; the prompt context is managed by the harness as in evaluation.
  - On our 44 runs (25 public games), 63% of attempts clear L1 + L2 within this budget (82% with no limit). 15 games have mixed outcomes (20–80%) and 5 always succeed; bp35 (0%) and sk48 (5%) give almost no signal. Analysis: [grpo-l1-l2-token-budget.md](grpo-l1-l2-token-budget.md).
  - The model spends a median of 575 tokens per human action on cleared L1 + L2, so 1,000 × h sits above most clears. The 100K cap binds on games with more than 100 human actions (dc22, ls20, ka59, m0r0, s5i5, lf52, tn36, wa30, g50t, sk48, tr87).
  - Open: games without human action counts (some training sets) need another source for h; the reward (human-relative level score within the budget); whether the model sees its budget.

## Data

The 25 public games are validation only; train on other games so the score measures generalisation to the 110 hidden games. Local copies sit next to this repo in `../`.

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
