# W2: RL for LLMs: algorithms, stability, and MoE/quantization-specific issues (sweep 2026-10-18)

Scope: which RL and preference-optimization methods are viable for Qwen3.8-Flash-Next (125B hybrid GDN/QSA MoE, 512 experts, top-10 + 1 shared, trained in BF16, served as Intel AutoRound W4A16 on our SGLang build), what they cost, and the pitfalls. Builds on `research/data/finetune-research-2026-10-17/training.md` §2-3, which had GiGPO, RAGEN, Router Replay and ReST-EM at abstract level only. Here the numbers come from the papers' bodies where I could read them. "[abstract]" means only the abstract or a search snippet was read, and "UNVERIFIED" means not confirmed in this sweep.

## 1. Algorithm landscape: what each method fixes, and what labs use for MoE

### Takeaway
For MoE models, labs in 2025-2026 stay critic-free and group-relative and add one of three stabilizers:
- sequence-level importance ratios: GSPO and its successor SAPO, both Qwen's;
- clipped IS weights that keep every token's gradient: CISPO (MiniMax), also used in Meta's ScaleRL;
- token-level PPO plus routing replay and mismatch masking: DeepSeek-V3.2 and Xiaomi's R3.

Plain GRPO is the baseline that crashes on MoE. Qwen trained Qwen3-Next with GSPO, so a GSPO-family loss is the natural default for Flash-Next.

### Cited Findings
- **GSPO** (arXiv 2507.18071, 2025-07-24): [HTML v2](https://arxiv.org/html/2507.18071v2)
  - The ratio is the sequence likelihood ratio raised to 1/|y|, the geometric mean of the token ratios (§4.1, Eq. 7).
  - Clip ranges are 3e-4 / 4e-4, vs GRPO's 0.2 / 0.27 (§5.1).
  - It clips about 100x more tokens than GRPO and still trains more efficiently (§5.2).
  - The GSPO-token variant allows per-token advantages for multi-turn use (§4.3).
  - In the 48-layer Qwen3-30B-A3B, "roughly 10% of the experts activated" change after each gradient update. GRPO needed Routing Replay to converge and GSPO doesn't (§5.3).
  - The paper argues GSPO is "more tolerant of precision discrepancies" (§5.4; argued, not shown).
- **Qwen3-Next model card:** "Leveraging GSPO, we have addressed the stability and efficiency challenges posed by the hybrid attention mechanism" with high-sparsity MoE in RL. [HF](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking)
- **No RL method has been published for Flash-Next.** The Qwen3.8-Next architecture paper (arXiv 2608.30320, 2026-08-31) covers only pretraining stability (the gated residual stays stable at 4x the optimal LR) in the part I read. UNVERIFIED which algorithm its post-training used. [HTML](https://arxiv.org/html/2608.30320)
- **SAPO** (Qwen, arXiv 2511.20347): a temperature-controlled soft gate replaces hard clipping. It down-weights only the off-policy tokens, and negative-advantage tokens get faster-decaying gates. Compared with GSPO and GRPO-R2 on Qwen3-VL-30B-A3B [abstract]. [arXiv](https://arxiv.org/abs/2511.20347)
- **CISPO** (MiniMax-M1, arXiv 2506.13585): [HTML](https://arxiv.org/html/2506.13585)
  - Clips the stop-gradient IS weight, so all tokens keep their gradients. No KL term; only ε_high is tuned (§3.1).
  - On Qwen2.5-32B it matched DAPO's AIME with 50% of the steps, a "2x speedup" (§3.1, Fig. 2).
- **ScaleRL** (Meta, arXiv 2510.13786, 2025-10-15; >400k GPU-h of ablations): [HTML](https://arxiv.org/html/2510.13786)
  - CISPO and GSPO both beat DAPO's asymptote "by a large margin", and CISPO is marginally ahead late in training (Fig. 5a).
  - The final recipe (§3.2, §4):
    - PipelineRL with off-policyness k=8;
    - forced end-of-thinking interruptions;
    - FP32 LM head;
    - CISPO;
    - prompt-level loss averaging;
    - batch-level advantage normalization;
    - zero-variance prompt filtering;
    - no-positive-resampling (prompts with pass rate ≥0.9 are dropped).
- **DAPO** (arXiv 2503.14476): 50 on AIME'24 from Qwen2.5-32B base. UNVERIFIED in this sweep, from memory: its four techniques are Clip-Higher (counters entropy collapse), Dynamic Sampling (drops all-0 and all-1 groups), token-level loss and overlong shaping. [arXiv](https://arxiv.org/abs/2503.14476)
- **Dr. GRPO** (arXiv 2503.20783): GRPO's bias "artificially increases response length (especially for incorrect outputs)". The fix "improves token efficiency". UNVERIFIED: the fix drops the 1/|o| and std normalizations. [arXiv](https://arxiv.org/abs/2503.20783)
- **VAPO** (value-based PPO, arXiv 2504.05118): 60.4 on AIME'24 on Qwen-32B in 5k steps, ">10 points" over DAPO, "no training crashes" [abstract]. It needs a critic as large as the policy. [arXiv](https://arxiv.org/abs/2504.05118)
- **GiGPO** (arXiv 2505.10978): [HTML](https://arxiv.org/html/2505.10978)
  - Step-level groups come from hashing identical environment states ("anchor states") across a group, with no extra rollouts. A = A_episode + ω·A_step, with ω=1 and γ=0.95.
  - Table 1 (ALFWorld All / WebShop success):

    | Model | PPO | RLOO | GRPO | GiGPO |
    |---|---|---|---|---|
    | 1.5B | 54.4 / 51.5 | 69.7 / 52.1 | 72.8 / 56.8 | 86.1 / 67.4 |
    | 7B | 80.4 / 68.7 | 75.5 / 65.7 | 77.6 / 66.1 | 90.2 / 75.2 |

  - More than 65% of states recur (Fig. 5). The overhead is under 0.002% of iteration time.
  - The setup used only a 2-step history and 2-4k-token prompts (App. E.1).
- **Entropy** (Cui et al., arXiv 2505.22617): R = -a·e^H + b. Entropy falls because of the covariance between log-prob and advantage. Clip-Cov and KL-Cov counter it [abstract]. [arXiv](https://arxiv.org/abs/2505.22617)
- **Turn-level credit methods** (Turn-PPO 2512.17008, TCPO 2608.01667, survey 2604.09459): search-level only, and none was tested on a large MoE. [survey](https://arxiv.org/html/2604.09459v2)

### Inferences
- **Default loss:** GSPO or GSPO-token (Qwen's method for this family), or CISPO. ms-swift's Flash-Next recipe uses `--loss_type grpo`. Check that its GSPO and CISPO options run on the Megatron backend before committing.
- **GiGPO's anchor states fit ARC-AGI-3,** because frames are exactly hashable. But our contexts differ across rollouts that reach the same frame, so grouping by frame is only approximate.
- **No critic.** PPO and VAPO need a second ~125B network, which one node can't hold.

### Gaps
- Flash-Next's own RL recipe is unpublished.
- I found no GSPO-vs-CISPO comparison on a hybrid linear-attention MoE.
- I didn't re-read REINFORCE++ (2501.03262).

## 2. MoE-specific: routing instability, routing replay, router freezing, load balance

### Takeaway
On MoE, routing that differs between rollout and training, and across updates, is a measured cause of RL collapse. Rollout Routing Replay (R3) fixes it at under 3% overhead, and ms-swift and NeMo RL both implement it, but only with vLLM rollouts. The load-balancing aux loss should be off during RL, yet ms-swift's Flash-Next recipe leaves it on.

### Cited Findings
- **R3 paper** (arXiv 2510.11370, v2 2025-10-21; Qwen3-30B-A3B, SGLang rollouts, Megatron training, VeRL): [HTML v2](https://arxiv.org/html/2510.11370v2)
  - About 10% of routers pick different experts than at inference, and 94% of tokens differ in at least one layer (§3.2).
  - Training-inference KL (k3) is 1.535e-3 for the MoE vs 6.4e-4 for the dense Qwen3-8B. With R3 it drops to 7.5e-4 (§3.1, §4.3).
  - Two repeated Megatron forward passes alone give KL 8.4e-4 (§3.3).
  - Table 1:
    - GRPO crashed in 3/3 runs (steps 60, 105, 120).
    - GRPO+TIS crashed in 1/2.
    - All 4 R3 runs stayed stable.
  - Multi-turn SWE (Table 2): GRPO crashed at step 90, while GRPO+R3 reached 38.6 vs 31.8.
  - Rollout latency overhead is under 3%, and routing masks are cached with the prefix KV cache (§4.1-4.2).
  - R2 (training-side replay) doesn't fix framework mismatch and is a no-op at one mini-step (§6.3).
- **DeepSeek-V3.2:**
  - "Keep Routing" (inference routing enforced in training) is "crucial" for MoE RL stability, and they have used it since V3-0324.
  - "Keep Sampling Mask" applies the rollout's top-p/top-k truncation masks to the training policy.
  - Off-policy sequence masking (§3.1).
  - [HTML](https://arxiv.org/html/2512.02556)
- **ms-swift:** `--router_replay_mode {disabled,R2,R3}`, Megatron only. R3 needs vLLM ≥0.14.0 (`routed_experts`) and megatron-core ≥0.16, and "R3+TIS may give no additional gains". [docs](https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/router_replay.html)
- **NeMo RL:** `policy.router_replay.enabled`, with vLLM `enable_return_routed_experts=True`. "Other generation backends are not wired into this path". [docs](https://docs.nvidia.com/nemo/rl/latest/guides/router-replay.html)
- **SGLang-side R3** exists in Miles (RadixArk, slime's successor, arXiv 2609.08368, 2026-09-08) via `--use-rollout-routing-replay`. Miles also lists LoRA RL and on-policy distillation [abstract/snippet]. [arXiv](https://arxiv.org/abs/2609.08368)
- **Aux loss:** NeMo RL's Qwen3.5 MoE guide says `router_aux_loss_coef` 0.001 left on during RL "severely degrades accuracy", and its recipes set it to 0.0. The router stays fp32 and unquantized. [NeMo RL Qwen3.5](https://docs.nvidia.com/nemo/rl/nightly/guides/models/qwen/qwen3-5.html)
- **ms-swift's Flash-Next GRPO recipe:**
  - `--moe_aux_loss_coeff 1e-3`;
  - LoRA r8/α32 on `in_proj out_proj linear_proj linear_qkv` (GDN and attention only);
  - no router-replay or IS settings.
  - [ms-swift](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html)

### Inferences
- **Router freezing comes free with LoRA, but routing still drifts.** No LoRA target touches `mlp.gate`. LoRA on attention and GDN changes the router's inputs, so R3 (or GSPO) is still needed.
- **Set the aux loss to 0 even with a frozen router.** Its gradient still flows through the hidden states into the upstream LoRA (from the computation graph).
- **Our serving stack matches the router today.** The leader arm `mem-repair-10x` sets no `SGLANG_ARC_MOE_ROUTING` (`kaggle/arms.json:1223` onward), so our server's routing equals the trained router. Any arm that enables that patch makes R3 mandatory.

### Gaps
- No public RL run on a 512-expert top-10 model with hyper-connections.
- Whether R3 composes with LoRA in ms-swift is undocumented.

## 3. Training-inference mismatch: can we roll out on W4A16 and train in BF16?

### Takeaway
Not naively:
- With post-training-quantized INT4 rollouts and a BF16 trainer, the mismatch grows during training.
- NVFP4 rollouts collapse.
- TIS and MIS fix INT8/FP8-sized gaps but have not been shown at INT4.
- The working INT4 recipe (LMSYS/slime) fake-quantizes in the trainer so both sides use the same weights.

Our cheaper variant, not yet tested by anyone: train LoRA against the dequantized W4A16 experts and adapt only BF16 modules. Weight mismatch is then zero and no re-quantization is needed. The Kaggle server also adds an FP8 KV cache, T=0.65 with top-p/top-k sampling, and no logprob logging; each needs handling.

### Cited Findings
- **Yao et al.** (OPT 2025, the work behind the "secretly off-policy" blog): [PDF](https://opt-ml.org/papers/2025/paper116.pdf)
  - With identical weights, vLLM and FSDP can give contradictory probabilities (π_vllm=1 where π_fsdp=0). This persists after exposing the true sampling probabilities and using an FP32 lm_head (§3.1, Fig. 1; Qwen2.5-32B).
  - TIS uses min(π_train/π_rollout, C) (§3.2).
  - INT8 rollouts degrade PPO on GSM8K, and TIS restores "similar" performance (§4.2, Fig. 2; 0.5B).
  - PPO-IS and untruncated IS reach about 0 accuracy with INT8 (§4.3, Fig. 3).
- **FlashRL** (INT8/FP8 rollouts + TIS, vLLM/verl): recommended for 14B+ models. No MoE and no INT4. [GitHub](https://github.com/yaof20/Flash-RL)
- **"When Speed Kills Stability"** (Liu, Li et al., 2025-09-17):
  - Token-level IS has a bias that grows with length. Sequence-level IS is unbiased but has exponential variance, and truncating or masking it balances the two.
  - Its geometric sequence masking was adopted by DeepSeek-V3.2 [snippet].
  - [blog](https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda)
- **ms-swift mismatch corrections:**
  - `--rollout_importance_sampling_mode {token,sequence}_{truncate,mask}`, threshold default 2;
  - `--off_policy_sequence_mask_delta` (masks only negative-advantage sequences);
  - logs `rollout_correction/k3_kl`, `chi2_seq`, ESS and `clipped_frac`.
  - [docs](https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/training_inference_mismatch.html)
- **FP32 LM head:**
  - MiniMax-M1: the training/inference probability correlation went from about 0.9x to 0.99x (§3.2).
  - ScaleRL: the asymptote rose from 0.52 to 0.61, and the head mattered more on the Scout MoE (§3.2, §4).
  - [MiniMax](https://arxiv.org/html/2506.13585), [ScaleRL](https://arxiv.org/html/2510.13786)
- **FP16 instead of BF16** (arXiv 2510.26788, 2025-10-30): BF16 rounding is the root cause, and FP16 "effectively eliminates" the mismatch [abstract]. [arXiv](https://arxiv.org/abs/2510.26788)
  - Counter-view (arXiv 2602.01826, 2026-02-02): mismatch grows "in tandem" with gradient noise, and IS can fail in long runs. Their fix is an LR decay triggered by response length. [arXiv](https://arxiv.org/abs/2602.01826)
- **LMSYS/slime INT4 QAT RL** (2026-01-26): [blog](https://www.lmsys.org/blog/2026-01-26-int4-qat)
  - Setup: Megatron fake-quant with STE (1×32 groups, range [-7,7]) and real W4A16 Marlin in SGLang, on Qwen3-235B-A22B and Kimi-K2-Thinking.
  - Without QAT, the logprob error "oscillate[s] and increase[s] with training steps".
  - With QAT, INT4 "nearly overlaps the BF16 baseline" and beats FP8. Plots only, no numbers.
- **QUADS** (arXiv 2607.15810, 2026-07-17): "NVFP4 rollout with BF16 training collapses after roughly 150 steps" on MoE, mainly from activation error, not weight error. The fix is trainer weight fake-quant plus activation compensation. [arXiv](https://arxiv.org/abs/2607.15810)
- **FP8 and KV cache:**
  - FP8-RL (arXiv 2601.18150): up to 44% rollout throughput "comparable to BF16" with W8A8, an FP8 KV cache with per-step QKV scale recalibration, and token TIS/MIS. [arXiv](https://arxiv.org/abs/2601.18150)
  - NeMo RL FP8 Qwen3.5-35B-A3B: train/gen KL about 0.0024-0.0031 (max 0.005), with nightly gates of 0.003 mean / 0.004 max. [docs](https://docs.nvidia.com/nemo/rl/nightly/guides/models/qwen/qwen3-5.html)
  - KV4-RL (arXiv 2610.14448): an INT4 KV cache hurts learning even with token TIS. [arXiv](https://arxiv.org/abs/2610.14448)
  - Full-pipeline FP8 RL (arXiv 2609.22870): entropy surges, fixed by "Calibrated Clipping". [arXiv](https://arxiv.org/abs/2609.22870)
- **Our stack (repo facts):**
  - Only routed experts are int4. Attention, GDN, shared expert, router, hyper-connections, MTP and PLE stay BF16 (`arms.md:634`).
  - The server runs `--dtype bfloat16 --quantization auto-round` with KV cache `fp8_e4m3` (`serving/kaggle/launch_sglang.py:37, 354-355`).
  - Sampling defaults are T=0.6, top-p 0.95, top-k 20 (`ARC3-Inference/inference/agent/tool_agent.py:1015-1017`), and the leader uses T=0.65 (`kaggle/arms.json:1232`).
  - No harness code requests logprobs (grep `logprob` over `ARC3-Inference/inference`: no hits).

### Inferences
- **Train against the served weights.** Load the AutoRound int4 experts dequantized and frozen, and put LoRA on BF16 modules only. Trainer and server then share weights, which is what QAT achieves, and the merged LoRA needs no re-quantization. This removes the plan's re-quant drift risk (risk 3).
  - What remains is kernel mismatch and the FP8 KV cache, the activation-side error that QUADS and KV4-RL flag.
  - TEST-FIRST (about 2 GPU-h): per-token KL between SGLang `return_logprob` and the trainer on about 200 logged decisions, at the same T with top-p/top-k off, bf16 vs fp8 KV. The bar is NeMo's ≤0.003.
- **Sampling consistency.** RL rollouts need top-p=1 and top-k=-1, or stored masks (Keep Sampling Mask), and trainer logprobs at the same temperature.
- **The ~180k logged calls have no behaviour logprobs.** They can't feed IS-corrected RL, but they are fine for RFT, SFT and DPO.
- **Kaggle rollouts can't be on-policy for online RL,** because a new policy can't be pushed to Kaggle each step. Kaggle provides off-policy one-step data (RFT, DPO pairs) and evaluation. Online RL rollouts must run on the Azure node.
- **FP16 training** of a BF16-pretrained checkpoint with hyper-connections risks overflow. Test it first.

### Gaps
- No published W4A16 (weight-only int4) rollout plus TIS/MIS measurement.
- Unknown whether mcore-bridge loads AutoRound int4 experts or needs an offline dequantize to BF16.
- The shared-expert module names in Megatron are unknown.

## 4. Cheaper alternatives: when do RFT, DPO, offline RL or distillation recover most of RL's gain?

### Takeaway
In the short-training regime our budget allows, rejection-sampling fine-tuning recovers most of the gain. Online or semi-online DPO matches GRPO, and purely offline methods fall short. On-policy distillation is about 10x cheaper than RL and is the documented repair for behaviour lost to fine-tuning. Thinking Machines warns that SFT on a model's own samples drifts, which weakens the prior plan's "30-50% self-generated replay" guard.

### Cited Findings
- **RAFT vs GRPO** (arXiv 2504.11343): RAFT (positive samples only) "performs competitively" with GRPO and PPO. GRPO's main benefit comes from discarding all-incorrect prompts, not from reward normalization [abstract]. [arXiv](https://arxiv.org/abs/2504.11343)
- **Online vs offline DPO** (Meta, arXiv 2506.21495): online and semi-online DPO reach "similar performance and convergence" to GRPO, and both clearly beat offline methods [abstract]. [arXiv](https://arxiv.org/abs/2506.21495)
- **DigiRL** (1.3B VLM agent): SFT on demos 17.7% → filtered BC 57.8% → offline-to-online RL 67.2%. Filtered BC got about 80% of the gain [abstract]. [arXiv](https://arxiv.org/abs/2406.11896)
- **On-policy distillation** (Thinking Machines, 2025-10-27): [blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
  - Qwen3 spent 17,920 GPU-h on RL vs 1,800 on distillation.
  - Instruction following lost to mid-training (IF-eval 85→45) was restored to 83.
  - "Running SFT on top of Qwen3-32B's own samples degrades performance", because finite batches turn it "into off-policy training over time".
  - No chat-data mix kept the original IF-eval (85→79 at a 70/30 mix).
- **LoRA Without Regret** (Thinking Machines, 2025-09-29): [blog](https://thinkingmachines.ai/blog/lora/)
  - LoRA "performs equivalently to FullFT for reinforcement learning even with small ranks", including rank 1.
  - The best LoRA LR is about 10x the full-FT LR (about 15x for runs under 100 steps).
  - "Attention-only LoRA significantly underperforms MLP-only LoRA".
  - LoRA tolerates large batches less well than full FT.
- **VinePPO:** Monte-Carlo values from intermediate states beat PPO in "up to 3.0x" less wall-clock [abstract]. [arXiv](https://arxiv.org/abs/2410.01679)
- **ReST-EM and SWE-Gym** (prior notes, `finetune-research-2026-10-17/training.md:38-39`).

### Inferences
- **Our planned SFT targets already are RAFT/ReST.** R3 (shortest correct) and R4 (stuck-level breakthroughs) are the right first move, and shortest-correct is ShorterBetter's reward target.
- **Add negatives cheaply.** Build same-state pairs (shortest-correct vs wrong or long), then do one semi-online DPO round after re-sampling with the stage-1 model.
- **LoRA targets.** ms-swift's attention-only targets are the weak case in Thinking Machines' study. The BF16 shared-expert MLP is an MLP target that still merges without re-quantization (test-first).
- **Replace or cap own-sample replay.** Use on-policy self-distillation to the frozen base, with LoRA-off as the teacher, on general prompts, or limit replay to 1 epoch at low LR.

### Gaps
- No comparison of RFT vs RL on multi-turn hidden-rule games.
- The anti-forgetting effect of distillation on a 125B MoE is unmeasured.

## 5. Pass-rate requirements, curricula, and published compute

### Takeaway
Group methods learn only where 0 < pass rate < 1, so labs filter zero-variance and nearly-always-solved prompts, or weight by p(1-p). Published MoE RL runs cost 10k-100k+ GPU-h. Our ceiling is about 2.9k GPU-h (one 8×H200 for 15 days) and the realistic budget 450-950 GPU-h. That means a few hundred RL steps at most, the regime where RFT is competitive.

### Cited Findings
- **Filtering:** ScaleRL excludes zero-variance prompts and permanently drops prompts with pass rate ≥0.9 (§3.2, Fig. 6). [HTML](https://arxiv.org/html/2510.13786)
- **LILO** (arXiv 2502.12272): p(1-p) sampling "consistently boosts" PPO and VinePPO [abstract]. [arXiv](https://arxiv.org/abs/2502.12272)
- **DeepSeek-V3.2** kept only agent environments with non-zero pass@100: 1,827 environments and 4,417 tasks (§3.2.3). [HTML](https://arxiv.org/html/2512.02556)
- **Published compute:**
  - MiniMax-M1 (456B-A45.9B): 512 H800 for about 3 weeks, $534,700. [MiniMax](https://arxiv.org/html/2506.13585)
  - ScaleRL: 3.5-4k GPU-h per 8B ablation, 16k per full 8B run, about 50k for Llama-4 Scout (17B×16). [ScaleRL](https://arxiv.org/html/2510.13786)
  - DeepSeek-V3.2: RL spending above 10% of pretraining. [DeepSeek](https://arxiv.org/html/2512.02556)
  - NeMo RL Qwen3.5-35B-A3B DAPO: 32 GPUs for 100 steps, accuracy 0.33→0.69 and length 5.5k→2.5k tokens. Qwen3.5-397B GRPO: 256 GPUs. [NeMo](https://docs.nvidia.com/nemo/rl/nightly/guides/models/qwen/qwen3-5.html)
- **ms-swift's Flash-Next GRPO recipe is smoke-test sized:**
  - `--max_length 1024`, `--max_completion_length 8192`, `--num_generations 4`, `--global_batch_size 8`;
  - vLLM colocate TP8 at 0.35 memory, with a vLLM >0.28 source build including PR #53896;
  - "8*135G".
  - [ms-swift](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html)
- **No Base checkpoint:** Hugging Face lists only `Qwen/Qwen3.8-Flash-Next` and `-FP8` (both 2026-08-24). Source: `https://huggingface.co/api/models?author=Qwen&search=Flash-Next`, queried 2026-10-18.

### Inferences
- **Full-game-from-reset episodes give near-zero group variance:** most public games are always-won or never-won.
- **The usable RL unit is a frozen decision state.** Take a logged context, draw k replies, verify them by engine replay, and keep states with 0<p<1. Our measurements back this: in 96% of stuck decisions an untried action reached a new board, and on cleared levels 81% of moves lie on the hindsight-shortest route.
- **Context length.** Prompts are about 55k tokens vs the recipe's 1k + 8k, so RL must use trimmed 8-16k windows at both rollout and training time. Plan risk 2 (train-short, play-long) applies to RL too.
- **Rough rollout cost:** 2.2k-token median reply × k=8 at about 690 tok/s is roughly 25 s per state, or about 140 states per Kaggle GPU-h. About 1.5k states a day fit the free quota: enough for one RFT or DPO round a day, not for online RL.

### Gaps
- No published RL compute for any GDN/QSA hybrid.
- Flash-Next BF16 vLLM rollout throughput on 8×H200 is unmeasured.

## 6. Reward design for game play

### Takeaway
Use verifiable per-decision rewards:
- level cleared;
- a board not yet seen on this level;
- a move on the hindsight-shortest route.

Add the length term only on correct samples. Sparse level-complete reward gives no gradient on stuck levels, which take 50-60% of our tokens. RL focused on hard, mostly failed states tends to lengthen replies, which works against our main lever, tokens per action.

### Cited Findings
- **Fatemi et al.** (arXiv 2504.05185): incorrect answers "inherently drive policies toward verbosity" under PPO and GRPO. A short RL phase on solvable problems "significantly reduces response length" while keeping accuracy [abstract]. [arXiv](https://arxiv.org/abs/2504.05185)
- **ShorterBetter** (arXiv 2504.21370): uses the shortest correct sample's length as the target, for 50-80% shorter outputs at equal accuracy. Naively adopting the shortest length destabilizes training [snippet]. [arXiv](https://arxiv.org/abs/2504.21370)
- **MARSHAL** (games): removing the length penalty raised length from 1,700 to 1,954 tokens and dropped Mini-Hanabi from 50.5 to 38.2 (prior notes `training.md:30`, abstract-level).
- **MiniMax-M1:** stops generation after 3,000 consecutive tokens with probability >0.99, to avoid repetition-gradient blow-ups (§3.2). [HTML](https://arxiv.org/html/2506.13585)
- **RAGEN "Echo Trap" and RAGEN-2 "template collapse"** (prior notes `training.md:27`).
- **Repo:** copying the previous frame matches only 2.1% of transitions. A median of 48 cells change per action, and 94.5% of actions change the frame (`arms.md:628`).

### Inferences
- **"Frame changed" is useless as a reward,** since it is true for about 94.5% of actions. Novelty must mean "grid hash not yet seen on this level", with HUD/budget bars excluded and GAME OVER/reset frames excluded, so the policy can't farm novelty by dying.
- **Other hacks to guard against:**
  - loops through transient frames;
  - replies shortened by skipping needed tool calls;
  - burning the action budget.
- **Length term only when the reward is positive,** with no 1/|o| normalization (Dr. GRPO-style), so failed long replies aren't favoured.

### Gaps
- No published LLM reward-design study for hidden-rule interactive games. The ARC-AGI-3 RL work found is non-LLM: graph exploration (arXiv 2512.24156) and the CNN+RL preview winner.

## 7. Implications for our plan (15 days, one 8×H200, about 11 Kaggle GPU-h a day)

1. **DO keep RL off the critical path, but make stage 1 RL-shaped.**
   - RFT on frozen decision states with 0<p<1, using zero-variance filtering, p(1-p) selection and the shortest-correct target.
   - Collect on the Kaggle W4A16 server. Off-policy, one-step data is Kaggle's only valid role.
2. **DO train against the served weights.**
   - Dequantize the int4 experts into the trainer, frozen. LoRA on attention, GDN and the shared-expert MLP (BF16 modules only).
   - Merge without re-quantization.
   - TEST-FIRST: trainer-vs-SGLang per-token KL on about 200 decisions, plus a merged-LoRA boot with an MTP accept check.
3. **DO add `return_logprob` logging to the state-lab resampler now.** Without behaviour logprobs, IS correction is impossible on any data we collect.
4. **If an RL step is attempted (only after merge→serve works; likely after 11-02):**
   - GSPO or GSPO-token (or CISPO);
   - R3 via ms-swift Megatron + vLLM ≥0.14;
   - `sequence_mask` / `off_policy_sequence_mask_delta`;
   - aux loss 0 (the recipe's 1e-3 contradicts NeMo RL);
   - FP32 LM head;
   - top-p=1/top-k=-1 at a fixed T;
   - rollouts on Azure at trimmed 8-16k contexts, k=8;
   - reward = level clear / unseen board / on-route, with a length term on correct samples only;
   - watch `k3_kl`, entropy and length; plan for a few hundred steps.
5. **DON'T:**
   - plain GRPO on this MoE (3/3 crashes without R3 on Qwen3-30B-A3B);
   - INT4 PTQ rollouts with a BF16-weight trainer;
   - critic-based PPO or VAPO;
   - full-game-from-reset episodes;
   - "frame changed" rewards;
   - untested FP16 training.
6. **Challenge to the prior plan.** Own-sample replay SFT drifts. Prefer a reverse-KL self-distillation anchor to the frozen base, or cap replay at 1 epoch and low LR, and measure the general-ability drop.
7. **Optional cheap upgrade.** One semi-online DPO round on same-state pairs after the stage-1 model is served. It performs about like GRPO at a fraction of the infrastructure.

## Fact-check (independent verifier, 2026-10-18)

Method: each claim was checked against the primary text (arXiv HTML including the math alt-text, the docs and blog pages), the Intel checkpoint's `config.json` and safetensors headers on Hugging Face, and the repo files at the cited lines. No browser was used.

### Claim verdicts
1. **R3 paper (2510.11370v2): PARTIALLY CONFIRMED.**
   - Confirmed:
     - KL 1.535e-3 for Qwen3-30B-A3B vs 6.4e-4 for Qwen3-8B (§3.1).
     - R3 lowers it "from 1.5e-3 to 7.5e-4" (§4.3).
     - Rollout latency overhead is "below 3%" (§4.1).
     - Setup: SGLang rollout, Megatron training, VeRL.
     - All three GRPO runs in Table 1 crashed, at steps 120, 60 and 105.
   - Correction: Table 1 has **six** R3 runs, not four (GRPO+R3 ×3, GSPO+R3 ×1, GRPO+TIS+R3 ×2), and none crashed. The three GRPO crashes come from three different settings (SFT with mini_step 8, SFT with mini_step 1, Base with mini_step 1), one run each. They are not repeats of one configuration.
   - Not in the notes: in Appendix B, Table 3 (Qwen3-30B-A3B-ReasoningSFT), **GSPO without R3 crashed at step 90** (mini_step 4) and GRPO crashed at step 40. Both GRPO+R3 runs stayed stable.
2. **GSPO (2507.18071v2): PARTIALLY CONFIRMED.**
   - Confirmed:
     - Eq. 7 is s_i = (π_θ(y_i|x)/π_old(y_i|x))^(1/|y_i|).
     - Clip ranges are 3e-4 / 4e-4, vs GRPO's 0.2 / 0.27 (§5.1).
     - In the 48-layer model, "roughly 10% of the experts activated" change after each update (§5.3).
     - The paper says GSPO "obviated the need" for Routing Replay.
   - Corrections:
     - The 10% figure is for Qwen3-30B-A3B-Base, with 4 mini-batches per rollout batch.
     - "Converges without Routing Replay" rests on the GSPO paper's own run. Three other sources report GSPO without routing replay collapsing on MoE:
       - R3 paper, App. B: crash at step 90 on Qwen3-30B-A3B.
       - Qwen's own SAPO paper (2511.20347), §5.1: "GSPO and GRPO-R2 ... experience early training collapse" on a Qwen3-30B-A3B-Base cold start.
       - ScaleRL (2510.13786), App. A.17: GSPO runs "diverged mid-training" on multiple occasions. On the Scout MoE, restarting from a stable checkpoint did not fix it.
     - So GSPO is not a safe substitute for R3 on Flash-Next.
3. **ms-swift Flash-Next GRPO recipe: CONFIRMED.**
   - Every listed flag matches: `--max_length 1024`, `--max_completion_length 8192`, `--num_generations 4`, `--global_batch_size 8`, `--vllm_mode colocate`, `--vllm_tensor_parallel_size 8`, LoRA r8/α32 on `in_proj out_proj linear_proj linear_qkv`, `--moe_aux_loss_coeff 1e-3`, and `# 8*135G`.
   - It has no router-replay or rollout-IS flags.
   - Other settings in the recipe: `--vllm_gpu_memory_utilization 0.35`, `--vllm_max_model_len 9216`, `--temperature 1.0`, `--loss_type grpo`, `--beta 0.0`, `--lr 5e-5`.
   - The note about vLLM PR #53896 and a source build >0.28.0 sits in the page's environment setup, so it applies to the whole page.
4. **NeMo RL aux loss: PARTIALLY CONFIRMED.**
   - Confirmed: the quote ("conflicts with the policy objective and severely degrades accuracy") and the 0.001 default.
   - Corrections:
     - The nightly page has no "MoE Router Notes" section. The text is under "100-Step Long-Run Results".
     - It is specific to the **AutoModel (DTensor) backend**, which reads `router_aux_loss_coef` from the HF config. The page says "the shipped Qwen3.5 AutoModel recipes set router_aux_loss_coef: 0.0" (since #3169). It says nothing about the Megatron recipes.
     - The `latest` version of the page doesn't contain this text.
   - Independent support: the R3 paper (§5.1) says "No auxiliary loss for expert balancing is introduced."
5. **LMSYS/slime INT4 QAT: CONFIRMED, with wording fixes.**
   - The quote "errors that oscillate and increase with training steps" is in the "Fake Quantization Ablations" section (the no-QAT, direct-INT4, PTQ-style case), not in the train-infer gap section.
   - The train-infer gap section says INT4 "almost overlaps" the BF16 baseline (not "nearly overlaps") and is "significantly lower than FP8", measured on Qwen3-30B and Qwen3-235B.
   - The blog's QAT fake-quantizes every trained weight (1×32 groups, range [-7,7]) and re-quantizes after each update. It does not test frozen dequantized int4 experts with LoRA on the BF16 modules, so that plan is still our own inference.
6. **Intel checkpoint and server flags: CONFIRMED.**
   - `arms.md:634` says this.
   - Checked independently against the HF `config.json` of `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`:
     - The quantization settings are auto-round, 4-bit, symmetric, group_size 128, `block_name_to_quantize` `model.language_model.layers`.
     - `extra_config` keeps these 16-bit: self_attn (including the QSA indexer), linear_attn (including in_proj_a/b), shared_expert and shared_expert_gate, mlp.gate, hyper_connection, mtp (including its 512 experts), ple, embed_tokens and visual.
   - In the safetensors headers of shards 3 and 4, the routed experts are `qweight` I32 with F16 `scales`. Attention, GDN, shared-expert, router, hyper-connection and PLE tensors are BF16.
   - `launch_sglang.py:37` sets `KVDTYPE="fp8_e4m3"`, and lines 354-355 pass `--dtype bfloat16 --quantization auto-round` and `--kv-cache-dtype CFG["KVDTYPE"]`. `mem-repair-10x` doesn't override KVDTYPE.
7. **Thinking Machines: CONFIRMED, with limits on scope.**
   - Confirmed quotes:
     - "LoRA performs equivalently to FullFT for reinforcement learning even with small ranks" (down to rank 1).
     - "Attention-only LoRA significantly underperforms MLP-only LoRA" (a figure caption). It holds for Llama-3.1-8B and Qwen3-30B-A3B-Base, and also for RL on MATH.
   - Scope:
     - The RL evidence is single-turn math on dense models: Llama-3.1-8B on MATH, and Qwen3-8B-Base on DeepMath at 8k tokens.
     - For the MoE model, "MLP" includes the routed experts, so LoRA on the shared expert alone is untested.
   - Self-distillation: the caption "Running SFT on top of Qwen3-32B's own samples degrades performance" is confirmed. The samples were unfiltered, at T=1.0, on Tulu3 prompts (KL=0 by construction). The finite-batch drift is offered only as "a possible explanation".
8. **RAFT (2504.11343): PARTIALLY CONFIRMED.**
   - The abstract wording holds.
   - The body qualifies it:
     - Average accuracy on Qwen2.5-Math-7B: vanilla RAFT 52.3, PPO 52.5, GRPO 56.3.
     - Only RAFT++, which adds importance sampling and clipping, reaches 56.1.
     - RAFT++ learns faster early, then its entropy collapses and it is "surpassed by GRPO in the later stage".
   - RAFT here is online and iterative: it re-samples from the current policy every iteration. It was tested on dense 7B/3B math models with 4k-token generations. It is not evidence for a single round of offline RFT from a frozen policy.
   - It does support RFT inside a short budget, since the advantage is in the first 100-200 iterations.

### Other errors found while skimming
- **§1 Takeaway and Inferences, §7 item 4 (GSPO as the default):** the recommendation leaves out the three GSPO-without-replay collapses listed under claim 2. R3 should be treated as required whichever loss is used.
- **§3 line 148 and line 160 (NeMo thresholds):** the page says "The nightly thresholds (0.004 / 0.003 on the 20-step mean)", right after the 35B (~0.0024) and 9B (~0.0014) figures. Both are 20-step-mean gates, most likely 35B / 9B, not "0.003 mean / 0.004 max". The matching bar for the 35B MoE is about ≤0.004. It was measured with FP8 weight rollouts and a BF16 KV cache, so it is only a loose analogue for W4A16 with an FP8 KV cache.
- **§4 line 182 (IF-eval):** the 45 belongs to the 100%-document mid-train. Distillation was shown only on the 70% mix (79 → 83), not on the model at 45.
- **§4 line 179 (DigiRL):** the 57.8% is "the prior best autonomous RL approach based on filtered behavior cloning". That is an online filtered-BC baseline, not filtered BC on static demonstrations.
- **§4 line 196 and §7 item 2 (shared-expert LoRA):** Thinking Machines' MLP-only result on the MoE included the routed experts. LoRA on the shared expert alone (1 of 11 active FFNs per token) is untested.
- **§5 line 214 (ScaleRL compute):** 16k GPU-h is the cost of one leave-one-out run. The full 8B ScaleRL run used 100k GPU-h. The Scout figure of 50k is correct.
- **§5 line 215 (DeepSeek-V3.2):** the paper says the post-training budget exceeds 10% of the pre-training cost. That covers all of post-training, not RL alone.
- **§1 line 21 (GSPO model):** the model is Qwen3-30B-A3B-Base.
- **Minor quote drift:**
  - Line 89: ms-swift says "R3 combined with TIS may not provide additional gains".
  - Line 90: NeMo says "Other inference/generation backends are not wired into this path and have not been tested with Router Replay."
- **§2 line 91 (Miles):** `--use-rollout-routing-replay` is confirmed in Miles' code (`miles/utils/arguments.py`). slime has the same flag, so SGLang-side R3 isn't unique to Miles.

### Spot-checked and fine
- **R3 paper:** §3.2 10% / 94%; §3.3 8.4e-4; Table 2 (31.8 vs 38.6, crash at step 90).
- **GSPO:** §5.2 and §5.4.
- **CISPO:** 2x speedup / 50% of the steps on Qwen2.5-32B.
- **MiniMax-M1:** $534,700 on 512 H800 in 3 weeks; FP32 head 0.9x → 0.99x; truncation after 3,000 consecutive tokens above 0.99.
- **ScaleRL:** the recipe, FP32 head 0.52 → 0.61, prompts with pass rate ≥0.9 dropped, k=8.
- **GiGPO:** all 16 numbers in Table 1, <0.002%, >65%, ω=1, γ=0.95, 2048/4096-token prompts.
- **DeepSeek-V3.2:** Keep Routing since V3-0324; 1,827 environments and 4,417 tasks.
- **Abstracts:** QUADS, KV4-RL, FP8-RL, 2609.22870, 2602.01826, 2510.26788, 2506.21495, VAPO, Fatemi, ShorterBetter, LILO, VinePPO, Dr. GRPO and the entropy paper.
- **Yao et al.:** Qwen2.5-32B in Fig. 1; 0.5B with INT8 on GSM8K.
- **ms-swift mismatch flags:** threshold default 2.
- **Hugging Face:** lists only Flash-Next and -FP8, both dated 2026-08-24.
- **Repo:**
  - `tool_agent.py:1015-1017` defaults.
  - `arms.json:1232` sets T=0.65.
  - No `SGLANG_ARC_MOE_ROUTING` in `mem-repair-10x`.
  - Zero `logprob` hits.
