# G-E (gap G11): R1, a self-distilled MTP draft head, and the mandatory MTP check after a trunk change

Researcher G-E, 2026-10-18. Scope: the R1 lossless-throughput lever (retrain Flash-Next's own MTP layer on its own ARC transcripts), how it fits our serving stack, the local accept-length → tok/s slope, and the exact MTP gate for any trunk LoRA. Local script: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/ce155f4d-514a-4385-ac65-a7e7e5a23e1c/scratchpad/mtp_slope.py` (run with `python3 -I mtp_slope.py runs/kaggle <run/ver>...`). In it, "clean" means a decode window with no Prefill line since the previous Decode line, the same rule as `kaggle/tput_rank.py` on tag `archive/arm/tput-levers`. Step time is 1000·bs·accept/tps, in ms per verify iteration (`kaggle/serve_report.py:68-72`).

## Q1. What does published work measure for self-distilled or in-domain draft-head training?

### Takeaway
Retraining a draft head on the target's own outputs reliably helps when the head started out weak or off-domain. The gain is much smaller for a head that was already trained for multiple steps. Our closest analogue is Qwen3-Next's native MTP, retrained in-domain on GSM8K: accept length rose about +0.21 (+7%). Botched head-only retrains do regress. The reported failures used hard labels, a high LR, little data and a mismatched sampling temperature.

### Cited Findings
- **EAGLE-3** has the target model regenerate the responses ("We call the target model to generate responses rather than using a fixed dataset"). It trains on ShareGPT (~68K) plus UltraChat-200K (~464K), and on OpenThoughts-114k-math for the R1-distill target. Mean τ on LLaMA-3.1-8B is 4.11 for EAGLE-2 vs 6.23 for EAGLE-3 at T=0, and 3.62 vs 4.92 at T=1. Training-time test unrolls simulated draft steps, and acceptance keeps rising with data (Fig. 1). SGLang throughput is 1.38× at batch 64. — [arXiv 2503.01840 HTML](https://arxiv.org/html/2503.01840v3)
- **HASS** adds a top-K distillation loss (K=10, w=1.0; its ablation peaks at K=5, w=0.5) and 3-step context alignment (5 steps is worse). Training costs ~66% more time than EAGLE-2. On LLaMA3-8B, τ is 4.52 → 5.08 at T=0 and 4.31 → 4.80 at T=1. **Self-distilled data (target-generated) raises HASS τ from 5.15 to 5.51 (+7%)** on LLaMA2-Chat-7B at T=0 (appendix). — [arXiv 2408.15766 HTML](https://arxiv.org/html/2408.15766v3)
- **Medusa:** self-distillation is needed when the training data does not match the model's output distribution (e.g. RLHF'd models). A self-generated set "is sufficient for training Medusa heads"; Medusa-2's backbone uses KL to the original model, because hard labels degrade it. — [arXiv 2401.10774](https://arxiv.org/html/2401.10774v3)
- **FastMTP** (MiMo-7B-RL, frozen trunk, one shared MTP head of 210.8M params) trains on 389.4K self-distilled samples generated at T=0.6 / top-k 20 / top-p 0.95. The setup: K=3 recursive steps, CE weighted by β^(k−1) with β=0.6, LR 5e-5, 3 epochs, <1 day on one H20 server. Per-position acceptance goes from ~70/10/0% to 80/56/36%, and τ from 1.83 (vanilla MTP) to 2.73. FR-style 32k vocab gives τ 2.66 but a larger speedup (2.03× vs 1.81×). — [arXiv 2509.18362](https://arxiv.org/html/2509.18362v1). The vanilla head was trained only for next-token use, so this gain overstates what an already multi-step head can gain.
- **Qwen3-Next-80B-A3B with vLLM Speculators 0.6.0** (FastMTP recipe; ~8,000 self-generated GSM8K samples, 3 steps, β=0.6, LR 1e-4, 3 epochs, seq 8,192): per-position acceptance goes from 0.897/0.719/0.476 to 0.912/0.776/0.616, with "up to 1.25×" lower median ITL. Summed, that is **3.09 → 3.30 accept length (+0.21)**. The Qwen3.5-9B GSM8K example took 443 s on 2×H200. The article reports no regressions and notes it may be undertrained. — [Red Hat, 2026-09-08](https://developers.redhat.com/articles/2026/09/08/optimize-vllm-speculative-decoding-fastmtp-heads)
- Flash-Next's card lists "MTP: 1 layer, trained with multi-steps" (W4-lab-pipelines.md:21, [HF card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)), so Qwen3-Next's native MTP is the right analogue, not FastMTP's vanilla head.
- **verl** recommends MTP-only training with the trunk frozen (`detach_encoder=True`). It reports that training the MTP during RL raises rollout acceptance by "around 14%". — [verl MTP guide](https://verl.readthedocs.io/en/latest/advance/mtp.html)
- **Regression, Holodeck-Lounge (Qwen3.5-9B merge):** a head-only retrain of a ~300M head on 5,000 greedy self-generated samples, with hard labels, LR 2e-4, seq 1,536 and 1 epoch (speculators `train.py`, one RTX 5090). Acceptance fell 58.55% → 49.75% under vLLM and mean accepted length 2.756 → 2.493; position 2 fell 41.4% → 29.5%. The eval used only 30 prompts. The author blames the high LR, hard labels instead of teacher distributions, and corpus shift. — [HF card](https://huggingface.co/Wiself/Holodeck-Lounge-Regressed-MTP)
- A community user reports that a head trained on greedy outputs reached 0.88 acceptance greedy but 0.22 at T=0.7. UNVERIFIED (user report). — [TENSELERATE #56](https://github.com/mintoriakamoto/TENSELERATE/issues/56)
- **Correction to the prior notes:** `finetune-research-2026-10-17/training.md:55` cites AngelSpec as "52.8% → 66.4%". Its abstract reports only "~30%" higher accepted length on Hy3-A21B and does not say the drafter was trained on target rollouts. Treat 52.8 → 66.4 as UNVERIFIED. — [arXiv 2607.25852](https://arxiv.org/abs/2607.25852)
- SpecForge PR #983 warns that an anchor/label off-by-one "once cost a DeepSeek-V4 DSpark fine-tune its serving acceptance". — [PR #983](https://github.com/sgl-project/SpecForge/pull/983)

### Inferences
- A realistic R1 target for our already-trained head is **+0.15 to +0.25** accept length (2.68 → ~2.85–2.95). The prior plan's 3.0–3.2 (`research/reports/rsi-proposals-2026-10-15.md:96`) is the optimistic tail, not the central case. Our domain shift from Qwen's post-training mix (long ARC reasoning, grid digits, tool calls) is larger than GSM8K's, which argues for the upper half of that range. Our much longer contexts (median 56.6k) argue against.
- Recipe lessons from the failures: use teacher distributions (top-K/KL) or many samples at the serving temperature; use a low LR (≤5e-5 for an already-trained head); train all 3 chained steps; evaluate on ≥ hundreds of held-out prompts.

### Gaps
- No published retrain of Flash-Next's own MTP, and no ARC-domain number.
- No per-position acceptance for our stack: SGLang logs only the mean accept length and rate.

## Q2. What is our measured accept-length → tok/s slope at 10–14 streams?

### Takeaway
At fixed SPEC_STEPS=3, the time per verify step does not depend on acceptance. **Clean decode tok/s is proportional to accept length (elasticity ≈ 0.95–1.0). Harness tok/s rises about 0.6× as much (+2.25% per +0.1 accept)** because prefill and gaps don't speed up. The cleanest natural experiment is relaxed acceptance (accept05): same steps, same drafter, same verify cost.

### Cited Findings (mtp_slope.py on `runs/kaggle/*/serve.log.xz`)
| Run | Accept (time-weighted) | Clean bs10 tok/s (accept) | Clean bs14 tok/s (accept) | Step ms bs10 / bs14 | Harness tok/s |
|---|---|---|---|---|---|
| v4-tput-panel-base/v1 | 2.643 | 820.0 (2.631) | — | 32.2 / — | 606.7 |
| v4-tput-panel-accept05/v1 | 3.061 | 944.8 (3.026) | — | 31.9 / — | 643.5 |
| v4-tput-panel-spec2/v1 (2 steps) | 2.294 | 784.5 (2.262) | — | 28.8 / — | 555.7 |
| v4-user-combo-10x v1 / v2 / v3 | 2.702 / 2.696 / 2.671 | 856 / 852 / 853 | 1008.7 / 1012.1 / 999.4 (2.664 / 2.668 / 2.645) | 31.5–31.7 / 36.8–37.0 | 698.1 / 698.1 / 703.4 |
| v4-user-combo-10x-accept05/v1 | 3.068 | 968.7 (3.101) | 1145.5 (3.044) | 32.0 / 37.3 | 759.5 |
| v4-mem-repair-10x/v1 (leader) | 2.684 | 852.7 | 1001.4 (2.649) | 31.7 / 37.0 | 687.9 |
| v4-control/v1 (10 streams) | 2.694 | 823.1 | — | 32.3 / — | 602.0 |

- **Panel, bs10:** accept ×1.150 (2.631 → 3.026) gives tok/s ×1.152 (820.0 → 944.8), with step time unchanged (32.2 → 31.9 ms).
- **Full runs, bs14:** accept ×1.145 (2.659 → 3.044, mean of the three user-combo runs) gives tok/s ×1.138 (1006.7 → 1145.5), with step time +1% (36.9 → 37.3 ms).
- **Harness tok/s:** 699.9 → 759.5 (+8.5%) for a time-weighted accept of 2.690 → 3.068 (+14.1%), so **+2.25% harness tok/s per +0.1 accept**. Time-weighted decode tok/s rose 720 → 783 (+8.7%).
- Within-run OLS on clean bs≥10 windows gives d ln(tps)/d accept = 0.29–0.31 (+2.9–3.2% per +0.1), below the proportional 0.37 because high-acceptance windows also step slightly slower (+2–3 ms per unit), a content confound. Use the cross-run figure.
- **Draft cost:** spec2 vs base at bs10 is 28.8 vs 32.2 ms, so one MTP draft step costs ≤3.4 ms, about 10% of a verify iteration. The 3 chained draft steps are ~30% of decode time. A better drafter at the same cost is the lever; a cheaper drafter is not on offer.
- **Ceiling:** the QSA cap keeps a verify block at 4 tokens (`serving/CUSTOM.md:24`, `arms.md:216`; launcher `--speculative-num-draft-tokens = SPEC_STEPS+1`, `serving/kaggle/launch_sglang.py:379-380`), so accept length can't exceed 4.0. With an iid per-token acceptance α, L = 1+α+α²+α³: 2.68 ↔ α 0.737, 2.88 ↔ 0.784, 3.0 ↔ 0.811, 3.2 ↔ 0.853. Greedy (T=0) on the current head gives 2.81 (`arms.md:292`), α ≈ 0.768.
- **Acceptance rule:** we serve `speculative_use_rejection_sampling: False`, thresholds 1.0/1.0, topk 1 (mem-repair serve.log server_args), which is SGLang's target-only verification. A chained draft token is accepted iff `coin < prob_acc / threshold_acc || target_prob_single >= threshold_single`, i.e. with probability p_target(draft) at the serving temperature. On rejection the server samples from relu(q − p) with the rejected token's mass removed. — [SGLang `sgl_kernel/speculative/sampling.cuh`](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/jit/include/sgl_kernel/speculative/sampling.cuh) lines 64-100 (upstream main; the pinned Pennyroyal copy is assumed identical, UNVERIFIED). The serving temperature is 0.65 (`kaggle/arms.json:1232`).

### Inferences
- With target-only verification, the best deterministic draft is the target's argmax. Acceptance is then capped by E[max p_T] at T=0.65, which no drafter can beat. Training should be mode-seeking: CE on the target's own samples at T=0.65, or top-K/KL to target logits. Greedy-only data is wrong for our serving temperature, as the TENSELERATE report shows.
- The accept05 run lost score (46.77 vs 56.94/52.70, `arms.md:362`) because relaxed acceptance changes which tokens are sampled. R1 is lossless, so its tokens are the same tokens, faster.

### Gaps
- E[max p_T] on our transcripts (the ceiling) is unmeasured. It needs top-1 logprobs from a prefill, which our logs don't hold.

## Q3. Can R1 train on one 96 GB or 141 GB GPU, and how does it stay compatible with albucino and Pennyroyal?

### Takeaway
Yes. With the trunk frozen and features cached, training is small. **Feature capture is the real cost and the main plumbing risk:** the MTP reads the 10,240-wide hyper-connection stream, and no ready capture hook exists for Qwen4Exp. Train only the MTP's own BF16 dense tensors (≈90M params) and keep albucino's INT4 g32 experts byte-identical. The new head is then a drop-in `mtp-dense.safetensors` that passes every launcher check.

### Cited Findings
- **Albucino drafter layout** (my parse of the `mtp-dense.safetensors` header, 2,723,937,440 B): 31 BF16 tensors, 1.362B params. `lm_head` and `embed_tokens` are 635.7M copies each (`mtp_use_dedicated_embeddings = False`). The rest is the MTP's own **≈90.6M**: `self_attn` 51.4M (q_proj [12288, 2560], o_proj [2560, 6144]), hyper-connection mixers 19.8M (`input_mix_weight_down` [320, 10240]), `fc_embedding` and `fc_hidden` 6.6M each, `shared_expert` 4.9M, router 1.3M, norms. Routed experts: 512 × 3 × 2560 × 640 = 2.52B params in 1,416,156,712 B of INT4 sym g32. `compact_sources.json`: "MTP routed experts locally RTN-quantized to INT4 symmetric group-32 with min/max scales; other MTP tensors unchanged", from RadixArk/Qwen3.8-Flash-Next-NVFP4 @7b71922. — [albucino repo](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE/tree/main/runtime/mtp-int4-g32); config: hidden 2560, `hc_count` 4, `hc_lowrank` 320, vocab 248,320, mtp `layer_types ['full_attention']` ([config.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/config.json)).
- **Launcher checks.** `prepare_draft_view` (`serving/kaggle/launch_sglang.py:101-155`) requires a compressed-tensors `mtp_routed_experts` group of symmetric INT4 g32, the expert regex `^mtp\.layers\.0\.mlp\.experts\...` and the dense-ignore regex (lines 51-52). A dense-only retrain that keeps the 31 names, shapes and BF16 dtype changes nothing these checks see. The Kaggle drafter is byte-identical to albucino's (`arms.md:666`).
- **What the MTP reads.** For Qwen4Exp, SGLang sets `output.hidden_states = hc_hidden_states`, the HC-flattened 4 × 2560 = **10,240-wide** last-layer stream. Aux-hidden-state capture is broken: the `layer_communicator` was deleted and the return slot collides. So "no DFlash or DSpark drafter can be served". — [SGLang #38589](https://github.com/sgl-project/sglang/issues/38589) (open). EAGLE-3 needs the same aux-layer capture, so I infer it is also unservable on this target. UNVERIFIED for EAGLE-3 specifically.
- **SpecForge PR #983** (open, 2026-10-16) adds a MoE DSpark drafter with MTP warm start for Flash-Next.
  - It says SGLang "has no hidden-state capture hook" for qwen4_exp, so it adds an HF backend (Transformers ≥5.16.1, BF16 reference, PLE table kept in host RAM) and a vLLM backend that needs a fork.
  - The HF backend cannot load the W8A8 checkpoint.
  - Serving a DSpark drafter on SGLang is listed as a follow-up.
  - Source: [PR #983](https://github.com/sgl-project/SpecForge/pull/983).
- The pinned build has `enable_return_hidden_states` (False in mem-repair serve.log server_args). Upstream it is a server flag ([sglang search](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/arg_groups/fields/exec_.py)). Whether it returns the HC stream for every prompt position at usable speed is UNVERIFIED; upstream converts the result to JSON lists.
- **ms-swift.** `--mtp_num_layers 1 --mtp_loss_scaling_factor 0.1` trains the checkpoint's MTP (`examples/models/qwen4_exp/megatron_sft.sh:4-5`, header "# 8*70G"). The relevant options:
  - `freeze_parameters_ratio 1` plus `trainable_parameters` / `trainable_parameters_regex` makes a head-only run possible.
  - `mtp_decoder_input_detach`.
  - `mtp_shared_weights` ("`--mtp_num_layers 3 --mtp_shared_weights true`", the GLM-5 scheme). This would mirror our 3-step chained use of one layer.
  - Source: [Megatron-SWIFT params](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Megatron-SWIFT/Command-line-parameters.md) lines 248-270.
  - The mcore-bridge MTP support is #204 (W6 fact-check row 3).
  - Whether `mtp_shared_weights` exports back to a single `mtp.layers.0` for qwen4_exp is UNVERIFIED.
  - ms-swift runs the **BF16** trunk on 8 GPUs and is measured only to 8k context (corrected fact).
- **Leader-run token budget** (mem-repair-10x serve.log `ReqTimeStats`, my parse): 2,636 requests, 146.0M input tokens, 130.2M cached (89.1%), **15.85M fresh prefill**, 5.29M output, median input 56,643. Prefill chunks of ≥8k tokens ran at a median **12.3k tok/s** (p25 11.1k, p75 13.0k; 1,224 lines).

### Inferences
- **Memory (dense-only, features cached):** 90.6M trainable with AdamW ≈1.5 GB; frozen experts 1.4 GB INT4 (dequantized per forward) or 5 GB BF16; the 64K FR-Spec `lm_head` slice 0.33 GB; activations for a 16k window × 3 steps ≈1 GB of HC stream plus 12.6 GB of fp32 logits unless the CE is chunked. Total <40 GB, so it fits a 96 GB RTX PRO 6000 easily. A full-MTP retrain (2.6B trainable, ≈42 GB of optimizer state) fits 96 GB tightly.
- **Don't retrain the experts.** A fine-tune's weight deltas at LR ≤5e-5 are far smaller than an INT4 g32 step, so RTN would mostly erase or randomise them. Experts would also need a new INT4 pack, which brings format risk.
- **Capture plan (preferred):**
  - Replay logged conversations in order on our own W4A16 Pennyroyal server with prefix caching and spec off, through a `serving/runtime_patches/` hook that `torch.save`s the 10,240-wide HC stream. These are the exact serving features.
  - One leader run is ≈21M unique positions (15.85M + 5.29M), so ≈29 min of prefill at 12.3k tok/s.
  - Storage is the constraint. At 10,240 × BF16 that is ≈420 GB per run, and ≈53 GB at 2,560 × FP8 if the MTP's input mixer is frozen and its 2,560-wide output is cached instead. The 2,560-wide output is inferred from `fc_hidden` [2560, 2560], UNVERIFIED.
  - So keep only a 4k-token window ending at each reply: ≈10.5M positions, ≈26 GB per run at FP8/2560. The trade-off is that the MTP's own attention then sees ≤4k tokens in training instead of ~57k at serving (risk, see Q4).
- **Fallback:** ms-swift Megatron on the Azure 8×H200 node (head-only regex, `--mtp_num_layers 3 --mtp_shared_weights true`). Simpler to plumb, but BF16 features at ≤8k context vs W4A16 at ~57k in serving, at $84.80/node-h.
- **Data:** 10x-family runs are sampled at T=0.65, the serving temperature; one leader run holds ≈5.3M generated tokens (FastMTP used 389K samples, Qwen3-Next gained from 8K). 3–5 runs is ample. Hold out ~5 public games to measure cross-game transfer.

### Gaps
- Kaggle session disk and host-RAM limits for an in-session capture-then-train run are UNVERIFIED, so a Kaggle-only R1 is uncertain. Azure RTX PRO 6000 spot (~$1/h) or on-demand (~$5.50/h) is the safe host.
- The MTP's attention in serving is QSA ("QSA MTP index sharing enabled… layers [0]", mem-repair serve.log). Training with dense attention is a small mismatch, UNVERIFIED.

## Q4. Expected gain, cost, recipe and risks

### Takeaway
Central case: +0.2 accept, which is +7% clean decode, +4.5% harness tok/s, about **+2 offline points and +1 to +2 LB**. It costs ≈2–4 engineering days and <15 GPU-h. It is lossless and stacks multiplicatively with every other lever. On its own it is far from LB 60.

### Cited Findings
- sim2 speed-scale s (decode speed) is from L3-headroom.md:114-128, confirmed by its fact-check row 5. For 25 games: s 1.0 → 50.93, 1.15 → 55.08. For 110 games: s 0.6 → 30.95, 0.7 → 35.49. P1 (LB ≈ 0.55 × offline) is the floor and P3 (hidden ≈ 0.6× compute) the ceiling (L3-headroom.md:146-157). With s = 1 + 0.95·Δaccept/2.68, linear interpolation gives:

| Δ accept | s | Offline | LB (P1) | LB (P3) |
|---|---|---|---|---|
| −0.25 (botched head) | 0.911 | −3.1 | −1.7 | −3.1 |
| +0.05 | 1.018 | +0.5 | +0.3 | +0.5 |
| +0.15 | 1.053 | +1.5 | +0.8 | +1.5 |
| **+0.20 (Qwen3-Next analogue)** | 1.071 | **+2.0** | **+1.1** | **+1.9** |
| +0.40 (= accept05 speed) | 1.142 | +3.9 | +2.2 | +3.9 |

- sim2 over-predicted the 27B by ~10 points (L3-headroom.md:132,168).

### Inferences: recipe (R1 v1)
1. **Data:** 3–5 recent 10x-family runs (T=0.65), re-rendered with the chat template (`preserve_thinking`); history is normalised (VERBATIM_HISTORY off), which is close enough for a drafter. Hold out 5 games.
2. **Capture:** Pennyroyal W4A16, spec off, prefix cache on, a runtime-patch dump of the HC stream (or the frozen-mixer 2,560 state) in FP8, keeping a 4k window ending at each reply. First validate that the albucino head reproduces its served acceptance on captured rows.
3. **Train:**
   - Initialise from albucino. Train only the 90.6M dense tensors; freeze `lm_head`, `embed_tokens` and the experts.
   - Unroll 3 chained steps with shared weights. Each step's input is the previous step's output stream plus the next ground-truth token's embedding (FastMTP/HASS).
   - Step weights β = 0.6. Loss: CE on the sampled token plus top-K KL to target logits. The target logits come from `lm_head` on the cached trunk state at t+1, so no extra capture is needed.
   - Restrict to the 64K FR-Spec vocab; that matches serving and is 4× cheaper.
   - LR 1e-5 to 5e-5 with cosine decay, 1–3 epochs.
   - Unit-test the anchor alignment: the label for draft step k at position t is token t+1+k.
4. **Offline select:** on held-out features, per-depth top-1 agreement with the target argmax and the expected accept E[L] = 1 + Σk Πj≤k p_T(d_j). Ship only if it beats albucino by ≥0.1 on the held-out games.
5. **Export:**
   - Rewrite `mtp-dense.safetensors`, keeping the 31 names, shapes and BF16 dtype.
   - Keep the index and the expert file byte-identical.
   - Upload as a new Kaggle model version, then gate per Q5.

### Inferences: cost
- Capture is ≈0.5 GPU-h per run, ≈2 GPU-h for 4 runs. Training is ≈2–8 GPU-h.
- On Azure RTX PRO 6000 that is <$15 at spot and <$80 on-demand. Kaggle validation adds a 25-min panel plus one 121-min run, ≈3 GPU-h of quota.
- Engineering is 2–4 days, mostly the capture hook and the exporter.

### Inferences: risks
- No SGLang capture hook (PR #983, #38589); train/serve mismatch (truncated MTP attention context, QSA vs dense attention, re-rendered prompts); hot-LR or hard-label regression (Holodeck); off-by-one alignment (PR #983); public-to-private transfer (measure on held-out games). **Any trunk change invalidates R1 v1,** because the features move.

### Gaps
- No per-game drift measurement of acceptance (also W10).

## Q5. The exact MTP acceptance check to run after any trunk LoRA

### Takeaway
A trunk LoRA leaves the MTP weights alone but moves the 10,240-wide state the MTP reads, and it may change the target's entropy. Under target-only verification, the second effect can raise or lower acceptance. So gate the joint effect on the real load, with the base trunk measured on the same load as the reference.

### Cited Findings
- Merge and export can silently drop or omit MTP tensors:
  - Transformers drops MTP weights on a LoRA merge-export ([ms-swift #9802](https://github.com/modelscope/ms-swift/issues/9802), Qwen3.6).
  - Derivative checkpoints missing `mtp.fc.weight` "draft from a random projection: no exception, no warning" under vLLM's quantized-load path ([FriendliAI model card](https://friendli.ai/models/PassingByPixels/Qwen3.6-27B-Architect-Polaris2-Fable-B-F451-NVFP4-MTP); vendor card, UNVERIFIED).
  - After an MTP-enabled SFT, a Qwen3.5-35B scored 94% with MTP off but 74% with it on, which points to a serving bug ([ms-swift #9283](https://github.com/modelscope/ms-swift/issues/9283), via W6).
- Our launcher loads the drafter from `DRAFT_MODEL_DIR`, separate from the target (`launch_sglang.py:379-385`). A trunk merge can't drop it, but it can mismatch it.

### Inferences: the check (G-MTP), in order
0. **Integrity, before boot:**
   - The SHA-256 of `mtp-dense.safetensors` and `mtp-routed-experts-int4.safetensors` equals the intended drafter's.
   - `prepare_draft_view` passes.
   - The serve log shows no missing or unexpected weight keys and `speculative_num_draft_tokens 4`.
1. **Losslessness (T=0):**
   - 64 held-out-game decision prompts (real 30–100k contexts), `max_tokens` 1024, spec on vs off, at bs 1 and bs 14, run identically on the base trunk.
   - Record identical-output rate and median first-divergence position.
   - Gate: the candidate's identity rate ≥ the base's − 5 pp, and no prompt diverges in its first 32 tokens unless it does so on the base too. The base's own rate is the numerics floor; it is unmeasured today.
   - This catches ms-swift-#9283-type bugs.
2. **Acceptance and speed (T=0.65, the harness's sampling settings):**
   - A 25-min throughput panel (the `tput-panel-base` design) on the candidate, with the base panel from the same day as the reference.
   - Use `serve_report.py` / `tput_rank.py`. Read the time-weighted accept length, the clean bs10 median tok/s (SE ≈1%) and the step time.
   - Gate: accept ≥ base − 0.05 (base 2.643 on this panel); clean bs10 ≥ 0.98 × base (≥ ~804 against 820.0); step time within ±1 ms of 32.2 ms. A step-time change means something other than the drafter changed.
3. **If accept drops by >0.05:** run R1 on the candidate trunk's own captured features (≈1 day). Re-run step 2 with the retrained head, and ship whichever head passes. If neither passes, the trunk's score gain must exceed ~0.3 offline points per −0.01 accept (Q4 slope: −0.1 accept ≈ −1 offline point).
4. **Real-run confirmation:** in the 121-min run's serve.log, the time-weighted accept must be ≥2.63 and the clean bs14 median ≥ ~980 tok/s (leader 2.684 / 1001.4 / 37.0 ms; harness 687.9).
5. **Diagnostics:** log the trunk's mean top-1 probability on the step-2 load, to split an acceptance change into drafter mismatch vs target entropy.

### Gaps
- How far acceptance falls for a low-rank non-expert LoRA on Flash-Next is unknown. No public report exists.

## Implications for our plan

### Takeaway
R1 is a small, safe and cheap multiplier. Expect about +1 to +2 LB, not +4. Its main value is that the same pipeline is mandatory after any trunk change, so build it once and early.

### Inferences
1. **DO** build the capture hook and the dense-only exporter now, while the trunk work is still in progress:
   - First test the albucino head against itself: the exported unchanged weights must be byte-identical, and accept must equal 2.68 on a panel.
   - Then train R1 on the current trunk. If the trunk ships first, train R1 on the shipped trunk instead.
   - The pipeline is the G-MTP gate's step 3.
2. **DO** keep the albucino INT4 experts frozen and byte-identical, and train only the ≈90.6M dense MTP tensors. A retrain then needs no new quantization and passes `prepare_draft_view` unchanged.
3. **DO** re-plan the expected gain. Use +0.15 to +0.25 accept (Qwen3-Next analogue +0.21; HASS self-distill +7% τ) → +2.25% harness tok/s per +0.1, ≈ +1.5 to +2.5 offline points. The 3.0–3.2 target is now the upside case.
4. **DO** run G-MTP (Q5) on every trunk candidate before any 121-min run. The gate from W6/L3 (≥2.6 accept, within 5% of ~690 tok/s) is too loose: a 0.1 accept loss is ~1 offline point. Tighten it to −0.05 accept and −2% clean decode.
5. **DON'T** pursue EAGLE-3, DFlash or DSpark drafters, deeper speculation (the QSA cap of 4 tokens), relaxed acceptance (accept05 lost 6–10 points), or expert retraining for the drafter before 11-02.
6. **CHEAP SIDE CHECK:** count how many generated tokens fall outside the 64K FR-Spec hot map (`pennyroyal-v253/hot_tokens_64k.pt`) on our corpus. If >1–2%, an ARC-derived hot map is a training-free acceptance lever. FR-Spec off cost −5.4% decode (`arms.md:362`). Its gain is UNVERIFIED.

## Fact-check (independent verifier, 2026-10-18)

Method: I re-parsed all nine `serve.log.xz` files and the run logs with my own stdlib parser (`fc/check.py`, `fc/reqs.py` in the session scratchpad, under `ge-fc/`). It uses the same clean-window rule and the 40-iteration window, but none of `mtp_slope.py`'s code. I ran `mtp_slope.py` only to read its OLS lines. I fetched the primary web sources with curl, trafilatura and `gh`, and read the albucino safetensors headers with HTTP range requests. I recomputed the sim2 interpolation from the L3 tables. All 12 key results were checked; none were skipped.

| # | Claim | Verdict | Correction / note |
|---|---|---|---|
| 1 | Cross-run slope: panel 820.0 @2.631 → 944.8 @3.026, 32.2 → 31.9 ms; bs14 1008.7 / 1012.1 / 999.4 @2.664 / 2.668 / 2.645 → 1145.5 @3.044, 36.9 → 37.3 ms | **Confirmed** | Every number reproduces exactly. The ratios are ×1.152 vs ×1.150 (panel) and ×1.138 vs ×1.145 (bs14). Per-run bs14 steps are 36.9 / 36.8 / 37.0 ms. All runs have `speculative_num_steps 3`, `num_draft_tokens 4` and the same draft-view hash. The accept05 runs use thresholds 0.5 / 0.5. |
| 2 | Harness 699.9 → 759.52 (+8.5%) for accept_tw 2.690 → 3.068 (+14.1%); +2.25% per +0.1; elasticity ~0.6; within-run OLS 0.29–0.31 | **Confirmed** | 698.07 / 698.1 / 703.43 (mean 699.87) and 759.52 are the last "job wallclock" lines. accept_tw is 2.702 / 2.696 / 2.671 → 3.068, which gives +2.25% per 0.1 and ln-elasticity 0.62. The OLS range 0.286–0.313 holds for the base-acceptance runs: user-combo ×3, mem-repair 0.306, panel-base 0.295, control 0.286. The accept05 runs give 0.243 / 0.235. Caveat: accept05 is lossy, so its harness tok/s also reflects different trajectories. That is one more reason to treat the 0.6 harness elasticity as approximate. |
| 3 | One draft step ≤3.4 ms of 32.2 ms (spec2 28.8 ms @2.262); 3 chained steps ~30% of decode | **Partially confirmed** | 28.8 vs 32.2 ms, 2.262 and spec2's `num_steps 2 / num_draft_tokens 3` all reproduce. But dropping a step also removes one verified token per request (30 vs 40 tokens in the bs10 target pass), so the 3.4 ms is an upper bound on one draft step. "3 chained steps are ~30% of decode time" should read **≤~32%, an upper bound, not a measurement**. It also leaves out the per-iteration draft-extend pass. |
| 4 | Leader: accept_tw 2.684, clean bs14 1001.4 @37.0 ms, harness 687.91; 2,636 req, 146.0M in, 130.2M cached (89.1%), 15.85M fresh, 5.29M out, median input 56,643; ≥8k prefill median 12.3k tok/s | **Confirmed** | All numbers reproduce. There are 2,639 ReqTimeStats lines, and 2,636 after `serve_report.py`'s input_len > 1000 warmup filter. Output is 5.288M. The ≥8k prefill lines (n = 1,224) have median 12,327, p25 11,131 and p75 13,043. |
| 5 | Target-only verification (rejection sampling False, 1.0/1.0, topk 1); accept iff coin < p_target(draft) at T = 0.65; argmax-optimal; cap E[max p_T]; ≤4.0 | **Confirmed** (one nuance) | The server_args match. `sampling.cuh` lines 64–100 on upstream main are `TreeSpeculativeSamplingTargetOnly`: with topk 1, `prob_acc` resets after each accept, so P(accept) = p_target(draft). `arms.json:1232` is mem-repair-10x's `LOCAL_ANALYZER_TEMPERATURE 0.65`. CUSTOM.md:24 and launch_sglang.py:379–385 match. Nuance: the harness also sends **top_p 0.95 and top_k 20** (`tool_agent.py:1016-1017` defaults, not overridden by the arm). Upstream `eagle_utils.py:975-992` applies top-k, then top-p renormalisation to `target_probs` before this kernel. So the cap is E[max p] of the T = 0.65, top-k-20, top-p-0.95 renormalised distribution, which is somewhat higher than raw softmax at 0.65. The pinned build is assumed identical (UNVERIFIED, as the note says). |
| 6 | Qwen3-Next-80B-A3B native MTP, ~8,000 GSM8K samples (3 steps, β 0.6, LR 1e-4, 3 epochs): 0.897 / 0.719 / 0.476 → 0.912 / 0.776 / 0.616, ~3.09 → 3.30; up to 1.25× lower median ITL | **Partially confirmed** | The per-position table, "around 8,000 samples from openai/gsm8k" and Qwen3-Next-80B-A3B-Instruct are in the text. "Up to 1.25x speedup" in median inter-token latency is in Figure 1's alt text. But **3 steps / β 0.6 / LR 1e-4 / 3 epochs / seq 8,192 are the flags of the article's Qwen3.5-9B example command** (`--max-samples 5000`). The article does not give the Qwen3-Next run's hyperparameters. The sums 3.09 → 3.30 assume cumulative per-position rates (vLLM's metric). +0.21 holds either way. |
| 7 | HASS 5.15 → 5.51 (self-distill, L2-7B, T = 0); K = 10, w = 1.0, 3 alignment steps; L3-8B 4.52 → 5.08. FastMTP: frozen trunk, 389.4K at T = 0.6, K = 3, β 0.6, LR 5e-5; τ 1.83 → 2.73; ~70/10/0 → 80/56/36% | **Confirmed** (minor) | HASS Table A.4 (F 5.15 → MG 5.51), Implementation (K = 10, w = 1.0, aligned for 3 steps) and Table 1 L3-8B T = 0 (EAGLE-2 4.52, HASS 5.08) all match. In FastMTP, the K = 3 Table-1 mean τ is 1.83 for vanilla MTP and 2.73 for self-data FT, and the hyperparameters match §3.1 and Appendix A. Minor: the abstract gives per-position **70 → 81, 11 → 56, 2 → 36%**, so the first position is 81, not 80. "~70 / 10 / 0" is the Fig. 4 prose. |
| 8 | Holodeck regression numbers; AngelSpec "52.8% → 66.4%" not in the abstract → UNVERIFIED | **Partially confirmed — AngelSpec half refuted** | Holodeck: 5,000 greedy self-distilled samples, LR 2e-4, seq 1536, 1 epoch, vLLM RS 58.55 → 49.75%, MAL 2.756 → 2.493, pos2 41.4 → 29.5%, 30 prompts (T = 0.7). All confirmed. The loss was `lk_hybrid` (KL + TV) on greedy hard targets. **AngelSpec: the full text (arXiv HTML v1 and v2, §2.5, Table 2 "Hy3 MTP acceptance") says: "the base MTP checkpoint … after TTT on the diverse target-model rollout mixture. At T = 0, average acceptance increases from 52.8% to 66.4% … MAL rises from 2.58 to 2.99. At T = 0.9 … 51.3% to 63.3% and MAL from 2.54 to 2.90"** (vLLM, D = 3). The prior note (`training.md:55`) was right. The abstract's "~30%" belongs to DFly, a different drafter. Remove the line-20 "correction". |
| 9 | Albucino `mtp-dense`: 31 BF16 tensors, 1.362B params, lm_head and embed 635.7M each; own ≈90.6M (attn 51.4M, HC 19.8M, fc 6.6M each, shared expert 4.9M, router 1.3M); experts 2.52B params in 1,416,156,712 B INT4 sym g32 | **Confirmed** | Header 3,736 B, 31 tensors, all BF16, 1,361,966,848 params. The own params are 90.57M: self_attn 51.45M including the 1.64M indexer, HC mixers and inject/norm 19.77M, fc 6.55M each, shared_expert 4.92M, gate 1.31M. Expert file: 4,608 tensors. Packed I32 is 1,258,291,200 B (= 2,516,582,400 params × 0.5 B), BF16 scales 157,286,400 B, header 554,528 B, total 1,416,156,712 B. config.json: group 32, 4 bit, symmetric, observer minmax. The compact_sources quote is verbatim. |
| 10 | Qwen4Exp `output.hidden_states` = 10,240-wide HC stream; aux capture broken, DFlash/DSpark unservable (issue open); SpecForge #983 (open, 2026-10-16): no capture hook, HF/vLLM-fork capture, MoE DSpark with MTP warm start, SGLang serving is a follow-up | **Confirmed** (one update) | #38589 is OPEN (2026-09-09), and its body has these statements verbatim. #983 is OPEN, created 2026-10-16T09:27Z; follow-up 5 is serving on SGLang; it trains on Ascend. Update: the fix PR #38642 was converted to draft on 2026-10-08, after #41550 ("capture auxiliary states at residual reads", merged 2026-09-29) changed upstream capture. Its author says the Qwen4-Exp problem remains on main: the aux stream is still HC-wide and not contracted. |
| 11 | ms-swift MTP flags, head-only via freeze ratio 1 plus trainable params, qwen4_exp example lines 1–5 ("# 8*70G"); verl recommends detach_encoder = True and reports ~14% higher rollout acceptance when MTP is trained during RL | **Partially confirmed** | ms-swift docs lines 248–253 and 268–270, and `megatron_sft.sh` lines 1–5, are verbatim. verl's detach_encoder recommendation is verbatim. But the 14% sentence reads "Enabling MTP improves the rollout acceptance rate by around 14%", under §4 "Performance Notes for MTP in Rollout Inference". It does not say the gain comes from training the MTP during RL. The same paragraph says throughput does not rise on H20 and falls ~50% for mimo-7B with SGLang. The experiments are on mimo-7B-math. |
| 12 | s = 1 + 0.95·Δ/2.68: +0.20 → s 1.071 → +2.0 offline, +1.1 / +1.9 LB; +0.40 → +3.9, +2.2 / +3.9; −0.25 → −3.1 | **Confirmed** | Recomputed: −0.25 → s 0.911, −3.11 / −1.70 / −3.14; +0.20 → 1.071, +1.96 / +1.07 / +1.93; +0.40 → 1.142, +3.92 / +2.15 / +3.86 (P1 = 0.548×; P3 at 0.6·s on the 110-game table). Caveats: the 0.95 factor is the clean-decode elasticity. The 110-game s = 0.5 and 0.7 points were not among those L3's fact-check re-ran (0.6 / 0.8 / 1.2 / 1.8). sim2 over-predicted the 27B. |

**Other errors found while checking:**
- **Q1 line 20, Q1 Inferences and Implications 3 (consequence of row 8).** AngelSpec is the closest published analogue: a production MoE MTP, D = 3, base MAL 2.54–2.58 (ours 2.68), TTT on target-model rollouts in the serving format, with long-context code-agent trajectories in the mix. It gained **+0.36 to +0.41 MAL**, about twice the Qwen3-Next +0.21. The "+0.15 to +0.25 central case, 3.0–3.2 is the optimistic tail" framing should cite it. At +0.4 the Q4 table gives about +3.9 offline and +2.2 / +3.9 LB. The ceiling caveat still applies: our base starts higher.
- **Line 69, "The Kaggle drafter is byte-identical to albucino's (`arms.md:666`)".** `arms.md:666` bases that on the file list and sizes, not hashes. Q5's step-0 SHA-256 check is the right fix.
- **Line 51** repeats row 3's "≈30% of decode time"; it is an upper bound. **Line 53 and Q2 Inferences**: add the top-k 20 / top-p 0.95 renormalisation from row 5. **Line 15**: the hyperparameters belong to the Qwen3.5-9B example (row 6). **Line 17**: the verl 14% context (row 11).
- Not checked: the EAGLE-3 and Medusa bullets, TENSELERATE #56, the FriendliAI card, ms-swift #9802 / #9283, the upstream `enable_return_hidden_states` flag, and the Q3–Q4 memory and cost inferences.
