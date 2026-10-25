# vLLM for GRPO rollouts on H200

These are candidates found on 2026-10-25 while trying to replace Lord Hansolo's runtime with an upstream vLLM nightly for the Kaggle RTX PRO 6000. That attempt was dropped; the RTX serving stays on Lord's runtime. Everything below comes from reading code and PRs, and **none of it has been measured on an H200**. Measure with [`bench_rollout.py`](../../ARC3-Inference/inference/grpo/bench_rollout.py) at C = 32/48/80 per TP=2 pair.

## Base runtime

**Don't reuse Lord's runtime.** It is vLLM `e975732` (09-23) plus an overlay tuned for the RTX PRO 6000:
- SM120 GEMM plans;
- exact CUDA graphs for 14 sequences × MTP 3;
- the launcher's `TORCH_CUDA_ARCH_LIST=12.0`;
- an NVFP4 MTP head, which runs FP4 natively only on SM120.

**Use an upstream nightly image instead.** It carries sm_90 kernels. A RunPod pod has internet, so the image can be pulled by digest; Kaggle GPU notebooks have none.

Merged upstream since 09-23 that Lord's base lacks and that matter on H200:

| Area | PRs |
|---|---|
| Hopper kernels | SM90 low-latency GEMM plans for Qwen4Exp; the H200 retune of the merged QSA projection (#60776); the HC up projection kept on those plans (#60027) |
| TP > 1 | Sequence parallelism for Qwen3.8-Flash-Next, `--enable-hc-sp` (#56322); QSA `head_dim` from the unsharded head count (#59945) |
| Speedups | merged QSA projections (#59533); fused QSA prepare (#57097); skip drafting when no request can use the drafts, e.g. non-final prefill chunks of long contexts (#62784) |
| Draft head | quantized draft lm_head (#59973): `fp8` runs natively on Hopper |
| Checkpoint | INC (AutoRound) PLE route (#59990) |
| RL weight transfer | NCCL many-to-N with per-parameter layouts (#60934), metadata sent once (#35911), receive-buffer reuse fix (#62555) |

## Must change for GRPO (correctness)

1. **Weight sync each step.** The LoRA weights (attention + GDN, BF16) are merged into the engine through vLLM's weight-transfer engines (`vllm/distributed/weight_transfer/`, IPC and NCCL many-to-N).
   - After every sync, reset the prefix cache (`/reset_prefix_cache`): cached KV, Mamba and RecoverSSM states belong to the old weights.
   - The quantized draft head (#59973) refreshes itself on weight reload.
   - The MTP head is not trained, so watch its acceptance as the policy drifts.
2. **Behavior logprobs.** Use `--logprobs-mode processed_logprobs`, i.e. logprobs after temperature 0.7, top-p 0.95 and top-k 20, and request `logprobs` (not `logprob_token_ids`, so #61219 isn't needed). Check them against the trainer's.
3. **Forced `</think>`.** The 10,240-token thinking budget is enforced in the sampler (`vllm/v1/sample/thinking_budget_state.py`), so that token isn't sampled from the policy. Mask it in the loss. vLLM doesn't flag it, so identify it by position: reasoning length equal to the budget.
4. **MoE routing gap.** `--enable-return-routed-experts` returns each token's top-10 of 512 experts, which allows routing replay in the trainer as an alternative to TIS or masking. Lord's base has it too. QSA's top-512 block selection has no such capture; that would be a vLLM addition, if the gap turns out large.
5. **FP8 KV and FP8 indexer cache** (decided). The trainer fake-quantizes K and V to FP8.

## Worth adding for throughput

1. **RecoverSSM for Qwen GDN (#58863, open).** No per-draft Mamba blocks, so more KV and more concurrent rollouts per engine.
   - It pairs with #61017, stacked on it: one set of commit launches for all GDN KV groups. The author measured 523 → 184 µs of host time per step.
   - And with #62625: the replay records move out of the Mamba page. With #58863 the upstream KV block was 1,856 tokens, against 1,664 in Lord's runtime.
   - Before trusting it, run a temperature-0 output check.
   - If it is combined with #63193 (mixed-batch GDN slicing), add a copy-back of the RecoverSSM output. Without it, the speculative rows' output is lost in every step that also prefills (found 10-25: the merge was textually clean).
2. **TP=2 vs DP per pair.** Candidates to try:
   - `--enable-hc-sp` (#56322);
   - #62961 (open): the PDL all-reduce saves 1.4–1.9 µs per all-reduce at TP 2–4, and `in_proj_ba` folds into `in_proj_qkvz`;
   - the QSA decode-context-parallel series (#59279, #61728, #61242; open), which shards 130K-token contexts across ranks.
3. **Host RAM with several engines per node.**
   - Pin the PLE table at its exact size: 95.4 instead of 128 GiB per engine. This is our patch (porting-advaith `02035725`), not upstream.
   - With vLLM data parallel inside one server, `EngramConfig.dp_shared_memory` shares one PLE host table across the DP ranks.
4. **CUDA graphs.** Leave the capture sizes to upstream's default, which now adds exact decode sizes n × (1 + K) on a request-count grid up to 512 tokens; C = 80 × 4 = 320 is covered. Don't pass Lord's `--max-cudagraph-capture-size`.
5. **Prefix caching across a group.**
   - The 8 rollouts of a group share the system prompt and the first frames, so start a group together.
   - `prefix_match_unit` must divide upstream's KV block size: 64 does, 128 failed at engine start on 10-25.
6. **GDN decode kernel.** Compare Triton against CUDA (`VLLM_GDN_DECODE_KERNEL`), and #59366 (open, NVIDIA author), which runs the RecoverSSM verify inside the fused CUDA MTP kernel.
7. **Drafter.** Use Intel's own BF16 MTP head, with `draft_lm_head_quantization: "fp8"`.

## Not needed on H200

- Lord's SM120 GEMM plans and his C14 capture sizes.
- The 32K draft vocabulary: Lord-only; #59973 replaces it.
- The `embed_tokens` host offload: there is room on 141 GB.
- The NVFP4 MTP head.

## Caveats

- **No maintainer approvals.** None of the open PRs above has one. Prefer merged ones, and check temperature-0 outputs before any speed number counts.
- **Different KV layout.** Upstream groups the KV cache as 6 groups × 1,856 tokens; Lord's runtime used 5 × 1,664, because it merges the PLE and GDN cache groups. So upstream fits fewer tokens per GB.

## First benchmark: TP=2, 40 concurrent rollouts (decided 2026-10-25, not yet run)

One GRPO-shaped step on one 2×H200 SXM pod: 40 replayed games at once, each run to its end. It measures step time and throughput before any GRPO code exists.

**Runtime: upstream nightly `vllm/vllm-openai:nightly-83c078d` (2026-10-23, amd64 `sha256:d428631b…`), unpatched.**
- Lord's runtime (09-23) predates the TP>1 fix for QSA's head dimension (#59945, 10-07) and sequence parallelism (#56322, 10-12).
- The "no new bundle" decision covers only the Kaggle RTX serving.
- RunPod pulls the image by digest; the pod has internet.

**Model.** `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` from Hugging Face (181 GB). Its own BF16 MTP layer (1,565 tensors) is the drafter, so no separate draft model.

**Server settings**

| Setting | Value | Reason |
|---|---|---|
| `--tensor-parallel-size` | 2 | the configuration under test |
| `--max-num-seqs` | 40 | the concurrency under test: every rollout of a step is in flight at once |
| `--max-model-len` | 147,072 | 130K context cap + 12,288-token reply + estimate error; same as Kaggle |
| `--max-num-batched-tokens` | 8,192 | Kaggle's 2,048 saved activation memory on a 96 GB card. H200 has room, and bigger chunks prefill 130K contexts in fewer steps. Decode at 40 × 4 = 160 tokens leaves ~8K tokens of prefill per step. |
| KV / indexer / Mamba cache dtypes | `fp8_e4m3` / `fp8` / `bfloat16` | the decided FP8 caches; Mamba state as on Kaggle |
| `--gpu-memory-utilization` | 0.92 (vLLM default) | KV is not the constraint. Per rank: ~129 GiB usable − ~35 GiB of weights (half of the 69.5 GiB at TP1) − activations ≈ 90 GiB of KV. At TP2 each rank holds half the KV and GDN heads, so that is roughly 13M tokens, against at most 40 × 130K = 5.2M. |
| Speculative decoding | `{"method": "mtp", "num_speculative_tokens": 3, "draft_lm_head_quantization": "fp8"}` | MTP 3 as on Kaggle. FP8 runs natively on Hopper. The quantized head only affects acceptance: the target verifies with its own BF16 head. |
| Prefix caching | on, `--prefix-match-unit 64` | Prefix caching reuses each game's growing context across turns. 64 divided upstream's 1,856-token KV blocks at TP1; the TP2 block size is unknown, hence the fallback. |
| `--logprobs-mode` | `processed_logprobs` | GRPO's behavior logprobs (after temperature, top-p, top-k); the benchmark requests them, so their cost is included |
| Parsers | `--reasoning-parser qwen3` (enables `thinking_token_budget`), `--tool-call-parser qwen3_coder`, `--enable-auto-tool-choice` | as on Kaggle |
| Template | Intel's `chat_template.jinja`, `--default-chat-template-kwargs {"preserve_thinking": true}`, `--generation-config vllm` | as on Kaggle |
| Images | `--mm-processor-kwargs {"max_pixels": 589824}`, `--limit-mm-per-prompt {"video": 0}` | 768² px = a 64×64 grid at upscale 12; the replayed runs used upscale 10 and 12 |
| CUDA graphs | upstream default (no flags) | It captures decode sizes n × (1 + 3) on a request-count grid; with 40 sequences the ceiling is min(40 × 4 × 2, 512) = 320 tokens. |
| PLE table | upstream default: Engram host offload | keeps ~95 GiB of BF16 table off the GPUs; host RAM is ample on a 2×H200 pod |
| Not used | SM120 GEMM, C14 capture sizes, draft vocabulary, `embed_tokens` offload, NVFP4 drafter, `--enable-hc-sp` | RTX-specific, or a second variable. Sequence parallelism comes as an optional second job. |

**Automatic fallbacks** (a fresh server start each time):
1. `Invalid prefix_match_unit` at startup → restart without `--prefix-match-unit`.
2. A startup error naming `draft_lm_head_quantization` → restart without it.
3. CUDA OOM at startup → restart with `--gpu-memory-utilization` 0.88.

**Workload** ([`bench_rollout.py`](../../ARC3-Inference/inference/grpo/bench_rollout.py), replay from [`build_replay.py`](../../ARC3-Inference/inference/grpo/build_replay.py)):
- **Rollouts:** 40 = 5 games × 8 rollouts (G = 8 as in GRPO): cn04, sc25, ka59, s5i5, dc22. These are the five pilot games closest to 50% within the budget (47–53%).
- **Attempts:** 8 of each game's 12 recorded untrimmed attempts (seed 0), so a group shares the system prompt and the first frames, as in GRPO.
- **Requests:** each asks for the recorded reply length at temperature 0.7, top-p 0.95, top-k 20, with a thinking budget of 10,240 and logprobs on. Tool and game time are not simulated: the server is the only clock.
- **Reported:**
  - step wall time and generated tok/s, both overall and until half the rollouts finished (steady state at C = 40);
  - median per-rollout tok/s;
  - peak KV usage, peak running and waiting requests, preemptions;
  - MTP tokens per step;
  - kernels and KV size from the server log.

**Pod and cost**
- **Hardware:** 2×H200 SXM in one pod (NVLink), RunPod community at $7.18/h for both, secure if community is out.
- **Disks:** 400 GB `/workspace` volume, 50 GB container disk.
- **Run:** one script for pull, download, server, benchmark, results and `POD_END=stop`, with a 120-minute self-stop. A local watcher pulls the results and stops the pod.
- **Time:** ~15 min setup (image, 181 GB download, load) + ~25 min for the 40-rollout step ≈ 40 min, about $5 community.
- **Optional job B:** the same step with `--enable-hc-sp`, +~20 min (+$2.40).
