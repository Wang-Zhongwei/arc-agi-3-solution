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
- **Rollouts:** 40 = 5 games × 8 rollouts (G = 8 as in GRPO): cn04, sc25, ka59, s5i5, dc22. These are the five pilot games closest to 50% within the budget (47–53% when chosen; 44–53% with images priced at each run's board size).
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

### Result (2026-10-25)

**Run:** pod `hl0b9na5elmeqw`, 2×H200 SXM. Community had no 2-GPU host, so it ran on secure at $10.58/h for 30.5 min, about $5.40; the pod was deleted at the end. Data in `runs/runpod/jonathanwang2022/grpo-rollout-bench/hl0b9na5elmeqw/`.

| | |
|---|---|
| Step: 40 rollouts (5 games × 8) | **960 s.** Rollouts ended at p10 444 s, p50 855 s; the last was s5i5 at 961 s (95K tokens). Mean end per game ranged from 470 s (sc25) to 873 s (s5i5). |
| Generated | 2.61M tokens. **2,713 tok/s** over the step, 2,814 tok/s until half the rollouts finished, **3,020 tok/s** while 36 or more ran. |
| Per rollout | median 89 tok/s |
| GRPO estimate | The replay re-prefills each recorded reply: 4.44M uncached prompt tokens, 1.7× the generated tokens, where GRPO would see about 0.7×. Across the 10-s samples with 36 or more running, generation falls by 133 tok/s per 1K prompt tok/s. At GRPO's ~1.9K prompt tok/s that gives about **3.5K tok/s**. |
| MTP | 2.82 tokens per step: Intel's BF16 MTP head with an FP8 draft lm_head. Lord's NVFP4 head on RTX gave 2.67–2.75. |
| KV cache | 79.3 GiB per rank, 9.34M tokens. Peak use **28%**, no preemptions: KV doesn't limit concurrency. |
| GPU | 93% busy, 569 W mean |
| Kernels | INT4 experts on `HUMMING` (Marlin on RTX), BF16 MTP experts on Triton, V2 model runner. KV blocks are 1,664 tokens at TP=2, so `prefix_match_unit` 64 is valid. |
| Startup | SSH after 277 s (image pull); 169 GB downloaded in 182 s; server ready in 310 s on the first try, no fallback needed |
| Errors | 3 of 1,448 requests, all ka59 near the cap. Real prompts of 147.9K and 148.4K tokens exceeded `max_model_len` 147,072; the third failed because vLLM shrank `max_tokens` below the requested `min_tokens`. |

**Findings**
- **Our token estimate ran low; fixed on main (755d4749).** It priced every image at 402 tokens, the cost of a 640-pixel board. 9 of the 12 replayed runs drew 768-pixel boards at 578 tokens each. In those runs real ÷ estimate had a median of 1.04, p99 1.14 and a maximum of 1.17 on image-heavy late requests. With images priced at each run's board size, it is median 0.97, p99 1.07, max 1.07.
- **For GRPO, check the 130K budget with the server's exact `prompt_tokens` and `completion_tokens`.** To cap `max_tokens` before sending, the harness's own estimate is enough. It recomputes characters per token from the previous request's real count, which is the same as the previous real count plus the new messages at that rate. On contexts above 60K it is median 0.999, between 0.94 and 1.04 ([details](grpo-l1-l2-token-budget.md#token-estimate)).
- **Throughput fell short of the estimate.** About 3.0–3.5K tok/s per TP=2 pair, against 3.5–6.5K estimated for C = 32–48 (about 1.7× an RTX PRO 6000 per GPU). The GPUs were busy, but KV stayed under 30%.
- **The longest rollout sets the step time**: ~95K tokens at ~99 tok/s ≈ 16 min.
- **Concurrency: 64 per TP=2 pair (projection; measured on 10-26: 3,920 tok/s, below).** A linear fit of step time to the 10-s samples (R² 0.55) gives 13.5 ms + 0.52 ms per running sequence + 10.8 µs per uncached prefill token, at 2.83 tokens per sequence per step. At GRPO's prefill (0.7 × generated) it projects:

  | Concurrency | 40 | 48 | 56 | 64 | 72 | 80 |
  |---|---:|---:|---:|---:|---:|---:|
  | Total tok/s per TP=2 pair | 3.2K | 3.5K | 3.6K | **3.8K** | 3.9K | 4.0K |
  | tok/s per rollout | 81 | 72 | 65 | **59** | 54 | 50 |
  | Minutes for a 95K-token rollout, all slots full (measured at 40: 16, as the step emptied) | 20 | 22 | 24 | **27** | 29 | 32 |
  | KV if every rollout reaches 130K (9.34M available) | 5.2M | 6.2M | 7.3M | **8.3M** | 9.4M | 10.4M |

  64 is the largest concurrency here that cannot run out of KV even if every rollout reaches the cap. Past it, each 8 more sequences add only 3–4%. An earlier fit with a larger prefill term gives 4.2K total and 65 per rollout at 64, so expect 3.8–4.2K total (+15% over 40) and about 60 tok/s per rollout. That is about the per-game speed of a Kaggle run (reset-baseline: 867 tok/s over 14 streams ≈ 62). Measured in this run, with its heavier re-prefill: 3.1K total and 79 per rollout with 40 running, 2.4K and 111 with 20. Averaged over a synchronous step the total is about 10% lower than at full batch (2.7K against 3.0K here), because the batch empties at the end. With synchronous steps the longest rollout sets the step time, so start the next group's rollouts as slots free up.
- **Next measurements:** concurrency 80; TP=2 vs 2 × TP=1 at the same concurrency. Concurrency 64, the MTP draft count and `--enable-hc-sp` were measured on 10-26 (below).

## Second benchmark: MTP draft count and `--enable-hc-sp` (2026-10-26)

**Run:** pod `4xquqsnaaubac9`, 2×H200 SXM on secure at $10.58/h (community had no 2-GPU host), about 45 min and $8, deleted at the end. Same nightly image as the first benchmark. Data in `runs/runpod/jonathanwang2022/spec-bench/4xquqsnaaubac9/`; script [`spec_bench_pod.sh`](../../ARC3-Inference/inference/grpo/pod/spec_bench_pod.sh).
- 64 concurrent rollouts: 8 games (cn04, dc22, ka59, s5i5, g50t, ls20, r11l, sp80) × 8 recorded attempts.
- Each rollout starts at its first request with at least 40K tokens of context, so decoding runs at GRPO-like lengths ([`bench_rollout.py`](../../ARC3-Inference/inference/grpo/bench_rollout.py) `--start-tokens`).
- Each configuration is measured over a 200 s full-batch window that opens once the opening prefill is done (`--window`).

| Configuration | Tokens/s | Tokens per step | Step |
|---|---:|---:|---:|
| **MTP 3 (current)** | **3,920** | 2.87 | 46.4 ms |
| MTP 4 | 3,834 (−2.2%) | 3.19 | 52.7 ms |
| MTP 5 | 3,851 (−1.8%) | 3.34 | 54.8 ms |
| MTP 3 + `--enable-hc-sp` | 3,771 (−3.8%) | 2.87 | 47.6 ms |
| MTP 4 + `--enable-hc-sp` | skipped: both parts lost on their own | | |

- The concurrency-64 projection held: 3,920 tokens/s at a 46.4 ms step (projected 3.8–4.2K and 47.8 ms).
- Conditional acceptance stays at 0.76–0.77 per draft position, so a 4th and 5th draft add the expected accepted tokens (+11%, +5%). At 64 sequences each extra draft also makes the target verify 64 more tokens per step and adds a draft pass, and the step grows more than the tokens.
- `--enable-hc-sp` was on (the server args show `enable_hc_sp: True`) and made the step 1.2 ms slower.
- **Decision:** keep MTP 3 and leave `--enable-hc-sp` off.

**Where a decode step goes** (MTP 3; a torch profile of 30 engine iterations, rank 1; [`profile_summary.py`](../../ARC3-Inference/inference/grpo/profile_summary.py)):

| Part of the step | ms | Share |
|---|---:|---:|
| Target forward (one CUDA graph) | 26.2 | 55% |
| Drafter passes (~5 small graphs) | 4.2 | 9% |
| Eager kernels (~360 per step: sampling, top-k/top-p, log-softmax, all-gathers, some MoE) | 7.5 | 16% |
| GPU idle | 10.0 | 21% |

- r, one draft token's cost against the target forward, is about 0.053.
- Inside the target forward, the INT4 MoE GEMMs (HUMMING) take 11 ms (42%), about 60% of the HBM bound for reading the expert weights.
- Rank 0's kernel time includes ~9 ms per step of all-reduce waiting for rank 1.
- **Next lever:** the 37% of the step outside the two graphs (eager kernels and idle GPU), which points at host-side work per step, not at the draft count.

**Temperature-0 check: inconclusive.**
- The 8 check requests were sent together, so each server split their prefills into chunks differently.
- Outputs then differed from the first token, which no draft setting touches.
- `bench_rollout.py --greedy` now sends the requests one at a time.
- None of these configurations is adopted, so nothing depends on the check.

## Third benchmark: tuned dense GEMMs, Marlin MoE, live harness (2026-10-26)

**Run:** pod `zn157yeall9q2i`, 2×H200 SXM on secure at $10.58/h (US-GA-2), about 70 min and $12, deleted at the end. Same image and server flags as the second benchmark (MTP 3, 64 sequences). Data in `runs/runpod/jonathanwang2022/rollout-env/zn157yeall9q2i/`.

### Tuned dense GEMMs at M = 256 (PyTorch TunableOp)

- A decode step verifies 64 sequences × 4 tokens = 256 rows, and vLLM's skinny-GEMM path only covers M ≤ 16. So the BF16 projections use cuBLASLt's heuristic algorithm (the idea comes from Lord's RTX tuning).
- TunableOp times the cuBLASLt algorithms for each shape and keeps the fastest.
- The tuned table came from a first run with tuning on. The measured run read it with tuning off.
- The replay and the 200 s window are the same as in the second benchmark.

| Configuration | Tokens/s | Tokens per step | Step |
|---|---:|---:|---:|
| Baseline (second benchmark) | 3,920 | 2.87 | 46.4 ms |
| Tuned table, tuning off | 3,949 (+0.7%) | 2.84 | 45.2 ms |
| Tuning on while serving | 1,621 (−59%) | 2.75 | 105.5 ms |

- Both ranks read the table: 4,433 shapes each, and 14 have M = 256, most of them with a non-default algorithm.
- The step is 1.2 ms shorter, but tokens per step came out lower, so tokens/s moved only +0.7%. With one run of each configuration, the gain is at most 0–3%.
- **Why tuning can't help much:**
  - At M = 256 a projection takes about 10 µs. The 1280×2560 one takes 10.6 µs, against 1.4 µs to read its weights from HBM.
  - The time is fixed per-call overhead, not tile choice. The ~550 such calls per step add up to the profile's 4.8 ms.
  - The remaining lever is fewer, larger GEMMs (fusing projections), not better tiles.
- With tuning on while serving, every new shape stalls to be tuned, and prefill chunks keep producing new shapes.
- **Decision:** not adopted.

### Marlin MoE

- `--moe-backend marlin` (draft kept on Triton) does not start: "WNA16 MoE backend 'MARLIN' does not support the deployment configuration since the MoeWNA16 checkpoint layout is not supported."
- The INT4 MoE stays on HUMMING.
- `--moe-backend triton` is untested.

### Live harness (partial)

**Setup:**
- Real games instead of the replay: the harness runs inline on the pod (`make interactive`) with [`configs/rollout-throughput/v1.json`](../../ARC3-Inference/configs/rollout-throughput/v1.json).
- The 8 games × 9 = 72 games share 64 slots, with a 30-minute cap.
- Script: [`rollout_env_pod.sh`](../../ARC3-Inference/inference/grpo/pod/rollout_env_pod.sh). Summary: [`rollout_env_report.py`](../../ARC3-Inference/inference/grpo/rollout_env_report.py).
- Stopped after 11 minutes, so the M = 256 test could run.

| Window | Tokens/s | Running (of 64) | Waiting | Tokens per step | Prefix hit |
|---|---:|---:|---:|---:|---:|
| Minutes 3–10 | 3,555 | 63.8 | 7.3 | 2.60 | 92% |
| Minutes 7–11 (end) | 3,855 | 63.8 | 7.2 | 2.62 | 94% |

- **Slots stay full.** With 72 games for 64 slots, about 7 requests are always waiting, there are never fewer than 62 running, and nothing is preempted.
- **The host is not the limit.** The machine is 98% idle and the harness uses about 10% of one core.
  - vLLM's engine core process uses 92% of one core. That is one single-threaded loop, which fits the 21% idle GPU time in the step profile.
- **GPU busy:** 94% on rank 0, 80% on rank 1.
- **Throughput over time:**
  - Tokens/s rose from 3.4K to 3.9K over the first 10 minutes, then stayed within 2% of the replay (3,855 vs 3,920).
  - Acceptance was lower (2.62 vs 2.87 tokens per step), and the effective step was shorter (~43 ms vs 46.4 ms) because contexts are still short 10 minutes into a game.
  - The replay's figure works as a planning number. How throughput holds as contexts grow to the GRPO cap is unmeasured.
