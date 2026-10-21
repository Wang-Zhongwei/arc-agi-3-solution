# arc3-idea-sweep: verify:1/10

Workflow run `wf_6b16ae20-e2c`, phase Verify. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "verdicts": [
  {
   "id": "relaxed-mtp-accept",
   "keep": true,
   "status_in_franzen": "present-but-off",
   "impact": 3,
   "effort_days": 0.25,
   "confidence": "medium",
   "first_step": "Add 20-min 'balanced' arms to arc-agi-3-v3/franzen-moe/kaggle/arms.json using the serve override {\"SPEC_ACCEPT_SINGLE\":0.3,\"SPEC_ACCEPT_ACC\":0.3} (also 0.5/1.0 and 0.5/0.5). Both keys already exist in the CFG dict in template.ipynb cell 12, so the builder's 'existing keys only' override works. Compare serve.log at bs 9-10 against fz-tput-base (accept len 2.58, 770 tok/s) and fz-arms-v4-control (2.66, 793 tok/s). Also compare the finish_reason=length rate against the 1.1% baseline (25 of 2,299 replies). Then run the best setting as a full 121-min 25-game arm.",
   "reason": "Not refuted. Thresholds are 1.0/1.0 in every run I could find: all 22 fz-* serve logs, dfranzen/skarin/amatlas notebooks, and rellik13's sglang_boot_result.json. build_bundle_pennyroyal.sh:1411 calls it a deliberate lossless choice. I read the kernel (speculative_sampling.cuh:80, accept if coin <= p/threshold_acc or p >= threshold_single). eagle_utils.py:775-830 passes both thresholds to it, applied to temperature-scaled, top-k/top-p-renormalized target probs. Gabriel's STATUS.md:50-52 and serve.sh:73-85 give C1 at temp 0.6 as 179 (1.0), 203 (0.5), 231 (0.3) tok/s. The ARC-Decode poster is real (icml.cc/virtual/2026/poster/66788, up to 1.6x over EAGLE-3 under sampling). No ablation exists in WRITEUP section 6 or arms.md; arms.md lists only greedy as untested, and tput-all-v7 greedy shows accept 2.87 against 2.58-2.71 sampled. Caveats: (1) Gabriel's baseline accept was 2.1-2.2 on chat, against 2.66 here, so expect +8-16% accept length, not +29%. (2) Throughput should convert to score. fz-experts16-full lost 17% of tokens (3.70M vs 4.48M) and scored -7.6 points, and the user's replay data suggests a similar slope. (3) It sharpens sampling. temp06 showed no change (45.92 vs 46.25, run-atlas findings sections 5-6), but the relaxed-decoding survey (arXiv 2607.08690) warns that relaxation needs capability evaluation. Neither the survey nor ARC-Decode tests long agentic runs. The batch-10 gain is not smaller than single-stream, because verify cost per step is fixed. No extra VRAM and no context cost, unlike the streams14 and ledger-streams12 arms."
  },
  {
   "id": "experts-top8",
   "keep": true,
   "status_in_franzen": "absent",
   "impact": 2,
   "effort_days": 0.25,
   "confidence": "medium",
   "first_step": "This arm is already pushed (mellamoa/fz-experts8, commit 09e7961, 20-min balanced panel; Kaggle status was QUEUED at review time). When it completes, run `kaggle kernels output mellamoa/fz-experts8 -p arc-agi-3-v3/kaggle-outputs/fz-experts8`. Compare bs10 median tok/s and accept len against fz-tput-base (770, 2.58). If decode is at least +6% and accept len is 2.55 or higher, add 'experts8-full' to franzen-moe/kaggle/arms.json: {\"serve\":{\"TEXT_CONFIG_OVERRIDE\":{\"num_experts_per_tok\":8}}} on all 25 games for 121 min.",
   "reason": "Not a new idea and not yet refuted: arms.md lists it as inconclusive, and the experts8 arm is already pushed with no output. The throughput math holds. fz-experts16-full serve.log gives bs10 median 625 tok/s against control 793, about -3.5% per extra expert, so top-8 should give about +7% decode. Accept len was unchanged at 2.67-2.78 even with the target changed. TEXT_CONFIG_OVERRIDE touches the target view only (franzen-moe/kaggle/build_notebook.py:115-132), the draft keeps its own config, and the router honours config.norm_topk_prob (qwen2_moe.py:313-314). Tokens correlate with score: experts16-full lost 17% of tokens and scored 38.69 against 46.25. Main risk is quality, which I could not settle. arXiv 2505.03531 says fewer active experts in fine-grained MoE gives at least 10% throughput with minor degradation. MoBiLE's abstract only claims negligible loss for token-adaptive halving, and I could not verify the idea's '8.5-11.6%' figure. experts16 also moved off-distribution (more experts than trained), so its -7.6 mixes lost tokens and lost quality. Impact is capped at 2 until the 25-game arm reads out."
  },
  {
   "id": "lowm-gemm-max-m",
   "keep": true,
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 0.3,
   "confidence": "low",
   "first_step": "In the notebook's launcher cell (cell 12), after the offline wheel install and before launching the server, add `sed -i 's/^_MAX_M = 32/_MAX_M = 64/' $VENV/lib/python*/site-packages/sglang/kernels/ops/gemm/sm120_lowm_bf16_gemm.py`. Set SGLANG_BF16_GEMM_LOG_SHAPES=1 for the first run to confirm m=36/40 shapes now take the Triton path. Run a 20-min arm and compare bs10 median against 770-793 tok/s; drop it if the gain is under +2%.",
   "reason": "The facts check out. _MAX_M = 32 is in build_bundle_pennyroyal.sh:102, and tactics are empty (_SM120_TUNED_TACTICS = {}), so every shape uses the generic (1,16,128) tactic. The verify graph is captured at bs=[1,2,4,7,8,9,10] with 4 tokens per request (serve.log:78), so bs 9 and 10 give m=36 and 40, which fall back to cuBLAS. Those batch sizes are 73% of control decode log lines (1457+2028 of about 4,800). Expected gain is much smaller than the idea's +5-8%. Gabriel's own patch 0004 bought only +3.6% at C1 (142.8 to 147.9 tok/s, PERF_CEILING.md), and the 30-50% bandwidth finding was measured at m<=8 with split-K WMMA kernels. At m=40 cuBLAS may already use proper tensor-core tiles, and Triton's BLOCK_M becomes 64, which is untuned. Smem is 80 KB at 4 stages, fine under about 99 KB. Realistic gain is +1-3% decode, or a loss. It costs a sed and a 20-min throughput arm, so it is a free rider. Do not spend a full 25-game arm on it."
  },
  {
   "id": "reasoning-effort-medium",
   "keep": false,
   "status_in_franzen": "tried-no-gain",
   "impact": 1,
   "effort_days": 0.1,
   "confidence": "high",
   "first_step": "Do not run. If you want the evidence, read arc-agi-3-v3/gb10-wt/token-efficiency/notes/gb10-arms/token-efficiency.md, lines 1-15 and 335-350.",
   "reason": "The user already tested it on Qwen3.8-Flash-Next (GB10 batch b6, token-efficiency-medium sub-arm) and it is listed in arms.md under old-harness 'Didn't work'. That note says 'medium' injects nothing in the checkpoint template and only removes the xhigh sentence. Reasoning tokens per request were inside the 7-control range (748 vs 633-861 on ar25, 702 vs 758-1,137 on sk48). The verdict was 'reasoning_effort=medium as a lever: no ... Do not take it to Kaggle.' Franzen's server passes only preserve_thinking, so the template default (xhigh) applies. Pennyroyal's 'medium' default is the same no-op against this template. The Kaggle variant (Franzen, SGLang, cached prefix) is not materially different, because the cache is unaffected either way. It cannot beat noise, and a 25-game arm costs 2 h of Kaggle GPU."
  },
  {
   "id": "reasoning-effort-low",
   "keep": false,
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 0.1,
   "confidence": "medium",
   "first_step": "Do not run unless a Kaggle slot is idle. If run: add \"reasoning_effort\":\"low\" to the --default-chat-template-kwargs string in cell 12 for one 25-game arm and judge it on tokens per cleared level (control 42.7k) and games >= 10, not on mean score.",
   "reason": "No evidence it helps, and the vendor warns against it. The HF model card (fetched) says lower effort in multi-turn agentic use 'can lead to insufficient analysis, more failures, and repeated retries, which may increase total latency and token consumption.' It was never tested: GB10 only tested 'medium', and its note names 'low' as the only candidate. The case for it is weak. Control replies are short (mean 5.2k reasoning chars, about 1.5k tokens, median 2.7k chars). Tokens per cleared level has sat at 39-48k across every harness change. Franzen reports that removing thinking from history was a large regression. Action slack is not the issue: 84% of cleared levels in control already have efficiency of at least 1. The ladder code exists but is off (ARC3_REASONING_EFFORT_LADDER, tool_agent.py:1778-1797, 9410-9430). Switching effort mid-run changes the first system-message sentence, which invalidates the whole cached prefix. Expected sign is about a coin flip with a negative lean, and it spends a 2 h arm that relaxed MTP or experts8 would use better."
  },
  {
   "id": "thinking-budget-forcing",
   "keep": false,
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 1,
   "confidence": "medium",
   "first_step": "Do not build Tier 1. If any piece is wanted, lower LOCAL_ANALYZER_MAX_OUTPUT from 12288 to about 10240 as a free rider in another arm. Do not enable ARC3_STUB_DEAD_REASONING.",
   "reason": "The counts hold up but the upside is tiny. In control, finish_reason=length is 25 of 2,299 replies, about 307k tokens (6.9% of all generated tokens). Capping at 8k tokens would save at most about 2.2% (excess reasoning chars over 28k, measured from the control transcripts); capping at 20k chars saves 6.9%, but only by cutting 105 replies that mostly finish with a tool call. The truncated reasoning is not lost: the harness yields on turn_token_budget, the next prompt says the last snippet executed no actions, and the model continues from its retained thinking. In bp35 the following reply was 495 chars of thought plus a tool call. The long replies are often the model parsing a board by hand in its head (37k chars in the bp35 example), which forced action would not fix. Stubbing discards reasoning the model acts on, and the HF card warns that cutting analysis raises failure and retry rates. Tier 1 is a day of work for at most +2-3% tokens and cannot be measured against noise. The SGLang issue cited (#25536) is real but shows thinking_budget failing regardless of speculative decoding, not specifically in EAGLE verify."
  },
  {
   "id": "hicache-host-tier",
   "keep": true,
   "status_in_franzen": "present-but-off",
   "impact": 2,
   "effort_days": 1.5,
   "confidence": "low",
   "first_step": "In cell 12's args, add `--enable-hierarchical-cache --hicache-size 32 --hicache-write-policy write_through --hicache-io-backend kernel` (the form at rellik13 nb.txt:661 and Pennyroyal serve-flash-next-frspec.sh:39-41; omit the NIXL storage flags). Set MAXREQ=13, CUDAGRAPH_MAXBS=13, MAMBA_CACHE=78, ARC3_MAX_ACTIVE_STREAMS=13, and keep context at 128Ki and drain at 58Ki. Run a 20-min balanced arm and check four things: the server boots, bs13 decode is about 850 tok/s or better, the cached-token fraction in the serve.log cache report is 93% or higher, and no cold re-prefill storms.",
   "reason": "Feasible but a low-confidence bet. HiCache works with extra_buffer plus NEXTN elsewhere: Pennyroyal's Flash-Next recipe uses it, and rellik13's scored notebook runs 48 GB write_through with extra_buffer and 16 requests. Host RAM in the control run is 115 of 176.9 GB (fz-arms.log), so 32 GB fits; 48 GB needs the FP8-PLE checkpoint. The mechanism is sound: the KV pool is 1,011,264 tokens (11.6 GB), 14 streams at about 87k mean context need about 1.22M, and the user's own streams14 and ledger-streams12 lost their gains by cutting context (46.38 vs 46.25; 48.08 vs 52.56). Ceiling is modest. From fz-streams14 serve.log, decode median is 838 at bs12, 882 at bs13 and 850 at bs14 against 793 at bs10, so +6-11%. Franzen's own patch (patch-sglang-mamba-final-prefill-v2.py) edits unified_radix_cache.py and says 'GPU integration testing is pending', so interaction with HiCache's host tier is unverified. Pennyroyal's docs say HiCache is not credited for decode throughput. I could not open the Kaggle discussion 744792 (JS-rendered) or find a verifiable writeup behind the claimed Siri 14.49 to 22.53 jump. FP8 weights (below) would free the VRAM without a host tier."
  },
  {
   "id": "fp8-weight-only-dense",
   "keep": true,
   "status_in_franzen": "partial",
   "impact": 3,
   "effort_days": 2.5,
   "confidence": "low",
   "first_step": "Stage 1 (about 0.5 d): build a runtime patch from competitors/sglang-flashnext-sm120/patches/0005-sm120-fp8-weight-only.patch. I checked with `git apply --check` in the scratchpad that its kernel-file and unquant.py hunks apply cleanly on top of Franzen's embedded 01-lowm patch (build_bundle_pennyroyal.sh:102); only the 4-line environ.py hunk needs a manual edit. Apply it in cell 12 before launch and set SGLANG_SM120_LOWM_FP8_WEIGHT=1. The notebook's existing env is a no-op on Pennyroyal. Run a 20-min arm and compare bs10 decode and the KV pool size in serve.log (control: 793 tok/s, 1,011,264 tokens). Stage 2 only if decode is at least +8%: free the BF16 weights after quantizing and add a dequant-to-BF16 scratch path for large-M prefill.",
   "reason": "Real upside, but the idea as written is incomplete. Gabriel's patches 0005/0006 exist and show C4 428 to 477 tok/s with accept unchanged (patch commit message). The BF16 dense stack is 8.7 GB per step against about 45 GB per step at bs10, which gives +8-10% decode. Freed BF16 would be worth about 370k KV tokens (KV is 11.58 GB for 1.01M tokens, about 11.5 KB per token), enough for 13-14 streams at 128k. But 0005 keeps BF16 copies: STATUS.md says '~3.6 GB VRAM' and 'MEMFRAC 0.96'. At that cost the KV pool would drop by about 290k tokens (about 28%), below the roughly 0.87M that 10 streams need, which would force context cuts and probably a net loss. The net-positive version has to release BF16 and handle large-M prefill itself, which the idea treats as small. A dequant-to-scratch GEMM for chunks above 64 rows adds roughly 7% to the dense prefill share, estimated, not measured. The user's MXFP8 result (-27%: 555 vs 765 tok/s; bs8 589 vs 728 even at greedy accept 2.87) came from Pennyroyal's single bundled switch (SGLANG_SM120_ONLINE_MXFP8: MXFP8 weights plus dynamic activation quantization, FP8 HC and lm_head), so it does not separate weight-only from the rest. Weight-only is materially different from what failed, so it is not dropped. Rank it behind relaxed MTP and experts8 because it is 2-3 days of work and kernel-level risk against a 5-day target."
  }
 ]
}
```
