# arc3-idea-sweep: sweep:serving

Workflow run `wf_6b16ae20-e2c`, phase Sweep. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "ideas": [
  {
   "title": "Relaxed MTP acceptance: speculative accept thresholds 1.0 \u2192 0.5 \u2192 0.3",
   "category": "serving",
   "description": "The launcher already has the knobs: CFG SPEC_ACCEPT_SINGLE / SPEC_ACCEPT_ACC (template.ipynb cell 12), which feed --speculative-accept-threshold-single/-acc. Both are pinned at the lossless 1.0. Run two arms: (a) single=0.5, acc=1.0, which always accepts a draft token the target gives \u22650.5 and otherwise keeps the normal coin flip; (b) single=0.3, acc=0.3, which is Gabriel Olympie's shipped setting. For each arm, compare the serve.log 'accept len' (baseline mean 2.69, 2.65 at bs10), harness tok/s and the 25-game score/trimmed mean. Also watch the harness degeneracy and repetition counters, because the change sharpens sampling. The thresholds cannot be combined with --speculative-use-rejection-sampling (the server refuses).",
   "mechanism": "Verify cost per step is fixed: bs\u00d74 draft tokens, with weights read once per step. Every extra accepted draft token is therefore close to a free output token. At temp 0.7 the target-only verifier still rejects a draft that matches the target's top token whenever the coin exceeds p. Forcing acceptance when p \u2265 \u03c4 raises accept length; greedy decoding alone moved it 2.67\u21922.81. On this exact model Gabriel measured +13% decode at \u03c4=0.5 and +29% at \u03c4=0.3 (temp 0.6). The output distribution sharpens toward high-probability tokens, much like a lower temperature, and the temp-0.6 arm showed no score change. Expect roughly +15\u201325% generated tokens per budget, and the atlas shows tokens drive levels cleared.",
   "evidence": [
    "https://github.com/gabrielolympie/sglang-flashnext-sm120/blob/main/scripts/serve.sh lines 73-77: 'Ladder measured at temp 0.6 (C1 median): 1.0 lossless = 179 - 0.5 = 203 - 0.3 = 231 tok/s'; 0.3 passes greedy/needles/8x GSM/code gates",
    "Pennyroyal d00d88e python/sglang/srt/server_args.py lines 2241-2250: threshold_single = accept if target prob > threshold; threshold_acc raises accept prob p to min(1, p/threshold_acc)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/franzen-moe/kaggle/template.ipynb cell 12: CFG SPEC_ACCEPT_SINGLE=1.0, SPEC_ACCEPT_ACC=1.0 passed to --speculative-accept-threshold-*",
    "daniel-franzen-sol/serving/build_bundle_pennyroyal.sh:1411 says to keep 'lossless acceptance thresholds'. This was a deliberate choice, not an ablation; WRITEUP.md \u00a76 never tested it",
    "arc-agi-3-v3/reports/run-atlas/findings.md \u00a76: greedy raised draft acceptance 2.67\u21922.81; versions.md: temp 0.6 arm 45.92 vs control 46.25 (no effect)",
    "arc-agi-3-v3/kaggle-outputs/fz-arms-v4-control/serve.log: mean accept len 2.69 of max 4 (3 steps; QSA caps draft tokens at 4), bs10 decode median 793 tok/s"
   ],
   "effort": "<2h",
   "expected_impact": "high",
   "status_in_franzen": "present-but-off"
  },
  {
   "title": "Fewer routed experts (top-10 \u2192 top-8), the opposite of the failed experts16",
   "category": "model-config",
   "description": "Use the existing TEXT_CONFIG_OVERRIDE in the arm builder. arms.json already defines an 'experts8' arm, but it was never run. Set num_experts_per_tok=8 on the target view only and leave the MTP draft at 10, since the draft cannot change output quality. Run the 20-min 'balanced' throughput check first (bs10 decode median vs 770\u2013793), then the full 25-game arm. If quality drops, try a dynamic version instead of fixed top-8. Patch SGLang's top-k (python/sglang/srt/layers/moe/topk.py) to drop routed experts whose renormalized gate weight is below \u03c4 (about 0.03\u20130.05), with a floor of 6. Map dropped ids to the invalid-expert id the fused MoE path skips; check that the Marlin MoE align step tolerates it. A per-layer k (LExI) is the other option.",
   "mechanism": "At 10 streams \u00d7 4 verify tokens = 40 tokens/step, reading routed-expert weights dominates the ~33 ms decode step. Under uniform routing the union of top-10 picks over 40 tokens covers about 55% of 512 experts (~30 GB of the ~60 GB W4A16 expert stack). Top-8 covers about 47%: roughly 15% fewer bytes and 20% fewer Marlin FLOPs. The user's experts16 run measured the slope: +60% active experts cost 25% of bs10 decode. Interpolating, top-8 should give about +6\u201310% decode. The shared expert plus renormalized top-8 keep most of the routing mass. Increasing the expert count is already tried-no-gain.",
   "evidence": [
    "kaggle-outputs/fz-experts16/serve.log: bs10 decode median 592 tok/s vs 770 (fz-tput-base) and 793 (control)",
    "kaggle-outputs/fz-experts16-full/summary.txt: 504.6 harness tok/s and mean 38.69 vs control 611 tok/s and 46.25 (atlas noise band is about \u00b11.1)",
    "arc-agi-3-v3/franzen-moe/kaggle/arms.json: 'experts8' arm defined (TEXT_CONFIG_OVERRIDE num_experts_per_tok=8); no output folder exists",
    "https://github.com/gabrielolympie/sglang-flashnext-sm120/blob/main/docs/PERF_CEILING.md: routed experts (512, top-10) are 72.98 GB on disk and 1.43 GB per token",
    "https://arxiv.org/abs/2505.03531: for fine-grained MoE, reducing activated experts gives substantial efficiency with only minor degradation; at least +10% throughput with no degradation",
    "https://arxiv.org/abs/2509.02753 (LExI): training-free layer-adaptive active-expert counts; at equal throughput, accuracy is 10% better than uniform pruning on Qwen1.5-MoE",
    "https://arxiv.org/pdf/2510.12357 (MoBiLE): naively halving active experts in Qwen MoE cost 8.5% GSM8K and 11.6% HumanEval, so do not go far below 8"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Let the low-M Triton BF16 GEMM run at the real verify batch (raise _MAX_M from 32 to 64)",
   "category": "bugfix-quickwin",
   "description": "Franzen's port of Gabriel's patch 0004 only sends a GEMM to the Triton kernel when m \u2264 32 tokens (`_MAX_M = 32` in sglang/kernels/ops/gemm/sm120_lowm_bf16_gemm.py). Target verify and draft-extend at bs 9\u201310 have m = bs\u00d74 = 36\u201340. Those steps fall back to cuBLAS for every BF16 projection: GDN in/out proj, QSA attention and the shared expert. The test is cheap. Before launch, sed the installed file in /tmp/sgl-intel/venv to `_MAX_M = 64`; no rebuild is needed. Run once with SGLANG_BF16_GEMM_LOG_SHAPES=1 to confirm which (m,n,k) shapes now take the Triton path, then A/B bs10 decode tok/s on a 20-min throughput arm. If it helps, add tuned (40,n,k) tactics to _SM120_TUNED_TACTICS so they precompile before graph capture.",
   "mechanism": "Gabriel measured cuBLAS-under-CUDA-graph on sm120 at only 30\u201350% of DRAM bandwidth on the mid shapes (n=2560\u00b7k=1536, n=1280, n=512), versus about 90% for the Triton kernel. About 8.7 GB of BF16 dense weights are read every verify step. If cuBLAS runs at about 60% at m=40 and Triton at about 85%, the saving is about 2\u20133 ms of a ~33 ms step, i.e. +5\u20138% decode. There is no quality impact. The gain is uncertain because cuBLAS may choose better kernels at m=40 than at m\u22648, so measure before committing.",
   "evidence": [
    "daniel-franzen-sol/serving/build_bundle_pennyroyal.sh:102 (embedded patch 01-lowm): `_MAX_M = 32` with the comment 'Covers decode (m = bs <= 4), bs=1 target-verify (m = 4-5) and bs<=4 C4 verify (m <= 16)'; dispatch uses m = x.numel() // x.shape[-1]",
    "kaggle-outputs/fz-arms-v4-control/serve.log: 3,485 of 4,803 decode log lines are at bs 9\u201310 (m=36\u201340); verify graphs are captured with num_tokens_per_req=4",
    "https://github.com/gabrielolympie/sglang-flashnext-sm120/blob/main/docs/PERF_CEILING.md: 'Mid shapes ... were genuinely bad (30-50%)'; the Triton kernel 'hits 90% BW'; bf16 per-step bytes are GDN 4.17 + QSA 1.34 + HC 1.32 + shared 0.61 GB",
    "daniel-franzen-sol/WRITEUP.md:95 (low-M GEMM adopted 'for faster matrix multiplications at small batch sizes')"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "Set reasoning_effort explicitly (medium vs the model default xhigh)",
   "category": "model-config",
   "description": "The server sends only {\"preserve_thinking\": true}. The harness adds reasoning_effort only when ARC3_REASONING_EFFORT_LADDER fires after a truncation (tool_agent.py:7210-7232). Every request therefore runs at the checkpoint's default effort, which the model docs list as xhigh. Add \"reasoning_effort\":\"medium\" to --default-chat-template-kwargs and keep it constant for the whole run so the system-prompt prefix stays cached. A/B it against the default on 25 games. Before doing so, check that the checkpoint's chat_template.jinja honours reasoning_effort and where it injects the effort sentence. Gotcha: if the effort text sits at the top of the system block, as in Pennyroyal's bundled template, each change of the per-request ladder invalidates the whole cached prefix (60\u2013116k tokens re-prefilled). Leave the ladder off.",
   "mechanism": "Tokens per cleared level sit at 39\u201348k across every arm, so efficiency per generated token is the lever. If medium effort cuts thinking per decision without hurting decision quality, the budget buys more decisions. It should also reduce the 12k-token runaway reasoning-only completions. The sign is not known in advance: xhigh may be what solves the hard levels, so this is strictly an A/B.",
   "evidence": [
    "https://unsloth.ai/docs/models/qwen3.8-next: reasoning_effort levels are xhigh (default), medium, low and none",
    "kaggle-outputs/fz-experts16-full/serve.log server_args: default_chat_template_kwargs={'preserve_thinking': True}, with no effort set",
    "daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py:7210-7232 (_harness_template_kwargs sends preserve_thinking plus a ladder rung only)",
    "https://github.com/jpezzulli/sglang-rtxpro6000/blob/d00d88efc8d6281b12be4f4073126aec95038c55/configs/pennyroyal/reasoning-effort.sh: Pennyroyal pins reasoning_effort=medium by default; its bundled template (configs/pennyroyal/templates/froggeric-v22.5.jinja:217) puts the effort sentence at the start of the system message",
    "arc-agi-3-v3/reports/run-atlas/findings.md \u00a71: about 42k generated tokens per cleared level; no arm went below ~39k"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "Stop runaway reasoning-only completions (\u22487% of all generated reasoning)",
   "category": "bugfix-quickwin",
   "description": "In every Franzen-era run, about 1% of completions (20\u201337 per run) hit max_tokens=12,288 with only reasoning and no tool call. With ARC3_STUB_DEAD_REASONING off, those ~36k-char dead traces are also replayed in context on later requests. Cheapest fix: LOCAL_ANALYZER_MAX_OUTPUT \u2248 9216 (about the p99 of successful completions) plus ARC3_STUB_DEAD_REASONING=1. Better fix, about half a day: on finish_reason=length with no tool call, continue the same assistant turn with the reasoning closed (append '\\n</think>\\n\\n' and use continue_final_message). The already-cached reasoning then produces an action. Do not use SGLang's built-in Qwen3ThinkingBudgetLogitProcessor as-is, for two reasons. It hardcodes Qwen3 ids 151667/151668, whereas Qwen3.8 uses 248068 (<think>) and 248069 (</think>), so a subclass with the right ids would be needed. Also, I found no custom-logit-processor hook in the EAGLE verify code, so it may silently do nothing under NEXTN.",
   "mechanism": "About 300k tokens per 25-game run (~7% of generation) currently end with no action. This change either saves them or turns them into actions. It also keeps 12k-token dead traces out of the context window, which means fewer trims and re-prefills. A smaller max-output reservation adds about 3k usable context per request.",
   "evidence": [
    "kaggle-outputs/*/transcripts [MODEL RESPONSE META] finish_reason=length: 25/2,299 completions (control), 24/2,414 (ledger-on-trim), 23 (temp06), 20 (streams14), 37 (ucb)",
    "Truncated reasoning was 873k of 11.9M reasoning chars in control (7.3%) and 1.27M of 11.8M in ucb (10.8%); each truncated completion is 26\u201341k chars",
    "Successful tool_call completions in control: p50 2,666, p95 17,590, p99 27,167, max 36,891 reasoning chars",
    "daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py:1210-1221: ARC3_STUB_DEAD_REASONING defaults to off; its docstring describes a dead trace 'observed at ~35k characters ... replayed as context'",
    "Pennyroyal python/sglang/srt/sampling/custom_logit_processor.py: Qwen3ThinkingBudgetLogitProcessor uses ids 151667/151668; the local Intel Qwen3.8 tokenizer.json maps <think>=248068 and </think>=248069"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "HiCache host-RAM tier, so 12\u201314 streams no longer cost context",
   "category": "serving",
   "description": "Add `--enable-hierarchical-cache --hicache-size 32 --hicache-host-memory-mode cache --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first` to the launcher (no NIXL or storage backend needed). Then raise MAXREQ / ARC3_MAX_ACTIVE_STREAMS to 12, with CUDAGRAPH_MAXBS 12 and MAMBA_CACHE 72 (about 6\u00d7 streams). Keep LOCAL_ANALYZER_CONTEXT_WINDOW at 128k and the drain at 58k. Smoke-test first, checking two things. (1) The startup log still prints 'Mamba sparse-prefill v2'. Franzen's patch edits unified_radix_cache.py and may not engage on the HiCache radix path. (2) The cached-token fraction stays at or above about 93%. Optional: switch to Albucino's FP8-PLE checkpoint, which the launcher already supports. That frees about 48 GB of host RAM and halves the bytes of the per-token PLE host\u2192GPU gather.",
   "mechanism": "Decode throughput keeps rising with batch size: bs10 793 \u2192 bs13 882 tok/s, and the streams14 run gained 15% harness tok/s. Both multi-stream runs, however, had to shrink per-game context to fit the ~1.01M-token GPU KV pool, which cancelled the gain. With a host tier, prefixes of streams parked in tool calls or behind the priority gate spill to RAM. They come back over PCIe (60k tokens \u00d7 ~12.4 KB \u2248 0.75 GB, well under a second) instead of a ~5 s re-prefill. More streams then stop costing context, and priority-gate handovers get cheaper.",
   "evidence": [
    "arc-agi-3-v3/reports/run-atlas/findings.md \u00a74: 14 streams gave 703 vs 611 harness tok/s with context cut 128k\u219296k (score flat); ledger+12 streams gave 644 tok/s with context cut to 110k (48.08 vs 52.56)",
    "kaggle-outputs/fz-streams14/serve.log: decode median bs13 882 and bs14 850, vs bs10 793 in control",
    "kaggle-outputs/fz-arms-v4-control/fz-arms.log: host RAM 106\u2013117 of 176.9 GB used throughout the run (about 60 GB free)",
    "fz-arms-v4-control/serve.log: KV pool 1,011,264 tokens; target K+V 11.58 GB plus draft 0.96 GB (\u224812.4 KB/token)",
    "https://github.com/jpezzulli/sglang-rtxpro6000/blob/d00d88efc8d6281b12be4f4073126aec95038c55/configs/pennyroyal/serve-flash-next-frspec.sh: Pennyroyal's Flash-Next recipe uses these exact HiCache flags (32 GB host tier); MEMORY-AND-PERSISTENCE.md says the host tier persists KV + GDN + QSA keys",
    "https://github.com/gabrielolympie/sglang-flashnext-sm120: '--max-mamba-cache-size must be ~6\u00d7 max-running-requests or the speculative CUDA graphs silently cap at bs=4'",
    "https://docs.sglang.io/docs/advanced_features/hicache_design (GPU L1 / host L2 hierarchy)",
    "daniel-franzen-sol/serving/patches/patch-sglang-mamba-final-prefill-v2.py: the policy is scoped to the FULL+MAMBA extra_buffer path in unified_radix_cache.py"
   ],
   "effort": "1 day",
   "expected_impact": "medium",
   "status_in_franzen": "present-but-off"
  },
  {
   "title": "FP8 weight-only dense layers that speed up decode at bs10 (fixing the MXFP8 regression)",
   "category": "serving",
   "description": "The user's online-MXFP8 test (SGLANG_SM120_ONLINE_MXFP8=1) freed 4.6 GB and grew KV from 1.01M to 1.36M tokens, but cut full-batch decode by 27%. That path uses FlashInfer CUTLASS MXFP8 with dynamic activation quantization, which Pennyroyal tuned and measured at single-request M\u22484. Proposed fix: port Gabriel's patch 0005 (W8A16 weight-only FP8 with per-row scales and a split-free Triton kernel at 89% of the halved-bytes floor) and 0006 (FP8 HC mix and lm_head) onto Pennyroyal, with three changes. Replace the BF16 weights rather than keeping BF16+FP8 copies (Gabriel's version costs about +3.3 GB). Extend the small-M path to M \u2264 64. Serve large-M prefill with a Triton W8A16 tiled GEMM that dequantizes in the K loop. Validate on greedy/needle checks, then a 25-game arm. Side note: the notebook still exports SGLANG_SM120_LOWM_FP8_WEIGHT=0 and SGLANG_SM120_LM_HEAD_FP8=0. Those are Gabriel's env names and are not defined in Pennyroyal v2.5.3's environ.py, so they do nothing.",
   "mechanism": "This gives two separate wins. First, about 4.3 GB fewer bytes per verify step (the dense BF16 stack is about 8.7 GB per step), for roughly +7\u201310% decode at bs10. Second, about 4.3 GB of VRAM freed, worth roughly +350k KV tokens (+35%). That is enough for 13\u201314 streams at the unchanged 128k context, a configuration the atlas never tested because both multi-stream runs cut context.",
   "evidence": [
    "https://github.com/gabrielolympie/sglang-flashnext-sm120/blob/main/patches/0005-sm120-fp8-weight-only.patch: 'C1 greedy median 147.9 -> 163.5 tok/s, C4 428 -> 477 ... (mem-fraction 0.96 to hold ~3GB of fp8 copies) ... accept length unchanged'",
    "https://github.com/gabrielolympie/sglang-flashnext-sm120/blob/main/docs/PERF_CEILING.md: the bf16 dense stack is 85% of per-step traffic at bs1; FP8 weight-only quality risk described as low (W8)",
    "arc-agi-3-v3/reports/run-atlas/findings.md \u00a76: MXFP8 'frees 4.6 GB (weights 69.85 \u2192 65.28 GB), KV pool 1.01M \u2192 1.36M tokens, but decode \u221227% at full batch (555 vs 765 tok/s)'",
    "https://github.com/jpezzulli/sglang-rtxpro6000/blob/d00d88efc8d6281b12be4f4073126aec95038c55/BACKENDS.md: online FP8 = FlashInfer CUTLASS MXFP8 weights + dynamic activations; README.md reports +28.3% single-request decode",
    "https://github.com/mratsim/sglang-qwen38fn-sm120-turbo: NVFP4 + MXFP8-at-load reaches about 1170 tok/s aggregate at 8 concurrent requests, so the regression looks specific to this configuration rather than inherent to FP8",
    "arc-agi-3-v3/franzen-moe/kaggle/template.ipynb cell 12 env: SGLANG_SM120_LOWM_FP8_WEIGHT=0 and SGLANG_SM120_LM_HEAD_FP8=0; neither appears in Pennyroyal d00d88e python/sglang/srt/environ.py"
   ],
   "effort": "2-3 days",
   "expected_impact": "medium",
   "status_in_franzen": "present-but-off"
  },
  {
   "title": "Cap the experts loaded per verify step across the batch (XShare / MoE-Spec style)",
   "category": "serving",
   "description": "Behind an env flag, change MoE top-k selection for the target-verify forward only (prefill stays untouched). In each layer, sum gate scores per expert over the batch's 40 tokens and keep the top-B experts, with B around 160\u2013200 of 512. Then rerun top-10 for each token over that allowed set (mask the rest to \u2212inf) and renormalize. Everything is static-shape (scatter_add, top-k over 512), so it can be captured inside the existing verify CUDA graphs. Sweep B on 20-min throughput runs, then run a 25-game arm at the best B.",
   "mechanism": "At 40 tokens per step the union of top-10 selections reads roughly half of all expert weights, and that read dominates step time. Capping the union lets tokens reuse experts that other tokens already pulled in. That cuts expert bytes by about 20\u201335%, while each token keeps its highest-scoring experts within the shared set. It is a larger lever than a fixed top-8, at somewhat more quality risk.",
   "evidence": [
    "https://arxiv.org/abs/2602.07265 (XShare): batch-aware expert selection, training-free; up to 30% fewer expert activations and up to 14% throughput gain in speculative decoding",
    "https://arxiv.org/abs/2602.16052 (MoE-Spec): a training-free per-layer expert budget at verification time gives 10\u201330% higher throughput than EAGLE-3 at comparable quality",
    "kaggle-outputs/fz-experts16/serve.log vs fz-tput-base: +60% active experts \u2192 \u221225% bs10 decode, which shows expert bytes dominate the step",
    "https://github.com/gabrielolympie/sglang-flashnext-sm120/blob/main/docs/PERF_CEILING.md bytes table: routed experts 72.98 GB on disk, 1.43 GB per token at top-10"
   ],
   "effort": "2-3 days",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Rebuild the FR-Spec draft vocabulary from the user's own ARC transcripts",
   "category": "serving",
   "description": "The 64k hot-token map (hot_tokens_64k.pt, sha becfa41\u2026) consists of the 32,768 lowest token ids plus the most frequent tokens from SGLang's own Python source (1.78M tokens). Run Pennyroyal's scripts/pennyroyal/frspec/build_token_map.py with --corpus-root pointing at the generated text (THINKING + tool-call code) from the existing run transcripts. Keep size 65536, or use 32768 for a cheaper draft head. Upload it as a dataset, update the TOKEN_MAP_SHA assert in the launcher (template.ipynb cell 12), and check accept len in serve.log.",
   "mechanism": "The draft can never propose a token outside the map, so each such token ends the draft chain. I measured the control run: 1.26% of its 4.58M generated tokens fall outside the map. The misses are dominated by tokens the draft would predict easily: 'Hmm', ' Hmm' and ' hmm' alone account for 23.5k occurrences, followed by 'Alternatively', '\u2713', '.ascii' and ' BFS'. The whole workload uses only 16,433 distinct tokens, so a workload-built map covers essentially everything. Expect about +1\u20133% accept length. It is lossless, because verification is unchanged.",
   "evidence": [
    "Measured with the local Intel tokenizer (tokenizers lib) on kaggle-outputs/fz-arms-v4-control/transcripts/*.txt: 57,893 of 4,582,008 generated tokens (1.26%) are outside the map; top misses 'Hmm' 10,127, ' Hmm' 9,769, ' hmm' 3,648, ' bead' 1,818, 'Alternatively' 1,344",
    "https://github.com/jpezzulli/sglang-rtxpro6000/blob/d00d88efc8d6281b12be4f4073126aec95038c55/configs/pennyroyal/frspec/flash-next-64k.manifest.json: base_count 32768; corpus = python/sglang/srt/** source, 1,780,923 tokens; map_sha256 becfa41\u2026 (matches the notebook's TOKEN_MAP_SHA)",
    "https://github.com/jpezzulli/sglang-rtxpro6000/blob/d00d88efc8d6281b12be4f4073126aec95038c55/scripts/pennyroyal/frspec/build_token_map.py (deterministic map builder from a text corpus)",
    "https://github.com/gabrielolympie/sglang-flashnext-sm120 serve.sh: adding FR-Spec gave C1 200.8\u2192219.4 and C4 541\u2192592 tok/s with accept length unchanged, so the map affects both draft cost and coverage"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "Lossless fallback: rejection-sampling verification made compatible with FR-Spec",
   "category": "serving",
   "description": "The fork already implements Leviathan-style chain rejection sampling for NEXTN/EAGLE with topk=1 (--speculative-use-rejection-sampling): the draft samples x~q at the request temperature and the verifier accepts with min(1, p/q). It currently refuses a reduced draft vocabulary; the FIXME in eagle_worker_v2.py says to scatter draft probs into the target vocab via the d2t map. It also refuses accept thresholds. Implement that scatter (about 20 lines: probs_full = zeros(bs, V); probs_full[:, hot_token_id] = probs_hot) where the draft probabilities are built, then A/B against the baseline. Use this path if the relaxed thresholds hurt score, since it leaves the output distribution exact.",
   "mechanism": "Target-only verification at T=0.7 accepts a greedy draft token with probability p_target(x). Rejection sampling accepts with \u03a3 min(p, q), which is higher when draft and target spread their mass over the same few tokens. There is no quality risk. The gain is uncertain (my estimate is 0\u20138% accept length), so it ranks below the relaxed thresholds. Running it without FR-Spec is not worth it: a full-vocab draft head adds about 1 GB of reads to each of the 3 draft steps.",
   "evidence": [
    "Pennyroyal d00d88e python/sglang/srt/arg_groups/speculative_hook.py lines 791-836: rejection sampling is supported only for EAGLE/NEXTN with topk=1 and raises if accept thresholds are not 1.0",
    "Pennyroyal d00d88e python/sglang/srt/speculative/eagle_worker_v2.py lines 228-240: raises 'requires the draft and target to share one vocab' with a FIXME to scatter via d2t",
    "Pennyroyal python/sglang/srt/speculative/spec_utils.py renorm_draft_probs / sample_draft_proposal: temperature-scaled q 'so the draft proposal q tracks the target sampling temperature (higher acceptance)'",
    "kaggle-outputs/fz-arms-v4-control/serve.log accept 2.69 (sampled, lossless) vs 2.81 (greedy, run-atlas \u00a76)"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "present-but-off"
  },
  {
   "title": "Fill the CUDA-graph batch-size gaps (3, 5, 6 and any new stream count)",
   "category": "bugfix-quickwin",
   "description": "The launcher captures verify, draft-decode and draft-extend graphs only for bs [1,2,4,7,8,9,10]. A batch of 5 or 6 therefore pads to 7 (3 pads to 4) and runs 28 verify tokens instead of 20\u201324, touching more expert weights. Set graph_bs = list(range(1, MAXREQ+1)) in template.ipynb cell 12. The cost is slightly longer capture and somewhat more graph memory; the three graph families currently use 0.48 + 0.37 + 0.29 GB.",
   "mechanism": "About 5% of decode steps in control were at bs 5\u20136, and more occur in the end-game when few games remain. Padding inflates those steps by roughly 10\u201325%. The overall gain is under 1%, but it costs nothing and matters more if the stream count changes (12 and 14 must be in the list).",
   "evidence": [
    "arc-agi-3-v3/franzen-moe/kaggle/template.ipynb cell 12: graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, MAXREQ, CUDAGRAPH_MAXBS})",
    "kaggle-outputs/fz-arms-v4-control/serve.log: 241 of 4,803 decode lines at bs 5\u20136; capture lines show cuda_graph bs [1,2,4,7,8,9,10] with mem usage 0.48/0.37/0.29 GB"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "Cut server cold start (~10 min) and audit the game-time budget",
   "category": "other",
   "description": "In control, the server was READY after 595 s: weight load took 386 s and target-verify CUDA-graph capture took 145 s, mostly first-time Triton/FlashInfer JIT (2:03 on the first batch). The benchmark started at 667 s. Persist the compile caches (TRITON_CACHE_DIR, FLASHINFER_WORKSPACE_BASE, SGLANG_JIT_CACHE_DIR and TORCHINDUCTOR_CACHE_DIR under /tmp/sgl-intel/cache) from one run into a Kaggle dataset, and copy them in before launch; this only works with the same image, driver and GPU. Then confirm the competition's rerun time cap. With TRUE_SUBMISSION the per-game deadline is started_at + 532 min and soft_end=None, so startup time saved only becomes play time if max_runtime_s_per_game is raised to match, and only if the platform cap allows it.",
   "mechanism": "Each extra minute of play is about 0.19% more tokens over 532 min. Shaving 2\u20133 min of JIT is worth about 0.5%. It also lowers the risk that the 12-min SERVER_STARTUP_TIMEOUT releases games while the server is still loading.",
   "evidence": [
    "kaggle-outputs/fz-arms-v4-control/serve.log: 'Load weight end. elapsed=386.02 s'; 'Capture target verify CUDA graph end. elapsed=144.80 s'",
    "kaggle-outputs/fz-arms-v4-control/fz-arms.log: 'READY after 595s' at t=665 s; 'Starting benchmark...' at t=667 s",
    "arc-agi-3-v3/franzen-moe/kaggle/template.ipynb cell 16 (TRUE_SUBMISSION: max_runtime_s_per_game = 532*60) and cell 20 (soft_end None when TRUE_SUBMISSION); cell 4 SERVER_STARTUP_TIMEOUT = 12*60"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  }
 ]
}
```
