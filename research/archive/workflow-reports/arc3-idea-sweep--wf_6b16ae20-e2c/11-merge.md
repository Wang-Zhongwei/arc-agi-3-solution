# arc3-idea-sweep: merge

Workflow run `wf_6b16ae20-e2c`, phase Merge. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "ideas": [
  {
   "id": "relaxed-mtp-accept",
   "title": "Relaxed MTP acceptance thresholds (single/acc 1.0 -> 0.5 -> 0.3)",
   "category": "serving",
   "description": "Change CFG SPEC_ACCEPT_SINGLE / SPEC_ACCEPT_ACC (template.ipynb cell 12, passed to --speculative-accept-threshold-single/-acc), both pinned at lossless 1.0. Arms: (a) single=0.5, acc=1.0; (b) single=0.3, acc=0.3 (Gabriel Olympie's shipped setting); optional mild (0.6/0.8) and aggressive (0.3/0.5) variants. Compare serve.log accept len (baseline 2.69), harness tok/s, 25-game score/trimmed mean, and degeneracy/repetition counters. Cannot combine with --speculative-use-rejection-sampling.",
   "mechanism": "Verify cost per step is fixed, so every extra accepted draft token is nearly free. Forcing acceptance when target p >= tau raises accept length; Gabriel measured +13% decode at 0.5 and +29% at 0.3 (single stream, temp 0.6). It sharpens sampling like a slightly lower temperature, and temp 0.6 vs 0.7 showed no score change. Batch-10 gain will be smaller than single-stream.",
   "evidence": [
    "gabrielolympie/sglang-flashnext-sm120 serve.sh:73-85 and STATUS.md:50-52: ladder 1.0=179, 0.5=203, 0.3=231 tok/s; 0.3 passes quality gates",
    "Pennyroyal server_args.py:2241-2250 threshold semantics",
    "template.ipynb cell 12 / nb.txt:10422-10423: both thresholds 1.0; build_bundle_pennyroyal.sh:1411 deliberate lossless choice, never ablated",
    "run-atlas findings.md section 6: greedy accept 2.67->2.81; temp06 45.92 vs control 46.25",
    "ARC-Decode (icml.cc poster 66788): relaxed acceptance up to 1.6x under sampling"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "present-but-off",
   "merged_from": [
    "Relaxed MTP acceptance 1.0->0.5->0.3",
    "Relax speculative-decoding acceptance thresholds",
    "Relaxed MTP acceptance (SPEC_ACCEPT 0.3)"
   ]
  },
  {
   "id": "experts-top8",
   "title": "Fewer routed experts (top-10 -> top-8) on target only",
   "category": "model-config",
   "description": "Use TEXT_CONFIG_OVERRIDE num_experts_per_tok=8 on the target view only (MTP draft stays 10). arms.json already defines an unrun 'experts8' arm. Run 20-min throughput check then the full 25-game arm. Fallback: dynamic per-token pruning by renormalized gate weight < ~0.03-0.05 with floor 6, or per-layer k (LExI).",
   "mechanism": "Expert weight reads dominate the ~33 ms step; top-8 touches ~47% vs ~55% of experts, ~15% fewer bytes. experts16 measured slope (+60% experts = -25% decode) interpolates to about +6-10% decode.",
   "evidence": [
    "fz-experts16 serve.log: bs10 592 vs 770/793 tok/s; full run 38.69 vs 46.25",
    "arms.json experts8 arm defined, never run",
    "arXiv 2505.03531; 2509.02753 (LExI); 2510.12357 (MoBiLE: halving costs 8.5-11.6%)"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "lowm-gemm-max-m",
   "title": "Raise low-M Triton BF16 GEMM _MAX_M from 32 to 64",
   "category": "bugfix-quickwin",
   "description": "Franzen's patch 0004 only uses the Triton kernel for m<=32, but verify at bs 9-10 has m=36-40, so BF16 projections fall back to cuBLAS. sed _MAX_M=64 in the installed sm120_lowm_bf16_gemm.py, confirm shapes with SGLANG_BF16_GEMM_LOG_SHAPES=1, A/B bs10 decode on a 20-min arm; add tuned (40,n,k) tactics if it helps.",
   "mechanism": "cuBLAS reaches only 30-50% of DRAM BW on mid shapes vs ~90% for Triton; ~8.7 GB BF16 read per step, so about 2-3 ms of ~33 ms (+5-8% decode), no quality impact. Uncertain.",
   "evidence": [
    "build_bundle_pennyroyal.sh:102 _MAX_M = 32",
    "fz-arms-v4-control serve.log: 3,485 of 4,803 decode lines at bs 9-10",
    "PERF_CEILING.md: mid shapes 30-50% BW, Triton 90%"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "partial",
   "merged_from": []
  },
  {
   "id": "reasoning-effort-medium",
   "title": "Set reasoning_effort=medium explicitly (default is xhigh)",
   "category": "model-config",
   "description": "Add reasoning_effort=medium to --default-chat-template-kwargs, constant for the run so the prefix stays cached; A/B on 25 games. Check chat_template.jinja honours it and where it injects the effort sentence; keep ARC3_REASONING_EFFORT_LADDER off because effort text at top of system block invalidates the whole cached prefix on each change.",
   "mechanism": "Tokens per cleared level sit at 39-48k across arms; cutting thinking per decision could buy more decisions and reduce 12k runaways. Sign unknown.",
   "evidence": [
    "unsloth.ai qwen3.8-next: xhigh default, medium, low, none",
    "fz-experts16-full serve.log: default kwargs only preserve_thinking",
    "tool_agent.py:7210-7232 ladder only",
    "Pennyroyal reasoning-effort.sh pins medium; froggeric template puts effort sentence at start of system message"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "partial",
   "merged_from": []
  },
  {
   "id": "reasoning-effort-low",
   "title": "Global reasoning_effort=low A/B",
   "category": "model-config",
   "description": "Run one arm with default-chat-template-kwargs reasoning_effort=low (constant per game, cache-neutral). 'medium' only removes the effort sentence (GB10: no shortening); 'low' injects 'keep thinking brief'. Keep the ladder off; if dynamic, switch only at drain/handover boundaries.",
   "mechanism": "If low cuts tokens per decision 20-40% at equal decision quality, more turns per budget. Model card warns low effort can hurt multi-turn agentic completion.",
   "evidence": [
    "HF Qwen3.8-Flash-Next chat_template.jinja: effort text at start of first system message",
    "HF model card warning on lower effort",
    "tool_agent.py:1778-1797, 7221-7232",
    "GB10-OVERNIGHT-RESULTS.md:47: medium does not shorten thinking"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "partial",
   "merged_from": []
  },
  {
   "id": "thinking-budget-forcing",
   "title": "Thinking budget / forced action instead of 12k-token reasoning-only truncations",
   "category": "model-config",
   "description": "About 1% of replies (20-37 per run, 184 over 17.4k) hit max_tokens=12,288 with only reasoning, ~7% of generated reasoning. Tier 0: LOCAL_ANALYZER_MAX_OUTPUT ~8192-9216 plus ARC3_STUB_DEAD_REASONING (note: transcripts show model acts on the kept reasoning, so stubbing is debatable) and a 'previous reply hit the limit, act now' opener line. Tier 1 (~1 day): cap stage 1 at ~6-8k, on length cut with no tool call send a continuation with reasoning closed (append Qwen early-stop text + </think>, continue_final_message=true, max_tokens ~2k) so the cached reasoning yields a tool call; reuse _recover_tool_calls_from_markup. Do not rely on SGLang thinking_budget/Qwen3ThinkingBudgetLogitProcessor (ids 151667/151668 vs Qwen3.8 248068/248069; not applied in EAGLE verify).",
   "mechanism": "~300k tokens/run (~7%) produce no action and put 12k-token blobs in context (earlier trims, more re-prefill). Budget forcing (s1) converts them to executed actions. Cutting at 8k saves ~4k tokens per truncation (~3% generation) at the cost of continuation requests for replies finishing in 8-12k.",
   "evidence": [
    "fz transcripts: finish_reason=length 25/2,299 control, 24, 23, 20, 37 (ucb); truncated 7.3% of reasoning chars",
    "successful replies p99 27k chars; 173 of 184 truncations followed by tool call",
    "tool_agent.py:1210-1221, 9410, 9472, 9570",
    "sglang issue 25536; arXiv 2501.19393 (s1); Qwen docs two-stage budget recipe"
   ],
   "effort": "1 day",
   "expected_impact": "medium",
   "status_in_franzen": "partial",
   "merged_from": [
    "Stop runaway reasoning-only completions",
    "Thinking budget-forcing",
    "Cap overlong deliberations earlier",
    "Two-stage thinking budget",
    "Per-request thinking cap with forced tool call"
   ]
  },
  {
   "id": "hicache-host-tier",
   "title": "HiCache host-RAM tier (with optional FP8-PLE) to enable 12-14 streams without cutting context",
   "category": "serving",
   "description": "Add --enable-hierarchical-cache --hicache-size 32 (48 with FP8 PLE, albucino checkpoint frees ~48 GB host RAM) --hicache-write-policy write_through --hicache-io-backend kernel (--hicache-host-memory-mode cache, page_first). Raise MAXREQ/MAX_ACTIVE_STREAMS/CUDAGRAPH_MAXBS to 12-14 with MAMBA_CACHE ~6x streams. Arms: (a) more streams at 128k; (b) 10 streams, shallower drain (~40Ki). Smoke-test that 'Mamba sparse-prefill v2' still engages on the HiCache path and cached fraction >=93%. Also check FP4 KV dtype support. Avoid expert pruning for memory.",
   "mechanism": "Decode rises with batch (bs10 793 -> bs13 882) but multi-stream runs had to cut context to fit the ~1.01M KV pool. A host tier spills parked-stream prefixes and reloads over PCIe (~0.75 GB, <1 s vs ~5 s re-prefill). Siriki's largest jump (14.49->22.53) came from freeing KV for longer history; LHS served 14 streams from 1.417M pool.",
   "evidence": [
    "findings.md section 4: 14 streams 703 vs 611 tok/s with context cut, score flat",
    "Pennyroyal serve-flash-next-frspec.sh uses these HiCache flags",
    "Kaggle discussion 744792 table; Siriki notebook flags and writeup",
    "patch-sglang-mamba-final-prefill-v2.py scoped to extra_buffer path",
    "fz-arms-v4-control: host RAM 106-117 of 176.9 GB, KV pool 1,011,264 tokens"
   ],
   "effort": "1 day",
   "expected_impact": "medium",
   "status_in_franzen": "present-but-off",
   "merged_from": [
    "HiCache host-RAM tier",
    "SGLang hierarchical cache on host RAM freed by FP8 PLE"
   ]
  },
  {
   "id": "fp8-weight-only-dense",
   "title": "FP8 weight-only dense stack, FP8 HC mix and lm_head (Gabriel patches 0005/0006) on Pennyroyal",
   "category": "serving",
   "description": "Port W8A16 weight-only FP8 with per-row scales and split-free Triton kernel (patch 0005) and FP8 HC mix/lm_head (0006) onto Pennyroyal v2.5.3. Replace BF16 weights rather than keep copies (+3.3 GB otherwise), extend small-M path to M<=64, use Triton W8A16 tiled GEMM for large-M prefill. Validate greedy/needle gates then 25-game arm. Note SGLANG_SM120_LOWM_FP8_WEIGHT/LM_HEAD_FP8 env vars in the notebook do nothing on Pennyroyal.",
   "mechanism": "~4.3 GB fewer bytes per verify step of ~8.7 GB dense BF16 (+7-10% decode at bs10) and ~4.3 GB VRAM freed (~+350k KV tokens), enough for 13-14 streams at 128k. The user's MXFP8 loss (-27%) came from dynamic activation quantization, not weight-only.",
   "evidence": [
    "0005 patch: C1 147.9->163.5, C4 428->477 tok/s, accept unchanged",
    "PERF_CEILING.md: dense BF16 stack 85% of bs1 traffic",
    "findings.md section 6: MXFP8 frees 4.6 GB but decode -27%",
    "Intel W4A16 keeps attn/linear_attn/shared/HC/lm_head in BF16",
    "mratsim turbo repo: MXFP8-at-load ~1170 tok/s at 8 concurrent"
   ],
   "effort": "2-3 days",
   "expected_impact": "medium",
   "status_in_franzen": "present-but-off",
   "merged_from": [
    "FP8 weight-only dense layers (fix MXFP8 regression)",
    "FP8 weight-only dense stack and FP8 lm_head/HC"
   ]
  },
  {
   "id": "batch-aware-expert-routing",
   "title": "Batch-aware expert routing (XShare / MoE-Spec / Opportunistic Expert Activation)",
   "category": "serving",
   "description": "Behind an env flag, change target-verify top-k only (static shapes, CUDA-graph safe). Variant A (XShare/MoE-Spec): sum gate scores per expert over the ~40 batch tokens, keep top-B experts (B~160-200 of 512), rerun top-10 per token over that set, renormalize. Variant B (OEA): take union of each token's top-k0 (k0~6), fill remaining slots up to 10 from the union, renormalize. Sweep on 20-min throughput runs, then a 25-game arm.",
   "mechanism": "At 34-40 tokens per step the union reads ~half of all experts and dominates step time; reuse of already-loaded experts cuts expert bytes 20-35% while keeping 10 experts per token (OEA paper: -39% MoE decode latency on Qwen3-30B at batch 16). Larger lever than fixed top-8, more quality risk.",
   "evidence": [
    "arXiv 2602.07265 XShare (up to 14% throughput in spec decoding)",
    "arXiv 2602.16052 MoE-Spec (10-30% over EAGLE-3)",
    "arXiv 2511.02237 OEA",
    "experts16 slope: +60% experts -> -25% decode",
    "fz-arms-v4-control serve.log: mean running-req 8.56, accept 2.69"
   ],
   "effort": "2-3 days",
   "expected_impact": "medium",
   "status_in_franzen": "absent",
   "merged_from": [
    "Cap experts loaded per verify step (XShare/MoE-Spec)",
    "Batch-aware expert routing instead of static top-k change"
   ]
  },
  {
   "id": "frspec-map-from-transcripts",
   "title": "Rebuild FR-Spec hot-token map from ARC transcripts",
   "category": "serving",
   "description": "Run Pennyroyal build_token_map.py (or Gabriel's make_hot_tokens.py) with corpus = generated text (thinking + python) from run transcripts (lordhansolo shipped a 32k map from 125 transcripts). Keep 65536 or try 32768, upload as dataset, update TOKEN_MAP_SHA assert, check accept len and decode tok/s.",
   "mechanism": "Draft can never propose out-of-map tokens; 1.26% of control's 4.58M generated tokens fall outside (Hmm/ Hmm/ hmm 23.5k, Alternatively, BFS). Workload uses only 16,433 distinct tokens. Expect +1-3% accept length, lossless; a smaller map could cut draft lm_head cost.",
   "evidence": [
    "Measured miss rate 57,893 of 4,582,008 on fz-arms-v4-control",
    "flash-next-64k.manifest.json: corpus is sglang python source",
    "lordhansolo configs/draft_vocab_32k.json",
    "Gabriel serve.sh: FR-Spec +9% decode vs full vocab"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "partial",
   "merged_from": [
    "Rebuild FR-Spec draft vocabulary from own ARC transcripts",
    "Rebuild FR-Spec hot-token map (smaller draft vocab)"
   ]
  },
  {
   "id": "rejection-sampling-frspec",
   "title": "Lossless rejection-sampling verification compatible with FR-Spec",
   "category": "serving",
   "description": "Implement the FIXME in eagle_worker_v2.py: scatter hot-vocab draft probs into the full target vocab via the d2t map (~20 lines), enabling --speculative-use-rejection-sampling with FR-Spec, then A/B. Use as lossless fallback if relaxed thresholds hurt score.",
   "mechanism": "Rejection sampling accepts with sum min(p,q), higher than target-only acceptance; no quality risk; estimated 0-8% accept length gain.",
   "evidence": [
    "speculative_hook.py:791-836 and eagle_worker_v2.py:228-240",
    "spec_utils.py renorm_draft_probs",
    "accept 2.69 sampled vs 2.81 greedy"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "present-but-off",
   "merged_from": []
  },
  {
   "id": "cudagraph-bs-gaps",
   "title": "Fill CUDA-graph batch-size gaps (3,5,6 and new stream counts)",
   "category": "bugfix-quickwin",
   "description": "Set graph_bs = list(range(1, MAXREQ+1)) in template cell 12 (currently [1,2,4,7,8,9,10]); must include 12/14 if streams change. Cost: slightly longer capture and graph memory (0.48+0.37+0.29 GB today).",
   "mechanism": "~5% of decode steps at bs 5-6 pad to 7 (28 verify tokens vs 20-24), +10-25% on those steps; overall <1% gain but free.",
   "evidence": [
    "template.ipynb cell 12 graph_bs",
    "serve.log: 241 of 4,803 decode lines at bs 5-6"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "partial",
   "merged_from": []
  },
  {
   "id": "startup-and-deadline",
   "title": "Cut server cold start (~10 min) and anchor per-game deadline to notebook start",
   "category": "other",
   "description": "Persist TRITON/FLASHINFER/SGLANG_JIT/TORCHINDUCTOR caches from one run into a dataset and copy in before launch (same image/GPU only); start shard precaching in parallel with wheel install (cell 6), drop '--warmups structured_output', reuse autotune cache. In TRUE_SUBMISSION replace fixed max_runtime_s_per_game=532*60 with NOTEBOOK_START + LIMIT - ~5 min margin - now, after confirming the competition's actual runtime cap.",
   "mechanism": "Weight load 386 s + graph capture 145 s; benchmark starts at 667 s. Each extra play minute ~0.19% more tokens; saved startup only becomes play time if deadline is anchored. Also reduces risk of the 12-min startup timeout.",
   "evidence": [
    "fz-arms-v4-control serve.log: load 386 s, capture 144.8 s; READY after 595 s",
    "template cell 16 (532*60), cell 20 soft_end None, cell 4 SERVER_STARTUP_TIMEOUT",
    "simulator: +2% minutes ~ +0.8 points (upper bound)"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "partial",
   "merged_from": [
    "Cut server cold start and audit game-time budget",
    "Anchor per-game deadline to notebook start and shave startup"
   ]
  },
  {
   "id": "mamba-state-precision",
   "title": "Mamba/GDN state: fp16 + stochastic rounding instead of bf16 RTN",
   "category": "model-config",
   "description": "Arm A: --mamba-ssm-dtype float16 --enable-mamba-cache-stochastic-rounding (same footprint). Confirm boot log shows SR covers GDN writes and smoke-test fp16 overflow. Fallback B: float32 (doubles state memory, competes with KV).",
   "mechanism": "36 GDN layers re-round state to 7 mantissa bits every step over 58-128k contexts; Nemotron 3 Ultra report finds fp16+SR preserves FP32 accuracy and verbosity. Zero throughput cost; effect uncertain.",
   "evidence": [
    "nb.txt:10418 SSM_DTYPE bfloat16; serve.log server_args SR False",
    "server_args.py:2640-2665, 6737-6770",
    "arXiv 2606.15007 sec 4.5"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "trim-time-reasoning-compaction",
   "title": "Trim-time reasoning compaction (strip old <think> only when drain fires)",
   "category": "memory-context",
   "description": "At drain, delete reasoning_content from assistant messages except the most recent K (6-8), keeping tool calls, code, results and prompts; then drop whole messages only if still over target. Implement in _persistent_history_messages (~8737) / _trim_messages_for_context when force_drain. New env ARC3_TRIM_KEEP_REASONING_TURNS; test alone and with ledger-on-trim.",
   "mechanism": "Past thinking is ~55-60% of retained context (5.2k chars/request vs 0.57k code); compaction at trim time keeps cache intact between trims yet lets the 58k block hold ~2x more action/observation turns and allows smaller drain. Differs from failed 'remove all thinking'.",
   "evidence": [
    "fz-arms-v4-control transcripts: reasoning mean 5,182 chars vs code 570, tool result 620",
    "WRITEUP.md:69, :97",
    "ledger-on-trim 52.56 vs 46.25",
    "GB10 token-efficiency notes: dropping ALL reasoning every request caused re-deriving"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "drain-time-image-stripping",
   "title": "Drop old board images only at drain time (cache-friendly history image window)",
   "category": "memory-context",
   "description": "ARC3_HISTORY_IMAGE_KEEP is applied every commit (tool_agent.py ~8737-8741, _apply_history_image_window :1309), invalidating the prefix each turn. Apply it only in the commit where token trim/drain fires (token_trim_dropped/force_drain), keeping the last K~4-6 image-bearing messages with the existing placeholder via _strip_message_images, then re-run the trim. Optionally combine with reasoning compaction.",
   "mechanism": "Images are ~10-20% of prompt tokens (402 each at 10x, ~1.6 per prompt). Removing old ones at trim time frees ~5-7k tokens per drain for text history with no cache cost since the prefix is rebuilt anyway.",
   "evidence": [
    "tool_agent.py:1227-1246, 8722-8771",
    "CONFIGURATION.md:94-96 warns image filtering hurts prefix reuse",
    "control transcripts calibration: images 10,050 of 84,483 prompt tokens",
    "fz transcripts: 4,927 images over 3,094 user prompts"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "present-but-off",
   "merged_from": [
    "Make ARC3_HISTORY_IMAGE_KEEP cache-friendly",
    "Strip old images only when a drain fires",
    "Trim-synchronized image stripping",
    "Strip old board images only when history is trimmed"
   ]
  },
  {
   "id": "resume-turn-diet",
   "title": "Resume-turn diet: ARC3_RESUME_PROMPT_IMAGES=0, short resume prompt, higher yield threshold",
   "category": "memory-context",
   "description": "Config-only sweep: ARC3_RESUME_PROMPT_IMAGES=0, ARC3_YIELD_RESUME_PROMPT=short, LOCAL_ANALYZER_YIELD_TOKENS 2048 vs 3072/6144/0. A resume only happens when nothing executed so the re-attached board is identical to one already in context. Cache-neutral (only new messages change). ARC3_PRUNE_RESUME_PROMPTS changes the prefix.",
   "mechanism": "31% of turns end in turn_token_budget yield (466 of 1,560 in control; 3,690 across runs), each adding ~750 text tokens plus a duplicate ~402-token image, ~190k context tokens per run, pushing trims sooner.",
   "evidence": [
    "fz transcripts: 434 'executed NO actions' openers mean 2,539 chars with 1 image",
    "tool_agent.py:1249-1260, 9213-9245, 9830",
    "WRITEUP.md:299 resume variants inconclusive"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "present-but-off",
   "merged_from": [
    "Resume-turn diet: stop re-attaching board image",
    "Stop re-sending duplicate context on yield-resumes"
   ]
  },
  {
   "id": "image-token-hygiene",
   "title": "Skip duplicate boards, HUD-only diff images; cheaper diff images",
   "category": "memory-context",
   "description": "(a) In _build_user_message skip the current-grid image when board is identical to the last image in context ('Board unchanged since last image'); (b) skip diff image when only border-band cells changed; (c) ARC3_DIFF_IMAGE_UPSCALE (e.g. 4x = 66 tokens) for diff images.",
   "mechanism": "~30% of openers re-attach an unchanged 402-token board (~185k image tokens per 25 games); full-size diffs cost another 402. Frees 5-10% of trim window and vision-encoder work.",
   "evidence": [
    "461 of 1,560 openers are no-execution and carry an image; ~79 of 912 diff images followed a single NO-OP",
    "tool_agent.py:5788-5799, 6573-6615"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "partial",
   "merged_from": []
  },
  {
   "id": "upscale-patch-aligned",
   "title": "Patch-aligned board upscale: A/B MULTIMODAL_UPSCALE 8x and 16x vs 10x",
   "category": "perception",
   "description": "Vision tower uses 16-px patches with 2x2 merge (32 px per token). At 10x each patch covers 1.6 cells. Arms: 8x (512 px, 258 tokens, 2x2 cells per patch), 16x (1024 px, 1026 tokens, 1 cell per patch), 10x control; sizes must be multiples of 32 to avoid smart_resize blur. Update _IMAGE_TOKENS_FALLBACK; optionally add 1-px gridlines at 8x. >=2 runs per arm (noise +-1.1).",
   "mechanism": "Aligned patches give translation-consistent single-cell inputs, improving color/object recognition; 8x saves 144 tokens per image (36%). Franzen's biggest image gain was 4x->10x.",
   "evidence": [
    "tool_agent.py:4082-4084 (32 px per token, 402 tokens at 10x)",
    "Qwen3.8 preprocessor_config patch_size 16 merge 2",
    "versions.md:55: Wang harness upscale 8 scored 19.43 vs 17.74 at 4x",
    "vision_context.py:21 default 16; VISTA renders 8x",
    "Kaggle discussion 717133, 739801"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "partial",
   "merged_from": [
    "Align board-image upscale to vision patch grid",
    "Patch-aligned board images: A/B upscale 8 and 16"
   ]
  },
  {
   "id": "temp-1-0",
   "title": "Official thinking-mode temperature 1.0 (top_p 0.95, top_k 20)",
   "category": "model-config",
   "description": "Set LOCAL_ANALYZER_TEMPERATURE=1.0 keeping TOP_P 0.95, TOP_K 20, provider vllm so top_k is sent. Watch accept len (expect ~3% drop) and score.",
   "mechanism": "Model card recommends T=1.0 for thinking; may improve hypothesis diversity on stalled levels. Score flat between 0.6 and 0.7, so small effect.",
   "evidence": [
    "HF card and unsloth docs: temperature 1.0, top_p 0.95, top_k 20",
    "nb.txt:10190-10192 uses 0.7",
    "temp06 45.92 vs 46.25",
    "openai_compat.py:155-159"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "partial",
   "merged_from": []
  },
  {
   "id": "no-model-swap",
   "title": "Do not spend the remaining days on a model swap",
   "category": "model-config",
   "description": "No open model that fits one RTX PRO 6000 offline beats Qwen3.8-Flash-Next on agentic/visual tasks; Kimi K3 and DeepSeek V4 Pro do not fit. Spend effort on serving/generation knobs instead.",
   "mechanism": "Porting a new architecture would consume most of the 5-day window with high regression risk.",
   "evidence": [
    "HF benchmark table vs Qwen3.8-27B",
    "kingy.ai open-weight landscape",
    "WRITEUP.md:15"
   ],
   "effort": ">3 days",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "solved-level-memory",
   "title": "Pinned solved-level memory (rule + winning action tail + closing reasoning), re-shown after trims",
   "category": "memory-context",
   "description": "Port Siriki/rellik13 M85. On level_transition record level, actions spent, last <=30 executed actions, last ~900 chars of reasoning of the clearing turn; ask for a 'Rule for level N:' paragraph (goal, controls, winning plan, <=5 lines), captured from content OR reasoning_content, at most 2 asks. Keep newest 3 levels full, older rule-only, ~1.5k tokens cap. For cache economy put it in the level-up opener and re-insert as first retained message at each drain (not the system prompt, which re-prefills ~116k per level-up). Write once per level, not rewritten every turn like the failed WM. Fold in ledger-on-trim; optional 'Level N notes' hindsight (win condition from last_transition.before_frame vs end of animation frames) parsed via _extract_labeled_blocks, flag ARC3_LEVEL_LEDGER.",
   "mechanism": "Drains to ~57k evict the turns that established earlier levels' mechanics, so the model re-derives them (cd82 L2 122 actions vs 8). Later levels weigh more; gains in M85 +2.82 LB (7.14->9.96), ledger-on-trim +6.3 offline. Harness-captured facts avoid the unreliable model-written WM.",
   "evidence": [
    "Siriki writeup and cells/42 M85 implementation; rellik13 cells/42",
    "Kaggle discussion 734843 (66.8% of replies have zero visible content)",
    "skarin audit: 50.88 of gap from incomplete levels",
    "ledger-on-trim 52.56 vs 46.25; kinetic v8 ARC3_X_SOLVED_MEMORY",
    "arXiv 2607.28287 Tycho; 2607.20709 NOOA; OpenAI compaction 13.3%->38.3%",
    "WRITEUP.md:295-297, :97; prompts.py:45-58; tool_agent.py:8564/8902"
   ],
   "effort": "1 day",
   "expected_impact": "high",
   "status_in_franzen": "partial",
   "merged_from": [
    "Solved-level memory (Siriki M85)",
    "Pin solved-level memory block (competitors)",
    "Level-completion hindsight + pinned per-game level ledger"
   ]
  },
  {
   "id": "level-debrief",
   "title": "Harness-written level-completion debrief plus winning-board image",
   "category": "memory-context",
   "description": "At level-up extend LEVEL_START_USER_PROMPT opener with a block built by the harness (no model call): 'Level N cleared in K actions (A attempts)', run-length-encoded action list (<=300 chars), last ~600 chars of pre-clear reasoning, one-line digests of earlier levels, and an image of the board before the winning action via _prepare_gameover_images. Sits at end of context so no prefix cost (~300-700 text + 402 image tokens per level). Optionally set ARC3_LEVEL_INVENTORY=1.",
   "mechanism": "Later levels cost a median 26.4k tokens vs 15.3k for level 1 in control; ledger-on-trim cut later levels to 18.8k, so carried knowledge lowers later-level cost. Objective facts at the moment needed; winning image shows what 'goal reached' looks like.",
   "evidence": [
    "benchmark.json control vs ledger medians",
    "findings.md section 3",
    "Tycho arXiv 2607.28287",
    "harness-feature-matrix C#2",
    "talk_by_tufa_lab.md",
    "prompts.py:45-58"
   ],
   "effort": "half-day",
   "expected_impact": "high",
   "status_in_franzen": "partial",
   "merged_from": []
  },
  {
   "id": "stall-audit-prompt",
   "title": "Stall-triggered strategy audit / trouble reflection prompt",
   "category": "prompting",
   "description": "Add a one-shot reflection block to the next opener only when a level stalls (gen tokens on level cross a multiple of ARC3_PRIORITY_TOKEN_SCALE 62k, ~115 actions, or >=50 actions since last marker on game over, plus no-progress after ~40k tokens and >=30 actions), at most once per interval, reset at level-up, flag ARC3_STRATEGY_AUDIT / ARC3_TROUBLE_PROMPT_ACTIONS. Text: LHS STRATEGY_AUDIT_PROMPT and Rodionov v1.5 trouble prompts (level-1 'missing simple visual cue' and later 'what is missing from your model'): analyze tries, state a new goal hypothesis, 2-3 competing hypotheses, cheapest discriminating action, prefer simpler mechanics.",
   "mechanism": "Stuck runs usually defend a wrong early goal hypothesis (83% of forks had already tried the decisive action). ~600 tokens fired only on stalls; complements the scheduler which only deprioritizes. Always-on hints and audit60k showed no gain, so gating on stagnation is the difference.",
   "evidence": [
    "LHS prompts.py STRATEGY_AUDIT_PROMPT; solver.py:239-249 (0.25 x runtime)",
    "Kaggle discussions 744792, 743723, 743060",
    "arXiv 2607.15439 (v1.5 74.78->82.01 RHAE, confounded); Reflexion 2303.11366",
    "arms: audit60k 44.10 vs 46.25; WRITEUP.md:303"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "absent",
   "merged_from": [
    "Stall-triggered strategy/assumption audit",
    "Stagnation-triggered trouble reflection prompt"
   ]
  },
  {
   "id": "sandbox-perception-helpers",
   "title": "Sandbox perception helpers and node fields (bbox/area/mask, labelled crops, runs, find, objects(), KeyError hints)",
   "category": "tools-repl",
   "description": "In segmentation.py add node bbox [r0,c0,r1,c1], center, h, w, area (alias pixels), touches_edge, mask (row strings when bbox<=~256 cells); return nodes as a dict subclass whose __missing__ names the right field (copy _NODE_FIELD_HINTS from Wang/Ludvig M90). In FrameView add .rows, __getitem__, .at(), show(r0,r1,c0,c1) with column ruler, runs(frame,r), find(frame,ch), objects(frame,bg,diagonal,multicolor) ARC-DSL style. Class must live in segmentation.py (spliced via inspect.getsource). Update STRUCTURED_RUNTIME_STATE_ADDENDUM and prompts.py:193; check frame_diff/JSON paths accept the subclass.",
   "mechanism": "Model hand-parses .ascii in 39% of calls and recomputes bboxes in 17%; asks for bbox/area/centroid and gets bare KeyErrors (198 code-error recoveries in 25 games for Siriki). Letter-run tokenization makes column counting unreliable. Fewer wrong coordinates, retries and boilerplate lowers tokens per cleared level.",
   "evidence": [
    "Siriki M75/M90 cells; LHS nodes carry bbox/pixel_count/mask",
    "fz transcripts: .ascii.split in 879/2275 calls, 113 'misread' mentions, 3x FrameView not subscriptable",
    "Flash-Next tokenizer: 27.7 tokens per 64-cell row",
    "segmentation.py:327-338; python_tool_sandbox.py:195-215, 774",
    "GB10 numpy-perception notes: non-acting calls 11->4",
    "Kaggle discussion 734369"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "partial",
   "merged_from": [
    "Add bbox/area/mask to segmentation nodes and field hints",
    "Sandbox perception helpers: labelled crops, run-length rows, frame indexing"
   ]
  },
  {
   "id": "truthful-prompt-hud-budget",
   "title": "Make the prompt truthful: edge bar is the action budget, real scoring rule, HUD panels as goal hints",
   "category": "prompting",
   "description": "Replace prompts.py:135-136 ('do not get distracted by the bar... DON'T DO THIS') with LHS wording: edge bar is a per-level action budget (every executed action incl. dead clicks draws it down; exhaustion = game over); side panels/legends often show target configuration, order or key shapes. Replace prompts.py:122 with the actual rule: uncleared level scores 0, later levels weigh more, score ~ (baseline/actions)^2, every game solvable. Fix tool schema text tool_agent.py:7200 'ephemeral' vs retained functions.",
   "mechanism": "Prompt currently tells the model to ignore and to budget the same bar; goal-bearing HUD content gets dismissed (sk48 bottom panel order, ls20 legend glyph 15-21k-char debates). Siri removed both lines; true scoring text + pacing line gave +28% hidden (with click hints). Net length ~unchanged.",
   "evidence": [
    "Kaggle discussion 744792 (rellik13), 743060, 718638, 728299; docs.arcprize.org/methodology",
    "Siriki v18.1gs changelog",
    "prompts.py:122,135,136; tool_agent.py:7200",
    "harness-feature-matrix C#1; fz control: 0 game overs in 18/25 games"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "partial",
   "merged_from": [
    "Make the prompt truthful",
    "Stop telling the model to ignore the HUD"
   ]
  },
  {
   "id": "lean-opener",
   "title": "Lean per-turn opener: send static reminder block once per context window",
   "category": "memory-context",
   "description": "_build_user_prompt repeats ~10 static lines plus TOOL_CALL_FORMAT_GUIDANCE (tool_agent.py:7096-7157) every opener/resume, and _retained_function_context (:6461) lists all retained signatures every turn. Move static lines into the system prompt (or emit only in the first opener after a drain/level change via _note_history_evicted flag); reduce per-turn text to a Wang/Ludvig-style 5-line message plus event blocks; list retained functions only when changed or after drain. Env flag for A/B; measure share of retained context first. Separate arm: ARC3_YIELD_RESUME_PROMPT=short.",
   "mechanism": "Static block median 1,725 chars = 62% of user-prompt chars (~450-550 tokens per turn), retained list +13%. Cutting slows context growth 10-15%, fewer drains/re-prefills, more history kept. Siriki: instruction bloat cost 2.8-4.3 LB; Wang 5-line port +3.44.",
   "evidence": [
    "Siriki writeup; M90 cell",
    "tool_agent.py:7096-7157, 6461",
    "measured 4,932 user prompts avg 2,761 chars",
    "WRITEUP.md:299"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "partial",
   "merged_from": [
    "Lean per-turn opener",
    "Send static per-turn instruction block once per context window"
   ]
  },
  {
   "id": "history-state-io",
   "title": "Stop O(history) state rewrite/re-parse on every action and python call",
   "category": "bugfix-quickwin",
   "description": "write_runtime_state (solver.py:1876) json.dumps(indent=2) the whole history after every action incl. each batch action; each python call re-parses it (tool_agent.py:7936 unused load, 7957) and rebuilds ascii+grid payloads for all frames (_ascii_history_view_payload :4060). Fix: keep history in memory (RuntimeState via callable), cache each immutable HistoryEntry payload, drop indent / store hex-row strings, delete unused load, write once per batch/tool call, send only newly appended entries and keep child FrameViews and segmentation cache (hash-keyed; ~47 ms per frame), honor solver.minimal_diagnostics, send grid only with ascii lazy. Log load/write time per call before/after; verify by replaying a recorded snippet and diffing globals.",
   "mechanism": "Per-action cost grows linearly: 0.10 s at 50 entries, 0.40 s at 200, 0.99 s at 500, 2.0 s at 1000 (rellik13: 0.8 s write at 800 actions; 36/150 action calls hit the 30 s timeout), all under one GIL shared by ~10-110 game threads, idling GPU slots and causing mid-plan timeouts on exactly the long high-weight games.",
   "evidence": [
    "rellik13 cell 38 M86 WHY block; Siriki M86/M111",
    "lordhansolo writeup: in-memory state removed most tool latency",
    "runtime_state.py:281; solver.py:801-810,1876,2144; tool_agent.py:7935-7982; python_tool_sandbox.py:419-447, 906",
    "local benchmarks; tr87 12 actions in 30 s at step 279; median ~188 actions per game, max ~545"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "absent",
   "merged_from": [
    "Stop rewriting and re-parsing whole game history (Siriki M86)",
    "Make per-action state refresh scale with new entries",
    "Incremental runtime-state transfer and segmentation cache",
    "Stop O(history) state rewrite; honor minimal_diagnostics"
   ]
  },
  {
   "id": "server-watchdog",
   "title": "Server health watchdog with bounded restarts, games pause on analyzer failures, server-argv audit",
   "category": "bugfix-quickwin",
   "description": "Background watchdog polling /v1/models (15 s); after 4 consecutive failures kill, confirm port free, relaunch identical argv (same chunked-prefill/context settings; Siriki's restart failed when they differed); max 2 restarts. Set ARC3_MAX_ANALYZER_FAILURES=-1 and long backoff so games wait out the ~9-min reboot. Add rellik13's check that final server argv equals the intended list.",
   "mechanism": "Insurance: server death today makes every game fail 10 times and stop, ending the run early with partial score. Several teams saw runs finish in 2-4 h around Oct 1-2, and private score comes from the same run.",
   "evidence": [
    "Kaggle discussion 744995, 729985",
    "Siriki v14.64 watchdog; Scott Le Grand taaf-flashnext notebook; keithtyser vLLM Duck cell 0/15",
    "skarin audit: GPU peak 94.5/125.5 GB",
    "CONFIGURATION.md section 13; franzen.txt:10701 single launch --watchdog-timeout 1800"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "absent",
   "merged_from": [
    "Server watchdog + games pause",
    "Guard the 9-hour run: watchdog and argument check"
   ]
  },
  {
   "id": "click-target-helper",
   "title": "Ranked click-target helper with HUD masking and dead-signature memory",
   "category": "perception",
   "description": "Sandbox click_candidates(frame=None, k=20): one representative interior cell per distinct object (cell nearest centroid) from segmentation, excluding the 4-cell HUD border and largest background component, grouped by shape_hash/color, ranked small/compact/rare-colored first (Reki/graph-explorer tiers), thin edge strips last, with center, color, area, shape_hash. Tag tried_no_change when an earlier MOUSE on same (shape_hash,color) gave gameplay_changed=False on this level; annotate 'already clicked: effect'. Optional one opener line listing top ~8 untried candidates; document as optional.",
   "mechanism": "Click games offer ~4096 actions per state; dead clicks on HUD/background cost score and level budget and tokens spent computing coordinates. Segmentation alone added levels in graph explorers (5 on vc33, 2 on ft09). Clicks are ~28% of actions, in 17 of 25 games.",
   "evidence": [
    "Kaggle discussion 743060; Reki Milestone-1 blog; StochasticGoose/Blind Squirrel preview learnings",
    "arXiv 2512.24156 frame processor tiers",
    "discussion 718638 dead clicks counted",
    "prompts.py:136; CONFIGURATION.md section 5"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "partial",
   "merged_from": [
    "Click-target helper with dead-signature memory",
    "Ranked click-target helper with HUD masking"
   ]
  },
  {
   "id": "state-graph-coverage",
   "title": "Observed state-graph frontier helpers and interaction-coverage ledger",
   "category": "planning-exploration",
   "description": "New module (hook next to _record_guard_observations tool_agent.py:7773, reuse _interior_state_hash :617 and segmentation) keeping per-level (1) object type (color+shape hash) x action kind -> effect classes counts and (2) directed graph state hash -> {action: successor} with no-op/fatal flags. Sandbox: untried(frame), known_path(target), goto(target) with per-step hash check, path_to_untested(), coverage(), graph_stats(). On a stall add one opener line 'Never-interacted object types: ...' (max 5).",
   "mechanism": "Catastrophic failures usually come from never testing the one interaction a level needs; harness bookkeeping is exact, token-free and survives drains. OPINE ablation 78.4 vs 58.2 RHAE; graph exploration solved 19 levels vs 5 for LLM+DSL.",
   "evidence": [
    "arXiv 2607.01531 OPINE-World; 2512.24156; 2510.19788 AutumnBench",
    "Blind Squirrel (wd13ca) and dolphin-in-a-coma just-explore repos",
    "schema-harness DC22/LF52",
    "sk48 transcript hand-written snapshot/UNDO rewind",
    "CONFIGURATION.md:78,125-126; noop_repeat_guard.py"
   ],
   "effort": "1 day",
   "expected_impact": "medium",
   "status_in_franzen": "partial",
   "merged_from": [
    "Interaction-coverage ledger + state-graph frontier",
    "Observed state-transition graph helpers"
   ]
  },
  {
   "id": "action-plan-annotations",
   "title": "Durable per-action `plan` field plus game_overs ledger",
   "category": "tools-repl",
   "description": "Add optional `plan` string to the python tool schema (tool_agent.py:7194-7203), normalized like lordhansolo _normalized_plan, stored on every HistoryEntry from that call and exposed as history[i].plan / transitions[i].plan; document with LHS line 'once a turn leaves your context this is the only record'. Add game_overs for current level (actions since level start/previous death, action_count, newest frame; cleared on level-up; LHS wording on fixed budgets). Optionally have the harness print a plan trail for the span a trim just removed (not in cached prefix except at trim).",
   "mechanism": "Blockwise drains discard half the context and the hypothesis behind each action; a one-line plan survives in Python state at ~20 tokens per call and supports predict-then-check. Equal action_count across deaths diagnoses budget deaths from data.",
   "evidence": [
    "LHS prompts.py history[i].plan and game_overs; tool_agent.py:734,1042,1070",
    "VISTA predict-then-act protocol; discussion 744792",
    "tool_agent.py:6961-6990, 7194-7203",
    "arms.md hypothesis-loop 93% compliance inconclusive"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "absent",
   "merged_from": [
    "Durable action annotations (plan + game_overs)",
    "Add optional plan field and plan trail after trims"
   ]
  },
  {
   "id": "animation-frames-last8",
   "title": "Keep in-between animation frames for last ~8 animated actions",
   "category": "perception",
   "description": "Port Siriki M111: keep GameState.all_frames in-between frames for the last 8 animated actions (<=32 each, dedupe consecutive, hex-row storage), expose as transitions[i].frames read-only views, add animation_frames count to action result; optional LHS tween/path/blink classifier. Lazy, out of the prompt.",
   "mechanism": "Animation access was Franzen's strongest perception gain but only the last animated action is retained. Siriki found mid-animation-only evidence in 22 of 25 games.",
   "evidence": [
    "Siriki M111 docstring; LHS .animation_frames",
    "discussion 734369",
    "WRITEUP.md:181-185,301; CONFIGURATION.md section 4"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "partial",
   "merged_from": []
  },
  {
   "id": "submission-protocol",
   "title": "Submission hygiene and decision protocol: pin Docker image, sanity-check runs, 4-game panel, paired seeds",
   "category": "other",
   "description": "Assert sys.version_info[:2]==(3,12) (default image moved to Py3.13 Oct 1; cp312 wheels); diff env/CFG vs original and check wall time for early termination; screen changes on the FT09/SP80/TN36/VC33 panel (reject only big regressions), promote on >=3 matched full-25 seeds with paired per-game deltas at equal generated tokens, using tools/significance.py; choose complementary final picks. Prioritize completion arms over efficiency arms (50.88 of 53.5 lost points are incomplete levels); six weak games take ~18% of tokens.",
   "mechanism": "LB SD ~3.9 per run (unchanged copies 25.77 mean; same version 27.80 then 21.01), so +3 is invisible on one draw; protocol avoids wasted slots and false rejections.",
   "evidence": [
    "discussion 745062, 745654, 729985",
    "skarin audit and 'why one run lies' notebooks (panel 0.953 median held-out correlation, worst fold 0.573)",
    "own submissions 22.88, 20.87",
    "inference/tools/significance.py"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "partial",
   "merged_from": [
    "Submission hygiene",
    "Adopt a decision protocol before spending remaining submissions"
   ]
  },
  {
   "id": "sim-110-scale-testing",
   "title": "Test scheduling changes at 110-game scale (replay sim, then 110-clone competition run)",
   "category": "other",
   "description": "Sweep ARC3_PRIORITY_* knobs in the replay simulator first, then confirm once with TAAF CompetitionArcadeServer.official_110() (or 50 clones for ~242 min), scored with make significance paired tests.",
   "mechanism": "In 25-game demos only ~15 games wait at handover vs ~100 in the real run, hiding priority differences; avoids false negatives.",
   "evidence": [
    "taaf competition_arcade.py official_110, clone_game_ids",
    "replay_scheduler.py docstring",
    "WRITEUP section 6; ARMS.md swings of 50 points"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "grid-lines-10x",
   "title": "Draw 1-px cell grid lines on the 10x board image",
   "category": "perception",
   "description": "In vision_context.py after upscaling overwrite last pixel row/col of each cell with a dark separator, optionally brighter line every 8 cells; same image size and token cost. A/B on click-heavy games.",
   "mechanism": "Cell boundaries help counting and MOUSE coordinate localization; Qwen is weak at inferring coordinates from images.",
   "evidence": [
    "VISTA 8x upscale with 1-px gridlines",
    "discussion 734369",
    "arXiv 2608.21170 grid lines +0.2 to +2.6 pp"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "labelled-axes",
   "title": "Labelled coordinate axes on board images (no per-cell glyphs)",
   "category": "perception",
   "description": "Flag MULTIMODAL_AXES: 32-px margin top/left with tick labels every 8 cells and faint major gridlines (8x image becomes 544x544 = 289 tokens); no per-cell letters; add image-geometry sentence and tell model to confirm exact coordinates in Python before MOUSE.",
   "mechanism": "Labelled axes anchor positions to cell indices (blank margin raises +-1 errors to 17-46%, labelled 4-5%); Qwen3.8 invents canvas sizes.",
   "evidence": [
    "arXiv 2608.21170; SCAFFOLD 2402.12058; voxel51 Qwen3.8 grounding",
    "fz transcripts '736 px' misreads",
    "versions.md:56 12x+ASCII overlay scored 5.60 (caution)"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "prediction-checked-batches",
   "title": "Prediction-checked batches: action(..., expect=...) stopping at first mismatch",
   "category": "tools-repl",
   "description": "Flag ARC3_PREDICTION_CHECK. Tier A: action(actions, expect='text') shown next opener as 'You predicted: ...' above observed change. Tier B: per-action expect as ascii with '?' wildcards or {'r,c': char}; in solver batch loop next to BATCH_NOOP_BLOCK (solver.py:1626) compare claimed interior cells, stop with stop_reason='prediction_mismatch' returning mismatched cells and crop. Sandbox checked(actions, predict) helper; clear _stale_state when a no-op was predicted.",
   "mechanism": "Score falls with square of actions; a wrong long batch currently runs to the end. Stopping at first divergence turns it into one informative probe (shared core of Schema commit_actions, Rodionov, Tycho, NOOA).",
   "evidence": [
    "schema-harness.github.io (42.83%->98.98%)",
    "arXiv 2605.05138, 2607.28287, 2607.20709",
    "solver.py:1626; python_tool_sandbox.py:524"
   ],
   "effort": "1 day",
   "expected_impact": "medium",
   "status_in_franzen": "partial",
   "merged_from": []
  },
  {
   "id": "world-model-kit",
   "title": "World-model kit: backtest/check_model, bounded bfs/search, partial-prediction contract",
   "category": "tools-repl",
   "description": "Add to bootstrap near frame_diff (python_tool_sandbox.py:400): backtest(predict, level, border, show) / check_model(parse, step) replaying every recorded transition of the level (skip auto RESET/level changes), scoring only claimed cells (None/'?' abstain), reporting accuracy, coverage, first mismatches as crops, optional (grid,'win'|'dead'); bfs/search(start, succ, is_goal, key, budget_s=12, max_nodes) returning path/stats and never hitting the 30 s kill; grid(frame) mutable copy. ~150 tokens of optional prompt; pair with ARC3_PERSISTENT_FUNCTIONS_SCOPE=game. Do not force it.",
   "mechanism": "Moves trial-and-error into a checked model: unverified simulators yield confident wrong plans that cost actions. Schema's exact-backtested models used 1.6-5x fewer actions than humans in 14/25 games; bounded search turns timeouts into partial results.",
   "evidence": [
    "arXiv 2607.15439 (replay verification ranked first; textual beat executable in some conditions)",
    "schema-harness run_backtest; Tycho 2607.28287; NOOA unbounded searches",
    "WorldCoder NeurIPS 2024; GIF-MCTS 2405.15383; PoE-World",
    "fz: 136 sim/bfs/step functions defined; 3 of 7 control timeouts were search blowups",
    "prompts.py:263-264; GB10 sandbox notes: provided path() used 0 times (adoption risk)"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "partial",
   "merged_from": [
    "World-model kit: backtest, bounded bfs",
    "World-model checker plus anytime search"
   ]
  },
  {
   "id": "seed-helper-library",
   "title": "Seed a small starter helper library through the retained-function channel",
   "category": "tools-repl",
   "description": "Mine transcripts for most-rewritten helpers (crop, find, bbox, objs, delta/frame_diff wrapper, bfs; 1,092 defs/25 games), curate 5-10 generic ones, unit-test on the 25 public games, preload into self._kept_functions at game start (restored into every snippet), list signatures labelled 'starter helpers; editable'. Keep tiny and optional; A/B with a flag.",
   "mechanism": "Re-deriving the same helpers per game wastes generation and adds bugs; saved tokens feed scheduler redistribution. Tufa warns hand-crafted toolkits can hurt, so match the model's habits.",
   "evidence": [
    "Voyager 2305.16291; Rodionov 2605.05138 future work",
    "fz-arms-v4-control: crop 31, bfs 18, snap 11 defs, 331 same-name redefinitions",
    "WRITEUP.md:20; Tufa milestone-1 blog; GB10 sandbox: find used 82x, path 0x"
   ],
   "effort": "1 day",
   "expected_impact": "low",
   "status_in_franzen": "partial",
   "merged_from": [
    "Seed a curated helper library (Voyager-style)",
    "Small starter helper set in model's own style"
   ]
  },
  {
   "id": "reset-exposure",
   "title": "A/B model RESET exposure with zero-action RESET guard and UNDO/RESET cost wording",
   "category": "bugfix-quickwin",
   "description": "Run EXPOSE_RESET=on with ARC3_ACTION_INFO=1. Add a guard (tool_agent.py:8071-8095) refusing RESET with no action on current level (in competition this is a no-op counted as an action; locally ONLY_RESET_LEVELS differs). Edit UNDO_INFO_ADDENDUM (prompts.py:156): UNDO costs one action and does not refund; RESET_INFO_ADDENDUM: RESET once instead of draining the bar; RESET plus replay of good prefix usually beats long chained UNDOs.",
   "mechanism": "Dead-end states otherwise cost 10-30 counted actions plus turns burning the bar (6-11 of 25 games per run plan to 'burn the bar'). Humans use resets 12.5% vs <7% for models.",
   "evidence": [
    "fz transcripts 'burn the bar' plans: 11/9/7/6/6 per run",
    "arc_agi scorecard.py:701-703, api.py:325-334; docs.arcprize.org start-or-reset",
    "AutumnBench 2510.19788; stridenote ARCBaseGame; arXiv 2605.05138 sec 4",
    "CONFIGURATION.md:27 EXPOSE_RESET default off; harness-feature-matrix Siri 46% of actions in failed attempts"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "present-but-off",
   "merged_from": [
    "A/B existing model RESET exposure plus guard",
    "Expose RESET, guard zero-action RESET, UNDO/RESET cost"
   ]
  },
  {
   "id": "actor-modeling-subcall",
   "title": "Actor-requested modeling subcall (Tycho orchestrator policy)",
   "category": "planning-exploration",
   "description": "Sandbox ask_modeler(task) triggers a host-side side completion not appended to main history: short modeling system prompt, current level transitions as compact frame_diffs, board crop and current predict() source; reply is Python predict()/is_goal() checked with backtest(). Cap ~6k tokens, 2-3 calls per level, count tokens in scheduler C term.",
   "mechanism": "Writing simulator code in a short clean context avoids context rot; actor-requested delegation scored highest in Tycho's matched comparison. Costs ~15-20k prefill per call; scaffold gains fall below a capability floor on small models.",
   "evidence": [
    "arXiv 2607.28287; 2605.09998 Continual Harness; 2605.09650 DreamTeam"
   ],
   "effort": "2-3 days",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "prolong-memory-search",
   "title": "Programmatic memory of evicted context: search_log plus per-action notes (PRO-LONG)",
   "category": "memory-context",
   "description": "When _drop_oldest_history_block / _trim_messages_for_context drop messages, append their text and tool outputs tagged with level/step to an in-memory per-game log; accept action(actions, note='why') stored in transitions[i].result['note']; expose search_log(regex,k,width) in the sandbox via a host query (like the animation fetch); one prompt line.",
   "mechanism": "Each drain discards ~70k tokens of reasoning; frames survive but conclusions do not. A lossless searchable archive restores recall without growing context or touching the prefix cache.",
   "evidence": [
    "arXiv 2607.20064 PRO-LONG",
    "OpenAI compaction post (13.3%->38.3%)",
    "stridenote ActionInput reasoning field",
    "tool_agent.py:8564"
   ],
   "effort": "1 day",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "memo-dict",
   "title": "Persistent JSON memo dict across python calls",
   "category": "tools-repl",
   "description": "Sandbox memo dict restored at call start and returned in 'final' and 'error' messages (python_tool_sandbox.py:772-800), JSON-safe, capped ~20-200KB, stored next to _kept_functions (tool_agent.py:6497-6571, 8161-8179), game-scoped, flagged at level change; one line in PERSISTENT_FUNCTIONS_LINE (prompts.py:355-365). Port of Kinetic v8 ARC3_X_MEMO. Optionally auto-restore small JSON-serializable top-level variables.",
   "mechanism": "Model keeps trying to persist data: 302 NameErrors on earlier variables, 30 'open', pickle 10, 'store' 4 per runs; each wastes a round trip and recomputation. Kinetic v8 carrying it scored 27.28 (no measurable gain), so expect small effect.",
   "evidence": [
    "fz transcripts NameError counts",
    "kinetic-learning-by-playing-v8.txt:10205-10297",
    "CONFIGURATION.md section 7; WRITEUP 'variables discarded'"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": [
    "Persistent JSON memo store",
    "Add a memo dict that persists variables across python calls"
   ]
  },
  {
   "id": "persistent-kernel",
   "title": "Persistent per-game Python kernel (true REPL)",
   "category": "tools-repl",
   "description": "Make _SANDBOX_BOOTSTRAP a long-lived child per game looping on exec messages and keeping runtime_globals; refresh provided globals each call with incremental history; soft SIGALRM timeouts so the kernel survives; on hard kill restart and rehydrate from retained-function pool; RLIMIT_AS 2-4 GiB, cap live kernels, kill kernels of gate-parked games. Prompt: variables persist, don't bind provided globals as defaults. Flag ARC3_PERSISTENT_KERNEL.",
   "mechanism": "Works with the model's habit of assuming persistence; removes recompute, NameErrors and 50-100 ms spawn plus full-state JSON per call. GB10: 0 python errors in 47 calls, shorter requests.",
   "evidence": [
    "CONFIGURATION.md section 7 (not a persistent kernel)",
    "fz 7 runs: 377 NameErrors (2.2%)",
    "GB10 sandbox notes (n=2 games)",
    "CodeAct arXiv 2402.01030"
   ],
   "effort": "1 day",
   "expected_impact": "medium",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "timeout-wipes-retained",
   "title": "Bug: tool timeout/crash wipes all retained functions and drops stdout",
   "category": "bugfix-quickwin",
   "description": "In _record_retained_functions (tool_agent.py:6512-6571), if sandbox_result lacks 'keepable_functions' (timeout/crash/invalid paths python_tool_sandbox.py:1001-1038, 1107-1112) return early and keep self._kept_functions unchanged, optionally noting retained functions were kept. ~5 lines. Pair with partial output below.",
   "mechanism": "One timeout drops the whole helper library (tr87 L6: ~40 helpers vanished, model redefined them with 4.8k reasoning chars); 51 timeouts in ~17.4k calls, concentrated late where regeneration is costliest.",
   "evidence": [
    "tool_agent.py:6529-6566; verified locally keys ['action_results','error','stdout']",
    "fz transcripts cn04:28608, g50t:25629, re86:18271, tr87:8092",
    "timeouts per run: 7,7,2,6,11,8,7"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": [
    "Don't wipe retained functions on timeout/crash",
    "Fix bug: timeout wipes retained functions and drops output"
   ]
  },
  {
   "id": "soft-timeouts",
   "title": "Soft, non-destructive tool timeouts: partial stdout, host action time not charged, >30 s allowed",
   "category": "tools-repl",
   "description": "Child arms signal.setitimer(ITIMER_REAL, timeout-2..3) raising ToolTimeout(BaseException) added to the except tuple (python_tool_sandbox.py:743/778-800) so it returns an 'error' with partial stdout, action_results, retention payload and 'executed before timeout: ...' summary; host deadline stays as hard backstop. Extend host deadline by time inside action_handler/animation_handler (:1063-1095, Note :906); remove min(30,...) at tool_agent.py:5131 and make prompts.py:378 truthful (try 45-60 s); expose a time shim (perf_counter/monotonic).",
   "mechanism": "~23 of ~50 timeouts happened in snippets that executed actions; model got empty stdout and unknown board state, risking more wasted actions. Host overhead 0.7-1.5 s per action under 10-14 streams kills long loops; timeouts rise with streams (11 at 12 streams vs 6-7).",
   "evidence": [
    "python_tool_sandbox.py:906-908, 1001-1011; tool_agent.py:5131; prompts.py:378",
    "sk48:8548-8732 (37 UNDOs ~0.8 s each), g50t:25629, tr87:8683",
    "'import time' blocked 47 times"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": [
    "Make tool timeouts soft and non-destructive",
    "Stop counting host-side action time against 30 s limit"
   ]
  },
  {
   "id": "cutoff-signal-fix",
   "title": "Fix dead 'reply was CUT OFF' signal and bogus resume text after truncation",
   "category": "bugfix-quickwin",
   "description": "A 12,288-token truncated reply always exceeds YIELD_TOKENS=2048 so control_yield_reason() (tool_agent.py:9566) yields before the CUT-OFF nudge (:9569-9578); the resume prompt (:536-563) instead claims a snippet ran or no tool call. Set a flag on finish_reason=='length' and lead the resume with the CUT-OFF sentence. With ARC3_MEMORY_SECTIONS=off, gate the follow-up nudge's seven memory-section clauses (:9605-9622; _advertised_section_labels :1870-1890 returns all 7) on _memory_sections_disabled().",
   "mechanism": "184 of 17,406 replies (1.06%, 7.3% of reasoning chars) hit the cap; 10 followed by another truncation; CUT-OFF text appears 0 times across runs. Telling the model length killed its call should shorten next deliberation; memory-section text invites long dumps.",
   "evidence": [
    "tool_agent.py:9566-9622, 536-563, 1870-1889, 7106-7116",
    "fz transcripts: 0 CUT-OFF occurrences, 184 length replies"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "sandbox-allowlist-gaps",
   "title": "Close sandbox allowlist/builtin gaps (class, globals, time, sys, recursion, pre-imports, missing attrs)",
   "category": "tools-repl",
   "description": "Add __build_class__, object, super, globals, id, setattr, vars, property, staticmethod, classmethod to SAFE_BUILTINS (python_tool_sandbox.py:107-176); add time shim (time/perf_counter/monotonic), sys shim (setrecursionlimit capped ~20000), dataclasses/typing/array/enum modules; pre-bind deque, defaultdict, Counter, heapq, itertools, json, math, re, functools, collections; add .step/.level to HistoryEntryView/TransitionView, FrameView.__getitem__, _LazyAnimation.items/values, make build_animation_view return error dict with steps/frames; update module list prompts.py:253 (also omits difflib).",
   "mechanism": "Each error costs a round trip (~1.5k reasoning tokens) and some fire after actions ran. ~114 ImportErrors (time 47, inspect 32, sys 16, pickle 10) and ~72 builtin NameErrors; recursion 1000 kills DFS; ~330 of ~4.7k calls ended in Python errors.",
   "evidence": [
    "python_tool_sandbox.py:85-176, 195-226, 313-317",
    "utils/animation.py:427-435",
    "fz transcripts error counts; local tests of class/recursion failures"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "partial",
   "merged_from": [
    "Close sandbox allowlist/builtin gaps",
    "Make the REPL tolerate common guesses: time/sys shims and missing attributes"
   ]
  },
  {
   "id": "inspect-source",
   "title": "Let the model read retained functions' source (source()/inspect.getsource shim)",
   "category": "tools-repl",
   "description": "Pass self._kept_functions into the child; provide source(name_or_fn) and an inspect shim (allow 'inspect' in _safe_import, python_tool_sandbox.py:313) with getsource/signature; append first docstring/comment line (~60 chars) of each function to signature list in _retained_function_context (tool_agent.py:6479-6489).",
   "mechanism": "After trimming only signatures remain, so the model misuses/redefines helpers or tries inspect.getsource (70 uses, 28 ImportErrors) and wastes a request.",
   "evidence": [
    "fz runs: 70 inspect.getsource uses; 28 ImportErrors",
    "tool_agent.py:6479-6489; python_tool_sandbox.py:85-106",
    "WRITEUP section 2 trims to ~57k"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "numpy-grid",
   "title": "Expose numpy and the numeric grid in the REPL",
   "category": "tools-repl",
   "description": "Sandbox runs python -I -S so numpy cannot import; add host numpy site path to sys.path in bootstrap and 'numpy' to SAFE_MODULES (:85); add FrameView.grid (tuple-of-tuples or ndarray); change 'raw numeric grid intentionally not exposed' lines (prompts.py:193, tool_agent.py:429). Best with persistent kernel. A/B, since Tufa/Franzen hid the grid on purpose.",
   "mechanism": "Replaces hand-written ascii parsing/color-search in ~30-40% of calls with np.argwhere, bbox, diffs: fewer code tokens and indexing bugs. GB10 numpy-perception: ar25 L2 cleared in 2/2 runs vs 2/8 controls.",
   "evidence": [
    "prompts.py:193; python_tool_sandbox.py:947",
    "fz-arms-v4-control: 38.6% of calls split .ascii, 28.8% color-search",
    "GB10-OVERNIGHT-RESULTS.md; sandbox notes find() used 82x"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "plain-text-tool-output",
   "title": "Render tool results as plain text instead of indented JSON",
   "category": "tools-repl",
   "description": "In _render_tool_payload (tool_agent.py:7476-7508, return json.dumps(indent=2) at :7508) emit stdout verbatim with real newlines plus compact error:/result:/function_retention:/truncated: lines; only non-string result values stay JSON. Keep truncation and board-dump guard. Flag ARC3_PLAIN_TOOL_OUTPUT.",
   "mechanism": "Multi-row crops reach the model as one line with \\n escapes, destroying column alignment (+11% chars; 14.5% of results have crops of 3+ rows). Plain text keeps geometry readable.",
   "evidence": [
    "tool_agent.py:7508, 9673-9680",
    "fz-arms-v4-control: 329 of 2,275 results with crops; 11.3% inflation",
    "GB10 sandbox notes (confounded); SWE-agent 2405.15793"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "effects-track-history",
   "title": "History effect table and object tracks: effects() and track()",
   "category": "tools-repl",
   "description": "Using stored per-action effect (or cached frame_diff per transition) add effects(level, action, last=30) returning one line per transition (index, action, effect summary, NO-OP/GAME OVER/LEVEL flags, bar delta) and track(hash/color/px) returning (step, bbox) positions of matching nodes across the level using cached segmentation.",
   "mechanism": "Rule induction currently needs a hand-written history parser each time (sk48 example), slow (47 ms/frame) and timeout-prone; one call shortens hypothesis cycles.",
   "evidence": [
    "sk48-d8078629_p0.txt:8380-8470",
    "7.9% of calls loop over history; replies >20k chars hold 24.5% of reasoning"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "partial",
   "merged_from": []
  },
  {
   "id": "auto-frame-diff",
   "title": "Turn on ARC3_AUTO_FRAME_DIFF and improve it (deltas, rigid grouping, HUD line)",
   "category": "perception",
   "description": "Set ARC3_AUTO_FRAME_DIFF=1 (budget 300), appended to new user message so cache-safe. Patch utils/frame_diff.py compute_frame_diff and renderer: (a) delta=[dr,dc] on moved entries; (b) merge touching moved components with same delta; (c) collapse border-band changes into one 'HUD: bar 40->39' line; (d) 'gone at/new at' for vanish/appear copies; (e) skip 1x1 same-hash matches jumping >6 cells. Own arm vs same-day control, >=2 runs each.",
   "mechanism": "Harness hands exact change facts; in GB10 on this checkpoint all 4 diff/perception runs cleared ar25 L2 vs 2 of 8 controls (pooled Fisher p~0.03, post hoc). Replaces inspection-only python calls (58% of calls) with 100-300 cheap prefill tokens; frame_diff() used in only 6% of calls.",
   "evidence": [
    "CONFIGURATION.md:76-77 (default Off); notebook submission flags lack AUTO_FRAME_DIFF; WRITEUP section 6 silent",
    "tool_agent.py:3689-3766, 3812-3819; frame_diff.py:243-246",
    "GB10-OVERNIGHT-RESULTS.md:59-66; numpy-perception matcher fixes (fake 1x1 moves 21/31->0/33)",
    "control: 2275 python calls, 1323 without action(), frame_diff in 140",
    "GPT-4o ARC 50% write-up (diff representation)"
   ],
   "effort": "half-day",
   "expected_impact": "high",
   "status_in_franzen": "present-but-off",
   "merged_from": []
  },
  {
   "id": "action-effect-echo",
   "title": "Per-action object-level effect and change magnitude in action() results and echo",
   "category": "perception",
   "description": "In solver._execute_action (solver.py:1776-1880) compute compute_frame_diff(prev,new) reusing after-frame segmentation as next before-frame; render via _render_auto_frame_diff/_diff_entry_line with ~40-60 tokens per action (fold HUD-border changes into 'bar -1'); store as individual_result['effect'] / transitions[i].result['effect']; extend _action_verdict/_build_action_echo/_build_action_trace (solver.py:248-375), e.g. 'RIGHT -> effect: B 16px (15,18)->(15,21); 109 cells', plus interior changed-cell count, bbox, top-3 color transitions, 'HUD-only', 'animated: N frames, T transient px'. Add lordhansolo magnitude fields (changed_pixels, largest_changed_regions, per_step_changed_pixels, reading guide) and optionally Scott Le Grand's per-level action ledger and 'UNTESTED on this level' line. Cap lines per snippet (~25); flag ARC3_ACTION_EFFECTS.",
   "mechanism": "Exact per-action change facts arrive inside the loop where the model acts, so code can branch without extra requests; bare 'NO-OP' labels mislead (sc25 clicks echoed NO-OP while red-flash animations occurred; 158 of 1019 clicks echoed NO-OP). Model already hand-writes summ()/diff wrappers. Ledger evidence mixed (Scott shipped it off).",
   "evidence": [
    "solver.py:246-330; ARC3_ACTION_ECHO=1 in submission",
    "ar25 transcript summ(tag); sc25 transcript; GB10 frame-diff notes (sk48 3 requests ~20 min)",
    "LHS runtime_state.py:154-200; prompts.py 'Result keys'",
    "Scott Le Grand taaf-flashnext notebook:817-900; Siriki ledger no gain",
    "GB10-OVERNIGHT-RESULTS.md"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "partial",
   "merged_from": [
    "Return short object-level effect per action",
    "Add per-action change summary and animation flag to action echo",
    "Harness-measured action effects: changed_pixels / UNTESTED line"
   ]
  },
  {
   "id": "scene-card",
   "title": "Deterministic 'scene card' at each level start",
   "category": "perception",
   "description": "Flag ARC3_SCENE_CARD: on turn 1 and first opener of each level insert <=250 tokens from segment_layer: background color, HUD candidates (thin edge-flush), objects grouped by shape_hash with count/color/hxw/bbox (<=~12, rarest first), enclosures, click candidates when MOUSE valid; tag new types via ARC3_LEVEL_INVENTORY diff (tool_agent.py:6073-6146). Never include the full board as text.",
   "mechanism": "Opening hypothesis largely decides the level; list removes 1-2 exploratory inspection calls (~1.5-2k tokens each). On GB10 the model copied the block's bboxes into its level-1 plan; capped and once per level because full-ASCII-every-turn lost badly.",
   "evidence": [
    "CONFIGURATION.md:78; GB10 numpy-perception notes (1052 vs 1472/2478 tokens per turn)",
    "discussion 743060; arXiv 2305.18354; arc-dsl",
    "versions.md:62-63 (ASCII 4.63, full ASCII+image 9.86 vs 17.74)"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "partial",
   "merged_from": []
  },
  {
   "id": "level-inventory-flag",
   "title": "Enable ARC3_LEVEL_INVENTORY=1 (flag-only 'what is new on this level')",
   "category": "perception",
   "description": "Set ARC3_LEVEL_INVENTORY=1 (not in notebook env; ARC3_DIFF_IMAGE already on) so _level_transition_diff_lines (tool_agent.py:6073) lists a START-vs-START diff (~300 tokens): 'appeared' types likely new mechanics, 'moved' for recurring. Can run env-only or with the debrief.",
   "mechanism": "Each new level costs ~4k median generated tokens before the first action (~19-20% of cleared-level tokens); a harness list shortens orientation and aims first probes at the new mechanic.",
   "evidence": [
    "tool_agent.py:5737-5779, 6073-6146; CONFIGURATION.md section 3",
    "control benchmark: median 3,853 tokens before first action; prompts.py:45-58"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "present-but-off",
   "merged_from": []
  },
  {
   "id": "budget-bar-readout",
   "title": "Harness-computed budget-bar readout",
   "category": "perception",
   "description": "Host-side detector: edge-band thin one-color components whose pixel count falls monotonically with actions; estimate drop per action and report 'Budget bar (top edge): ~N actions left' in the opener (and echo when N<15); reset on level change/death; after game over state whether bar was depleted (answers BAR RULE).",
   "mechanism": "Model re-derives the bar in ~16% of thinking blocks (371/2299) and still dies of budget exhaustion; a deterministic count prevents budget deaths and shortens game-over diagnosis.",
   "evidence": [
    "fz control: 371 of 2299 thinking blocks discuss bar/timer; 19 game-overs; sc25 'budget death'",
    "WRITEUP.md:223-229; prompts.py:135-136"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "hud-band-mask",
   "title": "Replace fixed 4-cell no-op border with a learned HUD-band mask",
   "category": "perception",
   "description": "Port Scott Le Grand's F5: band cell counts as HUD if it flips on >=90% of single actions with one fixed color transition (HUD_RATE 0.9, HUD_CELL_MIN 5, <=3 cells per action, HUD_BORDER 2); report board_changed_ex_hud, changed_bbox. First measure on transcripts how often guards fired where changes within 4 of the edge were not a ticking bar; build only if non-trivial.",
   "mechanism": "Avoids false 'no gameplay change' verdicts that truncate valid batches or block follow-up actions and false progress from HUD flicker outside the strip. Scott found a 1-2 cells/action budget bar in 25/25 public games.",
   "evidence": [
    "taaf-flashnext-sheetu12b-0922.txt:289-312,380-420",
    "CONFIGURATION.md ARC3_NOOP_GUARD_BORDER=4; WRITEUP section 5"
   ],
   "effort": "1 day",
   "expected_impact": "low",
   "status_in_franzen": "partial",
   "merged_from": []
  },
  {
   "id": "zoom-crops",
   "title": "Model-requested zoom() image and zoomed change crops",
   "category": "perception",
   "description": "Sandbox zoom(r0,r1,c0,c1) queues a crop at 16-32 px per cell (>=256 px per side, multiple of 32) attached to the next opener (max 1 per turn); when interior change bbox <=16x16 cells attach a labelled zoomed crop instead of full-board diff image. Off by default; test as an arm.",
   "mechanism": "At 10x one vision token covers 3.2x3.2 cells so 1-cell features fall below a token; zoom gives one cell per patch only where needed (VISTA inspect tool); small-change crop is cheaper than the full diff.",
   "evidence": [
    "vista-research.github.io (100% RHAE with Opus 5.0)",
    "vision_context.py:431-489; preprocessor_config",
    "discussion 717133 and WRITEUP.md:301 cautions"
   ],
   "effort": "1 day",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "reshow-level-start-board",
   "title": "Re-show current level's start board (and scene card) at head of history after each trim",
   "category": "memory-context",
   "description": "At drain insert one short user message after the system prompt with the level's start-frame image (self._cur_level_start_frame) and scene card if built, marked 'reference: starting layout'. Only at drain time (prefix-neutral); stack with ledger-on-trim.",
   "mechanism": "After a 58k drain the initial layout is lost though needed for goal inference and RESET/UNDO reasoning; 258-402 tokens per trim restores it cheaply.",
   "evidence": [
    "findings.md:30-44 ledger-on-trim 52.56 vs 46.25, 38.7k vs 42.7k tokens/level",
    "tool_agent.py:5760-5770"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "level-aware-trim",
   "title": "Level-aware history trimming, gated by a sawtooth natural-experiment check",
   "category": "memory-context",
   "description": "Run rellik13's analysis/sawtooth.py on own transcripts (visible in-level history vs clear within W turns). If positive, change _drop_oldest_history_block (tool_agent.py:8564-8617, plain pop(0)) to remove whole completed-level blocks first (keeping the solved-level block) and preserve the first K turns of the current level plus recent ones.",
   "mechanism": "Long hard levels lose their earliest evidence first under level-blind trimming; rellik13's retention fix took 14.49->22.53. Diagnostic prevents another unvalidated memory change.",
   "evidence": [
    "rellik13-milestone2/analysis/sawtooth.py:1-12; WRITEUP FP8 KV and longer history",
    "tool_agent.py:8564-8617"
   ],
   "effort": "2-3 days",
   "expected_impact": "medium",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "drain-depth-retune",
   "title": "Retune history drain (118k -> ~80k instead of ~59k)",
   "category": "model-config",
   "description": "Sweep ARC3_CONTEXT_DRAIN_TOKENS so post-trim window ~80k (LHS 126k->81.5k; rellik13 57k->45k); compare in-window history, trim frequency, prefill share (15% of wall time) and completions at equal tokens. Run only if the sawtooth check shows more history helps.",
   "mechanism": "Shallower drain keeps ~18% more history but trims ~25% more often; empirical. Earlier 52k drain and streams14 (96k/42k) showed no clear difference.",
   "evidence": [
    "tong-m2-comparison SUMMARY.md",
    "LHS tool_agent.py:458,1291",
    "arms.md streams14 46.38 vs 46.25"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "tried-no-gain",
   "merged_from": []
  },
  {
   "id": "scheduler-recalibration",
   "title": "Recalibrate scheduler: drop h=25 action penalty, halve tail B",
   "category": "scheduling",
   "description": "Set ARC3_PRIORITY_HUMAN_ACTIONS=300-1000 (A ~ level x 55/W). Add ARC3_PRIORITY_TAIL_SCALE multiplier in priority_value() (priority_scheduler.py ~240-252) with TAIL_LOOKUP=remaining scale 0.5 (4/3.5/2.5/0), or env-only unset TAIL_LOOKUP (base 2, cap 4). Keep SCORE_NORMALIZATION=1, REFRESH_QUEUE=1, TAIL_FADE=0.2, TOKEN_SCALE 62k. Validate with the replay simulator.",
   "mechanism": "Real per-level score is min(1.15,(b/a)^2) x level/W; measured efficiency 1.05-1.10 at 10-60 actions vs h=25 formula's 0.31-0.09, so A undervalues finishing deep high-weight levels by 3-10x and flat B=8 dominates. Sim: +5.3 to +5.6 (40/40 reps), +2.1-2.9 with 40% unclearable; expect ~60% on private scale.",
   "evidence": [
    "taaf game.py:381-413; docs.arcprize.org/methodology",
    "priority_scheduler.py:224; template cell 4",
    "replay_scheduler.py + levels.csv sims (mysim4/5); 82% of 838 completed levels <= baseline"
   ],
   "effort": "<2h",
   "expected_impact": "high",
   "status_in_franzen": "partial",
   "merged_from": []
  },
  {
   "id": "scheduler-level-count",
   "title": "Generalize scheduler level-count handling beyond [6,10] clamp",
   "category": "scheduling",
   "description": "parse_total_levels (priority_scheduler.py:63-84) clips N to [6,10]. Compute 'remaining' tail directly from N-level (>=3 ->8, 2->7, 1->5, 0->0) and normalize A with true N; keep clip only for the empirical 80k table.",
   "mechanism": "For N<6 the last level gets B=5 instead of 0; for N>10 tail drops too early. Free safety if private set has other level counts.",
   "evidence": [
    "priority_scheduler.py:58-60, 63-84, 238-241",
    "observed number_of_levels only 6-10"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "partial",
   "merged_from": []
  },
  {
   "id": "kv-budget-admission",
   "title": "KV-budget-aware admission: 12-14 streams while contexts are short",
   "category": "scheduling",
   "description": "Replace fixed ARC3_MAX_ACTIVE_STREAMS=10 with a gate (_PriorityGate, tool_agent.py ~2761) admitting another game while summed estimated contexts (_estimate_request_input_tokens) plus reply reserve stay under ~0.95M of the 1.01M KV pool; hard cap ARC3_MAX_ACTIVE_STREAMS_HARD=14 with MAXREQ=14, CUDAGRAPH_MAXBS=14, MAMBA_CACHE=84; keep 128k window for all games.",
   "mechanism": "Decode rises with batch (611/644/703 tok/s at 10/12/14 streams); context binds only when most games near 116k. First-pass waves cover ~30% of the run at about half-window, giving ~+4-5% tokens with no history loss.",
   "evidence": [
    "fz-arms-v4-control, ledger-streams12, streams14 runs (streams14 cut context and was flat 46.38)",
    "ARMS.md KV pool ~1.01M; first actions of second wave at ~890-1060 s",
    "sim +5% tokens ~ +1.97 pts (optimistic)",
    "PyTorch blog: HiCache for hybrid models future work"
   ],
   "effort": "1 day",
   "expected_impact": "medium",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "endgame-bank-it",
   "title": "Endgame 'bank it' mode in the final ~10-15% of runtime",
   "category": "scheduling",
   "description": "When time_remaining_seconds < ~0.12 x budget add a prompt line ('unfinished level scores 0, finished scores even if inefficient; commit to best hypothesis, batch top candidates, keep reasoning short'), disable BATCH_NOOP_BLOCK/STALE_STATE_BLOCK guards and lower LOCAL_ANALYZER_YIELD_TOKENS.",
   "mechanism": "Near deadline an uncompleted level scores 0 while completion at 3x baseline still banks 0.11 x level/W; actions are cheap vs ~850 tokens deliberation per action. ~30% of tokens go to levels unsolved at end.",
   "evidence": [
    "game.py:381-413",
    "ar25 transcript:267 time_remaining_seconds unused in prompts",
    "9.7M of 32.1M tokens on unsolved level at end; CONFIGURATION section 11 tail fade"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "action-economy-prompt",
   "title": "Level-weighted action-for-token trade prompt (probe cheaply early, thrifty late)",
   "category": "prompting",
   "description": "Replace prompts.py:122 and WORLD_MODEL_FREE line (:313-317) with scoring facts plus per-level policy: uncleared scores 0, later levels weigh more, clearing comes first, 1-3 action experiments beat further deliberation, on level 1 test every control freely, on later levels plan routes before executing. Add to state line 'Actions on this level so far: k (level L of N; weight L/(N(N+1)/2))' using _priority_total_levels (tool_agent.py:8773). Reconcile with PREFER_TOOL_CALLS wording.",
   "mechanism": "84% of cleared levels already beat baseline actions (median ratio 1.57); tokens, not actions, bind. Level-1 +8 actions costs only -0.23 mean while +20% on every level costs -1.65. Old-harness spend-actions/scoring-guide arms were negative or mixed, so measure trimmed mean and levels cleared.",
   "evidence": [
    "control benchmark: 105 cleared levels, 84% >=1.0; rescoring via game.py",
    "docs.arcprize.org methodology; arms.md failed arms c5cfbb0, b4",
    "harness-feature-matrix: Hansolo scoring explanation; 30% tokens on unsolved final levels",
    "prompts.py:122, 313-317"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "absent",
   "merged_from": [
    "Action-for-token trade prompt",
    "Level-weighted action-economy prompt"
   ]
  },
  {
   "id": "decisiveness-flags",
   "title": "Enable ARC3_PREFER_TOOL_CALLS and ARC3_OPENER_COMMIT_HINT",
   "category": "prompting",
   "description": "Env-only arm: ARC3_PREFER_TOOL_CALLS=1 (PREFER_TOOL_CALLS_LINE prompts.py:336-349: prefer another tool call over more reasoning) and ARC3_OPENER_COMMIT_HINT=1 (last opener line 'call action(actions) with the sequence your evidence supports'). Neither set in notebook nor in WRITEUP section 6.",
   "mechanism": "Targets the deliberation tail: top 10% of requests hold 41% of reasoning chars; 36% of executing turns run a single action; post-resume batches 6.7-6.9 vs 3.9-5.4 actions.",
   "evidence": [
    "tool_agent.py:2515-2533, 2586-2602; CONFIGURATION.md section 2",
    "control transcripts and benchmark.json"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "present-but-off",
   "merged_from": []
  },
  {
   "id": "auto-probe-sweep",
   "title": "Free control-mapping sweep at game start (harness presses each key once on level 1)",
   "category": "tools-repl",
   "description": "In solver._play_inner before the first analyzer turn at level 1 run each valid simple action once via step_env (skip MOUSE; send UNDO after each probe when valid; abort on game_over/level_completed; at most 5-6 actions), compute frame_diff, put a compact table in the first opener; flag ARC3_AUTO_PROBE. Prompt-only fallback: exemplar probing snippet in the action_num==0 opener. Skips 7/25 MOUSE-only games.",
   "mechanism": "First actions are the most token-expensive (median 6.8k tokens before 5th action, 12.0k before 10th; level 1 median ~17k). Level 1 weight is 1/W so +4/+8 actions cost -0.13/-0.23 mean; saved tokens go to later levels. Risk: irreversible state change on level 1.",
   "evidence": [
    "control benchmark; rescoring +4/+8/+15 -> 46.12/46.02/45.88",
    "game.py:381-413; solver.py:1480-1640; frame_diff.py",
    "18/25 directional, 7/25 MOUSE-only"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "absent",
   "merged_from": [
    "Harness auto-probe at game start",
    "Free control-mapping sweep at game start"
   ]
  },
  {
   "id": "fresh-eyes-reset",
   "title": "Fresh-eyes reset for stalled levels at next trim",
   "category": "planning-exploration",
   "description": "When a level has used >~60k generated tokens uncleared, the next natural trim drains deeper (~25k) and puts at the head of retained history: level facts (actions, resets, deaths), RLE action sequences per attempt, solved-level debriefs/ledger, and 'these approaches have not cleared the level; your goal model is probably wrong, re-derive from the board'. At most 1-2 times per level; ~0 cache cost since trim invalidates prefix anyway.",
   "mechanism": "28% of tokens go to never-cleared levels and completion odds fall with tokens; context full of failed hypotheses anchors the model. Prompt-only audit failed (audit60k 44.10) but trim-injected memory helped (+6.3). Larger context helped difficult games, so gate tightly.",
   "evidence": [
    "benchmark.json unsolved-level token share 0.281/0.286",
    "WRITEUP section 3 C calibration 62k",
    "arms.md audit60k, ledger-on-trim; talk_by_tufa_lab.md"
   ],
   "effort": "1 day",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "level-goal-function",
   "title": "Executable goal hypothesis: retained level_goal(frame) scored and falsified by the harness",
   "category": "planning-exploration",
   "description": "Add rule to COMPACT_TOOL_SESSION_ADDENDUM and LEVEL_START_USER_PROMPT: encode current goal as retained function level_goal(frame)->float (1.0 = reached). In _build_user_prompt, if in self._kept_functions, run it in a tiny sandbox call on previous/current frame and report 'level_goal(): 0.40 -> 0.65'; if >=1.0 but level not complete add 'your goal hypothesis is wrong or missing a condition'; at level-up say re-check; report errors once.",
   "mechanism": "Goal hypothesis currently lives only in prose thinking and is never tested (ls20: 15-21k-char debates on legend glyph). Code-level goal is falsified in one turn and gives search a heuristic; uses the channel the model writes reliably (11 of 2,299 replies have visible content), unlike prose-structure arms that failed.",
   "evidence": [
    "Tycho arXiv 2607.28287",
    "control: content_chars>0 in 11/2,299; ls20 20 retained functions",
    "arms.md failed thinking guide / scratchpad; hypothesis-loop inconclusive",
    "prompts.py:268; tool_agent.py:6461-6496"
   ],
   "effort": "1 day",
   "expected_impact": "medium",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "level-start-state-line",
   "title": "Show 'level L of N' and actions spent on this level in the per-turn state line",
   "category": "prompting",
   "description": "State line (tool_agent.py:7073-7077) shows only global step and level; its 'out of observed max level' branch is dead (observed_max_level from history frames, :6812-6815). Replace with 'level L of N, A actions spent on this level' (N from _priority_total_levels :8773, A from run.actions_per_level); optionally one system line that per-level score falls with square of actions.",
   "mechanism": "Model cannot see per-level spend or levels remaining, so cannot balance probing vs committing; prefix-safe (only newest message changes).",
   "evidence": [
    "tool_agent.py:7073-7077, 6812-6815, 8773-8795",
    "taaf game.py:380-410"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "truncated-prompt-text-bugs",
   "title": "Fix system-prompt text bugs and state image geometry",
   "category": "bugfix-quickwin",
   "description": "prompts.py:132 lacks trailing newline so '...1x1 tokens.- Some games are logic...' merges bullets; fix grammar ('are usually be rendered', 'Sometime'); prompts.py:136 paragraph lacks '- '; tool_agent.py:3814/3822-3828 diff-image sentences appended with leading space instead of bullets. In MULTIMODAL_CONTEXT_ADDENDUM (prompts.py:241-246) add 'board image is the 64x64 grid at N px per cell (640x640); cell (row,col) occupies y in [N*row,N*row+N); a 0-1000 position maps to cell floor(v*64/1000); use Python for exact coordinates' with N from MULTIMODAL_UPSCALE; reword opener line 'raw numeric color IDs not available' (tool_agent.py:7100) if helpers land.",
   "mechanism": "Zero-cost fix of text read every request; the 'no explicit player avatar' rule is buried after a typo'd bullet; model guesses image size wrongly ('736 px ... ~11.5 px/cell'). System prompt constant so cache unaffected.",
   "evidence": [
    "prompts.py:127-136, 241-246; tool_agent.py:3812-3828",
    "rendered competition system prompt and transcripts show run-on",
    "fz transcripts image-size misreads; voxel51 Qwen3.8 0-1000 grid"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": [
    "Fix system-prompt text bugs",
    "Perception prompt fixes: state image geometry",
    "Fix prompt and tool-doc consistency bugs (a)"
   ]
  },
  {
   "id": "tool-doc-contradictions",
   "title": "Remove contradictions between tool schema/prompts and competition config",
   "category": "prompting",
   "description": "(a) With ARC3_PERSISTENT_FUNCTIONS=1 tool description says 'ephemeral snippet' (tool_agent.py:420-431, 7199-7201); say 'variables reset; top-level functions are retained' and add frame_diff and last_animation_frames. (b) With ARC3_MEMORY_SECTIONS=off, prompts still say 'update your working world model' (prompts.py:119, tool_agent.py:6983, 7034); reword to 'state it in your reasoning'. (c) Keep prompts.py:378 in sync with the real timeout. Also tool_agent.py:420 omits frame_diff/last_animation_*.",
   "mechanism": "Tool-schema text sits beside the call; 'not saved' pushes redefining helpers, wasting retention savings. Franzen measured a broken WM instruction dropped replies carrying text from 76% to 43%.",
   "evidence": [
    "tool_agent.py:420-431, 7199-7201, 7106-7116, 6983, 7034",
    "prompts.py:119, 355-365, 378",
    "CONFIGURATION/WRITEUP"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": []
  },
  {
   "id": "stale-guard-mouse",
   "title": "Refine stale-state guard for MOUSE/SPACE select actions",
   "category": "bugfix-quickwin",
   "description": "Sandbox arms the stale flag after any interior no-op (python_tool_sandbox.py:581-602; refused in solver.py:1530-1590). Flag-gated: do not arm after MOUSE/SPACE when board_changed (any cell) was true or animation had transient pixels, or arm only after directional no-ops. A/B with the paired significance test.",
   "mechanism": "145 of ~310 stale-guard firings followed a MOUSE no-op vs ~150 for all four directions; refused follow-up clicks force a new LLM round trip (~1-3k tokens). Effect uncertain.",
   "evidence": [
    "python_tool_sandbox.py:581-602; solver.py:1530-1590",
    "fz transcripts firing breakdown",
    "WRITEUP section 5"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "partial",
   "merged_from": []
  },
  {
   "id": "skip-list-notes",
   "title": "Skip list: audited issues inert under the competition config",
   "category": "other",
   "description": "Checked against actual notebook env and judged not worth fixing: _build_user_message dropping images without grid image (tool_agent.py:6606), _summary_turn_context_lines wrong keys (:6736-6760, summaries off), OpenRouter streaming response.json (tools/chat.py:147, offline vllm), unused DEATH_GUARD/NOOP_GUARD addenda (prompts.py:29, 319-334, guards off), exact-string prompt rewrites (verified they fire; re-verify after edits), multi-tool-call discard (:9682; 14 of ~17.4k), board-dump guard off (:2165; 0.1%), segmentation cost (20-200 ms), retry backoff, 80k-table truthiness, frame_diff rotation modulus, viewer notes. Also: XML vs JSON tool-call format is not a problem (0 markup recoveries/malformed in 2,299 requests).",
   "mechanism": "Saves limited days for the items that matter.",
   "evidence": [
    "nb.txt:10175-10260 competition env",
    "fz transcripts tool_call_count 2 only 14 times; 3 of 4,521 results with full-width dumps",
    "local segmentation benchmark; arms.md xml-format arm no effect"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent",
   "merged_from": [
    "Skip list: Notes that do not cost score",
    "(d) skip tool-call-format work"
   ]
  },
  {
   "id": "module-library-with-tests",
   "title": "LHS-style saved modules with frozen-observation regression tests",
   "category": "tools-repl",
   "description": "Port lordhansolo python_workspace.py (301 lines): save_module(name, source, tests, replace_tests), load_module, read_module, saved_modules; tests run check(actual, expected, label) against frozen observations; a save breaking earlier suites is rejected unless replace_tests=True; show module signature catalog each turn. Opt-in arm alongside function retention.",
   "mechanism": "Makes learned simulators cumulative across levels/trims without trusting memory. Rodionov finds executable models not universally beneficial and lordhansolo cannot confirm any gain, so expect small, uncertain effect for a large port.",
   "evidence": [
    "LHS python_workspace.py",
    "lordhansolo writeup",
    "arXiv 2607.15439"
   ],
   "effort": "2-3 days",
   "expected_impact": "low",
   "status_in_franzen": "partial",
   "merged_from": []
  },
  {
   "id": "mechanics-catalog-circuit-breaker",
   "title": "A/B short cached mechanics catalog and no-action circuit breaker",
   "category": "prompting",
   "description": "Kinetic v8's 19-line knowledge pack in the cached system prompt plus mechanics catalog sent once per game and level (fixed-step movement, push/carry, select-then-act, linked objects, cycle controls, goal types, budgets, hazards) and a rule 'if three calls in a row sent no action, the next must act'; harness nudge after K consecutive no-action tool calls (Scott's F4 used K=6).",
   "mechanism": "A prior over mechanic families could shorten hypothesis search; breaker cuts long no-action deliberation (unsolved levels spent 76% of calls with no action vs 55% cleared). But extra standing text crowds history; Kinetic v8 scored 27.28 (flat), rellik13's text additions dropped scores, audit60k/scoring guide found nothing. Low priority.",
   "evidence": [
    "kinetic-learning-by-playing-v8.txt:10114-10175, 10600",
    "Siriki writeup 'More text made the agent worse'",
    "arms.md audit60k 44.10 vs 46.25"
   ],
   "effort": "1 day",
   "expected_impact": "low",
   "status_in_franzen": "tried-no-gain",
   "merged_from": []
  }
 ]
}
```
