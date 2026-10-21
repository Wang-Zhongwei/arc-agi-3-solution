# arc3-idea-sweep: sweep:model-config

Workflow run `wf_6b16ae20-e2c`, phase Sweep. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "ideas": [
  {
   "title": "Relax speculative-decoding acceptance thresholds (accept more MTP drafts)",
   "category": "serving",
   "description": "Franzen's launch sets --speculative-accept-threshold-single 1.0 and --speculative-accept-threshold-acc 1.0 (CFG SPEC_ACCEPT_SINGLE/SPEC_ACCEPT_ACC). In Pennyroyal's verify kernel, a draft token is accepted when coin <= p_target/threshold_acc OR p_target >= threshold_single, so 1.0/1.0 means strict target-only sampling. Sweep two arms on the 25-game, 121-min demo: (a) single=0.6, acc=0.8 (mild) and (b) single=0.3, acc=0.5 (aggressive). Change only the two CFG values; nothing else changes. Read the gain in serve.log ('accept len', 'gen throughput') and the score. Second-tier knobs, probably not worth it: --speculative-use-rejection-sampling (lossless) exists, but it needs a target-vocab draft distribution, so the FR-Spec 64k token map would have to be turned off. That likely costs more than it gains.",
   "mechanism": "Decode is about 85% of wall time. The MTP chain currently accepts 2.69 tokens/step at T=0.7 (per-draft acceptance about 0.72). Greedy only reaches 2.81, so draft quality, not sampling, is the main limit. Relaxed thresholds also accept drafts the target ranks second but still gives high probability. Expect accept len of about 2.9-3.1, so roughly +5-15% generated tokens in the same runtime. The quality risk is a mild bias toward the draft's argmax (like a slightly lower temperature). The user's runs show the score is insensitive to temperature (0.6 vs 0.7 gave 45.92 vs 46.25), so the bias should be harmless.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/external/dfranzen/nb.txt:10422-10423 (SPEC_ACCEPT_SINGLE=1.0, SPEC_ACCEPT_ACC=1.0) and :10699-10700 (passed to sglang serve)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs/fz-arms-v4-control/serve.log server_args: speculative_accept_threshold_single 1.0, _acc 1.0, speculative_use_rejection_sampling False (confirms flags exist in Pennyroyal v2.5.3)",
    "Mean 'accept len' from serve.logs: control 2.691, temp06 2.722, greedy+mxfp8 (fz-tput-all-v7) 2.809 -> sampling temperature barely moves acceptance",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/external/sirikilohit/duck-qwen38-nvfp4-mtp-vllm-smoke-v1/src/sglang-rtxpro6000/src/sglang/srt/server_args.py:2241-2250 (threshold semantics: accept if p > single; accept prob raised from p to min(1, p/acc))",
    "same tree: speculative/eagle_utils.py:~760-850 (tree_speculative_sampling_target_only with draft_probs=zeros unless rejection sampling)",
    "https://icml.cc/virtual/2026/poster/66788 (ARC-Decode: relaxed acceptance up to 1.6x over EAGLE-3 under sampling with negligible quality change)"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Fix Mamba/GDN state precision: fp16 + stochastic rounding instead of bf16 round-to-nearest",
   "category": "model-config",
   "description": "Franzen serves with --mamba-ssm-dtype bfloat16 and leaves Pennyroyal's --enable-mamba-cache-stochastic-rounding off (default). The recurrent state of the 36 Gated-DeltaNet layers (75% of layers) is therefore re-rounded to 7 mantissa bits on every decode step, over contexts of 58k-128k tokens. Arm A: --mamba-ssm-dtype float16 --enable-mamba-cache-stochastic-rounding (same 16-bit footprint, 10 mantissa bits, unbiased rounding; the flag requires fp16 + CUDA and supports the flashinfer mamba backend). Check the boot log to confirm stochastic rounding actually covers the linear-attention (GDN) state writes, and smoke-test for fp16 overflow. Fallback arm B: --mamba-ssm-dtype float32. B doubles the state memory, which competes with the KV pool (KV already peaks at 0.99 usage), so prefer A.",
   "mechanism": "NVIDIA's Nemotron 3 Ultra report (hybrid Mamba MoE) found that mantissa precision and stochastic rounding are the key ingredients for keeping FP32-cache accuracy and verbosity. Naive 16-bit round-to-nearest state storage shows 'recurrent rounding bias'. If bf16 RTN state slightly degrades long-context recall or inflates reasoning length, fixing it gives better decisions and fewer tokens per decision for zero throughput cost. The expected effect is uncertain: short vLLM tests reported identical outputs with bf16 state.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/external/dfranzen/nb.txt:10418 (SSM_DTYPE='bfloat16')",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs/fz-arms-v4-control/serve.log server_args: mamba_ssm_dtype 'bfloat16', enable_mamba_cache_stochastic_rounding False, mamba_cache_philox_rounds 0",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/external/sirikilohit/duck-qwen38-nvfp4-mtp-vllm-smoke-v1/src/sglang-rtxpro6000/src/sglang/srt/server_args.py:2640-2665, 6737-6770 (flag requires --mamba-ssm-dtype float16; flashinfer supported)",
    "https://arxiv.org/abs/2606.15007 Sec. 4.5 'SSM Cache Optimization': 'FP16 with stochastic rounding preserves FP32-cache accuracy and verbosity'; 'mitigating the recurrent rounding bias observed with naive FP16 round-to-nearest cache storage'",
    "https://huggingface.co/Qwen/Qwen3.8-Flash-Next (12 x (3 x Gated DeltaNet -> MoE, 1 x sparse attention -> MoE): ~75% recurrent layers)"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Thinking budget-forcing: turn runaway thinking into an action instead of a 12k-token truncation",
   "category": "model-config",
   "description": "In the user's control run, 25 of 2,299 responses hit finish_reason=length. All 25 had 0 tool calls and 25.8k-41.2k chars of pure (non-degenerate) thinking, and other arms show 20-37 per run. The reply is kept as reasoning_content, the turn yields on the token budget, and the resume opener only says 'produced no tool call'. Tier 0 (<2h): set a flag when the last reply was length-truncated (tool_agent.py ~9410 already detects finish_reason=='length'). Add one line to the next opener: 'your previous reply hit the output limit while thinking; your reasoning is above; act on it now with one compact snippet'. Optionally lower LOCAL_ANALYZER_MAX_OUTPUT from 12288 to ~8192 so the cap works as a budget. Tier 1 (~1 day): on a length cut with an open thinking block, send one continuation request: messages + {assistant, reasoning_content: partial + ' Thinking budget reached; acting on best hypothesis now.', content: ''} with continue_final_message=true, add_generation_prompt=false, max_tokens~2048. The template then closes </think> and the model emits the tool call. Reuse _recover_tool_calls_from_markup if the qwen3 reasoning parser files the output under reasoning. Do NOT rely on SGLang's native thinking_budget: it is reported broken for Qwen3.6-vocab models, and custom logit processors are not applied in the EAGLE verify path.",
   "mechanism": "25 x 12,288 is about 307k of 4.48M generated tokens (~7%) per demo run, spent on replies that execute nothing. Each one also costs that game ~3 minutes of decode and puts a 12k-token blob into its context, which speeds up the next trim. Budget forcing (s1) converts the same thinking into an executed action, recovering most of that compute and context.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs/fz-arms-v4-control/transcripts/*.txt: 25 'finish_reason: length' vs 2274 tool_calls; truncated replies have tool_call_count 0, reasoning_chars 25,850-41,241; 23/25 followed by a tool call on the next request",
    "Same pattern in other arms: temp06 23, ledger-on-trim 24, streams14 20, ucb 37, audit60k 28, ledger-streams12 26 truncations",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py:9410,9472,9570 (truncated_tool_call is only set for malformed calls, never for reasoning-only truncation)",
    "https://arxiv.org/abs/2501.19393 (s1: budget forcing by appending the end-of-thinking delimiter)",
    "https://github.com/sgl-project/sglang/issues/25536 (thinking_budget / --enable-strict-thinking not enforced for Qwen3.6; token ids 248068/248069 differ from Qwen3's)",
    "local sglang speculative/eagle_utils.py verify path applies penalties/logit_bias/grammar mask only, not custom logit processors"
   ],
   "effort": "1 day",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "Trim-time reasoning compaction: strip old <think> blocks only when the drain fires",
   "category": "memory-context",
   "description": "When the token trimmer drains (the prefix is invalidated anyway), first delete reasoning_content/reasoning from all assistant messages except the most recent K (e.g. K=6-8), keeping their tool_calls, code, tool results and user prompts. Only then drop whole messages if the context is still over the drain target. Implement in _persistent_history_messages (tool_agent.py ~8737) and the request-time _trim_messages_for_context path when force_drain or token_trim_dropped is true. Optionally leave a 1-line stub ('earlier reasoning elided'). Add a new env ARC3_TRIM_KEEP_REASONING_TURNS. Test alone and combined with the user's best arm (ledger-on-trim, 52.56).",
   "mechanism": "Past thinking is roughly 55-60% of retained context: about 5.2k reasoning chars/request vs 0.57k code, 0.62k tool output and ~1.8k prompt chars. Compaction at trim time keeps the cache intact between trims, yet lets the ~58k retained block hold about 2x more action/observation turns. Those turns are the evidence the model needs not to re-test mechanics. It also allows a smaller drain target (fewer trims). Trims account for most of the ~10.7M prefill tokens per run and evict prefixes from a KV pool that peaks at 0.99. Unlike Franzen's failed 'remove all thinking' and the GB10 preserve_thinking=false arm (which caused re-deriving), recent reasoning stays intact.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs/fz-arms-v4-control/transcripts: mean reasoning 5,182 chars/request (p90 12,759), code mean 570 chars, tool result mean 620 chars",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs/fz-arms-v4-control/serve.log: prefill cached fraction 0.936, 10.68M new prefill tokens; 'full token usage' peaks at 0.99",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/WRITEUP.md:69 (removing thinking entirely -> large regression) and :97 (58k retained after trim)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs/fz-ledger-on-trim/summary.txt (52.56 vs control 46.25: information that survives trims helps)",
    "GB10 note (final-c1 branch gb10/token-efficiency notes/gb10-arms/token-efficiency.md): dropping ALL earlier reasoning every request -> model re-derived mechanics"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Make ARC3_HISTORY_IMAGE_KEEP cache-friendly: drop old images only at drain time",
   "category": "memory-context",
   "description": "ARC3_HISTORY_IMAGE_KEEP exists but is applied at the end of every analyzer turn (_persistent_history_messages, tool_agent.py ~8737-8741). Every new image-bearing message therefore pushes an older image out of the window, which changes a mid-history message and invalidates the cached prefix each turn. That is why it can't be used as-is. Move _apply_history_image_window after the token trim and run it only when the trim actually dropped messages (token_trim_dropped / force_drain). Then set ARC3_HISTORY_IMAGE_KEEP=4-6. Pairs naturally with the reasoning-compaction idea.",
   "mechanism": "Images are about 12% of prompt tokens late in a game (402 tokens each at 10x). Removing them from the older part of the retained block at trim time frees ~5-7k tokens per drain for text history, or lengthens the interval between trims. The cache-hit rate is unchanged, because the prefix is already being rebuilt at that moment.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py:1227-1246 (_history_image_keep), :8737-8741 (applied every turn before trimming)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/CONFIGURATION.md:94-96 ('Filtering historical images changes request prefixes and can affect prefix reuse')",
    "control transcripts text_token_calibration lines: images 10,050 of 84,483 prompt tokens; 7,638 of 60,170; 5,226 of 57,586"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "present-but-off"
  },
  {
   "title": "Align board-image upscale to the vision patch grid (try 16x and 8x instead of 10x)",
   "category": "perception",
   "description": "Qwen3.8-Flash-Next's vision tower is Qwen3-VL style: 16-px patches with a 2x2 merge, so 32 px per LLM token (the harness hard-codes this in its estimator). At MULTIMODAL_UPSCALE=10, a 64x64 board becomes 640 px, and each 16-px ViT patch straddles 1.6 cells, mixing neighbouring cells' colors inside one patch embedding. Arms: 16x gives 1024 px, exactly one cell per ViT patch and 2x2 cells per LLM token, at 1026 tokens/image (16x is the harness's source default; Franzen only reported 4x vs 10x). 8x gives 512 px, aligned 2x2 cells per patch, at 258 tokens/image (36% cheaper than 10x). Run each with all else equal; combine 16x with drain-time image stripping to contain context growth.",
   "mechanism": "Franzen's single biggest image-related gain was 4x->10x upscaling. Grid-aligned patches give the encoder pure single-cell or exactly-2x2 inputs. That should make cell counting, positions and thin-wall perception more reliable, with fewer wasted probe actions and less re-checking via ascii. 8x tests whether alignment can match 10x at lower token cost.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py:4082-4084 (_VISION_PATCH_PIXELS=32 '16px patches, 2x2 merge'; 402 tokens at 10x)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/external/sirikilohit/duck-qwen38-nvfp4-mtp-vllm-smoke-v1/src/sglang-rtxpro6000/src/sglang/srt/configs/qwen4_exp.py:7-12 (Qwen4ExpVisionConfig subclasses Qwen3VLVisionConfig)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/WRITEUP.md:203 (biggest image gain: 4x -> 10x)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/vision_context.py:21 (MULTIMODAL_UPSCALE source default 16)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/external/dfranzen/nb.txt:10217 region ('MULTIMODAL_UPSCALE': '10')"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "Global reasoning_effort=low A/B (and never switch effort mid-prefix)",
   "category": "model-config",
   "description": "The Qwen3.8-Flash-Next template defaults to xhigh and injects 'think carefully..., validate key assumptions, consider plausible alternatives...' at the very top of the system message. 'medium' only removes that sentence (the GB10 test showed no shortening). 'low' injects 'Keep your thinking brief and focused'. Run one arm with --default-chat-template-kwargs '{\"preserve_thinking\":true,\"reasoning_effort\":\"low\"}', which is constant per game and so cache-neutral. Gotcha: because the effort text is the first system-prompt content, the existing ARC3_REASONING_EFFORT_LADDER invalidates a game's entire prefix (a full ~100k re-prefill) on every switch and again when it resets after an action. Keep it off. If a dynamic variant is wanted, switch effort only at drain/handover boundaries, e.g. low once tail fade starts in the final 20%.",
   "mechanism": "Generation is the main cost. If 'low' cuts tokens per decision by 20-40% while keeping decision quality, games get more model turns and levels per budget. The model card warns lower effort can hurt multi-turn agentic completion, so this is a measured gamble. It is cheap to test.",
   "evidence": [
    "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/chat_template.jinja (reasoning_instructions inserted at the start of the first system message; xhigh/low texts; medium adds nothing)",
    "https://huggingface.co/Qwen/Qwen3.8-Flash-Next ('lower reasoning effort does not always reduce overall task completion time')",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py:1778-1797 and :7221-7232 (ladder puts reasoning_effort into chat_template_kwargs)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/GB10-OVERNIGHT-RESULTS.md:47 ('medium does not shorten thinking (only low asks for brevity)')"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "Use the official thinking-mode temperature 1.0 (top_p 0.95, top_k 20)",
   "category": "model-config",
   "description": "Set LOCAL_ANALYZER_TEMPERATURE=1.0 (keep TOP_P 0.95, TOP_K 20, provider 'vllm' so top_k is actually sent). This is the Qwen3.8-Flash-Next model card's recommended thinking-mode setting; Franzen used 0.7 and the user already tested 0.6. Watch serve.log accept len (expect a small drop) alongside the score.",
   "mechanism": "The model was presumably RL-tuned and validated at T=1.0. Sampling nearer its training temperature can improve hypothesis diversity and reduce getting stuck re-trying one idea on stalled levels. The cost is about 3% lower MTP acceptance. The user's data shows the score is flat between 0.6 and 0.7, so the expected effect is small either way, but it is a one-line arm.",
   "evidence": [
    "https://huggingface.co/Qwen/Qwen3.8-Flash-Next (thinking mode: temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0)",
    "https://unsloth.ai/docs/models/qwen3.8-next (same recommendation)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/external/dfranzen/nb.txt:10190-10192 (temperature 0.7, top_p 0.95, top_k 20)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs/fz-arms-v5-temp06/summary.txt (45.92) vs fz-arms-v4-control/summary.txt (46.25)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/utils/openai_compat.py:155-159 (top_k only sent when provider normalizes to vllm)"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "Batch-aware expert routing instead of a static num_experts_per_tok change",
   "category": "serving",
   "description": "Static top-k changes are already being tested: experts16 was slower and worse (38.69 vs 46.25; 504 vs 611 tok/s), and experts8 is pending. A finer-grained alternative is Opportunistic Expert Activation. At each decode/verify step, take the union of each token's top-k0 experts (e.g. k0=6) across the batch (~34 tokens = 8.5 streams x 4 verify tokens). Then let each token fill its remaining slots up to k=10 from experts already in that union, renormalizing the router weights. Implement in SGLang's top-k selection (pure tensor ops, so CUDA-graph safe) behind an env flag. Validate tokens/s and accept len on a 20-min smoke run before a full arm.",
   "mechanism": "With 512 experts, 10 per token and ~34 tokens per step, about 250 distinct experts' INT4 weights are streamed per MoE layer per step. Decode is memory-bandwidth bound, so cutting distinct experts cuts step time while each token still gets 10 experts. The paper reports 39% lower MoE decode latency on Qwen3-30B at batch 16 with no statistically significant accuracy loss. That would beat static k=8 on the quality-for-speed trade-off.",
   "evidence": [
    "https://arxiv.org/abs/2511.02237 (Opportunistic Expert Activation: 39% MoE-layer decode latency reduction on Qwen3-30B, 15% on Qwen3-235B, batch 16, no retraining)",
    "https://huggingface.co/Qwen/Qwen3.8-Flash-Next (512 experts, 10 routed + 1 shared)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs/fz-experts16-full/summary.txt (38.69, 504.58 tok/s) vs fz-arms-v4-control/summary.txt (46.25, 611.12 tok/s)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/franzen-moe/kaggle/arms.json (experts8 arm via TEXT_CONFIG_OVERRIDE)",
    "fz-arms-v4-control/serve.log: mean #running-req 8.56, accept len 2.69 (~34 tokens per verify step)"
   ],
   "effort": "2-3 days",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "Resume-turn diet: stop re-attaching the unchanged board image on token-budget yields",
   "category": "prompting",
   "description": "Set ARC3_RESUME_PROMPT_IMAGES=0. A resume only happens when no action executed, so the board equals the one already in context. Optionally pair it with LOCAL_ANALYZER_YIELD_TOKENS=3072 (from 2048), since 30% of turns end by token-budget yield. This is a zero-code arm. It is cache-neutral because it only changes newly appended messages.",
   "mechanism": "The control run had 466 turn_token_budget yields across 1,560 turns. Each appended a ~2.5k-char opener plus a 402-token image identical to the previous one, about 190k context tokens per 25-game run. Removing the duplicates slows context growth and so delays trims (each trim re-prefills ~58k tokens), with no loss of information. Franzen found prompt-shortening variants inconclusive, so expect a small gain.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs/fz-arms-v4-control/transcripts: 466 'turn_token_budget' of 1,560 turns; 434 'executed NO actions' openers, mean 2,539 chars, 1 image each",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py:1249-1260 (_resume_prompt_images_enabled, default True)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/WRITEUP.md:299 (resume prompt/yield variants inconclusive)"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "present-but-off"
  },
  {
   "title": "Do not spend the remaining days on a model swap",
   "category": "model-config",
   "description": "Searches found no open model that fits offline on one RTX PRO 6000 and has evidence of beating Qwen3.8-Flash-Next on agentic or visual tasks. Its card shows it ahead of Qwen3.8-27B on AndroidWorld (84.5 vs 81.9), ClawEval-MM (64.4 vs 57.4), RealWorldQA (88.5 vs 85.9) and CoWorkBench (73.9 vs 70.7). The newer large open models (Kimi K3, DeepSeek V4 Pro) do not fit. Spend the effort on the generation and serving knobs above, which tune the existing model, rather than porting a new architecture into Pennyroyal.",
   "mechanism": "Franzen's score gains came from model upgrades. But the only candidates newer than Flash-Next do not fit the hardware, and porting one would use most of the 5-day window with high risk of a regression in throughput or prefix caching.",
   "evidence": [
    "https://huggingface.co/Qwen/Qwen3.8-Flash-Next (benchmark table vs Qwen3.8-27B and Claude-Opus-4.6)",
    "https://kingy.ai/ai/best-open-weight-ai-models-2026/ (2026 open-weight landscape; Kimi K3 2.8T, DeepSeek V4 Pro 1.6T, both far beyond 96 GB)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/WRITEUP.md:15 (each model switch helped; Flash-Next is the latest)"
   ],
   "effort": ">3 days",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  }
 ]
}
```
