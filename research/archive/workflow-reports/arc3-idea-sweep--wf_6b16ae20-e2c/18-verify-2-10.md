# arc3-idea-sweep: verify:2/10

Workflow run `wf_6b16ae20-e2c`, phase Verify. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "verdicts": [
  {
   "id": "batch-aware-expert-routing",
   "keep": true,
   "reason": "Survives refutation. Absent: no routing/top-k change in serving/patches or build_bundle_pennyroyal.sh; the notebook uses moe_runner_backend auto (GPTQMarlinMoE, serve.log:42). The user's experts16 arm is a different idea (static top-k up), and experts8 (static down) has no output. Citations are real: XShare 2602.07265 (up to 14% spec-decode throughput), MoE-Spec 2602.16052 (10-30% over EAGLE-3), OEA 2511.02237 (-39% MoE-layer latency on Qwen3-30B, but only -15% on Qwen3-235B, so the 39% is a best case). Config fetched from HF: 512 experts, top-10, 48 layers. I corrected the slope claim: serve.log decode at bs10 is 588 tok/s (experts16-full) vs 711 (control), which is -17%, not -25%; accept len is unchanged (2.694 vs 2.691). Gabriel's PERF_CEILING.md shows MoE at 26% of verify time at bs1; at bs10 (about 40 verify tokens, roughly 250 unique experts per layer) it should be about 60-65%, so cutting unique experts 25-30% is worth about +15% decode before ~1 ms of extra small kernels per step. Risks: (a) the routing change also perturbs the KV/GDN state written during verify, not just the logits; (b) padded CUDA-graph rows (bs5/6 pad to 7) must be masked out of the score aggregation; (c) the real unique-expert count may be lower than uniform routing predicts because the 4 tokens of a request are correlated. The user's own data says +15% throughput (streams14) did not visibly move score (46.38 vs 46.25), so the payoff is about +2 LB points at best, with quality unknown. Patch topk.py in the venv after the wheel install (no wheel rebuild needed) and A/B in throughput runs first.",
   "status_in_franzen": "absent",
   "impact": 3,
   "effort_days": 2.5,
   "confidence": "low",
   "first_step": "Run the already-pushed experts8 arm (arms.json:131-139, kernel mellamoa/fz-experts8) as a probe of quality sensitivity to fewer experts, and log mean unique experts per layer per verify step (add a counter in python/sglang/srt/layers/moe/topk.py in verify mode, or use the expert-distribution recorder). If the mean is below about 220, stop. Otherwise add env-gated Variant A: aggregate softmax gate probs over non-padded tokens, torch.topk to B=192, masked_fill the other router logits to -inf, then the normal top-10."
  },
  {
   "id": "frspec-map-from-transcripts",
   "keep": true,
   "reason": "Real but much smaller than claimed. FR-Spec is already ON (notebook cell 12: FRSPEC=True, hot_tokens_64k.pt, TOKEN_MAP_SHA), so only the map is generic. I re-tokenized the control transcripts (THINKING plus python calls, 4.56M tokens) with the Flash-Next tokenizer: 1.268% fall outside the 64k map (the idea says 1.26%), led by Hmm / ' Hmm' / ' hmm' (about 23.5k). I then cross-validated by game (5 folds, map built from the other games' transcripts, 32768 lowest ids plus top transcript tokens, size 65536): held-out miss rate 0.34%, or 0.71% without the base ids. The saving is about 0.9 pts of miss per draft position, roughly 4 * 0.9% * 0.74 = +1% accept length, so the idea's +1-3% is optimistic. A 32k map saves about 0.7% of step time (lm_head bytes) but raises misses. Lossless and low-risk, but throughput gain is at most 1-1.5%. lordhansolo's configs/draft_vocab_32k.json (23,853 ids from 125 transcripts) exists but there is no measured effect in tong-m2-comparison. Gabriel's FR-Spec +9% is vs the full vocab, not vs this map.",
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 0.5,
   "confidence": "medium",
   "first_step": "Extract THINKING plus TOOL CALL text from the fz-*/transcripts dirs, build a 65536 map (32768 base ids plus top transcript tokens plus specials), torch.save the id list, upload it as a dataset, update TOKEN_MAP_SHA in template cell 12, and compare accept len in the serve.log of a 20-min run against 2.69."
  },
  {
   "id": "rejection-sampling-frspec",
   "keep": false,
   "reason": "Drop. The FIXME is real (eagle_worker_v2.py:228-240 raises ValueError when the draft vocab differs from the target vocab) and the flag is off (server_args speculative_use_rejection_sampling False). But the gain is unproven and could be negative. Current verify with thresholds 1.0/1.0 and an argmax (topk=1) draft is already exact target-only acceptance with accept prob p(d). Rejection sampling gives sum min(p,q) only if the draft SAMPLES from q, and it is lower than p(d) whenever the draft is flatter than the temperature-0.7 and top-k20 target (p=.9, q=.4/.3/.3 gives .5 vs .9). The observable ceiling is the greedy gap (2.81 vs 2.69, 4%, arms.md:119), minus extra draft sampling and [bs,4,248k] prob traffic. It needs a serving patch plus a wheelhouse rebuild for at most +2-3% throughput, which is under 1 offline point and unmeasurable in score. Not worth the days against a +12 LB target.",
   "status_in_franzen": "present-but-off",
   "impact": 1,
   "effort_days": 1,
   "confidence": "medium",
   "first_step": "None recommended. If revisited: scatter the draft hot-vocab probs through the d2t map in eagle_worker_v2.py init_token_map and run a 20-min throughput A/B against the control accept len of 2.69."
  },
  {
   "id": "cudagraph-bs-gaps",
   "keep": true,
   "reason": "Verified but nearly free of upside. graph_bs = sorted({1,2,4,7,8,9,10}) in template cell 12 and in Franzen's nb.txt:10663. Control serve.log has 4,803 decode lines: bs5=71, bs6=170, bs3=11 (5.0% at bs5-6), so padding to 7 happens on about 5% of steps. A 28-token vs 20-token verify costs about 15% more MoE time, which is worth about +0.5-0.8% overall at best. Costs: capture is 144.8 s for 7 sizes (about +40-60 s startup, 0.15-0.2% of the run) and about +0.3-0.5 GB graph memory off the KV pool (KV usage in the control: mean 65%, p90 86%, max 99%). Net is about +0.3-0.6% throughput, so score impact is negligible. Add only 5 and 6 (skip 3); if stream count changes, the list must cover the new MAXREQ.",
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 0.1,
   "confidence": "high",
   "first_step": "In template cell 12 change the graph_bs set to {1,2,4,5,6,7,8,9,10}, run a 20-min throughput check, and confirm in serve.log that the capture time and avail mem after capture (4.25 GB today) did not regress."
  },
  {
   "id": "startup-and-deadline",
   "keep": true,
   "reason": "Keep only the trivial part; the rest is not worth the risk. Verified numbers: weight load 386 s (serve.log:47), verify-graph capture 144.8 s (:94), READY after 595 s and Starting benchmark at t=667 s (fz-arms.log), SERVER_STARTUP_TIMEOUT 12 min (cell 4), 532*60 (cell 16), soft_end None in TRUE_SUBMISSION (cell 20). Each play minute is 0.19% of 532. But the load is I/O-bound (the 185 GB precache took 353 s, about 0.52 GB/s, and ends at the same time as the weight load), so persisting the Triton/FlashInfer caches cannot beat that floor and risks a stale-cache launch failure on the real rerun. The safe gain is starting the model precache at t~20 s instead of t~130 s (server launch +60 s delay), worth about 1 min, or +0.2% runtime, about +0.1 offline point. Dropping --warmups structured_output saves about 10-15 s (the harness uses no grammar). Re-anchoring the deadline: game started_at is set at benchmark start, so the nominal game end (667+31920 s) is already about 3 min past the 32400 s budget the notebook asserts. It is unclear whether Kaggle scores a notebook killed at the cap. Franzen's and the user's submissions did score, so the deadline is only a 0.5% tail-fade timing issue. Confirm from a real submission log before touching it.",
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 0.15,
   "confidence": "medium",
   "first_step": "In template cells 6 and 12 start precache_model_thread right after the wheelhouse precache (delay=0, 3-4 threads) and remove '--warmups structured_output'; compare READY time to 595 s."
  },
  {
   "id": "mamba-state-precision",
   "keep": false,
   "reason": "Refuted at the implementation level; drop. The Nemotron citation is real: arXiv 2606.15007 section 4.5 (checked in the PDF) says FP16 SSM cache with stochastic rounding preserves FP32-cache accuracy and verbosity, and that naive FP16 round-to-nearest is biased. But (1) in the Pennyroyal source, --enable-mamba-cache-stochastic-rounding is consumed only by the Mamba2 selective_state_update path (kernels/ops/mamba/triton_ops/ssu_dispatch.py and mamba_ssm.py; server_args.py:2656 says 'FP16 Mamba SSM cache states'); nothing reads it in layers/attention/linear/gdn_backend.py. Flash-Next has GDN layers, so the flag would not touch the state this model rounds. (2) The serve.log kernel dispatch is FlashInferGDNKernel verify/decode, and the vendored kernel (kernels/ops/attention/cutedsl_gdn_mtp_ring.py) is 'BF16-STATE only', so --mamba-ssm-dtype float16 would hit an unsupported or slow Triton path. (3) The fallback float32 adds +3.2 GB (ssm_state is 3.22 GB at bf16, serve.log:60) out of a 12.5 GB KV pool whose p90 usage is already 86%, and it loses the bf16 CuTe MTP verify kernel. Enabling SR on GDN means writing a stochastic-rounding epilogue in a CuTe kernel, which is well over 3 days with no evidence that bf16 state limits score. I checked the HEAD clone, not the pinned d00d88e, but the boot-log fields and kernel dispatch match.",
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 4,
   "confidence": "medium",
   "first_step": "None. If revisited, first measure whether bf16 vs fp32 state changes accept length or output quality offline on one prompt before any kernel work."
  },
  {
   "id": "trim-time-reasoning-compaction",
   "keep": true,
   "reason": "Plausible and genuinely untested, and it is the user's own stated next step (GB10 notes: reasoning-summary 'Next: condense only at compaction time'). Absent: CONFIGURATION.md lists only ARC3_STUB_DEAD_REASONING (dead replies) and ARC3_PRESERVE_THINKING defaults on; the notebook env sets neither. I verified the premise on the control transcripts: reasoning averages 5,186 chars per turn vs code 657 and tool result 622, about 63% of non-image history text, so after a drain the retained 58k block only holds about 20-37 turns (about 3k tokens per turn); halving the per-turn weight about doubles the memory horizon. Support: ledger-on-trim 52.56 vs 46.25 shows what a trim discards matters. Risks, which cap the impact: (1) Franzen (WRITEUP.md:69) reports removing thinking gave a large regression; (2) the GB10 token-efficiency arm that dropped all stale reasoning generated about 30% more tokens per turn on sk48 (2,166 vs 1,621-1,729) from re-deriving, n=2; (3) many empty <think> blocks in history may teach the model to think less. Materially different from those failures: only older turns, only at drain (the prefix is rebuilt anyway), keep the last 6-8 turns whole, and keep a ~500-char tail of each older reasoning (where the conclusion usually sits) instead of deleting it. Context size looks weakly sensitive (streams14 at 96k and ledger-streams12 at 110k showed no change), so I rate this lower than the idea does.",
   "status_in_franzen": "absent",
   "impact": 2,
   "effort_days": 1,
   "confidence": "low",
   "first_step": "In inference/agent/tool_agent.py _trim_messages_for_context, at the first transition to draining (line ~9025, before _drop_oldest_history_block) add an env-gated ARC3_TRIM_REASONING_KEEP_TURNS=8 and ARC3_TRIM_REASONING_TAIL_CHARS=500 step that copies older assistant messages with reasoning_content cut to its tail, then continue the drain loop. Run a 25-game 121-min arm, twice (noise is about 3 points), alone and with ledger-on-trim."
  },
  {
   "id": "drain-time-image-stripping",
   "keep": true,
   "reason": "Valid mechanism, tiny and low-risk, and best done together with the reasoning-compaction hook. ARC3_HISTORY_IMAGE_KEEP is unset in Franzen's notebook env (present-but-off), and CONFIGURATION.md:96 warns that filtering historical images changes request prefixes. I traced _persistent_history_messages (tool_agent.py:8737-8741): it strips on every commit, so a stripped image N messages back invalidates the cache from that point onward each turn. A drain-only variant is new and has no cache cost, since the prefix is rebuilt at a drain anyway. Gain is small: images are about 12% of prompt tokens (control calibration 10,050 of 84,483; 402 tokens each at 10x), so the retained 58k block gains roughly 4-7k tokens of text, or +8-12% history depth. Given the weak context sensitivity above, expect no measurable score change; it is a cheap hedge, not a lever. A related config-only knob, ARC3_RESUME_PROMPT_IMAGES=0, would drop about 434 redundant board images per run (28% of the 1,560 user prompts are 'executed NO actions' resumes) for about 2% of tokens. Do not use a large K with the every-commit window, because it still invalidates the prefix each turn.",
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 0.25,
   "confidence": "medium",
   "first_step": "In the same drain hook in _trim_messages_for_context (first time draining flips True), call _apply_history_image_window(history, K=6) once, gated by a new env ARC3_DRAIN_IMAGE_KEEP; leave ARC3_HISTORY_IMAGE_KEEP unset so non-drain commits stay cache-stable. Compare tokens kept after drain and prefix-cache hit rate in serve.log."
  }
 ]
}
```
