# arc3-idea-sweep: verify:3/10

Workflow run `wf_6b16ae20-e2c`, phase Verify. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "verdicts": [
  {
   "id": "resume-turn-diet",
   "keep": false,
   "status_in_franzen": "tried-no-gain",
   "impact": 1,
   "effort_days": 0.1,
   "confidence": "high",
   "first_step": "Do not run. If you still want a fragment, change only LOCAL_ANALYZER_YIELD_TOKENS (2048 to 4096) and leave images and prompt untouched, but expect a result below the \u00b11.1 noise floor.",
   "reason": "Franzen already built and tested the knobs: ARC3_RESUME_PROMPT_IMAGES (tool_agent.py:1249-1260), ARC3_YIELD_RESUME_PROMPT short/state_only, ARC3_YIELD_RESUME_TONE. WRITEUP.md:299 says shorter prompts, different tones and removing some reinsertion gave no consistent gain. His own docstring (tool_agent.py:1087-1091, also in the nb.txt patch near line 2060) says 'short' held up while images were attached and degraded without them. That is exactly the combination proposed here (images=0 plus short), and state_only was his answer to it. I reproduced the stats from fz-arms-v4-control transcripts: 466 turn_token_budget yields of 1,560 turns, 434 'executed NO actions' openers averaging 2,540 chars with 1 image. The upside is about 20k duplicate tokens per game (about 8% of context growth), which only means slightly fewer trims. That is not measurable against noise: control-like runs span 44.1-46.3 (findings.md section 7). The yield also acts as a forcing function ('call action(actions) with your best next action'), so removing or softening it could raise tokens per level. Notebook config (nb.txt ~10190) leaves all three knobs at defaults, so this is untested by Franzen on the yield-threshold axis only."
  },
  {
   "id": "image-token-hygiene",
   "keep": false,
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 0.5,
   "confidence": "high",
   "first_step": "Do not build. Part (a) is already the existing flag ARC3_RESUME_PROMPT_IMAGES=0. If anything, test it alone under resume-turn-diet's caveats.",
   "reason": "(a) Skipping the duplicate board on resume is already implemented as ARC3_RESUME_PROMPT_IMAGES (tool_agent.py:1249-1260, resume_frame logic at 9218-9226), and Franzen's own docstring (1087-1091) reports degradation when the image is dropped from short resumes. Numbers check out (434 NO-ACTIONS openers, 912 diff images, 1,560 openers in control). (b) HUD-only diff skip is genuinely absent: tool_agent.py:5788-5796 attaches a diff image whenever grids differ at all, and only the animation-image path drops it (6573-6615). But 79 of 912 images is under 9%, about 32k tokens per 25-game run, under 1%. (c) ARC3_DIFF_IMAGE_UPSCALE does not exist: vision_context.py diff_image_part reuses current_grid_image_upscale(). At 4x each 32-px token spans 8x8 cells, so a 1-3 cell change shrinks to a speck. That risks the diff-image perception gain Franzen credits (WRITEUP.md:17,201). Total saving is about 3-5% of context, below detectability, with a perception downside."
  },
  {
   "id": "upscale-patch-aligned",
   "keep": true,
   "status_in_franzen": "partial",
   "impact": 2,
   "effort_days": 0.1,
   "confidence": "low",
   "first_step": "Set MULTIMODAL_UPSCALE=8 in the setup_env block of your notebook builder (arms.json arm 'up8', env only). ARC3_IMAGE_TOKEN_ESTIMATE already derives tokens from PNG size via ceil(w/32)*ceil(h/32)+2 (tool_agent.py:4225-4238), so no code change is needed. Run it at least twice and only after the higher-value memory arms. Drop the 16x arm.",
   "reason": "The geometry checks out. HF config.json and preprocessor_config.json for Qwen/Qwen3.8-Flash-Next give patch_size 16 and merge 2, which means 32 px per token. 10x gives 402 tokens, 8x gives 258 and 16x gives 1,026. But the premise is weaker than claimed. 64*u is a multiple of 32 for every integer u, so 10x (640 px) is not smart_resize-blurred either, and 'sizes must be multiples of 32' discriminates nothing. Cell-to-patch alignment holds only for u dividing 16, and the Duck default 4x is perfectly aligned (4 cells per patch) yet Franzen found 4x to 10x the biggest image gain (WRITEUP.md:203). So px-per-cell, not alignment, is what mattered. Supporting data is thin: Wang c1 8x scored 19.43 vs 17.74 at 4x (versions.md c1 row), +1.7, within noise, and it was never compared with 10x. I could not find discussions 717133 or 739801 saying anything about patch alignment. 16x would cost 2.55x image tokens: with about 99 images per game, roughly 100k tokens against 40k now. Your c2 (12x plus ASCII) scored 5.6 with 117k tokens per level, so drop 16x. The one safe benefit of 8x is -144 tokens per image, about 4-5% of context. Impact is 2, not medium, because the effect is probably below the \u00b11.1 offline noise."
  },
  {
   "id": "temp-1-0",
   "keep": false,
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 0.05,
   "confidence": "high",
   "first_step": "Do not spend a Kaggle run. Leave LOCAL_ANALYZER_TEMPERATURE at 0.7 (nb.txt notebook config).",
   "reason": "The HF Qwen3.8-Flash-Next card does say thinking mode: temperature=1.0, top_p=0.95, top_k=20 (fetched, quoted). But it is a generic recommendation with no evidence for this harness. Your temp06 arm scored 45.92 vs 46.25 (versions.md), and serve-log numbers show the response is flat: harness tok/s 606 vs 611 and accept len 2.72 vs 2.69. Franzen chose 0.7 and the Duck-era notebooks use 0.6, so no known data point favours 1.0. The openai_compat.py:155-159 top_k claim is true: top_k is sent for provider vllm, which the notebook sets. Going to 1.0 lowers draft acceptance (greedy shows 2.81, so each step up costs accept len) for an unmeasurable effect. A single 2-hour run cannot detect a gain under about 3 points. It also risks longer, noisier thinking. The 'diversity on stalled levels' mechanism is speculative; none of your 25-game transcripts show the repetition failures it would fix (0 degenerate_content_stubbed and 0 'CUT OFF' in control and temp06)."
  },
  {
   "id": "no-model-swap",
   "keep": false,
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 0,
   "confidence": "high",
   "first_step": "None. Keep Qwen3.8-Flash-Next and the existing serving bundle.",
   "reason": "This is a resource-allocation note, not a change that raises score. The conclusion is right but the cited evidence is weak. WRITEUP.md:15 supports only that each Qwen switch helped, which is a reason a better model would matter, not proof none exists. rellik13's WRITEUP.md says Flash-Next is 'the strongest model that fits on one RTX PRO 6000'. My web search for newer 96GB-fit open models returned noisy aggregator pages (one claimed GLM-5.2 runs 'on a single 24GB GPU'), so I could not verify the kingy.ai or HF-table claims. A swap is also over 3 days and needs a new SGLang sm120 port, as serving/README.md patches show. Keep the content out of the idea list and treat it as a planning constraint."
  },
  {
   "id": "solved-level-memory",
   "keep": true,
   "status_in_franzen": "partial",
   "impact": 3,
   "effort_days": 1.5,
   "confidence": "medium",
   "first_step": "First do backlog #1: cherry-pick 98cc560 (ARC3_LEDGER_ON_TRIM) from /Users/advaith/Documents/studying-stuff/arc-agi-3-v3/franzen (branch franzen/base) onto an arm/solved-level-memory branch off main in daniel-franzen-sol. Then extend MemoryLedger.on_level_change() in inference/agent/memory_ledger.py to store {level, n_actions, actions[-30:], last ~900 chars of the clearing turn's content/reasoning_content} from tool_agent history, and render it inside the existing <memory> block. Add a 'Rule for level N:' ask (max 2 turns) to LEVEL_START_USER_PROMPT and capture it with the same tolerant regex as absorb(). Keep the block at or below 1.5k tokens, and do not put it in the system prompt.",
   "reason": "Franzen's repo has no ledger, pinned memory or summary beyond the failed ARC3_SUMMARY_* (WRITEUP.md:297). But your own franzen/base ledger-on-trim already carries <level_note>, world_model and hypotheses, plus a summarize_think() fallback folded in at on_level_change. That is why it is 'partial': this is an extension of backlog #1, not a separate new idea. The new part is the harness-captured winning tail, closing reasoning and explicit 'Rule for level N:' paragraph. Verified evidence: (1) the M85 code in sirikilohit nb.txt:3902-4103 matches the description. It pins to the SYSTEM prompt, which would force a full re-prefill on every level-up in Franzen's 128k setup, so use the trim-head slot as proposed. (2) rellik13 WRITEUP.md:31-32 and 69-71 report 7.14 to 9.96 (+2.8) from solved-level memory, but at 37-49k context with the world model wiped each level and one run per point. Franzen's 128k context and off world model should shrink that gain. (3) ledger-on-trim 52.56 vs control 46.25, 8 vs 7 games won (versions.md); one run, so the 3-point noise caveat applies. (4) I recomputed per-level generated tokens from benchmark.json: control L1 median 15.3k, later levels 26.4k, ledger later levels 18.8k, but temp06 25.3k and audit60k 22.5k, and this counts cleared levels only. These support carrying knowledge, but one run is not proof. Citation problems: Tycho arXiv 2607.28287's abstract says nothing about cross-level memory, and OpenAI's 13.3% to 38.3% is retained reasoning plus compaction together. Franzen already retains reasoning (ARC3_REASONING_HISTORY_KEY=reasoning_content), so that does not support this idea. Risk: rellik13 WRITEUP.md:57-62 saw a level-start advice block drop 13.40 to 9.98 under a small cap. Stay inside the head-of-history slot and be wary of text volume. Impact 3 rather than high, because it is an increment over a ledger that is not yet ported."
  },
  {
   "id": "level-debrief",
   "keep": false,
   "status_in_franzen": "partial",
   "impact": 2,
   "effort_days": 0.5,
   "confidence": "medium",
   "first_step": "Fold the run-length-encoded action tail into solved-level-memory's ledger record instead of building this separately. Optional micro-test: ARC3_LEVEL_INVENTORY=1, which already exists (tool_agent.py:1743, 6073-6146) and costs about 100-300 tokens only at level-up.",
   "reason": "The 15.3k vs 26.4k median claim reproduces exactly from fz-arms-v4-control/benchmark.json, with ledger later levels at 18.8k, so the data is real. But a block at the END of context (the level-up opener) mostly duplicates what is still in retained history. At level-up the clearing turns are inside the 58-116k window unless a trim just happened. The win condition, last actions and reasoning are already visible, and the opener already carries LEVEL_START_USER_PROMPT with ARC3_LEVEL_TRANSFER_GUIDANCE=1 on (nb.txt:~10215). The value appears only after a trim evicts them, which requires a pinned or head-of-history placement. That is solved-level-memory's design, so this idea is subsumed. The winning-board image adds 402 tokens per level for a board already imaged at the clearing opener. rellik13 reports that a level-start advice block hurt (13.40 to 9.98, WRITEUP.md:57-62), so per-level text on the opener is the risky placement. Residual: ARC3_LEVEL_INVENTORY=1 (new-object-type diff at level start) is a built, off-by-default flag with no recorded result in WRITEUP or arms.md, so it is an optional free test, not a debrief."
  },
  {
   "id": "stall-audit-prompt",
   "keep": false,
   "status_in_franzen": "tried-no-gain",
   "impact": 1,
   "effort_days": 0.25,
   "confidence": "high",
   "first_step": "Do not rebuild. The code already exists on franzen/base (ARC3_STRATEGY_AUDIT_TOKENS, tool_agent.py:254-275, STRATEGY_AUDIT_PROMPT at prompts.py:72). If you retest anything, only the early-trigger variant: ARC3_STRATEGY_AUDIT_TOKENS=40000 combined with an action-count gate.",
   "reason": "The idea's key claim is false. It says audit60k showed no gain and that 'gating on stagnation is the difference'. But audit60k WAS stagnation-gated: _strategy_audit_due counts generated tokens from the later of level start and the last audit, only on an uncleared level, and resets on level-up. It used lordhansolo's STRATEGY_AUDIT_PROMPT (the same text proposed here) at 60k vs the proposed 62k (ARC3_PRIORITY_TOKEN_SCALE). Result: 44.10 vs control 46.25, fired 30 times (arms.md line 85, versions.md audit row, findings.md section 5). That is already inside the \u00b13 noise band with a slightly negative sign. What remains different is a ~40k-token or 115-action trigger and Rodionov's trouble text. I fetched and parsed arXiv 2607.15439 (Rodionov). Section 8.1 says the trouble prompt fires at GAME OVER resets when more than 50 additional actions have accumulated on the level, on a GPT-5.5 xhigh Codex agent. The 74.78 to 82.01 RHAE jump is explicitly 'not a controlled estimate' because v1.5 also accelerated the game client. The text asks the agent to analyze prior attempts and form a new hypothesis, and Franzen already issues a long game-over diagnosis and revise-plan prompt (WRITEUP.md:230-233). The 83% 'decisive action already tried' figure I could not verify. A gain over 3 points from a copy of an arm that fired 30 times with no effect is unlikely, and measuring it needs a run you cannot afford."
  }
 ]
}
```
