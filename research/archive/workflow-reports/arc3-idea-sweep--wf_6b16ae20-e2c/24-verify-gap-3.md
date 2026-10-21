# arc3-idea-sweep: verify-gap:3

Workflow run `wf_6b16ae20-e2c`, phase Verify. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "verdicts": [
  {
   "id": "gap-8",
   "keep": true,
   "reason": "Keep, narrowed to a quick transcript-based analysis (hours, not a polished tool). Refs verified: python_tool_sandbox.py:76-95 (difflib '32 transcripts, imported 9 times'), tool_agent.py:8296-8303 ('clamped from'), notebook cell 2 sets TAAF_MINIMAL_DIAGNOSTICS=1 only in a real rerun, so the miner can only run on offline runs. Two corrections to the idea. (1) The data already exists: /Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs/fz-{arms-v4-control,ledger-on-trim,arms-v5-temp06,arms-v6-audit60k,streams14,ledger-streams12,ucb}/transcripts/*_p0.txt plus benchmark.json, so there is no need to wait for a new run. (2) The cited inputs are not available there. requests.jsonl needs save_request_logs, which is false in configs/inference.json:66, and traces.py reads artifacts/*viewer_data.json, which the Kaggle outputs lack. The transcripts do carry [MODEL RESPONSE META] (finish_reason, tool_call_count, reasoning_chars), Traceback blocks, '-> NO-OP' action lines, and text_token_calibration/request_timeout status lines. benchmark.json gives per-action generated_tokens and wallclock. A first grep on the control run found: 25 finish_reason=length turns, about 0.27M of 4.48M tokens (about 6%), and 24/23/37 such turns in ledger-on-trim/temp06/ucb. Franzen's notebook cell 4 env dict leaves the matching ARC3_STUB_DEAD_REASONING and ARC3_REASONING_EFFORT_LADDER unset. It also found 202 Tracebacks in 2,275 tool calls (NameError 50, StaleStateActionError 38 which is the guard working, ImportError 17 including 'time' x7 and 'sys' x5, KeyError 27), 395 NO-OP actions, and 29-31% of tokens in each game's final uncleared level (levels.csv). Caveat that caps impact: the user's own tool-error-analysis.md (experiments/arc-agi-3/docs/analysis) found all 62 old-harness tool errors cost about 1.06 game budgets and the addressable ones about 0.46, 'under the noise floor'. So this prioritizes work rather than being a lever, and it cannot hurt score.",
   "status_in_franzen": "partial",
   "impact": 2,
   "effort_days": 0.5,
   "confidence": "medium",
   "first_step": "Write /Users/advaith/Documents/studying-stuff/arc-agi-3-v3/franzen/analysis/failure_taxonomy.py next to level_dataset.py. Split each kaggle-outputs/fz-*/transcripts/<game>_p0.txt on '--- analysis_step=' blocks. Regex finish_reason: length, tool_call_count: 0, ^Traceback plus the error class, 'NO-OP', 'request_timeout' and 'clamped from'. Join with benchmark.json history (generated_tokens, wallclock) and levels.csv. Print tokens and actions wasted by cause per run, starting with the length-truncation sink."
  },
  {
   "id": "gap-9",
   "keep": false,
   "reason": "Drop. Mostly already built, and the one new piece is statistically invalid. (a) Equal-token comparison already exists in TAAF: tufa-arc-agi-framework/src/taaf/diagnostics.py:2495-2560 (generate_comparison_html uses _min_tokens_per_game_budget, _global_ceiling_for_budget, _capped_score_fn), with _capped_pass_curve and a score-vs-generated-tokens curve (diagnostics.py:917-1026), paired t and permutation tests, and a tokens-vs-wallclock curve at :858. ActionRecord.generated_tokens (game.py:~225-240) and GameRun.final_generated_tokens already feed it. (b) The user's own arc-agi-3-v3/franzen/analysis/level_dataset.py + fit_levels.py already build per-level token costs with right-censoring and Kaplan-Meier curves. run-atlas findings \u00a71/\u00a78 already use tokens per cleared level (42.7k control vs 38.7k ledger) as the equal-compute metric. (c) The new proposal, a paired bootstrap over about 100-150 level attempts, is wrong. Level attempts inside a game are not independent (level k+1 exists only if k was cleared, and games differ in difficulty). The noise that matters is run-to-run trajectory randomness (findings \u00a77: 50-80 point per-game swings, near-identical runs spanning about 44-46 mean, gaps under 3 are noise). Re-slicing one run cannot shrink that, and a cluster bootstrap by game collapses back to n=25. Only repeat runs help, and those are the Kaggle GPU cost the idea hoped to avoid. A conditional-on-completion efficiency metric adds nothing, since score caps at completed weight share (findings \u00a72). Pure eval tooling, no direct score effect.",
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 0.5,
   "confidence": "high",
   "first_step": "No new code. Run generate_comparison_html on two fz-* benchmark.json files (e.g. fz-arms-v4-control vs fz-ledger-on-trim) to see the existing capped-token comparison. If a missing slice is wanted, add a 30-line cumulative-tokens-at-level-clear table to analysis/level_dataset.py."
  },
  {
   "id": "gap-10",
   "keep": false,
   "reason": "Drop. Verified absent (vision_context.py:431-487 diff_image_part renders the full board with no crop) and ARC3_DIFF_IMAGE=1, ARC3_GAMEOVER_DIFF_IMAGE=1 and MULTIMODAL_UPSCALE=10 are on in Franzen's own notebook (arc-agi-3-v3/external/dfranzen/arc-agi-3-milestone-2-solution.ipynb cell 4). The upside is smaller than claimed. Images cost 402 tokens each (transcript 'images 402'), and the control run attached 912 diff images over 2,299 model turns, so a crop saves about 300 tokens on about 40% of turns, under about 8% of retained context. The user's streams14 arm, which cut context 25% (128k to 96k, trim 58k to 42k), was neutral (46.38 vs 46.25), so a saving this size is worth about 0. Prefill saving is negligible because images are prefilled once and then prefix-cached. The perceptual gain is unprovable: a sub-1-point effect is below the +/-3 offline noise. The downside is plausible. A varying-size crop drops the absolute-position cue that the full-board diff gives for free, and the caption has to compensate for a weak spot in vision models. Related negative evidence: Franzen \u00a76 says composite animation images hurt some games, and the user's old-harness b3 (labeled grid + changed-cells diff image) scored 12.343 vs 17.743, though confounded. The idea's evidence 'WRITEUP \u00a74 credits diff images with a perception gain' is not supported. WRITEUP credits animations as strongest and 10x upscale as the biggest image gain, with no per-feature diff-image number. If context saving is the goal, the existing off-by-default ARC3_HISTORY_IMAGE_KEEP (CONFIGURATION.md section 4) is the cheaper knob, with the stated prefix-reuse cost.",
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 0.5,
   "confidence": "high",
   "first_step": "Do not build. If anything is tested, run the existing flag with ARC3_HISTORY_IMAGE_KEEP=<N> (and ARC3_RESUME_DIFF_IMAGE=1 if resumed turns need the diff) as a single Kaggle arm, rather than a new crop renderer in vision_context.py."
  },
  {
   "id": "gap-11",
   "keep": false,
   "reason": "Drop: 2-3 days of a roughly 5-day window for an eval accelerator that cannot resolve the effects the user is chasing. Feasibility is partly shaky. solver.py:813 seed_initial_history only appends one initial frame entry, and the cited lines 962-985 are play(), so a prefix-replay path through TAAF GameAPI/step_env would be wholly new code. Statistical power is the real killer. Per-game swings of 50-80 points between near-identical runs (findings \u00a77) mean a 6-10 game late-level panel resolves only very large effects. The best arm found so far (ledger-on-trim) was +6 across all 25 games, so a panel of 6-10 games could not see it. Validity is also off. The system prompt says levels build on earlier mechanics, and the model would start level k+1 with no retained functions or reasoning, measuring a 'cold start mid-game', not real late-level play. The motivation is partly real. Per levels.csv (869 attempts, 7 Franzen-era runs), levels 5-8 carry about 50% of earned weight (0.185+0.152+0.095+0.072), and game.py:381-413 weights level L by L. But the saving is at most about 2x: median tokens to clear levels 1-4 are about 17k+25k+28k+21k = about 91k of about 180k per game, and per-level cost is not higher late (median 15-30k at levels 5-8). Conditional clear rates are 0.69-0.86, so late levels are not disproportionately expensive. The same effect is cheaper by reading per-level outcomes from full runs (level_dataset.py already produces them).",
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 2.5,
   "confidence": "high",
   "first_step": "Do not build. For late-level signal, extend /Users/advaith/Documents/studying-stuff/arc-agi-3-v3/franzen/analysis/level_dataset.py to report per-level clear rate and tokens for levels >=5 across the existing fz-* runs."
  }
 ]
}
```
