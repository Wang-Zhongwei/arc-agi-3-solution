# arc3-idea-sweep: verify:9/10

Workflow run `wf_6b16ae20-e2c`, phase Verify. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "verdicts": [
  {
   "id": "scheduler-level-count",
   "keep": false,
   "reason": "The cited evidence undercuts the claim. fz-arms-v4-control benchmark.json shows all 25 public games have number_of_levels in 6..10 (6:9, 7:5, 8:6, 9:4, 10:1; sum 183, which matches the public level count found on the web). So the [6,10] clip in priority_scheduler.py parse_total_levels never binds. tufa-arc-agi-framework game_api.py:231 sets number_of_levels from initial.win_levels, so the count is available in submission mode. For a hypothetical 5-level game the error is B=5 instead of 0 on one level, a bounded shift in that level's priority. Expected score effect is about 0, and a rewrite risks a KeyError on TAIL_LOOKUP_REMAINING[total_levels]. Only do it if idle (about 15 min).",
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 0.1,
   "confidence": "high",
   "first_step": "If done anyway: in inference/agent/priority_scheduler.py priority_value, compute the tail directly from the raw N (N-level >=3 -> 8, 2 -> 7, 1 -> 5, 0 -> 0) and keep parse_total_levels clipping only for the 80k table lookup."
  },
  {
   "id": "kv-budget-admission",
   "keep": false,
   "reason": "The premise is wrong. serve.log in the user's runs shows the KV pool is 1,011,264 tokens at MAMBA_CACHE=60, 961,344 at 72, and 911,424 at 84 (about 4.2k KV tokens per Mamba slot). So 'MAMBA_CACHE=84 with a 1.01M pool' is really a 10% smaller pool for the whole run. Decode-batch logs for control show mean 9.0 running requests. KV usage is 0.44-0.54 in the first two deciles, then about 0.72 mean with 0.92-0.99 peaks. The second wave of games starts at 889-1124 s and the third at 1864-2003 s, so real headroom exists mainly early (the +4-5% token estimate is plausible; score-vs-time slope near the end is about 0.27 offline pts per 1% of time). But the core lever, more streams, was already run twice. streams14 gave +14.6% tokens (5.14M vs 4.48M) and scored 46.38 vs 46.25. ledger-streams12 gave +9% tokens and scored 48.08 vs 52.56 for ledger-on-trim alone. Tokens did not convert to score. In steady state a 12-stream gate would reproduce ledger-streams12, because 12 x 87k mean context (about 1.04M) exceeds the 961k pool. The best case is about +0.6 LB, below the +/-3-5 offline noise. The PyTorch SGLang hybrid blog does list HiCache integration as future work, and it confirms one Mamba slot per MTP draft token (this matches the 4 slots per running request seen in the logs).",
   "status_in_franzen": "tried-no-gain",
   "impact": 2,
   "effort_days": 1,
   "confidence": "medium",
   "first_step": "Do not build the gate. If one slot is spare, run an env-only arm: ARC3_MAX_ACTIVE_STREAMS=11, MAXREQ=11, MAMBA_CACHE=66, window unchanged at 128Ki."
  },
  {
   "id": "endgame-bank-it",
   "keep": true,
   "reason": "Not refuted but small, and the idea's scope should be trimmed. Verified: the scoring formula is in game.py _compute_final_score (min(115,(base/actions)^2*100), weights l/W, so 3x baseline banks 0.11 of the level weight). Control unsolved final-level tokens are 28.1% (1.26M of 4.48M). But only 8.1% of tokens are spent beyond 60k tokens on unsolved levels, so most of that 28% is ordinary work in progress at the deadline. Tail fade (ARC3_PRIORITY_TAIL_FADE=1, FRACTION 0.2, in the competition env) already shifts scheduling toward near-complete levels. From benchmark.json, the last 10% of the run adds +2.7 (control) and +3.7 (ledger-on-trim) offline points, so a 10-20% productivity change there is about +0.3-0.6 offline (+0.2-0.3 LB). The 'time_remaining_seconds unused in prompts' claim is half right. It is in the step payload (solver.py:859, tool_agent.py:7884) but the model only sees it if it prints last_action_call_result. The ar25 transcript:267 reference is just a board row; the real hit is line 20568. Drop the sub-ideas of disabling BATCH_NOOP/STALE_STATE guards (no evidence they hurt; they stop wasted actions) and lowering YIELD_TOKENS (read at startup). Put the line in the per-turn user message, not the system prompt, to avoid a cache flush. Keep the wording about time and commitment, not scoring math. The old spend-actions prompt c5cfbb0 was negative.",
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 0.3,
   "confidence": "low",
   "first_step": "In ToolAgent._build_user_prompt (tool_agent.py ~L6763, near the closing-guidance block ~L7137), append one line when remaining/(elapsed+remaining) < 0.12, using gate._clock_start/_clock_deadline or self._game_elapsed_seconds plus max_runtime_s_per_game. Gate it behind a new env flag."
  },
  {
   "id": "action-economy-prompt",
   "keep": false,
   "reason": "This duplicates the user's own failed arms. wang-duck-harness commit c5cfbb0 ('spend actions to finish levels') used the same argument (29/49 cleared levels at the 115 cap, about 0.72x human actions, scoring rule including level weighting, replacing the 'as few actions as possible' line). It scored 10.921 vs 13.645 (trimmed 8.43 vs 6.344, mixed) and was reverted. The b4 scoring-guide arm scored 11.12 vs 17.743. A per-level policy plus an actions-so-far counter is not materially different. My rescoring of the control reproduces the idea's numbers: 84% of 105 cleared levels have ratio >=1 with median 1.57 (mean cleared-level score 103.9); level-1 +4/+8/+15 actions gives 46.12/46.02/45.88; +20% actions everywhere gives 44.59. That proves action headroom exists, not that a prompt converts it into saved tokens. The model also tends to ignore guidance (0/28 Predict lines, 0 scratchpad tags). prompts.py:122 and :313-317 are live in the competition config (MEMORY_SECTIONS=off). The added prompt text and per-turn counter cost tokens and could hurt.",
   "status_in_franzen": "tried-no-gain",
   "impact": 1,
   "effort_days": 0.1,
   "confidence": "medium",
   "first_step": "Do not pursue. The only untested fragment is deleting prompts.py:122 and nothing else, which cannot be measured at the current noise level."
  },
  {
   "id": "decisiveness-flags",
   "keep": true,
   "reason": "Not refuted. Verified absent from the competition env (template notebook setup_env has neither flag; defaults are False at tool_agent.py:2533 and 2601). WRITEUP section 6 covers only resume-tone variants and the step-verification hint, which are adjacent but not these flags. The user has not tried them. Evidence checks pass on the control transcripts: 2,299 replies, top 10% hold 41% of reasoning chars, 11 of 2,299 have visible content. The commit-hint docstring (tool_agent.py:2515-2533) cites post-resume batches of 6.73/6.91 actions vs 3.91/5.43 after fresh openers. Weaknesses: prompt-guidance arms on this model have mostly been ignored or null (thinking guide 0/28 Predict lines, token-efficiency no speed-up, reasoning_effort no effect). The system prompt already says 'Do not ration tool calls', and the 'commit' resume tone is already the default. Larger batches also mean more blind actions (+10% actions everywhere costs -0.7 offline, from rescoring). Expected effect is about 0 to slightly positive, at zero engineering cost. Bundle it with another env-only arm instead of spending a Kaggle slot on it alone.",
   "status_in_franzen": "present-but-off",
   "impact": 1,
   "effort_days": 0.1,
   "confidence": "low",
   "first_step": "Add 'ARC3_PREFER_TOOL_CALLS': '1' and 'ARC3_OPENER_COMMIT_HINT': '1' to setup_env in the notebook (or an arms.json env entry). Judge by tokens per action and levels cleared, not mean score."
  },
  {
   "id": "auto-probe-sweep",
   "keep": false,
   "reason": "Premises verified but the value is overstated. Median cumulative generated tokens before action 1/5/10 are 1.7k/6.8k/12.0k (benchmark.json history), and the +4/+8/+15 rescoring reproduces. step_env (solver.py:1463) can be called from _play_inner like the warmup RESET. But the model's first inspection turn (about 1.7k tokens) stays regardless. Only the roughly 5k tokens of probe turns 2-5 on about 16-18 directional games would be saved, about 1.1-1.3% of total tokens, worth about +0.3 offline. The extra level-1 actions cost about -0.15 to -0.25. Net is about +0.1-0.2 offline (+0.1 LB), not 'medium'. Long pre-action stalls (bp35 12.3k before action 2, g50t 14.7k before action 4) come from analysis, which a probe table does not remove. Implementation gotchas: BATCH_NOOP_BLOCK is on from level 1 in the competition env, so a probe batch would stop at the first no-op. Each probe must be its own step_env call. SPACE/ACTION5 may be irreversible, and undo doubles the action cost. tool_agent's action_num, _last_step_summary and actions_at_level_start won't see harness-run actions, so the action_num==0 opener branch and priority snapshots need care. About 7-8 of 25 public games are MOUSE-only. Cost is about 1 day for roughly zero net gain, with an irreversible-state risk. Not worth it.",
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 1,
   "confidence": "low",
   "first_step": "If revived: in solver.py _play_inner after the warmup block, loop over non-MOUSE, non-ACTION5 simple actions on level 1 with separate self.step_env({'action': a}) calls behind ARC3_AUTO_PROBE. Abort on game_over or level_completed, and inject the compact frame_diff table via the action_num==0 opener in tool_agent._build_user_prompt."
  },
  {
   "id": "fresh-eyes-reset",
   "keep": false,
   "reason": "The control data refutes the trigger. In fz-arms-v4-control benchmark.json only 8.1% of generated tokens (365k, 11 levels) are spent beyond 60k tokens on final-unsolved levels. Meanwhile 14 levels were cleared after more than 60k tokens (255k tokens beyond 60k). So 14 of the 25 levels that run past 60k still clear. A deep drain to about 25k at that point discards context exactly where persistence pays (WRITEUP section 2: larger context helps hard levels). The '28% of tokens on unsolved levels' figure is verified at 28.1%, but it is mostly in-progress work at the deadline, not stalls. The prompt half is the user's audit60k arm (44.10 vs 46.25, fired 30 times, no gain). ledger-on-trim's +6.3 is a single run of model-written <world_model>/<hypothesis> tags (memory_ledger.py in the user's fork), not harness-computed attempt logs, and it has not been ported to arc-agi-v4 yet. The unproven new element, the stall-gated deep drain, is the part most likely to hurt. If the harness-computed attempt/death log is wanted, test it as its own arm after the ledger-on-trim port, at every trim and not stall-gated.",
   "status_in_franzen": "tried-no-gain",
   "impact": 1,
   "effort_days": 1,
   "confidence": "medium",
   "first_step": "Do not build. Port ledger-on-trim (arms.md backlog #1) first."
  },
  {
   "id": "level-goal-function",
   "keep": false,
   "reason": "The evidence does not transfer. Tycho (arXiv 2607.28287, Lehmann/Aioanei/Vahdati, 30 Jul 2026) is real. It uses only frontier models (Claude Opus 4.8, GPT-5.6 Sol, Opus 5) and builds full executable transition models with an outcome() classifier, not a level_goal(frame)->float. Its auto-repair policy scored lower (83.07 vs 88.49 RHAE). A companion study (arXiv 2607.15439, GPT-5.4/5.5) finds requiring executable code 'is not universally beneficial'. Neither evaluates small open-weight models. The same family was already tried by the user: hypothesis-loop (bec652f) had the model commit a falsifiable claim via a tool parameter and the harness grade it with a CONFIRMED/WRONG verdict line. It had 93% compliance and was inconclusive. Franzen's finding that turning off structured world-model scaffolding was his biggest win in that area points the same way. Retained functions are already on (game scope; ls20 lists 20 helper functions), so the model can already write goal checks. ls20's long reasoning (replies of 15-21k chars, verified) is about what the goal is, which cannot be coded before it is understood. The harness already reports level_completed as the falsifier. Extra per-level function writing costs tokens, which are the binding constraint, and false 'goal wrong' alarms are possible. P(success) is about 10-15%.",
   "status_in_franzen": "partial",
   "impact": 2,
   "effort_days": 1.5,
   "confidence": "low",
   "first_step": "Do not build now. If revisited, prototype prompt-only: add the rule to COMPACT_TOOL_SESSION_ADDENDUM and log how often level_goal is defined and ever reaches 1.0 before level_completed on saved transcripts, before writing any harness-side evaluation."
  }
 ]
}
```
