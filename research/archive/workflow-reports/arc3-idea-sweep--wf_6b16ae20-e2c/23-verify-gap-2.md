# arc3-idea-sweep: verify-gap:2

Workflow run `wf_6b16ae20-e2c`, phase Verify. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "verdicts": [
  {
   "id": "gap-4",
   "keep": false,
   "status_in_franzen": "tried-no-gain",
   "impact": 1,
   "effort_days": 0.5,
   "confidence": "high",
   "first_step": "Nothing to build. If someone insists on closing it out, a single control-vs-SPEC_STEPS=2 Kaggle arm is a one-line CFG change in /Users/advaith/Documents/studying-stuff/arc-agi-3-v3/franzen/kaggle/template.ipynb cell 12 (SPEC_STEPS=3 is already a CFG knob, draft tokens = steps+1, topk 1), but expect <=0.",
   "reason": "Refuted by the user's own Kaggle run and by two other sources. (1) The adaptive half was already run on the exact Franzen/Pennyroyal build as arm fz-spec-adaptive. serve.log:59 shows 'AdaptiveSpeculativeParams initialized: steps=3, candidate_steps=[1, 2, 3]', and lines 141-231 show it flapping 3->2->3 several times per second. Parsing the decode lines (fz-spec-adaptive/serve.log vs fz-arms-v4-control/serve.log) gives accept length 2.23-2.25 at batch 8-10 against 2.66-2.71 for static 3 steps. Mean decode tok/s at batch 10 was 662 vs 711, about -7%. The -36% in arms.md (377 vs 587) is a job-wallclock figure on a 20-minute panel and overstates the steady-state gap, but the direction is clearly negative. SGLang's docs (https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding, fetched) confirm the cited limits: EAGLE/EAGLE3 and topk 1 only, defaults [1,3,7] for BS 1-7 and [1,3] for BS 8-31, no mention of hybrid/Mamba models. (2) Static depth is already pinned. 4 steps is impossible because Qwen QSA caps speculation at 4 draft tokens (reports/run-atlas/findings.md section 6). The GB10 scan (GB10-OVERNIGHT-RESULTS.md:54) at 8 real requests gave 0:34.8, 2:39.2, 3:42.0, 4:41.8 tok/s, so 3 is the optimum. That leaves only 'steps 2 vs 3', where 3 won by 7% on GB10. At batch 10 with 10 routed experts of 512, nearly every expert is touched at either depth, so a longer draft is the cheaper one. (3) Eagle topk 2 tree drafts are not a documented path for the GDN/Mamba MTP: SGLang's own Qwen3-Next recipe uses num-steps 3, topk 1, draft-tokens 4. I could not confirm a hard block, so that part is unverified, but there is no evidence it helps. Franzen's notebook has relaxed acceptance disabled (thresholds 1.0). Residual upside is at most a couple of percent of tok/s. The user's streams14 arm showed +15% tok/s moved the score by +0.1, so it would be invisible."
  },
  {
   "id": "gap-5",
   "keep": false,
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 0.1,
   "confidence": "medium",
   "first_step": "Only if wanted as hygiene: add os.nice(5) at the top of _set_limits() inside _SANDBOX_BOOTSTRAP (python_tool_sandbox.py ~line 320), in the child, not via Popen preexec_fn. Also log os.cpu_count() once in the notebook. Do not expect a score change.",
   "reason": "The premise (sandbox children starving SGLang) has no support in the data, and the suggested mechanism is risky. Real bug check: python_tool_sandbox.py:946-957 does spawn 'python -I -S' with no nice level, and RLIMIT_CPU is timeout+1 at :320-333, so the claim about the code is accurate. But the user's saved Franzen-control transcripts (kaggle-outputs/fz-arms-v4-control/transcripts, 25 games, 121 min, about 1,560 analysis steps and 2,299 model replies) contain only 7 'Tool timed out after 30s' events. That is at most about 210 CPU-seconds, about 3% of one core over 7,260 s. Typical tool calls are sub-second (WRITEUP section 2). With 10 streams each generating about 2-3k tokens per turn at 60-70 tok/s per stream, average live sandbox children is far below 1. A saturated-CPU scenario would need a heavy BFS in several streams simultaneously, and the data shows that is rare. I could not find the Kaggle VM core count: the logs do not record it, and a web search returned only cloud-vendor configs. The recommended implementation, subprocess preexec_fn, is documented by CPython as unsafe in multithreaded programs (deadlock risk), and this harness runs one worker thread per game. A semaphore capping concurrency would add queueing latency to tool calls for no measured benefit. Net: nothing to gain unless a replay shows decode tok/s dipping when children are live, which no existing log suggests."
  },
  {
   "id": "gap-6",
   "keep": false,
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 0.1,
   "confidence": "high",
   "first_step": "None needed. The code gotcha is real (tool_agent.py:9681-9690 sets preserve_history=False when an action executes before the last tool call, and :9774-9779 restores previous_history_messages). If wanted as a 15-minute robustness fix, trim assistant_message['tool_calls'] to tool_calls[:tool_index+1] instead of rolling back.",
   "reason": "The code reading is correct, but the frequency question the idea left open is answered by the user's saved transcripts: the trigger almost never happens. Each '[MODEL RESPONSE META]' block records tool_call_count. Across 10 saved Kaggle runs (about 20k replies), multi-call replies are 1 of 2,299 in fz-arms-v4-control, 1 of ~2,350 (temp06), 4 of ~2,230 (audit60k), 5 of ~2,280 (ucb), 2 of ~2,615 (ledger-streams12), 1 of ~2,717 (streams14), and 0 in ledger-on-trim, tput-all-v7 and b1. That is 0-0.2% of replies, and the rollback applies only if an action ran in a non-last call, so fewer still. The prompt (prompts.py:371-376) says 'as many times as you want per step', but the model in practice emits one tool call per reply and iterates across replies, which the loop handles correctly. Losing about 1-4 turns per 2,300 is not detectable against the +/-3 offline noise floor. It is harmless to fix, but it would not move score, so it should not take a slot."
  },
  {
   "id": "gap-7",
   "keep": false,
   "status_in_franzen": "absent",
   "impact": 2,
   "effort_days": 1.5,
   "confidence": "medium",
   "first_step": "Do not build as proposed. If the user still wants a prompt lever, derive 3-5 one-line additions from failure modes seen in their own transcripts (like Franzen's game-over and level-transfer prompts), not a genre primer, and ship them without an A/B.",
   "reason": "Likely wasted effort and unmeasurable, with a poor track record for this class of change. (1) Not testable: the primer would be distilled from the same 25 public games the A/B panel uses, so any offline gain is contaminated. Offline run noise is +/-1.1 on the mean (gaps under about 3 are noise per findings.md section 7), per-game scores swing 50-80 points between identical runs, and two parallel Kaggle sessions allow only a handful of arms before the deadline. The Milestone 1 winners reported that 'local public-game checks weren't a reliable leaderboard proxy' (https://arcprize.org/blog/arc-prize-2026-milestone-1, fetched), and Tufa Labs said hand-crafted additions hurt and a lightweight generic harness worked better. (2) Prompt-only arms have a bad record. Franzen WRITEUP section 6: stronger step-verification hint and resume-prompt variants gave no gain. The user's own arms.md: scoring guide/math policy b4 11.12 vs 17.74, thinking guide 0/28 uptake, spend-actions prompt 10.92 vs 13.65, prompt dedup 5.25 (worst run), strategy-audit 44.10 vs 46.25, 'harness owns facts' v9 cleared 2 levels in total. Franzen's prompt wins were all targeted at an observed failure (game-over confusion, repeated openers, level transfer), not general priors. (3) The mechanism is weaker than stated. Scoring (tufa-arc-agi-framework/src/taaf/game.py:375-415) weights level k by k, so level 1 is only 1/45 of a 9-level game, and the claimed saving is concentrated on the lowest-weight level. The existing VISUAL_GAME_ADDENDUM (prompts.py:116-142) already contains scene, HUD-bar, no-avatar and level-transfer priors, so the marginal content is smaller than implied. (4) It can hurt: a hedged genre list can anchor the model on the wrong goal type in a novel game, and the cost shows up as extra tokens per level, which findings.md section 1 says have never been beaten below about 39k. I checked the code references (prompts.py, kaggle.py game ids, solver.py 'probes hardest' comment at line 204, not 168-176 as cited; game.py scoring), and they are real. There is no evidence of a gain."
  }
 ]
}
```
