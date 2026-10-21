# arc3-idea-sweep: verify:6/10

Workflow run `wf_6b16ae20-e2c`, phase Verify. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "verdicts": [
  {
   "id": "reset-exposure",
   "keep": true,
   "reason": "Confirmed present-but-off. Franzen's real competition env block (arc-agi-3-v3/franzen/kaggle/template.ipynb cell 4) sets EXPOSE_UNDO=on and ARC3_ACTION_INFO=1 but never sets EXPOSE_RESET (default off, action_names.py:39). arms.md and arms.json/fz-* logs contain no EXPOSE_RESET run, and WRITEUP section 6 is silent on it, so it is neither done nor reported as failed. Evidence holds: in the fz transcripts 16 of 75 sampled games say 'no reset available', and ka59 (control) reasons 'trigger a reset efficiently: just burn actions with no-ops? No-ops might not decrement the bar', which the batch no-op guard makes hard. Games with 'die on purpose to reset' reasoning (ls20, sc25, su15, wa30, tu93, dc22) recur across runs, 5-12 games per run (the idea's 6-11 is about right; the narrow 'burn...bar' regex gives 3-8). scorecard.py inc_reset_count adds an action, and UNDO (id 7) also goes through take_action, so the UNDO/RESET cost wording is factually right; the sk48 transcript shows the model chaining 32-37 counted UNDOs and timing out. Scoring is (baseline/actions)^2 (scorecard.py:168-171), so burned actions are punished quadratically. Two corrections. (1) The 'humans reset 12.5% vs <7%' stat is not in AutumnBench arXiv 2510.19788 (abstract and PDF checked), so drop it. (2) docs.arcprize.org/api-reference/commands/start-or-reset-game-instance.md says a RESET with no action since the last level transition resets the ENTIRE game. The local arc_agi competition_mode branch (api.py:325-334) turns that into a counted no-op, but the Kaggle gateway is a separate container, so the zero-action guard is a safety requirement, not only an efficiency tweak. The guard must track engine actions since the last RESET (auto-reset after death too) or level change, not 'actions on this level', because an auto-reset zeroes the engine's _action_count. Risk: the 'start a different approach' wording may cause over-resetting (each RESET costs 1 action and refunds nothing). Impact is 2: the dead-end fraction is real but it only recovers the remaining bar on already-struggling levels, and an effect below about 3 points cannot be measured in one offline run, so decide from a transcript check of RESET usage.",
   "status_in_franzen": "present-but-off",
   "impact": 2,
   "effort_days": 0.5,
   "confidence": "medium",
   "first_step": "In the Kaggle notebook setup_env add 'EXPOSE_RESET':'on' (ARC3_ACTION_INFO is already '1'). In ARC3-Inference/inference/agent/tool_agent.py, inside the `if reset_exposed():` block near line 8073, add a counter of engine actions since the last RESET or level start (reset it on any RESET, auto-reset after game over, and level change) and refuse RESET when it is 0 with a stop_reason like 'reset_at_start'. Then edit UNDO_INFO_ADDENDUM and RESET_INFO_ADDENDUM in agent/prompts.py (lines 156-168): UNDO counts as one action and does not refund the reversed action; once a level is unrecoverable, RESET once instead of draining the bar; RESET plus replaying a known-good prefix usually beats long UNDO chains."
  },
  {
   "id": "actor-modeling-subcall",
   "keep": false,
   "reason": "Absent in Franzen (grep for modeler/subcall/predict/backtest finds nothing), but the evidence does not transfer. I read the Tycho paper text (arXiv 2607.28287 PDF): the 88.49 RHAE actor-requested-delegation result is one matched run per policy on Claude Opus 4.8, then GPT-5.6 Sol and Opus 5 at 100 RHAE with ~90M-token, $250-750 budgets. It contains no small or open-weight model results and no 'capability floor' statement; that claim is not in the paper. The Qwen3.8-Flash-Next harness spends about 180k tokens per game. Franzen's REPL with persistent functions already lets the model write and test simulators in its own context, so the only novelty is context isolation. That costs 15-20k uncached prefill per call under a different prefix, which competes for the 60 Mamba cache slots at 10 streams. The user's related attempts (hypothesis-loop inconclusive, transition_log rejected, v9 'harness owns facts' cleared 2 levels) and backlog items 5-6 already cover this space. At 2-3 days it eats half of the remaining schedule with a real chance of net harm.",
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 2.5,
   "confidence": "high",
   "first_step": "Do not build. If revisited, first replay saved fz transcripts offline to measure how often the model's own in-context predict()/simulator code is wrong, which is the only gain this would deliver."
  },
  {
   "id": "prolong-memory-search",
   "keep": false,
   "reason": "Absent in Franzen, but weak transfer. PRO-LONG (arXiv 2607.20064) was tested only on GPT-5.5, Opus 4.6/4.8 and Fable 5 with 500-2,000-action budgets and near-million-token contexts; there is no small-model evidence. The mechanism needs the model to decide to search its log, and the user's own results show this model ignores optional model-initiated features: the scratchpad tag was written 0 times, thinking-guide 0/28 Predict lines, and the action-tools `act` tool was called 0 times. The one clear memory win (ledger-on-trim, +6.3 offline, 8 games won) is automatic and needs no model initiative, and porting it is already backlog item 1. Franzen's summaries and the user's reasoning-summary showed no gain or were inconclusive. A sub-variant (action note stored in transitions) has no consumer without search. 1 day for an unproven change that duplicates the proven design. Better use of the time is porting and repeating ledger-on-trim.",
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 1,
   "confidence": "medium",
   "first_step": "Do not build. Port ledger-on-trim (franzen/base 98cc560) first; only consider search_log as an add-on if the ledger repeat run confirms memory recall is the bottleneck."
  },
  {
   "id": "memo-dict",
   "keep": true,
   "reason": "Keep as a tiny, low-risk change, but low impact. Absent in Franzen: WRITEUP says ordinary variables are discarded and CONFIGURATION section 7 says retention is not a persistent kernel. NameErrors are real: 380 across the 7 fz runs (about 2.2% of ~17.4k calls). The model clearly tries to persist (open/pickle/store attempts), so adoption is plausible, unlike the optional-tool failures. Correction to the evidence: Kinetic v8 contains ARC3_X_MEMO code (kinetic-learning-by-playing-v8.txt:10205-10297, 10437-10439), but its notebook config only sets ARC3_X_CATALOG=1 (line 10600), so the memo was OFF and the 27.28 score says nothing about it. It is untested, not 'no gain'. Upper bound on gain: each NameError costs one round trip, and re-typing data is under 1% of tokens, so impact is 1. The memo is exchanged between sandbox and host only, so there is no per-turn token cost. Risks: JSON-safety and size caps, and shipping a prompt line the model may not use.",
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 0.25,
   "confidence": "medium",
   "first_step": "Port the ARC3_X_MEMO hunks from competitors/kaggle/kinetic-learning-by-playing-v8/kinetic-learning-by-playing-v8.txt (lines ~10205-10297 sandbox side, ~10437-10439 host side) into agent/python_tool_sandbox.py and agent/tool_agent.py next to _kept_functions. Make the memo game-scoped and capped at about 100KB, and add one sentence to PERSISTENT_FUNCTIONS_LINE in prompts.py (~355-365)."
  },
  {
   "id": "persistent-kernel",
   "keep": false,
   "reason": "The motivation is real but small. NameErrors are 2.2% of calls (380 across 7 runs), 'not retained' rejections are only about 60 across 7 runs, and timeouts are about 7 per run, so the saving is under about 2% of calls and under 1% of tokens. The risks are high: a long-lived child per game with 10-14 concurrent streams (memory, kill and restart), model variables shadowing provided globals (history, transitions, previous_frame), stale-state bugs, and breakage of the retained-function pipeline that Franzen's guards are built around. The evidence is thin: GB10 '0 python errors in 47 calls' is n=2 games and came from the sandbox-v2 REPL with delta(), not from persistence per se. CodeAct (arXiv 2402.01030) supports persistent interpreters in general but not this gain size. The memo-dict covers most of the benefit in under 2 hours. At 1 day plus debugging it is not worth the schedule.",
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 1,
   "confidence": "medium",
   "first_step": "Do not build. Ship memo-dict instead, and if variable-persistence errors still matter, add an auto-restore of small JSON-safe top-level variables inside the same memo mechanism."
  },
  {
   "id": "timeout-wipes-retained",
   "keep": true,
   "reason": "Verified real bug. run_sandboxed_python (python_tool_sandbox.py:1001-1038, 1107-1112) returns timeout, crash and invalid-response dicts with no 'keepable_functions' key, while only the 'final'/'error' messages include it. _record_retained_functions (tool_agent.py:6512-6570) is called unconditionally at line 8179 and does `kept = {}` from `sandbox_result.get('keepable_functions') or []`, so after a timeout self._kept_functions is replaced by {} and the model is told every function is 'no longer retained'. Transcript check across the 7 fz runs: 50 'Tool timed out after 30s' events (matches the idea's 7,7,2,6,11,8,7), and in 41 of them the retained-function listing was 8+ before the timeout and mostly 0-9 at the next listing (examples: cn04 39->7, tr87 88->37, m0r0 54->1, vc33 60->0). Sandbox crashes were 0, so only timeouts matter. Impact is 1: about 6 events per run, and each costs one redefinition round (a few k tokens), concentrated late in games like tr87, cn04 and m0r0. The fix is about 5 lines, has no downside (functions retained from earlier successful calls are still valid), and the key's absence uniquely identifies the failure paths.",
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 0.1,
   "confidence": "high",
   "first_step": "In agent/tool_agent.py _record_retained_functions (~line 6531), after the _persistent_functions() check, add `if 'keepable_functions' not in sandbox_result: return` so self._kept_functions is left untouched on timeout, crash or invalid response. Optionally add a note that retained functions were kept."
  },
  {
   "id": "soft-timeouts",
   "keep": false,
   "reason": "Low value and partly subsumed. The wipe half is fixed by timeout-wipes-retained. The rest touches only about 7 timeouts per run out of roughly 2.5k python calls (0.3%). Host-side executed actions are already surfaced after a timeout: the sandbox timeout return carries host_action_results, which feeds _last_step_summary ('The code executed N actions'). The sk48 transcript (control, line 8548) confirms the model ran a 32-UNDO loop that timed out, but that is a rare pattern. Raising 30s to 45-60s would only rescue the roughly 27 compute-only timeouts per 7 runs that a model would otherwise rewrite with pruning. A SIGALRM raising a BaseException in the child interrupts the pipe read mid-action exchange and risks host/child protocol desync, and a model `except BaseException` or bare except can swallow it. The 'import time blocked 47 times' claim is also not clearly reproduced in the transcripts (models do call time.time() in snippets). Not worth the half-day and the risk.",
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 0.5,
   "confidence": "medium",
   "first_step": "Do not build beyond timeout-wipes-retained. If revisited, only add a host-side timeout message that lists executed action count and a note that retained functions were kept."
  },
  {
   "id": "cutoff-signal-fix",
   "keep": true,
   "reason": "Core claim verified; one half of the idea is dead and should be dropped. Verified: control_yield_reason() (tool_agent.py:9288) yields once the turn's generated tokens reach YIELD_TOKENS=2048 (competition env LOCAL_ANALYZER_YIELD_TOKENS=2048, MAX_OUTPUT=12288), so a 12k-token truncated reply always yields before the follow-up nudge at 9569-9578. 'CUT OFF' appears 0 times in the control and streams14 transcripts although 'finish_reason: length' appears 25 and 20 times. `truncated_tool_call` (9410-9472) is only set when a malformed tool call exists, so the common reasoning-only truncation (bp35 transcript: tool_call_count 0, 37k reasoning chars, finish_reason length) never sets it. The next prompt then says 'Your last python snippet executed NO actions (it only computed or inspected state - that is fine)', which is false. Truncation matters: 184 truncated replies across 7 runs x 12,288 tokens is about 7% of all generated tokens (4.48M per run). Drop the memory-sections half: the follow-up nudge with 'include memory update lines' appeared 0 times in 3 of 4 runs checked and once in streams14, because the yield pre-empts it, and the world model is off, so gating it is worthless. WRITEUP section 6 says resume-prompt variants (shorter, tone, less reinsertion) gave no consistent gain, but this is a factual correction of a false sentence, not a tone variant. Impact is 1 because it is about 1% of replies and the model may keep rambling anyway.",
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 0.15,
   "confidence": "medium",
   "first_step": "In tool_agent.py near line 9411, set a per-turn flag whenever finish_reason=='length' (regardless of tool calls). Pass it into the resume-prompt builder (_stale_summary_lines, ~line 536) as stale_reason='cut_off', so the opener leads with: 'Your previous reply hit the 12k-token output limit and was CUT OFF before any tool call; keep reasoning short and emit the python call early.' Skip the memory-section follow-up change."
  }
 ]
}
```
