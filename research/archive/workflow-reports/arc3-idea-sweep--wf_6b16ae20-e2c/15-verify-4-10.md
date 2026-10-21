# arc3-idea-sweep: verify:4/10

Workflow run `wf_6b16ae20-e2c`, phase Verify. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "verdicts": [
  {
   "id": "sandbox-perception-helpers",
   "keep": true,
   "status_in_franzen": "partial",
   "impact": 2,
   "effort_days": 0.5,
   "confidence": "low",
   "first_step": "inference/utils/segmentation.py ~L327-338: add bbox/center/h/w/area to each node dict plus a node_cells(id) accessor (not an inline mask, to avoid bloating printed nodes). inference/agent/python_tool_sandbox.py L195-215: add FrameView.rows and show(r0,r1,c0,c1) with a column ruler. Add 'time' to SAFE_MODULES. Update STRUCTURED_RUNTIME_STATE_ADDENDUM in prompts.py L184-193 by one line. Skip the KeyError-hint dict subclass.",
   "reason": "Verified in code: nodes at segmentation.py:327-338 have no bbox/center/cells (only boundary corners), and FrameView (sandbox:195-215) exposes only ascii/step/level/shape/segmentation. I parsed the user's fz-arms-v4-control transcripts (2275 python calls): .ascii.split appears in 879 calls (matches the idea), 'boundary' in 391, 'misread' 113 times in thinking. But the pain is weaker in Franzen's harness than the idea claims. Only ~8/2275 calls fail on a missing perception API (FrameView not subscriptable x3, HistoryEntryView has no .step x2, KeyError 'color' x2). The '198 recoveries' figure is Siriki's Wang sandbox. Retained functions (ARC3_PERSISTENT_FUNCTIONS_SCOPE=game is on) already amortize hand-written helpers: bbox shows up in the retained-function list 246 times, crop 249. The 17 ImportErrors in the control are time(7)/sys(5)/pickle(3)/inspect(2), a cheap fix. User evidence: GB10 sandbox-v2 had 0 errors in 47 calls and used helpers in 34, but only 1 run on 2 games (GB10-OVERNIGHT-RESULTS.md). The user's backlog items #3 and #6 are the same family. Keep a trimmed version; the full spec risks output-token bloat."
  },
  {
   "id": "truthful-prompt-hud-budget",
   "keep": true,
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 0.1,
   "confidence": "low",
   "first_step": "prompts.py L135-136: delete 'If you identify such a bar, do not get distracted by it...' and 'DON'T DO THIS!' and replace with one equal-length line saying the edge bar usually counts down a per-level action budget and exhausting it ends the attempt. Also delete 'and whether the last action affected gameplay state or only HUD elements such as countdown bars' (COMPACT_TOOL_SESSION_ADDENDUM). Fix the 'ephemeral' wording in tool_agent.py L421 and L7200 when persistent functions are on. Do NOT add the scoring-rule text.",
   "reason": "Verified: prompts.py:135-136 tells the model to ignore the edge bar, while the flag-semantics line in PYTHON_ADDENDUM_TAIL says the bar is a budget, so the prompt contradicts itself. Siriki deleted the same phrase in v18.1gs (competitors/rellik13-milestone2/cells/40_patch_cell_v18_0_m90...py:498-508), but that was one of five fixes in a 13.40 to 14.49 step on a noisy leaderboard (his own identical notebook scored 5.02/5.50/6.42), so it is not attributable. Control transcripts show only 19 game-over prompts across 25 games (0 in 18 games). In sk48 the model did use the bottom HUD as the goal order, so 'HUD dismissed' is not evidenced. SCORING HALF DROPPED. The user's b4 scoring-guide arm scored 11.12 vs 17.743 and GB10 recorded 'no effect' (guidance used 1 time in 36). docs.arcprize.org/methodology confirms (baseline/actions)^2 with level-index weights. But taaf/game.py:385-411 caps each game at the weight of completed levels, so completing levels dominates efficiency, and a frugality push risks under-exploration. The '+28% hidden' claim and Kaggle discussion 744792 could not be fetched (JS page) and are unverified. Keep only the two-phrase deletion plus the tool-schema consistency fix at net-zero length."
  },
  {
   "id": "lean-opener",
   "keep": false,
   "status_in_franzen": "tried-no-gain",
   "impact": 1,
   "effort_days": 0.5,
   "confidence": "high",
   "first_step": "If ever retried: gate the static lines in _build_user_prompt (tool_agent.py ~L7096-7157) behind a new env flag. Not recommended.",
   "reason": "WRITEUP.md \u00a76 says shorter resume prompts, different tones and 'removing some reinsertion' gave no consistent gain, and ARC3_YIELD_RESUME_PROMPT=short/state_only already exists (CONFIGURATION.md:373). The user's arms.md has token-efficiency (prompts 33% smaller, no speed-up) and Prompt dedup (A3, 5.25, the worst run on record). My measurement of the control transcripts: user prompts average 2,701 chars and appear on 1560 of 2275 calls, versus 5,392 chars of thinking per call, so the opener is about 22% of per-call context growth and the static part about 13%. The best case is 10-13% slower context growth and fewer drains. The prefix is cached, so there is no compute saving. Siriki's writeup says prompt text hurt at his 37k context and that the 'last jump changed no prompts'. At Franzen's 128k window the pressure is lower and the reminders (use segmentation, never print the full board) likely shape behavior. Drop."
  },
  {
   "id": "history-state-io",
   "keep": false,
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 0.5,
   "confidence": "high",
   "first_step": "Not recommended. If done: cache per-entry payloads in tool_agent._ascii_history_view_payload (L4060) and return in-memory history from load_runtime_state (runtime_state.py L224) instead of re-parsing the file.",
   "reason": "The mechanism is real but measured as small. From the user's fz-arms-v4-control/benchmark.json, I computed latency between consecutive actions inside one snippet by history index. Median is 0.105 s (<100 entries), 0.33 (100-200), 0.54 (200-300), 0.73 (300-400), 0.86 (400-500), 0.92 (500+). My microbenchmark of write+load+payload+dumps gave 0.04 s at 50 entries and 0.40 s at 500. rellik13's cell-38 docstring (36/150 action calls timed out) is a longer-game regime than Franzen's (max 545 actions, mean ~225). Total in-snippet overhead is about 2,000 s of 165,436 game-seconds (1.2%), maybe 3-4% of active stream time with non-acting calls. There were only ~8 'Tool timed out' in 2275 control calls. The user's streams14 arm gave +15% throughput and zero score change (46.38 vs 46.25), so a 3-4% gain is invisible. Two sub-claims are wrong. 'Honor solver.minimal_diagnostics' does not apply, because the state file is functional state, not diagnostics. A segmentation cache is moot, because the sandbox child is a fresh process per call and segments lazily. The cost is half a day plus risk of state divergence. Dropping indent=2 alone saves only ~10-15% of the dumps/loads time. Drop."
  },
  {
   "id": "server-watchdog",
   "keep": true,
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 0.5,
   "confidence": "medium",
   "first_step": "Franzen launcher cell (arc-agi-3-v3/franzen/kaggle/template.ipynb cell 12, after the Popen at 'proc = subprocess.Popen(args...)'): add a daemon thread that relaunches the identical args only when proc.poll() is not None (plus 4 consecutive /health failures). Kill the process group, wait for the port and GPU memory to free, allow at most 2 restarts. Set ARC3_MAX_ANALYZER_FAILURES=-1 (or ~200) in setup_env so games outlive the ~9-min reboot.",
   "reason": "Verified gap: the launcher does a single Popen with only SGLang's own --watchdog-timeout 1800 and no restart. solver.py:1066-1081 ends a game after 10 consecutive analyzer failures ('nothing here can restart it'; CONFIGURATION.md:413), so a server death would cascade through all 110 games within minutes. Evidence from competitors: rellik13's v14.64 notebook says 'v1 lost the server mid-run and the harness spun on Connection refused', and the taaf-flashnext notebook has a 4-failure/max-2-restart watchdog (both local files). Counter-evidence: the user's ~9 baseline-config full runs (control, temp06, audit60k, streams14, ucb, ledger*) have 0 tracebacks/OOM in serve.log. Crashes occurred only in experimental configs (spec-adaptive OOM, nvfp4 KV, QSA cap). The two early 'never became healthy' failures were a startup problem fixed in later versions. Expected value is roughly P(crash) ~3% x ~60% of the score, about +0.5 pt, so it is insurance rather than lift. Restart only on a dead process, not on /v1/models latency, to avoid killing a healthy loaded server. Discussions 744995/729985 were not fetchable and are unverified."
  },
  {
   "id": "click-target-helper",
   "keep": false,
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 0.5,
   "confidence": "medium",
   "first_step": "Not recommended. If pursued: add click_candidates() in the sandbox bootstrap built on current_frame.segmentation, excluding the 4-cell border (ARC3_NOOP_GUARD_BORDER).",
   "reason": "Claims verified in the control transcripts: 17/25 games use MOUSE, and 157 of 1036 single-line MOUSE actions are NO-OPs (~15%). But no-ops are concentrated in games that fail for understanding reasons, not target selection: sc25 34/74 (score 0), tn36 (score 0), dc22 14/63. Click-heavy games that succeed have low dead rates (lp85 2/131 at score 100, r11l 5/103 at 100). The offline scorer (taaf/game.py:385-411) caps per-game credit at the completed-level weight, so saving ~6% of clicks buys little. The model already enumerates segmentation nodes (448/2275 calls). The 'tried_no_change by (shape_hash,color)' memory is the same family as Franzen's remembered-no-op guard (WRITEUP \u00a76: no clear benefit). A type-level tag could also mislead on state-dependent buttons. The supporting evidence (Reki, StochasticGoose, arXiv 2512.24156) is non-LLM graph explorers: the arXiv abstract confirms salience-based action prioritisation and a median 30/52 levels, but not 'tiers' or the 5-vs-19 numbers. Drop."
  },
  {
   "id": "state-graph-coverage",
   "keep": false,
   "status_in_franzen": "partial",
   "impact": 2,
   "effort_days": 1.5,
   "confidence": "medium",
   "first_step": "Not recommended as specified. The only salvageable piece is a harness-computed per-object-type x action effect table appended to the ledger-on-trim block once the user ports ledger-on-trim (backlog #1). Hook near _record_guard_observations (tool_agent.py L7773), reusing _interior_state_hash (L617).",
   "reason": "Hook points are real, but the idea reproduces several things already tried without gain. Franzen has repeated-state, remembered-no-op, death-repeat and death-ledger guards (CONFIGURATION.md:125-129, off, WRITEUP \u00a76 'no clear benefit') and ARC3_LEVEL_INVENTORY (:78, off). The user's own arms are close analogs: state-manager (model cites it in ~6% of requests), hypothesis-loop ('the model never reads the ledgered refutations'), action-tools (act called 0 times, so models do not adopt new APIs), and audit60k (a stall-triggered nudge fired 30 times, 44.10 vs 46.25). The proposed 'Never-interacted object types' stall line is exactly that audit-style nudge. Cited evidence is not comparable. arXiv 2607.01531 (OPINE-World) is real (I fetched the abstract: Claude Opus 4.8, 1.5% to 78.40% RHAE), but it is a frontier model writing a programmatic world model. arXiv 2512.24156 is a non-LLM explorer. The 78.4 vs 58.2 ablation and '19 vs 5 levels' were not verifiable from the abstracts. Skimming the control's sc25 and sk48 transcripts, failures are deep mechanics misunderstandings after extensive probing, not untested interactions. Effort is 1.5 days with an adoption risk. Drop."
  },
  {
   "id": "action-plan-annotations",
   "keep": true,
   "status_in_franzen": "absent",
   "impact": 2,
   "effort_days": 0.5,
   "confidence": "low",
   "first_step": "tool_agent.py L7194-7203: add an optional string 'plan' parameter beside 'code'. Thread it through _run_python_tool/_handle_action into the step callback payload. In solver.py ~L1860-1870 add 'plan' to the result_keys whitelist so it lands in HistoryEntry.result (HistoryEntry is a frozen dataclass, so use result and avoid schema changes). Add one prompt line. Then, once ledger-on-trim (franzen/base 98cc560) is ported, print the plan trail for the span a trim removed inside that same block. Skip the game_overs ledger.",
   "reason": "Verified absent: the python tool schema (tool_agent.py:7194-7203) has only 'code' and HistoryEntry holds action/frame/result. lordhansolo's source (competitors/kaggle-datasets/lordhansolo-taaf-kaggle-source/.../prompts.py:75,139) really has history[i].plan and game_overs, but LHS scored 23.84 vs Franzen 27.89 on a different stack, so the plan field is not shown to help. The user's analogs temper it. Hypothesis-loop reached 93% tool-parameter compliance yet was inconclusive, and the model rarely re-reads history (history[ appears in only 96/2275 control calls). The mechanism with a real shot is the auto-printed trail at a trim. It complements the user's only proven win, ledger-on-trim (+6.3 offline, single run), is harness-held rather than relying on model-written tags, costs ~20 tokens/call, and falls where the cached prefix is already invalidated. The game_overs half is dropped: the control had 19 game-over prompts in total (0 in 18/25 games). The plan param could also nudge tool-call generation, so A/B it after the ledger port. Franzen's summaries (no gain) are a caution, but a plan trail is not a summary."
  }
 ]
}
```
