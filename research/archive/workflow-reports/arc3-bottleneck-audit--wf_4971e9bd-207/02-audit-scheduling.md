# arc3-bottleneck-audit: audit:scheduling

Workflow run `wf_4971e9bd-207`, phase Audit. Audit the whole Franzen harness (inference, serving, TAAF, notebook, configs) for bottlenecks; verify each; classify DEFINITELY vs MAYBE for GitHub issues

---

```json
{
 "findings": [
  {
   "title": "Gate slots == max-running-requests, so GPU batch is under-filled by holders' non-request time",
   "location": "tool_agent.py::_max_active_streams (main L2119-2128), _PriorityGate (L1976-2117); solver.py::play (L603-634); notebook cell 4 ARC3_MAX_ACTIVE_STREAMS=10 (nb L10248), cell 12 CFG MAXREQ=10 / --max-running-requests (nb L10421, L10688)",
   "resource": "gpu-throughput",
   "problem": "The gate admits exactly 10 games and SGLang runs at most 10 requests, so any moment a slot-holder is not inside a server request (python tool exec, env step, JSON/image build, HTTP, wake-up after handover) leaves a hole in the decode batch. The gate keeps the slot across tool calls by design (WRITEUP L125 'except during short interruptions for tool calls'), and the server queue never has anyone to backfill because admission is capped at the same number as the server batch. The slot-holder count is fine (about 10 holders 79% of the time, at least 9 for 90%), so the loss is purely holder idle time (about 3 s per 28 s request).",
   "evidence": "Control run kaggle-outputs/fz-arms-v4-control/serve.log (25 games on 10 slots, same gate config). Time-weighted decode #running-req: 10 only 36.8% of the time, 9 for 31.1%, 8 for 15.7%, mean 8.75. ReqTimeStats in-flight mean 8.97 (44.5% at 10). Throughput by batch size: 10 -> 710 tok/s, 9 -> 697, 8 -> 682, 7 -> 638; time-weighted mean 679, i.e. 4.4% below an always-full batch. It worsens over the run: avg running 9.7 (t<600 s) -> 9.4 (1200-2400 s) -> 8.2-8.3 (t>6000 s) with tput 716 -> 648-657 (-9%), even though 18 unfinished games were always available to fill slots. Competition mode adds a gateway HTTP round trip per action that the offline local env does not, so idle is likely at least as large online.",
   "estimated_cost": "about 4-5% of decode throughput over the whole run, about 9% late in the run (roughly 30-60 tok/s of about 700). Binomial model with the measured tok/s-vs-batch curve: 12 slots over a 10-request server gives +2.2% at the average duty cycle (0.897) and +4.7% at the late-run duty cycle (0.82).",
   "fix": "Set ARC3_MAX_ACTIVE_STREAMS=11 or 12 while keeping --max-running-requests 10 and CUDAGRAPH_MAXBS 10, so the server queue absorbs holder jitter and the batch stays at 10. Context size is unchanged because the extra games are queued, not running, so this avoids the KV-sharing loss seen in the atlas '12 streams' arm. Watch --schedule-policy lpm: a queued full-miss request can wait behind warm ones; fcfs is safer if queue_duration rises.",
   "proposed_class": "DEFINITELY"
  },
  {
   "title": "Gate-blocked threads send a full-cost request after their deadline (no stop check after the gate wait)",
   "location": "tool_agent.py::analyze loop (main L6737 control_yield_reason, then L6744 self._maybe_handover()); _PriorityGate._enqueue_and_wait (L2080-2097, bare self._cond.wait() at L2095, no timeout or stop check); _PriorityGate.release (L2063); solver.py::_play_inner L679 (request_timeout_seconds computed before the wait); solver.py::play L633-634",
   "resource": "startup-teardown",
   "problem": "should_stop()/deadline is only evaluated at the top of the loop, before _maybe_handover(). A game parked in the gate cannot see its deadline. When the deadline hits, the 10 active games exit, each release() admits a waiter, and the waiter wakes and sends its request immediately. That request is a full 52-60K-token cache-miss prefill plus up to several thousand generated tokens, with an HTTP timeout computed before the multi-hour wait. Only after this response does the loop top see stop and exit, releasing the next waiter. This is a serial drain of about (waiters / 10) waves, and bm.run cannot return until every thread exits.",
   "evidence": "Control run: deadline 03:27:54 (start 01:26:54 + 7260 s). serve.log shows 8 requests STARTED 3-6 s after the deadline, all with input 52-60K and cached_input_len 0 or 4672: 407,714 new prefill tokens, 13,109 generated tokens, 42 s prefill, 104 s queue, last response 03:29:04, benchmark ended 03:29:08 (+74 s). Separately 9 requests straddled the deadline (25,986 tokens, the 'analyzer request timed out ... yielding' lines in fz-arms.log). In the 110-game config about 60-70 games are still waiting at the deadline (control: 8 of 18 unfinished), so expect roughly 6-7 waves of 45-55 s, about 3.3M prefill tokens (about 330-370 s of GPU) and roughly 5-8 min of post-deadline teardown.",
   "estimated_cost": "No score cost if the run is allowed to finish, but several minutes of teardown after the deadline and about 3M tokens of pure waste. If Kaggle's 540-min kill lands first (see next finding), the scorecard close at the end of the drain is at risk.",
   "fix": "Re-check should_stop() / deadline immediately after _maybe_handover() returns and skip the request. Give _enqueue_and_wait a timed wait that returns when the stop event or deadline passes, and have _pump drop expired waiters instead of admitting them. Recompute request_timeout_seconds after the gate wait.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Per-game 532-min clock starts after kernel boot with no soft end, so the real deadline (and tail-fade clock) overshoots the 540-min kernel limit",
   "location": "notebook cell 16 (nb L10786-10806: max_runtime_s_per_game = 532*60 for TRUE_SUBMISSION) and cell 20 (soft_end=None unless offline); solver.py::play L614-625 (configure_clock) and runtime_limit_reached L521-526; game_api.py::_CompetitionScorecard.finish_run (L79-89)",
   "resource": "reliability",
   "problem": "Each game's budget is 532 min from its thread start, which is bm.run start, not notebook start. The offline logs show bm.run starting about 666 s (11.1 min) after kernel start (benchmark.solver print at stream time 666 s), and the real run also waits for the gateway and allows up to 12 min of model load. The last game therefore stops at about 543+ min, past the 32400 s kernel budget the notebook itself asserts. The offline branch builds a soft_end = budget - 10 min, but TRUE_SUBMISSION passes soft_end=None, so there is no graceful margin. The gate's tail-fade deadline is the same shifted value, and the shared scorecard only closes when the last game thread finishes, after any post-deadline drain (previous finding).",
   "evidence": "fz-arms-v4-control/fz-arms.log: 'benchmark.solver: ...' at t=666.69 s. The notebook asserts target.max_runtime_s == 32400 only in the non-submission branch. 666 + 31920 = 32586 s > 32400 s by 186 s before any drain; with a 5-8 min drain the final close lands about 8-11 min past the limit. How Kaggle scores a killed run is not visible in the repo.",
   "estimated_cost": "If the kill discards unclosed-scorecard results: potentially the whole score. If the gateway scores incrementally: only a few minutes of tail play are lost. Treat as a risk, not a measured loss.",
   "fix": "Anchor the deadline to NOTEBOOK_START_EPOCH rather than game start (deadline = kernel start + 540 min - drain margin of 10-12 min), pass a soft_end in TRUE_SUBMISSION too, and make the gate/clock use that same absolute deadline.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Priority is only re-evaluated at trim, so stuck games hold slots for entire slices with no preemption",
   "location": "tool_agent.py::_maybe_handover (main L6372-6417; gated on _context_was_trimmed at L6380, set only via _note_history_evicted L6307); _PriorityGate.handover (L2068-2078); notebook ARC3_CONTEXT_DRAIN_TOKENS=58K, ARC3_HISTORY_ASSISTANT_TURNS=150",
   "resource": "tokens",
   "problem": "A game yields its slot only when its context is token-trimmed (or hits the 150-turn cap), a trigger measured in context tokens, not slot-time or progress. A game that just won re-admission therefore holds the slot for a whole slice even if it becomes hopeless early in it, and token-light, tool-heavy games hold slots longest per unit of GPU use. The documented hazard term C only affects who gets the NEXT slot, not whether the current holder keeps its slot.",
   "evidence": "Control run, reconstructed from transcripts plus benchmark.json history (60 slices, 73,028 busy slot-seconds): later slices median 746 s, mean 1171 s, p90 2454 s, max 3231 s; slices over 2000 s hold 36% of busy slot-time. 30% of slot-time (22,175 s) was spent after the game's last level-up, i.e. on the final never-cleared level; 14% (10,314 s) was in slices that cleared no level. This agrees with the atlas finding that 24-28% of tokens go to the final uncleared level. Measured yield: first slices 4.20 level-weighted points per 1000 slot-s, later slices 4.31, so the average is similar and the waste is concentrated in the stuck tail.",
   "estimated_cost": "Part of the 30% post-last-level-up and 14% zero-progress slot-time is avoidable; reallocating even a third of the post-last-level-up time at the measured average yield would be about +10% level-weighted score, but the true marginal yield is lower, so treat as a few % to ~10% upside.",
   "fix": "Add a mid-slice handover trigger at a tool-call boundary: when tokens-on-level or slice-seconds exceed a threshold and a waiting game's frozen priority exceeds the holder's refreshed priority, call gate.handover() (cost is one extra re-prefill of about 5 s, small against 1000-3000 s slices). Alternatively cap slice length in seconds.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Frozen priority with no aging parks early-stuck games at the gate for most of their lifetime",
   "location": "tool_agent.py::_PriorityGate._pump (L2099-2117), _enqueue_and_wait snapshot at L2080-2088, _maybe_handover L6395-6417; priority_scheduler.py::priority_value (L80-134)",
   "resource": "tokens",
   "problem": "A waiting game's snapshot (level, actions, tokens) is frozen at its trim; only the tail-fade term changes while it waits. Games whose priority drops after one slice (level 1, many actions and tokens spent) lose every subsequent slot contest to games that keep re-entering with higher scores, so they are never re-evaluated and can sit for the rest of the run. The 2M 'untrimmed' band additionally makes every never-started game outrank every returning game.",
   "evidence": "Control run transcripts: tn36 (0/7 levels) had its last request at t=1051 s and then waited about 6283 s of its 7334 s life (86%) with 68K tokens total; sk48 waited 5370 s (0/8, 117K tokens); sc25 4154 s (0/6, 94K); sp80 3397 s (1/6, 139K); ls20 about 2300 s at the end. Four of 25 games (16%) got only one slice after the first wave. Average cost per cleared level is about 42K tokens (atlas), so these games stopped at about 1.5-3 level-costs. The real run has 110 games on 10 slots (about 4.4x contention vs 2.5x offline), so this effect is larger there.",
   "estimated_cost": "Uncertain: these may be genuinely hard games and the hazard term could be right. The tradeoff is up to 12-16% of games frozen after a single slice, with no mechanism to retry a game stuck at 0 levels.",
   "fix": "Add aging, or a minimum-service rule such as every game must get a second slice before any game gets a fourth. Alternatively make the hazard C decay slower for games that have cleared 0 levels so a late-bloomer is retried. Evaluate on the scaled offline harness before shipping.",
   "proposed_class": "MAYBE"
  }
 ]
}
```
