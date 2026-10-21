# arc3-bottleneck-audit: audit:taaf

Workflow run `wf_4971e9bd-207`, phase Audit. Audit the whole Franzen harness (inference, serving, TAAF, notebook, configs) for bottlenecks; verify each; classify DEFINITELY vs MAYBE for GitHub issues

---

```json
{
 "findings": [
  {
   "title": "One failed game-start HTTP call aborts the whole 110-game submission",
   "location": "tufa-arc-agi-framework/src/taaf/benchmark.py::Benchmark.run (main L148-163 serial start loop, L188 finally); game_api.py::GameAPI._start_game (main L204-228)",
   "resource": "reliability",
   "problem": "Before the solver task exists, Benchmark.run starts all 110 games one after another. Each start_game is 2 unretried HTTP calls with timeout=10: a metadata GET made with a bare requests.get, then the make-time RESET POST. A failure is turned into a RuntimeError by GameAPI._start_game: 'Arcade.make returned None' (L213) or 'observation_space is None' (L228). No except wraps that loop (the handlers at L174/L177 only cover `await solver_task`), so the exception runs the finally block and propagates out of bm.run. The notebook cell then fails and no game is ever played. Retrying make() cannot repair a lost RESET response in competition mode: arc_agi/api.py::_get_or_create_environment refuses a second make for a game id already in the competition scorecard, and the client never learns the guid. The only safe retry is for connections that never opened, and main has none. The user's fork added exactly that (commit 1f499ed, 'same fix as lordhansolo's Milestone 2 notebook'), which suggests refused connects do happen in real reruns.",
   "evidence": "About 224 sequential unretried calls: 110 metadata GETs + 110 RESET POSTs + scorecard open + Arcade constructor GETs. arc_agi RemoteEnvironmentWrapper.reset swallows RequestException and returns None (remote_wrapper.py L127-137). Latency is NOT the issue: I measured a local reference REST gateway at about 30 ms per remote make(), so about 3 s for 110 games. Expected loss is about 224*p*S, where p is the per-call failure probability and S is the full score (offline control mean 46.25). If p=1e-4 that is about 1 point; if p=1e-3 about 10 points. No gateway-HTTP failure data exists in the user's runs, because all fz-* runs are OFFLINE and use in-process LocalEnvironmentWrapper.",
   "estimated_cost": "Tail risk of losing 100% of the submission. Roughly 2% of the score per 1e-4 per-call failure rate. Startup latency itself is only about 3-20 s.",
   "fix": "In Benchmark.run, wrap each start_game in try/except. On failure mark that run 'crashed' and continue with the other games, losing 1/110 instead of 110/110. Mount a connect-only urllib3.Retry(connect=3, read=False) on Arcade._session and on each env._session, as the user's fork does. Retry make() a few times with backoff, which is safe because the server refuses duplicate makes.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Gateway step/RESET failures: unguarded auto-reset kills the game; read timeouts desync state",
   "location": "game_api.py::GameAPI._execute_action (main L248-264); ARC3-Inference/inference/framework/solver.py::_HarnessGameSession._play_inner (main L651, L660 auto-reset, L731-741 except), ::step_env handler L1103-1114",
   "resource": "reliability",
   "problem": "arc_agi RemoteEnvironmentWrapper.step uses timeout=10 (connect plus read), no retry, and converts ANY requests error (connect refused, read timeout, reset, HTTP 4xx/5xx) into None. GameAPI._execute_action raises RuntimeError('env.step returned None'). (1) For a normal action, step_env catches it (L1110) and returns an error payload, which costs the model a turn. If the failure was a read timeout the server already applied the action (arc_agi/api.py::cmd), but TAAF's _current_state and actions_per_level stay stale. The model then re-issues the move on a board it has not seen, so an extra action is spent; level score goes as 1/actions^2. (2) _execute_auto_reset (after GAME_OVER, L660) has no handler of its own. Any failure reaches the outer except at L731 and sets state='crashed' plus finish_game(), so the game is permanently finished with all remaining levels at 0. The warmup RESET at L651 sits outside the try and dies the same way via _play_one -> _finish_after_error. The audited main lacks the user's connect-retry (commit 1f499ed in the franzen fork). That retry also covers only never-opened connections, not read timeouts or 5xx.",
   "evidence": "Exposure scaled from the offline control run (fz-arms-v4-control: 5632 actions, 20 RESETs over 25 games): about 24.8k step calls and about 88 auto-resets in a 110-game run. At p=1e-4 per call that is about 2.5 desync/error turns and about a 1% chance of losing one whole game. The user's commit message says: 'one refused connect to the competition gateway ends the game and zeroes its remaining levels.' The patched solver in the notebook (hunks at L8459+) adds no HTTP guard.",
   "estimated_cost": "Each error costs one wasted LLM turn (about 2-4k tokens, under 0.5% of tokens overall). Each lost game forfeits all remaining weighted levels. Each desync costs one or two extra actions on a level.",
   "fix": "Wrap both _execute_auto_reset call sites in a retry loop (5 tries, 2-10 s backoff) before crashing the game. Mount the connect-only Retry on env._session (the user's _retry_unsent_requests). Raise the read timeout above 10 s with an HTTPAdapter subclass that forces timeout=(5,60). Log every None-step loudly so desyncs are visible.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Per-game deadline ends 48-187 s after Kaggle's 32,400 s limit, and there is no cancel backstop in submission",
   "location": "ARC3-Inference notebook cells (notebook_cells.py L10800 `max_runtime_s_per_game = 532*60`, L10992-10999 soft_end=None when TRUE_SUBMISSION); taaf/benchmark.py::Benchmark.run (main L169-170, deadline task only if soft_end_time is not None); solver.py::_HarnessGameSession.started_at/runtime_limit_reached (main L480, L521-526)",
   "resource": "startup-teardown",
   "problem": "In a real rerun the soft deadline is disabled (soft_end=None). The only stop is each game's max_runtime_s_per_game = 31,920 s, measured from the game session's start_time, which is after the notebook's setup, the SGLang load wait and the serial game starts. The notebook itself asserts a 32,400 s budget. Games therefore stop after the hard limit instead of before it, and on a Kaggle kill the final close_scorecard and teardown never run. With soft_end=None, _cancel_at/_drain/_finish_remaining never arm. Any single hung game thread would also keep bm.run from returning and the scorecard from being closed.",
   "evidence": "'Starting benchmark...' appears at 528 s, 576 s, 612 s and 667 s of notebook time in fz-tput-all-v7, fz-ucb, fz-streams14 and fz-arms-v4-control (READY after 472-643 s). Deadline = start + 31,920 s = 32,448 / 32,496 / 32,533 / 32,587 s, which is 48 / 96 / 133 / 187 s past 32,400. The designed margin is 480 s but real startup is 528-667 s. The offline path does arm soft_end = start + budget - min(600, budget/2); the submission path does not (it is skipped on purpose).",
   "estimated_cost": "The last 48-187 s of every game (0.15-0.6% of compute) is unusable. If Kaggle voids a timed-out rerun, the whole score is at risk. That depends on the real rerun limit being 32,400 s, which I could not verify.",
   "fix": "Derive the limit at run time: max_runtime_s_per_game = 32400 - (time.time() - NOTEBOOK_START_EPOCH) - ~600. Or pass soft_end_time = NOTEBOOK_START_EPOCH + 32400 - 600 to bm.run for TRUE_SUBMISSION too, which also gives the _cancel_at backstop, a graceful drain and a guaranteed close_scorecard.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "The score-finalising close_scorecard is a single unretried call, and 'closed' is set before it succeeds",
   "location": "game_api.py::_CompetitionScorecard.finish_run (main L79-87); GameAPI._finish_game (main L266-296)",
   "resource": "reliability",
   "problem": "The shared competition scorecard is closed by whichever game finishes last. finish_run sets self.closed=True BEFORE calling arcade.close_scorecard (L86-87). That call is one HTTP POST with timeout=10 (arc_agi base.py L486). Any exception is swallowed by the try/except in _finish_game (a warnings.warn). The scorecard is then never retried and not marked failed. The server-side close in competition mode first make()s every game missing from the card, then builds two full EnvironmentScorecard serialisations (arc_agi/api.py::close_scorecard), so a slow response is plausible. All 110 games finish at about the same instant, at the common deadline.",
   "evidence": "Code path confirmed for the competition spec (operation_mode COMPETITION, `_competition_scorecard` shared). A reference gateway auto-closes idle cards after 15 minutes (STALE_MINUTES=15, scorecard_cleanup_loop), but the notebook exits before then, so whether that rescues a failed close is unknown. No run data exists for this path.",
   "estimated_cost": "Unknown probability. If a closed card is needed for scoring it is the single most valuable call in the run (100% of the score), so it deserves retries.",
   "fix": "Make finish_run retry close_scorecard 3-5 times with backoff, and set closed=True only on success. Log a loud error if it ultimately fails.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "HarnessSolver ignores minimal_diagnostics and rewrites the full frame history as pure-Python indent=2 JSON on every action",
   "location": "taaf/solver.py::Solver.minimal_diagnostics contract (main L45-47, L54); ARC3-Inference/inference/framework/solver.py::_HarnessGameSession.write_runtime_state (main L503-508; called at L628, L669 and unconditionally per action at L1342); agent/runtime_state.py::write_runtime_state (main L127-140, json.dumps(indent=2)); agent/tool_agent.py L5727/L5735/L6587 (load_runtime_state on each python tool call)",
   "resource": "wall-clock-cpu-io",
   "problem": "Benchmark.run stamps minimal_diagnostics=True on the solver in submission, and the Solver docstring says solvers must then skip optional writes. HarnessSolver never reads the flag (zero references in ARC3-Inference; the TAAF_MINIMAL_DIAGNOSTICS env var is set but never read anywhere). The runtime-state file holds EVERY history entry's full 64x64 grid (history_entries is never trimmed) and is rewritten after each executed action and before each analysis step. json.dumps with indent=2 bypasses the C encoder on CPython 3.12. Each stream thread does this synchronously while it holds its priority-gate slot, so its LLM request is idle. The tool agent also re-reads and re-parses the whole file on every python tool call. Per-game cost is O(actions^2).",
   "evidence": "Microbenchmark of the exact functions from main (macOS arm64, CPython 3.12): write = 0.04 s / 3.2 MB at 50 entries, 0.12 s / 9.5 MB at 150, 0.24 s / 19 MB at 300; load = 0.02 / 0.05 / 0.11 s. Using the recorded per-game action counts of the offline control run, action-driven writes alone total 685 s for 25 games (27 s/game) and 54 GB written, i.e. about 3,000 s and about 240 GB for 110 games. That is 0.9% of the 10 streams x 31,920 s = 319k stream-seconds. Analysis-step writes and tool-call re-parses add about as much, and Kaggle CPUs are slower than an M-series.",
   "estimated_cost": "About 1-3% of stream time (about 0.5-2% of decode throughput, since SGLang batches 9 instead of 10), plus GIL pressure and heavy disk churn in /kaggle/working.",
   "fix": "Gate optional writes on solver.minimal_diagnostics. Make the runtime state incremental: append one JSONL line per action, or keep only the last K frames. Drop indent so the C encoder is used. Let _serialized_runtime_state use the in-memory history instead of re-reading the file.",
   "proposed_class": "DEFINITELY"
  }
 ]
}
```
