# arc3-bottleneck-audit: audit:solver-loop

Workflow run `wf_4971e9bd-207`, phase Audit. Audit the whole Franzen harness (inference, serving, TAAF, notebook, configs) for bottlenecks; verify each; classify DEFINITELY vs MAYBE for GitHub issues

---

```json
{
 "findings": [
  {
   "title": "Per-action O(history) state rewrite, reload and ship to sandbox",
   "location": "ARC3-Inference/inference/framework/solver.py::_HarnessGameSession._execute_action (main 1234-1346, write_runtime_state at 1342), write_runtime_state (503-508), _play_inner (669); inference/agent/runtime_state.py::write_runtime_state (127-140, json.dumps(indent=2) at 139); inference/agent/tool_agent.py::_run_python_tool (5727, 5735, 5885-5899) and _ascii_history_view_payload (2843-2853); inference/agent/python_tool_sandbox.py::run_sandboxed_python (866-876, 926-954)",
   "resource": "wall-clock-cpu-io",
   "problem": "history_entries is never trimmed and every HistoryEntry carries a full 64x64 grid. write_runtime_state rewrites the entire history with indent=2, which forces the pure-Python JSON encoder on Python 3.11-3.13. It runs once per executed action, including every action inside a batch, and again at every _play_inner iteration even for yield continuations and right after an auto-reset. After each action() call the host re-reads that file (load_runtime_state), ASCII-formats and list-copies all N frames, json-dumps them and pipes them to the sandbox, which parses them again. The sandbox's 30 s deadline includes this host handler time, so long loops of single actions time out. The state file is written and immediately re-read in the same process, which is a pointless filesystem round trip.",
   "evidence": "Microbenchmark (Python 3.12, M-series, 64x64 frames), per action, N = history length 50/150/300/500: write 39/116/240/435 ms, load 17/53/107/190 ms, ascii+copy 28/83/173/311 ms, dumps 8/24/42/75 ms. Total 91/276/562/1011 ms; pipe payload 0.9/2.6/5.2/8.6 MB. indent=2 is about 4.4x slower than compact JSON. Control run fz-arms-v4-control: 5632 actions, about 4017 action() calls, 1560 analyze() calls, 25-545 actions per game. Estimated total about 1.2e3 s of mostly GIL-held CPU, about 16% of one core over the 7254 s run, about 47 s per game. It is quadratic, so the 400-545-action games pay 0.8-1.0 s per action. 7 'Tool timed out after 30s' events in the control transcripts; the tr87 snippet looped about 100 single action([m]) calls.",
   "estimated_cost": "About 2-4% of each stream's wall time late in games, and at most 1% of decode tok/s, since the control run's decode throughput is nearly flat from 7 to 10 running requests (643 to 711 tok/s). It caps single-action loops in a snippet at about 25-45 calls at N=300-500, each cap hit costing a lost turn. Kaggle CPUs are probably slower than this laptop, so the real cost is likely higher.",
   "fix": "Write compact JSON (indent=None, about 4-5x faster). Write once per step_env instead of per batch action, and skip writes when nothing changed. Pass history to the analyzer in memory instead of through the file. Send the sandbox only the last K frames or new transitions, or cache the serialized older frames, and compute ASCII lazily.",
   "proposed_class": "DEFINITELY"
  },
  {
   "title": "Game deadline is anchored to game start, not to the 9 h notebook limit",
   "location": "ARC3-Inference/inference/framework/solver.py::_HarnessGameSession.runtime_limit_reached (main 521-526), started_at (480), play() clock/gate setup (614-625), request_timeout_seconds (536-553); notebook_cells.py 10799-10800 (max_runtime_s_per_game = 532*60) and 10992-10999 (soft_end is None when TRUE_SUBMISSION)",
   "resource": "startup-teardown",
   "problem": "In the true submission there is no soft_end. Each game stops 532 min (31920 s) after its session object is created, which happens after the server load, the gateway wait and 110 start_game() calls. Startup is about 11 min, so the games' own deadline falls after the 32400 s notebook limit. The 480 s margin is smaller than the startup. The tail-fade window and time_remaining_seconds are also computed against this late deadline.",
   "evidence": "fz-arms-v4-control log: SGLang READY at 665 s, 'Starting benchmark' at 667 s, all games stop at 7928 s = 667 + 7260, which is the same 532*60*25/110 budget. For the true submission this puts the deadline at about 32587 s or later, at least 187 s past 32400 s. The Tufa notebook also sets soft_end=None for submissions, so nothing else enforces a margin. Not verified: how Kaggle scores a notebook killed at the limit; Franzen's real submission was scored, so it may be benign.",
   "estimated_cost": "At least the last 3 minutes of GPU work (about 0.6% of the budget) are at risk, because the games' own timers have not fired when the notebook is killed. If the kill prevents the scorecard from closing or voids the run, the cost is the whole score.",
   "fix": "After server READY, set max_runtime_s_per_game (or soft_end) from the notebook start: 32400 - (now - NOTEBOOK_START_EPOCH) - margin of at least 300 s. Keep the tail-fade deadline equal to that value.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Env HTTP steps have no retry, and the auto-RESET path is unguarded",
   "location": "tufa-arc-agi-framework/src/taaf/game_api.py::GameAPI._execute_action (main 248-264) and _start_game (162-246); ARC3-Inference/inference/framework/solver.py::_play_inner (main 651, 656-661, except at 731-736), _execute_auto_reset (1228-1232), step_env except (1110-1114); arc_agi 0.9.9 remote_wrapper.step/reset (site-packages, not in repo)",
   "resource": "reliability",
   "problem": "In COMPETITION mode every action is an HTTP POST to the gateway with a 10 s timeout. RemoteEnvironmentWrapper.step swallows RequestException and returns None, so GameAPI raises RuntimeError('env.step returned None'), with no retry. On the model path step_env turns this into an error payload, costing a wasted turn. If the request timed out after the gateway had applied the action, the harness history and the scorecard diverge by one action. The auto-RESET after a game over (_execute_auto_reset, run at loop top and for the warmup) is not wrapped like that path. A single failure there raises into the outer except Exception, marks the game 'crashed' and calls finish_game(), so the game is finished permanently and its remaining budget is lost. Sessions for all 110 games are also opened at t0 while later cohorts first act up to about 2.7 h later; an idle-session expiry on the gateway would fail their first step.",
   "evidence": "Control run: 5632 actions per 25 games, which extrapolates to about 24.8k step calls per 110-game run. 19 deaths in 25 games, which extrapolates to about 84 auto-resets, each a one-shot kill path. Late game start times in the control (cohorts at +0, +15 and +30 min) imply the last of 110 games starts about 2.7 h in. The user's own fork added _retry_unsent_requests with test_game_api_unsent_retry.py (franzen/kaggle/ARMS.md line 21), which suggests refused or unsent connections to the gateway were seen. Offline runs use the local wrapper, so this path has no measurement here.",
   "estimated_cost": "Each auto-reset failure forfeits one game: its remaining scheduled share, about 49 min of a stream, plus its score potential. Model-path failures cost about 2-3k tokens per occurrence plus possible desync. The probability per call is unknown.",
   "fix": "Retry connect and 5xx failures on step and reset, but never retry a request that reached the server. Wrap _execute_auto_reset in a bounded retry. After any failed step, re-read the engine state instead of trusting the local state.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Analyzer failure handling is asymmetric and gives up permanently",
   "location": "ARC3-Inference/inference/framework/solver.py::_play_inner (main 689-690, 691-712, sleep at 711, break at 704-707, except at 731-736), constants 79-81 and 105; inference/agent/tool_agent.py analyze() (7143-7160 retryable_failure, 7161-7179 except Exception returns None), _post_with_retries (557-662)",
   "resource": "reliability",
   "problem": "Request failures return retryable_failure and the solver retries up to 10 times, sleeping 10 s each time while the thread keeps its stream slot. Any other exception inside analyze() makes it return None, and the solver raises RuntimeError, which crashes and finishes the game with no retry at all. After 11 consecutive request failures, each costing about 15 s of in-place HTTP retries, the 10 s sleep and the request time (roughly 25 s per cycle, about 4.5 min in all), the game is finished permanently with no requeue. A deterministic failure, such as a non-retryable 4xx treated as retryable with history rolled back, replays the identical request 11 times first.",
   "evidence": "Code-confirmed. Across about 13 offline runs (about 300 game-runs) in kaggle-outputs there were 0 crashed games and no error notes, so the empirical rate is below 1%. The competition has 110 games.",
   "estimated_cost": "Expected loss is under one game per run; larger only if the endpoint stalls for about 5 min, which would permanently end the 10 in-flight games. Slot idle time during retries is small.",
   "fix": "Classify errors: retry non-request exceptions a few times before crashing. On persistent failure, hand over the slot and requeue the game with backoff until the deadline, instead of finishing it.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "minimal_diagnostics is ignored: per-step viewer and transcript artifacts written in submission",
   "location": "ARC3-Inference/inference/framework/solver.py::write_viewer_payload (main 848-895), _base_viewer_event (782-792), _append_*_viewer_event (794-846), _read_transcript_bytes/_transcript_delta_since (762-780, used at 670-688), _write_analysis_html (753-760, run at 743); tufa-arc-agi-framework/src/taaf/benchmark.py 130 and taaf/solver.py 45-47",
   "resource": "wall-clock-cpu-io",
   "problem": "Benchmark.run sets solver.minimal_diagnostics=True in the submission, and the Solver docstring says solvers should skip optional capture. HarnessSolver never reads the flag; there are 0 hits in main. job_dir is /kaggle/working, so every action builds a 64x64 board list plus ASCII and appends a viewer event to a jsonl file. Every analyze() call reads the whole transcript twice and stores the transcript delta, which includes the system prompt re-appended each call, in memory and on disk. An HTML copy of the transcript is written at game end. None of this is used in submission.",
   "evidence": "Control transcripts average 2.3 MB (max 3.6 MB). Estimated about 10-12 MB per game and about 1.2 GB of /kaggle/working for 110 games. Viewer events add about 40 KB of RAM per event, about 1.3 GB RAM total. CPU about 5-8 ms per action plus about 10 ms per analyze, about 2-3 s per game.",
   "estimated_cost": "Under 0.2% of throughput and some disk and RAM. Low impact; it is dead work in the hot path.",
   "fix": "Guard these writes with self.solver.minimal_diagnostics: skip viewer events and payload, transcript-delta reads and HTML, and keep only the transcript append if needed.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Out-of-range MOUSE coordinates are silently clamped and burn a counted action",
   "location": "ARC3-Inference/inference/framework/solver.py::_normalize_actions (main 936-937) and step_env requested_displays (1013-1016); inference/agent/tool_agent.py::_normalize_python_actions (5410-5446)",
   "resource": "actions",
   "problem": "Rows and columns are clamped with max(0, min(63, int(...))) instead of rejected, and no earlier layer validates the range. A model slip such as row=64 or col=-1 executes a real click at the clamped cell and consumes one counted action. It could be fatal. The echoed display uses the clamped values, so the model is never told its request was altered.",
   "evidence": "Code-confirmed. Frequency is unmeasured: the transcripts only show clamped values, so out-of-range requests cannot be detected from them.",
   "estimated_cost": "Probably rare. Each occurrence costs 1 action, plus an unnoticed wrong belief and possibly a death and reset.",
   "fix": "Return an error payload ('row/col must be 0-63') without executing, as is already done for unknown actions.",
   "proposed_class": "MAYBE"
  }
 ]
}
```
