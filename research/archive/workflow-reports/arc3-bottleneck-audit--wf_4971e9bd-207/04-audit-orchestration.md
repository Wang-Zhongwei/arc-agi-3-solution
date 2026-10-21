# arc3-bottleneck-audit: audit:orchestration

Workflow run `wf_4971e9bd-207`, phase Audit. Audit the whole Franzen harness (inference, serving, TAAF, notebook, configs) for bottlenecks; verify each; classify DEFINITELY vs MAYBE for GitHub issues

---

```json
{
 "findings": [
  {
   "title": "No SGLang supervision/restart; startup crash or >4.6 min outage forfeits the run",
   "location": "notebook cell 12 launch+health loop (notebook_cells.py:10724-10757, `proc` never polled again); ARC3-Inference/inference/framework/solver.py::_HarnessGameSession._play_inner (main 686-711; MAX_CONSECUTIVE_ANALYZER_FAILURES main 105, ANALYZER_RETRY_BACKOFF_SECONDS main 79/711); ARC3-Inference/inference/agent/tool_agent.py::_post_with_retries (main 557-655) and ToolAgent._chat_completion initial grace (main 5284-5298)",
   "resource": "reliability",
   "problem": "After the /health loop nothing watches the server (no thread, no relaunch; the system monitor only exists when not TRUE_SUBMISSION). (a) If SGLang exits during load, the loop just `break`s and the benchmark starts against a dead port: each of the 110 games burns a fresh 900 s initial-retry grace (ARC3_HTTP_RETRY_INITIAL_SECONDS, per ToolAgent) plus 11 failed cycles x ~25 s (3x5 s HTTP retries + 10 s backoff) before giving up, 10 at a time (gate), i.e. ~11 rounds x ~1160 s = ~3.5 h of spinning, then the run ends with ~0 score and ~5.5 h unused, never relaunching the server. (b) Mid-run, ARC3_MAX_ANALYZER_FAILURES is not set in cell 4 so it defaults to 10: a game permanently abandons itself after 11 consecutive failed requests (~275 s), well under a server restart (~7-10 min). The gate then admits the next waiter, which fails the same way, so the whole 110-game run drains in ~50 min. The 10 s backoff also sleeps while holding a gate slot.",
   "evidence": "Code confirmed in main and in the notebook. solver.py comment itself says 'nothing here can restart it'. No server crash seen in the 14 logged fz-* serve.logs (so probability looks low), but the user's atlas (reports/run-atlas/findings.md s10) records 'a server crash during startup used to leave the games running against nothing' and two cold-starting notebooks both hanging on the 190 GB load. 1 of 14 runs (fz-spec-adaptive) hit the 12-min SERVER_STARTUP_TIMEOUT (READY at 879 s), so slow/stuck loads do happen.",
   "estimated_cost": "Up to 100% of remaining score on any such event (expected loss = P(event) x remaining score); observed event rate 0/14 offline runs, so low-probability/high-impact.",
   "fix": "In TRUE_SUBMISSION run a supervisor thread polling `proc` and /health every ~10 s; on exit relaunch the identical args (JIT/page caches warm, ~5-8 min) up to 2-3 times, and retry the launch before releasing the benchmark if startup fails. Set ARC3_MAX_ANALYZER_FAILURES=-1 (unlimited) so games wait out a restart instead of abandoning.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Cold JIT in CUDA-graph capture adds ~150 s to every startup; caches never persist",
   "location": "notebook cell 12 (notebook_cells.py: CFG AUTOTUNE=True :10428, cache dirs under /tmp/sgl-intel/cache :10635-10648, --cuda-graph-bs-decode :10695-10697, launch :10724-10730)",
   "resource": "startup-teardown",
   "problem": "Every Kaggle container starts with empty Triton/FlashInfer/SGLang-JIT/TileLang caches (all pointed at /tmp/sgl-intel/cache), so the first target-verify CUDA-graph capture compiles from scratch and the GPU sits idle ~2.5 min after weights are already loaded. Total startup (notebook start -> first game request) is 9-15 min with the GPU idle the whole time.",
   "evidence": "serve.log 'Engine startup timings' in 13 fz-* runs: cuda_graph target_verify = 144.6-151.9 s (+7.0-7.9 s draft decode), vs ~1 s for each later batch size (control run: first of 7 sizes took 139 s, 'Capturing batches 1/7 [02:19]'; TileLang compile logged 01:25:58->01:26:01, so ~125 s of silent compile precedes it). scheduler_e2e 440-775 s. Notebook start -> 'Starting benchmark' in 14 runs: 528-730 s (median ~596 s; one run READY at 879 s). The run is ended by the 9 h kill (see deadline finding), so each startup second removes a second of play: median startup = 1.84% of 32,400 s, JIT alone = 0.45%.",
   "estimated_cost": "~145-152 s per run (~0.45% of the 9 h budget, ~0.5% of play tokens); startup as a whole ~1.6-2.7% of budget.",
   "fix": "Build the caches once on the identical Kaggle image/GPU (RTX PRO 6000, pinned docker image) and ship /tmp/sgl-intel/cache (triton, flashinfer, sglang jit, ~/.tilelang) in the wheelhouse dataset; copy it into place before launch and set TILELANG_CACHE_DIR. Verify the second launch's target_verify capture drops to ~8 s. Also test whether --disable-flashinfer-autotune accounts for the silent ~125 s (unverified).",
   "proposed_class": "DEFINITELY"
  },
  {
   "title": "185 GB model read starts ~110 s late; serial pre-launch work and mismatched warmup",
   "location": "notebook cell 6 (notebook_cells.py:10336 precache_model_thread, delay=60, threads=3) + cell 12 (:10661 sglang/torch import check, :10725 precache_model_thread.start(), :10701 --warmups structured_output)",
   "resource": "startup-teardown",
   "problem": "The critical path of startup is reading ~185 GB of checkpoints at ~330-725 MB/s. The dedicated model precache thread is only started right before Popen (t~70 s) and then sleeps another 60 s, so the first read happens at ~130 s; the server's own loader starts reading at ~96 s. Between ~19 s (wheel precache done) and ~96 s the disk is idle while a serial 45 s `import sglang, torch; cuda check` runs. Separately, the only server warmup is `--warmups structured_output` (a 64-token path the harness never uses, ~8 s), so the first real 5.5k-token multimodal prefill runs cold at ~490 tok/s (~11 s, all 10 streams stall) vs 7-16k tok/s steady.",
   "evidence": "Control fz-arms-v4-control/fz-arms.log: wheel precache done 18.7 s; import check 24.5->69.3 s; launch 69.8 s; 'Precaching: Started' 129.8 s; model precache 'loaded 62 files (185.40 GB) in 353 s' (=525 MB/s) ending 483 s; load_weight 387 s. Load time across runs 255-589 s (storage-bound, run-to-run spread). serve.log: first Prefill batch #new-token 5504 at 489.51 tok/s then 7166 tok/s. GPU weights are only 69.85 GB, so ~110 GB of the read is mostly the BF16 PLE table offloaded to host RAM (inference).",
   "estimated_cost": "Up to ~75-110 s of startup (0.2-0.35% of budget) if the load is storage-bound, plus ~10-20 s of cold-start stall.",
   "fix": "Start the model precache right after the wheel precache (t~20 s, delay=0, 8-16 threads) and move/drop the 45 s import check (the server pays the import anyway). Replace --warmups structured_output with a realistic ~8k-token image+text warm-up request in the health loop. Optional, quality trade-off unverified: the Albucino FP8-PLE checkpoint (already listed as acceptable in cell 12) would cut a large share of the bytes read.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Per-game 532-min clock starts after startup, so the plan overruns the 9 h kernel limit",
   "location": "notebook cell 16 (notebook_cells.py:10795-10805 max_runtime_s_per_game=532*60, 32400 s asserted only offline) + cell 20 (:10992-10999 soft_end=None in TRUE_SUBMISSION); ARC3-Inference/inference/framework/solver.py::_HarnessGameSession (started_at default_factory main 480; runtime_limit_reached main 521-526; request_timeout_seconds main 536-553; play() clock main 612-625); tufa-arc-agi-framework/src/taaf/game_api.py::_CompetitionScorecard.finish_run (main 80-87)",
   "resource": "reliability",
   "problem": "Each game's deadline is 31,920 s after its session object is created (bm.run start), not after notebook start, and a real submission has no soft deadline. 532 min + startup exceeds Kaggle's 540 min unless startup <= 8 min. Measured startup never was. The shared competition scorecard is only closed after the last game finishes (finish_run), which cannot happen before the kill. The priority scheduler's tail-fade window (20% of the run, ends at the per-game deadline) is also anchored to the same wrong end. If startup exceeds the 12-min cap (1/14 runs) the benchmark is released anyway and the game clocks run while the server still loads.",
   "evidence": "benchmark start 528-730 s after notebook start in 14/14 runs => per-game window ends 48-250 s (median ~116 s) past 32,400 s, before the drain tail (next finding). Kaggle's own clock likely also counts container boot/dataset mount before cell 4 (not measured). Counter-evidence on severity: Wang-lineage notes (arc-agi-3-v3/wang-duck-harness/notes/serving/ucb-v05-disk-and-retry-loop.md, docs/allocation.md) say a run that hit the 9 h limit ('ran the full 9 h as a submission') still got a valid score; whether the gateway needs the scorecard closed is listed there as 'not verified'.",
   "estimated_cost": "Planned play is 532 min but real play is 540 min - startup (525-531 min). Direct token loss is zero (the kill ends play either way); the cost is risk that an un-closed scorecard/killed notebook is scored differently, plus a slightly mistimed tail fade (~0.5-4 min of a 106 min window).",
   "fix": "Anchor the deadline to the Kaggle clock in cell 16 (which runs after READY): max_runtime_s_per_game = min(532*60, 32400 - (time.time()-NOTEBOOK_START_TIME) - margin), or give TRUE_SUBMISSION a soft_end like the offline branch. This trades a few minutes of play for a clean finish, so only do it together with the drain fix below and a shorter startup (JIT cache) to recover the minutes.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "End-of-run drain: gate waiters fire stale full-context requests after the deadline",
   "location": "ARC3-Inference/inference/agent/tool_agent.py::ToolAgent.analyze loop (should_stop check main 6719-6725, _maybe_handover call main 6744, _maybe_handover main 6372-6417, request sent right after) and _PriorityGate._enqueue_and_wait (main 2080-2097, `self._cond.wait()` with no timeout main 2095); solver.py::_HarnessGameSession.request_timeout_seconds (main 536-553, computed once per analyze() call, main 679)",
   "resource": "startup-teardown",
   "problem": "A game that trimmed its context blocks in gate.handover(); control flow resumes straight into _chat_completion with no should_stop() re-check, and with the request timeout computed before the wait (up to 900 s). When the per-game deadline passes, the active games exit and release their slots, then each waiter is admitted one by one, re-prefills its ~58k-token context cold and generates a response that step_env will refuse to act on (should_stop is true). With ~100 waiters in the 110-game run this drain serializes on GPU prefill.",
   "evidence": "Control run (fz-arms-v4-control/benchmark.json, 25 games, deadline 7260 s): 9 games ended at 7260.3-7261.7 s, the other 9 unfinished games ended late at 7270, 7299, 7302, 7310, 7313, 7314, 7316, 7324, 7334 s, i.e. ~4-8 s per waiter, run duration 7334 s vs 7260 s. Extrapolated to ~100 waiters: ~10 min of tail.",
   "estimated_cost": "No score loss by itself today (it happens after the point where Kaggle's 9 h kill already hit), but ~10 min of wasted cold prefills and it guarantees the notebook cannot exit cleanly before the limit; it blocks any fix that anchors the deadline to 9 h (previous finding).",
   "fix": "Re-check should_stop() (and recompute the remaining-time request timeout) after _maybe_handover returns; make _enqueue_and_wait use cond.wait(timeout) and abort when the session deadline passes or a global stop flag is set, so waiters wake and exit without issuing requests.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "110 gateway games are started sequentially before any request is issued",
   "location": "tufa-arc-agi-framework/src/taaf/benchmark.py::Benchmark.run (main 135-163: deepcopy + `game_copy.start_game(session)` loop before `asyncio.create_task(played_solver.run_games(...))`); tufa-arc-agi-framework/src/taaf/game_api.py::GameAPI._start_game (main 162-230, arcade.make main 205); notebook cell 20 (notebook_cells.py:10978-10999)",
   "resource": "wall-clock-cpu-io",
   "problem": "bm.run starts all 110 games (arcade.make -> remote-wrapper __init__ -> HTTP RESET, 10 s timeout each, under the Arcade lock) one after another on the event-loop thread before the solver task is created. The server is already READY (GPU idle) during this serial phase, so the first 10 streams cannot begin until all 110 gateway round trips finish.",
   "evidence": "Offline in-process engine: 'Starting benchmark' 666.9 s -> 'priority gate active' 668.2 s for 25 games (~50 ms/game). The real gateway per-call latency is unmeasured; arc_agi remote_wrapper.reset() is a blocking requests.post per game (arc-agi-3-v2 venv site-packages/arc_agi/remote_wrapper.py:60-105).",
   "estimated_cost": "Unmeasured; ~5-60 s of idle GPU at start if each gateway call costs 50-500 ms (<=0.2% of budget).",
   "fix": "Start the games concurrently (thread pool) or lazily in _play_one so games 0-9 begin immediately; also overlap game listing/arcade creation with the tail of server loading.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Hard-coded concurrency=120 silently starves any games beyond 120",
   "location": "notebook cell 16 (notebook_cells.py:10799 `bm.solver.concurrency = 120  # game count should be 110`); ARC3-Inference/inference/framework/solver.py::HarnessSolver._run_games (semaphore main 1524-1531) and _setup (ThreadPoolExecutor(max_workers=concurrency) main 1503-1515)",
   "resource": "correctness",
   "problem": "Concurrency is fixed before the game list is known (cell 20 builds it later from the gateway). Because games run until their deadline, a semaphore of 120 means any game index >=120 would never start, scoring zero. All of the 110-game design (gate, ThreadPool size, timers) assumes the count is exactly <=120.",
   "evidence": "Comment in cell 16 admits the assumption; OFFICIAL_110_RUN_COUNT=110 in competition_arcade.py. Code path confirmed; actual rerun game count not verifiable here.",
   "estimated_cost": "0 if the count is 110; if the gateway exposes N>120 games, each extra game is ~1/N (~0.8%) of the score.",
   "fix": "After bm.games is built in cell 20 set bm.solver.concurrency = max(120, len(bm.games)) (or len(bm.games)), and assert len(bm.games) <= concurrency.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "ANSWER to user question: yes, Franzen changed Tufa's prompts substantially (not a bottleneck)",
   "location": "ARC3-Inference/inference/agent/prompts.py (git diff 7652836..main: +254 lines changed, 19 new constants) and tool_agent.py::_build_system_prompt (main 2653-2750) + user-turn builders (e.g. LEVEL_START_USER_PROMPT main 4959)",
   "resource": "tokens",
   "problem": "Question: did Franzen change the prompts from Tufa's? Answer: yes. Core design is kept (single `python` tool, system prompt skeleton, tool-call format, BFS/search advice) but the text is extended and rewritten, mostly flag-gated on in his notebook (cell 4): (1) ARC3_ACTION_INFO adds UP/DOWN/LEFT/RIGHT/SPACE/MOUSE meanings plus UNDO and RESET notes (EXPOSE_UNDO=on exposes undo, which Tufa advertised but did not allow); (2) ARC3_EXPLAIN_GAMEPLAY_CHANGED + ARC3_NEW_CHANGED_PROMPTS reword board_changed vs gameplay_changed and the no-op/changed messages; (3) ARC3_LEVEL_TRANSFER_GUIDANCE adds a system bullet and a long level-up user prompt; (4) new game-over system flag semantics and a long game-over user prompt (diagnose cause, do not replay the fatal sequence); repeated-opener prompts now say 'no actions executed' and label old outcomes as reminders; (5) ARC3_FRAME_DIFF_HINT documents frame_diff(); ARC3_ANIMATION adds animation variables/timeline text; persistent-function lines (ARC3_PERSISTENT_FUNCTIONS, scope=game, imports on); (6) the segmentation description gains rotation/pose_hash/shape_hash and history/transition results are rewritten (last_action_call_result, per-transition .result, automatic RESET as its own transition); (7) ARC3_MEMORY_SECTIONS=off swaps the world-model instructions for a world-model-free line. Present in code but OFF in the competition config: step-verification hint, prefer-tool-calls line, death/no-op guard addenda, summary prompt.",
   "evidence": "Measured by building _build_system_prompt under both trees with the notebook's env flags (scratch measure_prompt.py): Tufa 12,358 chars (~3.1k tokens) -> Franzen 18,619 chars (~4.7k tokens), +50% (~+1.6k tokens). Prompt constants overall 12.8k -> 23.0k chars (8 -> 27 constants, several unused/gated). WRITEUP.md section 4 'Significant prompt changes' documents (1)-(3) and states the combination of undo, action info, level-transfer guidance and persistent functions gave a large gain on the competition set.",
   "estimated_cost": "Negligible: the added ~1.6k tokens sit in the static, prefix-cached system prompt (~1.3% of the 128k window, paid once per cold prefill); the longer game-over (~330 words) and level-up (~190 words) user prompts occur once per event. Informational item, not a throughput bottleneck.",
   "fix": "None needed for efficiency. If trimming tokens, the largest enabled blocks are FRAME_DIFF_HINT (1.7k chars) and ANIMATION (~1.4k chars); changing them would alter behaviour Franzen tuned, so only ablate with scores.",
   "proposed_class": "MAYBE"
  }
 ]
}
```
