# idle-slot-and-tail-streams-arms: review:tail-streams

Workflow run `wf_d74eec12-ff3`, phase Review. Implement #2 history-state perf fix and throughput-aware tail streams in separate worktrees; review and fix each

---

**Verdict: no defects, ranked by severity. I found no correctness, concurrency or flag-off-identity bugs; the items below are low-severity nits and test gaps.**

**What I ran**
- The 105 tests passed on all 25 serial runs and all 15 runs under CPU saturation (22 busy-loops on 11 cores). They also passed with `ARC3_TAIL_STREAMS=5` and `ARC3_MAX_ACTIVE_STREAMS=3` set in the environment. Imports resolve to the worktree.
- My own threaded stress used the real gate on the real clock: 14 games, 6 slots, cap 3, step and ramp, with `ARC3_PRIORITY_REFRESH_QUEUE=1`, tail fade and snapshots on. There were no hangs and no thread exceptions. Holders and waiters returned to 0 in every run. No request went out after the deadline. Running count never exceeded the cap except from games admitted before it engaged, as expected without preemption.
- The `bundle-base..HEAD` patch passes `git apply --check`.
- `build_notebook.py --arm tail-streams7`, run in a scratch clone, writes `setup_env['ARC3_TAIL_STREAMS']='7'` and `FROM='0.8'` before `os.environ.update`, with `demo_excluded_games=[]` and `max_runtime_s_per_game=int(121*60)`. The template sets `ARC3_MAX_ACTIVE_STREAMS=10`, so the gate exists and the arm is not inert.

**Why I found nothing wrong**
- **Flag-off identity holds.** With no setting, no clock, or a target at or above the slot count, `_tail_active()` is False. Capacity then equals the slot count, waits are untimed, `should_stop` is never called, and `acquire`/`handover`/`_maybe_handover` return True. The `_free` to `_holders` change is arithmetic-equivalent to the old counter.
- **No deadlock or starvation.**
  - Capacity only ever shrinks as time passes, including after a later `configure_clock`, so pumping on release and handover is enough.
  - If every holder hands over, holders reach 0 and `cap` waiters are admitted.
  - `should_stop` runs under the gate lock, which is an RLock. It only reads the game's own plain attributes (`game_run`, `current_state`, the stop event, monotonic time, token counters), with no lock or IO. No lock cycle is possible.
- **Withdraw and admission are mutually exclusive per token.** Both happen under the lock, so a withdrawn game cannot also be admitted. The phantom slot a stopped game holds above the cap is released in `play()`'s `finally`. A game stopped at the initial `acquire` still runs `_play_inner`, so its cleanup and finish run.
- **A False handover ends the turn cleanly.** `_maybe_handover` has the only call site. `latest_request_*` is guarded for `None` after the loop, and the unwind path is the same as the existing top-of-loop `stop_requested` break.
- **Ramp and fraction maths are correct.** The ramp holds 10, 9, 8, 7 at equal intervals, and `FROM` uses the gate's `_clock_start` and `_clock_deadline`.
- **Hot-path cost with the flag off is negligible.** One extra method call per `_pump`.

**Findings (all low)**
1. **Docs wording, `CONFIGURATION.md:304` and `tool_agent.py:2041`.**
   - They say waiters poll "while the cap is active".
   - The code polls from t=0 once the flag is on and a clock is configured (`tool_agent.py:2263`), not from `FROM`. A stopped game can also get `False` from `acquire` or `handover` before `FROM`.
   - Admission before `FROM` is still identical, and the early polling is harmless.
   - Fix: reword to "whenever the flag is on and the run has a deadline".
2. **`tool_agent.py:2342` (`_priority_gate`).** With `ARC3_MAX_ACTIVE_STREAMS` at 0 there is no gate, so `ARC3_TAIL_STREAMS` is silently ignored. It is documented but not logged, and this arm is unaffected. Fix: warn once there if `ARC3_TAIL_STREAMS` is set.
3. **`tool_agent.py:2076-2079`.** The `_free` property is unused outside `test_free_counts_unheld_slots`. Remove it, or accept it as test scaffolding.
4. **Test gaps.**
   - Nothing exercises the `solver.py:628` change. A test should check that `play()` releases the slot exactly once when `acquire` returns False.
   - No test enables the refresh queue together with the tail cap. My stress run covered it and was clean.
5. **Already noted by the implementer, not a new defect.** A stale request timeout lets a request admitted just before the deadline run past it. The flag-off late request is also unchanged.

**Scope**
Your relayed request asked for fixes to issues #2, #7 and #9. This branch fixes none of them, as the implementer said, and tail streams deliberately leaves slots idle late in the run, which opposes #7. I did not review the other worktrees (`history-perf`, `gate11`, `stub-dead`).

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/tail-streams/`:
- `ARC3-Inference/inference/agent/tool_agent.py`
- `ARC3-Inference/inference/framework/solver.py`
- `ARC3-Inference/CONFIGURATION.md`
- `ARC3-Inference/tests/test_priority_gate.py`
- `kaggle/arms.json`
