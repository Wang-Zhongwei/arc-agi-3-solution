# idle-slot-and-tail-streams-arms: implement:tail-streams

Workflow run `wf_d74eec12-ff3`, phase Implement. Implement #2 history-state perf fix and throughput-aware tail streams in separate worktrees; review and fix each

---

Tail streams are in on `arm/tail-streams`, with all 105 tests passing. With the flag off, admission matches the old gate exactly. One thing to check first: your request names issues #2, #7 and #9, and this branch fixes none of them. Tail streams deliberately leaves slots empty late in the run, which is the opposite of #7. The #2/#7/#9 work is in the other worktrees (`history-perf`, `gate11`, `stub-dead`).

**Branch:** `arm/tail-streams`, worktree `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/tail-streams`. Nothing pushed to Kaggle or GitHub.

**`git log --oneline arm/kaggle-builder..HEAD`:**
```
1f67876 Add the tail-streams7 arm
5f505c0 Document the tail-stream settings
ff53060 Let games parked by the tail cap stop instead of waiting for a slot
6a5b8b7 Cap gate admissions late in the run with ARC3_TAIL_STREAMS
68335f9 Count gate holders against a slot total instead of a free counter
```

**pytest:** `105 passed in 0.49s`. This branch had no tests before, so all 105 are new, in `ARC3-Inference/tests/test_priority_gate.py`. They pass 8 runs in a row and 6 runs in parallel. Imports resolve to the worktree.

**Measurements.** These come from a threaded simulation running the real gate: 25 games, 10 slots, 121 min compressed, ~28 s requests, a handover every 30–50 requests, 6 seeds. Script: `scratchpad/ts_sim/gate_sim.py`.

| | Mean holders, last 24.2 min | Time to reach 7 holders after the cap starts | Requests sent after the deadline | Run end past deadline |
|---|---|---|---|---|
| Flag off | 10.00 | – | 15 | +105 s |
| Step to 7 | 7.52 | 5.9 min | 0 | +55 s |
| Ramp to 7 | 8.73 | 20.6 min | 0 | +48 s |

- **The cap bites late.** Nothing is preempted, so slots only drain as games hand over or finish. With step 7 at FROM 0.8, the cap is reached around minute 102.7, giving about 18 effective minutes, not 24.2. The simulator agent should model this lag.
- **Flag-off problem reproduced.** The 15 late requests reproduce the audit's finding that gate-blocked games send a request after their deadline. With the cap on there are none.
- **Throughput (worked out from your tok/s figures, not measured):** about 7.5 streams in the tail means roughly 11% fewer total tokens there (about 2% of the whole run). Each running game decodes about 20% faster.
- **Kaggle arm:** `python kaggle/build_notebook.py --arm tail-streams7` builds. The notebook carries both env values, and its patch applies cleanly to `bundle-base`.

**Decisions:**
- **Settings.**
  - `ARC3_TAIL_STREAMS` unset or 0 means off. A value at or above the slot count is a no-op with one warning. A non-integer or negative value, or `FROM` outside [0, 1), raises `ValueError` when the gate is built. That matches how the existing tail-fade setting fails.
  - The ramp splits the time from FROM to the deadline into equal parts. For 10 to 7 that is 10, 9, 8, 7 slots for 6.05 min each in a 121-min run.
- **Waking at the deadline (cap on only).** A queued game checks its stop condition once a second. If it should stop, it unwinds without sending a request, and acquire/handover return False. To keep the acquire/release pairing intact, that game briefly counts as holding a slot above the cap and sends nothing. The tool agent ends the turn as `stop_requested` before building the request. The solver passes `should_stop` and otherwise behaves as before.
- **Flag-off identity.** With the cap off or no deadline, waits never time out and never call `should_stop`, so both calls return True as before. A test replays random acquire/handover/release scripts against a copy of the old admission rule and gets identical admissions.
- **Logging.** One warning when the cap engages and one per capacity change, with holder and waiter counts.

**Limitations:**
- **Flag-off late request not fixed.** A gate-blocked game can still send one request after its deadline when the flag is off. I left it because the task required identical behaviour with the flag off; it could be its own small arm.
- **Stale request timeout.** A request's timeout is fixed when the turn starts (analyzer timeout 900 s). A game parked in the tail and let in just before the deadline can therefore run a request past it. It cannot act afterwards, because `step_env` refuses once `should_stop` is true, but it uses server time.
- **Engagement timing.** The cap is checked only when a game releases, hands over or is polling in the queue, so the engagement log can lag slightly.
- **Merge conflicts in `kaggle/arms.json`.** The `gate11` and `stub-dead-reasoning` branches reformat the `panel-control` entry, so stacking them will need a trivial hand merge.
- **Not measured on hardware.** The simulation doesn't model games finishing or throughput, so the drain times are an upper bound.

Files are in `ARC3-Inference/`:
- `inference/agent/tool_agent.py` (`_TailStreams`, `_PriorityGate`, `ToolAgent._maybe_handover`)
- `inference/framework/solver.py`
- `CONFIGURATION.md` (new "Tail streams" section)
- `tests/test_priority_gate.py`

The arm is in `kaggle/arms.json`.
