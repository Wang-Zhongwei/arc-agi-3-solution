# idle-slot-and-tail-streams-arms: fix:tail-streams

Workflow run `wf_d74eec12-ff3`, phase Fix. Implement #2 history-state perf fix and throughput-aware tail streams in separate worktrees; review and fix each

---

I checked all five findings against the code. Findings 1 to 4 were real and are fixed in five small commits; finding 5 needs no change. All 132 tests pass. This branch does not fix issues #2, #7 or #9 from your request (see the last section).

**Branch:** `arm/tail-streams`
**Worktree:** `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/tail-streams`

**Per finding**

1. **Docs wording: fixed (7dd2720).** The reviewer was right: waiters check their stop condition from the start of the run once the setting is on and the run has a deadline, not only after `FROM`.
   - I renamed `_tail_active` to `_tail_armed`, because "active" was the misleading word.
   - I corrected the class docstring, the poll comment, the `acquire`, `_stop_due` and `_maybe_handover` docstrings, the comment in `solver.py`, and the Deadlines bullet in `CONFIGURATION.md`.
   - The behavior itself stays: a stopped game leaving the queue early does no harm, and admission before `FROM` is unchanged.
   - A new test pins it: a stopped waiter at t=100, with full capacity, leaves the queue and gets False.
2. **Setting silently ignored without a gate: fixed (84c2376).** When `ARC3_MAX_ACTIVE_STREAMS` is 0 and `ARC3_TAIL_STREAMS` is non-zero, `_priority_gate()` now logs one warning. It reads the raw string rather than parsing it, so an invalid value is not newly turned into an error when there is no gate. The extra cost on every call with the flag off is one global read. `CONFIGURATION.md` is updated, and a test covers unset, 0 and 7.
3. **Unused `_free` property: fixed (b0b81a8).** I removed it, and the test now checks `_holders` directly.
4. **Test gaps: fixed (dbc9b54, e5a2ef6).**
   - The `solver.py` change is now tested. The test runs the real `_HarnessGameSession.play()` against a real tail-capped gate with the game stopped while queued. It checks that `acquire` returns False, the slot is released exactly once, no analyzer call is made, and the game finishes as cancelled.
   - `play()` runs in a thread with a join timeout, so a regression fails the test instead of hanging it.
   - I checked the test against three deliberate breakages in `solver.py`: not passing `should_stop`, releasing twice, and skipping `_play_inner`. Each one failed it.
   - The refresh queue is now tested with the tail cap. The random cap test runs with `ARC3_PRIORITY_REFRESH_QUEUE` on and handovers carrying snapshots. A new fixed-order test shows that under the cap, the waiter with the highest refreshed price gets the slot, where without refresh the highest queued price would have.
   - Adding the refresh cases exposed a weak check in the random test, not a gate bug. With seed 4 and refresh on, every game had finished before the tail started, so nothing could be admitted there. A released game now queues again later. Over a 2000-run sweep, the cap held in every run and every run admitted at least one game in the tail.
5. **A request let in just before the deadline can run past it: no change.** The request timeout is worked out before the gate wait, so it can be stale. This is behavior that existed before this branch and also applies with the setting off. The reviewer agreed it is not a new defect. Fixing it would change request timeouts for every policy, which is beyond this branch.

**`git log --oneline arm/kaggle-builder..HEAD`**
```
e5a2ef6 Test the tail cap with the refresh queue on
dbc9b54 Test that a game stopped in the gate queue releases its slot once
84c2376 Warn once when ARC3_TAIL_STREAMS is set without a gate
7dd2720 Say the tail-stream stop poll runs from the start of the run
b0b81a8 Drop the unused _free property from the priority gate
1f67876 Add the tail-streams7 arm
5f505c0 Document the tail-stream settings
ff53060 Let games parked by the tail cap stop instead of waiting for a slot
6a5b8b7 Cap gate admissions late in the run with ARC3_TAIL_STREAMS
68335f9 Count gate holders against a slot total instead of a free counter
```

**Pytest:** `132 passed, 14 warnings in 1.33s`
- The import resolves to the worktree.
- The same result held on 10 repeat runs, and once with `ARC3_TAIL_STREAMS=5 ARC3_MAX_ACTIVE_STREAMS=3 ARC3_PRIORITY_REFRESH_QUEUE=1` set.
- The 14 warnings are deprecation warnings from matplotlib's own pyparsing use. They appear because the new solver test imports `inference.framework.solver`, which loads matplotlib.

**Your request (#2, #7, #9):** this worktree is the tail-streams arm. Those issues are not addressed here, and the reviewer flagged the same thing. They appear to belong to the sibling worktrees (`gate11`, `history-state-perf`, `stub-dead-reasoning`), which I did not touch. Tail streams also leaves slots empty late in the run on purpose, which works against #7. It is off unless `ARC3_TAIL_STREAMS` is set, and only the `tail-streams7` arm in `kaggle/arms.json` turns it on. Nothing was pushed to Kaggle.

**Files changed:**
- `ARC3-Inference/inference/agent/tool_agent.py`
- `ARC3-Inference/inference/framework/solver.py`
- `ARC3-Inference/CONFIGURATION.md`
- `ARC3-Inference/tests/test_priority_gate.py`
