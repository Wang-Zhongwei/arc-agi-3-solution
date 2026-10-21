# facts-arms-and-eviction-trap: review:eviction-trap

Workflow run `wf_d828aa65-8a9`, phase Review. Revise attention-prompt from advice to harness-computed steering facts, build indicator-notes and budget-alert arms, investigate and file the memory pin/eviction trap issue; review, fix, integrate, PRs

---

REVIEW: eviction trap (issue #58). Read-only; nothing edited or committed.

VERDICT: The issue's mechanism and numbers are accurate. I re-traced the code and re-ran the scans and the simulation niced, and everything reproduced. arm/stack-v2 is not affected. There is no fix branch (no wt/eviction-fix worktree, no evict/trap branch), and since nothing is broken that is defensible. The defects are in the regression-test spec, which can't be implemented as written and doesn't cover enough paths, and in some rule wording.

CLAIMS CHECKED (stack-v2 @ d55df75, tool_agent.py unless noted)
- **Slot events: true.** There are only three: solver.py:625 acquire, solver.py:634 release, and 6499 `gate.handover`. `handover` (2109-2119) releases the slot and re-enqueues in one `_cond` section.
- **Request order: true.** Each request runs trim 6825, then `_apply_due_ledger` 6826, then `_maybe_handover` 6827. `_maybe_handover` consumes `_context_was_trimmed` (6462-6464).
- **Bottom band: true.** The handover priority is below `_PRIORITY_UNTRIMMED_BASE` = 2M (2191), the band games that have not started yet acquire at (solver.py:625).
- **Flag sources: true.** `_note_history_evicted` (6375-6380) is called only from 6408, 6431 (`history != original_history`), 6627, 6642 and 6247 (summary-replace). The trimmer is called from 6405, 6781, 6825 and 6894, so all of them feed 6627.
- **Head merge: true.** `_with_ledger_head` (142-163) rewrites the first user message. So a false eviction with the ledger on becomes a real full re-prefill.
- **Fable's journal was counted as an eviction: true.** Their handbook says the journal used Franzen's summary-replace path, with `_note_history_evicted` called after the replacement.
- **Fable's pin dedupe: true.** `_fl_restore_pinned` (fable_layer_patch.py:219-230) dedupes on exact content (`m.get("content") == text`). The pin plus ledger loop therefore reproduces.
- **Real logs (fz-ledger-on-trim-r2, 5 games): reproduced.**
  - 400 requests; 15 head divergences, all at `request_index_within_turn` = 1 and all with the old head dropped.
  - 14 of 14 logged head divergences missed the cache.
  - 0 mid-history divergences, 0 ledger-only head rewrites, 0 of 378 append-only requests missed the cache.
  - Turn times: 13 trim turns, median 283 s, mean 1,084 s; other turns median 42 s.
- **Simulation: reproduced.** base 8, ledger 9, pin+ledger 27 (16 without a drop), delta_strip_counted 30 (21 without a drop). This matches sim_results2.jsonl.
- **Cross-references: correct.** #8 and #11 exist with the titles described.

DEFECTS, most severe first

1. **(Medium) The test spec can't be implemented on the committed harness.** The issue and the injected text say to "drive with tests/memory_sim.py" and to wrap the agent's methods and requests_seen. But `run()` in arm/memory-v2 @ d01f845 (`tests/memory_sim.py:258-308`) builds the agent and requests_seen internally and has no hooks. The `setup` and `before_turn` hooks exist only in the scratch copy (`scratchpad/eviction-check/sim/memory_sim.py:264-265, 330-331, 338-339`).
   - Fix: have the spec add `setup(agent, requests_seen)` and `before_turn(agent, turn)` parameters to `memory_sim.run()`, copied from the scratch file, or point to `sim/evict_sim.py` as the starting point.

2. **(Medium) The test covers only part of the Kaggle path.** The scripted model's `fake_chat` always returns an action call with `usage={}` (`memory_sim.py:308`), so every turn is exactly one request with no yields. In the real r2 logs, 119 of 400 requests were the second or later in their turn. Never exercised:
   - the in-loop trim, ledger and handover at 6825-6827 on a non-first request;
   - the overflow retry at 6894-6901 (`force_drain` and `_force_reduce_messages`);
   - resume openers (Kaggle sets `LOCAL_ANALYZER_YIELD_TOKENS=2048`; memory-v2 §6 adds no deltas to resume openers);
   - rollback at 7275 (memory-v2 restores shown_version there).

   Two smaller problems:
   - memory_sim's KAGGLE_ENV leaves out `ARC3_PRIORITY_REFRESH_QUEUE=1`. With it set, line 6487 calls `gate._snapshot_priority`, which the fake gate lacks, so the fake gate must define it.
   - Assertion (2) may fail falsely on the request after a rollback: its new opener can differ at d = len(prev) - 1 with no trim.

   Fix: add a non-action python turn, a one-off context-length `RequestException`, a usage value above the yield limit, and a `fail_turns` rollback. Exempt the request right after a rollback from assertion (2).

3. **(Low-Med) Rule 1's wording invites hiding the edit.** Issue Rule 1 says "Never call it, or reach one of its checks ... by ... stripping", and injected rule 1 says "in a way either check can see". Read literally, that allows the delta_strip_uncounted variant, which the issue's own table measures as worse: 21 of 31 unpriced re-prefills and 0 append-only requests. Rules 2 and 3 forbid it, but rule 1 contradicts them.
   - Fix: "Memory text must never change an already-sent message. The only legal prefix break is a real trim, and the checks must see it."

4. **(Low-Med) The Fable evidence overstates causation.** The injected text says "30% of game-minutes became pauses, and the score fell 28.65 to 10.49". It leaves out that Fable's baseline already paused 18% (187 of 1,043 minutes). It also leaves out Fable's own conclusions:
   - p.82: active minutes fell 11% while levels per 100 active minutes fell 44%, so "the queue explains only a small part" of the loss.
   - p.36: removing every pause gives +3–10% tokens, under 1 point.

   Our 283 s vs 42 s is per-game latency, mostly queue wait behind games that haven't started (2M band). It is not a measured score cost.
   - Fix: say the score cost of a false handover is probably small; the real cost is the forced full re-prefill when the ledger is on. MAYBE remains the right label.

5. **(Low) Rule 5 (reserve the card's tokens via `extra_safety_tokens`) is unverified and buys little.**
   - Analytically it moves the trim threshold and the landing point down by the same amount, so the spacing between trims doesn't change.
   - At drain 0 the cascade happens without any card: base_drain0 already has 20 back-to-back full re-prefills.
   - `_ledger_due` is set inside the same trim call, so the caller can't make the reserve depend on it, and reserving on every request shifts every trim earlier.
   - Under the 58k drain the whole rule is moot.
   - Fix: drop the rule, or run a sim variant with the reserve, or subtract the card only from the drain target.

6. **(Low) Drop recording can mask a handover without a drop.** Wrapping `_drop_oldest_history_block` records floor-mode probe drops that are later thrown away (6604-6612), which can hide a handover that had no real drop. Floor mode is off in Kaggle.
   - Fix: record a drop by comparing the trimmer's input and output lengths, and the persist step's `original_history` against the stored history.

7. **(Low) No guard landed, and the label is off.** For "maybe fix it", the cheap concrete fix is to land `tests/test_memory_eviction.py` now on arm/memory-v2, where memory_sim.py already lives. It would cover flags-off, ledger-on-trim and the negative control, and costs under 1 s CPU per config. Until then nothing enforces the rule except the memory-v2 implementation prompt. Separately, the `bottleneck` label is odd on an issue whose own Status says nothing current is affected.

Files:
- Scripts and results: /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/eviction-check/ (`real_scan.py`, `turn_waits.py`, `sim/evict_sim.py`, `sim/memory_sim.py`, `sim_results2.jsonl`, `issue_body.md`)
- Code reviewed: /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/stack-v2/ARC3-Inference/inference/agent/tool_agent.py
