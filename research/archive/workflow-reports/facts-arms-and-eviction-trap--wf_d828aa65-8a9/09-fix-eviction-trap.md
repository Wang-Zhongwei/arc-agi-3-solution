# facts-arms-and-eviction-trap: fix:eviction-trap

Workflow run `wf_d828aa65-8a9`, phase Fix. Revise attention-prompt from advice to harness-computed steering facts, build indicator-notes and budget-alert arms, investigate and file the memory pin/eviction trap issue; review, fix, integrate, PRs

---

**Issue #58: guard test landed on a new branch, and the issue is corrected.** All seven review findings were real; I fixed or applied six and partly fixed the seventh. Two things to know first:
- **No worktree or branch existed.** The `wt/eviction-fix` worktree didn't exist and there was no eviction branch, so I created `wt/eviction-fix` on a new branch `arm/eviction-guard`.
- **It is based on `arm/memory-v2` @ d01f845, not `arm/stack-v2`.** That is the only branch with `tests/memory_sim.py`, so `arm/stack-v2..HEAD` also includes d01f845.

**Merge risk:** the memory-v2 implementation prompt also asks for `tests/test_memory_eviction.py`. If that agent writes one, the integrate step should keep this version.

**Findings**
1. **Test spec can't be built (fixed, f18f8f1).** `memory_sim.run()` had no hooks. It now takes `setup(agent, requests_seen)` and `before_turn(agent, turn)`. Both default to off, and the flags-off identity golden still passes.
2. **Test covered only part of the Kaggle path (fixed, b89c93e and 8fb1ec6).** I re-counted the real logs: 119 of 400 requests were second or later in their turn. `tests/eviction_sim.py` now adds:
   - a python call that doesn't act;
   - a reply over the per-turn token budget, which yields and then sends a resume opener;
   - one context-length rejection, which exercises the force-drain retry;
   - a failure after a successful request, which rolls the turn back.

   Each request is compared with the last accepted request of its turn, or else of the last turn that was kept, so the request after a rollback no longer fails assertion 2 falsely. The env adds `ARC3_PRIORITY_REFRESH_QUEUE=1`, and the fake gate defines `_snapshot_priority`. The test fails if a run never reaches a later-in-turn handover, two resumes and a rollback.
3. **Rule 1 wording (fixed in the issue comment).** It now reads: "Memory text never changes an already-sent message; the only legal prefix break is a real trim, and the checks must see it."
4. **Fable's evidence overstated the cost (fixed in the comment).** I checked the handbook: Fable's baseline already paused 18%. Active minutes fell 11% while levels per 100 active minutes fell 44% (p.82), and removing every pause is worth under 1 point (p.36). Our 283 s figure is latency, not a measured score cost. MAYBE stays.
5. **Rule 5, reserve the card's tokens (applied: rule dropped).** It is worse than useless. A 1,300-token reserve on every trim (3,900-char card, 4k drain) gave 17 handovers and 6 back-to-back full re-prefills, against 11 and 0 without it. At drain 0 it gave 27 against 26, and flags off already gives 20 there. What matters is keeping the card far below the drain, which Kaggle's 58k drain does.
6. **Drop recording could mask a handover (partly fixed).** Drops are now measured only where the trimmer, force-reduce or turn-end persist return fewer messages. The masking case is narrower than the review said: a floor-mode probe always shares its index with a real drop, except for an overflow retry that re-raises.
7. **No guard landed, and the label was wrong (fixed).** The test is in. I removed the `bottleneck` label; MAYBE stays.

**Side finding:** with floor mode on (off on Kaggle), ledger-on-trim gave 1 back-to-back full re-prefill against 0 with flags off. That config is left out of the guarded set. It's recorded in the issue comment.

Issue comment: https://github.com/adsdemaybe/daniel-franzen-sol/issues/58#issuecomment-6009032585

**git log arm/stack-v2..HEAD**
```
1e21ccd Document the slot-handover rule for memory features
8fb1ec6 Test that only real history drops hand the GPU slot over (#58)
b89c93e Add eviction_sim: does anything but a real drop hand the slot over?
f18f8f1 Add setup and before_turn hooks to memory_sim.run()
d01f845 Snapshot what stack-v2 sends the model, for a flags-off identity test
```

**Targeted pytest** (niced, imports resolve to the worktree), `tests/test_memory_eviction.py` plus `tests/test_memory_flags_off_identity.py`:
`13 passed, 3 skipped in 3.54s`. The 3 skips are the memory-v2 placement configs; they run once `ARC3_MEMORY_PLACEMENT` exists on the branch.

**PR summary**
- **Hypothesis:** memory text that edits already-sent history either gives the GPU slot away every turn (Fable's trap) or breaks the cached prefix with no handover. Nothing currently enforced against either.
- **Change:** test-only, no behaviour change. Hooks in `memory_sim.run()`, a new `tests/eviction_sim.py` sim, `tests/test_memory_eviction.py`, and a paragraph in section 11 of `CONFIGURATION.md`. No new flags.
- **Checks per config:** every handover follows a real drop; every prefix break has a handover; no back-to-back full re-prefills; at most one `<memory>` block per request.
- **Configs:** flags off, floor mode, ledger-on-trim, ledger plus death ledger, and memory-v2 trim/delta/both (inert until that arm merges). Each runs in its own niced subprocess, about 2 s for the whole file.
- **Results:**
  - Flags off: 9 handovers, floor mode 13, ledger 8, ledger plus death ledger 8. All follow real drops, with 0 unpriced breaks and 0 cascades.
  - Delta stripped and counted: 30 handovers, 21 without a drop.
  - Delta stripped, not counted: 23 unpriced breaks.
  - Fable's pin plus ledger: 15 handovers without a drop, up to 4 `<memory>` blocks and 4 pin copies in one request.
- **Separate check:** the card-reserve rule was simulated and dropped (11 to 17 handovers, 0 to 6 cascades).

Files are in /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad:
- wt/eviction-fix/ARC3-Inference/tests/eviction_sim.py
- wt/eviction-fix/ARC3-Inference/tests/test_memory_eviction.py
- eviction-check/guard/issue_comment.md
- eviction-check/guard/reserve_sim.py
