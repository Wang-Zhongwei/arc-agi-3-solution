# verify-prs-for-merge: decide

Workflow run `wf_4b117a88-1cb`, phase Decide. Adversarially verify which open PRs are certainly correct to merge (builder, ledger-on-trim, docstrings, fork README, #4/#5 fixes) before merging

---

**Merge plan, as of 2026-10-07. Only #34 is certain enough to merge now. Everything else stays open.**

## Merge now: #34 only (arm/kaggle-builder → main, head 4d63980)

**Why it's certain:**
- It is tooling only. It adds `.gitignore` and `kaggle/*` (10 files, +752 lines) and touches nothing under `ARC3-Inference/` or `tufa-arc-agi-framework/`, so main's default behaviour still matches Franzen.
- The same builder code has already pushed v4 kernels that ran on Kaggle from start to finish. v4-stack-v1 played all 25 games with 0 tracebacks, and v4-panel-control also ran.
- A merge into main (615944f) succeeds with no conflicts.

**Steps:**
1. Make sure the head hasn't moved, then merge:
   `gh pr merge 34 --merge --match-head-commit 4d63980415211fc09a9dab54b22a950983650c23`
   - Use a merge commit, not squash or rebase. All 11 child branches contain 4d63980, and a merge commit keeps that history, so their diffs against main stay correct.
   - Don't pass `--delete-branch`. The repo already has `delete_branch_on_merge=false`, so `arm/kaggle-builder` stays.
2. No conflicts need resolving.
3. Retarget the PRs based on `arm/kaggle-builder`, so that a later merge lands on main and not on a stale branch:
   `for n in 35 38 39 40 42 43 45 51 52 53; do gh pr edit $n --base main; done`
   - Their diffs don't change. Each contains 4d63980, and main's own commits only touch `research/`, `runs/` and two README lines, which no arm branch conflicts with.
   - Add #44 to the list only if you keep it open (see the last section).
4. Follow-ups after the merge (not blockers):
   - The top-level `README.md` won't mention `kaggle/`, which CLAUDE.md requires. Fix it with the edited #49 described below.
   - Don't run `kaggle/push_queue.sh` while main is checked out. It runs `git commit` on `runs.json` in whatever branch is current, and `runs.json` on main starts empty (`[]`).
   - The `build_notebook.py` docstring mentions a `franzen-m2` tag that exists only locally. Optionally `git push origin franzen-m2`.
   - Note in `arms.md` that the builder is now on main.

## Leave open (not certain)

- **#45 ledger-on-trim:** The port is faithful, off by default, and has run on GPU inside stack-v1. The gain is not proven, though:
  - The fz runs fail `make significance` (P=0.896 and 0.651 against a 0.90 bar), and r2's +2.89 is under the noise floor.
  - The v4 base lacks fz's connect-retry fix.
  - The only v4 run that included it scored 43.32, and that run was confounded by other changes.
  - Merge it after two paired v4 runs of `ledger-on-trim` vs `control` from one commit both come out higher and pass `make significance`. Retarget it in step 3 either way.
- **#1 docstrings:** The code's syntax tree is unchanged, but it is not free of runtime effects:
  - The sandbox bootstrap grows by 16 KB. It is passed through `python -c`, so headroom before Linux's argument-size limit (E2BIG) drops from about 65 KB to about 49 KB.
  - Harness line numbers in rare fallback tracebacks shift.
  - The PR's "bootstrap byte-identical" claim is wrong, and only 5 claims across about 9.3k lines of docs were spot-checked.
  - Merging now adds conflicts to 22 of 33 arm branches, inside behaviour-critical code. Rebase it once the arm stack has landed.
- **#49 fork README:** It is docs only, but wrong for main:
  - The intro says every change is off by default, but #43, #51 and #52 change defaults.
  - Links to `serving/CUSTOM.md` and `analysis/README.md` return 404 on main.
  - The stack-v1 cause is out of date, and #43 has no warning about the suspected sandbox errors.
  - Its head has also moved from the checked 77c4b1a to 45e29c3, and those faults are still there.
  - Apply the verifier's edits 1–8, re-check, and merge after #34 so it describes the builder.
- **#51 fix-5:** Its tests are good (5 of 8 fail on the old code). But a helper redefined in a call that later times out now quietly falls back to the old version, where it used to raise a clear NameError, and it has never run on GPU. It needs either a note to the model plus a test, or paired runs.
- **#52 fix-4:** It adds an unmeasured 3x wall-clock cap (3x the actions in a runaway loop), a new thread, lock and message type in the sandbox, and changes to the error text the model sees. It has never run on GPU. Needs paired runs.
- **#43 fix-2 (#2 history perf):** Main suspect for sandbox IndexError and KeyError rates rising 2–3x in v4-stack-v1. stack-v2 leaves it out. Keep it until a D′ + ledger run without #2 is compared with one that has it.
- **#53 fix-3 (verbatim history and prefix diagnostics):** Behind flags, never run on GPU, and not checked in this pass.
- **#38 gate11 and #39 stub-dead:** Config-only arms that have never run. Simulated gate11 gains are only an upper bound. Both are bundled in #69.
- **#35 easy-first:** The smoke test isn't a verdict, the replay simulator puts it within noise, and its code sits under #36, #37, #55 and #63.
- **#36 D′ (form-priority):** Its only real loss came from the confounded stack-v1 run. sim-v2 recommends one more test, D′ + ledger without #2, before dropping it.
- **#37 replay-sim:** Analysis tooling stacked on the D′ and easy-first harness code. Its D′ win didn't hold up in the real run.
- **#40 animation-digest:** Still in progress and never run.
- **#42 custom-serving:** Changes the serving launcher, which matters at runtime on Kaggle. Not checked in this pass.
- **#47 nvfp4:** Never run. It is paired against stack-v1, which lost, and #68's `qwen38fn-nvfp4` profile reproduces its launch. Re-pair it against stack-v2 or control if you pursue it.
- **#48 moe-routing:** The `base_norm` routing has never run. The experts16 loss was measured without it.
- **#50 serving-bench, #54 fix-6 (JIT cache), #68 model-profiles:** Serving tooling, none of it run on Kaggle (no quota).
- **#56 stack-v2:** The submission candidate, never run. It contains #45, #51 and #52, so it inherits their uncertainty.
- **#57, #59, #60, #62 (facts arms):** Never run. #62 is a draft.
- **#61 eviction-guard:** Tests only, but stacked on stack-v2. Merging it would bring unverified #51 and #52 into main.
- **#63 sim-v2:** Useful simulator, but its branch chain carries the D′ and easy-first harness code into main.
- **#64, #65, #66, #67 (memory-v2/xml/level-path/v3):** Never run. #67 is a draft and passes the #58 eviction guard only with a 15,000-token simulation window.
- **#69 throughput-stack:** Never run (no quota).

## Suggestion only: could be closed without merging (keep the branches)

- **#44 tail-streams (draft):** Every stream cap lost in simulation, and a real measurement backs this: total decode speed falls from 789 to 677 tok/s at 7 streams. `arms.md` says not to spend Kaggle compute on it. Nothing builds on it.
- **#46 stack-v1:** Lost in a real run (43.32 vs 46.25 control) and is superseded by stack-v2, which drops D′ and #2. The next D′ test sim-v2 recommends is a different combination. Keep `arm/stack-v1`, because #47 targets it and its `runs.json` holds the v4 run records.
- **#55 sched-policies (weaker case):** `arms.md` marks it superseded by sim-v2, and its "nothing beats D′" result no longer matters now that D′ lost. #63 already contains its commits, so closing it only tidies the list.

All verified heads still match (#1 297b9b0, #45 7f8beba, #51 5406c50, #52 38df48d); only #49 has moved. No `merge-verify` worktrees are left.
