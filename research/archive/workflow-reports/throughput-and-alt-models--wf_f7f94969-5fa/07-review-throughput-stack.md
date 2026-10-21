# throughput-and-alt-models: review:throughput-stack

Workflow run `wf_f7f94969-5fa`, phase Review. Win 2+ more games via throughput (sim-v2 sweep + a throughput stack of built-but-unrun fixes) and serve an alternative model (GLM-5.3-Flash EXL3, ISTA Qwen3.8-Flash-Next GSQ-RCO GGUF) by refactoring the RTX Pro 6000 SGLang launcher into per-model profiles

---

I found no defect that would break a run or change the default behaviour. There are 2 medium and 2 low defects, plus 3 nits. Everything below is from local runs on the CPU. I edited nothing in the worktree, and `git status` there is clean.

## Checks that passed

- **Default behaviour is unchanged.**
  - I made a fresh copy of `d55df75` (stack-v2) and re-ran `request_snapshot.py` there. The defaults, kaggle and stack-v2 environments produce the same hashes that `EXPECTED` hard-codes in `ARC3-Inference/tests/test_flags_off_requests.py:25-28`. The branch gives the same 3 hashes. `tput-safe` gives a different hash, as it should.
  - I rendered the `control` and `stack-v2` serving cells with stack-v2's own builder. The server argv and server env are identical to this branch's.
  - The rest of the serving cell differs only in blocks that are off by default (JIT cache, runtime-patch table, the `CUDAGRAPH_BS_EXTRA` assert).
  - Setting the `CUDAGRAPH_BS_EXTRA` default to `[3]` makes `test_control_launch_is_franzens` fail, so that test does catch a changed default.
- **Launch arguments are valid for SGLang 2.5.3.** I checked against `competitors/sglang-rtxpro6000`, whose Python code is the same as at d00d88ef.
  - `cuda_graph_bs_decode` is an `Optional[List[int]]` with no restriction on the values (`server_args.py:1968, 4961`).
  - The acceptance thresholds go to the target-only sampling kernel, so 0.5 works alongside FR-Spec (`eagle_utils.py:770-848`).
  - `tput-safe-lowm` uses the same argv as `tput-safe`.
- **Memory is not a concern.**
  - In the `fz-arms-v4-control` serve.log, 4.25 GB is free after graph capture.
  - The batch-10 graphs take about 0.41, 0.40 and 0.28 GB. All the smaller buckets together take about 0.05 GB, and capturing them took about 6 s.
  - So 3 extra buckets cost well under 0.1 GB and roughly 5–8 s of startup.
  - Nothing else changes the KV, Mamba or context sizing. Gate slots (11) are above MAXREQ (10), which equals the largest CUDA graph batch (10), so every decode batch has a graph.
- **Offline only:** the launcher, bench, JIT cache and bench patches contain no external URLs and no downloads.
- **Builds:**
  - All 15 arms and the 3 benches build in memory. The largest is 639 KB, under the 950 KB limit.
  - Every code cell parses.
  - The harness patch passes `git apply --check` on `bundle-base`.
- **Tests:** all 185 targeted tests pass (run with `nice`, one file at a time). ruff finds no undefined names or redefinitions (F821/F811) in `tool_agent.py`, `prefix_reuse.py` or `build_notebook.py`.
- **Merges:** every merged branch is the same commit as its `origin/` copy. The `tool_agent.py` merge resolution keeps all 124 lines of fix-3's change.

## Defects, most severe first

1. **Medium: the bench can't check the low-M gate at batch 7 or 8.**
   - `kaggle/bench.json:51` gives C0 `sweep1/4/7`; `:59` gives C3 `sweep1/4/8`. So C3's batch-8 sweep has no C0 counterpart, and C0's batch-7 sweep has none in C3.
   - The `tput-safe-lowm` gate ("no slower at batch 8 or below", `kaggle/README.md:78`) can only be compared at batches 1 and 4.
   - That matters because `nosplitk` changes every untuned shape, not only M > 32.
   - The test at `kaggle/tests/test_throughput_arms.py:224` says it covers batches up to 8, but it only checks C3's phases.
   - **Fix:** add `"sweep8"` to C0's phases, and `"sweep7"` to C3's (+65 s each). Change the test to assert that every sweep phase in C3 also runs in C0.
2. **Medium-low: plan step 6 (repeat `stack-v2`) can't be pushed.**
   - `_check_unique` (`build_notebook.py:591-600`) refuses any notebook identical to a recorded run.
   - `stack-v2` and `ledger-on-trim` also have the same fingerprint on this branch (`2f13a41dd06d01fb`), so pushing either one blocks the other.
   - **Fix now, while there's no quota:** add an arms.json `"replica"` key or a `--replica N` option. It would set an inert env var such as `TAAF_REPLICA=N`, which the fingerprint then includes. Alternatively, add a `stack-v2-r2` arm with that env var. Either way, add a test.
3. **Low: the bench gates aren't enforced.** Nothing stops a push of `tput-safe-lowm` or `tput-accept05` before the bench results are in. **Fix:** add an arms.json key such as `"gated_by"`, and have the builder refuse `--push` for those arms unless `--gate-passed` is given.
4. **Low (docs/process):**
   - The fork overview lives on `docs/fork-readme`. It has no row for the `throughput-stack` arms or for the `--bench` tooling, and CLAUDE.md requires the README update in the same PR.
   - `arm/throughput-stack` has no upstream and no PR.
   - **Fix:** add the rows on `docs/fork-readme`, push the branch, and open a PR (a git push needs no Kaggle quota).

## Nits

5. `serving/CUSTOM.md:21` says the extra graphs cost "about 70 MB per graph". The logs above show under 10 MB per extra bucket, plus about 2 s of capture each. The startup cost only matters in the competition run, where readiness already takes 472–700 s against the 720 s `SERVER_STARTUP_TIMEOUT`.
6. `serving/kaggle/jit_cache.py:41` leaves `SPEC_ACCEPT_SINGLE` and `SPEC_ACCEPT_ACC` out of `CFG_IGNORED`, although they can't change compiled kernels. As a result, `tput-accept05` can't reuse the cache that `tput-coldstart-save` saves.
   - **Optional:** add them to `CFG_IGNORED`, and add the embedded runtime-patch digests to the config snapshot. That digest gap predates this branch.
7. No test exercises ledger-on-trim with `ARC3_VERBATIM_HISTORY` across an actual trim. The 6-request snapshot never trims, so the merged ledger lines are only tested for absorbing replies, not for the ledger being re-shown after a trim.

The review scripts and their outputs are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/review-tput/`.
