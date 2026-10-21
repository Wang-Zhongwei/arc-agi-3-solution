# throughput-and-alt-models: fix:throughput-stack

Workflow run `wf_f7f94969-5fa`, phase Fix. Win 2+ more games via throughput (sim-v2 sweep + a throughput stack of built-but-unrun fixes) and serve an alternative model (GLM-5.3-Flash EXL3, ISTA Qwen3.8-Flash-Next GSQ-RCO GGUF) by refactoring the RTX Pro 6000 SGLang launcher into per-model profiles

---

Six of the seven findings are fixed and one part was rejected. All six fix commits are on `arm/throughput-stack`, plus one README commit on `docs/fork-readme`. Nothing was pushed, no PR was opened and nothing ran on a GPU. Both worktrees are clean.

**Per finding**

1. **Bench can't check batch 7/8 (medium): fixed** in `996cee9`.
   - C0, C3 and the disabled C4 now all sweep batches 1, 4, 7 and 8. That adds about 65 s per config: `tput-gates` is about 70 min of its 90, `levers` about 145 of its 150.
   - The test now requires C3 and C0 to run the same sweeps. The README gate row says which batches are compared.
2. **Repeat `stack-v2` run can't be pushed (medium-low): fixed** in `2183f56`.
   - New `--replica N` option (N ≥ 2). It adds an env var, `TAAF_REPLICA=N`, that nothing in the harness or framework reads, so the fingerprint changes. `runs.json` records `"replica": N`.
   - Run 1 builds exactly as before (tested). It also works for `--bench`, and `push_queue.sh` accepts `<arm>@N`.
   - New tests are in `kaggle/tests/test_builder_rules.py`. A mutation check (making the option do nothing) fails them.
   - `stack-v2` and `ledger-on-trim` sharing a fingerprint is correct (they are the same run), so I documented it rather than changed it.
3. **Gates not enforced (low): fixed** in `ba9156a`.
   - New arms.json `"gate"` key on `tput-safe-lowm` and `tput-accept05`.
   - The builder still builds a gated arm and prints its gate, but refuses `--push` without `--gate-passed`. It also refuses `--gate-passed` on an ungated arm or a bench. `push_queue.sh` passes the flag for arms listed in `GATE_PASSED="..."`.
   - I checked `main()` with every non-git subprocess blocked: a gated push is refused, and with `--gate-passed` it reaches the push call, where the block stopped it.
4. **Docs/process (low): partly fixed.**
   - Fixed: rows for the serving bench (PR #50) and the `tput-safe` arms are added on `docs/fork-readme` as `86c1526`, in the existing worktree `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/readme`. It is local only, so integrate needs to push that branch too.
   - Rejected: pushing the branch and opening the PR. The task rules leave that to integrate.
5. **Nit, graph memory figure: fixed** in `ea3f021`. The `fz-arms-v4-control` serve.log shows the batch-10 graphs taking 0.41, 0.37 and 0.28 GB. Each smaller size adds at most 10 MB and about 1 s per capture phase. `CUSTOM.md` now says about 10 MB and 3 s per extra size, so `[3,5,6]` costs about 30 MB and 8 s.
6. **Nit, JIT cache hash: thresholds fixed, patch digests rejected.**
   - Fixed in `63ef8cf`: `SPEC_ACCEPT_SINGLE` and `SPEC_ACCEPT_ACC` are now ignored by the cache hash. In SGLang they are plain float arguments to a prebuilt sgl-kernel op, called outside the CUDA graphs, and `set_internal_state` can change them on a live server. `tput-accept05` can now restore `tput-coldstart-save`'s cache (tested). No cache has been published yet, so no saved cache goes stale.
   - Rejected: adding the runtime-patch digests to the hash. The kernel caches key each kernel by its own source, so a patch cannot make a restored kernel wrong. Adding the digests would only force a separate cold-start-save run for `tput-safe-lowm`.
7. **Nit, ledger and verbatim history never tested across a trim: fixed** in `519c4f5`.
   - The new test runs two scripted turns with the assistant-turn cap at 2. Turn 1 shows no ledger. Turn 2 opens with the `<memory>` block on the first kept user message, and the kept reply still has its blank lines.
   - It fails with verbatim history off, and with the ledger's line that reads the reply's content removed.

**Git log `arm/stack-v2..HEAD` (first-parent)**
```
519c4f5 Test the ledger and verbatim history together across a trim
63ef8cf Leave the MTP acceptance thresholds out of the JIT cache hash
ea3f021 Correct the cost of extra CUDA-graph sizes in CUSTOM.md
ba9156a Refuse to push a gated arm without --gate-passed
2183f56 Add --replica N to repeat a run past the duplicate check
996cee9 Run the batch 7 and 8 sweeps in both C0 and C3
6d946d3 Add the tput-gates bench: the configs that gate the tput arms
6a8511a Add the throughput-stack arms
edd3c23 Pin the flags-off request bodies to arm/stack-v2's
56ab8cd Let an arm apply serving/bench_patches/ files to SGLang
cbc112a Read bench runtime patches through one helper
5f64e58 Add CFG CUDAGRAPH_BS_EXTRA for extra CUDA-graph batch sizes
e99e18f Merge arm/fix-3-prefix-reuse into throughput-stack
080082c Merge arm/stub-dead-reasoning into throughput-stack
d730c1d Merge arm/gate11 into throughput-stack
6ced208 Merge arm/fix-6-cold-start into throughput-stack
504fc26 Merge arm/serving-bench into throughput-stack
```
Every new commit ends with the two required trailer lines.

**Tests** (each file on its own under `nice`, after clearing `__pycache__`)

| File | Passed |
|---|---|
| `test_verbatim_history` | 9 |
| `test_prefix_reuse_probe` | 23 |
| `test_ledger_on_trim` | 7 |
| `test_flags_off_requests` | 4 |
| `test_retained_functions_survive_failures` | 8 |
| `test_tool_timeout_result` | 2 |
| `test_sandbox_deadline` | 16 |
| `test_bench` | 33 |
| `test_jit_cache` | 33 |
| `test_launcher_jit_cache` | 6 |
| `test_build_serving` | 31 |
| `test_throughput_arms` | 14 |
| `test_builder_rules` | 8 |

That is 194 in total.
- ruff finds no F-errors.
- All 15 arms and 3 benches build, with no syntax errors. The largest notebook is 640,657 bytes.
- The harness patch passes `git apply --check` on `bundle-base`.
- The flags-off request hashes still match stack-v2's.

One thing to know: `--submission` of an arm that already ran as an experiment is refused as a duplicate. This was already the case before my changes; `--replica` gets around it.

**PR summary**
- **Hypothesis:** stacking the throughput levers on stack-v2 raises decode tok/s without changing the default behaviour. The levers are gate11, stub-dead-reasoning, verbatim history and CUDA graphs for batches 3/5/6. The low-M GEMM patches and relaxed acceptance stay gated on a bench.
- **Merges:** serving-bench, fix-6, gate11, stub-dead-reasoning and fix-3 into arm/stack-v2. The one real conflict, in `tool_agent.py`, keeps both sides.
- **New:** CFG `CUDAGRAPH_BS_EXTRA`, and an arms.json `runtime_patches` key, both off by default.
- **Arms:** `tput-safe`, `tput-safe-lowm` (gated), `tput-accept05` (gated, lossy), `panel-tput-safe` and `tput-coldstart-save`.
- **Bench:** `tput-gates` (C0/C3/C1, about 70 min). C0 and C3 both sweep batches 1/4/7/8.
- **Builder:** `--replica N` for paired repeat runs, and an arms.json `gate` key, so a gated arm pushes only with `--gate-passed`.
- **JIT cache:** the acceptance thresholds are left out of the hash, so one `tput-coldstart-save` cache serves every `tput-*` arm.
- **Checks:** flags off, request bodies hash the same as stack-v2's, and the control/stack-v2 server argv and env are the same as under stack-v2's builder. 194 targeted tests pass, and every notebook builds under the 950 KB limit.
- **Cost:** the extra graphs add about 30 MB of GPU memory and about 8 s to startup. Expected gain is at most ×1.06 throughput, about +1.7 offline points (sim-v2 upper bound).
- **Not run (no quota).** Order: bench `tput-gates`, then `tput-coldstart-save`, `panel-tput-safe`, `stack-v2` + `tput-safe`, gated arms if they pass, then `stack-v2@2` + `tput-safe@2`.
- **Docs:** `kaggle/README.md`, `serving/CUSTOM.md` and the fork README row on `docs/fork-readme`. The review fixes are also logged in `arms.md`.

All paths are under `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/throughput-stack`:
- `kaggle/build_notebook.py`
- `kaggle/arms.json`
- `kaggle/bench.json`
- `kaggle/push_queue.sh`
- `kaggle/README.md`
- `kaggle/tests/test_builder_rules.py`
- `kaggle/tests/test_throughput_arms.py`
- `kaggle/tests/test_jit_cache.py`
- `serving/kaggle/jit_cache.py`
- `serving/CUSTOM.md`
- `ARC3-Inference/tests/test_ledger_on_trim.py`
