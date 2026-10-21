# throughput-and-alt-models: fix:model-profiles

Workflow run `wf_f7f94969-5fa`, phase Fix. Win 2+ more games via throughput (sim-v2 sweep + a throughput stack of built-but-unrun fixes) and serve an alternative model (GLM-5.3-Flash EXL3, ISTA Qwen3.8-Flash-Next GSQ-RCO GGUF) by refactoring the RTX Pro 6000 SGLang launcher into per-model profiles

---

Five of the seven review findings are fixed in six commits, including one docs commit. The JIT-cache nit is rejected, and the throughput-stack merge is partly fixed, with the rest left as merge notes below. Nothing was pushed, and no PR was opened.

**Per finding**

1. **Fixed (`3f76fa9`): a bench could not clean up a server that died during startup.** Confirmed: before the fix the bench's namespace had no `proc` after a failed start. The launcher now sets `globals()["proc"] = proc` right after `Popen` in `launch()`. I kept the local name, so the log-streaming thread still watches its own process. `launch_harness.launch`/`record_bench` take an optional `bench_ns`. Two new tests, one on the launcher alone and one running the real built bench notebook's launcher through `bench.launch()`, both fail without the fix and pass with it.
2. **Fixed in docs (`4022071` and `arms.md`): the GSQ run order mixed model and harness.** `gsq` now pairs with `control` and `gsq-panel` with `panel-control`, from the same batch. This is written in `serving/CUSTOM.md`, `serving/profiles/README.md`, `kaggle/README.md` and the `arms.md` run order. `arms.md` also says that a GSQ test on the throughput harness needs `gsq-tput-safe` (tput-safe's env plus the profile) against `tput-safe`, after merging into throughput-stack. I did not define that arm here because tput-safe does not exist on this branch.
3. **Partly fixed (`49c776e`): the throughput-stack merge needs more than the listed resolution.** `_serving_launcher(*, serve, serve_env, profile)` is now keyword-only, with a test, so a positional call raises `TypeError` instead of being read as `serve`. The moe-routing conflict in `launch_sglang.py` is confirmed with `git merge-tree`. The rest is merge-time work, listed in the merge notes below.
4. **Fixed (`911fd8d`): C8's batch sweeps did not match C0's.** C8 now sweeps 1/4/7 like C0, and the humming bench is about 49 of its 75 minutes. A new test, `test_c8_measures_every_batch_size_c0_measures`, keeps C8's measured phases equal to C0's, so a merge that changes C0's sweeps fails until C8 follows. Running C8 inside `tput-gates` after the merge is noted in `arms.md`.
5. **Nits:**
   - **JIT-cache hash ignores the profile: rejected.** No arm or bench config sets `JIT_CACHE_DIR` with a non-default profile, so nothing would restore under Humming. A restore only adds files, and Humming compiles its own kernels as on a cold start. The nvfp4 and gsq profiles already hash differently because their checkpoints' `config.json` differs. Adding the profile to the snapshot would either change the hash of every config, which invalidates any published cache, or need a special case.
   - **`check_checkpoint` gaps: fixed (`e97038f`).** Checked against Pennyroyal d00d88ef. `MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]`; `gptq_marlin` refuses other sizes and auto-round falls back to the slower MoeWNA16 kernel. Humming reads `config["format"]` with no default. The checker now fails a Marlin launch with an unsupported or undeclared group size, and a Humming compressed-tensors config without `format`. Intel's real headers still pass ("group_size [128]: a Marlin size", 68.50 GiB, 1,011,264-token pool).
   - **`_check_context` crash: fixed (`94c5198`).** A non-integer `LOCAL_ANALYZER_CONTEXT_WINDOW` now stops the build with `SystemExit("env ... must be an integer, got '96*1024'")`. The harness also reads it with `int()`.

**Merge notes for integrate**
- **Into `arm/throughput-stack`:**
  - **`launch_sglang.py`:** put `CUDAGRAPH_BS_EXTRA=[]` and its assert in CFG, and add `*cfg["CUDAGRAPH_BS_EXTRA"]` to `graph_bs` in `build_args`.
  - **`build_notebook.py`:** `_arm_config` keeps `profile` and gains their `runtime_patches` and `gate`. The signature becomes `_serving_launcher(*, serve=None, serve_env=None, profile=None, arm_patches=None)`, with their patch-merge body. `build_notebook` passes `arm_patches=`. `arm_notebook` must pass the config's runtime patches so `launch_harness.record_arm`, which goes through it, records them in the `tput-*` goldens. `main` keeps `--replica` and `--gate-passed`.
  - **Tests:** in `test_build_serving.py:42-91`, `_serving_launcher(x)` becomes `_serving_launcher(arm_patches=x)`. In `test_throughput_arms.py:140`, the monkeypatch becomes `lambda **kw: _template_serving_cell()`.
  - **`bench.json`:** C0 takes throughput-stack's sweep1/4/7/8. Set C8's phases to match and add `sweep8` to the literal list in the new C8 test.
  - **Then:** run `launch_harness.py --write` and review the golden diff. Every golden's `cfg` gains `CUDAGRAPH_BS_EXTRA`.
- **Into `arm/moe-routing`:** in `launch_sglang.py`, put its two-line "MoE experts per token" print after this branch's `Target:` print in the validate section. In `arms.json`, keep both sides.

**Final git log against the base `3db0db4`**
```
4022071 Document the GSQ arms' pairing and the new checkpoint checks
e97038f check_checkpoint: check Marlin group sizes and Humming's format key
94c5198 Refuse a non-integer context window env with a clear message
49c776e Make the builder's _serving_launcher options keyword-only
911fd8d Run bench C8 at the batch sizes C0 measures
3f76fa9 Publish the server process before the startup wait
57ecdab Document model profiles, the checkpoint checker and the Humming bench
8e17f78 Add bench C8 (Humming vs Marlin) and the humming and baseline benches
eb69ec0 Add serving/tools/check_checkpoint.py: vet a checkpoint on the CPU
e2bb3d8 Add the qwen38fn-gsq profile slot, its arms, and Humming-safe AutoRound views
2ab41cd Add the qwen38fn-nvfp4 profile and the nvfp4 arm
b9bbf9a Builder: take inputs, kernel sources and the drafter from the arm's profile
37dadb6 Serve the model from a profile: serving/profiles/qwen38fn-w4a16.json
38bbfb5 Move the builder's launcher edits into the launcher and split it into functions
9519beb Record what every arm and bench config launches as goldens
e891a19 Builder: factor the arm and bench notebook builds out of main
```

**Tests** (niced, main repo venv, targeted files only)
- `test_launcher`, `test_launch_golden`, `test_build_serving`, `test_build_profiles`, `test_check_checkpoint`, `test_profiles`, `test_bench`, `test_launcher_jit_cache`: 160 passed in 9.41s.
- The goldens are unchanged.
- Local builds, none pushed: `--arm control` built; `--arm gsq` built and warned that its input is TBD, so `--push` is refused; `--bench humming` built at 596,527 bytes. The worktree is clean at `4022071`.

**PR summary**
```
arm/model-profiles: serve any checkpoint from a per-model profile (serving/profiles/<name>.json).
- Launcher split into functions; builder's code splices moved into it. Goldens pin argv/env/patches/CFG:
  control, panel-control, coldstart-save and every bench config launch byte-identically.
- Profiles: qwen38fn-w4a16 (default), qwen38fn-nvfp4 (arm nvfp4, matches arm/nvfp4's launch),
  qwen38fn-gsq (arms gsq-panel vs panel-control, gsq vs control; model input TBD, push refused),
  qwen38fn-w4a16-humming (bench C8; a config view skips AutoRound's BF16 layers for Humming).
- serving/tools/check_checkpoint.py: CPU-only check of profile match, Marlin group sizes, Humming keys,
  experts-only quantization, tokenizer/template SHAs and KV pool (matches the W4A16's measured 1,011,264).
- Benches: humming (C0 Marlin vs C8 Humming at bs 1/4/7/10), baseline (C0 with --profile).
- Review fixes: bench kills a server that dies while starting; keyword-only _serving_launcher; clear
  error on a non-integer context window env.
- GLM-5.3-Flash and the published GSQ GGUF are logged as not servable (arms.md "Not pursuing").
- Not run on GPU (no quota). Order: --bench humming; check the GSQ export; baseline --profile gsq; gsq-panel; gsq.
- Tests: kaggle/tests targeted files 160 passed; docs: CUSTOM.md §6, profiles/README, kaggle/README, README.
```

The worktree is `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/model-profiles`, and `arms.md` is updated at `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/arms.md` (model-profiles entry under In progress).
