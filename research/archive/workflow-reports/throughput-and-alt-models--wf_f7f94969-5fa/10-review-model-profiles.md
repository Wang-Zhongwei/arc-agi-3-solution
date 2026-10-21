# throughput-and-alt-models: review:model-profiles

Workflow run `wf_f7f94969-5fa`, phase Review. Win 2+ more games via throughput (sim-v2 sweep + a throughput stack of built-but-unrun fixes) and serve an alternative model (GLM-5.3-Flash EXL3, ISTA Qwen3.8-Flash-Next GSQ-RCO GGUF) by refactoring the RTX Pro 6000 SGLang launcher into per-model profiles

---

I found no blocking defects in `arm/model-profiles`: the default launch is unchanged, the flags are valid for SGLang 2.5.3, everything runs offline and all notebooks build. I did find one real regression in the bench path, one run-order mistake and three smaller issues, ranked below with fixes.

## What I verified (no defects)

- **Default behaviour unchanged.**
  - The goldens were recorded at 9519beb, before the refactor, and are byte-identical in 38bbfb5, 37dadb6 and b9bbf9a. `launch_harness.py` changed only additively, and the test requires the same keys.
  - I also checked it independently. I built `control`, `panel-control`, `coldstart-save` and `--bench smoke` at the base (3db0db4) and at HEAD (57ecdab) in temporary worktrees, since removed.
    - Only the serving cell differs, and `kernel-metadata.json` is identical.
    - I ran the base notebooks' serving cells through the harness: argv, env, runtime patches, CFG and files all equal the `control`, `coldstart-save`, `panel-control` and `bench-C0` goldens.
    - The CFG keys and their order are unchanged, so the JIT cache config hash is preserved.
- **Tests.** `kaggle/tests` under nice: 187 passed.
- **Launch args are valid for Pennyroyal 2.5.3.** I read the source at d00d88ef (`competitors/sglang-rtxpro6000`).
  - `humming`, `gptq_marlin`, `compressed-tensors`, `modelopt_fp4` and `unquant` are all accepted values (`server_args.py:152-188`).
  - `--json-model-override-args`, `--speculative-draft-model-quantization`, `--gdn-mtp-cache-mode` and `--ple-offload-embedding` all exist as options.
  - `--quantization humming` on an auto-round checkpoint passes the quantization check: `HummingConfig.override_quantization_method` returns "humming", and no earlier override fires.
  - Humming accepts `--moe-runner-backend auto` (`humming.py:1236`).
  - `humming-kernels[cu13]==0.1.12` is a declared Pennyroyal dependency in `pyproject.toml:38`, so the offline install includes it.
- **The Humming view is correct.** On Intel's real config it reduces to 12 `-:` patterns. Against Intel's real safetensors headers, every BF16 2-D weight is covered except `lm_head`, which Humming skips anyway. The routed experts are the only INT tensors.
- **NVFP4.** The profile reproduces fee2a6c's flags exactly (same flags, different order).
- **FP8-PLE checkpoint.** The albucino FP8-PLE checkpoint still passes the new bit check: its FP8 is set in `text_config`, not `extra_config`.
- **Memory math.**
  - `check_checkpoint` on the real Intel headers gives 68.50 GiB of weights and a 1,011,264-token pool, matching the measured run.
  - The per-token KV cost of (12 + 1) layers × 2 × 2 heads × 256 = 13,312 B is correct.
  - The NVFP4 and INT3 adjustments are reasonable.
- **Builds.**
  - Arms `nvfp4`, `gsq-panel` and `gsq` build at about 338 KB.
  - Benches `humming`, `levers` and `baseline` (with `--profile` gsq, nvfp4 and w4a16-humming) build at about 596 KB.
  - TBD pushes are refused before any Kaggle call.
- **Throughput levers.** gate11 (11 harness streams over MAXREQ 10), CUDA-graph sizes up to 10 with the `[3,5,6]` extras, and MAMBA_CACHE 60 don't conflict. The new CTX = harness window + 8Ki check passes every throughput-stack arm.
- **Docs and arms.md.** CUSTOM.md §6, `kaggle/README`, `profiles/README`, `serving/README` and the top-level README are updated. arms.md has the In progress and Not pursuing entries.

## Defects, most severe first

1. **Medium: a bench can no longer clean up a server that dies during startup.**
   - `serving/kaggle/launch_sglang.py:528` keeps `proc` local to `launch()`; the module-level `proc` is only set at line 721, after `launch()` returns. If the server exits during startup, `launch()` raises (line 575) and `ns["proc"]` is never set. `bench.run_bench`'s `finally: stop_server(ns.get("proc"))` (`bench.py:1194`) then cannot `killpg` leftover SGLang workers that still hold GPU memory and the port.
   - The bench docstring at `bench.py:1066-1067` promises this cleanup.
   - Reproduced with the harness and a dead server: base `ns has proc: True`, HEAD `ns has proc: False`. This is most likely on C8, the first Humming launch ever.
   - **Fix:** publish `proc` before waiting, e.g. `globals()["proc"] = proc` right after `Popen`, or split `launch()` into a module-level start and a wait. Add a bench test that a failed start leaves `ns["proc"]` set.

2. **Medium: the GSQ run order mixes model and harness.**
   - arms.md:146 says "`gsq` against the best throughput arm". The `gsq` and `gsq-panel` arms in `kaggle/arms.json:22-30` run this branch's control harness (fix-6), not stack-v2 or throughput-stack.
   - That pairing changes both the model and the harness at once. Also, if a throughput-stack batch is open, the builder refuses the build unless `--new-batch`, which closes that batch.
   - **Fix:** either compare `gsq` only with `control`, and `gsq-panel` with `panel-control`, on this branch; or, after merging into throughput-stack, define `gsq-tput-safe` with tput-safe's env plus `"profile": "qwen38fn-gsq"` and pair it with `tput-safe`. Update arms.md:146 to match.

3. **Low-medium: the throughput-stack merge needs more than the listed resolution.**
   - Beyond the textual conflicts, throughput-stack's tests call `_serving_launcher(arm_patches)` positionally (`test_build_serving.py:42-91`) and monkeypatch it as `lambda arm_patches=None` (`test_throughput_arms.py:140`).
   - After the merge, `_serving_launcher(serve, serve_env, profile)` (`build_notebook.py:256`) would take the patches dict as `serve`. That fails loudly, but `git merge-tree` doesn't show it and the implementer's resolution notes don't cover it.
   - **Fix:** make the parameters keyword-only (`def _serving_launcher(*, serve=None, serve_env=None, profile=None, arm_patches=None)`) and update those two test files. `launch_harness.record_arm` must also pass the arm's `runtime_patches` so the `tput-*` goldens include them.
   - The merge into `arm/moe-routing` also conflicts in `launch_sglang.py`, which the report doesn't list.

4. **Low: C8's batch sweeps don't match C0's.**
   - In `kaggle/bench.json:75`, C8 runs sweep1, sweep4 and sweep8. C0 on this branch runs sweep1, sweep4 and sweep7, so the humming bench has no Marlin number at batch 8 and no Humming number at batch 7.
   - **Fix:** give C8 the same sweeps as C0 (add sweep7 now; after the merge, use throughput-stack's sweep1/4/7/8).
   - Efficiency: running `humming` (C0, C8) and `tput-gates` (C0, C3, C1) as separate sessions repeats C0, about 26 minutes plus notebook startup. After the merge, one combined C0/C3/C1/C8 bench would save that GPU quota.

5. **Nits.**
   - The JIT-cache hash ignores the profile, so a Humming launch and a Marlin launch of the same checkpoint share a hash. This is harmless, but a restore would be logged as a match with no Humming kernels in it.
   - `check_checkpoint` doesn't check `group_size` (Marlin supports −1, 32, 64 and 128) or that a compressed-tensors config has the top-level `format` key Humming reads.
   - An env `LOCAL_ANALYZER_CONTEXT_WINDOW` given as an expression string makes `_check_context` crash with `ValueError` instead of a clear `SystemExit`.

The reviewed worktree is clean at 57ecdab: /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/model-profiles

Supporting files are in /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/mp-review/ (Pennyroyal `humming.py`, `model_config.py`, `qwen4_exp.py`, and albucino's FP8-PLE config).
