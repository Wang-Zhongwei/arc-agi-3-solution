# fix-definitely-issues: implement:#6

Workflow run `wf_800f9fbb-745`, phase Implement. Fix DEFINITELY issues #5, #4, #3, #6 one at a time: implement in a worktree, Sonnet review, fix

---

Issue #6 is fixed on branch `arm/fix-6-cold-start`; nothing has run on Kaggle yet. The control notebook still behaves the same, all 54 tests pass, and nothing was pushed. Whether the restored cache actually removes the ~145 s compile is the main thing still unproven.

**Worktree:** /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/fix-6

**git log --oneline arm/custom-serving..HEAD**
```
259feb9 Test the launcher's JIT cache blocks with SGLang stubbed (#6)
f580018 Tie the JIT cache to the target model's config (#6)
59577cf Document the cold-start flags and how to publish the JIT cache (#6)
9d867c9 Add coldstart-save, coldstart-overlap and autotune-off arms (#6)
a16d8ed Let an arm mount extra Kaggle datasets (#6)
3efc744 Save and restore the JIT caches behind CFG flags (#6)
a3c89d8 Add CFG STARTUP_WEIGHT_LOAD_MODE to the launcher (#6)
081cb6f Embed the JIT cache helpers in the serving cell (#6)
440fb63 Add JIT cache snapshot and restore helpers for the launcher (#6)
```

**pytest:** `54 passed in 0.19s`, run from ARC3-Inference as `-m pytest -q ../kaggle/tests`. The import check resolves to the worktree (`.../wt/fix-6/ARC3-Inference/inference/__init__.py`).
- The tests are in `kaggle/tests/`, not `ARC3-Inference/tests`. Anything under ARC3-Inference goes into the harness patch every notebook carries, so it would have changed the control and forced a new batch. The `arm/serving-bench` branch already keeps its tests in `kaggle/tests/`.
- This branch has no `ARC3-Inference/tests`, so the command as written fails with "no such directory".
- The flags are documented in `serving/CUSTOM.md`, not `ARC3-Inference/CONFIGURATION.md`, which explicitly excludes serving settings.

**What I added**
- **The cache module** is `serving/kaggle/jit_cache.py`, which the builder embeds in the serving cell.
  - It tars the triton, flashinfer, sglang/jit, torchinductor and cuda caches plus `~/.tilelang`, keeping file times and skipping lock files and Triton's half-written entries.
  - The manifest records the GPU, CUDA, Python, torch, SGLang and kernel-package versions. It also hashes the serving config, the kernel-related env vars and the target model's `config.json`.
  - The cache is restored only if the manifest matches. Otherwise the log says why and the launch continues cold.
  - Kaggle may unpack a `.tar` on upload, so the restore also accepts the unpacked folder.
- **The arms**:
  - `coldstart-save`: saves the cache, 5-minute panel.
  - `coldstart-overlap`: overlap weight loading, 20-minute panel.
  - `autotune-off`: no FlashInfer autotune, 20-minute panel.
- **`"datasets"`**: an arm can now list extra datasets to mount. Arms without the key get the same metadata as before.
- **A consumer arm that loads the cache is not added**, because the dataset doesn't exist yet. `serving/CUSTOM.md` has the arm snippet and the publish steps.

**What changes for the model:** nothing with the defaults.
- I dry-built `control` and `panel-control`. Only the serving cell differs, and every difference is an added line:
  - three new settings, off by default;
  - the embedded helper module;
  - one check;
  - blocks that only run when a flag is set.
- Launch arguments, server env, kernel metadata and the harness patch are unchanged. The code hash is unchanged because nothing under ARC3-Inference changed.
- Every arm's fingerprint changes because the serving cell text changed.
- New `runs.json` records will carry `"datasets": []`.

**The flagged arms**
- `coldstart-overlap` passes `--startup-weight-load-mode overlap`. It hasn't been tested with AutoRound, the NEXTN draft model or `--ple-offload-embedding`, and Franzen's prefetch patch edits the same loader code.
- `autotune-off` passes `--disable-flashinfer-autotune`, which may lower decode speed.
- `coldstart-save` adds a probe of the runtime versions (a few seconds before launch). It then writes the tar in the foreground once the server is ready, which delays the games by however long that takes.

**Still needs a Kaggle run**
1. **Does a restored cache work in a fresh container?** Run `coldstart-save`, publish the cache, then run a consumer arm. Success is `cuda_graph target_verify` in serve.log falling from ~145 s to a few seconds.
   - FlashInfer's and SGLang's compiled modules may be rebuilt anyway. Their build tool compares file times with the freshly installed package sources.
   - Triton may rerun its autotune benchmarks even on cached kernels.
2. **Is the cache complete?** Check the save log's `not in the snapshot:` lines for other cache folders that hold compiled kernels.
3. **Is overlap loading correct and faster?** Check `coldstart-overlap`'s serve.log for errors, plus its accept length and tok/s against `panel-control`.
4. **How much of the silent ~125 s is autotune, and what does turning it off cost?** Compare `autotune-off`'s startup timings and tok/s with `panel-control`.
