# wf_57f4538b-24c: integrate

Workflow run `wf_57f4538b-24c`, phase Integrate. 

---

Everything up to the Kaggle pushes is done; nothing was pushed to Kaggle. Tests and notebook checks pass, the branch is pushed, PR #70 is open, and `arms.md` and the fork README are updated. Nothing has run on a GPU yet, and the LUT kernel has still never been compiled.

**Test results** (all niced, on `arm/runner-gsq` at `9562eb7`):
- **`kaggle/tests`:** 290 passed in 26.65s, with torch, the real F001 source and the real pennyroyal-v253 wheel. Without torch, 2 test files skip, giving 272 passed and 6 skipped.
- **Converter tests:** 42 passed and 1 skipped (the HF-download test) in 2.29s, with `LLAMA_CPP_DIR` pointing at the local llama.cpp b11476 copy. Without it, 36 passed and 7 skipped.
- **`pin.py --check`:** all gsq patch pins are current.
- **nbcheck:** passes on `v4-fable-f001-gsq.ipynb`, `gsq-coder-convert.ipynb` and `gsq-engine-build.ipynb`.
- **Rebuilds:** all three notebooks rebuild byte-identical from `9562eb7`. The test notebook is 905,009 bytes and its metadata also matches.
- **Kernel metadata:** `mellamoa/v4-fable-f001-gsq` is on `NvidiaRtxPro6000` with internet on, which it needs to download the weights at run time. Apart from the id, title and file name, the internet flag is the only difference from F001. The conversion and engine notebooks also have internet on, on the same machine.

**Git and PR:**
- `arm/runner-gsq` is pushed to origin (28 commits).
- **PR:** https://github.com/adsdemaybe/daniel-franzen-sol/pull/70, based on `arm/model-profiles`. The body quotes your requests and covers the competitor optimizations adopted, deferred and rejected, the conversion error, the GLM verdict with numbers, tests, and the test plan against F001 v2 (29.93, 32 levels at matched tokens). It is marked "Not run yet".
- **README:** `docs/fork-readme` got commit `58ce1c2`. It adds a tooling row and a `gsq-coder` / `gsq-coder-panel` arm row, and corrects the old `gsq` row, which said the GSQ Coder GGUF can't be served.

**`arms.md`** (backup at `<scratchpad>/arms.md.bak-runner`; only lines added, plus the PR number in the runner-gsq heading):
- The runner-gsq entry now shows PR #70, the competitor optimizations and the integration results.
- A new In-progress entry, "GSQ Fable test", covers the hypothesis, what cell 14 does, the baseline and the push commands.
- The GLM entry under Serving has the re-check numbers: 175.6 GB checkpoint against 95.6 GiB of VRAM, offload at 17.6–30.6 tok/s, and turboderp's 2.05 bpw at 85.1 GB with KL 0.19 and an estimated 150–400 tok/s at 10 streams.
- The GSQ "as published" entry notes that runner-gsq now converts the model itself, so the friend's re-export is no longer needed.

**Commands to push the notebooks** (each needs your approval):

1. **F001 test on the GSQ runner (run this first).** Expect about 60 GPU minutes, with a hard cap of about 90. F001 v2 took 37 minutes, and this adds the download, conversion and self-test stages, each with a hard time budget.
   ```
   kaggle kernels push -p /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/push-fable-f001-gsq
   kaggle kernels output mellamoa/v4-fable-f001-gsq -p /Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/runs/kaggle/v4-fable-f001-gsq/v1
   ```

2. **Conversion notebook (only if the test holds up).** The copy built now converts and checks only; it uploads nothing. To publish, rebuild with `--publish` and add the `KAGGLE_USERNAME` and `KAGGLE_KEY` secrets in the notebook's Kaggle UI. Expect about 15–30 GPU minutes without publishing. With publishing, about 25–70: the 45.6 GB upload time is an unmeasured guess.
   ```
   cd /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/runner-gsq
   python kaggle/build_bundle_notebooks.py --convert --publish
   kaggle kernels push -p kaggle/out/gsq-coder-convert
   ```
   Then set `inputs.model.source` in the profile to `mellamoa/qwen3.8-flash-next-gsq-rco-coder/Transformers/lut4/1`.

3. **Engine notebook.** Expect about 5–10 GPU minutes; it only repackages the wheel.
   ```
   kaggle kernels push -p /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/runner-gsq/kaggle/out/gsq-engine-build
   kaggle kernels output mellamoa/gsq-engine-build -p /tmp/gsq-engine
   kaggle datasets create -p /tmp/gsq-engine/gsq-engine/bundle-gsq-overlay
   ```
   Then set `inputs.engine` in the profile.

The conversion and engine notebooks don't use the GPU but still run on the RTX Pro 6000 machine type, so they count against GPU quota.

Files:
- PR body: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/integrate/pr_runner_gsq.md`
- `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/arms.md`
