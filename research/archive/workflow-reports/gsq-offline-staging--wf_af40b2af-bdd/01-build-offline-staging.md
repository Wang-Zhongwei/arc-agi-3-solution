# gsq-offline-staging: build:offline-staging

Workflow run `wf_af40b2af-bdd`, phase Build. Make the F001-on-GSQ test run offline: CPU staging notebooks that fetch the Coder GGUF shard 1 in parts as notebook outputs, and an offline test notebook that mounts them; review and fix

---

The offline F001 x GSQ test is built and committed, and nothing was pushed to Kaggle or to git. The generated notebooks pass `nbcheck.py` and are under 1 MB. The staging download code also worked against the real HF file on a 1 MiB read.

**What changed**
- **Staging notebooks:** the shard is split into 3 parts (part 1: 9,869,197,312 bytes, part 2 the same, part 3: 9,870,051,872). Each CPU notebook checks the HF API (it returned the pinned sha256 `e11083ba…` and size 29,608,446,496). It then downloads its byte range with 16 parallel Range requests, refusing any answer that is not a 206 for exactly that range. Requests are retried, the download resumes within a run, and the budget is 3 hours. Only `<file>.partK` and `manifest.json` end up in `/kaggle/working`; a failed run deletes its partial file.
- **Offline test notebook:** it finds each part under `/kaggle/input/notebooks/<owner>/<slug>` or `/kaggle/input/<slug>`, then by name anywhere under `/kaggle/input`. It checks each manifest and file size, and stops in seconds naming any missing part. Next it runs the plan check on part 1's GGUF header and the patch check. Then it reassembles the shard into `/tmp/gsq/shard1.gguf`, checking each part's sha256 and the whole file's in one read (15-minute budget, replacing the download). Convert, self-test, serve and play are unchanged.
- **Test notebook metadata:** GPU NvidiaRtxPro6000, internet off, the competition, both datasets and both models, plus `kernel_sources` = `mellamoa/gsq-coder-stage-p1..p3`.
- **Unchanged from the F001 source:** only cells 4 and 14 differ. Cells 2 (harness patch), 6 (Fable layer) and 18 (panel, 10 streams) are byte-identical, and so is the notebook-level metadata.

**Commits on `arm/runner-gsq`** (6 ahead of origin, not pushed, so PR #70 doesn't show them yet):
- `845491f` Add stage_gsq: download one byte range of the GGUF shard with a manifest
- `2208fcf` Add stage_gsq.locate and reassemble for the offline test notebook
- `e041ed9` Add the staging notebook generator for GGUF shard 1
- `fefde27` stage_gsq.locate: resolve the input root at call time
- `86b28f6` build_fable_gsq: reassemble shard 1 from the staging notebooks, internet off
- `91a565f` Document the staged GGUF and the offline F001 x GSQ test (`serving/CUSTOM.md` section 7, `kaggle/README.md`, the bundle README, the top-level README)

**Tests** (run one file at a time, niced; every `kaggle/tests` file passes):
- **`kaggle/tests/test_stage_gsq.py`** (43 passed, 1 opt-in):
  - The split covers the file with no gaps or overlaps, for 1–5 parts and several sizes, including the real one.
  - Downloads work through injected 503s, resume from the state file, and stop at the deadline. A server that ignores Range or reports the wrong file size is refused.
  - The manifest format is exact, the HF API check and its fallback to the pinned values work, an oversized part is refused, and a failed run leaves no output.
  - Parts are found in both mount layouts, mixed layouts and a deeper fallback location; a missing part fails fast and is named; a short part or a different split is rejected.
  - Parts given in any order reassemble correctly; a corrupt part, a wrong whole-file sha256, a gap, a missed deadline and a full disk are each caught.
  - `GSQ_HTTP_TESTS=1` runs the real HF read; it passed.
- **`kaggle/tests/test_build_gsq_stage.py`** (6 passed): staging metadata has no GPU, no machine shape or image, internet on and nothing attached. The built cells run against a stand-in HF and leave only the part and manifest. The `--check-manifests` option is covered.
- **`kaggle/tests/test_build_fable_gsq.py`** (12 passed with `FABLE_F001_NOTEBOOK` set): stage order; metadata with internet off, the competition, datasets, models and `kernel_sources`; metadata missing the GPU, competition or inputs is refused. One end-to-end run uses real staged parts in both mount layouts and checks the converter reads the exact reassembled bytes.
- **Real HF check:** the code embedded in the generated p2 notebook read 1 MiB at the start of part 2 (offset 9,869,197,312), byte-equal to a plain request.

**Push order** (`S=/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad`):
1. `kaggle kernels push -p $S/push-gsq-stage-p1`, then the same for `push-gsq-stage-p2` and `push-gsq-stage-p3`. They can run at the same time.
2. `kaggle kernels status mellamoa/gsq-coder-stage-p1` (and p2, p3) until each shows COMPLETE. Optionally, `kaggle kernels files mellamoa/gsq-coder-stage-pK` lists the part and its size.
3. For K = 1, 2, 3: `kaggle kernels output mellamoa/gsq-coder-stage-pK -p $S/gsq-stage-manifests/pK --file-pattern 'manifest\.json$'`
4. From the worktree: `python kaggle/build_gsq_stage.py --check-manifests $S/gsq-stage-manifests --parts 3`
5. `kaggle kernels push -p $S/push-fable-f001-gsq` — only after all three staging runs are COMPLETE, because an attached notebook mounts its latest finished output.
6. `kaggle kernels output mellamoa/v4-fable-f001-gsq -p runs/kaggle/v4-fable-f001-gsq/v1`

**Expected time per staging notebook:** about 5–12 minutes. That is 1–2 min to start, 1–5 min to download 9.87 GB (5 min at 33 MB/s), about 0.5 min to hash, and 2–5 min for Kaggle to save the output. The download is capped at 3 hours. In the test notebook, reassembly should take 1–4 minutes if `/kaggle/input` reads at 150–500 MB/s.

**Risks:**
- **Mount read speed:** the 15-minute reassembly budget needs at least 33 MB/s from the mounted outputs; I haven't measured that.
- **Output size:** each output is about 9.9 GB against the ~19.5 GB cap. Saving and mounting outputs that large is untested. Two parts (14.80 GB each) would also fit, but with less margin.
- **Wrong bytes from HF:** a part's sha256 only proves Kaggle stored what the staging notebook downloaded. Only the whole-file check in the GPU run proves the bytes match HF, and a failure there costs about 2–4 GPU minutes. The plan check confirms part 1's GGUF header before reassembly.
- **Disk on `/tmp`:** the GPU machine needs the same space as before, 29.6 GB for the shard plus the conversion output. Free space for the shard is checked up front.
- **HF rate limits (429s):** handled by retries and extra rounds within the 3-hour budget.
- **Test-order bug (pre-existing):** running `test_build_fable_gsq.py` and `test_acquire_gsq.py` in one pytest process fails the miniature-model test with "No module named gsq_coder.tests". It also fails on the commit before this work, and each file passes when run alone.

Files are in `$S/wt/runner-gsq`:
- `serving/kaggle/stage_gsq.py`
- `kaggle/build_gsq_stage.py`
- `kaggle/build_fable_gsq.py`
- `kaggle/tests/test_stage_gsq.py`
- `kaggle/tests/test_build_gsq_stage.py`
- `kaggle/tests/test_build_fable_gsq.py`

Generated folders in `$S`:
- `push-gsq-stage-p1`
- `push-gsq-stage-p2`
- `push-gsq-stage-p3`
- `push-fable-f001-gsq` (916,755 bytes)
