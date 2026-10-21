# fix-definitely-issues: review:#6

Workflow run `wf_800f9fbb-745`, phase Review. Fix DEFINITELY issues #5, #4, #3, #6 one at a time: implement in a worktree, Sonnet review, fix

---

VERDICT: 2 defects need fixing (the `coldstart-overlap` arm cannot work, and `autotune-off` is probably a no-op), plus 2 minor ones. The cache save/restore code and the defaults-unchanged claim hold up.

SGLang source references are to the pinned commit `d00d88ef` in `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/sglang-rtxpro6000`. Its `server_args.py:3363` is the same line the issue cites.

**Verified, no defect**
- **Tests:** `python -m pytest -q ../kaggle/tests` gives 54 passed. The spec's `tests` path fails because `ARC3-Inference/tests` does not exist. The implementer's reason holds: files under ARC3-Inference go into the harness patch and would change the code hash.
- **Control and panel-control:** I built both at `arm/custom-serving` and at HEAD and diffed every cell. Only the serving cell differs, with 445 added lines and 0 removed. Metadata and the other notebook keys are identical, and the harness patch hash is the same. The notebook grows from 301,878 to 322,519 bytes, under the 950,000 limit.
- **Arms:** all five arms build and the serving cells compile. `coldstart-save` overrides `JIT_CACHE_SAVE`, `coldstart-overlap` overrides `STARTUP_WEIGHT_LOAD_MODE`, and `autotune-off` overrides `AUTOTUNE`. No extra dataset sources appear for any arm.
- **`/health` timing:** the launcher's claim is correct. `/health` returns 503 while the server status is `Starting`, and the status becomes `Up` only after the warmup request (`http_server.py:674`, `2319`). Saving at `/health` is therefore saving after warmup.
- **sglang/jit restore:** SGLang's JIT cache is content-addressed (`kernels/jit/utils/compile/cache.py`), so restoring it does not depend on file times.

**Defects, most severe first**

1. **`coldstart-overlap` cannot work, and the server will die at model load.**
   - Where: `kaggle/arms.json:19-24`, `serving/CUSTOM.md:29` and `:85`, `serving/kaggle/launch_sglang.py:370`.
   - Why: `StartupWeightLoadManager._get_unsupported_reason` (`startup_weight_load.py`) rejects this launch config on several independent rules, so it raises `ValueError: --startup-weight-load-mode=overlap is not supported: ...`.
     - **Speculative decoding** (line 360): NEXTN is on.
     - **Page-cache dropping** (line 394): `DROP_CACHE_AFTER_LOAD` defaults to True.
     - **Quantization** (line 411): auto-round.
     - **Multimodal** (line 416): the model takes images.
     - **Architecture allowlist** (lines 38-43): Llama, Qwen2 and Qwen3 only, so Qwen3-Next is not allowed.
     - **Draft worker:** draft workers are also rejected.
   - The call site (`load_model_utils.py:318-325`) does not catch the error and there is no fallback to serial. The builder turns the server exit into a RuntimeError, so the arm burns a GPU session and produces no games.
   - The docs say "Unverified with AutoRound, the NEXTN draft...". The source shows a hard reject, and it also kills issue #6 proposal item 2.
   - Fix:
     - Delete the arm from `arms.json`.
     - Rewrite `CUSTOM.md:29` and `:85` to say overlap is rejected on this stack and list the reasons above.
     - Optionally replace the assert at `launch_sglang.py:169` with one that rejects `"overlap"` with that reason list, so it fails in seconds instead of mid-launch.
     - Note on issue #6 that hiding the capture behind the weight read must come from the cache restore instead.

2. **`autotune-off` is very likely a no-op, so the arm and its docs are misleading.**
   - Where: `kaggle/arms.json:25-30`, `serving/CUSTOM.md:24` and `:84`.
   - Why: FlashInfer autotune only runs if `should_run_flashinfer_autotune` (`runner/flashinfer_autotune.py:53-126`) returns True. That needs the MoE runner to be one of the `flashinfer_*` backends (line 88), or a modelopt fp4/fp8 model (the checks before line 122). Otherwise it returns False at lines 122-123.
   - This launch uses `--moe-runner-backend auto` with `--quantization auto-round`. `arg_groups/overrides.py:2775-2834` leaves `auto` unchanged for that quantization. So startup should be identical with `AUTOTUNE` on or off.
   - The ~125 s silent period is then not autotune, and the "autotune picks the kernels, compare tok/s" note in `CUSTOM.md:84` is wrong.
   - I found no serve.log locally to confirm, so this rests on the source. The cheap check is to grep an existing serve.log for "Running FlashInfer autotune".
   - Fix: drop the arm unless that grep shows autotune running, and correct `CUSTOM.md:24` and `:84`. If autotune does run, its cache is written to `cache/sglang/flashinfer/autotune/` (default `SGLANG_FLASHINFER_AUTOTUNE_CACHE` is True). That path is not in `CACHE_SUBDIRS`, so it would also need adding.

3. **A failed save leaves a partial tar in the kernel output (minor).**
   - Where: `serving/kaggle/jit_cache.py:200-208`.
   - Why: any exception inside `tarfile.open(tmp, "w")` leaves `sgl-jit-cache.tar.partial` in `/kaggle/working`. I reproduced this by making `tar.add` raise on the second file, which left a `.partial` file behind.
   - One cache file vanishing between the file listing and the tar (for example a build finishing) also aborts the whole snapshot, and there is no retry. `.staging-*` build directories from SGLang's JIT cache are not skipped.
   - Fix: wrap the write in try/except and unlink `tmp` on failure. Skip files that raise `FileNotFoundError`. Add `.staging-` to `_skipped`.

4. **The publish steps would upload the kernel log too (minor).**
   - Where: `serving/CUSTOM.md:112`.
   - Why: `fetch_version_output.py` also writes `<slug>.log` into the output directory, so `kaggle datasets create -p /tmp/sgl-jit-cache` publishes `v4-coldstart-save.log` into the cache dataset.
   - Fix: add a step to delete that log, or fetch into a subfolder and publish only the tar and manifest.

**Not defects, still unproven on Kaggle (the implementer already lists these)**
- Whether FlashInfer's ninja build treats the restored objects as up to date.
- Whether Triton reruns its benchmarks on cached kernels.
- Whether the snapshot is complete. The `not in the snapshot:` log lines cover only directories under `PREFIX/cache`.

Worktree untouched: `git status` is clean at `259feb9`.
