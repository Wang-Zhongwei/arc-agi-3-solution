# wf_57f4538b-24c: fix:runner-gsq

Workflow run `wf_57f4538b-24c`, phase Fix. 

---

I fixed all six findings in 11 commits on `arm/runner-gsq`. Two of them (1a and 5) can't happen on today's Kaggle inputs, which I checked directly; I added cheap guards for them anyway. Nothing was pushed to git or Kaggle. HEAD is `9562eb7`, 28 commits on top of `eb7f340`, and the worktree is clean. The LUT kernel still has never been compiled or run on a GPU.

## Per-finding verdicts

**1a. Patch hashes only checked after download (fixed as a guard; not failing today).**
- I downloaded the real `dfranzen/pennyroyal-v253` SGLang wheel (`sglang-0.5.19+gd00d88efc8d6`, 24.8 MB) and hashed the 8 files the patches edit. All 8 match the pinned "before" hashes, so the mismatch can't happen with today's wheel.
- The notebook mounts the dataset's latest version, so a future update could still break it. New `acquire_gsq.patch_check()` applies gsq/01-03 to copies of those files from the wheel, right after `plan_check` and before the download. It takes seconds.
- A test runs it against the real wheel when `PENNYROYAL_WHEELHOUSE` is set; it passes.

**1b. Chat template check (fixed; not failing today).**
- dfranzen's Kaggle copy of `chat_template.jinja` is byte-identical to Intel's (`c3cf9e34…`, fetched by a single-file download).
- `check_gsq` now compares the converted checkpoint's template with the `--base` template it was copied from. Without `--base` it still compares with Intel's. A non-Intel base template is now only the general warning.

**2. Stage budgets weren't hard limits (fixed).**
- `_run` now has a watchdog that kills the child's whole process group at the deadline, so a silent converter or `check_checkpoint` stops on time.
- The download raises at its deadline instead of waiting out retries, and workers stop retrying once it has passed.
- The smoke subprocess has a new `smoke: 10 min` budget.
- The self-test is time-boxed: 15 min at most, and in the F001 test it leaves 10 of the 25 launch minutes for the weight load and graph capture.
- The new tests fail on the old code: the old download overran a 2-second deadline by about 4.5 minutes.

**3. Holes in the int8 fallback (fixed).**
- `verify_gsq_engine.py` now writes its report before the first case and after every case, marked incomplete with the case in progress until it finishes.
- The serve glue runs the self-test in its own process group. If a LUT case crashes or hangs after the stock cases passed, the server runs `int8` on the same sources. Any other failure reverts gsq/03 and gsq/02 and reruns the stock cases. The chosen mode is written back to `verify_gsq_engine.json` as `served_mode`.
- Revert now works without `.gsq-orig` copies (the overlay case): it undoes each edit, checks the result against the shipped hash, and writes nothing unless every file can be restored.
- I did not split the stock and LUT cases into separate processes. Writing the report after each case covers the same failures.

**4. Smoke test too strict (fixed).**
- Each check now has a status. "fail" means the server didn't serve: an HTTP error or timeout, an empty answer, a response that doesn't parse, or no call under `tool_choice: required`. "warn" means a wrong answer.
- The games stop only on a "fail", or when text, vision and tool are all wrong at once (the server is producing noise).
- The `AttributeError` on a string `detail` is fixed, and a missing `smoke_gsq.json` is handled.

**5. Base layout unchecked (fixed as a guard; not triggered on Kaggle).**
- I built a headers-only copy of dfranzen's Kaggle base by range requests (38 shard files) and ran `--plan-only` on it. It copies 10.67 GB in 3 files and links all 22 PLE shards, the same as the HF layout, so the 45.6 GB and 5–8 minute estimates hold.
- `plan_check` now stops before the download if a plan would copy more than 16 GB.

**6. `PUBLISH` needed a hand edit (fixed).** `build_bundle_notebooks.py --convert --publish` now builds the notebook with `PUBLISH = 1`. The default still converts and checks only.

**Precache doc mismatch.** The repo docs never claimed a precache thread during conversion. I added a sentence to CUSTOM.md saying precache warms the linked PLE shards at launch and nothing warms them during conversion. I also removed the incorrect "38-shard layout" risk from the docs.

## Final git log (new commits)
```
9562eb7 Document the review fixes and the measured Kaggle base and wheel
c84a628 build_bundle_notebooks: add --publish instead of a hand-edited PUBLISH
cdc3467 plan_check: refuse a base whose copy would exceed 16 GB
1b6cf53 check_checkpoint: compare a gsq checkpoint's template with its base's
8338e7e Check the gsq runtime patches against the wheelhouse before the download
14a1e7c Bound the F001 cell's smoke stage and read its report defensively
c236cba smoke_gsq: stop the games only when the server does not serve
4d5c88f Time-box the gsq self-test and serve int8 when it dies after the stock cases
6012b2d verify_gsq_engine: write the report after every case
745d530 Revert gsq patches by undoing their edits when no .gsq-orig copy exists
de8e38d Make the acquire stages' budgets hard limits
76c3cf0 Document the GSQ-RCO Coder runner: build, publish and run   (previous HEAD)
```

## Tests (all run with `nice`)
- `kaggle/tests` with the real F001 source and the real wheel: `290 passed in 27.08s`.
- Converter tests in `serving/convert`: `36 passed, 7 skipped in 2.24s`.
- `pin.py --check`: `all gsq patch pins are current`.
- pyflakes is clean on every changed `.py` file, and every changed JSON file parses.
- nbcheck: the F001 test, conversion and engine notebooks all check, exit 0.
- Byte-identical builds against `76c3cf0`: `coldstart-save`, `control`, `gsq`, `gsq-panel`, `nvfp4` and `panel-control`, and benches `baseline`, `humming`, `levers` and `smoke`. Only `gsq-coder`, `gsq-coder-panel` and bench `gsq-coder` changed.
- In the launch goldens only the three gsq patch hashes changed; the argv and env are unchanged.

## Rebuilt test notebook
`/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/push-fable-f001-gsq/v4-fable-f001-gsq.ipynb`, built from the committed script at `9562eb7`:
- 905,009 bytes, under the builder's 950,000-byte limit.
- Only cells 4 and 14 differ from F001.
- Same metadata as before: internet on, `NvidiaRtxPro6000`, same docker image.

The conversion and engine notebooks in `kaggle/out/` are rebuilt too.

## Final serving cell (cell 14)
Cell 14 is 192,691 characters, mostly embedded source. A readable copy with the base64 removed is at `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/cell14_readable.py`. The parts around the launcher, verbatim:
```python
GSQ_BUDGET = {'download': 15 * 60, 'convert': 12 * 60, 'launch': 25 * 60, 'smoke': 10 * 60}
GSQ_BASE_DIR = MODEL_DIR                       # Intel's W4A16 mount (cell 4): the base the experts are served on
GSQ_GGUF, GSQ_MODEL = Path("/tmp/gsq/shard1.gguf"), Path("/tmp/gsq/model")
try:
    gsq.plan_check(GSQ_SRC, GSQ_BASE_DIR)      # fails here, before 30 GB of download, if Intel's base cannot be served
    gsq.patch_check(GSQ_SRC, WHEELHOUSE_DIR)   # ... or if runtime patches gsq/01-03 do not apply to the wheel's files
    _gsq_installer = gsq.start(gsq.install_wheelhouse, WHEELHOUSE_DIR)
    gsq.download(GGUF_SHARD1_URL, GSQ_GGUF, size=GGUF_SHARD1_SIZE, deadline=time.time() + GSQ_BUDGET["download"])
    _gsq_hasher = gsq.start(gsq.verify_sha256, GSQ_GGUF, GGUF_SHARD1_SHA256)
    _gsq_convert_deadline = time.time() + GSQ_BUDGET["convert"]
    gsq.convert(GSQ_SRC, GSQ_GGUF, GSQ_BASE_DIR, GSQ_MODEL, deadline=_gsq_convert_deadline, log_path=WORKING_DIR / "convert.log")
    gsq.check(GSQ_SRC, GSQ_MODEL, GSQ_BASE_DIR)
    gsq.join(_gsq_hasher, timeout=max(30.0, _gsq_convert_deadline - time.time()))
    gsq.join(_gsq_installer, timeout=600)
finally:
    gsq.write_timings(GSQ_TIMINGS)
MODEL_DIR = str(GSQ_MODEL)
_gsq_launch_start = time.time()
GSQ_LAUNCH_DEADLINE = _gsq_launch_start + GSQ_BUDGET["launch"]
SERVER_STARTUP_TIMEOUT = max(SERVER_STARTUP_TIMEOUT, GSQ_LAUNCH_DEADLINE - NOTEBOOK_START_TIME)
# ---- launch_sglang.py, profile qwen38fn-gsq-coder; its serve glue now calls:
env.update(serve_glue.prepare_env(python=PYTHON, env=env, working_dir=WORKING_DIR, prefix=PREFIX, venv=VENV,
                                  deadline=globals().get("GSQ_LAUNCH_DEADLINE")))
# ---- after the launch
gsq.wait_healthy(SERVED_MODEL_PORT, GSQ_LAUNCH_DEADLINE, proc)
_gsq_smoke = subprocess.run([... "smoke_gsq.py" ...], capture_output=True, text=True, timeout=GSQ_BUDGET["smoke"])  # TimeoutExpired -> RuntimeError
gsq.TIMINGS["smoke"] = {r["test"]: r.get("status", "ok" if r.get("ok") else "fail") for r in _gsq_results}
if _gsq_smoke.returncode: raise RuntimeError(f"smoke_gsq failed ({gsq.TIMINGS['smoke'] or 'no report'}); not starting the games ...")
```

## Launch command (unchanged from `76c3cf0`; golden `launch_gsq-coder.json`)
```
/tmp/sgl-intel/venv/bin/sglang serve --model-path $MODEL_DIR --load-format safetensors --model-loader-extra-config '{"enable_multithread_load":false}' --served-model-name flashnext --host 127.0.0.1 --port 8001 --tensor-parallel-size 1 --dtype bfloat16 --quantization gsq_lut --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.96 --context-length 139264 --page-size 64 --max-running-requests 10 --chunked-prefill-size 8192 --max-prefill-tokens 16384 --cuda-graph-max-bs-decode 10 --cuda-graph-bs-decode 1 2 4 7 8 9 10 --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 60 --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 64 --mamba-backend flashinfer --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer --moe-runner-backend auto --ple-offload-embedding --trust-remote-code --mm-feature-transport cpu --image-processor-backend pil --reasoning-parser qwen3 --tool-call-parser qwen3_coder --default-chat-template-kwargs '{"preserve_thinking":true}' --watchdog-timeout 1800 --schedule-policy lpm --warmups structured_output --enable-cache-report --enable-metrics --enable-request-time-stats-logging --weight-loader-prefetch-checkpoints --weight-loader-drop-cache-after-load --chat-template $MODEL_DIR/chat_template.jinja --gdn-mtp-cache-mode none --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --speculative-draft-model-path /tmp/sgl-intel/draft-view-<id> --speculative-draft-model-quantization compressed-tensors --speculative-moe-runner-backend auto --speculative-draft-kv-cache-dtype fp8_e4m3 --speculative-accept-threshold-single 1.0 --speculative-accept-threshold-acc 1.0 --speculative-token-map $WHEELHOUSE_DIR/hot_tokens_64k.pt
```
`SGLANG_GSQ_KERNEL` starts as `auto`, and the self-test sets it to `lut`, `lut_iq2s_int8` or `int8`. In the F001 test, `$MODEL_DIR` is `/tmp/gsq/model`.

## PR summary (draft)
- **Hypothesis:** ISTA's GSQ-RCO Coder GGUF experts, served by a lookup-table Marlin MoE kernel on Pennyroyal on top of Intel's exact BF16 base, run at about W4A16 speed (estimate 650–850 tok/s against F001 v2's 760–785). Quality is the open question.
- **Converter:** `serving/convert/gsq_coder` turns the GGUF into a `gsq_lut` checkpoint. The decoder matches gguf-py bit for bit, encoding error is at most 0.175 %, and the router mask matches Intel's rows bit for bit in all 48 layers.
- **Engine:** runtime patches gsq/01-03 add `--quantization gsq_lut`, per-GEMM bit widths and a 16-entry LUT dequant. Only the `qwen38fn-gsq-coder` profile embeds them; the pins match the real pennyroyal-v253 wheel.
- **Runner:** profile `qwen38fn-gsq-coder`, serve glue `gsq_serve`, a GPU self-test that picks `lut`, `lut_iq2s_int8` or `int8`, a smoke probe, arms `gsq-coder` and `gsq-coder-panel`, and bench `gsq-coder`.
- **Test notebook:** `v4-fable-f001-gsq` is F001 with only cells 4 and 14 changed. It checks the plan and the patches, downloads and converts at run time with internet on, then launches, smoke-tests and plays the 10-game panel.
- **Robustness:**
  - Every stage stops at a hard budget (15 / 12 / 25 / 10 min).
  - A LUT crash or hang falls back to int8.
  - Patches revert without backup copies.
  - Wrong answers in the smoke test only warn.
  - The plan stops above a 16 GB base copy.
- **Measured on the laptop:** GPU weights and FP8 KV pool are 42.50 GiB / 3.11M tokens for `lut` and 69.65 GiB / 0.92M for `int8`, against 68.50 GiB / 1.01M for W4A16. Kaggle's base plan copies 10.67 GB and links all 22 PLE shards.
- **Tested:**
  - `kaggle/tests`: 290 passed with the real F001 source and wheel.
  - Converter: 36 passed, 7 skipped.
  - Pins current, nbcheck clean.
  - Every non-gsq arm and bench notebook is byte-identical.
- **Not measured:** anything on a GPU. Score and compute cost come from the F001 test against F001 v2 at matched tokens, using `make significance`.
- **Docs:** `serving/CUSTOM.md` section 7, the bundle, converter and patch READMEs, and `kaggle/README.md`.

I also added a review-fixes line under the `runner-gsq` entry in `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/arms.md`. A backup of the previous version is at `<scratchpad>/arms.md.bak-gsqfix`. The run, publish and upload commands from the previous report are unchanged and each still needs your approval. The only difference is that publishing the model now uses `--convert --publish` instead of editing `PUBLISH` by hand.
