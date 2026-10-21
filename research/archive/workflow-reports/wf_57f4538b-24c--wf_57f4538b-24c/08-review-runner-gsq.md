# wf_57f4538b-24c: review:runner-gsq

Workflow run `wf_57f4538b-24c`, phase Review. 

---

I found no correctness defects in the conversion, the name map, the Marlin layout, the patches or the launch path. I did find 5 robustness defects and 1 minor one. Each robustness defect can throw away a GPU-hour test run (mostly after the 30 GB download and conversion), or let a stage run past its budget. I reviewed read-only at `76c3cf0`; the worktree is clean.

## Checked and correct

- **Decoding real blocks.** I fetched expert 137 (the implementer sampled 0 and 255) of six real tensors over HTTP and decoded them two ways: with the branch's decoder and with gguf-py b11476. For Q2_0 I used a direct transcription of `dequantize_row_q2_0` from llama.cpp b11476. They match bit for bit for all six types.
  - Served error: 0.156–0.171%, or 0.221–0.241% with each weight rounded to BF16. Q2_0 is 0.162% either way.
  - The GPTQ pack and unpack round trip is exact.
  - Scripts: `<scratch>/review-gsq/indep_check.py` and `indep_expert_map.py`.
- **Expert identity and orientation.** GGUF router row 137 in layer 1 equals Intel row 261 exactly.
  - The GGUF expert against Intel's dequantized expert 261 correlates at 0.889 (gate), 0.894 (up) and 0.856 (down).
  - GGUF gate against Intel up gives −0.003, and against a different Intel expert gives 0.002.
  - So experts are not mixed up, gate and up are not swapped, and nothing is transposed. The GDN head reordering doesn't matter because only experts and routers are taken from the GGUF.
- **Layout against Pennyroyal d00d88ef.**
  - Codes are int32 `[K/8,N]` with nibble j of word i at k = 8i+j. Scales are BF16 `[K/G,N]` and go through `marlin_moe_permute_scales`. Stock `uint4b8` and `uint8b128` already cover groups of 32 and 64. Group 16 exists only in the LUT module.
  - The LUT output order matches stock `kU4B8`. It uses `__byte_perm` without masking, the same way llama.cpp's `get_int_from_table_16` does.
  - In the patched `.cuh`, all four template declarations and four calls carry `kHasLut`, and the table travels in `scale2_ptr`, which the host code always passes.
- **Runtime fit.**
  - The FusedMoE weight loader handles `w13_qweight` (gate in the first half) and `scales`.
  - No other quantization method claims `gsq_lut`.
  - Shared-expert fusion stays off on CUDA, so the shared expert is not folded into the GSQ expert tensors.
  - Bench G2's expert recorder is safe because `qwen4_exp_mtp.py:203` disables recording in the drafter.
  - The draft view keeps albucino's 512-expert config.
- **Launch settings.** The launch arguments and every server environment key equal F001 v2's (checked against v2's `serve.log` `server_args`), except `--quantization gsq_lut`, the model path and `SGLANG_GSQ_KERNEL`. The memory figures are consistent.
- **Byte-identical builds.** I rebuilt all 6 existing arm notebooks from `git archive` of both branches; they are byte-identical. The W4A16 goldens are unchanged.
- **Tests.** `kaggle/tests` gives 271 passed, 1 skipped. `nbcheck` is clean. The test notebook differs from F001 only in cells 4 and 14, and its Fable and harness-patch cells are byte-identical. The synthetic converter output loads with the real safetensors 0.8.0.
- **Offline builds.** Arm builds take `enable_internet: false` from the template. Neither the launcher nor the serve glue downloads anything.

## Defects, most severe first

1. **Checks that fail only after the ~25 minutes of download and conversion (Medium-High).**
   - (a) The "before" hashes the patches check against (`01-gsq-lut-quant.py:41-47` for `server_args.py`, plus the other targets) come from a rebuilt copy: d00d88ef plus patch 05 from `build_bundle_pennyroyal.sh` (see `scratchpad/shipped`). They were never hashed from the actual `dfranzen/pennyroyal-v253` wheel. A mismatch makes gsq/01 exit 1 only when the launcher runs, after `build_fable_gsq.py:118-131` has finished downloading and converting.
   - (b) `check_checkpoint.py:344-345` fails the run if the copied `chat_template.jinja` isn't Intel's Hugging Face copy (`c3cf9e34…`). The converter copies it from dfranzen's resharded Kaggle copy, which nobody has hashed. The older base check at :459-464 only warns about the same difference.
   - **Fix:** run `gsq/0{1,2,3}-*.py <site-packages> --check` in the installer thread and join it before `download()`. Compare the checkpoint's template with `--base`'s template, hash the base's template in `plan_check`, and make the Hugging Face hash comparison a warning.
2. **Stage budgets aren't hard limits (Medium).**
   - `acquire_gsq.py:87-89` checks its timeout only when the child prints a line. Conversion is silent while it computes the mask and while it copies the base after the experts, and the check step is silent too.
   - The download retry loop (`acquire_gsq.py:155-173`) ignores the deadline. That allows up to six 120 s timeouts plus backoff per worker past budget.
   - The smoke subprocess (`build_fable_gsq.py:150`) has no timeout.
   - The self-test `_verify` uses `timeout=1800` (`gsq_serve.py:146`), longer than the 25-minute launch budget.
   - **Fix:** a watchdog thread that kills the child at the deadline; a deadline check inside the retry loop; and `subprocess.run(..., timeout=deadline - now)` for both the smoke test and the self-test.
3. **The int8 fallback has holes (Medium).**
   - `verify_gsq_engine.py:193` writes its report only at the end, and `gsq_serve.py:184` reverts and falls back only when `stock_ok is False`. If the new kernel crashes the self-test process hard or hangs until the timeout, there is no report. The run then raises with no games, even though int8 on stock kernels would work.
   - Separately, once the engine overlay is published, the wheel has no `.gsq-orig` copies, so `revert_kernel_patches` fails.
   - **Fix:** write the report after every case; run the stock and LUT cases in separate processes; treat a missing report or a timeout as a LUT failure, revert, and rerun with `--int8-only`.
4. **The smoke test is too strict to gate a quality test (Low-Medium).**
   - Fatal checks require exactly "4", exactly "red", and a python tool call within 2 samples at temperature 0.7 (`smoke_gsq.py:121-174`). A model that serves but answers worse — the thing this test exists to measure — would end the run with no score.
   - `build_fable_gsq.py:155` calls `.get` on `detail`, which is a string when the first check failed. That raises `AttributeError` and hides the "smoke_gsq failed" message.
   - **Fix:** make only HTTP, empty-output or parse failures fatal; record wrong answers as warnings; guard with `isinstance(detail, dict)`.
5. **Kaggle base layout is unchecked (Low).** The base copy has 38 shards. If the n-gram (PLE) tensors share shards with experts, up to 102 GB gets copied instead of linked (`convert.py:275-284`). That breaks the 45.6 GB and 5–8 minute estimates and bloats the published model. 38 = 16 + 22 hints that only Hugging Face's single n-gram (PLE) shard was split (then nothing breaks), but `plan_check` (`acquire_gsq.py:120-123`) only prints the copy size. **Fix:** fail or stretch the budgets in `plan_check` when `copied_bytes` exceeds about 15 GB.
6. **Minor.** Publishing from the conversion notebook means hand-editing `PUBLISH = 0` in the generated `.ipynb` (`build_bundle_notebooks.py:137`); add a `--publish` flag.

Also, the design said a precache thread would warm the Intel n-gram (PLE) shards during conversion; the build only re-points F001's precache at launch time. That is a doc mismatch, not a defect.

Nothing was pushed or published. All the run, publish and upload commands still need your approval.
