# throughput-and-alt-models: build:throughput-stack

Workflow run `wf_f7f94969-5fa`, phase Build. Win 2+ more games via throughput (sim-v2 sweep + a throughput stack of built-but-unrun fixes) and serve an alternative model (GLM-5.3-Flash EXL3, ISTA Qwen3.8-Flash-Next GSQ-RCO GGUF) by refactoring the RTX Pro 6000 SGLang launcher into per-model profiles

---

Branch `arm/throughput-stack` (off `arm/stack-v2`) now has the five lever branches merged, the new launcher and builder options, six new arms and a gating bench. Everything is committed and the worktree is clean. Nothing was pushed, no PR was opened and nothing ran on a GPU. 185 targeted tests pass, and every new arm and bench notebook builds locally.

**Worktree:** `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/throughput-stack`

**Commits** (`arm/stack-v2..HEAD`, first-parent; all end with the required two trailer lines)
- `504fc26` Merge arm/serving-bench (clean)
- `6ced208` Merge arm/fix-6-cold-start (conflict only in arms.json; kept both sides)
- `d730c1d` Merge arm/gate11 (arms.json only)
- `080082c` Merge arm/stub-dead-reasoning (arms.json only)
- `e99e18f` Merge arm/fix-3-prefix-reuse. One real conflict, in `tool_agent.py` where the reply's history message is built: I kept both stack-v2's ledger `absorb`/`note_think` lines and `reply_message = assistant_message`. arms.json kept all arms.
- `5f64e58` Add CFG `CUDAGRAPH_BS_EXTRA` (default `[]`; must be ints from 1 to `CUDAGRAPH_MAXBS`), documented in `serving/CUSTOM.md`
- `cbc112a` Refactor only: read bench patches through one helper, `_bench_patches`
- `56ab8cd` New arms.json `"runtime_patches"` key: an arm picks `serving/bench_patches/` files, applied after the shared patches. The builder refuses missing files, name clashes and names that would sort before a shared patch. Arms without the key build exactly as before. Tests and docs (`kaggle/README.md`, `CUSTOM.md`, the bench_patches README) are in the same commit.
- `edd3c23` Flags-off check against stack-v2: `tests/request_snapshot.py` and `tests/test_flags_off_requests.py`
- `6a8511a` The arms (`tput-safe`, `tput-safe-lowm`, `tput-accept05`, `panel-tput-safe`, `tput-coldstart-save`), `kaggle/tests/test_throughput_arms.py`, and a throughput section in `kaggle/README.md`
- `6d946d3` Bench `tput-gates` (C0, C3, C1; about 70 min estimated, against 150 for the `levers` bench)

**Arms**
- **`tput-safe`:** env `ARC3_LEDGER_ON_TRIM=1`, `ARC3_MAX_ACTIVE_STREAMS=11`, `ARC3_STUB_DEAD_REASONING=1`, `ARC3_VERBATIM_HISTORY=1`, `ARC3_PREFIX_REUSE_DIAG=1`; serve `CUDAGRAPH_BS_EXTRA=[3,5,6]`; `MAMBA_CACHE` stays 60.
- **`tput-safe-lowm`:** `tput-safe` plus `runtime_patches` `m64.py` and `nosplitk.py`. Push only if bench C3 gains at least 1% at batch 10 and loses nothing at batch 8 or below.
- **`tput-accept05`:** `tput-safe` plus `SPEC_ACCEPT_SINGLE` and `SPEC_ACCEPT_ACC` at 0.5. Push only if bench C1 gains at least 3%. It changes outputs, so it needs its own score comparison.
- **`panel-tput-safe`:** `tput-safe` on the balanced 20-min panel with 4 streams (a smoke test; it can't test gate11).
- **`tput-coldstart-save`:** see the second deviation below.
- **Baseline:** the existing `stack-v2` arm on this branch.

**Tests** (run under `nice`, one file at a time)

| Area | Passed |
|---|---|
| verbatim history | 9 |
| prefix-reuse probe | 23 |
| ledger-on-trim | 6 |
| flags-off requests | 4 |
| retained functions | 8 |
| tool timeout | 2 |
| sandbox deadline | 16 |
| bench | 33 |
| JIT cache | 33 |
| launcher JIT cache | 6 |
| build serving | 31 |
| throughput arms | 14 |

- **Harness, flags off:** every request body of a scripted 6-request turn (taken at `requests.post`) has the same hash on the stack-v2 tree (d55df75) and on this branch, under the default, Kaggle and stack-v2 environments. With `tput-safe`'s flags on, only the stored replies change: blank lines are kept and the cut-off reasoning-only reply is replaced by the stub.
- **Launch:** `control` and `stack-v2` render the same server argv and server env as stack-v2's own builder, and the other cells match too.
- **Mutation checks:** defaulting `CUDAGRAPH_BS_EXTRA` to `[3]`, or `ARC3_VERBATIM_HISTORY` to on, makes the identity tests fail, so they do catch changes.

**Launch command** (rendered from the built `tput-safe` cell; paths are stubbed)
```
sglang serve --model-path MODEL_DIR --load-format safetensors --model-loader-extra-config '{"enable_multithread_load":false}' --served-model-name flashnext --host 127.0.0.1 --port PORT --tensor-parallel-size 1 --dtype bfloat16 --quantization auto-round --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.96 --context-length 139264 --page-size 64 --max-running-requests 10 --chunked-prefill-size 8192 --max-prefill-tokens 16384 --cuda-graph-max-bs-decode 10 --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 60 --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 64 --mamba-backend flashinfer --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer --moe-runner-backend auto --ple-offload-embedding --trust-remote-code --mm-feature-transport cpu --image-processor-backend pil --reasoning-parser qwen3 --tool-call-parser qwen3_coder --default-chat-template-kwargs '{"preserve_thinking":true}' --watchdog-timeout 1800 --schedule-policy lpm --warmups structured_output --enable-cache-report --enable-metrics --enable-request-time-stats-logging --weight-loader-prefetch-checkpoints --weight-loader-drop-cache-after-load --chat-template MODEL_DIR/chat_template.jinja --gdn-mtp-cache-mode none --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --speculative-draft-model-path DRAFT_VIEW --speculative-draft-model-quantization compressed-tensors --speculative-moe-runner-backend auto --speculative-draft-kv-cache-dtype fp8_e4m3 --speculative-accept-threshold-single 1.0 --speculative-accept-threshold-acc 1.0 --speculative-token-map WHEELHOUSE/hot_tokens_64k.pt
```
The other arms differ from this only as follows:
- **`control` / `stack-v2`:** `--cuda-graph-bs-decode 1 2 4 7 8 9 10`.
- **`tput-accept05`:** both acceptance thresholds are `0.5`.
- **`tput-safe-lowm`:** the same argv, plus the two embedded patches.
- **`panel-tput-safe` and `tput-coldstart-save`:** the same argv as `tput-safe`. The save arm also writes the JIT cache once the server is ready.

**Local builds** (`python3 kaggle/build_notebook.py --arm <arm>`, no push)

| Notebook | Size |
|---|---|
| `v4-tput-safe` | 371,867 bytes |
| `v4-tput-safe-lowm` | 373,197 bytes |
| `v4-tput-accept05` | 371,932 bytes |
| `v4-panel-tput-safe` | 371,989 bytes |
| `v4-tput-coldstart-save` | 371,851 bytes |
| `v4-stack-v2` | 371,456 bytes |
| `v4-control` | 371,415 bytes |
| bench `tput-gates` | 638,279 bytes |
| bench `smoke` | 636,065 bytes |
| bench `levers` | 639,447 bytes |

All cells compile, and the embedded harness patch applies cleanly to `bundle-base` (`git apply --check`).

**Where I departed from the plan**
- **The "one-block fix":** the research report's one-block fix turned out to be the fix-3 conflict resolution itself. "History keeps blank lines" is just what the flag does, so no extra code was needed. The new test confirms the blank lines survive.
- **Two extra arms and a bench:**
  - `tput-coldstart-save`: a JIT cache only restores under the exact settings it was saved with, so `coldstart-save`'s cache would not fit `tput-safe`. A test confirms this.
  - `panel-tput-safe`: for step 3 of the run order.
  - `tput-gates`: so step 1 costs about 70 min instead of 150.
- **`JIT_CACHE_DIR` is not set on `tput-safe`:** the cache dataset doesn't exist yet. The README says how to add a consumer arm once it does.
- **Full suite not run**, per the laptop rule; only the targeted files above.
- **No top-level README change:** this base still has Franzen's README, and the fork overview lives on `docs/fork-readme`. `CONFIGURATION.md` is unchanged because no new harness flag was added. I logged the arm under "In progress" in `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/arms.md`.

**Limitations**
- **Repeat runs are blocked for `stack-v2`:** the builder refuses a notebook identical to an earlier run. `tput-safe` can repeat by dropping `ARC3_PREFIX_REUSE_DIAG`, but a second `stack-v2` run needs a builder change, such as an inert replica env var.
- **The gates are not enforced:** nothing stops `tput-safe-lowm` or `tput-accept05` being pushed before the bench results are in. That relies on the docs.
- **Merging `arm/model-profiles` will conflict:** a test merge conflicts in `kaggle/build_notebook.py` and `serving/kaggle/launch_sglang.py`, because that branch is mid-refactor. After the merge, `CUDAGRAPH_BS_EXTRA` and the `runtime_patches` key need moving into its profile/config structure, and goldens are needed for the new arms.
- **Runtime patches don't change the JIT cache's config hash** (this predates my changes). Triton keys its kernels by source, so the risk looks low.
- **GPU-side effects are unverified:** the memory cost of the three extra CUDA graphs (about 70 MB each, per the research), gate11's effect on KV pressure (peaks of 0.99), and every throughput number. The sim-v2 estimate of ×1.06 is an upper bound.

**Key files**
- `serving/kaggle/launch_sglang.py`
- `kaggle/build_notebook.py`
- `kaggle/arms.json`
- `kaggle/bench.json`
- `kaggle/tests/test_throughput_arms.py`
- `ARC3-Inference/tests/test_flags_off_requests.py`
- `ARC3-Inference/tests/request_snapshot.py`
- `kaggle/README.md`
- `serving/CUSTOM.md`

All are under the worktree path above.
