# Our serving stack: what we can modify

Serving is Franzen's Pennyroyal SGLang 2.5.3 setup:
- The model is the **W4A16 AutoRound Qwen3.8-Flash-Next** plus the albucino INT4 MTP drafter.
- It runs on one RTX Pro 6000, installed offline from the `dfranzen/pennyroyal-v253` wheelhouse.

The launcher is now ours: `serving/kaggle/launch_sglang.py`. It started as a verbatim copy of his notebook cell, and `kaggle/build_notebook.py` builds the notebook's serving cell from it. Change serving with ordinary commits on an `arm/` branch, like the harness.

The layers are ordered from cheapest to change to most expensive.

To measure a change before spending a game run on it, put it in a config of `kaggle/bench.json` and build a serving benchmark (`kaggle/build_notebook.py --bench`, see `kaggle/README.md`). A bench config takes CFG keys, server env, launch-argument edits and patches from `serving/bench_patches/`.

## 1. Launch config: `CFG` in the launcher, or an arm's `"serve": {...}` override

| Key | Now | What it controls | What we know |
|---|---|---|---|
| `CTX` | 136 Ki | Server context length (harness window 128k + 8k margin) | Shrinking context to add streams hurt (streams14, ledger-streams12) |
| `MEMFRAC` | 0.96 | GPU memory for weights + KV | KV pool ≈ 1.01M tokens at FP8; peaks 0.99 used |
| `MAXREQ` | 10 | `--max-running-requests` (decode batch) | Batch averages 8.9/10 (issue #7); see `gate11` arm |
| `CUDAGRAPH_MAXBS` | 10 | Largest CUDA-graph decode batch | Must cover `MAXREQ` |
| `CUDAGRAPH_BS_EXTRA` | `[]` | Extra decode batch sizes to capture CUDA graphs for, on top of the fixed 1/2/4/7/8/9/10 | Without a graph, batch 3 pads to 4 and 5/6 pad to 7. `[3, 5, 6]` (arm `tput-safe`) costs about 10 MB and 3 s of startup per size: the graphs below batch 10 share its memory pool, and in the `fz-arms-v4-control` serve.log each one took ≤10 MB and ~1 s in each of target verify, draft decode and GDN recovery. Expected gain ≤0.3% (`research/archive/reports/throughput-900.md` on `main`) |
| `MAMBA_CACHE` | 60 | Mamba state slots (≈ 6 per stream) | Peaks at 0.67 used; not the limiter |
| `CHUNK` / `MAX_PREFILL` | 8192 / 16384 | Chunked-prefill size and per-batch prefill cap | Untested; larger = faster trim re-prefills, more decode stalls |
| `SPEC`, `SPEC_STEPS` | on, 3 | MTP speculative decoding | 3 is best (GB10 scan); 4 can't run (Qwen QSA caps at 4 draft tokens) |
| `SPEC_ACCEPT_SINGLE` / `_ACC` | 1.0 / 1.0 | Draft acceptance thresholds (lossless at 1.0) | **Never tested.** Sweep Tier 1: 0.5/0.3 could give +8–16% accept length |
| `FRSPEC` | on | Frequency-ranked speculative vocab map | Needs the pinned tokenizer |
| `AUTOTUNE` | on | FlashInfer autotune at start | No-op on this stack: SGLang skips autotune for our MoE runner and quantization, so it is not part of the cold start (see [Cold start](#cold-start-jit-cache-issue-6)) |
| `KVDTYPE` / `SSM_DTYPE` | fp8_e4m3 / bf16 | KV-cache and Mamba state dtypes | nvfp4 KV crashes (`float4_e2m1fn_x2`) |
| `MAMBA_RADIX` | extra_buffer | Mamba prefix-cache strategy | Prefix hit falls back to prompt-end in ~45–60% of turns (issue #3) |
| `LINEAR_BACKEND` | flashinfer | Linear-attention / Mamba kernels | |
| `PREFETCH_CHECKPOINTS`, `DROP_CACHE_AFTER_LOAD` | on | Checkpoint loading | Startup time |
| `JIT_CACHE_SAVE` / `JIT_CACHE_DIR` | off / None | Save the JIT caches once the server is ready; restore them from a mounted dataset before launch | See [Cold start](#cold-start-jit-cache-issue-6) |
| `GDN_MTP_CACHE_MODE` | none | GDN cache for MTP | |
| Builder-added keys: `DRAFT_KVDTYPE`, `ATTN_BACKEND`, `TEXT_CONFIG_OVERRIDE`, `SPEC_ADAPTIVE` | | Draft KV dtype, attention backend, model `text_config` override (e.g. `num_experts_per_tok`), adaptive speculation | experts16: −7.6; experts8: no speed-up; adaptive spec: −36% throughput |

## 2. Server process environment: an arm's `"serve_env": {...}`

These are set in the launcher's `env.update(...)`:
- `SGLANG_ENABLE_SM120_LOWM_BF16_GEMM` (on)
- `SGLANG_SM120_ONLINE_MXFP8` (off; on was −27% decode)
- `SGLANG_SM120_LOWM_FP8_WEIGHT`, `SGLANG_SM120_LM_HEAD_FP8` (off; untested)
- `SGLANG_MAMBA_CONV_DTYPE`, `SGLANG_MM_PREPROCESS_DEVICE` (cpu)
- Thread counts and cache paths, including `SGLANG_JIT_CACHE_DIR`, `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR`. `JIT_CACHE_SAVE` / `JIT_CACHE_DIR` persist these across containers (see [Cold start](#cold-start-jit-cache-issue-6)).

## 3. Launch arguments: edit `args` in the launcher

These are fixed today but are ours to change:
- **Scheduling:** `--schedule-policy lpm` (try `fcfs`), `--page-size 64`, `--mamba-track-interval 64`.
- **Reasoning and tools:** `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`, `--default-chat-template-kwargs '{"preserve_thinking":true}'`. The chat template can take `reasoning_effort`; sweep Tier 1 suggests trying `low` from game start.
- **CUDA graphs:** the `--cuda-graph-bs-decode` batch list. `CUDAGRAPH_BS_EXTRA` adds sizes to it.
- **Images:** `--image-processor-backend pil`, `--mm-feature-transport cpu`.
- **Logging:** `--watchdog-timeout`, `--warmups`, and the metrics / request-time logging flags. Turning off the request logs saves CPU.

## 4. SGLang's own Python code: `serving/runtime_patches/`

Anything in SGLang above the compiled kernels can be changed here. Drop a unified diff (`*.patch`, paths like `sglang/srt/...`) or a patch script (`*.py`, run with the site-packages path) into `serving/runtime_patches/`. The builder embeds it, and the launcher applies it to the installed SGLang before launch, once per install.

Candidates:
- The scheduler: request ordering, preemption, how queued requests backfill.
- The radix and Mamba prefix-cache policy: which checkpoints to keep, LRU refresh. This is where issue #3's prompt-end fallbacks live.
- Speculative-decoding accept logic.
- Logits processing: stop conditions, a soft output cap or streaming abort for runaway reasoning (issue #9).
- Request priority: honour the harness `priority` field.
- Image preprocessing and caching (issue #24).
- Metrics.

Keep each patch small and pinned to SGLang 2.5.3 (`d00d88ef`). A patch that no longer applies fails the launch loudly.

A patch can also stay arm-only: an arm's `"runtime_patches"` in `kaggle/arms.json` names files of `serving/bench_patches/`, which the builder embeds after the shared ones for that arm only (see `kaggle/README.md`).

### MoE routing variants (`runtime_patches/20-moe-routing.py`)
The patch wraps SGLang's `select_experts`. Without `SGLANG_ARC_MOE_ROUTING` set in an arm's `"serve_env"`, it changes nothing. With it set, it re-weights the experts the router already chose, using the router's own softmax probabilities `p`:

| `SGLANG_ARC_MOE_ROUTING` | Weights | Extra settings |
|---|---|---|
| `base_norm` | `p_i / sum(top-K0 p)`. The trained top-K0 keep exactly their trained weights; experts beyond K0 add their raw probability on top. | `SGLANG_ARC_MOE_BASE_K` (K0, default 10 = the checkpoint's experts per token) |
| `nucleus` | Fewest experts reaching `PARAM` of the chosen mass, renormalised (at least `MIN_K`). | `SGLANG_ARC_MOE_PARAM` (default 0.9), `SGLANG_ARC_MOE_MIN_K` (default 1) |
| `temp` | `p ** (1/PARAM)`, renormalised (sharper below 1). | `SGLANG_ARC_MOE_PARAM` |

- **Combine with more experts per token** (`"serve": {"TEXT_CONFIG_OVERRIDE": {"num_experts_per_tok": 16}}`) to test more experts without SGLang's default renormalisation. That renormalisation shrinks every trained expert's weight and probably contributed to experts16's −7.6. Compute follows k, so k=16 still costs about 17% throughput.
- **It skips, with a one-time `[arc-moe-routing]` log line, wherever the formula doesn't fit:** draft (MTP) MoE blocks unless `SGLANG_ARC_MOE_DRAFT=1`, non-softmax scoring, grouped top-k, bias, fused shared experts, routed scaling, and non-standard output formats. Padded tokens are left alone.
- **The launcher logs the checkpoint's experts per token,** to confirm K0.
- **Arms:** `experts16-basenorm`, `experts12-basenorm`, `experts13-basenorm` (the user's 13).
- **Tests:** `serving/tests/test_moe_routing.py` (needs torch). It runs against the real v2.5.3 `topk.py` with `SGLANG_TOPK_PY` set.

## 5. Kernels and wheels: rebuild the wheelhouse

These need `serving/build_bundle_pennyroyal.sh`, run in the Kaggle GPU Docker image or an internet-enabled Kaggle notebook. Upload the result as our own dataset, then point `WHEELHOUSE_DIR` and `kaggle/template-metadata.json` at it.
- CUDA kernels: low-M GEMM tuning (issue #11), Marlin, FlashInfer, sgl-kernel.
- torch/CUDA versions, and Franzen's five compiled-in patches.

## 6. Model artifacts (the W4A16 Flash-Next stays fixed)

- `config.json` / `text_config` overrides through a model view (`TEXT_CONFIG_OVERRIDE`).
- The chat template (`chat_template.jinja`), which could also be served from a view.
- The draft model and the FR-Spec token map.
- Changing the quantization or the base model is out of scope by default.

## Startup deadline: offline runs wait for the server

The launcher waits for SGLang's `/health` before the benchmark starts, up to a deadline:

- **Competition rerun** (`TRUE_SUBMISSION`): `SERVER_STARTUP_TIMEOUT` (12 min from notebook start, template cell 4), as in Franzen's notebook. Past it the benchmark is released with the server still loading, because Kaggle may cancel a rerun that takes no action for too long.
- **Offline test run:** `OFFLINE_STARTUP_TIMEOUT` (45 min). An early release there only starts every game's 121-min clock with no model: `snowiers/v4-final-hold-lap-10x` (2026-10-17) loaded its weights in 1,299 s (the 90 earlier runs: 255-801 s) and lost about 15 min of play per game. Past 45 min the run is released as before, so a server that never comes up still shows its errors.

`kaggle/tests/test_offline_wait_ready.py` pins both rules.

## Cold start: JIT cache (issue #6)

Every container starts with empty kernel caches, so the first target-verify CUDA-graph capture compiles from scratch: `cuda_graph target_verify` in serve.log's `Engine startup timings` is 145-152 s in every run. The fix is to carry the JIT caches across containers. The arm **`coldstart-save`** (`JIT_CACHE_SAVE: true`) produces the cache; it runs the balanced panel with 4 streams, like `panel-control`, for 5 minutes.

FlashInfer autotune (issue #6 proposal 3) does not run on this stack, so `AUTOTUNE: false` changes nothing and the silent ~125 s before TileLang's compile is not autotune.
- `should_run_flashinfer_autotune` (`model_executor/runner/flashinfer_autotune.py`) runs it only for a `flashinfer_*` MoE runner, or for ModelOpt FP4/FP8 and MXFP8 GEMMs.
- We launch with `--moe-runner-backend auto` and `--speculative-moe-runner-backend auto` under AutoRound, and SGLang's overrides leave both at `auto` for that quantization.
- This is read from the source, not from a log. Any serve.log should confirm it by having no `Running FlashInfer autotune` line.

Overlap loading (`--startup-weight-load-mode overlap`, issue #6 proposal 2) cannot hide the capture behind the ~387 s weight read on this stack. SGLang 2.5.3 rejects it for our launch: `StartupWeightLoadManager._get_unsupported_reason` (`model_runner_components/startup_weight_load.py`) fails several independent rules:
- speculative decoding (NEXTN), and draft workers;
- `--weight-loader-drop-cache-after-load` (`DROP_CACHE_AFTER_LOAD`);
- any quantization (AutoRound);
- multimodal models;
- an architecture allowlist of Llama, Qwen2 and Qwen3 only.

The model load raises `ValueError: --startup-weight-load-mode=overlap is not supported: ...` with no fallback to `serial`, so the server exits. Franzen's prefetch-lookahead patch does not touch these rules. The launcher therefore has no knob for it, and the JIT cache is the only way to shorten the capture.

### What the JIT cache flags do

The launcher embeds `serving/kaggle/jit_cache.py` as the module `jit_cache`.

- **`JIT_CACHE_SAVE`** waits for `/health`. SGLang answers it only after graph capture and its warmup request. The launcher then writes `/kaggle/working/sgl-jit-cache.tar` and `sgl-jit-cache.manifest.json`.
  - The tar holds `PREFIX/cache/{triton,flashinfer,sglang/jit,torchinductor,cuda}` and `~/.tilelang`, with modification times kept. Lock files, Triton's half-written `tmp.pid_*` entries and SGLang's in-progress `.staging-*` builds are skipped, as is any file removed while the tar is written. A failed save leaves no partial tar behind.
  - The manifest records:
    - the GPU name and compute capability;
    - the Python, torch, torch-CUDA and nvcc versions;
    - the sglang, sgl-kernel, triton, flashinfer, tilelang and tvm-ffi versions;
    - a hash of `CFG`, the target's `config.json` and the server's `SGLANG_*`/`FLASHINFER_*`/`TRITON_*`/`TORCHINDUCTOR_*` env.
  - The hash leaves out the keys that cannot change a kernel: the cache flags, `PREFETCH_CHECKPOINTS`, `DROP_CACHE_AFTER_LOAD` and the acceptance thresholds `SPEC_ACCEPT_SINGLE`/`SPEC_ACCEPT_ACC` (float arguments of a prebuilt kernel, called outside the CUDA graphs).
  - The log line `JIT cache saved to ... N files, X MB` reports the result. Any `not in the snapshot: <dir>` lines list other cache directories the server filled. If one of them holds kernels, add it to `CACHE_SUBDIRS`.
- **`JIT_CACHE_DIR`** points at the mounted dataset. Both Kaggle layouts work, `/kaggle/input/datasets/<owner>/<slug>` and `/kaggle/input/<slug>`.
  - The launcher finds the tar, or the tree Kaggle made if it unpacked the tar on upload, and compares the manifest with this runtime.
  - It unpacks into a staging directory and only then moves the files into place.
  - Any mismatch or error prints `JIT cache NOT restored, starting cold ...: <reason>` and the launch continues cold, exactly as without the flag.
  - Changing the wheelhouse, the GPU, the model or a kernel-relevant `CFG`/env key therefore needs a new snapshot.

### Publishing the cache

1. Push the producer: `python kaggle/build_notebook.py --arm coldstart-save --push`.
2. When it finishes, download the two cache files. The fetcher also writes the kernel log, `v4-coldstart-save.log`, into the same folder, so fetch into a scratch folder and check that log for `JIT cache saved to`:

       python kaggle/fetch_version_output.py mellamoa v4-coldstart-save <version> /tmp/coldstart-save 'sgl-jit-cache\.(tar|manifest\.json)$'

3. Move only the tar and its manifest into the dataset folder, so the log is not published, and publish them as a private dataset:

       mkdir -p /tmp/sgl-jit-cache && mv /tmp/coldstart-save/sgl-jit-cache.tar /tmp/coldstart-save/sgl-jit-cache.manifest.json /tmp/sgl-jit-cache/
       kaggle datasets init -p /tmp/sgl-jit-cache      # set "id": "mellamoa/sgl-jit-cache" and a title
       kaggle datasets create -p /tmp/sgl-jit-cache    # later snapshots: kaggle datasets version -p ... -m "..."

   Kaggle may unpack the tar on upload. The restore handles both forms.
4. Add a consumer arm. `"datasets"` mounts the dataset (the builder appends it to the kernel's `dataset_sources`):

       "coldstart-cache": {
         "demo_games": "balanced",
         "demo_minutes": 20,
         "env": {"ARC3_MAX_ACTIVE_STREAMS": "4"},
         "datasets": ["mellamoa/sgl-jit-cache"],
         "serve": {"JIT_CACHE_DIR": "/kaggle/input/datasets/mellamoa/sgl-jit-cache"}
       }

5. In its log, look for `JIT cache restored in Xs`. `cuda_graph target_verify` should then fall from ~145 s to a few seconds.

Not yet verified on Kaggle:
- Whether a cache from one container is valid in the next. The issue leaves this open even for the same GPU and CUDA.
- Whether FlashInfer's and SGLang's ninja-built JIT modules count as up to date. Ninja compares their modification times with the freshly installed package sources and its own build log, so it may rebuild them anyway.
- Whether Triton's autotuner reruns its benchmarks on cached kernels.

The restore log line and the startup timings answer all three.

Startup time becomes play time only while the per-game clock is not capped. The clock starts after startup, so once the `532*60` cap binds, a faster start no longer adds play.
