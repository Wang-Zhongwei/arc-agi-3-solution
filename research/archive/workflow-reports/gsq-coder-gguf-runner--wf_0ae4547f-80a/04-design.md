# gsq-coder-gguf-runner: design

Workflow run `wf_0ae4547f-80a`, phase Design. Serve the ISTA GSQ-RCO Coder GGUF the way its Hugging Face repo documents (llama.cpp or Strata, both shards, 256 experts, mmproj), validate the harness contract locally, build staging/engine/test notebooks and the 2-hour final-run notebook

---

DESIGN: arm/runner-gguf, branched off arm/runner-gsq (worktree scratchpad/wt/runner-gguf; the PR targets arm/runner-gsq)

## 0. Decisions, each checked against the HF card and the b11476 source
- **Engine: llama.cpp b11476 (commit 988190680), not Strata.**
  - The card gives llama.cpp first, with an exact recipe: "loads the pair when given the first [shard]".
  - The card says shard 2 (the n-gram table) "may be served from disk", and b11476 does exactly that by default. `--lazy-mode auto` (arg.cpp:2713) reads tensors over 4 GiB on demand from the mmap. It logs "enabling prefetch for 'per_layer_token_embd.weight'" (llama-model.cpp:1802).
  - `-np 10` gives the harness its 10 streams, the same as control and F001. Strata caps at 8 (serve/server.py:2066, PARALLEL_MAX).
  - Strata's speed lead comes from MTP drafts, which batched slots do not use by default (docs/BATCHING.md:67).
  - Strata also needs two CUDA builds, iq_pack, an MTP head built from base-model weights outside the Coder repo, and an image-plus-parallel-slots path that is untested upstream (open hang #1317).
  - llama.cpp is one build, already scripted and tested in a container. Strata stays a follow-up arm (arm/runner-strata) that reuses the same staged files.
- **Files**, re-verified today with HF paths-info at revision 5348543e:
  - shard1: IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf, 29,608,446,496 B, e11083ba… (staged; p1..p3 COMPLETE).
  - shard2: IQ1_M/...-00002-of-00002.gguf, 28,800,138,432 B, 316b46f3….
  - mmproj: mmproj-Qwen3.8-Flash-Next-BF16.gguf, 907,543,008 B, b1a82259….
- **Template: the one embedded in the GGUF** (tokenizer pre qwen35). No `--chat-template-file` on Kaggle.
  - Its effort is `reasoning_effort|default('xhigh')`, and it maps 'high' to xhigh.
  - The harness never sends reasoning_effort (`ARC3_REASONING_EFFORT_LADDER` is unset in F001), so every request runs at the card's xhigh.
- **Sampling** stays the harness's, for comparability: T 0.7, top_p 0.95, top_k 20. These override the GGUF defaults (T 1.0). `--min-p 0` matches SGLang's default; llama.cpp's own default is 0.05.
- **No proxy.** There is one harness change (the context-overflow wording, §3). Everything else is launch flags and notebook env.

## 1. Engine build: CPU notebook mellamoa/gguf-engine-build
- **Script:** `serving/gguf/build_llama_server.sh`, the research script with its logic unchanged:
  - pip CUDA 13.0.88 toolchain, and a stub libcuda from a sha-pinned NVIDIA .deb;
  - `-DCMAKE_CUDA_ARCHITECTURES=120a-real`, CCCL v3.4.3 plus a probe that it wins over the wheel's copy;
  - `GGML_NATIVE=OFF` with AVX2, RUNPATH `$ORIGIN`, and `-rpath-link`;
  - targets llama-server and llama-batched-bench;
  - bundles libcudart/libcublas/libcublasLt 13 and libgomp, behind the `llama-server.sh` wrapper.
- **Generator:** `kaggle/build_gguf_engine.py` writes `push-gguf-engine-build/{gguf-engine-build.ipynb, kernel-metadata.json}`.
  - CPU, internet on, nothing attached, `docker_image` pinned to F001's (glibc 2.35).
  - One cell embeds the script (gz+b64, with its sha256 in a comment), runs it under `timeout 75m` and tees build-llama.log.
  - Output is only `llama-b11476-cu130-sm120a.tar.gz` (~0.5 GB), its `.json` manifest and the log.
- **Verify step:** the script exits non-zero unless all of these pass:
  - glibc ≤ 2.35 and GLIBCXX ≤ 3.4.30;
  - RUNPATH `$ORIGIN` on every library;
  - the CUB `_V_300403_` DeviceTopKKernel is in libggml-cuda;
  - `--version` reports 9881906, and ldd is clean;
  - a CPU smoke test serves stories260K (CUDA init fails cleanly on the stub driver).
  - Locally, `build_gguf_engine.py --check-manifest <dir>` re-checks the downloaded .json: every check true, tag, commit, archs 120a-real, bytes and sha256.
- **Fallbacks:**
  - Exit code 3 within seconds means Kaggle swapped the image. Rebuild with `--image latest --bundle-glibc`. That was tested in a 22.04 container, but not with the 580 driver.
  - Route B: the same notebook on RtxPro6000 with no competition and internet on. About 10 GPU minutes; it also runs `--list-devices`.

## 2. Staging shard 2 and the mmproj (reusing stage_gsq)
- **Pins:** `serving/gguf/__init__.py` holds GGUF_FILES.
  - shard1 is re-exported from gsq_coder, so there is one source of truth; shard2 and mmproj are new.
  - Each entry has path, size, sha256, a url at 5348543e, a slug template and a part count.
  - Also LLAMA_TAG/COMMIT and EMBEDDED_TEMPLATE_SHA256 (computed from runner-research/chat_template.jinja, which is the shard-1 header's template).
- **stage_gsq.py (refactor only):** `kernel_refs(owner, parts, slug=STAGE_SLUG)`, and docstrings say "a pinned file". The shard-1 notebooks must rebuild byte-identically (test).
- **`build_gsq_stage.py --file {shard1,shard2,mmproj}`:**
  - shard2 goes to mellamoa/gsq-coder-shard2-p1..p3: 3 × 9.60 GB, the part size shard 1 already proved. Two parts of 14.4 GB would need a second copy on a CPU disk that has not been tested at that size.
  - mmproj goes to mellamoa/gsq-coder-mmproj-p1 (one part).
  - Same `stage()` as shard 1: HF API size and sha256 check, 16-way Range download, resumable, 180-min deadline, part sha256, manifest.
  - `--check-manifests DIR --file X --parts N` checks them.
- **Shard-2 fallback** (`build_fable_gguf.py --shard2-from clothespin`): mount clothespin/strata-qfn-q2-0-assets, check the sha256 of its `…Q2_0-00002-of-00002.gguf` (the same bytes, 316b46f3…), and symlink it under the IQ1_M name. It is not the default because their README reports large mounts that never started.

## 3. Runner: `serving/kaggle/gguf_serve.py` (standard library only)
- **`place()`:**
  - `stage_gsq.locate` finds every staged part under either mount layout.
  - Shard1 and shard2 are reassembled in two threads, then the mmproj, into /tmp/gguf/ under the **original HF names**. llama.cpp builds shard 2's path from `-m …-00001-of-00002.gguf` (llama-model-loader.cpp:596-672).
  - Every part's sha256 and the whole file's are checked in the same read. Shard 1 measured 157 s at 189 MB/s on the GPU host.
  - The writes leave shard 2 in the page cache (176.9 GB RAM), so the on-demand n-gram reads start warm. Cell 8's model precache thread is never started.
- **`check_headers()`:** shard1's header sha256 is d663d111…, split.count 2, expert_count 256, arch qwen4exp; shard2 has split.no 1. It reuses the gsq_coder/gguf.py reader.
- **`install_engine()`:** the research `gpu_install_llama.py` as a function, run in parallel with `place()`.
  - The tarball's sha256 and size must match its manifest; it is extracted to /tmp/lcpp.
  - `--version` must run, and `--list-devices` must show CUDA0.
- **`launch_args(cfg)`** is the only place the argv is built (golden-tested). It is passed as an argv list, so the JSON needs no shell quoting:
  - `llama-server.sh -m /tmp/gguf/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --mmproj /tmp/gguf/mmproj-Qwen3.8-Flash-Next-BF16.gguf`
  - `--host 127.0.0.1 --port 8001 --alias flashnext -ngl all -fit off -fa on -ctk f16 -ctv f16 --lazy-mode auto`
  - `-np 10 --no-kv-unified -c 1392640 -b 2048 -ub 512 --jinja --reasoning-format deepseek --min-p 0 --no-prefill-assistant`
  - `--chat-template-kwargs {"preserve_thinking":true} -ctxcp 8 -cram {8192 smoke | 32768 final} --metrics -to 3600`
- **Why these values:**
  - 1,392,640 / 10 = 139,264 tokens per slot, the same per-request window as SGLang.
  - f16 KV is required: b11476's sparse-attention gather only runs with unquantized K and V.
  - VRAM: about 34 GB of KV plus 29.6 GB of weights plus the mmproj and buffers, roughly 70 of 95.6 GB. The n-gram table stays on the host by design.
  - In the final run, 25 games rotate through 10 slots and each idle game's state is about 2.5–3 GB, so the RAM cache gets 32 GiB.
  - No speculative decoding: the Coder GGUF has no MTP tensors.
- **`wait_healthy(deadline, proc)`:** waits for /health 200. If the process exits, it raises with the tail of serve.log.
- **`check_serve_log()`:**
  - requires `n_slots = 10, n_ctx_slot = 139264`;
  - requires `offloaded N/N layers to GPU` with numerator equal to denominator;
  - requires the per_layer_token_embd prefetch line;
  - rejects "CUDA error" and "GGML_ASSERT".
- **`check_props()`:**
  - /props: the chat template's sha256 equals EMBEDDED_TEMPLATE_SHA256, and vision is enabled;
  - POST /apply-template with a harness-shaped message must contain "Reasoning effort is set to xhigh".
- **`patch_context_error(bundle_dir)`:** an anchor edit of the bundle's tool_agent.py `_is_context_length_error`.
  - The anchor `        or "context_length_exceeded" in message\n` must occur exactly once.
  - It adds "exceeds the available context size", "exceed_context_size_error" and "is larger than the max context size" (server-context.cpp:3411 and 3420, server-common.cpp:69).
  - It runs before cell 16 imports the harness, the same mechanism the Fable layer uses.
  - The same lines go into ARC3-Inference/inference/agent/tool_agent.py as their own commit; under SGLang they never match.
- **`MetricsPoller`:** every 60 s it reads /metrics (llamacpp: tokens_predicted_total, prompt_tokens_total, n_decode_total, requests_processing, requests_deferred) plus a /slots summary, and appends to /kaggle/working/llama_metrics.jsonl.
  - `kaggle/gguf_report.py` turns that, timings.json and smoke_gguf.json into decode and prefill tok/s per window. serve_report.py stays SGLang-only.
- **Env is unchanged from F001:**
  - `LOCAL_ANALYZER_PROVIDER=vllm`, so top_k and chat_template_kwargs are sent;
  - `ARC3_REASONING_HISTORY_KEY=reasoning_content`;
  - `ARC3_MAX_ACTIVE_STREAMS=10`;
  - `ARC3_HTTP_RETRY_INITIAL_SECONDS=900`.

## 4. GPU notebooks: `kaggle/build_fable_gguf.py --mode smoke|final`
- **Source:** F001 as we ran it (sha256 prefix 86f9cc40b371d9e6).
  - The builder refuses any change outside cells 4, 14 and 18.
  - It asserts that cell 2 (harness patch) and cell 6 (Fable layer) are byte-identical, that the cell count is unchanged, and that the notebook is under 1 MB.
- **Cell 4, three anchor edits:**
  - `SERVER_STARTUP_TIMEOUT` 12 → 45 min.
  - `MODEL_DIR = _resolve_input(MODEL_DIR)` → `MODEL_DIR = '/tmp/gguf'`.
  - The same for `DRAFT_MODEL_DIR`. Cell 8 stays byte-identical, and its thread is never started.
- **Cell 14:** an embedded payload (stage_gsq, the pins, gguf_serve, the gsq_coder gguf reader, smoke_gsq, smoke_gguf), then these stages. Each has a hard budget and raises when it runs over, so games never start on a broken server. All stages are timed into timings.json.

  | Stage | Budget |
  |---|---|
  | locate | 2 min |
  | install_engine in parallel with place | 20 min |
  | check_headers | 1 min |
  | patch_context_error | — |
  | launch and /health | 15 min |
  | check_serve_log and check_props | 1 min |
  | smoke_gguf | 12 min |
  | start MetricsPoller | — |

  Expected total 12–18 min (estimate).
- **`serving/tools/smoke_gguf.py`** reuses smoke_gsq's checks:
  - text, vision, tool_choice required, and a harness-shaped request. A failure stops the games; a wrong answer is only a warning.
  - single-stream tok/s on a 512-token reply.
  - **New, parallel:** 10 concurrent harness-shaped requests, each with a 640 px PNG and the python tool, thinking on, max_tokens 1024. All must return 200 within 8 min; the aggregate tok/s is recorded.
  - **New, overflow:** one text request of about 150K tokens must return HTTP 400 with a body the patched matcher recognises.
  - **Floor:** single stream ≥ 10 tok/s and aggregate ≥ 40 tok/s, or the notebook stops. This is set to catch misconfiguration, not to judge quality.
- **Cell 18:**
  - `analyzer_timeout` 900 → 1800, because a 12,288-token reply at about 10–30 tok/s per stream can take longer than 900 s.
  - SMOKE: F001's balanced 10-game panel, unchanged except 25 → 15 min per game. That keeps all 10 slots busy, which is the concurrency the final run needs, in the same wall time 3 games would take.
  - FINAL: `demo_excluded_games = []` and `121*60`. The cell's own formula then gives concurrency 25, i.e. the `all` panel in which control scored 46.25.
- **Metadata:** F001's, with these changes:
  - kept: RtxPro6000, the competition, the pinned docker_image, pennyroyal-v253 and the taaf bundle;
  - `model_sources: []`, so the 190 GB model mounts are gone;
  - internet off;
  - kernel_sources: gguf-engine-build, gsq-coder-stage-p1..p3, gsq-coder-shard2-p1..p3, gsq-coder-mmproj-p1;
  - ids mellamoa/v4-gguf-smoke and mellamoa/v4-gguf-final.
- **Outputs:** serve.log, timings.json, smoke_gguf.json, llama_metrics.jsonl, the census lines, request logs, fable_layer_log.jsonl, benchmark.json and submission.parquet.

## 5. CPU tests (kaggle/tests; run niced, one suite at a time)
- **test_stage_gsq:**
  - shard2 splits into 3 parts, each ≤ 15 GB, on MiB boundaries; the mmproj is one part;
  - the slug refs and per-file `check_manifest` are right;
  - the shard-1 notebooks are byte-identical to the ones already pushed.
- **test_build_gsq_stage:** the shard2 and mmproj notebooks embed the right pins and slugs; metadata is CPU, internet on, no competition.
- **test_build_gguf_engine:**
  - `bash -n` passes on the script, and the notebook embeds it byte for byte;
  - metadata pins the image, has internet on and no GPU;
  - `--check-manifest` accepts the fixture and rejects glibc 2.38, a missing TopK kernel and a wrong commit.
- **test_gguf_serve:**
  - `launch_args` matches a golden for both smoke and final;
  - `install_engine` raises on a sha256 mismatch and when CUDA0 is missing;
  - `place()` on synthetic parts gets the names, sha256 checks and deadline right;
  - `wait_healthy` against a stub server: 503 then 200 passes, a process exit raises, the deadline raises;
  - the serve-log, props and metrics parsers pass on fixtures captured in §6.
- **test_context_error:**
  - the three llama.cpp messages, copied verbatim from b11476 with file:line in the test, match both the repo's `_is_context_length_error` and the anchor-edited F001 harness;
  - the SGLang and vLLM wordings still match, and unrelated 400s don't.
- **test_build_fable_gguf** (on the real F001 source):
  - only cells 4, 14 and 18 differ; cells 2 and 6 are identical; cell 14 compiles;
  - smoke and final differ only in cell 18, the id and `-cram`;
  - kernel_sources are in order, there are no model_sources, and enable_internet is false.
- **test_smoke_gguf:** the parallel and overflow checks against a fake OpenAI server. A pass passes, a hang fails within its budget, and wrong wording fails.
- The existing 290 tests must stay green.

## 6. Local validation (Mac, Metal, niced, one heavy job at a time; under 1 GB downloaded)
- **L1, build:**
  - cmake and ninja are not installed, so they go into the scratchpad with pip `--target` or `uv tool run`.
  - Configure: `nice -n 19 cmake -S runner-research/llama.cpp-b11476 -B scratchpad/lcpp-mac -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF`. That source is already at HEAD 988190680.
  - Build only llama-server with `-j4`; `--version` must show 9881906.
- **L2, model (737 MB, sha256-pinned from the HF API):**
  - unsloth/Qwen3.5-0.8B-GGUF: Qwen3.5-0.8B-Q4_K_M.gguf (532,517,120 B, bd258782…) and mmproj-F16.gguf (204,987,232 B, 56e4c6cf…).
  - It is the same tokenizer family as the Coder (pre qwen35) and uses the same XML tool-call format and vision path. Its hybrid layout (arch qwen35) is confirmed from its header.
- **L3, serve:** `gguf_serve.launch_args(local=True)` with the Kaggle flags, except:
  - its own `-m` and `--mmproj`, `-np 2` and `-c 32768`;
  - `--chat-template-file` set to the Coder's embedded template (sha256 EMBEDDED_TEMPLATE_SHA256), so llama.cpp picks the same parser and renders the same prompts as on Kaggle.
- **L4, protocol:** smoke_gguf (parallel 2, overflow), check_props, and the apply-template xhigh check. The protocol checks must pass; answers may only warn, since this is a 0.8B model.
- **L5, the real harness** on 2 offline games (sk48 and ft09, from the environment files in arc-prize-2026-arc-agi-3.zip), in a worktree venv (`uv sync --locked`):
  - Env is F001 cell 4's `setup_env`, read from the notebook with `ast` so it cannot drift. Base URL 127.0.0.1:8001, `ARC3_MAX_ACTIVE_STREAMS=2`, 6 min, request logs on.
  - `LOCAL_ANALYZER_CONTEXT_WINDOW=20480` against a 16,384-token slot, to force overflows.
  - Pass criteria:
    - at least 10 turns per game;
    - at least 1 python call executed per game;
    - reasoning_content is non-empty and is sent back in history;
    - every frame request carries an image, and `usage.prompt_tokens` grows by about 400 per 640 px image;
    - at least 1 overflow is recovered by trim and retry with the patch;
    - negative control: the unpatched harness logs that overflow as a failure;
    - no 5xx responses.
- **L6:** save the serve log, /metrics, /props and /slots samples as test fixtures, then delete the build directory and the model.
- **Not covered locally** (left to the GPU smoke): the sm_120a kernels, qwen4exp on the real weights, on-demand reads of the 28.8 GB table, and 10-slot memory and speed.

## 7. Push order for the main session (every push needs the user's approval)
1. **Local gate:** CPU tests green and L1–L6 passing; commits pushed to arm/runner-gguf; PR opened.
2. **Build the push dirs:**
   - `build_gguf_engine.py --out-root X`
   - `build_gsq_stage.py --file shard2 --parts 3 --out-root X`
   - `build_gsq_stage.py --file mmproj --parts 1 --out-root X`
3. **Push the CPU notebooks** (no GPU quota), the engine first because it fails in seconds if the image pin is refused: gguf-engine-build, gsq-coder-shard2-p1, -p2, -p3, gsq-coder-mmproj-p1. Don't re-push gsq-coder-stage-p1..p3; they are COMPLETE.
4. **Wait for COMPLETE and check the outputs:**
   - Fetch only the manifests and logs (`--file-pattern 'manifest\.json$'`, or `'\.json$|\.log$'` for the engine).
   - Run `--check-manifests` for shard2 and mmproj, and `build_gguf_engine.py --check-manifest` for the engine.
5. **Smoke run:** `build_fable_gguf.py --mode smoke` and `--mode final`, then push v4-gguf-smoke. About 35 GPU minutes; cell 14 stops the notebook itself on any failed gate.
6. **Read the smoke outputs** with gguf_report.py. Go to the final run only if all of these hold:
   - startup ≤ 30 min, and every cell-14 gate passed;
   - no CUDA error, GGML_ASSERT or server exit in serve.log;
   - HTTP errors under 2% of requests, and no analyzer timeouts;
   - tool calls parse on ≥ 90% of assistant turns that contain `<tool_call>`;
   - reasoning_content is present on ≥ 95% of thinking turns;
   - requests_deferred is about 0;
   - VRAM peak under 93 GB and RAM under 160 GB.

   Levels are recorded but not gated. If any check fails: fix, rebuild and smoke again.
7. **Final run:** push v4-gguf-final with no other GPU session running. About 15 min startup + 121 min of play + teardown, roughly 2.4 GPU-hours.
   - Fetch the outputs to runs/kaggle/v4-gguf-final/v1 and score with `inference.tools.eval`.
   - Run `make significance` against control 46.25. That comparison changes the model, the engine and the Fable layer at once, so it measures this stack and cannot attribute the difference.
   - Record the run in runs.json, arms.md (Worked or Didn't work) and the PR.

## Commits (one logical change each; docs in the same branch)
1. stage_gsq: name the staging notebooks per file (refactor)
2. Add the GGUF pins for shard 2 and the mmproj
3. build_gsq_stage: stage shard 2 and the mmproj
4. Add the llama-server build script for sm_120a
5. Add the engine build notebook generator with --check-manifest
6. Add gguf_serve: place, install, launch, checks, metrics
7. Add smoke_gguf with parallel and overflow checks
8. Recognise llama-server context-overflow errors (tool_agent.py)
9. Add the F001 smoke and final notebook builder for the GGUF runner
10. Add gguf_report
11. Add the local llama.cpp validation script and its fixtures
12. Docs:
    - README: a runner-gguf row;
    - kaggle/README: the GGUF notebooks and the push order;
    - serving/CUSTOM.md §8: the llama.cpp runner, its flags and the reasons for them;
    - CONFIGURATION.md: the overflow wordings (there are no new flags);
    - analysis/README: gguf_report.

The arms.md entry (outside the repo) moves from "in progress" to its result.

## Risks
- **Speed** is only projected: about 90–400 tok/s total, against SGLang W4A16's about 760. Fewer tokens per game means a lower score; the final run measures how much.
- **Multi-slot stability:** b11476's multi-slot sparse-attention cache code is days old (race fixes #29958 and #29994). The smoke run's 10-stream play is the test. Fallbacks, in order:
  1. `-np 8` with `ARC3_MAX_ACTIVE_STREAMS=8`;
  2. the Strata arm.
- **Coder quality:** HF discussions report repeat loops (#18) and weak agentic runs (#14). Nothing in this run mitigates that, because sampling is kept the same for comparability.
- **Pinned image:** whether Kaggle runs the pinned image in a CPU session is untested. It fails fast, and the fallbacks are in §1.
- **Tool-call edge cases:** llama.cpp's lazy grammar forces a wrong tool name to `python`, and a tool call cut off at max tokens is dropped as malformed. Check both in the smoke transcripts.
- **Kaggle input limit:** if Kaggle caps kernel_sources below 8, restage shard 2 in 2 parts.
