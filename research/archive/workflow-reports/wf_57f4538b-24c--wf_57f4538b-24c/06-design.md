# wf_57f4538b-24c: design

Workflow run `wf_57f4538b-24c`, phase Design. 

---

Design: a custom GSQ-RCO Coder runner for Pennyroyal, plus the F001 test notebook. Nothing below has been built or run yet.

Repo root is `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol`. The branch is `arm/runner-gsq`, cut from `arm/model-profiles` (PR #68), in a new worktree `<scratch>/wt/runner-gsq`. `<scratch>` is `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad`.

## 0. Verdict
- **Custom runner is feasible.** It is not llama.cpp. It is Pennyroyal d00d88ef plus three runtime patches pinned to that commit, a 256-expert checkpoint converted from the GGUF, and its own profile.
- **What changes.** The routed experts are served from GSQ's own 4-bit codes, losing only BF16 scale rounding. A 16-entry lookup table is added to Pennyroyal's Marlin MoE kernel, which is compiled on the GPU at startup.
- **What stays.** Everything else is the exact BF16 base, taken from the mounted Intel model.
- **Gains.** About 26 GiB of GPU memory is freed. GSQ's quality is tested on its real weights. It is not a speed win by itself (see section 10).
- **GLM: not built** (section 7).

## 1. Weight path
**Format `gsq_lut`, safetensors.** The code is lifted from `<scratch>/runner-research/gsq/convert_gsq.py`.
- Per expert: `{gate,up,down}_proj.qweight` holds int32 4-bit codes packed like GPTQ along K, so the stock `gptq_marlin_moe_repack` works. `.scales` is BF16 `[K/G, N]`, stored as Marlin uses it.
- Per-layer tables and group sizes live in `config.quantization_config.layers`.

**Encoding by GGUF type.**

| GGUF type | Matrices | Served as | Group | Bits/weight | Error vs GGUF |
|---|---|---|---|---|---|
| IQ4_NL, IQ4_XS | 41 | LUT | 32 | 4.5 | BF16 scale rounding only, measured 0.16–0.17% |
| IQ3_S, IQ3_XXS | 54 | LUT | 32 | 4.5 | same |
| Q2_0 | 9 down | stock `uint4b8` (exact) | 64 | 4.25 | same |
| IQ2_S | 40 gate/up | LUT | 16 | 5.0 | same |
| IQ2_S fallback | 40 gate/up | int8 | 32 | 8.5 | about 0.34% (my estimate) |

- **IQ2_S fallback.** It is used only if the GPU self-test fails the 16-wide-group BF16-scale kernel. It merges the two 16-weight sub-scales at load time and costs +7.3 GB of experts.
- **Experts total** 34.9 GB (4.62 bits per weight) against Intel's 62.76 GB.

**Error budget.** The converter checks every matrix against llama.cpp-exact dequantisation and aborts above the limit. Results go into `gsq_manifest.json`.
- At most 0.25% relative Frobenius error for the LUT and Q2_0 matrices.
- At most 0.5% for the IQ2_S int8 fallback.
- Router: bit-exact.

**Everything except the experts is BF16 from Intel's checkpoint.** That covers dense, GDN, QSA, HC, shared expert, embeddings, `lm_head`, vision tower, tokenizer and the base chat template.
- This is strictly closer to BF16 than the GGUF, whose dense tensors are Q4_K–Q6_K (about 2% error) and whose n-gram table is IQ4_NL (7–9% error).
- So the served model is "GSQ-RCO experts and mask on the exact base". It is not bit-identical to what ISTA evaluated.

**PLE (n-gram) table.** The 102.4 GB BF16 `ple_embedding.ngram_embedding.shard_*` tensors are linked from Intel's mounted shards through a view built at serve time.
- The converter checks that those shards hold only PLE tensors. In the HF layout they do (`model-00016`). Franzen's Kaggle copy is resharded and unchecked.
- If they are mixed, `--embed-ple` copies the table into the model (+102 GB).

**Where conversion runs.**
- **Test:** inside the test notebook, with internet on. It downloads only GGUF shard 1 (29.61 GB, HF revision `5348543e`, Apache-2.0, not gated). Shard 2 and the mmproj are not used.
- **Submission:** once, in a conversion notebook, then published as a Kaggle Model.
- **Speed:** numpy encodes at 56M weights/s per core, so about 1–2 minutes on 16+ processes under `nice`. Output is about 45.6 GB to `/tmp`.

**Mode `SGLANG_GSQ_KERNEL=int8`.** The same checkpoint can be expanded to int8 g32 at load time and run on stock Marlin with no CUDA change. That is a bring-up and correctness path (62 GB experts, slower) and the automatic fallback.

## 2. qwen4exp mapping (`serving/convert/gsq_coder/convert.py`)
- **Experts.** `blk.L.ffn_{gate,up,down}_exps[:,:,j]` maps to `model.language_model.layers.L.mlp.experts.{j}.{gate,up,down}_proj.{qweight,scales}`, for j in 0..255.
- **Mask.** `mask[L]` comes from matching the rows of the GGUF `blk.L.ffn_gate_inp` (BF16, 256×2560) bit-for-bit against base `mlp.gate.weight` (512 rows). The converter asserts one match per row and ascending order, and writes the sliced base rows as the router.
- **Config.** `text_config.num_experts=256`. Pennyroyal's `qwen4_exp.load_weights` builds its expert mapping from `config.num_experts`, so no model-code patch is needed.
- **Dropped.** All other GGUF tensors (attention, SSM, HC, shared expert, indexer, norms, `token_embd`, `output`, `per_layer_token_embd`, mmproj) are not read. GDN V-heads are permuted in the GGUF (head i is base head 3·(i%16)+i//16); that matters only for the spot-check. Intel's `mtp.*` and its 512-expert tensors are dropped too.
- **Copied byte-identical:** `tokenizer.json` (sha `06b95…`), `tokenizer_config.json`, `chat_template.jinja` (Intel's, not the Unsloth template embedded in the GGUF), `preprocessor_config.json` and `generation_config.json`.

## 3. Runtime patches (profile-scoped; pinned to d00d88ef; idempotent; fail loudly)
- **R0 (refactor commit, no behaviour change).** New profile key `runtime_patches` lists files under `serving/runtime_patches/<subdir>/`, and only that profile's notebooks embed them. Root-level patches still go to every arm.
  - So W4A16 notebooks, their goldens and their JIT-cache config hash stay byte-identical.
  - Each script checks the target file's sha256 equals the d00d88ef file, or that its own marker is present (then it does nothing). Otherwise it exits 1, and the launcher already stops on that.
- **`gsq/01-gsq-lut-quant.py` (Python).** Adds `layers/quantization/gsq_lut.py` and one registry line.
  - `GSQLutConfig`: linear layers stay unquantized BF16; FusedMoE gets `GSQLutMoEMethod`.
  - `create_weights` reads the per-layer types and group sizes.
  - `process_weights_after_loading` runs the repack, permutes the scales, and expands one layer at a time to int8 in int8 mode.
  - `apply` calls `fused_marlin_moe` with a type and table per GEMM.
- **`gsq/02-marlin-moe-per-gemm-type.py` (Python).** `fused_marlin_moe` gets the optional arguments `w1_q_type`, `w2_q_type`, `w1_lut` and `w2_lut`.
  - Today one `num_bits` drives both GEMMs (`fused_marlin_moe.py`, `get_scalar_type` ×2).
  - With the defaults set to None, the call is unchanged.
- **`gsq/03-marlin-moe-lut-kernel.py` (JIT CUDA source).**
  - `marlin/dequant.h`: adds `dequant_lut()`. It does a 16-entry byte lookup with `__byte_perm` (llama.cpp's `get_int_from_table_16`), then reuses the existing `uint8b128`-to-BF16 conversion. Element order matched stock on 24K random words in the numpy emulation.
  - `marlin_moe/marlin_template.h`, `kernel.h` and `moe_wna16_marlin.cuh`: a compile-time `kHasLut` flag, added to the JIT module key next to `kIsEP`/`kHasBias` in `ops/moe/moe_wna16_marlin.py`, plus a `uint32x4` LUT argument.
  - Instantiates `kU4B8` with LUT at group blocks 2 (g32) and 1 (g16, BF16 scales). The g16 case is the main kernel risk.
  - Instantiations without the LUT keep the same template source.
- **No patch** to the qwen4_exp model code, the scheduler, or the drafter (the albucino view keeps its own 512-expert config).

## 4. Engine bundle: `mellamoa/pennyroyal-v253-gsq`, an overlay on `dfranzen/pennyroyal-v253`
- **No ahead-of-time compiled code changes.** sgl-kernel and FlashInfer are untouched, and Marlin MoE is compiled on the GPU from source shipped in the sglang wheel. So we don't rerun the 1,660-line source build.
- **`serving/build_engine_gsq.sh`** runs in the Kaggle GPU image or an internet-enabled notebook. It needs no GPU and takes minutes. It:
  1. checks the base lock and sglang wheel sha256;
  2. runs `wheel unpack`, then applies `runtime_patches/gsq/*.py` to the unpacked tree (the same scripts);
  3. sets the version to `0.5.19+gd00d88efc8d6.gsq1`;
  4. runs `wheel pack` with `SOURCE_DATE_EPOCH`, so the output is reproducible;
  5. writes `requirements.lock` (base lock with the sglang line swapped), `manifest.json` (base lock sha, patch shas, builder git sha) and `SHA256SUMS`;
  6. installs offline into a fresh venv with `--no-index`, pointing at both the overlay and base wheels;
  7. imports sglang, checks the markers, checks that every profile flag appears in `sglang serve --help` (as the user's `verify_vllm_flags.py` does), and checks that `--quantization gsq_lut` resolves.
- **`serving/tools/verify_gsq_engine.py`** runs on the GPU, in the bench or test notebook before launch. It:
  - checks sm120 and the markers;
  - tests `fused_marlin_moe` with the LUT against a torch dequant-then-matmul reference, per type, for M in {1, 10, 40, 64, 8192}, E=256, top-10; the bound is 2e-2 relative, and LUT against int8 must agree within 1e-2;
  - picks the IQ2_S mode, prints the JIT compile time, and writes `verify_gsq_engine.json`.
- **Launcher.** `install()` accepts the overlay dir (profile `inputs.engine`). When the wheel already carries the markers, the runtime patches do nothing.

## 5. Model bundle: Kaggle Model `mellamoa/qwen3.8-flash-next-gsq-rco-coder`, Transformers, instance `lut4`
- **Converter.** `serving/convert/gsq_coder/` contains `gguf.py` (header and tensor-offset reader with range reads), `blocks.py` (decoders, including Q2_0 which gguf-py lacks; the ggml grids are vendored with a sha check against b11476), `convert.py` and a README.
  - CLI: `convert.py --gguf <shard1> --base <Intel dir> --out <dir> [--procs N] [--embed-ple]`.
  - It writes `gsq_manifest.json`: sources, revisions, GGUF header sha, mask, per-layer types, error stats, the PLE link list and the converter's git sha.
- **Checker.** `serving/tools/check_checkpoint.py` learns `gsq_lut`:
  - per-layer types and group sizes in {16, 32, 64};
  - `num_experts` equals the router rows (256);
  - exact tokenizer and template shas;
  - PLE links resolve against `--base`;
  - GPU-bytes and KV-pool estimates for both lut and int8 modes.
- **Notebook.** `kaggle/build_bundle_notebooks.py --convert` builds the conversion notebook (internet on, GPU shape for the ~1 TB `/` and 48 vCPU, which is an inference). It runs convert, then check.
  - Its last cell is gated by `PUBLISH=0`. With `PUBLISH=1` and Kaggle secrets it runs `kaggle models create` and then `kaggle models instances create`.
  - `/kaggle/working` (19.5 GB) can't hold the 45.6 GB output, so the upload has to happen inside the notebook.

## 6. Profile `serving/profiles/qwen38fn-gsq-coder.json`
- **Inputs.** `model` is the published `lut4` instance; it stays TBD, so pushes are refused, until it is published. New `inputs.base` is the Intel W4A16 mount, used for the PLE link and the tokenizer check. `draft` is albucino. `wheelhouse` is pennyroyal-v253, plus `inputs.engine` for the overlay (optional).
- `validate`: `{quant_method: ["gsq_lut"]}`.
- `args`: `quantization gsq_lut`, `qwen3` / `qwen3_coder`, `preserve_thinking`, `chat_template.jinja`. No JSON override; `num_experts` is already in config.
- `hybrid` and `spec` are the same as W4A16: NEXTN, albucino, compressed-tensors draft, FP8 draft KV.
- `frspec` uses the same map (`becfa41d…`) and tokenizer sha, with `on_mismatch: raise`.
- `runtime_patches` lists `gsq/01`, `02` and `03`.
- `serve_env`: `{SGLANG_GSQ_KERNEL: "auto"}`. `auto` means the lookup-table kernel if `verify_gsq_engine` passes, otherwise int8; the choice is printed and saved.
- `cfg` is `{}`. To compare directly with F001 v2, this keeps:

  | Setting | Value |
  |---|---|
  | `CTX` | 136K |
  | `MAXREQ` / `CUDAGRAPH_MAXBS` | 10 / 10 |
  | KV cache | FP8 |
  | `MEMFRAC` | 0.96 |
  | `MAMBA_CACHE` | 60 |
  | `CHUNK` / `MAX_PREFILL` | 8192 / 16384 |
  | `SPEC_STEPS` | 3 (4 draft tokens) |
  | Accept thresholds | 1.0 |

- `startup_timeout_s`: 1080, because the LUT kernel compiles at startup.

**KV and Mamba at 10 streams × 136K (1.36M tokens worst case):**
- **LUT mode:** GPU weights are 42.5 GiB, so the FP8 pool is about 3.11M tokens (2.3× worst case) and nothing can be retracted.
- **IQ2_S int8 fallback:** 49.3 GiB of weights, pool about 2.56M.
- **int8 mode:** 69.8 GiB, pool about 0.91M (today 1.01M).
- **Mamba:** 60 slots, about 6 per stream (Gabriel: below that, the speculative graphs silently cap at 4). Peak use was 0.67.
- **Later knobs:** BF16 KV (1.55M also fits) and 12–14 streams. Neither is in the first test.

**Kept from competitors** (all already in our stack):
- Gabriel 0003 FP32 prefill state; 0004 low-M BF16 GEMM.
- RecoverSSM and the WY verify path.
- FP8 QSA KV (Pennyroyal / mratsim 0001).
- KV-budget fix (mratsim 0002).
- Sampler all-reduce skip (mratsim 0006); grammar warmup (0007).
- `expandable_segments` (mratsim).
- FR-Spec 64K (+9.3%).
- NEXTN 3/4: QSA caps draft tokens at 4.
- `--moe-runner-backend auto`, which means Marlin with no autotune. mratsim's `flashinfer_cutlass` autotunes for about 20 minutes on every new container.

**Not adopted:**
- Online MXFP8: −27% at batch 10, because of its 32- and 16-row caps.
- Relaxed acceptance 0.5/0.3: lossy at temperature 0.7, so it needs its own arm.
- NVFP4 experts: −22.9%.
- `extra_buffer_lazy`, HiCache.
- `SGLANG_FORWARD_UNKNOWN_TOOLS`: it changes tool behaviour against F001 v2.
- Next engine step (v2): port Gabriel 0005/0006 (FP8 weights for dense, HC mix and `lm_head`) with the row caps raised to 64. Estimated +8–15%; it needs a bench gate.

**Runner tools:**
- `serving/tools/smoke_gsq.py`, in the style of `check_flash_next_smoke.py`: text 2+2, a red 128 px image, one tool call, and one harness-shaped call that must return `finish_reason=tool_calls` with non-empty `reasoning_content`, plus a 640 px image. It records timings and time-to-first-token.
- `kaggle/bench.json` bench `gsq-coder`: G0 (int8) and G1 (lut), at batch sizes 1, 4, 8 and 10, reporting startup, tok/s and accept length. G1 also runs `--expert-distribution-recorder-mode stat` to measure real routing skew.
- Optional control profile `qwen38fn-w4a16-codermask`: Intel's own INT4 experts restricted to the Coder mask, on the stock path. It separates the effect of pruning from GSQ's quantization.

## 7. GLM-5.3-Flash: not built
- **The user's 4bpw checkpoint doesn't fit.** It is 175.6 GB (163.6 GiB) against 95.6 GiB.
- **Offload is blocked by its format.** ExLlamaV3 offloads only `mul1` experts ("experts are not mul1, CPU offload skipped"). Offload builds that do run measured 17.6–30.6 tok/s on one stream.
- **The one build that fits is worse.** turboderp's 2.05 bpw is 79.4 GiB, with mean KL 0.19 against NVFP4's 0.045. It runs 93.65 tok/s on one stream; I estimate 150–400 tok/s at 10 streams, against our 744 median. Forum users report loops.
- **Conclusion.** A run would mix model, quantisation damage and a 2–5× token shortfall. Not credible on one card. Already recorded under Not pursuing in `arms.md`; CUSTOM.md will say the same.

## 8. F001 test notebook (`<scratch>/push-fable-f001-gsq/`, id `mellamoa/v4-fable-f001-gsq`)
- **Generator.** `kaggle/build_fable_gsq.py --source <scratch>/push-fable-f001/v4-fable-f001.ipynb --out <scratch>/push-fable-f001-gsq` pins the source sha256 prefix `86f9cc40b371d9e6`.
  - Cells 0–3, 5–13 and 15–24 stay byte-identical, including harness patch cell 2, Fable cell 6, precache cell 8, the balanced panel, 25 minutes and concurrency 10 in cell 18. Builds refuse any other diff.
- **Cell 4.** Only `SERVER_STARTUP_TIMEOUT` changes, from 12 to 45 minutes. `MODEL_DIR` stays the Intel mount, which becomes the base.
- **Cell 14** is replaced by our launcher (`launch_sglang.py` with this profile and the gsq patches) and a new embedded `serving/kaggle/acquire_gsq.py`. Stages, each with its own guard; any overrun raises instead of releasing the games:
  1. Download GGUF shard 1, in parallel with the wheelhouse install: budget 15 minutes.
  2. Convert and check: budget 12 minutes. A precache thread warms the Intel PLE shards and the drafter, not Intel's 512 experts.
  3. Patches and `verify_gsq_engine`, which picks lut or int8.
  4. Launch and `/health`: budget 25 minutes.
  5. `smoke_gsq`.
- **Timings.** The cell prints download, convert, verify, load (taken from serve.log startup timings) and first-token, and writes `timings.json`.
- **Logs.** `serve.log` (already in `/kaggle/working`), `convert.log`, `gsq_manifest.json` and `verify_gsq_engine.json`.
- **Kernel metadata.** Same Docker image sha, `NvidiaRtxPro6000`, the same two datasets and two models (Intel is needed as the base), and `enable_internet: true`.

## 9. Tests (local, `nice`, run one at a time)
- **`convert/tests/test_codec.py`:**
  - Synthetic random blocks per type: codes×table×scale equals the decoder with FP32 scales, and stays within budget with BF16 scales.
  - The 12 real sampled experts, as a small committed fixture plus a skippable HTTP range test: within budget.
  - GPTQ pack/unpack round trip; Marlin element order equals stock in the emulation.
  - Decoders cross-checked against gguf-py where it has the type.
- **`test_namemap.py`:**
  - All 1,224 GGUF tensors are classified (expert, router, base-sourced, unused), with zero unknown.
  - Every one of Intel's 224,280 keys is copied, linked or dropped, with zero unknown.
  - The output weight map equals Intel's keys minus Intel's experts minus `mtp.*`, plus 48×256×3×2 expert tensors.
- **`kaggle/tests`:**
  - New golden argv for `qwen38fn-gsq-coder`.
  - The W4A16 golden and notebook stay byte-identical, with no gsq patch embedded.
  - `check_profile` accepts `runtime_patches`, `inputs.base` and `inputs.engine`.
  - Each patch applies to vendored d00d88ef copies (sha-pinned), does nothing on a second run, exits 1 on a changed anchor, and the patched Python compiles.
- **`test_build_fable_gsq.py`:** only cells 4 and 14 differ, the metadata is right, the guards are present, and `python3 <scratch>/nbcheck.py <notebook>` reports zero errors.

## 10. Expected throughput and quality against our W4A16 (estimates, none measured)
**Today (F001 v2):** 760–785 tok/s at batch 9–10, accept length 2.64, 599 tok/s averaged over the whole job.

**LUT mode.**
- Expert bytes per verify step (M=40) fall from 34.2 to 27.8 GB with uniform routing. That is about −14% step time at most.
- The saving shrinks with routing skew, and per-expert bytes are +8% (4.62 against about 4.28 bits per weight).
- The albucino drafter was trained on the full model, so I expect accept length 2.3–2.5.
- Net estimate: about −15% to +8%, central about −5%, roughly 650–850 tok/s.
- MoE prefill kernel time is about +10–20%.

**int8 mode:** about −30% to −37%, roughly 490–560 tok/s.

**Quality: likely below the W4A16; nothing measured.**
- GSQ's experts are 1.4–4× further from BF16 than Intel's, measured per weight (IQ2_S layers 27% against 11%), though GSQ optimises output KL instead.
- The 512 → 256 prune used a coding calibration set, and ARC games are outside that domain.
- History: our experts16 arm lost 7.6 points from routing drift. On the Coder repo, users report repeat loops (#18) and zero tasks solved on SWE-bench Verified (#14).
- Read the result at matched tokens against F001 v2 (29.93, 32 levels) with `make significance`. A single 10-game run is noisy.

## 11. Risks
1. The g16 BF16-scale Marlin instantiation for IQ2_S may not work. Fallback is int8 g32 for those 40 matrices (+7.3 GB, still about lossless).
2. The LUT element order is only emulated, not run on a GPU; `verify` gates it and int8 is the fallback.
3. Accept length could fall below 2.2. Then add bench G1-s2 with `SPEC_STEPS=2`.
4. HF download speed from Kaggle is unknown; the 15-minute guard covers it.
5. Franzen's resharded Kaggle copy may mix PLE tensors with others in one shard. Then the converter copies the table (`--embed-ple`).
6. Cold start: about 146 s graph capture plus LUT JIT compilation. A new JIT-cache snapshot is needed, since its config hash changes.
7. Publishing 45.6 GB needs Kaggle secrets inside a notebook.
8. Sliced-router softmax over 256 experts should equal llama.cpp's behaviour; `norm_topk_prob` comes from the config.

## 12. Commits and the commands that need your approval
**Commits, in order:**
1. R0: profile-scoped runtime patches (refactor; goldens unchanged).
2. The GSQ codec and GGUF reader, with tests.
3. The converter and manifest.
4. The `gsq_lut` quant method patch (int8 mode).
5. Per-GEMM quant types for `fused_marlin_moe`.
6. LUT dequant in the Marlin MoE JIT kernel.
7. `check_checkpoint` support for `gsq_lut`.
8. The profile, base input, PLE view and goldens.
9. The engine overlay builder, verify script and lock.
10. The conversion notebook generator.
11. The smoke probe and bench `gsq-coder`.
12. The F001 GSQ notebook generator.
13. Docs: `serving/CUSTOM.md` §7 "Custom bundles" (build, publish, run), `serving/bundles/qwen38fn-gsq-coder/README.md`, `profiles/README.md`, `kaggle/README.md`, the top-level README, and the `arms.md` entry `runner-gsq` under Not yet implemented.

**Effort:** about 3 days to a runnable int8 path, plus 2–4 days for the LUT kernel.

**Prepared commands; none run without your approval:**
- **Test (needs no publishing; run first):** `kaggle kernels push -p <scratch>/push-fable-f001-gsq`, then `kaggle kernels output mellamoa/v4-fable-f001-gsq -p runs/kaggle/v4-fable-f001-gsq/v1`.
- **Bench (optional, only if the test's numbers are unclear):** `python kaggle/build_notebook.py --bench gsq-coder --push`.
- **Engine:** `kaggle kernels push -p kaggle/out/gsq-engine-build`, then `kaggle kernels output mellamoa/gsq-engine-build -p /tmp/gsq-engine`, then `kaggle datasets create -p /tmp/gsq-engine/bundle-gsq-overlay`.
- **Model:** add the `KAGGLE_USERNAME` and `KAGGLE_KEY` secrets in the notebook's UI, then `kaggle kernels push -p kaggle/out/gsq-coder-convert` with `PUBLISH=1`. That runs `kaggle models create` and `kaggle models instances create -p /tmp/gsq/model` inside the notebook.
