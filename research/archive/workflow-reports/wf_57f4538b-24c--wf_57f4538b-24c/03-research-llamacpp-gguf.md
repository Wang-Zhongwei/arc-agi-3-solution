# wf_57f4538b-24c: research:llamacpp-gguf

Workflow run `wf_57f4538b-24c`, phase Research. 

---

The 59.3 GB GGUF can be served on Kaggle's RTX Pro 6000 with llama.cpp and 10 slots of 139,264 tokens each, and the Fable test is buildable. It will run well below our SGLang W4A16 stack, though. Nothing was built or run on Kaggle; everything below is from source reading, GGUF headers, the release binaries and published reports.

No llama.cpp build or cmake/nvcc process is running on this machine (checked with `ps`).

## Verdict
- **It can run.** The arch is fully supported in tag **b11476**. Every quant type in the file has CUDA kernels, and the memory fits with f16 KV (about 73–85 of 95.6 GiB).
- **It will be slow.** Expect roughly 2–5× lower aggregate decode and 3–8× lower prefill than our SGLang (about 720 tok/s aggregate decode, 11–14K tok/s prefill). Fewer levels in the 25-min panel follows from that.
- **Correctness risk is medium.** The multi-slot sparse-attention cache code was rewritten on 10-01 and its last race fix landed on 10-06. Nobody has published a 10-slot CUDA validation of this arch.
- **The prebuilt release binaries don't run on Kaggle.** They need glibc 2.38 and GLIBCXX_3.4.32; the Kaggle image is Ubuntu 22.04.5 with glibc 2.35. So build from source.
- **This is not a competition path.** It's fine as a feature test.

## 1. Version support and GGUF header
- **Arch support:** PR #27742 merged 2026-08-27; the first tag containing it is b10660. The rewrite that fixed this arch, #29751, first appears in b11342. Pin **b11476** (2026-10-07, commit 988190680). It has the later sparse-attention, indexer, memory and race fixes: #28770, #29824, #29825, #29901, #29958, #29994, #29184.
  - v0.6.0 (10-05) is older than b11476.
  - conda-forge's latest build (11351) lacks everything after #29751.
- **Header (shard 1, read by range request):**
  - `general.architecture=qwen4exp`.
  - 48 blocks: 12 full-attention plus 36 Gated-DeltaNet layers.
  - Full attention: 24 heads, 2 KV heads, head dimension 256. The sparse-attention indexer has 4 heads, key length 128, top_k 2048, compress ratio 4.
  - 256 experts with 10 used per token; hyper-connections 4.
  - The n-gram table (PLE): n=3, 16 heads × 160 dimensions.
  - Native context 262144.
- **Split:** 2 files, 1224 tensors. The second shard is only the n-gram table `per_layer_token_embd`, IQ4_NL, 28.80 GB. `-m` on shard 1 loads both.
- **Tensor types:**
  - IQ4_NL 38.05 GB (28.80 GB of it is the n-gram table), IQ3_XXS 5.46, IQ2_S 5.37, IQ3_S 3.60, Q6_K 1.88, BF16 1.47, Q2_0 1.06, IQ4_XS 0.95, Q5_K 0.35, Q4_K 0.20 GB.
  - There are no IQ1_M tensors; "IQ1_M" is only the folder label. All of these types have CUDA kernels for both prefill and decode, Q2_0 included.
- **Chat template is embedded** (9,993 chars). It is an Unsloth-modified Qwen template, not the base one our Intel model ships:
  - It merges leading system/developer messages.
  - It maps reasoning effort "high" to "xhigh".
  - It raises an error on string tool arguments.
  - Pass the base template with `--chat-template-file` to match the SGLang runs.
- **mmproj:** `qwen3vl_merger`, BF16, patch 16 with 2×2 merge. That is 32 px per token, so a 640 px image is 400 tokens.

## 2. Getting a working binary on Kaggle
- **Official release assets** (`llama-b11476-bin-ubuntu-cuda-12.8-x64` and `-13.4-x64`, plus `cudart-*`): they include sm_120a code but need glibc 2.38. A fallback is to download Ubuntu 24.04's libc6/libstdc++6, run `ld-linux-x86-64.so.2 --library-path ... llama-server` from inside the binary directory, and set `LD_LIBRARY_PATH` to the cudart dir. That is untested.
- **Pip wheels:** llama-cpp-python is a binding without `llama-server`, and oobabooga's binaries stop at 2026-06-01, before this arch existed.
- **Recommended: build from source on the Kaggle machine.**
  - Use the CUDA 13.0.88 toolkit wheels from our `pennyroyal-v253` dataset. Our bundle build already compiles sm_120 with that nvcc, and Kaggle uses it with g++-11.
  - Compile for `120a-real` only.
  - Use `-DGGML_CUDA_CCCL_VERSION=v3.4.3`, as the official releases do. Without CCCL 3.4.3 or newer, the fast top-k path the sparse-attention indexer uses is compiled out.
  - Expect about 5–12 minutes. That is my estimate: a published guide gives 4–12 minutes on 8 cores for a single-arch build, and the Kaggle box is probably a 48-vCPU GCP g4-standard-48 (inferred from its 177 GB RAM, not verified).
  - Run the 59 GB download in parallel with the build, onto `/` (about 1 TB free; `/kaggle/working` is capped at 19.5 GB).

## 3. Server flags and memory
- **Slots and context:**
  - `-np 10 --no-kv-unified -c 1392640` gives exactly 139,264 tokens per slot (n_ctx ÷ slots).
  - A unified cache (`-kvu`) shares one pool across slots. It is the default only when `-np` is left on auto, which also means 4 slots.
- **Attention and KV cache:**
  - Keep `-fa on` with f16 KV. The sparse flash-attention gather that reads only the selected KV cells is enabled for head dimension 256 but only for unquantized K and V.
  - q8_0 KV falls back to dense attention: one report (#28734, A100) measured decode 24–37% slower at 128K–250K.
- **Tool calls and reasoning:** `--jinja` (now the default) is required. Without it, tool names come back empty. Use `--reasoning-format deepseek`, which returns `reasoning_content`, the field the harness echoes back.
- **Other flags:** `--alias flashnext --port 8001 -ngl all -fit off -b 2048 -ub 512 -ctxcp 16 --metrics`.
- **Endpoints:** `/health` (503 while loading, 200 when ready), `/metrics`, `/slots`, `/v1/models`.

**VRAM estimate:**

| Item | Size |
|---|---|
| GPU weights (shard 1 minus `token_embd`) | 27.3 GiB |
| mmproj | 0.85 GiB |
| f16 KV (30–33 KiB/token, #27901) × 1,392,640 tokens | 40–44 GiB |
| Recurrent state (computed 113 MiB/slot; #27901 measured 0.22 GiB at one slot) | 1.1–2.2 GiB |
| Compute buffer (unmeasured; grows with total cached context × ubatch) | 4–10 GiB |
| **Total** | **≈73–85 GiB of 95.6** |

- The n-gram table stays in host RAM (177 GB available) and is read lazily from disk; the fresh download should still be in the page cache.
- **If it OOMs at load:** use 10 slots × 112K (`-c 1146880`), or 8 × 139K.

## 4. Known bugs
- **#29166 (open PR):** it patches a function that #29751 removed, so it no longer applies. The related unified-KV "amnesia" bug (#27994) was fixed in #27941. I'd still use the non-unified cache.
- **#28019 (open):** the broken rollback is only armed for MTP, EAGLE3, DFLASH or DSPARK speculation, so it can't trigger without speculation.
- **#28286 (open):** MTP with more than one slot mixes content across slots. It doesn't apply here: the Coder has no MTP tensors, and the shared MTP head fails on it with "expert_count mismatch (model: 256, mtp: 512)" (HF Coder discussion #21).
- **Speculation:** don't enable n-gram speculation either. #28019 measured 50 vs 66–70 tok/s at 3 streams with it on.
- **#30078:** closed 2026-10-07. The reporter's own local patch caused it, not llama.cpp, so there is no known upstream tool-call bug with `--jinja`.
- **#27911, #27835, #27901 (open):** a CUDA grid-size crash under concurrency. The tensor reshape that triggered it is gone in b11476. Not re-verified on hardware.
- **Still open and worth watching:** #27388 (server stops generating mid-decode, generic CUDA) and #28280 (slot livelock during context checkpointing, seen on Vulkan).

## 5. Expected speed
**Published data points** (none is a 10-slot CUDA measurement):
- RTX PRO 6000, one stream: prefill 3867 → 4038 tok/s at 128K after #29901.
- RTX PRO 6000, one stream (#28734 goodbadwolf, MTP on, q8_0 KV): prefill 1983/1684/1201 tok/s at 9K/106K/351K; decode 46/31/16 steps/s.
- Batch scaling (#29335, M3 Ultra, 32K, `llama-batched-bench`): aggregate decode 29.6 / 38.2 / 43.8 tok/s at 1 / 2 / 4 sequences.
- 6×3090, 5 users at 58K (lukolszewski in #28734, 10-02 master): 12.4–15.6 tok/s each. Prefill halves once other slots hold context, because the indexer scores every slot's cached blocks for every token.

**My estimate for 10 slots on b11476 with f16 KV:** about 150–400 tok/s aggregate decode and about 1.5–4K tok/s prefill. This is a rough calculation from memory bandwidth plus the overhead of the indexer's host-side mask fill, not a measurement.

## Recipe
A full draft cell (untested) is in `llamacpp_serving_cell.py`; this is the core:
```
# CUDA 13.0 toolkit from the pennyroyal-v253 wheelhouse
uv venv --python $(python -c 'import sys;print(sys.executable)') /tmp/lcpp/venv
uv pip install --python /tmp/lcpp/venv/bin/python --no-index --find-links $WHEELHOUSE/wheels \
  nvidia-cuda-nvcc nvidia-cuda-crt nvidia-nvvm nvidia-cuda-runtime nvidia-cuda-cccl nvidia-cublas
uv pip install --python /tmp/lcpp/venv/bin/python cmake ninja
CUDA_HOME=/tmp/lcpp/venv/lib/python3.12/site-packages/nvidia/cu13
# then: lib64->lib, unversioned .so links, lib/libcuda.so -> driver (same as Fable cell 14)

# GGUF download in the background
python -c "from huggingface_hub import snapshot_download as s; s('ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF', local_dir='/tmp/lcpp/gguf', max_workers=16, allow_patterns=['IQ1_M/*.gguf','mmproj-*.gguf'])" &

# build
git clone --depth 1 --branch b11476 https://github.com/ggml-org/llama.cpp /tmp/lcpp/src
cmake -S /tmp/lcpp/src -B /tmp/lcpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DGGML_CUDA_NCCL=OFF \
  -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON \
  -DLLAMA_OPENSSL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DCUDAToolkit_ROOT=$CUDA_HOME -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=$(which g++-11)
cmake --build /tmp/lcpp/build -j$(nproc) --target llama-server llama-batched-bench

# serve
LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH /tmp/lcpp/build/bin/llama-server \
  -m /tmp/lcpp/gguf/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
  --mmproj /tmp/lcpp/gguf/mmproj-Qwen3.8-Flash-Next-BF16.gguf \
  --alias flashnext --host 127.0.0.1 --port 8001 -ngl all -fit off -fa on -ctk f16 -ctv f16 \
  -np 10 --no-kv-unified -c 1392640 -b 2048 -ub 512 -t 16 \
  --jinja --reasoning-format deepseek --chat-template-kwargs '{"preserve_thinking":true}' \
  --chat-template-file $MODEL_DIR/chat_template.jinja -ctxcp 16 -cram 8192 --metrics --no-ui -to 1800
```

**Notebook changes:**
- Kernel metadata: set `enable_internet` to true. Keep `pennyroyal-v253` (CUDA wheels) and the W4A16 model mount; it only supplies the base `chat_template.jinja`, and leaving it mounted keeps cell 4's input check from failing.
- Drop the drafter's precache.
- Raise `SERVER_STARTUP_TIMEOUT` to about 45 minutes. Otherwise the games are released while the server is still downloading and building, and burn their 25-minute budget.
- Before releasing the harness, send one python tool call and check for `finish_reason=tool_calls` with non-empty `reasoning_content`, then send one 640×640 image request.
- The harness's context-overflow detector doesn't match llama.cpp's wording ("exceeds the available context size"). It shouldn't trigger, because each slot is the 128K window plus 8K.
- `serve_report.py` parses SGLang logs only, so for llama.cpp read `/metrics` or the `print_timing` lines.

**Quality caveat (anecdotal):** users on the Coder repo report repeat loops (#18), "one too many experts stripped" (#11) and "zero tasks accomplished" on SWE-bench Verified (#14).

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/runner-research/`:
- `llamacpp-NOTES.md` – findings with source pointers
- `llamacpp_serving_cell.py` – draft replacement for Fable cell 14
- `shard1.kv.json` – GGUF metadata
- `shard1.tensors.tsv` – tensor list
- `chat_template.jinja` – the template embedded in the GGUF
- `tmpl_Qwen_Qwen3.8-Flash-Next.jinja` – the base Qwen template
- `fable_cell14.py` – the Fable serving cell as run
- `llama.cpp-b11476/` – b11476 source
- `gh/` – GitHub issue and PR text
