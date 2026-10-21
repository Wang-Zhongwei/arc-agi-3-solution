# gsq-coder-gguf-runner: research:strata-vs-llamacpp

Workflow run `wf_0ae4547f-80a`, phase Research. Serve the ISTA GSQ-RCO Coder GGUF the way its Hugging Face repo documents (llama.cpp or Strata, both shards, 256 experts, mmproj), validate the harness contract locally, build staging/engine/test notebooks and the 2-hour final-run notebook

---

**Recommendation: use Strata v0.1.40.3 (commit d5ea713), with `"parallel": 8` and the harness capped at 8 active streams. Keep llama.cpp b11476 as the fallback.**

Strata can serve the Coder GGUF on one RTX Pro 6000 at about 128K context, offline on Kaggle. It cannot run 10 streams at once, though: the engine holds at most 8. Neither engine has been measured at 8–10 streams with this model, and both should come in far below our SGLang W4A16 (about 785 tok/s at 10 streams, prefill about 10.4K tok/s). Nothing was built or run; all of this is from source, docs, GitHub and Hugging Face issues, the HF API and the Kaggle CLI.

## Can Strata serve it?
- **Coder support: yes.** The Coder is a built-in model family in Strata, pinned to the same revision as our staged shard 1 (5348543e, sha e11083ba…).
  - It ships an expert-ranking file for the 256 kept experts (`data/expert-profile-coder.bin`).
  - It runs the base model's 512-expert draft head against the 256-expert Coder, so speculative decoding works. llama.cpp refuses this pairing ("expert_count mismatch", HF Coder discussion #21).
  - Measured draft acceptance on the Coder is 0.62–0.79 (`bench/results/2026-09-28-coder`).
  - It loads shard 1 and reads shard 2 (the n-gram table) from disk through `--ple-gguf`, which is how the model card says to run it. The card names Strata as the engine users report good results with.
- **10 concurrent streams: no.** The engine caps parallel slots at 8 (`kVerifyMaxT = 8`, `generate.cpp:3799-3802`). Further requests wait in the server queue.
  - New prompts are read one at a time. While a long prompt is read, the other slots slow sharply: a 32K read left them at about 32 tokens/s combined (issue #1369).
  - With MTP drafting inside the parallel batch turned on (`--batch-mtp`), each slot uses 2 of the 8 rows, so only 4 slots run together.
- **128K context: yes.** One limit applies to every slot. Set `--max-context 147456` (131,072 plus the 12,288-token reply, plus slack). That is roughly 2.5 GiB per slot with 8-bit KV, about 23 GiB for 9 sessions. Total VRAM should be around 60 of 96 GB.
- **Offline on Kaggle: yes, but it has to be built from source.**
  - GitHub releases only have Windows binaries.
  - Build with nvcc 13.0, which our `pennyroyal-v253` wheels have. Avoid 13.2.0 and 13.2.1: they miscompile some kernels for this GPU (`docs/TROUBLESHOOTING.md:127`).
  - The engine also needs llama.cpp at commit 3cf03257 for its ggml code. It is C++20 and the docs list Ubuntu 22.04 as supported.
  - The build needs no GPU (the Dockerfile builds without one), so it can run in a CPU notebook. The competitor's Strata binary was built in the Kaggle image, which shows it compiles there.
- **API fit with the harness:** it returns `reasoning_content`, `tool_calls`, usage (including cached tokens) and `finish_reason`. It accepts `chat_template_kwargs` for thinking on/off and reasoning effort, and images as data URLs (a 640×640 image is about 400 tokens).
  - `preserve_thinking` is ignored, but the template keeps past thinking by default anyway.
  - `top_k` is capped at 64; we send 20.
  - A prompt plus `max_tokens` over the context returns a 400 that our context-error matcher won't recognise ("prompt (N tokens) + max tokens (M) exceeds the context (C)…"). Either add that string or start the server with `--fit-max-tokens`.

## Is it faster than llama.cpp?
| | Strata | llama.cpp b11476 |
|---|---|---|
| One stream, measured on RTX PRO 6000 | 130 tok/s without MTP, 258 with MTP (Unsloth Q4_K_XL, 8K, `bench/results/2026-10-06-community-rtxpro-v0140`) | 46 tok/s at 9K and 31 at 106K (#28734) |
| Multi-stream scaling, measured on other single GPUs | 0.87–1.3× its own one-stream speed (#857, #1121, #1118) | 1.48× at 4 streams (#29335); 5 users on 6×3090 got 62–78 tok/s total |
| Prefill | 5.7–5.9K tok/s at 32K on an RTX 5090 (#1369) | about 4K tok/s at 128K for one stream; halves once other slots hold context |
| Decode as context grows | Roughly flat | Drops (46 to 31 tok/s from 9K to 106K) |
| My projection for 8–10 streams | 200–330 tok/s total | 90–150 tok/s total |

Strata's own head-to-head shows it 3–5× faster at decode and 1.6–9× at prefill. That was on 12–16 GB cards that hold only part of the model, so it doesn't reflect our setup. The projections in the last row are mine, not measurements.

## Main risks
1. **Images with parallel slots are untested upstream.** `serve/test_parallel.py` has no image case. The server runs the image encoder while other slots keep decoding, and its own comment says doing that in single-request mode left a request stuck for good (`server.py:3015-3021`, `3254`). Open issue #1317 reports an intermittent hang reading the engine/vision pipe. Fallback: run the encoder on the CPU with `"max_tokens": 1024`, at about 3–4 s per image, one image at a time.
2. **The parallel code is very new.** A stall while reading a long prompt was fixed in 0.1.40.2 (#1118, released Oct 7). A crash when 7–8 slots decode together ("batch instantiate: out of memory", #997) was fixed in commit 3bcfd8f. As extra safety, pass `--vram-reserve-mib 1700`. Leave `--batch-mtp` off for the first run.
3. **There is no shared prefix cache across slots** beyond 6 RAM checkpoints, plus optional RAM "parking" of finished conversations (`--conversation-cache-mib N --conversation-cache-slots K`). Games that switch slots will re-read their prompts.

## The competitor's Kaggle use
- `clothespin` doesn't run Strata in the ARC competition. They run the unpruned Q2_0 on two T4s, exposed to the internet through a chisel reverse tunnel, at about 36 tok/s. Their binary is built for the T4 (sm75, CUDA 12.8), so we can't reuse it.
- No public notebook uses their datasets (`kaggle kernels list -s strata` and `--dataset` found nothing relevant).
- **Two of their files are reusable:**
  - **Shard 2:** `clothespin/strata-qfn-q2-0-assets` contains `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf`. HF metadata shows the same sha256 as the Coder's shard 2 (316b46f3…, 28,800,138,432 bytes) at both pinned revisions. The vision file (`mmproj-…-BF16.gguf`) is also identical across the two repos (b1a82259…).
  - **Draft head:** their `mtp_rt__*` files are exactly what Strata's setup builds for every model, the Coder included (q2_0 experts from Qwen revision de4b8e4d, the 106,299-token CJK draft vocabulary = 425,196 bytes).
  - Their README warns that sessions mounting large dataset sets sometimes never started. I'd prefer to stage our own copies.

## What to stage (nothing was pushed)
1. **Shard 1:** done (`mellamoa/gsq-coder-stage-p1..p3`).
2. **Shard 2 and the vision file:** stage with `stage_gsq.py`, about 4 parts in total.
3. **Draft head:** copy clothespin's 4 `mtp_rt` files, or rebuild them in a CPU notebook with `tools/mtp_fetch.py fetch` (about 5 GB of range reads), then `tools/mtp_pack.py --experts q2_0`, then `tools/mtp_rt.py`.
4. **Engine build (CPU notebook):**
   - Engine: `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_GGML_DIR=<llama.cpp@3cf03257>`
   - Image encoder: `tools/vision` with `-DSTRATA_VISION_CUDA=ON`.
   - Output the binaries, the `serve/`, `tools/` and `data/` folders, and llama.cpp's `gguf-py`. The engine links against libcudart and libcublas 13.
5. **On the GPU notebook:** build the model pack with `tools/iq_pack.py --gguf <shard1> --out <pack>` (seconds). Then start the server:
   - Command: `python -m serve.server --engine strata --config strata-coder.json --host 127.0.0.1 --port 8001 --fit-max-tokens`
   - Config: `"parallel": 8`, `"aliases": ["flashnext"]`, and a `vision` section.
   - Engine args: `--pack P --native SHARD1 --ple-gguf SHARD2 --expert-profile data/expert-profile-coder.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp RT --max-context 147456 --kv int8 --vram-reserve-mib 1700`
   - Add `--ple-io ram` if the notebook allows locking about 29 GB of RAM; otherwise keep the default disk reads.
   - The port only opens once the model is loaded, so the existing `/health` wait works; also check `"loaded": true`.
6. **Harness:** set `ARC3_MAX_ACTIVE_STREAMS=8`, and add the Strata wording to `_is_context_length_error` (or rely on `--fit-max-tokens`).
7. **Gate before the 2-hour run:**
   - Run `tools/batch_test.py --batch 8 --n 8`, which checks that 8 parallel greedy answers match their solo runs token for token.
   - Then send 8 concurrent requests, each with an image and a tool call, and require no hang.
   - Require an aggregate of at least about 200 tok/s. Otherwise switch to llama.cpp (flags are in `gguf-plan/llamacpp-gguf.md`).

Everything is in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/gguf-plan/research/strata/`:
- `src/` – Strata clone at d5ea713
- `coder-tree.json`, `base-tree.json` – HF file lists with sha256
- `ds/`, `ds-t4/`, `ds-lite/` – the competitor datasets' small text files
- `issues/{857,997,1118,1121,1317,1369}.txt`
- `disc-{6,18,21}.json` – HF Coder discussions
