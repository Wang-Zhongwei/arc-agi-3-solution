# Plan: Lord's model on Daniel's SGLang

2026-10-06. Follows [NVFP4_FOLLOWUP_EXPERIMENT.md](NVFP4_FOLLOWUP_EXPERIMENT.md).

## Where we are

The first swap (Kaggle `lord-nvfp4-on-pennyroyal` v1 vs `pennyroyal-w4a16-baseline` v1) changed only the model. Lord's model decoded 11–17% slower at every stream count (633 vs 723 tok/s at 10 streams; same draft acceptance) and loaded weights 3.5× slower (1,432 s vs 413 s).

The engine is fine; its tuning is for Daniel's model. Daniel's patches speed up BF16 layers and INT4 Marlin experts. Lord's model stores attention in FP8 and experts in NVFP4, so those layers run on untuned paths.

## Daniel's patches against Lord's model

| # | Patch | Targets | Daniel's model | Lord's model | What to do |
|---|---|---|---|---|---|
| 1 | Low-M BF16 GEMM | Small BF16 matmuls (≤ 32 tokens) | Speeds up all attention projections | Mostly bypassed: attention is FP8 | Give FP8 attention a small-batch path; widen the BF16 path to 56 for the layers still in BF16 |
| 2 | Spec-state memory budget | Draft-head memory | Applies | Applies | Keep |
| 3 | Marlin scale dtype | Marlin expert loading | Experts on Marlin W4A16 | Experts on FlashInfer CUTLASS NVFP4 | Run NVFP4 experts on Marlin (the fork has it, but not for Lord's file format) |
| 4 | Prefix-cache retention | Cache policy | Works (94.5% hit) | Works (93.3% hit) | Keep |
| 5 | Prefetch lookahead | Startup loading | On | Off; Lord's prefetch window used instead | Loading looks CPU-bound, not disk-bound: find the slow step; start games only after the server is ready |

## Order of work

Fix the target at **14 streams**, as Lord runs. His model's real advantage is KV capacity: 1.42M tokens on vLLM against Daniel's 1.01M. On SGLang it got only 0.92M so far, because `embed_tokens` stays on the GPU and memory is budgeted Daniel's way. So KV room comes first, then every patch is tuned for 14 streams. With 4 verify tokens per stream, small batches now reach 56 tokens, above the current 32-token limit.

1. **KV room for 14 streams.** Move `embed_tokens` to host memory (Lord's trick) and retune the memory fraction. Raise the stream count from 10 to 14 everywhere it is set: the harness gate (`ARC3_MAX_ACTIVE_STREAMS`) and the server (`MAXREQ`, `CUDAGRAPH_MAXBS`).
2. **Experts on Marlin** (replaces what patch 3 did for Daniel). Experts are most of the weights read per step.
3. **Small-batch paths for 14 streams** (patch 1, retuned): an FP8 path for attention, and the BF16 path widened for the layers still in BF16. Raise the limit from 32 to 56 tokens and tune tile settings for 56 only (all 14 streams verifying). Capture a CUDA graph at 14 so batches aren't padded to 16. Smaller batch sizes (48, 52) and larger concurrency come later, if 56 pays off.
4. **Loading** (patch 5, retuned): fix the slow step, and start games only after the server is ready.

All four changes go into one run, at 14 streams, compared against the v1 numbers. Each change has its own config flag, so any one can be switched off later to measure it alone.

## Implementation

Config [`lord-nvfp4-pennyroyal-tuned/v1.json`](../../ARC3-Inference/configs/lord-nvfp4-pennyroyal-tuned/v1.json): the v1 candidate plus the settings below. Nothing has been timed on a GPU yet: tile settings are chosen by rule, and the self-checks below guard correctness.

The launcher patches the installed wheel ([`pennyroyal_patches.py`](../../ARC3-Inference/inference/framework/pennyroyal_patches.py)) before starting the server. Each edit matches one exact anchor in Daniel's d00d88e + five-patch source; if any anchor is missing, nothing is changed and the server runs untuned. Each change is behind its own environment flag. If the tuned server exits before it is ready, the launcher starts it again once without the flags. `pennyroyal-startup.json` records which happened (`tuneup`).

| Step | Setting | What it does |
|---|---|---|
| 1. KV room | `EMBED_TOKENS_HOST`, `MEMFRAC` 0.97, `MAXREQ`/`CUDAGRAPH_MAXBS` 14, `ARC3_MAX_ACTIVE_STREAMS` 14 | `embed_tokens` goes into pinned host memory and is read by the PLE gather kernel; the draft shares it. Frees ~1.2 GiB, plus ~1 GiB from the memory fraction. CUDA graphs: 1, 2, 4, 7, 8, 9, 10, 14. |
| 2. Experts on Marlin | `MOE_MARLIN` | The compressed-tensors NVFP4 MoE scheme gets a Marlin branch: gate/up loaded gate-first, global scale inverted to ModelOpt's `weight_scale_2`, then the fork's `prepare_moe_nvfp4_layer_for_marlin`. The first MoE layer is checked against a dequantized reference (two experts); a mismatch stops the server. |
| 3. Small batches | `LOWM_FP8`, `LOWM_MAX_M` 56 | FP8 attention and linear-attention projections with ≤ 56 rows use a new weight-only split-K Triton GEMM (BF16 activations, FP8 weights widened in registers, per-row scale). This skips the per-token quant kernel. Daniel's BF16 path goes up to 56 with 64-wide tiles above 32 rows. Each shape bucket is checked once against a dense reference outside graph capture; a bucket that fails uses the stock path. |
| 4. Loading | `server_startup_timeout_s` 3600 | Games start only when the server is ready (v1 released them at 720 s, and loading slowed to ~60 s per shard). If no server came up, the setup step fails instead of starting games. Expert names are looked up by key instead of scanning 1,536 mapping entries per tensor. |

What the logs show about loading: an expert shard (1.3 GiB, 6,144 tensors) takes ~14 s even when its prefetch finished long before. That is ~84 MB/s against Daniel's ~0.7 GB/s, so the cost is per tensor and on the CPU.

## Tuned v1 result (2026-10-06)

Kaggle `lord-nvfp4-pennyroyal-tuned` v1 (commit 61be503) never served a request:

1. **Marlin self-check failed.** After the full load (1,389 s, no faster than v1), the first MoE layer's Marlin output was effectively zero against the dequantized reference (relative error 1.0). The server stopped as designed. The cause is not yet known; the scale handling matches both the fork's dense NVFP4 Marlin test and ModelOpt's Marlin path.
2. **The untuned fallback segfaulted** (exit -11) 2 minutes into its load, while the stack-dump watchdog was writing (the dump ends mid-frame). `faulthandler.dump_traceback_later` reads other threads' frames without the GIL; the option is removed.
3. **The launcher still released the games**, which all failed to connect; the notebook was cancelled.

The stack dumps (94 samples over the tuned load) located the slow loading steps: ~47% in per-expert host-to-GPU copies (`FusedMoE._load_w13` / `_load_w2`), ~34% in copying FP8 PLE rows into the pinned table (`copy_ple_rows_to_tp_embedding`). The expert-name lookup was not the bottleneck.

## Marlin root cause

The bisect ([`pennyroyal_marlin_debug.py`](../../ARC3-Inference/inference/framework/pennyroyal_marlin_debug.py), RunPod) passed all 23 cases: dense and MoE NVFP4 Marlin, synthetic and layer 0 of the real checkpoint, M = 1, 8, 56. The bug was the self-check. SGLang runs weight post-processing under `set_default_torch_dtype(bf16)`, so the check's `torch.ones()` top-k weights were BF16. Marlin reads them as float32, which zeroed half the rows. Fixed in 03ad1a4 (explicit float32); serving was never affected.

## RunPod v3 and v4

Model mounts come from Hugging Face. `primitive-ai@07915ee+mtp_nvfp4` rebuilds lordhansolo's package (a new `+overlay` mapping).

- **v3:** all self-checks passed in-server, with a KV pool of 1,156,544 tokens. SGLang capped requests at 12 (60 Mamba slots at 5 per request). At `MEMFRAC` 0.97 the first image prefill ran out of memory.
- **v4** (`MEMFRAC` 0.955, `MAMBA_CACHE` 80): ran cleanly at 14 slots with 965,888 KV tokens. Decode was 15–35% above v1 at up to 7 running requests, but about 25% below at 9–10. Those batches exceed 32 tokens and use the new 64-wide tiles, whose sizes were never timed.

## Memory compared with lordhansolo

From his vLLM log (notebook v293) and the [Milestone 2 comparison](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792):

| GiB | lordhansolo, vLLM | v4, SGLang |
|---|---|---|
| Weights (target + draft) | 71.94 | 69.95 + 2.68 (the draft carries a 1.18 GiB LM head it never loads) |
| KV + Mamba | 19.69, one shared pool | 11.98 KV + 0.69 QSA index + 4.50 Mamba |
| CUDA graphs | 0.46 | 1.28 |
| Left free | 1.90 | 3.41 |
| Tokens | 1,417,100 | 965,888 |

vLLM sizes Mamba pages to match attention pages and serves both from that 19.69 GiB pool, so its token count is a pool size. SGLang reports KV only and keeps Mamba separate. SGLang costs 14,140 B per KV token: 12,288 target KV (FP8), 1,024 draft KV, and about 768 for the BF16 QSA index (vLLM keeps the indexer in FP8).

## v5: lordhansolo's exact model and drafter

[`v5.json`](../../ARC3-Inference/configs/lord-nvfp4-pennyroyal-tuned/v5.json) differs from v4 as follows:

- **Model:** his BF16 PLE table (`PLE_BF16`; v1–v4 used RadixArk's FP8 table).
- **Drafter:** his 32K draft vocabulary (`DRAFT_VOCAB: configs/draft_vocab_32k.json`, from his source) with the embedded NVFP4 MTP head and 3 draft tokens.
- **Memory, with no numerics change:**
  - `MTP_SHARED_HEAD`: +1.18 GiB.
  - Mamba at 3 slots per request: `extra_buffer_lazy` + `MAMBA_SKIP_DECODE_LOCK`, `MAMBA_CACHE` 48.
  - His 2,048-token prefill chunk, so `MEMFRAC` can rise to 0.967.
- **Speed:** `LOWM_MAX_M` back to 32.
- **Load:** 2 passes over the 10 public games, so 14 streams fill.

Expected: about 1.27M KV tokens plus 2.7 GiB of Mamba. That is 19.5 GiB of KV + Mamba, the same as his 19.69 GiB pool.

At FP8 KV, 1.5M SGLang tokens would need 19.75 GiB for KV alone, more than remains after weights, Mamba, graphs and headroom. Options beyond v5:

- QSA index in FP8, as his indexer is: ~0.54 GiB at 1.5M; needs kernel changes.
- Smaller CUDA graphs: 1.28 GiB vs his 0.46.
- An NVFP4 KV cache: about 2M tokens. It changes numerics, and QSA support is unverified.
