# Plan: Lord's model on Daniel's SGLang

2026-10-06. Follows [NVFP4_FOLLOWUP_EXPERIMENT.md](NVFP4_FOLLOWUP_EXPERIMENT.md).

## Where we are

The first swap (Kaggle `lord-nvfp4-on-pennyroyal` v1 vs `pennyroyal-w4a16-baseline` v1) changed only the model. Lord's model decoded 11–17% slower at every stream count (633 vs 723 tok/s at 10 streams; same draft acceptance) and loaded weights 3.5× slower (1,432 s vs 413 s).

The engine is fine; its tuning is for Daniel's model. Daniel's patches speed up BF16 layers and INT4 Marlin experts. Lord's model stores attention in FP8 and experts in NVFP4, so those layers run on untuned paths.

## Daniel's patches against Lord's model

| # | Patch | Targets | Daniel's model | Lord's model | What to do |
|---|---|---|---|---|---|
| 1 | Low-M BF16 GEMM | Small BF16 matmuls (≤ 32 tokens) | Speeds up all attention projections | Mostly bypassed: attention is FP8 | Give FP8 attention a small-batch path; widen the BF16 path to 64 for the layers still in BF16 |
| 2 | Spec-state memory budget | Draft-head memory | Applies | Applies | Keep |
| 3 | Marlin scale dtype | Marlin expert loading | Experts on Marlin W4A16 | Experts on FlashInfer CUTLASS NVFP4 | Run NVFP4 experts on Marlin (the fork has it, but not for Lord's file format) |
| 4 | Prefix-cache retention | Cache policy | Works (94.5% hit) | Works (93.3% hit) | Keep |
| 5 | Prefetch lookahead | Startup loading | On | Off; Lord's prefetch window used instead | Loading looks CPU-bound, not disk-bound: find the slow step; start games only after the server is ready |

## Order of work

Fix the target at **14 streams**, as Lord runs. His model's real advantage is KV capacity: 1.42M tokens on vLLM against Daniel's 1.01M. On SGLang it got only 0.92M so far, because `embed_tokens` stays on the GPU and memory is budgeted Daniel's way. So KV room comes first, then every patch is tuned for 14 streams. With 4 verify tokens per stream, small batches now reach 56 tokens, above the current 32-token limit.

1. **KV room for 14 streams.** Move `embed_tokens` to host memory (Lord's trick) and retune the memory fraction and CUDA-graph sizes for 14 streams.
2. **Experts on Marlin** (replaces what patch 3 did for Daniel). Experts are most of the weights read per step.
3. **Small-batch paths for 14 streams** (patch 1, retuned): an FP8 path for attention, and the BF16 path widened for the layers still in BF16. Raise the limit from 32 to 64 tokens. Tune tile settings for 56 first (all 14 streams verifying), then 48 and 52 (12–13 streams). 64 only needs to not be slow; capture a CUDA graph at 14 so batches aren't padded to 16. Streams never exceed 14, so nothing above 64 is needed unless concurrency is raised later.
4. **Loading** (patch 5, retuned): fix the slow step, and start games only after the server is ready.

One change per run, all at 14 streams, compared against the v1 numbers. Run on RunPod (no queue). Confirm the final setup on Kaggle.
