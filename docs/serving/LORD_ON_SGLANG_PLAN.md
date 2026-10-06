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

Not a patch, but also needed: free GPU memory for the KV cache (Lord's model had 9% less; Lord moves `embed_tokens` to host memory), so more streams fit.

## Order of work

0. **Measure once.** One profiling run of Lord's model at 1, 4, 10 and 16 streams, plus stack samples while it loads. This shows where each step's time goes and how far each part is from its memory-bandwidth limit. Use it to confirm or reorder steps 1–5.
1. **Experts on Marlin.** Expected to be the biggest win: experts are most of the weights read per step.
2. **FP8 attention small-batch path.**
3. **Widen the small-batch BF16 path to 64.**
4. **Loading.** Fix the slow step; never release games before the server is ready.
5. **KV room, then more streams** (Lord ran 14).

One change per run, compared against the v1 tok/s table at the same stream counts. Run on RunPod (no queue). Confirm the final setup on Kaggle.

## What to expect

Lord's expert weights are about 8% larger than Daniel's, and his attention weights are about half the size. Fully tuned, Lord's model should match Daniel's speed per stream, not beat it clearly. Lord's higher vLLM throughput came from running 14 streams (about 68 tok/s per stream vs Daniel's 81). So the real gain is from steps 3 and 5, which would also help Daniel's model.

Stop rule: if step 0 shows that fixing every untuned path would still leave Lord's model behind Daniel's, skip steps 1–2 and go straight to more streams.
