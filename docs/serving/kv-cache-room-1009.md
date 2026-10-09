# More KV room for 14 streams (2026-10-09)

Follows [concurrency-14-streams](../../ARC3-Inference/configs/concurrency-14-streams/README.md). Constraint: **no new quantization** (weights, KV, PLE, index); numerics stay as in new-baseline v3.

## Why

At 14 streams Daniel's engine decodes 921 / 953 / 985 tok/s at 12 / 13 / 14 running (793 at 10), and ran 808 vs 667 tok/s effective for the first 15 minutes. Then the 1,181,312-token KV pool filled: 14 games × ~85K tokens. The prefix cache thrashed (3.86M tokens re-prefilled from the system prompt, hit rate 95% → 5–38%) and the 23-minute run ended at 602 vs 668 tok/s. Over a full history-drain cycle (59K → 118K, ~88K average) 14 streams need ~1.23M tokens; 12 need ~1.06M.

## GPU budget at 14 streams (v2 run, 94.97 GiB)

| Item | GiB |
|---|---|
| Target weights | 68.66 |
| Draft weights (albucino INT4 MTP) | 2.62 |
| KV pool, 1,181,312 tokens (12,288 B target + 1,024 B draft per token) | 14.64 |
| Mamba state, 76 slots | ~4.2 |
| CUDA graphs | ~1.3 |
| Free after graph capture (lowest during a 4,096-token prefill: 0.25 GiB) | 2.20 |

One GiB of KV ≈ 80K tokens.

## Options without numerics changes

| # | Method | Gain | Cost / risk |
|---|---|---|---|
| 1 | **Draft LM head on the meta device** (tuneup's `MTP_SHARED_HEAD`, ported to Daniel's draft) | ~1.18 GiB → **+90K tokens (+7–8%)** | None to outputs (see below). One patch to port. |
| 2 | 2,048-token prefill chunk, then `MEMFRAC` ~0.976 | ~0.7 GiB → +55K (+5%) | Prefill ~10–15% slower → ~2% tok/s |
| 3 | Mamba slots 76 → 68 | 0.44 GiB → +33K (+3%) | Spare slots hold prefix-cache checkpoints; tuneup v7 lost its hit rate with too few. Measure. |
| 4 | Fewer CUDA graph sizes | 0.2–0.4 GiB | Small; SGLang's separate draft graphs set the floor (ours 1.3 vs lordhansolo's 0.46) |
| 5 | **Host-RAM KV + Mamba tier (HiCache)**, as sirikilohit (48 GB on Pennyroyal v2.5.0) | No GPU tokens; evicted prefixes come back by copy (~1 GB in tens of ms) instead of re-prefill (~85K tokens, ~10 s) | Kaggle has ~132 GiB host RAM and BF16 PLE takes 95.4, so ~15–20 GiB for a tier (~1.2M+ tokens). sirikilohit made room with FP8 PLE, which is excluded here. Host OOM is the risk. |

1 + 2 ≈ 1.33M tokens; with 3 and 4 ≈ 1.38M. Enough for 14 streams on average (~1.23M), not for every synchronized peak; option 5 makes those peaks cheap rather than impossible.

Excluded (change numerics): NVFP4 KV (~2× tokens), FP8 QSA index (tuneup v10: +3% KV, −4% decode), online MXFP8 (−30% decode), FP8 PLE (frees host RAM only).

From the [Milestone 2 comparison](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792): lordhansolo's 1.417M-token pool comes from a built-in MTP head (no separate drafter, +1.70 GiB), vLLM's profiled 0.98 budget (+2.35 GiB; SGLang must keep prefill headroom outside its fraction, which is what OOMed concurrency-14-streams v1), and smaller graphs (+0.68 GiB). Option 1 is our share of the first. rellik13 reported the same failure we saw: pool 91–97% full, requests waiting for cache space.

**Recommendation:** options 1 + 5 at 14 streams; fall back to 12–13 streams if a host tier does not fit.

## Option 1 in detail: the draft's dead LM head

**Evidence.** albucino's `runtime/mtp-int4-g32` index (4,639 tensors, 4.14 GB) holds `lm_head.weight` and `model.language_model.embed_tokens.weight` (~1.27 GB each, BF16) next to the `mtp.*` layer. On the GPU the draft is 2.62 GB at "Load weight end", before the KV pool is sized: the MTP layer (~1.35 GB) plus the head (the embedding is already shared through `OFFLOAD_EMBED_TOKENS`). The tuneup branch put the same class's head (`Qwen4ExpForCausalLMMTP`, same pinned wheel) on the meta device for lordhansolo's model and gained exactly 1.18 GiB of KV (lord-nvfp4-pennyroyal-tuned v5).

**Why outputs cannot change.** Per the tuneup source notes, `load_weights` keeps only `mtp.*` tensors, so albucino's `lm_head.weight` is never loaded, and the speculative worker installs the target's head (its FR-Spec 64K hot rows) after the pool is sized. Even if that were wrong, every draft token is verified by the target, so a bad draft head lowers the accept length; it cannot change the generated text.

**Change.**

1. Port tuneup's `QWEN4_EXP_MTP` edit (build `ParallelLMHead` under `torch.device("meta")`, then an empty real `(0, hidden)` placeholder weight so post-load staging accepts it) into new-baseline's `pennyroyal_patch.py` as a third hook, `--draft-head`, gated at runtime by `ARC3_MTP_SHARED_HEAD=1`. Same rules as the other hooks: exact anchor once, refuse an unexpected prior edit, validate before writing. Config switch `kaggle.sglang.DRAFT_HEAD_ON_META` (the tuneup name `MTP_SHARED_HEAD` stays tied to `pennyroyal_patches.py`, which the launcher refuses to combine).
2. Unit test against the anchor, like `test_pennyroyal_optional_patches.py`.
3. GPU checks, in the pre-server step next to the GEMM check or right after startup:
   - the draft's `lm_head.weight` after the swap is the target head's hot-row slice (same values, shape 64K × hidden), not the placeholder;
   - log the draft weight size (expect ~1.4 GB instead of 2.62) and the KV pool (expect ~1.27M tokens at `MEMFRAC` 0.968).

**Run.** `concurrency-14-streams` v3 = v2 + `DRAFT_HEAD_ON_META`, everything else unchanged (14 streams, 4,096 chunk, 0.968). Pass criteria: accept length ≈ 2.6 (v2: 2.63), KV ≈ 1.27M tokens, no OOM, and the thrash onset later than v2's ~15 minutes. Run 40+ minutes so it passes the first history drain.

**Risk to watch.** After the swap, the original head's 1.27 GB may return to PyTorch's allocator cache, where prefill activations can reuse it. That would be hidden headroom that the v2 run relied on (lowest device-level free was 0.25 GiB). Moving it into the KV pool removes that slack: watch the lowest free memory, and if prefill OOMs, give part back (`MEMFRAC` 0.964) or use the 2,048 chunk (option 2).
