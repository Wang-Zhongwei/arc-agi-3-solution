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
| 1 | ~~Draft LM head on the meta device~~ (tuneup's `MTP_SHARED_HEAD`) | **None at equal headroom** (see below) | Same trade as raising `MEMFRAC` by ~0.009 |
| 2 | 2,048-token prefill chunk, then `MEMFRAC` ~0.976 | ~0.7 GiB → +55K (+5%) | Prefill ~10–15% slower → ~2% tok/s |
| 3 | Mamba slots 76 → 70, the floor for 14 streams (see [Mamba slots](#mamba-slots)) | 0.33 GiB → +27K (+2%) | Spare slots hold prefix-cache checkpoints (20 → 14 unlocked at 14 running). Measure the hit rate. |
| 4 | Fewer CUDA graph sizes | 0.2–0.4 GiB | Small; SGLang's separate draft graphs set the floor (ours 1.3 vs lordhansolo's 0.46) |
| 5 | **Host-RAM KV + Mamba tier (HiCache)**, as sirikilohit (48 GB on Pennyroyal v2.5.0) | No GPU tokens; evicted prefixes come back by copy (~1 GB in tens of ms) instead of re-prefill (~85K tokens, ~10 s) | Kaggle has ~132 GiB host RAM and BF16 PLE takes 95.4, so ~15–20 GiB for a tier (~1.2M+ tokens). sirikilohit made room with FP8 PLE, which is excluded here. Host OOM is the risk. |

2 ≈ 1.25M tokens; with 3 and 4 ≈ 1.29M (+9%). Near the 14-stream average (~1.23M), but not the synchronized ramp of a fresh run: all games start together and approach the 118K drain point together (v3 at 10 streams peaked at ~102K per running request), so 14 × ~105K ≈ 1.47M before the first drain. Only admission that backs off when the pool is nearly full, or option 5, handles that.

Excluded (change numerics): NVFP4 KV (~2× tokens), FP8 QSA index (tuneup v10: +3% KV, −4% decode), online MXFP8 (−30% decode), FP8 PLE (frees host RAM only).

From the [Milestone 2 comparison](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792): lordhansolo's 1.417M-token pool comes from a built-in MTP head (no separate drafter, +1.70 GiB), vLLM's profiled 0.98 budget (+2.35 GiB; SGLang must keep prefill headroom outside its fraction, which is what OOMed concurrency-14-streams v1), and smaller graphs (+0.68 GiB). Option 1 is our share of the first. rellik13 reported the same failure we saw: pool 91–97% full, requests waiting for cache space.

**Recommendation:** KV-aware admission (harness: hold new turns when server KV usage > ~0.9) or option 5 at 14 streams; fall back to 12–13 streams. Option 2 adds ~5% on top.

## Option 1 checked: not a gain (correction, same day)

The first version of this note counted the draft's dead LM head as +1.18 GiB. Reading the pinned wheel (`sglang-0.5.19+gd00d88efc8d6`, from `dfranzen/pennyroyal-v253`) shows it is already returned:

- albucino's `runtime/mtp-int4-g32` does ship `lm_head.weight` and `embed_tokens.weight` (~1.18 GiB each, BF16), but `Qwen3_5ForCausalLMMTP.load_weights` skips every tensor without `mtp` in its name, so the head is allocated and never loaded.
- `EAGLEWorkerV2.alloc_memory_pool` sizes the draft pools and only then calls `init_lm_head()` (`eagle_worker_v2.py:220–226`), whose `set_embed_and_head` deletes the draft head, installs the target head's 64K FR-Spec rows (0.31 GiB) and calls `torch.cuda.empty_cache()`.
- So the pools are sized with the dead head resident, and ~0.87 GiB comes back as free memory right after. That memory is part of the 2.20 GB left after graph capture, which the 4,096-token prefill uses (lowest free during v2: 0.25 GiB).

Building the head on the meta device grows the pool by 1.18 GiB and shrinks that headroom by the same amount minus nothing: ~2.20 → ~1.3 GB, below the ~1.95 GB a 4,096 prefill needs. Keeping the headroom means lowering `MEMFRAC` by the same amount, which cancels the gain; raising `MEMFRAC` by ~0.009 without the patch gives the identical layout. Tuneup's +1.18 GiB was measured at a fixed fraction and paid for in headroom. The real lever is the prefill headroom itself (option 2).

## Mamba slots

Most of the model's layers are Mamba-style (linear attention): instead of keeping every past token like the KV cache, each conversation keeps one fixed-size state, ~55 MB here (4.22 GiB / 76 slots), however long it is. A slot holds one such state. A running request locks several: its live state plus ping-pong buffers for speculative decoding. Slots that are not locked keep the checkpoint states of games between turns, so a returning game resumes from its saved state instead of re-prefilling the whole conversation. The cached KV alone is useless for this hybrid model without the matching state, so too few spare slots collapses the prefix hit rate (tuneup v5/v7).

| | Daniel / new-baseline v3 | concurrency-14-streams v1/v2 |
|---|---|---|
| Running requests | 10 | 14 |
| Slots locked at peak (4 each, from `serve.log` mamba usage) | 40 | 56 (0.74 of 76) |
| Unlocked, for prefix-cache checkpoints | 20 | 20 |
| `MAMBA_CACHE` | 60 | **76** |
| Memory | ~3.4 GiB | ~4.2 GiB |

At 14 streams the hit rate held at 94–98% until the KV pool filled, so 20 unlocked slots were enough; the v2 thrash came from KV, not Mamba.

**SGLang caps running requests at `MAMBA_CACHE // 5`.** In the pinned wheel (`kv_cache_configurator.py`, `_calculate_mamba_ratio`) the reservation per request is `MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO` (3) + 2 ping-pong slots for `extra_buffer` with the overlap scheduler = 5, although only 4 stay locked during decode. Hence:

| `MAMBA_CACHE` | Max running requests | Unlocked at 14 running | GiB freed vs 76 | KV tokens |
|---|---|---|---|---|
| 76 | 15 | 20 | — | — |
| 70 | 14 | 14 | 0.33 | +27K |
| 68 | 13 | — | 0.44 | capped at 13 |
| 56 | 11 | — | 1.11 | capped at 11 |

So 70 is the floor for 14 streams; below it SGLang silently lowers `max_running_requests` (it logs "max_running_requests is capped to N by the mamba state cache"). 70 gains only ~2% KV and leaves 14 unlocked slots for checkpoints; worth it only alongside a measured hit rate.
