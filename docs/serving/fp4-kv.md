# 4-bit KV: competitor approaches and generation speed

## Competitor approaches

Public leaderboard as of 2026-11-03; the private leaderboard is not out. Entries are self-reported in
[discussion 755202](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/755202),
[discussion 755466](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/755466) and
[Tong Hui Kang's writeup](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/writeups/public-lb-41-cache-eviction-and-fp4-kv-cache),
unless marked. A dash means not stated. Ranks 1–3 (Yi-Chia Chen 75.56, Tufa Labs 70.24, NVARC3 60.95) have no
serving details in the threads read.

| Rank / score | Team                                  | KV                                        | Weights / experts                                                                                 | Draft                     | Concurrency                 | Reported tok/s (basis)                                |
| ------------ | ------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------- | --------------------------- | ----------------------------------------------------- |
| 4 / 58.59    | xz                                    | —                                         | FP8 input projections hurt less than output projections; expert pruning loses about what it gains | —                         | —; context ≈ 80K            | ~1,700 (unstated)                                     |
| 5 / 56.02    | Team Turducken (Son Pham)             | —                                         | REAP 512→448 (reported by Team Pandora)                                                           | —                         | 32                          | 2,350 (prefill + decode)                              |
| 6 / 54.55    | Grigoriy Zatravkin                    | int4, all KV in RAM                       | 400 experts, 512 after 6 h (50→54)                                                                | Tuned, stochastic drafter | 110                         | ~3,300–3,400 ("considering prefill steps"; undefined) |
| 18 / 43.88   | Michael Poluektov                     | int4, HiCache                             | REAP 512→448                                                                                      | —                         | 36; 200K max, ~100K average | ~1,400 (decode)                                       |
| 32 / 39.22   | zhangzibo54                           | FP4 middle layers, FP8 first 3 and last 3 | Mixed 3–4 bit                                                                                     | —                         | 20; 140K context            | ~1,000 (unstated)                                     |
| 41 / 37.92   | Tong Hui Kang                         | fp4, cache eviction                       | Light fine-tune                                                                                   | None                      | 64 decoding, 66 active      | 760→1,300 (unstated)                                  |
| 45 / 37.08   | Ours (main-tuned-drafter v6 settings) | FP8                                       | Intel W4A16                                                                                       | Tuned MTP-3               | 14                          | 917 (generated, 25-game run)                          |
| 80 / 35.33   | Mosamosa                              | —                                         | 384 experts                                                                                       | —                         | 17                          | ~900 (unstated)                                       |

- The tok/s column is not comparable across teams because the bases differ.
- The v6 settings are not confirmed as the submission that scored 37.08.
- Poluektov credits Richard Csaky's public milestone solution as the reference for REAP and int4 KV.
- REAP: noted as not worth it, but the evidence is mixed. Ranks 5 and 6 pruned, xz reports no net gain, and
Zatravkin gained after restoring 512 experts late.

## Generation speed

| Symbol | Meaning |
|---|---|
| gen | Generated tokens per second of wall-clock time |
| r_decode | Generated tok/s if prefill cost nothing |
| r_prefill | Prefill tok/s if the GPU only prefilled |
| ρ | Prefill tokens per generated token |

```
1/gen = 1/r_decode + ρ/r_prefill
prefill rate = ρ × gen
prefill + decode = (1 + ρ) × gen
```

`1/total = 1/prefill + 1/decode` is the special case ρ = 1. Prefill and generated token counts are independent, so
ρ must be measured.

`prefill + decode` counts tokens touched, not capacity. It can rise while gen falls, so compare runs on gen and
report ρ alongside it.

### v06 values

From 660 10-second lines in `vllm-openai-server.log`, and the step fit in [drafters.md](drafters.md):

| Quantity | Value | Source |
|---|---|---|
| gen | 930 tok/s | Log mean (917 at job level) |
| Prefill rate | 2,473 tok/s | Log mean; median 1,699 |
| ρ | 2.66 | 2,473 ÷ 930 |
| Prefill + decode | ~3,400 tok/s | Sum |
| Prefill cost | 73 µs/token | Step fit: 35.5 ms + 73.0 µs × prefill tokens, R² = 0.82 |
| r_prefill | ~13,700 tok/s | 1 ÷ 73 µs; extrapolated, the log never reaches it |
| r_decode | ~1,135 tok/s | Solved from the equation |

Time per generated token is ~881 µs decode (82%) plus ~194 µs prefill (18%).

| ρ | gen | Gain |
|---|---|---|
| 2.66 (v06) | ~930 | n/a |
| 1.33 | ~1,022 | +10% |
| 0 (unreachable) | ~1,135 | +22% |

These assume fixed r_decode, so they exclude any gain from more streams. Not checked: whether the log's prompt
counter excludes prefix-cache hits; if it includes them, true prefill and ρ are lower.

## Things they likely used but I didn't know

- HiCache of Sglang and KV RAM offload for vLLM
- Expert masking strategy and (448->512) at hour 6. 
- FP4/int4 KV cache (does vLLM and SGlang support them? How much accuracy they will affect? I'm surprised to see they didn't cause much degradation  
I remember it is not a good idea to quantize projection heads of attentions because primitive-ai's nvfp4 model is dumber than nvdia, but looks like KV itself are not much? Also the location of KV matters, first 3 and last 3 layers of KV need to stay FP8. It minds me the difference of using FP8 PLE and BF16, while they have no difference at all. So I think we can conclude that projection especially output projection matters more than the representation themselves (KV cache) I wonder is that because of error will accumulate across layers? But I wanna do a toy experiment to see that. How would you design it)