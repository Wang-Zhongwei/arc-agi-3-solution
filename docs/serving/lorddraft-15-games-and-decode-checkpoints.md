# lorddraft: 15 games on 14 sequences, and decode checkpoints

**Decision (2026-10-09):** run 15 harness games against vLLM `max_num_seqs` 14 (`ARC3_MAX_ACTIVE_STREAMS` 15). It gave +3.0% generation throughput with no KV or prefix-cache cost. Keeping decode-time Mamba checkpoints (`--prefix-cache-retention-interval 1664`) gave nothing; the diagnosis below explains why, and what a working version needs.

Baseline: [matijaludvig/arc3-vcintel-lorddraft-25x1](https://www.kaggle.com/code/matijaludvig/arc3-vcintel-lorddraft-25x1) v1 (Intel W4A16 target, lordhansolo's NVFP4 MTP draft, 14 games, 14 sequences; 803 tok/s, score 56.89). Both variants are forks of that exact notebook built by [`build_fork.py`](../../ARC3-Inference/experiments/lorddraft-tps/build_fork.py), which only applies the listed one-line edits and keeps v1's datasets (source bundle v1, runtime `lordhansolo/vllm-main-e975732-arc3` v3, overlay `e3a6fe0d`), models, image and machine.

## Why these two

In lorddraft v1 a decode step costs about 19 ms + 1.27 ms per running sequence + 69 µs per prefill token (fit on the 10 s engine log intervals, R² 0.75). Two things kept the GPU from its best rate:

- **Idle sequence slots.** The harness gate holds a slot per *game* through tool runs and game steps, so the engine averaged 12.35 running requests of 14 even with 17–25 games alive, and sat at the cap only 39% of the time.
- **Re-prefilled completions.** In every one of 2,905 growing turns, the prefix hit stopped at the previous prompt's end (median 192 tokens short). The model's own previous answer (median 1,201 tokens) was prefilled again: 6.35M tokens, 37% of all prefill.

## Runs

Both were cancelled early on purpose; outputs survive a cancel. Comparisons use the same minutes of play (from the first busy engine step) in lorddraft v1.

| Kaggle notebook (v1) | Change | Played |
|---|---|---|
| [jonathanwang2022/lorddraft-15-games-14-seqs](https://www.kaggle.com/code/jonathanwang2022/lorddraft-15-games-14-seqs) | `ARC3_MAX_ACTIVE_STREAMS` 14 → 15, `max_num_seqs` stays 14 | 51 min |
| [jonathanwang2022/lorddraft-retain-decode-checkpoints](https://www.kaggle.com/code/jonathanwang2022/lorddraft-retain-decode-checkpoints) | `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` 0 → 1664 | 33 min |

| Same window as the variant | v1 (51 min) | **15 games** | v1 (33 min) | Retain 1664 |
|---|---|---|---|---|
| Generation tok/s | 870 | **896 (+3.0%)** | 892 | 886 (−0.7%) |
| Running requests (mean) | 13.33 | 13.75 | 13.54 | 13.64 |
| Time at 14 running | 58% | 88% | 66% | 74% |
| KV usage max | 90.5% | 87.5% | 90.5% | 89.1% |
| Prefix hit rate | 92.9% | 92.6% | 93.5% | 92.1% |
| MTP acceptance length | 2.70 | 2.71 | 2.68 | 2.68 |

Paired 5-minute bins: 15 games was higher in 7 of 10 bins, mean +26 tok/s, t = 2.59 (df 9, one-sided p ≈ 0.015). Retain 1664: mean −2 tok/s, t = −0.10 (df 5). Bins of one run are correlated and each side is one run on one host, so treat the p-value as a guide.

## Finding 1: one extra game fills the engine

With 15 games and 14 sequences, a game in a tool run no longer leaves a sequence idle: the engine sat at 14 running 88% of the time (v1: 58%). The extra game's prefix did not cost cache: KV peaked lower and the hit rate held. Two extra games would not fit: at v1's KV peak only about 180K tokens are free, roughly two average contexts, so the extra prefixes would be evicted and re-prefilled (about 5 s of engine time per 75K rebuild).

## Finding 2: why only 2% of turns reused a decode checkpoint

RecoverSSM's commit kernel already writes the exact GDN state at every 1664-token Mamba block boundary that decode crosses; retention 0 just dropped those states. With retention 1664 the next turn could, in principle, resume inside the previous answer. It did in 33 of 942 turns.

What it is **not**:

- **Not the history.** In 932 of 942 consecutive requests the next request's messages extend the previous ones exactly; the other 10 are context trims (excluded above).
- **Not the chat template.** Re-tokenizing the stored answer as the template renders it (`<think>\n` + trimmed reasoning + `\n</think>\n\n` + tool call + `<|im_end|>`) gives exactly the generated token count for 911 of 928 tool-call answers (98.2%), and no reasoning had whitespace that `trim` removes. The next prompt carries the answer token for token.

What it **is**:

| The previous answer crossed | Turns | Resumed inside it | Landed at |
|---|---|---|---|
| no 1664 boundary | 397 | 0% | – |
| 1 boundary | 340 | 2% | `b` ×6 |
| 2+ boundaries | 195 | 14% | `b − 1664` ×23, `b` ×4 |

1. **MTP back-off of a whole Mamba block.** With speculative decoding the Mamba group finds its newest checkpoint `b` and then searches again below `b − 1664` (`find_longest_cache_hit` with `drop_eagle_block`, [kv_cache_coordinator.py](https://github.com/vllm-project/vllm) in the e975732 overlay). Prompt tails dodge this by registering their checkpoint one 128-token unit lower; decode checkpoints exist only every 1664 tokens, so the back-off lands before the previous prompt for any answer that crosses fewer than two boundaries — 79% of turns. The few hits exactly at `b` are turns where an attention group had already applied the back-off.
2. **Even long answers mostly miss.** Only 14% of answers crossing two or more boundaries resumed. The newest boundary state is usually not found at lookup, so the back-off falls two blocks. Not yet traced in the code.

Ceiling on lorddraft v1's 17.3M prefill tokens:

| Decode checkpoints | Saves |
|---|---|
| every 1664 tokens, with the one-block back-off (this run, if cause 2 were fixed) | 10.9% |
| every 1664 tokens, no back-off | 23.0% |
| every 128 tokens, back-off of 128 | 31.4% |

So retention alone cannot give much. The useful version is the SGLang one: SGLang's Mamba radix cache inserts prompt + output when a request finishes, with the state tracked by two ping-pong slots during decode. On vLLM that means writing the GDN state at the last 128-token boundary during the RecoverSSM commit and registering it like the prompt tail, so the back-off is one 128-token unit. Prefill is about 20% of engine time, so 31% less prefill is worth roughly +6%.

## Not tested

- Score: both runs were cut before the games finished.
- 15 games over a full two-hour run, where v1 dropped to ~11 running after minute 80.
- Cause 2 above, and the 128-token decode checkpoint, which needs a runtime patch and GPU testing.
