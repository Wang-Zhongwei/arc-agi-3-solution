# Token budget for GRPO on levels 1–2

**Decision (2026-10-23): min(1,750 × human actions of L1 + L2, 130K) context tokens per episode.** It replaces the 2026-10-21 rule of min(1,000 × h, 100K) generated tokens. Recorded in [decisions.md](decisions.md). The numbers were recomputed on 2026-10-25 with images priced at each run's board size ([Token estimate](#token-estimate)); the choice stands.

GRPO rollouts on L1–2 never trim the context, so one sequence holds everything the model saw and wrote: system prompt, observations, tool results, frame images, and every generated token (thinking, replies, tool code). One number now caps that sequence. It is both the rollout budget and the trainer's memory limit: one H200 trains up to about 133K tokens (the 123K request peaked at 129 GiB), and 130K stays under it. With h = the human action count of L1 + L2:

| Rule | Measured on | Attempts within budget | Games at 20–80% | Games at ≥ 95% | Games at ≤ 5% | Mean budget |
|---|---|---:|---:|---:|---:|---:|
| **min(130K, 1,750 · h)**, chosen | context | **51%** | **15** | 4 | 2 | 113K |
| min(130K, 1,500 · h) | context | 46% | 14 | 4 | 2 | 107K |
| 1,750 · h, uncapped | context | 66% | 13 | 5 | 1 | 178K |
| Earlier: min(100K, 1,000 · h) | generated | 62% | 15 | 5 | 2 | 78K |

82% of attempts clear L1 + L2 with no limit. A game at 20–80% gives GRPO groups with mixed outcomes, which is where it learns most; at ≥ 95% or ≤ 5% the groups rarely have mixed outcomes.

**Why 1,750:**
- It gives the most games with mixed outcomes under the 130K cap (calibration below).
- It is the earlier rule translated: 1,000 generated tokens per human action × the median context-to-generated ratio of 1.83 ≈ 1,830.
- 1,500 is close (14 games) but drops vc33 to 23%, tu93 to 7% and sc25 to 21%: on games that are short for humans, the fixed start (system prompt, tools and first frame: 5.4K tokens by vLLM's count) and the frame images take a larger share.

**The cap is tighter than before.** 130K of context is about 70K generated tokens at the median ratio, where the earlier cap was 100K generated. The cap binds on 14 games, and games with long L2s lose the most: g50t 47% → 23%, wa30 37% → 21%, ls20 67% → 42%, dc22 74% → 44%. 27% of cleared attempts reached more than 130K of context.

Scripts: [scripts/l1_l2_context_tokens.py](scripts/l1_l2_context_tokens.py) rebuilds each attempt's context into [data/l1_l2_context.json](data/l1_l2_context.json); [scripts/l1_l2_token_budget.py](scripts/l1_l2_token_budget.py) reads it, writes the figures below and prints the numbers in this doc. Context: [RFT v1 A/B](rft-v1-kaggle-ab.md), the reason we move to GRPO on the early levels.

## Data

- 43 Kaggle runs of 25 games each, all with the base model, 10-01 to 10-16, 1,075 attempts (one game in one run). A run counts if it scored ≥ 30, which drops crashed and 5-minute smoke runs; the RFT runs are excluded. One more run (lorddraft-baseline v01) has no transcripts.
- The runs trimmed their context; GRPO will not. So the context is rebuilt from each run's transcripts with every message kept (`inference/rft/context_rebuild.py`): the size of the whole message stream when the first L2 request is sent (L1 cleared) and when the first L3 request is sent (L2 cleared). Token counts are the rebuild's estimates, with each image priced at its run's board size. Above 60K of context, real ÷ estimate is 0.99 on the median and 0.92–1.07 at the extremes ([Token estimate](#token-estimate)). The rebuilt generated tokens match `evaluation.json` within about ±10%.
- Clears agree with `evaluation.json` on all 1,075 attempts. An attempt that did not clear L2 is counted at the end of its game: time ran out or the scheduler parked it. That number is a lower bound on what clearing would have cost.
- The harness versions and serving quantizations vary between runs, so this is the spread across our recent setups, not one configuration.
- Human action counts come from `benchmark.json` (`base_actions_per_level`).

## Calibration

| Rule | Attempts within budget | Games at 20–80% | Games at ≥ 95% | Games at ≤ 5% | Games where the cap binds |
|---|---:|---:|---:|---:|---:|
| min(130K, 1,000 · h) | 33% | 8 | 1 | 8 | 7 |
| min(130K, 1,250 · h) | 40% | 11 | 4 | 4 | 10 |
| min(130K, 1,500 · h) | 46% | 14 | 4 | 2 | 12 |
| **min(130K, 1,750 · h)** | **51%** | **15** | 4 | 2 | 14 |
| min(130K, 2,000 · h) | 56% | 14 | 4 | 2 | 16 |
| min(130K, 2,500 · h) | 59% | 12 | 5 | 2 | 21 |

| Measure | Value |
|---|---|
| Context ÷ generated tokens at the L2 clear | median 1.83, p10 1.45, p90 2.54 |
| Context per human action, cleared L1 + L2 | median 1,049, p10 525, p90 2,178 |

![Context tokens per human action](images/l1_l2_tokens_per_human_action.png)

The ratio of context to generated tokens is higher for short episodes, where the fixed system prompt weighs more (ft09 2.06, re86 2.45, sb26 2.58), and lowest for long ones (g50t 1.60, sk48 1.50, r11l 1.46).

## Context at the L2 clear, per game

![Context tokens at the level-2 clear per game](images/l1_l2_tokens_per_game.png)

How to read it:
- Each attempt is a dot (cleared) or a cross (not cleared). Games are sorted by median context.
- The black tick is 1,750 · h; the shorter gray tick is 1,500 · h. A tick past 300K is drawn as a triangle at the edge with its value. The dashed line is the 130K cap.
- The columns at the right give each game's share of attempts within budget under the three context rules. The earlier generated-token rule is in the table below.

| Game | Human actions L1 + L2 | Budget | Within min(130K, 1,750·h) | Within min(130K, 1,500·h) | Within 1,750·h, uncapped | Within earlier: min(100K, 1,000·h) generated | Cleared at all | Median context (cleared) | p90 context (cleared) | Context per human action (median) | Context / generated (median) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ft09 | 43 + 12 | 96K | 100% | 100% | 100% | 100% | 100% | 25K | 39K | 449 | 2.06 |
| lp85 | 17 + 38 | 96K | 100% | 100% | 100% | 100% | 100% | 44K | 55K | 798 | 1.58 |
| re86 | 26 + 42 | 119K | 100% | 98% | 100% | 100% | 100% | 44K | 69K | 647 | 2.45 |
| tr87 | 54 + 58 | 130K | 98% | 98% | 98% | 98% | 98% | 74K | 107K | 660 | 1.61 |
| ar25 | 32 + 50 | 130K | 93% | 93% | 98% | 95% | 100% | 51K | 89K | 626 | 2.45 |
| sb26 | 18 + 28 | 80K | 84% | 72% | 84% | 93% | 100% | 55K | 103K | 1,193 | 2.58 |
| m0r0 | 30 + 111 | 130K | 70% | 70% | 93% | 86% | 93% | 100K | 158K | 706 | 1.62 |
| su15 | 22 + 42 | 112K | 70% | 67% | 70% | 72% | 95% | 85K | 155K | 1,333 | 2.06 |
| r11l | 22 + 33 | 96K | 58% | 30% | 58% | 37% | 100% | 93K | 138K | 1,697 | 1.46 |
| vc33 | 7 + 18 | 44K | 56% | 23% | 56% | 88% | 100% | 43K | 53K | 1,703 | 2.32 |
| ka59 | 28 + 109 | 130K | 53% | 53% | 91% | 72% | 93% | 125K | 179K | 909 | 1.68 |
| s5i5 | 20 + 89 | 130K | 51% | 51% | 79% | 65% | 88% | 115K | 194K | 1,059 | 1.64 |
| cn04 | 29 + 54 | 130K | 47% | 47% | 51% | 49% | 72% | 113K | 240K | 1,361 | 1.68 |
| dc22 | 59 + 102 | 130K | 44% | 44% | 88% | 74% | 88% | 132K | 190K | 822 | 1.86 |
| sc25 | 36 + 6 | 74K | 44% | 21% | 44% | 63% | 91% | 74K | 154K | 1,761 | 2.20 |
| ls20 | 22 + 123 | 130K | 42% | 42% | 91% | 67% | 91% | 134K | 174K | 924 | 1.55 |
| sp80 | 39 + 58 | 130K | 37% | 37% | 51% | 49% | 74% | 123K | 309K | 1,270 | 1.67 |
| cd82 | 55 + 8 | 110K | 35% | 19% | 35% | 53% | 81% | 116K | 186K | 1,835 | 2.02 |
| tu93 | 19 + 16 | 61K | 33% | 7% | 33% | 51% | 100% | 68K | 91K | 1,930 | 1.91 |
| g50t | 78 + 175 | 130K | 23% | 23% | 72% | 47% | 72% | 145K | 249K | 573 | 1.60 |
| wa30 | 71 + 119 | 130K | 21% | 21% | 70% | 37% | 70% | 167K | 238K | 880 | 1.80 |
| lf52 | 32 + 81 | 130K | 14% | 14% | 35% | 37% | 53% | 155K | 256K | 1,372 | 2.19 |
| tn36 | 32 + 72 | 130K | 14% | 14% | 26% | 21% | 49% | 180K | 349K | 1,729 | 1.79 |
| bp35 | 21 + 48 | 121K | 0% | 0% | 0% | 0% | 26% | 217K | 358K | 3,148 | 1.62 |
| sk48 | 61 + 177 | 130K | 0% | 0% | 19% | 5% | 19% | 186K | 351K | 781 | 1.50 |

## What the new rule does

- **Always within budget (the guard games):** ft09, lp85, re86 and tr87. ar25 drops from 95% to 93%, so it gives some signal now.
- **Little signal:** bp35 (0%, median clear 217K of context) and sk48 (0%). With per-level reward credit they still give signal on L1: sk48 has mixed groups 98% of the time at 8 rollouts; bp35 44% (69% at 16 rollouts), since 93% of its attempts clear L1 within budget but almost none clear L2.
- **The bp35 exception is gone.** It got a flat 100K generated tokens under the earlier rule; now its 1,750 · h is 121K, close to the cap, and a flat 130K would not change its 0%.
- **Two kinds of failure, as before.** Where the cap binds (g50t, wa30, lf52, tn36, ls20, dc22), most failures are long L2s that ran past 130K. Where 1,750 · h binds (tu93, vc33, sc25, cd82, r11l), failures are slow clears of levels that are short for humans. Both push toward fewer tokens per level, which is the efficiency we want, but also toward shorter thinking, which is what hurt [RFT v1](rft-v1-kaggle-ab.md). Watch thinking per turn and draft acceptance.

## Level 1 vs level 2

![Context on level 1 and level 2 per game](images/l1_l2_tokens_by_level.png)

Values past 200K are drawn at 200K. The level-2 panel only includes attempts that reached L2, and counts the context added from the L1 clear onward.

| | Level 1 | Level 2 (attempts that reached it) |
|---|---|---|
| Cleared at all | 95% | 86% |
| Cleared within 50K of context on that level | 62% | 50% |
| Cleared within 100K of context on that level | 88% | 74% |
| Median context of a clear | 36K at the clear (incl. the ~7.6K first request) | 43K added |

- **L2 takes the larger share.** For a median cleared attempt, 55% of the context at the L2 clear was added on L2.
- **L1 alone can exceed the budget:** 81 of 1,075 attempts were past 100K of context at the L1 clear.

Not decided yet: whether the model sees its budget.

## Token estimate

The rebuild prices text at 0.333 tokens per character plus 520 per request, fitted on request-logged runs at `MULTIMODAL_UPSCALE` 10. Images are priced the way the harness prices them: one token per 32-pixel patch plus 2. That is 258 tokens at upscale 8, 402 at 10 and 578 at 12. Of the 43 runs, 27 drew boards at upscale 10, 12 at 12 and 4 at 8. In every run, this price matches the image cost the harness logged for each game's first frame.

**Fixed 2026-10-25 (main 755d4749): every image used to be priced at 402 tokens.** 402 is the cost of a 640-pixel board. A 768-pixel board at upscale 12 costs 578, so the upscale-12 runs were undercounted, and the more images a context held, the worse. vLLM's prompt counts for the 1,445 requests replayed in the [H200 benchmark](h200-rollout-serving.md#result-2026-10-25):

| Runs | Requests | Real ÷ estimate at 402 per image | At the run's board size |
|---|---:|---|---|
| Upscale 12 (9 runs) | 1,149 | median 1.04, p99 1.14, max 1.17 | median 0.97, p99 1.07, max 1.07 |
| Upscale 10 (3 runs) | 296 | median 0.96, max 1.02 | unchanged |

- In the upscale-12 runs the shortfall grew from a median of 2% with 2–19 images in context to 6% with 20–59.
- The fix moved context at the L2 clear by a median factor of 1.000 (upscale-10 runs are unchanged), at most 1.11. Within budget went from 53% to 51%. The same 15 games stay at 20–80%, and the cap binds on the same 14.
- Short contexts still read high. The first request (system prompt, tools, first frame) is 5.4K tokens by vLLM's count and 7.6K by the estimate, because its prose runs at 0.235 tokens per character. Later content (frames, tool output, code) is denser than 0.333 per character, so on long contexts the two errors partly cancel.

**Real counts beat the formula wherever they exist.** vLLM returns the exact `prompt_tokens` and `completion_tokens` of every request. The table covers contexts above 60K (603 of the same requests):

| Estimate of request k | Median real ÷ estimate | Min | p99 | Max |
|---|---:|---:|---:|---:|
| Formula only | 0.990 | 0.916 | 1.066 | 1.073 |
| Real count of the first request + formula for everything since | 1.012 | 0.944 | 1.085 | 1.115 |
| **Real count of request k − 1 + formula for the messages added since** | **1.000** | 0.967 | 1.013 | 1.041 |

- **GRPO rollouts:** each response returns the exact counts, so the budget check needs no estimate. An estimate is needed only before a request is sent, to cap `max_tokens` so the reply cannot cross 130K; use the last row. A live rollout also knows the previous reply's exact token count, which the replay did not use, so it should be at least this close.
- **Offline (this analysis, RFT data):** our runs log a real count only for the first request: the harness's `text_token_calibration` line, present in all 43 runs. Per-request counts are in neither the transcripts nor the server logs. Anchoring on that one count is no better than the formula (second row), so the rebuild keeps the formula. If every request's `prompt_tokens` and `completion_tokens` were logged, later rebuilds could anchor every step.
- The live harness already reads these counts, but only to recalibrate its characters-per-token rate. It caps that rate at 3.3 characters per token, so text is over-counted (`tool_agent._calibrate_from_usage`).
