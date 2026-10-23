# Token budget for GRPO on levels 1–2

**Decision (2026-10-23): min(1,750 × human actions of L1 + L2, 130K) context tokens per episode.** It replaces the 2026-10-21 rule of min(1,000 × h, 100K) generated tokens. Recorded in [decisions.md](decisions.md).

GRPO rollouts on L1–2 never trim the context, so one sequence holds everything the model saw and wrote: system prompt, observations, tool results, frame images, and every generated token (thinking, replies, tool code). One number now caps that sequence. It is both the rollout budget and the trainer's memory limit: one H200 trains up to about 133K tokens (the 123K request peaked at 129 GiB), and 130K stays under it. With h = the human action count of L1 + L2:

| Rule | Measured on | Attempts within budget | Games at 20–80% | Games at ≥ 95% | Games at ≤ 5% | Mean budget |
|---|---|---:|---:|---:|---:|---:|
| **min(130K, 1,750 · h)**, chosen | context | **53%** | **15** | 4 | 2 | 113K |
| min(130K, 1,500 · h) | context | 46% | 14 | 4 | 2 | 107K |
| 1,750 · h, uncapped | context | 67% | 13 | 5 | 1 | 178K |
| Earlier: min(100K, 1,000 · h) | generated | 62% | 15 | 5 | 2 | 78K |

82% of attempts clear L1 + L2 with no limit. A game at 20–80% gives GRPO groups with mixed outcomes, which is where it learns most; at ≥ 95% or ≤ 5% the groups rarely have mixed outcomes.

**Why 1,750:**
- It gives the most games with mixed outcomes under the 130K cap (calibration below).
- It is the earlier rule translated: 1,000 generated tokens per human action × the median context-to-generated ratio of 1.79 ≈ 1,790.
- 1,500 is close (14 games) but drops vc33 to 23%, tu93 to 9% and sc25 to 21%: on games that are short for humans, the fixed ~7.4K system prompt and the frame images take a larger share.

**The cap is tighter than before.** 130K of context is about 70K generated tokens at the median ratio, where the earlier cap was 100K generated. The cap binds on 14 games, and games with long L2s lose the most: g50t 47% → 21%, wa30 37% → 21%, ls20 67% → 44%, dc22 74% → 47%. 26% of cleared attempts reached more than 130K of context.

Scripts: [scripts/l1_l2_context_tokens.py](scripts/l1_l2_context_tokens.py) rebuilds each attempt's context into [data/l1_l2_context.json](data/l1_l2_context.json); [scripts/l1_l2_token_budget.py](scripts/l1_l2_token_budget.py) reads it, writes the figures below and prints the numbers in this doc. Context: [RFT v1 A/B](rft-v1-kaggle-ab.md), the reason we move to GRPO on the early levels.

## Data

- 43 Kaggle runs of 25 games each, all with the base model, 10-01 to 10-16, 1,075 attempts (one game in one run). A run counts if it scored ≥ 30, which drops crashed and 5-minute smoke runs; the RFT runs are excluded. One more run (lorddraft-baseline v01) has no transcripts.
- The runs trimmed their context; GRPO will not. So the context is rebuilt from each run's transcripts with every message kept (`inference/rft/context_rebuild.py`): the size of the whole message stream when the first L2 request is sent (L1 cleared) and when the first L3 request is sent (L2 cleared). Token counts are the rebuild's estimates, within about 3% of request logs. The rebuilt generated tokens match `evaluation.json` within about ±10%.
- Clears agree with `evaluation.json` on all 1,075 attempts. An attempt that did not clear L2 is counted at the end of its game: time ran out or the scheduler parked it. That number is a lower bound on what clearing would have cost.
- The harness versions and serving quantizations vary between runs, so this is the spread across our recent setups, not one configuration.
- Human action counts come from `benchmark.json` (`base_actions_per_level`).

## Calibration

| Rule | Attempts within budget | Games at 20–80% | Games at ≥ 95% | Games at ≤ 5% | Games where the cap binds |
|---|---:|---:|---:|---:|---:|
| min(130K, 1,000 · h) | 34% | 8 | 1 | 8 | 7 |
| min(130K, 1,250 · h) | 41% | 11 | 4 | 4 | 10 |
| min(130K, 1,500 · h) | 46% | 14 | 4 | 2 | 12 |
| **min(130K, 1,750 · h)** | **53%** | **15** | 4 | 2 | 14 |
| min(130K, 2,000 · h) | 57% | 13 | 4 | 2 | 16 |
| min(130K, 2,500 · h) | 60% | 11 | 5 | 2 | 21 |

| Measure | Value |
|---|---|
| Context ÷ generated tokens at the L2 clear | median 1.79, p10 1.43, p90 2.50 |
| Context per human action, cleared L1 + L2 | median 1,032, p10 525, p90 2,160 |

![Context tokens per human action](images/l1_l2_tokens_per_human_action.png)

The ratio of context to generated tokens is higher for short episodes, where the fixed system prompt weighs more (ft09 2.06, re86 2.42, sb26 2.49), and lowest for long ones (g50t 1.54, sk48 1.45, r11l 1.42).

## Context at the L2 clear, per game

![Context tokens at the level-2 clear per game](images/l1_l2_tokens_per_game.png)

How to read it:
- Each attempt is a dot (cleared) or a cross (not cleared). Games are sorted by median context.
- The black tick is 1,750 · h; the shorter gray tick is 1,500 · h. A tick past 300K is drawn as a triangle at the edge with its value. The dashed line is the 130K cap.
- The columns at the right give each game's share of attempts within budget under the four rules; the last one is the earlier generated-token rule.

| Game | Human actions L1 + L2 | Budget | Within min(130K, 1,750·h) | Within min(130K, 1,500·h) | Within 1,750·h, uncapped | Within earlier: min(100K, 1,000·h) generated | Cleared at all | Median context (cleared) | p90 context (cleared) | Context per human action (median) | Context / generated (median) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ft09 | 43 + 12 | 96K | 100% | 100% | 100% | 100% | 100% | 25K | 38K | 449 | 2.06 |
| lp85 | 17 + 38 | 96K | 100% | 100% | 100% | 100% | 100% | 44K | 54K | 791 | 1.58 |
| re86 | 26 + 42 | 119K | 100% | 98% | 100% | 100% | 100% | 43K | 72K | 639 | 2.42 |
| tr87 | 54 + 58 | 130K | 98% | 98% | 98% | 98% | 98% | 75K | 106K | 669 | 1.57 |
| ar25 | 32 + 50 | 130K | 93% | 93% | 98% | 95% | 100% | 51K | 89K | 626 | 2.41 |
| sb26 | 18 + 28 | 80K | 84% | 74% | 84% | 93% | 100% | 56K | 103K | 1,228 | 2.49 |
| m0r0 | 30 + 111 | 130K | 70% | 70% | 93% | 86% | 93% | 97K | 155K | 685 | 1.62 |
| su15 | 22 + 42 | 112K | 70% | 63% | 70% | 72% | 95% | 85K | 155K | 1,333 | 2.02 |
| vc33 | 7 + 18 | 44K | 70% | 23% | 70% | 88% | 100% | 41K | 52K | 1,643 | 2.30 |
| r11l | 22 + 33 | 96K | 60% | 30% | 60% | 37% | 100% | 90K | 139K | 1,629 | 1.42 |
| ka59 | 28 + 109 | 130K | 53% | 53% | 91% | 72% | 93% | 126K | 179K | 919 | 1.62 |
| s5i5 | 20 + 89 | 130K | 53% | 53% | 79% | 65% | 88% | 112K | 193K | 1,032 | 1.62 |
| sc25 | 36 + 6 | 74K | 51% | 21% | 51% | 63% | 91% | 73K | 143K | 1,729 | 2.20 |
| cn04 | 29 + 54 | 130K | 49% | 49% | 51% | 49% | 72% | 108K | 230K | 1,304 | 1.62 |
| dc22 | 59 + 102 | 130K | 47% | 47% | 88% | 74% | 88% | 128K | 190K | 796 | 1.82 |
| ls20 | 22 + 123 | 130K | 44% | 44% | 91% | 67% | 91% | 131K | 184K | 903 | 1.55 |
| sp80 | 39 + 58 | 130K | 40% | 40% | 51% | 49% | 74% | 119K | 294K | 1,230 | 1.64 |
| cd82 | 55 + 8 | 110K | 37% | 19% | 37% | 53% | 81% | 117K | 178K | 1,857 | 1.92 |
| tu93 | 19 + 16 | 61K | 33% | 9% | 33% | 51% | 100% | 66K | 97K | 1,889 | 1.89 |
| g50t | 78 + 175 | 130K | 21% | 21% | 72% | 47% | 72% | 145K | 246K | 572 | 1.54 |
| wa30 | 71 + 119 | 130K | 21% | 21% | 70% | 37% | 70% | 158K | 221K | 832 | 1.71 |
| lf52 | 32 + 81 | 130K | 16% | 16% | 37% | 37% | 53% | 154K | 263K | 1,364 | 2.15 |
| tn36 | 32 + 72 | 130K | 14% | 14% | 26% | 21% | 49% | 180K | 332K | 1,729 | 1.78 |
| sk48 | 61 + 177 | 130K | 5% | 5% | 19% | 5% | 19% | 181K | 334K | 760 | 1.45 |
| bp35 | 21 + 48 | 121K | 0% | 0% | 0% | 0% | 26% | 217K | 358K | 3,148 | 1.62 |

## What the new rule does

- **Always within budget (the guard games):** ft09, lp85, re86 and tr87. ar25 drops from 95% to 93%, so it gives some signal now.
- **Little signal:** bp35 (0%, median clear 217K of context) and sk48 (5%). With per-level reward credit they still give signal on L1: sk48 has mixed groups 98% of the time at 8 rollouts; bp35 44% (69% at 16 rollouts), since 93% of its attempts clear L1 within budget but almost none clear L2.
- **The bp35 exception is gone.** It got a flat 100K generated tokens under the earlier rule; now its 1,750 · h is 121K, close to the cap, and a flat 130K would not change its 0%.
- **Two kinds of failure, as before.** Where the cap binds (g50t, wa30, lf52, tn36, ls20, dc22), most failures are long L2s that ran past 130K. Where 1,750 · h binds (tu93, vc33, sc25, cd82, r11l), failures are slow clears of levels that are short for humans. Both push toward fewer tokens per level, which is the efficiency we want, but also toward shorter thinking, which is what hurt [RFT v1](rft-v1-kaggle-ab.md). Watch thinking per turn and draft acceptance.

## Level 1 vs level 2

![Context on level 1 and level 2 per game](images/l1_l2_tokens_by_level.png)

Values past 200K are drawn at 200K. The level-2 panel only includes attempts that reached L2, and counts the context added from the L1 clear onward.

| | Level 1 | Level 2 (attempts that reached it) |
|---|---|---|
| Cleared at all | 95% | 86% |
| Cleared within 50K of context on that level | 63% | 50% |
| Cleared within 100K of context on that level | 88% | 74% |
| Median context of a clear | 36K at the clear (incl. the ~7.4K system prompt) | 43K added |

- **L2 takes the larger share.** For a median cleared attempt, 55% of the context at the L2 clear was added on L2.
- **L1 alone can exceed the budget:** 81 of 1,075 attempts were past 100K of context at the L1 clear.

Not decided yet: whether the model sees its budget.
