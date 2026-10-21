# Token budget for GRPO on levels 1–2 (2026-10-21)

A GRPO rollout must clear levels 1 and 2 within a budget of generated tokens (thinking, replies and tool code). With h = the human action count of L1 + L2, three rules are compared:

| Rule | Budget | Attempts within budget | Games at 20–80% | Games at ≥ 95% | Games at ≤ 5% |
|---|---|---:|---:|---:|---:|
| Proposed | min(100K, 2,000 · h) | 70% | 11 | 8 | 2 |
| Tighter, capped | min(100K, 1,000 · h) | 63% | 15 | 5 | 2 |
| **Human-scaled** | **1,000 · h** | **68%** | **12** | **5** | **1** |

For reference, 82% of attempts clear L1 + L2 with no limit. A game at 20–80% gives GRPO groups with mixed outcomes, which is where it learns most. At ≥ 95% every rollout succeeds, and at ≤ 5% almost none do; either way the group teaches nothing.

**Verdict:**
- **min(100K, 2,000 · h) is effectively a flat 100K.** The model spends a median of 575 tokens per human action, not the assumed 1,000–2,000, so 2,000 · h exceeds 100K for 21 of 25 games.
- **1,000 · h without a cap is reasonable, and closer to the intent.** Its average budget is 102K, about the same as the flat 100K, but it is spread by how much work the game takes a human. It keeps 68% of attempts, gives mixed outcomes on 12 games, and drops only bp35.

Script: [scripts/l1_l2_token_budget.py](scripts/l1_l2_token_budget.py). It writes the figures below and prints the numbers in this doc. Context: [RFT v1 A/B](rft-v1-kaggle-ab.md), the reason we move to GRPO on the early levels.

## Data

- 44 Kaggle runs of 25 games each, all with the base model, 10-01 to 10-16. A run counts if it scored ≥ 30, which drops crashed and 5-minute smoke runs; the RFT runs are excluded. That gives 1,100 attempts (one game in one run).
- The harness versions and serving quantizations (Lord NVFP4, Intel W4A16) vary between runs, so this is the spread across our recent setups, not one configuration.
- Tokens come from each run's `evaluation.json` progression (cumulative generated tokens at each level clear). Human action counts come from `benchmark.json` (`base_actions_per_level`).
- A cleared attempt is counted at the L2 clear. An attempt that did not clear L2 is counted at the end of its game: time ran out or the scheduler parked it. That number is a lower bound on what clearing would have cost.

## Tokens per action

| Measure | Value |
|---|---|
| Generated tokens per model action (run totals) | median 824, range 665–934 |
| Generated tokens per human action, cleared L1 + L2 | median 575, p10 256, p90 1,247 |
| Cleared attempts below 1,000 per human action | 83% |
| Cleared attempts between 1,000 and 2,000 per human action | 14% |

![Generated tokens per human action](images/l1_l2_tokens_per_human_action.png)

Only bp35 (2,062), r11l (1,173) and tu93 (1,005) have game medians above 1,000. Under the 1,000 · h rule, an attempt is within budget exactly when it cleared L1 + L2 using at most 1,000 tokens per human action.

## Tokens to clear levels 1–2, per game

![Generated tokens to clear levels 1-2 per game](images/l1_l2_tokens_per_game.png)

How to read it:
- Each attempt is a dot (cleared) or a cross (not cleared). Games are sorted by median tokens.
- The black tick is 1,000 · h; the shorter gray tick is 2,000 · h. A tick past 300K is drawn as a triangle at the edge with its value.
- The columns at the right give each game's share of attempts within budget under the three rules.

| Game | Human actions L1 + L2 | 1,000 · h | Within 1,000 · h | Within min(100K, 1,000 · h) | Within min(100K, 2,000 · h) | Cleared at all | Median tokens (cleared) | p90 tokens (cleared) | Tokens per human action (median) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ft09 | 43 + 12 | 55K | 100% | 100% | 100% | 100% | 11K | 22K | 209 |
| lp85 | 17 + 38 | 55K | 100% | 100% | 100% | 100% | 28K | 35K | 505 |
| re86 | 26 + 42 | 68K | 100% | 100% | 100% | 100% | 17K | 40K | 248 |
| tr87 | 54 + 58 | 112K | 98% | 98% | 98% | 98% | 47K | 74K | 422 |
| ar25 | 32 + 50 | 82K | 95% | 95% | 100% | 100% | 21K | 50K | 256 |
| m0r0 | 30 + 111 | 141K | 93% | 86% | 86% | 93% | 59K | 99K | 417 |
| sb26 | 18 + 28 | 46K | 93% | 93% | 100% | 100% | 23K | 41K | 498 |
| dc22 | 59 + 102 | 161K | 89% | 73% | 73% | 89% | 71K | 113K | 440 |
| vc33 | 7 + 18 | 25K | 89% | 89% | 100% | 100% | 18K | 26K | 718 |
| ls20 | 22 + 123 | 145K | 86% | 68% | 68% | 91% | 78K | 118K | 535 |
| ka59 | 28 + 109 | 137K | 84% | 70% | 70% | 93% | 77K | 143K | 563 |
| g50t | 78 + 175 | 253K | 73% | 48% | 48% | 73% | 90K | 166K | 357 |
| s5i5 | 20 + 89 | 109K | 73% | 66% | 66% | 89% | 65K | 136K | 595 |
| su15 | 22 + 42 | 64K | 73% | 73% | 89% | 95% | 41K | 90K | 640 |
| wa30 | 71 + 119 | 190K | 68% | 36% | 36% | 68% | 94K | 147K | 496 |
| sc25 | 36 + 6 | 42K | 64% | 64% | 89% | 91% | 33K | 74K | 782 |
| cd82 | 55 + 8 | 63K | 55% | 55% | 73% | 82% | 56K | 122K | 895 |
| cn04 | 29 + 54 | 83K | 50% | 50% | 55% | 73% | 68K | 159K | 819 |
| sp80 | 39 + 58 | 97K | 50% | 50% | 52% | 75% | 63K | 214K | 648 |
| tu93 | 19 + 16 | 35K | 50% | 50% | 98% | 100% | 35K | 49K | 1,005 |
| lf52 | 32 + 81 | 113K | 45% | 39% | 39% | 55% | 73K | 127K | 644 |
| r11l | 22 + 33 | 55K | 36% | 36% | 89% | 100% | 65K | 109K | 1,173 |
| tn36 | 32 + 72 | 104K | 30% | 23% | 23% | 50% | 103K | 194K | 986 |
| sk48 | 61 + 177 | 238K | 18% | 5% | 5% | 18% | 126K | 236K | 531 |
| bp35 | 21 + 48 | 69K | 0% | 0% | 5% | 25% | 142K | 234K | 2,062 |

## Is 1,000 · h reasonable?

Yes, and it matches what we want from the budget better than a flat cap.

- **Same average, redistributed.** The 25 budgets average 102K (25K for vc33 up to 253K for g50t), about the same as a flat 100K. Large games get more room and small ones less.
- **It moves the budget toward human effort in both directions:**
  - Loosened on games that take humans many actions: g50t 48 → 73%, wa30 36 → 68%, ls20 68 → 86%, dc22 73 → 89%, ka59 70 → 84%, sk48 5 → 18%. On these games the 100K cap mostly failed attempts that would have cleared in a human-reasonable number of tokens.
  - Tightened on games that are short for humans but long for the model: r11l 89 → 36%, tu93 98 → 50%, sc25 89 → 64%, cd82 73 → 55%, su15 89 → 73%. These are the "easy for a human, slow for the model" levels this plan targets.
- **Two kinds of failure.** Where "within budget" equals "cleared at all" (g50t, wa30, sk48, tr87), the budget never binds and failures are levels the model doesn't solve. Where it is much lower (r11l, tu93, sc25, cd82, cn04, sp80), failures are slow clears. GRPO then trains speed, not solving. That is the efficiency we want, but it also pushes toward shorter thinking, which is what hurt [RFT v1](rft-v1-kaggle-ab.md). Watch thinking per turn and draft acceptance.
- **bp35 gets no signal:** 0% within budget, since it needs a median 2,062 tokens per human action. Drop it, or give it its own budget.
- **Cost.** Rollouts on g50t, sk48, wa30 and dc22 can run 160–253K generated tokens, past the 116K prompt budget, so the harness trims the context during the rollout as it does in normal runs. Total generation per step is about the same as with a flat 100K, because the average budget is the same.

## Level 1 vs level 2

![Generated tokens on level 1 and level 2 per game](images/l1_l2_tokens_by_level.png)

Values past 200K are drawn at 200K. The level-2 panel only includes attempts that reached L2, and counts tokens from the L1 clear onward.

| | Level 1 | Level 2 (attempts that reached it) |
|---|---|---|
| Cleared at all | 95% | 87% |
| Cleared within 50K tokens on that level | 84% | 66% |
| Cleared within 100K tokens on that level | 94% | 82% |
| Median tokens of a clear | 16K | 25K |

- **L2 takes most of the budget.** For a median cleared attempt, 64% of the L1 + L2 tokens go to L2, and only 16 of 1,100 attempts spent more than 100K on L1 alone.
- **L1 is hard only in a few games.** L1 clear rates over the same 44 runs: sk48 48%, g50t 82%, cd82 86%, dc22 89%, sc25 91%; every other game ≥ 93%.
- **bp35's problem is L2, not L1:** its L1 clears at 25–80K, but L2 is cleared in only 26% of the attempts that reach it.
- **A per-level split of 1,000 · h** (1,000 × L1 human actions for L1, the same for L2) would keep a slow L1 from eating into L2. The per-level data above would size it.

Not decided yet: the reward (human-relative level score within the budget), the held-out games, whether the model sees its budget, and one combined budget vs per-level budgets.
