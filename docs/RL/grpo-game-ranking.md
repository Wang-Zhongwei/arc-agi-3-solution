# GRPO games ranked by signal per group

Round 1 trains 8 games, one group of 8 rollouts each per wave ([decisions.md](decisions.md), Games). A B200 node has room for more rollouts per wave, so this ranks all 25 public games to pick the next ones. The B200 run keeps round 1's 8 and adds su15 and lf52 (decided 2026-10-27): [configs/grpo-b200/v1.json](../../ARC3-Inference/configs/grpo-b200/v1.json).

## Measure

A group gives a gradient only when its rewards differ, and its advantages (R − group mean) are larger when they spread more. So a game's signal per group is the variance of its reward over attempts.

- **Reward:** the decided one without the thinking factor: R = ⅓ · (1 · [L1 within budget] · min(1, (h₁/a₁)²) + 2 · [L2 within budget] · min(1, (h₂/a₂)²)). Budget = min(1,750 × human actions of L1 + L2, 130K) context tokens.
- **Data:** the 1,075 attempts (43 runs × 25 games) in [l1_l2_context.json](data/l1_l2_context.json), with the actions at each level clear from each run's `evaluation.json`. The runs span many harness versions, not v6 alone.
- **Check:** with the level reward alone (0, ⅓ or 1), the 8 round-1 games average 0.092, and 0.125 with dc22 + m0r0 in place of bp35 + sk48: the numbers in decisions.md. Their wave holds 7.6M context tokens here, as there.
- **Cost:** a rollout ends at the L2 clear or at the budget. The wave lasts about as long as the budget lets the longest rollouts run; trainer time goes with the mean length.

Script: [grpo_game_ranking.py](scripts/grpo_game_ranking.py) → [grpo_game_ranking.json](data/grpo_game_ranking.json).

```bash
python3 docs/RL/scripts/grpo_game_ranking.py ARC3-Inference/runs/kaggle
```

## Ranking

| Rank | Game | Budget | L1 | L1 + L2 | Var(level) | Var(reward) | Mean reward | P(flat group) | Mean tokens | Var per 100K tokens | Note |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | sc25 | 74K | 60% | 44% | 0.214 | 0.204 | 0.47 | 0.00 | 68K | 0.298 |  |
| 2 | dc22 | 130K | 86% | 44% | 0.150 | 0.166 | 0.54 | 0.00 | 120K | 0.138 |  |
| 3 | m0r0 | 130K | 86% | 70% | 0.150 | 0.161 | 0.72 | 0.06 | 101K | 0.160 |  |
| 4 | g50t | 130K | 58% | 23% | 0.150 | 0.150 | 0.35 | 0.00 | 126K | 0.119 | round 1 |
| 5 | ka59 | 130K | 98% | 54% | 0.119 | 0.136 | 0.63 | 0.01 | 120K | 0.114 |  |
| 6 | ls20 | 130K | 98% | 42% | 0.115 | 0.133 | 0.51 | 0.01 | 117K | 0.114 |  |
| 7 | s5i5 | 130K | 100% | 51% | 0.111 | 0.130 | 0.59 | 0.01 | 109K | 0.120 | round 1 |
| 8 | vc33 | 44K | 100% | 56% | 0.110 | 0.127 | 0.51 | 0.01 | 40K | 0.315 |  |
| 9 | cn04 | 130K | 98% | 46% | 0.118 | 0.120 | 0.60 | 0.01 | 115K | 0.104 |  |
| 10 | r11l | 96K | 100% | 58% | 0.108 | 0.110 | 0.72 | 0.01 | 86K | 0.127 |  |
| 11 | sp80 | 130K | 100% | 37% | 0.104 | 0.104 | 0.58 | 0.03 | 110K | 0.095 | round 1 |
| 12 | cd82 | 110K | 58% | 35% | 0.193 | 0.102 | 0.31 | 0.00 | 104K | 0.098 | round 1 |
| 13 | su15 | 112K | 100% | 70% | 0.094 | 0.094 | 0.80 | 0.06 | 91K | 0.103 |  |
| 14 | tu93 | 61K | 98% | 33% | 0.103 | 0.080 | 0.50 | 0.03 | 59K | 0.135 |  |
| 15 | wa30 | 130K | 100% | 21% | 0.074 | 0.075 | 0.46 | 0.15 | 127K | 0.059 | round 1 |
| 16 | tn36 | 130K | 91% | 14% | 0.069 | 0.074 | 0.38 | 0.12 | 124K | 0.059 | round 1 |
| 17 | sb26 | 80K | 100% | 84% | 0.061 | 0.059 | 0.79 | 0.24 | 59K | 0.100 |  |
| 18 | lf52 | 130K | 100% | 14% | 0.053 | 0.049 | 0.41 | 0.30 | 126K | 0.039 |  |
| 19 | tr87 | 130K | 100% | 98% | 0.010 | 0.035 | 0.92 | 0.83 | 79K | 0.044 | guard |
| 20 | ar25 | 130K | 100% | 93% | 0.029 | 0.029 | 0.95 | 0.56 | 57K | 0.050 |  |
| 21 | sk48 | 130K | 40% | 0% | 0.027 | 0.022 | 0.11 | 0.02 | 130K | 0.017 | round 1 |
| 22 | re86 | 119K | 100% | 100% | 0.000 | 0.010 | 0.96 | 1.00 | 47K | 0.021 | guard |
| 23 | bp35 | 121K | 93% | 0% | 0.007 | 0.010 | 0.21 | 0.56 | 121K | 0.008 | round 1 |
| 24 | ft09 | 96K | 100% | 100% | 0.000 | 0.007 | 0.98 | 1.00 | 27K | 0.027 | guard |
| 25 | lp85 | 96K | 100% | 100% | 0.000 | 0.000 | 1.00 | 1.00 | 44K | 0.000 | guard |

L1, L1 + L2: share of attempts that cleared within the budget. P(flat group): chance that all 8 rollouts land in the same outcome (no L1, L1 only, L1 + L2), so the level reward gives no signal. Var per 100K tokens: signal per trainer cost.

## Picks

| Set (guards excluded) | Games | Mean Var(reward) | Rollouts | Context tokens per wave |
|---|---|---:|---:|---:|
| Round 1 (decided 2026-10-26) | bp35 cd82 g50t s5i5 sk48 sp80 tn36 wa30 | 0.083 | 64 | 7.6M |
| **B200 run (decided 2026-10-27)** | round 1 + su15 lf52 | 0.081 | 80 | 9.3M |

- su15 (rank 13) clears L1 + L2 within budget in 70% of attempts, lf52 (rank 18) in 14%; both always clear L1. Their reward variance is 0.094 and 0.049, so the set's mean stays at round 1's level.
- The highest-signal games not trained: sc25 and dc22 (ranks 1 and 2, 0.204 and 0.166), then m0r0, ka59, ls20.
- In round 1, bp35 and sk48 never clear L2 within budget (ranks 23 and 21) and train L1 only. cd82's level outcomes vary most after sc25, but its L2 takes 8 human actions, so its clears earn little efficiency credit.
- The guards (ft09, lp85, re86, tr87) stay untrained; they are near 100% anyway. The other 11 untrained games measure transfer to new games.

## Caveats

- Rates come from evaluation runs with context trimming, many harness versions and Kaggle's time limits, not from GRPO rollouts of the v6 harness. Wave 0 measures the real spread: the loop logs each game's rewards per wave.
- The ranking moves as the policy learns: a game whose group reaches all-clear stops giving level signal (P(flat group) rises), though efficiency and the thinking factor still vary.
