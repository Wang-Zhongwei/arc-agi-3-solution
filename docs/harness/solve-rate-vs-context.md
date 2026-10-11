# Level 1-3 solve rate vs untrimmed context

Question: on the 15 harder public games, does the chance of solving the current level drop once the context fills and trimming starts (the 118,272-token budget)?

![Solve rate vs untrimmed context](images/solve-rate-vs-context.png)

**Answer: level 1 dips at the trimming line; levels 2 and 3 do not.** Level 1 holds at roughly 15-25% per 10k tokens from 30k to 118k, then drops to about 8-9% in the next two windows. Measured from the start of each level, level 2 ramps up over the first 40k to about 8-10%, holds there to about 100k, then sits near 5-7%. Level 3 ramps up over the first 50k, peaks at 13-18% around 60-80k, holds at about 10-15% to 130k, then falls off. Their trimming points vary with how full the context was when the level began, so the budget line applies to level 1 only.

The level-1 dip: of attempts unsolved at 90k, 55/112 (49%) solved before 118k; of those unsolved at 118k, 16/54 (30%) solved before 150k (Fisher's exact on that 2x2 table: odds ratio 2.29, p = 0.020, optimistic because an attempt can be in both rows). It is not uniform: bp35 20/25 -> 2/5, g50t 10/19 -> 2/7 and cd82 6/12 -> 2/6 drop, while sk48 rises 4/26 -> 6/21. Part of it comes from which games are still on level 1 by then: 21 of the 54 attempts past 118k are sk48. Comparing within the same game, the odds ratio is 0.60 (95% CI 0.28-1.27, p = 0.18). The direction fits trimming hurting level 1, but this sample cannot separate it from hard games remaining.

Before trimming, level 1 gets easier as context grows. From 30k to 118k, within the same game, the odds of solving in the next 10k rise 1.24x per +10k of untrimmed context (95% CI 1.12-1.36, p = 2e-5, errors clustered by attempt). Pooled across games the trend is flat (0.99, CI 0.92-1.08), because the mix shifts toward hard games as context grows. Together with the dip after the line, this points to retained context helping on level 1. A direct test would be an A/B run with a larger context budget on the hard games, or with the first trim delayed.

## Levels 2-3 vs current context

The figure above cannot show trimming for levels 2 and 3, because each attempt starts at a different point of the context sawtooth. This one uses the current (trimmed) context instead, which cycles between about 58k and 118k, and keeps only turns after the game's first trim.

![Levels 2-3 vs current context](images/solve-rate-vs-current-context.png)

**No trend with current context.**
- Level 2 stays at about 0.06-0.08 level-ups per 10k of added context across 50-110k, with the two highest windows slightly lower (0.060, 0.051).
- Level 3 moves between 0.06 and 0.12 with no direction.
- Within game and level, the rate changes by a factor of 0.97 per +10k of current context (95% CI 0.92-1.03, p = 0.33; 10,899 turns, 384 level-ups).
- On the runs that log real prompt sizes, the factor is 1.13 (CI 0.97-1.30, p = 0.11; 1,690 turns, 73 level-ups).

The current context is simulated for most runs, so it is noisy. That noise pulls any real trend toward flat. On real prompt sizes the hint is toward more context helping, the same direction as level 1, but it is not significant.

How it is measured:
- **x:** the current context at the start of each turn. Only the request-log runs record it, so for the 30 runs it is simulated from the untrimmed estimate: it grows with each turn, and once it passes 118k it drops to 58k. Those values were fitted against server `prompt_tokens`: median error 3k tokens, but correlation only 0.60 inside the 50-118k band, because each trim lands a little differently and the phase drifts.
- **y:** level-ups per 10k of context added while in that window. The bars are exact Poisson 95% intervals.
- **Test:** a Poisson model with game and level fixed effects, offset by added context, errors clustered by game run.

## Does keeping the last level-up in context help?

For levels 2 and 3, each turn is marked by whether the turn that completed the previous level is still in the trimmed context. After a trim, the context keeps roughly the newest (current - base) tokens, so the level-up is kept if the untrimmed context added since it fits in that.

At the same tokens since the level-up, some attempts have already lost it to a trim and others have not, depending on where in the sawtooth the level began. The comparison uses only 20k bands where both cases occur, with game and level fixed effects, offset by added context, errors clustered by game run.

![Last level-up kept vs trimmed away](images/levelup-in-context.png)

| Data | Level-up rate, kept vs trimmed away | 95% CI | p | Level-ups |
|---|---:|---:|---:|---:|
| Simulated trimming, 30 runs | 1.07 | 0.78-1.45 | 0.68 | 300 |
| Real prompt sizes, request-log runs | 0.50 | 0.19-1.30 | 0.16 | 58 |

**No detectable effect.** The larger simulated sample centres on no difference, the small real-data sample leans the other way, and the bands disagree in direction (simulated, per 10k added: 60-80k kept 0.116 vs trimmed 0.088; 80-100k kept 0.070 vs trimmed 0.105). Simulated trims mislabel some turns, which pulls the estimate toward 1. An A/B run of `context.retain_last_level_up` (PR #31) would settle it. Script: [`scripts/levelup_in_context.py`](scripts/levelup_in_context.py).

## Data

- **Games:** the later 15 public games (bp35 cd82 cn04 dc22 g50t ka59 lf52 ls20 m0r0 s5i5 sk48 sp80 su15 tn36 wa30). The 10-game runs used the other 10.
- **Runs:** every run in `ARC3-Inference/runs/kaggle` on Daniel's harness: 118,272-token budget, 3,072-token tool output, no sampled-animation or intermediate-frame images. That is 30 runs and 320 game runs. Short runs and the early-level-recovery arms are included; a run that ends mid-level counts that level as unsolved at that point.
- **Excluded:** main's harness runs (1,024-token tool output, sampled animations or intermediate frames), the `bp35-lordhansolo` series, and runs with other budgets.
- **Sample size:** level 1 has 407 attempts (373 solved), level 2 has 373 (273 solved), level 3 has 273 (215 solved).

## Method

**x-axis.** x is the untrimmed context added since the level started: everything that entered the conversation since the previous level-up, as if history were never trimmed. Level 1 counts from game start, so it includes the base prompt.
- Only five runs log server `prompt_tokens`, so x is rebuilt from each transcript. It counts user prompts, thinking, replies, tool calls and tool results, plus a cost per attached image. The per-turn copy of the system prompt and the status blocks are excluded.
- The fit (`base + chars/2.92 + 350 * images`) uses the runs that saved request logs, at each turn's first request before the game's first trim (n = 2,347).
- Median absolute error is 3.3%, and the 90th percentile is 8.5%.

**Level-ups.** Level-ups come from each game's events file. A level-up at action *a* ends at the untrimmed context reached by the end of the turn that executed *a*. The next level starts there.

**y-axis.** For each 10k window: among attempts on that level that started before the window and were unsolved at its start, the share solved inside it. Windows with fewer than 20 attempts at risk are omitted. The bars are 95% Wilson intervals.

## Caveats

- Later windows hold the attempts that are still unsolved, which are mostly the hard games. A falling raw curve can therefore come from the mix of games rather than from the context.
- One attempt counts in every window it spans, so the windows are not independent, and the intervals are narrower than they should be.

## Reproduce

From the repo root:

```bash
RUNS_DIR=<main checkout>/ARC3-Inference/runs/kaggle python3 docs/harness/scripts/solve_rate_vs_context.py
```

[`scripts/solve_rate_vs_context.py`](scripts/solve_rate_vs_context.py) writes both figures and prints the calibration, the run list and the per-window counts.
