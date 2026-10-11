# Level 1-3 solve rate vs untrimmed context

Question: on the 15 harder public games, does the chance of solving the current level drop once the context fills and trimming starts (the 118,272-token budget)?

![Solve rate vs untrimmed context](images/solve-rate-vs-context.png)

**Answer: level 1 dips at the trimming line; levels 2 and 3 do not.** Level 1 holds at roughly 15-25% per 10k tokens from 30k to 118k, then drops to about 8-9% in the next two windows. Measured from the start of each level, level 2 ramps up over the first 40k to about 8-10%, holds there to about 100k, then sits near 5-7%. Level 3 ramps up over the first 50k, peaks at 13-18% around 60-80k, holds at about 10-15% to 130k, then falls off. Their trimming points vary with how full the context was when the level began, so the budget line applies to level 1 only.

The level-1 dip: of attempts unsolved at 90k, 55/112 (49%) solved before 118k; of those unsolved at 118k, 16/54 (30%) solved before 150k (Fisher's exact on that 2x2 table: odds ratio 2.29, p = 0.020, optimistic because an attempt can be in both rows). It is not uniform: bp35 20/25 -> 2/5, g50t 10/19 -> 2/7 and cd82 6/12 -> 2/6 drop, while sk48 rises 4/26 -> 6/21. Part of it comes from which games are still on level 1 by then: 21 of the 54 attempts past 118k are sk48. Comparing within the same game, the odds ratio is 0.60 (95% CI 0.28-1.27, p = 0.18). The direction fits trimming hurting level 1, but this sample cannot separate it from hard games remaining.

Before trimming, level 1 gets easier as context grows. From 30k to 118k, within the same game, the odds of solving in the next 10k rise 1.24x per +10k of untrimmed context (95% CI 1.12-1.36, p = 2e-5, errors clustered by attempt). Pooled across games the trend is flat (0.99, CI 0.92-1.08), because the mix shifts toward hard games as context grows. Together with the dip after the line, this points to retained context helping on level 1. A direct test would be an A/B run with a larger context budget on the hard games, or with the first trim delayed.

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

[`scripts/solve_rate_vs_context.py`](scripts/solve_rate_vs_context.py) writes the figure and prints the calibration, the run list and the per-window counts.
