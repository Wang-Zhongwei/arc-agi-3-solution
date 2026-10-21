# Research

Restarted on 2026-10-14. Everything before that date is in [`archive/`](archive/README.md), unchanged.
This page holds only what the next round of work starts from.

## Where we are

- **Goal:** 10 of the 25 public games fully won in the 121-minute offline run, which stands in for the
  110 private games in Kaggle's 9-hour run. Score is reported with wins and time-to-win first.
- **Best run:** `mem-repair-10x` (`mellamoa/v4-mem-repair-10x` v1, 2026-10-14):
  - 54.69 mean score, 116 levels, 5.03M tokens.
  - **10 games won**: ar25, cd82, cn04, ft09, lp85, r11l, sb26, sc25, tr87 and vc33.
  - The exact notebook is in [`kaggle/notebooks/`](../kaggle/notebooks/README.md).
  - The code is on `main` (PR #174, merged 2026-10-14, carrying mem-unified → stall-fix → mem-repair).
  - It is one run. By the repeat rule, it is not yet an alpha over `mem-unified-10x` (56.09, 8 wins) or
    `user-combo-10x` (56.94, 10 wins).
- **Leaderboard:** best public score **38.22**, from `mem-repair-10x` v1 (mellamoa, submission 57200258, 2026-10-15;
  the first attempt hit a Kaggle system error and was resubmitted). Before that: 34.04 (snowiers, 2026-10-14).
- **Report on the best run:** [`reports/mem-repair-2026-10-14.md`](reports/mem-repair-2026-10-14.md). It covers:
  - the result;
  - checks that the code is game- and time-agnostic;
  - the su15 regression;
  - the next fixes.

- **Second audit of the best run (2026-10-15):** `mem-repair2-10x` (PR #178,
  [`reports/mem-repair2-2026-10-15.md`](reports/mem-repair2-2026-10-15.md)) did not beat it: 8 wins, 49.41, 104 levels in
  one run (paired −5.29 vs mem-repair, P(better) 0.08). Its flags fired as built; only 2 of the 7 targeted stalls it
  reached converted.
- **Model prong (2026-10-15):** [`reports/rsi-proposals-2026-10-15.md`](reports/rsi-proposals-2026-10-15.md) proposes six
  ways to put the gains in the weights (R1 self-distilled draft head first). The self-improvement loop and its model
  version control are on `arm/rsi` (PR #177, `game-reasoning/gamereason/rsi/`).
- **Qwen3.8-27B instead of Flash-Next (2026-10-17):** `q27-dflash2-10x` (PR #179, FP8 + DFlash2) scored 14.51 with 0 wins
  and 52 levels in one run (paired −40.2 vs mem-repair). It needs as many actions per level as Flash-Next (49.6 vs 49.8)
  but thinks twice as long per action (1,793 vs 872 tokens), and serving speed was about level (628 vs ~690 tok/s), so it
  played half as far. Flash-Next stays.
- **Final-level hold and victory lap (2026-10-17, PRs #180-#182, merged; both flags OFF by default):**
  - `final-hold-lap-10x` = mem-repair-10x + `ARC3_SCHED_FINAL_HOLD=1` + `ARC3_VICTORY_LAP=1`, two runs on snowiers.
    v1 is invalid: a 1,299 s weight load released the benchmark with the server still loading and every game lost
    about 15 of its 121 min (47.23). v2 (clean start): **55.08, 9 wins, 117 levels**, paired +0.39 vs mem-repair
    (t +0.06): a tie.
  - **Hold** (`ARC3_SCHED_FINAL_HOLD`): no measurable help or harm in two runs; it also keeps the slot of games stuck
    on their last level (r11l, tu93 in both runs). Not in the recommended config. Over 35 full runs, 90% of clears of
    a game's last two levels came within 68k tokens on the level (95% within 98k): a release point for a future
    limiter.
  - **Victory lap** (`ARC3_VICTORY_LAP`): 16 wins, 16 laps skipped (nothing shorter to replay), 0 errors, 0 points. The
    replay has not yet executed on Kaggle. It can only add on a won game whose failed attempts it can cut.
  - **Always on (not a flag):** offline test runs now wait up to 45 min for the model server before the benchmark
    starts (#182). Competition reruns keep the 12-min release.
  - Recommended config unchanged: **mem-repair-10x**.

- **New goal (2026-10-17): a consistent leaderboard 60 through fine-tuning.** Plan:
  [`reports/finetune-plan-2026-10-17.md`](reports/finetune-plan-2026-10-17.md), with raw research in
  [`data/finetune-research-2026-10-17/`](data/finetune-research-2026-10-17/).
  - Keep Qwen3.8-Flash-Next: it is the only model with evidence of a leaderboard score of 30 or more.
  - Train it with ms-swift Megatron LoRA on 8× H200.
  - Targets: tokens per action and exploration on stuck levels.
  - Scheme: one mixed SFT stage in the MiniCPM decay-mix style, then one rejection-sampling round.
  - The deadline (~11-02) rules out RL.

## What we know (carried over from the archive)

**Noise and judging**
- One full run moves about ±2.7 on the mean, and a single game swings by 50–80 points.
- An arm counts only if repeated runs score consistently higher.
- Compare runs at matched wall-clock time, and use `make significance` for paired per-game tests.

**What helps the model**
- The model (Qwen3.8-Flash-Next W4A16, local) uses facts the harness computes and writes into tool
  results or turn messages.
- It ignores, or is hurt by, advice text, optional structure and free-form helpers.
- In state-lab tests these facts moved the right next move:

  | Fact | Right next move (out of 20) |
  |---|---|
  | Moved objects | 0 → 16 |
  | Touching objects split | 5 → 10 |
  | The model's own quoted rules | 3 → 9 |

**What the memory does wrong**
- Where the memory hurts, it is because a line is wrong, stale or noisy, or a correct fact is dropped.
  Adding more lines is not the lever. ([archive/reports/sf-memory-reads-2026-10-14.md](archive/reports/sf-memory-reads-2026-10-14.md))
- About 30 levels per run stall (at least 60k tokens or 45 min).
- After a context trim, games wait 20–80 min for a slot. These waits are the largest wall-clock loss.
  They are zero-sum: the server runs 13–14 requests throughout. An oracle reallocation bound is about
  +4.9 points. ([archive/reports/stalls-2026-10-14.md](archive/reports/stalls-2026-10-14.md))

**How often each game is won** (over the 14 full runs of the user-combo and memory arms, 2026-10-09 to 10-14)

| Games | Wins out of 14 |
|---|---|
| ar25, cd82, ft09, lp85 | 14 |
| sb26 | 13 |
| tr87 | 11 |
| r11l | 7 |
| sc25, vc33 | 6 |
| tu93 | 5 |
| tn36, cn04 | 2 |
| su15 | 1 |

## Layout

- `reports/`: write-ups from 2026-10-14 on.
- `data/`: the scripts and tables behind them.
- `archive/`: the earlier material. It holds:
  - the old index ([`archive/README.md`](archive/README.md));
  - reports, data and workflow agent reports;
  - studies of the old Wang/Duck harness;
  - a snapshot of the arms log.
- The living notes (`arms.md`, `scratchpad.md`, `research-sweep.md`) sit outside the repo. Raw run
  logs are in [`runs/`](../runs/).
