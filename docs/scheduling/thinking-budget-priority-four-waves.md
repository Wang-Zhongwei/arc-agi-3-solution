# Thinking budget with the priority allocator in four waves

Notebook: [`jonathanwang2022/thinking-budget-priority-four-waves`](https://www.kaggle.com/code/jonathanwang2022/thinking-budget-priority-four-waves)
Config: `ARC3-Inference/configs/thinking-budget-priority-four-waves/v1.json`

## Question

[`ticbudja/thinking-budget-and-user-prompt-trim`](https://www.kaggle.com/code/ticbudja/thinking-budget-and-user-prompt-trim) v3 (thinking budget 8192, user-prompt trim) ran the Yield-UCB scheduler with every game in one wave. This notebook runs the same source with the original allocator and the 1001 run shape, so the two isolate the scheduling change.

## What differs from v3

| | v3 | this notebook |
|---|---|---|
| Source | `e7e80c3` | `e7e80c3` + soft stop (this branch) |
| `ARC3_SCHEDULER` | `ucb` (actions, c = 0.0002, recent weight 0.5) | `priority` (UCB settings removed) |
| Lanes (`concurrent_jobs`) | 120 | 28 |
| Per-game cap | 532 min | 132 min |
| Waves on the ~110 hidden games | 1 | 4 |
| `ARC3_GAME_SOFT_STOP_SECONDS` | unset | 300 |
| Public run length (`MAX_EXPERIMENT_RUNTIME_MINUTES`) | 150 | 150 |

Everything else in `harness_env`, `analyzer` and `kaggle` is copied from `configs/thinking-budget-and-user-prompt-trim/v3.json`.

The Makefile at this commit pins the one-wave shape on `kaggle-duck`, so the four-wave shape is set on the command line:

```
make kaggle-duck CONFIG_PATH=configs/thinking-budget-priority-four-waves/v1.json \
  KAGGLE_USERNAME=jonathanwang2022 CONCURRENT_JOBS=28 MAX_RUNTIME_MINUTES=132
```

## Soft stop at the wave boundary

In four waves each lane plays about four games back to back, each capped at 132 min. A game used to be cut wherever it was at the cap, usually between thinking continuations of a turn that had not acted yet. With `ARC3_GAME_SOFT_STOP_SECONDS=300`, from minute 127 a game starts no new turn, but the turn in flight finishes (continuations, actions, and no gate handover). Only minute 132 cuts it. See CONFIGURATION.md, "Soft stop at the per-game cap".

The cap is not extended, because the hidden run has no slack: 4 × 132 min plus about 9 min of server startup is 537 of the notebook's 540 minutes.

## Caveats

- The 25 public games fit in 28 lanes, so the public run is one wave under either shape. Waves only differ on the hidden set. On the public run, the differences are the allocator, the 132-min cap and the soft stop.
- The gate keeps one shared clock: the earliest game start and the earliest deadline. With `ARC3_PRIORITY_TAIL_FADE=1`, games in waves 2 to 4 see that deadline as passed, so their tail term B is faded to zero. This was also true of earlier four-wave runs and is left unchanged here.

## Results

Version 1 was pushed on 2026-10-04 from commit `458e575` (clean tree), with 28 lanes, a 132-min per-game cap and a 150-min public run. It was cancelled while queued, because its public run would not finish in time to submit that day.

Version 2 was pushed the same day from `eea6f57`, with `MAX_EXPERIMENT_RUNTIME_MINUTES=30` added to the command above. That makes the public run a quick health check: it stops 20 min after notebook start, which is about 10 min of play after server startup. It cannot be compared with v3's 150-min public run. The competition run ignores this soft end, so the hidden set still plays 28 lanes × 132 min in four waves. Record scores here after `make pull-kaggle NOTEBOOK=jonathanwang2022/thinking-budget-priority-four-waves VERSION=2`.
