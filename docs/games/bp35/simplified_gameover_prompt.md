# bp35: simplified game-over prompt

The candidate shortens the user prompt sent after a game over: the old prompt and the new one are in [user_prompt_on_death.md](user_prompt_on_death.md), and the code change is commit `58b909f`. It cuts the long death explanation, the separate "fatal action" paragraph and the "do not replay" paragraph. It keeps the history indices to inspect, the BAR RULE and the per-action trace.

## Setup

- Game: `bp35-0a0ad940` × 10 passes, 200k generated tokens per game.
- Configs: [v1.json](../../../ARC3-Inference/configs/bp25/v1.json) and [v2.json](../../../ARC3-Inference/configs/bp25/v2.json). They are identical, so the game-over prompt is the only difference.
- Notebook: [jonathanwang2022/bp25-experiment](https://www.kaggle.com/code/jonathanwang2022/bp25-experiment).

| Kaggle version | Arm | Code |
|---|---|---|
| 1 | — | Failed before any game ran (`games: 0`): `main` lacks the offline-games notebook fix (`5b99998`). |
| 2 | Baseline | `main` at `bb1a2c0`, plus `5b99998` |
| 3 | Candidate | version 2's code, plus the simplified prompt |

## Results

| | Baseline (version 2) | Candidate (version 3) | p (Welch / Mann-Whitney) |
|---|---:|---:|---|
| Mean score | 2.82 ± 1.98 | 3.27 ± 3.30 | 0.73 / 0.49 |
| Median score | 2.22 | 2.03 | |
| Levels cleared, mean | 1.2 | 1.5 | 0.27 / 0.32 |
| Actions per pass | 67 | 107 | 0.013 / 0.021 |
| Analyzer turns per pass | 28 | 41 | 0.007 / 0.015 |
| Tokens per pass | ~170k | ~193k | |

Levels cleared per pass:

- Baseline: 1, 1, 1, 1, 1, 1, 1, 1, 2, 2.
- Candidate: 1, 1, 1, 1, 1, 1, 2, 2, 2, 3.

The candidate's only 3-level pass scored 11.66.

There is only one game, so the across-game test from the [t-test guide](../../rules/t-test.md) does not apply. The p-values compare the 10 pass scores of each arm.

## Conclusion

The score is slightly higher but not significant. The clear effect is efficiency: on a similar token budget, the candidate took 59% more actions over 43% more turns. A shorter death prompt leaves more room for play. Whether that turns into more cleared levels needs the full game set, where the across-game test applies.

## Data

Both run folders, with `evaluation.json`:

- `ARC3-Inference/runs/kaggle/jonathanwang2022/bp25-experiment/v02/` (baseline)
- `ARC3-Inference/runs/kaggle/jonathanwang2022/bp25-experiment/v03/` (candidate)
