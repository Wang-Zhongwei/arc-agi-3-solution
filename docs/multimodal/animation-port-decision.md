# Port sampled animation frames into new-baseline

Enable the animation-only treatment in
[`new-baseline/v4.json`](../../ARC3-Inference/configs/new-baseline/v4.json).
It keeps the existing `ARC3_ANIMATION_IMAGE` and
`ARC3_ANIMATION_DEATH_IMAGE` knobs: `sampled` attaches up to eight frames from
the last action, and `off` disables those attachments. The frames use upscale
2; the current board stays at upscale 8. Intermediate action boards remain
disabled. No shared prompt or frame-rendering code changes are needed.

## Evidence from completed runs

These are three **single-pass, 25-game** Kaggle runs using Daniel/Pennyroyal,
the same game list, 10 active streams, and the same 120-minute public budget.
The [animation-only v2](../../ARC3-Inference/runs/kaggle/jonathanwang2022/animation-frames/v02/evaluation.json)
and [intermediate-only v1](../../ARC3-Inference/runs/kaggle/jonathanwang2022/intermediate-frames/v01/evaluation.json)
configs differ in which extra frame type they attach. The
[combined v1](../../ARC3-Inference/runs/kaggle/jonathanwang2022/intermediate-animation-frames/v01/evaluation.json)
run also raises the regular board upscale from 8 to 10, so it does not isolate
the value of adding intermediate boards to animation boards.

| Run | Extra images | Mean framework score | Won / cancelled | Actions | Generated tokens |
|---|---|---:|---:|---:|---:|
| `intermediate-frames/v01` | Every intermediate action board, upscale 2 | 44.5643 | 9 / 16 | 4,979 | 4,063,507 |
| `animation-frames/v02` | Up to eight sampled animation frames, upscale 2 | 48.8424 | 8 / 17 | 5,327 | 4,050,430 |
| `intermediate-animation-frames/v01` | Both types, upscale 2; current board upscale 10 | 49.4942 | 9 / 16 | 4,647 | 4,011,260 |

Pairing the same 25 games by ID, animation-only scored **4.28 points higher**
on average than intermediate-only (9 games higher, 7 lower, 9 tied). The
combined run scored **0.65 points higher** than animation-only (7 higher,
7 lower, 11 tied). Two-sided paired across-game t-tests give p = 0.375 and
p = 0.860, respectively; their 95% intervals for the mean differences are
[-5.48, 14.04] and [-6.88, 8.19]. These data do not establish a reliable
score advantage for either frame type. The small combined mean gain is also
confounded by its larger current board.

Animation-only is the narrower port: it had the better observed mean of the
two isolated treatments and avoids attaching every board in a long action
batch. The toggle preserves a clean off setting for a future repeat-run A/B
test. This choice is directional, not a claim that animation frames improved
the underlying model or will reproduce the observed score.

`new-baseline/v02` did not finish a comparable 25-game run.
`new-baseline/v03` scored 40.3289 in a complete run, but also changed
embedding offload and startup behavior relative to the frame experiments'
v2 control; it is context, not an isolated animation-frame baseline. The new
v4 config inherits v3's offload and startup timeout. Its 2,048-token analyzer
yield setting remains unchanged, so the v4 config has not been evaluated as
the combined set of settings.

The per-game scores and usage numbers above come from the three exact
`evaluation.json` files. The t-tests use the 25 per-game `score` values in
those files, paired by game ID (`scipy.stats.ttest_rel`, two-sided). The
[animation-frame README](../../ARC3-Inference/configs/animation-frames/README.md)
records the exact Kaggle v2 version and attachment semantics; the
[combined README](../../ARC3-Inference/configs/intermediate-animation-frames/README.md)
records its submitted v1 source and the current-frame upscale difference.
