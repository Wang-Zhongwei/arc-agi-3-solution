# animation-frames

[v1.json](v1.json) uses [new-baseline v2](../new-baseline/v2.json) as its control. Compare separately with [intermediate-frames](../intermediate-frames/README.md).

Adds up to eight evenly sampled animation frames from the last action, at upscale 2. Sampling covers `last_animation_frames[:-1]`: the final frame is already shown as the current or death board. Consecutive duplicate frames are collapsed, and samples retain their original indices and chronological order. Fatal-action animations are included. Boards between actions are disabled.

Current boards remain at upscale 8. Existing diff images, Daniel's model, 10 active streams, game list, and budgets match the baseline. Upscale 2 produces 128×128 PNGs for 64×64 boards; Daniel's processor enlarges them to 256×256 (64 visual tokens each). See [multimodal previews](../../../docs/multimodal/README.md).

See the [attachment preview](../../../docs/multimodal/frame-attachments.html) for the current shared prompt template, illustrated with synthetic boards in animation-only, intermediate-only, and combined modes. Current board and diff follow the action text; sampled frames follow the animation text. State and event guidance follow the images. On death, the fatal board and animation precede the reset board and fatal diff.

Submitted as [Kaggle v1](https://www.kaggle.com/code/jonathanwang2022/animation-frames) on 2026-10-08; failed during setup because the competition wheelhouse was missing at the expected mount path. The model and frame experiment did not start. Local artifacts: `runs/kaggle/jonathanwang2022/animation-frames/v01/`. This replaces the combined intermediate-animation-frames experiment.

[v2.json](v2.json) was submitted as Kaggle v2 on 2026-10-08, preserving v1's settings and retrying with setup that accepts both namespaced and flat competition mounts. Wheel installation and offline game lookup use the same resolved mount; 29 relevant tests passed. Local artifacts: `runs/kaggle/jonathanwang2022/animation-frames/v02/`.

Downloaded and evaluated the exact [Kaggle v2 run, script version 356571727](https://www.kaggle.com/code/jonathanwang2022/animation-frames?scriptVersionId=356571727): framework score **48.842383**, with **8 won and 17 cancelled** across 25 games in one pass. Usage: 5,327 actions, 1,186 analyzer turns, and 4,050,430 generated tokens. All 187 output files and the kernel log are saved locally; scores and viewer curves are in that run's `evaluation.json`. This run predates the shared prompt update shown in the preview.

From `ARC3-Inference/`:

```sh
make kaggle-duck CONFIG_PATH=configs/animation-frames/v2.json KAGGLE_USERNAME=jonathanwang2022 KAGGLE_KERNEL_SLUG=animation-frames KAGGLE_DATASET_REF=jonathanwang2022/animation-frames-src
```
