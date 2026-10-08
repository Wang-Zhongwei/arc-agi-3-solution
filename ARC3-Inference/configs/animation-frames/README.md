# animation-frames

[v1.json](v1.json) uses [new-baseline v2](../new-baseline/v2.json) as its control. Compare separately with [intermediate-frames](../intermediate-frames/README.md).

Adds up to eight sampled animation frames from the last action, at upscale 2. Samples stay in chronological order, collapse consecutive duplicates, and exclude boards matching the start or end. Fatal-action animations are included. Boards between actions are disabled.

Current boards remain at upscale 8. Existing diff images, Daniel's model, 10 active streams, game list, and budgets match the baseline. Upscale 2 produces 128×128 PNGs for 64×64 boards; Daniel's processor enlarges them to 256×256 (64 visual tokens each). See [multimodal previews](../../../docs/multimodal/README.md).

See the [attachment preview](../../../docs/multimodal/frame-attachments.html) for the exact text and image order, illustrated with synthetic boards. Extra images follow the related action or animation text; the current board sits beside the current-state text.

Prepared locally; not uploaded or evaluated. This replaces the combined intermediate-animation-frames experiment.

From `ARC3-Inference/`:

```sh
make kaggle-duck CONFIG_PATH=configs/animation-frames/v1.json KAGGLE_USERNAME=jonathanwang2022 KAGGLE_KERNEL_SLUG=animation-frames KAGGLE_DATASET_REF=jonathanwang2022/animation-frames-src KAGGLE_DRY_RUN=true
```
