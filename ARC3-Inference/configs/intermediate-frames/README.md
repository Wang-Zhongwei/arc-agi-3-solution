# intermediate-frames

[v1.json](v1.json) uses [new-baseline v2](../new-baseline/v2.json) as its control. Compare separately with [animation-frames](../animation-frames/README.md).

Adds boards between actions in the previous Python call, at upscale 2, excluding its final board. Includes the fatal board before automatic RESET. Automatic animation images are disabled; baseline animation metadata and Python inspection remain available.

Current boards remain at upscale 8. Existing diff images, Daniel's model, 10 active streams, game list, and budgets match the baseline. Upscale 2 produces 128×128 PNGs for 64×64 boards; Daniel's processor enlarges them to 256×256 (64 visual tokens each). See [multimodal previews](../../../docs/multimodal/README.md).

See the [attachment preview](../../../docs/multimodal/frame-attachments.html) for the exact text and image order, illustrated with synthetic boards. Extra images follow the related action or animation text; the current board sits beside the current-state text.

Prepared locally; not uploaded or evaluated. This replaces the combined intermediate-animation-frames experiment.

From `ARC3-Inference/`:

```sh
make kaggle-duck CONFIG_PATH=configs/intermediate-frames/v1.json KAGGLE_USERNAME=jonathanwang2022 KAGGLE_KERNEL_SLUG=intermediate-frames KAGGLE_DATASET_REF=jonathanwang2022/intermediate-frames-src KAGGLE_DRY_RUN=true
```
