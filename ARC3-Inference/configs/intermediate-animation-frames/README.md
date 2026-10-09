# intermediate-animation-frames

[v1.json](v1.json) combines the [intermediate-frames](../intermediate-frames/README.md) and [animation-frames](../animation-frames/README.md) experiments. It starts from `animation-frames/v2.json`, enables every intermediate board at upscale 2, and sets the regular current-board upscale to **10**. The last action's animation contributes up to eight evenly sampled frames at upscale 2; its final frame is already shown as the current or death board. The [combined preview](../../../docs/multimodal/frame-attachments.html#both) shows the shared prompt order.

The model, game list, 10 active streams, budgets, and Pennyroyal serving settings match the two single-feature configs. Embedding-table offload is **off**: this config has no `OFFLOAD_EMBED_TOKENS` override, and the launcher defaults it to `false`. Because the regular board upscale also changes from 8 to 10, this is a combined treatment rather than an isolated test of adding the second frame type.

Submitted as [Kaggle v1](https://www.kaggle.com/code/jonathanwang2022/intermediate-animation-frames) on 2026-10-09, using source commit `2a7dfc2` and dataset `jonathanwang2022/intermediate-animation-frames-src`. Kaggle accepted the launch; no result has been downloaded or evaluated yet. The exact local submitted bundle is under `/private/tmp/arc3-intermediate-animation-prep/kaggle/jonathanwang2022/intermediate-animation-frames/v01/`. The earlier packaging dry run also succeeded.

From `ARC3-Inference/`, the submission command was:

```sh
make kaggle-duck CONFIG_PATH=configs/intermediate-animation-frames/v1.json KAGGLE_USERNAME=jonathanwang2022 KAGGLE_KERNEL_SLUG=intermediate-animation-frames KAGGLE_DATASET_REF=jonathanwang2022/intermediate-animation-frames-src EXPERIMENTS_DIR=/private/tmp/arc3-intermediate-animation-prep
```
