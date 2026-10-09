# intermediate-frames

[v1.json](v1.json) uses [new-baseline v2](../new-baseline/v2.json) as its control. Compare separately with [animation-frames](../animation-frames/README.md).

Adds boards between actions in the previous Python call, at upscale 2, excluding its final board. Includes the fatal board before automatic RESET. Automatic animation images are disabled; baseline animation metadata and Python inspection remain available.

Current boards remain at upscale 8. Existing diff images, Daniel's model, 10 active streams, game list, and budgets match the baseline. Upscale 2 produces 128×128 PNGs for 64×64 boards; Daniel's processor enlarges them to 256×256 (64 visual tokens each). See [multimodal previews](../../../docs/multimodal/README.md).

See the [attachment preview](../../../docs/multimodal/frame-attachments.html) for the exact text and image order, illustrated with synthetic boards. Every intermediate board appears in action order with a full caption such as `history[254].frame (after UP):`, followed by the current board and diff. The executed actions are listed in the same order, with consecutive repeats grouped; if that list is long, its middle groups are abbreviated, while every image remains labeled. Current-state text follows the images.

The current prompt template is shared with animation-only and combined modes. Animation evidence stays beside its summary; level-up, death, and guard-stop turns use their own concise guidance. A fatal board is shown once before the automatic RESET board. The submitted v1 run predates this prompt update.

Submitted as [Kaggle v1](https://www.kaggle.com/code/jonathanwang2022/intermediate-frames) on 2026-10-08; completed all 25 games, with 9 wins and a harness mean score of 44.56. This is the local public-game summary, not a competition score or a baseline comparison. Local artifacts: `runs/kaggle/jonathanwang2022/intermediate-frames/v01/`. This replaces the combined intermediate-animation-frames experiment.

Downloaded and evaluated with `make evaluate_run`: framework score **44.564306**, with 9 won and 16 cancelled. Scores and viewer curves are saved in that run's `evaluation.json`.

From `ARC3-Inference/`:

```sh
make kaggle-duck CONFIG_PATH=configs/intermediate-frames/v1.json KAGGLE_USERNAME=jonathanwang2022 KAGGLE_KERNEL_SLUG=intermediate-frames KAGGLE_DATASET_REF=jonathanwang2022/intermediate-frames-src
```
