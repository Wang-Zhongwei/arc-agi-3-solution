# Multimodal upscale examples

Three starting boards from `jonathanwang2022/bp35-lordhansolo` v12, at
`multimodal.upscale = 1, 4, 8, 10`. Each link opens one PNG. Dimensions are
stored pixel dimensions; your viewer may display them at a different zoom.

## Original request PNGs

| Game | Upscale 1 | Upscale 4 | Upscale 8 | Upscale 10 |
|---|---|---|---|---|
| bp35 | [64 × 64](images/bp35-0a0ad940-upscale-1.png) | [256 × 256](images/bp35-0a0ad940-upscale-4.png) | [512 × 512](images/bp35-0a0ad940-upscale-8.png) | [640 × 640](images/bp35-0a0ad940-upscale-10.png) |
| ls20 | [64 × 64](images/ls20-9607627b-upscale-1.png) | [256 × 256](images/ls20-9607627b-upscale-4.png) | [512 × 512](images/ls20-9607627b-upscale-8.png) | [640 × 640](images/ls20-9607627b-upscale-10.png) |
| vc33 | [64 × 64](images/vc33-5430563c-upscale-1.png) | [256 × 256](images/vc33-5430563c-upscale-4.png) | [512 × 512](images/vc33-5430563c-upscale-8.png) | [640 × 640](images/vc33-5430563c-upscale-10.png) |

## Reconstructed processor RGB previews

| Game | Upscale 1 | Upscale 4 | Upscale 8 | Upscale 10 |
|---|---|---|---|---|
| bp35 | [256 × 256](processor-images/bp35-0a0ad940-upscale-1.png) | [256 × 256](processor-images/bp35-0a0ad940-upscale-4.png) | [512 × 512](processor-images/bp35-0a0ad940-upscale-8.png) | [640 × 640](processor-images/bp35-0a0ad940-upscale-10.png) |
| ls20 | [256 × 256](processor-images/ls20-9607627b-upscale-1.png) | [256 × 256](processor-images/ls20-9607627b-upscale-4.png) | [512 × 512](processor-images/ls20-9607627b-upscale-8.png) | [640 × 640](processor-images/ls20-9607627b-upscale-10.png) |
| vc33 | [256 × 256](processor-images/vc33-5430563c-upscale-1.png) | [256 × 256](processor-images/vc33-5430563c-upscale-4.png) | [512 × 512](processor-images/vc33-5430563c-upscale-8.png) | [640 × 640](processor-images/vc33-5430563c-upscale-10.png) |

The request renderer uses nearest-neighbor scaling. The processor previews
reconstruct the PIL resize stage using the [pinned Intel checkpoint metadata](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/blob/4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897/preprocessor_config.json):
bicubic resizing, minimum area 65,536 pixels, and dimensions aligned to 32 pixels.
Upscale 1 becomes 256 × 256 with softened edges; upscale 4 retains sharp blocks.

Previews show RGB before normalization and patch extraction. They were not
captured from a live server and do not show internal model embeddings.

## Animation and intermediate-board attachments

[Open the attachment preview](frame-attachments.html) to compare animation only, intermediate boards only, and [both included](frame-attachments.html#both). All three use one shared prompt template. Every intermediate board is attached in action order, with its full `history[i].frame (after ACTION)` caption, so the agent can trace each action to its result. Animation samples keep their `last_animation_frames` indices. Synthetic boards show image order, labels, and PNG dimensions before model preprocessing; the [ten-action example](frame-attachments.html#long-intermediate) shows all nine intermediate boards, and expandable cases cover level-up, death/reset, animation with unchanged interior, and guard stops.

Validation: 105 image attachment/scaling/sampling, shared event prompt, animation-description, and frame-diff tests passed for this prompt update. See the [intermediate](../../ARC3-Inference/configs/intermediate-frames/README.md) and [animation](../../ARC3-Inference/configs/animation-frames/README.md) experiment READMEs for submitted versions and results.
