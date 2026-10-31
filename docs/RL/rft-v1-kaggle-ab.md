# RFT v1 on Kaggle (2026-10-21)

The first RFT model, our W4A16 checkpoint with the 2-epoch LoRA merged in, scores **8.6 points per game below** the base model on the 25 public games. The paired t-test p is 0.003 and the Wilcoxon p is 0.004. This is a regression, not a null result. It is concentrated on the games and levels we trained on.

Training and data: [RL README](../../ARC3-Inference/inference/rft/README.md), [dataset rft_v1](../../ARC3-Inference/inference/rft/data/rft_v1.md), [Logs](../../ARC3-Inference/inference/rft/Logs.md).

## Runs

Both RFT runs use `configs/rft-model/v1.json`, which is the reset-daniel-context v1 config with two changes:

- `model_source` = `jonathanwang2022/qwen3-8-flash-next-w4a16-rft-v1/Transformers/default/1`;
- `kaggle.ple_model_source` links the base model's PLE files.

v2 (`configs/rft-model/v2.json`) also lowers `multimodal.upscale` from 12 to 10; v2 vs v1 is −2.95 (p = 0.50). The Lord NVFP4 MTP draft is unchanged.

The baselines are the five most recent runs of the same harness family. The first two run the RFT runs' exact harness and config, with the base model.

| Run | Kaggle notebook | Code | Mean | Median | Levels | Actions | Generated | tok/s (server mean) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| RDC (tic) | `ticbudja/reset-daniel-context` v1 | 039d528 | 56.57 | 58.33 | 121 | 6,697 | 5.71M | 878 |
| RDC | `jonathanwang2022/reset-daniel-context` v1 | d2d6cf4 | 59.07 | 58.33 | 126 | 6,489 | 5.34M | 828 |
| reset-baseline | `ticbudja/reset-baseline` v1 | 8db9b61 | 58.13 | 58.33 | 124 | 6,743 | 5.66M | 880 |
| animation-min-frames | `ticbudja/animation-min-frames` v1 | b3579b9 | 56.73 | 58.33 | 126 | 7,236 | 5.72M | 869 |
| world-model-notes | `ticbudja/world-model-notes` v1 | f406cd0 | 56.04 | 53.54 | 122 | 6,700 | 5.56M | 865 |
| **RFT v1** | `jonathanwang2022/rft-model` v1 | 7298a46 | **50.21** | 47.24 | 117 | 7,569 | 5.57M | 903 |
| **RFT v2** | `jonathanwang2022/rft-model` v2 | 7298a46 + v2 config (7c4ef43) | **47.26** | 44.69 | 109 | 8,285 | 5.66M | 908 |

Paired comparisons over the 25 games. "Worse / better" counts games that moved by more than 0.5 points.

| Comparison | Δ | t-test p | Wilcoxon p | Worse / better |
|---|---:|---:|---:|---|
| RFT v1 vs RDC (tic) | −6.35 | 0.13 | 0.053 | 14 / 5 |
| RFT v1 vs RDC | −8.86 | 0.037 | 0.039 | 14 / 6 |
| RFT v2 vs RDC (tic) | −9.31 | 0.049 | 0.068 | 12 / 9 |
| RFT v2 vs RDC | −11.81 | 0.021 | 0.016 | 13 / 5 |
| **Mean of both RFT runs vs mean of the 5 baselines** | **−8.57** | **0.003** | **0.004** | 17 / 4 |

Both runs were healthy: no server errors, and the PLE link loaded. MTP draft acceptance rose to 62.9% and 63.4%, from 57.6–58.4% across eight baseline runs (the five above plus world-model-notes v2, last-animation v1 and context-64-128k v1). Mean acceptance length went from 2.73 (RDC tic) to 2.89 and 2.90. The jump is far outside the baseline spread, so the model's outputs really changed. It bought only ~3% tok/s.

## Where the loss is

**Trained vs untrained games.** The 17 games with training samples lost 12.1 points on average. The 8 games without any lost 1.1. The baseline score does not predict the drop (Spearman 0.17, p = 0.40), so this is not just the easy games sitting at the ceiling.

**The most-trained games regressed in both runs.** The levels below are levels completed: five baseline runs in the order of the table above, then the two RFT runs.

| Game | Training samples (rank of 17) | Trained levels | Baselines | RFT v1, v2 |
|---|---:|---|---|---|
| ls20 | 41 (1) | L4, L5 | 5 5 5 4 5 | 2, 1 |
| wa30 | 39 (2) | L2, L3, L4 | 4 3 3 4 4 | 2, 2 |
| sp80 | 27 (4) | L2, L4, L5 | 4 3 4 4 3 | 1, 1 |
| ka59 | 13 (13) | L4, L6 | 5 4 4 6 6 | 3, 3 |

**They stall on the trained levels themselves.** Below are clears out of reaches, over ten baseline runs (the five above plus world-model-notes v2, last-animation v1, context-60-110k v1, context-64-128k v1 and level-up-board v1) vs the two RFT runs:

- ka59 L4: 10/10 → 0/2
- sp80 L2: 7/10 → 0/2
- wa30 L3: 7/10 → 0/2
- sk48 L2: 3/7 → 0/2
- re86 L7: 3/8 → 0/2

In sp80, both RFT runs spent 67–70 turns and ~65 actions on L2 without clearing it. The baselines spent 24–32 turns. ls20 in v2 never left L1 (48 turns, 318 actions), which every baseline clears in about 20 actions. In that run the model saw the key icon change when it stepped on the "+" and called it "an animated mascot, not gameplay state". The word "mascot" appears 57 times in that transcript and in no other ls20 run.

Summed over all levels, trained levels fall from a 0.65 to a 0.59 clear rate. Untrained levels are unchanged: 0.89 → 0.89 in trained games and 0.86 → 0.87 in untrained games.

**More games stuck at level 1?** Not by count. Each RFT run has 4 games with at most one level cleared; baselines have 1–4, and no run has a game with zero levels. The difference is which games: sp80 (both runs) and ls20 (v2) normally reach 3–5 levels.

## Behaviour

Totals over all 25 games, from the `analysis` events in `artifacts/*_events.jsonl`:

| Run | Turns | Actions | Actions / turn | Thinking chars / turn | Levels / 100 actions | RESET calls |
|---|---:|---:|---:|---:|---:|---:|
| RDC (tic) | 2,054 | 6,697 | 3.26 | 9.3K | 1.81 | 50 |
| RDC | 2,038 | 6,489 | 3.18 | 8.6K | 1.94 | 41 |
| reset-baseline | 2,081 | 6,743 | 3.24 | 9.0K | 1.84 | 100 |
| animation-min-frames | 2,161 | 7,236 | 3.35 | 8.6K | 1.74 | 64 |
| RFT v1 | 2,262 | 7,569 | 3.35 | 8.2K | 1.55 | 70 |
| RFT v2 | 2,374 | 8,285 | 3.49 | 7.8K | 1.32 | 109 |

The RFT model thinks about 10% less per turn, takes 13–24% more actions and clears fewer levels per action. Score charges for actions, so this costs twice: fewer levels, and each one less efficient.

## Why it probably happened

1. **The selection rule taught a style, not a skill.** Per level, we kept clears in the best 35% by actions *and* with generated tokens at or below the level's median, on levels cleared in fewer than 80% of attempts. On hard levels, those are often the clears where a short guess happened to work. Training on them shifts the model toward deciding sooner with less thinking: −10% thinking per turn, more actions. It does not carry over whatever made those particular guesses right, and the damage is largest where it saw the most such examples.
2. **Little to learn from the data.** The replies are the model's own, so the loss starts low. Train loss went 0.285 → 0.277 over two epochs; val loss went 0.310 → 0.288 after epoch 1, then 0.287 after epoch 2. Most trained tokens are thinking narration; the action choices that decide a level are a small fraction. The update changed how the model reasons more than what it knows, and untrained games show no gain (−1.1). The val drop does not predict play: val is the next-best clears of 12 *trained* levels, scored with the real context fed token by token. Imitating short clears more closely lowers that loss, and that same imitation plays worse. It was also measured on the unmerged adapter, not the served checkpoint.
3. **Less varied outputs.** Draft acceptance rose by about 5 points with the draft model unchanged, so the RFT model's tokens became more predictable. That fits the long stalls on hard levels, where the model keeps refining one hypothesis. This is supporting evidence, not a measurement of diversity.
4. **The merge is unchecked (secondary).** Merging into BF16 kept the update's size (median 103% of the intended update, minimum 97.7%) but rounded its direction: cosine to the intended update is 0.95 median and 0.80 minimum (o_proj). The merged model's val loss was never measured on a GPU (`pod/verify_merged_pod.sh` exists for this). This is unlikely to be the main cause, since the damage follows the training data, but it has not been ruled out.

We also trained and scored on the same 25 games, while the leaderboard uses hidden ones. The untrained-game number (−1.1) is the closest proxy for the leaderboard, and it shows no gain.

## Next

- Hold out games: train on some games and score the others.
- Stop selecting by fewest actions and short thinking. Use all clears weighted by how much better they did than the level's average (close to GRPO), or add failed attempts as negatives.
- Before the next Kaggle run, serve the adapter directly or verify the merged model's val loss on a GPU.

Not competition-submitted.
