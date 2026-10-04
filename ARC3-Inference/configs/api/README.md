# API comparison configs

Four variants of the Kaggle notebook `api-signatures-and-frame-diff`. Queue waits moved v2/v3 to the `ticbudja` account, so Kaggle version numbers do not follow config names; see Runs.

| Config | Code | `ARC3_AUTO_FRAME_DIFF` | `ARC3_API_SIGNATURES` | `ARC3_HUD_DISPLAY` |
| --- | --- | --- | --- | --- |
| `v0.json` | `main` | Off | — | — |
| `v1.json` | `main` | On | — | — |
| `v2.json` | `game_over_feedbacks_and_hash_board_transform_summary` | Off | Off | Off |
| `v3.json` | `game_over_feedbacks_and_hash_board_transform_summary` | Off | On | On |

## Runs

| Config | Notebook | Version | Status | Score (10 passes) |
| --- | --- | --- | --- | --- |
| `v0.json` | `jonathanwang2022/api-signatures-and-frame-diff` | 3 | Complete | 30.06 ± 12.67; 3.9/7 levels |
| `v1.json` | `jonathanwang2022/api-signatures-and-frame-diff` | — | Skipped | — |
| `v2.json` | `ticbudja/api-signatures-and-frame-diff` | 1 | Queued | — |
| `v3.json` | `ticbudja/api-signatures-and-frame-diff` | 2 | Not pushed | — |

`jonathanwang2022` versions 1–2 (v0, v1) played no games because of the `'__auto__'` notebook bug, version 4 (v1) was cancelled while queued, and version 5 (v1) was abandoned in the queue. Pulled runs live in `runs/kaggle/<owner>/api-signatures-and-frame-diff/vNN/` with `evaluation.json` from `make score_run`.

Base: `configs/scheduling/v2.json` (the 1003 one-wave Yield-UCB submission): Flash Next on Pennyroyal SGLang and its full `harness_env`, including game-scoped persistent functions, which `hud()` needs. Overrides: `kaggle.public_games = ["ls20-9607627b"]`, 10 passes, 230,000 generated tokens per game/pass, 240-minute per-game and notebook caps, so the token budget is the binding limit.

Pennyroyal notebooks receive only `harness_env`, so the variant flags live there; the matching `analyzer.*` keys only keep local Make exports consistent.

JSON does not select source code: launch v0/v1 from `main` and v2/v3 from the branch, each with

```bash
make kaggle-duck CONFIG_PATH=configs/api/vN.json N_PASSES=10 KAGGLE_USERNAME=<you>
```

`kaggle-duck` forces one pass unless `N_PASSES` is given; the per-game cap, concurrency and played games come from the `kaggle` section. Offline runs need TAAF's standard notebook to rebuild games from the competition's `environment_files` (fixed alongside these configs); without it every game fails with the `'__auto__'` env-files error and the run reports `games: 0`.
