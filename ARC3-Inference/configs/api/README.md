# API comparison configs

One Kaggle notebook, `api-signatures-and-frame-diff`, run four times. Kaggle numbers versions from 1, so config `vN` is notebook version N+1.

| Config | Code | `ARC3_AUTO_FRAME_DIFF` | `ARC3_API_SIGNATURES` | `ARC3_HUD_DISPLAY` |
| --- | --- | --- | --- | --- |
| `v0.json` | `main` | Off | — | — |
| `v1.json` | `main` | On | — | — |
| `v2.json` | `game_over_feedbacks_and_hash_board_transform_summary` | Off | Off | Off |
| `v3.json` | `game_over_feedbacks_and_hash_board_transform_summary` | Off | On | On |

Base: `configs/scheduling/v2.json` (the 1003 one-wave Yield-UCB submission): Flash Next on Pennyroyal SGLang and its full `harness_env`, including game-scoped persistent functions, which `hud()` needs. Overrides: `kaggle.public_games = ["ls20-9607627b"]`, 10 passes, 230,000 generated tokens per game/pass, 240-minute per-game and notebook caps, so the token budget is the binding limit.

Pennyroyal notebooks receive only `harness_env`, so the variant flags live there; the matching `analyzer.*` keys only keep local Make exports consistent.

JSON does not select source code: launch v0/v1 from `main` and v2/v3 from the branch, each with

```bash
make kaggle-duck CONFIG_PATH=configs/api/vN.json N_PASSES=10 KAGGLE_USERNAME=<you>
```

`kaggle-duck` forces one pass unless `N_PASSES` is given; the per-game cap, concurrency and played games come from the `kaggle` section. Offline runs need TAAF's standard notebook to rebuild games from the competition's `environment_files` (fixed alongside these configs); without it every game fails with the `'__auto__'` env-files error and the run reports `games: 0`.
