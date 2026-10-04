# API comparison configs

One Kaggle notebook, `api-signatures-and-frame-diff`, run four times. Kaggle numbers versions from 1, so config `vN` is notebook version N+1.

| Config | Code | `ARC3_AUTO_FRAME_DIFF` | `ARC3_API_SIGNATURES` | `ARC3_HUD_DISPLAY` |
| --- | --- | --- | --- | --- |
| `v0.json` | `main` | Off | — | — |
| `v1.json` | `main` | On | — | — |
| `v2.json` | `game_over_feedbacks_and_hash_board_transform_summary` | Off | Off | Off |
| `v3.json` | `game_over_feedbacks_and_hash_board_transform_summary` | Off | On | On |

Base: `configs/scheduling/v2.json` (the 1003 one-wave Yield-UCB submission): Flash Next on Pennyroyal SGLang and its full `harness_env`, including game-scoped persistent functions, which `hud()` needs. Overrides: `ls20-9607627b`, 10 passes, 230,000 generated tokens per game/pass, 240-minute per-game and notebook caps, so the token budget is the binding limit.

Pennyroyal notebooks receive only `harness_env`, so the variant flags live there; the matching `analyzer.*` keys only keep local Make exports consistent.

JSON does not select source code: launch v0/v1 from `main` and v2/v3 from the branch, with `KAGGLE_DUCK_PUBLIC_HARNESS=false GAME=ls20-9607627b N_PASSES=10 MAX_RUNTIME_MINUTES=240 CONCURRENT_JOBS=10` (the `kaggle-duck` target otherwise overrides the JSON run shape).
