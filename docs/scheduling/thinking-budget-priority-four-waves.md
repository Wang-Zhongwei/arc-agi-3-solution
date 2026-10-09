# Thinking budget with the priority allocator in four waves

Notebook: [`jonathanwang2022/thinking-budget-priority-four-waves`](https://www.kaggle.com/code/jonathanwang2022/thinking-budget-priority-four-waves)
Config: `ARC3-Inference/configs/thinking-budget-priority-four-waves/v1.json`

## Question

[`ticbudja/thinking-budget-and-user-prompt-trim`](https://www.kaggle.com/code/ticbudja/thinking-budget-and-user-prompt-trim) v3 (thinking budget 8192, user-prompt trim) ran the Yield-UCB scheduler with every game in one wave. This notebook runs the same source with the original allocator and the 1001 run shape, so the two isolate the scheduling change.

## What differs from v3

| | v3 | this notebook |
|---|---|---|
| Source | `e7e80c3` | `e7e80c3` + soft stop (this branch) |
| `ARC3_SCHEDULER` | `ucb` (actions, c = 0.0002, recent weight 0.5) | `priority` (UCB settings removed) |
| Lanes (`concurrent_jobs`) | 120 | 28 |
| Per-game cap | 532 min | 132 min |
| Waves on the ~110 hidden games | 1 | 4 |
| `ARC3_GAME_SOFT_STOP_SECONDS` | unset | 300 |
| Public run length (`MAX_EXPERIMENT_RUNTIME_MINUTES`) | 150 | 150 |

Everything else in `harness_env`, `analyzer` and `kaggle` is copied from `configs/thinking-budget-and-user-prompt-trim/v3.json`.

The Makefile at this commit pins the one-wave shape on `kaggle-duck`, so the four-wave shape is set on the command line:

```
make kaggle-duck CONFIG_PATH=configs/thinking-budget-priority-four-waves/v1.json \
  KAGGLE_USERNAME=jonathanwang2022 CONCURRENT_JOBS=28 MAX_RUNTIME_MINUTES=132
```

## Soft stop at the wave boundary

In four waves each lane plays about four games back to back, each capped at 132 min. A game used to be cut wherever it was at the cap, usually between thinking continuations of a turn that had not acted yet. With `ARC3_GAME_SOFT_STOP_SECONDS=300`, from minute 127 a game starts no new turn, but the turn in flight finishes (continuations, actions, and no gate handover). Only minute 132 cuts it. See CONFIGURATION.md, "Soft stop at the per-game cap".

The cap is not extended, because the hidden run has no slack: 4 × 132 min plus about 9 min of server startup is 537 of the notebook's 540 minutes.

## Caveats

- The 25 public games fit in 28 lanes, so the public run is one wave under either shape. Waves only differ on the hidden set. On the public run, the differences are the allocator, the 132-min cap and the soft stop.
- The gate keeps one shared clock: the earliest game start and the earliest deadline. With `ARC3_PRIORITY_TAIL_FADE=1`, games in waves 2 to 4 see that deadline as passed, so their tail term B is faded to zero. This was also true of earlier four-wave runs and is left unchanged here.

## Results

Version 1 was pushed on 2026-10-04 from commit `458e575` (clean tree), with 28 lanes and a 132-min per-game cap. It was QUEUED at the time of writing. Record its scores here after `make pull-kaggle NOTEBOOK=jonathanwang2022/thinking-budget-priority-four-waves VERSION=1`.

## Version 3: allocation finding, one wave

Config: `ARC3-Inference/configs/thinking-budget-priority-four-waves/v3.json`. The source is the same as v2 (`eea6f57`). The only changes are the allocation settings from [the priority replay](../allocation/priority-replay.md#decision): rank by $R = v\,C$ (no tail B, efficiency ablated), with 14 active streams and one wave.

| | v2 | v3 |
|---|---|---|
| `ARC3_PRIORITY_TAIL_LOOKUP` | `remaining` (B = 8/7/5/0) | unset |
| `ARC3_PRIORITY_TAIL_CAP` | unset (4, unused under `remaining`) | 0, so B = 0 |
| `ARC3_PRIORITY_HUMAN_ACTIONS` | unset (25) | `1e9`, so e ≈ 1 |
| `ARC3_PRIORITY_SCORE_NORMALIZATION` | 1 | 1 |
| `ARC3_MAX_ACTIVE_STREAMS` | 10 | 14 |
| Lanes × per-game cap | 28 × 132 min (4 waves) | 120 × 532 min (1 wave, the Makefile default at this commit) |
| Public run length | 30 min | 30 min |

With these settings, `priority_value` at this source returns $\lfloor 55\,v\,C \rfloor$ (checked on five snapshots). The tail fade and the soft stop are left as in v2; with B = 0 the fade has no effect.

```
make kaggle-duck CONFIG_PATH=configs/thinking-budget-priority-four-waves/v3.json \
  KAGGLE_USERNAME=jonathanwang2022 MAX_EXPERIMENT_RUNTIME_MINUTES=30
```

Version 3 was pushed on 2026-10-08 from `808f025`, a clean tree on branch `allocation-vc-one-wave` (`eea6f57` plus this config and section). It reported `Passes: 1; concurrency: 120; waves: 1`, with a 532-min per-game cap. The generated notebook is cell-for-cell identical to v2's. `setup_commands.json` differs from v2 only in the four env settings above. Every source file that v2 bundled is byte-identical in v3's bundle.

Two packaging quirks of the worktree push:

- `kaggle kernels push` printed `Expecting value: line 1 column 1` and exited 1, but Kaggle did create version 3, and it started running.
- The bundle's `git_status.txt` names the main checkout (`416cba3`, dirty), and `src/` also contains 21 files from that checkout that `eea6f57` lacks: newer configs, `inference/framework/lordhansolo/`, `pennyroyal_{patch,wide_gemm,bf16_embedding}.py`, `benchmark_pennyroyal_optional.py`, runpod tools, and `deploy_runpod.py`/`runpod_pod.py`. Nothing at `eea6f57` or in the generated notebook references them, so they are not loaded.

Record scores here after `make pull-kaggle NOTEBOOK=jonathanwang2022/thinking-budget-priority-four-waves VERSION=3`. The 30-min public run is a health check, not a score comparison.

## Version 4: v3 with 10 active streams

Config: `ARC3-Inference/configs/thinking-budget-priority-four-waves/v4.json`. Same source and settings as v3, except `ARC3_MAX_ACTIVE_STREAMS` is 10 (as in v1 and v2) instead of 14. v3's 14 came from the replay doc's settings line, not from the intended plan.

The public run is set for about 10 min of play: `MAX_EXPERIMENT_RUNTIME_MINUTES=31`. At this source the soft end is the budget minus min(10 min, budget / 2), counted from notebook start, and v2's server was ready after 625 s (about 11 min in). So 31 ends at minute 21, about 10 min of play. A literal 10 would end at minute 5, before the server is up, and play nothing.

```
make kaggle-duck CONFIG_PATH=configs/thinking-budget-priority-four-waves/v4.json \
  KAGGLE_USERNAME=jonathanwang2022 MAX_EXPERIMENT_RUNTIME_MINUTES=31
```

The v4 config first ran on 2026-10-08 as `jonathanwang2022/thinking-budget-priority-four-waves` version 1 (a new notebook after the original was renamed), with `MAX_EXPERIMENT_RUNTIME_MINUTES=31`. On 2026-10-09 jonathanwang2022 had used its 45-hour weekly GPU quota, so the 120-min-play run was pushed as [`ticbudja/thinking-budget-priority-one-wave`](https://www.kaggle.com/code/ticbudja/thinking-budget-priority-one-wave) version 1 from `92dece7`, with `kaggle-as tic` and `MAX_EXPERIMENT_RUNTIME_MINUTES=141` (soft end at minute 131, about 120 min of play). Fetch it with `make pull-kaggle NOTEBOOK=ticbudja/thinking-budget-priority-one-wave VERSION=1`.

## Version 5: v4 without the 2,048-token turn yield

Config: `ARC3-Inference/configs/thinking-budget-priority-four-waves/v5.json`. Same source and settings as v4, except `LOCAL_ANALYZER_YIELD_TOKENS` is 0 instead of 2048, which disables the per-turn yield (see [the 1009 error analysis](../tools/1009-error-analysis.md)). The per-request limits stay at `max_tokens: 12288` and a thinking budget of 8,192. It was pushed on 2026-10-09 from `cb02f7f` as `ticbudja/thinking-budget-priority-one-wave` version 2 (bundled `LOCAL_ANALYZER_YIELD_TOKENS: "0"`), with the same 120-min play window as that notebook's v1, to compare against it:

```
kaggle-as tic make kaggle-duck CONFIG_PATH=configs/thinking-budget-priority-four-waves/v5.json \
  KAGGLE_USERNAME=ticbudja MAX_EXPERIMENT_RUNTIME_MINUTES=141
```

On 2026-10-09 the ticbudja notebook was renamed `thinking-budget-priority-one-wave` (title "Thinking budget priority one wave"), because it runs one wave; its local runs moved to `ARC3-Inference/runs/kaggle/ticbudja/thinking-budget-priority-one-wave/`. `v5.json` now targets that slug. The source dataset keeps its name, `ticbudja/thinking-budget-priority-four-waves-src`.

## Merge of main (2026-10-09)

`main` (`67d3490`) was merged into this branch, with this branch's prompts and tools kept as they were.

- **Kept from this branch:** `prompts.py`, the turn prompts and captions in `tool_agent.py`, the Python sandbox's module list, and the `animation.py`, `segmentation.py` and `frame_diff.py` outputs the model reads. Main's prompt rewrites (`6303467` game-over and level-start text, `0371330`, `d62d95f` turn messages and captions, `0ee3507` file-I/O guidance), its segmentation `bbox` change with `pose_hash`/`shape_hash` dropped, and its wider sandbox module list are not taken.
- **Taken from main:** serving, Kaggle and config fixes (one place per setting, the credential handoff, private reruns ignore token budgets, the 6 precache + 8 prefetch model read), the evaluator and serving metrics, RunPod and Lord Hansolo serving, and the opt-in options `ARC3_FRAME_DIFF` (default on), `ARC3_PRIORITY_THINKING` (default off), `ARC3_ANIMATION_IMAGE`/`ARC3_ANIMATION_DEATH_IMAGE=sampled` and `multimodal.intermediate_frames` (default off). The frame options are ported at `e26d05d`, before main's later layout and caption changes.
- **Configs:** main rejects a typed setting repeated in `harness_env`. In v1, v3, v4 and v5, the `LOCAL_ANALYZER_*` and `MULTIMODAL_*` entries moved to their `analyzer`, `shared` and `multimodal` fields, with the values that reached the harness on Kaggle (temperature 0.7, `max_output` 12288, `tool_output_tokens` 3072, `upscale` 10).

Checks: with v5's serving env, the system prompt is byte-identical before and after the merge. A dry-run v5 bundle matches the pushed v5 bundle (`ticbudja/thinking-budget-priority-one-wave` v2) in serving spec, harness env (plus `MULTIMODAL_INTERMEDIATE_FRAMES=false`) and run config (1 pass, 120 lanes, one wave, 532 min). The branch's own tests and the ported options' tests pass. Main's tests that assert its prompt wording (`test_animation_description.py`), segmentation fields (`test_segmentation.py`) and final frame-image layout (`test_intermediate_images.py`) fail here by design.

## One-wave v3: animation frames in context (prepared, not pushed)

Config: `ARC3-Inference/configs/thinking-budget-priority-one-wave/v3.json`, for the next version (Kaggle v3) of the renamed notebook `ticbudja/thinking-budget-priority-one-wave`. It is v5 with sampled animation frames turned on: `ARC3_ANIMATION_IMAGE` and `ARC3_ANIMATION_DEATH_IMAGE` set to `sampled`, `multimodal.animation_upscale: 2`, and `multimodal.intermediate_frames: false`. The last action's animation, and a fatal action's, send up to eight frames at 128×128. The current board stays at upscale 10. The run shape now lives in the config: `kaggle.public_runtime_minutes: 131` (about 120 min of public play), `kaggle.private_runtime_minutes: 532`, and `kaggle.concurrent_jobs: 120`.

A `KAGGLE_DRY_RUN=true` build matches the pushed one-wave v2 bundle in serving spec and harness env, except for the four added keys (`ARC3_ANIMATION_IMAGE`, `ARC3_ANIMATION_DEATH_IMAGE`, `MULTIMODAL_ANIMATION_UPSCALE=2`, `MULTIMODAL_INTERMEDIATE_FRAMES=false`). The run shape is 1 pass, 120 lanes, one wave, 532 min per game, and a 141-min public notebook budget. [frame-attachments-one-wave.html](../multimodal/frame-attachments-one-wave.html) shows the resulting per-turn messages. It was rendered with `docs/multimodal/scripts/render_frame_attachments.py` from that dry-run bundle.

Rename the Kaggle notebook to "Thinking budget priority one wave" before pushing, or the push creates a new notebook at version 1. To push:

```
kaggle-as tic make kaggle-duck CONFIG_PATH=configs/thinking-budget-priority-one-wave/v3.json KAGGLE_USERNAME=ticbudja
```

The `python` tool description, sent with every request, now ends with two lines: "No I/O is allowed: do not read or write files." and "Only use the exposed packages; any other import fails." The sandbox already rejects `open()` and imports outside its module list, so these lines only tell the model what the sandbox enforces. They apply to one-wave v3 onwards.
