# Transition-reminder validation

The feature adds independent, default-off `context.retain_last_level_up` and `context.retain_last_game_over` options. The agreed [design](trimming.md) specifies pre-action reasoning retained verbatim, capped at 2,048 tokens by keeping its first and last 1,024 tokens, and historical board images before level-up and on both sides of a fatal action.

Focused tests cover the reasoning boundaries, tokenizer failure, independent flags, event ordering and replacement, final win, session reset, fatal versus automatic RESET images, reminder budgeting, duplicate prevention, image-retention interaction, unchanged scheduler state on an identical rebuild, overflow recovery, complete tool exchanges, and capture from the action-producing assistant response.

Serving-handoff tests verify that both typed config fields reach the harness under vLLM, Pennyroyal, and Lord serving. Existing image, thinking-budget, and soft-stop tests are included in the regression run. Ruff, Python syntax, and whitespace checks are also run.

Validation passed: **187 tests passed, 40 skipped** by the existing branch-divergent test list. Ruff, Python syntax, and `git diff --check` passed. A further 22 focused tests passed in the `context-vs-levelup` worktree after copying the verified files there.

## Kaggle test

Config: [`configs/context-transition-reminders/v1.json`](../../ARC3-Inference/configs/context-transition-reminders/v1.json).

The test derives from `configs/priority-one-wave-lorddraft/v1.json`. It enables both reminders, lowers the harness context from 131,072 to 65,536 tokens to exercise trimming sooner, and enables request logs. The model, serving settings, one-wave priority scheduler, 14 active streams, 25 public games, one pass, and 24-minute public play budget remain the same. The notebook budget is 34 minutes including the existing 10-minute buffer.

Destination: `jonathanwang2022/context-transition-reminders`; private source dataset: `jonathanwang2022/context-transition-reminders-src`.

This is a functional test. The smaller context and extra logging mean its score is not a controlled A/B measure of the reminder feature. No performance improvement is claimed.

Kaggle **version 1** was published on 2026-10-10 from commit `0dc3b80f4e6ee2242137ee7e341e64e4fd7fa79e` and confirmed `RUNNING`. Packaged feature files match that commit; setup exports both reminder flags as `true` and the context as `65536`. [Notebook](https://www.kaggle.com/code/jonathanwang2022/context-transition-reminders). No evaluated result is available at publication.

A persistent completion monitor checks exact version `v1` every minute and retrieves output, logs, source provenance, and evaluation when it finishes. Its status is saved with the Drive-backed run in `runs/kaggle/jonathanwang2022/context-transition-reminders/v01/followup_status.json`; terminal failures also trigger artifact retrieval.
