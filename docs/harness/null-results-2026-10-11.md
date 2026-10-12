# Null results, 2026-10-11

Four context and history experiments did not beat the baseline. Their branches are deleted. The code and configs are kept under `archive/<branch>` tags.

All runs: lorddraft vLLM serving, 25 public games, one pass, 120-minute soft end. The p values are paired t-tests over the 25 games.

| Run | Change vs no-io baseline | Streams | Score | vs no-io | vs level-up-board | Gen tok/s (mean) | KV peak |
|---|---|---:|---:|---:|---:|---:|---:|
| `lorddraft-baseline-no-io` v1 | baseline (58-118K history) | 15/14 | 50.17 | | | 830 | 92.8% |
| `level-up-board` v1 | `ARC3_LEVELUP_BOARD_IMAGE=1` (on main, PR #29) | 15/14 | 55.94 | +5.77, p=0.37 | | 838 | 93.0% |
| `view-tool` v2 | `ARC3_VIEW_TOOL=1` | 15/14 | 51.25 | +1.08, p=0.79 | | 806 | 91.7% |
| `context-100-130k` v1 | level-up board + 100-130K history | 10/10 | 47.78 | | -8.16, p=0.09 | 625 | 80.8% |
| `context-48-96k` v1 | view tool + level-up board + 48-96K history | 16/16 | 51.84 | +1.67, p=0.74 | -4.09, p=0.49 | 901 | 91.4% |

Notebooks: `jonathanwang2022/<run>` on Kaggle. Streams: game slots / engine sequences.

## View tool

`view(indices, bbox=None)` shows `history` boards as images, optionally cropped. After a trim, a note names the `history` range that left the conversation. **+1.08 over no-io (p=0.79): no effect.** It cost about 3% generation throughput.

v1 was cancelled before the priority gate rotated in the last 10 games. The v2 rerun is the one scored here. The stall audit behind the idea is in `docs/harness/stall-recovery-design.md` on the tag. It also proposes high-thinking-per-action reminders, which were never implemented.

Tag: `archive/harness/stall-recovery-design` (`ARC3_VIEW_TOOL`, `configs/view-tool/`, `tests/test_view_tool.py`).

## Context 100-130K at 10 streams

History is trimmed at 130K and drained back to 100K, instead of 118K to 58K. There are 10 streams instead of 15/14. **47.78, 8.16 below level-up-board (p=0.09).**

Fewer streams cut throughput by 25% (625 vs 838 tok/s). Extra history did not make up for less play. KV peaked at only 80.8%.

Tag: `archive/context-100-130k`.

## Context 48-96K at 16 streams

History is trimmed at 96K and drained to 48K. Streams are 16/16. The view tool and the level-up board are on. **51.84: +1.67 over no-io (p=0.74) and 4.09 below level-up-board alone (p=0.49).**

Throughput rose 8% (901 vs 838 tok/s), and KV peaked at 91.4%, as estimated. The extra throughput did not turn into score. Shorter history, possibly together with the view tool, gave back the level-up board's gain.

v1 ran from `9bb7186`. Its score comes from the evaluator run on the notebook's `benchmark.json`/`games.pkl`; the full output is not in `runs/`.

Tag: `archive/context-48-96k`.

## Level 1-3 solve rate vs context

Offline analysis of 30 Daniel-harness runs (320 game runs), merged to main as [solve-rate-vs-context.md](solve-rate-vs-context.md) (PR #32, branch `low-level-tokens`).

- **Level 1** dips after the first trim (49% → 30% solved per window, p=0.02 pooled). Within the same game the effect is not significant (OR 0.60, CI 0.28-1.27). Hard games that are still on level 1 explain much of the dip.
- **Current context 50-118K**: no optimal range for levels 1-3 (rate ratios 0.96-0.98 per +10K, all CIs include 1).
- **Keeping the last level-up in context**: no significant effect (levels 2-3 1.11x, CI 0.80-1.55).

The two A/B runs above agree: neither longer history (100-130K) nor shorter history plus more streams (48-96K) moved the score.

## Takeaways

- Keep the 58-118K band at 15/14 streams. The level-up board is the only change from this round worth keeping.
- More throughput from more streams did not raise the score at 16 streams. Less throughput from fewer streams lowered it at 10.
- With n=25 games and one pass, a difference under about 8 points is not detectable. These are null results, not proof of no effect.

## Other archived branches

Tags of the form `archive/<branch>` also hold these branches, deleted on the same day:

- `inference-tuneup`: Pennyroyal/SGLang tuning, NVFP4 Marlin debugging, concurrency-14-streams v2-v4 (also in `docs/serving/concurrency-12-streams.md` on `simple-baseline`).
- `priority-one-wave-lord-vllm`: the 10-08 harness on Lord vLLM; its record is [../submission/1009.md](../submission/1009.md).
- Older experiment branches whose PRs merged or whose arms finished (`yield-tokens`, `simple-event-prompts`, `harness/early-level-recovery`, `adaptive-context`, `vision-memory`, ...).

`simple-baseline` is the former `main` (`60811ed`): Daniel's harness with lorddraft serving. `main` is now the former `allocation-vc-one-wave`.

List the tags with `git tag -l 'archive/*'`. Restore a branch with `git switch -c <name> archive/<name>`.
