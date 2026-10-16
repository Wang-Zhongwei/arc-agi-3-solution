# Animation guards (2026-10-15)

An action that only animated (cells changed and changed back, at least 200 of them) no longer counts as a no-op for the stale-state, batch-stop and known-no-op guards. It was tested against **reset-baseline v1** (`ticbudja/reset-baseline` v1: 58.13 mean / 58.33 median, 124 levels, 6,743 actions, 867 tok/s). It did not beat the baseline and has not been merged; the code stays on branch `harness/animation-guards`.

Related: [world-model notes](world-model-notes.md), [last-animation](last-animation.md).

| Run | Kaggle notebook | Code | Mean | Median | Δ vs RB (paired) | t-test p | Wilcoxon p | Worse / better games | Levels | Actions |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| reset-baseline (RB) | `ticbudja/reset-baseline` v1 | main 8db9b61 | **58.13** | 58.33 | — | — | — | — | 124 | 6,743 |
| animation-guards (AG) | `ticbudja/animation-guards` v1 | `harness/animation-guards` 5a43bc4 | 51.60 | 58.33 | −6.53 | 0.27 | 0.11 | 11 / 5 | 113 | 6,694 |

Both runs used the same 25 games, one pass, a 120-minute public budget, and the same serving and scheduler setup:

- Lord vLLM with the NVFP4 MTP draft model
- 15 active streams / 14 sequences
- 131K context window and a 59,392-token drain

The treatment config `configs/animation-guards/v1.json` is a copy of `configs/reset-baseline/v1.json` with only the slug, title and dataset renamed. "Worse / better" counts games whose score moved by more than 0.5 points.

**Confound:** reset-baseline v1 ran with the RESET sentence in the system prompt ("no diff image after a RESET you issue"). Commit 693ad8e, which is on the animation-guards branch, removed it. The comparison therefore measures the guards *plus* the sentence removal. There is no 120-minute control run without the sentence; reset-baseline v2 was only a 5-minute smoke run.

---


### What changed

Before this branch, "no-op" meant `gameplay_changed is False`: the board interior (excluding the 4-cell border) was identical before and after the action. Animations did not count. Three guards use that verdict:

1. **Stale-state refusal** (`ARC3_STALE_STATE_BLOCK`, from level `ARC3_GUARDS_FROM_LEVEL`=2 on). After a no-op, the sandbox arms `_stale_state`. The *next* `action()` call in the same snippet then raises `StaleStateActionError: <ACTION> was not executed…`.
2. **Batch no-op stop** (`ARC3_BATCH_NOOP_BLOCK`, any level). The rest of a batch is skipped after a no-op.
3. **Known-no-op memory** (`no_op_per_action`).

The branch adds three commits to the RB code, plus the commit that drops the RESET sentence:

| Commit | Change |
|---|---|
| **519b508** *An action that only animated is not a no-op for the guards* | The solver's per-action `no_op` now also requires that **no interior cell changed and changed back** during the action's animation. All three guards use this verdict. Only a true no-op arms the stale flag, and an animated action clears it. The echo label changes from `-> NO-OP (executed, no change inside the board area)` to `-> effect (animation only; the board ended unchanged)`. A *refused* `action()` call no longer overwrites the last executed action's animation chain or change flag. **Why:** tn36's run button and dc22's 2,995-cell flash (turn 64) armed the stale flag, so deliberate repeats were refused. About 40 stale refusals in four earlier runs blamed an animated action, and agents wrote retry loops around them (tn36 `act()` with tries=3). |
| **b1c9463** *Count an animation as an effect only when it changes 200+ cells* | A less aggressive version of 519b508. An animation counts as an effect only if at least `ARC3_ANIMATION_EFFECT_CELLS` (default **200**) interior cells change and change back. The threshold was checked against 93 hand-labelled actions from reset-daniel-context v1 that animated but left the board unchanged. It keeps 42 of 45 real mechanics (tn36 run button, sp80 beam, dc22 wipe) and 0 of 45 "nothing happened" responses (su15 click ripples, lf52 highlight on an immovable piece, bp35 and ls20 wall bumps). It misses sc25's 36–81-cell pattern flashes. |
| **d0f7e5e** *Report a repeated guard or no-op stop once, with its count* | When a snippet keeps calling `action()` after a refusal, the identical stop notes are merged into one note in the turn summary, ending "This … N times in that turn." Example: tn36 in last-animation v1, turn 40, where the stale-state note had been printed three times. |
| 693ad8e *(confound)* | Drops the RESET sentence from the system prompt. |

The opener still calls animations under 200 cells "not a no-op", while the guards treat them as no-ops. The user reviewed this wording and accepted it.

### How much the change actually fired

These counts come from the run transcripts (`runs/kaggle/ticbudja/{reset-baseline,animation-guards}/v01/transcripts/*.txt`):

| | RB | AG |
|---|---:|---:|
| Actions labelled `effect (animation only…)` | 0 | **20**, in 3 games: dc22 8, sp80 8, tn36 4 |
| Actions labelled `NO-OP (executed…)` | 372 | 290 |
| `StaleStateActionError` refusals | 22 in 9 games | 17 in 8 games |
| Merged "N times in that turn" notes | 0 | 15 (m0r0 6, lf52 4, lp85 3, re86 2) |

The 200-cell rule relabelled only 20 of 6,694 actions. Stale refusals were about as frequent with the change as without it (17 vs 22).

### Scores on the games the change touched

| Game | RB score / levels / actions | AG score / levels / actions | Δ | Guard footprint (RB → AG) |
|---|---|---|---:|---|
| **tn36** (target game) | 74.81 / 6 / 355 | 20.53 / 3 / 233 | **−54.28** | anim-effect 0→4; NO-OP 67→4; stale refusals **0→0**; tokens 170K→293K |
| **sp80** (beam) | 47.62 / 4 / 212 | 28.57 / 3 / 160 | −19.05 | anim-effect 0→8; NO-OP 17→4; stale 0→1 |
| **dc22** (wipe) | 47.62 / 4 / 361 | 47.62 / 4 / 312 | 0.00 | anim-effect 0→8; NO-OP 43→33; stale 0→2 |
| ls20 | 41.14 / 5 / 529 | 19.99 / 4 / 602 | −21.15 | stale 2→8 (all from non-animated no-ops) |
| m0r0 | 100.00 / 6 / 316 | 67.54 / 5 / 319 | −32.46 | stale 0→1; 6 merged-repeat notes |

- **The three games where an animation counted as an effect averaged −24.4.** The other 22 games averaged −4.1.
- **But the guard fix never applied to tn36.** tn36 is the game the change was built for, yet it had **zero** stale refusals in RB, so the refusal path the change fixes never blocked it there. In AG it used 72% more tokens and reached only 3 levels.
- **The relabelling may have changed how the model treated the run button.** The "NO-OP" signal on the run button was gone, and it is plausible the model reasoned differently about it. With one run, this cannot be separated from run-to-run variance or from the dropped RESET sentence.

### Largest swings in games with no animation-guard events

| Game | RB | AG | Δ | Note |
|---|---:|---:|---:|---|
| su15 | 76.28 | 13.33 | **−62.95** | no anim-effect, no stale refusals in either run |
| cn04 | 47.62 | 0.00 | −47.62 | stale 4→0 |
| g50t | 43.79 | 10.71 | −33.08 | stale 4→0 |
| sc25 | 0.00 | 82.08 | **+82.08** | reverses RB's 100→0 (RB got stuck at level 0) |
| ka59 | 24.90 | 62.48 | +37.58 | — |

The biggest moves happened in games the change could not have affected. That shows how large the per-game noise is (see Lessons).

---

## Lessons

1. **A single-pass 25-game A/B can only detect large effects.**
   - The paired per-game differences here have SD ≈ 29 points, so the SE of the mean difference is ≈ 5.7.
   - At 80% power, the smallest effect such a run can reliably detect is about **±16 points**. The −6.5 result is inside the noise.
   - Single games swing by 50–80 points with no mechanism behind them: su15 −63 and sc25 +82 in AG, where the guard never fired.
   - **Rule:** do not attribute a single game's swing to a change unless that game's transcript shows the change firing.

2. **Measure the footprint before reading the score.** The animation-guard change relabelled 20 of 6,694 actions in 3 games, and the stale-refusal count barely moved (22 → 17). A change that touches under 1% of actions cannot be expected to move a 25-game mean measurably. Every report should open with a footprint table: labels, refusals, note lines.

3. **Change one variable at a time.**
   - This experiment also dropped the RESET sentence. world-model-notes v2, which also ran without it, scored below RB too (51.9 vs 58.1). This may be noise or a real effect of the sentence.
   - The result cannot be read until there is a 120-minute control run of main a0b99dc (no sentence, nothing else changed).

4. **Status.** The guards are off and not merged; they live on `harness/animation-guards` (5a43bc4).

## Reproduce

```bash
cd ARC3-Inference   # from the main checkout (runs/ is the Drive symlink)
kaggle-as tic make pull-kaggle NOTEBOOK=ticbudja/reset-baseline VERSION=1
kaggle-as tic make pull-kaggle NOTEBOOK=ticbudja/animation-guards VERSION=1
make score_run SCORE_RUN_DIR=runs/kaggle/ticbudja/animation-guards/v01   # the eval target ignores the run folder
```

Data sources:

- **Per-game scores:** each run's `summary.txt`.
- **Guard footprints:** `grep` over `transcripts/*.txt` for `-> effect (animation only`, `-> NO-OP (executed`, `^StaleStateActionError: \w+ was not executed` and `times in that turn`.
