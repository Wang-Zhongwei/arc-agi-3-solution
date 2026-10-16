# Last animation (2026-10-14/15)

Two harness changes were tested together against **reset-daniel-context v1** (`jonathanwang2022/reset-daniel-context` v1: 59.07 mean / 58.33 median, 126 levels, 6,489 actions, 816 tok/s):

- Report the call's **last animated action**, not only an animated final action.
- After a deliberate RESET, **diff from the board the RESET returned**, not across it.

The run did not beat the control. Main was reverted to the control's code (b6922a0), and the work stays on branch `last-animation` (b1b1744).

Related: [animation guards](animation-guards.md), [world-model notes](world-model-notes.md).

| Run | Kaggle notebook | Code | Mean | Median | Δ vs RDC (paired) | t-test p | Wilcoxon p | Worse / better games | Levels | Actions | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| reset-daniel-context (RDC) | `jonathanwang2022/reset-daniel-context` v1 | d2d6cf4 | **59.07** | 58.33 | — | — | — | — | 126 | 6,489 | 816 |
| last-animation (LA) | `ticbudja/last-animation` v1 | `last-animation` b1b1744 | 53.94 | 58.33 | −5.14 | 0.30 | 0.24 | 11 / 8 | 116 | 6,638 | 804 |

`configs/last-animation/v1.json` is the reset-daniel-context v1 config with only the slug, title and dataset renamed. Both runs used:

- 25 games, one pass, a 120-minute public budget
- 15 active streams / 14 sequences
- 131K context window and a 59,392-token drain

The code is d2d6cf4 plus the two commits below. LA ran on the `ticbudja` account because `jonathanwang2022` had hit the weekly GPU quota. Generated tokens were 5.37M vs 5.34M. "Worse / better" counts games whose score moved by more than 0.5 points.

---

## What changed

### 45b2789: report the call's last animated action, not only an animated final action

**Before:** the animation record (`last_action_call_result['animation']`, `last_animation_frames`, `last_animation_timeline`, and the opener line) was replaced by *every* executed action. If a call ended with inert actions after an animated one, the animation was lost. For example, in cd82 the call MOUSE, SPACE, LEFT, LEFT left no opener line and an empty `last_animation_timeline`.

**After:**

- The record is cleared once per call and replaced only by actions that animate.
- `last_action_call_result['animation']` now also carries `action` and `position` (1-based within the call).
- When the animated action was not the final one, the opener names it: "Action 9 of 11 (`DOWN`) animated over 21 frames. 40 cells changed and changed back around rows 35-55, cols 38-58. `last_animation_timeline` shows them." Otherwise the opener keeps "The last executed action…".

The system-prompt bullet changed accordingly:

```diff
- When the final executed action of the latest model-issued call animated, `last_action_call_result['animation']`
- reports `frames`, `transient_pixels` and `transient_bbox`.
+ When an action of the latest model-issued call animated, `last_action_call_result['animation']`
+ reports the last such action's `action`, `position` (1-based in the call), `frames`, `transient_pixels`
+ and `transient_bbox`; it need not be the final action.
```

Files changed: `prompts.py`, `tool_agent.py`, `solver.py`, `utils/animation.py`. Tests: `tests/test_last_animated_action.py` (109 lines).

### 59879eb: diff from the board a deliberate RESET returned, not across it

**Before:** the diff image and the auto frame diff ran from the previous turn's board. After a RESET, the diff therefore mixed what the RESET undid with what came after it. In RDC, 25 of 71 deliberate-RESET turns were a lone RESET. tn36 in RDC looped on RESETs.

**After:**

- `_last_manual_reset_frame()` finds this turn's last model-issued RESET; automatic resets after a death are excluded.
- The diff starts from the board that RESET returned, and its caption reads "cells that changed since your **RESET**".
- A turn that ends on RESET gets no diff image.
- There is no system-prompt change.

Tests: `tests/test_reset_diff_baseline.py`.

Main later took a simpler variant, b5ffe54 ("Attach no diff after a turn in which the model issued RESET"): no diff at all after any turn containing a model-issued RESET. That variant is part of reset-baseline v1.

---

## How much the change actually fired

These counts come from the run transcripts:

| | RDC | LA |
|---|---:|---:|
| Openers naming a **non-final** animated action ("Action N of M (`X`) animated…") | 0 | **10**, in 4 games: re86 5, g50t 2, ls20 2, ka59 1 |
| All animation opener lines ("animated over N frames") | 586 | 548 |
| Turns whose executed actions include a deliberate RESET | 66 | 56 |
| Diff images captioned "since your RESET" | 0 | **34**, in 13 games |

- **The last-animated-action change surfaced an animation only 10 times** in a 25-game run. In every other turn the opener is identical to RDC's.
- **The RESET-diff change touched 34 turns.** tn36's RESET loop disappeared (27 RESET turns → 2), but its score stayed near zero (2.29 → 3.57).

## Scores by game, with footprint

| Game | RDC | LA | Δ | Non-final openers | RESET turns RDC → LA | "since your RESET" diffs | Levels RDC → LA |
|---|---:|---:|---:|---:|---|---:|---|
| ar25 | 100.00 | 100.00 | 0.00 | 0 | 0 → 0 | 0 | 8 → 8 |
| bp35 | 6.67 | 1.17 | −5.50 | 0 | 0 → 0 | 0 | 2 → 1 |
| cd82 | 84.97 | 100.00 | +15.03 | 0 | 1 → 3 | 1 | 6 → 6 |
| cn04 | 71.43 | 71.43 | 0.00 | 0 | 2 → 0 | 0 | 5 → 5 |
| dc22 | 47.62 | 37.06 | −10.56 | 0 | 2 → 5 | 3 | 4 → 4 |
| ft09 | 100.00 | 100.00 | 0.00 | 0 | 0 → 0 | 0 | 6 → 6 |
| **g50t** | 53.57 | 10.71 | **−42.86** | **2** | 3 → 5 | **10** | 5 → 2 |
| **ka59** | 23.90 | 68.76 | **+44.86** | **1** | 3 → 2 | 0 | 4 → 6 |
| lf52 | 7.96 | 1.82 | −6.14 | 0 | 5 → 1 | 1 | 3 → 1 |
| lp85 | 100.00 | 100.00 | 0.00 | 0 | 0 → 0 | 0 | 8 → 8 |
| **ls20** | 45.70 | 30.34 | −15.36 | **2** | 5 → 9 | 1 | 5 → 4 |
| m0r0 | 100.00 | 71.43 | −28.57 | 0 | 0 → 0 | 0 | 6 → 5 |
| r11l | 71.43 | 100.00 | +28.57 | 0 | 0 → 0 | 0 | 5 → 6 |
| **re86** | 58.33 | 57.97 | −0.36 | **5** | 0 → 2 | 1 | 6 → 6 |
| s5i5 | 16.67 | 58.33 | +41.66 | 0 | 0 → 0 | 0 | 3 → 6 |
| sb26 | 96.94 | 100.00 | +3.06 | 0 | 1 → 0 | 0 | 8 → 8 |
| **sc25** | 100.00 | 36.97 | **−63.03** | 0 | 0 → 8 | **7** | 6 → 4 |
| sk48 | 0.00 | 0.00 | 0.00 | 0 | 1 → 0 | 0 | 0 → 0 |
| sp80 | 28.57 | 4.76 | −23.81 | 0 | 1 → 2 | 2 | 3 → 1 |
| su15 | 54.32 | 22.22 | −32.10 | 0 | 11 → 10 | 1 | 7 → 4 |
| tn36 | 2.29 | 3.57 | +1.28 | 0 | 27 → 2 | 1 | 1 → 1 |
| tr87 | 100.00 | 89.39 | −10.61 | 0 | 0 → 0 | 2 | 6 → 6 |
| tu93 | 96.87 | 100.00 | +3.13 | 0 | 0 → 1 | 1 | 9 → 9 |
| vc33 | 100.00 | 69.14 | −30.86 | 0 | 0 → 4 | 3 | 7 → 6 |
| wa30 | 9.54 | 13.33 | +3.79 | 0 | 4 → 2 | 0 | 3 → 3 |

**Games with non-final openers** (re86, g50t, ls20, ka59) averaged −3.4: g50t −43 and ka59 +45 cancel. With 1–5 events per game, a 40-point swing cannot be pinned on the change.

**Games with RESET-diff turns:**

- **sc25 (−63) and g50t (−43) are the two worst games, and also the two with the most "since your RESET" diffs (7 and 10).** But they also issued more RESETs than in RDC (sc25 0 → 8). Losing a level and resetting more is as likely a cause of those diffs as a consequence of them.
- **tn36 is the case the change was built for.** Its RESET loop stopped (27 → 2 RESET turns), but the score did not move.

**Large swings with no footprint at all:** s5i5 +42, r11l +29, m0r0 −29. These mark the noise floor.

---

## Lessons

1. **This run was the first sign that single runs cannot resolve small effects.**
   - Paired per-game differences have SD ≈ 24 points, so the SE of the mean difference is ≈ 4.8.
   - At 80% power, the smallest effect such a run can reliably detect is about **±13 points**.
   - The −5.1 result, and the comparison with batch-animations (PR #34, 44.62; LA is +9.3 above it, p = 0.16), are both inside the noise.
   - The games that collapsed in batch-animations recovered here (cd82, ft09 and tu93 to 100; vc33 to 69), while sc25 and g50t dropped instead: swings of 30–60 points between near-identical configs.
2. **Measure the footprint before reading the score.** The last-animated-action change fired 10 times in 6,638 actions. A change this rare cannot move a 25-game mean; its value has to be judged from transcripts of the turns where it fired.
3. **Do not bundle a rare change with a frequent one.** The RESET-diff change (34 turns) and the last-animated-action change (10 turns) were tested together, so neither can be credited or blamed separately.
4. **Status.**
   - Main was reverted to d2d6cf4 content (b6922a0, "the fixes are all null"). The work lives on branch `last-animation` (b1b1744).
   - The RESET idea survives on main in the simpler form b5ffe54 (no diff after a RESET turn), which is part of reset-baseline v1.
   - The last-animated-action reporting is not on main.

## Reproduce

```bash
cd ARC3-Inference   # from the main checkout (runs/ is the Drive symlink)
make pull-kaggle NOTEBOOK=jonathanwang2022/reset-daniel-context VERSION=1
kaggle-as tic make pull-kaggle NOTEBOOK=ticbudja/last-animation VERSION=1
make score_run SCORE_RUN_DIR=runs/kaggle/ticbudja/last-animation/v01   # the eval target ignores the run folder
```

Data sources:

- **Per-game scores:** each run's `summary.txt`.
- **Footprints:** `grep` over `transcripts/*.txt`:
  - non-final openers: `^Action \d+ of \d+ \(\``
  - all opener lines: `animated over \d+ frames`
  - RESET turns: `^Executed actions: .*RESET`
  - RESET diffs: `cells that changed since your RESET`
