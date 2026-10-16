# World-model notes (2026-10-15/16)

`ARC3_WORLD_MODEL_NOTES` gives the model a per-game `world_model.md`. It writes updates to the file, the harness tags every line with level and action count, and the model reads the file back from Python when it chooses. The notes are never injected into the prompt. The feature was tested against **reset-baseline v1** (`ticbudja/reset-baseline` v1: 58.13 mean / 58.33 median, 124 levels, 6,743 actions, 867 tok/s) and **concluded negative** on 2026-10-16. It stays on branch `harness/world-model-notes`, with the flag defaulting to off.

Related: [animation guards](animation-guards.md), [last-animation](last-animation.md).

| Run | Kaggle notebook | Code | Mean | Median | Δ vs RB (paired) | t-test p | Wilcoxon p | Worse / better games | Levels | Actions |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| reset-baseline (RB) | `ticbudja/reset-baseline` v1 | main 8db9b61 | **58.13** | 58.33 | — | — | — | — | 124 | 6,743 |
| world-model-notes v2 (WM v2) | `ticbudja/world-model-notes` v2 | `harness/world-model-notes` 4fd3b22 | 51.94 | 47.48 | −6.19 | 0.24 | 0.23 | 10 / 6 | 115 | 6,229 |

**world-model-notes v1 was thrown out because of a bug.** In `ticbudja/world-model-notes` v1 (f406cd0), `world_model.md` was read-only, so every write the model attempted failed with `PermissionError` (28 in 18 games) and the feature never worked as designed. Its score (56.04) is not used anywhere in this report. v2 is the fixed version.

Both runs used the same 25 games, one pass, a 120-minute public budget, and the same serving and scheduler setup:

- Lord vLLM with the NVFP4 MTP draft model
- 15 active streams / 14 sequences
- 131K context window and a 59,392-token drain

The treatment config `configs/world-model-notes/v2.json` is a copy of `configs/reset-baseline/v1.json` with only the slug, title and dataset renamed. It also sets `ARC3_WORLD_MODEL_NOTES=1`. "Worse / better" counts games whose score moved by more than 0.5 points.

**Confound:** reset-baseline v1 ran with the RESET sentence in the system prompt ("no diff image after a RESET you issue"). The world-model branch starts from a0b99dc, which no longer has it. The comparison therefore measures the notes *plus* the sentence removal. There is no 120-minute control run without the sentence; reset-baseline v2 was only a 5-minute smoke run.

---


The run reported here is v2 (`ticbudja/world-model-notes` v2, 4fd3b22). v1 (f406cd0) is excluded because of the bug described above.

### What changed

The work is on branch `harness/world-model-notes`. It adds `inference/agent/world_model_notes.py` and `tests/test_world_model_notes.py` (242 lines of tests). The rest is flag-gated code in `tool_agent.py`, `python_tool_sandbox.py` and `prompts.py`. In total the branch adds 699 lines and removes 5. The config is `configs/world-model-notes/v2.json`, which is reset-baseline v1 plus `ARC3_WORLD_MODEL_NOTES=1`.

**Prompt addendum** (`WORLD_MODEL_NOTES_ADDENDUM`, appended to the system prompt):

> World model:
> - When your understanding of the game changes - a rule confirmed or refuted, a new mechanic, a revised goal - record the update: emit a `<world_model>...</world_model>` block in your thinking or your reply, or append it from Python with `open('world_model.md', 'a').write(...)`. Write only what changed and the evidence for it.
> - `world_model.md` keeps your updates for this game, every line tagged automatically with the level and action count it was written at, e.g. `[L2 s57]`. Read it with `open('world_model.md')` when you need what you established earlier; filter by tag or content with `re`.

**Harness behaviour**

- **Two ways to write.**
  - A `<world_model>` block: each completion contributes at most one update, the **last complete** block. A block in the reply beats one in the thinking. Backticked mentions, blocks inside `<tool_call>` bodies, and unclosed blocks cut off by the output cap are ignored.
  - A Python write to `world_model.md`: any write mode appends.
  - Either way, an update identical to the previous one is skipped. So are lines already in the file and lines carrying copied harness tags, so a read-edit-write-back does not duplicate anything.
- **Storage.** Each game keeps a list of `(level, step, body)` entries, rendered as `world_model.md`. **Every line** is prefixed `[L<level> s<action count>]` for the point where it was written, so each line is a record the model can filter. The file is also saved as `artifacts/<game>_p0_world_model.md`.
- **Never injected.** The notes are never put back into the prompt; the model must `open()` them from Python. This was deliberate: an earlier design re-injected notes every turn, and that anchored the model on stale beliefs.
- **Sandbox files.** All file access is in memory, with no disk I/O.
  - `world_model.md` is readable and writable.
  - `/dev/null` accepts and discards writes.
  - Any other path is a scratch file kept for the rest of the game (1 MB per file, 4 MB total, text mode only).
  - Replaying every `open()` snippet the model wrote in the buggy v1 run (72 snippets) gave 0 errors.

### Usage

| | WM v2 |
|---|---:|
| Games with any note | 25 / 25 |
| Update points (distinct `[L s]` tags) | 313 |
| Note lines | 530 |
| `<world_model>` blocks extracted | **0** (every update was a Python write) |
| Python writes to `world_model.md` | ~350 |
| Reads of `world_model.md` | **~22** |
| Scratch-file opens | 133 |
| Tool calls that only wrote notes | 140 (~160K chars of code) |

Write and read counts were verified by hand from tool-call code. Plain regex counts over the transcripts are inflated, because the prompt text itself contains `open('world_model.md')`.

### Scores by game, with notes volume

| Game | RB | WM v2 | Note lines / updates | Δ |
|---|---:|---:|---|---:|
| ar25 | 100.00 | 100.00 | 11 / 6 | 0.0 |
| bp35 | 2.22 | 1.85 | 17 / 3 | −0.4 |
| cd82 | 89.28 | 100.00 | 10 / 10 | +10.7 |
| cn04 | 47.62 | 47.48 | 18 / 6 | −0.1 |
| dc22 | 47.62 | 47.09 | 37 / 10 | −0.5 |
| ft09 | 100.00 | 100.00 | 10 / 9 | 0.0 |
| g50t | 43.79 | 21.43 | 21 / 11 | −22.4 |
| ka59 | 24.90 | 32.88 | **71 / 35** | +8.0 |
| lf52 | 1.82 | 1.82 | 18 / 6 | 0.0 |
| lp85 | 100.00 | 100.00 | 19 / 15 | 0.0 |
| ls20 | 41.14 | 25.16 | 15 / 14 | −16.0 |
| m0r0 | 100.00 | 68.53 | 21 / 19 | −31.5 |
| r11l | 100.00 | 100.00 | 10 / 5 | 0.0 |
| re86 | 58.33 | 77.78 | 13 / 10 | +19.4 |
| s5i5 | 16.67 | 8.33 | 2 / 2 | −8.3 |
| sb26 | 100.00 | 100.00 | 14 / 8 | 0.0 |
| sc25 | 0.00 | 61.06 | 17 / 16 | +61.1 |
| sk48 | 0.56 | 2.78 | 11 / 3 | +2.2 |
| sp80 | 47.62 | **4.76** | 30 / 12 | **−42.9** |
| su15 | 76.28 | **22.22** | 39 / 13 | **−54.1** |
| tn36 | 74.81 | 65.62 | **46 / 45** | −9.2 |
| tr87 | 100.00 | 100.00 | 10 / 3 | 0.0 |
| tu93 | 67.31 | 80.00 | 40 / 33 | +12.7 |
| vc33 | 100.00 | **24.22** | 16 / 7 | **−75.8** |
| wa30 | 13.33 | 5.59 | 14 / 12 | −7.7 |

- **More notes went with slightly worse scores.** Every game wrote notes. Notes volume vs Δ score has Spearman ρ = **−0.26** (p = 0.21), weakly negative.
- **Writing notes cost play.** WM v2 played 514 fewer actions than RB (6,229 vs 6,743) and reached 9 fewer levels (115 vs 124), consistent with turns spent writing instead of acting.
- **The losses are concentrated in a few games.** The biggest were vc33 (−75.8), su15 (−54.1), sp80 (−42.9) and g50t (−22.4). The biggest gains were sc25 (+61.1) and re86 (+19.4).
- **vc33 dropped from 100 to 24 despite good notes.** Its notes were high quality (mechanics, invariants, per-level solutions), so good notes were not enough.

---

## Lessons

1. **A single-pass 25-game A/B can only detect large effects.**
   - The paired per-game differences here have SD ≈ 26 points, so the SE of the mean difference is ≈ 5.2.
   - At 80% power, the smallest effect such a run can reliably detect is about **±14 points**. The −6.2 result is inside the noise.
   - Single games swing by 50–80 points between near-identical configs (see [animation guards](animation-guards.md)), so vc33 −76 or sc25 +61 here is not evidence on its own.
   - **Rule:** do not attribute a single game's swing to a change unless that game's transcript shows the change firing.

2. **Change one variable at a time.**
   - This run also dropped the RESET sentence. animation-guards, which also ran without it, scored below RB too (51.6; this run 51.9; RB 58.1). This may be noise or a real effect of the sentence.
   - The result cannot be read until there is a 120-minute control run of main a0b99dc (no sentence, nothing else changed).

3. **The model uses tools it already knows, not new tags.** In the buggy v1 run the model kept trying `open('world_model.md','a')`. Once writing worked (v2), it never used the `<world_model>` tag: 0 blocks and about 350 writes. When designing an affordance, replay what the model already tries and make that path work.

4. **Memory that is written a lot but rarely read is pure cost.**
   - WM2 wrote about 350 times and read about 22 times, so the notes were rarely consulted. Notes that are not read cannot help, while writing them used tokens and actions (514 fewer actions and 9 fewer levels than RB).
   - A memory feature needs a read path that does not depend on the model choosing to read. One option is injecting the notes only at discontinuities: context trim, RESET, level start. Every-turn re-injection is already known to anchor the model.

5. **Status.** World-model notes were concluded negative on 2026-10-16. Do not propose self-written notes again unless there is a new approach to reading them.

## Reproduce

```bash
cd ARC3-Inference   # from the main checkout (runs/ is the Drive symlink)
kaggle-as tic make pull-kaggle NOTEBOOK=ticbudja/reset-baseline VERSION=1
kaggle-as tic make pull-kaggle NOTEBOOK=ticbudja/world-model-notes VERSION=2
make score_run SCORE_RUN_DIR=runs/kaggle/ticbudja/world-model-notes/v02   # the eval target ignores the run folder
```

Data sources:

- **Per-game scores:** each run's `summary.txt`.
- **Notes volume:** `artifacts/*_world_model.md`.
