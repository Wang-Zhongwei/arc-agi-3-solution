# Confirmed rules audit

2026-11-01. Compares the confirmed-rule filters in [`confirmed_rules.py`](../../ARC3-Inference/inference/agent/confirmed_rules.py) with what Kaggle run `jonathanwang2022/main-tuned-drafter` v06 actually pinned: 330 lines from 25 games, 315 confirmed and 15 ruled out.

**Decision.** Confirmed lines go through the filters below and show only for cleared levels. Ruled-out lines stay as they were. This is committed in `875ff4b8` and always on under `ARC3_CONFIRMED_RULES`. It has no Kaggle score yet: `v7.json` has the same settings as `v6.json`, and only the code differs.

## v06 against the filters

Two Claude readers judged each line from the section alone, without the transcript. One was placed on the line's own level, the other on a later level. "Meaningful" is their label. **It doesn't mean true**: the check of v06's lines against the transcripts was cancelled.

| Phase | Reader | v06 lines | v06 meaningful | New code lines | New code meaningful |
|---|---|---|---|---|---|
| Current: level being played | Same level | 315 | 219 (70%) | 0 | none shown |
| Past: level cleared | Later level | 221 | 102 (46%) | 136 | 75 (55%) |

The past row is the 221 of the 315 lines whose level the run later cleared.

For reference, the earlier candidate rule (rule v9) would have kept 159 current lines, 118 of them meaningful (74%). That version also used two current-only filters, `agitated` and `impossibility`.

What the decision gives up:

- **Past:** of the 85 lines dropped, 27 were meaningful. All 27 match `hedged_next`.
- **Current:** 219 lines the same-level reader found meaningful are no longer pinned while that level is played. A line is only pinned once a trim removes it from the conversation, so before that trim the model can still read it.
- **Never-cleared levels:** 94 lines from 12 games never show, 65 of them meaningful. These levels also have more wrong lines (see [Earlier labelled runs](#earlier-labelled-runs)).

## The filters

A confirmed line passes these steps in order. Sentences are split from the thinking of one reply.

| Step | Since | Drops or does |
|---|---|---|
| Find | v6 | Keeps sentences with the word "confirmed". Drops any with `?`, a hedge or conditional word (if, whether, would, let me...), or fewer than 5 words |
| Content filter | v7 | Tidies markup and joins a "...:" head with up to 4 following lines (400 characters). Drops subject-less fragments and sentences with fewer than 2 content words |
| Cut | v6 | Cuts each line to 500 characters |
| Line filters | New | The seven filters below |
| Release | v6 | A line waits until a trim removes its reply, and repeats are dropped |
| Cleared-level gate | New | A released confirmed line waits for the first trim after its level is cleared |
| Cap | v6 | Keeps the section to 8,000 characters by dropping the lines released first |

The seven line filters, with their catches among v06's 221 past lines (a line can match several):

| Filter | Drops | Example | Lines | Meaningful |
|---|---|---|---|---|
| `hedged_next` | The next sentence (first 600 characters) doubts or re-checks this one | "...confirmed. Wait, maybe that only happens once." | 69 | 27 |
| `generic_artifact` | A plan, model or map is called confirmed without saying what it is | "So the rotation model is confirmed." | 3 | 0 |
| `fragment` | A heading whose list is gone, or a short sentence (up to 12 words) starting with a cell | "I've now confirmed all buttons' effects:" | 4 | 0 |
| `snapshot` | Where something is at one moment | "Fresh state confirmed: P (2,3), O (9,6), bar 64." | 5 | 1 |
| `dead_coords` | A place with nothing said to happen there | "Confirmed: row 4 col 2 = M (wall), col 3 = B (open)." | 8 | 1 |
| `dead_layout` | One level's map or layout checked, with numbers | "The maze map is confirmed with 3 rooms." | 0 | 0 |
| `dead_literal` | A private symbol set to a value, a set formula or a quoted grid | "B = {r+c<=9} level 2 confirmed it." | 0 | 0 |

The v7 content filter accounts for another 13 past lines, none meaningful. `dead_layout` and `dead_literal` caught nothing here and at most 1 line on v01, so they are effectively untested.

**Ruled-out lines** use a strong refutation word (refuted, disproved, falsified, debunked, ruled out). They get the find, content filter, cut, release and cap steps, but no line filters and no cleared-level gate. v06 had only 15, in 9 games.

**Left out:** `agitated` (uppercase `CONFIRMED` or two ✓) and `impossibility` were only for the level being played, which no longer shows confirmed lines. `agitated` also failed on v01: 19 of the 23 lines it caught were useful.

## Earlier labelled runs

v06's lines have no truth labels, so truth comes from v02 and v01. Verifier agents checked those lines against the transcripts. The filters were designed on v02 and frozen before v01 was labelled. The old filter version included `agitated`, which doesn't affect the past rows below. Full tables are in [`8aa82362`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/8aa82362/docs/harness/confirmed-rules-audit.md).

| Held-out v01, past phase | v7 extraction | With filters | Change, game bootstrap 95% |
|---|---|---|---|
| Useful | 52% | 64% | +11.9 [+5.1, +20.2] |
| Noise | 38% | 28% | −9.2 [−16.9, −1.4] |
| Bad | 11% | 8% | −2.8 [−7.5, +1.2] |

Bad means wrong or misleading. Useful means correct and meaningful. Noise is everything else.

- **Accuracy:** 76% of v01's confirmed lines are correct, 13% partly and 9% wrong (v02: 77%, 11%, 11%).
- **Next-sentence hedge:** lines followed by one are 22% wrong, against 7% otherwise (v02: 27%, 7%).
- **Level status:** lines on never-cleared levels are 13% wrong, against 7% on cleared levels (v02: 17%, 7%).
- **Wrong lines stay:** the filters mostly remove useless lines. Wrong claims read just like good rules, and the bad-line change above could be zero.
- **Ruled-out accuracy:** 17 of 18 correct on v02 didn't replicate on v01 (5 of 9 correct, 3 partly, 1 wrong).

## Limits

- **No score evidence.** v06 scored 60.80 against 65.68 for v03: paired difference −4.89, SD 21.98, t = −1.11, p = 0.28, 25 games. A single pass only detects about ±13 points (80% power, two-sided 5%), so a small gain or loss from the new code won't be visible.
- **Agent labels.** Readers and verifiers are Claude agents, and nobody has hand-checked a sample. Readers also vary between runs, so compare shares within a run.
- **Text differences.** Readers judged v06's text, while the filters run on the v7-tidied text. v7 extraction could add up to 12 colon-joined lines that v06 never showed, and they have no labels.
- **Same games.** v01, v02 and v06 share the same 25 games.
- **The section changes the thinking.** After the section appears in v06, the model writes "confirmed" 1.79× more often than v03 does at the same point (95% [1.28, 2.51]), while placebo words stay flat. Hiding current-level lines may change this, and that hasn't been measured.
- **Open bug.** `refuted_sentences` lets negated refutations through, such as "doesn't disprove", "didn't rule out" and "doesn't refute".

## Files

Scripts are in [`scripts/rules_audit/`](scripts/rules_audit/). Data is in `ARC3-Inference/runs/kaggle/jonathanwang2022/main-tuned-drafter/v06/rules_audit/` on Drive.

| Script | Does | Output |
|---|---|---|
| [`prepare_v06.py`](scripts/rules_audit/prepare_v06.py) | Takes the last copy of the section in each v06 transcript and maps each line back to its thinking | `claims_v06.json` |
| [`cold_read_workflow.js`](scripts/rules_audit/cold_read_workflow.js) | Runs the readers | `cold_result_v06.json` |
| [`score_v06.py`](scripts/rules_audit/score_v06.py) | Prints the tables above, using the harness filter code | stdout |

To rerun the tables, from `ARC3-Inference/`:

```bash
.venv/bin/python ../docs/harness/scripts/rules_audit/score_v06.py
```
