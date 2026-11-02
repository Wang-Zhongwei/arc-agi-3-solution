# Confirmed rules audit

2026-11-01. Branch `confirmed-refuted-rules`, extraction code at `a71991a2` (`ARC3-Inference/inference/agent/confirmed_rules.py`).

The pinned section lists sentences from the model's own thinking that say "confirmed" (and, with `ARC3_REFUTED_RULES`, "refuted"). This audit asks whether those lines are true and whether a later reader can use them. Labels come from Claude verifier and reader agents, not from people. See [Limits](#limits).

**Status.** The numbers below are in-sample on `main-tuned-drafter` v02. Two candidate filters, v8 and v9, are frozen but not yet tested on held-out v01, and neither is in the harness.

## Main findings

1. **No score gain so far.** v06 scored 60.80 against 65.68 for v03: paired difference −4.89, SD 21.98, t = −1.11, p = 0.28, 25 games. This does not rule out either a gain or a loss: at this SD, a single pass only detects about ±13 points (80% power, two-sided 5%). Configs differ only by `ARC3_STUCK_CONTEXT_RESET=0`, `ARC3_CONFIRMED_RULES=1`, `ARC3_REFUTED_RULES=1`.
2. **Most lines are true, but 11% are wrong.** Of 235 confirmed lines: 182 correct (77%), 25 partly correct (11%), 26 wrong (11%), 1 unverifiable, 1 not a claim. 16 of the 26 wrong lines are in bp35, ka59 and g50t.
3. **Readers can't spot the bad lines.** Of 40 wrong or misleading lines, the same-level reader rated 33 meaningful. A section with wrong lines costs more than its share suggests.
4. **Wrong lines cluster where the model is stuck.** See the level table below.
5. **Hedging in the next sentence is a usable signal.** Lines followed by a hedge are 27% wrong, against 7% otherwise.
6. **Don't attach the sentence after a confirmation.** Consequences are only 47% true (33% wrong).
7. **The section changes what the model writes.** "Confirmed" is said 1.79× more often after the section appears, with flat placebo words. Accuracy measured on v02, which never had a section, may not carry over.
8. **Rules can remove useless lines, but not wrong ones.** After v8, 28 of the 32 current-phase noise lines are correct: they go stale or a reader can't tell what they refer to. v9's new filters remove about half of that noise for 1 useful line, and almost no bad lines, because wrong claims read just like good rules.

## Accuracy

### Confirmed and refuted

| Set | Lines | Correct | Partly | Wrong | Other |
|---|---|---|---|---|---|
| Confirmed | 235 | 182 (77%) | 25 (11%) | 26 (11%) | 2 |
| Refuted, real claims | 18 | 17 (94%) | 0 | 1 (6%) | 0 |
| Refuted, in section | 6 | 5 | 0 | 1 | 0 |

Another 7 of the 25 refuted sentences are not claims at all. The existing hedge filter drops all 7. Four of the 6 refuted lines in the section are not meaningful to a same-level reader, and 3 of the 6 are from tn36. Refuted is accurate but rare: v06's final sections hold 315 confirmed and 15 ruled-out lines, across 25 games, with 9 games having any ruled-out line.

### Level status

"Cleared" includes the last level of a won game.

| Level status | Lines | Levels | Lines per level | Agitated | Correct | Wrong |
|---|---|---|---|---|---|---|
| Cleared | 133 | 69 | 1.9 | 6% | 83% | 7% |
| Never cleared | 102 | 14 | 7.3 | 26% | 70% | 17% |
| Never cleared, agitated | 27 | | | | 59% | 30% |
| Never cleared, calm | 75 | | | | 73% | 12% |
| Cleared, agitated | 8 | | | | 100% | 0% |

Agitated means uppercase `CONFIRMED` or at least two ✓. The same marks are harmless once a level is cleared, so the signal is only used on the level being played.

### What follows a confirmation

| Next sentence | Confirmed lines | Refuted lines |
|---|---|---|
| Elaboration | 81 (34%) | 5 (20%) |
| Consequence | 78 (33%) | 2 (8%) |
| Moves on | 63 (27%) | 8 (32%) |
| Doubt | 13 (6%) | 9 (36%) |

- **Truth of the next sentence:** consequences 37/78 correct, 26/78 wrong. Elaborations 70/81 correct, but mostly stale detail.
- **Doubted lines:** 5/13 wrong. Too few to act on by themselves.
- **Hedge regex on the next sentence (`HEDGE`, broader than doubt):** kept 183 lines, 81% correct, 7% wrong. Dropped 52 lines, 63% correct, 27% wrong, costing 33 correct lines to remove 14 wrong ones.
- **Refuted:** a hedge after it doesn't predict error (6/6 correct without, 11/12 with). The filter would empty most "Ruled out" lists, so it applies to confirmed only.

## Can a reader use the line?

Two readers saw only the section, with no access to the transcript: one placed on the line's own level, one on a later level. Of 235 lines, the same-level reader found 170 meaningful (72%) and the later-level reader 115 (49%). They rated 23% and 17% fully clear, and agreed on "meaningful" for 75% of lines.

| Kind (same-level reader's label) | Lines | Meaningful, own level | Meaningful, later level | Correct |
|---|---|---|---|---|
| Game mechanic | 117 (50%) | 94% | 87% | 78% |
| Level fact | 64 (27%) | 83% | 14% | 70% |
| State snapshot | 30 (13%) | 17% | 7% | 90% |
| Empty or meta | 16 (7%) | 0% | 0% | 75% |
| Plan or progress | 8 (3%) | 25% | 25% | 88% |

Grouped by the later reader's own labels, the later-level column reads 91%, 7%, 6%, 0%, 0%. Snapshots are the most accurate kind and among the least useful: accuracy doesn't imply usefulness.

## Candidate filter

[`rule_v8.py`](scripts/rules_audit/rule_v8.py) runs after v7's extraction. It is frozen at sha256 `a0becc609406b3eecf6709b23a8484f9cd13c00279279f63db114919cdb060ac` for the held-out test. It isn't in the harness.

| Filter | Applies | Drops |
|---|---|---|
| H | always | Next sentence hedges or doubts |
| M | always | Generic artifact called confirmed with no content ("the plan is confirmed") |
| A | current level only | Uppercase `CONFIRMED` or at least two ✓ |
| I | current level only | Impossibility verdicts (unsolvable, dead end, trap, no exit…) |

Lines have two phases. **Current** is while the level is being played, judged by the same-level reader. **Past** is after it's cleared, judged by the later-level reader. Held lines would be labelled unverified while current, and A and I lines would return once the level is cleared.

- **Useful:** correct, or partly correct and not misleading; meaningful to that phase's reader; not bad.
- **Bad:** wrong, or rated misleading.
- **Noise:** everything else.

| Phase | Rule | Lines | Useful | Bad | Noise | Correct | Wrong |
|---|---|---|---|---|---|---|---|
| Current | v7 | 235 | 136 (58%) | 40 (17%) | 59 (25%) | 77% | 11% |
| Current | v8 | 149 | 101 (68%) | 16 (11%) | 32 (21%) | 84% | 5% |
| Past | v7 | 125 | 56 (45%) | 14 (11%) | 55 (44%) | 82% | 7% |
| Past | v8 | 100 | 51 (51%) | 7 (7%) | 42 (42%) | 87% | 2% |

Change from v7 to v8, game bootstrap, 95% interval:

| Phase | Bad share | Useful share |
|---|---|---|
| Current | −6.1 [−12.4, −0.3] pts | +9.8 [+4.2, +15.9] pts |
| Past | −4.0 [−9.4, +0.1] pts | +6.0 [+1.7, +11.4] pts |

What each filter catches (lines can match several):

| Filter | Phase | Lines | Useful | Bad | Noise |
|---|---|---|---|---|---|
| H | Current | 52 | 22 | 15 | 15 |
| M | Current | 7 | 0 | 0 | 7 |
| A | Current | 35 | 16 | 11 | 8 |
| I | Current | 11 | 2 | 7 | 2 |
| H | Past | 23 | 5 | 7 | 11 |
| M | Past | 4 | 0 | 0 | 4 |
| A | Past | 7 | 6 | 0 | 1 |
| I | Past | 1 | 0 | 1 | 0 |

The filter costs useful lines. In the current phase, v8 gives up 35 useful lines to remove 24 bad and 27 noise lines. Counting bad lines alone, that pays off only if a wrong line costs at least about 1.5× what a useful line gives (H and M 22/15, A and I 13/9). Noise would lower that if it costs anything.

## Useless lines left after v8

| Kind | Current | Past | Example |
|---|---|---|---|
| State snapshot | 18 | 15 | "Fresh state confirmed: P (2,3), O (9,6), bar 64." (`wa30-259`) |
| Level fact | 6 | 17 | "The plug's offset is (0,+12) confirmed." (`cn04-028`) |
| Game mechanic, too terse | 2 | 5 | "L- = +1 along BL confirmed by the unique O." (`lp85-112`) |
| Plan or progress | 3 | 2 | "Then (1,3)→(1,5) [confirmed], or (1,4)→(1,2)." (`lf52-102`) |
| Empty or meta | 3 | 3 | "I've now confirmed all buttons' effects:" (`dc22-058`) |
| Total | 32 | 42 | |

Coordinates can't serve as the filter: in the current phase, 34 of the 56 lines with coordinates are useful. [`noise_scan.py`](scripts/rules_audit/noise_scan.py) scores the first feature attempts.

## Candidate filter v9

[`rule_v9.py`](scripts/rules_audit/rule_v9.py) is v8 plus text patterns for useless lines. It is frozen at sha256 `d2b4c7104eac4356ca7ebcbc04a69e39fb2ad6d6365f73cd28d66eee18966d4c`.

Four agents designed filters independently, each for one kind of useless line. A fifth tried to break whatever passed, using reworded versions of the useful lines and 66 invented sentences. The bar, fixed in advance, was at least 4 lines across at least 3 games, with at least 70% of them noise or bad. Most candidates were rejected. Their code and the reasons are in [`noise_candidates/`](scripts/rules_audit/noise_candidates/) and `noise_design_result.json`.

| Filter | Applies | Drops | Example |
|---|---|---|---|
| fragment | always | Ends in a colon, or opens with a cell and has at most 12 words | "(17,2) unreachable ✓ confirmed." |
| snapshot | always | A right-now word with a place, a bare location, or a step tie in the main clause, with no input or general condition | "Confirmed: the plus is at rows 15-23, cols 6-14…" |
| dead_coords | past only | A place in the main clause with nothing said to happen there | "Confirmed: row 4 col 2 = M (wall), col 3 = B (open)." |
| dead_layout | past only | A layout word next to "confirmed", plus a number | "Confirmed the lower region layout (cols offset 2):" |
| dead_literal | past only | A private symbol set to a literal, a set formula, or a quoted grid | "B = {r+c<=9} — assumed by symmetry…" |

| Phase | Rule | Lines | Useful | Bad | Noise | Correct | Wrong |
|---|---|---|---|---|---|---|---|
| Current | v7 | 235 | 136 (58%) | 40 (17%) | 59 (25%) | 77% | 11% |
| Current | v8 | 149 | 101 (68%) | 16 (11%) | 32 (21%) | 84% | 5% |
| Current | v9 | 131 | 100 (76%) | 15 (11%) | 16 (12%) | 85% | 5% |
| Past | v7 | 125 | 56 (45%) | 14 (11%) | 55 (44%) | 82% | 7% |
| Past | v8 | 100 | 51 (51%) | 7 (7%) | 42 (42%) | 87% | 2% |
| Past | v9 | 76 | 51 (67%) | 6 (8%) | 19 (25%) | 87% | 3% |

Change from v8 to v9, game bootstrap, 95% interval:

| Phase | Bad share | Useful share |
|---|---|---|
| Current | +0.7 [−0.2, +1.8] pts | +8.4 [+4.2, +13.3] pts |
| Past | +0.9 [−1.5, +2.5] pts | +16.0 [+9.5, +24.3] pts |

What each new filter catches among the lines v8 keeps:

| Filter | Phase | Lines | Useful | Bad | Noise |
|---|---|---|---|---|---|
| fragment | Current | 8 | 1 | 0 | 7 |
| snapshot | Current | 10 | 0 | 1 | 9 |
| fragment | Past | 6 | 0 | 0 | 6 |
| snapshot | Past | 7 | 0 | 1 | 6 |
| dead_coords | Past | 13 | 0 | 0 | 13 |
| dead_layout | Past | 4 | 0 | 0 | 4 |
| dead_literal | Past | 5 | 0 | 0 | 5 |

The one useful line lost is `lp85-114`. Filters overlap, so the union removes 18 lines in the current phase and 24 in the past phase.

These numbers are upper bounds, since the patterns were written after reading the labels. The weak points:

- **`dead_coords`** matched 7 of the 66 invented sentences, such as "Hazard confirmed at (8,8): anything entering it dies".
- **`dead_layout` and `dead_literal`** fall below 4 lines without their largest game. Drop these first if the held-out test is weak.

### What rules can't catch

| Kind | Why | Evidence |
|---|---|---|
| Wrong or misleading claims | They read like good rules, so they need game ground truth | 15 bad lines remain in the current phase, 6 in the past phase |
| Repeats | Word overlap catches chance matches, and readers often rate real repeats useful | Word-overlap (Jaccard) matches at 30%: 18% junk in the current phase |
| One-event records vs rules shown by example | Same grammar | Past-tense event lines: 20% junk in the current phase, 33% in the past phase |
| Undefined private names | Good rules use them too | 21% junk in the current phase |
| Coordinates on a live level | Readers want them while playing | 53% junk with 15 useful lost when there's no cause word |

16 noise lines remain in the current phase and 19 in the past phase. The evidence column comes from the designers' and critic's scorer runs in `noise_design_result.json`; `reproduce.py` doesn't recompute it.

## The section changes the thinking

Each v06 game is split where its section first appears, and the control runs are split at the same thinking-character count. Rates are per 100K thinking characters, over the 23 games with thinking after the split in every run.

| Word | v06 before → after | v03 before → after | Ratio of ratios vs v03 | vs v02 |
|---|---|---|---|---|
| "confirmed" | 2.52 → 4.48 | 2.44 → 2.40 | 1.79 [1.28, 2.51] | 1.96 [1.36, 2.84] |
| hmm (placebo) | 215 → 315 | 201 → 296 | 1.00 [0.84, 1.19] | 0.98 [0.86, 1.12] |
| hypothes* (placebo) | 8.53 → 5.81 | 8.20 → 5.70 | 1.03 [0.77, 1.35] | 1.23 [0.89, 1.66] |

The refuted family is too rare to judge (vs v03: 3.33 [0.65, 15.16]).

## Code issues

- **Open:** `refuted_sentences` at `a71991a2` lets negated refutations through. "The second test doesn't disprove the ring theory at all here." lands under "Ruled out", as do "didn't rule out" and "doesn't refute".
- **v6 to v7:** the content filter changes extraction on v02 from 252 confirmed and 12 refuted to 232 and 6 ([`v6_vs_v7.py`](scripts/rules_audit/v6_vs_v7.py)). v7 has no Kaggle run.

## Limits

- Verdicts and readers are Claude agents. Nobody has hand-checked a sample. Each claim got two verifiers, one reading this run and one also using other runs of the game, with a judge on disputes: they agreed on 240/260 verdicts, and 51 claims went to the judge.
- v02 had no section. Finding 7 means the same extraction on a run with the section may behave differently.
- The audit covers every line v7 extracts from v02, whether or not a trim would have released it.
- v8 and v9 were designed on these same labels. Treat these numbers as optimistic until the v01 test is in. The designer agents had no access to v01 labels. A few v01 reader lines (bp35) were seen in the main session before v9 was frozen, and weren't used to design it.
- One run, 25 games. The bootstrap resamples games, not lines.

## How to check this

From the repo root, with the run folders synced from Drive:

```bash
cd docs/harness/scripts/rules_audit
python3 reproduce.py                 # the tables above, except the section-change one and the evidence for what rules can't catch
python3 noise_harness.py noise_candidates/cand_shortlist.py --list   # lines each v9 filter catches
python3 priming.py ../../../../ARC3-Inference/runs/kaggle/jonathanwang2022/main-tuned-drafter v06 v03 v02
python3 show_claim.py bp35-016       # one claim: labels, evidence and the transcript around it
```

Saved outputs are in the data folder: `reproduce_output.txt`, `priming_output.txt`, `v6_vs_v7_output.txt`. To check the files themselves, run `shasum -a 256 -c MANIFEST.sha256` there.

`reproduce.py` also checks that the joined labels match their sources and that each claim is at its transcript location (260/260).

To rebuild the claims from the raw transcripts (byte-identical on the machine that built them, since each record stores an absolute transcript path):

```bash
RUN="$(cd ARC3-Inference/runs/kaggle/jonathanwang2022/main-tuned-drafter/v02 && pwd -P)"
mkdir -p /tmp/rules_v7 && git archive a71991a2 ARC3-Inference/inference/agent/confirmed_rules.py | tar -x -C /tmp/rules_v7
python3 docs/harness/scripts/rules_audit/build_claims.py /tmp/rules_v7/ARC3-Inference "$RUN" /tmp/rules_v7/claims.json
cmp /tmp/rules_v7/claims.json "$RUN/rules_audit/claims_v02.json"
```

Claims worth spot-checking with `show_claim.py`:

| ID | Why |
|---|---|
| `sk48-180` | Correct, reusable mechanic |
| `bp35-016` | Wrong "level is unsolvable" chain, followed by a hedge |
| `g50t-071` | Wrong, and the next sentence says "wait, let me re-read" |
| `ka59-094` | Wrong, and the next sentence plans a double check |
| `wa30-259` | True, but a useless snapshot |
| `dc22-058` | Heading that lost its content |
| `sp80-194` | The one wrong refuted line in the section, followed by "but wait!" |

## Files

Data is in Drive at `ARC3-Inference/runs/kaggle/jonathanwang2022/main-tuned-drafter/v02/rules_audit/`, because run artifacts stay out of Git:

| File | Contents |
|---|---|
| `claims_v02.json` | The 260 extracted claims |
| `batches_v02.json`, `cross_runs.json`, `batch_ids.json` | Verifier inputs |
| `wf_result.json` | Raw verifier and judge output |
| `verified_v02.json` | Claims with final verdicts and evidence |
| `cold_args.json`, `cold_result.json` | Reader inputs and raw output |
| `joined_v02.json` | Verdicts plus both readers: start here |
| `dev_lines_v02.json` | Lines v8 keeps with labels, as given to the rule designers |
| `noise_design_result.json` | Designer and critic output, with reasons for each rejected rule |

Code is in [`scripts/rules_audit/`](scripts/rules_audit/):

| File | Purpose |
|---|---|
| `reproduce.py`, `priming.py`, `show_claim.py` | Recompute the numbers; inspect a claim |
| `rule_v8.py`, `rule_v9.py` | The frozen candidate filters |
| `build_claims.py`, `make_batches.py`, `cross_index.py` | Build the claims and verifier inputs |
| `aggregate.py`, `cold_agg.py`, `rule_eval.py` | Original tabulations |
| `holdout_eval.py`, `make_batches_v01.py` | Held-out scoring on v01 |
| `noise_harness.py`, `noise_scan.py`, `noise_v02.py` | Score filters for useless lines |
| `noise_candidates/` | Each designer's candidates and the critic's shortlist. `critic/` holds its helpers as they ran, with temp paths |
| `v6_vs_v7.py` | Extraction counts by code version |
| `verify_claims_workflow.js`, `cold_read_workflow.js`, `noise_rule_design_workflow.js`, and `*_v01.js` | Agent prompts and orchestration as they ran. Their input paths point to a deleted temp folder; use the Drive data folder |
| `audit_paths.py` | Data locations (`RULES_AUDIT_DATA` overrides) |

## Still open

- **Held-out test.** Score frozen v8 and v9 on v01 with `holdout_eval.py`. Verification is still running.
- **Harness change.** Nothing here is implemented in `confirmed_rules.py` yet. It would need the next sentence at harvest and the current level at render.
