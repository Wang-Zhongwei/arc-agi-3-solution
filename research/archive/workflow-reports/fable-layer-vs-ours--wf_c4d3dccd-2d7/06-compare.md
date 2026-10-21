# fable-layer-vs-ours: compare

Workflow run `wf_c4d3dccd-2d7`, phase Compare. Compare leoprovorov Fable & Astra notebook (handbook + Fable layer on Franzen) with our stack: read, measure claims on our runs, compare, skeptic-verify

---

# Fable & Astra notebook vs ours: what is actually better

2026-10-05. This answers the user's request to "compare how this notebook is better than our notebook" (https://www.kaggle.com/code/leoprovorov/fable-astra-play-arc-3-handbook-harness).

**Tags**
- **[verified]**: checked against code or files, or recomputed.
- **[their claim, unverified]**: their figure, not reproduced.
- **[inferred]**: my reasoning.

**Path keys**
- `FC` = `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/fable-compare/`. Every script and output cited below lives here.
- `SV2` = `.../scratchpad/wt/stack-v2/ARC3-Inference/`
  - `TA` = `SV2/inference/agent/tool_agent.py`
  - `PR` = `SV2/inference/agent/prompts.py`
- `C` / `P` = `FC/fable_layer_core.py` / `FC/fable_layer_patch.py`. These are their notebook cells 41 / 42 and match `FC/cell41.txt` / `cell42.txt` line for line.
- `HB p.N` = handbook PDF page (text in `FC/pages.txt`).
- `NB n` = `competitors/leoprovorov-fable-astra/narrative.md`, cell n.
- `RW` = `competitors/rellik13-milestone2/WRITEUP.md`
- `MV2D` / `MV2M` = `scratchpad/memory-v2-design.md` / `scratchpad/memory-v2-measure.md`
- r1 = run `fz-ledger-on-trim`, r2 = `fz-ledger-on-trim-r2`, control = `fz-arms-v4-control`. All under `/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs/`.

## 0. Short answer

**As a submission, their notebook has not been shown to be better.**
- Its leaderboard score of 31.73 comes from V1, which is an unmodified copy of Franzen's notebook.
- The only version that contains the Fable layer is V2, and it ran for 10 seconds (Kaggle version list, seen today). [verified on the page; the attribution is inferred]
- Their handbook lists the pinning variants as "not yet run" (HB p.84). [verified quote]
- Their only measured change was a journal that replaced history. It cut cleared levels from 46 to 23 (HB p.78). [their claim, unverified]

**On the leaderboard they are ahead of us today, but not because of the layer.**
- Their 31.73 against our best of 27.62 (submission 56838702, 2026-10-05, from `kaggle competitions submissions`). [verified]
- Resubmitting the same notebook moves its score by about ±4 (HB p.5). [their claim]
- Franzen's notebook has itself scored both 27.89 and 31.47 (arms.md:148). [verified]
- stack-v2 has never been run or submitted. The leaderboard is about 0.56× the offline score (arms.md:22), which would put stack-v2 near 28. [inferred]

**Where they are genuinely ahead:**
1. **Harness-computed facts.** The harness works facts out of the frames, so nothing depends on the model's cooperation. Our memory depends on model-written tags, and only 0.9–2.1% of calls contain one (MV2M:22). [verified]
2. **The clock/bar diagnosis.** They found that the model reads the step bar as a wall-clock timer, because `time_remaining_seconds` appears in every action result (HB p.15).
   - New today: our runs show the same thing. In each of r1, r2 and control, the model proposes "the bar is a real-time timer" in 12 of 25 games (`FC/clock_confusion.out`). [verified snippets; the regex count is approximate]
3. **Test method.** Running copies of one level, with minimum-detectable-effect (MDE) maths and decision thresholds set in advance, is more sensitive for changes that act within one level.
4. **Diagnosis.** A careful per-level written analysis: the 90-page handbook.

**Where we are ahead:**
- A measured memory gain that has been replicated: ledger-on-trim scored 52.56 and 49.14 against control 46.25 (arms.md:53-54).
- A head placement that adds no prefill and no slot handover (verified by simulation).
- Paired 25-game, 121-minute evaluation, and unit tests on every branch.
- Their layer has about 12 bugs or design gaps. Two of them break when combined with our ledger.

**New measurements made for this comparison** (each run under `nice`, one at a time):
- `FC/action_penalty.py` → `action_penalty.out`: how much score the action-count term costs.
  - It reproduces the official run means exactly: 46.25, 45.92, 44.10, 52.56, 49.14 and 43.32.
- `FC/gate_repeats.py` → `gate_repeats.out`: where memory-v2's targets sit relative to tokens spent on the level (r2).
  - It reproduces MV2M's count of 504 exact repeats (8.95% of actions).
- `FC/clock_confusion.py` → `clock_confusion.out`: model reasoning that treats the step bar as a real-time timer, across 3 full runs.
  - `clock_words.py` is an earlier, noisier pass. Ignore it.
- r2 transcripts: `grep -o "NameError: name '…' is not defined"` gives 44 NameErrors against 2,217 `[TOOL CALL: python]` blocks.

## 1. Verdict table

| Dimension | Theirs (F001 = Franzen M2 + Fable layer) | Ours (stack-v2) | Who is ahead | Evidence |
|---|---|---|---|---|
| Base harness, model, serving, config | Franzen M2, unchanged | Franzen M2, unchanged | Tie | F001 cells 2 and 4 are byte-identical to our `kaggle/template.ipynb` cells 2 and 4. [verified, diffed in the layer pass] |
| Measured gain from the added layer | None. V2 ran for 10 s, and the pinning variants are "not yet run" | ledger-on-trim scored 52.56 and 49.14 vs control 46.25 (mean +4.6) on 25 games at 121 min | **Ours** | HB p.84 [verified quote]; Kaggle version list; arms.md:53-54 [verified] |
| Public leaderboard today | 31.73, from V1 (the unmodified Franzen copy) | Best is 27.62, from a notebook the CLI doesn't record. stack-v2 has not been submitted | **Theirs** on the board, but not from the layer | `kaggle competitions submissions` [verified]; ±4 per resubmission (HB p.5) [their claim]; Franzen 27.89 / 31.47 (arms.md:148) [verified] |
| Facts the harness computes from frames | Indicator notes, per-key tallies, a step-bar estimate and a cleared-level digest. None of it needs the model | A diff image and a per-action echo. Memory comes from model tags (in 0.9–2.1% of calls) plus thinking fragments. memory-v2 is designed, not built | **Theirs**, by design; unmeasured | C:151-540; MV2M:22, 36-38 [verified] |
| Step budget | A bar detector plus an alert at ≤25% left or ≤12 actions. Bugs: it fires about once per level (B1) and can be 5 frames stale (B2) | The prompt says "count remaining budget into your plans" (PR:247) but also "do not get distracted by it" (PR:105) | **Theirs**, slightly | The alert would have been news on only 1 of 35 levels; on the other 34 the model mentioned the bar first (`FC/claims_analysis.out`) [verified] |
| Clock mistaken for the step bar | Diagnosed (HB p.15). A fix (hiding the clock) is proposed but not run (HB p.31, 76) | We show the same field (TA:5763-5765). The model calls the bar a real-time timer in 12 of 25 games per run | **Theirs** for the diagnosis; neither side has fixed it | `FC/clock_confusion.out` [verified snippets] |
| Indicator / tray tracking | Works on the sk48 trays (rows 58-59). But 34% of MOUSE events sit on the click point, and 4 of 10 sampled events were not indicators | Nothing equivalent. The diff image shows the changed cells each turn | Mixed | `FC/claims_analysis.out`, `FC/fp_samples/` [counts verified; the labels are my judgement] |
| Memory across context trims | A LEVEL LOG pinned at the head after trims. Unmeasured | The ledger placed at the head after trims. Measured at +3 to +6 | **Ours** | arms.md:53-54; MV2M:129 |
| Cache and slot hygiene | A refresh deletes the old pin from mid-history, which forces a full re-prefill without the trim flag. That happens 18 times in the r2 replay, at about 4.7 s each. The content-matched restore breaks when combined with our ledger | The head edit rides on the trim's own re-prefill and handover; no extra flags | **Ours** | `FC/replay_fz-ledger-on-trim-r2.json`, `FC/sim_stackv2_conflict.py`, `FC/sim_eviction_flags.py` [verified] |
| "Facts, not advice" | Stated as a rule, yet the log ends with 5 advice lines and an assistant turn the model never wrote | Franzen's ~4.1k-token advice prompt. attention-prompt was advice from 19:31 to 20:44 today (commit 1406ec5) and is now a harness-computed `Salient:` line | Tie: both are mixed | C:507-516 [verified]; attention-prompt commits 9c3a40b..f43ca80 (20:44-20:45) [verified git log] |
| Code safety | Anchored patch that skips itself if any anchor is missing. Every hook is wrapped in try/except and logs only at `log.debug`, so failures are silent | Branch commits with tests (32 on stack-v2) and tests that flags-off leaves output byte-identical | **Ours** for testing. Theirs is easier to drop into someone else's notebook | P:281-313; arms.md:190 |
| Evaluation method | N copies of one level, MDE maths, thresholds set in advance. But one run per setting, on 15 hard games at 72 min | Paired 25-game, 121-min runs with `make significance`; the ledger result is replicated; a replay simulator | Different strengths (§3) | HB p.17, p.86; NB 38; arms.md:20-22 |
| Diagnosis | 13 levels read by hand. On sk48, the point where Fable and the model diverge is traced to a single turn | Quantitative data cards (MV2M) and this report | Theirs goes deeper per level; ours covers more | HB p.15-26; MV2M |
| Kaggle robustness | Falls back to a Python 3.12 venv | A pinned byod image, fail-fast when the server dies, and serve.log streaming | **Ours**, for our image | Layer pass §1 items 3 and 6 |

## 2. Ranked arms to port or adapt

**What sets the size of any expected effect:**
- **Token budget.** Each of our games gets about 180k generated tokens in 121 minutes (4.48M across 25 games, arms.md:87). Their hidden-run budget is about 181k (HB p.4). [verified arithmetic]
- **Elasticity.** Near that budget, score rises about 0.25% for each 1% more tokens (NB 13). [their claim, unverified] Our streams14 run agrees: +15% tok/s gave +0.13 points (arms.md:89).
  - So an arm that only saves 10% of tokens is worth about +1 point. [inferred]
- **Action-count penalty.** It costs 1.4–2.2 points in the control and ledger runs, and 1.0–3.9 across all 6 runs (`FC/action_penalty.out`). [verified]
  - Each cleared level scores min(115, (human/actions)²·100) (`daniel-franzen-sol/tufa-arc-agi-framework/src/taaf/game.py:381-416`). [verified] Every action on the level counts, including those in dead attempts (game.py:259). So MV2M:75's "actions are not penalised" is wrong.
  - Arms that only make play more efficient are therefore capped near 2 points. Anything bigger has to come from clearing more levels. [inferred]
- **Compute.** We are out of Kaggle compute. Each test plan therefore lists what can be checked offline now, and what goes into the next Kaggle batch.

### 1. `budget-truth`: hide the run clock and delete the "don't get distracted by the bar" sentence

**What.** One arm with two flags:
- (a) Stop copying `run_elapsed_seconds` and `time_remaining_seconds` into the action result the model sees.
- (b) Remove "If you identify such a bar, do not get distracted by it…" from Franzen's visual-game text. Keep the next sentence, which says not to click through the bar's segments.

**Where.**
- (a) `_compact_action_result`, TA:5763-5765.
  - The endgame clock reads the raw payload (TA:5771-5773), so `_in_endgame` (TA:6503) is unaffected. [verified]
  - The sandbox's `last_action_call_result` is the same compact dict (TA:5937-5950), so the clock is hidden there too. [verified]
- (b) PR:105.

**Why.**
- Their p.15 finding reproduces on our runs. In each of r1, r2 and control, the model proposes that the bar is a real-time timer in 12 of 25 games, often citing `time_remaining_seconds`. [verified snippets in `FC/clock_confusion.out`]
- At least three of the first-mention snippets conclude that there is no action budget:
  - control dc22: "it's not a hard action budget … probably a real-time timer"
  - control re86: "there's no strict action budget (the bar is real-time)"
  - r1 g50t: "Not step-based. Good: I have plenty of actions"
- rellik13 deleted the same "ignore the bar" line as a fix (RW:84-85). [verified quote; his effect wasn't isolated]
- Our prompt contradicts itself: PR:105 against PR:247. [verified]
- In the hidden run each game has 532 minutes, so the clock values the model sees offline (at most 7,260 s) don't match the real run anyway. [inferred]

**Expected effect.** Small: well under 1–2 points, and not detectable in a single run. [inferred]
- Unintended budget deaths number 2–6 per run. Of the 17 budget deaths across r2 and stack-v1, 4 were clearly unintended and 8 unclear (`FC/death_intent.py`).
- A dead attempt averages 43.5 actions and about 0.6% of a run's generated tokens (15.2% across 24 deaths in r2, MV2M:76, 88).
- The cost of the clock confusion elsewhere, in reasoning spent on it, is not measured.

**Cost.** Zero tokens; it actually removes about 20 tokens per action result. About 10 lines of code.

**Risk.** Low. The model loses run-level pacing, which the harness's endgame phase already handles.

**Test plan.**
- **Offline now:**
  - A unit test that the compact result has no timing keys and that `_game_elapsed_seconds` still updates.
  - A flags-off identity test.
- **On Kaggle:** no dedicated run. It rides along in the next treatment stack (see arm 2). Read-outs:
  - `clock_confusion.py` matches should fall from about 12 of 25 games to about 0.
  - Budget deaths should not rise (`frames_pass.py` death classes).

### 2. `mem-facts-gated`: memory-v2 B+C+D plus a step-bar fact, with per-turn lines held until 15k tokens on the level

**What.** memory-v2 as designed (MV2D §3-6), with the §4 edits:
- A hold on the per-turn lines until the level has used 15k tokens.
- A step-bar rate fact from a fixed version of Fable's detector.
- Counts in place of the label "inert".
- A single head card.

**Where.**
- `_mem_observe`, right after `_death_ledger_observe` (TA:5153).
- Deltas at the end of the opener.
- The card through `_apply_due_ledger` (TA:6382-6393), called after the trim in the request loop (TA:6825-6826).
- The level-change hook, next to `self._ledger.on_level_change` (TA:3912-3916).
- The rollback snapshot at TA:6697.

**Why.**
- It goes after the largest measured losses: exact (state, action) repeats are 9% of actions, 27% of actions sit on loops, and dead attempts take 15% of tokens (MV2M:62-71, 88).
- In r2, 85% of exact repeats and 75% of revisits happen after the level passes 15k tokens (`FC/gate_repeats.out`). [verified] The repeat rate by tokens on the level:

  | Tokens on level | Repeat rate |
  |---|---|
  | under 15k | 3.0% |
  | 15–30k | 11.4% |
  | over 30k | 15.3% |

- Fable's harness-computed facts point the same way. rellik13's evidence is that facts helped and advice hurt (RW:59-71).

**Expected effect.** Uncertain: 0 to +3. [inferred]
- Halving the repeats and loop actions after 15k would save maybe 3–6% of tokens. That is about +0.4–0.7 by elasticity, plus at most about 1 point from the action term.
- Anything more has to come from breaking loops on stuck levels: 53% of the tokens spent after a level passes 30k go to levels that are never cleared (`FC/gate_share.py`).
- The ledger's +3 to +6 shows that text shown at the right moment can move the score more than token arithmetic predicts. Why it does is unexplained (MV2M:129).

**Cost.**
- About 1–3 ms of CPU per call. Fable's comparable code measured 0.3–2.6 ms mean in our replay. [verified for their code]
- At most about 12k prompt tokens per game, roughly 1% of input (MV2D:105). [design estimate] Gating makes it smaller.

**Risk.**
- Facts can anchor exploration (MV2D:117-118).
- A wrong "no effect" fact is exactly what sank their journal. Of its 28 "no effect" entries, all 15 that were retried later changed the board, and none was confirmed (HB p.84). [their claim]
- Mitigation: state counts together with the boards they were seen on (§4 E3).

**Test plan.**
- **Offline now:**
  - The design's tests (MV2D §10), plus E4–E6 below.
  - Replay on the r2 and stack-v1 frames to count lines per game.
  - Check 30 rendered lines by hand for truth.
- **Kaggle, when compute returns:**
  - stack-v2 against stack-v2 plus this arm (plus arm 1), from the same commit, at 25 games × 121 min, run twice.
  - `make significance`, plus the read-outs in MV2D:116 and §4 E7.

### 3. `mem-solved`: memory-v2 (A), the analogue of rellik13's +2.8

**What.** The harness writes one `<solved>` record per cleared level:
- The loop-free winning path, with click coordinates.
- Actions spent and deaths.
- The end-state diff.
- One "how" sentence.

It appears on the next level's first opener and in every head card.

**Where.** Written in the level-change hook (TA:3910-3916). Shown through the opener delta and `_apply_due_ledger`.

**Why.**
- rellik13's only text addition that helped was a pinned record of cleared levels: +2.8 (7.14 → 9.96), from one submission (RW:69-71). His same-notebook resubmissions scored 5.02, 5.50 and 6.42 (RW:17). [verified quotes]
- His code pins harness facts (action count, the last 30 winning actions, about 900 chars of the clearing turn's reasoning) plus a model-written "Rule for level N" (`rellik13-milestone2/cells/42_…`). [verified in the earlier pass]
- Fable's digest is weaker: it drops click coordinates (B9).

**Expected effect.** Small to moderate. [inferred]
- The next level is cheaper in only 34% of pairs (MV2M:102).
- On later levels the model already mentions reuse in 44–50% of first calls (MV2M:104).

**Cost.** At most 165 chars per level (MV2D:29).

**Risk.** Low. A stale path could be replayed blindly on a level that has changed.

**Test plan.**
- Run it in the same Kaggle pair as arm 2. A and B–D are separate flags, so ablate only if the stack wins.
- Read-outs: the next-level-cheaper rate (34% today) and the number of calls before the first action on level 2 and later.

### 4. `attention-salient`: keep the facts version that landed today

**What changed today.** At 20:44–20:45, `arm/attention-prompt` replaced the "Where to look" advice with:
- A harness-computed `Salient:` line in each turn message.
- Two descriptive system-prompt sentences.

These are commits 9c3a40b, 1f2a1a5, 8bacd3d and f43ca80. [verified git log] The old advice included "set aside changes confined to a timer or remaining-steps bar at the edge" (1406ec5, prompts.py:156); that text is no longer in the branch.

**Recommendation.**
- Run only the facts version.
- Keep `ARC3_ATTENTION_AFTER_TOKENS=0`. These facts are about new elements, which matter at the start of a level. 44% of actions, and 51% of no-ops, come before a level reaches 15k tokens (`FC/gate_repeats.out`). [counts verified; "matters early" is inferred]
- Update PR #57's description and arms.md:135. Both still describe the advice version.

**For the record, the evidence against the advice version:**
- rellik13: extra text in every prompt dropped his score from 6.42 to 4.16, and a level-start advice block dropped it from 13.40 to 9.98 (RW:59).
  - Both results are confounded. His offline practice scores went the other way: the 9.98 notebook had scored 37 in practice and the 13.40 notebook had scored 32 (RW:18). [inferred to be the same pair]
- Ours: audit60k scored 44.10 against 46.25 (arms.md:92), which is inside the noise band.
- Handbook: there is "no data in favour" of prompts built from strong-agent lessons (HB p.25). [their claim]

**Test plan.** Agree a shared per-turn block with arm 2 first (§4 E8). Then include it in the same treatment stack, or in a second pair if compute allows.

### 5. `indicator-facts`: Fable's indicator notes, rewritten as plain facts

**What.** Report a small changed patch far from the main change, with no interpretation. Example: "rows 58-59, cols 28-29: red → white after action 15; back to red after action 20".

**Where.** In the opener, after the valid-actions line (Fable edit 2, TA:5148). Feed it from `_mem_observe`, so there is only one frame ledger.

**Changes from Fable:**
- Drop patches within 2 cells of a click; 34% of MOUSE events are on the click point.
- Drop "Legends, trays…" and "A mark you earned may have been lost".
- Never add a negative sentence that "nothing moved it".
- Keep their 15k-token gate (P:134).

**Expected effect.** Near zero overall. [inferred]
- Lost marks track tokens on sk48 (ρ 0.80, n=6, p=0.07), but only weakly when all games are pooled (ρ 0.26, n=330).
- The causal direction is untested (`FC/claims_analysis.out`). [verified]

**Gate before any Kaggle run.** Relabel a 30-event replay sample by hand. Continue only if at least 70% are real indicators or informative far-off changes. In today's 10-event sample, 6 of 10 were.

### 6. `persist-sandbox-vars`: keep variables between python calls

**Why.**
- In their baseline, 15 of 39 tool errors were "name not found" (HB p.28). [their claim]
- In r2 we see 44 NameErrors in 2,217 python calls, about 2%. [verified count]
- About 10–12% of our calls redefine a retained function (MV2M:52).

**Expected effect.** Small. Low priority, behind arms 1–5.

### 7. Test infrastructure: a copies panel (not an arm)

**What.** A builder option that runs N copies of one game for M minutes on the 10 slots, with a decision rule set before the run, as on HB p.86.

**Use it only for** mechanisms that act within one level and are expected to have large effects:
- Step-bar facts on wa30 and vc33, which had 5 and 3 budget deaths.
- Death facts on bp35 level 2 and tu93 level 2, where the first-death site recurs in 8 of 8 runs (MV2M:92).
- Trays on sk48 level 1.

**Power.** With n=10 per arm it detects about 0.87 levels (NB 38) [verified arithmetic]; with n=30, about 0.5.

**Cost.** About one 48-minute session per level studied, against more than 2 hours for a full run. Whether a run can start at a fixed level has not been checked. [inferred]

## 3. Method: theirs vs ours

| Aspect | Theirs | Ours | Take |
|---|---|---|---|
| Unit of test | 10 copies of sk48 level 1, 48 min each, no queue (HB p.17, p.86) | 25 games × 121 min on 10 shared slots, paired per-game test | Theirs isolates effects within one level and removes scheduler noise. Ours measures the thing we are optimizing |
| Power and decision rule | MDE of 0.87 levels at n=10. Thresholds set in advance: a mean of at least 2.4 levels counts as a benefit, at most 0.6 as harm (HB p.86; NB 38) | A noise band of about 3 points (arms.md:22); 1–2 runs per arm; `make significance` | Adopt a decision rule written down before each run |
| Normalization | Elasticity curve and budget per game (HB p.4; NB 12-13); levels per 100 active game-minutes (HB p.82) | Tokens per cleared level (arms.md:53), levels.csv, sim-v2 calibration | Use elasticity to bound arms that only save tokens. Add levels per active minute when an arm may change pauses |
| Simulator | None | Replay simulator, within about 1.5 points of real runs. But D′ won in simulation (+5) and lost for real (stack-v1 43.32, arms.md:83-87) | Ours is useful for scheduling only, and not for start order |
| Checks before scoring | A scripted model: flag off leaves openers identical; the pin is restored after trims (NB 39) | Unit tests per branch, flags-off identity snapshots, replays on recorded frames | Similar; ours is more extensive |
| Qualitative | 13 levels read by hand; the Fable vs model divergence (HB p.15-26) | Data cards built from transcripts (MV2M) | Complementary. Reading the levels where an arm regressed by hand is cheap; do it after each run |
| Runs actually done | B003 vs B005, one run each (HB p.78) | 11 full runs, including the replicated ledger | Ours has more evidence |

**What to change in our method:**
1. Write the keep/drop rule into the PR before each run.
2. Report per-level read-outs alongside the game mean: P(clear | the level reached k tokens) and tokens per cleared level, from levels.csv. That gives about 330 level units per run pair, against 25 games.
3. Report levels per 100 active game-minutes. That metric is what exposed B005's pause cost: it fell from 5.4 to 3.0 (HB p.82).
4. Before spending compute on an arm that only saves tokens, bound its gain with the elasticity: +10% tokens is about +1 point.
5. Don't trust the simulator for start order.

## 4. Lessons for our pending work

### memory-v2: edits to make before the implementation runs

There is no separate memory-v2 implementation-prompt file in the scratchpad (I listed `scratchpad/` and searched for one). `memory-v2-design.md` is the spec the prompt will be built from, so each edit below names the MV2D line it changes.

**E1. Hold the per-turn lines until 15k tokens on the level** (MV2D:7-9 knobs, MV2D:57 for D, MV2D:84 for deltas).
- Add `ARC3_MEM_LINES_AFTER_TOKENS`, default 15000. Below it, nothing is shown, but the recorders still run, as the death-ledger lesson requires (MV2D:13-14).
- Measure tokens on the level as `self._session_generated_tokens - self._tokens_at_level_start` (TA:3911), the same quantity Fable gates on (P:53-54).
- These bypass the hold:
  - the `<solved>` delta on a new level's first opener (MV2D:97)
  - first death and first clear items
  - the bar fact (E2)
  - the trim card
- Evidence: 85% of exact repeats and 75% of revisits come after 15k, and the repeat rate below 15k is 3.0% (`FC/gate_repeats.out`) [verified]. About a third of level attempts never reach 15k (`FC/gate_share.py`). Below 15k, the action echo already reports per-action no-ops (`SV2/inference/framework/solver.py:132, 186-230`).
- If compute allows, add an arm variant with the knob at 0.

**E2. Add `ARC3_MEM_BUDGET`, a step-bar fact, inside `_mem_observe` rather than as a second frame ledger** (MV2D:7 flags, MV2D:13-23 pipeline, MV2D:48 and 62 hazard).
- Port Fable's `detect_budget` (C:192-269) with these fixes:
  - (i) Keep its state per attempt, resetting it on death or RESET. This fixes B1 (`FC/test_budget_attempts.py`: attempts 2 and 3 never alert).
  - (ii) Recompute on every new frame, with no 5-frame cache. This fixes B2.
  - (iii) Stay silent when the bar refills or changes colour. The detector misses ls20 and sc25.
  - (iv) Compute the rate over every action in the attempt, UNDO included. UNDO moved the bar on 9 of 22 presses.
- Text: `bar row 63: 20 of 64 cells left, -2.0 per action this attempt (about 10 actions at this rate)`.
- Show it once per attempt, at first detection, and again when 25% or less, or 12 actions or less, remain.
- Hazard lines (MV2D:48 says "never a cause"): allow one cause label, `bar empty`, when the detected bar reached 0 at the fatal action. That is a frame fact, not a guess. In r2 plus stack-v1, 17 of 44 deaths came with the bar empty or nearly empty (`FC/claims_analysis.out`) [verified].
- Why: the model mentions the bar early on 85% of levels, but in about half the games it suspects the bar is a wall-clock timer (§2 arm 1). A per-action rate measured by the harness settles that question with a fact.

**E3. State counts and boards, never verdicts** (MV2D:43 example, MV2D:48 "inert" rule).
- Render "no change on 4 presses across 2 boards" instead of "inert" or "(inert so far)".
- Ban these words in harness text: inert, refuted, impossible, useless, cannot.
- Evidence: of their journal's 28 "no effect" entries, all 15 that were retried later changed the board (HB p.84). Fable's own "no action of yours has moved it" comes from a detector that throws away busy frames (B8).

**E4. Exactly one head card, identified by its marker** (MV2D:61, MV2D:77).
- Add three invariants, each with a test:
  - (a) No code path inserts a separate pinned head message.
  - (b) A re-render replaces only the block that `_LEDGER_BLOCK_RE` (TA:139) matches. Never search for the previous text.
  - (c) After a trim the card appears once. The next request without a trim has a byte-identical `messages[1]` and exactly one `<memory>` block.
- Any future arm that pins head content (indicators, attention) writes into this card.
- Evidence: Fable restores its pin by matching content, and our ledger merges into the head. Together they produced 2 copies of the LEVEL LOG in 6 of 12 simulated requests (`FC/sim_stackv2_conflict.py`). [verified]

**E5. No edits to retained history between trims, in any placement, and nothing per-game in the system message** (MV2D:91-94; extend test 7 at MV2D:111 from delta mode to trim and both).
- Fable's refresh deletes the old pin from mid-history. That breaks the prefix without setting the trim flag, 18 times per r2 replay at about 4.7 s of prefill each. [verified count and prefill time; the slot effect is inferred]
- rellik13 pinned his solved-level rule at the end of the system prompt. In our code that would change `messages[0]` on every level clear. The flag check compares only `messages[1:]` (TA:6430), so the result is a full re-prefill with no flag. [inferred from code] Keep `<solved>` in the delta and the head card.

**E6. Rollback when a trim happened in the failed turn** (MV2D:93-94, test 8 at MV2D:111).
- Add a test case: a request fails after the trim and the card. The retried request shows the card exactly once, and the delta queue is not lost.
- This is the analogue of Fable's B4, which leaves the pin state set after a failure.

**E7. Read-outs and kill rule** (MV2D:114-119; also correct MV2M:75).
- Make stack-v2 the baseline. Compare against ledger-on-trim and control as well, not instead.
- Add four read-outs:
  - points lost to the action term (`FC/action_penalty.py`)
  - P(clear | k tokens on the level) and tokens per cleared level (levels.csv)
  - levels per 100 active game-minutes
  - `clock_confusion.py` matches, when E2 or arm 1 is on
- Write the keep/drop rule down before the run, for example: keep if the mean of 2 runs is at least stack-v2 + 3 and exact repeats are at most 6%; drop if it is at most stack-v2 − 1. [the thresholds are my suggestion]
- Correct MV2M:75. Actions are penalised, quadratically above the human baseline (game.py:381-416).

**E8. One per-turn block shared with `Salient:`** (MV2D:57 order, MV2D:84 cap).
- Fix the order: Salient, then seen-state, then deltas, then the bar fact.
- Set a combined cap of about 600 chars per opener.
- Remove duplicates: Salient's "objects that appeared during this level" against C's `MOUSE@<colour>` rows.
- `salient_facts.py` (attention-prompt branch) already leaves "use known controls first" to memory-v2's table and far-off colour changes to "the indicator-notes arm" (its docstring, lines 14-18). [verified]
- Keep Salient ungated (E1 applies only to memory-v2's loop facts).

**E9. Keep hashes, not grids** (MV2D:10, MV2D:51).
- AttemptTracker and ActionEffects store only hashes and counts. The `end` diff keeps 2 grids per level.
- Why: Fable keeps up to 40 before/after grid pairs per key per level for the whole game. In r2 one game reached 457 samples, about 33 MB per game by my estimate.

**E10. No invented turns and no advice in the card** (MV2D:74-75).
- Never append an assistant message the model didn't write; Fable's ACK at C:507 does exactly that.
- Never put "how strong runs work" lines in the card (C:509-516).
- Rewrite MEMORY_CARD_ADDENDUM as two descriptive sentences, as `SALIENT_FACTS_ADDENDUM` does. Example: "Lines in `<memory>` tagged [harness] are counted from the frames; lines tagged [note] are your own notes."

**E11. Keep what the design already gets right, and test it.**
- The clearing action is classed as `clear`, not as a no-op. Fable P:88-92 gets this wrong.
- Solved paths keep `M(r,c)` click coordinates. Fable B9 drops them.
- SeenStates hashes with border 4 (MV2D:19-22).

### attention-prompt

- Advice in the system prompt is static and cached, so rellik13's crowding-out explanation doesn't apply to it. But his "advice also made it doubt rules it had right" does (RW:63). So does the bar conflict: the advice version told the model to set the bar aside, which is the line rellik13 deleted (RW:84-85).
- The branch now follows both Fable's rule and rellik13's. Measure the facts version, and don't spend compute on the advice version (commit 1406ec5).

## 5. What NOT to copy

| Item (their source) | Why not | Evidence |
|---|---|---|
| The five "How strong runs work…" protocol lines and the "Noted: …" assistant turn the model never wrote (C:507-516) | This is advice plus a made-up assistant turn, against their own rule | rellik13's advice results (RW:59) [confounded]; our audit60k 44.10 vs 46.25 (arms.md:92) |
| Restoring the pin at the head after a trim by matching its content (P:219-231) | It collides with our ledger head merge and duplicates the log | 6 of 12 requests in `FC/sim_stackv2_conflict.py` [verified] |
| Refreshing by deleting the old pin from mid-history (P:156-169, 206-207) | Full re-prefill without the trim flag; the game keeps its slot meanwhile | 18 per r2 replay at about 4.7 s each [verified] |
| A journal that replaces history (their B005) | Cleared levels fell from 46 to 23 | HB p.78 [their claim] |
| Interpretive indicator text ("Legends, trays and counters often report progress", "A mark you earned may have been lost") | 4 of 10 sampled events weren't indicators; 34% of MOUSE events are at the click point | `FC/claims_analysis.out`, `FC/fp_samples/` |
| Negative claims and centroid "typical effect" lines ("no action of yours has moved it"; "SPACE: white cells shifted down 9 and left 22") | Results that hold only under conditions, written as general rules: the failure they diagnosed in their own journal | HB p.84; B8 examples [verified] |
| The budget alert as implemented (P:112-121, 141-151) | Fires about once per level (B1) and can be 5 frames stale (B2) | `FC/test_budget_attempts.py` [verified] |
| The digest's "last 10 actions" without click coordinates; logging the clearing action as a no-op (P:88-98) | Useless for click games, and it miscounts | B9 [verified output] |
| `_fl_log` writing `/kaggle/working/fable_layer_log.jsonl`, on by default | Writes files during the competition rerun | B12 [verified code] |
| Storing up to 40 grid pairs per key per level | Host RAM risk at 110 games | B-memory estimate [inferred] |
| Silent `log.debug` on every hook failure | A broken layer looks like a null result | P hooks [verified]. Count failures in the run report instead |
| The Python 3.12 venv fallback | Not needed with our pinned byod image | Layer pass §1 item 3 [inferred]; harmless |

## 6. Suggested arms.md updates (not applied here: arms.md is outside this task's write scope)

- **Not yet implemented, ranked:**
  1. `budget-truth`
  2. `mem-facts-gated` (memory-v2 with E1–E11)
  3. `mem-solved`
  4. `indicator-facts` (only after the 30-event precision gate)
  5. `persist-sandbox-vars`
  6. The copies-panel builder option
- **Update the attention-prompt entry (arms.md:135)** to the `Salient:` facts version: commits 9c3a40b..f43ca80, `ARC3_ATTENTION_AFTER_TOKENS`.
- **Note under the scoring guidance:** actions cost score through (human/actions)² (game.py:381-416). That costs 1.4–2.2 points in the control and ledger runs.

## Caveats

- **Fable layer effects are counterfactual.** The trajectories come from runs without the layer. The replay shows which text would have appeared, not its effect on play.
- **`clock_confusion.py` is a regex.** I read the first match per game for 3 runs (36 snippets). Most of them raise the timer idea as a hypothesis, and some reject it later. So "12 of 25 games" counts games where the idea was raised, not games where it was believed.
- **Token gates are only r2 and only approximate.** `requests.csv` has per-step tokens only for r2 among the full runs with boards, and a step's tokens are credited before its actions.
- **Every handbook and narrative figure is their own, from their Kaggle runs**, unless marked verified.
