# fable-layer-vs-ours: read:ours+rellik

Workflow run `wf_c4d3dccd-2d7`, phase Read. Compare leoprovorov Fable & Astra notebook (handbook + Fable layer on Franzen) with our stack: read, measure claims on our runs, compare, skeptic-verify

---

## Fable layer vs our stack-v2: the comparison from our side

Our run of record is stack-v2 under the competition config.

**Path abbreviations**
- `SV2` = `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/stack-v2/ARC3-Inference/`
- `ATT` = the same path under `wt/attention-prompt/ARC3-Inference/`
- `TA` = `inference/agent/tool_agent.py`, `PR` = `prompts.py`, `ML` = `memory_ledger.py`, `SOL` = `inference/framework/solver.py`
- `C41` / `C42` = their notebook cells 41 / 42. Line numbers point into the copies at `.../scratchpad/fable-compare/cell41.txt` and `cell42.txt`.
- `NB` = narrative.md cell number, `RW` = rellik13 `WRITEUP.md`, `HB` = handbook PDF page.
- `MV2D` = `memory-v2-design.md`, `MV2M` = `memory-v2-measure.md` (both in the scratchpad).

### 0. Shared base
- **Same base config.** Their F001 notebook's config cell 4 and harness-patch cell 2 are byte-identical to our `kaggle/template.ipynb` cells 4 and 2 [verified, diffed]. Their arm is therefore our control plus cell 6 ("0b", the Fable layer). Our arm is our control plus `ARC3_LEDGER_ON_TRIM=1` and the #4/#5 fixes (`kaggle/arms.json`).
- **Their patch would apply to our code.** All five Fable anchors (C42 EDITS) match exactly once in stack-v2, attention-prompt and main `tool_agent.py` [verified, `fable-compare/anchor_check.py`].
- **Possible clash if both layers are stacked** [inferred from code].
  - Their `_fl_restore_pinned` and `_fl_remove_pinned` (C42:156-232) find the pinned log by exact content match.
  - Our `_apply_due_ledger` (SV2 TA:6382-6393, 6826) prefixes `<memory>` onto the first user message after a trim. That message would be their restored pin.
  - Result: duplicated or unremovable LEVEL LOGs.

### 1. Mechanism-by-mechanism table

| Fable mechanism (source) | What stack-v2 already does, as configured | memory-v2 design adds | attention-prompt adds |
|---|---|---|---|
| **Indicator notes**: small changed patches (2–16 cells, at most 6×6) at least 8 cells from the main change; marked first, lost or again; tied to an action index; up to 2 per turn after 15k tokens on a level (C41:151-189, 328-391, 398-417; C42:123-154) | **Partial.** (a) Diff image each turn: changed cells in true colour, the rest dimmed to navy (TA:4066-4078; `ARC3_DIFF_IMAGE=1`, cell 4) [verified]. It covers the whole turn, is not tied to a single action, keeps no history and has no lost/again logic. (b) `frame_diff()` exists but the model must call it (PR:250-259) [verified]. (c) The text frame diff is off: `ARC3_AUTO_FRAME_DIFF` defaults False (TA:332) and is not set in cell 4 [verified]. Nothing marks a patch as "far from the main change". | Nothing that detects indicators. The seen-board note and the effect classes (MV2D:17-18, 50-58) work on whole-board hashes and do not look for small patches [verified, design only]. | A rule only, asking the model to look: "the cells that changed color show what the action touched…" (ATT PR:143-162) [verified]. No computation. |
| **Per-key counts in the log**: pressed, changed the board, did nothing, first no-op, typical shift, MOUSE targets, keys not yet tried (C41:419-472) | **Per action, never totalled.** The action echo prints `[action] X -> effect / NO-OP / GAME OVER / LEVEL COMPLETED` (SOL:132, 186-230; `ARC3_ACTION_ECHO=1`). The batch no-op stop adds a per-action trace to the turn message (TA:4943-4969) [verified]. No-op means the same 4-cell interior border in both systems (SOL:119; C41:26, 53) [verified]. No per-key tally and no shift signature. | **Equivalent and stronger** (C, MV2D:36-49): per key per class (noop, hud, shift, change, death, clear) for this level and the whole game; MOUSE grouped by clicked colour; untested keys listed; deaths tied to boards. Shown as a full card at trims plus every-turn deltas [verified, design only]. | Rule 5: "make progress with the actions whose effects you know" (ATT PR:160-161). Advice that assumes the model already keeps this tally. |
| **Indicator history in the log** (C41:474-486) | None [verified]. | None [verified]. | None. |
| **UNDO count**: "every press counts as one action" (C41:526-527) | None. UNDO is exposed, but the addendum never says it costs an action (PR:126-129) [verified]. **Barely matters for us:** UNDO was 2 of 5,654 actions in r2, used in 2 of 25 games [verified, `fable-compare/undo_count.py`]. Their sk48 copy 9 pressed UNDO 127 times in 317 actions (NB 33) [their claim, unverified]. | None. | None. |
| **Step budget**: detect the bar from frames, estimate actions left, alert at ≤25% of cells or ≤12 actions (C41:192-269; C42:112-121, 141-151) | **Text rules only. No detector.** One system line says "watch for a bar… count remaining budget into your plans" (PR:247). The game-over BAR RULE applies only after a death (TA:5031-5085). `gameplay_changed` ignores the 4-cell edge (PR:140-149) [verified]. **Against it:** our shipped system prompt still says "If you identify such a bar, do not get distracted by it…" (PR:105, present in the rendered control prompt) [verified]. That is the line rellik13 deleted as a fix (RW:83-84). | None in the design; only the death count and the hazard lines (MV2D:62). By our own count, 11% of 54 hand-labelled deaths were unintended budget run-outs (MV2M:79-84) [verified]. | **Opposite direction**: "set aside changes confined to a timer or remaining-steps bar at the edge" (ATT PR:156-157) [verified]. |
| **Digest of cleared levels**: actions used, shift signatures, last 10 actions (C41:488-505) | **Partial, written by the model.** Ledger `<level_note>` plus `note_think` fragments folded in at the level change (ML:216-250; TA:3912-3916), shown only after a trim (TA:6375-6393). Measured: 87% of level notes are cut-off thinking fragments; 61% start with filler (MV2M:36-38) [verified]. The level-start prompt is advice text (PR:35-48). | **Equivalent and stronger** (A, MV2D:24-35): `<solved>` = loop-free winning path, spent, deaths, end-state diff, the "how" sentence; never evicted; at most 165 chars per level [verified, design only]. | Rule 4: test new elements first (ATT PR:158-159). Advice. |
| **Five protocol lines inside the log** (C41:509-516; 174 tokens) | Already covered by Franzen's own advice text: VISUAL_GAME (PR:99-112, 441 tokens), PYTHON_TAIL search advice (PR:232-249), the game-over "Do NOT re-submit…" text (TA:5099-5106) [verified]. Step-verification hint is off. | Asks for descriptive wording with no "do not" (MV2D:55, 74-75). | **Same kind as these lines** (advice); static, 283 tokens. |
| **Gating by tokens on the level**: notes from 15k, log from 30k, refresh every +40k; levels solved fast get nothing (C42:134, 176-186) | `_tokens_at_level_start` exists (TA:3911) but only the scheduler reads it (TA:2278-2318, 6466-6478). The ledger is gated by trims, not tokens [verified]. **What the gates would touch in our runs:** 64–71% of levels reach 15k, 41–49% reach 30k, 13–19% reach 70k; 40–47% of all generated tokens are spent after a level passes 30k; levels past 30k have a median of 1 trim (6 full runs) [verified, `fable-compare/gate_share.py` on `levels.csv`]. sk48 level 1: cleared in 5 of 11 of our full runs, at 29k–128k tokens [verified]. | None. Deltas go into every turn message; the seen-board note appears on about 19% of turns (MV2D:57-58, 84). | Always on. |
| **Placement and cache**: the log is appended as a user/assistant pair (suffix); put back at the head after a trim, after the eviction check, so no slot handover. Refresh deletes the old pair mid-history (C42:171-217, 219-231, 271-275) | **Same rule as ledger-on-trim**: shown only after a trim, which already invalidated the prefix; added after `_trim_messages_for_context` has called `_note_history_evicted`, so no extra handover (TA:128-136, 6375-6393, 6826-6827) [verified]. Ours costs no extra cache; their refresh breaks the prefix once per 40k tokens (NB 30) [their claim, consistent with C42:207]. | "both" mode: a head card at each trim plus deltas only in the new suffix, so it is cache-safe (MV2D:76-94). | Lives in the static system prefix, shared and cached [verified]. |
| **Never remove history** (NB 16, 30) | Same base: trims drop from the front with hysteresis to about 58k (template cell 4); thinking is kept. Summaries are off and do not replace history (TA:1036, 1156). The ledger edits the head message and removes nothing [verified]. | Keeps this (MV2D:88-90). | n/a |
| **Facts, not advice** (NB 16) | **Mixed.** Ledger content is model-written plus harness thinking fragments; only 2.7–3.9 tags per 100 calls (MV2M:10-27). The system prompt is about 4.1k tokens, mostly Franzen advice [verified, rendered]. **Their layer is mixed too:** protocol lines plus advice clauses inside the notes ("note which action did this", "compare what you did…", C41:406, 416) [verified]. | Moves to harness-made facts; model tags shrink to "beliefs" (MV2D:2-5, 59-75). | **Pure advice.** |

**Sizes, measured with the scratchpad `tokenizer.json`** [verified, `fable-compare/sizes.py`, `sysprompt_size.py`]:

| Item | Tokens |
|---|---|
| System prompt: control | 4,134 |
| System prompt: stack-v2 | 4,361 (+227) |
| System prompt: attention-prompt | 4,417 (+283) |
| System prompt: stack-v2 + attention | 4,644 |
| Fable LEVEL LOG, synthetic (5 keys, 3 events, 2 digests) | 647 (they claim 0.6–1k, NB 30) |
| One Fable indicator note | 59 |
| Our audit60k prompt | 377 |

### 2. rellik13 citations, checked against RW

| Fable claim (NB 16, HB 25/29/39) | What RW says | Verdict |
|---|---|---|
| Advice cut the score from 6.42 to 4.16 | RW:59 "Adding text to every prompt dropped the score from 6.42 to 4.16." | The number is right, but RW says "text", not specifically advice [verified]. Content is not given; RW:140-141 lists the failures as "advice or summaries… including per-action 'what changed' notes and a level-start comparison block". **Size is not stated anywhere.** Repeated in every turn message [inferred from RW:61 "every repeated line… pushed out… recent turns"]. Context was about 37k (RW:60). In his base, 34.2% of every prompt was already duplicated instruction lines (`rellik13-milestone2/cells/21_…py:41-44`) [verified]. The baseline 6.42 is the top of his repeat triple 5.02 / 5.50 / 6.42 (RW:17) [inferred to be the same notebook]. |
| Advice cut the score from 13.40 to 9.98 | RW:59 "Later, a level-start advice block dropped it from 13.40 to 9.98." | Shown once per level, not every turn [verified]. **Contrary evidence in RW itself:** RW:18 "A notebook that scored 37 on my 25-game practice run got 9.98 on the leaderboard; one that scored 32 in practice got 13.40" [verified quote]. That reads as the same pair [inferred], so offline the advice block was +5. It may also be the "level-start comparison block" of RW:141, which would make it facts, not advice [inferred, ambiguous]. |
| +8.0 from keeping more history | RW:8-10, 35 (14.49 → 22.53 = +8.04) | **Confounded** [verified]: FP8 KV, trims 36,864→26,624 changed to 57,344→45,056, context 49k→70k, **concurrency 15→16** (RW:98-101), "decoded faster than with BF16" (RW:124). One LB submission. His attribution "so the effect comes from memory" (RW:103) is [their claim]. |
| +2.8 from a rule pinned from the previous level | RW:69-71 "the rule the agent wrote for a cleared level stays pinned… The only text addition that helped (+2.8)…" (7.14 → 9.96) | Number right [verified]. But the code (`cells/42_…solved_level_memory.py:1-24`) pins **harness facts** (level, action count, last 30 winning actions, about 900 chars of the clearing turn's reasoning) **plus** a model-written "Rule for level N" (at most 600 chars), at the end of the **system** prompt. Fable's paraphrase drops the facts half. It is closest to their `digest_lines` and to our memory-v2 (A). |
| Also relevant | RW:63 "Advice also made it doubt rules it had right." RW:145 "A ledger for proven and refuted assumptions. The agent never used it." RW:83-84 the deleted bar line. | [verified quotes] |

**Is our attention-prompt the same kind of change?**
- **Same by content**: general behavioural advice, not computed facts [verified].
- **Different by placement and cost** [verified]:
  - It is static system-prompt text, added once and cached, not repeated in history.
  - +283 tokens is about 0.5% of our roughly 59k window after a trim, against about 37k (RW:60) in his setup.
- **Consequence** [inferred]:
  - The crowding-out explanation does not apply to it.
  - The "doubt rules it had right" risk (RW:63) does apply.
  - So does the conflict on the step bar.
- Handbook p.25 also says that rewriting the system instruction from strong-agent lessons has "no data in favour… data against it", and that "a run of the baseline system with two such prompts also changed nothing" [their claim, unverified].

### 3. Our own evidence on advice-style arms (`arc-agi-v4/arms.md`)

**Negative or null**
- **audit60k** (arms.md:92): 44.10 vs control 46.25, "Fired 30 times, no gain". This is a 377-token advice block that fires every 60k tokens on a stalled level, our closest analogue to Fable's token-gated protocol [verified: summary.txt 44.10; 30 firings found in transcripts; prompt from franzen `69f6ced`]. The gap is inside the 3-point noise band (arms.md:20).
- **Franzen:** a stronger step-verification hint gave "no gain" (arms.md:78; Franzen WRITEUP:303). Summaries gave no gain (arms.md:75). The world-model sections were worth more off than on (arms.md:47).
- **Franzen's own docstring:** the per-turn `gameplay_changed` line "scored 2.30 on competition where the run without it scored 4.05", with the guard wording changed in the same run (TA:2455-2459; SOL:121-127) [verified quote; confounded].
- **Old harness** (vs 17.743 unless noted):

| Arm | Result (arms.md line) |
|---|---|
| Scoring guide | 11.12 (:103) |
| Thinking guide | dropped; 0/28 Predict lines (:110) |
| Spend-actions | 10.921 vs 13.645 (:115) |
| Labeled grid + diff image | 12.343 (:102) |
| "harness owns facts" v9 | 2 levels in total (:117) [our log; details not re-found] |

**Positive**
- **Franzen's level-transfer guidance** (level-start advice) is part of the bundle that "produced a large improvement", though not isolated (arms.md:43-45; Franzen WRITEUP:236-250, 266). That runs against rellik13's 13.40 → 9.98 result.
- **ledger-on-trim** (model memory): 52.56 and 49.14 vs 46.25 (arms.md:52-55). Our data card says the gain "is not explained by trim recovery" (MV2M:129).
- **Old-harness memory ledger:** 13.645 vs 9.856 (arms.md:61).
- **Per-turn fact blocks** (frame-diff text, numpy perception): 2 of 2 on GB10 only, never run on Kaggle (arms.md:62-63).
- **attention-prompt** has not been run (arms.md:135).
- **Harness-flagged new elements**, the facts version of attention rules 1 and 4, is backlog only (arms.md:197).

**Side flag** [verified]: `MV2M:75` says "actions are not penalised". The audit60k summary shows efficiency-scaled scores (bp35 0.43 with 1 of 9 levels, sb26 93.82 with 8 of 8). Actions do count once a level takes more than the human baseline.

Scripts and extracts are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/fable-compare/`: `cell41.txt`, `cell42.txt`, `handbook.txt`, `anchor_check.py`, `gate_share.py`, `undo_count.py`, `sizes.py`, `sysprompt_size.py`, `sys_*.txt` (rendered system prompts), `audit_prompts.py`.
