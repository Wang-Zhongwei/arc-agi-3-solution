# Experiments

Kaggle notebook `api-signatures-and-frame-diff`, game `ls20-9607627b` (7 levels), 10 passes per arm, 230,000 generated tokens per pass. Configs and flags are in `ARC3-Inference/configs/api/` on `api-comparison-branch`. Every pass in every arm ends `gave_up` at the token cap, so the token budget binds.


| Arm | Code   | What changes for the agent                                                                                                                                                                                                                                                                                                                                            | Run                                                 | Score (mean ± sample sd) |
| --- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------ |
| v0  | `main` | Baseline ("original API"): frames expose `.ascii` and `.segmentation`; a diff image each turn.                                                                                                                                                                                                                                                                        | `jonathanwang2022/api-signatures-and-frame-diff` v3 | **30.06 ± 13.35**        |
| v1  | `main` | `ARC3_AUTO_FRAME_DIFF=1`: each turn's message adds an "Auto frame diff" object summary (about 2.2k chars on about 60% of turns), and the prompt calls it ground truth and asks the agent to consult it before acting.                                                                                                                                                 | `jonathanwang2022/api-signatures-and-frame-diff` v5 | **22.65 ± 10.39**        |
| v2  | branch | "Enhanced API", all flags off: the branch's always-on API changes (below).                                                                                                                                                                                                                                                                                            | `ticbudja/api-signatures-and-frame-diff` v1         | **22.79 ± 11.36**        |
| v3  | branch | v2 plus `ARC3_API_SIGNATURES=1` and `ARC3_HUD_DISPLAY=1`: a typed API reference (about 5.7k chars, roughly 1,400 prompt tokens per turn) listing every runtime object, `action()`'s return keys and `frame_diff`'s schema; `FrameView` is listed without `.ascii`. HUD display: once the agent defines a `hud()` helper, each later turn's message shows its reading. | `ticbudja/api-signatures-and-frame-diff` v2         | **19.40 ± 10.69**        |


**v2 is not "the branch with features off."** With every flag off, the branch (`game_over_feedbacks_and_hash_board_transform_summary`) still rewrites the runtime section of the system prompt:

- `.ascii` is dropped from the documented frame API and replaced by cropping (`frame[r0:r1, c0:c1]`, clipped components, crop-local IDs). `.ascii` still works; it is just not mentioned.
- Segmentation nodes gain `mirrored`, `form_hash`, `silhouette_hash`, `scale`, `bbox` and `parent`, and reflection changes the meaning of `hash` and `shape_hash`.
- `frame_diff` returns a new schema (`mirrored`, `scaled`, before/after IDs and bboxes), and history entries lose `.result`.
- The warning not to treat edge HUD/timer bars as clickable objects is replaced by advice to write a `hud()` helper.
- Prompts suggest stepping one action at a time when searching for a target state, and stopping on `level_completed`.

So v0 → v2 measures that whole package. It does not isolate any single feature.


|                                  | v0    | v1    | v2    | v3    |
| -------------------------------- | ----- | ----- | ----- | ----- |
| Levels cleared (10 passes)       | 39    | 35    | 36    | 31    |
| Deaths                           | 5     | 4     | 10    | 9     |
| Actions                          | 3,640 | 3,931 | 4,096 | 3,469 |
| Turns                            | 851   | 964   | 926   | 800   |
| Mean actions per `action()` call | 4.4   | 1.6   | 2.9   | 10.7  |
| No-op actions                    | 0.9%  | 0.9%  | 1.0%  | 0.9%  |


v3 pass 7 stopped early on `stop_requested` at action 120 with 147k of its 230k tokens used and 1 level (score 1.8); the cause is not in the logs. Every other pass in all four arms ran to the token cap. Without pass 7, v3 averages 21.36.

Mean actions spent on each level (passes that cleared it):


| Level | v0       | v1      | v2           | v3      |
| ----- | -------- | ------- | ------------ | ------- |
| L1    | 27 (10)  | 28 (10) | 40 (10)      | 30 (10) |
| L2    | 126 (10) | 111 (9) | **169** (10) | 113 (8) |
| L3    | 96 (10)  | 103 (9) | 85 (9)       | 116 (8) |
| L4    | 69 (7)   | 95 (6)  | 85 (7)       | 85 (5)  |
| L5    | 76 (2)   | 135 (1) | —            | —       |


Significance (unpaired permutation test on the 10 pass scores; with one game, the across-game test from [t-test guidance](../../rules/t-test.md) is not possible):


| Comparison                     | Δ score | p (two-sided) |
| ------------------------------ | ------- | ------------- |
| v1 − v0 (frame change summary) | −7.41   | 0.18          |
| v2 − v0 (enhanced API)         | −7.26   | 0.20          |
| v2 − v1                        | +0.14   | 0.98          |
| v3 − v2 (API signatures + HUD) | −3.39   | 0.50          |
| v3 − v0                        | −10.65  | 0.07          |


# Findings

1. **Original API vs enhanced API: the original scores higher.** Enhanced API must have been too distracting. It provides a more general tool for no place to use.
2. **Frame change summary in the user prompt: no benefit, and a −7.4 drop that is not significant (p = 0.18).** "No good at all" is too strong for n = 10; "no evidence it helps" is supported. That's probably the reason why Daniel turn that option off. 
3. **New errors are not the cause.** The tool-error rate is flat (8.4–9.5%) in every arm, and errors cost tool calls, not game moves. The new features add little:
  - v2 adds about one API-misuse error per pass.
  - v3 adds one real bug: the signature lists optional `action()` keys as always present, giving 16 `KeyError`s, each recovered. (to be fixed if we pick it up later. 

## Lessons learned

- Don't create unnecessary tools if agent is never gonna use them. 
- If agent understands the current tools well, don't fix it. .ascii -> frame[r0:r1, c0:c1]
- Look deeper into what agent really needs to level up. 
- Prioritize exposing accurate and complete information of the game instead of providing complicated tools. Agent can create them if they want to use. 

# Error analysis

Tool-call errors, classified, as a share of all tool calls:


| Error kind                                         | v0         | v1         | v2         | v3         |
| -------------------------------------------------- | ---------- | ---------- | ---------- | ---------- |
| All errors                                         | 8.4% (110) | 9.5% (138) | 9.2% (128) | 8.7% (107) |
| API misuse (harness objects used wrongly)          | 0.8% (10)  | 1.0% (15)  | 1.5% (21)  | 0.9% (11)  |
| Agent's own code bugs (NameError, IndexError, ...) | 5.6% (73)  | 4.7% (68)  | 4.4% (62)  | 6.1% (75)  |
| Harness refusals (repeat after a no-op)            | 1.2% (16)  | 2.8% (41)  | 1.6% (22)  | 0.8% (10)  |
| Sandbox limits (blocked imports, `open`)           | 0.8% (11)  | 1.0% (14)  | 1.6% (23)  | 0.9% (11)  |
| *of which: documented `action()` key missing*      | 0          | 0          | 1          | 16         |




**Small, visible API confusion in v2 (enhanced API), too small to explain the drop.** v2 adds about 11 API-misuse errors over 10 passes: crop syntax (`Use frame[r0:r1, c0:c1] with two slices`, 6), `len()` on the integer changed-cell count in animation steps (4), and subscripting `HistoryEntryView` (3). It also adds about 12 blocked `inspect`/`time`/`sys` imports, mostly the agent trying to introspect the new API. One pass built shape helpers on crops and got wrong answers because crops clip objects at the edge. Altogether that is about 1–2% of v2's tool calls and tokens, while v2 lost about 25% of the score.

**A change in how v2 looks at the board.** v2 reads `.ascii` in 20% of tool calls, against 45% in v0 and 48% in v1, and uses segmentation (30%) and crops instead. This is consistent with v2's slower level 1 (40 actions against 27) and level 2 (169 against 126), but this data can't show cause. (Isn't this supposed to be 0?)

**v1 (frame change summary) acts in smaller steps rather than making more errors.** With the auto frame diff it averages 1.6 actions per `action()` call (v0: 4.4), takes 13% more turns for the same tokens, and hits the no-op repeat guard 2.6 times as often (41 against 16). Refused actions are not executed, so they waste tokens, not moves. The summary itself is compact and mostly accurate. It does list noise such as `resized: color G 892->892px`, a "resize" with no change in size.

**Where v2 actually loses: level 2.** Seven of v2's 10 deaths are on level 2 (v0: 4 of 5). The transcripts show game-reasoning failures in the move-budget-and-lives mechanic:

- Pass 8: the agent's own BFS checked the 21-move budget after applying a refill rather than before, and the plan ran out of moves at action 133.
- Pass 5: it acted on a wrong guess ("any press unlocks the door") and ran out of budget testing it at action 152.
- Pass 1 spent 122 actions on level 1 working out the rule (match the panel shape at the marker, then enter the box), with only a few, quickly recovered tool errors.

v0's prompt warned about HUD/timer edge bars and v2's does not, which may matter on this budget-heavy level. That is a hypothesis, not something these runs show.

**v3 (API signatures + HUD) introduces one real bug: the signature promises keys that `action()` does not always return.** The typed reference lists `action()`'s return as one flat dict in which every key looks present. In practice `executed_count` is missing from single-action results, `stop_reason` appears only when a batch stopped early, and `animation` appears only when something animated. The agent trusted the signature and indexed them directly, giving 16 `KeyError`s in 9 of 10 passes (`executed_count` 7, `animation` 5, `stop_reason` 4), against 0 in v0/v1 and 1 in v2. Fix: mark these keys optional in the signature (`NotRequired[...]`, or "present only when ...") or always return them. The cost is small (1.3% of tool calls, each recovered with `.get()`), so this bug does not explain the score.

**v3 confusion: the typed `FrameView` drops `.ascii` from sight entirely.** v3 uses `.ascii` in 0% of tool calls (v2: 20%, v0: 45%). Instead it reads colours through one-cell crops (`f[r:r+1, c:c+1].segmentation`) twice as often as v2 (2.2% of calls against 0.9%). Pass 7's last turn builds a `cc(r, c)` helper that crops about 150 single cells to print a 14 × 11 patch, rebuilding the old text view the expensive way. The reference also costs about 1,400 extra prompt tokens per turn, and v3 takes the fewest turns (800) for the same generated-token budget.

**v3's HUD display barely helps here.** The agent defined `hud()` in time for its readings to appear on 217 of 800 turns (27%). The readings are whatever the agent's helper returns, often raw bounding boxes such as `{"Y":[61,13,62,54],"R":[61,62,62,63]}` instead of moves or lives left, so they don't spare the agent the budget arithmetic it gets wrong on level 2.

**v3 batches far more aggressively.** It averages 10.7 actions per `action()` call (v2: 2.9, v0: 4.4), hits the no-op guard least (10), and still dies 9 times (7 on level 2). Longer blind batches are consistent with more mid-plan budget deaths, but these runs can't separate that from the level-2 reasoning failures seen in v2.

**Verdict.** Error counts don't explain the drops. v1 and v2 introduce no meaningful new errors; v3 introduces one real but cheap bug (optional `action()` keys documented as always present). All three lose through behaviour: v2 and v3 read the board through segmentation and crops instead of `.ascii` and die more on level 2; v1 moves in smaller steps; v3 batches long and has one pass cut short. Only v3 − v0 (−10.7, p = 0.07) comes close to significance at n = 10.

## Reproduce

From the repo root, with the pulled runs under `ARC3-Inference/runs/kaggle/` (or set `ARC3_RUNS_DIR`):

```bash
python3 docs/games/ls20/scripts/analyze.py      # per-pass turns, actions, deaths, levels, tool errors; writes summary.json
python3 docs/games/ls20/scripts/levels.py       # actions per level, deaths, permutation tests (needs summary.json)
python3 docs/games/ls20/scripts/classify.py     # error classes
python3 docs/games/ls20/scripts/peek.py ticbudja/api-signatures-and-frame-diff/v01 8 100 140   # read a pass's turns
```

To include v3, pass its run path relative to `runs/kaggle` as the first argument to `analyze.py`, `levels.py` and `classify.py`: `ticbudja/api-signatures-and-frame-diff/v02`.