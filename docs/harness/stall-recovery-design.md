# Recovering useful history and moving from thought to action

Status: 2026-10-11. Base: `allocation-vc-one-wave` at `d49a940` (PR #30 reverted).
The [view tool](#view-tool-implemented) is implemented behind `ARC3_VIEW_TOOL`; its
[test config](../../ARC3-Inference/configs/view-tool/README.md) uses a 120-minute public
runtime. The other reminders below are proposals and are not implemented.

When conversation is trimmed, tell the agent which part of `history` left it and how to view it. Also trigger a review when level 1 remains uncleared, even though no previous level-up exists. If the review still produces no level-up, repeat the cue at the next actual history trim. When thinking per executed action becomes high, encourage a useful action. Tie the reminders to a concrete winning condition and a plan or search toward it.

The change is an **addition to the regular user prompt**. Keep the normal state, action results, and event guidance, then append only the relevant reminder. Ordinary requests receive no extra text. Expose the history-viewing capability once in the tool reference.

## Evidence and initial threshold

The audit covers the newer 15 games in 49 exact archived Kaggle versions: 735 game attempts, with mixed prompts, models, schedules, and budgets. A stall means an uncleared final observed level with at least 20 analyzer invocations after the analyzer step containing the latest clear. Yield resumptions can be separate invocations; this is a lack-of-progress flag, not proof of an action loop.

Among the 363 flagged stalls, estimated thinking totals 38,947,611 tokens for 37,991 executed non-RESET actions: **1,025 thinking tokens per action** pooled. The per-attempt median is 1,168 and upper quartile is 1,858. Start with **2,000 thinking tokens per action after at least 10 executed actions**. Applied retrospectively to flagged tails, this selects 71/363 stalled attempts (19.6%). It is a provisional, selective reminder threshold; these observations do not establish an optimal threshold or a causal benefit.

Older runtime history is accessed during 336/363 stalls, but explicit successful review of previous-clear boards appears in only 41/309 stalls eligible for a prior clear (13.3%). In the seven versions with request logs, 84/105 attempts show definite old-message eviction; only 6/84 subsequently show direct successful boundary-board review. Broader prior-level/boundary review appears in 34/84. These conservative code detectors miss aliases, manual indexes, cached snapshots, and mental recall. The gap motivates a specific range cue rather than another generic instruction to inspect history.

| Game | Stalled attempts / 49 | Mean thinking tokens per action on stalled level | Previous-clear board review / eligible stalls |
| --- | ---: | ---: | ---: |
| tn36 | 22 | 987 | 4/22 |
| lf52 | 32 | 1,101 | 7/31 |
| cn04 | 26 | 1,510 | 7/26 |
| bp35 | 28 | 2,345 | 4/25 |
| wa30 | 23 | 402 | 0/23 |
| sp80 | 24 | 1,653 | 1/24 |
| m0r0 | 15 | 1,257 | 5/11 |
| ka59 | 23 | 553 | 2/21 |
| sk48 | 34 | 1,103 | 3/14 |
| dc22 | 29 | 998 | 1/23 |
| cd82 | 9 | 918 | 0/1 |
| g50t | 26 | 817 | 2/16 |
| ls20 | 23 | 500 | 3/23 |
| s5i5 | 21 | 2,547 | 2/21 |
| su15 | 28 | 1,101 | 0/28 |

The thinking column is total estimated thinking divided by total executed non-RESET actions in that game's flagged tails. It includes inspection-only and yielded reasoning. It is an action-weighted average, not tokens per turn or an average of level totals. Transcript estimates use the Qwen3.8 tokenizer; checks against logged usage on 255 calls agreed within 99.77–99.97%. Archive tails start after the progress-producing analyzer step; the proposed runtime counter below resets at the actual clear, so the two intervals are close but not identical.

The prompt should encourage *informative* action, not merely increase the denominator. In `jonathanwang2022/animations-in-context/v01`, dc22 spent 424 actions in eight long LEFT/RIGHT alternations, each cycling between two visible states after removing the top/bottom four rows. Conversely, `ticbudja/thinking-budget-priority-four-waves/v03` solved m0r0 with a search over 288,283 states, a 38-action plan, and execution checks. In the same notebook's v01 sp80 run, all four inferred targets were simultaneously filled in animation frames 24–31 without completion: the winning condition needed revision. These cases support a direction of **specify the goal, test the model, search, execute, verify**.

## View tool (implemented)

Behind `ARC3_VIEW_TOOL` (off by default). It extends `view_frames` from archived commit `52c97a4` with crops and renames it `view`:

```python
view(120)                             # one full board
view(120, bbox=(8, 12, 31, 43))       # one cropped board
view(range(117, 121))                 # several boards; stop is exclusive
view(range(117, 121), bbox=(8, 12, 31, 43))
```

A bbox is `(min_row, min_col, max_row, max_col)`, inclusive, matching segmentation coordinates. Labels keep the history index, action, level, step, completion flags, and crop. Each image uses 64 visual tokens; crops get a larger integer scale within that budget, and blue margins pad shapes that do not fill the aligned canvas. Limits are four images per Python call and eight per analyzer invocation. Viewing spends no game actions.

**Trim note.** The agent is not told about its last level-up or game over. Each opener records the latest history index when it was created. Once the oldest retained opener starts at `history[s]` with `s > 0`, a note follows the system prompt:

> Earlier conversation has been trimmed. It covered history[0:s], which remains in `history`. To review key moments there, such as level-ups or game overs, use `view` with a bbox.

The note carries no reasoning or images. It is built per request rather than stored, counts toward the context estimate, and changes only when trimming moves the start, so prefix caching is unaffected between trims. It replaces the latest-clear cue proposed earlier and the transition reminders of PR #30.

`history[k - 1]` is the board before a winning action at `k`; `history[k]` is normally the next level's initial board, not a saved solved board. Runtime history persists after chat trimming; fatal actions and automatic RESET have separate entries.

## Further proposals (not implemented)

**Level-1 stall trigger:** after 20 completed analyzer invocations with accepted solver work and no level clear, queue the initial level-1 review cue once for that uncleared level; later cues follow the trim recurrence rule. Count each invocation once if it contains at least one accepted solver response. Include inspection-only and yield continuations so a thinking-only stall can trigger it; scheduler waits, invocations with only unsent/failed requests, and summary generation add zero. Death and RESET keep this count. This is a lack-of-clear trigger, not a claim that every invocation was unproductive. It operates independently of trimming and the thinking/action threshold.

Without a prior clear, point to `history[0:min(4, len(history))]`: the initial board and earliest action results. It can contain only the initial frame when no action has executed; `view(0)` remains valid. Label it as opening evidence, not a previous success. Compare that evidence with the current state and reconsider the model or winning condition. Cancel a pending level-1 cue if the level clears before delivery.

**Repeat at later trims until a level-up:** a delivered recovery reminder starts a review round for the current level. If the resulting solver work still produces no level-up, each subsequent actual history trim queues another similar reminder. Point to the range named by the trim note, or to the opening range on level 1. Ask the agent to compare that evidence with observations since the previous review and revise its model, winning condition, or plan. Absence of a level-up does not establish that the preceding work was useless.

An actual trim removes old conversation exchanges or replaces them with a summary; a trim-helper call with no eviction, image-window update, or redundant-control-only pruning does not count. Coalesce all cuts and overflow retries while preparing one inference request into one cue. After an accepted response, a later real trim can begin the next round. Failed or unsent requests retain the pending cue. A successful clear ends the round and cancels pending follow-ups; death and RESET preserve it.

## Thinking per action

For the current level, maintain `T` = reasoning tokens from all solver responses and `A` = actually executed non-RESET actions. The metric is `T / A`. Accumulate inspection-only responses and yield continuations; count each executed batch primitive, including executed no-ops. Refused/skipped actions add zero to `A`. Exclude summary-generation tokens. Death, RESET, chat trimming, and scheduler handover keep the current-level counters; a successful clear resets them for the new level. Charge a response's reasoning and its winning action to the old level before resetting. Count actual reasoning expenditure once per response, even if a later tool failure rolls back other bookkeeping.

Use exact usage through `_usage_reasoning_tokens`; `_last_call_reasoning_tokens` alone misses earlier inspection and continuation calls. If any solver response lacks reasoning accounting, treat the interval as unknown and omit the numeric warning rather than substitute generated tokens or undercount it.

At the next safe model-request boundary, append the initial action reminder if `A ≥ 10` and `T / A > 2,000`. Evaluate this independently of trimming, and send the standalone warning at most once per level. At each later trim-triggered review, re-evaluate the same current-level average and include an updated numeric line only if the threshold still holds. The numeric reminder waits for ten actions while inspection-only thinking accumulates in `T`; the separate level-1 stall cue can fire without any action. It advises the agent; it neither forces actions nor reduces its thinking budget. The retrospective 71/363 count does not predict online trigger frequency, and a low ratio does not rule out an action loop.

## Extra user-prompt text

For an ordinary turn, the current regular guidance is:

> Inspect the evidence from the newest history, and search or score candidate actions or short sequences against the current goal as you currently understand it. Separate gameplay-object changes from HUD changes.

Keep that regular prompt and append these lines only when their events occur. The numbers below are illustrative substitutions.

**Thinking/action threshold crossed:**

> You have averaged 2,450 thinking tokens per executed action on this level (29,400 tokens / 12 actions). Prefer a short, verified action sequence now; if a rule is uncertain, choose a small action that tests it.

**Level 1 remains uncleared:**

> Level 1 remains uncleared after 20 analyzer invocations. Revisit its starting layout and earliest action effects with view(range(0, 4)), and compare them with current_frame.

Use the available stop index in this example; if only the initial frame exists, suggest `view(0)`. A crop is optional in every history cue, for example `view(0, bbox=(8, 12, 31, 43))` when those bounds match the region of interest.

**Next trim with no level-up since the previous review:**

> This level remains uncleared since the previous review, and more conversation history is being trimmed. Revisit history[117:121] with view(range(117, 121)), compare it with current_frame and observations since that review, and reconsider your winning condition or action plan.

On level 1, substitute the opening-evidence range. Include the refreshed thinking/action line only when its threshold still holds.

**Append once per reminder request:**

> Make the winning condition explicit. Use the mechanics supported by your observations to plan or search for a reachable state that satisfies it. Execute a short prefix and check the result; revise the model or goal when predictions fail.

If multiple events occur in the same request, combine their lines with one planning sentence. This wording encourages action toward a goal while leaving the agent free to choose its state representation, search method, and probes.

## Integration and evaluation

Attach the extra text after choosing the actual normal/short/state-only opener, and after a complete assistant/tool group on within-turn continuations. Adding it only inside `_build_user_prompt` would miss yield resumptions. Keep pending events and the active review round outside removable chat messages, account for their input tokens in the final trim estimate, and ensure the selected cue survives that request. Cover token trimming, assistant-turn retention, summary replacement, control pruning that removes the source exchange, and server-overflow fallback. Preserve atomic tool exchanges and strip private provenance from wire messages. Mark a reminder delivered after an accepted inference response; failed or unsent requests leave it pending. Track the level, last delivered review, and last handled trim episode so repeated trim passes cannot duplicate a cue.

Implementation touches the prompt constants, tool sandbox, and request/trim bookkeeping in [tool_agent.py](../../ARC3-Inference/inference/agent/tool_agent.py), plus persistent counter/event state. Reuse the archived tool and provenance approach without restoring its whole reasoning/image reminder feature. Put effective tool limits and the opt-in reminder/threshold settings in [CONFIGURATION.md](../../ARC3-Inference/CONFIGURATION.md) when implemented.

Before an experiment, check initial delivery, another cue at the next actual trim without a clear, cancellation after a clear, and one cue per trim episode despite repeated cuts/retries. Also check threshold crossings and refreshed numeric lines, level-1 stalls with and without actions, tool-call pairing, yield/overflow retries, level-clear versus RESET counter behavior, missing usage, and single-index/range/bbox labels and caps. Compare baseline against the history cues + tool + shared goal/search sentence first, then add the independent thinking/action reminder as a second arm. Keep model, per-game budgets, scheduler, and concurrency fixed. Evaluate all 15 games with repeated passes: levels cleared/wins and action efficiency first; thinking/action, prior-clear review, and recovery after the cue second. A lower ratio alone is not success. Follow the [A/B comparison guide](../rules/t-test.md) and report per-game differences before any merge.

## Evidence files and source references

The [attempt metrics](data/stall-recovery-attempts.csv) retain the exact owner/notebook/version and stall numerator/denominator for all 735 attempts. The [game summary](data/stall-recovery-by-game.csv) retains unrounded means and history-review counts. These are derived analysis tables; raw run artifacts remain outside Git under `ARC3-Inference/runs/kaggle/<owner>/<notebook>/vNN/`.

The stall definition is sensitive to the invocation cutoff: 10 flags 482/735 attempts; 40 flags 141/735. Definite trimming is established only for the seven request-logged versions, by removed older tool-call IDs while the newest previous ID remains. History-review counts use model-issued Python code, exclude duplicate diagnostics and detected uncaught errors, and are conservative detections rather than exhaustive recall measures. This mixed archive supports the design rationale, not a controlled model comparison.

Reviewed case pointers, relative to the exact run folder:

- dc22, `jonathanwang2022/animations-in-context/v01`: `artifacts/dc22-fdcac232_p0_events.jsonl:520–629` (longest cycle); `solver_analysis/dc22-fdcac232_p0.html:23411` (useful search also occurs).
- m0r0, `ticbudja/thinking-budget-priority-four-waves/v03`: `solver_analysis/m0r0-492f87ba_p0.html:23083` (search), `:23536` (execution).
- sp80, `ticbudja/thinking-budget-priority-four-waves/v01`: `solver_analysis/sp80-589a99af_p0.html:25892` (animation review), `:26052` (filled inferred targets without a clear).

Current contracts: [history and regular prompts](../../ARC3-Inference/inference/agent/prompts.py), [runtime views](../../ARC3-Inference/inference/agent/python_tool_sandbox.py), [history persistence](../../ARC3-Inference/inference/agent/runtime_state.py). Archived references: `52c97a4184ad81bc0cb13e5d1ce72d202e597a6d` (`view_frames`), `0dc3b80f4e6ee2242137ee7e341e64e4fd7fa79e` (source-exchange provenance). The existing [thinking/action analysis](../reasoning/thinking-vs-actions.md) also distinguishes reasoning accounting from calls that emit no actions.
