# Early-level recovery

When an agent spends a large budget on the first mechanics or objective, it can keep extending a search built on a wrong assumption. This option adds a one-time progress check to the user prompt. It does not diagnose failure from token count alone: some successful levels also need substantial work.

The branch starts from local `allocation-vc-one-wave` at `1d35483` and includes the latest allocation baseline at `7ce6732`, including [PR #27](https://github.com/Wang-Zhongwei/arc-agi-3-solution/pull/27). Existing harness prompts, tools and scheduling remain the baseline when the new option is off.

## Trigger

- `ARC3_EARLY_LEVEL_RECOVERY_TOKENS=0` disables it; `32768` enables the initial treatment.
- Count accepted generated tokens since the current level started: thinking plus response/tool-call text, including rolling summaries. Input and image tokens are excluded.
- Only levels 1 and 2 are eligible. Deliver at most once on each level in a game, at the next user opener after reaching the threshold.
- A real level transition resets the count. Deaths, RESET, yields and history eviction preserve it. A new game resets delivery markers too.
- Rendering a prompt does not consume it. A successful response with committed history does; failed requests with rolled-back history and stops before a response do not.

32,768 is a starting hypothesis, not an optimized cutoff. It allows substantial exploration before intervening and measures actual model output rather than counting cheap inspection calls or repeated input context. Limiting the intervention to two reminders avoids repeatedly interrupting legitimate planning. Levels 1–2 target foundational mistakes; later levels often require longer searches using mechanics already learned.

## Prompt

> Progress check: {generated_tokens} generated tokens on level {level} without a level-up. Reconsider an uncertain assumption about the controls, mechanics, or goal. Check exact board/history data in Python and consider an alternative explanation; choose a small test that distinguishes them before extending the same search. Keep verified facts. If your plan's predictions match observations and it is making progress, continue it.

This is appended to the ordinary user opener, including short/state-only resumptions. It does not change the system prompt, force a RESET, start a separate critic, or tell the model to think less. A threshold crossed during a long invocation waits for the next opener; the baseline's 2,048-token yield provides regular opportunities.

## A/B submission settings

A: [`early-level-recovery-baseline/v1.json`](../../ARC3-Inference/configs/early-level-recovery-baseline/v1.json). B: [`early-level-recovery/v1.json`](../../ARC3-Inference/configs/early-level-recovery/v1.json). Both use the same code snapshot. Their only operational difference is `ARC3_EARLY_LEVEL_RECOVERY_TOKENS`: `0` vs `32768`; notebook titles/slugs and source datasets identify the two arms.

| Setting | Both arms |
|---|---|
| Harness | Allocation one-wave baseline; priority scheduler, 120 lanes, one public pass |
| Serving | PR #27 lorddraft: Intel W4A16 target, Lord NVFP4 MTP draft, Lord vLLM runtime |
| Request limits | Harness active streams 14; vLLM `max_num_seqs=14`, prefill 2,048 tokens |
| Model limits | Harness context 131,072; server max length 147,072; max output 12,288; thinking budget 8,192 |
| Prompt/tools | Baseline wording, 3,072 returned-tool tokens, retained functions, current board at upscale 10 |
| Images | Current board and baseline diff/death images; animation images off, intermediate boards off |
| Yield | 2,048 generated tokens per analyzer invocation; no time yield |
| Run window | Public soft end at notebook minute 131; notebook cap 141 minutes; per-game cap 532 minutes; no per-game token cap |
| Games | The same 15 newer public games, excluding the original 10 |

Game pool: `tn36`, `lf52`, `cn04`, `bp35`, `wa30`, `sp80`, `m0r0`, `ka59`, `sk48`, `dc22`, `cd82`, `g50t`, `ls20`, `s5i5`, `su15`. The JSON files retain exact game IDs and order.

The PR's 24-minute public setting was a serving smoke test; it is too short for this quality comparison. Both arms use the longer baseline window. Startup consumes part of that window, so record actual playing time and tokens when comparing. Animation API/text remains available as in the baseline; sampled animation images are explicitly disabled.

The lorddraft reference achieved 803 vs vcintel's 725 generated tokens/s with similar scores, according to the PR. Its 14-request configuration is a different engine from the earlier Pennyroyal cache-thrashing experiment. Our harness's memory footprint still needs confirmation from the submitted logs.

## Validation and comparison

Focused tests cover trigger boundaries, generated-token accounting, level/death/session lifecycle, delivery in all opener modes, threshold crossing at a yield, and failed/unsent requests. Disabled and below-threshold prompts and the system prompt must match the baseline exactly. The config tests execute the real Make-to-serving settings handoff and assert that the two arms differ only in the recovery flag.

No gameplay improvement is established locally. After both exact versions finish, retrieve source and outputs and evaluate them. Compare paired per-game scores, aggregate score, completed levels/wins, generated tokens, thinking/code breakdown where available, actual play time, serving throughput and request failures. In B's transcripts, inspect each `Progress check:` and the subsequent tests/actions: did a weak assumption change, and did the agent progress? One pass per arm is exploratory evidence; across-game pairing does not isolate sampling variance or differing allocation between games. Repeat a promising result before claiming a general gain.

Local checks on 2026-10-09: full suite **400 passed, 64 existing branch-divergence skips**; the focused recovery/config/MTP suite run through Make **30 passed**; Ruff and `git diff --check` passed. Comparing against the live `origin/allocation-vc-one-wave` source at `7ce6732`, the disabled feature produced byte-identical system prompts, user prompts and tool schemas across nine level/event cases (levels 1–3, ordinary/death/level-transition openers).

Submitted on 2026-10-10 from clean source commit `143902e`, using the user's `jonathanwang2022` account:

| Arm | Exact notebook version | Private source dataset | Status after submission |
|---|---|---|---|
| A | [early-level-recovery-baseline v1](https://www.kaggle.com/code/jonathanwang2022/early-level-recovery-baseline) | `jonathanwang2022/early-level-recovery-baseline-src` v1 | RUNNING |
| B | [early-level-recovery v1](https://www.kaggle.com/code/jonathanwang2022/early-level-recovery) | `jonathanwang2022/early-level-recovery-src` v1 | RUNNING |

Both pushes explicitly reported `Kernel version 1 successfully pushed`. The packaged deployment targets each contain the intended 15-game pool and a 141-minute notebook cap. The launch command lists 25 initial games; the notebook narrows them using the packaged `public_games` list. The two packaged `tool_agent.py` files have the same SHA256, `0a715ead3c74666e668e8e831c4d4a8c6fa0bf9add4ae7b24268a53d410d9241`. Their rendered setup commands contain recovery thresholds `0` and `32768`, respectively.

A local follow-up monitor is running in managed execution session `50317`. First completion check: **2026-10-10 06:37 UTC / 02:37 Eastern**; subsequent checks every five minutes while queued/running. It retrieves exact v1 outputs and source, then runs `make evaluate_run`; failed runs have their available outputs/logs retrieved. Timer/script/log/status files are under `/private/tmp/arc3-early-level-recovery-followup/`. The current source fetcher recognizes only `taaf-kaggle-source*` automatically, so the monitor supplies the exact named dataset reference and dataset version through its existing `fetch_source` API when that default fails.

Remote RUNNING status is submission evidence, not a gameplay score. Results are pending.

## Independent overnight arm: v2

Config `configs/early-level-recovery/v2.json` retains the same notebook/source dataset names, game set and runtime as v1. The arm is rebased directly on the allocation tip, then adds action coverage to the existing 32,768-token, level-1/2, once-per-level reminder. Other arms' animation guidance, image settings, simpler event wording, and 4,096-token yield are absent.

At reminder time, derive untried currently valid action types from the full runtime execution history. Group MOUSE coordinates as the same action type, normalize engine/model labels, and exclude RESET from suggestions. Count only actual model actions on the current level: refused/skipped actions and automatic RESET do not count. Attribute the action that completes a level to its pre-action level. Deaths, context eviction and prompt-history trimming do not erase runtime coverage. Say that trying a type does not cover every target or interaction; if all currently valid types were used, report that without forcing extra exploration.

All overnight arms start directly from `allocation-vc-one-wave` at `b96b113b4af948e55369d5739f56bcf6aa126ccb`, fetched from origin on 2026-10-10. No arm includes another arm's treatment. The two initial recovery runs must finish before this queue starts; thereafter no more than two notebooks may be RUNNING/QUEUED/PENDING on `jonathanwang2022` at once. Submission is already authorized and uses the normal `make kaggle-duck` target directly.

Settings held fixed: the same 15 exact game IDs and order as `early-level-recovery-baseline` v1; one public pass; priority one-wave scheduling; 14 active model streams and vLLM sequences; Lord vLLM `e975732`, Intel W4A16 target with Lord NVFP4 MTP draft (3 tokens); 131-minute public soft end plus 10-minute notebook buffer; context 131,072, server max length 147,072, thinking budget 8,192, max response 12,288 and tool returns 3,072. Intermediate action boards are off. Current/diff/death board images remain enabled. No generated-token cap per game. The two-job queue limit is Kaggle notebook concurrency, separate from each notebook's 14 inference streams.

Games: `tn36`, `lf52`, `cn04`, `bp35`, `wa30`, `sp80`, `m0r0`, `ka59`, `sk48`, `dc22`, `cd82`, `g50t`, `ls20`, `s5i5`, `su15`. Config files retain their exact IDs.

The allocation tip includes newer bounded startup prefetch than the already running baseline v1. All four new arms inherit it equally. When comparing to that earlier baseline, record actual startup/play time and generated tokens rather than attributing a startup-time difference to prompt behavior. One pass is exploratory; compare per-game scores, levels, tokens, request failures, and intervention-adjacent transcripts before claiming a gain.

Validation on 2026-10-10: Ruff passed; full suite **422 passed, 64 existing branch-divergence skips**; real Make-to-serving handoff and rendered ordinary/death prompts passed. Animation validation decoded every attached PNG and confirmed 256×256 for a 64×64 board at scale 4. Recovery validation confirms unused types in all opener modes and correct execution/level attribution. Simpler-prompt validation confirms the system prompt is byte-identical to the allocation baseline and the representative death opener shrinks from 2,912 to 1,654 characters. Normalized configs are identical after removing only the arm's named treatment and notebook/source metadata. No gameplay gain has been measured yet.
