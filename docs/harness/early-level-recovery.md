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

Intended destinations: A `ticbudja/early-level-recovery-baseline`, dataset `ticbudja/early-level-recovery-baseline-src`; B `ticbudja/early-level-recovery`, dataset `ticbudja/early-level-recovery-src`. Exact push versions and remote status will be recorded after submission.
