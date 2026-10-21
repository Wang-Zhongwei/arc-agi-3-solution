# State lab, round 1: do harness facts and guards keep the model on the right idea?

2026-10-11. The plan: take the worst moments from the [mem-v2 reasoning audit](memv2-reasoning-audit.md), rebuild each on the exact engine version, re-sample the model's next call about 8 times with and without a change, run every reply's code on the engine, and count how often it keeps the right idea. The goal was to choose the audit fixes for the 3-hour stack before spending GPU. Tooling: `analysis/state_lab/` on `tooling/state-lab` (with `analyze.py` for the statistics). States and success criteria were fixed before sampling, in `analysis/state_lab/states/round1.json`. Results page: `kaggle-outputs/replays/state-lab/index.html`. Raw rows: `kaggle-outputs/replays/state-lab/runs/round1/` (`stats.json` holds the numbers below).

## Bottom line

**The round stopped early and cannot rank any variant.** OpenRouter refused calls once the account ran out of credit. The key showed $9.83 left, but that was the key's own limit: `/api/v1/credits` shows total_credits $0 and usage $0.174, so the free-tier account is at -$0.17. Only 22 of the planned ~290 samples were graded. Six variants were tried on one state each, 2 samples against 2 for control. Three variants (win_snapshot, crops, reset_confirm) were never sampled. With 2 samples against 2, the smallest possible p is 0.33, so no result here could reach significance whatever the model did. What follows is mechanism plus anecdote. Spend: $0.0934 of the workflow's $6.00 budget.

What the pilot did establish is that the pipeline works:
- Each control's prompt equals Kaggle's logged prompt minus 11 tokens.
- Alibaba's prefix cache brings a repeat sample down to about $0.002.
- Replies execute on the rebuilt engine. One tr87 control re-ran the run's loop into the same GAME OVER.

## Results

Success is k/n with the exact (Clopper-Pearson) 95% interval; p is Fisher's exact test against the same state's control. "Acted" counts the samples whose code executed at least one game action.

| state | variant | success | 95% CI | vs control | p | acted | RESETs | GAME OVER |
|---|---|---|---|---|---|---|---|---|
| tr87_L5_loop | control | 1/2 | 1-99% | | | 2/2 | 0 | 1 |
| tr87_L5_loop | loop_guard | 2/2 | 16-100% | +50% | 1.00 | 0/2 | 0 | 0 |
| tr87_L5_loop | advice | 2/2 | 16-100% | +50% | 1.00 | 1/2 | 0 | 0 |
| lf52_L5_reset | control | 1/2 | 1-99% | | | 1/2 | 0 | 0 |
| lf52_L5_reset | budget_time | 0/2 | 0-84% | -50% | 1.00 | 0/2 | 0 | 0 |
| lf52_L5_reset | untried_reset_guard | 0/2 | 0-84% | -50% | 1.00 | 2/2 | 2 | 0 |
| m0r0_L6_selected | control | 0/2 | 0-84% | | | 1/2 | 0 | 0 |
| m0r0_L6_selected | facts | 1/2 | 1-99% | +50% | 1.00 | 1/2 | 0 | 0 |
| m0r0_L6_selected | memory_layout | 0/2 | 0-84% | 0 | 1.00 | 0/2 | 0 | 0 |
| ar25_L3_blocked | control | 1/1 | 2-100% | | | 1/1 | 0 | 0 |
| sb26_L2_tryH | control | 1/1 | 2-100% | | | 1/1 | 0 | 0 |
| sp80_L4_finish | control | 0/1 | 0-98% | | | 1/1 | 0 | 0 |
| m0r0_L6_reset | control | 0/1 | 0-98% | | | 0/1 | 0 | 0 |

No sample gained a level.

**Pooled across states.** The test is an exact stratified permutation test: labels are shuffled only within a state, so each variant is compared with control state by state. Every variant was sampled on just one state, so the pooled test reduces to Fisher's exact test on that state, and all six give p = 1.00. Planned against sampled states: advice 1 of 3, budget_time 1 of 4, facts 1 of 5, loop_guard 1 of 2, memory_layout 1 of 2, untried_reset_guard 1 of 2; crops 0 of 1, reset_confirm 0 of 1, win_snapshot 0 of 2.

## What each variant did

- **loop_guard: mechanically solid.** Replaying the run's own code on the engine, the guard (stop at the second visit to a board within a snippet) halts tr87 L5 after 13 actions. Without it the run went to 102 actions and GAME OVER, and one of the two control samples did the same again. Both samples that followed the guard read the refusal and switched to a read-only diagnosis. One wrote *"... never equals 6 → infinite loop. That matches the observed behavior!"* Both then crashed on `NameError: TGT`, a constant from the stopped snippet; the sandbox keeps only functions between calls. So the 2/2 is mechanical: neither sample took an action.
- **untried_reset_guard: weak sign against.** Both follow-up samples re-issued RESET after reading the untried list. One wrote *"From this board, LEFT is untried ... Given the deadlock, RESET is right ... Let me just confirm RESET."* One control sample pressed LEFT ×6 with no guard at all. So the soft confirm may cost a call and change nothing.
- **facts (the model's own rule from before the trim, quoted back): the only variant that produced the target move.** One sample wrote *"with the marker selected, move it LEFT 3 ... Per L3: yes."* and pressed LEFT ×3. The other five m0r0 samples (both controls, both memory_layout samples and the second facts sample) clicked once or only inspected the board, and two of them guessed at a new "colour toggle" mechanic.
- **budget_time: untested where it matters.** It was sampled only on lf52, where the bar is not the problem. The states it targets were never sampled: sp80 with 2 minutes left, and wa30 twice. The one sp80 control repeated the run's mistake, pressing SPACE without the DOWN that has to come first.
- **memory_layout: no difference.** Both samples inspected the board, as the controls did.
- **advice: used once, in a single sample.** One sample wrote *"Cap total actions at ~30"*, echoing the advice sentence, and ran 6 actions safely. The other crashed (`open` is not defined). A control sample capped its own loop without any advice.

## Recommendation for stack-3h (`arm/stack-3h`, every flag off by default)

| variant | stack-3h flag | call |
|---|---|---|
| loop_guard | `ARC3_REPEAT_STATE_GUARD` (already built: stops an action repeated from a board already seen in the snippet) | **Turn it on in the 3-hour config.** The benefit is mechanical and engine-checked. Open risk: it could cut off a correct long batch; ar25 L3 was meant to check that and was not sampled. Persistent sandbox globals would fix the follow-up crash it caused. |
| facts / rule ledger | none (`ARC3_MEM_EVERY_CALL` and `ARC3_LEDGER_ON_TRIM` carry the model's own notes, not verified rules) | **Sample more first.** This is the most promising lead and the first priority when the run resumes, across all 5 planned states. |
| win_snapshot | none (`ARC3_GAMEOVER_DIFF` covers deaths only) | **Sample more first.** It is the audit's top fix and was never sampled. |
| budget_time | `ARC3_BUDGET_ALERT` (bar only) | **Sample more**, on sp80 and wa30. The bar line is already in the stack; the time-left line is new. |
| untried_reset_guard / reset_confirm | none (`EXPOSE_RESET` has only the reset-with-no-actions refusal) | **Hold.** Sample m0r0_L6_reset, and the untried line on its own without a refusal. |
| memory_layout | `ARC3_MEM_EVERY_CALL`, `ARC3_MEM_LEGIBLE` | **No call from this lab.** It is its own arm; judge it by repeated Kaggle runs. |
| advice | (`ARC3_STEP_VERIFICATION_HINT` is similar) | **Drop from the stack.** The prior holds: Flash-Next ignores free advice. It is cheap to keep in the resumed lab run. |
| crops | none | **Sample more** (cd82 L2). |

Any variant that goes into the stack still has to pass repeated Kaggle runs (the alpha rule). This lab only decides what is worth those runs.

## Power and caveats

- **Power at the planned N=8 per cell**, with the pooled test at α=0.05 (2,000 simulations): one state detects 20%→70% only 41% of the time. Two states give 74%, three give 93%. For 20%→50%: 11%, 29% and 48%. A variant tested on a single state needs about N=20 per cell for 80% power at 20%→70%. So resume with at least 3 states per variant, or move N to the variants that are on few states.
- **Model.** OpenRouter's Alibaba Qwen3.8 Flash runs at unknown precision and without the MTP drafter. It is not our W4A16 Qwen3.8-Flash-Next. The sampling parameters match, and prompt tokens equal Kaggle's minus 11. Use these rates to rank changes, not to predict Kaggle scores.
- **One step.** Each sample is one model call and its one snippet. Many snippets only inspect the board, so "tries X in the first N actions" fails on them and "no GAME OVER" passes trivially; read the acted column alongside success. The follow-up variants sample the model's next call after a guard refused the run's own reply, so they compare a different exchange than control does.
- **Grading and timeouts.** Several criteria are regular expressions on the reasoning, so they are noisy. On this laptop, lf52 snippets can hit the 30 s sandbox limit; one control sample timed out after LEFT ×6.

## To finish the round

1. Add about $2 of credit to the OpenRouter account.
2. In the `tooling/state-lab` worktree, run:

```
nice -n 15 $PY -m analysis.state_lab.run --states analysis/state_lab/states/round1.json --exp round1 --n 8 --budget 6.0 --out kaggle-outputs/replays/state-lab/runs
```

3. Then run:

```
python -m analysis.state_lab.analyze --states analysis/state_lab/states/round1.json --runs kaggle-outputs/replays/state-lab/runs --exp round1 --html kaggle-outputs/replays/state-lab/index.html
```

The run skips the 22 samples already done. Expect it to cost $0.75–1.00; the $6 ledger cap still covers it.
