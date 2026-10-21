# fable-layer-vs-ours: verify

Workflow run `wf_c4d3dccd-2d7`, phase Verify. Compare leoprovorov Fable & Astra notebook (handbook + Fable layer on Franzen) with our stack: read, measure claims on our runs, compare, skeptic-verify

---

**Skeptic review: the Fable & Astra comparison report (2026-10-05)**

The report's main conclusion holds: as a submission, their notebook has not been shown to be better. But several of its numbers and confidence labels are off, and three are wrong.

Tags: K = kept, W = weakened, R = refuted. All new files are in `FC/skeptic/` (`FC` = `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/fable-compare`):
- `paired.py`: the paired per-game test.
- `arc-prize-2026-arc-agi-3-publicleaderboard-2026-10-06T03:59:17.csv`: the full public leaderboard.

Heavy jobs ran one at a time under `nice -n 15`.

### A. Numbers I re-checked at the source (12)

1. **Leaderboard, theirs 31.73 vs ours 27.62: K, with context.** [verified]
   - CLI: our submission 56838702 scored 27.62.
   - Leaderboard CSV: their team AlekseiProvorov (leoprovorov) has 31.73 at rank 69, as the best of 7 submissions. We are at rank 463.
   - Three teams sit at exactly 27.62, ours included (rows 462–464). arms.md:148 gives 27.62 as another team's unchanged resubmission of Franzen's notebook, so ours is probably an unchanged Franzen copy too. [inferred]
   - HB p.5 says their 31.73 is also "a copy of this notebook". [their claim]
   - So the 4.1-point gap compares one Franzen copy with another. It is resubmission noise, and on their side it is also a best-of-7 pick.
2. **Ledger-on-trim 52.56 / 49.14 vs control 46.25: W.** I re-ran `action_penalty.py` and the means reproduce exactly. But with my paired per-game test (`FC/skeptic/paired.py`):

   | Run vs control | Mean diff | p | Games better / worse |
   |---|---|---|---|
   | r1 | +6.31 | ≈0.23 | 10 / 7 |
   | r2 | +2.89 | ≈0.70 | 9 / 10 |

   - The better evidence is at run level. Both ledger runs cleared 112 levels, against 101–107 in every control-like run.
   - Both ledger means sit above all 4 control-like means (44.10–46.38). That ordering has a rank p of about 1/15 = 0.07.
   - So the gain is "likely real". "Replicated" overstates it, because r2 alone is inside the 3-point noise band.
3. **Action-count penalty, 1.4–2.2 points (control and ledger) and 1.0–3.9 (all 6 runs): K.** [verified by re-run]
   - The formula is at game.py:381-416.
   - game.py:259 is the docstring invariant `sum(actions_per_level) == len(history)`, which implies that actions in dead attempts count.
   - So MV2M:75's "actions are not penalised" is indeed wrong.
4. **Clock confusion in 12 of 25 games per run: W.**
   - The counts are right in all 3 runs.
   - At least 2 of the 36 first snippets are false positives that reason correctly about the run clock: r2 ft09 and r1 re86 ("time_remaining_seconds: 537 → … be efficient").
   - The true figure is about 11 of 25, counting games where the idea was raised.
5. **Repeats by tokens on the level (85% of repeats and 75% of revisits after 15k; repeat rate 3.0 / 11.4 / 15.3%): K as arithmetic** (`gate_repeats.out`). The interpretation is weakened under C2.
6. **Fable pin plus our ledger gives 2 copies of the log in 6 of 12 requests: K qualitatively.**
   - I re-ran `sim_stackv2_conflict.py` and it reproduces 6 duplicates and 6 head changes with no trim flag.
   - The rate is an artifact of the setup: a 6k budget that forces a trim every other request.
   - In real runs the duplicate would last from one trim to the next.
7. **18 refreshes at about 4.7 s each: W.**
   - The 18 is verified (`replay_…r2.json`, sum of `refreshs`).
   - I found no source for 4.7 s anywhere in FC. The "[verified prefill time]" tag is wrong.
8. **44 NameErrors in 2,217 python calls: K** (grep re-run).
   - But 7 of the 44 are `open`, which the sandbox blocks, and several are missing imports (Counter, deque, pd).
9. **Budget alert would have been news on 1 of 35 levels: K** (recomputed from `claims_analysis.out`; the one level is tu93 L4).
10. **Bar empty at 17 of 44 deaths: K** (claims_analysis.out:116).
11. **MDE of 0.87 at n=10 and 0.50 at n=30: K** (2·0.97·√(2/N)).
12. **Token budgets: K.** 4.48M / 25 ≈ 179k per game, against 181k at HB p.4. The hidden run's 532 min per game is confirmed by `kaggle/template.ipynb` cell 16 (`532*60`).

### B. Claims that "theirs is better"

- **Harness-computed facts: W.**
  - The layer is unmeasured.
  - Our base already computes facts from the frames: the per-action echo (`ARC3_ACTION_ECHO=1` in the template), `gameplay_changed`, the diff image, the animation timeline and the death ledger.
  - Their notes appear only after 15k tokens on the level, and the log only after 30k.
  - Their one tested fact, a per-turn budget line, had "no effect visible: losses by budget 6 and 6" (HB p.84). It was confounded with the journal.
- **Step budget, "theirs slightly ahead": R.**
  - Their own data shows no effect (above). Ours shows the model mentions the bar first on 34 of 35 alert levels.
  - B1 is real: the alert state is keyed per level, not per attempt (P:141-151).
  - But the report's test claim is wrong. `test_budget_attempts.py`'s own output reads "attempts with an alert: [1, 2]": attempt 2 does alert, late, with 4 of 64 cells left. Only attempt 3 never alerts.
  - B2 (results up to 5 frames stale) is verified in the trace: at k=60 the detector shows 8 cells left when 4 remain.
- **Clock diagnosis: K** (HB p.15).
  - One correction: the report says hiding the clock was "proposed, not run". HB p.31 says the clock display was switched off in their 10-copy experiment. It was used as a fixed condition, never A/B-tested.
  - The causal story is incomplete:
    - Our own prompt calls the bar a "timer" at PR:105, 106 and 241, and says "step/time budget" at PR:247.
    - Many snippets infer "timer" from irregular bar decrements (r1 cn04, control s5i5).
- **Test method (copies panel): K, with caveats.**
  - Their level-1 criterion hit a ceiling (9 of 10 copies cleared it) and had to be rewritten after the baseline run (HB p.86).
  - The panel only applies to mechanisms that act within one level.
- **Diagnosis depth: K** (qualitative).
- **"Their only measured change was a journal": W.**
  - That run combined the journal and the budget line (HB p.78).
  - They also measured a 10-copy sk48 baseline (HB p.86).
- **"V2 contains the layer and ran 10 s": W.** I could not verify it from the CLI, and the point doesn't matter.
  - The public kernel is a CPU display notebook: `enable_gpu` is false, and its code cells are plots plus the layer's source (cells 41 and 42). No harness runs in it.
  - The layer ships in a separate notebook, F001. Whether F001 was ever submitted is unknown; they have 7 submissions.

### C. Claims that "we are ahead"

- **Memory gain, "measured and replicated": W.** See A2. Call it likely real, at run-level p ≈ 0.07.
- **The ledger head adds no extra prefill or handover: K.** I re-ran `sim_eviction_flags.py`: handovers happen only at trims (case C: 5). Adding Fable's tail pin adds 1 (case D: 6).
- **Unit tests and paired evaluation: K.** arms.md:190 records 32 tests on stack-v2. Note that `make significance` was never run on the ledger result.
- **"About 12 bugs":**
  - B9 is verified: `key_of` drops click coordinates, and the clearing action is fed as (pg, pg), so it is logged as a no-op (P:88-92).
  - B12 is weakened: F001 does set `FABLE_WRITE_LOG=True`, but writing to `/kaggle/working` is allowed.
  - The RAM risk is weakened: about 33 MB per game × 10 concurrent games is roughly 0.3 GB.

### D. Expected effects of the arms to port

1. **budget-truth.** The small effect is K. The expected read-out of about 12 → about 0 matches is R.
   - The prompt's "timer" and "step/time budget" wording, and the irregular bar rates, would remain.
   - If the arm runs, it should also change PR:105, 106, 241 and 247 to "step bar". That is a prompt-behaviour change, so make it on purpose.
   - Unintended budget deaths, re-run with `death_intent.py`: 4 clear plus 3 more with no bar detected, across 2 runs.
2. **mem-facts-gated, "0 to +3": W, to roughly 0 to +1.**
   - MV2M:66: 97.8% of repeats across calls were of actions still in the context window. The model wasn't missing the information.
   - MV2M:65: 26% of exact repeats are replays after a death, which are often needed.
   - MV2M:70: the model already mentions revisits 45% of the time.
   - The "53% of post-30k tokens go to uncleared levels" figure is not reproduced. I get 44% (control), 49% (r1, r2) and 64% (stack-v1).
   - "+3" requires the unexplained ledger mechanism to carry over.
3. **mem-solved, "small to moderate": W.**
   - MV2D:118-119 itself says "expect A to move little".
   - rellik13's +2.8 came from a single submission on a 37k-context Wang harness that had no level memory (RW:60, 69-71).
   - Franzen already retains functions and level-transfer guidance (WRITEUP:257, 266).
4. **Elasticity bound (10% of tokens ≈ +1 point): W.**
   - "streams14 agrees" is R. At elasticity 0.25, its +15% would predict about +1.7 points; it got +0.13.
   - experts16 implies an elasticity of about 1.0, though that run is confounded.
   - Our own runs bound elasticity only to somewhere between 0 and 1.
5. **persist-sandbox-vars: W further.** Franzen discards ordinary variables on purpose, so stale board data can't persist (WRITEUP:257). At most about 30 of the 44 errors are addressable, roughly 1.4% of calls.
6. **UNDO moved the bar on 9 of 22 presses: R.** The sum in claims_analysis.out:379-390 is 10 of 22.
   - It depends on the game: sk48 is 0 of 4, which matches their cell 35; lf52 is 4 of 4.
   - The direction (drain or refill) was never established.
7. **Copies-panel arithmetic and indicator counts: K.** 34% (152 of 449) of MOUSE events are at the click point. ρ = 0.80 on sk48 (p = 0.07); ρ = 0.26 pooled (n = 330).

### Corrected bottom line

1. As a submission, theirs is not better. Their 31.73 and our 27.62 both come from copies of Franzen's notebook; theirs is a best-of-7 pick, and the gap is within the ±4 resubmission noise. The layer has no submitted or measured result.
2. Their one tested change (journal plus budget line) halved cleared levels (46 → 23). Its budget line showed no effect.
3. The diagnosis that the model reads the bar as a clock holds, at about 11 of 25 games per run here. Our own prompt calls the bar a "timer" in 4 places, so hiding the clock alone won't fix it.
4. The ledger gain is likely real but has not been shown statistically:
   - +7 cleared levels in both runs, and a run-level rank p of about 0.07.
   - Per-game paired tests give p of about 0.23 and 0.70.
   - Run `make significance` before relying on "replicated".
5. Every ported arm is small in expectation:
   - budget-truth: under 0.5 points.
   - mem-facts-gated: about 0 to +1. Repeats are of things still in the window, and some are needed replays after a death.
   - mem-solved: little, as the design itself says.
   - The action term caps efficiency-only arms near 2 points.
6. What is worth taking from them:
   - Hide the clock and reword "timer" to "step bar" as one cheap prompt and harness change.
   - Use per-level read-outs, levels per active minute, and a keep/drop rule written down before each run.
   - Use a copies panel only for mechanisms that act within one level.
7. What not to take: the advice protocol and the invented "Noted" turn, the content-matched pin restore (it duplicates the log with our ledger), and the per-level budget alert as written.
8. Wrong in the report and to fix there:
   - "attempts 2 and 3 never alert": attempt 2 does.
   - UNDO 9 of 22 should be 10 of 22.
   - 53% is not reproduced (44–64%).
   - The 4.7 s prefill figure has no source.
   - "streams14 agrees" with elasticity: it doesn't.
   - "Clock-hiding not run": it was used as a condition in their 10-copy run (HB p.31).
