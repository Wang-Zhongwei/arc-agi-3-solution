# Win reliability: what it takes to win 12 games in one run

2026-10-10. The user's target: "the goal is for 12 games to be solved at the very least".

Data: the nine 121-minute, 25-game runs below. Each of the 12 games that has ever been won got a per-game failure analysis (decisive level, what separated winners from losers, time and tokens, the scheduler's role, model behaviour, levers). Those analyses were run as workflow `winrel` over `runs/kaggle/*/{benchmark.json.xz, evaluation.json, score_timeline.jsonl, *.log.xz, *_requests.jsonl.xz, artifacts/*_events.jsonl.xz, sched_decisions.jsonl.xz}`. The arithmetic is in [`research/data/win-reliability/levers.py`](../data/win-reliability/levers.py), with its output in `levers-output.txt` and the win counter in `wincount.py`.

Runs (letter codes are used in the tables):

| Code | Run | Mean | Score-100 wins | All levels cleared |
|---|---|---|---|---|
| C | v4-control/v1 | 50.62 | 7 | 7 |
| A | v4-user-combo-10x/v1 | 56.94 | 9 | 10 |
| B | v4-user-combo-10x/v2-snowiers | 52.70 | 4 | 7 |
| D | v4-user-combo-10x/v3-snowiers | 49.09 | 5 | 6 |
| U | v4-user-combo/v1 | 49.61 | 8 | 9 |
| T | v4-user-combo-13x/v1 | 47.34 | 6 | 8 |
| K | v4-sched-curriculum/v1 | 47.83 | 5 | 9 |
| X | v4-user-combo-10x-accept05/v1 | 46.77 | 6 | 7 |
| H | v4-hats/v1 | 37.11 | 2 | 4 |

## Bottom line

- **The current stack wins about 6 games per run, not 9.** The pooled 9-run rate is 5.78 score-100 wins per run. `user-combo-10x` alone (A, B, D) gives 6.0. Counting a game as won when every level is cleared, the rates are 7.44 and 7.67. The 9-win run was the top of a wide spread: identical code produced 9, 4 and 5.
- **Twelve is out of reach on reliability alone.** Stacking every lever found here raises the expected count to about 8.8 score-100 wins per run (9.4 on the 10x runs only), or about 9.8 full clears (10.4). P(all 12 in one run) stays at 2–15%.
- **What 12 actually needs.** Twelve games at p = 0.95 each gives P(12) = 0.54. Twelve at p = 0.90 plus 4 more games at p = 0.30 gives P(≥12) = 0.66. So the target needs both:
  - **The reliability levers below.** They take the 12 known games from about 0.48 to about 0.73 mean p.
  - **One or two new games** pushed to about 0.3–0.5 per run. Candidates: re86 (7/8 levels once, usually 5–6), su15 (8/9 once), ka59 (6/7 once), dc22 (4/6 consistently).
- **Losses split into three kinds, and the levers follow them:**
  - **Knowledge lost at trims and level changes.** The memory card holds 200-character reasoning fragments instead of the verified rules. This is the largest lever: cd82, m0r0, tu93, tr87, sc25, tn36.
  - **Committing without checking.** The model submits or executes before simulating its plan in code, or abandons the right goal when its planner says the goal is infeasible: sb26, cd82, vc33, sc25.
  - **The scheduler starving a game.** This happens on its final level (B = 0 makes the price collapse) or early, when a level is long: vc33, tr87, sc25, r11l, tn36, m0r0.
  - **New object classes nobody clicks.** A smaller kind: ft09, vc33, m0r0.
- **No-effect levers.** Earlier start, RESET-to-retry, avoiding deaths, goal transfer, more time late (the tail fade), macros and the action scorer did not appear in any decisive failure. On these 12 games they are worth about 0.

## Baseline per game

P(win) now = wins / runs. "10x" means A, B and D only (3 runs, so read it as a direction, not an estimate).

| Game | Score 100 (of 9) | All levels (of 9) | Score 100, 10x (of 3) | Decisive point |
|---|---|---|---|---|
| ar25 | 9 | 9 | 3 | never lost |
| lp85 | 8 | 9 | 3 | lost only under hats |
| ft09 | 7 | 8 | 3 | L5: three checkerboard buttons nobody clicks |
| r11l | 6 | 6 | 1 | L2/L3 parked or planning marathon, then out of clock on L5/L6 |
| sb26 | 5 | 9 | 1 | L8: the first SPACE is right or wrong (all 9 cleared; 4 lost on efficiency) |
| m0r0 | 4 | 4 | 0 | L6: MOUSE-select blocker forgotten; goal "overlap" misread |
| tr87 | 4 | 6 | 2 | L6/L5: handed out on the last level, price 18, never readmitted |
| vc33 | 3 | 3 | 1 | L4: orange gate never re-clicked; L7: parked with a plan |
| cd82 | 2 | 7 | 2 | L2/L4/L5 efficiency: an incomplete pose graph means probing |
| tu93 | 2 | 2 | 1 | L9: walked into a known guard ray after the rule was trimmed away |
| sc25 | 1 | 3 | 0 | L6: parked before the insight, or reached L6 late after early parks |
| tn36 | 1 | 1 | 1 | L2: the demo panel's program never copied |
| **Sum / 9** | **5.78** | **7.44** | **6.00** | |

## Ranked levers

**Method.** For each losing (game, run) pair, the per-game analysis gave each lever a chance of flipping that loss to a win: about 0.5–0.8 where it said "likely", 0.3–0.5 for "plausibly" and 0.1–0.3 for "possible" or "partial". The full matrix is `F` in `levers.py`. A lever's value for one game is P_with − P_now, where P_with = (wins + Σ p_flip) / runs. When levers are stacked, each loss's flip chance is 1 − Π(1 − p_lever), so overlapping levers are not double-counted. These are judgement-weighted counts from one run per configuration, so read them as a ranking, not as forecasts.

| # | Lever | Games helped (Δ P(win), 9-run) | Extra wins / run: 9-run (10x) | Existing arm or branch | Cheapest test |
|---|---|---|---|---|---|
| 1 | **Verified mechanics, hazard and goal card that survives trims and level changes.** At each level clear, and at each trim mid-level, store the model's verified rules (e.g. "MOUSE-click 2x2 blue selects it; arrows move only it"; "red guard kills along its facing; land on it from the side"; "goal = same cell, a touching 4x8 blob is not a merge"; "cup orbits 8 poses, 45° apart") instead of the 200-character `level_note` fragment. | cd82 +0.25, tu93 +0.19, m0r0 +0.17, sc25 +0.08, tn36 +0.06, r11l +0.05, tr87 +0.05 | **+0.85 (+1.05)** | memory-v2 card (`ARC3_MEM_CARD`, `ARC3_MEM_SOLVED`, `ARC3_MEM_REPLY_WINS=1`; never run on the 10x stack) plus level-debrief (`ARC3_LEVEL_DEBRIEF=1`, being built). **Gap:** a trim-time digest of the current level. tu93's guard-ray rule and tr87's box-shift rule were lost mid-level, and thinking-digest (#136) never fired. ledger-on-trim is already in the stack and did not carry them. | CPU first: run memory-v2's card builder (memory_sim) over the logged transcripts and check whether the decisive facts (m0r0 L6 select line, tu93 L9 guard ray, sc25 L5 teleport code, cd82 orbit) are in the card at the decisive level's start. Then a reliability panel (see Test plan) on cd82, m0r0, tu93, tr87 and sc25. |
| 2 | **Simulate before committing, and suspect the model before the goal.** Before an irreversible or efficiency-scored commit (sb26's SPACE, a long plan run), the harness requires a simulation in code, under the carried-forward rule from the previous level plus any new hypothesis. Submit the placement that is valid under the most hypotheses. When the planner says the known goal is infeasible, re-parse the target in code and test the transition model's unverified assumptions (e.g. a two-object interaction test) before changing the goal. | sb26 +0.18, cd82 +0.18, vc33 +0.07, sc25 +0.06 (score-100 only), tr87 +0.03 | **+0.54 (+0.72)** | mdl-tournament plus reversal-check (executable rules scored on transitions; an invented recursion guard costs description length). Goal transfer + search (backlog, not built). sandbox-robust keeps the simulator's data. **Warning:** multi-trajectory showed 0 of 10 games ever defined `simulate`, so this must be a harness check or a fact, not an optional interface. | CPU Stage-0 on logs: on sb26 L8, does the L7 traversal run unchanged reject every losing first placement and accept every winning one? (`sb26_sim.py` in the winrel scratchpad says yes.) On vc33 L7 and cd82 L4, does a harness "infeasible means test assumptions" fact fire at the logged moment? Then a reliability panel on sb26, cd82 and vc33. |
| 3 | **Final-level hold.** On a game's last level (and the second-to-last in the second half of the run), give B a floor of about 3 instead of 0, and do not hand the game over at a trim when its price is within a margin of the best waiter. | vc33 +0.16, sc25 +0.11, tr87 +0.10, tu93 +0.06, r11l +0.01 | **+0.44 (+0.58)**, plus protection for 6 wins that came back only after the tail fade (sc25 A/B, tr87 B/U/T, tu93 D, ft09 D) | **New scheduler flag** on Franzen's `priority_scheduler.py`. Neither curriculum nor UCB does this; the curriculum tilt only nudged level-6+ games. | CPU: re-price every final-level park in the 9 runs (tr87 D 18 → about 215 with floor 3, inside the winners' 143–282), then sim-v2 with a **win-count** objective, not mean score; earlier scheduler sims all optimised the mean. Then full 25-game runs only, since an uncontended panel makes it inert. |
| 4 | **Cap early parks.** Set a maximum wait (about 10 min) or an aging bonus for a game handed out at its first L2/L3 trim, and a fair-share floor for games that are still clearing levels. | sc25 +0.19, r11l +0.15, m0r0 +0.11, tn36 +0.11, cd82/tu93/vc33 +0.02 | **+0.62 gross (+0.57)**, about **+0.3 net**: zero-sum, because the minutes come from other games | **New scheduler flag.** sched-hidden and sched-curriculum stall and round-robin rules lost on mean score (−0.87 to −2.9), so it must be re-priced on win count. | sim-v2 with the win-count objective; keep only if the mean score loss is ≤ 0.5. |
| 5 | **Novel-object and re-probe facts.** A per-level registry of object classes by appearance signature and whether each was clicked. One fact line after a NO-OP or a failed target: "never clicked on this level: N/M checker ×3 at …". A re-probe flag when an object first clicked as a NO-OP changes appearance ("gate was w → NO-OP; now O; untested"). Report any colour first seen in an intermediate frame of a batch. MOUSE feedback names the object under the click and warns when every coordinate is under about 16 (a grid-index leak). | vc33 +0.18, m0r0 +0.11, ft09 +0.09, r11l +0.02 | **+0.41 (+0.42)** | Partly the attention-prompt `Salient:` line (#57; appeared objects, first-seen colours) and memory-v2 action-effects (keyed by cell, not object). Harness-systems-gap survivor 4 (object-level surprise line). **The registry, re-probe flag and intermediate-frame alert are new.** | CPU replay of the logged frames: does the line fire at ft09 L5 (H, D), vc33 L4 (K act 66, D step 32) and m0r0 L6 (D grid clicks)? Then a reliability panel on ft09, vc33 and m0r0. |
| 6 | **Forced action after a no-action streak.** After 4–5 consecutive turn-budget yields with no action, require the best probe or the first part of the plan. This also delays the first trim, so the game hands over at a higher level and price. | r11l +0.11, ft09 +0.02, sc25 +0.01 | **+0.14 (+0.27)** | **New.** Harness-systems-gap rated a "model-authored probe on a gated inspect streak" sign-uncertain and rejected an inspect-budget yield, so treat this as a risk. | CPU: count streaks of 5+ yields per level in the 9 runs and how they end. Gate on the panel. |
| 7 | **Twin-panel / demo diff plus a small-control sweep** (tn36). When a non-editable twin panel changes under a control, give the full displayed configuration as a fact. When an expected effect is missing on a ≤6-toggle glyph, sweep the subsets in Gray order. | tn36 +0.26 | **+0.26 (+0.22)** | **New.** The sweep could run on program-induction's policy executor. Low hidden-set transfer (tn36-specific). | Low priority: build only if the cheap levers are done. |
| 8 | **Sandbox state** that persists plain-data variables, a live `current_frame` (warn on capture in a default argument), and no silent loss of mutable defaults. | tr87 +0.07, m0r0 +0.03 | **+0.10 (+0.13)** | **sandbox-robust** (#76, `ARC3_SANDBOX_PERSIST_VARS`, `ARC3_SANDBOX_SHIMS`), built and not run. The `current_frame` warning is new and small. | Cheapest of all: add the flags to the next 10x run. |
| 9 | **Bar-aware execution.** When the plan length is at least the remaining step bar minus a margin, RESET and run from a full bar; a "budget exhausted" death-cause fact. | tu93 +0.07, tn36 +0.03, sc25 +0.01 | **+0.11 (+0.03)** | expose-reset is already on in `user-combo-10x`. budget-alert (#60) and budget-truth (#74) give the bar facts; the plan-vs-bar check is new and small. | Fold into the next stack run; judge from transcripts (did RESET fire before long plans). |
| 10 | **Separate same-colour touching objects** ("S at (34,14) and S at (38,14), adjacent, not overlapping"). | m0r0 +0.06 | **+0.06 (+0.17)** | **New** perception. Related: the controllable-entity tracker (harness-systems-gap 1) and program-induction's `avatar()`. | CPU replay on m0r0 L6 frames. |
| 11 | **No-op repeat guard.** Refuse an action already seen as a NO-OP from the same board within the level. | cd82 +0.06 (efficiency only) | **+0.06 (+0.02)** | Franzen's `ARC3_NOOP_REPEAT_GUARD` (exists, off) plus memory-v2 seen-state. | Turn the flag on in the next stack run. |
| – | Earlier start, RESET-to-retry a level, avoiding deaths, goal transfer, more time late (tail fade), learned macros, action scorer, program induction (except as a sweep executor for lever 7) | none | **0** | – | – |

Notes on the ranking:

- **On the 10x runs alone the order is the same.** Card +1.05, execute-and-check +0.72, final hold +0.58, park cap +0.57 (gross), novelty +0.42. The extra weight on m0r0 comes from 10x losing it 0 for 3, and r11l weighs more for the same reason.
- **Lever 4 is ranked below lever 3 despite the larger gross number.** It is zero-sum against other games, and every stall or round-robin rule tested so far lost on mean score. Lever 3 costs less: few games are on their final level at once, and they need 10–20 minutes.
- **Levers 1 and 2 interact.** A card with wrong rules makes simulation confidently wrong (cd82 U "provably impossible"). Lever 2's "suspect the model before the goal" rule is the guard against that.
- **Score 100 versus all levels cleared.** For a full clear, lever 2 loses most of its value (sb26 and cd82 are already cleared 9/9 and 7/9) and the scheduler levers gain. All levers stacked give 9.76 full clears per run (10.41 on 10x).

## Combinations

Expected wins per run and the chance of 10 or more, or all 12, from a Poisson-binomial over the 12 games' P_with:

| Stack | Score 100, 9-run: E | P(≥10) | P(12) | Score 100, 10x: E | All levels cleared, 9-run: E | P(≥11) |
|---|---|---|---|---|---|---|
| now | 5.78 | 0.004 | 0.000 | 6.00 | 7.44 | 0.007 |
| 1 + 3 (card + final hold) | 7.03 | 0.047 | 0.001 | 7.56 | 8.46 | 0.053 |
| 1 + 2 + 3 | 7.47 | 0.082 | 0.001 | 8.18 | 8.53 | 0.059 |
| 1 + 2 + 3 + 5 | 7.83 | 0.121 | 0.002 | 8.52 | 8.86 | 0.090 |
| + 4 (park cap) | 8.29 | 0.202 | 0.006 | 8.92 | 9.32 | 0.168 |
| all 11 | 8.75 | 0.311 | 0.017 | 9.35 | 9.76 | 0.285 |

Per-game P_with for all levers (9-run): ar25 1.00, lp85 0.89, r11l 0.90, ft09 0.89, m0r0 0.81, sb26 0.73, vc33 0.72, tr87 0.63, cd82 0.61, tu93 0.54, sc25 0.52, tn36 0.49. The weakest four (tn36, sc25, tu93, cd82 for score 100) set the ceiling. Their remaining losses are mostly late-game time (sc25, tu93) or game-specific insight (tn36 L2, cd82 per-level efficiency).

**The combination that plausibly reaches ≥ 12**, in order of build:

1. **Card (1) + sandbox-robust (8) + no-op guard (11) + bar facts (9).** These are mostly built: `ARC3_MEM_CARD`, `ARC3_MEM_SOLVED`, `ARC3_MEM_REPLY_WINS`, `ARC3_LEVEL_DEBRIEF`, `ARC3_SANDBOX_PERSIST_VARS`, `ARC3_SANDBOX_SHIMS`, `ARC3_NOOP_REPEAT_GUARD` and budget-alert on top of `user-combo-10x`. Missing piece: the trim-time digest of the current level. Expected about +1.0 (to about 7.0).
2. **Final-level hold (3)**, plus the early-park cap (4) if sim-v2 shows it costs ≤ 0.5 mean. Expected +0.5 to +0.8 (to about 7.7).
3. **Execute-and-check (2) + novelty facts (5).** Expected +0.6 to +0.9 (to about 8.5).
4. **New games.** Even after steps 1–3 the expected count is about 8.5–9 score-100 wins, or 9.5–10 full clears. Reaching 12 needs 2 or 3 more games at p ≥ 0.3. The next analysis should be this same per-game study on re86, su15, ka59 and dc22 (best levels 7/8, 8/9, 6/7 and 4/6).

## Test plan, cheapest first

1. **CPU only, no GPU (laptop, one job at a time, `nice`).**
   - Novelty-line replay (5), the card-content check (1), the sb26 executable-rule check (2), the touching-object split (10) and the streak count (6), all on the logged frames and transcripts of these 9 runs.
   - Final-level re-pricing (3), and sim-v2 re-scored on win count for levers 3 and 4.
   - Each check has a pass bar: the fact fires at the logged decisive moment in at least half the losing runs and in under 10% of all turns.
2. **Reliability panel: new, small tooling.** Add a panel to `kaggle/build_notebook.py` with 7 fragile games × 2 copies, using `run.py --n-passes 2`, which already exists. Run it at 14 streams for 90 min. Suggested games: cd82, m0r0, sb26, tu93, tr87, vc33, ft09.
   - With 14 runs on 14 slots nothing waits, so this measures the model-side win probability per game with the scheduler removed. Use it for levers 1, 2, 5, 8, 9, 10 and 11, paired against the same panel on `user-combo-10x`.
   - One session gives 14 game samples, against 12 per full run (and 25-game runs spend half their samples on games that are never won). Three paired sessions per arm (42 game samples each) detect a mean p change of about 0.3 at 80% power, or 0.2 at 50%.
   - Caveat: games get more slot time than in a 25-game run, so absolute p is inflated in both arms.
3. **Full 25-game, 121-min runs** for the scheduler levers (3, 4) and for the final stack. By the user's rule, an alpha needs repeated runs, so at least 3 runs of the final stack. Compare with `make significance` and with per-run win counts.

## Caveats

- **The flip chances are judgement weights from single runs.** The ranking is more robust than the absolute numbers. The 10x-only column uses 3 runs per game, and its P = 1.00 entries (ft09, lp85, ar25) are optimistic.
- **The pooled 9-run baseline includes configurations already rejected** (hats, accept05, 13x, curriculum). Some of the flips it counts (hats lp85, accept05 everywhere) would not occur on the current stack. The 10x columns exclude them.
- **"Fill is not use."** Levers that rely on the model choosing to use a new field (2, 6) have failed before (hypothesis loops, multi-trajectory, confidence modes). Prefer harness-computed facts and harness checks.
- **The hidden set** is 110 games, each played once, at 11 games per slot. Levers 1, 3, 5, 8 and 9 are game-agnostic and should transfer. Lever 7 is tn36-specific. Under heavier contention the scheduler levers matter more, but so does their zero-sum cost.
