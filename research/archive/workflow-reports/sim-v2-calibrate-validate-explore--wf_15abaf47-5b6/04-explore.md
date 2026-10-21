# sim-v2-calibrate-validate-explore: explore

Workflow run `wf_15abaf47-5b6`, phase Explore. Calibrate a continuous-time scheduler simulator from real runs, validate it against held-out real policies (UCB, D-prime), then explore annealed round-robin to D-prime/UCB schedulers

---

**Phase 4: scheduler exploration in simulator v2**

None of these results is a reason to switch away from Franzen. The validation failed on D′ (the simulator puts D′ about 1.25 above Franzen; the real D′ run was 7.5 below). Every policy here except UCB moves compute between games or reorders starts, which is exactly the D′ case. So every Δ vs Franzen below repeats that known error, and every difference within the D′ family is under 1.5 points, below the 2-point "unresolved" rule. I kept the exploration narrow and read it only as "which variants cost or don't cost score in the simulator".

Six commits are on `arm/sim-v2` (41c2eb0 … 664b7e5). All tests pass (29 + 11 new). Simulation CPU was about 28 min, one job at a time under `nice -n 10`. Nothing was pushed and the main repo was not touched.

**Setup**
- **Level costs:** I added a depth-neutral prior (`--deep-pool 1`) and used it for this phase. The old backoff pooled by depth and put medians of 31–39k on unreached levels 5–7, a rising curve the cleared-level data does not show. The Franzen fit barely moves (46.70 / 51.30 against real 45.43 / 50.85), and UCB − Franzen becomes −1.76 against the real −1.59. Per-(game, level) costs are kept.
- **Screen:** 68 configurations, 150 replicates offline and 20 at the 110-game setting.
- **Finals:** 27 configurations on fresh seeds: 60 replicates at 110 games / 532 min and 200 offline (25 games / 121 min), each with and without the ledger.
- **Robustness:** two other level models: any level unclearable with probability 0.04 ("stuck floor"), and the old by-depth prior.
- **Observability:** policies see only runtime state. A twin-game test checks that hidden level costs never change a price.

**Ranked table (finals; Δ vs Franzen with replicates won; last three columns are the mean Δ vs D′ over the four settings)**

| # | Policy | 110 games, ledger | 110 games, no ledger | Offline, ledger | Offline, no ledger | Δ vs D′ | Stuck floor 0.04 | By-depth prior |
|---|---|---|---|---|---|---|---|---|
| 1 | D′, stall patience 7.5 × own cost (shrunk) | +3.14 (60/60) | +2.75 (60/60) | +1.36 (142/200) | +2.68 (177/200) | +0.06 | +0.13 | +0.01 |
| 2 | D′, own cost shrunk toward 30k | +3.27 | +2.53 | +1.46 | +2.59 | +0.04 | +0.13 | −0.04 |
| 3 | Blend, round robin → D′ over 0–40% of run | +3.11 | +2.68 | +1.36 | +2.58 | +0.01 | −0.01 | +0.01 |
| 4 | D′ (harness) | +3.12 (60/60) | +2.69 (60/60) | +1.36 (137/200) | +2.51 (173/200) | 0 | 0 | 0 |
| 7 | Blend, round robin → D′ over 0–80% of run | +3.10 | +2.69 | +1.33 | +2.20 | −0.09 | −0.01 | −0.02 |
| 8 | Blend, logistic 20–60% | +3.07 | +2.61 | +1.12 | +1.85 | −0.26 | −0.11 | −0.18 |
| 9 | Round robin for first 20%, then D′ | +2.74 | +2.20 | +1.08 | +2.26 | −0.35 | −0.18 | −0.28 |
| 10 | Affectify form v1, no fresh-first | +2.88 | +2.62 | +0.97 | +1.65 | −0.39 | −0.30 | −0.46 |
| 11 | (a) D′ fresh-first | +2.58 | +2.09 | +1.15 | +2.12 | −0.43 | −0.17 | −0.29 |
| 15 | Expected remaining weighted score (D′ depth values) | +3.56 | +1.05 | +1.16 | +0.61 | −0.83 | −0.59 | −1.18 |
| 16–18 | (b) minimum budget: 100k tokens, or until first clear | +0.9 to +1.2 | +1.3 | +0.4 to +0.5 | +1.4 to +1.6 | −1.3 | −0.9 to −1.3 | −1.0 to −1.1 |
| 20 | Franzen | 0 | 0 | 0 | 0 | −2.42 | −1.39 | −2.08 |
| 25 | UCB | +0.22 (34/60) | −2.34 (3/60) | −1.10 (71/200) | −2.09 (33/200) | −3.75 | −3.49 | −3.80 |
| 27 | D′ with tail ∝ remaining level weight | −6.08 | −5.30 | −5.23 | −5.28 | −7.89 | | |

**Start times (finals)**

| Policy | Offline: starts after 30 min | Offline: last start | 110 games: starts after 160 min | 110 games: never started |
|---|---|---|---|---|
| Franzen / UCB / D′ fresh-first | 4.3 | 33 min | 8.7 | 0 |
| D′ | 8.7 | 78 min | 49 | 5.4 |
| Blend 0–40% | 5.4 | 77 min | 35 | 5.4 |
| Blend 0–80% | 4.4 | 51 min | 16 | 5.3 |
| Blend, logistic 20–60% | 4.3 | 33 min | 5.2 | 4.5 |
| Round robin 30%, then D′ | 4.3 | 33 min | 8.4 | 0.5 |

**Findings**
1. **Round robin at the start can only ever be one round.** A game's first context segment generates about 63k tokens before it can be switched, each game gets about 180k in total, and starting every game alone takes 27% (offline) and 32% (110 games) of the run. So:
   - (a) fresh-first, then D′, is that round.
   - (b) minimum budgets of 15k, 30k or 50k never bind; 100k or "until first clear" cost about 1 point against D′, and 150k costs about 4.
   - (c) blends starting at t=0 tie D′ (−0.09 to +0.01); starting later costs more.
   - (d) each extra 10% of pure round robin costs about 0.2.
   - On UCB, every variant equals UCB, because UCB already starts every game first.
2. **What the blends buy is the start schedule.** They bring starts close to Franzen's at almost no simulated cost. But the real D′ diagnosis put only about 0.5 points on late starts, so this hedges a small risk, not the stuck-level loss.
3. **Neither Franzen nor D′ treats depth as difficulty.** Both use the level index only as the current level's score weight, plus a tail by number of levels left.
   - Valuing a game by the weight of the levels still ahead (its "unlock value") loses 5–9 points on D′ and up to 1.5 on Franzen: stuck shallow games hog the slots.
   - The expected-remaining-weighted-score policy wins with the ledger (+0.44 against D′ at 110 games) but loses 1.6–1.9 without it, and leaves 14–16 games never started. It is not robust.
4. **D′'s own stall patience (about 7.5 × level cost) is the peak** of the patience sweep, including under the stuck-floor model. Shrinking a game's own cost estimate toward the population is neutral (+0.04).

**Recommendation**
- **Keep Franzen for submissions.**
- **Don't spend a real run on a round-robin or minimum-budget variant for its score.** In the simulator the best ones only tie D′.
- **The most informative real run is D′ + ledger without the #2 history-state fix,** paired with Franzen + ledger on the same notebook. That separates the scheduler from the sandbox-error lead found in phase 3.
- **Only if D′ then ties or beats Franzen,** submit the 0–80% blend (`blend-lin:0-0.8:dprime`) instead of plain D′ for its Franzen-like starts, or the 0–40% blend (`blend-lin:0-0.4:dprime`) as the second choice. Neither fixes the roughly 5 games D′'s late tail fade leaves unstarted at 110 games; fresh-first does, at −0.2 to −0.6.

I did not edit `arms.md`, since my instructions limited me to the worktree; these results still need logging there.

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/sim-v2/analysis/`:
- sim2_RESULTS.md
- sim2_policies.py
- explore_sim2.py
- test_sim2_policies.py
- sim2.py
- README.md

The raw replicate data (not committed) is in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/explore-out/` (`screen.jsonl`, `final.jsonl`, `robust.jsonl`).
