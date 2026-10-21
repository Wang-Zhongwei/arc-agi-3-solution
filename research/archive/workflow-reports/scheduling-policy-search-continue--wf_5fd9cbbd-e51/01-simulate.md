# scheduling-policy-search-continue: simulate

Workflow run `wf_5fd9cbbd-e51`, phase Simulate. Continue the scheduling policy simulation from existing commits on arm/sched-policies, then verify the top results

---

**Nothing beats D′ robustly.** Every candidate that beats it at never 0 and 0.4 loses or ties at never 0.8. Part of the best one's lead turned out to be a simulator artifact. The tests passed from the start (21/21, now 23/23), and no policy reads a level's cost or whether it can clear.

**What changed in this session.** The previous agent had already screened 121 configurations and run the finals before the network error, with the same committed code. I re-ran four of them and got identical numbers, so I reused those and screened the other 221 configurations. Sim CPU this session was about 27 min, run one process at a time with `nice -n 10`. With the previous agent's ~17 min the total is about 44 min, inside the 60-min budget.

**Leak check (step 2).** A new test builds twin games that look the same to the harness but differ in the current level's cost, whether it clears, and the levels after it. All configurations (now 337 plus 9 new M43 variants) rank, hold and size grants identically for both twins. The test fails when I plant a policy that reads the cost.

The one leak is in the shared simulator, not in any policy:
- **Action count.** When a stuck level is drawn as unclearable, its action count stops at 50 after 30k tokens. A clearable one keeps counting. D′'s action terms therefore slightly favour unclearable levels.
- **Fix kept off by default.** I added `--censored-actions observed`, where a stuck level gains actions at the rate its source run showed, so the count no longer reveals clearability. The default is unchanged and reproduces the cached D′ and Franzen scores exactly.
- **Second option.** `--resid-scale` makes stuck levels need 0.5× or 2× the remaining tokens the Kaplan-Meier (KM) curve draws. This tests policies that rank with that same curve.

**Screen (never 0.4, 40 reps; difference from D′ (wins) / difference from Franzen (wins)).** The `p3000`/`p6000` configurations and their baselines run at that switch penalty.

| Policy | vs D′ | vs Franzen |
|---|---|---|
| `dl-hRg:pn0.4` | +0.87 (32) | +3.26 (40) |
| `dl-hQ:pn0` | +0.69 (28) | +3.08 (39) |
| `aff-v1:nofresh` | +0.47 (29) | +2.86 (38) |
| `dl-hQ:pn0.4` | +0.47 (24) | +2.86 (39) |
| `end-srpt-km:e0.3,f0.03,p3000` | +0.45 (27) | +3.48 (39) |

Each other family's best result against D′:

| Family | Best vs D′ |
|---|---|
| additive offsets | +0.08 |
| NB knapsack | −0.04 |
| deadline chain, max over horizons | −0.12 |
| endgame with Beta hazard | −0.24 |
| chance-term swaps | −0.32 to −0.76 |
| 50k-first hybrid | −0.77 |
| ASHA | −0.78 |
| Wang UCB | −1.61 |
| deadline chain plus D′ tail | −1.99 |
| budget fade | −2.7 to −4.8 |
| FIFO waves | −4.0 |
| one-level deadline | −10.0 |
| M43 with unstarted games first | −12.5 |
| chain tail | −16 to −20 |
| M43 | −24.2 |

`dprime-float` scores −0.00 against D′, which confirms the port.

**Finals, no switch penalty (200 reps; vs D′ (wins) / vs Franzen (wins)).**

| Policy | never 0 | never 0.4 | never 0.8 |
|---|---|---|---|
| D′ vs Franzen | +5.12 (200) | +2.56 (194) | +0.94 (174) |
| `dl-hRg:pn0.4` | +1.81 (181) / +6.93 (200) | +0.83 (152) / +3.39 (200) | −0.25 (88) / +0.69 (140) |
| `dl-hQ:pn0` | +1.94 (183) / +7.07 (200) | +0.54 (129) / +3.10 (195) | −1.06 (42) / −0.12 (97) |
| `aff-v1:nofresh` | +0.87 (145) / +6.00 (200) | +0.45 (138) / +3.01 (196) | −0.09 (95) / +0.85 (162) |
| `dl-hQ:pn0.4` | +1.12 / +6.24 | +0.38 / +2.94 | −0.57 / +0.37 |
| `end-fox-km:e0.3,f0,p0` | +0.33 / +5.45 | +0.10 / +2.66 | −0.14 / +0.80 |
| `end-srpt-km:e0.3,f0.03` at penalty 0 | −0.19 / +4.93 | −0.05 / +2.51 | +0.07 / +1.01 |

**Finals at a 3,000-token switch penalty (200 reps; baselines pay it too).**

| Policy | never 0 | never 0.4 | never 0.8 |
|---|---|---|---|
| D′ vs Franzen | +6.16 (200) | +3.32 (199) | +1.33 (181) |
| `dl-hRg:pn0.4` | +1.76 (182) / +7.92 | +0.67 (138) / +3.99 | −0.70 (56) / +0.64 |
| `dl-hQ:pn0` | +2.87 (195) / +9.03 | +1.11 (162) / +4.43 | −0.88 (57) / +0.46 |
| `aff-v1:nofresh` | +1.64 (185) / +7.79 | +0.91 (163) / +4.24 | +0.06 (108) / +1.39 |
| `end-srpt-km:e0.3,f0.03,p3000` | +0.19 (115) / +6.35 | +0.27 (118) / +3.59 | +0.23 (125) / +1.56 |
| `end-fox-km:e0.3,f0.03,p3000` | +0.54 (144) / +6.70 | +0.14 (114) / +3.46 | −0.29 (72) / +1.04 |

**Sensitivity (100 reps; vs D′ (wins); the first row is D′ vs Franzen).**

| Policy | observed actions, never 0 | observed, 0.4 | observed, 0.8 | tail ×0.5, never 0.4 | tail ×2, never 0.4 |
|---|---|---|---|---|---|
| D′ vs Franzen | +4.60 | +3.11 | +1.85 | +3.23 | +2.44 |
| `dl-hRg:pn0.4` | +1.08 (81) | +0.27 (61) | −0.81 (21) | +0.39 (61) | +0.45 (67) |
| `dl-hQ:pn0` | +1.25 (80) | +0.06 (58) | −1.53 (12) | +0.37 (59) | +0.17 (55) |
| `aff-v1:nofresh` | +0.23 (53) | +0.22 (60) | −0.05 (52) | +0.22 (56) | +0.12 (53) |
| `end-srpt-km` freeze, 3k penalty | +0.03 | +0.08 | +0.07 | — | — |

**Does anything beat D′?** No.
- **Deadline chain (`dl-hRg`, `dl-hQ`).** Biggest gains at never 0, but it loses at 0.8. Once actions stop revealing clearability, D′ gains 0.5–0.9 on Franzen and the chain's lead at 0.4 drops from +0.83 to +0.27. Its never prior is a guess the run can't check, and it swings `dl-hRg` by 2.4 points (−1.50 with 0, +0.87 with 0.4).
- **`aff-v1:nofresh`.** The most robust candidate: it never trails D′ by more than 0.1 in any check. Its lead is small, only about +0.2 with observed actions.
- **Endgame SRPT with a 3% freeze.** The only configuration ahead at all three never values, and only at the 3k penalty (+0.19 to +0.27). Without the penalty, or with observed actions, it ties.
- **Why the winners win.** They are stickier than D′: they hand the slot back to the same game on 56–80% of trims, against D′'s 52%. They also avoid D′'s late churn, where 72% of trims after 80% of the run switch to a different game. That is why a switch penalty widens their lead.
- **Calibration point.** Affectify's author reports D′ about +1.5 ahead of his form v1 in his own simulator. Ours does not reproduce that.
- **Recommendation.** Keep D′. If a switch is known to cost a re-prefill, `aff-v1:nofresh` is the candidate worth a real A/B run.

**Where `policies.py` doesn't follow the design literally:**
- **Deadline chain.**
  - A stuck level is unclearable with probability p_never × π(t), where π(t) is the share of attempts still uncleared at t that their run ended on. The design's flat p_never would make 40% of fresh levels unclearable.
  - The token grid stops at 300k and the horizon at 16 trims. Actions are ignored.
  - The index is scaled so a fresh ten-level game starts at 1, not in D′ units. That only matters for the variant that adds D′'s tail.
- **Affectify v1.** The design asks for no unstarted-first gate; that is `aff-v1:nofresh`. `aff-v1` keeps the notebook's gate and scores −0.20.
- **M43.** Implemented as specified, but the tier ranking means games that never cleared a level never start (34 of 110 in one rep). I added `m43-fresh` with Franzen's start gate; it scores −12.5 instead of −24. Under the default action model the a > 60 throttle has no effect on score at 10 or 20 actions per grant.
- **Wang UCB.** The exploration floor counted one trim per exploration slice, so long slices explored 3–6× past the target share. Fixed in `efa1c29` and re-screened; the family still loses by 1.6 to 12.4.
- **ASHA.** A game's rung score is recorded at its first ranking after crossing the rung, not exactly at the rung. A game can't be promoted until `reduction` games have reached that rung.
- **NB-KNAP Thompson variants.** All draws come from one seeded generator, so results depend on rep order. They are still deterministic per run.
- **FIFO waves.** The cap applies per grant, and the game then requeues behind games with fewer grants instead of retiring.
- **`bud-c+fade`.** An extra diagnostic not in the design.

**`git log arm/replay-sim..HEAD`** (branch `arm/sched-policies`, not pushed):
```
774820b README: results of the 2026-10-05 policy sweep
487abc8 README: list the sweep tools and two simulator limits
0c5e123 M43: variant that starts every game first, as Franzen's gate does
3d5c3a3 Simulator: optional observed-rate actions and a scaled KM tail for stuck levels
efa1c29 WANG-UCB: count a whole slice against the exploration floor
81585a5 test_policies: every configuration ignores hidden level state
a58bb33 screen_policies.py: --penalty runs every configuration at one switch penalty
beb9c0c Add screen_policies.py: paired screening against D' and Franzen
09788b0 Add the candidate scheduling policies of the 2026-10-05 sweep
70c949e Simulator: run-level context, per-game grants and an additive incumbent offset
```

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/`:
- `wt/replay-sim/analysis/README.md` (results section added)
- `wt/replay-sim/analysis/policies.py`
- `wt/replay-sim/analysis/replay_scheduler.py`
- `wt/replay-sim/analysis/screen_policies.py`
- `wt/replay-sim/analysis/test_policies.py`
- `sched_results/`: raw outputs (`gap_screen.jsonl`, `final.jsonl`, `final_p3000.jsonl`, `sens.jsonl`, `misc.jsonl`, `screen_merged.json`, `trace.log`), the previous agent's results (`prev_*.jsonl`) and the baseline cache (`cache.json`)
