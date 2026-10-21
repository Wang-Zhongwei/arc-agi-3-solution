# sim-v2-calibrate-validate-explore: verify

Workflow run `wf_15abaf47-5b6`, phase Verify. Calibrate a continuous-time scheduler simulator from real runs, validate it against held-out real policies (UCB, D-prime), then explore annealed round-robin to D-prime/UCB schedulers

---

PHASE 5 SKEPTIC VERDICT

Summary: the simulator's held-out data path is clean. The D′-family's sign over Franzen survives every perturbation I tried. But that gain is unvalidated, and the differences inside the D′ family are noise. I recommend keeping Franzen. Nothing was committed, pushed or edited. The `sim-v2` worktree and the main repo are clean. Total sim CPU was about 25 min, one job at a time under `nice -n 10`.

**1. Were the held-out predictions tuned on the held-out runs?**

Mostly no, with one bug and one weak spot.

- **Data path is clean.**
  - I deleted every held-out run's rows from `levels`, `requests`, `segments`, `reqstats` and `serve_decode`, keeping the 5 calibration runs.
  - Predictions were bit-identical on four held-out setups (fz-ucb, v4-stack-v1, fz-streams14, the D′ panel) over 25 seeds each. Nothing held-out reaches a fitted piece.
- **Independent re-run, fresh seeds, 200 reps:**
  - D′ 52.35 vs Franzen 51.25 on v4-stack-v1's setup. Only 4.0% of simulated D′ runs score ≤ 43.32.
  - UCB 42.78 vs Franzen 45.25.
  - Both match the report's tables. The 29 and 11 unit tests pass.
- **Changes made after the held-out numbers were visible:**
  - The arrival model (commit 630301d) was kept on first-wave likelihood. Its effect on D′ − Franzen is +0.03.
  - `deep_pool=1` was introduced in phase 4. I re-ran all 9 runs with it. It helps UCB (−1.0 to +0.35 error) but worsens D′, 14 slots and 12 slots. RMSE over the 9 runs goes from 3.95 to 4.21, so it was not selected for held-out fit.
  - The stuck-floor and shrink options were tried and rejected even though floor 0.04 would have matched the D′ run.
- **Weak spot: the pass is low-power.**
  - The sim's own sd is 4–5, so z-scores tolerate errors of 8 points. UCB is the only passing held-out test, and its sim-vs-real gap (−2.5 vs −1.6) has a real spread near 2.
  - Absolute held-out errors (predicted minus real): UCB −1.0, 14 slots +1.9, 12 slots +6.3, D′ +8.3. The mean is +3.9, against −0.6 for leave-one-out on the calibration runs.
  - Both held-out ledger runs are over-predicted by 6–8 points, so the 2-run ledger cost pool is probably optimistic.
- **Bug in the never-cleared check.**
  - `tail="pooled"` is a no-op in the ledger condition. The pooled draw only goes beyond `km.max_time`, and the KM already contains the prior's largest attempt. It converted 0 of 4,133 unclearable draws (7.5% of draws) for the ledger, and 27% of the 10.1% unclearable for no-ledger.
  - So the "pooled tail" rows in `sim2_VALIDATION.md` section F (ledger D′ − Franzen +1.25, identical to default) tested nothing.
  - I replaced it with a pooled-clear-beyond-own-max draw (below).

**2. Are the gains robust to the uncertain calibration pieces?**

The sign of every D′-family gain over Franzen is robust. The size is not. Their differences from D′ are noise.

I ran paired sweeps with common random numbers:
- 150 reps at the 25-game setting (121 min).
- 20 reps at the 110-game setting (532 min).
- Fresh seeds, paired with Franzen on the same draws.

Perturbations (26 with ledger costs, 24 without; the 110-game setting got 8, with ledger and without):
- client pause ×0.5 / ×2
- re-prefill ×0.5 / ×2 / ×4
- all prefill ×0.5 / ×2
- level-cost backoff: by-depth prior, prior at every level, prior weight 0 / 3, prior only below 2 attempts
- never-cleared: pooled-beyond-own-max clears, floor 0.02 / 0.04 / 0.08, shrink 0.5
- leave-one-run-out of each calibration run, including each ledger run
- two combined worst-case runs
- costlier / stuck-prone "frontier" levels the runs barely observed
- clears beyond 60k tokens made 1.5× to 8× slower

All numbers below are paired differences in score points. "n/n positive" means the sign held in every perturbation run.

| Scheduler | vs Franzen | vs D′ |
|---|---|---|
| D′ (harness) | Offline, ledger: +1.12 base (se 0.20), range +0.12 to +2.25 (floor 0.08 and gap ×2), 26/26 positive. Offline, no ledger: +2.42, range +1.02 to +3.76. 110 games: +2.81 / +2.68, ranges +0.93 to +3.25 and +1.54 to +3.02. | reference |
| dprime-pat:7.5 | Offline, ledger +1.19, 26/26 positive (+0.25 to +2.24). | Ledger +0.07 (−0.06 to +0.30). No ledger +0.11 (−0.39 to +0.32; −0.39 is the exclude-control run). Not run at 110 games. |
| dprime-shrink | Offline, ledger +1.22, 26/26 positive (+0.29 to +2.30). | Ledger +0.10 (+0.05 to +0.31). No ledger +0.02 (−0.05 to +0.15). 110 games: +0.28 ledger, −0.24 no ledger, so the sign flips with the cost pool. |
| blend-lin:0-0.4:dprime | Offline, ledger +1.00, 26/26 positive (+0.05 to +2.16). | Ledger −0.12 (−0.30 to +0.11). No ledger −0.06. 110 games within ±0.2. |
| blend-lin:0-0.8:dprime (the report's recommendation) | Offline, ledger +1.16, 26/26 positive (+0.17 to +2.27). | Ledger +0.04 (−0.32 to +0.19). No ledger −0.13 (−0.66 to +0.07). 110 games −0.02 / +0.02. |
| dprime-ff (fresh-first D′) | Offline, ledger +0.95, 26/26 positive. | Ledger −0.17, no ledger −0.33 (down to −0.92). 110 games −0.58 / −0.51. Costs score but starts every game. |

The D′ edge shrinks as unclearable mass rises:
- Ledger D′ − Franzen falls from +1.12 to +0.41 at floor 0.04 and +0.12 at floor 0.08.
- It also falls to +0.21 with prior weight 0 and +0.15 with the prior only below 2 attempts.
- My corrected pooled-tail variant gives +1.95 with ledger and +3.2 without.
- Dropping a ledger run leaves +0.63 (without r1) or +1.41 (without r2).

The D′ edge survives everything else I perturbed:
- Pessimistic frontier levels still leave +1.2 to +2.6.
- Slowing clears beyond 60k by 8× (Franzen falls to about 38–40) still leaves +0.6 / +0.9. So D′'s patience is not the source of its simulated edge.
- Pause scaling moves it between +0.47 and +2.25. Prefill scaling moves it by less than 0.4.
- UCB stays below Franzen in 25 of 26 ledger perturbations (median −1.1) and 22 of 24 no-ledger ones (median −2.2).

Within the D′ family, the best variants sit 0.04–0.12 above D′. They flip sign across perturbations and are 8–50 times below the 2-point rule. The phase-4 screen's +0.20 for pat7.5 shrank to +0.06, which is winner's curse. The starts-schedule effect is real: late starts at 110 games are 15 for the 0-0.8 blend vs 46–50 for D′ and 8.4 for Franzen. But never-started games stay at about 5.5–6.2 for D′ and both blends, against 0 for Franzen and D′-ff.

At 110 games the D′ family also clears fewer levels than Franzen (443 vs 450 with ledger, 412 vs 416 without) yet scores higher, so the gain is a depth-weighting effect.

The perturbations cannot explain the real D′ loss. Whatever caused it sits outside the sim's structure. Candidates are luck, the #2 history-state fix, or scheduler-dependent level costs.

**3. Implementable in `_PriorityGate` with runtime data?**

All are implementable, and none reads hidden state. The twin-game test in `test_sim2_policies.py` covers pricing and holds. The sim's `Gate` matches the harness `_pump` and handover semantics. The sim loads the same `priority_scheduler.py` as arm/stack-v1 (same hash).

- **D′:** already there (`ARC3_PRIORITY_POLICY=dprime`, fade 0.4). Its snapshot fields match the sim's.
- **D′ fresh-first:** a one-line flag in `solver.py play()` (lines ~616–631), using the non-D′ acquire.
- **dprime-shrink:** `level_cost` is already in `PrioritySnapshot`. It needs a shrunk geometric-mean cost, so the agent must track per-level costs or a log-sum next to `_easy_cleared_*` (tool_agent.py ~3918). About 10 lines, no gate change.
- **dprime-pat:7.5:** the same, plus `D_PRIME["rel_pow"]=1.0`.
- **Blends:** needs more.
  - Add total generated tokens to the snapshot (`_session_generated_tokens`, available per game).
  - `_pump` (tool_agent.py ~2101) must rank all waiters jointly (percentile ranks) and use w(t) from `_clock_start` / `_clock_deadline`. About 40–50 lines.
  - The price scale (~1e9 in the sim) clashes with `_PRIORITY_BAND=1e6` in the `_priority_next` request-priority hint. Rescale to below 1e6. This matters only if the server queue ever forms.

**Verdict per scheduler**

- **D′:** the sim gain is sign-robust across all perturbations, so I would not call it an artifact of calibration errors. But the one real full run is −7.5 on a real spread of 1.7–2.4 (the spread rests on 3 degrees of freedom; with a wider sd, z is about −1.7 rather than −4.3). A rough Bayes combination (prior +1.25 ± 2, one real pair at −7.5 ± 3.4) gives about −1 ± 1.7, or −0.2 ± 1.8 if the #2 fix can explain a few points. The sim gain is not evidence of improvement.
- **dprime-pat:7.5 and dprime-shrink:** reject. They sit 0.0–0.3 above D′ with sign flips (pat −0.39 under exclude-control; shrink −0.24 at 110 games without ledger). Cheap to build, but unmeasurable.
- **blend-lin:0-0.4:** reject. It ties D′ (−0.12 to +0.1) and leaves 35 late starts at 110 games.
- **blend-lin:0-0.8 (the report's recommendation):** keep only as a conditional hedge. It ties D′ (±0.3) and cuts late starts, but does not fix never-started games. If you want that fixed, D′-ff does it at −0.2 to −1.0. The start-schedule effect was worth about 0.5 in the real run.
- **Run design:** the report's suggested run (D′ + ledger without the #2 fix, against Franzen + ledger) can tell whether the −7.5 was the scheduler. It cannot confirm a +1 gain: a single paired run has sd of about 3.4, so confirming +1.1 would take about 75 pairs. If D′ were even moderately likely (above about 18%) to be truly harmful at −5, its expected value is negative against the simulated +1.1.

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/skeptic/`:
- `leak_test.py`
- `repro_c.py`
- `deeppool.py`
- `robust.py`
- `robust2.py`
- `robust3.py`
- `robust4.py`
- `summ.py`
- `agg.py`
- Raw data: `off.jsonl`, `off2.jsonl`, `comp.jsonl`, `fr.jsonl`, `tail.jsonl`
