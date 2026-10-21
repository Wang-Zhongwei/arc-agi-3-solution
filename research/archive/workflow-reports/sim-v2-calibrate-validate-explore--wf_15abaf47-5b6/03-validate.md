# sim-v2-calibrate-validate-explore: validate

Workflow run `wf_15abaf47-5b6`, phase Validate. Calibrate a continuous-time scheduler simulator from real runs, validate it against held-out real policies (UCB, D-prime), then explore annealed round-robin to D-prime/UCB schedulers

---

I validated simulator v2 against held-out runs. It predicts UCB and the slot-count runs well, but it does not reproduce D′'s real loss: in every variant fitted on the Franzen runs it ranks D′ above Franzen, while the real D′ run came in 7.5 points below. I could not find a scheduling mechanism the simulator is missing. The whole D′ gap is in which levels got stuck on the short games. The simulator treats level costs as independent of the scheduler, so it cannot tell luck from a D′-caused effect.

Six new commits are on `arm/sim-v2` (ed3ec03 … 0633924); the working tree is clean. All 29 unit tests pass, both with plain `python3` and under pytest. Nothing was pushed, and the main repo's working tree was not touched. Total simulation CPU was about 18 min, one job at a time under `nice -n 10`.

**Resumed work.** Both leftover pieces were sound and are committed:
- the `sim2.py` change that fits on the Franzen runs only (`--calib franzen`, now the default; `validate_sim2.py` pinned to `--calib all`);
- `holdout_sim2.py`, since extended with the arrival, stuck-level, D′, noise and robustness sections.

**Calibration set:** control, temp06 and audit60k without the ledger; ledger-on-trim r1 and r2 with it. Everything else is held out.

### Iterations

| # | Change | Evidence from the calibration runs | Effect on held-out runs | Kept? |
|---|---|---|---|---|
| 0 | State as resumed | LOO z mean +0.12 | D′ predicted 52.15 (z −1.74); D′ panel 5.59 (z +2.88); D′−F ledger +1.22 | base |
| 1 | Arrival order: U(0,3), plus 20% of games up to 10 places later | The old U(0,3) model made 3 of the 5 real first waves impossible; log-likelihood −16.7 vs −38.9 | D′ panel 6.31 (z +2.26); D′ last start 75.9 min (real 71.6, was 79.4); D′−F unchanged (+1.25) | yes |
| 2 | Stuck floor 0.04 (best per-level log score) | Level log score −2.414 vs −2.511, but LOO run scores 7.0 too low (z +1.29); in-sample 38.8 / 43.9 vs real 45.4 / 50.9 | D′ 43.93 matches only because every run drops (UCB 35.8, 14 slots 40.3) | no, reverted |
| 3 | Stuck shrinkage toward the depth mean (total stuck mass kept) | Log score −2.461, but LOO run scores still 1.6 too low (z +0.28) | D′−F +0.96 | no |

Iterations 2 and 3 stay available as `--stuck-floor` and `--stuck-shrink` for sensitivity checks. Iteration 2 was rejected on its calibration-run bias; its held-out numbers came out of the same run and point the same way.

### Check 1: Franzen, 25 games, 121 min (final model)

| Condition | Real (sd) | Sim (sd) | Levels real / sim | Tokens M | Starts >30 min | Last start | Per-game rank corr. |
|---|---|---|---|---|---|---|---|
| No ledger | 45.43 (1.16) | 45.99 (3.78) | 104 / 104.7 | 4.60 / 4.51 | 3.3 / 4.3 | 32.2 / 32.5 | +0.98 |
| Ledger | 50.85 (2.42) | 50.95 (4.66) | 112 / 112.1 | 4.52 / 4.48 | 2.5 / 4.5 | 31.7 / 32.7 | +0.96 |

- **Leave-one-out:** z = +0.07, +0.29, −0.24, +0.91, −0.40 (mean +0.13, sd 0.46); per-game rank correlation 0.61–0.84.
- **Replaying each run's own level costs:** error RMSE 1.08.

### Check 2: held-out runs (no retuning)

| Run | Real | Predicted (sd) | z | Replay of own costs | Rank corr. | Starts >30 min | Last start (min) |
|---|---|---|---|---|---|---|---|
| fz-ucb | 43.84 | 43.54 (4.29) | +0.07 | 44.69 | +0.79 | 4 / 4.3 | 33.0 / 32.5 |
| v4-stack-v1 (D′) | 43.32 | 52.20 (5.09) | −1.74 | 43.99 | +0.69 | 10 / 8.7 | 71.6 / 75.9 |
| fz-streams14 | 46.38 | 48.44 (4.25) | −0.48 | 46.60 | +0.84 | 0 / 0 | 14.9 / 15.7 |
| fz-ledger-streams12 | 48.09 | 54.81 (5.03) | −1.34 | 49.28 | +0.86 | 0 / 0 | 27.1 / 27.4 |
| Panel, control | 5.13 | 4.98 (0.73) | +0.20 | 4.75 | +0.94 | | |
| Panel, easy-first | 7.51 | 5.47 (1.19) | +1.71 | 6.05 | +0.78 | | |
| Panel, D′ | 10.77 | 6.31 (1.97) | +2.26 | 6.81 | +0.65 | | |

### Key test

| Comparison | Sim difference (se) | Real difference | z, sim noise | z, real spread (sd 1.69) |
|---|---|---|---|---|
| D′ − Franzen, ledger | +1.25 (0.15) | −7.53 | −1.45 | −4.25 |
| UCB − Franzen, no ledger | −2.46 (0.14) | −1.59 | +0.18 | +0.45 |

- **UCB:** passes; the simulator puts it below Franzen, as reality did.
- **D′:** fails. Across 12 modelling variants fitted on the Franzen runs, D′ − Franzen stays +0.05 to +1.29 on ledger costs and +1.0 to +2.8 without. Only fitting on every run, the held-out runs' costs included, brings it to +0.17.

### Where the D′ gap sits

| Candidate mechanism | Real | Simulated | Finding |
|---|---|---|---|
| Prefill cost or pauses (compute lost) | 4.62M tokens (Franzen 4.51 / 4.53) | 4.48M | Ruled out: D′ lost no compute |
| Late starts (9 games delayed >5 min) | −0.50 pts | −0.52 pts | Reproduced, small |
| Switch-outs / never re-admitted | 43 / 9 | 38.2 / 8.3 | Reproduced |
| Arrival-order variants | – | +1.22 to +1.25 | No effect |
| The other 12 short (6–7 level) games | −6.65 pts | +2.04 pts | **The whole gap** |
| Tokens sunk into levels never cleared | 1.88M | 1.33M (real is at the 99.5th percentile) | |

- **Replay of the D′ run's costs:** D′ 43.99, Franzen 44.76 on the same costs. That Franzen score is at the 10th percentile of sampled runs.
- **The stuck levels:** tr87 L2 (all 8 other runs cleared it in 4.7–36k tokens) and m0r0 L3 have probabilities of 0.04 and 0.01 under the ledger level model. D′'s count of such "surprise" stucks (2) is normal for a new run (UCB, 14-slot and 12-slot runs had 3 each); they just landed on high-value short games.
- **A possible non-scheduler cause, checked by hand:** the D′ run also carried the #2 history-state fix. Its sandbox `IndexError` rate was 0.025 per step against 0.009 in r2, and `KeyError` 0.028 against 0.011. That is one pair of runs, so treat it as a lead only.

### Noise

The simulated run-to-run sd is 3.8 (no ledger) and 4.7 (ledger), against a real spread of 1.16 and 2.42 (chi-square P = 0.07). Per-game spread matches (22.4 real vs 21.7 simulated), but real games offset each other within a run and simulated games don't (permutation P = 0.06). The cause is unresolved. Across all 9 full runs, the z-scores have mean −0.32 and sd 0.77.

### Trust verdict

- **Trust it for:** given a run's own level costs, the scoring, scheduling, serving and context mechanics (within 1.8 points on every run). Also for throughput, slot-count, context and Franzen-family priority changes, and for UCB-type comparisons.
- **Don't trust it for:** ranking policies that reallocate compute among games or reorder starts (D′-like). It ranked D′ above Franzen in every variant and the only full-length D′ run came in 7.5 below. Its D′ edge also shrinks from +1.25 to +0.05 as stuck levels become more common.
- **Its own error bars are probably about 2× too wide**, so "within noise" calls from them are too lenient.
- **Rule:** treat simulated differences under about 2 points as unresolved, and confirm any simulated win for a reallocation policy with paired real runs.
- **Biggest open limit:** it assumes level costs don't depend on the scheduler, which cannot be tested here.

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/sim-v2/analysis/`:
- sim2_VALIDATION.md
- holdout_sim2.py
- sim2.py
- test_sim2.py
- validate_sim2.py
- README.md
