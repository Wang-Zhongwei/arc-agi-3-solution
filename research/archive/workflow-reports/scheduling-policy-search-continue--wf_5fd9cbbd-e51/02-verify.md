# scheduling-policy-search-continue: verify

Workflow run `wf_5fd9cbbd-e51`, phase Verify. Continue the scheduling policy simulation from existing commits on arm/sched-policies, then verify the top results

---

**Verdict: none of the three survives as a robust replacement for D′.** `dl-hRg:pn0.4` and `dl-hQ:pn0` fail clearly. `aff-v1:nofresh` is the only one that holds up, and only conditionally.

My driver reproduces all 9 reported base cells exactly (never 0 / 0.4 / 0.8). I changed no repo file (`git status` is clean). Everything below is paired against D′ at the same setting.

## (1) Runtime observability: all three pass

- **Code read.**
  - `dl-*` use only level, total levels, tokens on level, mean cleared-level cost, and a table fitted on other runs. `dl-hQ` is a static function of those four, with no clock and no actions. `dl-hRg` also reads R/G.
  - `aff-v1:nofresh` is a closed form of level, N, actions, tokens and mean cleared-level cost. It has no tables.
- **Runtime tripwire.**
  - Method: full 110-game sims at never 0 and 0.8, where any policy read of an uncleared level raises, plus planted-leak controls.
  - Result: all three pass; the four planted leaks were caught. Only the simulator's own action proxy `_level_actions` reads the current level.
  - File: `verify/tripwire.py`.
- **Tables fitted leave-one-game-out instead of leave-one-run-out.** No own-game leak: the leads do not shrink.

  | Policy | Leave-one-game-out (never 0 / 0.4 / 0.8) | Original leave-one-run-out |
  |---|---|---|
  | `dl-hRg:pn0.4` | +2.00 / +1.13 / +0.11 | +1.81 / +0.83 / −0.25 |
  | `dl-hQ:pn0` | +2.86 / +1.28 / −0.49 | +1.94 / +0.54 / −1.06 |

- **Remaining optimism that is not a leak.**
  - The `dl` tables use the generator's own Kaplan-Meier law, and `pn0.4` matches the simulated never value 0.4 exactly. That is an oracle match.
  - Config selection (best of 337 on the same seeds) costs about 0.1 to 0.3. On fresh seeds (reps 1000–1199) the three score:

    | Policy | Fresh seeds (never 0 / 0.4 / 0.8) |
    |---|---|
    | `dl-hRg:pn0.4` | +1.56 / +0.53 / −0.62 |
    | `dl-hQ:pn0` | +1.77 / +0.27 / −1.31 |
    | `aff-v1:nofresh` | +0.86 / +0.37 / −0.16 |

## (2) Robustness (200 reps unless noted; vs D′, never 0 / 0.4 / 0.8; standard error 0.06 to 0.17)

| Setting | `dl-hRg:pn0.4` | `dl-hQ:pn0` | `aff-v1:nofresh` |
|---|---|---|---|
| Base | +1.81 / +0.83 / −0.25 | +1.94 / +0.54 / −1.06 | +0.87 / +0.45 / −0.09 |
| Switch penalty 6000 | +1.55 / +0.35 / −1.18 | +3.50 / +1.58 / −0.62 | +2.03 / +1.23 / +0.21 |
| Quantum 15k | +1.66 / +0.78 / −0.05 | +2.25 / +0.99 / −0.30 | +0.43 / +0.14 / −0.08 |
| Quantum 40k | +1.00 / **−1.41 / −3.30** | +1.33 / −0.25 / **−2.06** | +0.58 / +0.10 / −0.46 |
| Arrival-order starts, vs D′ ranked starts | −0.22 / −0.14 / −0.31 | +0.55 / +0.18 / −0.29 | −0.61 / −0.28 / −0.14 |
| Arrival-order starts, vs D′ also arrival-order | +0.50 / +0.17 / −0.20 | +1.26 / +0.50 / −0.18 | +0.11 / +0.03 / −0.04 |
| Observed actions + penalty 6000 | +1.29 / −0.06 / −1.60 | +3.25 / +1.18 / −1.05 | **+1.91 / +1.28 / +0.56** |
| Observed actions + quantum 40k (never 0.4 / 0.8 only) | −1.76 / −3.63 | −0.60 / −2.39 | +0.15 / −0.18 |

Arrival-order starts mean every unstarted game starts before any ranking. D′ itself loses 0.71 / 0.31 / 0.11 when forced into arrival order. The earlier "no change" held only against Franzen.

Two further stresses (100 reps, never 0.4 unless noted):

- **Token-rate error (`dl-hRg` only; the policy sees R × 0.6 or × 1.6).**
  - R × 0.6: +1.19 / +0.46 / −0.63.
  - R × 1.6: +1.06 / **−2.13 / −4.77**.
- **Level-cost scale (all costs × s; tables and constants unchanged).** All three lose to D′ once costs move off the calibration (`dl-hRg` / `dl-hQ` / `aff-v1:nofresh`):

  | Cost scale | `dl-hRg:pn0.4` | `dl-hQ:pn0` | `aff-v1:nofresh` |
  |---|---|---|---|
  | × 0.75 | −0.24 | −0.17 | −0.35 |
  | × 1.25 | −0.73 | −0.66 | −0.49 |
  | × 1.5 | −2.31 | −2.26 | −1.53 |
  | × 2 | −2.62 | −2.38 | −0.81 |

  At × 2 across never 0 / 0.8: `dl-hRg` −2.22 / −3.03, `dl-hQ` −1.89 / −2.86, `aff-v1:nofresh` −0.72 / −0.89. If actions scale with cost, × 1.5 gives −1.20 / −1.15 / −0.98 and × 2 gives −0.30 / −0.07 / −0.10. At × 1.5 with penalty 6000: −2.16 / −1.44 / −0.79.

  D′ stays +5.5 to +5.8 ahead of Franzen at × 1.5 and × 2 (never 0.4), against +2.56 at × 1. Simulated means fall to about 32–39, nearer the real leaderboard score of 27.89.

**Per policy:**
- **`dl-hRg:pn0.4`: does not survive.** It is negative at never 0.8 in nearly every setting. Its 0.4 lead flips at quantum 40k, a token-rate overestimate, cost × 1.25 or more, and observed actions with a 6000 penalty.
- **`dl-hQ:pn0`: does not survive.** It has the biggest gains when switching is costly (+3.50 / +1.58 at penalty 6000). It is negative at never 0.8 everywhere, flips at quantum 40k, and loses 0.7 to 2.9 once costs are scaled up.
- **`aff-v1:nofresh`: partial survival.**
  - Holds: it never trails D′ by more than 0.1 at the calibrated base. With a switch penalty it beats D′ at all three never values (+2.03 / +1.23 / +0.21 at 6000, and +1.91 / +1.28 / +0.56 with observed actions).
  - Fails: it ties D′ under arrival-order starts (+0.03 against D′ also arrival-order at 0.4). It loses 0.46 at quantum 40k and never 0.8, and 0.5 to 1.5 at cost × 1.25 or more.
  - Verdict: a small conditional edge (about 0 to +1.3, and only with a switch cost of at least 3k tokens and costs near the data), not a robust win.
- **Bonus, `end-srpt-km:e0.3,f0.03` at penalty 3000.** It ties D′ (+0.19 / +0.27 / +0.23 at base; +0.33 / −0.01 at quantum 40k at never 0.4 / 0.8). It loses 1.06 at cost × 2 and 1.16 under arrival-order starts.

The "never 0.8" cells are where the `dl` family loses. Real runs have a standard deviation of about 3.9 (Franzen copies), so leads of 1 point or less cannot be measured with one real submission.

## (3) Implementability in `_PriorityGate`

The gate ranks by `_snapshot_priority(snapshot, now)` and `_pump` re-prices all waiters. The snapshot carries level, actions on level, tokens on level, total levels and mean cleared-level cost. The harness runs 110 games as live threads on 10 slots.

- **`aff-v1:nofresh`: yes, trivially.**
  - Add a policy branch in `_snapshot_priority` and in the fresh-game branch of `solver.py` (`play()`, which ranks unstarted games via the D′ branch).
  - Extend the `level_cost` and cleared-token tracking from "easy or D′" to the new flag, and compute `total_levels` unconditionally.
  - Use an integer scale of about ×10⁴ so priorities stay below the 1,000,000 band.
- **`dl-hQ:pn0`: yes.** It needs one embedded lookup table (total levels × level × cost class × tokens grid, about 3k floats) fitted from `levels.csv`. It needs no clock, G or actions.
- **`dl-hRg:pn0.4`: yes, with new plumbing.**
  - It needs tokens left over all slots and the count of live games.
  - Tokens left needs a token-rate estimate, because the gate has none. Deadline, now and slot count are available.
  - Live games would be counted from `acquire` minus `release` calls.
  - That rate error is exactly the stress that flips it (R × 1.6).

**Recommendation:** keep D′. The only candidate worth a real A/B is `aff-v1:nofresh`, and only if a switch is measured to cost 3k or more tokens of lost slot time.

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/verify/`:
- `vrun.py` (driver)
- `tripwire.py`
- `results.jsonl` (raw results)
- `pivot.py`
- `run_verify.sh`, `run_verify2.sh`, `run3.sh`, `run4.sh`
- `log.txt`, `log2.txt`, `log3.txt`, `log4.txt`
