# scheduling-policy-search: research:prior

Workflow run `wf_f55d767d-e13`, phase Research. Find new slot-scheduling policies (competitors, theory, prior work), simulate them against D-prime, verify the best

---

**Paths:** `ALLOC`=/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/gb10-wt/alloc-policies/ARC3-Inference, `ALLOCN`=…/alloc-policies/notes, `FZ`=/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/franzen/ARC3-Inference, `ATLAS`=/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/reports/run-atlas. All read-only.

## A. Branch gb10/alloc-policies (tip 9405c1a; Wang-fork harness: 25 games, 8 slots, 132 min, 300 s slices)
Shared notation: N levels, k cleared, share(k)=k/(N(N+1)/2), next-level share=(k+1)/(N(N+1)/2) (`ALLOC/inference/framework/allocation.py:105`). Every policy keeps a warm-up slice per game and the exploration floor. While exploration seconds < f·total, the least-served game gets the slice (allocation.py:184-226). f was raised 0.25→0.375 in 4029245.

| Policy | Exact rule (allocation.py) | Inputs |
|---|---|---|
| ucb (default) | priority=0.5·(lifetime+recent) weighted-levels per generated token (recent window 32k, :131-172), score=priority+c·√(ln(1+elapsed_min)/actions), c=1e-4 (:98, :579) | tokens, level-weighted reward, action count |
| ucb_marginal (4029245) | UCB priority × N·nextshare = 2(k+1)/(N+1) (:627-642) | adds k, N |
| ucb_knapsack (4029245, never ran) | density=share/max(1, share·seconds_per_weight). Cost used the wrong units, so share cancelled and it ranked by lifetime rate. Now an alias for `knapsack` | seconds-per-weight |
| knapsack (4305704, :644) | argmax over candidates of V/q, q=min(300, time left). V=Σ_{j=1..N-k} (k+j)/(N(N+1)/2)·P(≥j levels in q), negative-binomial from a Gamma posterior (:430). Shape=0.5+Σexp(−(S−t_i)/1800); rate=0.5/λ_pool+1800(1−e^{−S/1800}); λ_pool=(Σlevels+1)/(Σservice+300) (:295-300, :362-415). Ties go to least-served | game service seconds S, level-completion times, N, pooled rate |
| thompson (:671) | same V/q, but λ drawn from the Gamma posterior | same |
| halving (:702) | rungs 25→13→8 (ceil(n/2), floor at slots). Per-game rung budget=remaining·slots/rungs_left/members. Least-served first within the rung; ranked by knapsack V/q; winners backfilled; idle slots filled (`halving_fill`) | same |
| exp3 (:794) | γ=min(1,√(K lnK/((e−1)H))), H=remaining·slots/q. Reward=min(1, Δshare/0.25 · quantum scaling). logw += γ/K·x/p | share earned per slice |

Key design points: the 1800 s e-fold on evidence retires stalled games with no timer. Waiting time is not aged. Won games stay in the pooled prior. The last slice is priced for the seconds actually left.

**Results (simulated only, `ALLOCN/allocation/policy-simulation.md`, `gb10-arms/alloc-policies.md`).** Traces: 8 full Wang-era runs. Each replicate replays a random full run, with the censored tail from Kaplan-Meier (KM) (`exp` or `none`). 400 paired replicates.
- Main model (replay, exp tail), mean at f=0.375 unless noted: ucb 12.84; ucb@.25 13.14; round-robin (explore=1) 11.55; ucb_marginal 13.04 (+0.19); **knapsack 14.00 (+1.16±0.07, boot +0.89 SD 0.40)**; thompson 13.27; halving 12.20; exp3 11.70.
- At lower f (main model): knapsack@.25 14.49 (+1.64), knapsack@0 14.62. thompson@0 14.67 (+1.82, trim 9.32); halving@.125 13.65 (trim 9.14).
- Robustness, knapsack@.375 − ucb@.375: replay/no-tail +0.09; KM/exp +0.64; KM/no-tail +0.30. Versus ucb@.25: +0.86/−0.06/+0.40/+0.28.
- Mean gain comes from wins (1.04 vs 0.67 per run). Cost: −0.7 games >0 (19.4 vs 20.1). Trimmed mean +0.43 to −0.22.
- With no tail, knapsack@0.125 or 0 loses up to 2.3 trim and 9 games >0, so the 0.375 floor was kept. exp3 is round robin with noise (0.07 wins). Round robin never wins a game.
- Hyperparameters were checked, not tuned: memory in {0..3600} moved the mean ≤0.5; a 4-pseudo-level prior cost up to 0.8 in the no-tail model.
- Caveats from the author: the gain depends on winners clearing levels beyond any observed service time (the tail assumption). Progress is assumed to depend only on service seconds. Variants are pooled as exchangeable. Score assumes efficiency 100.
- Validation: a differential test pins UCB parity (97bce2a). A race fix in `served_seconds()` (7c80bc4). The simulator reproduces its numbers exactly. Calibration: simulated ucb@.25 gives 13.14 against 13.51 real, which is partly circular.
- **Never run on real Kaggle.** The a1 "marginal" and a2 "knapsack" runs (1.18, 1.26 mean; 30-min budget) silently ran `uniform` (no `allocation_events.jsonl`; `ALLOCN/allocation/policy-simulation.md` §1). The proposed Kaggle arm was never launched.
- Also on the branch: `stall` (continuous-clock extension from a teammate's bundle, `StallAllocator`, allocation.py:29-95; `ALLOCN/allocation/stall-allocation-policy.md`) and `rolling_extension`. Neither has a score result.

## B. Branch franzen/base (priority gate, 10 slots, token-quantum trims; the D' setting)
- **UCB (ff0630c)**, `FZ/inference/agent/priority_scheduler.py:285-337`: index=(1−w)·life+w·recent + c·√(ln(1+T/1e5)/(1+n/1e5)), w=0.5, c=0.1. Reward=c(c+1)/(N(N+1)) (`level_weighted_reward`, :288). Rates are per 100k tokens, recent window 64k, scaled ×1e5 and capped at 400k. n=game tokens, T=total tokens. The gate, slots and "unstarted first" are unchanged.
  - Real run `fz-ucb` (arm `ucb`, `ARC3_UCB_C=0.1`): **43.84 mean / 38.95 trim / 24 >0 / 6 won** vs control 46.25 / 41.58 / 22 / 7. It sent 61% of tokens to games ending ≥50% complete vs 74% for Franzen (`ATLAS/findings.md:66`, `ATLAS/versions.md:79`). The atlas treats gaps under ~3 as noise (findings.md:80-83), so this is "not better", not proven worse.
- **Fitted index (474a0ab)**, priority_scheduler.py:345-433: index=P_soon(t)·(level·eff + cont)/(N(N+1)/2).
  - P_soon = KM P(clear in next 30k | t spent): (0,.53) (20k,.53) (30k,.52) (40k,.48) (60k,.34) (80k,.32) (100k,.29) (150k,.24), linearly interpolated.
  - eff=min(1,(25/(25+a))²).
  - cont=Σ_{j>k} Π P_next·j·0.9, with P_next 0.73/0.77/0.83 for 1/2/3 levels left and 0.86 for 4 or more. It fades with tail_fraction.
  - Pooled parameters only, fit on 869 attempts (`franzen/analysis/levels.csv`, `fit_levels.py`).
- **Stall swap (474a0ab)**, `FZ/inference/agent/tool_agent.py:8679`: yield the slot when ≥60k tokens have been spent on the level and the best already-started waiter has ≥2× my priority. Checked every 10k tokens; unstarted games are excluded. Cost: one re-prefill.
- **Replay simulator (9f64abb, `franzen/analysis/replay_scheduler.py`)** policies: round-robin, franzen, fitted, gittins, gittins+stall thresholds. Gittins index = max over windows τ≤150k (step 5k) of P_clear(τ)·value/E[tokens spent]. Rescoring reproduces all 175 real scores; replays are within ~1.5 points of real runs. Numbers were not recorded in the commit (see your results above).
- Real runs `fz-ledger-fitted` and `fz-ledger-stallswap` were pushed 2026-10-04 (runs.json, `batch_open: true`). No outputs are on disk (`franzen/kaggle/out/*` holds only the notebook and metadata). `ATLAS/findings.md:97-101` §8 says "being tested". So there is **no real-run result** for either.

## C. Other real-run scheduling evidence (`ATLAS/versions.md`, Wang era, noisy single runs)
- e0 uniform allocation 4.68 mean / 7 >0 and e1 uniform+predict 2.97, against UCB siblings at 13-19.
- explore 0.5: 14.04 vs memory-fixed 13.64. explore 0.25: 9.17 vs baseline 9.86.
- f1 "stop investing after score 35": 8.08 vs b1 17.74.
- Run-to-run noise: ±1.1 on the offline mean; single games swing 50-80 points (findings.md:78-83).

## D. Not yet tried in the 110-game replay sim (relative to your D' results)
1. Knapsack, Bayesian value/second: negative-binomial multi-level slice value and an evidence-decay stall detector. This is the closest relative of D', because both price depth.
2. Thompson draw (the best trimmed mean in the Wang-era simulation).
3. Successive halving, with rungs never below the slot count.
4. A least-served exploration floor (f=0.375, then 0.25 in the sim). It fits your finding that starts are low-yield quanta.
5. ucb_marginal's depth tilt of 2(k+1)/(N+1), roughly the M in D'.

Portability: all of the Wang-era policies are in service seconds and have no token-quantum or trim gate. Mapping them onto Franzen's gate needs tokens (or tokens/rate) in place of `service_seconds`.
