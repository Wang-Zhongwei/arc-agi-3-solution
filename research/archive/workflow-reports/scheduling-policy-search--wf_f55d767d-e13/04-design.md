# scheduling-policy-search: design

Workflow run `wf_f55d767d-e13`, phase Design. Find new slot-scheduling policies (competitors, theory, prior work), simulate them against D-prime, verify the best

---

NOTATION (all observable at runtime). l = level (1-based), N = total levels, m = max(0, N-l), t = tokens on this level, a = actions on this level, Q = 25k.
cbar = mean tokens per cleared level (30k if none cleared), rho = clip(cbar/30k, .5, 2), rem = fraction of run left.
R = rem*slots*minutes*60*rate (19.5M at start), G = unfinished games, R_g = R/G (177k at start), T_g = total tokens in the game.
norm = 55/(n(n+1)/2) with n clamped to 6..10, v(L) = (1+0.5(L-1))*norm, eta(a) = (300/(300+a))^2.5.
D' = A*M*C + B*phi with B = 16/14/10/0 for m >= 3/2/1/0 and phi = min(1, rem/0.4).
Port facts I checked in replay_scheduler.py and priority_scheduler.py:
- D' is run with fresh_first=False, so a fresh game is priced at 17.0 (l=1, a=t=0).
- A*M*C only spans about 0.25-3.5, while B is 10-16. D' therefore orders games by B class first, then A*M*C.
- Additive offsets of 0.25-1 are only 1.5-6% of B. Multiplicative bonuses on B-sized values are what caused the earlier hogging.
SIM HOOKS NEEDED:
- ctx gets R, G, T_total and elapsed.
- Game gets clear_tokens (token timestamps of each level-up) and a grants counter.
- simulate() gets an additive incumbent offset, a per-game quantum function, and hold(g, ctx).
- Fit every KM, hazard or NB table leave-one-run-out over the 7 runs in levels.csv. In-sample fits flatter any table-driven policy.
- Report the delta vs D' (not only vs Franzen) at never = 0/0.4/0.8 with 200 paired reps. Policies that depend on switching also run at switch_penalty = 3000.

--- A. MINIMAL D' EDITS ---
1. D'-BUD (budget-aware tail). priority = A*M*C + B_g*phi_g. Three variants:
   1a: phi_g = clip(R_g/(kappa*min(m,3)*c0), 0, 1), c0 = cbar or 30k. It replaces the clock fade phi = rem/0.4.
   1b: phi_g * min(1, rem/0.1), a short final fade on top of 1a.
   1c: chain tail B_g = beta0 * sum_{i=1..min(m,4)} v(l+i) * q^i * min(1, R_g/(i*c0)), with beta0 = 2.2 (equals 16 at l=3, N=10, R_g = inf).
   Caveat on 1c: B_g is about 11 at l=1 and 21 at l=7, against a flat 16 in D', so it tilts toward deep games. It also tests whether depth-increasing B helps.
   Uses: l, N, cbar, rem, active slots, rate, unfinished count. No hazards.
   Grid: kappa in {0.5, 1, 2}; q in {.73, .8, .86} (1c); beta0 in {1.6, 2.2, 2.8} (1c).
   Hypothesis: the clock fade penalizes cheap-level and costly-level games alike. A budget cut keeps patience where the remaining levels are affordable and releases slots from expensive deep games earlier. Decaying B by stall always lost, but this decay comes from global budget, not stall. phi_g is 1 until R_g < 90k, then falls.
2. D'-C (alternative chance term). Keep A, M and B*phi. Replace C = 0.1*max(.1, 1-a/115) + 0.9*max(.1, 1-t/T).
   2a, Beta/hyperbolic: C_t = 1/(1 + t/(lam*Q*rho)). Half-life is lam*Q*rho, so 125k at lam = 5, rho = 1. Heavy tail instead of a linear floor.
   2b, KM marginal-utility: C_t = h_d(t)/h_d(0), where h_d(t) = KM P(clear within next Q | stuck t) by depth bucket d (L1-2, L3-4, L5+).
     Shrink L5+ toward pooled with prior n = 20, because L5+ has few samples (.09 at 100k) against .44 for L3-4.
   2c, exponential: C_t = exp(-t/(tau*rho)).
   The action part keeps its 0.1 weight in every variant. Floor 0.1 as in D'.
   Grid: lam in {2, 5, 10}; tau in {75k, 150k}; the 2b table is fitted leave-one-run-out.
   Hypothesis: linear C hits its floor at about 200k and stall detection is crude. A smoother, data-shaped C re-ranks within B classes only. Expect +-1. If 2a matches D' with fewer constants, adopt it.
3. D'-ADD (additive categorical offsets, no multipliers). rank_i = D'_i + delta*1[i = incumbent and mid-level] - sigma*1[i fresh].
   Grid: delta in {0.25, 0.5, 1, 2} (D' float units); sigma in {0, 0.5, 1, 2}; switch_penalty in {0, 3000, 6000}.
   Hypothesis: multiplicative bonus x1.25 on a B of about 16 is +4 units, which is why it hogged slots (-10 to -22). An additive offset <= 1 only breaks near-ties toward the incumbent, so D's 34.6% switch share falls.
   The sigma offset shifts quanta from starts (0.042 per quantum) to repicks (0.065). The sim charges 0 per switch by default, so expect about 0 at penalty 0 and +0.5-1.5 at 3-6k. The sigma sweep may also lose the cheap early levels of unstarted games.
4. D'-END (endgame marginal-utility / SRPT switch). For rem < e, replace the whole priority with an expected-gain rate; for rem >= e use D' unchanged.
   4a, Fox: 100*v(l)*eta(a)*P_clear(t, H)/H.
   4b, SRPT: v(l)*eta(a)*P_clear(t, H)/m(t, H), with m = truncated mean residual life = integral_t^{t+H} S(u) du / S(t) from KM.
   H = min(Q, rem*minutes*60*rate). P_clear is the KM table by depth bucket. B = 0 because the tail is unrealisable.
   Optional freeze: for rem < f the incumbent always holds the slot (no handover).
   Grid: e in {.1, .2, .3}; f in {0, .03, .06}; P_clear source in {KM, Beta h = 2.5/(2.5 + 2.5*rho + t/Q)}.
   Hypothesis: D' switches 79% of quanta in the 80-90% decile. The rotation comes from the stall-decaying C while B has faded away. Ranking by expected gain within the actual remaining horizon should stop the churn.
   Freezing only pays if switching costs tokens, so run it at penalty 3000.

--- B. STRUCTURALLY DIFFERENT ---
5. NB-KNAP (Bayesian knapsack with a Gamma-Poisson game clear rate). This is Wang-era `knapsack`/`thompson` (+1.16 in its own 25-game sim) ported to token units.
   lam0 = (total clears + 1)/(total tokens + 30k), pooled over all games.
   Game posterior: shape s = s0 + sum over clears of exp(-(T_g - T_clear)/Mem); rate r = s0/lam0 + Mem*(1 - exp(-T_g/Mem)); lambda_g = s/r.
   V = norm * sum_{j=1..min(m+1,3)} (1+0.5(l+j-2)) * eta_bar * P(>=j clears in Q), with P from NB(s, r). Scale by 1/V_ref (V of a no-evidence game at l=1).
   priority = V/V_ref + B*phi (B undecayed) for 5a, or V/V_ref alone for 5b.
   5c, Thompson: draw lambda_g ~ Gamma(s, r) at each rank call.
   Uses: level-up history, tokens, N, l, pooled counts.
   Grid: Mem in {110k, 220k, inf}; s0 in {.5, 2}; mode in {mean, TS}; B in {on, off}.
   Hypothesis: game-level clear-rate evidence shrunk to the pool is a better "chance" than a per-level stall counter. 5a tests this with patience preserved. 5b, with B off, is expected to lose like fitted/Gittins; it is a control for whether B is what matters.
6. DEADLINE-CHAIN (rested rising-bandit / single-arm Whittle with a hard deadline). Offline table E[Y(h)] by (l, N bucket, t grid at 12.5k steps, rho in {.5, 1, 2}), built from a leave-one-run-out KM curve and cleared-duration pools.
   Residual time on the current level is drawn from S(.|>t) (never with prob p_never). Later levels are drawn from the pool of cleared durations times rho.
   Y(h) = sum_j v(j)*eta_bar*1[levels l..j all cleared within h tokens], over 200 common-random-number draws.
   index = K * max over h in {Q, 2Q, 4Q, ..., floor(R_g/Q)*Q} of E[Y(h)]/h, with K scaling the index into D' units.
   6b, fixed-H rising bandit: H in {Q, 100k, R_g}. 6c: single-level reward (no chain), to isolate cause. 6d: index + B*phi (chain plus D' tail).
   Uses: KM and duration pools, l, N, t, cbar, R_g.
   Grid: p_never in {0, .4}; H as above.
   Hypothesis: prior Gittins and fitted index lost because they used a one-level reward and no deadline. Comparing 6 with 6c separates the two causes. If the best of 6/6b is still worse than -5, then ranking by hazard-decayed rate is the problem.
   If 6d beats D', then the deadline chain gives a better B. Expect -1 to -4 for 6/6b, because the index is still hazard-decayed.
7. ASHA-D' (asynchronous successive halving over D'). Rungs r_k = r0*3^k tokens per game (k = 0, 1, 2, ...).
   Score Z_g = (sum_{j<l} j*eta_j/(N(N+1)/2))/(T_g + Q). A game that reaches rung k is promoted only if Z_g is in the top 1/3 of all games that have reached rung k.
   Unpromoted games are parked (ranked last, so they still run when slots would idle, or at x0.25 rank). Rank everything else by D'.
   Uses: tokens, cleared-level efficiencies, l, N.
   Grid: r0 in {25k, 50k, 75k}; reduction in {2, 3}; parked in {last, x0.25}.
   Hypothesis: prunes unlucky stuck games earlier than D''s linear C. But the objective is best-arm identification, not summed reward, so it should favor fast shallow games. Expect <= 0, in line with easy-first about 0 and the tail caps negative. Included as a structural-diversity control.
8. WANG-UCB (rellik13 / Wang `ucb` plus warm-up slices). Priority is the one-pool UCB with an exploration floor.
   r_life = (sum_{j<l} j/T_N)/T_g (T_N = N(N+1)/2); r_recent = the same over the last 32k tokens of the game.
   idx = 1e5 * (0.5*r_life + 0.5*r_recent) + c*sqrt(ln(1 + T_total/1e5)/(1 + T_g/1e5)).
   Warm-up: every game gets W tokens first. Optional exploration floor f: while exploration tokens < f * total, the least-served game gets the quantum.
   Slice: holds the slot for S tokens (hold callback), mimicking rellik's 2400 s slices.
   Uses: level-up history, tokens, N. Never used in the replay sim; the real run (43.84 vs 46.25) used c = 0.1 in Franzen units, not Wang's 1e-4 constants.
   Grid: c in {0, .03, .1, .3}; W in {25k, 50k, 75k}; f in {0, .25}; S in {25k, 75k, 150k}. A full 110-game 2400 s warm-up is infeasible (264k s of a 319k-slot-second budget).
   Hypothesis: UCB's mean-reward index over-rewards early cheap clears and under-values patience, so expect it near round-robin (-3.6) or worse. This puts a simulated number on a result that was previously a single real run.
9. M43-PERMITS (rellik13 two-tier action-quantum permits). Documented in rellik's notes as a dead path in the final 22.53 submission (0 grants in m43_summary.json), so it is a design, not a proven LB policy.
   Permit = X actions (X = 20, the median level-1 clear cost) then park, and park at once on a level-up.
   Rank = (tier = 1 if >=1 level cleared, fairness, l/T_N/max(a,10)^2).
   Throttle: once a > 60 on this level, the grant drops to 4 actions (about 6k tokens at 25k per 40 actions).
   Per-game quantum function; actions are prorated as in _level_actions.
   Uses: l, N, a, grant count, cleared-level count.
   Grid: X in {10, 20, 40}; throttle in {off, 4 actions, 8 actions}; switch_penalty = 3000 (needed or throttle looks free).
   Hypothesis: the 1/a^2 cliff plus short grants is a harsh stall decay like the decaying-tail loss (-0.3 to -2.9). Expect negative. Cheap to run once the per-game quantum hook exists.
10. COMPETITOR ANCHORS (sim validation and lower bounds).
   10a, Affectify form v1: e^{0.2(l-1)} * norm(N) * 2^{-a/100} * pace * 2^{-(t/124000)^2} * (1 + bonus/8).
     pace = clip((30000/cbar)^0.25, .25, 4); bonus = 4 / 3.5 / 2.5 when 3+ / 2 / 1 levels remain (check the exact offset in the notebook's _FP_SOURCE).
     fresh_first=False, whole value decayed, no undecayed tail. The author reports D' about +1.5 over v1 in his private sim. If our sim reproduces that gap, it agrees with his; if not, our sim differs.
   10b, lordhansolo capped FIFO waves: slot held until the game has used cap K = R_g(0) (177k) or finishes; no ranking.
     10b-rolling: K grows by +Q per level-up. This is a Duck-era 5-6 LB design; expect strongly negative.
     10b' hybrid: every game gets min(K, 50k) contiguous first, then D' ranking.
   Uses: l, N, a, t, cbar, level-up count.
   Grid: K in {100k, 177k, 250k}.
   Hypothesis: 10a is a calibration point for the simulator. 10b is a floor for how much any ranking is worth. 10b' tests whether an uninterrupted start is better than D''s quantum-by-quantum start.

NOT SIMULATED, with reasons:
- Affectify lottery: its "alive" ticket uses board-hash novelty over the last 20 actions, which levels.csv does not hold. Only a softmax-over-D'^kappa version could be run; D' values span 13-18, so kappa would need to be large and it would add noise only.
- Luby hold and retirement with an endogenous lambda: equal to a bounded incumbent offset (item 3) and to greedy ranking.
- Whittle with a switch cost: needs a simulator that charges re-prefill. Item 3 at penalty 3000-6000 is the proxy.

SUGGESTED RUN ORDER (by expected information and chance of beating D'): 3, 1a/1c, 2a/2b, 4, 5a, 6, 10a, 8, 7, 9, 10b.
Paths: replay sim at /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/replay-sim/analysis (replay_scheduler.py, levels.csv).
D' source: .../replay-sim/ARC3-Inference/inference/agent/priority_scheduler.py.
