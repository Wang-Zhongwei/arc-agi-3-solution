# scheduling-policy-search: research:theory

Workflow run `wf_f55d767d-e13`, phase Research. Find new slot-scheduling policies (competitors, theory, prior work), simulate them against D-prime, verify the best

---

# Slot-scheduling theory for the ARC-3 harness (read-only survey)

**Notation (all observable).** Game g is on level ℓ of N, with t tokens and a actions on this level. c̄ is `level_cost`, n_c the number of cleared levels, Q = 25k tokens.
- R is the remaining run tokens (T_rem · 61 tok/s · 10 slots) and G the number of unfinished games. R_g = R/G is a fair share: about 177k at the start, roughly 7 quanta or 6 mean level-costs. The budget is tight, so this is a knapsack, not an infinite horizon.
- w_ℓ = ℓ/(N(N+1)/2) and η(a) = (300/(300+a))^2.5 (D′'s efficiency term).
- Pace ρ = (n_c·c̄ + 3·30k)/((n_c+3)·30k). The log-cost autocorrelation within a game is 0.37 (n=577), so pace is informative.
- Hazard h_g(t) = P(clear within next Q | stuck at t) = α/(α+β·ρ+t/Q), a Beta(α,β) per-quantum clear rate with (α,β) = (2.5,2.5).
  - At t = 0/25/50/100/150k it gives .50/.42/.36/.28/.23. KM on `levels.csv` gives .52/.47/.37/.25/.33 for L1–2.
  - L5+ KM falls to about .09 past 100k, but n is small.
- Continuation K_g = Σ_{j>ℓ} w_j·η̄·q^(j−ℓ)·min(1, R_g/((j−ℓ)c̄)), with q ≈ 0.8 (data: .73–.86). V_g = w_ℓ·η(a+â) + K_g.
- Other data: cleared-level median 22k, p90 64k; censored-attempt median 89k.

**Unifying frame.** Every index below is I_g(τ) = E[score gained in next τ tokens]/τ.
- τ = Q gives marginal-utility greedy.
- τ = H_g (horizon) gives rising bandit, SERPT and knapsack density.
- τ = argmax over [Q, H_g] gives deadline-Gittins or the single-arm Whittle index.
- D′ is roughly marginal-utility greedy with a hand-set chance term C ≈ h, an undecayed B ≈ K, and φ ≈ a budget truncation.

**Why Gittins lost (hypotheses, untested).**
1. Decreasing hazard makes a one-level Gittins index fall with attained service (Aalto–Ayesta–Righter 2009; Scully & Harchol-Balter [arXiv:2111.10703](https://arxiv.org/abs/2111.10703)). That is least-attained-service, which churns toward fresh or shallow games. It is the same failure as Franzen's 17.5k switches into L1–2 games at 0.03 per quantum, and why round-robin scored −3.6.
2. The reward was the current level only. Hazard resets to about .5 after every clear, so the right arm is the renewal chain of levels, whose rate is about w̄η/c̄, not h·w.
3. Gittins assumes an infinite horizon with discounting and one server. Here there is a hard 532-minute deadline, 10 servers and about 7 quanta per arm (Berry–Fristedt 1985; Niño-Mora 2011, finite-horizon index).
4. The fit comes from censored runs made under another scheduler, and the "never" mass is unidentifiable.
- Every loser (Gittins, fitted, decaying-tail) decays its continuation value by hazard. The winner does not. This is consistent with the hypotheses, not proof.
- **Deadline-aware variant:** G_g = max_{k=1..⌊R_g/Q⌋} E[Y_g(kQ)]/(kQ), where Y = Σ_j w_j·η·1{levels ℓ..j all cleared within kQ tokens}. Level ℓ's remaining time is drawn from S_g(·|>t). Later levels are drawn from `levels.csv` cleared-token durations scaled by ρ_g (about 200 Monte-Carlo draws).

## Candidate policies

1. **Marginal-utility greedy** (Fox, *Mgmt Sci* 13(3) 1966; metareasoning: Hay et al. [arXiv:1207.5879](https://arxiv.org/abs/1207.5879), Dean–Boddy 1988).
   - I = max(h_g,0.2)·V_g/Q. For separable concave returns, equalizing V·f is optimal.
   - The floor at 0.2 matters because decaying the tail by stall lost (−0.3 to −2.9). It is basically D′ with fewer hand constants, so expect D′ ±1.
2. **Rising (rested) bandits** (Heidari–Kearns–Roth IJCAI'16; Metelli et al. [arXiv:2212.03798](https://arxiv.org/abs/2212.03798), R-ed-UCB: μ̂+(H−n)γ̂+bonus).
   - Games are rested: state freezes when off a slot, and V·F_g(t) is increasing and concave.
   - Level-ups are too sparse to estimate the slope γ̂, so use the model slope: I = V_g[F_g(t+H_g) − F_g(t)]/H_g, with F_g = 1 − Ŝ(t/ρ).
   - H is a patience knob: H = Q gives marginal-utility greedy, and H = R_g flattens the decay.
3. **Budgeted and knapsack bandits** (Tran-Thanh et al. AAAI'12; Badanidiyuru–Kleinberg–Slivkins [arXiv:1305.2545](https://arxiv.org/abs/1305.2545); Xia et al. [arXiv:1505.00146](https://arxiv.org/abs/1505.00146); Gupta et al. [arXiv:1102.3749](https://arxiv.org/abs/1102.3749), non-martingale Markov bandits, the closest model).
   - Density d_g = h_g·V_g/E[tokens used]; play by density.
   - Start rule: start a new game only if d_start ≥ λ, where λ is the density of the 10th-ranked running game. This explains why skipping about 4% of starts is rational.
   - Heavier option: solve the Gupta-style LP over states (ℓ, k quanta, ρ bucket) once offline and use the dual prices as the priority.
4. **Successive halving, Hyperband, ASHA, successive rejects** (Jamieson–Talwalkar [arXiv:1502.07943](https://arxiv.org/abs/1502.07943); Hyperband [arXiv:1603.06560](https://arxiv.org/abs/1603.06560); ASHA [arXiv:1810.05934](https://arxiv.org/abs/1810.05934); Audibert–Bubeck–Munos COLT'10).
   - Rungs r_k = 25k·3^k, and 225k is D′'s own T. Promote game g when Z_g = Σ_{j<ℓ} w_j·η_j / (total tokens) ranks in the top 1/3 of the games at that rung. ASHA's asynchronous promotion suits 10 slots.
   - The objective is wrong: best-arm identification, not the sum of rewards. It favors fast shallow games, so expect ≤ 0, in line with easy-first ≈ 0 and the negative stall caps.
5. **Whittle index, restless** (Whittle 1988 [JSTOR 3214163](https://www.jstor.org/stable/3214163); Weber–Weiss 1990, asymptotically optimal at fixed m/n, here 10/110; Yu–Xu–Tong, deadline scheduling [arXiv:1610.00399](https://arxiv.org/abs/1610.00399)).
   - For rested arms Whittle equals Gittins. It differs only through passive dynamics: the shrinking deadline and the context warmth lost when a game is off a slot.
   - W_g = min λ such that Q_active(s;λ) ≤ Q_passive(s;λ), by backward DP over (ℓ, k, τ_left).
   - Practical form: I = h̃V/Q + (λ·s_g/Q)·1[incumbent], with s_g the re-prefill tokens of the game's context. That is an additive switch cost, unlike the multiplicative incumbent bonus that lost −10 to −22.
   - The simulator charges 0 for switches, so it cannot score this. Expect small gains at best.
6. **Optimal stopping and retirement** (Whittle 1980 retirement view).
   - Stay while h_g(t)·V_g ≥ λQ. With the Beta hazard this gives t*_g = Q·(αV_g/(λQ) − α − βρ).
   - λ is the 10th-largest I among ready games, an endogenous shadow price. Deep games have a larger V, so they get a later t* (patience), and shallow games release early.
   - This replaces the hand-set 0.4 tail fade.
7. **SRPT / shortest expected remaining** (Schrage, *Oper. Res.* 16(3) 1968; Smith 1956; SOAP [arXiv:1712.00790](https://arxiv.org/abs/1712.00790)).
   - I = V_g·P(clear within H)/m_g(t,H), with truncated mean residual life m_g = ∫_t^{t+H} S_g(u)du / S_g(t).
   - Truncating at H guards against the infinite mean residual life that comes from "never" levels.
   - Decreasing hazard makes the mean residual life grow with t, so this carries the same bias against stuck games as Gittins. Use it only in the endgame, on last-level games.
8. **Portfolio and restarts** (Huberman–Lukose–Hogg, *Science* 1997; Gomes–Selman, *AIJ* 2001; Luby–Sinclair–Zuckerman, *IPL* 1993; Streeter–Golovin–Smith AAAI'07).
   - A bounded Luby hold: the i-th time a stuck game is admitted, let it keep the slot for L_i = Q·luby(i) = 1,1,2,1,1,2,4,… quanta. It is within O(log) of the best cutoff when the distribution is unknown, and it is bounded, unlike the −10 to −22 incumbent bonus.
   - Variance diversification likely adds little at about 7 quanta per game.
9. **Last-mile deadline allocation** (Dean–Goemans–Vondrák, *MOR* 2008, stochastic knapsack with adaptivity; Fox 1966).
   - Truncate K by budget (the min(1, R_g/…) term) instead of the fixed fade φ.
   - In the last X% of the run, choose r_g in units of Q to maximize Σ_g w_ℓ·η·[F_g(t_g+r_g) − F_g(t_g)] subject to Σ r_g = slots·τ_left. Separable and concave, so greedy is optimal.
   - Use Q_eff = min(Q, τ_left), and hold slots once τ_left < 2Q. D′'s switch share rises to 79% in the 80–90% decile.
10. **Thompson sampling** (Russo et al. [arXiv:1707.02038](https://arxiv.org/abs/1707.02038); budgeted version in item 3).
    - θ̃_g ~ Beta(α + n_clears, βρ + quanta failed on this level); I = θ̃_g·V_g/Q. This is the item-1 hazard, randomized.
    - It only breaks ties and thrash. Use it for starts: start w.p. P(θ̃_start·V_start/Q > λ).
11. **Kaplan–Meier hazard allocation** (Kaplan–Meier, *JASA* 1958).
    - Use Nelson–Aalen by depth bucket (L1–2 / L3–4 / L5+). The fresh-level hazard per 25k is .52/.46/.43. At 100k it is .25/.44/.09.
    - This suggests deep-level patience should end near 100k.
    - Replace D′'s linear C with C = h_g(t)/h_g(0). `fit_levels.py` can check this offline.

## Suggested experiments (expected sign, given what the sim has shown)
1. D′ with the budget-aware K_g in place of B·φ (items 1 and 9). It is cheap, and the most likely small win.
2. D′ with additive hysteresis λ·s, in a simulator that charges switches (item 5).
3. Chain-based deadline-Gittins or Whittle (the Gittins section above). If it beats −5, the chain reward is the cause of the difference.
4. Beta-hazard marginal-utility greedy with the 0.2 floor (item 1). It should match D′ with fewer constants.
5. Luby bounded hold (item 8), then retirement with endogenous λ (item 6).
6. ASHA rungs and successive rejects (item 4). Expect ≤ 0.

Link-checked: all arXiv links above, JSTOR 3214163 and Hyperband in JMLR 18. The other venue/year citations (Gittins 1979, Niño-Mora 2011, Weber–Weiss 1990, Tran-Thanh 2012, Fox 1966, Schrage 1968, Luby 1993 and the rest) are from memory and not link-checked. The `levels.csv` figures were recomputed from `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/replay-sim/analysis/levels.csv`.
