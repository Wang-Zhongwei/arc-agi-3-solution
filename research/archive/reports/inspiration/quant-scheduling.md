# Quant / operations-research ideas for the slot scheduler (research brief, 2026-10-08)

Scope: what quantitative finance, quant-dev practice and adjacent operations research offer for allocating ~10 LLM slots over 110 games (hidden) or 25 games (offline), with the owner's rule applied: **each design is judged by whether our own real-run data (7 full runs, 869 level attempts) can check it, and designs that rest on assumptions the data cannot check are ranked down.**

Provenance. Claims about the literature are linked. Numbers marked "own analysis" were computed in this session from `analysis/levels.csv` and `analysis/calibration/*.csv` on branch `origin/arm/sim-v2` (copied to `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/inspiration/work/`, scripts `a1.py` ... `a7.py` in the same folder). They use only attempts from the 7 `fz-*` runs in `levels.csv` (not the D' run `v4-stack-v1` or ledger r2, which that file omits). Facts about the repo come from `arms.md`, `priority_scheduler.py`, `sim2_VALIDATION.md`, `sim2_RESULTS.md`, `analysis/README.md` and the failure-taxonomy report on `origin/arm/failure-taxonomy`.

---

## 0. Bottom line

1. **The best-supported quant idea is not a new policy but better inputs.** A held-out check shows the in-run pace of a game (how costly its earlier cleared levels were) predicts whether the current level clears soon, beyond the pooled curve (own analysis, +0.030 log-lik per point, se 0.005, 2,440 leave-one-run-out points). Franzen's hand-set Gaussian falloff predicts slightly worse than the pooled Kaplan-Meier curve (own analysis, -0.011). This is checkable on our data and testable in the simulator the team already trusts (Franzen-family tweaks).
2. **Gittins theory says what the right rule looks like, and it differs from Franzen's falloff in one measurable way**: the empirical per-token clear rate declines slowly with age (index at 100k tokens is about 0.5x its peak; Franzen's token `chance` is 0.165 there, 0.047 at 130k; own analysis). But whether that matters depends on the share of levels that are truly unclearable, **which our data cannot identify** (section 3). So the "stale level" part of a Gittins design is ranked down.
3. **The transaction-cost analogy is empty on the GPU side.** In the only run with request logs, a trim request re-prefills about 54k uncached tokens whether or not the game handed over its slot (own analysis, n=104; section 6). A no-trade band would be solving a cost that is about zero.
4. **Validation, not policy, is the larger lever.** With the real run-to-run sd of 1.7 points, two paired full runs per arm can only detect effects of about 10 points; a simulated +1.25 (D') would need about 29 runs per arm (own analysis). D' could not have been confirmed or rejected as a +1.25 policy with any plausible GPU budget. What the single real run could do was act as a harm tripwire, and it did (z = -4.5 against the Franzen runs' spread).

---

## 1. Gittins index and scheduling with unknown job sizes

**What it is.** For a set of independent "projects" where you may work on one at a time, the Gittins index of a project in its current state is the best achievable reward-per-unit-of-effort over all stopping rules; playing the highest-index project is optimal for discounted infinite-horizon bandits ([Gittins 1979, JRSS B](https://academic.oup.com/jrsssb/article/42/2/143/7027598); short proof by [Weber 1992](https://www.statslab.cam.ac.uk/~rrw1/abstracts/w92c.html); [Weber's lecture notes](https://www.statslab.cam.ac.uk/~rrw1/oc/ocgittins.pdf)). For jobs with unknown size and tail function F, the rank of a job of age a is `r_G(a) = inf_{b>a} integral_a^b F(t)dt / (F(a) - F(b))` (expected service spent per completion, minimised over the quantum b), which is the reciprocal of "completion probability per unit of expected effort" ([Scully, Harchol-Balter 2020, SERPT paper](https://www.alphaxiv.org/abs/1907.10792.md)).

**Evidence.**
- The Gittins policy minimises mean response time in the preemptive M/G/1 queue when only the size distribution is known ([Scully and Harchol-Balter, "Simple near-optimal scheduling for the M/G/1"](https://arxiv.org/pdf/1907.10792); [Scully, "The Gittins Policy in the M/G/1 Queue"](https://arxiv.org/pdf/2111.10703); [Aalto, Ayesta, Righter 2009](https://research.aalto.fi/en/publications/on-the-gittins-index-in-the-mg1-queue/)).
- Hazard-rate structure decides what Gittins does. For decreasing-hazard-rate sizes it equals foreground-background (serve the youngest job); for new-better-than-used-in-expectation sizes it equals FCFS (run to completion) ([Aalto and Ayesta overview](https://www.irit.fr/~Urtzi.Ayesta/argi/2011-aaltio-GittinsPEISrev.pdf)). Real cost distributions with a hump (rising then falling hazard) give a mixed rule: commit for an initial quantum, then rotate.
- The simpler "shortest expected remaining processing time" rule can be badly suboptimal; its monotone envelope (M-SERPT) is within a factor 3 to 5 of Gittins ([SERPT paper](https://arxiv.org/pdf/1907.10792)). The SOAP framework gives one analysis for the whole family of age-based priority policies, including Gittins, so a candidate rank function of age can be analysed without bespoke simulation ([SOAP, Scully, Harchol-Balter, Scheller-Wolf](https://www.alphaxiv.org/abs/1712.00790)).
- Multiserver case (we have 10 slots): Scully et al. have a paper on unknown sizes with several servers ([Performance 2020](https://www.performance2020.deib.polimi.it/wp-content/uploads/2020/11/Performance_2020_paper_107.pdf); abstract not read for this brief).

**Where our problem departs from the theorem** (so the optimality claim does not transfer):
- The M/G/1 result is for Poisson arrivals, all jobs eventually finishing, minimising mean response time. Ours is a fixed-deadline reward-maximisation with weights (level i is worth i), jobs that are abandoned, and a sequence of dependent jobs per game. The index form (reward over expected effort, optimised over a quantum) still applies as a heuristic; optimality does not.
- Switching costs break index optimality in general ([Banks and Sundaram 1994](https://authors.library.caltech.edu/67329); survey [Jun 2004](https://link.springer.com/doi/10.1007/s10645-004-2477-z)); but see section 6, our switching cost appears to be about zero.
- Finite horizon: the infinite-horizon index is optimal only for the discounted infinite problem; the finite-horizon analogue has exact algorithms with O(T^2 n^3) time ([Niño-Mora, finite-horizon index](https://ar5iv.labs.arxiv.org/html/2207.14189)).

**Computation.** Gittins indices for a discrete-state project come from the retirement formulation: the index of state x is the retirement value M at which continuing and retiring are equally good; Niño-Mora's algorithm is O(N^3) for infinite-horizon projects ([same source](https://ar5iv.labs.arxiv.org/html/2207.14189)); model-free learners exist ([QGI](https://arxiv.org/html/2405.01157v4)). **Our state graph is acyclic** (age and level only increase), so the stopping problem is solved by exact backward induction on a grid, then bisected over the price; I did this (own analysis, `a6.py`), 2k-token age bins, 8 levels, a few seconds.

**What the data say (own analysis, 869 attempts, pooled Kaplan-Meier over tokens spent on a level, uncleared attempts censored):**

| Token bin on a level | Clears | Exposure (10k-token units) | Hazard per 10k tokens |
|---|---|---|---|
| 0-10k | 167 | 793 | 0.21 |
| 10-20k | 178 | 594 | 0.30 |
| 20-30k | 113 | 450 | 0.25 |
| 30-45k | 118 | 488 | 0.24 |
| 45-60k | 80 | 318 | 0.25 |
| 60-80k | 42 | 282 | 0.15 |
| 80-100k | 22 | 188 | 0.12 |
| 100-130k | 16 | 126 | 0.13 |
| 130-200k | 5 | 57 | 0.09 |

Cleared levels: median 22.2k tokens, p90 64k, p99 114k; 127 of 869 attempts uncleared (median spend 88.7k). The hazard rises over the first 20k, is flat to about 60k, then halves. A Weibull fit has shape 0.99 (it collapses to an exponential), while a log-normal fits better (leave-one-run-out log-lik per attempt: log-normal -9.9448, Weibull -9.9939, log-logistic -9.9503, log-normal with cure fraction -9.9454, 2-component mixture with cure -9.9431 using 6 parameters: no meaningful gain from complexity).

**Single-level Gittins index implied by that curve** (clears per token, relative to age 0; own analysis, log-normal fit):

| Age (tokens) | Gittins index relative to fresh | Franzen token `chance` (0.5^((a/62000)^2)) |
|---|---|---|
| 10k | 1.19 (peak) | 0.982 |
| 40k | 0.87 | 0.749 |
| 60k | 0.71 | 0.522 |
| 80k | 0.60 | 0.315 |
| 100k | 0.52 | 0.165 |
| 130k | 0.42 | 0.047 |

So the empirical rule sits on a plateau for about 30-40k tokens, then decays slowly, while Franzen's falloff goes to near zero by 130k. (Franzen's actual P also multiplies in an action-count term and a pace divisor, so this is the token part only.) Important caveat: the tail of this curve rests on 85 events past 60k tokens and on attempts the scheduler chose to keep running (informative censoring).

**Mapping to our scheduler.** Replace `chance/pace` in `priority_value` with `max over Δ of P(clear within Δ | state) / E[tokens spent within Δ | state]`, times the A+B reward. Inputs: per-level-attempt token costs with censoring (`levels.csv`), a hazard model with in-run pace covariate, the reward weights already in the code. The quantum Δ also answers "how long before the next re-evaluation" (here: re-evaluation at trims only, since slots hand over there).

**Validation on held-out data.** (a) Hazard calibration, leave-one-run-out (done, section 2). (b) Replay in sim-v2 as a Franzen-family priority change (the owner-stated trust boundary). (c) Real runs only after (a) and (b), with a power calculation (section 9). **The part not checkable**: the share of levels that are unclearable (section 3).

---

## 2. Survival / hazard modelling and empirical Bayes shrinkage

**What it is.** Kaplan-Meier handles right-censored attempts; parametric forms (log-normal, Weibull, log-logistic, mixtures, cure models) give smooth extrapolation; empirical Bayes (James-Stein) shrinks noisy per-group estimates toward the pooled mean, with the shrinkage weight set by the ratio of within- to between-group variance ([Efron and Morris 1975](https://www.medicine.mcgill.ca/epidemiology/hanley/bios602/MultilevelData/EfronMorrisJASA1975.pdf)). Quants apply the same idea to covariance and return estimates ([Ledoit and Wolf 2004](https://econ-papers.upf.edu/papers/691.pdf)).

**Evidence from our data (own analysis).**
- Variance decomposition of log tokens of cleared levels (intraclass correlation): by game 0.37, by level index 0.009, by run -0.004, by game-level pair 0.74. Level depth carries essentially no information (consistent with `arms.md`: median cleared cost is flat across levels 1-8); the identity of the game-level carries most.
- Predicting "clears within the next 20k tokens | alive at age a" (checkpoints 0 to 100k; 3,008 held-out points; leave-one-run-out; mean log-lik, higher better; pooled Kaplan-Meier = -0.6683):

| Predictor | Δ log-lik vs pooled | Valid for unseen games? |
|---|---|---|
| By-level Kaplan-Meier blended 50/50 | +0.003 | yes |
| Franzen-shaped decay with scale fitted on train runs | -0.011 | yes |
| Game identity from other runs, shrunk (K=5 pseudo-attempts) | +0.066 | **no** |
| Game-level identity from other runs, K=2 | +0.137 | **no** |
| In-run pace tercile (earlier cleared levels of the same game in the same run), 2,440 points | +0.030 (se 0.005) | yes |

- The in-run pace feature is real: correlation of pace with the next cleared level's log cost is 0.42 (n=577); the slope against the pooled mean is 0.20 with one earlier level, 0.79 with two, 0.83 with three, 0.68 with four. The share of reached levels eventually cleared is 0.935 / 0.87 / 0.69 for fast / mid / slow tercile pace; median cleared cost 14.8k / 29.3k / 35.7k tokens. Franzen's `ProgressPace` already uses this idea with confidence n/(n+3), clipped to a factor of 0.5 to 2 (`priority_scheduler.py`). The empirical slopes suggest an optimal weight rising faster with n than n/(n+3), but the references differ (Franzen's reference tokens versus my pooled log mean), so treat this as a lead, not a result.

**Reconciling with `arms.md`'s "consecutive level costs correlate only 0.05"**: that statistic is a level-to-level cost ratio; the pace feature is the game's mean level cost, which carries the game identity. Both are true.

**Leakage warning (quant "purging" analogue).** Leave-one-run-out lets other runs of the same game inform the held-out run. If the hidden games are not the public games (the code docstring says the scheduler "never reads hidden game metadata"), any feature using game identity must be validated **leave-one-game-out**, or it will look 2x better than it can be. Run-level splits are correct for hazard shape and for pace; wrong for game priors. See [purged and combinatorial purged cross-validation](https://www.luxalgo.com/library/concept/purged-cross-validation/) (secondary source) for the general rule: remove from training anything that shares information with the test block.

**Mapping.** Inputs: censored token costs, game-run level history, `total_levels` (already used). Output: a calibrated `P(clear in next Δ)`; the shrinkage weight is a single parameter fitted by the same held-out score. Use cure fractions only with a sensitivity sweep (section 3).

---

## 3. The unclearable-level problem (what the data cannot identify)

Of 147 game-level pairs seen in the 7 runs, 15 were never cleared in any attempt, 10 of those with at least one attempt past 60k tokens; but of the 133 pairs with at least 3 attempts only 4 were never cleared (own analysis). A hierarchical maximum-likelihood fit (game-level log-median ~ Normal, within-game-level sd, plus a point mass of unclearable game-levels) put **zero** mass on the unclearable class and absorbed the stuck levels into a heavy tail (between-game-level sd of log cost 0.98, within 0.57). A log-normal-with-cure fit gives a 1.3% cure fraction, the 2-component mixture 3.2%. The likelihood is flat in this direction because the scheduler itself stops spending on stuck levels, so "very slow" and "never" look the same in censored data.

Why it matters: a Gittins/Bayes index is only as good as the posterior probability that a long-stuck level is hopeless. Under the fitted heavy-tail prior the single-level index at 150k is still about 0.54 of its value at age 0 (1.03 vs 1.91 clears per 100k tokens); under a "30% of stuck levels are unclearable" prior it would collapse. The taxonomy report puts 32.8% of tokens in never-cleared levels (repo file `research/reports/failure-taxonomy.md` on branch `origin/arm/failure-taxonomy`, PR #77; local, no public URL). The sim-v2 validation also shows the simulated D' edge falls from +1.25 to +0.05 as stuck mass rises (`sim2_VALIDATION.md`, iteration 2). `arms.md` also lists about 7 unfinished games per run on levels no run has cleared.

Consequence for ranking: any design whose benefit depends on spending more or less on stale levels (anything Gittins-like at large age, "stuck patience", budget caps) is **ranked down**, because this prior is not identifiable from our censored data. The repo's earlier failures (UCB twice, D', budget caps) are all of this kind.

---

## 4. Whittle index, restless bandits, Lagrangian relaxation, Bandits with Knapsacks

- **What it is.** Whittle's relaxation replaces the "at most m of n arms active" constraint with an average constraint priced by a Lagrange multiplier; the resulting index policy is asymptotically optimal as n and m grow with m/n fixed under extra conditions ([Whittle 1988 and Weber-Weiss 1990](https://www.dpmms.cam.ac.uk/~rrw1/abstracts/w90c.html)). Bandits with Knapsacks adds consumable budgets (time, tokens) to bandit learning, with primal-dual and balanced-exploration algorithms and near-optimal regret ([Badanidiyuru, Kleinberg, Slivkins](https://arxiv.org/pdf/1305.2545)).
- **Mapping.** Our games are not restless in the Whittle sense (a stopped game's state is frozen apart from its deadline clock), so plain Gittins applies, not Whittle. The valuable piece is the Lagrangian view: one **price per slot-token** λ(t); a game continues while its index exceeds λ(t); λ is set so total demand equals remaining supply. Franzen's priority queue already realises this implicitly (waiters are the price). The new use is explicit: compute the Lagrangian upper bound of achievable score given fitted hazards to know the **headroom** of any scheduler (a yardstick, not a policy). With 110 games on 10 slots the Weber-Weiss regime (many arms, fixed ratio) is closer to true than with 25 on 10.
- **Validation.** The bound is computable from the 869 attempts plus the fitted hazard; cross-check by comparing the bound with each real run's realised score. Not done in this brief (see Gaps).

---

## 5. Optimal stopping, secretary-type rules, Pandora's box, real options

- **Pandora's box** ([Weitzman 1979](https://scholar.harvard.edu/sites/scholar.harvard.edu/files/weitzman/files/optimalsearchbestalternative.pdf)): sequentially open boxes with known cost and reward distribution; optimal rule assigns each box a reservation value solving `E[(X - z)+] = cost`, opens in decreasing order and stops when the best seen beats all remaining reservation values. Mapping: deciding whether to start an unstarted game versus continue an aged level near the end. But all unstarted games are exchangeable apart from public metadata, so the reservation values tie, and the start-order effect is exactly what the simulator is not trusted for (`sim2_VALIDATION.md`: "do not use it ... reordering starts").
- **Secretary-type rules** (irrevocable accept/reject with no recall) do not fit: every game can be resumed later at a cost, so there is recall.
- **Real options** ([McDonald and Siegel 1986](https://www.nber.org/system/files/working_papers/w1019/w1019.pdf); [Dixit and Pindyck](https://ideas.repec.org:443/b/pup/pbooks/5474.html)): the value of waiting is the value of information obtained before an irreversible commitment. Our commitment is reversible at a trim and the information (how a level goes) arrives only by spending tokens, so abandonment is a Gittins-type optimal stopping problem, not a waiting problem. The one place the option view helps: the **end of the run**. A level's continuation value is `P(clear within remaining budget) x weight`, which replaces Franzen's linear tail fade (last 20%) with a quantity computable from the hazard and the decode rate. Taxonomy: 3.2% of tokens are produced after the last action, and `arms.md` notes near-wins (vc33, cd82, tu93, m0r0, tr87, sc25) that lacked slot time at the end.

---

## 6. Portfolio ideas: shrinkage, Kelly, robust allocation, transaction costs, market making

- **Estimation error dominates (DeMiguel, Garlappi, Uppal).** Across 14 optimising models and 7 datasets none beat the equal-weight rule out of sample; the needed estimation window is about 3,000 months for 25 assets ([RFS 2009](https://ideas.repec.org/a/oup/rfinst/v22y2009i5p1915-1953.html)). Mapping: Franzen's coarse B table (8/7/5/0) is already a shrunk "1/N" valuation of future levels. With 742 clears and 127 censored attempts, the data support about three or four fitted parameters (age decay scale and shape, one pace slope, one tail constant), not a per-level, per-depth table. This is an argument against the "unlock-value" family (tested in sim-v2, loses: `arms.md`).
- **Kelly and fractional Kelly.** Full Kelly maximises long-run growth but is very risky short term, and errors in means matter about 20x more than errors in variances ([MacLean, Thorp, Ziemba](https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf)). Mapping: our per-game "bet size" is slot-share; score is additive across games, not multiplicative, so Kelly's log-wealth objective is the wrong one. The usable lesson is shrink the allocation toward uniform in proportion to estimation uncertainty: use a lower confidence bound on the clear hazard of stale levels. Same ranking argument as section 3.
- **Robust optimisation** (worst case over a confidence set for hazards) is the formal version of that haircut; no direct evidence for our problem. Rank low.
- **Transaction-cost-aware rebalancing** ([Garleanu and Pedersen 2013](https://www.nber.org/papers/w15205): trade partway toward an aim portfolio; no-trade regions with proportional costs from Davis-Norman). Mapping: re-prefill cost on a slot switch.
  - **Own check on the only run with request logs (`fz-ledger-on-trim-r2`, `requests.csv`, 104 trim requests):** handed-over trims re-prefill a median 53.3k uncached tokens (cached median 0), kept-slot trims re-prefill 54.1k; prefill time 4.9 s vs 4.5 s. A trim rewrites the prompt, so the re-prefill is paid either way. The extra GPU cost of a switch is about 0.35 s.
  - The real cost of a switch is the **waiting game's** lost wall-clock: across the sim-v2 calibration segments, 433 switch-outs had a median wait of 1,093 s (mean 1,410 s) (own analysis, `segments.csv`).
  - Conclusion: a no-trade band with hysteresis `challenger > incumbent x (1 + κ)` is justified only through the waiter's cost, a second-order effect. `arms.md` already records easy-first (a hold rule) as neutral and the yield-time ideas as env-only. Rank low.
- **Market-making inventory control** ([Avellaneda and Stoikov 2008](https://math.nyu.edu/~avellane/HighFrequencyTrading.pdf)): quote around a reservation price shifted by inventory times risk aversion. Analogy: a game's "inventory" is its sunk spend; the reservation-price shift is the age penalty. It gives the functional form "price = value - γ x inventory", which is what Franzen's stall decay and D''s linear stall decay already are. No new information beyond the Gittins view.

---

## 7. Quant-dev practice: event-driven systems and simulator fidelity

- Event-driven backtesters that model latency and queue position explicitly (for example [hftbacktest](https://hftbacktest.readthedocs.io/)) exist because small edges vanish when mechanics are mis-modelled. sim-v2 follows this philosophy and passes the equivalent test: given a run's own level costs it reproduces every run's score within 1.8 points (RMSE about 1; `sim2_VALIDATION.md`). Keep treating it as a mechanics model.
- What quants do when the simulator cannot model market impact: they treat sim edge as an upper bound, apply a haircut, and require live confirmation at small size. The sim-v2 record gives an empirical haircut table: D' simulated +4.9 (v1) and +1.25 (v2) versus real -7.5; UCB simulated -2.5 versus real -1.6; 16 experts predicted -3.3 versus real -7.6; 14 slots predicted +0.13 versus real; 12 slots with ledger predicted 54.8 versus real 48.1 (from `arms.md` and `sim2_VALIDATION.md`). Out-of-sample sim-to-real error for policies is of the order of 3 to 8 points, larger than any scheduling effect being chased.
- The failure mode the sim cannot test is **policy-dependent outcomes** (a level's cost depends on how the game is interleaved). Quants call the analogue "market impact": your own orders move the market. Detect it with natural experiments, not with the sim.

---

## 8. Backtesting discipline and multiple testing

- **Deflated Sharpe ratio** ([Bailey and Lopez de Prado 2014](https://papers.ssrn.com/abstract=2460551)) corrects a selected strategy's Sharpe for the number of trials, the variance across trials, and non-normality: expected maximum over N independent trials ≈ σ x [(1-γ) Φ⁻¹(1-1/N) + γ Φ⁻¹(1-1/(Ne))] ([formula summary](https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio)). For N=337 screened configurations this is 2.93 sd (own analysis). With a simulated per-policy standard error of 0.1 to 0.25 points the pure selection inflation is under 0.75 points; the D' miss (+1.25 predicted, -7.5 real) is far larger and **common-mode** (model error shared by the whole D'-like family), which deflation cannot fix.
- **Probability of backtest overfitting** via combinatorially symmetric cross-validation, and minimum backtest length ([Bailey et al., SSRN](https://papers.ssrn.com/abstract=2326253); [Pseudo-mathematics and financial charlatanism](https://scholarworks.wmich.edu/math_pubs/40/)): the more configurations tried, the more likely the best is overfit; the number tried must be reported. Action: keep a trial log of every configuration ever screened (337 already, plus sim-v2's 68 + 27).
- **Multiple-testing thresholds in finance** ([Harvey, Liu, Zhu](https://www.nber.org/papers/w20592)): the usual t = 2 is inadequate after heavy data mining; t > 3 is the hurdle. Use the stricter bar when a policy is picked from a large screened set.
- **Data-snooping tests** ([White 2000 reality check; Hansen 2005 SPA; Romano and Wolf 2005 stepwise](https://www.econometricsociety.org/publications/econometrica/browse/2005/07/01/stepwise-multiple-testing-formalized-data-snooping)): bootstrap the best-of-N statistic against the benchmark. Mapping: paired per-game differences against Franzen resampled by game, joint across all candidates.
- **Off-policy evaluation**: with logged data from a known randomised policy, the replay method gives unbiased estimates of a new policy ([Li et al., replay](https://arxiv.org/pdf/1003.5956)); doubly robust estimators combine a reward model and propensities ([Dudík, Langford, Li](https://www.arxiv.org/abs/1103.4601)). Mapping: our logs come from a deterministic scheduler, so propensities are 0 or 1 and replay is biased. Adding a small randomisation (e.g. 10-20% of handover decisions pick the second-ranked waiter) in future real runs would create estimable data; with about 45 switch-outs per run that is about 5 to 10 randomised decisions per run, so it only accumulates over many runs. Low value per run; note it for the final runs that matter.
- **Scheduling with misestimated sizes** ([Mitzenmacher 2020](https://arxiv.org/abs/1902.00732)): the cost of using predicted instead of true job sizes can be quantified analytically ("price of misprediction"); useful to bound how wrong the hazard may be before an index rule is worse than FCFS-like rules.

---

## 9. Validation protocol that would have caught D'

D' history (from `sim2_VALIDATION.md`, `arms.md`, taxonomy): replay sim v1 predicted +4.9 (25 games); sim-v2 later +1.25; real run (`v4-stack-v1`) 43.32, 7.5 below the ledger Franzen runs (z = -4.25 under the pooled run-to-run sd of 1.69, own check -4.5 using the 7.53 gap), with 40.7% of tokens in never-cleared levels and a 20.5% wait for a first slot (taxonomy). The run bundled D', the ledger and the #2 history fix.

**Power facts (own analysis, two-sided t test, alpha 0.05, power 0.8, true sd per run 1.7 / 2.4 / 3.0):**

| Runs per arm | MDE at sd 1.7 | at 2.4 | at 3.0 |
|---|---|---|---|
| 2 | 9.7 | 13.6 | 17.0 |
| 3 | 5.3 | 7.4 | 9.3 |
| 4 | 4.1 | 5.8 | 7.2 |
| 6 | 3.1 | 4.4 | 5.4 |

(sd 1.69 is the pooled within-condition sd of the five calibration runs; 1.16 without the ledger, 2.42 with.) To confirm a +1.25 effect: about 29 runs per arm. Hence a simulated difference below about 3 points is **not confirmable with real runs**, so the simulator cannot be the sole basis for a decision and real runs can only reject large harm.

**Protocol (each step is cheap relative to a GPU run; steps 1, 2, 4, 5 use only existing data):**

1. **Pre-register** per candidate: the hypothesis, the number of configurations tried so far, the primary metric (offline score), secondary metrics (tokens on never-cleared levels, wait for first slot, clears in the final 20%), the kill rule.
2. **Estimator gate (no scheduler simulation).** Every input the policy uses (hazard, pace, stall scale) must beat the incumbent's implied probabilities in leave-one-run-out held-out log-likelihood or Brier score, and must pass **leave-one-game-out** if it uses game identity. D' had no such gate; its stall scale and pace boost were never scored on held-out clears.
3. **Simulator known-answer test per policy family.** The simulator must reproduce the sign and rough size of at least one real contrast in the same family before it ranks that family. UCB passed (43.5 vs 43.8); D'-like had no prior real contrast, so the correct status was "unvalidated", and the decision rule in `sim2_VALIDATION.md` ("treat any simulated difference under about 2 points as unresolved") is the right one and should be applied to family-level ranking too.
4. **Sensitivity sweep on the unidentifiable quantity.** Re-run each candidate with the unclearable share at 0, 0.1, 0.3 (the repo's `--never` and `--stuck-floor` options exist). Reject if the sign flips. The D' edge went from +1.25 to +0.05 along this axis.
5. **Power gate.** Compute the MDE from the table; if the claimed gain is below it, the real run is only a harm test. Set the harm margin (for example -3) and a go/no-go before running.
6. **Factorial, not stacked, real runs.** One factor per paired run: D' + ledger vs ledger alone (this is already named the most informative run in `arms.md`). Bundling D', ledger and #2 fix made the 7.5-point loss unattributable.
7. **Endogeneity check from natural experiments.** Compare per-game-level clear probability at equal spend between runs with different interleaving (UCB, 12/14 slots, D' run) with a paired test. The sim assumes costs are exogenous; this is the only available evidence on whether they are.
8. **Report with a trial ledger and deflation.** State N tried, apply the expected-maximum correction (2.93 sd at N=337) and use paired bootstrap by game (joint across candidates) for the final claim.
9. **Shadow logging.** In every real run, log each candidate's priority beside the actual at each handover. Costs nothing; builds a dataset for step 2 on later runs.

**Would it have caught D'?** Step 2 would have flagged missing evidence for D''s stall/pace parameters; step 3 would have labelled the sim result "unvalidated for this family"; step 4 would have shown the edge shrinking to +0.05; step 5 would have shown a +1.25 claim is unprovable. Together they say "D' is not worth a GPU run as a gain; if run at all, run it as a bounded-harm test, one factor at a time". That is what ultimately happened, at the cost of one run and a confounded one. The honest limit: nothing in the data available then would have predicted -7.5; it was a 4.3-4.5 sd event and possible confounding by the #2 fix (higher IndexError/KeyError rates) is unresolved.

---

## 10. Ranked scheduler designs

Ranking rule: first by whether the design's key assumptions are checkable on our 7 runs and 869 attempts; then by expected effect. "Sim-OK" means the sim-v2 trust boundary (Franzen-family priority tweaks) covers it.

| # | Design | Formula / algorithm | Inputs from our data | Held-out validation | Expected effect | Risk | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | **Calibrated hazard + in-run pace inside Franzen's P** (keep A+B, queue, tail) | Replace the Gaussian `chance` and `ProgressPace` by `P(clear in next Δ | age, shrunk pace)`: pooled/log-normal hazard times a pace factor with shrinkage weight n/(n+k), k and slope fitted by leave-one-run-out score | Censored token costs per attempt, level history of the same game in the same run, `total_levels` | Done: pace tercile +0.030 log-lik (se 0.005, 2,440 pts); Franzen shape -0.011 vs pooled. Next: leave-one-game-out of the pace slope; sim-v2 replay (Sim-OK); then 2 paired real runs | Small, 0 to +1.5 points; prediction quality should improve, score uncertain | Low (only 3-4 parameters, small ranking change) | Prediction better: moderate-high. Score gain: low |
| 2 | **Deadline-aware tail** (replace the linear fade over the last 20%) | Value of a level at time t = weight x `P(clear within remaining tokens)` from the hazard and decode rate; start no level that cannot finish | Hazard (as #1), measured per-stream decode speed, per-run timeline of when clears occur | Count clears and tokens in the last 20% in the 7 runs (not yet computed); sim-v2 tail variants (Sim-OK, tail fade is a Franzen parameter); near-wins list in `arms.md` | Plausibly +0.5 to +2 (near-wins lacked end-of-run slot time) | Medium: D' needed its 0.4 fade (-7.3 without) in replay v1, so the tail is a high-leverage knob | Low-moderate |
| 3 | **Headroom bound first (diagnostic, not a policy)** | Lagrangian/clairvoyant upper bound: with realised costs of each run, best prefix-knapsack allocation; compare with realised score | 869 attempts, realised per-run costs, censoring | Compare bound to each of the 7 realised scores | Tells whether scheduler work can matter (if headroom is under 2 points, stop) | None | High that it is informative |
| 4 | **Single-level Gittins/marginal-productivity rank for aged levels** | `reward x max_Δ P(clear in Δ)/E[cost in Δ]` with a slower age decay than Franzen's | Same hazard; cure fraction as swept parameter | Hazard calibration at ages 60k+ is the weak part (85 events); sim sweep over unclearable share 0 to 0.3 | Unknown sign: more spend on stale levels is the 32.8% sink | **High**: unclearable share not identifiable (section 3) | Low |
| 5 | **Full chain Gittins (levels in sequence, depth-weighted rewards)** | Exact backward induction on (level, age) DAG, bisection on price | Hazards for deep levels | Deep-level hazards cannot be validated: attempts per level 175/165/144/125/105/85/37/27/6 for levels 1-9, level 9 never cleared; with depth-neutral hazard the index of deep levels saturates high (own analysis), contradicting sim-v2's finding that remaining-weight valuations lose | Unknown | High: assumption-driven | Low |
| 6 | **Game-identity priors (empirical Bayes across runs)** | Shrink a game's cost toward its cross-run history, K about 2 to 5 | Game identity | Leave-one-game-out is required; in leave-one-run-out it gives +0.07 to +0.14 but that is leakage if hidden games differ | Works only if hidden games are the public ones; otherwise nil | Overfits the 25 public games | Do not use unless hidden-set overlap is known |
| 7 | **No-trade band on slot switches** | Challenger must exceed incumbent x (1+κ) | Re-prefill cost per switch | Done: switching adds about 0.35 s of prefill; the cost is the waiter's lost wall-clock | About 0 | Low | High that the effect is small |
| 8 | **Start-order optimisation (Weitzman / Pandora)** | Reservation values for unstarted games | Public metadata only | Sim not trusted for start ordering | Unknown | Medium | Low |

Items 4 to 8 are ranked down by the owner's rule; their value rests on quantities our data cannot check (unclearable share, deep-level hazard, hidden-game overlap, start-order effects).

---

## Gaps

- **Not done:** the headroom bound (design 3), the clears-in-last-20% tabulation (design 2), the leave-one-game-out version of the pace slope, the policy-dependence test from natural experiments (step 7). All need only data already in `analysis/` or the Kaggle outputs; the timelines for the last two sit outside `levels.csv`.
- `levels.csv` omits the D' run and the ledger r2 run (7 runs, 869 attempts), so those could not be used as held-out sets for level hazards.
- Informative censoring: the scheduler's own stopping rule determines which long attempts exist; the hazard beyond about 60k tokens is estimated from 85 events and biased by selection. A proper treatment needs inverse-probability-of-censoring weights from the priority at each handover; not done.
- Not read in full: the multiserver Gittins abstract; Niño-Mora's finite-horizon algorithm was read only through the summary. PDFs of Scully's papers could not be extracted by the fetch tool (summaries came from the arXiv/alphaxiv versions).
- Davis-Norman 1990 and the Disruptor/LMAX event-driven material were not retrieved as primary sources.
- Whether the hidden games overlap the public 25 is unknown; it determines whether design 6 is usable.
