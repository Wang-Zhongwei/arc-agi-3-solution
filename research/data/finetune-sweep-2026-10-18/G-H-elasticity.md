# G-H: Compute elasticity: how score responds to more effective compute (existing data only)

Researcher G-H, 2026-10-18. Local analysis only; no new runs. Paths without a prefix are relative to `daniel-franzen-sol/`. "SP" means the session scratchpad, `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/ce155f4d-514a-4385-ac65-a7e7e5a23e1c/scratchpad/gh/`.

**Method (reproducible).**
- **Timelines.** I rebuilt per-minute score, level, token and action timelines for 49 full runs with `nice -n 15 python3 analysis/score_timeline.py --runs $(cat SP/runs.txt) --out SP/tl` (bench clock, 60 s step).
  - Where a `summary.txt` exists (the fz runs, v4-control, tput-safe, stack-v1), the tool's cross-check matched every game.
  - Elsewhere the final values equal the live `score_timeline.jsonl` rows, for example MR 54.69.
- **Reference set REF.** 23 runs from the 10x family at stock serving: 10 experts, 14 streams, a 98k window, no output cap and stock acceptance. This is L3's list of 26 minus `user-combo-13x`, `uc-11x` and `pf-solved-cap8k` (the list is in `SP/tl_analyze.py`).
  - REF final scores: mean **51.57**, sd 3.26.
  - REF generated tokens: mean 5.18M, sd 0.17M.
- **Scripts** (stdlib only, run with `python3 -I`):
  - `SP/tl_analyze.py`: the pooled curve, slopes and fits.
  - `SP/natexp.py` and `SP/natexp2.py`: the natural experiments at matched compute.
  - `SP/late_gain.py`: where late gains come from.
  - sim2 grids (`analysis/sim2.py --policy franzen --ledger on`, 200 reps for 25 games, 40 reps for 110 games): `SP/sim/grid_{s,min,110_noledger}.txt`.
- **Time-equivalence assumption.** A run at uniform compute ×m for 121 min ≈ the same run at ×1 for 121·m min. The deadline is the solver's `max_runtime_s_per_game` and the priority tail fade is a fraction of it, so both scale with `demo_minutes` (`ARC3-Inference/inference/framework/solver.py:810-823`; `kaggle/build_notebook.py:619-622`).
  - **Down:** a run with less compute should score like a reference run truncated at the same cumulative tokens.
  - **Up:** a run with more compute is an extrapolation of that curve.
  - Q1 tests the assumption.

## Q1. What do the natural experiments say? Is the time/token curve the right yardstick?

### Takeaway
- **Downward:** all 8 runs that lost compute without a known change in decisions land on the reference curve at their own token (or action) count. Mean residual is **−0.01**, sd 1.68.
- **Upward:** all 4 runs that bought more tokens did so by changing streams, window or sampling, and every one scored below the curve (−4.3 to −23.2). **No clean upward test exists.**
- **The 27B:** at matched actions it scores 9–12 below the curve. Its decisions are worse, so this is not a compute effect.

### Cited Findings
Prediction = the mean, over the reference runs, of each one's score at the test run's final tokens or actions. Above a reference's own total, the prediction extrapolates its last-20-min slope. Output of `SP/natexp.py` and `SP/natexp2.py`.

| Test run | What changed | Compute ratio | Real | Curve | Residual | Ref (mean final) |
|---|---|---|---|---|---|---|
| `fz-experts16-full/v1` | 16 experts/token, 504.6 vs 611.1 tok/s (`arms.md:211`) | tokens 0.826 | 38.69 | 39.14 | **−0.45** | fz-arms v4/v5/v6 (45.42) |
| `v4-user-combo-13x/v1` | 13 experts, −12% decode (`arms.md:111`) | 0.821 | 47.34 | 47.65 | **−0.32** | uc-10x ×3 (52.91) |
| `v4-user-combo/v1` | 13 experts, base_norm (`arms.md:366`) | 0.788 | 49.61 | 46.53 | +3.08 | uc-10x ×3 |
| `v4-uc-11x/v1-snowiers` | 11 experts, −10% decode (`arms.md:111`) | 0.877 | 47.19 | 49.38 | −2.19 | uc-10x ×3 |
| `v4-uc-11x-16s/v1` | 11x + 16 streams; lost compute | 0.551 | 37.58 | 37.54 | **+0.04** | uc-10x ×3 |
| `v4-pf-solved-cap8k-10x` | 8k output cap, −9% tokens/action (`arms.md:102`) | actions 0.943 | 49.89 | 50.51 | −0.63 | pf-solved (55.55) |
| `v4-control/v1` | Franzen at 10 streams | tokens 0.860 | 50.62 | 48.90 | +1.72 | uc-10x ×3 |
| `v4-gguf-final/v1` | llama.cpp, 175 tok/s, stopped at ~61 min (`arms.md:218`) | 0.138 | 2.33 | 3.64 | −1.31 | fz-arms ×3 |
| `fz-streams14/v1` | 14 streams, 96k ctx, +15% tok/s (`arms.md:206`) | **1.145** | 46.38 | 50.68 | **−4.30** | fz-arms ×3 |
| `fz-ledger-streams12/v1` | 12 streams, 110k ctx | **1.083** | 48.08 | 52.85 | −4.76 | ledger-on-trim ×2 |
| `v4-user-combo-10x-accept05` | relaxed spec acceptance 0.5/0.5, lossy (`arms.md:362`) | **1.085** | 46.77 | 55.32 | **−8.56** | uc-10x ×3 |
| `v4-uc-18s-64k/v1` | 18 streams, 64k window (`arms.md:110`) | **1.221** | 36.01 | 59.24 | −23.23 | uc-10x ×3 |
| `v4-q27-dflash2-10x/v1` | Qwen3.8-27B, 2.06× tokens/action (`arms.md:101`) | actions 0.447 | 14.51 | 23.41 | −8.90 | MR (54.69) |
| same, vs REF | | actions 0.468 | 14.51 | 26.51 | −12.00 | REF (51.57) |
| `v4-tput-safe/v1` | gate11 queueing, 66.6 vs 7.1 min of waiting (`arms.md:544`) | tokens 0.983 | 42.09 | 50.06 | −7.97 | v4-control (50.62) |

- **The 8 compute-loss rows** have mean residual −0.01 and sd 1.68 (`SP/natexp*.py`). That sd is below the one-run sd of 3.2 (L3 fact-check §7).
- experts16 lost −6.74 against 3 controls. The curve predicts −6.28.
- **The 4 upward rows** have mean residual −10.2, or −5.9 without the 64k-window run.
- The project log already says: "throughput has not converted to score so far (streams14 +15% tok/s → 46.38 vs 46.25)" (`arms.md:693`).
- In `cap8k`, completion tokens fell 14% but actions fell only 5.7%. The cap enlarged the prompt budget, and cache hits dropped from 91.0% to 83.6% (`arms.md:102`). On the actions axis the run sits on the curve.

### Inferences
- **Uniform decode slowdowns cost what the time curve says, to within about ±2 points.** Fewer experts, a slower engine or lost compute all qualify, so the curve is a validated yardstick on the downside.
- **Every upward lever we have tried also changed what each stream sees or samples:**
  - smaller windows or more trims (streams14, 18s-64k);
  - a new token distribution (accept05).
- So "more tokens" has never been tested alone. R3 (shorter thinking) is a uniform per-game speed-up, the mirror image of experts16. It is also a distribution change like accept05, and accept05 lost 8.6 points more than its compute was worth.

### Gaps
- No run has more compute through a uniform speed-up, or through a longer budget. `_critic.md:6`: no arm ran longer than 121 min.

## Q2. Within-run timelines: how flat is the curve at minute 121?

### Takeaway
On REF the curve is **log-linear in time from minute 40 to 121**: dS/d ln t ≈ 29–31, with no visible bend at the end. The late slope is 0.25–0.29 points per minute.
- **Down, measured:** S(121) − S(110) = **2.77** (se 0.43). That is the time-equivalent of a 9.1% compute cut.
- **Up, extrapolated:** S(133.1) − S(121) = **+2.6 to +2.9**.
- **Supply of clearable levels:** at the deadline, 15.6 of 17.3 live games per run are on a level that some other REF run has cleared.

### Cited Findings (all `SP/tl_analyze.py`, `SP/late_gain.py` on REF, n = 23)
- **Pooled mean S(t):**

  | t (min) | 40 | 60 | 80 | 90 | 100 | 110 | 121 |
  |---|---|---|---|---|---|---|---|
  | S(t) | 19.07 | 30.45 | 39.67 | 42.65 | 45.69 | 48.79 | 51.57 |

- **Late slopes** (points per minute, mean with sd in brackets):

  | Window | REF (n = 23) | FZ-era (n = 7) |
  |---|---|---|
  | 80–121 | 0.290 (0.087) | 0.261 |
  | 100–121 | 0.280 (0.126) | 0.237 |
  | 110–121 | 0.252 (0.186) | 0.209 |

  - The REF values are close to the earlier fz-era report's 0.16–0.40 (`research/archive/reports/score-timeline.md:12`).
- **dS/d ln t:**
  - REF: 30.1 (60–121), 28.8 (80–121), 30.8 (100–121). On the token axis it is 29.1.
  - FZ-era: 29.3, 25.8 and 26.1 over the same windows, so it bends slightly.
- **Time-equivalent cuts** (REF: S(121) − S(121/m), with the per-run sd):

  | m | Cut | Loss | Per-run sd |
  |---|---|---|---|
  | 1.1 | −9% | 2.77 | 2.05 |
  | 1.25 | −20% | 6.60 | 2.59 |
  | 1.333 | −25% | 8.66 | 2.79 |
  | 1.5 | −33% | 11.67 | 3.52 |
  | 2 | −50% | 20.85 | 4.23 |

- **Fits on the pooled REF mean, minutes 40–121:**
  - log: S = −91.87 + 29.93 ln t (SSE 4.02);
  - saturating exponential: S∞ 70, t0 14, τ 80 (SSE 4.64);
  - a quadratic in ln t has a curvature term of **+0.54**, so there is no concavity.
  - **Holdout** (fit on 40–100, predict 121; actual 51.57): log 51.64, exp 50.88, power law 50.31. On FZ the log fit over-predicts by 2.0 (48.71 vs 46.72).
- **Extrapolated S(T)** (log / exp), each fit's own S(121) being 51.64 / 51.62:

  | T (min) | 133.1 | 151.25 | 161 | 182 | 242 |
  |---|---|---|---|---|---|
  | log fit | 54.50 | 58.32 | 60.19 | 63.86 | 72.39 |
  | exp fit | 54.20 | 57.41 | 58.85 | 61.43 | 65.95 |

- **Where late gains come from** (window 90.75–121):
  - +8.65 per run from 14.2 clears;
  - 32% from clears that finish a game, 72% from the 14 games ever won in REF, 68% from levels ≥ 5.
- **State at the deadline** (per run): 7.7 games won, 17.3 live. Of the live ones, 15.6 are on a level cleared in some REF run and 1.8 are on a never-cleared level.
  - This is consistent with L3: 10% of stuck levels are walls, and P(later clear | ≥ 60k tokens spent) is 58% (`L3-headroom.md`, Q2).

### Inferences
- The marginal minute at 121 is still worth 0.25–0.29 points, mostly from deep levels of winnable games.
- Upward per +10% (ln 1.1): log fit +2.86, exp fit +2.58, against 2.77 measured downward.
- Saturation is not visible by 121 min. The log/exp spread is 1.3 points at 161 min and 6.4 points by 242 min.

### Gaps
- The curve shape above 121 min is unobserved. The holdout validates extrapolating by 21% of time, and 161 min needs 33%.

## Q3. sim2's predictions against reality

### Takeaway
On the token axis sim2 is about 15% steeper than the real curve. It was wrong in both directions in its predictions:
- It under-predicted the compute-loss arms, because its speed inputs came from decode benchmarks while the real token losses were larger.
- It over-predicted every upward lever and the 27B.

L3's +7.6 offline for −25% tokens per action (sim2 at s = 1.333) is therefore about 20% high.

### Cited Findings (`SP/sim/*.txt` unless noted)

| Case | sim2 input → predicted Δ | Real Δ |
|---|---|---|
| experts16 | s = 0.826, no-ledger fit → −4.96; earlier sweep said −3.3 (`arms.md:522`) | −6.74 vs 3 controls (−7.56 vs control) |
| streams14 | 14 slots, no-ledger → +2.48 | +0.96 vs 3 controls (+0.13 vs control) |
| 13 experts | s = 0.88 → −3.91 | −4.43 (mean of 2 runs) |
| 11 experts | s = 0.90 → −3.18 | −5.72 |
| accept05 | s ≈ 1.15 → +4.15; +4.0 in `arms.md:534` | −6.15 |
| tput-safe | +2.1 (`arms.md:534`) | −8.53 |
| 27B | s = 0.44 → 24.30 | 14.51 |

- **Time-truncation deltas, sim2 vs real.** For m = 1.1, 1.25, 1.333, 1.5 and 2, sim2's `--minutes` grid gives 3.32 / 7.88 / 10.07 / 14.08 / 24.05. REF gives 2.77 / 6.60 / 8.66 / 11.67 / 20.85, so the real/sim ratio is **0.83–0.87**.
- **sim2 upward, from its s = 1 base of 50.93:**
  - Longer deadline (`--minutes`): 133.1 → 54.22, 151.25 → 58.05, 161 → 59.58, 182 → 62.27, 242 → 65.63.
  - Speed scale (decode only): s = 1.1 → 53.76, 1.25 → 57.13, 1.333 → 58.49.
- **Calibration and stated scope.**
  - sim2 is calibrated on the fz-era runs (`analysis/sim2.py:666`; `analysis/calibration/levels.csv`). At s = 1 it makes 4.48M tokens, against 5.18M in REF.
  - Its own scope says to trust it for mechanics, not for "policies that move compute between games" (`arms.md:518-519`).

### Inferences
- Fed the actual token ratios, sim2 tracks the downside roughly but too steeply.
- sim2 saturates at ~66 (s = 4 → 66.07, `L3-headroom.md:118`) because of its Kaplan–Meier "never" mass. The REF curve shows no saturation through 121 min. At 242 min sim2 (65.6) and the exp fit (66.0) agree; the log fit says 72.4.

## Q4. Elasticity estimate and what R3's token cuts are worth

### Takeaway
**Offline elasticity at 25 games / 121 min: +2.7 points per +10% effective compute, 95% CI about 1.9–3.5.** That is about 28 points per unit of ln compute.
- The interval combines the per-run increment se (0.43) with the spread between the log and exp fits.
- It is validated only downward. Upward conversion has never been observed cleanly.

For −25% tokens per action at equal decisions:

| | Offline | LB under P1 | LB under P3 |
|---|---|---|---|
| This estimate | **+6.3** (range 3.3–7.6) | **+3.5** (1.8–4.1) | **+6.5** (3.3–8.4) |
| L3 | +7.6 | +4.1 | +8.3 |

### Cited Findings and derivation
- **Effective compute.** A cut x in tokens per action is decode speed s = 1/(1−x). Prefill and client gaps do not shrink, so the wall-clock-equivalent multiplier m_eff is smaller than s:
  - sim2's own pass-through gives generated tokens ×**1.138 / 1.254 / 1.319** for x = 15% / 25% / 30% (interpolated from `SP/sim/grid_s.txt`; L3: s = 1.333 → 1.25×, `L3-headroom.md:122`).
  - A low pass-through, with decode at 60% of per-action wall time (G-E: harness tok/s rises about 0.6× as much as clean decode, `G-E-mtp-head.md:34`), gives 1.099 / 1.176 / 1.220.
- **Offline gain** = E_ln × ln(m_eff), where E_ln is points per unit of ln compute:
  - central: E_ln = 28 with sim2 pass-through;
  - low: E_ln = 20.3 (the lower 95% bound of the measured 2.77 ± 0.84 per 10%) with low pass-through;
  - high: sim2 directly (s = 1.176 / 1.333 / 1.429 → +4.68 / +7.56 / +8.82).
- **P1:** LB gain = 0.548 × offline gain (`L3-headroom.md:137,142`).
- **P3, L3's version:** hidden games ≈ public games at 0.6× compute, on sim2's 110-game, 532-min curve.
  - Re-run at 40 reps: s = 0.6 → 30.74; 0.706 → 35.60; 0.8 → 39.12; 0.857 → 41.07.
- **P3, real-curve version:** LB ≈ 0.9 × REF curve at the compute that gives 31.
  - The 0.9 factor is sim2's 110-game/25-game ratio, 45.76/50.93.
  - 0.9 × S(69 min) = 0.9 × 34.66 = 31.2.
  - The local slope there is about 32 per ln unit (S(60) = 30.45, S(80) = 39.67), so the LB gain ≈ 28.8 × ln(m_eff).

| Tokens per action | Offline, central [range] | LB under P1 | LB under P3, real curve [low – L3's sim2] |
|---|---|---|---|
| −15% | **+3.6** [1.9–4.7] | **+2.0** [1.0–2.6] | **+3.7** [1.9–4.9] |
| −25% | **+6.3** [3.3–7.6] | **+3.5** [1.8–4.1] | **+6.5** [3.3–8.4] |
| −30% | **+7.8** [4.0–8.8] | **+4.3** [2.2–4.8] | **+8.0** [4.0–10.3] |

- Per +10% effective compute, on the LB: P1 ≈ +1.5; P3 ≈ +2.7 (real curve) to +2.75 (sim2: s = 0.6 → 0.66).
- **For LB 60 from efficiency alone** (+29 LB):
  - P3, real curve: ln m = 29/28.8, so m ≈ 2.7×, and only if the log-linear shape holds that far. sim2 saturates first. This is consistent with L3's 2.85×.
  - P1 (16.4 per ln unit): about 6×.

### Inferences
- The real curve is log-linear down to 40 min, so P3 implies LB gain ≈ 1.0× offline gain, not the ~1.1× that sim2's convexity gave. P1 stays at 0.55×.
- **The cuts most likely to be achieved (−15 to −25%) are worth +2 to +6.5 LB.** Since L3 bracketed −25% at +4 to +8, that is a modest downgrade.
- Because uniform slowdowns are costed exactly (Q1), any checkpoint that loses 10% effective throughput costs about −2.7 offline, −1.5 LB (P1) or −2.7 (P3). This backs L3's ±5% throughput gate.

### Gaps
- The P1 vs P3 split is unmeasured, and the 0.9 factor comes from sim2.
- m_eff ignores slower context growth when thinking is shorter (thinking stays in history; `WRITEUP.md:69`), which is a small upside.

## Q5. Cheapest real measurement, and what would falsify R3

### Takeaway
Run the leader's exact configuration once with **`demo_minutes` 242: about 4.2 Kaggle GPU-h, 25 games.** A 161-min run is the cheapest alternative at about 2.9 GPU-h.
- The run's own `score_timeline.jsonl` yields S(121), S(133), S(151), S(161), S(182) and S(242) as within-run increments. Each increment is paired, so between-run noise (sd 3.3) drops out.
- **The R3 compute premise is falsified if S(151) − S(121) < +2.5 or S(161) − S(121) < +3.5.**

### Cited Findings
- **Cost.**
  - A 121-min run occupies the notebook for about 131 min: MR's bench ended at notebook second 7852 (`runs/kaggle/v4-mem-repair-10x/v1/score_timeline.jsonl`, last row), and the bench starts 9.2–12.2 min after the notebook (`score-timeline.md`).
  - So 161 min ≈ 171 min of notebook time ≈ **2.9 GPU-h**, and 242 min ≈ 252 min ≈ **4.2 GPU-h**.
  - Either fits one day's cap (mellamoa 6 h, snowiers 5 h; project `CLAUDE.md`). Kaggle's 12-h session limit is UNVERIFIED for these commits.
  - It is a new configuration, not an exact repeat (`demo_minutes` changes; `kaggle/build_notebook.py:619-622`).
- **Predicted within-run increments over S(121):**

  | Increment | Log fit | Exp fit | sim2 (time) | Noise sd |
  |---|---|---|---|---|
  | to 133 min | +2.9 | +2.6 | +3.3 | 2.05 |
  | to 151 min | +6.7 | +5.8 | +7.1 | 2.59 |
  | to 161 min | +8.5 | +7.2 | +8.7 | 2.79 |
  | to 242 min | +20.8 | +14.3 | +14.7 | 4.23 |

  - The noise sd is the REF per-run sd of the mirror-image downward increment (Q2).
- **The truncation validity check is built in.** The run's S(121) should sit within noise of REF (51.57 ± 3.26) or MR (54.69). This is the same equivalence Q1 validated downward.
- **Optional LB discriminator between P1 and P3.** It costs no GPU quota (scoring reruns are free, project `CLAUDE.md`) but uses 1 of about 15 remaining daily submissions per draw.
  - Submit the leader with the private-run deadline halved (266 of 532 min). Predictions:
    - P1: −0.548 × 20.85 = **−11.4**;
    - P3, real curve: 0.9 × (34.66 − 15.12) = **−17.6**;
    - P3, sim2 (110 games, s = 0.6 at 266 min → 11.94): **−18.8**.
  - The gap of about 6–7 LB points is about 1.9 draw-sd (sd 3.5, L3 fact-check §7) per draw.
  - A 0.75× throttle separates the models less: P1 −4.7 vs P3 −7.8 to −9.3.

### Inferences: falsification thresholds
1. **R3's compute premise.** It says a uniform ×1.25–1.33 buys about +6 to +8.7 offline.
   - **Falsified** if S(151) − S(121) < +2.5 or S(161) − S(121) < +3.5. Either means E_ln ≤ about 12 (≤ 1.2 per +10%). Then −25% tokens per action is worth ≤ +2.7 offline and ≤ +1.5 / +2.7 LB, which does not pay for the training prong.
   - If the true value is +7.2, the chance of a false falsification is about 9%.
   - **Supported** if S(161) − S(121) ≥ +6 (E_ln ≥ 21).
2. **Ceiling.**
   - S(242) − S(121) ≤ +10 means saturation is sooner than sim2's: a −50% cut would be worth ≤ +10 offline.
   - ≥ +18 means log-linear continues, and big cuts keep paying on the public set.
3. **Equal decisions, which is the accept05 risk.** The first full run of an R3 checkpoint must land on the REF curve at its measured tokens per action and harness throughput (predicted S = REF S(121·m_eff)).
   - A residual below −5 falsifies "equal decisions", whatever the state-lab screen showed. accept05 had −8.6.
4. **Hidden-set conversion.** At least 3 LB draws of a checkpoint with a verified −25% tokens per action gaining < +2 over the 30.9 mean means P1 or worse. The efficiency prong then cannot exceed about +4 LB.

### Gaps
- One 242-min run is a single draw of the curve shape. Two runs at 161 min, or one at 242, give noise sd about 2–2.8 on the R3-relevant increment.

## Implications for our plan
1. **Do:** run `demo_minutes` 242 on the leader (about 4.2 GPU-h) before committing training budget to R3. The 161-min run (about 2.9 GPU-h) is the fallback. This is the only way to measure upward elasticity with no model change. The rule for reading it is fixed in advance: S(151) − S(121) < +2.5 or S(161) − S(121) < +3.5 → deprioritise R3.
2. **Revise the R3 payoff:**
   - −25% tokens per action ≈ **+6.3 offline (3.3–7.6)**, **+3.5 LB under P1 (1.8–4.1)**, **+6.5 LB under P3 (3.3–8.4)**;
   - −15% ≈ +2.0 / +3.7 LB;
   - −30% ≈ +4.3 / +8.0 LB.
   - L3's +7.6 offline (+4.1 / +8.3 LB) is about 15–25% high, because sim2 is about 15% steeper than real runs.
3. **Gate on the curve, not on tokens:**
   - Every uniform slowdown landed on the REF curve (8/8, residual sd 1.7).
   - Every lever that bought tokens while changing context or sampling fell below it (4/4, −4 to −23).
   - So an R3 checkpoint must show a full-run residual of at least −3 at its measured tokens per action. Fewer tokens alone is not enough.
4. **Keep the throughput gate.** Each −10% of effective compute in a fine-tuned checkpoint (MTP drift, re-quantisation) costs about 2.7 offline and 1.5–2.7 LB, measured, not simulated.
5. **Efficiency cannot reach LB 60 on its own:**
   - It would need about 2.7× effective compute under P3 if the log-linear shape held that far (sim2 saturates first), and about 6× under P1.
   - The 1.8 walls per run sit at the deadline, while 15.6 games per run are on clearable levels. So public-set gains are compute-bound, while the hidden-set gap (P1 vs P3) is the open question.
   - If 2 submission slots can be spared, the 0.5×-deadline throttle draw (P1 −11 vs P3 −18) is the cheapest way to settle how much of R3 reaches the LB.

## Fact-check (independent verifier, 2026-10-18)

**What I re-ran.** All under `nice -n 15`, with scratch scripts in the session scratchpad (`vf_finals.py`, `vf_live.py`, `vf_more.py`, `vf_sim/`).
- I re-ran `SP/tl_analyze.py`, `SP/natexp.py`, `SP/natexp2.py` and `SP/late_gain.py`. Every printed number reproduces.
- **Independent rebuild, bypassing `SP/tl/*.csv`.** I read the `kind: run` rows of each REF run's live `runs/kaggle/*/score_timeline.jsonl` on `t_bench_s` and recomputed the pooled curve, cuts, slopes and log fit from them.
- **Final values.** For all 49 runs in `SP/runs.txt`, the timeline finals equal `runs/index.csv`. Where a live timeline exists, they also equal its last row.
- **sim2 re-runs** (`analysis/sim2.py --policy franzen --ledger on`): 25 games at 161 min (200 reps); 110 games with s = 0.6 at 266 and at 532 min; plus two diagnostic points (110 games with s = 1 at 266 min, and s = 0.3 at 532 min).
- **Not re-run:**
  - the full sim2 grids (I spot-checked 3 of their points, and all matched);
  - `analysis/score_timeline.py` itself (I checked its output against the live timelines instead);
  - the fz-era timelines, which have no live jsonl and were checked on finals only.

**Verdicts**
1. **REF pooled curve: CONFIRMED.** The independent rebuild from live rows gives:
   - S(40/60/80/90/100/110/121) = 19.07/30.45/39.69/42.65/45.69/48.79/51.56;
   - final 51.57 (sd 3.26) and 5.178M tokens;
   - slopes 0.290/0.279/0.252;
   - dS/d ln t = 30.10/28.69/30.79.

   Each is within 0.07 of the note.
2. **Time-equivalent cuts: CONFIRMED.** Independent values: 2.77 (sd 2.04, se 0.43), 6.60 (2.58), 8.66 (2.79), 11.66 (3.51), 20.84 (4.23).
3. **Curve fits: PARTIALLY CONFIRMED.**
   - The log fit reproduces independently: −91.84 + 29.92 ln t, SSE 3.98. So do the holdout (51.63) and the log extrapolations (54.49/60.18/72.38).
   - The exp-fit numbers come from a coarse grid: S∞ in steps of 1, t0 in steps of 2, τ in steps of 5.
   - A finer grid (S∞ 0.25, t0 0.5, τ 1) gives S∞ 68.75, t0 15.5, τ 76.
     - Its **SSE is 2.90, so the exp fit fits better than the log fit (4.02), not worse** (4.64 in the note).
     - Its extrapolations are 54.12/58.61/65.26 at 133.1/161/242 min (note: 54.20/58.85/65.95).
     - The exp increments over S(121) become +2.53/+5.63/+7.02/+13.66 for 133/151/161/242 min.
   - So "log-linear, no visible bend" is not preferred by fit quality. The quadratic-in-ln-t term (+0.54) and the saturating exponential point in opposite directions, so the shape above 121 min is less settled than Q2 says.
4. **Downward natural experiments: PARTIALLY CONFIRMED.**
   - The numbers reproduce exactly: mean −0.007, sd 1.676, n = 8.
   - **The stated condition is not met.** These are not all runs "without a known change in decisions":
     - `v4-control` vs uc-10x differs in the user-combo harness stack and runs 10 vs 14 streams (`arms.md:366,544`). Q1 itself treats a stream change as contaminating on the upward side.
     - The 13x, 11x and 16-expert runs and the GGUF run also change the model's numerics.
   - **The set depends on what was left out:**
     - two other downward runs fall well below the curve: `user-combo-16x` (0.818× tokens, −7.98, OOM) and `tput-safe` (0.983×, −7.97);
     - cap8k's residual is +2.63 on the token axis, against −0.63 on the actions axis the note uses.

     With all 10 downward rows the mean is −1.60, sd 3.67. The claim holds for uniform slowdowns, defined so as to exclude stream, window and gating changes.
5. **Upward natural experiments: CONFIRMED.** All residuals reproduce: −4.30, −4.76, −8.56, −23.23 and the 27B −8.90 (−12.00 vs REF).
   - Caveat: every upward "curve" value is an extrapolation past the reference runs' own totals, using their last-20-min token slope.
   - The FZ reference set (fz-arms v4/v5/v6) includes temp06 and audit60k, which are not pure controls.
6. **sim2 steeper than real runs: PARTIALLY CONFIRMED.**
   - Every sim2 number reproduces (the re-run at 161 min gave 59.58), and real/sim = 0.83–0.87.
   - **But sim/real is 1.15–1.21 (mean 1.18)**, so "about 15% steeper" is the low end; 15–20% is closer.
   - Q3's takeaway says "on the token axis", but this is the time axis. **On the token axis sim2 is about 25–28% steeper**: 36–37 points per ln-token unit (sim2's tokens grow more slowly than time, for example 3.63M vs 4.48M at 96.8 vs 121 min), against REF's 29.1.
   - Against the fz-era real curve (2.30/5.91/7.73/10.34/20.40), sim2's deltas are 18–44% steeper.
7. **sim2 vs reality: CONFIRMED,** with two nits.
   - The 13x real Δ is −4.44 (the mean of −5.58 and −3.31), not −4.43.
   - Q3's reason ("speed inputs came from decode benchmarks while real token losses were larger") does not fit experts16. Its decode ratio (504.6/611.1 = 0.826, `arms.md:211`) equals its real token ratio (0.826), yet sim2 at s = 0.826 cuts tokens only to 0.849× (3.83M/4.51M, no-ledger grid) and under-predicts the loss: −4.96 vs −6.74.
8. **Elasticity: CONFIRMED.**
   - 2.77 per ln 1.1 is 29.1 per ln unit, and the measured dS/d ln t is 28.7–30.8. "About 28" therefore corresponds to the rounded 2.7, which is slightly conservative.
   - The CI is 2.7 ± 1.96 × 0.43 = 1.86–3.54.
   - Log upward is +2.86; exp upward is +2.58 on the coarse grid, +2.53 on the fine one.
9. **R3 token-cut gains: CONFIRMED.** I re-derived all of it:
   - m_eff 1.138/1.254/1.320 from `grid_s.txt` interpolation;
   - the low pass-through 1/(0.4 + 0.6/s) = 1.099/1.176/1.220;
   - 28 ln m = 3.62/6.34/7.75;
   - the low end 20.3 ln m = 1.92/3.29/4.04;
   - sim2 high = 4.68/7.56/8.82;
   - P1 ×0.548 = 1.98/3.47/4.25;
   - P3 28.8 ln m = 3.72/6.52/7.97;
   - the 110-game sim2 rows 35.60/39.12/41.07 − 30.74.

   The cited L3 lines (122, 137, 142–147) and G-E:34 match.
10. **Deadline state and late gains: CONFIRMED.** Re-run output:
    - 7.7 won and 17.3 live per run (15.6 soft, 1.8 wall);
    - +8.65 per run from 14.2 clears in 90.75–121 min;
    - 72% of that gain from the 14 ever-won games, 68% from levels ≥ 5.
11. **Measurement design: PARTIALLY CONFIRMED.**
    - Confirmed:
      - the MR last row has t_notebook_s 7852.4 (130.9 min);
      - `kaggle/build_notebook.py:619-622` sets `max_runtime_s_per_game = demo_minutes*60`;
      - the leader's tail fade is a fraction (0.2) of the shared run duration (`ARC3_PRIORITY_TAIL_FADE_FRACTION`, `ARC3-Inference/inference/agent/tool_agent.py:3119-3133`), so it scales;
      - the endgame phase is off by default;
      - the log, exp and sim2 increments and the noise sds.
    - **Cost is slightly low.** The project's GPU-budget auto-memory counts a full 121-min arm at ~2.3 GPU-h, from push to completion and including model load. That makes 161 min ≈ 3.0 GPU-h and 242 min ≈ 4.3 GPU-h, not 2.9 and 4.2. Both still fit under one day's cap.
    - **False falsification is higher than stated.** The stated rule fires if *either* threshold is crossed. If the truth follows the exp path (+5.8 at 151 min, +7.2 at 161 min), with the empirical correlation ρ = 0.88 between the mirror increments, the chance of a false falsification is **~13%**: 10.1% for the 151 rule alone, 9.3% for the 161 rule alone. The note's 9% is the 161-only figure.
    - With the fine exp grid the exp column becomes +5.6 / +7.0 / +13.7.
12. **LB P1 vs P3 discriminator: CONFIRMED.**
    - The sim2 re-run reproduces 11.94 (266 min) and 30.74 (532 min).
    - S(34.5) = 15.12 and S(69) = 34.65 reproduce from the live rows.
    - −11.43 / −17.58 / −18.80 are correct.
    - **Possible confound, checked.** The 532 min is one shared wall-clock deadline for all 110 games (`ARC3-Inference/CONFIGURATION.md:828`). At 266 min, sim2 never starts 4.8 of the 110 games, so a halved deadline might not equal halved compute. sim2 says the two are nearly equivalent:
      - s = 0.3 at 532 min gives 12.09, against 11.94 for s = 0.6 at 266 min;
      - s = 1 at 266 min gives 23.44 against 45.76 at 532 min, a −48.8% drop, against −47% for 25 games.
    - **The noise figure is optimistic.** The gap is 6.2–7.4 LB, which is 1.77–2.11 sd at 3.5. A single draw is compared with a baseline mean of about 4 draws, so that mean's se (about 1.75) adds in, and the separation per draw is ≈1.6–1.9 sd.

**Other errors found**
- Line 17 cites `solver.py:810-823` for "the priority tail fade is a fraction of it". Those lines hold only the per-game runtime limit. The fade is in `tool_agent.py:3119-3133`, and the leader sets the fraction to 0.2. The assumption holds; the citation is incomplete.
- Implications §2 says L3's figures are "about 15–25% high". The ratios are 7.6/6.3 = 1.21, 4.1/3.5 = 1.17 and 8.3/6.5 = 1.28, so 17–28% high.
