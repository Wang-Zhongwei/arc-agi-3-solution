# "Hmm" density vs leveling up

Question: does the model say "hmm" more when it is stuck? Concretely, does a
higher rate of "hmm" in recent thinking go with a lower chance of clearing the
level within the next 10k thinking tokens? And if so, should the priority
allocator use it?

Follows [thinking-vs-level-up.md](thinking-vs-level-up.md), which found the same
direction for thinking *volume*.

## Data

Every Kaggle run under `ARC3-Inference/runs/kaggle/` with both transcripts and
event sidecars: 61 runs, 1,102 game runs (pass 0), 108,235 model calls.

- **Hmm count:** matches of `\bh+m{2,}\b` (case-insensitive: Hmm, hmm, Hmmm)
  in each call's `[THINKING]` block. Prompts, history, tool-call code and tool
  results are not counted.
- **Clock:** thinking tokens only, encoded with the Qwen3.8-Flash-Next
  tokenizer. Tool results are input, not generated, and never enter. Tool-call
  code is generated but is not reasoning, so it is left out to keep the signal
  pure.
- **Level-ups:** from `*_events.jsonl`, an action whose `level` exceeds the
  previous one (or whose state turns `WIN`), credited to the end of the last
  call of its `analysis_step`. Totals match `summary.txt` on 98% of games; the
  rest are multi-pass means or summaries written before the last level-up.

Per-call table: [`data/hmm_calls.csv.gz`](data/hmm_calls.csv.gz).

## Definitions

For a call that ends at thinking-token position $\tau$ on level $k$:

- $H$, $T$: hmms and thinking tokens of the calls on level $k$ within
  $(\tau - 10\text{k}, \tau]$ thinking tokens. The current call always counts.
- $d = H / T$: the **hmm density**, hmms per thinking token. Values below are
  quoted per 1k thinking tokens. Calls with $T < 1\text{k}$ are dropped. Pooled
  mean $\mu = 5.8$ per 1k, SD $2.70$. Run means range from $4.2$ to $9.1$ per 1k.
- **Outcome:** a level-up at a call ending in $[\tau, \tau + 10\text{k}]$
  thinking tokens, which includes the current call's own action. Calls whose
  window runs past the end of an unfinished game are dropped as censored. This
  leaves 95,630 calls, with a base rate of 33.8%.

## Results

### 1. Level-up within the next 10k thinking tokens

Quintiles of $d$, ranked within each game:

| Quintile | Median $d$ per 1k | P(level-up) | Excluding current call |
|---|---:|---:|---:|
| 1 (fewest) | 3.0 | 43.2% | 37.8% |
| 2 | 4.6 | 40.0% | 35.6% |
| 3 | 5.6 | 35.2% | 31.9% |
| 4 | 6.7 | 30.0% | 27.4% |
| 5 (most) | 8.7 | 20.8% | 19.2% |

Odds ratio per 1 SD of $d$, with SEs clustered by game run:

| Model | OR per SD |
|---|---:|
| game + run + level fixed effects | 0.57 [0.54, 0.61] |
| + log thinking volume, log tokens on level | 0.52 [0.49, 0.56] |
| + "wait" density | 0.52 [0.48, 0.56] ("wait": 1.08) |
| excluding the current call's action | 0.58 [0.55, 0.62] |
| conditional logit within each (game run, level), density only | 0.69 [0.67, 0.72] |
| conditional logit, + log thinking volume, log tokens on level | 0.84 [0.81, 0.88] |
| single-call density (≥ 300 thinking tokens) | 0.60 [0.58, 0.62] |

- **Within a level attempt:** a linear probability model with (game run, level)
  fixed effects and the same controls gives −9.1 pp per SD [−10.1, −8.0] on the
  33.8% base. The agent clears levels when its "hmm" rate is at its lowest for
  that level. The conditional logit fits warn of numerical overflow on very long
  levels, so the linear model is the more reliable within-level estimate.
- **Not a length effect:** ranked within each game, $d$ alone gives an AUC of
  0.60. Thinking volume (0.49) and tokens on the level (0.51) carry no signal on
  their own here. Given $d$, more thinking volume slightly *raises* the odds.
- **Specific to "hmm":** "wait" density does not substitute for it.
- **Holds across runs:** per-run within-level odds ratios (controlling for
  tokens on level) fall below 1 in 41 of 50 runs (22 with p < 0.05), with a
  median of 0.72.
- **Other horizons:** at 5k thinking tokens, 0.46 per SD with controls and
  −9.8 pp within level (base 23.0%). At 20k it is 0.60 across games, but only
  −5.9 pp within level (base 49.4%), with a per-run median of 0.92. The signal
  is about the near term.
- **Clock check:** counting tool-call code in the clock as well gives nearly the
  same result: 39.8% → 18.1% across quintiles, 0.50 per SD with controls, and
  −8.6 pp within level.

### 2. Does the level eventually clear?

The allocator does not need to know *when* a level clears, only *whether* more
tokens will clear it. Taking each level once, at its first call past a
checkpoint of thinking tokens spent on it, and using game + level fixed effects:

| Thinking tokens on level | Levels | Cleared | Low / mid / high $d$ tercile | OR per SD |
|---|---:|---:|---|---:|
| 10k | 3,564 | 83% | 87 / 84 / 77% | 0.78 [0.68, 0.90] |
| 20k | 2,420 | 77% | 84 / 80 / 69% | 0.69 [0.59, 0.80] |
| 40k | 1,263 | 65% | 69 / 64 / 63% | 0.80 [0.69, 0.94] |
| 80k | 449 | 47% | 48 / 47 / 45% | fit is singular |

This is real but much weaker than the short-horizon effect: a spread of 3–15
points between terciles, narrowing on long levels. "Cleared" also depends on how
many tokens each game run's own scheduler gave it.

## An allocator term

The deployed ranking (`configs/level-up-board/v1.json`, `mamba-fp32-state`) is
$R = v\,C$ from [priority-replay.md](../allocation/priority-replay.md):
$B = 0$ via `ARC3_PRIORITY_TAIL_CAP=0`, efficiency off via
`ARC3_PRIORITY_HUMAN_ACTIONS=1e9`, and the default $C$. It is recomputed at
each context-trim handover. The candidate term is a confusion multiplier $u$ on
$R$, of the same shape as the existing `ARC3_PRIORITY_THINKING` multiplier:

$$
u = \operatorname{clip}\!\left((\mu/d)^{\alpha},\ \tfrac12,\ 2\right),\qquad
R = v\,C\,u .
$$

- $d = H / T$ as above, over the last $W$ thinking tokens on the current level.
- $\mu = 0.0058$ hmms per thinking token (5.8 per 1k), the pooled density. Units
  cancel in $\mu/d$. A game at the usual rate keeps $u = 1$, twice the usual rate
  gives $2^{-\alpha}$, half gives $2^{\alpha}$. $\mu$ scales every game alike,
  so it matters only through the clip.
- $u = 1$ while $T < T_{\min}$: too little thinking on the level to judge. A
  level with no "hmm" yet ($d = 0$) gets the cap, $u = 2$.
- The signal is thinking-only, but the allocation budget stays in generated
  tokens (thinking + tool-call code), since that is what the server spends.

### Calibrated $\alpha$

If $u$ tracks how the chance of success changes with $d$, then
$P \propto d^{-\alpha}$. A relative-risk fit (Poisson, log link, SE clustered by
game run) on $\log d$ gives:

| Outcome | $\alpha$ |
|---|---:|
| level-up within the next 10k thinking tokens (game + run + level FE) | 0.46 [0.41, 0.52] |
| eventually clears the level, at 10k thinking tokens on it | 0.13 [0.09, 0.17] |
| same, at 20k | 0.17 [0.13, 0.22] |
| same, at 40k | 0.09 [0.01, 0.18] |

With game + level fixed effects, the eventual-clear values are 0.06, 0.13 and
0.08. The allocator needs the eventual-clear chance (section 2), so the
calibrated value is about $\alpha \approx 0.1$–$0.2$.

### Replay

The same replay as [priority-replay.md](../allocation/priority-replay.md):
275 trajectories from 11 runs. Calls are placed on each trajectory's
generated-token axis by piecewise-linear interpolation between level-up knots.
Full output in [`data/hmm_replay.md`](data/hmm_replay.md).

**$\alpha$ sweep** on the deployed score, $R = v\,C\,u$ with
$W = 10\text{k}$, $T_{\min} = 1\text{k}$ and clip $[\tfrac12, 2]$. $\alpha = 0$
is $R = v\,C$. The $K = 14$ columns are paired differences over the 11 runs
($\gamma = 35\text{k}$). The pool column is the private-pool shape: $K = 15$ on
200 pools of 110 games, $\beta = 150\text{k}$, the same pools for every
$\alpha$, ± leave-one-run-out jackknife SE.

| $\alpha$ | $K = 14$, $\beta = 100\text{k}$ | $K = 14$, $\beta = 150\text{k}$ | $K = 15$ pools, $\beta = 150\text{k}$ |
|---|---|---|---|
| 0 ($R = v\,C$) | 29.42 | 40.33 | 41.84 |
| 1/8 | −0.34 ± 0.26 (3/11) | −0.11 ± 0.07 (2/11) | +0.02 ± 0.18 |
| 1/4 | −0.27 ± 0.23 (3/11) | +0.20 ± 0.14 (5/11) | −0.02 ± 0.19 |
| 3/8 | +0.33 ± 0.40 (5/11) | −0.10 ± 0.20 (4/11) | −0.17 ± 0.22 |
| 1/2 | +0.12 ± 0.43 (5/11) | −0.11 ± 0.21 (4/11) | −0.26 ± 0.29 |
| 3/4 | +0.33 ± 0.39 (7/11) | +0.01 ± 0.17 (3/11) | −0.64 ± 0.41 |
| 1 | +0.03 ± 0.45 (6/11) | +0.04 ± 0.16 (3/11) | −0.75 ± 0.44 |
| 3/2 | +0.11 ± 0.47 (6/11) | −0.49 ± 0.30 (2/11) | −0.88 ± 0.49 |
| 2 | +0.27 ± 0.47 (7/11) | −0.66 ± 0.47 (3/11) | −1.06 ± 0.54 |

- **Best $\alpha$ is about 0.** On the pools the score is flat up to
  $\alpha = 1/4$ (+0.02, −0.02) and falls steadily after it, to −1.06 at
  $\alpha = 2$. The $K = 14$ columns are noise around zero.
- **Selection check:** picking $\alpha$ on pools from runs 1–6 gives
  $\alpha = 0$. Picking it on runs 7–11 gives $1/8$ (+0.33 in sample), which
  scores −0.18 on runs 1–6.
- **Agrees with the calibration:** the flat region, $\alpha \le 1/4$, contains
  the calibrated eventual-clear value of 0.1–0.2. Larger $\alpha$ acts on the
  short-horizon signal, which only shifts *when* a level clears.

Other settings on the pools, difference vs $v\,C$ (jackknife ± SE):

| Setting | vs $v\,C$ |
|---|---|
| best of a $W \in \{5, 10, 20\}\text{k}$, $T_{\min} \in \{1, 3\}\text{k}$, 6-value $\alpha$ grid at $K = 14$ ($W = 20\text{k}$, $T_{\min} = 3\text{k}$, $\alpha = 1$) | −0.55 ± 0.52 |
| earlier form that shrank $d$ toward $\mu$ instead of using $T_{\min}$ ($\alpha = 1/2$, $N_0 = 3\text{k}$) | −0.09 ± 0.32 |
| uniform | −5.30 ± 1.03 |

- Changing $W$ or $T_{\min}$ does not help, and neither do shorter grants: at
  $\gamma = 10\text{k}$ and $20\text{k}$, $\alpha = 1/2$ is within ±0.5 of
  $v\,C$ and $\alpha = 2$ loses 0.16–1.81 points.
- The clip matters: widening it to $[\tfrac14, 4]$ or $[\tfrac18, 8]$ loses
  0.2–0.4 points at $\alpha = 1$.
- Using thinking volume instead of "hmm" density also ties.
- Why a strong predictor does not help: in the replay, holding back a
  high-"hmm" game only delays a level-up it would still reach. The allocator
  gains only from signals about whether a level *ever* clears within budget
  (section 2). That signal is weaker, and the token falloff in $C$ already
  carries most of it.

## Conclusion

"Hmm" density is a clean, length-independent marker of confusion. A game with a
high rate is about half as likely, per SD, to clear its level in the next 10k
thinking tokens, and the effect holds within a level, across games and in 41 of
50 runs. It is a correlation: it does not show that suppressing "hmm" would
help.

As an allocator term it does not help: the best $\alpha$ on the existing score
is about 0. $\alpha \le 1/4$ is harmless and $\alpha \ge 1/2$ costs points.
The most it could be is $\alpha = 1/4$, $W = 10\text{k}$,
$T_{\min} = 1\text{k}$, clip $[\tfrac12, 2]$ (replay −0.02 ± 0.19 on the
private-pool shape; calibrated value 0.1–0.2), which moves a typical game only
between $0.90\times$ ($d = 8.7$ per 1k) and $1.18\times$ ($d = 3.0$ per 1k).

**Decision (2026-10-12): no Kaggle A/B.** The replay expects no gain, so an
allocator arm is not worth a run. I think the signal is more likely to pay off
inside the agent, for example as a stuck detector that changes its prompt when
the density stays high late in a level, than in the allocator.

## Reproduce

From the main checkout, where `ARC3-Inference/runs/` exists:

```sh
pip install tokenizers   # tokenizer file: see scripts/extract_transcript_calls.py
python3 docs/reasoning/scripts/hmm_extract.py                   # -> data/hmm_calls.csv.gz
python3 docs/reasoning/scripts/hmm_vs_level_up.py 10000 think   # section 1 (also 5000, 20000; `out` adds tool-call code to the clock)
python3 docs/reasoning/scripts/hmm_within_level.py 10000 think  # within-level LPM, AUC, section 2, calibrated alpha
python3 docs/reasoning/scripts/hmm_replay.py                    # replay and alpha sweep -> data/hmm_replay.md
```
