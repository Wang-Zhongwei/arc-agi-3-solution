# Priority allocation: offline replay of recorded trajectories

Question: under a fixed generated-token budget, which ranking of games
maximizes the mean game score, and how does the shipped priority compare with
uniform advancement?

Summary (replay, 11 runs, 14 slots, paired across runs):

- The shipped priority beats uniform advancement by $+3.0 \pm 0.4$ points at
  $\beta = 100\text{k}$ tokens per game and $+1.9 \pm 0.4$ at $150\text{k}$.
- Setting the future-level tail $B = 0$ raises this to $+5.0 \pm 0.9$ and
  $+2.6 \pm 0.6$. The level value $v$ is what helps; the probability term
  ($C$, or a refitted $\hat P$) alone gives about $+3$ / $+1.5$.
- $C$ is a calibrated completion rate. On recent runs it matches the measured
  rate up to about $60\text{k}$ tokens on a level and is 2–5× too low beyond
  $75\text{k}$; replacing it with the refit changes the replay by $-0.8$ to
  $+0.5$ points, within noise.
- Projecting the human baseline per level in the efficiency term does not
  help; dropping efficiency ties at $35\text{k}$ grants and wins at
  $15\text{k}$ grants.
- Finer switching is not monotonically better: the optimum grant is about
  $15\text{k}$ for $v \cdot C$ and $35\text{k}$ for the shipped rule.
- The score-vs-tokens viewer curve cannot show any of this: it evaluates every
  game at the same per-game budget, i.e. it is the uniform policy by
  construction (`ARC3-Inference/viewer/curves.html`, `pool()`).

## 1. Data

| Run (Kaggle notebook, version) | Games × passes | Used for |
| --- | --- | --- |
| `jonathanwang2022/bp35-lordhansolo` v02, v03, v07, v08, v09 | 25 × 1 each | replay |
| `jonathanwang2022/persistent-animations` v01 | 25 × 1 | replay |
| `ticbudja/thinking-budget-and-user-prompt-trim` v01, v02, v03 | 25 × 1 each | replay |
| `ticbudja/yield-ucb-scheduling-1001` v01 | 25 × 1 | replay |
| `dangthuygiang/arc3-v207-danibase-base-dfranzen` v05 | 25 × 1 | replay |
| `jonathanwang2022/bp35-lordhansolo` v07 | 25 × 1 | gate waits, vLLM throughput |

Per game and pass, `evaluation.json → progression_lanes` stores the score as
a step function of the game's own generated tokens and of its own actions,
with one knot per level-up. The two lanes share their knots (checked for all
275 trajectories). Human baselines `base_actions_per_level` come from
`benchmark.json`. Extracted data: [`data/trajectories.json`](data/trajectories.json),
[`data/baselines.json`](data/baselines.json),
[`data/gate_waits_v07.csv`](data/gate_waits_v07.csv),
[`data/vllm_throughput_v07.json`](data/vllm_throughput_v07.json).
All tables below are printed by [`scripts/replay.py`](scripts/replay.py) into
[`data/results.md`](data/results.md).

## 2. Notation

Games $g = 1,\dots,G$ ($G = 25$ per run) with $n_g$ levels. The official
score of a game (`tufa-arc-agi-framework/src/taaf/game.py`, `_compute_final_score`) is

$$
S_g = \frac{\sum_{k=1}^{n_g} k \cdot \mathbb 1[\text{level } k \text{ cleared}] \cdot
\min\!\left(115,\ 100\,(b_{g,k}/a_{g,k})^2\right)}{n_g(n_g+1)/2},
$$

capped at $100 \cdot \sum_{k:\,\text{scored}} k \,/\, \tfrac{n_g(n_g+1)}{2}$, where
$b_{g,k}$ is the human baseline and $a_{g,k}$ the actions used on level $k$.
The benchmark score is $\bar S = \frac1G \sum_g S_g$.

Let $S_g(\tau)$ be the recorded score after the game has generated $\tau$
tokens, and $T_g$ the last recorded token count. At $\tau$:

- $L = L_g(\tau)$: levels completed; the current level is $k = L + 1$.
- $s = s_g(\tau)$: tokens since the last level-up (tokens on the current level).
- $a = a_g(\tau)$: actions since the last level-up, linearly interpolated in
  $\tau$ between knots.

An allocation gives game $g$ a total of $\tau_g$ tokens with
$\sum_g \tau_g = G\beta$. The objective is

$$
\max_{\tau_1,\dots,\tau_G}\ \frac1G \sum_g S_g(\tau_g)
\quad\text{s.t.}\quad \sum_g \tau_g = G\beta .
$$

**Assumption (A1).** $S_g(\cdot)$ does not depend on when the tokens are
generated: pausing a game does not change what its next tokens do.

## 3. The shipped scheduler

Settings of the Lord Hansolo configuration (`configs/lordhansolo/v11.json`
`harness_env`, not tracked): `ARC3_SCHEDULER=priority`,
`ARC3_MAX_ACTIVE_STREAMS=14`, `ARC3_PRIORITY_REFRESH_QUEUE=1`,
`ARC3_PRIORITY_TAIL_FADE=1`, `ARC3_PRIORITY_TAIL_FADE_FRACTION=0.2`,
`ARC3_PRIORITY_TAIL_LOOKUP=remaining`, `ARC3_PRIORITY_SCORE_NORMALIZATION=1`.

**Gate.** At most $K = 14$ games generate. A game keeps its slot until its
context is trimmed, then re-enters the queue (`_maybe_handover`,
`tool_agent.py:6580`). Unstarted games enter with priority
$2{,}000{,}000 - \text{queue position}$ (`solver.py:630`), so every game starts
before any trimmed game resumes. Only $G - K$ games can wait at any time.

**Priority** (`scheduling/priority.py:97`):

$$
\rho = \max\!\big(1,\ \lfloor 100\,(A + B)\,C \rfloor\big),\qquad
A = k\, e(a)\, \frac{55}{n(n+1)/2},\qquad e(a) = \left(\frac{25}{25 + a}\right)^2,
$$

$$
C = 0.25 \cdot 2^{-(a/115)^2} + 0.75 \cdot 2^{-(s/62000)^2},\qquad
B = \begin{cases} 8 & n - k \ge 3\\ 7 & n - k = 2\\ 5 & n - k = 1\\ 0 & n - k = 0\end{cases}
\times \min\!\Big(1, \frac{r}{0.2\,T}\Big),
$$

with $r$ the time remaining on the gate's clock and $T$ its total duration.
$C$ was calibrated on 725 level attempts from four earlier runs as the
completion rate per window of spend among attempts still at risk
([WRITEUP.md](../../WRITEUP.md#3-allocate-computation-where-it-can-still-help)).

**Observed grants** (v07, transcript timestamps; a gap $> 150$ s between
steps counts as a wait): games waited 794 of 2,142 game-minutes (37%);
110 grants with median $\approx 34\text{k}$ tokens, the first grant per game
$46$–$108\text{k}$. Games stuck at 0–1 levels (sk48, cn04, lf52, bp35, s5i5)
waited longest, 46–69 min of about 100, but still received
$159$–$263\text{k}$ tokens ([`data/gate_waits_v07.csv`](data/gate_waits_v07.csv)).

**Private waves** (branch `bp35`, `_configure_private_waves` in
`taaf_kaggle_run.ipynb`): lanes $= \lceil \text{trials}/4 \rceil$, per-game cap
$= (\text{remaining budget})/4$, one shared gate. `configure_clock`
(`tool_agent.py:2081`) keeps $\min$ of all deadlines, i.e. wave 1's. For games
of waves 2–4, $r = 0$ and therefore $B = 0$ throughout.

## 4. Replay

**Algorithm.** For one run, start with $\tau_g = 0$. Each round, let
$\mathcal L = \{g : g \text{ not won},\ \tau_g < T_g\}$. Rank $\mathcal L$ by
$(\mathbb 1[\tau_g = 0],\ R(g, \tau_g))$ descending, give the top $K$ games
$\gamma$ tokens each, and stop when $\sum_g \tau_g = G\beta$ or
$\mathcal L = \varnothing$. Report $\frac1G \sum_g S_g(\tau_g)$.

- $K = 14$ matches the gate; "free choice" uses $K = 1$.
- $\gamma = 35\text{k}$ approximates the observed grant (§3); $5$, $15$, $70\text{k}$
  are tested in §5.5.
- A game that reaches $T_g$ without winning leaves $\mathcal L$: there is no
  data beyond it. Table T1 gives the coverage $\sum_g T_g / (G\beta)$.

**T1. Recorded tokens / replay budget**

| $\beta$ (tokens/game) | $\sum_g T_g / (G\beta)$ |
|---|---|
| 100k | 1.82 |
| 150k | 1.22 |
| 200k | 0.91 |

At $\beta = 200\text{k}$ the data run out before the budget, all policies
converge, and the replay cannot rank them. Results use $\beta \in \{100, 150\}\text{k}$.

**Statistics.** For runs $r = 1,\dots,11$ and two policies $X, Y$:
$d_r = \bar S^{(r)}_X - \bar S^{(r)}_Y$, reported as
$\bar d \pm \mathrm{sd}(d)/\sqrt{11}$ and the number of runs with $d_r > 0$.

**Fitted level-up probability.** With stall bins
$\mathcal B = \{[0,20), [20,40), \dots, [100,130), [130,160), [160,\infty)\}\text{k}$,
sample every trajectory at $\tau \in \{0, 5\text{k}, 10\text{k}, \dots\}$ and set

$$
\hat P(j, \ell) = \frac{h_{j\ell} + 0.5}{m_{j\ell} + 2},\qquad
h_{j\ell} = \#\{\text{samples in bin } j,\ \mathbb 1[L \ge 1] = \ell,\ \text{level-up in } (\tau, \tau + D]\},
$$

where $m_{j\ell}$ also counts samples with no level-up that were observed for
the full window $D = 35\text{k}$ (censored samples are dropped). In the replay
$\hat P$ is fitted on the other 10 runs.

**Indices $R(g, \tau)$.** With $v = 100\,k / \tfrac{n(n+1)}{2}$, the score points
of clearing the current level at full efficiency:

| Name | $R$ |
|---|---|
| uniform | $-\tau$ |
| shipped | $\rho$ above with $r/(0.2T) = 1$ (no fade) |
| lookup off | $(A + \min(4, k + 1))\, C$, the code default tail |
| $B = 0$ | $v\, e(a)\, C \ \propto A\,C$ |
| $B = 0$, no efficiency | $v\, C$ |
| $C$, $\hat P$ | $C$, $\hat P$ |
| $v \hat P$, $v e \hat P$ | $v\,\hat P$, $v\,e(a)\,\hat P$ |
| stall | $-s$ |
| TOKEN_SCALE 150k | shipped with $0.5^{(s/150000)^2}$ and no action term |
| lookahead | $S_g(\tau + \gamma) - S_g(\tau)$ (uses the future; myopic oracle) |

## 5. Results

### 5.1 Policies vs uniform ($\gamma = 35\text{k}$)

Cells: mean $\bar S$, then $\bar d$ vs uniform ± SE (runs better / 11).

| Policy | $K=14$, $\beta = 100\text{k}$ | $K=14$, $\beta = 150\text{k}$ | $K=1$, $\beta = 100\text{k}$ | $K=1$, $\beta = 150\text{k}$ |
|---|---|---|---|---|
| uniform | 24.6 | 37.3 | 24.6 | 37.3 |
| shipped | 27.6, +3.01 ± 0.40 (11) | 39.2, +1.88 ± 0.42 (9) | 28.9, +4.33 ± 0.51 (11) | 38.3, +1.05 ± 0.42 (8) |
| lookup off | 29.6, +5.07 ± 0.76 (11) | 39.3, +2.04 ± 0.49 (9) | 30.1, +5.54 ± 0.62 (11) | 39.6, +2.33 ± 0.51 (9) |
| $B=0$ | 29.5, +4.95 ± 0.86 (10) | 39.9, +2.64 ± 0.56 (9) | 30.8, +6.20 ± 0.76 (11) | 40.4, +3.06 ± 0.57 (9) |
| $B=0$, no efficiency | 29.4, +4.84 ± 0.91 (11) | 40.3, +3.03 ± 0.84 (9) | 30.4, +5.78 ± 0.91 (11) | 41.3, +4.01 ± 0.80 (9) |
| $v\hat P$ | 29.9, +5.31 ± 0.88 (11) | 40.3, +3.00 ± 0.80 (9) | 30.4, +5.84 ± 0.92 (11) | 40.5, +3.24 ± 0.87 (9) |
| stall | 27.7, +3.12 ± 0.70 (10) | 38.6, +1.26 ± 0.48 (7) | 29.1, +4.53 ± 0.94 (10) | 38.9, +1.59 ± 0.69 (6) |
| TOKEN_SCALE 150k | 27.9, +3.31 ± 0.45 (11) | 37.8, +0.47 ± 0.36 (5) | 29.5, +4.93 ± 0.59 (11) | 37.3, −0.03 ± 0.40 (6) |
| lookahead | 31.6, +6.99 ± 0.74 (11) | 38.9, +1.61 ± 0.51 (8) | 33.2, +8.59 ± 0.90 (11) | 39.5, +2.16 ± 0.54 (8) |

The shipped tail is almost constant ($B = 8$ while three or more levels
remain) and larger than $A$, so $\rho \approx 8\,C$: the order is set by $C$ and
the level depth barely matters. The docstring of `_priority_tail_cap`
(`tool_agent.py:2290`) reports the same from an earlier replay: a flat tail was
the worst option at every constrained budget. Flattening $C$
(TOKEN_SCALE 150k) removes the part that works.

### 5.2 Which factor helps

$\bar d$ vs uniform, $K = 14$, $\gamma = 35\text{k}$:

| Index | $\beta = 100\text{k}$ | $\beta = 150\text{k}$ |
|---|---|---|
| $C$ | +3.58 ± 0.53 (11) | +1.65 ± 0.44 (9) |
| $\hat P$ | +3.14 ± 0.79 (10) | +1.49 ± 0.34 (9) |
| $v\,e\,C$ ($B = 0$) | +4.95 ± 0.86 (10) | +2.64 ± 0.56 (9) |
| $v\,C$ | +4.84 ± 0.91 (11) | +3.03 ± 0.84 (9) |
| $v\,\hat P$ | +5.31 ± 0.88 (11) | +3.00 ± 0.80 (9) |
| $v\,e\,\hat P$ | +4.19 ± 0.70 (10) | +2.86 ± 0.62 (9) |

$\hat P$ is the probability of a level-up per $D$ tokens, so
$v \hat P / D$ is the expected score per token of the next grant. The
probability term alone ($C$ or $\hat P$) only orders games by how stalled they
are; multiplying by $v$ adds $1.3$–$2.2$ points. Replacing $C$ by $\hat P$
changes the result by $-0.8$ to $+0.5$, within about one SE.

### 5.3 Calibration of $C$

Life table on the same metric as the original calibration: level attempts
(1,326, of which 1,130 completed) at risk at the start of each window of spend,
$\hat h = d / (n_{\text{risk}} - c/2)$ with $d$ completions and $c$ attempts
whose run ended in the window. The last column is $C$'s token curve scaled to
the first window, $\hat h_0\, 2^{-(x/62000)^2}$ at the window midpoint $x$.

| Tokens on level | At risk | Completed | Censored | $\hat h$ | $C$ curve |
|---|---|---|---|---|---|
| 0–15k | 1326 | 412 | 22 | 31.3% | 31.0% |
| 15–30k | 892 | 322 | 14 | 36.4% | 28.6% |
| 30–45k | 556 | 167 | 18 | 30.5% | 24.3% |
| 45–60k | 371 | 90 | 22 | 25.0% | 19.1% |
| 60–75k | 259 | 52 | 21 | 20.9% | 13.8% |
| 75–90k | 186 | 37 | 21 | 21.1% | 9.2% |
| 90–105k | 128 | 20 | 17 | 16.7% | 5.6% |
| 105–120k | 91 | 9 | 16 | 10.8% | 3.2% |
| 120–135k | 66 | 10 | 9 | 16.3% | 1.7% |
| 135–150k | 47 | 1 | 13 | 2.5% | 0.8% |

Actions (windows of 25, $\hat h_0\, 2^{-(x/115)^2}$):

| Actions on level | At risk | Completed | Censored | $\hat h$ | $C$ curve |
|---|---|---|---|---|---|
| 0–25 | 1326 | 472 | 34 | 36.1% | 35.8% |
| 25–50 | 820 | 356 | 37 | 44.4% | 33.5% |
| 50–75 | 427 | 149 | 27 | 36.0% | 29.4% |
| 75–100 | 251 | 80 | 32 | 34.0% | 24.1% |
| 100–125 | 139 | 25 | 29 | 20.1% | 18.6% |
| 125–150 | 85 | 23 | 10 | 28.7% | 13.4% |
| 150–175 | 52 | 6 | 10 | 12.8% | 9.0% |
| 175–200 | 36 | 10 | 6 | 30.3% | 5.7% |
| 200–225 | 20 | 4 | 4 | 22.2% | 3.4% |

Up to about $60\text{k}$ tokens, where 80% of the at-risk observations lie,
$C$ agrees within a few points. Beyond $75\text{k}$ the measured rate stays at
11–21% per window while the Gaussian falls to 2–9%. The fitted $\hat P$ on all
runs ($D = 35\text{k}$; a 15k-window rate $h$ corresponds to
$1 - (1 - h)^{35/15}$):

| | 0–20k | 20–40k | 40–60k | 60–80k | 80–100k | 100–130k | 130–160k | 160k+ |
|---|---|---|---|---|---|---|---|---|
| $L = 0$ | 0.77 | 0.71 | 0.69 | 0.36 | 0.33 | 0.29 | 0.29 | 0.12 |
| $L \ge 1$ | 0.60 | 0.52 | 0.47 | 0.44 | 0.34 | 0.33 | 0.30 | 0.38 |

A floored form $C' = c_0 + (1 - c_0)\, 2^{-(s/62000)^2}$ with
$c_0 \approx 0.45$ would match the tail; by §5.2 the replay cannot
distinguish it from $C$.

### 5.4 Efficiency term

Human baselines grow with the level (25 public games):

| Level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Median $b_k$ | 30 | 54 | 51 | 54 | 96 | 80 | 86 | 92 | 163 | 225 |
| Games | 25 | 25 | 25 | 25 | 25 | 25 | 16 | 11 | 5 | 1 |

Variants of $e$ inside $R = v\, e\, C$, with $\hat p_k$ the median baseline
at level $k$ over the other 24 games (levels $\ge 7$ pooled). Cells:
$\bar d$ vs $e = (25/(25+a))^2$.

| $e$ | $\gamma = 35\text{k}$, $\beta = 100\text{k}$ | $\gamma = 35\text{k}$, $\beta = 150\text{k}$ | $\gamma = 15\text{k}$, $\beta = 100\text{k}$ | $\gamma = 15\text{k}$, $\beta = 150\text{k}$ |
|---|---|---|---|---|
| $1$ (none) | −0.11 ± 0.62 (6) | +0.39 ± 0.45 (5) | +2.32 ± 0.83 (9) | +1.20 ± 0.34 (9) |
| $(\hat p_k/(\hat p_k + a))^2$ | −0.00 ± 0.62 (4) | −0.45 ± 0.33 (3) | −0.11 ± 0.52 (4) | −0.76 ± 0.26 (1) |
| $\min(1, (\hat p_k/a)^2)$ | +0.30 ± 0.75 (7) | −0.06 ± 0.31 (4) | +0.80 ± 0.46 (8) | −0.16 ± 0.22 (4) |
| $\min(1, (b_{g,k}/a)^2)$, true $b$ | −1.31 ± 0.59 (3) | −0.24 ± 0.59 (4) | +0.95 ± 0.36 (9) | +0.08 ± 0.62 (6) |

Even the true baseline does not help. $e$ decreases in $a$, and $a$ grows
while a level is stuck, so $e$ repeats the stall information already in $C$.
With $B = 0$ it also lowers games stuck on an expensive level whose clearance
unlocks deeper, higher-weight levels. `ARC3_PRIORITY_HUMAN_ACTIONS` has no
upper clamp, so $10^9$ sets $e \approx 1$ without code changes.

### 5.5 Grant size

Mean $\bar S$, $K = 14$:

| $\beta$ | Policy | $\gamma=5\text{k}$ | $15\text{k}$ | $35\text{k}$ | $70\text{k}$ |
|---|---|---|---|---|---|
| 100k | uniform | 24.7 | 24.3 | 24.6 | 24.7 |
| 100k | shipped | 26.7 | 27.0 | 27.6 | 27.4 |
| 100k | $v\,e\,C$ | 30.4 | 29.6 | 29.5 | 27.6 |
| 100k | $v\,C$ | 30.3 | 32.0 | 29.4 | 28.7 |
| 100k | lookahead | 26.7 | 28.0 | 31.6 | 32.5 |
| 150k | uniform | 37.4 | 37.2 | 37.3 | 36.0 |
| 150k | shipped | 38.3 | 38.3 | 39.2 | 38.2 |
| 150k | $v\,e\,C$ | 40.5 | 40.3 | 39.9 | 39.1 |
| 150k | $v\,C$ | 40.8 | 41.5 | 40.3 | 39.9 |
| 150k | lookahead | 37.8 | 37.2 | 38.9 | 40.2 |

Finer grants are optimal for an index policy only when switching is free and
the index values the whole remaining reward of a game (Gittins). The indices
here are myopic and decrease in $s$, so very small grants abandon partly-solved
levels; the lookahead oracle is the extreme case, losing as $\gamma$ shrinks.

**Switching cost.** At a trim the prefix is already invalid, so switching
there is free. A preemption inside a context costs a re-prefill of about
$X \approx 80\text{k}$ tokens. With v07's peak prompt throughput
$\approx 15.1\text{k}$ tokens/s and generation $\approx 895$ tokens/s
([`data/vllm_throughput_v07.json`](data/vllm_throughput_v07.json)), one forced
switch displaces

$$
c \approx \frac{X}{15.1\text{k}} \times 895 \approx 4.7\text{k generated tokens}.
$$

With a fraction $f$ of $15\text{k}$ grants ending in a forced switch and
about $218\text{k}$ tokens per game (v07), the cost is
$f \cdot \tfrac{218}{15} \cdot 4.7\text{k} / 218\text{k} \approx 0.31 f$ of the
budget. The marginal return at $150$–$300\text{k}$ tokens is $18$–$22$ points
per $100\text{k}$ tokens per game (T9 below), so the $+1.2$ to $+2.6$ point
gain of $\gamma = 15\text{k}$ is worth 2.5–7% of the budget, and it pays only
for $f \lesssim 0.08$–$0.21$.

**T9. Score points per 100k tokens, by tokens the game had already generated**
(game-equivalents of data in the bin in parentheses):

| Runs | 0–50k | 50–100k | 100–150k | 150–200k | 200–300k |
|---|---|---|---|---|---|
| all 11 | 21.7 (275) | 28.3 (256) | 26.6 (212) | 22.3 (151) | 20.0 (50) |
| bp35-lordhansolo | 21.9 (125) | 26.3 (121) | 21.3 (113) | 18.2 (89) | 18.5 (27) |

## 6. Proposal

1. Set $B = 0$: `ARC3_PRIORITY_TAIL_LOOKUP` off and `ARC3_PRIORITY_TAIL_CAP=0`
   (config only). The ranking becomes $A\,C \propto v\,e\,C$; tail fade and
   the stale wave clock (§3) no longer change anything.
2. Optionally `ARC3_PRIORITY_HUMAN_ACTIONS=1e9` ($e \approx 1$): neutral at
   today's trim-aligned grants, better with finer grants.
3. Keep switching at trims. Add mid-context preemption only for a stalled
   running game ($s \gtrsim 40\text{k}$) when a waiting game ranks at least
   about $2\times$ higher, and measure $f$ and the prefill cost on a GPU run.
4. Refitting $C$ (floored tail) or replacing it with $\hat P$ shows no
   measurable gain in the replay ($-0.8$ to $+0.5$).

## 7. Limitations

- (A1) is assumed, not tested. The replay ignores wall-clock throughput,
  prefill, prefix-cache effects, per-game time caps, and waves.
- Rounds are synchronous; the real gate switches one game at a time.
- A grant that overshoots $T_g$ is counted in full.
- Actions within a level are interpolated linearly in tokens.
- Policies that would give a game more than $T_g$ are cut off, which
  understates concentrating policies. Real runs use about $220\text{k}$ tokens
  per game, beyond what the replay can rank (T1).
- Differences of 1–2 points are visible only because comparisons are paired.
  A single 25-game Kaggle run has a standard deviation of roughly 3–5 points;
  a Kaggle A/B needs several paired runs ([t-test guide](../rules/t-test.md)).

## 8. Reproduce

From the main checkout, where `ARC3-Inference/runs/` exists:

```sh
python docs/allocation/scripts/extract_trajectories.py --runs-root ARC3-Inference/runs
python docs/allocation/scripts/extract_gate_waits.py \
    --run-dir ARC3-Inference/runs/kaggle/jonathanwang2022/bp35-lordhansolo/v07
python docs/allocation/scripts/replay.py   # writes data/results.md
```

`replay.py` needs only `data/` and imports `priority_value` from
`ARC3-Inference/inference/agent/scheduling/priority.py`.
