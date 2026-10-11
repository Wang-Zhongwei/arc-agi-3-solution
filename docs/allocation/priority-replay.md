# Priority allocation: offline replay

Question: with a fixed generated-token budget, does dropping the future-level
tail $B$, or the efficiency term, improve the priority ranking?

## Data

Recorded per-game score trajectories from 11 public Kaggle runs (25 games × 1
pass each, 275 trajectories): `jonathanwang2022/bp35-lordhansolo` v02, v03,
v07, v08, v09; `jonathanwang2022/persistent-animations` v01;
`ticbudja/thinking-budget-and-user-prompt-trim` v01–v03;
`ticbudja/yield-ucb-scheduling-1001` v01;
`dangthuygiang/arc3-v207-danibase-base-dfranzen` v05. Each run's
`evaluation.json → progression_lanes` stores a game's score as a step function
of its own generated tokens and of its own actions, with one knot per level-up.
Extracted to [`data/trajectories.json`](data/trajectories.json).

Thinking per turn comes from the transcripts: the `reasoning_chars` of each
turn's last model call (the call that emitted its actions), at 4 characters
per token, placed on the game's token axis with `benchmark.json`
`history[].generated_tokens`. Extracted to
[`data/thinking.json`](data/thinking.json) (14,883 turns).

## Model

For game $g$ with $n$ levels, $S_g(\tau)$ is the recorded score after the game
has generated $\tau$ tokens, and $T_g$ is the last recorded token count. At
$\tau$, let $L$ be the levels completed ($k = L + 1$ is the current level), $s$
the tokens and $a$ the actions spent since the last level-up. An allocation
gives game $g$ a total of $\tau_g$ tokens, and the objective is

$$
\max_{\tau}\ \bar S = \frac1G \sum_{g=1}^{G} S_g(\tau_g)
\quad\text{s.t.}\quad \sum_g \tau_g = G\beta .
$$

The replay assumes $S_g(\cdot)$ does not depend on when the tokens are generated.

**Rankings.** With

$$
v = \frac{100\,k}{n(n+1)/2},\qquad
e = \left(\frac{25}{25 + a}\right)^2,\qquad
C = 0.25 \cdot 2^{-(a/115)^2} + 0.75 \cdot 2^{-(s/62000)^2},
$$

| Policy | Index $R$ |
|---|---|
| current | $\lfloor 100\,(A + B)\,C \rfloor$, $A = k\,e\,\frac{55}{n(n+1)/2}$, $B = 8, 7, 5, 0$ for $n - k \ge 3, 2, 1, 0$ |
| $B = 0$ | $v\,e\,C \ \propto A\,C$ |
| $B = 0$, efficiency ablated | $v\,C$ |
| $B = 0$ + pace | $v\,e\,C_p / p$ |
| $B = 0$ + thinking | $v\,e\,C\,m$ |
| uniform (baseline) | $-\tau$ |

**Pace** (`ARC3_PRIORITY_PACE`, `ProgressPace` in `priority.py`). When the
game completes level $j$ after $t_j$ tokens on it, with reference cost $r_j$
(`priority_pace_reference.json`: $27\text{k}$ for level 1, $49$–$67\text{k}$ for
levels 2–6, $45.9\text{k}$ fallback),

$$
x_j = \log\frac{t_j}{r_j},\qquad
\ell_1 = x_1,\quad \ell_j = 0.6\,\ell_{j-1} + 0.4\,x_j,\qquad
p = \exp\!\Big(\operatorname{clip}\big(\tfrac{m}{m+3}\,\ell_m,\ \log\tfrac12,\ \log 2\big)\Big),
$$

with $m$ the levels completed so far ($p = 1$ before the first). A slow game
($p > 1$) has its token falloff stretched and its value divided:

$$
C_p = 0.25 \cdot 2^{-(a/115)^2} + 0.75 \cdot 2^{-(s/(62000\,p))^2},\qquad R = v\,e\,C_p / p .
$$

On these runs $p$ has median $0.80$ (10th–90th percentile $0.50$–$1.06$; 18% at
the $0.5$ clip): recent games clear levels faster than the reference, so pace
mostly raises their value and shortens their falloff.

**Thinking** (`ARC3_PRIORITY_THINKING`, `thinking_multiplier` in
`priority.py`). With $t$ the reasoning tokens of the game's most recent call
and reference $2048$,

$$
m = \operatorname{clip}\!\left(\sqrt{2048 / t},\ \tfrac12,\ 2\right),
$$

so $m = 2$ at $t = 512$, $1$ at $2048$, $\tfrac12$ at $8192$ ($m = 1$ before the
first call). On these runs the median $t$ is $709$, and 42% of calls are at
or below $512$ ($m = 2$). Turns that complete a level have median $t = 162$
against $788$ for other turns.

"current" calls `priority_value` from
`ARC3-Inference/inference/agent/scheduling/priority.py` with the Lord Hansolo
settings (`ARC3_PRIORITY_TAIL_LOOKUP=remaining`,
`ARC3_PRIORITY_SCORE_NORMALIZATION=1`), without tail fade.

**Replay.** Start at $\tau_g = 0$. Each round, the live games
($g$ not won and $\tau_g < T_g$) are sorted by $(\mathbb 1[\tau_g = 0],\ R)$,
and the top $K = 14$ (`ARC3_MAX_ACTIVE_STREAMS`) receive $\gamma$ tokens each,
until $\sum_g \tau_g = G\beta$. The result is $\bar S$ at the final $\tau_g$.
$\gamma = 35\text{k}$ is close to the median tokens between context trims in
v07 (where the gate switches games today). Budgets are
$\beta = 100\text{k}$ and $150\text{k}$ tokens per game, where the recorded data
cover $1.82\times$ and $1.22\times$ the budget. At $200\text{k}$ the coverage
is $0.91\times$, every policy converges, and the replay cannot rank them.

**Statistics.** Every policy is replayed on the same 11 runs. For a
comparison $X - Y$, $d_r = \bar S^{(r)}_X - \bar S^{(r)}_Y$, reported as
$\bar d \pm \operatorname{sd}(d)/\sqrt{11}$ with the number of runs where $d_r > 0$.

## Results

Mean $\bar S$, then the paired difference vs the uniform baseline.

| Policy | $\beta = 100\text{k}$, $\gamma = 35\text{k}$ | $\beta = 150\text{k}$, $\gamma = 35\text{k}$ |
|---|---|---|
| uniform (baseline) | 24.6 | 37.3 |
| current | 27.6, +3.01 ± 0.40 (11/11) | 39.2, +1.88 ± 0.42 (9/11) |
| $B = 0$ | 29.5, +4.95 ± 0.86 (10/11) | 39.9, +2.64 ± 0.56 (9/11) |
| $B = 0$, efficiency ablated | 29.4, +4.84 ± 0.91 (11/11) | 40.3, +3.03 ± 0.84 (9/11) |
| $B = 0$ + pace | 29.3, +4.72 ± 0.88 (11/11) | 40.0, +2.73 ± 0.55 (9/11) |
| $B = 0$ + thinking | 29.7, +5.08 ± 0.74 (11/11) | 39.8, +2.52 ± 0.59 (8/11) |

- $B$ is nearly constant ($8$ while three or more levels remain) and larger
  than $A$, so the current ranking is close to $8\,C$ and the level value
  barely matters. Removing it gains $+1.9$ / $+0.8$ points over current.
- Ablating the efficiency term ties $B = 0$ ($-0.1$ / $+0.4$). $e$ falls as
  actions accumulate on a stuck level, which repeats the stall signal in $C$.
- Pace ties $B = 0$ ($-0.2$ / $+0.1$). It rescales every game by its own
  speed, and $C$ already demotes games that are slow on the current level.
- Thinking ties $B = 0$ ($+0.1$ / $-0.1$; with the reference set to the data
  median $709$: $+0.2$ / $0.0$). Low thinking marks the turn that completes a
  level; by the next ranking the game is on a fresh level, where $C = 1$
  already ranks it high.

## Concurrency

Same replay with our choice ($B = 0$, efficiency ablated, $R = v\,C$) and
$K$ concurrent slots. The active fraction $K/G$ is the share of the games in a
wave ($G$) that can generate at once. The control is $K = 14$, used in every table above.
The total token budget $G\beta$ is the same for every $K$, so this measures
only how $K$ changes which games receive the tokens. It does not model the
higher batch throughput of larger $K$, nor its KV- and prefix-cache costs.

Mean $\bar S$, then the paired difference vs $K = 14$ ($\gamma = 35\text{k}$):

| Slots | $K/G$ | $\beta = 100\text{k}$, $\gamma = 35\text{k}$ | $\beta = 150\text{k}$, $\gamma = 35\text{k}$ |
|---|---|---|---|
| $K = 1$ | 0.04 | 30.4, +0.93 ± 0.84 (7/11) | 41.3, +0.98 ± 0.34 (9/11) |
| $K = 2$ | 0.08 | 30.7, +1.30 ± 0.83 (7/11) | 41.4, +1.05 ± 0.29 (9/11) |
| $K = 3$ | 0.12 | 30.5, +1.03 ± 0.74 (7/11) | 41.6, +1.24 ± 0.39 (9/11) |
| $K = 5$ | 0.20 | 30.5, +1.07 ± 0.57 (7/11) | 41.2, +0.84 ± 0.31 (8/11) |
| $K = 7$ | 0.28 | 30.3, +0.91 ± 0.54 (6/11) | 41.3, +0.96 ± 0.33 (9/11) |
| $K = 10$ | 0.40 | 30.1, +0.70 ± 0.53 (5/11) | 40.6, +0.23 ± 0.26 (4/11) |
| $K = 14$ (control) | 0.56 | 29.4 | 40.3 |
| $K = 20$ | 0.80 | 29.0, −0.44 ± 0.91 (4/11) | 39.2, −1.14 ± 0.58 (4/11) |
| $K = 25$ | 1.00 | 25.1, −4.34 ± 0.87 (1/11) | 38.2, −2.15 ± 0.75 (2/11) |

- Higher concurrency lowers the score. With $K$ slots, at most $G - K$ games
  can be held back, so the ranking has less to decide. At $K = G = 25$ every
  live game gets a grant each round, and the result is close to uniform
  ($24.6$ / $37.3$).
- $K = 10$ is level with $K = 14$ ($+0.7$ / $+0.2$, within one SE). From
  $K = 7$ down to $K = 1$ the gain stays at about $+1$ point: once the ranking
  can hold back most games, further restriction adds nothing.
- Fewer slots only pays if the server keeps its total generation rate. With
  fewer concurrent sequences it usually generates fewer tokens per second,
  which costs budget $\beta$ directly; this replay cannot weigh that.

## Token-only decay

Our choice with the action term of $C$ removed (`ARC3_PRIORITY_ACTION_WEIGHT=0`,
token weight $1$) and the token half-life $h$ (`ARC3_PRIORITY_TOKEN_SCALE`) swept:

$$
R = v \cdot 2^{-(s/h)^2}.
$$

Budget $\beta = 150\text{k}$, $\gamma = 35\text{k}$. $K = 3$ of $25$ ($K/G = 0.12$) is
closest to the planned 14 streams on about 110 games. Mean $\bar S$, then the paired
difference vs $v\,C$ (action weight $0.25$, $115$ actions, $62\text{k}$ tokens) at the same $K$:

| Ranking | $K = 3$ | $K = 7$ | $K = 14$ |
|---|---|---|---|
| $v\,C$ (control) | 41.6 | 41.3 | 40.3 |
| $h = 46\text{k}$ | 40.3, −1.30 ± 0.51 (3/11) | 40.4, −0.93 ± 0.41 (3/11) | 40.0, −0.36 ± 0.39 (6/11) |
| $h = 55\text{k}$ | 41.1, −0.47 ± 0.55 (6/11) | 40.4, −0.89 ± 0.38 (3/11) | 39.9, −0.41 ± 0.47 (3/11) |
| $h = 62\text{k}$ | 41.1, −0.42 ± 0.47 (4/11) | 40.7, −0.54 ± 0.28 (2/11) | 40.2, −0.10 ± 0.24 (3/11) |
| $h = 70\text{k}$ | 41.2, −0.33 ± 0.38 (4/11) | 41.2, −0.10 ± 0.22 (3/11) | 40.3, −0.00 ± 0.18 (5/11) |
| $h = 78\text{k}$ | 41.3, −0.31 ± 0.21 (3/11) | 41.0, −0.30 ± 0.17 (2/11) | 40.2, −0.17 ± 0.23 (3/11) |
| $h = 86\text{k}$ | 41.3, −0.30 ± 0.28 (3/11) | 40.9, −0.39 ± 0.22 (2/11) | 40.3, −0.01 ± 0.31 (4/11) |
| $h = 93\text{k}$ | 41.1, −0.51 ± 0.29 (3/11) | 41.0, −0.27 ± 0.17 (2/11) | 40.5, +0.19 ± 0.32 (4/11) |
| $h = 108\text{k}$ | 40.6, −0.99 ± 0.56 (1/11) | 40.9, −0.41 ± 0.37 (3/11) | 40.4, +0.08 ± 0.28 (6/11) |
| $h = 124\text{k}$ | 40.2, −1.38 ± 0.55 (0/11) | 40.7, −0.57 ± 0.38 (1/11) | 40.3, −0.05 ± 0.32 (4/11) |

- The plateau is $h = 70$–$93\text{k}$, within $0.5$ of the control at every $K$
  and within one SE. $h = 70\text{k}$ has the best mean over the three $K$
  ($-0.33$, $-0.10$, $-0.00$).
- The action term is worth at most about $0.3$ points; without it the token
  half-life should be slightly longer than $62\text{k}$, since the action term
  no longer shortens the falloff.
- Both ends lose: $h \le 46\text{k}$ abandons levels too early, and
  $h \ge 108\text{k}$ keeps stalled games ranked high. With no decay at all
  ($R = v$) the loss is $-1.35$ at $K = 3$ and $-0.52$ at $K = 14$.
- The shape does not help either (exploratory, $h$ swept $10$–$248\text{k}$):
  an exponential falloff $2^{-s/h}$ peaks at $h = 31\text{k}$ ($-0.47$ at
  $K = 3$) and $46\text{k}$ ($+0.16$ at $K = 14$); a sharper $2^{-(s/h)^4}$
  peaks at $93\text{k}$ ($-0.59$ / $-0.06$). Neither is available as a setting.
- The production `priority_value` (integer index, floored at 1) reproduces the
  float replay to $0.1$ at $h = 62$, $78$ and $93\text{k}$.

**Action-only.** Action weight $1$ (`ARC3_PRIORITY_ACTION_WEIGHT=1`), action
half-life $h_a$ (`ARC3_PRIORITY_ACTION_SCALE`) swept: $R = v \cdot 2^{-(a/h_a)^2}$.
Same budget, slots and control:

| Ranking | $K = 3$ | $K = 7$ | $K = 14$ |
|---|---|---|---|
| $v\,C$ (control) | 41.6 | 41.3 | 40.3 |
| token-only, $h = 70\text{k}$ | 41.2, −0.33 ± 0.38 (4/11) | 41.2, −0.10 ± 0.22 (3/11) | 40.3, −0.00 ± 0.18 (5/11) |
| $h_a = 29$ | 38.7, −2.89 ± 1.06 (1/11) | 39.1, −2.15 ± 0.81 (1/11) | 39.6, −0.71 ± 0.58 (4/11) |
| $h_a = 58$ | 40.3, −1.27 ± 0.57 (1/11) | 40.6, −0.70 ± 0.38 (2/11) | 40.2, −0.14 ± 0.35 (3/11) |
| $h_a = 72$ | 40.9, −0.69 ± 0.51 (3/11) | 41.2, −0.12 ± 0.24 (3/11) | 40.5, +0.19 ± 0.29 (5/11) |
| $h_a = 86$ | 40.9, −0.71 ± 0.61 (1/11) | 41.3, −0.03 ± 0.35 (3/11) | 40.7, +0.33 ± 0.33 (5/11) |
| $h_a = 93$ | 41.0, −0.60 ± 0.55 (3/11) | 41.3, −0.03 ± 0.44 (2/11) | 40.7, +0.36 ± 0.32 (5/11) |
| $h_a = 115$ | 40.4, −1.21 ± 0.57 (1/11) | 40.5, −0.76 ± 0.41 (1/11) | 40.2, −0.16 ± 0.29 (4/11) |
| $h_a = 173$ | 39.5, −2.08 ± 0.65 (0/11) | 39.8, −1.52 ± 0.49 (0/11) | 39.7, −0.67 ± 0.29 (2/11) |
| $h_a = 345$ | 39.6, −1.97 ± 0.59 (0/11) | 40.0, −1.24 ± 0.39 (0/11) | 39.8, −0.55 ± 0.32 (2/11) |

- The best band is $h_a = 72$–$93$. Against token-only $70\text{k}$ it is
  $-0.3$ to $-0.4$ at $K = 3$ and $+0.2$ to $+0.4$ at $K = 14$, within one SE
  both ways (paired SE $0.2$–$0.35$).
- The band is narrower than the token one: $h_a = 58$ and $115$ already lose
  about $1$ point at $K = 3$. The integer production index moves it by up to
  $0.3$, because an action-only $C$ reaches the floor of $1$ sooner.

**Equal weights.** $C = 0.5 \cdot 2^{-(a/h_a)^2} + 0.5 \cdot 2^{-(s/h)^2}$
(`ARC3_PRIORITY_ACTION_WEIGHT=0.5`). Difference vs $v\,C$ at $K = 3$ / $7$ / $14$:

| | $h = 46\text{k}$ | $h = 62\text{k}$ | $h = 70\text{k}$ | $h = 86\text{k}$ | $h = 108\text{k}$ |
|---|---|---|---|---|---|
| $h_a = 58$ | −0.78 / −0.58 / −0.05 | −0.37 / −0.26 / −0.29 | −1.02 / −0.57 / −0.13 | −0.52 / −0.61 / +0.14 | −0.81 / −0.01 / +0.25 |
| $h_a = 86$ | −0.42 / −0.16 / +0.18 | −0.10 / −0.47 / +0.23 | −0.81 / −0.37 / +0.12 | −0.66 / −0.38 / +0.24 | −0.64 / −0.65 / −0.01 |
| $h_a = 115$ | −0.15 / −0.05 / +0.28 | −0.97 / −0.15 / +0.14 | −0.80 / −0.68 / +0.22 | −0.94 / −0.99 / −0.02 | −1.40 / −0.89 / −0.20 |
| $h_a = 173$ | −1.03 / −0.60 / +0.00 | −1.14 / −0.50 / +0.06 | −1.54 / −0.77 / −0.19 | −1.45 / −0.84 / −0.28 | −1.68 / −1.23 / −0.33 |
| $h_a = 345$ | −1.35 / −0.70 / −0.04 | −1.53 / −0.73 / −0.04 | −1.60 / −1.04 / −0.63 | −1.61 / −0.94 / −0.38 | −1.96 / −1.52 / −0.66 |

- No cell beats the control at $K = 3$. The best, $h_a = 115$, $h = 46\text{k}$
  ($-0.15$ / $-0.05$ / $+0.28$) and $h_a = 86$, $h = 62\text{k}$
  ($-0.10$ / $-0.47$ / $+0.23$), tie token-only $70\text{k}$.
- Neighbouring cells differ by up to $0.8$ (e.g. $h_a = 115$: $-0.15$ at
  $46\text{k}$, $-0.97$ at $62\text{k}$), more than any setting gains. With
  $35\text{k}$ grants a small change in $C$ moves which games get the next
  grant, so a single cell's result is mostly noise.
- Long action half-lives ($h_a \ge 173$) lose at every $h$.

**Summary.** Near its best half-life, every weighting (actions only, tokens only,
$0.25$, $0.5$) is within about $0.5$ of the others, and the replay cannot rank
them.

## Decay grid at $K = 15$, $G = 110$

All three parameters of $C$ searched together:

$$
C = w \cdot 2^{-(a/h_a)^2} + (1 - w) \cdot 2^{-(s/h)^2},\qquad R = v\,C,
$$

with $w \in \{0, 0.25, 0.5, 0.75, 1\}$, $h_a \in \{43, \dots, 345\}$ and
$h \in \{31\text{k}, \dots, 186\text{k}\}$ (10 values each, 320 settings). To
match 15 streams on the roughly 110-game private pool ($K/G = 0.14$), each
replay uses $K = 15$ on a pool of $G = 110$ trajectories drawn without
replacement from the 275 (the same 25 games recur about 4 times per pool).
Every setting is replayed on the same 100 pools; $\beta = 150\text{k}$,
$\gamma = 35\text{k}$. The control is the current $C$ ($w = 0.25$, $h_a = 115$,
$h = 62\text{k}$): 41.94, against 36.66 for uniform. Full tables in
[`data/decay_grid.md`](data/decay_grid.md).

Pools share trajectories, so the spread over pools understates the error.
Uncertainties below are a leave-one-run-out jackknife (30 pools from the
other 10 runs per replicate).

| $w$ | Best setting | $\bar S$ | vs control |
|---|---|---|---|
| 0 (tokens only) | $h = 100\text{k}$ | 41.72 | −0.22 (jackknife −0.18 ± 0.13) |
| 0.25 | $h_a = 144$, $h = 31\text{k}$ | 42.10 | +0.16 |
| 0.5 | $h_a = 86$, $h = 46\text{k}$ | 42.11 | +0.17 (jackknife +0.19 ± 0.34) |
| 0.75 | $h_a = 72$, $h = 55\text{k}$ | 41.96 | +0.01 |
| 1 (actions only) | $h_a = 72$ | 41.77 | −0.18 (jackknife −0.18 ± 0.31) |

Single-term profiles, difference vs control:

| tokens only, $h$ | 31k | 46k | 55k | 62k | 70k | 78k | 86k | 100k | 124k | 186k |
|---|---|---|---|---|---|---|---|---|---|---|
| | −2.53 | −1.48 | −0.83 | −0.54 | −0.35 | −0.25 | −0.32 | −0.22 | −0.48 | −0.92 |

| actions only, $h_a$ | 43 | 58 | 72 | 86 | 100 | 115 | 144 | 173 | 230 | 345 |
|---|---|---|---|---|---|---|---|---|---|---|
| | −1.77 | −0.57 | −0.18 | −0.19 | −0.31 | −0.42 | −0.52 | −0.73 | −0.98 | −1.06 |

- The surface is flat near the control: the best of 320 settings is $+0.17$
  ($+0.19 \pm 0.34$ by jackknife), and the control ranks 57th, within
  $0.17$ of the top. At $w = 0.25$ and $0.5$, every cell with
  $h_a = 86$–$144$ and $h = 31$–$86\text{k}$ is within $\pm 0.2$.
- Selection does not carry over. Picking the best setting on runs 1–6 and
  scoring it on runs 7–11 gives $+0.10$; the reverse gives $+0.64$ in sample
  and $-0.53$ held out.
- A single term costs about $0.2$: tokens only is best at $h = 78$–$100\text{k}$,
  actions only at $h_a = 72$–$86$. Token-only $70\text{k}$ (the earlier choice)
  is $-0.32 \pm 0.25$.
- Short half-lives lose the most ($h \le 46\text{k}$ alone, or $h_a = 43$ at
  any $w$: $-1$ to $-2.5$), and $h \ge 124\text{k}$ or $h_a \ge 173$ loses
  $0.5$–$1$.

## Yield-UCB

`ARC3_SCHEDULER=ucb` (`ARC3-Inference/inference/agent/scheduling/ucb.py`)
ranks by observed yield plus an exploration bonus. With $W(L) = L(L+1)/2$ the
unnormalized weight of $L$ completed levels, $W_{32}$ the weight $32\text{k}$
tokens ago, and $N$ the game's total actions (`ARC3_UCB_DENOMINATOR=actions`)
or total tokens (`tokens`),

$$
\text{rate} = \tfrac12\,\frac{W(L)}{\tau} + \tfrac12\,\frac{W(L) - W_{32}}{\min(\tau, 32\text{k})},\qquad
R = \text{rate} + c\,\sqrt{\frac{\log(1 + t_{\min})}{N}},
$$

where $t_{\min}$ is wall-clock minutes since the run started. The replay has
no clock, so $t_{\min}$ = tokens spent in the run $/\ 52\text{k}$ (v07: 25 games ×
$218\text{k}$ tokens in about 105 minutes). The bonus depends on it only through
$\sqrt{\log(1 + t_{\min})}$. Defaults: $c = 0.0002$ for actions, $0.005$ for tokens.

Mean $\bar S$, then the paired difference vs the uniform baseline ($\gamma = 35\text{k}$):

| Policy | $\beta = 100\text{k}$, $\gamma = 35\text{k}$ | $\beta = 150\text{k}$, $\gamma = 35\text{k}$ |
|---|---|---|
| uniform (baseline) | 24.6 | 37.3 |
| yield only ($c = 0$) | 29.1, +4.52 ± 0.67 (11/11) | 38.3, +0.99 ± 0.61 (6/11) |
| actions, $c = 0.00005$ (¼× default) | 29.3, +4.72 ± 0.52 (11/11) | 39.0, +1.70 ± 0.56 (7/11) |
| actions, $c = 0.0002$ (default) | 28.2, +3.64 ± 0.48 (11/11) | 39.1, +1.80 ± 0.73 (7/11) |
| actions, $c = 0.0008$ (4×) | 27.1, +2.52 ± 0.35 (11/11) | 38.1, +0.77 ± 0.44 (6/11) |
| actions, $c = 0.0032$ (16×) | 24.4, −0.22 ± 0.49 (4/11) | 36.7, −0.57 ± 0.56 (3/11) |
| tokens, $c = 0.00125$ (¼×) | 29.1, +4.52 ± 0.67 (11/11) | 38.6, +1.31 ± 0.49 (7/11) |
| tokens, $c = 0.005$ (default) | 28.9, +4.37 ± 0.73 (11/11) | 40.0, +2.68 ± 0.85 (8/11) |
| tokens, $c = 0.02$ (4×) | 27.8, +3.24 ± 0.59 (10/11) | 38.8, +1.48 ± 0.39 (8/11) |
| tokens, $c = 0.08$ (16×) | 26.9, +2.34 ± 0.55 (10/11) | 38.4, +1.08 ± 0.31 (9/11) |

- Yield alone works at $100\text{k}$ but not at $150\text{k}$: a game that has not
  levelled up recently has rate near $0$ and is starved, even when it is about
  to level up.
- A small bonus fixes that. The token-count default reaches $+4.4$ / $+2.7$,
  level with $B = 0$ ($+5.0$ / $+2.6$). The action-count default, used in the
  1001 and 1003 submissions, reaches $+3.6$ / $+1.8$, level with current.
- Large $c$ lets the bonus dominate and the ranking drifts toward
  fewest-actions-first; at 16× the action default it falls below uniform.

## Decision

Use $B = 0$ with the efficiency term ablated, $R = v\,C$. I think this is
probably as far as allocation can take us.

A lower active fraction $K/G$ is better for allocation. $K = 2$ of $25$
($K/G = 0.08$) scored highest at $\beta = 100\text{k}$ ($K = 3$ at $150\text{k}$),
although every $K/G \le 0.28$ is within noise of it (about $+1$ point over $K = 14$). On the roughly 110-game private pool,
$0.08$ corresponds to about $8.8$ active games, at a cost in throughput. So
14 concurrent streams with 1 wave ($K/G = 14/110 \approx 0.13$) looks like the
best choice; 4 waves gives $14/28 = 0.5$, near the control.

Keep the current $C$ ($w = 0.25$, $h_a = 115$, $h = 62\text{k}$). The
three-parameter grid at $K = 15$, $G = 110$ finds nothing better outside
noise (best $+0.19 \pm 0.34$), and the selected settings do not hold up on
held-out runs. Tokens only is the simpler alternative at about $-0.2$; if
used, set $h = 100\text{k}$ ($-0.18 \pm 0.13$) rather than $70\text{k}$
($-0.32 \pm 0.25$).

Settings: unset `ARC3_PRIORITY_TAIL_LOOKUP`, `ARC3_PRIORITY_TAIL_CAP=0`,
`ARC3_PRIORITY_HUMAN_ACTIONS=1e9` (makes $e \approx 1$), keep the default
falloff (`ARC3_PRIORITY_ACTION_WEIGHT=0.25`, `ARC3_PRIORITY_ACTION_SCALE=115`,
`ARC3_PRIORITY_TOKEN_SCALE=62000`), keep
`ARC3_PRIORITY_SCORE_NORMALIZATION=1` (gives $A \propto v$),
`ARC3_MAX_ACTIVE_STREAMS=14`, and `kaggle.private_waves: 1`.

## Limitations

Two caveats matter most.

**1. No extrapolation past the recording.** A game that reaches its recorded
end $T_g$ without winning has no data beyond it, so it receives no more tokens
and its share goes to the other games. Share of games affected ($\gamma = 35\text{k}$):

| $\beta$ | uniform, $K = 14$ | $v\,C$, $K = 1$ | $v\,C$, $K = 2$ | $v\,C$, $K = 7$ | $v\,C$, $K = 14$ | $v\,C$, $K = 25$ |
|---|---|---|---|---|---|---|
| 100k | 8% | 12% | 12% | 9% | 5% | 8% |
| 150k | 25% | 36% | 35% | 33% | 31% | 24% |

- At $\beta = 100\text{k}$ this is infrequent (5–12% of games), so the results
  there remain mostly valid. At $150\text{k}$ it reaches a quarter to a third of
  the games; read that column as indicative.
- Truncation is within a few points across rankings at the same $K$, so
  comparisons between rankings are less affected than absolute scores.
- It grows as tokens are concentrated (36% at $K = 1$ against 31% at
  $K = 14$). The flat gain below $K = 7$ in the concurrency study is therefore
  partly an artifact: the top games run out of data, and the true effect of
  very low $K$ could be larger or smaller.
- How long each game was recorded was decided by the original run's own
  scheduler.
- The private run's budget is beyond what the replay can rank: at about
  $800$ generated tokens/s for $530$ minutes over $110$ games,
  $\beta \approx 800 \times 530 \times 60 / 110 \approx 231\text{k}$ tokens per
  game, where the recorded data cover less than the budget.

**2. No throughput.** Every policy and every $K$ gets the same token budget
$G\beta$. The replay ignores that fewer concurrent streams generate fewer
tokens per second, that switching a game mid-context costs a re-prefill, and
prefix caching, per-game time caps and waves. This matters most for the
concurrency conclusion.

Smaller simplifications: the trajectory $S_g(\tau)$ is assumed not to depend on
when the tokens are generated; rounds are synchronous, while the real gate
switches one game at a time; actions are interpolated linearly in tokens within
a level.

## Reproduce

```sh
# from the main checkout, where ARC3-Inference/runs/ exists
python docs/allocation/scripts/extract_trajectories.py --runs-root ARC3-Inference/runs
python docs/allocation/scripts/extract_thinking.py --runs-root ARC3-Inference/runs
python docs/allocation/scripts/replay.py   # writes data/results.md
python docs/allocation/scripts/decay_grid.py   # writes data/decay_grid.md (about 5 minutes)
```
