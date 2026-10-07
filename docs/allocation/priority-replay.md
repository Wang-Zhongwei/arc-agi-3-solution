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

Settings: unset `ARC3_PRIORITY_TAIL_LOOKUP`, `ARC3_PRIORITY_TAIL_CAP=0`,
`ARC3_PRIORITY_HUMAN_ACTIONS=1e9` (makes $e \approx 1$), keep
`ARC3_PRIORITY_SCORE_NORMALIZATION=1` (gives $A \propto v$),
`ARC3_MAX_ACTIVE_STREAMS=14`, and `kaggle.private_waves: 1`.

## Limitations

The replay ignores wall-clock throughput, prefill, prefix caching, per-game
time caps and waves. Rounds are synchronous, while the real gate switches one
game at a time. Actions are interpolated linearly in tokens within a level.
A game cannot receive tokens beyond $T_g$, which understates rankings that
concentrate tokens. Real runs spend about $220\text{k}$ tokens per game, beyond
what the replay can rank.

## Reproduce

```sh
# from the main checkout, where ARC3-Inference/runs/ exists
python docs/allocation/scripts/extract_trajectories.py --runs-root ARC3-Inference/runs
python docs/allocation/scripts/extract_thinking.py --runs-root ARC3-Inference/runs
python docs/allocation/scripts/replay.py   # writes data/results.md
```
