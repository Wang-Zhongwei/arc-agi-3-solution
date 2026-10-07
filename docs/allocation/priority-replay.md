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
