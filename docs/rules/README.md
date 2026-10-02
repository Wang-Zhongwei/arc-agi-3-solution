# t-test guidance

For a `n` game x `k` pass runs, there are 2 ways of doing the formal [t-test](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm) 

1. Across-game t-test
2. Across-pass t-test

To illustrate the difference, let's see a toy example: 3 games × 2 passes

| A (baseline) | Pass 1 | Pass 2 |
| --- | ---: | ---: |
| Game 1 | 20 | 20 |
| Game 2 | 20 | 20 |
| Game 3 | 20 | 20 |

| B (candidate) | Pass 1 | Pass 2 |
| --- | ---: | ---: |
| Game 1 | 28 | 32 |
| Game 2 | 16 | 20 |
| Game 3 | 22 | 26 |

| D = B − A | Pass 1 | Pass 2 | Game mean |
| --- | ---: | ---: | ---: |
| Game 1 | +8 | +12 | +10 |
| Game 2 | −4 | 0 | −2 |
| Game 3 | +2 | +6 | +4 |
| Pass mean | +2 | +6 | +4 |

**1. Across games:** $d = (10, -2, 4)$, $N = 3$.

$$
\bar d = \frac{10 - 2 + 4}{3} = 4, \qquad
s_d = \sqrt{\frac{(10-4)^2 + (-2-4)^2 + (4-4)^2}{3-1}} = \sqrt{36} = 6
$$

$$
t = \frac{\bar d}{s_d/\sqrt{N}} = \frac{4}{6/\sqrt{3}} \approx 1.155, \qquad \mathrm{df} = 2, \qquad p_{\text{one-sided}} \approx 0.184
$$

**2. Across passes:** $d = (2, 6)$, $k = 2$.

$$
\bar d = \frac{2 + 6}{2} = 4, \qquad
s_d = \sqrt{\frac{(2-4)^2 + (6-4)^2}{2-1}} = \sqrt{8} \approx 2.828
$$

$$
t = \frac{\bar d}{s_d/\sqrt{k}} = \frac{4}{\sqrt{8}/\sqrt{2}} = 2, \qquad \mathrm{df} = 1, \qquad p_{\text{one-sided}} \approx 0.148
$$

I think **across-games** is a more of a fit to this competition. In other words, if an approach has improved performance on most games,
 we tend to believe this is the better approach. 

