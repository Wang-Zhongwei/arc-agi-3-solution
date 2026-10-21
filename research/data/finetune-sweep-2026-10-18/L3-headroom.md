# L3: Headroom model: what has to change, and by how much, to reach LB 60

Researcher L3, 2026-10-18. Local analysis only. Paths without a prefix are relative to `daniel-franzen-sol/`.

**Data and method (reproducible).**
- **Runs.** 26 full 25-game, 121-min Flash-Next runs of the 10x family: `runs/kaggle/{v4-user-combo-10x/v1,v2-snowiers,v3-snowiers; v4-user-combo-13x/v1; v4-sched-curriculum/v1; v4-uc-11x/v1-snowiers; v4-level-path-10x/v1; v4-mem-layout-10x/v1; v4-fable-facts-10x/v1-snowiers; v4-probe-lap-10x/v1-snowiers; v4-mem-recap-10x/v1,v2; v4-mem-solved-10x/v1,v2-snowiers; v4-expert-stats-10x/v1; v4-mem-recap-keep-10x/v1; v4-prompt-facts-10x/v1-snowiers; v4-pf-solved-10x/v1; v4-pf-keep-10x/v1; v4-pf-solved-cap8k-10x/v1-snowiers; v4-pf-keep-v2-all-10x/v1; v4-mem-unified-10x/v1; v4-mem-unified-core-10x/v1-snowiers; v4-stall-fix-10x/v1-snowiers; v4-mem-repair-10x/v1; v4-mem-repair2-10x/v1-snowiers}/benchmark.json.xz`, plus `v4-q27-dflash2-10x/v1` analysed separately.
  - `v4-mem-recap-10x-r2/v1` was dropped: its data is identical to `v4-mem-recap-10x/v2`.
  - That leaves 650 game-runs, 2,942 cleared levels and 132.3M generated tokens.
- **Scoring.** I recomputed every game score with the official formula (below). It matches the logged `final_score` for **675/675** game-runs. Some runs store the score ×100.
- **Simulator.** `analysis/sim2.py` (calibrated, event-driven harness simulator), run with `--policy franzen --ledger on` and `--speed-scale s`.
  - `--speed-scale s` multiplies decode speed. Level costs are in tokens, so s ≈ the compute multiplier at equal decisions.
  - Settings: 200 reps for 25 games / 121 min, and 30–40 reps for 110 games / 532 min.
  - The scripts are in the session scratchpad (`/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/ce155f4d-514a-4385-ac65-a7e7e5a23e1c/scratchpad/l3/{extract,common,analyze,analyze2}.py`). They use the standard library only.
- **LB pairs.** These come from `runs/submissions.csv`, whose descriptions state each notebook's offline score.

## Q1. How exactly is the competition score computed?

### Takeaway
The score is the mean over games of a level-index-weighted average of per-level scores. Each level scores `min(115, 100·(human/AI actions)²)`. The game total is capped at the weighted share of levels cleared. Unreached levels score 0, and the weights are 1..n.

So a game that clears its first k of n levels at human efficiency scores 100·k(k+1)/(n(n+1)), which is roughly the square of the share of levels cleared. **Depth dominates. Per-level action efficiency is nearly saturated for us.**

### Cited Findings
- **Per-level score.**
  - Formula: `level_score = (human_baseline_actions / ai_actions)^2`, capped at 1.15. [ARC docs, Methodology › Per-Level Scoring / Per-Level Score Cap](https://docs.arcprize.org/methodology)
  - Code: `score = ((baseline_actions / actions_taken) ** 2) * 100; score = min(score, 115.0)`. `ARC3-Inference/.venv/lib/python3.12/site-packages/arc_agi/scorecard.py:170-171` (arc-agi 0.9.9).
- **Game score.**
  - It is the weighted average of per-level scores with weight = the 1-indexed level number. "To unlock a maximum game score of 100%, the AI must complete all levels". Example: completing 4 of 5 levels caps the game at 10/15 = 66.7%. [ARC docs, Per-Game Aggregation](https://docs.arcprize.org/methodology)
  - Code: `scorecard.py:193-206` (`weight = self.level_indices[i]`; `max_score = max_weights / total_weights * 100`; `score = min(score, max_score)`). Uncleared levels append 0 (`scorecard.py:178-183`).
  - Our harness mirror: `tufa-arc-agi-framework/src/taaf/game.py:381-413`.
- **Total score** is the mean of game scores. [ARC docs, Total Score](https://docs.arcprize.org/methodology); `scorecard.py:612-616`.
- **Repeat plays and resets.**
  - A game played several times scores its best play: `EnvironmentScoreList.score = max(run.score …)` (`scorecard.py:241`). The scored play is the one with the most levels (`scorecard.py:567-574`).
  - **A RESET counts as an action**: `inc_reset_count` increments `actions` (`scorecard.py:701-704`).
- **Human baseline.** It is the "upper median human (by fewest actions) per level" among first-time players [ARC docs, Human Baseline](https://docs.arcprize.org/methodology). Exactly 10 members of the public were tested per environment, and an environment needs ≥2 full human solves to be included. [ARC-AGI-3 technical report, arXiv 2603.24621, §4.2 and §5](https://arxiv.org/html/2603.24621)
- **Sets.**
  - Public demo has 25 environments, semi-private 55, fully private 55. Every environment has "at least six levels". [arXiv 2603.24621, §3.4 and §3.6 table](https://arxiv.org/html/2603.24621)
  - The public set is "intentionally easier for both humans and AI". The private set is "significantly more difficult … intentionally out-of-distribution … limited overlap with the mechanics found in the public environments". [arXiv 2603.24621, §3.6](https://arxiv.org/html/2603.24621)
- **Public LB coverage.** The Kaggle public LB "is calculated with approximately 50% of the test data". [Kaggle leaderboard page, search snippet](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)
  - That is ~55 of the 110 hidden games (also stated in `research/archive/reports/leaderboard-35-45.md:16`).
  - I could not load the page text itself, which is JS-rendered.
- **Compute per game is matched offline and on the LB.** The offline run is "25 offline games for 121 min. That is Franzen's 532 min × 25/110" (`kaggle/README.md:78`), and the private run is 110 games in Kaggle's 9-hour run (`README.md:14`). So the LB/offline gap is not a compute-per-game gap.
- **Our efficiency on cleared levels (computed):**
  - The median actions/baseline ratio is **0.64** (p25 0.45, p75 0.92).
  - **82.5%** of cleared levels score ≥100, and the mean level score is 103.7.
  - This matches "Points lost to extra actions on cleared levels: 3.9 (median 0.65×)" in `research/reports/rsi-proposals-2026-10-15.md:25`.

### Inferences
- Squaring only punishes inefficiency, and our cleared levels are mostly under the human baseline. **The level weighting combined with the depth cap is what makes the score roughly quadratic in depth.**
  - Worked example: with 7 levels, clearing 4 gives 36, 5 gives 54, 6 gives 75 and 7 gives 100.
  - So the last levels of each game carry most of the score.
- Training should not target action efficiency on levels the agent already clears. RESET-heavy strategies cost score only on levels that are eventually cleared.

### Gaps
- The exact Kaggle scorer is not public. I assume it matches `arc_agi` 0.9.9; our harness logs match it for all 675 game-runs.
- UNVERIFIED: how Kaggle splits the 110 games between public and private, and whether "110" is the semi-private plus fully-private sets of the technical report.

## Q2. What do tokens per action, stuck-level clears and efficiency buy, offline and on the LB? What are the ceilings?

### Takeaway
On the public 25:
- At equal decision quality, −25% tokens per action is worth about **+7.6 offline** and −50% about **+13.4 offline**. Gains saturate near +15 (about 66 offline), because some levels no observed run ever clears.
- One extra cleared level per non-won game is worth **+10.4 offline**.
- Clearing every reached level, plus the stuck one, at human efficiency gives 63.7. Every game reaching its own best-ever depth gives 75.4.

How this maps to the LB depends on why the LB is 0.55× offline. The two models below bracket it: +4 to +8 LB for −25% tokens, and +7 to +20 LB for −50%.

### Cited Findings (all computed from the 26 runs unless a source is given)
- **Baseline structure.**
  - Mean 51.17 (run sd 3.28, range 45.4–56.9).
  - 30.6% of game-runs won; 4.53 of 7.32 levels cleared per game.
  - The median cleared level costs 22.4k tokens (p90 70k) and 28 actions.
  - The median final stuck level has absorbed 84.8k tokens.
  - Tokens per action are 909 pooled (MR 872, `runs/kaggle/v4-mem-repair-10x/v1`).
  - Levels per M tokens average 22.3 (19.9–26.6), matching "21.3–24.1, no trend" (`research/reports/rsi-proposals-2026-10-15.md:26`).
  - Run score vs total levels has slope **0.53 points per level** (r = 0.88).
- **Where the tokens go.**
  - **80% of actions carry 0 tokens**: the model emits action sequences. Per deciding turn, the mean is 4,411 tokens, the median 2,234 and p90 10,958 (MR). MU is similar.
  - The top 25% of turns (≥ ~5k tokens) hold **69–70%** of all tokens, and the top 10% (≥ ~11k) hold 42–45%.
  - Clipping every turn at 4k / 6k / 8k tokens would remove **48% / 35% / 26%** of tokens (MR; MU 47/36/28).
  - On a narrow definition, 22–29% of tokens go to the game's final uncleared level. Counting every stalled level, "24–36 stalled levels per run, holding 2.4–3.2M of ~5M tokens" (`research/reports/mem-repair2-2026-10-15.md:38-39`).
- **Counterfactual mean scores** (public 25, offline scale):

  | Counterfactual | Offline score |
  |---|---|
  | Actual | 51.17 |
  | Cleared levels at exactly human efficiency | 53.06 |
  | +1 level per non-won game, at the median efficiency observed for that game-level | **61.55** |
  | Same, at human efficiency | 61.77 |
  | All reached levels plus the stuck level at human efficiency | **63.65** |
  | Every game at its best-ever depth across the 26 runs | **75.38** |
  | All levels at observed efficiency | 98.47 |

- **Value of clearing the final stuck level, by depth.**

  | Stuck level | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|
  | Game-score gain | +3.2 | +5.9 | +9.6 | +11.0 | +19.1 | +24.7 | +21.2 |

  69.4% of game-runs are non-won, so the average is +10.4 per game, or about **+1.04 offline per +10 percentage points of currently-stuck final levels cleared**. That count excludes further levels a faster clear would unlock.
- **Stuck levels are mostly "soft", not walls.**
  - Of 451 final stuck levels, only 45 (10%) were never cleared in any of the 26 runs. These walls are bp35 L5, dc22 L5, g50t L6, ka59 L7, lf52 L5, ls20 L6, re86 L8, s5i5 L7, sk48 L3, sp80 L6 and wa30 L5.
  - The soft ones carry +8.9 of the +10.4.
  - 14 of the 25 games have been fully cleared at least once.
- **More compute still clears long-stuck levels.**

  | Already spent on the level | P(it later clears) | Attempts |
  |---|---|---|
  | ≥ 60k tokens | 58% | 686 |
  | ≥ 85k tokens | 47% | 420 |
  | ≥ 125k tokens | 27% | 157 |

  Among those that clear, the median extra cost is about 21–25k tokens. The rates are censored by the deadline, so they are a lower bound.
- **Compute elasticity (sim2, public 25 games, 121 min).**

  | Speed-scale s | 0.5 | 0.8 | 1.0 | 1.15 | 1.333 | 1.5 | 2 | 3 | 4 |
  |---|---|---|---|---|---|---|---|---|---|
  | Mean score | 28.86 | 43.92 | **50.93** | 55.08 | 58.49 | 60.69 | 64.35 | 65.88 | 66.07 |
  | Levels | 74.7 | | 112.0 | | 123.8 | | 131.9 | | 134.3 |

  - Saturation comes from the Kaplan-Meier "unclearable" mass: `tail=never`, and `--tail pooled` gave identical numbers (`analysis/sim2.py:54-56, 439-443`).
  - In the sim, s = 1.333 gave 1.25× generated tokens, because prefill and client gaps do not speed up.
  - The sim keeps prompt growth fixed, but thinking is kept in history (`WRITEUP.md:69`), so shorter thinking would also slow context growth and trims. That is a small conservative bias.
- **110 games / 532 min (sim2).**

  | s | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 | 1.2 | 1.333 | 1.5 | 1.8 | 2.0 | 2.4 | 3.0 | 4.0 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | Mean score | 18.83 | 25.05 | **30.95** | 35.49 | 39.29 | 42.66 | 45.76 | 51.43 | 54.36 | 57.95 | 60.98 | 61.71 | 62.81 | 63.27 | 63.34 |

  At s = 1.0 the 110-copy setting scores about 10% below the 25-game setting (45.8 vs 50.9).
- **Sim validity check against the 27B.**
  - The 27B had the same actions per level, 2.06× the tokens per action and about 0.91× the throughput, so its effective s ≈ 0.44. Real outcome: **14.51 offline, 52 levels**; sim2 predicts 24.3 and 67.6 levels.
  - The sim is therefore optimistic in the low-compute direction. The 27B also hit the 12,288 cap 75 times vs 27, and never cleared L1 of g50t, re86 or sk48 (`arms.md:101`).
  - Upward evidence is thin: `fz-streams14` ran +15% tokens and finished level with control, 46.38 vs 46.25, but that comparison is confounded by a smaller window (`research/archive/reports/score-timeline.md:27,40`).
  - Real runs gain 0.16–0.40 points per minute over minutes 80–121 (`score-timeline.md:12`). That is consistent with the sim's slope of about +0.28 per 1% compute near s = 1.
- **The LB/offline relation.**
  - 10x-family notebooks: LB/offline = **0.548** (sd 0.051, n = 12).
  - The old taaf/Wang harness, offline 9–19: **0.30**. Pairs: taaf-b1 17.74→5.08 mean; c1 19.43→5.04; wang-memory 13.65→4.68; explore 9.16→3.27; 0922 9.86→2.40 (`runs/index.csv`; `runs/submissions.csv`; the taaf runs used a 7,920-s per-game budget).
  - Fit over all 17 notebooks: LB = −4.38 + 0.629·offline (r = 0.98). Within the 10x family: LB = 7.5 + 0.405·offline (r = 0.44).

### Inferences: two models that bracket the hidden set
- **P1, "walls / proportional".** LB gain = 0.548 × offline gain. Results:
  - −25% tokens per action: **+4.1 LB**. −50%: **+7.4**. −75%: +8.3, a ceiling of about 39 LB.
  - +1 level per non-won game: +5.7.
  - The best-ever-depth reliability ceiling: +13.3.
- **P3, "compute-difficulty".** Hidden games behave like public games played at about 0.6× compute; in the 110-game sim, s = 0.6 gives 30.95, our LB mean. Results:
  - −25% tokens per action (s = 0.8): **+8.3 LB**.
  - −33% (0.9): +11.7.
  - −50% (1.2): **+20.5**.
  - −60% (1.5): +27.0.
  - −67% (1.8): +30.0, reaching LB 61.
  - Saturation at about **63.3** LB.
- **Which model fits better.** One factor of about 0.5 compute reproduces both observed ratios:
  - 25-game sim at 0.5 vs 1.0: 28.86/50.93 = 0.57, against 0.55 observed.
  - At 0.15 vs 0.3: 4.28/13.49 = 0.32, against 0.30 observed.

  A constant-ratio (P1) model cannot produce a ratio that rises from 0.30 to 0.55 with skill. So P3 fits the convexity better. But the low-end points come from a different harness, and the technical report says the private set has new mechanics, which are walls in P1's sense. **Treat P1 as the floor and P3 as the ceiling.**
- **P4, "level shift"** (shift every game's depth on the public data):

  | Δ levels per game | −2 | −1 | 0 | +1 | +2 |
  |---|---|---|---|---|---|
  | Mean score | 23.7 | 35.9 | 51.2 | 61.6 | 71.4 |

  - LB 31 corresponds to Δ ≈ −1.4. So each +1 level per game on the hidden set is worth about **+13 to +15 LB**.

### Gaps
- No per-game hidden-set data exists, so P1, P3 and P4 are proxies. They cannot be validated until a lever with a known compute effect is drawn on the LB.
- sim2 over-predicted the 27B by about 10 points at s ≈ 0.44, and it is untested above s = 1.15. Its own README says to trust it for throughput, slot and context mechanics, not for policies that move compute between games (`analysis/README.md`, "Simulator v2").

## Q3. What combination of improvements does LB 60 need, and is 60 consistent with this setup's physics?

### Takeaway
LB 60 is **not reachable by improving today's agent along the axes we have measured** while the hidden/public gap stays as it is:
- Under P1 it needs offline ~102–109, which is impossible.
- Under P3 it needs about a **2.85× compute-equivalent gain**, and P3's ceiling of 63 is barely above 60.
- "Consistently ≥60" with an LB draw sd of about 3.2 needs a mean of about **64**. That is above P3's ceiling, so it needs new capability: clearing levels that no current run clears, in the private set's out-of-distribution mechanics.

LB 60 therefore needs a qualitatively stronger agent. Under P4 it means about +2.3 levels per game on hidden games, which is more depth on hidden games (about 5.4 levels per game) than we reach on the easier public games today (4.53).

### Cited Findings
- Top public LB on 2026-10-17: Tufa Labs 68.91, Yi-Chia Chen 67.49, #3 50.37. Methods undisclosed (project `CLAUDE.md`, "Where we are"). So 60+ is physically achievable on these hidden games with some system on this hardware.
- Our LB draws: 10x family n = 19, mean 29.12, sd 3.32 (22.88–38.22). mem-repair-10x drew 38.22 / 28.73 / 29.54 / 26.90 (`runs/submissions.csv`).
- From Q2: action efficiency headroom ≤ +1.9 offline; walls are 10% of public stuck levels; the reliability ceiling is 75.4 offline.
- Published self-training on shortest self-generated reasoning: "30% reduction in output tokens on average, across five model families on GSM8K and MATH, while maintaining average accuracy" (Munkhbat et al., 2025-02). [arXiv 2502.20122](https://arxiv.org/abs/2502.20122) This is math, not agents, so treat it as inspiration only.
- Our toy RSI loop doubled its score mainly by answering in fewer tokens (`research/reports/finetune-plan-2026-10-17.md` §9a).
- The 27B vs Flash-Next comparison (same actions per level, 2× tokens per action) shows that tokens per action varies 2× across models in this family (`arms.md:101`).
- Throughput is near its ceiling: decode peaks at 12–13 running requests (~926 tok/s), and MTP accept length is 2.68 (`research/reports/rsi-proposals-2026-10-15.md:28,32`).

### Lever table (LB effects: P1 floor / P3 ceiling)

| Lever | Change needed | LB effect P1 / P3 | Plausibility in 15 days (reasoning) |
|---|---|---|---|
| Tokens per action at equal decisions (R3 shortest-correct) | −25% | +4 / +8 | **Medium.** ~30% is the typical published SFT result. 70% of our tokens sit in turns ≥5k tokens, so the tail is where cuts must come from. Risk: thinking is the in-history memory (`WRITEUP.md:69`), and a hard cap hurt (`arms.md:102`, cap8k −5.7, confounded) |
| Same | −50% | +7 / +20 | **Low to medium.** It needs every turn effectively ≤4k tokens with the same actions (clip@4k = −48%) |
| Same | −67% | +8 / +30 (LB about 61) | **Low.** No precedent at equal quality |
| Throughput (MTP head retrained to the fine-tuned model, cleaner serving) | +10–15% | +2 / +4 | **Low upside, real downside.** Re-quantization or MTP drift could cut 10% and cost the same amount. Gate on ±5% (plan §3 stage 4) |
| Stuck-level exploration: fewer actions to find the goal, about 1.5× compute-equivalent on hard levels (R4) | 1.5× | +5 / +12 | **Unknown.** In 96% of stuck decisions an untried action reached a new board (`archive/reports/graph-exploration-bandits-memory.md:221`), so the agent explores too little. No evidence yet that training transfers to new mechanics |
| Stuck clear rate, direct | +10 pp of final stuck levels cleared | +0.6 / about +1.3 (P4) | Small per unit, so it needs tens of points of clear rate |
| Reliability (reach best-ever depth) | Public ceiling +24 offline | ≤ +13 (P1) | **Unknown on hidden**, since the public games are tuned and repeated 26×. Win reliability: about 6 wins per run, and identical code gave 9, 4 and 5 (`archive/reports/win-reliability.md:23`) |
| New-mechanic capability (walls) | Clear levels no current run clears | Required beyond about 63 | **Low by 11-02.** This is what training on diverse external games targets; the private set is out-of-distribution by design |
| Cleared-level action efficiency | n/a | ≤ +1 | **Don't target.** Median 0.64× human |

### Inferences
- **What has to stack for LB 60 under P3.** Compute-equivalent factors multiply. An example is −50% tokens per action (2.0×) × 1.4× exploration efficiency on stuck levels ≈ 2.8×, which gives ≈ 60.
  - Under P1 the same stack gives ~40.
  - Each factor must hold on new mechanics, not only on the public 25.
- **Even a perfect efficiency stack sits at P3's ceiling.** So the plan must also contain a capability component: diverse external training games, and per-level or new-board rewards. Otherwise LB 60 can at best be a lucky draw.
- **A cheaper sanity target.** LB 40–45 needs about 1.5–1.7× compute-equivalent under P3 (−33% to −40% tokens per action at equal decisions). Under P1 it is not reachable from efficiency alone.

### Gaps
- What Tufa Labs and Yi-Chia Chen do is unknown. Yi-Chia Chen publishes `opd-32b-v33-s200-gptq-w4a16` plus a DFlash drafter (`research/archive/reports/leaderboard-35-45.md:13`), which suggests a distilled 32B, UNVERIFIED.
- The share of hidden-set loss that is walls vs compute is unmeasured.

## Q4. Noise and evaluation: sample sizes, cheapest trustworthy screens, offline→LB validity

### Takeaway
Measured noise:
- Offline run sd is about **3.2**. The 25-game paired per-game test is conservative, because games share compute.
- LB draw sd is about **3.0–3.3**.

At 80% power and one-sided α = 0.05, two independent arms need:
- +5 LB: about **5 draws per arm**, or 2–3 draws against the established 19-draw baseline.
- +10 LB: 1–2.
- +20 LB: 1.

Offline predicts LB only for large differences: r = 0.44 within the 10x family, r = 0.98 across harness generations. So screens should measure the mechanism (tokens per turn at matched decisions, branch clear rates), then confirm with full runs.

### Cited Findings
- **Same-arm offline repeats.** user-combo-10x 56.94 / 52.70 / 49.09; mem-solved 53.20 / 53.77; mem-recap 45.39 / 49.55. Pooled within-arm sd = **3.15** (df 4).
  - Across the 26 configs, sd = 3.28.
  - The pf family's pooled sd is 2.39 (`arms.md:288`).
- **Per-game noise.** Per-game paired-difference sd between runs has median **27.2** (IQR 24.0–31.4), and the mean per-game sd across runs is 16.9.
  - The iid paired formula then implies sd 5.4 for a one-pair 25-game difference, vs √2·3.15 = 4.5 empirically. Games compete for shared compute, so `make significance`'s per-game test overstates noise.
  - Per-game swings of 50–80 points between identical runs: `runs/README.md`, "Scoring a run".
- **LB within-notebook sd.** 3.04, pooled over df 10 from repeat draws in `runs/submissions.csv`. All 10x-family draws: sd 3.32.
  - The prior plan assumed sd ~5 and so needed ~16 draws for +5 (`research/reports/finetune-plan-2026-10-17.md:190-191`). At sd 5 the requirements are 12 / 3 / 1 draws per arm for +5 / +10 / +20.
- **Offline runs needed per arm** (sd 3.2):

  | Offline gain to detect | +3 | +5 | +10 | +20 |
  |---|---|---|---|---|
  | Runs per arm | ~13 | ~5 | ~1.3 (1 paired run each gives z ≈ 2.2) | 1 |

  Under P1, LB +5 / +10 / +20 corresponds to offline +9 / +18 / +36.
- **Panels.** The minute-25 score predicts the final ranking at only r = 0.21 (Spearman 0.18) across 11 full runs, rising to 0.63 at 60 min and 0.90 at 110 min (`research/archive/reports/score-timeline.md:13`). A 20-min panel is fine for throughput and tokens, and not for score.
- **Frozen-decision lab cost.** The state-lab Kaggle round ran 400 samples in 14.9 min (638k tokens, 27 GPU-min) (`arms.md:313`).
- **Checkpoint-branch bench** (not built): branch at levels that clear about 50% of the time (tn36 L2, cn04 L2, m0r0 L6, re86 L6, sc25 L6, vc33 L7), 25 min per branch (`arms.md:575`).
- **Offline→LB.** The best offline config, mem-unified 56.09, has LB mean 30.91. mem-unified-core, at 51.15 offline, drew 34.04 (`runs/submissions.csv`). Below about ±5 offline, offline rank says nothing about LB rank.

### Inferences
- **Screen 1: tokens.**
  - Mechanism: frozen-decision resampling of the fine-tuned checkpoint vs the base on about 300–500 logged states, weighted toward long-turn states and stuck states.
  - Metrics: tokens per turn; agreement with the logged action, or with an engine-verified "reaches a new board / on the hindsight route" action.
  - Per-turn tokens have CV ≈ 1.2, so detecting −25% needs roughly 2·(2.49·1.2/0.25)² ≈ 290 samples per arm: one state-lab round of about 0.5 GPU-h.
  - Gate: tokens −25% at action agreement no worse than base-vs-base resample agreement.
- **Screen 2: clear rate.**
  - Mechanism: the checkpoint-branch bench at ~50% levels.
  - Detecting p 0.5→0.7 needs about 73 branches per arm (binomial, 80% power), roughly 5 GPU-h per arm at 12 concurrent streams.
  - By comparison, a ~+2 offline effect would need ~45 full runs per arm.
- **Screen 3: the score.**
  - 2 full 121-min runs on held-out games plus external validation games, at matched wall-clock. The 10-game held-out public subset is noisier: about 4–5 paired runs to detect +10, from per-game sd 27.
  - Then LB draws: 2–3 draws suffice against the 19-draw baseline for +5. A +20 jump shows in one draw.
- **External games as a proxy.** They are untested as a predictor of LB.
  - Test first: run the current leader once on the validation external games. If it scores near LB (~31) rather than near public (~51), external games are the better proxy for the out-of-distribution private set.

### Gaps
- Run-to-run sd of a fine-tuned checkpoint may differ: for example, shorter, more decisive thinking may raise variance.
- No LB data exists for any compute-shifted variant, so P1 and P3 cannot yet be told apart on the LB.

## Implications for our plan
1. **Do:** make tokens per turn at equal decisions the primary, measurable training target (R3), aimed at the long tail. 70% of tokens are in turns ≥5k tokens; clip@4k would remove 48%.
   - Expected payoff: +4 to +8 LB at −25%, and +7 to +20 LB at −50%.
   - It is the only lever whose effect can be screened cheaply on our W4A16 setup before a full run.
2. **Do:** train stuck-level breakthroughs (R4) on non-public, diverse games. Even a perfect efficiency stack saturates at about 63 LB under the optimistic model, so LB 60 consistently (mean ≈ 64) needs depth on new mechanics.
3. **Don't:** target per-level action efficiency on cleared levels (≤ +1 LB). Don't accept any checkpoint that loses more than about 5% throughput, since −10% compute costs about 3–4 LB under P3.
4. **Test first:**
   - Before any training, run the leader on the external validation games to see which proxy they track.
   - After stage 1, gate on the state-lab token and agreement screen, then the branch bench, then 2 full runs, then LB draws (2–3 suffice for +5 against the 19-draw baseline).
5. **Set expectations:**
   - Realistic 15-day outcomes are LB ~35–45 if the token lever works (P3-leaning) and ~33–38 if hidden losses are mostly walls (P1).
   - LB 60 requires about a 2.8× compute-equivalent gain plus new capability, which is far beyond what any lever we have measured has delivered.
   - The first LB draw of a token-cut checkpoint discriminates P1 from P3: with LB sd ≈ 3.2, a +4 vs +8 difference is visible in 2–3 draws. That result should set how much of the remaining budget goes to efficiency vs capability.

## Fact-check (independent verifier, 2026-10-18)

Method: re-read the primary sources (arc_agi 0.9.9 `scorecard.py`, docs.arcprize.org/methodology raw markdown, arXiv 2603.24621 HTML, arXiv 2502.20122 abstract), re-extracted all 27 `benchmark.json.xz` files with a separate stdlib script, recomputed scores, ratios and token shares, and re-ran `analysis/sim2.py` at every quoted setting. Scripts are in the session scratchpad: `fc/fc_l3_{extract,analyze,more}.py`.

| # | Claim | Verdict | Correction / note |
|---|---|---|---|
| 1 | Per-level `min(115, 100·(b/a)²)`; level-index-weighted game mean capped at 100·(scored weight)/(total weight); total = mean over games | **Confirmed** | `scorecard.py:170-171`, `:197`, `:205-206`, `:614-616`; `level_index = level_idx + 1` (`:487`). Docs give the same formula, the 1.15 cap, the 10/15 = 66.7% example and the "average of all game scores" total. |
| 2 | Cleared levels: median 0.64× human actions, 82.5% score ≥100; human efficiency raises the mean only 51.17 → 53.06 | **Confirmed** | Recomputed: median 0.644 (p25 0.446, p75 0.918), 82.53% ≥100, mean level score 103.74, 51.172 → 53.057; score formula matches the logged `final_score` for 675/675 game-runs. Note on the corroboration: `rsi-proposals-2026-10-15.md:25`'s "3.9 points" is the *gross* shortfall below 100 on mem-repair alone (it ignores levels above 100 offsetting levels below it within a capped game). The net human-efficiency gain on that run is 1.90, the same as the notes' net figure. Only the median (0.65 vs 0.64) corroborates directly. |
| 3 | LB/offline 0.548 (sd 0.051) over 12 10x-family notebooks vs ~0.30 for taaf/Wang at 9–19 offline | **Confirmed** | Reproduced exactly: 0.5483, sd 0.0514, n = 12. The taaf ratios are 0.29 / 0.26 / 0.34 / 0.36 / 0.24 (mean 0.298). The 17-notebook fit is LB = −4.38 + 0.629·offline (r 0.978). Two caveats. (a) The 12 "10x-family" notebooks include the Franzen-era `fz-ledger-on-trim` and `fz-ledger-on-trim-r2`; without them the ratio is 0.559 (sd 0.041, n = 10). (b) The taaf rows in `submissions.csv` have an empty description, so their offline scores come only from `runs/index.csv` (the score column), matched by notebook name. |
| 4 | Public set "intentionally easier for both humans and AI"; private set "significantly more difficult… intentionally out-of-distribution… limited overlap"; 25 / 55 / 55 sets; ≥6 levels each | **Confirmed** | All quotes are verbatim in §3.6, the table gives 25 / 55 / 55, and §3.4 says "at least six levels per environment". All 25 public games have 6–10 levels. Related to the Q1 gap: §3.6 says the *fully private* set (55) is "used for the competition" and the semi-private set is for testing frontier models behind an API. So the report does not itself support "110 hidden games = semi-private + fully private". |
| 5 | sim2 equal-decision numbers (25 games: 50.93 / 58.49 / 64.35 / 66.07; 110 games: 30.95 / 39.29 / 51.43 / 60.98 / 63.34) | **Confirmed** | Every 25-game value reproduces exactly at 200 reps, and s = 0.5 gives 28.86. For 110 games at `--reps 30`, s = 0.6 / 0.8 / 1.2 / 1.8 reproduce exactly, but s = 4.0 gives 63.37 (63.34 needs `--reps 40`) and s = 1.0 gives 46.07 (45.76 at 40 reps). The differences are within one se. `--tail pooled` gives identical numbers, and s = 1.333 gives 1.25× tokens (5.62M vs 4.48M). |
| 6 | sim2 over-predicts the 27B: 14.51 offline / 52 levels vs sim2 at s = 0.44 → 24.3 / 67.6 | **Confirmed** | Recomputed: 27B 14.512 and 52 levels; tokens per action 1,793.2 vs MR 871.9 (2.06×). Throughput 628 vs ~690 tok/s ("91%", `arms.md:101`), and actions per level 49.6 vs 49.8 (same line). sim2 at s = 0.44 gives 24.30 and 67.6 levels. The s = 0.44 mapping treats the 27B as pure decode slow-down, but it also differs in KV and prefill (32 vs ~12 KiB/token), so this is not a clean test of the sim. |
| 7 | LB draw noise: pooled within-notebook sd 3.04 (df 10); 19 10x-family draws mean 29.12, sd 3.32; offline repeats pooled sd 3.15 | **Partially confirmed** | 29.12 / 3.32 (n = 19) and the offline 3.15 (df 4) reproduce. But **3.04 (df 10) pools the taaf repeat draws** (taaf-b1 3.95 / 4.21 / 7.07 and wang-memory 4.08 / 5.27). These sit at LB 4–7 with a within-sd of about 1.5 and pull the estimate down. **The 10x-family repeats alone** (mem-repair ×4, user-combo-10x ×3, mem-unified ×2, fz-ledger-on-trim-r2 ×2) **give 3.50 (df 7)**. At sd 3.5, +5 LB needs about 6 draws per arm (not 5), and "consistent ≥60" needs a mean of about 64.5. The 19 draws also include the 3 fz-ledger draws, and their sd mixes in between-notebook variance. |
| 8 | ~70% of tokens are in turns ≥~5k; clipping at 4k removes ~48% (MR), 47% (MU) | **Confirmed** | MR: the top-quartile turns (≥5,559) hold 69.0% of tokens, and turns ≥5,000 hold 72.6%. MU: the top quartile (≥4,934) holds 69.8%, and turns ≥5,000 hold 69.2%. Clipping at 4k / 6k / 8k removes 47.7 / 35.1 / 26.3% (MR) and 47.4 / 35.8 / 27.6% (MU). 80.2% of MR actions carry 0 tokens. |

**Other errors found while skimming:**
- **Q4 Screen 1, "Per-turn tokens have CV ≈ 1.2".** Across deciding turns in the run logs, the CV is **1.54 (MR) and 1.45 (MU)**. With the same unpaired formula, −25% then needs about **420–470 samples per arm, not ~290**, which is about 1 GPU-h for both arms at the round-2 rate (400 samples in 27 GPU-min), not 0.5. A paired design on the same states would need fewer.
- **Q4 Screen 2, "a ~+2 offline effect would need ~45 full runs per arm".** That figure uses the iid per-game paired formula (per-game sd 27.2, `analyze.py:171-172`), which the notes themselves say overstates noise. The run-level sd of 3.2 used in the table just above it, with the same one-sided 5% / 80% power, gives **about 32**.
- **Q4, "The pf family's pooled sd is 2.39 (`arms.md:288`)".** 2.39 is the pooled run-to-run sd over 6 replicate groups, defined at `arms.md:102` and only cited in the pf entry. The six pf-family runs are different configs, not repeats (53.20–55.79, sd ≈ 1.05).
- **Spot-checked and correct:** 650 game-runs, 2,942 levels, 132.28M tokens; 30.6% won; 4.53 of 7.32 levels; median cleared level 22.4k tokens and 28 actions; median final stuck level 84.8k; 22.3 levels per M tokens (19.9–26.6); slope 0.53 per level (r 0.88); 451 stuck levels, of which 45 are walls (the same 11 game-levels); 14 games ever fully cleared; P(later clear) 57.7 / 46.7 / 27.4% at n 686 / 420 / 157; P4 shift table; Munkhbat et al. "30% reduction… five model families on GSM8K and MATH"; Kaggle "approximately 50% of the test data"; and all local line citations checked (`kaggle/README.md:78`, `WRITEUP.md:69`, `arms.md:101-102, 313, 575`, `score-timeline.md:12-13, 27, 40`, `leaderboard-35-45.md:13, 16`, `mem-repair2:38-39`, `finetune-plan:190-191, §9a`, `win-reliability.md:23`, `graph-exploration…:221`, `sim2.py:54-56, 439-443`, `analysis/README.md:72-73`, `runs/README.md:48`).
