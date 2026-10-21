# Graph-based exploration, bandits and the memory card

Date: 2026-10-10. Workflow wf_504a1893 (research only, no harness code). Model: Qwen3.8-Flash-Next W4A16 + MTP.

**Nothing in this report has been run on Kaggle.** Every number comes from two places:

- the logs of four Kaggle runs that already exist: v4-user-combo-10x v1/v2/v3 (56.94 / 52.70 / 49.09) and v4-control v1 (50.62);
- CPU-only replays and simulations of the 25 public games on the laptop (single process, `nice -n 15`).

The arms this report recommends running first (memory-v2 and level-path) are built but have no scored Kaggle run. memory-v2-both has one 20-min smoke panel (5.95, not comparable to anything) and one full run that was stopped at about 50 min. level-path has never run.

Scratch data and scripts are in the session scratchpad (`.../72bf2785-.../scratchpad/gbm/`, full path in section 6). They are not in the repo and may be cleared.

## 0. The questions

1. "what if we try doing Graph-Based Exploration? the key is balancing the agent's exploration vs exploitation."
2. "also what about things like Gradient Bandit Algorithm?"
3. "rather than having these fields. what could be better to have?" The fields are today's memory card: **world model** (what the board is and what the actions do), **hypotheses** (confirmed and refuted; a refuted one is meant to stop the model retrying a dead idea) and **level notes** (one compressed line per level, distilled from the model's `<think>` trace, so a level change does not erase everything learned).
4. Added later: "to control the explore exploit, there has to be a way to calculate optimal budget. ... utility(a) = P(solve gain) + lambda*IG(a) - mu*action cost - rho*death risk. Most agents probably overexplore because 'learning the mechanic' feels productive even when the move isn't score-efficient. ... hopefully by doing this the agent can autonomously handle when to explore or exploit."

## 1. Short answers

### 1.1 Graph-based exploration

- **As the policy that picks actions: no.** A re-implementation of the Rudakov et al. explorer (arXiv 2512.24156, no LLM) scored **0.11** on the 25 public games at 5,000 actions per game, against about 50 for our LLM runs. It cleared 20 of 183 levels, and only 3 of those within 5x the human action count. The level score squares the human/agent action ratio, so brute-force clears are worth almost nothing.
- **As memory: yes, and it is already built.** memory-v2 (PR #64: seen-state, action-effects, solved levels) and level-path (PR #66: board links and BFS routes) record the per-level state-action graph and show facts from it. Neither has a scored run.
- **On the balance itself, the data says the model already explores well.** 91% of its actions try a new (state, action) pair, and on cleared levels its one-step choices beat every harness-computed score (on open levels the utility finds slightly more new boards, at twice the no-op rate). On levels it never clears it does not run out of ideas; it runs out of time (about 37 s per action, and every visited state still had untried actions).

### 1.2 Gradient bandit

It does not fit any decision left in the harness.

- **Game scheduler.** About 2–2.6 pulls per game per run, rewards that rise and then rot, and an Exp3 regret bound larger than the number of pulls. Bandit-type schedulers already lost: UCB −1.73 to −2.42 offline, and a real UCB run scored 43.84 vs 46.25.
- **Actions inside a level.** Transitions are about 97–98% deterministic and depend on the board, so the right object is a (state, action) graph, not a mean reward per arm. The level-clear reward arrives about once per 30 actions, and a dense novelty reward barely separates action types (0.79–0.90).
- **Choosing among model-proposed trajectories.** The model never proposed any (multi-trajectory panel: 0 of 10 games; PR #142 closed).
- The one place a bandit's assumptions hold is choosing which Kaggle arm to run or repeat next. The skeptic found too few such decisions for a tool to be worth building; `make significance` plus the user's repeated-runs rule already covers it.

### 1.3 Better fields than world model / hypotheses / level notes

Replace model-written prose with facts the harness computes from the state graph:

| Today's field | Replace with | Where it already exists |
|---|---|---|
| World model (model-written) | **Effects table**: per key, and per clicked colour, how often it moved something, changed the board, did nothing, killed or cleared, this level and this game | memory-v2 `ARC3_MEM_EFFECTS` (`<actions>`) |
| Hypotheses, confirmed / refuted (model-written) | **Seen-before line** tied to one board (what was tried from here and what happened), plus **death lines** (which action from which board killed) and a **route** back | memory-v2 `ARC3_MEM_SEEN`, `<hazard>`; level-path `level_path`, `level_path_before_death` |
| Level notes (clipped from the model's reasoning) | **Solved-level record**: loop-free winning path in run-length form, actions spent, deaths, the clearing turn's last whole sentence; later a computed **win signature** | memory-v2 `ARC3_MEM_SOLVED`; win signature is systems-gap rank 5 (not built) |

Why: today's fields are almost empty. The model writes a memory tag in 1.1% of replies, the refuted list was empty in all 79 cards of v4-user-combo-10x v1, and 82% of card lines are level notes clipped mid-sentence from the model's reasoning. Harness-computed facts do get read ("Seen before" was quoted in the next reasoning in 5 of 8 cases).

A model-written goal line was proposed as the one field the graph cannot supply. The skeptics dropped it: a tag nobody writes stays empty.

### 1.4 The utility and the action budget

- **The model does not over-explore at the action level.** On cleared levels 81% of decisions lie on the hindsight-shortest route. On never-cleared levels it acts too little, not too much.
- **λ (value of information) cannot be identified from logs.** Fitting the utility to the winning routes gives λ ≈ −0.15 game points per nat, because route actions are the predictable ones. A positive λ needs an online A/B test.
- **μ (cost of one more action) comes from the scorer, not from fitting.** It is 0 while the level is still under the 115 cap (75% of cleared levels finish there) and 0.03–0.14 game points per exploratory action otherwise (0.001–0.006 run points). One extra action on every cleared level costs 0.53–0.62 run points per run.
- **ρ (death weight) = 1 by accounting.** A death costs the re-walk to the board before it (mean 15.6 actions). The fit gives ρ ≈ 0 because deaths are rare (57 in 4 runs).
- **Per-level budget: no stopping point in the data for the model's own probes.** Gain/cost is 1.5–47 at every depth once a level has had 10 actions, up to 200. In the first ~10 actions the myopic formula says "don't explore" (gain/cost 0–0.9), but those moves make later clears possible, so it should not gate them. Blind harness probes never pay (gain/cost 0.006–0.04).
- **So the utility should not choose, veto or cap the model's actions.** The switch the user wants is better served by giving stuck levels more actions per hour.
- **Where the hunch holds:** no-op probes are 6–10% of exploratory actions in mixed keyboard+click games. Removing them with hindsight is worth at most +0.4 to +0.8 run points, and the harness cannot predict which untried action will be a no-op (its p_new is badly calibrated).

## 2. The data

### 2.1 Method

- **Runs.** v4-user-combo-10x v1 (main graph analysis and card audit), v2, v3, and v4-control v1 (calibration and bandit statistics). The score recomputed from per-level actions, min(115, 100·(b/n)²) weighted by level index, matches `evaluation.json` exactly.
- **State.** The board hash with each game's step/timer bar masked. The bar was found automatically in 25/25 games and checked by hand on tn36, sk48, sb26 and ls20. Masking cuts distinct states from 5,383 to 4,702. Some analyses used memory-v2's 4-cell edge crop instead; tables say which.
- **Click actions** are keyed by the clicked one-colour connected object.
- **Calibration replay.** Every recorded action of the 4 runs was replayed in the local engine (21,901 decisions, 24 games). At each decision the game was deep-copied and every candidate action stepped once (546k candidates), which gives the true one-step outcome of each action not taken. Replay matched the logged boards exactly on the 24 games. tn36 is excluded: deep-copying it leaks state back into the main game.

### 2.2 The state graph of v4-user-combo-10x v1

25 games, 138 levels (123 cleared, 15 never cleared), 5,692 actions.

| | Cleared levels (123) | Never-cleared levels (15) |
|---|---|---|
| Actions | 5,031 | 661 |
| Transition to a new state | 82.7% | 85.6% |
| Return to an earlier state | 12.0% | 8.5% |
| No-op (only the timer changed) | 5.3% | 6.0% |
| Repeat of an already-taken (state, action) pair | 9.2% | 4.9% |
| Repeat of a pair known to be a no-op / death | 48 / 2 | 2 / 0 |
| Distinct actions tried per visited state | 1.10 | 1.09 |
| Visited states with ≥ 2 actions tried | 7.7% | 9.8% |
| Keyboard (state, action) pairs never tried at visited states | 77.0% | 78.0% |
| Game time / generated tokens | 34.4 h / 3.90M | 6.8 h / 1.21M |

- **Near-deterministic.** With the bar masked, 8 of 352 repeated (state, action) pairs had more than one outcome. With the 4-cell crop, 97.9% (v1) and 96.5% (control) of repeats gave the same next board.
- **The agent walks a chain.** 92% of the states it acted from had only one distinct action tried. Distinct states per level: median 24, mean 34, max 161. Clicks are 31% of actions.
- **It does not miss wins.** It takes the winning action the first time it reaches the winning pre-state, except sp80 L1 and L2. In the calibration replay, 417 of 443 levels took a one-step win the first time one was available; 25 took it a median of 3 actions late; 1 never did (m0r0 L6, 1 of 53 click candidates).

Repeat and loop rates from the different analyses (they use different hashes, so they differ a little):

| Measure | Value | Run, hash | Source |
|---|---|---|---|
| Exact (state, action) repeats | 9.0% | r2, 4-cell crop | `memory-v2-measurement.md` |
| | 8.2% / 7.3% | ucx v1 / control v1, 4-cell crop | gradient-bandit analysis |
| Repeats of a pair already known to be a no-op or death | 52 of 5,692 (0.9%) | ucx v1, bar mask | graph analysis |
| | 37 (0.7%), 28 failed again | ucx v1, 4-cell crop | memory audit |
| | 183 over 4 runs, 161 failed again (88%) | 4 runs, bar mask | calibration |
| No-death loops | 3.0% / 4.1% of cleared-level actions; 3.9% / 9.8% of open-level actions | ucx v1 / control v1 | gradient-bandit analysis |
| Actions on net-zero loops | 27% | r2 | `memory-v2-measurement.md` |
| Moves landing on a board already seen this level | 14.2% | r2 | `memory-v2-measurement.md` |

### 2.3 What exploiting the graph could save

- **Hindsight shortest path** on cleared levels: 5,031 actions → 3,390 (−32.6%), score 56.94 → 58.80 (**+1.86**). This needs the solution in advance, so it is an upper bound. 37 of 123 levels are already optimal and 56 are within 10%; median ratio 0.86. Ten levels hold 53% of the savings (ls20 L1–4, g50t L4–5, dc22 L1, wa30 L2–3, sk48 L1, sc25 L6, lf52 L3). By game type the ratio is 0.61 keyboard-only, 0.68 mixed, 0.81 click-only.
- **Realistic (no hindsight):** keep every new experiment the agent ran, but replace each stretch of already-known moves with the shortest route through the graph known at that moment. Only 491 cleared-level actions re-walk known moves; 152 are saved (3.0%), **+0.24**, mostly ls20 (16.0 → 21.4).
- **Tokens.** The first estimate put 0.82M generated tokens (16%) on re-walk actions. A skeptic regrouped actions into model calls (about 4.7 actions per call): 440k of the 0.82M is in calls that go on to new actions, where the reasoning is about the new experiment. Pure re-walk calls are 51 (381k tokens), 31 of them single-action deliberations (286k). Multi-action pure re-walks are 20 calls, about 95k tokens (**1.9%**). That is the realistic ceiling for a route macro.
- The level-path replay (r2) found a route shorter than the winning attempt on only 2 of 112 clears.
- **Unlocking levels is worth far more.** If each of the 15 unfinished games cleared one more level, the mean score would be:

| Actions on the new level | Mean score | Gain |
|---|---|---|
| 1x human baseline | 66.11 | +9.17 |
| 2x | 60.21 | +3.27 |
| 3x | 58.94 | +2.00 |
| 5x | 58.28 | +1.34 |

- On the Franzen control run (46.25), 97% of lost points (51.98 of 53.75) are uncleared levels; only 1.77 is efficiency below the cap.
- Never-cleared levels used 6.8 of 41.1 game-hours and 1.21M of 5.10M tokens for 661 actions, about one action per 37 s. Every visited state there still had untried actions (e.g. re86 L6: 366 of 470 pairs untried; wa30 L4: 442 of 560).

### 2.4 The graph explorer as the policy

Re-implementation of Rudakov et al.: single-colour segmentation, status-bar masking, salience tiers, a directed state graph, untried actions first, then the shortest path to the nearest state with untried actions. No LLM, one seed.

- **5,000 actions per game, 25 games:** 20 of 183 levels cleared, mostly L1 (L2 in ft09, lp85, tu93, vc33; L3 only in tu93). Within 5x human: vc33 L1 1.5x, lp85 L1 2.3x, r11l L1 3.0x. The rest: 13.7x–278x. Mean game score **0.11**. The engine ran 400–5,000 steps/s on one core; all 25 games took 2 min 15 s.
- **20,000 actions on 9 games where the LLM stalls early:** cleared sp80 L2 (5,698 actions vs human 18), lf52 L2 (13,842 vs 81), dc22 L2 (15,212 vs 117), cn04 L1 (8,449 vs 16). Still failed tn36 L1, g50t L1 (stalled at 42 distinct states), bp35 L1, wa30 L1.
- **All of these clears are on levels v1 already clears.** v1's 15 never-cleared levels are L2–L9 (mostly L4–L6). On those 15 games the 5k explorer cleared 0 levels in 5 games, 1 in 9 and 3 in tu93 (whose stall is L9), so it never reached a single stall level.
- **The LLM is far more action-efficient.** Over 10 full runs it cleared 1,053 levels at a median 0.67x the human count (q90 1.29x); none above 5x.
- **Why the paper's numbers don't carry over.** Its median 30/52 levels on the 6 preview games came from preview scoring (levels, with actions only a tiebreaker) and up to 96,000 steps per game. Here every action counts, squared.
- **5x cutoff.** The verified leaderboard cuts an agent off at 5x the human count per level. The local scorer (arc_agi 0.9.9) does not (su15 L7 reached 13.4x). Whether the Kaggle scorer does is unverified.
- **Competition mode.** Every explorer action would be a real API call; the game cannot be cloned. The Kaggle step rate is unmeasured; 400–5,000 steps/s is an upper bound.

### 2.5 Bandit signal

- **Sparse reward:** 0.020 / 0.019 level clears per action (ucx v1 / control v1). A cleared level takes a median 28 / 31 actions (p90 91 / 75) and 7 model calls (p90 16).
- **Action types look alike on a novelty reward:** the share of each type's uses that reach a new board is 0.79–0.90 / 0.77–0.88. Within a game it is similar too (ar25 0.95–1.0, re86 0.92–1.0, wa30 0.89–0.98). Only 6 action types with ≥ 10 uses fall below 0.6 (dc22 MOUSE 0.47, sp80 SPACE 0.33, sk48 LEFT 0.57, ls20 UP/DOWN 0.54–0.58, g50t SPACE 0.56).
- **It drifts:** for (level, type) cells with ≥ 10 pulls, the mean absolute change between a level's first and second half is 0.15; 22% / 26% of cells move by ≥ 0.25.
- **The clearing action is spread out:** CLICK 41, LEFT 20, RIGHT 15, SPACE 15, DOWN 13, UP 9 (ucx v1).
- **Click games have many arms:** 18 of 25 games use clicks, a median 8 / 6 distinct click cells per cleared click level (p90 18 / 16).
- **Scheduler pulls:** Franzen's scheduler hands over 51.1 times per 121-min run (25 games, 10 slots), about 2 per game; 290.3 at the hidden ratio (110 games, 532 min), about 2.6 per game. Exp3's bound, 2.63·√(T·K·ln K), is about 169 at T = 51, K = 25 and about 1,019 at T = 290, K = 110: larger than T, so it guarantees nothing.

### 2.6 How today's memory card is used (v4-user-combo-10x v1)

Main's card (`ARC3_LEDGER_ON_TRIM`, `inference/agent/memory_ledger.py`) is a `<memory>` block placed once at the head of the kept history after a trim. A replay of `memory_ledger.py` over the request logs reproduces all 79 cards the model saw exactly.

- **Writes.** 31 of 2,752 replies (1.1%) contain a tag, 58 tags in all: world_model 35, level_note 10, confirmed 12, refuted 1, open 0, scratchpad 0. 31 of the 58 were inside python code. All 16 python calls that held a tag failed with a SyntaxError, so the call was wasted and the tag lost. Only 27 tags from 15 replies (0.5% of calls) reached the ledger. 78 replies say they will record something; 8 (10%) contain a tag the ledger can read.
- **Contents.** 79 cards, 301 lines: 248 (82%) are level notes the harness clipped from reasoning, 32 world_model, 10 model level notes, 11 confirmed, **0 refuted**, 0 scratchpad. Of 109 distinct note lines, 97% are cut at 200 chars, 70% are narration or hedging ("Let me test UP quickly to confirm..."), 43% contain coordinates. Model tags are long (median 238 chars; 34 of 58 exceed the 200-char cap).
- **Stale items.** 5 of 20 model-written items that reached a card carry positions from an earlier level and are shown on later levels (r11l, m0r0, vc33, dc22). No item was ever revised or removed.
- **Cost.** A card is in 68% of requests, about 257 tokens when present, 175 averaged (0.31% of the prompt), plus a 291-token cached system addendum.
- **Reads.** 4.6% of replies with a card mention reading memory (vs 0.1% without). 6 replies call the notes garbled, muddled or sparse. bp35 took the prompt's example tag for a real note.
- **What does get used is harness data and code.** Retained python functions are called in 54% of python calls; 25% call one whose defining turn was already trimmed. The sandbox's `transitions` appears in 8.1% of code blocks (22 games), `history` in 6.4% (20 games). The model wrote its own BFS/visited/graph code in 152 blocks across 15 games. memory-v2's "Seen before" was quoted in the next reasoning in 5 of 8 cases; once its "Untried:" list led the model to probe arrow keys that had been no-ops on level 1.
- **Fill is not use.** thinking-digest v1 (optional): 0 of 347 calls. v2 (required parameter): 88% filled, item ids referenced in 1 of 323 replies, 3 of 87 open hypotheses ever resolved. hypothesis-loop: 93% filled, 0 of 27 references.
- **Other harnesses.** Fable: model-written "no effect" notes were wrong in 15 of 15 retested cases; a journal that replaced history cut levels from 46 to 23. rellik13: "a ledger for proven and refuted assumptions. The agent never used it"; the one text addition that helped (+2.8, 7.14 → 9.96) was solved-level memory built from harness facts. Franzen: turning off the text world model "worked best". Tycho's executable world model: 3.53 vs 29.22 on the same panel.

### 2.7 The utility, calibrated

The user's utility in a form the harness can compute, in game-score points (divide by 25 for run points):

```
U(a|s) = V_L * Pgain(a) + λ * IG(a) - Ccost(L,k) * [ c(a) + ρ * Pdeath(a) * (1 + d0(s)) ]
```

| Term | Estimator | What the data says |
|---|---|---|
| V_L, value of clearing level L | E[w_L · S at the clear] by depth | L1 3.3, L2–3 9.4, L4+ 19.3 game points |
| Pgain(a) | ĥ(chooser, L) · p_new(a); ĥ = clears per novel board | Depends on who picks. Model: 0.031 / 0.022 / 0.016 clears per action (L1 / L2–3 / L4+). Blind explorer: 0.00025 / 0.00008, 25–100x lower. p_new is badly calibrated: predicted 0.79 → observed 0.28; still non-monotone after recalibration. 60% of untried candidates are no-ops (65% in click games). |
| IG(a) | entropy of (no-op, change, death) for an untried class; 0 for a tried pair | Cannot be identified from logs. |
| Ccost(L,k), μ | P_clear(L,k) · E[w_L(S(b,N) − S(b,N+1)) \| clear], at the level's final count N | 0 below the cap (N ≤ b/√1.15 ≈ 0.93b), where 75% of cleared levels finish; median agent/human ratio 0.63. Otherwise 0.03–0.14 game points per action. One extra action on every cleared level: 0.53 / 0.53 / 0.58 / 0.62 run points per run. P_clear: L1 0.97 → 0.88 by k = 50; L2–3 0.92 → 0.80 by k = 100; L4+ 0.80 → 0.46 by k = 100. |
| c(a) | 1 for a step; d_f + 1 to walk to the nearest frontier state and probe | — |
| Pdeath(a), ρ | known 0/1 for a tried pair, else the class death rate; death costs d0 + 1 actions, so ρ = 1 | Well calibrated: predicted 0.001 / 0.011 / 0.032 vs observed 0.0005 / 0.016 / 0.027. 57 deaths in 4 runs, 53 on new pairs. Mean d0 at death 15.6 actions. |

Note: the calibrated μ replaces the earlier estimate of about 0.9–1.4 run points per extra probe per cleared level (systems-gap G12). The sign and the conclusion (harness probes on levels the model will clear cost points) are unchanged.

**Fitted weights** (conditional logit on 16,973 decisions on route states of cleared levels, leave one game out, 24 folds): p_new +3.16 ± 0.17, IG −1.91 ± 0.11, −death cost −0.14 ± 0.02. In the user's units: **λ ≈ −0.15 game points per nat**, **ρ ≈ 0**, μ not identifiable this way (every one-step candidate costs one action, so it cancels).

**Ranking on held-out games** (top-1 hit on the hindsight route; biased toward the model, since the routes are made of its own edges):

| Chooser | All | Keyboard | Click | Mixed |
|---|---|---|---|---|
| Utility (user's 3-term form) | 0.216 | 0.245 | 0.240 | 0.187 |
| Novelty only | 0.246 | 0.269 | 0.262 | 0.225 |
| Random | 0.113 | 0.230 | 0.042 | 0.049 |
| Model's own choice | 0.839 | 0.831 | 0.888 | 0.829 |

**Unbiased check: one-step outcome of the chosen action** (all 21,901 decisions; P(new board) / P(no-op)):

| Chooser | Cleared levels | Never-cleared levels |
|---|---|---|
| Model | 0.815 / 0.050 | 0.740 / 0.068 |
| Utility | 0.657 / 0.205 | 0.772 / 0.138 |
| Best untried by p_new | 0.662 / 0.199 | 0.752 / 0.151 |
| Random untried | 0.441 / 0.464 | 0.513 / 0.387 |

At least one candidate reaches a new board in 96–97% of decisions. The model picks better everywhere except open levels, where the utility finds slightly more new boards (+3.2 points) at twice the no-op rate.

**Budget table** (4 runs pooled, game points per exploratory action, k = probe index on the level):

| Depth | k | Model hazard | Gain (model) | Gain (blind) | Cost | Gain/cost (model) | Break-even λ |
|---|---|---|---|---|---|---|---|
| L1 | 0–4 | 0.010 | 0.035 | 0.0008 | 0.039 | 0.9 | 0.055 |
| L1 | 10–19 | 0.039 | 0.121 | 0.0008 | 0.048 | 2.5 | 0.067 |
| L1 | 45–59 | 0.026 | 0.040 | 0.0004 | 0.025 | 1.6 | 0.035 |
| L2–3 | 0–4 | 0.003 | 0.032 | 0.0007 | 0.109 | 0.3 | 0.155 |
| L2–3 | 10–19 | 0.029 | 0.274 | 0.0008 | 0.070 | 3.9 | 0.099 |
| L2–3 | 60–89 | 0.033 | 0.262 | 0.0006 | 0.040 | 6.6 | 0.056 |
| L2–3 | 130–199 | 0.017 | 0.099 | 0.0005 | 0.041 | 2.4 | 0.058 |
| L4+ | 0–4 | 0.000 | 0.000 | 0.0015 | 0.141 | 0 | 0.199 |
| L4+ | 10–19 | 0.026 | 0.498 | 0.0015 | 0.132 | 3.8 | 0.186 |
| L4+ | 45–59 | 0.022 | 0.408 | 0.0015 | 0.037 | 11 | 0.051 |
| L4+ | 90–129 | 0.008 | 0.091 | 0.0010 | 0.035 | 2.6 | 0.049 |

Break-even λ (game points per nat, IG ≈ 0.7 nats) is what a blind probe would need to pay. By game type, model gain/cost at k ≥ 10 is 1.2–8.3 (keyboard), 5–12 (click), 3.7–10 (mixed).

**Over-exploration check** (4 runs pooled):

- 91% of actions try a new (state, action) pair (92.5% on cleared levels, 86% on open ones). 87% of those reach a new board, 5% are no-ops, 0.27% are deaths.
- Exploratory actions pay: P(level cleared within 10 / 20 actions) is 0.26 / 0.47 after an exploratory action vs 0.06 / 0.23 after a repeat (cleared levels).
- On cleared levels 81% of decisions are on the hindsight-shortest route; off-route exploratory actions are 16% (2,815 of 17,477), 745 of them no-ops. Deleting with hindsight (not achievable): all off-route exploration +1.6 to +2.9 run points; only the no-op probes +0.4 to +0.8; exact known-bad repeats +0.03 to +0.13; full hindsight shortest path +2.5 to +5.3.
- On open levels the shortfall is volume: 96% of decisions had an untried action reaching a new board; 31.7% of all untried actions reach one (73.5% in keyboard games).

**What logs cannot give:** outcomes of untried actions beyond one step; λ; the time cost (shadow price of wall-clock) of model actions vs harness probes; anything counterfactual on the hidden set; whether Kaggle applies the 5x cutoff (if it does, Ccost spikes at N = 5b).

## 3. Proposals and skeptic verdicts

Each proposal was checked by a skeptic against the data above.

| # | Proposal | Verdict | Proposer's expected effect | What survives |
|---|---|---|---|---|
| 0 | Budget gate (`ARC3_BUDGET_GATE`) | refuted | 0 alone (enabler for 1a/1b) | Log its features as passive telemetry only |
| 1a | Graph memory (`ARC3_GRAPH_MEM`) | refuted: redundant | +0.3 to +0.9 | Run memory-v2 + level-path; later a HUD mask for sb26/sk48 and object-keyed click edges |
| 1b | Explorer burst on parked games (`ARC3_GRAPH_BURST`) | refuted | +0.0 to +0.7 | Optional CPU replay as a measurement |
| 2 | Gradient bandit: reject in harness; Thompson run allocator | survives with fix | 0 | The rejection only; drop the allocator |
| 3 | Episode store + computed card (`ARC3_EPISODE_STORE`) | refuted: redundant | +0.3 to +1.0 | Tag-strip fix (already on `arm/memory-v2`) |
| 4 | Memory card redesign | survives with fix | (as 3) | Cut down to memory-v2 + level-path; change one thing at a time |

### 3.0 Budget gate: the user's utility as the explore/exploit switch (refuted)

- **Mechanism.** A per-game estimator computes U(a|s) after every action, never shown to the model. Because the model out-picks every computable utility, the gate never restricts or advises the model. It only decides what the harness does: stay silent (EXPLOIT), emit graph frontier facts (STUCK-FACTS, 1a), or run a CPU explorer burst while the game is parked or stuck (1b).
- **Explore/exploit rule.** EXPLOIT for the first 10 actions of a level, and while the model's expected gain beats Ccost or the novelty rate over the last 10 actions is ≥ 0.5. STUCK-FACTS when k ≥ 10, novelty < 0.3 for 2 turns and P_clear below the depth median. BURST when the game is parked, or > 600 s since the last new board and P_clear·E[μ] < ĥ_blind·p_new·V_L. Burst size B* from the utility; caps 200 per park, 600 per level; stop on a clear, 2 deaths or 25 probes with no new board.
- **Channel.** None in the prompt; one line per action to `*_gate.jsonl`.
- **Cost.** 0 tokens; about 0.2 ms CPU per action; 1 day to build.
- **Stage 0.** Leave-one-game-out over 24 games x 4 runs: P_clear Brier ≥ 5% better than a per-depth constant; never opens in the cap-free region; open ≥ 60% of never-cleared wall-clock after first opening; opens before the clear on ≤ 10% of cleared levels, costing ≤ 0.15 run points per run.
- **Kill rule.** 25-min balanced panel vs gate-off: any decision in a level's first 10 actions; the gate opens on a level the model clears within 20 actions in ≥ 2 cases; missing log; any prefix break.
- **Verdict: refuted.**
  - It puts nothing in front of the model and expects 0 on its own, so it has no testable score effect.
  - Blind probes lose at every calibrated depth (gain/cost 0.006–0.04; 0.10 only for L4+ at k 130–199 on 130 decisions). P_clear would have to fall to 1–4% of its bin average for a burst to pay.
  - Its stuck signals cannot tell a stuck level from one about to clear. Over the 4 runs, "k ≥ 10 and > 600 s since the last new board" fired on 99 levels; 70 (71%) were later cleared, 24 within 20 actions. Adding 200 probes to those levels costs a median 88 game points per level (unweighted). "Novelty < 0.3" fired on 24 of 66 never-cleared levels and on 62 of 444 cleared ones.
  - Novelty measures the wrong thing: never-cleared levels keep finding new boards (median rate 0.80 vs 0.87 on cleared levels). The bottleneck is inferring the goal (Makarov: 83% of stuck forks had already tried the decisive action; from a discussion snippet that could not be fetched).
  - Even the claimed upside of 1a and 1b together (+0.3 to +1.6) could not be confirmed with affordable runs against a run sd of 1.7–4.
- **Fix.** Don't build it. If a stuck signal is wanted later, log the features (novelty rate, seconds since the last new board, k, tokens on the level) in `events.jsonl` as passive telemetry, and fit a classifier only after 10+ runs. Scripts: `scratchpad/gbm/util/skeptic_gate.py`, `skeptic_gate2.py`.

### 3.1a Graph memory: per-level state graph as facts and a sandbox API (refuted: redundant)

- **Mechanism.** A per-level StateGraph keyed on the board hash with an auto-detected HUD mask; click edges keyed by object (colour + shape + bbox); edges store outcome (no-op / moved / changed / died / cleared); frontier = untried edges ranked by salience and p_new; BFS routes to the level start, the board before a death, the nearest frontier or any board. Pushed facts: a "Seen before" line on a revisit (≤ 160 chars), a death line with a route back, and in STUCK mode only a frontier line (never a bare "Untried: UP DOWN..." list). Everything else queryable as `memory.*` in the sandbox.
- **Explore/exploit rule.** From gate 0. EXPLOIT: revisit and death lines only. STUCK-FACTS: add the frontier line. The harness never moves the avatar; routes are offered as action lists for `action(memory.route_to(...))`.
- **Channel.** Tail text in tool results, plus the sandbox `memory` object; one fixed system sentence.
- **Cost.** About 6–10 tokens per action averaged; about 120 cached system tokens; 1–3 ms CPU per action; about 2 days.
- **Expected (proposer).** +0.3 to +0.9: +0.24 navigation, ≤ +0.13 known-bad repeats, about +0.5 from saved re-walk tokens.
- **Stage 0.** Replay 4 runs: HUD mask matches on ≥ 23/25 games and the graph is ≥ 97% deterministic; revisit lines on 10–14% of actions; nav saving ≥ +0.2; `route_to('death:-1')` no longer than the model's re-walk in ≥ 90% of cases; frontier lines name a class that reaches a new board in ≥ 60% of firings vs the 31.7% base rate.
- **Kill rule.** 25-min panel vs memory-v2-both: `memory.*`/`level_path` in < 1% of snippets and exact repeats > 7%; loop share not down ≥ 3 points; any prefix break; > +60 tokens per executing turn.
- **Verdict: refuted as redundant.**
  - The revisit line is memory-v2 `ARC3_MEM_SEEN`; the death route is level-path `level_path_before_death` plus memory-v2 `<hazard>`; BFS routes as an action list are `level_path`. New are only the HUD mask, object-keyed click edges, routes to arbitrary targets and the frontier line.
  - The HUD fix is mostly moot: memory-v2 and level-path hash with a 4-cell crop, which covers the bar in 23 of 25 games (not sb26 or sk48, bar at row 53).
  - The sandbox already exposes the full `history` and `transitions` with before/after frames (`python_tool_sandbox.py:490-506`), and the model uses them (8.1% / 6.4% of code blocks) and writes its own BFS (152 blocks).
  - The token saving is miscounted: about 95k tokens (1.9%), not 0.82M (section 2.3).
  - The rest is under +0.4 even with perfect use, against a run sd of 1.7–4. A null arm would pass the 2-run keep rule about 25% of the time.
  - The frontier line is the risky part: every visited state on open levels has untried actions, the ranking has little signal, and the "Untried:" list already drew no-op probes once. (The 31.7% base rate the skeptic questioned does have a source: the calibration replay.)
  - `arm/memory-v2` and `arm/level-path` are each about 245 commits behind main, so porting them is real work, and memory-v2-both has no completed run to compare against.
- **Fix.** Don't build 1a. Rebase memory-v2 and level-path onto main and run them (section 5). Only if `level_path` reaches 1% of executing snippets, add (a) an online HUD-mask detector (matters for sb26 and sk48) and (b) click edges keyed by object instance (su15 repeat share 22.6% by object vs 0% by pixel). Drop the frontier line and the token-saving claim.

### 3.1b Explorer burst on parked games (refuted)

- **Mechanism.** When gate 0 allows, a harness thread runs a Rudakov-style explorer on the game's current level using 1a's graph: untried edges at the current node by salience, then the shortest route to the nearest frontier. Never repeats a known no-op/death pair. Stops at B*, a clear, 2 deaths or 25 probes with no new board; walks back if the route is ≤ 10 actions. The model's next tool result gets one summary line (probes, new boards, deaths, new effects, route ids).
- **Explore/exploit rule.** Only when U says blind probes pay, or the game is parked (GPU busy with other games). Never in a level's first 10 actions; never preempts a model turn. A saturated level is flagged "explored-out".
- **Channel.** Harness-executed actions plus one summary line (≤ 240 chars).
- **Cost.** 60–120 tokens per summary, ≤ 3 bursts per game, 0 GPU; burst actions count in the level's squared score; about 2 days on top of 1a.
- **Expected (proposer).** +0.0 to +0.7: about +0.34 per rescue at L2 of a 7-level game, mostly through unlocking the next level.
- **Stage 0.** Local replay to every point where the gate would open; count rescues on never-cleared levels and re-score cleared levels with the burst added. Pass if net ≥ +0.3 per run in ≥ 3 of 4 runs and the loss on cleared levels ≤ 0.15.
- **Kill rule.** Median API step latency > 1 s; bursts on < 2 games; > 10% of actions on later-cleared levels from bursts; any thread race; summary quoted in < 1 of 5 resumed turns.
- **Verdict: refuted.**
  - No measured rescue is on a level the best stack fails. Every 5k/20k explorer clear is on a level v1 already clears; under a 600 cap only sk48 L1 is reachable, which the model clears near human cost.
  - A rescue is worth less than claimed. A clear at about 600 actions on a level with baseline ~100 scores about 3/100. Several stall levels were reached late in the run, with little time left on them (time spent on the level: g50t L6 155 s, ls20 L5 213 s, dc22 L4 253 s), so an unlock there could not be used. The truly stuck ones (bp35 L3, ka59 L4, m0r0 L6, re86 L6, su15 L4) are mostly click levels with 500–866 untried click-object pairs, the regime where the method degrades.
  - Bursts won't stop early: 83% of transitions reach a new board, so "25 probes with no new board" rarely fires. Re-scoring v1 with one burst on a later-cleared level costs 0.16 / 0.27 / 0.37 / 0.44 / 0.50 run points at +25 / 50 / 100 / 200 / 600 actions. One false fire per run cancels a rescue.
  - It is the rejected G12 affordance sweep at a different moment, and goal judgement is the bottleneck, not novelty.
  - Unbounded risks: the Kaggle 5x cutoff is unverified, and 600 is above 5x the baseline on most stall levels (bp35 44, ka59 51, g50t 54, ls20 96); at the hidden ratio about 100 games sit parked at once, so bursts would hit the API in parallel at an unknown rate.
- **Fix.** Drop it. The one step worth keeping is a CPU replay run as a measurement: from the model's last board on each never-cleared level, count rescues at ≤ 5x the human baseline. Revisit only if it finds ≥ 3 rescues across 4 runs.

### 3.2 Gradient bandit (survives with fix: keep the rejection, build nothing)

- **Mechanism.** Reject all three in-harness uses: (a) game scheduler (2–2.6 pulls per game, non-stationary, Exp3 bound vacuous, softmax stays near uniform at α = 0.1 after 2–3 pulls, so it behaves like round robin, which lost −2.89); (b) actions inside a level (deterministic, state-dependent, sparse, novelty doesn't separate arms); (c) choosing among model trajectories (none proposed). Use top-two Thompson sampling (Russo 2016) on run scores to choose which Kaggle arm to run or repeat next.
- **Explore/exploit rule.** Run allocation only: sample arm means, run the best sample, or with probability 0.5 the runner-up. Stop an arm at P(> incumbent) < 0.1 after ≥ 2 runs; promote at > 0.9 after ≥ 3.
- **Channel / cache / cost.** Offline script; no harness change; 0 tokens; half a day.
- **Expected.** 0 points per run; the claim was 20–30% fewer runs.
- **Stage 0.** Feed the recorded run scores in order and check it reproduces every keep/drop call with no more runs.
- **Verdict: survives with fix.**
  - The in-harness rejection holds, and every number checks against `arms.md` (curriculum 20/20 fail; UCB −1.73/−2.03/−2.42 offline, −1.13/−2.60/−3.59 hidden; real UCB 43.84; Gittins −0.6; Duck uniform 4.678 vs 17.743; UCB floor 9.165 vs 9.856; Exp3 bound 169 and about 1,018).
  - The allocator fails its own Stage 0: hats (37.11), accept05 (46.77) and tput-safe (42.09) were rightly dropped after 1 run, which its 2-run minimum forbids.
  - Its calls depend on the sd assumed and conflict with the user's rule: at sd 1.7, user-combo-10x vs control gives P = 0.88, nearly "promote", where the user ruled "not an alpha". At sd 3.93 it would ask for a third 13-expert run the user rightly skipped.
  - The "20–30% fewer runs" claim has no source for 1–3 runs per arm and a drifting baseline, and there are only about 8 scored v4 full runs to allocate.
- **Fix.** Add the rejection to `arms.md` (gradient bandit / Exp3 / softmax scheduler and in-level action bandit, with the evidence above). Build nothing; don't run the `sched_curriculum` GradientBandit replay. For repeats, use `make significance` per game and the user's "every repeat higher" rule.

### 3.3 Episode store + computed card (refuted: redundant)

- **Mechanism.** A per-game append-only EpisodeStore records every action across trims and levels and serves it as `memory` in the sandbox. The trim card is computed from it: controls, solved levels with win signatures, here/frontier, deaths, and one model-written goal line. It replaces the ledger's world model, hypotheses and clipped level notes. Two fixes ship with it: strip memory tags out of python code before exec, and read a `<goal>` tag from the reply or the thinking.
- **Explore/exploit rule.** None of its own; it is the shared state for gate 0 and 1a/1b.
- **Channel.** Card at trims; facts in tool results; `memory` object in the sandbox. Model-written: the goal line only.
- **Cost.** Card ≤ 1,600 chars (~480 tokens), median ~270 tokens vs the ledger's ~257; about 3 MB per 25-game run; 1–3 ms per action; about 3 days.
- **Expected (proposer).** +0.3 to +1.0, assuming about 2 more levels unlocked per run.
- **Stage 0.** Replay at the 79 trim points: every non-goal line harness-verifiable; ≤ 1,600 chars each; a solved record for every cleared level ≤ 60 tokens; ≥ 90% of later known-bad repeats and post-death re-walks covered; ≤ 5 ms p99 per action; historical snippets unchanged with `memory` present.
- **Kill rule.** Any tag-in-code SyntaxError; card-reference rate < 4.6%; < 3 of 10 games call `memory.*` and reference rate < 6%; any prefix break; prompt tokens up > 2%.
- **Verdict: refuted as redundant.**
  - memory-v2 already has solved lines, the effect table with hazard lines, "Seen before" with tried/untried, the 4-cell crop and a capped card. level-path has the routes and per-action store.
  - The tag-strip fix already exists on `arm/memory-v2` (`_mem_absorb_code` → `memory_ledger.strip_tags_in_code`), gated on `ARC3_MEM_CARD`. It is worth about 16 calls in 2,752 (0.6%).
  - The new parts were already dropped by the systems-gap skeptics (sandbox globals, the free-text per-level ask) or will go unused (tags in 1.1% of replies). The kill rule joins two conditions with AND, so the sandbox API is never killed on its own non-use.
  - The frontier distance is almost always 0 (every visited state on open levels has untried actions) and invites probes.
  - Dropping the clipped notes removes most of the content of the only broad local memory result (ledger-on-trim 52.56 / 49.14 vs 46.25, not significant), and those notes are read about as often as model-written items.
  - Every Stage-0 check tests presence, not use.
- **Fix.** Don't build EpisodeStore. Rebase memory-v2 and level-path, turn on `ARC3_MEM_REPLY_WINS=1`, move `strip_tags_in_code` out from under the `ARC3_MEM_CARD` gate, run solved-levels first, then seen-state + action-effects and level-path (section 5).

### 3.4 Memory card redesign (survives with fix)

The detailed card design behind proposal 3 (layout in section 4.2), judged as a card rather than as a new arm.

- **Design rules.** The harness stores and computes; the model only reads. Push only facts tied to the current moment (revisit, death, clear, card at a trim); keep the rest queryable. Never show a stale general rule: every entry carries its scope (level, board or object class) and counts. The model writes one thing, the goal.
- **Verdict: survives with fix.**
  - Most of it is built and unscored (memory-v2, level-path). New are the win signature, object deltas, auto HUD mask, `memory.*` API, goal tag, ledger removal and the gate/frontier.
  - The channel is plausible: the model uses harness data (`segmentation` 25.3% of python calls, `frame_diff` 10.4%, `transitions` 8.1%, `history` 6.3%) and quoted "Seen before" 5 of 8 times. A one-sentence `memory.*` API is closer to the optional helpers it ignores (`trajectories.propose` 0 of 10 games).
  - The goal line contradicts the design's own rule 1 (tags in 1.1% of replies; required fields filled but unused).
  - Click edges keyed by object **class** would create false "tried" facts: 72% of clicked objects have a same-colour, same-shape twin on the board (median 4; ft09 14.5, tn36 17). That is the Fable failure mode. A cross-level "controls" table has the same risk, since effects drift within a level, and it needs the controllable-entity tracker (not built).
  - Removing the ledger in the same arm makes any result unattributable; the ledger is part of the best stack and has never run alone on v4.
  - A 25-min panel can't see the card: the first trim card comes at a median request 65 of about 203 per game.
  - Opaque board ids ("b37") are worse than memory-v2's "the board after action #10".
  - Cache and cost are fine (card where the ledger already sits; lines at the tail; under 1% of the prompt).
- **Fix.**
  1. Port `arm/memory-v2` (memory-v2-both) and `arm/level-path` onto main. Leave `ARC3_LEDGER_ON_TRIM` on, set `ARC3_MEM_REPLY_WINS=1`, keep flags-off byte-identical. Run paired full 121-min runs against user-combo-10x; judge with `make significance` and repeats.
  2. Only then add: a computed win signature in the solved line; click keys by object **instance** (position), with class only as counts in the effect rows; an append-only `<game>_episode.jsonl` action store for logging and analysis only.
  3. Drop from this arm: the goal tag, the cross-level controls typical-delta, the frontier line and burst summaries, and removing the ledger (test ledger-off as its own arm later).
  4. Keep memory-v2's "board after action #N" wording. Put any `memory.*` API behind its own flag with a kill rule of < 3 of 10 games calling it.

## 4. The memory card

### 4.1 Today (main, `ARC3_LEDGER_ON_TRIM`)

Shown once at the head of the kept history after each trim (`memory_ledger.py` `render()`):

```
<memory>
  <world_model>...</world_model>                      # model tags, ≤ 6 entries
  <hypothesis status="confirmed">...</hypothesis>     # ≤ 8
  <hypothesis status="refuted">...</hypothesis>       # ≤ 8
  <level_note>level N: ...</level_note>               # ≤ 6; mostly clipped from <think>
  <scratchpad>
  ...                                                 # ≤ 1,200 chars, verbatim
  </scratchpad>
</memory>
```

Every entry is cut to one line of at most 200 chars with "...". Open hypotheses are not kept as items. A new entry only replaces an exact duplicate; otherwise the oldest is dropped; there is no revise or delete. The ledger resets per game, not per level. At a level change the harness joins up to 8 one-sentence summaries of the model's reasoning and cuts them at 200 chars, so the note is the oldest of the last 8 findings, not the insight that cleared the level.

Measured contents (v1, section 2.6): 82% clipped notes, 0 refuted entries in 79 cards, model tags in 1.1% of replies.

### 4.2 The proposed redesign (as designed, before the skeptics)

```
<memory game="ls20" level="4" spent="37" deaths="1" boards="29">
<controls>                       # HARNESS · ≤ 6 lines · ≤ 90 chars · whole game
UP/DOWN/LEFT/RIGHT: moves orange 3x3 4 cells (68/71; 3 blocked at walls); 0 deaths
CLICK blue 3x3: recolours it blue→red (L2-L4, 9/9)
SPACE: no change on boards b1-b40 (12/12)  [L1-L4]
</controls>
<solved>                         # HARNESS · newest 3 levels in full (≤ 200 chars each)
L3: 18 acts, 2 deaths; route [U4 R3 D2 CLICK yellow-2x2]; win: red count 3→0, then CLICK yellow
</solved>
<here>                           # HARNESS · current level · ≤ 3 lines
board b37 visited 3x; tried UP(→b38) LEFT(GAME OVER); route from start 9 moves
[STUCK mode only] nearest untried: CLICK teal 3x1 at b40 (2 moves); novelty last 10 actions: 1/10
</here>
<deaths>                         # HARNESS · newest 3 on this level
b52 RIGHT (into red cell) after 14 moves
</deaths>
<goal by="model, unverified">reach the yellow door with the key carried</goal>
</memory>
```

Plus pushed lines in tool results ("Seen before: board b37 ...", "Died: RIGHT from b52 ... route back: 14 actions", "Solved L3 in 18 actions ...", a burst summary and a STUCK-mode frontier line) and a sandbox API (`memory.transitions`, `effects`, `untried`, `route_to`, `deaths`, `solved`, `board`, `visits`, `here`). Cap 1,600 chars.

### 4.3 What to run first (memory-v2 + level-path as built, after the skeptics)

Card at each trim (memory-v2 `ARC3_MEMORY_PLACEMENT=both`, cap 3,900 chars), harness facts first and the existing ledger entries after them:

```
<memory>
<solved>L1: 13 acts [L3 U4 R3 U3], spent 40, 0 deaths | end: block moved (10,4)->(34,20) | how: take the key before the door</solved>
<actions>
UP shift 13 noop 2 clear 1 | game shift 31 noop 6 clear 2
RIGHT shift 6 noop 2 | game shift 12 noop 4 death 1 (fatal in 1 board)
SPACE untested this level | game noop 2/2 (inert so far)
</actions>
<hazard>L2: RIGHT from the board first reached at action 14 killed attempt 2 (31 acts)</hazard>
<world_model>...</world_model>                     # ledger entries kept (model tags)
<hypothesis status="confirmed">...</hypothesis>
<hypothesis status="refuted">...</hypothesis>
</memory>
```

Click games get colour-grouped rows instead of key rows, e.g. `MOUSE@R 7 targets: change 5 noop 2, median 12 cells, e.g. (16,22) change, (38,38) noop`.

Per-turn lines (tail text, cache-safe), held back until a level has used 15k generated tokens (`ARC3_MEM_LINES_AFTER_TOKENS`):

```
Seen before (board interior matches): after action #6 and #8 of this attempt. From it you tried: DOWN -> back to a seen board (loop). Untried: UP, LEFT, RIGHT.
<memory_update>
+ solved L2: 21 acts [R3 U2 D4 ...], spent 52, 1 death | end: ... | how: ...
+ RIGHT: first death (attempt 2, after 31 acts); UP: first noop
</memory_update>
Shortest route on record from this level's start board to the current board: 3 actions [R U2]; this attempt so far: 5 actions. Route length after each action since the last message: 1 2 1 2 3.
```

Each `action(...)` echo ends with a short route suffix (`| route 1 (shorter than 3 on record)`), and the sandbox gains `level_path`, `level_path_len` and `level_path_before_death` (lists `action()` accepts).

Later additions, one at a time and only after the step above is scored: a computed win signature in `<solved>`; click effects keyed by object instance; an episode jsonl for analysis. The "Untried:" part of the seen-before line is the one piece with observed harm (it drew no-op arrow probes once); watch it in the read-out.

### 4.4 Field-by-field

| Today | First test (built) | Later, if the first test holds | Dropped |
|---|---|---|---|
| `<world_model>` (model tags, 1.1% of replies) | Kept, plus `<actions>` effect table (harness counts, this level / game) | Effects keyed by object instance | Cross-level "controls" typical-delta (needs the entity tracker; effects drift) |
| `<hypothesis>` confirmed / refuted (refuted empty in 79/79 cards) | Kept, plus "Seen before" per board, `<hazard>` death lines, `level_path_before_death` | Object-instance click keys in tried lists | Frontier / "nearest untried" line; opaque board ids |
| `<level_note>` (82% of lines, clipped) | Kept as is (ledger-off is its own arm), plus `<solved>` with path, spent, deaths, end, how | Computed win signature in `<solved>` | Removing the ledger in the same arm |
| `<scratchpad>` (never written) | Kept | — | — |
| — | Tag-strip for tags written inside python code (fixes 16 SyntaxErrors per run) | — | Model-written `<goal>` tag; `memory.*` sandbox API (own flag only) |

## 5. What to build first, and why

**Why this order.** The large lever is clearing more levels (+9.17 if each unfinished game cleared one more level at human cost), not action efficiency (+0.24 realistic, +1.86 with hindsight). Stuck levels are time-bound, not short of options. Harness-computed facts are the one memory channel this model reliably reads, and memory-v2 + level-path are built, tested (flags-off byte-identical, #58 eviction guard) and unscored. Running them costs Kaggle GPU time but no new code beyond a rebase.

1. **Rebase `arm/memory-v2` and `arm/level-path` onto main** (the user-combo-10x stack, ledger on). They are about 245 commits behind. Set `ARC3_MEM_REPLY_WINS=1`. Move `strip_tags_in_code` out from under the `ARC3_MEM_CARD` gate so tags written in code no longer crash the call. Keep flags off byte-identical to main.
2. **Run `solved-levels` (`ARC3_MEM_SOLVED`) on user-combo-10x first** (systems-gap step 0). Full 121-min runs, paired with user-combo-10x, `make significance`, repeated. 25-min panels cannot see the card (first trim at about request 65 of 203).
3. **Then `seen-state` + `action-effects`, and `level-path`, against the same-commit stack.** Pre-registered kill rules: loop share must fall ≥ 3 points; `level_path` in ≥ 1% of executing snippets; exact (state, action) repeats ≤ 6%.
4. **Only if `level_path` is used:** an online HUD-mask detector (matters for sb26 and sk48) and object-instance click keys.
5. **Only if solved-levels at least matches the stack:** the computed win-signature card (systems-gap G8-reduced), with its own Stage 0 (separating predicate ≥ 60%, true on ≤ 5% of non-clearing frames).
6. **Free telemetry:** log novelty rate, seconds since the last new board, k and tokens on the level in `events.jsonl`, with no action taken. This lets a stuck classifier be fitted later on 10+ runs.
7. **Bookkeeping:** add the gradient-bandit rejection to `arms.md` under Didn't work / not pursued.

**Not building:** the budget gate, the explorer burst, the EpisodeStore, a `memory.*` sandbox API (unless behind its own flag), the goal tag, any in-harness bandit, and the Thompson run allocator. Optional measurement only: the CPU replay of explorer rescues from the model's last board on never-cleared levels at ≤ 5x baseline; revisit the burst only if it finds ≥ 3 rescues across 4 runs.

**Detection power.** The expected effects of steps 2–3 are small next to a run sd of 1.7–4 (navigation +0.24, known-bad repeats ≤ +0.13, token saving ≤ 2%). The usage kill rules (loop share, `level_path` share, repeats) will give a clear signal within 1–2 runs; a score claim will need several paired repeats under the user's rule. The solved-level record is the part that could matter more, because it is the closest thing to rellik13's +2.8 and aims at unlocking levels.

## 6. Sources

Papers and docs:
- Rudakov, Shock, Cowley, "Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks", arXiv 2512.24156 (AAAI-26 workshop): https://arxiv.org/abs/2512.24156 ; code https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore
- ARC-AGI-3 technical report: https://arxiv.org/html/2603.24621v1 ; methodology and scoring: https://docs.arcprize.org/methodology
- Preview results: https://arcprize.org/competitions/arc-agi-3-preview-agents ; https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings
- Go-Explore, Ecoffet et al., arXiv 1901.10995; Intelligent Go-Explore, Lu, Hu, Clune, ICLR 2025, arXiv 2405.15143
- Tang et al., "#Exploration", arXiv 1611.04717; Yamauchi, frontier-based exploration, CIRA 1997; Agent57, Badia et al., arXiv 2003.13350
- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., §2.8 (gradient bandit): http://incompleteideas.net/book/RLbook2020.pdf
- Mei et al., stochastic gradient bandit convergence, ICML 2023, arXiv 2402.17235
- Auer, Cesa-Bianchi, Freund, Schapire, Exp3, SIAM J. Comput. 32(1), 2002
- Russo, top-two Thompson sampling, arXiv 1602.08448; Russo et al., Thompson sampling tutorial, arXiv 1707.02038
- NVIDIA DreamTeam, arXiv 2605.09650

Competitor code (read-only, `arc-agi-v4/competitors/`): `dream-team/arc_agi_3/exploration_strategy.py`, `config.py`, `arc_league/collapse_gates.py`; `ARC3-solution` (StochasticGoose); Blind Squirrel https://github.com/wd13ca/ARC-AGI-3-Agents; `kaggle/arc-agi-3-champion-solution` (hard-coded public-game fingerprints); `rellik13-milestone2/WRITEUP.md`; `solve-arc-agi-3/docs/PLAN.md`; `arc-agi-3-prime-agent/README.md`.

Repo files:
- `ARC3-Inference/inference/agent/memory_ledger.py` (caps, `render`, `summarize_think`), `tool_agent.py:485-498` (state hash, `ARC3_STATE_IDENTITY_BORDER=0` hashes the HUD), `python_tool_sandbox.py:490-506` (`history`, `transitions`)
- `tufa-arc-agi-framework/src/taaf/game.py:381-413`, `.venv/.../arc_agi/scorecard.py:170-171`, `arc_agi/base.py:157-167, 431-432` (competition mode)
- `research/reports/memory-v2-design.md`, `memory-v2-measurement.md`, `harness-systems-gap.md`, `fable-handbook-digest.md`, `tycho-world-model-port-design.md`, `scheduling-curriculum.md` (on `analysis/sched-curriculum`)
- `research/notes/arms.md` (memory-v2, level-path, memory-v3, thinking-digest, hypothesis-loop, scheduler and bandit entries); `WRITEUP.md` L20, L297; `analysis/calibration/levels.csv`
- PRs #64 (memory-v2), #66 (level-path), #142 (multi-trajectory, closed)
- Run logs: `runs/kaggle/v4-user-combo-10x/{v1,v2-snowiers,v3-snowiers}`, `runs/kaggle/v4-control/v1`, `runs/kaggle/v4-panel-memory-v2-both/v1`, `runs/kaggle/v4-panel-thinking-digest/{v1,v2}`

Scratch analysis (not in the repo): `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/gbm/`
- Explorer: `scripts/graph_explore.py`, `out_5k.json`, `out_20k.json`, reference code `ext/just-explore/`
- Graph: `graph_opportunity.md`, `graph_stats_per_level.csv`, `nav_savings_per_level.csv`, `graph_summary.json`, `action_novelty.json`, `hud_mask.json`; scripts `graph_stats.py`, `nav_savings.py`, `aggregate.py`, `action_novelty.py`, `hud_probe.py`, `inverse_check.py`
- Bandit statistics: `action_stats.py`, `frontier.py`, `ucx_v1*.json`, `ctl_v1_b4.json`, `frontier_*.json`
- Card audit: `scripts/{writes,intent,analyze,strict,hits,panel,graph}.py`, `data/`
- Calibration: `util/sim_decisions.py`, `build_table.py`, `analyze.py`, `budget.py`, `extra.py`; `util/out/{an_pooled,an_ucx1,budget_final,extra_pooled}.json`, per-run decision files `util/out/*.jsonl.gz`, `ucx1.pkl`
- Skeptic checks: `util/skeptic_gate.py`, `skeptic_gate2.py`; `../skeptic1a/calls.py`; `../skeptic/{globals_use,trans_use,first_card,click_class}.py`
