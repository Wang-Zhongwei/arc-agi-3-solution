# Next moves: what to do with three weeks left

2026-10-10. The question (the user): "ok i think we optimized for tok/sec, and even for the harness. i have no idea how to improve. what can we do?"

How this was made: five lenses (where points are lost, scoring rules, untried ideas, model side, test method) produced a draft plan. Two critiques re-ran its numbers and checked feasibility. This is the revised plan. Scripts and outputs are in [`../data/next-moves/`](../data/next-moves/) (see the last section). Numbers are on the 25-game, 121-min offline proxy unless marked LB (Kaggle leaderboard). Guesses are marked as guesses.

**Late result folded in:** `mem-v2-10x` finished at 22:36 UTC and scored **43.81** (106 levels, 6 wins). That is below all three user-combo-10x references (56.94 / 52.70 / 49.09), even though its 25-min screen led them (11.74 vs 8.80). It generated more tokens (5.40M vs 5.10–5.17M) and cleared fewer levels per 1M tokens (19.6 vs 21.3–24.1; `final/run_summary.py`). The rule for this run was written before the score was read (`final/R0-mem-v2-rule.txt`): below 44 means memory-v2 leaves the factorial and gets its own diagnosis. It applies.

## The answer

Throughput and action efficiency are used up. Almost every missing point is a level we never clear. What's left is to take the points the scoring rules give for free, and to stop wasting runs on tests too noisy to decide anything. In order:

1. **Build the "victory lap" (about 1 day).** After a game is won, RESET once and replay each level's shortest recorded route with no LLM. The scorer keeps the better play. This is worth +0.34 per run on average and +0.84 on our best stack, and it cannot lower a score if play 2 finishes. It rides along on every later run.
2. **Label every submission and pick the final two on purpose.** Kaggle counts the better private score of the two you select. Within one run, public and private halves don't predict each other. So select two copies of one frozen config, never by public score. This is worth +0.2 to +1.2 LB directly on the private score, and costs no GPU (scoring reruns don't use quota; checked with `kaggle quota`).
3. **Fix how we decide.** The current stack's run-to-run sd is about 3.3–3.9, not the pooled 2.47. At that noise, +3 needs 15–21 runs per arm, and +2 can't be confirmed by Nov 2. Grow a reference pool first, write the decision rules down before each test, and use a per-level token test as a second check.
4. **Screen the built flags together in one 16-run factorial on mellamoa.** Factors: debrief, before/after, Fable facts, hypothesis tags, dyn-temp, sandbox, and an 8k output cap. The memory-v2 column becomes a placebo. This needs 16 runs instead of about 66 one at a time. Expect +0.4 to +1.5 (a guess, from our base rate of arms that worked).
5. **If the user will spend about $140, rent RTX PRO 6000 pods.** This is the only way to add statistical power before the deadline. It doubles the factorial to 32 runs (power for a +3 flag goes from 0.45–0.56 to 0.70–0.81) and gives the final stack 12 confirmation runs.
6. **Rebuild the FR-Spec draft-token map (lossless, half a day).** It enters the final config after a fixed-prompt serving bench, not through score runs.
7. **After the freeze, build the restart and novelty ideas.** These are a stuck-level restart at 40 active minutes, level-start novelty facts, and a checkpoint-branch bench. Each runs as a CPU check first. None can be confirmed before Nov 2.

**What to expect (a guess):** about +0.5 to +2.5 on the proxy by Nov 2. That is about +0.2 to +1 LB, plus the final-pick gain of +0.2 to +1.2 on the private score. The victory lap is the only item that can be proven to work, and only because of how the scorer is built. On mellamoa alone, nothing else can meet the user's rule that repeated runs are consistently higher, unless its effect is about +4 or more. 12 wins in every run is out of reach: user-combo-10x wins 10 / 7 / 6 (mean 7.7, sd 2.1).

**Tonight, before 00:00 UTC:** the push queue (`push3.py`, PID 80736) will start pushing at the daily budget reset. In order, it holds fable-facts-10x, uc-11x-16s, expert-stats-10x, panel-scorer, panel-debrief, panel-dyntemp, panel-dyntemp-split and mem-solved-10x. Recommended:
- Keep expert-stats-10x, and add one user-combo-10x reference run.
- Cancel the four panels, fable-facts-10x and mem-solved-10x. The factorial covers them, and single-flag runs from other commits can't enter its fit.
- Run uc-11x-16s only if the user still wants it.

The user decides.

## 1. Where the points are lost

Best stack, user-combo-10x v1–v3, mean 52.91, so 47.09 points per run are missing. Scoring follows `taaf/game.py:381-413`, which mirrors `arc_agi/scorecard.py:168-171`. The script is `where-lost/decomp.py`.

| Part | Points per run | One more level only | What it is |
|---|---|---|---|
| Inefficiency on cleared levels | 2.17 | – | Actions above what each level needed. 1.81 of it is actions in attempts that ended in a RESET or death. Only 0.56 is above 1.5× the human count. |
| Clock | 4.38 | 1.25 | Usually-cleared level, run spent less than the median clear time on it (6.4–9.3 at looser thresholds). |
| Stuck | 15.76 | about 4.1 | Usually-cleared level, run spent at least the median time and failed. |
| Wall | 18.4 | 3.1 | Level cleared in under half the runs that reach it. |
| Frontier | 6.34 | 2.67 | Never cleared in any of 14 v4 runs. |

- **Efficiency is used up:**
  - Cleared levels take 0.57–0.64× the human actions. 82% of 563 cleared levels are at or under the human count, and 75% of them score the 115 cap (`where-lost/decomp.txt`).
  - Spending 1.25× the actions everywhere would cost 2.4–3.5 points.
  - level-path cut inefficiency as designed (1.25 vs 2.17) but scored 48.31. Its tokens per cleared level did not change (ratio 1.00, p = 0.64).
- **Walls cluster at L2 and at new mechanics** (`where-lost/walls.txt`, 14 runs). The top 10 walls are worth about +6.75 if always cleared:

  | Level | Cleared / reached |
  |---|---|
  | tn36 L2 | 7/14 |
  | cn04 L2 | 7/14 |
  | su15 L4 | 3/14 |
  | s5i5 L4 | 3/10 |
  | dc22 L5 | 0/10 |
  | ka59 L6 | 1/8 |
  | re86 L7 | 1/8 |

- **The clear rate decays with time** (`where-lost/hazard.txt`). On usually-cleared levels, the chance of clearing in the next 10 active minutes is:

  | Active minutes on the level | Chance of clearing in the next 10 min |
  |---|---|
  | 0 | 0.66 |
  | 20 | 0.44 |
  | 25 | 0.45 |
  | 30 | 0.38 |
  | 40 | 0.10 (n = 39) |

  The median clear time is 6.1 min and the p95 is 26.
- **Variance:** the best score per game across four 10x-stack runs averages 68.2, against a mean of 51.8. The stack can reach these outcomes but doesn't repeat them.

## 2. What the scoring rules allow

Sources: arc_agi 0.9.9 in `ARC3-Inference/.venv` (scorecard.py and api.py are byte-identical to 0.9.8), arcengine `base_game.py`, and docs.arcprize.org/methodology.

- **Level score:** min(115, 100·(baseline/actions)²) (`scorecard.py:166-183`).
- **Game score:** the level-weighted mean of the level scores, capped at the weighted share of cleared levels (`scorecard.py:191-206`).
- **Run score:** the plain mean over games (`scorecard.py:612-616`).
- **A game can have several plays, and the game keeps the max** (`scorecard.py:237-241`). The docstring at line 240 says "average"; the code returns `max`.
- **What counts as an action:**
  - Every ACTION1–7 counts 1.
  - A level RESET counts 1 and does not clear the count (`scorecard.py:701-704`).
  - In competition mode, a RESET with zero actions is intercepted but still counts 1 (`api.py:296-316`).
- **The only full reset in competition mode is a RESET after WIN** (`base_game.py:311-314`). It starts a new play with fresh counters (`scorecard.py:839-843`).
- **No action cap.** An unfinished game scores 0 on its current level, with no penalty for actions spent there.
- **Baselines are hidden from the agent in submission mode** (`taaf/game.py:252-255`). Nothing can depend on knowing the human counts.

**Victory lap evidence** (`scoring/victory_lap.py`, `vl_best.jsonl`, `vl_rest.jsonl`, `e2e_competition.py`):
- All 77 recorded wins in 11 v4 runs replay exactly after the full reset.
- 11 of the 77 wins gain anything, and 7 gain more than 1 point. All gains are on sc25, cd82 and sb26:
  - sc25: 86.31→100, 79.89→100, 88.83→100
  - cd82: 71.36→91.94, 78.65→93.48, 74.49→77.16
  - sb26: 91.95→100
- Gain per run: +0.341 averaged over 11 runs; +0.548 / +1.642 / +0.322 on user-combo-10x v1–v3. v4-control gains 0.00.
- A replay is 161 actions per won game (median), 449 at most.
- End to end against arc_agi 0.9.9's `CompetitionArcadeServer`, run locally:
  - cd82 plays scored 71.22 and 91.94, and the official score was 91.94. The 0.14 below 71.36 is the intercepted warm-up RESET.
  - sc25 (server `ONLY_RESET_LEVELS=true`) plays scored 86.31 and 100, and the official score was 100.

**Why the lap is safe without seeing the gateway.** Suppose play 2 finishes and uses no more actions than play 1 on any level. Then play 2 scores at least play 1. The game score then doesn't drop under max (as coded), mean (as the docstring says) or last-play scoring. Under mean scoring the gain halves. Only one case can lower a score: the replay diverges mid-way on a non-deterministic hidden game, and the gateway averages or keeps the last play. All 25 public games are deterministic.

## 3. The measurement wall

**Noise** (`critique/pw.py`):

| Group | Runs | sd |
|---|---|---|
| Pooled, all groups (alpha-table.md:10) | 11, 7 dof | 2.47 (95% CI 1.63–5.02) |
| Franzen control | 46.25 / 45.92 / 44.10 / 46.38 | 1.06 |
| user-combo-10x (the stack) | 56.94 / 52.70 / 49.09 | 3.93 (95% CI 2.05–24.7) |
| Current stack pooled (10x + 13x) | 5 | 3.34 |

The F test of stack vs Franzen control gives F = 13.8, p = 0.06. The noise of the stack being tested is likely larger than 2.47. Its sd is the biggest unknown in planning, which is why the reference pool comes first.

**Power** (one-sided α 0.05; `final/power2.py`, `critique/pw.py`):

| | sd 2.47 | sd 3.34 | sd 3.93 |
|---|---|---|---|
| Runs per arm to detect +3 (two groups, 80%) | 8.4 | 15.3 | 21.2 |
| 16-run factorial, power for a +3 flag (z / t7) | 0.78 / 0.71 | 0.56 / 0.49 | 0.45 / 0.40 |
| 32-run factorial, power for +3 | 0.96 | 0.81 | 0.70 |
| Adopt rule β > 1.5: P(adopt) for a flag that does nothing / +3 / −1.5 | 0.11 / 0.89 / 0.008 | 0.18 / 0.82 / 0.036 | 0.22 / 0.78 / 0.063 |
| Expected false adoptions among 8 no-effect flags (score only / with token test) | 0.90 / 0.54 | 1.48 / 0.89 | 1.78 / 1.07 |
| An observed +3, shrunk with a 6% prior | +1.65 | +0.73 | +0.51 |
| Stack confirmation, 6 runs vs pool of 4, power for +2 | 0.35 | 0.24 | 0.20 |
| 8 runs vs pool of 8, power for +2 | 0.49 | 0.33 | 0.27 |
| 12 runs vs pool of 10, power for +2 / +3 | 0.60 / 0.88 | 0.40 / 0.67 | 0.32 / 0.55 |
| Sequential rule (looks at 1, 3, 6 runs), P(adopt +3), pool 4 / 10 | 0.71 / 0.82 | – | 0.44 / 0.52 |

**Other metrics are worse:**
- Wins per run are 10 / 7 / 6 (sd 2.1). Detecting +1 win per run takes about 54 runs per arm.
- Score at minute 25 correlates with the final score at only r = 0.12–0.21 (arms.md). Tonight's mem-v2-10x shows this again: it led at 25 min and finished last.
- Panels give the token test an se of 0.50–0.80 per pair, against 0.14–0.16 for full runs. That is 3–10× less information per GPU-hour. The "+3.4 for a no-effect arm" on panels is one arm (macros, n = 1).

**Per-level token test** (stratified Cox on tokens to clear, uncleared levels treated as censored; `cox.py`):
- Over 9 placebo contrasts (about 7 independent), rms z is 1.00. The 95% CI on the dispersion factor is about 0.66–2.03.
- It is more sensitive than score in one case: ledger z +3.08 vs score z +2.42.
- 13 experts: HR 1.60, z 4.42.
- It cannot see throughput, scheduling, prompt-side cost (flags that add prompt text move cost into prefill and earlier trims), or calls made after a clear. So it is a second check that must not contradict the score, never the main endpoint.

**Does throughput turn into score?** Measured within each run, the last 15% of time adds:

| Run | Score at 0.85T | Final | Gain |
|---|---|---|---|
| user-combo-10x v1 | 53.83 | 56.94 | +3.1 |
| v2 | 47.06 | 52.70 | +5.6 |
| v3 | 44.83 | 49.09 | +4.3 |
| 13x | 44.19 | 47.34 | +3.2 |

So extra tokens spent like a run's last minutes are worth up to about 0.25 points per 1% (`final/late_slope.py`). There is no clean A/B test:
- streams14 (+15% tok/s, +0.13) also cut the window to 96k and the drain to 42k.
- accept05 was lossy.
- user-combo's +17% gave +2.3, which is not significant.

Use 0.05–0.25 points per 1% of per-stream decode for lossless levers (a guess). Treat concurrency gains (more streams) as near zero.

## 4. The ranked plan

### 1. Victory lap
- **What:**
  - After a WIN, a harness thread (no LLM, no slot) sends RESET and checks that levels_completed is 0.
  - It then replays, for each level, the shortest route through play 1's observed state graph, built from frame hashes seen through the gateway, and checks every frame.
  - It starts only if the route is shorter than play 1 on at least one level and not longer on any, and if the remaining wall-clock is at least 10× the expected replay time.
  - It stops cleanly on a mismatch.
  - The game is then marked done, and the scheduler never gives it LLM time again.
  - taaf scores max over plays, and keeps a play-1 column so old runs stay comparable.
- **Why:** sections 2 and 3: the rules, 77/77 replays, the end-to-end check, and the safety argument.
- **Expected gain:**
  - +0.34 per run (11-run mean, range 0 to +1.64); +0.84 on the best stack.
  - LB, a guess: +0.05 to +0.3. Hidden runs have fewer wins (LB about 29 vs about 53 offline).
  - Half that if the gateway averages plays.
- **Cost:** about 1 build day. 0 dedicated GPU-h. Seconds of gateway calls per win, no tokens.
- **Confidence:** high on the local scorer. The hidden gateway's version can't be seen, and competition mode blocks reading the scorecard (`api.py:103-107`).
- **First step:**
  - Create branch `arm/victory-lap`.
  - Build the route from gateway frame hashes, not the local engine. That is a different code path from the 77/77 test.
  - Run an in-harness CPU end-to-end test: TAAF offline mode, the real solver, scheduler and threads, and a scripted fake model replaying the recorded wins from v2 (cd82, sc25, sb26).
  - Turn it on in the Oct 12 reference runs. It doesn't change play 1.
- **Kill rule:**
  - **Before shipping**, fix if any of these happens:
    - an LLM request goes to a won game after its RESET;
    - a lap is left unfinished at the end of the run;
    - a route differs from `victory_lap.py`'s on the recorded wins.
  - **On Kaggle**, drop it if any of these happens:
    - play 2 uses more actions than play 1 on any level;
    - a lap fails on a public (deterministic) game;
    - the first lap-on submission errors or times out.

### 2. Final-pick protocol and labelled submissions
- **What:**
  - Submit with `kaggle competitions submit -k mellamoa/v4-<arm> -v N -f submission.parquet -m "<arm> v<N> fp<fingerprint>"`.
  - Submit copies of the base, then copies of the final config.
  - On Nov 2, select two completed copies of the config the pre-registered offline rule favours. Do not use public score, and don't let Kaggle auto-pick.
- **Why:**
  - Kaggle counts the better private score of the 2 selected. This is the standard rule (discussion 432319); the host hasn't confirmed it here.
  - Split-half correlation within a run is −0.45 to −0.48 (`ana/halves.py`; `splithalf_null.py`, P = 0.05; 95% CI about −0.85 to +0.3). So a public score says roughly nothing about the private half.
  - Reruns are free: `kaggle quota` shows 9.97 h used at 22:35 UTC Oct 10. That equals the pushed sessions in the GPU ledger, even though submission 57028045 ran at 02:12 UTC.
  - The Oct 9 (29.75) and Oct 10 (28.86) submissions have empty descriptions.
- **Expected gain:**
  - +0.2 to +1.2 private LB against pairing the best config with a weaker one; probably the low end.
  - Max of two copies is μ + 0.564σ. With σ = 1.3–2 LB per copy from our own data, that is +0.75 to +1.1 over one draw.
- **Cost:** 0 build. 0 GPU-h. One submission a day. The final must be frozen by Oct 27 to get 6+ copies.
- **Confidence:** medium.
- **First step:**
  - Oct 11: make a labelled submission.
  - A Sonnet browser subagent maps 56993850 and 57028045 to kernel versions. It also reads the rules page for "best of selected" and "one account per participant".
  - Record the results in `runs/submissions.csv`.
- **Kill rule:**
  - If only one selected submission counts, choose by offline evidence alone.
  - Leave out any copy that errored or ran short.
  - Use a different config for the second pick only if it beats the final by more than 0.5 LB-equivalent on pre-registered offline evidence.

### 3. Decision rules, reference pool, token test
- **What:**
  - Grow the user-combo-10x reference pool to 6–8 runs by Oct 21. expert-stats-10x counts as one, and lap-on runs count because play 1 is unchanged.
  - Put the power table in arms.md.
  - Add `make hazard` (`cox.py`) as the second endpoint.
  - Pre-register the rules in section 6.
- **Why:** section 3. The stack's sd decides whether anything below +3 can be tested at all.
- **Expected gain:** indirect, from fewer false adoptions and kills. A guess: +0.2 to +0.5.
- **Cost:** 0.5 day of CPU. Each reference run is 2.3 GPU-h and doubles as a lap check.
- **Confidence:** high for the pool and tables; medium for the token test.
- **First step:** Oct 11: expert-stats-10x plus 1 reference. Move `cox.py` and `strat_tok.py` into `analysis/`.
- **Kill rule:** the token test becomes report-only if reference-vs-reference and all-off-vs-pool contrasts give rms z > 1.5 or any |z| > 2.5 over at least 6 contrasts.

### 4. Factorial-1: 16 runs, 8 columns
- **What:** a 2^(8-4) resolution IV design (`final/design2.txt`).
  - It runs in four 2-day periods of 4 runs on mellamoa. The all-on and all-off runs go first, and the all-on run is inspected before the other 14 are pushed. The code fingerprint is frozen for all 16.
  - Factors:
    - A: placebo. memory-v2 left after tonight's 43.81, by the pre-set rule.
    - B: DEBRIEF (#156)
    - C: BEFORE_AFTER (#156)
    - D: Fable facts (#149)
    - E: hypothesis tags (#151). Check that its memory-main patch doesn't switch memory-v2 on.
    - F: sandbox-robust plus the no-op guard (#76). A dummy if its rebase isn't through the suite by Oct 12.
    - G: dyn-temp phase (#154)
    - H: `LOCAL_ANALYZER_MAX_OUTPUT=8192` vs 12288, env only. A dummy if its checks fail.
  - The periods are blocked (AD, AG, DG). Every main effect is orthogonal to the others and to the periods. MEM×DEBRIEF (now placebo×DEBRIEF), MEM×HYP and BA×FABLE can still be estimated.
- **Why:**
  - One flag at a time at equal power needs about 66 runs (`power.py`).
  - stack-v1 changed 3 things in one run, scored −7.5, and the cause was never found (arms.md).
  - Per-flag se is sd/2, whatever the number of factors.
- **Expected gain:** +0.4 to +1.5 on the proxy, a guess.
  - Prior: 4–8% per flag that it is a real +3. arms.md's Franzen-era Didn't-work list has 28 entries against one proven behaviour arm (ledger-on-trim). memory-v2 was the best-evidenced lever and has just lost.
  - That prior times P(adopt | +3) of 0.78–0.89 gives 0.75–1.7, before allowing for flags under +3.
  - LB, a guess: ×0.25–0.56.
- **Cost:**
  - The merge is 2–3 days. `git merge-tree` from origin/main conflicts on fable-facts-main, level-debrief, dyn-temp and sandbox-robust. sandbox-robust is 246 commits behind main.
  - H needs no build.
  - 16 × 2.3 = 36.8 GPU-h, 8 days at 2 runs a day.
- **Confidence:** medium. The design is sound; the prior is low.
- **First step:**
  - Create branch `arm/factorial-1` from user-combo-10x plus the victory lap.
  - Merge hyp-outcome-tags (clean), then the four branches that conflict. Nested flags become one factor.
  - Run the suite at low priority inside the lock.
  - Loop sim: the all-off run's prompts must be byte-identical to user-combo-10x, so the old references join the pool. Measure the all-on run's prompt tokens per request and trims against user-combo-10x.
- **Kill rule:** rules R2–R4 in section 6.

### 5. Rent RTX PRO 6000 pods (if the user approves the money)
- **What:**
  - Run the unchanged notebook on rented pods with at least 128 GB of host RAM. The BF16 PLE table takes about 95.4 GiB of host memory (`--ple-offload-embedding`).
  - Use them to replicate the factorial to 32 runs and give the final stack 12 confirmation runs.
- **Why:** the only lever that adds statistical power before Nov 2 (power table). It also removes any dependence on snowiers.
- **Expected gain:** 0 directly. A guess: +0.3 to +1 through verdicts that can actually be trusted.
- **Cost:**
  - 2–3 build days: the Pennyroyal wheelhouse is built for Kaggle's image, and the weights (119 GB) take about 15 min to pull.
  - About $3–4 per 121-min run (RunPod $1.69–2.1/h; Vast $0.74–1.34/h; listing sites, approximate). About $140 for 16 replicates + 12 confirmations + 2 calibration runs + storage.
- **Confidence:** medium. Makarov's "reproduces almost one-to-one" (discussion 743723) is about his vLLM runtime, not our SGLang sm120 build with MTP and FR-Spec. Host RAM and PCIe bandwidth also vary by provider, and the PLE table is gathered from host memory every token.
- **First step:** ask the user for a budget. Then run one pod's decode bench against Kaggle's 895–909 tok/s at full batch, then 2 reference runs.
- **Kill rule:** stop if decode is more than 5% off Kaggle. Pod runs always enter fits with a platform term, never pooled raw.

### 6. Rebuild the FR-Spec draft-token map (lossless)
- **What:** keep the 64k size, and swap the 1,679 entries our transcripts never use for the generated tokens the map lacks ("Hmm" variants alone are 0.49% of tokens).
- **Why:**
  - The current map covers 98.92–98.94% of generated tokens.
  - A map built from v1 covers 99.908% of v2, which it never saw (`model-side/hotcov.py`, `hotcov2.py`).
  - Turning FR-Spec off cost 5.4% decode (arms.md:273, tput-levers).
- **Expected gain:** 0 to +0.25, a guess: about +1% acceptance at 0.05–0.25 per 1%.
- **Cost:** 0.5 day. It needs a new dataset version and a change to the launcher's pinned `TOKEN_MAP_SHA`. One fixed-prompt A/B serving bench (0.75 GPU-h).
- **Confidence:** medium.
- **First step:** build the map from all 14 runs, holding out v3 for coverage. Bench both maps on the same replayed requests at batch size 14.
- **Kill rule:**
  - Revert if the bench shows under +0.5% accept length, or any drop in decode tok/s.
  - Don't judge across runs from serve.log: identical runs already differ by 0.9% (2.678 / 2.674 / 2.653).
  - It enters at S1 (Oct 21), never mid-factorial.

### 7. Checkpoint-branch bench at decisive levels (new; not on the critical path)
- **What:**
  - Rebuild a logged run's game state (replaying its actions) and its conversation (from `*_requests.jsonl`) at the start of a level that clears about half the time:
    - tn36 L2 7/14
    - cn04 L2 7/14
    - m0r0 L6 5/10
    - re86 L6 8/14
    - sc25 L6 5/9
    - vc33 L7 4/8
  - Run arm and control branches from the identical state for 25 minutes each.
- **Why:** 44.9 of the 47.1 missing points are uncleared levels, and 95% of clears happen within 26 active minutes. Branching where the points are decided should take far fewer GPU-hours than full runs.
- **Expected gain:** 0 directly. A guess: it detects a +0.15 to +0.2 change in clear rate on those levels (about +1 to +2 points) in 4–6 sessions (3–4.5 GPU-h). It cannot see throughput or scheduling, and clear rates are inflated without contention.
- **Cost:** 2–3 build days. A CPU Stage-0 check of about 1 h.
- **Confidence:** low to medium.
- **First step:** Stage-0 on CPU: the rebuilt frame and prompt must match the logged ones byte for byte.
- **Kill rule:** drop it if the rebuild doesn't match, or if two control-vs-control sessions differ beyond binomial noise.

### 8. Stuck-level restart at 40 active minutes (trigger moved from 25)
- **What:** once per level, cut the conversation back to the opener plus harness facts: the ledger, the refuted hypotheses and the current frame. The board is not reset.
- **Why the trigger moved:**
  - At 25 minutes the 10-minute clear chance is still 0.45, the same as at 20.
  - At the 25-minute trigger, 78 usually-cleared levels went on to clear anyway, against 73 that never did (`critique/h25.py`). That fails the draft's own build rule.
  - At 40 minutes the chance is 0.10 (n = 39). Across all levels, 60 attempts reach 40 minutes (4.3 per run): 10 cleared later and 50 never did.
- **Expected gain:** 0 to +0.5, a guess. Little time may be left after 40 active minutes.
- **Cost:** 1–2 build days. After the freeze.
- **Confidence:** low.
- **First step:** CPU replay: how much slot time do those games have left after minute 40? The hidden ratio is about 68 slot-minutes per game.
- **Kill rule:** don't build it if the median at-risk level has under 10 minutes of slot time left.

### 9. Level-start novelty facts (CPU gate only)
- **What:** at each level start, the harness diffs colour-component classes against the previous level's start. It states which classes are new, until each has been probed once.
- **Why:**
  - Walls hold 18.4 points per run.
  - The near-miss report names unprobed new objects in ft09, vc33, m0r0, su15 and ka59.
  - The planned vehicle, the entity tracker, failed its Stage-0 gate (`ef25f59d`): lexicon +3 F1 against a +10 gate, MobileSAM 40–60× over the latency budget, player identification 48% against 80%.
- **Expected gain:** +0.5 to +1.5, a guess.
- **Cost:** 2–3 build days. After the freeze.
- **Confidence:** low.
- **First step:** a CPU gate on logged frames.
- **Kill rule:** ship it only if the line fires on a wall level's first frame in more than 50% of losing runs, and on fewer than 10% of all turns.

### 10. Fine-tune the MTP draft head (lossless; pods only, deferred)
- **What:** self-distil the dense parts of the albucino MTP layer on our own transcripts.
- **Why:**
  - Red Hat's fine-tune of Qwen3-Next's draft head raised per-position acceptance from 0.897/0.719/0.476 to 0.912/0.776/0.616 (developers.redhat.com, 2026-09-08).
  - Our accept05 panel showed acceptance 2.64→3.06 gave +15.2% decode.
- **Expected gain:** +0.3 to +1.5 if decode rises about 6%, a guess. It depends on the unconfirmed decode-to-score rate.
- **Cost:**
  - 4–7 build days.
  - 3–5 GPU-h per training attempt, which can't share a 6-hour day with 2 full runs.
  - Hidden states are about 20 KB per token, about 20 GB per 1M tokens, so the dump and the training must run in one session.
- **Confidence:** low to medium.
- **First step:** only if pods are approved: a teacher-forced acceptance baseline on 200k held-out tokens.
- **Kill rule:** stop if it gains under +3% teacher-forced acceptance by Oct 22, or isn't booting in Pennyroyal by Oct 25.

## 5. Stop doing

- **Single runs and 25-min panels as verdicts.**
  - mem-v2-10x led at 25 min (11.74 vs 8.80) and finished at 43.81, the lowest of the stack.
  - Panels carry 3–10× less information per GPU-hour.
  - Cancel the queued panel-debrief, panel-dyntemp, panel-dyntemp-split and panel-scorer.
- **Single-flag full runs from separate commits** (fable-facts-10x and mem-solved-10x are queued). They can't enter the factorial fit. Spend those slots on reference runs.
- **Testing one flag at a time:** about 66 runs against 16 for 8 flags.
- **Action-efficiency arms:**
  - The whole ceiling is 2.17 points per run, and only 0.56 of it is above 1.5× the human count.
  - level-path scored 48.31, macros had 0 invocations, and trajectory picking is the same kind of arm.
  - They can come back only through more levels cleared.
- **Serving sweeps on streams, experts or acceptance:**
  - 11 experts: 47.19.
  - 13 experts: 49.61 and 47.34.
  - 18 streams with a 64k window: 36.01.
  - 16 streams with an 80k window: out of memory.
  - accept 0.5: 46.77.
  - uc-11x-16s: at most once, under the sequential rule.
- **Scheduler arms.** Freeze Franzen's scheduler. sim-v2 agrees with replays at only Spearman +0.53. D′ scored −2.9. final-level-hold is −0.32 at the hidden ratio. Use sim-v2 only as a veto.
- **Deciding on wins, full clears or early cuts.** +1 win per run needs about 54 runs per arm. Minute 25 correlates with the final at only r = 0.12–0.21.
- **Don't start:**
  - global `reasoning_effort=low` (the 13-expert natural experiment says it's a wash);
  - expert pruning;
  - shared sampling seeds (they need deterministic mode, which turns off the radix cache);
  - dropping the previous level's thinking;
  - main-model LoRA or RL before Nov 2, unless funded separately. Public results on Flash-Next are negative (discussions 742835, 743319).
- **Unlabelled submissions.**
- **Counting on snowiers in the schedule** (open question 1).

## 6. Test method (pre-registered)

- **R0, tonight's mem-v2-10x** (written before reading the score, `final/R0-mem-v2-rule.txt`):
  - One run never moves a flag into the base.
  - memory-v2 stays factor A unless the run crashed or ran out of memory, or scored below 44 (1.2% chance at sd 3.93 if it has no effect).
  - **It scored 43.81, so A becomes a placebo.**
  - memory-v2's own look is a CPU diagnosis before any more GPU: prompt tokens per request, trims per run, card size, and the token test against the pool. Its 19.6 levels per 1M tokens is the lowest of the stack.
- **R1, all-off identity:** the all-off config's prompts must be byte-identical to user-combo-10x in the loop sim. If they are, all user-combo-10x references join the pool.
- **R2, all-on inspection** (a full 121-min run, not a 15-min smoke). Fix or drop a flag, then re-run all-on, if:
  - it fired 0 times;
  - a crash is traced to it;
  - prompt tokens per request rose more than 10%, or trims per run more than 15%, against user-combo-10x;
  - H is cut on more than 5% of replies, or fewer than 80% of the replies after a cut act. A full run has about 80 such events.

  Also, the victory lap must have fired on every win. After the all-on inspection, no code changes are allowed. A failed run is re-run on the same fingerprint. One missing run costs ×1.12 on the se.
- **R3, adoption:** adopt a flag only if all of these hold:
  - its score effect β is above +1.5;
  - the token test's logHR is at least 0;
  - for flags that add prompt text, prompt tokens per request rose no more than 10% and trims per run no more than 15%.

  The expected errors are in the power table. The report shows both raw and shrunk effects (a 6% prior of +3).
- **R4, interactions:** if AB, AE or CD is below −3, don't adopt both of its flags. Keep the one with the larger main effect. Stop the design if the all-off run lands outside 44–62 (2.2% chance with no change).
- **R5, freeze on Oct 27:**
  - The final is S1 if S1 minus the pool is above 0 with P ≥ 0.8 (one-sided t, pooled sd) and the token test is at least 0.
  - Otherwise the final is user-combo-10x + lap (+ FR-Spec map if its bench passed).
  - Futility: stop S1 early if it trails the pool by more than 1 after 3 runs.
- **R6, Nov 2:** select two completed copies of the R5 config. Re-check R5 with all runs through Oct 31. Switch config only if the sign has reversed.

**Factorial table** (`final/design2.txt`):

| Run | Period | Flags on |
|---|---|---|
| 1 | P1 | all (ABCDEFGH): inspect first |
| 2 | P1 | none (user-combo-10x + lap) |
| 3 | P1 | BCFH |
| 4 | P1 | ADEG |
| 5 | P2 | BEGH |
| 6 | P2 | CEFG |
| 7 | P2 | ABDH |
| 8 | P2 | ACDF |
| 9 | P3 | AFGH |
| 10 | P3 | ABCG |
| 11 | P3 | DEFH |
| 12 | P3 | BCDE |
| 13 | P4 | ABEF |
| 14 | P4 | ACEH |
| 15 | P4 | BDFG |
| 16 | P4 | CDGH |

## 7. GPU schedule (Kaggle quota weeks run Saturday 00:00 UTC to Saturday)

Caps: mellamoa has 45 h a week and the user's cap of 6 h a day. That allows 2 full runs (4.6 h) plus 1.4 h of benches a day. A full run is 2.3 GPU-h. `kaggle quota` at 22:35 UTC Oct 10 shows 35.03 h left until Oct 17 00:00. Submissions are 1 a day, and reruns are free. snowiers is not used.

**Week A (Oct 11–16, 35.0 h available)**

| Day | GPU | Build / CPU | Submission |
|---|---|---|---|
| Sun Oct 11 | expert-stats-10x (reference 4) + user-combo-10x reference 5 (4.6 h) | Victory lap and its in-harness test; start the factorial merge; `make hazard`; FR-Spec map coverage; memory-v2 CPU diagnosis | user-combo-10x v1, labelled (unless the Oct 9/10 ones already were it) |
| Mon Oct 12 | 2 lap-on references (4.6 h) + FR-Spec fixed-prompt bench (0.75 h) | Finish the merge; suite; loop sim (R1, prompt length) | Lap-on reference: checks the lap on 110 hidden games |
| Tue Oct 13 | Factorial 1 (all-on) + 2 (all-off) (4.6 h) | Inspect all-on (R2) overnight; freeze the fingerprint | Factorial all-on: a 9-h, 110-game check of every flag |
| Wed–Fri Oct 14–16 | Factorial 3–8 (13.8 h) | Optional: checkpoint-bench Stage-0; stuck-restart replay | Base + lap copies, labelled |

Week A total is 28.35 h, leaving 6.7 h of slack (about 2 failed runs). If the merge slips a day, Oct 13 runs 2 more references and everything shifts by a day.

**Week B (Oct 17–23, up to 42 h)**

| Day | GPU | Notes |
|---|---|---|
| Oct 17–20 | Factorial 9–16 (18.4 h) | – |
| Oct 21 | Fit (R3, R4). Build S1 = base + lap + adopted flags + FR-Spec map. S1 run 1 + 1 reference (4.6 h) | From here on, every run, references included, has the map |
| Oct 22–23 | S1 runs 2–5 (9.2 h) | Submit S1 daily, labelled |

Week B total is 32.2 h, leaving about 10 h of slack. If no flag is adopted, S1 = base + lap + map, and the same slots go to repeats and references.

**Week C (Oct 24–30, up to 42 h)**

| Day | GPU | Notes |
|---|---|---|
| Oct 24–26 | S1 runs 6–8 + 3 references (13.8 h) | – |
| Oct 27 | Freeze by R5 | From here on, submit the final daily (Oct 27 – Nov 2, 7 copies) |
| Oct 27–30 | Alternate the final and references (8 runs, 18.4 h) | Serves the user's repeat rule and R6 |

**Nov 2:** R6. Select two copies.

In total that is about 40 full runs and 92 GPU-h on mellamoa, inside the caps.

**If something slips**, drop in this order:
1. Week C's extra references.
2. S1 runs beyond 6.
3. The factorial start, by a day.

The Oct 27 freeze does not move. If S1 has fewer than 4 runs by Oct 26, the final is base + lap.

**With pods:** port Oct 12–14, calibrate Oct 15, replicate the factorial on 6–8 pods Oct 16–18, and fit 32 runs on Oct 21 (with a platform term). Then run S1 12 times on pods by Oct 25.

## 8. Open questions

1. **Is snowiers the user's own second account?** The memory file calls it "the user's second account", and user-combo-10x v2 and v3 ran on it.
   - Kaggle's standard competition rules allow one account per participant. I have not read this competition's rules page. That risk falls on mellamoa and its submissions.
   - If it is a teammate's account, merge teams before the merger deadline.
   - The plan doesn't use it either way.
2. **Will the user spend about $140 on pods?** This is the only way to add power before Nov 2.
3. **Which configs produced submissions 56993850 (29.75) and 57028045 (28.86)?**
4. **The hidden gateway's play scoring (max, mean or last) can't be seen.** The lap is safe under all three if play 2 finishes, and the gain halves under mean. Is that acceptable without proof?
5. **Does the user still want uc-11x-16s?** 11 experts showed no per-token gain, and 16 streams are expected at about 850 against about 900 tok/s.
6. **Will the user accept R5 (P ≥ 0.8, plus the token test) as "repeated runs consistently higher"?** At sd 3.3–3.9 only effects of about +4 can meet a stricter bar by Nov 2.
7. **Should main-model fine-tuning run as a separately funded long shot?** About $1–3k and 7–12 build days, with negative public evidence on Flash-Next. It is the only lever whose ceiling matches the leaders (LB 56–59; at our 0.56 ratio that implies more than 100 offline).
8. **Should progress toward 12 wins be tracked through mean score?** Wins per run are too noisy to compare arms.
9. **The mem-v2-10x entry in arms.md needs its result (43.81) recorded by whoever owns that run.** This report does not edit other entries.

## 9. What the review changed

- **Noise:** sd 2.47 was replaced by a range of 2.47 / 3.34 / 3.93, and all power figures were recomputed. Detecting +3 takes 8–21 runs per arm, not 6.
- **Factorial expected gain:** +1 to +2.5 became +0.4 to +1.5 (base rate). Error rates and shrinkage for the adopt rule are now stated.
- **Victory lap:**
  - Gating added: play 2 must finish and be no worse on any level, or not start.
  - The route is built from gateway hashes, with an in-harness test.
  - The kill rules can now fire.
  - The safety argument now covers max, mean and last-play scoring.
  - The LB guess fell to +0.05 to +0.3.
  - The win count is 11 of 77 that gain anything (7 by more than 1 point), not 8.
- **Final pick:**
  - The gain is measured against best + weaker (+0.2 to +1.2), not against one counted draw.
  - Quota use was verified as zero.
  - The split-half CI is stated.
- **Tail cap:** the split-request build was replaced by the env-only output cap, which needs no build. Its value was cut to 0 to +0.5, because freed tokens go back to the same stuck level. The cap also frees 4k of context budget, because the reply reserve shrinks; that is part of the factor.
- **FR-Spec:** checked by a fixed-prompt bench and entered at S1, not judged across runs from serve.log.
- **Stuck restart:** the trigger moved from 25 to 40 active minutes, because the draft's own rule fails at 25.
- **Novelty facts:** CPU gate only. The entity tracker failed Stage-0.
- **Factorial-2:** dropped. With 8 runs, power for +3 is about 0.33–0.53, and it would have finished the day before the freeze. Its runs go to S1 and references.
- **MTP head:** moved to pods only.
- **Pods:** ranked above the MTP head.
- **The 15-min smoke** was replaced by inspecting a full all-on run.
- **The schedule now:**
  - follows Kaggle's Saturday quota week;
  - starts the factorial on Oct 13 at the earliest;
  - has slack and a drop order;
  - doesn't use snowiers.
- **Corrected facts:**
  - 7.7 wins per run, not 6.
  - "+3.4 on panels" is n = 1.
  - Panel information is 3–10× lower, not 3–8×.
  - mem-v2-10x is folded in (43.81).

## Scripts and outputs

Copied to [`../data/next-moves/`](../data/next-moves/) from the session scratchpad. They read run data from `runs/kaggle/` and the session's `runs-out/`. Paths inside them point at the scratchpad.

| Folder | Contents |
|---|---|
| top level | `load.py`, `variance.py`, `perm.py`, `splithalf_null.py`, `metrics.py`, `cox.py`, `cox_panels.py`, `levelvar.py`, `paired_lt.py`, `power.py`, `design.py`, `design.txt` (method lens) |
| `where-lost/` | Decomposition, walls, hazard, clock and slot tables, stratified token tests (losses lens) |
| `scoring/` | `victory_lap.py`, `e2e_competition.py`, `vl_*.jsonl`, `reset_cost.py`, `decomp.py` (scoring lens) |
| `model-side/` | Thinking-length, cap, prompt-composition and FR-Spec coverage scripts (model lens) |
| `ana/` | `halves.py`, `stuck_rate.py`, `lvl_tokens.py` (untried-ideas lens) |
| `critique/` | `pw.py`, `h25.py` (the first critique's checks) |
| `final/` | `power2.py`/`.txt`, `design2.py`/`.txt`, `late_slope.py`, `run_summary.py`, `R0-mem-v2-rule.txt` (this revision) |
