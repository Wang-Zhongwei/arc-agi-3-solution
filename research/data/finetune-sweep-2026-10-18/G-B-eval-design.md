# G-B: Evaluation protocol for fine-tuned Flash-Next checkpoints (G8, G22)

Researcher G-B, 2026-10-18. Local analysis plus a few primary web reads. Paths without a prefix are relative to `daniel-franzen-sol/`. Sweep notes are cited by short name (L1 = `research/data/finetune-sweep-2026-10-18/L1-data-inventory.md`, and so on), with fact-check corrections applied.

**Scripts** (stdlib or the repo's `.venv` numpy, all in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/ce155f4d-514a-4385-ac65-a7e7e5a23e1c/scratchpad/gb/`):
- `g8.py`: subset noise over L3's `games.csv`, 26 runs, scores recomputed with the official formula.
- `pick.py`: held-out selection.
- `q2.py` and `h10.py`: training value, read from `decisions2_joined.json` (the 85-run corpus).
- `q5.py`: screen back-test.
- `picks.py`: final-pick simulation and power.
- `ext_games.py` and `ext2.py`: external-game scan.

## Q1 (G8). How noisy are 10-game subsets, and which 10 games should be held out?

### Takeaway
- **Noise.** A 10-game subset's per-run mean has sd **5.8** (the median over all 3,268,760 subsets), against 3.28 for all 25 games. One paired difference therefore has sd ≈ **8.2**.
- **Proposed held-out set, H10:**
  - always won: **lp85, tr87**
  - sometimes won: **cn04, sc25, vc33**
  - never won: **dc22, ka59, lf52, re86, wa30**
- **Power, scored against the 26-run pool** (one-sided α 0.05, 80% power): +10 offline needs about 3 candidate runs, +5 about 13.
- **A trap.** H10 must not be scored inside a run that also contains the 15 trained games. The scheduler hands their freed compute to the other games.

### Cited Findings
- **Baseline.** 26 runs: mean 51.17, run sd 3.28 across configs, within-replicate sd 3.15 (df 4). This reproduces `L3-headroom.md:229-230` (`g8.py`).
- **All 10-game subsets.**
  - Run sd across configs: p5 4.24, p25 5.15, **median 5.80**, p75 6.44, p95 7.44.
  - Within-replicate pooled sd: median 5.43 (df 4; p5–p95 2.50–8.94).
  - Correlation with the 25-game mean: median 0.57.
  - Paired sd: √2 × 5.80 = **8.20**. The iid per-game formula gives 27.2/√10 = 8.61. Both agree with the critic's "about 7–9" (`_critic.md:109`).
- **Strata** (win rate over the 26 runs):
  - Always (≥0.8): lp85 1.00, ar25 .96, sb26 .96, ft09 .88, cd82 .88, tr87 .85.
  - Sometimes: r11l .62, vc33 .46, sc25 .42, tu93 .19, tn36 .15, m0r0 .12, cn04 .12.
  - Never (<0.1): su15 .04, plus 11 games at 0.
  - A proportional 10-game set is therefore 2 / 3 / 5.
- **Selection** (`pick.py`):
  - 415,800 sets meet the 2/3/5 quota.
  - 22,239 also match the pool: mean within ±1 of 51.17, win rate within ±0.03 of 0.306, and an action mix of 4–6 keyboard+click, 1–2 keyboard and 3–4 click (tags from the public `metadata.json` files).
  - 5,721 also hold 15–19 of the 42 conditional frontier cells (Q2) and at least 3 of the 6 branch-bench games (`arms.md:575`).
  - Tie-break: correlation with the other 15 games' mean. H10 ranks 2nd. The 1st-ranked set swaps wa30 for bp35 and holds 3 fewer frontier cells.
- **H10's statistics.**
  - Per-run mean 51.84 (range 43.30–63.34); sd 5.13 across configs, 3.98 within replicates; all-pairs difference sd 7.25.
  - Win rate 0.285: 2.85 wins per run, against 7.65 over 25 games. Correlation with the other 15 games is 0.31.
  - Action modes: 5 keyboard+click, 3 click, 2 keyboard.
  - It holds 5 of the 11 walls (dc22 L5, ka59 L7, lf52 L5, re86 L8, wa30 L5; `L3:102`) and 4 branch-bench levels (cn04 L2, re86 L6, sc25 L6, vc33 L7).
  - The leader `v4-mem-repair-10x/v1` scores **63.34** on H10, the highest of the 26 runs (`h10.py`). Compare against the pool, not the leader run.
- **Power** (`picks.py`), using a planning sd of 5.8 rather than H10's in-sample 5.13:

  | Offline effect | Runs per arm, two arms | Candidate runs vs the 26-run pool |
  |---|---|---|
  | +5 | 16.6 | 12.2 |
  | +10 | 4.2 | 2.3 |

  A 25-game panel at sd 3.3 needs 5.4 per arm (3.0 vs the pool) for +5, and 1.3 (0.7) for +10.
- **Compute coupling.**
  - "If some of the games finish early, their remaining budget is automatically redistributed to other games" (`WRITEUP.md:125`).
  - L3's simulator puts +33% decode speed at 50.93 → 58.49 on 25 games (`L3:116-118`).

### Inferences
- **How big the contamination could be.** The 15 training games (T15) hold 60% of cleared-level tokens (105.4M of 176.7M).
  - A 25% token cut on T15 frees about 15% of run compute.
  - That is about +37% compute for H10, worth roughly **+5–8 on H10 with zero transfer**, the same size as the effect we want to detect.
  - Per-game caps exist (`max_runtime_s_per_game`, `max_generated_tokens_per_game`, `solver.py:1998,2004`), but they change both arms' policy.
- **Recommended eval run, "U25":** H10 plus 15 never-trained external games, in the standard 121-min, 25-game layout.
  - Every game is unseen, as on the leaderboard.
  - It keeps the 25-game structure. Its sd is unknown until base runs exist, probably about 3.3–4.5.
  - It doubles as G7's proxy test.
  - The 26-run pool does not apply to it, so it needs at least 3 fresh base runs.
  - The pool comparison for H10 stays valid only for harness-only arms.

### Gaps
- U25's sd is unmeasured until base runs exist.
- The within-replicate sd has df 4.
- The 26 configs differ slightly; the true config sd on the 25-game score is about 0.9.

## Q2. What do the other 15 public games add to training, and should we train on public games at all?

### Takeaway
T15 holds most of the immediately usable data:
- 93 of 151 distinct cleared levels, 3,616 of 6,034 clears and 230 of 354 wins (9 of the 14 winnable games);
- 24 of 42 conditional frontier cells (39 of 62 unconditional) and 77 of 123 stuck-vs-cleared pairs;
- 105.4M of 176.7M cleared-level tokens.

"Never train on public" leaves no logged training data at all, because the external games have never been run. **Recommendation: the 15/10 split, with U25 plus the leaderboard as the only valid evaluation for trained checkpoints.**

### Cited Findings
- The corpus reproduces L1 exactly: 85 runs, 1,642 game-runs, 6,034 clears, 354 wins, 151 distinct levels and 123 pairs (`L1:63-65,186`; `q2.py`).
- **Frontier cells:**
  - conditional: clear rate given the level was reached, in [0.2, 0.8], with at least 5 reaches → 42;
  - unconditional: clears per game-run in [0.2, 0.8] → 62.
- **What H10 takes out of training** (`h10.py`): 58 distinct levels, 2,418 clears, 124 wins from 5 games, frontier 18/42 (23/62), 46 pairs, 71.3M tokens and 22,076 decision steps.
- **The two positions.**
  - rsi-proposals: "Never train on the 25 public games. They are the eval set" (`rsi-proposals-2026-10-15.md:44`).
  - Prior plan: hold out 10 at 2/4/4 (`finetune-plan-2026-10-17.md:178`).
- **The external pool has never run in our harness** (`L1:230`). Kaggle produces about 12 game-runs per GPU-h (`_critic.md:222`).
- **Contamination guard.** arc-interactive's `vc33/9851e02b` is another version of the held-out vc33, so exclude it from training (`W9:302`).

### Inferences
- **What 15/10 costs evaluation.** For trained checkpoints, these stop being valid: the 25-game score, the 10-win target, the 26-run reference and the 0.548 LB/offline calibration (`L3:137`).
- **What "never train on public" costs.** Re-creating T15's roughly 980 game-runs on external games would take about 80 GPU-h, roughly half of the Kaggle quota left. That is not affordable before 11-02.
- **H10 tests transfer across games, not out-of-distribution.** The public set is "intentionally easier" (`L3:38`), so external games and the leaderboard remain necessary.
- **Don't refit on all 25 games before 11-02.** Doing so removes the only in-distribution check.

### Gaps
- How frontier cells convert into training benefit is unmeasured.

## Q3. Which external games can serve as validation, and what does a panel cost?

### Takeaway
- **DreamTeam:** all 25 qualify (8 levels each, 64×64 render, embedded solvers, Apache-2.0).
- **arc-interactive:** mostly small logic puzzles with placeholder baselines. Only about 8 games with 6 or more levels have a logical grid of 16 or larger.
- **Use:** 15 DreamTeam games go into U25, and the other 10 DreamTeam games are kept as a single-use "fresh" panel for the final confirmation. arc-interactive goes to the training pool.
- **Cost:** about 2.3 Kaggle GPU-h per panel, or about $13 on Azure on-demand and about $2.5 on spot.

### Cited Findings
- **DreamTeam.**
  - Each `metadata.json` has 8 baseline entries.
  - Per game, the median is 145 actions (53–345), against 638 for the public games (`W9:68-73`). These are solver lengths: 20 of 25 match exactly (`W9:339-345`).
  - `GRID_SIZE` is 16, 32 or 64 in 11 games, and each game renders a 64×64 frame (`competitors/dream-team/.../al7306.py:22,732`).
  - Action modes: 14 click, 8 keyboard, 3 mixed (`W9:75`).
  - The games load via `arcengine_adapter` (`dream-team/README.md:207`).
  - Report levels on these games, not RHAE (`W9:305`).
- **arc-interactive.**
  - `GAMES.md` has 249 rows. Levels: 5 in 192 rows, 7 in 52, 6 in 2.
  - Only 27 of the 252 `baseline_actions` arrays have 6 or more entries, and they look like placeholders (e.g. ab01 `[1,2,3,4,6]`).
  - 54 rows list 6 or more levels. Their grids: 8×8 ×14, 12×12 ×13, 10×10 ×11, 64×64 ×3, 16×16 ×3, 32×32 ×2.
  - Games with 6+ levels and a grid of 16 or more: bn04, ll01, mm01, mm04, mm05, pu04, sf04 and wm04 (`ext_games.py`, `ext2.py`). The wm family references ls20 (`W9:81`).
- **Prices.** A 25-game panel is about 2.3 GPU-h (`L1:344`). Azure NC144lds costs $5.50/h on-demand and $1.02/h on spot (`W6:207,279`).

### Inferences
- **Pre-registered DreamTeam selection.** Run all 25 once with the base model (2.3 GPU-h; this closes G7). Then:
  - put into U25 the 15 games with the most cells whose clear rate is in 0.2–0.8, keeping all 3 mixed-mode games;
  - keep the other 10 for one final confirmation, which guards against overfitting a reused validation set.
- **If DreamTeam saturates or floors**, fall back to H10 plus the leaderboard.
- **Keep DreamTeam out of training.**

### Gaps
- arc-interactive level counts need each game loaded (UNVERIFIED; I did not execute their code).
- Flash-Next's clear rates on these games are unknown.

## Q4. How should the ~15 leaderboard draws be spent, and when do we adopt a fine-tuned model?

### Takeaway
- **Slots.** 16 mellamoa slots run from 10-18 to 11-02 if today's is unused.
- **Allocation:**

  | Use | Draws |
  |---|---|
  | Fallback anchors | 3 |
  | Zero-delta FT-path draw | 1 |
  | Base model under the FT's harness settings | 1 |
  | Best harness-only candidate | 1 |
  | FT checkpoints (3 or more per finalist, at most 2 checkpoints) | 7 (one is a reserve) |
  | Top-ups | 2 |
  | Final day | 1 |

- **Pick rule** (the private score is the best of the 2 picks): let Δ* = 0.6 × (FT draw mean − fallback draw mean).
  - Pick both FT draws if Δ* ≥ +0.5.
  - Pick one FT and one fallback draw if Δ* is within ±0.5.
  - Pick both fallback draws if Δ* ≤ −0.5, or if any FT draw errored.

### Cited Findings
- **Rules:** 1 submission a day, 2 final selections, 2026-11-02 23:59 UTC; entry deadline 10-26 (`W5b:113-114,147`).
- **Best-of-two scoring:** this is Kaggle's usual rule ([DataCamp](https://campus.datacamp.com/courses/winning-a-kaggle-competition-in-python/modeling?ex=14); [DrivenData forum](https://community.drivendata.org/t/setting-final-submissions/5150)). This competition's own text was not read (UNVERIFIED).
- **Our submission list.** The `kaggle competitions submissions` CLI (2026-10-18, mellamoa) shows the last submission at 10-17 02:39 UTC and no snowiers rows, so snowiers is a separate leaderboard entry.
- **Noise.**
  - Per-draw sd is 3.50 (df 7) (`L3:295`). The family's 19 draws average 29.12.
  - mem-repair-10x drew 38.22, 28.73, 29.54 and 26.90 (`runs/submissions.csv`).
- **Don't pick by public score.** The within-run split-half correlation is −0.45 to −0.48, with a 95% CI of about −0.85 to +0.3 (`research/archive/reports/next-moves.md:195`). Hence "select ... never by public score" (`arms.md:569`).
- **Draws needed** (`picks.py`): +5 LB needs 6.1 draws per arm; +8 needs 1.5–2.4; +10 needs about 1.
- **Pick simulation** (σ = 3.5; 3 FT draws vs 6 fallback draws, so the posterior sd is 2.47). Expected best-of-two score relative to the fallback mean:

  | Posterior Δ | Both fallback | One each | Both FT |
  |---|---|---|---|
  | −1 | +1.98 | +1.75 | +0.96 |
  | 0 | +1.97 | +2.20 | +1.95 |
  | +1 | +1.96 | +2.71 | +2.93 |

### Calendar (UTC)
| Date | Draw |
|---|---|
| 10-18, 10-19 | Fallback anchor: the frozen `mem-repair-10x` v1 (brings it to 6 draws) |
| 10-20 | Best harness-only candidate |
| 10-21 | Zero-delta draw: the FT loading path with an identity adapter (tests the 9-h rerun path) |
| 10-22 | Base model under the FT's harness settings, e.g. `ARC3_VERBATIM_HISTORY=1` (`_critic.md` C8). This is the FT's true comparator |
| 10-23 | Fallback anchor, or a second draw of 10-22's config |
| 10-24 → 10-26 | FT1 ×3, only after gates G0–G3 (see Implications) |
| 10-27 | Reserve: rerun an errored draw, or FT1 #4 |
| 10-28 → 10-30 | FT2 ×3 if it beats FT1 offline; otherwise more FT1 or fallback draws |
| 10-31, 11-01 | Top-ups for whichever config sits near a threshold |
| 11-02 | By about 12:00 UTC, one more copy of the leading config; finals selected before 23:59 |

If FT1 isn't ready, its slots go to the harness prong. Never leave a slot unused, since leaderboard draws cost no GPU quota (`arms.md:569`).

### Inferences
- **The leaderboard can confirm only one candidate,** and only for effects of about +5 LB or more (roughly ≥ +9 offline). Choose the checkpoint offline.
- **The 0.6 shrinkage factor** comes from a skeptical N(0, 3²) prior, chosen because no public evidence shows fine-tuning helping (`W5b`).
- **Claiming an alpha** (the user's rule) needs a stricter test: every FT draw above the fallback mean and Δ ≥ 2·SE.

### Gaps
- Whether a still-running submission can be selected as a final pick.
- Whether snowiers is on the team.
- The size of the private half (critic G18).

## Q5. Did any screen-like metric in our logs predict the full-run ranking?

### Takeaway
**Mostly, the data can't answer this.**
- **Within the 10x family**, a single run's score has reliability **0.08** (variance across configs 10.76 vs within 9.92), which caps any screen's correlation with it at about 0.28.
- **Across a wider set of 48 runs** (reliability 0.62), no log metric predicts score in the direction a token-cut screen assumes.
- **The state-lab right-move rate can't be back-tested at all.**

### Cited Findings
(`q5.py`; Spearman ρ with bootstrap 95% CI. Wide set = 48 runs of 25 games scoring ≥30, from `runs/index.csv`.)

| Metric | 10x family | Wide set |
|---|---|---|
| L1 tokens-to-clear (log, normalised per game) | +0.33 [−0.10, +0.65] | **+0.34 [+0.07, +0.57]** |
| Tokens per decision | +0.06 | +0.22 [−0.04, +0.46] |
| No-tool-call rate | −0.14 | +0.12 |
| L1 actions-to-clear | +0.03 | +0.06 |

- **Partial runs:** the score at minute 25 correlates with the final ranking at r = 0.21, rising to 0.63 at minute 60, 0.77 at 90 and 0.90 at 110 (`research/archive/reports/score-timeline.md:13`).
- **The 27B:** same actions per level but 2.06× tokens per action, and it scored 14.51 offline (`L3:132,294`). That is one contrast where tokens at matched actions pointed the right way (n = 1).
- **State lab:** rounds 2–3 used hand-checked stand-in facts at single states, and none was run as a full arm (`arms.md:313-322`).

### Inferences
- **Use screens as regression gates, not predictors.** The positive sign on tokens means past token variation came with harness capability, so a token cut must be judged at matched decisions.

### Gaps
- Distinct actions at stuck states were not computed; that needs the 3,022 event files.

## Q6 (G22). What general-ability / forgetting canary can run in 30–60 minutes?

### Takeaway
About 50 minutes on the same SGLang server, with the base model measured in the same session.

### Cited Findings
- **Base finish reasons:** 139,883 tool calls, 1,514 `length` and 86 `stop` out of 141,483 responses, so **1.13%** of responses carry no tool call (`L1:52-57`). The family's per-run mean is 1.24% (`q5.py`).
- **Model-card scores** ([HF card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), BF16): GPQA-Diamond 91.7, IFBench 81.3, LiveCodeBench v6 91.9.
- **GPQA** is CC-BY-4.0 and gated, and asks users to "do not reveal examples ... online" ([HF](https://huggingface.co/datasets/Idavidrein/gpqa)). Diamond has 198 questions ([arXiv 2311.12022](https://arxiv.org/abs/2311.12022)).
- **IFEval** checks verifiable instructions with code ([arXiv 2311.07911](https://arxiv.org/abs/2311.07911)).
- **State-lab throughput:** 730–960 samples per GPU-h (`_critic.md:296`).
- **The critic's rule:** gate on game metrics; the general canary is a back-off trigger (`_critic.md:225`).

### Inferences: the design

| Part | Content | Time | Fail / back-off rule |
|---|---|---|---|
| A, gate | 300 logged states (on-route, stuck, ≥80k context), 1 sample each at deployed sampling | 19–25 min | No tool call >9/300 (P ≈ 0.01 at base 1.13%); malformed call or SyntaxError, or a sandbox exception, more than 2 pp above base (paired McNemar); valid-action rate down >3 pp; repetition loops more than 1 pp above base |
| B, back-off | GPQA-Diamond, fixed 100 questions | 10–19 min (tokens per question UNVERIFIED) | A drop >6 pp (about 2 SE paired) → λ-merge toward base |
| B, back-off | IFEval, fixed 150 prompts | about 5 min | Strict accuracy down >5 pp → back off |

- Calibrate the thresholds with a base-vs-base repeat on day 1.
- Keep the GPQA dataset private on Kaggle.

### Gaps
- Real thresholds come only from the base-vs-base noise.

## Q7. How does this change if we can rent Azure RTX PRO 6000 VMs?

### Takeaway
- **SKU:** `Standard_NC144lds_xl_RTXPRO6000BSE_v6`, at about $13 per run on-demand and about $2.5 on spot. Its 264 GiB of host RAM covers the 95.4 GiB BF16 PLE table.
- **What it buys:** replicate runs, so offline +5 becomes detectable.
- **What it doesn't change:** the leaderboard stays the arbiter.

### Cited Findings
- **SKUs** (`W6:207-213,279`):
  - NC144lds: 1 GPU, 144 vCPU, 264 GiB, $5.50/h on-demand, $1.02/h spot.
  - NC144ds: 516 GiB at $6.38/h, which isn't needed.
  - The fractional sizes (¼ and ½ GPU) can't hold the model.
- **PLE host memory:** 95.4 GiB (`next-moves.md:260`).
- **Calibration:** benchmark decode against Kaggle's 895–909 tok/s at full batch, and stop if it is more than 5% off. Add a platform term and never pool raw runs. The weights (119 GB) take about 15 minutes to pull (`next-moves.md:263-269`).
- **Quota risk:** high-demand GPU quota may not be granted to new subscriptions (`W6:220`).

### Inferences
- **What the money buys:**

  | Item | On-demand | Spot |
  |---|---|---|
  | 6–8 base runs on U25 | about $100 | about $20 |
  | 6 U25 runs per checkpoint (+5 detectable at sd 3.3) | about $80 | about $15 |
  | A 532-min, 110-game dress rehearsal (public + external games), for reliability | about $50 | — |
  | Branch bench (73 branches per arm, about 5 GPU-h; `L3:257`) | about $28 | — |

- **It also frees the Kaggle quota for data generation.**
- **Platform check.** The Azure card is the BSE (Blackwell Server Edition); Kaggle's variant is UNVERIFIED. Run 2 reference runs on the public 25 and compare them with 51.17 ± 3.28.

## Implications for our plan
1. **Adopt the 15/10 split with H10** (lp85, tr87, cn04, sc25, vc33, dc22, ka59, lf52, re86, wa30).
   - Retire the 25-game score and the 10-win target for trained checkpoints.
   - Exclude arc-interactive's vc33 copy from training.
2. **Never score H10 in a run that contains trained games.** Evaluate on U25 (H10 plus 15 DreamTeam games), with at least 3 base runs. Keep the other 10 DreamTeam games as a single-use final panel.
3. **Gates before any leaderboard draw:**
   - G0, serving parity: decode and MTP acceptance at least 95% of base, and prompt-token parity.
   - G1, the canary (Q6).
   - G2, a paired frozen-state screen on 420–470 states: no drop in agreement or right-move rate.
   - G3, 2 U25 runs scoring at least the base mean − 2, with no errors.
4. **Leaderboard:** follow the calendar; give one FT checkpoint at least 3 draws; pick finals by the Δ* rule, never by a draw's public score.
5. **Treat screens as regression gates.** No log metric predicted rankings. A minute-25 panel predicts at r = 0.21.
6. **Azure, if approved:** calibrate first, then buy replicates and a 9-h dress rehearsal. Don't build the schedule around quota arriving.

## Fact-check (independent verifier, 2026-10-18)

I re-checked all 13 key results. My scripts are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/ce155f4d-514a-4385-ac65-a7e7e5a23e1c/scratchpad/gbv/` (`v1.py`–`v8.py`, written from scratch). They recompute scores with `arc_agi` 0.9.9's formula (`scorecard.py:168-206`), which matches the logged score for all 700 rows of `games.csv`. I did not re-run `picks.py`'s Monte Carlo; I used closed forms instead. I also did not check IFEval timing, whether the 16- and 32-grid DreamTeam games render at 64×64, or arc-interactive level counts by loading the games.

| # | Result | Verdict | Correction / note |
|---|---|---|---|
| 1 | Noise: 51.17, sd 3.28, within 3.15 (df 4); 10-subset sd p5 4.24 / med 5.80 / p95 7.44; 8.20 vs 8.61 | **Confirmed** | Exact: 51.172, 3.280, 3.149; 3,268,760 subsets; 4.24 / 5.15 / 5.80 / 6.44 / 7.44; √2·5.80 = 8.20; median per-game paired sd 27.24 → 8.61. Also reproduced: within-replicate median 5.43 (2.50–8.94), corr with the 25-game mean 0.57. |
| 2 | H10 statistics and selection | **Partially confirmed** | All statistics are exact: mean 51.84 (43.30–63.34), sd 5.13, within 3.98, win 0.285, corr with the other 15 = 0.311, all-pairs sd 7.25, 2.85 vs 7.65 wins, modes 5/3/2, 4 bench games, 5 walls. The funnel 415,800 → 22,239 → 5,721 reproduces. But 5,721 **also requires ≥3 of the 6 branch-bench games** (line 39 says so; the key-result summary leaves it out). Without that condition the count is 13,390. **ft09's `metadata.json` has empty `tags`**, and `pick.py` hard-codes it as click, so "tags from the public metadata.json files" is not quite right. With ft09 untagged the funnel gives 21,096 → 5,436, and H10 still ranks 2nd. H10 is the 2nd-ranked set by correlation, not the top one: 1st is 0.322 with bp35 in place of wa30 and 15 frontier cells, against H10's 0.311 and 18. |
| 3 | Leader 63.34 on H10, highest of 26 | **Confirmed** | Next are mem-unified-10x 59.71, prompt-facts 59.25 and pf-keep 58.94. |
| 4 | Power table | **Confirmed** | z = 2.486. At sd 5.8: 16.64 / 12.23 (+5) and 4.16 / 2.26 (+10). At sd 3.3: 5.39 / 3.00 and 1.35 / 0.69. On the LB at sd 3.5: +5 → 6.06; +8 → 2.37 two-arm and 1.47 vs 6 draws; +10 → 0.87–1.51. |
| 5 | Compute-coupling contamination, +5 to +8 on H10 | **Partially confirmed** | The sources check out: the quote is `WRITEUP.md:125` verbatim; 50.93 → 58.49 is at `L3:118`; 105.4 / 176.7 = 59.7%. T15 also holds 60.2% of *all* level tokens (155.8M of 258.8M), so "frees about 15% of run compute" holds. But "+37% for H10" assumes all of the freed budget goes to H10. The quoted mechanism hands it to *any* unfinished game. Per run, about 10.2 T15 games are unfinished against about 7.2 H10 games, since T15 averages 4.80 wins. An even split gives H10 about +15%, worth roughly +4 at L3's elasticity (s = 1.15 → +4.15). Also, sim2's elasticity is a whole-run speed-up at equal decisions, not a reallocation. So **treat +5–8 as an upper bound; about +3–8 is the honest range.** The recommendation (don't score H10 next to trained games) stands. |
| 6 | 85-run corpus split | **Confirmed** | All exact: 85 runs, 1,642 game-runs, 6,034 / 354 / 151 / 123; T15 93, 3,616, 230 (9 games), 24/42, 39/62, 77, 105.41M; H10 58, 2,418, 124 (5 games), 18/42, 23/62, 46, 71.26M, 22,076 steps. |
| 7 | arc-interactive / DreamTeam scan | **Partially confirmed** | 249 rows, 54 with 6+ levels (38 of them 8×8–12×12), the 8 named games, 27 of 252 arrays with 6+ entries, and DreamTeam 25 × 8 entries all reproduce. But **ab01 is a bad example**: `ab01-v1` has 5 entries `[1,2,3,4,6]`, so it is not one of the 27. Of the 27, 24 are `[1,2,3,4,5,6]` placeholders. The other 3 carry real baselines, because they are the copies of public games: ft09 `[15,7,15,16,21,17]`, ls20 (7 entries) and vc33 (7 entries). |
| 8 | 10x back-test: reliability 0.08; ρ +0.33 / +0.06 / −0.14 | **Confirmed** | Variances 10.76 / 9.92 → 0.078. ρ values are exact. My bootstrap CI for L1 is [−0.08, +0.67] (different seed). |
| 9 | Wide-set back-test (48 runs) | **Confirmed** | n = 48, sd 5.13 → 0.62. L1 +0.34 [+0.07, +0.57], tokens per decision +0.22 [−0.06, +0.46], no-tool +0.12 [−0.19, +0.41]. Caveat: 0.62 assumes the family's within-sd of 3.15 also applies to the wider set. `arms.md:313-322` supports "no lab result has a full run". |
| 10 | Pick simulation and Δ* rule | **Partially confirmed** | Closed forms: both-fallback = σ/√π = 1.97 at every Δ; one-each 1.74 / 2.21 / 2.74; both-FT 0.97 / 1.97 / 2.97. The note's Monte Carlo values are within 0.05 of these. At s = 2.47 the crossovers fall at Δ = −0.48 and +0.48, which matches the ±0.5 band. However, **2.47 is the sampling SE.** Under the N(0, 3²) prior that produces the 0.6 shrink (0.595), the posterior sd is 1.91, and the band narrows to **±0.29**. The expected-score difference is under 0.1 point either way. |
| 11 | mellamoa submissions; rules | **Confirmed** | The CLI (2026-10-18) lists 10-17 02:39:13 as the latest row. snowiers' 28.73, 29.54, 34.04 and 28.06 (`runs/submissions.csv`) are absent from it. `W5b:113-114` and `:147` give 1 a day, 2 finals, and 11:59 PM UTC. See the multi-account note below. |
| 12 | Canary baselines, model card, GPQA | **Confirmed** | 1,600 / 141,483 = 1.131%. The HF card's table shows Flash-Next IFBench 81.3, GPQA Diamond 91.7 and LCB v6 91.9. The card doesn't label the scores' precision; the published safetensors are BF16. GPQA: `cc-by-4.0`, gated (`auto`), with the prompt "You agree to NOT reveal examples from this dataset in plain text or images online". The paper's Diamond set is "composed of 198 questions". |
| 13 | Azure NC144lds | **Confirmed** | `W6:207` (1 GPU, 144 vCPU, 264 GiB, $5.50, $1.02 spot) and `W6:279` (API-confirmed); `next-moves.md:260-269` (95.4 GiB PLE, 895–909 tok/s, 5% kill rule, platform term, 119 GB pull in about 15 min). The cost arithmetic works out: 2.3 h × $5.50 ≈ $13, and 8.87 h ≈ $49. |

**Other errors found**
- Line 119: only **10** DreamTeam games define `GRID_SIZE` (1 at 16, 4 at 32, 5 at 64), not 11. The other two matches are `observable.py` and `palette.py`, which are not games.
- Line 250: at a 1.13% base rate, P(>9 of 300 with no tool call) is **0.0025**. P ≈ 0.01 corresponds to >8 (0.0078). Either the threshold should read "≥9" or the probability should read 0.0025.
- Line 100: T15 has **1,024** game-runs in the corpus (H10 has 618), not about 980, which is the proportional 1,642 × 15/25. At 12 per GPU-h that is about 85 GPU-h, not 80. The conclusion doesn't change.
- Lines 95 and 113 disagree slightly: 25 game-runs per 2.3 GPU-h is 10.9 per GPU-h, against the 12 cited from `_critic.md:222`.
- Line 165 / Q4 gaps: `G-F-intel-rules.md:23,36` quotes the rule "you cannot enter or submit from multiple accounts" and calls the two-account setup a disqualification risk. Reading snowiers as a "separate leaderboard entry" makes that risk concrete, and the note should flag it. The calendar counts only mellamoa slots, which is the safe choice. 2 of the 4 mem-repair draws in the noise figures are snowiers draws; they are still valid as noise data.
