# fable-layer-vs-ours: web:kaggle

Workflow run `wf_c4d3dccd-2d7`, phase Read. Compare leoprovorov Fable & Astra notebook (handbook + Fable layer on Franzen) with our stack: read, measure claims on our runs, compare, skeptic-verify

---

Facts only, observed 2026-10-05. I did not log in, post, vote or comment.

**(1) Notebook: https://www.kaggle.com/code/leoprovorov/fable-astra-play-arc-3-handbook-harness**
- Title is "Fable & Astra Play ARC-3: Handbook + Harness". Author is AlekseiProvorov (@leoprovorov). It shows 91 views, 9 votes, 9 copies and an Apache 2.0 license.
- It is "Copied from Daniel Franzen" (https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution), with a diff of +1715/-11001.
- It has 4 versions, and V4 was updated about 4 hours ago:
  - "Copied from notebook": 6 days ago, ran 34m12s (Franzen's original).
  - V1: "Save & Run All", 4 days ago, diff +0/-0, ran 35m28s.
  - V2: Quick Version, about 4 hours ago, diff +1715/-11001, ran 10s.
  - V3: Quick Version, diff +0/-0, ran 9s.
  - V4: Quick Version, diff +0/-0, ran 7s.
- The page shows "Best Score 31.73 V1". That is V1, the unmodified Franzen copy. The only version containing the Fable layer is V2, and it ran for 10 seconds, so it never produced a real run. That implies the 31.73 is not a Fable-layer result. This is my inference from the version history, not something the page states.
- Inputs are the competition, the dataset below, and the GPU tag. The Comments tab shows 0 comments, so there is no thread, no LB report and no author replies.
- The body is an article plus a patch notebook. It describes a 90-page handbook, "the math behind the scoring", and "a small harness layer", and it ends with seven open questions.
- Lineage as the notebook states it: Duck harness, then Franzen M2, then this solution. The model, server, schedule and system prompt are identical to Franzen's. The only addition is a "Fable layer" of about 200 patch lines plus a ~540-line module. Every hook sits behind an env flag, and an error means "do nothing". These are the components:
  - **Indicator notes:** the harness reports a changed legend or tray patch and which action changed it. It fires after 15k tokens on a level.
  - **Pinned level log:** what each key did, in which board state, plus indicator history and UNDO count. It fires after 30k tokens on a level and refreshes every 40k.
  - **Budget estimate:** an alert only when ≤25% of the budget or ≤12 actions are left.
  - **Five protocol lines:** added inside the level log.
- Its stated rules are: add facts not advice; never remove history; touch only levels that drag on. It borrows evidence from a third-place M2 write-up by rellik13 (advice blocks cut score 6.42 to 4.16 and 13.40 to 9.98; keeping more recent history gave +8.0; a pinned rule gave +2.8).
- Its stated scoring math is below. It also estimates about 181k generated tokens per game (626 tok/s × 31,800 s ÷ 110 games):
  - Level score s_i = min(1.15, (h_i/a_i)^2). Game score weights levels by index and is capped by the share of weight cleared.
  - Elasticity of score to tokens: about 1.5–1.7 below 60k tokens per game, and 0.27/0.24 above 160k.

**(2) Dataset: https://www.kaggle.com/datasets/leoprovorov/arc-agi-3-fable-harness-analytical-handbook**
- It has "No description available", 1 version (V1, 5.51 MB), 0 votes, a usability score of 1.88, and was updated about 5 hours ago.
- Its three files are `ARC-AGI-3_handbook_en.pdf` (2.52 MB), `F001_fable_layer_notebook.ipynb` and `handbook_reader.html`.
- It has 4 views and 5 downloads. The one related notebook is the Fable & Astra one.

**(3) Profile: https://www.kaggle.com/leoprovorov**
- AlekseiProvorov is a Data Scientist at Consructor Group in Saint Petersburg, and a Notebooks Expert ranked 344 of 60,286.
- There are 8 public notebooks. The only ARC-AGI-3 one is the Fable & Astra notebook, scored 31.73. The others are other competitions and Gemma notebooks.
- There is no separate F001 submission notebook with a score. F001 exists only as a file inside the dataset.

**(4) Leaderboard: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard**
- This is the Public tab, computed on about 50% of the test data, with 3,845 teams and "A month to go".
- Leoprovorov's team "AlekseiProvorov" is rank 69, score 31.73, 7 entries, last submission 11h ago.
- Top 10 on the public leaderboard:

| Rank | Team | Score |
|---|---|---|
| 1 | Tufa Labs | 55.89 |
| 2 | Yi-Chia Chen | 48.59 |
| 3 | the last dance | 39.30 |
| 4 | mtg | 38.33 |
| 5 | face-of-agi | 37.94 |
| 6 | gng | 36.78 |
| 7 | Lord Han Solo | 36.61 |
| 8 | YUTO KOJIMA | 36.14 |
| 9 | Andrew Reed | 36.14 |
| 10 | lalalia | 35.79 |

- Ranks 11–49 run from 35.44 down to 32.24, and rank 50 and below is under 32.24.
- The page also shows a personal-best banner at rank 460 ("speed, my mom is kinda homless", 27.62, up from 22.88). It appears to be the logged-in account's own entry.

**(5) The "sovereign agent" notebook and its 13 points**
- The Fable notebook says: "The curve below has 13 points printed by a public Kaggle notebook (the 'sovereign agent' notebook) over the 25 public games". The 13 points are in its code cell "In [3]" as `CURVE`, in thousands of tokens per game and mean score on the 25 public games:
  - (14.8, 1.55), (30.2, 5.21), (45.8, 9.82), (60.7, 15.86), (74.8, 25.87), (90.6, 29.99), (105.1, 32.31)
  - (117.5, 36.51), (132.7, 39.07), (145.3, 43.93), (162.8, 47.24), (173.4, 48.05), (192.5, 49.25)
- The best match for that notebook is "ARC Prize 2026 Sovereign Agent" by bang1850: https://www.kaggle.com/code/bang1850/arc-prize-2026-sovereign-agent. The Fable page does not link it, so the match rests on the title and the "sovereign agent" search.
  - It has 47 votes, 52 copies, 555 views and 7 versions. Public score is 19.22, best score 29.08 (V4), with a Bronze medal.
  - It is a Franzen/Tufa-based bundle with an added harness patch and a known-no-op repeat guard. It uses Qwen3.8-Flash-Next W4A16 with the albucino MTP drafter. It runs on an RTX Pro 6000 for 2h25m, and its Comments tab shows 0 comments.
  - Its logged non-submission run on the 25 public games gave mean score 34.26, median 21.43, 3.52M total tokens and 2h14m49s duration, with 3 of 25 games won.
- I did not find the 13-point curve or the token series printed in that notebook's visible log. I only read the section 9 and 10 summary there, so the attribution of the curve to it is unverified.
- Context for the user's comparison: the Fable notebook scores the same as Franzen's baseline on the leaderboard, which is its V1 and not the Fable layer. Its value is the analysis, the elasticity data and the opt-in harness hooks. The only open-model number it adds is that curve, whose elasticity is about 0.25 near the real ~181k-token budget.
