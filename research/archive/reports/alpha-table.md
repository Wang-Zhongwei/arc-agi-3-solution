# Which arms are real? Run-level and per-game significance (2026-10-10)

Full 121-min runs on the 25 public games only. Reproduce with `python3 analysis/alpha_table.py` (reads `runs/kaggle/*/benchmark.json.xz`).

- **Noise model:** run-level difference over the pooled run-to-run sd from replicate groups (three user-combo-10x runs, four Franzen control-like runs, the two ledger-on-trim runs, the two 13-expert runs), with a t threshold at the pooled degrees of freedom. Per-game: candidate minus reference mean per game, games won/lost, and a bootstrap 95% CI of the mean.
- **Placebo:** each user-combo-10x run against the other two (same config). None should be significant; the largest gap is ±6 points (t = 2.0), which is the size of difference one run can show by chance.
- **Panels** (10 games, 25 min) are not in this table: there is one reference panel, and arms that did nothing (macros, 0 invocations) scored +3.4 over it, so panel noise is about ±3.

```
pooled run-to-run sd 2.47 (dof 7); groups: [56.94, 52.7, 49.09]; [46.25, 45.92, 44.1, 46.38]; [52.56, 49.14]; [49.61, 47.34]
(games played: )

== placebo: identical config, one run vs the other two (should NOT be significant) ==
placebo 10x v1                     runs 56.94            ref mean 50.90  diff  +6.04  t +2.00  sig(95%) no   all above  games +16/-6  per-game 95% CI [-2.4, +15.2]
placebo 10x v2-snowiers            runs 52.70            ref mean 53.02  diff  -0.32  t -0.10  sig(95%) no   overlap    games +9/-13  per-game 95% CI [-9.6, +9.8]
placebo 10x v3-snowiers            runs 49.09            ref mean 54.82  diff  -5.73  t -1.89  sig(95%) no   all below  games +10/-12  per-game 95% CI [-17.5, +4.4]

== Franzen-era arms vs Franzen control runs (44.10-46.38) ==
ledger-on-trim (2 runs)            runs 52.56, 49.14     ref mean 45.66  diff  +5.19  t +2.43  sig(95%) YES  all above  games +13/-11  per-game 95% CI [-1.3, +12.0]
fz-ledger-streams12/v1             runs 48.08            ref mean 45.66  diff  +2.42  t +0.88  sig(95%) no   all above  games +14/-9  per-game 95% CI [-3.4, +8.5]
fz-ledger-stallswap/v1             runs 45.22            ref mean 45.66  diff  -0.45  t -0.16  sig(95%) no   overlap    games +14/-9  per-game 95% CI [-10.9, +9.3]
fz-experts16-full/v1               runs 38.69            ref mean 45.66  diff  -6.98  t -2.53  sig(95%) YES  all below  games +9/-15  per-game 95% CI [-14.9, +0.5]
fz-ucb/v1                          runs 43.84            ref mean 45.66  diff  -1.83  t -0.66  sig(95%) no   all below  games +9/-14  per-game 95% CI [-10.6, +6.4]
v4-stack-v1/v1                     runs 43.32            ref mean 45.66  diff  -2.34  t -0.85  sig(95%) no   all below  games +14/-9  per-game 95% CI [-13.3, +6.7]

== v4 stack vs controls ==
user-combo-10x vs fz control       runs 56.94, 52.70, 49.09 ref mean 45.66  diff  +7.25  t +3.84  sig(95%) YES  all above  games +15/-8  per-game 95% CI [+1.2, +13.7]
user-combo-10x vs v4-control       runs 56.94, 52.70, 49.09 ref mean 50.62  diff  +2.29  t +0.80  sig(95%) no   overlap    games +11/-11  per-game 95% CI [-7.0, +11.9]
v4-control vs fz control           runs 50.62            ref mean 45.66  diff  +4.96  t +1.80  sig(95%) no   all above  games +12/-11  per-game 95% CI [-3.0, +12.7]
tput-safe vs v4-control            runs 42.09            ref mean 50.62  diff  -8.53  t -2.44  sig(95%) YES  all below  games +6/-10  per-game 95% CI [-17.6, -0.2]

== arms on the user-combo-10x stack vs its 3 runs ==
13 experts (2 runs)                runs 49.61, 47.34     ref mean 52.91  diff  -4.44  t -1.97  sig(95%) no   overlap    games +6/-16  per-game 95% CI [-10.9, +2.1]
v4-user-combo-10x-accept05/v1      runs 46.77            ref mean 52.91  diff  -6.15  t -2.16  sig(95%) no   all below  games +9/-13  per-game 95% CI [-14.4, +1.5]
v4-user-combo-16x/v1               runs 39.65            ref mean 52.91  diff -13.27  t -4.65  sig(95%) YES  all below  games +7/-15  per-game 95% CI [-23.9, -4.0]
v4-hats/v1                         runs 37.11            ref mean 52.91  diff -15.80  t -5.54  sig(95%) YES  all below  games +9/-15  per-game 95% CI [-26.1, -6.6]
v4-sched-curriculum/v1             runs 47.83            ref mean 52.91  diff  -5.09  t -1.78  sig(95%) no   all below  games +11/-12  per-game 95% CI [-13.7, +3.1]
v4-uc-18s-64k/v1                   runs 36.01            ref mean 52.91  diff -16.91  t -5.93  sig(95%) YES  all below  games +3/-19  per-game 95% CI [-24.2, -10.3]
v4-uc-11x/v1-snowiers              runs 47.19            ref mean 52.91  diff  -5.72  t -2.01  sig(95%) no   all below  games +12/-11  per-game 95% CI [-17.9, +5.3]
```
