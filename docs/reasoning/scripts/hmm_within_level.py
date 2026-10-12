"""Within-level linear probability model and AUC; run hmm_vs_level_up.py H first.

    python3 docs/reasoning/scripts/hmm_within_level.py 10000 [think|out]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
H = sys.argv[1]
CLOCK = sys.argv[2] if len(sys.argv) > 2 else "think"
x = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / f"hmm_rows_H{H}_{CLOCK}.csv.gz")
x["grp"] = x.gid + "#" + x.level.astype(str)
# within-level-attempt LPM: demean y and regressors by (game run, level)
cols = ["z", "log_think", "log_on_level"]
dm = x[["y"] + cols].sub(x.groupby("grp")[["y"] + cols].transform("mean"))
m = sm.OLS(dm.y, dm[cols]).fit(cov_type="cluster", cov_kwds={"groups": x.gid.astype("category").cat.codes})
print(f"H={H}, clock={CLOCK} within-level LPM: +1 SD hmm density -> {100*m.params['z']:+.1f} pp level-up prob "
      f"[{100*m.conf_int().loc['z',0]:+.1f}, {100*m.conf_int().loc['z',1]:+.1f}], p={m.pvalues['z']:.1e} (base {100*x.y.mean():.1f}%)")
# AUC of density alone, within game (rank within game name)
r = x.groupby("game").dens.rank(pct=True)
print(f"  AUC (lower density -> level-up), within-game rank: {roc_auc_score(x.y, -r):.3f}; "
      f"thinking volume alone: {roc_auc_score(x.y, -x.groupby('game').log_think.rank(pct=True)):.3f}; "
      f"tokens-on-level alone: {roc_auc_score(x.y, -x.groupby('game').log_on_level.rank(pct=True)):.3f}")

# Is the current level eventually cleared? One row per (game run, level) at its first call past
# each checkpoint of tokens on the level, so long levels are not over-weighted.
import statsmodels.formula.api as smf
ups = pd.read_csv(Path(__file__).resolve().parents[1] / "data/hmm_calls.csv.gz").groupby("gid").levelups.sum()
x["cleared"] = (x.level <= x.gid.map(ups)).astype(int)
print("\nEventually clears the current level, by hmm density at a checkpoint (game + level FE):")
for cp in (10_000, 20_000, 40_000, 80_000):
    s = x[x.on_level >= cp].sort_values("on_level").groupby(["gid", "level"]).head(1).copy()
    s["z"] = s.dens / x.dens.std()
    q = s.groupby(pd.qcut(s.dens, 3, labels=["low", "mid", "high"]), observed=True).cleared.mean()
    head = f"  {cp // 1000}k on level: n={len(s)}, cleared {s.cleared.mean():.2f}, by tercile {q.round(2).to_dict()}"
    try:
        m = smf.logit("cleared ~ z + log_think + C(game) + C(lvl)", s).fit(
            disp=0, method="bfgs", maxiter=500, cov_type="cluster", cov_kwds={"groups": s.gid.astype("category").cat.codes})
    except np.linalg.LinAlgError:
        print(head + ", fit failed (singular)")
        continue
    lo, hi = np.exp(m.conf_int().loc["z"])
    print(head + f", OR/SD {np.exp(m.params['z']):.2f} [{lo:.2f}, {hi:.2f}] p={m.pvalues['z']:.1e}")

# Calibrated alpha: fit P ∝ d^-alpha by relative-risk (Poisson, log link) regression on log d.
# Windows with no hmm (d = 0, 0.2% of calls) are dropped.
import statsmodels.api as sm
c = x[x.tr_hmm > 0].copy()
c["log_d"] = np.log(c.dens)


def alpha(formula, data, label):
    m = smf.glm(formula, data, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": data.gid.astype("category").cat.codes})
    lo, hi = m.conf_int().loc["log_d"]
    print(f"  {label:<40} alpha = {-m.params['log_d']:.2f} [{-hi:.2f}, {-lo:.2f}] (n={len(data):,})")


print("\nCalibrated alpha, P ∝ d^-alpha (SE clustered by game run):")
alpha("y ~ log_d + C(game) + C(run) + C(lvl)", c, f"level-up within {int(H) // 1000}k, FE")
for cp in (10_000, 20_000, 40_000):
    s = c[c.on_level >= cp].sort_values("on_level").groupby(["gid", "level"]).head(1)
    alpha("cleared ~ log_d", s, f"eventually clears, {cp // 1000}k on level")
    alpha("cleared ~ log_d + C(game) + C(lvl)", s, f"eventually clears, {cp // 1000}k on level, FE")
