"""hmm density vs level-up within the next H generated tokens (default 10k).

    python3 docs/reasoning/scripts/hmm_vs_level_up.py [H] [think|out]

The clock is thinking tokens (think, default) or thinking + tool-call arguments (out).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit

H = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
CLOCK = sys.argv[2] if len(sys.argv) > 2 else "think"
COL = {"think": "think_tok", "out": "out_tok"}[CLOCK]
TRAIL = 10_000
UNIT = 1_000  # densities are reported per 1k thinking tokens (display only)
MIN_THINK = 1_000  # trailing thinking tokens needed for a usable density

DATA = Path(__file__).resolve().parents[1] / "data"
d = pd.read_csv(DATA / "hmm_calls.csv.gz")
rows = []
for gid, g in d.groupby("gid", sort=False):
    g = g.sort_values("call").copy()
    end = g[COL].cumsum().to_numpy()
    start = end - g[COL].to_numpy()
    ups = g.levelups.to_numpy()
    total, won = end[-1], g.won.iloc[0]
    level = 1 + np.concatenate([[0], np.cumsum(ups)[:-1]])
    up_pos = end[ups > 0]
    up_idx = np.flatnonzero(ups > 0)
    hmm, wait, think = g.hmm.to_numpy(), g.wait.to_numpy(), g.think_tok.to_numpy()
    level_start = {}
    for i in range(len(g)):
        lv = level[i]
        level_start.setdefault(lv, start[i])
        # trailing window: calls on this level ending within TRAIL tokens of end_i
        j = i
        while j - 1 >= 0 and level[j - 1] == lv and end[i] - start[j - 1] <= TRAIL:
            j -= 1
        nxt = up_idx[up_idx >= i]
        y = int(len(nxt) > 0 and end[nxt[0]] <= end[i] + H)
        nxt_future = up_idx[up_idx > i]
        y_future = int(len(nxt_future) > 0 and end[nxt_future[0]] <= end[i] + H)
        censored = (not y) and end[i] + H > total and not won
        rows.append(dict(
            gid=gid, run=g.run.iloc[0], game=g.game.iloc[0], level=lv, call=i,
            y=y, y_future=y_future, now=int(ups[i] > 0), censored=censored,
            call_hmm=hmm[i], call_think=think[i],
            tr_hmm=hmm[j:i + 1].sum(), tr_wait=wait[j:i + 1].sum(), tr_think=think[j:i + 1].sum(),
            on_level=end[i] - level_start[lv],
        ))
x = pd.DataFrame(rows)
x = x[~x.censored]
x["dens"] = UNIT * x.tr_hmm / x.tr_think.clip(lower=1)
x["wdens"] = UNIT * x.tr_wait / x.tr_think.clip(lower=1)
x["call_dens"] = UNIT * x.call_hmm / x.call_think.clip(lower=1)
x = x[x.tr_think >= MIN_THINK].copy()
x["lvl"] = x.level.clip(upper=7).astype(str)
x["log_think"] = np.log(x.tr_think)
x["log_on_level"] = np.log1p(x.on_level)
sd = x.dens.std()
x["z"] = x.dens / sd
x["zw"] = x.wdens / x.wdens.std()
print(f"H={H}, clock={CLOCK}: calls {len(x):,}, games {x.gid.nunique():,}, runs {x.run.nunique()}, "
      f"base rate {x.y.mean():.3f}, mean density {x.dens.mean():.2f}/1k (SD {sd:.2f})")

# 1. descriptive: density quintile within game name (pooled over runs)
x["q"] = x.groupby("game").dens.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)) + 1
print("\nP(level-up within H) by trailing hmm-density quintile (within game):")
t = x.groupby("q").agg(n=("y", "size"), dens=("dens", "median"), p=("y", "mean"), p_future=("y_future", "mean"))
print(t.round(3).to_string())


def logit(formula, data, label):
    m = smf.logit(formula, data).fit(disp=0, cov_type="cluster", cov_kwds={"groups": data.gid.astype("category").cat.codes})
    for k in ("z", "zw", "log_think"):
        if k in m.params:
            lo, hi = np.exp(m.conf_int().loc[k])
            print(f"  {label:<38} {k:<9} OR {np.exp(m.params[k]):.3f} [{lo:.3f}, {hi:.3f}] p={m.pvalues[k]:.1e}")


print("\nLogistic regression, odds ratio per 1 SD of hmm density (SE clustered by game run):")
fe = "C(game) + C(run) + C(lvl)"
logit(f"y ~ z + {fe}", x, "game+run+level FE")
logit(f"y ~ z + log_think + log_on_level + {fe}", x, "+ thinking volume, tokens on level")
logit(f"y ~ z + zw + log_think + log_on_level + {fe}", x, "+ 'wait' density")
logit(f"y_future ~ z + log_think + log_on_level + {fe}", x, "excluding current call's action")

print("\nConditional logit within each (game run, level) attempt:")
x["grp"] = x.gid + "#" + x.level.astype(str)
for cols, label in ((["z"], "density only"), (["z", "log_think", "log_on_level"], "+ volume, tokens on level")):
    m = ConditionalLogit(x.y, x[cols], groups=x.grp).fit(disp=0)
    lo, hi = np.exp(m.conf_int().loc["z"])
    print(f"  {label:<38} OR {np.exp(m.params['z']):.3f} [{lo:.3f}, {hi:.3f}] p={m.pvalues['z']:.1e}")

# per-call density (no trailing window), calls with >= 300 thinking tokens
c = x[x.call_think >= 300].copy()
c["z"] = c.call_dens / c.call_dens.std()
c["log_think"] = np.log(c.call_think)
print(f"\nSingle-call density (calls with >=300 thinking tokens, n={len(c):,}):")
logit(f"y ~ z + log_think + log_on_level + {fe}", c, "+ volume, tokens on level")

# heterogeneity by run: per-run within-level estimate
print("\nPer-run conditional logit OR (density + tokens on level), runs with >=100 level-ups in sample:")
out, failed = [], 0
for run, r in x.groupby("run"):
    if r.y.sum() < 100:
        continue
    try:
        m = ConditionalLogit(r.y, r[["z", "log_on_level"]], groups=r.grp).fit(disp=0)
    except (np.linalg.LinAlgError, ValueError):
        failed += 1
        continue
    out.append((run, np.exp(m.params["z"]), m.pvalues["z"]))
o = pd.DataFrame(out, columns=["run", "OR", "p"])
print(f"  {len(o)} runs, OR<1 in {(o.OR < 1).sum()}, median OR {o.OR.median():.3f}, p<0.05 & OR<1 in {((o.p < .05) & (o.OR < 1)).sum()}"
      + (f" ({failed} fits failed)" if failed else ""))
x.to_csv(DATA / f"hmm_rows_H{H}_{CLOCK}.csv.gz", index=False)
