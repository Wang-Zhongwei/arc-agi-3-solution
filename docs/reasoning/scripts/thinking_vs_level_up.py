"""Thinking per turn vs leveling up on that turn, ls20 api-signatures study.

One game, 7 levels, 30 passes (3 arms x 10), so turns on the same level are
directly comparable without normalizing for difficulty. Per turn
(analysis_step) it sums `reasoning_chars` over the turn's model calls (about
4 characters per token) and marks a level-up when one of the turn's actions
has `level_completed`. Writes data/ls20_turns.csv and
figures/thinking_vs_level_up_ls20.png and prints the tests.

Needs the pulled runs under ARC3-Inference/runs/kaggle (or ARC3_RUNS_DIR).

    python3 docs/reasoning/scripts/thinking_vs_level_up.py
"""

import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORT = Path(__file__).resolve().parents[1]
K = Path(os.environ.get("ARC3_RUNS_DIR", REPORT.parents[1] / "ARC3-Inference/runs")) / "kaggle"
ARMS = {"v0": K / "jonathanwang2022/api-signatures-and-frame-diff/v03",
        "v1": K / "jonathanwang2022/api-signatures-and-frame-diff/v05",
        "v2": K / "ticbudja/api-signatures-and-frame-diff/v01"}
DATA = REPORT / "data/ls20_turns.csv"
FIG = REPORT / "figures/thinking_vs_level_up_ls20.png"

RC = re.compile(r"^reasoning_chars: (\d+)$", re.M)
FR = re.compile(r"^finish_reason: (\w+)$", re.M)
rows = []
for arm, run in ARMS.items():
    for p in sorted(run.glob("artifacts/ls20-9607627b_p*_events.jsonl")):
        pas = int(p.stem.split("_p")[1].split("_")[0])
        ev = [json.loads(l) for l in open(p)]
        acts = defaultdict(list)
        for e in ev:
            if e.get("type") == "action": acts[e["analysis_step"]].append(e)
        think = defaultdict(lambda: [0, 0, 0])
        for e in ev:
            if e.get("type") == "analysis":
                t = think[e["analysis_step"]]; rc = [int(x) for x in RC.findall(e["transcript"])]
                t[0] += sum(rc); t[1] += len(rc); t[2] += FR.findall(e["transcript"]).count("length")
        level = 1
        for s in sorted(think):
            a = acts.get(s, []); up = sum(bool(x.get("level_completed")) for x in a)
            rows.append(dict(arm=arm, pass_=pas, step=s, level=level, think_chars=think[s][0], calls=think[s][1],
                             capped=think[s][2], actions=len(a), level_up=int(up > 0),
                             death=int(any(x.get("game_over") for x in a))))
            level += up
with open(DATA, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

d = pd.DataFrame(rows)
d["log_t"] = np.log(d.think_chars)
L = d[d.level <= 4].copy()  # levels with >= 20 level-ups
UP, OTHER, INK, MUTED, GRID, SURF = "#2a78d6", "#a9a8a3", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF})

# ---- stats
print("Per level: median thinking chars, level-up turns vs other turns (Mann-Whitney)")
for lv, x in L.groupby("level"):
    u, o = x[x.level_up == 1].think_chars, x[x.level_up == 0].think_chars
    print(f"  L{lv}: up {u.median():6.0f} (n={len(u)})  other {o.median():6.0f} (n={len(o)})  ratio {u.median()/o.median():.2f}  p={stats.mannwhitneyu(u, o).pvalue:.2g}")
L["turns_on_level"] = L.groupby(["arm", "pass_", "level"]).cumcount()
for f in ["level_up ~ log_t + C(level)", "level_up ~ log_t + np.log1p(actions) + C(level) + C(arm)",
          "level_up ~ log_t + np.log1p(turns_on_level) + np.log1p(actions) + C(level) + C(arm)"]:
    m = smf.logit(f, L).fit(disp=0); ci = np.exp(m.conf_int().loc["log_t"])
    print(f"  {f:<85} OR per doubling of thinking = {2**m.params['log_t']:.2f} [{2**np.log(ci[0]):.2f},{2**np.log(ci[1]):.2f}] p={m.pvalues['log_t']:.2g}")
for arm, x in L.groupby("arm"):
    m = smf.logit("level_up ~ log_t + C(level)", x).fit(disp=0)
    print(f"  arm {arm}: OR per doubling = {2**m.params['log_t']:.2f} p={m.pvalues['log_t']:.2g}")

fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
rng = np.random.default_rng(0)
# (a) raw thinking per turn, by level, level-up turns vs others
for lv in range(1, 5):
    x = L[L.level == lv]
    for j, (flag, c, lab) in enumerate([(0, OTHER, "other turns"), (1, UP, "level-up turn")]):
        y = x[x.level_up == flag].think_chars; pos = lv + (j - .5) * .38
        ax[0].scatter(pos + rng.uniform(-.13, .13, len(y)), y, s=10 if flag == 0 else 26, color=c, alpha=.45 if flag == 0 else .9,
                      edgecolor=SURF if flag else "none", linewidth=.8, label=lab if lv == 1 else None, zorder=2 + flag)
        ax[0].plot([pos - .17, pos + .17], [y.median()] * 2, color=INK, lw=2, zorder=5, solid_capstyle="round")
ax[0].set(yscale="log", xticks=range(1, 5), xticklabels=[f"L{i}" for i in range(1, 5)], xlabel="Level of ls20",
          ylabel="Thinking characters in the turn (≈ 4 per token)", title="(a) Every turn of 30 ls20 passes, split by level\n(black bar = median)")
ax[0].legend(frameon=False, fontsize=9, loc="lower left", markerscale=1.5)

# (b) P(level-up) by thinking quintile, quintiles cut within each level
L["q"] = L.groupby("level").think_chars.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
r = L.groupby("q").level_up.agg(["mean", "sum", "count"])
ci = np.array([stats.binomtest(int(k), int(n)).proportion_ci() for k, n in zip(r["sum"], r["count"])])
ax[1].bar(r.index, r["mean"], width=.62, color=UP, edgecolor=SURF, linewidth=2)
ax[1].errorbar(r.index, r["mean"], yerr=[r["mean"] - ci[:, 0], ci[:, 1] - r["mean"]], fmt="none", ecolor=INK, elinewidth=1)
for i, row in r.iterrows():
    ax[1].text(i, ci[int(i), 1] + .006, f"{row['mean']:.0%}\n{int(row['sum'])}/{int(row['count'])}", ha="center", va="bottom", fontsize=8.5)
base = L.level_up.mean()
ax[1].axhline(base, color=MUTED, lw=1, ls="--"); ax[1].text(3, base + .004, f"base rate {base:.0%}", ha="center", va="bottom", color=MUTED, fontsize=8.5)
ax[1].set(xticks=range(5), xticklabels=["least\n20%", "20–40%", "40–60%", "60–80%", "most\n20%"], ylim=(0, ci[:, 1].max() + .07),
          xlabel="Turn's thinking, ranked among all turns on the same level", ylabel="Share of turns that level up",
          title="(b) Chance the turn clears the level,\nby how much it thought (L1–L4)")
ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))

# (c) thinking around each level-up, as ratio to the median turn on the same level
L["rel"] = L.think_chars / L.groupby("level").think_chars.transform("median")
rows = []
for (arm, p), y in d.groupby(["arm", "pass_"]):
    y = y.reset_index(drop=True); y["rel"] = y.think_chars / y.level.map(d.groupby("level").think_chars.median())
    for i in y.index[(y.level_up == 1) & (y.level <= 4)]:
        for k in range(-5, 4):
            if 0 <= i + k < len(y): rows.append(dict(k=k, lr=np.log2(y.rel[i + k])))
e = pd.DataFrame(rows).groupby("k").lr.agg(["mean", "sem"])
ax[2].axhline(0, color=MUTED, lw=1); ax[2].axvspan(.5, 3.5, color=GRID, alpha=.6, lw=0)
ax[2].text(2, .1 + (e["mean"] + 1.96 * e["sem"]).max(), "next level", ha="center", color=MUTED, fontsize=8.5)
ax[2].fill_between(e.index, e["mean"] - 1.96 * e["sem"], e["mean"] + 1.96 * e["sem"], color=UP, alpha=.18, lw=0)
ax[2].plot(e.index, e["mean"], color=UP, lw=2, marker="o", ms=7)
ax[2].annotate("level-up turn", (0, e.loc[0, "mean"]), (-3.3, e.loc[0, "mean"] + .15), fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=.8))
ax[2].set(xticks=range(-5, 4), xlabel="Turns relative to the level-up turn", ylabel="Thinking vs median turn on that level (log₂)",
          title="(c) Thinking dips on the level-up turn,\nthen spikes on the new level")
ax[2].set_yticks([-2, -1, 0, 1, 2]); ax[2].set_yticklabels(["¼×", "½×", "1×", "2×", "4×"])
for a in ax: a.grid(color=GRID, lw=.6); a.set_axisbelow(True)
fig.tight_layout(); fig.savefig(FIG, dpi=150)
print("quintile rates:", r["mean"].round(3).tolist(), "event:", e["mean"].round(2).tolist())
