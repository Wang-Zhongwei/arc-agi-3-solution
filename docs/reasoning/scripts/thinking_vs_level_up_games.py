"""Thinking per turn vs leveling up across games (cross-check for the ls20 study).

Three runs that log exact `usage.reasoning_tokens` per request (60 games). Per
turn (analysis_step) it sums the thinking tokens of the turn's model calls and
marks a level-up when the next turn shows a higher level. Games differ in
difficulty, so thinking is compared only within a game or within a level.
Writes data/games_turns.csv and figures/thinking_vs_level_up_games.png and
prints the tests.

Needs `*_p0_requests.jsonl` under ARC3-Inference/runs/kaggle (or ARC3_RUNS_DIR);
the duck-harness copy of these runs has them.

    python3 docs/reasoning/scripts/thinking_vs_level_up_games.py
"""

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
RUNS = {"daniel_v05": ("Daniel v05", "#2a78d6", K / "dangthuygiang/arc3-v207-danibase-base-dfranzen/v05"),
        "ucb_v01": ("UCB v01", "#eb6834", K / "ticbudja/yield-ucb-scheduling-v1/v01"),
        "m2_v01": ("Milestone-2 v01", "#1baf7a", K / "dfranzen/arc-agi-3-milestone-2-solution/v01")}
DATA = REPORT / "data/games_turns.csv"
FIG = REPORT / "figures/thinking_vs_level_up_games.png"
STATE = re.compile(r"Current state: step (\d+), level (\d+)")
SUMMARY = re.compile(r"\s+(\S+): score=.*levels=([\d.]+)/(\d+), actions=(\d+)")


def text(content):
    return content if isinstance(content, str) else " ".join(x.get("text", "") for x in content if isinstance(x, dict))


def final_levels(run):
    """Levels completed and total actions per game, to score the last turn."""
    out = {}
    for line in (run / "summary.txt").read_text().splitlines():
        m = SUMMARY.match(line)
        if m:
            out[m.group(1)] = (int(float(m.group(2))), int(m.group(4)))
    return out


def turns():
    rows = []
    for name, (_, _, run) in RUNS.items():
        fin = final_levels(run)
        for path in sorted(run.glob("*_p0_requests.jsonl")):
            game = path.name.split("_p0")[0]
            t = defaultdict(lambda: dict(think=0, calls=0, level=None, action=None))
            for line in open(path):
                r = json.loads(line)
                turn = t[r["analysis_step"]]
                turn["action"] = r["action"]
                if r["event"] == "request" and turn["level"] is None:
                    s = STATE.search(text(r["messages"][-1]["content"]))
                    if s:
                        turn["level"] = int(s.group(2))
                if r["event"] == "response":
                    turn["think"] += (r.get("usage") or {}).get("reasoning_tokens") or 0
                    turn["calls"] += 1
            steps = sorted(t)
            done, total_actions = fin[game]
            for i, s in enumerate(steps):
                nxt = t[steps[i + 1]] if i + 1 < len(steps) else dict(level=done + 1, action=total_actions + 1)
                rows.append(dict(run=name, game=game[:4], step=s, level=t[s]["level"], think=t[s]["think"],
                                 calls=t[s]["calls"], actions=nxt["action"] - t[s]["action"],
                                 level_up=int(nxt["level"] > t[s]["level"])))
    return pd.DataFrame(rows)


d = turns()
d.to_csv(DATA, index=False)
d["logT"] = np.log1p(d.think)
g = d.groupby(["run", "game"])
d["z"] = (d.logT - g.logT.transform("mean")) / g.logT.transform("std")

print("Logistic regression with game fixed effects, per run (games with >= 1 level-up):")
for run, x in d.groupby("run", sort=False):
    x = x[x.groupby("game").level_up.transform("sum") > 0]
    m = smf.logit("level_up ~ z + np.log1p(actions) + C(game)", x).fit(disp=0)
    ci = np.exp(m.conf_int().loc["z"])
    print(f"  {run}: {len(x)} turns, {x.level_up.sum()} level-ups, OR per 1 SD of within-game thinking = "
          f"{np.exp(m.params['z']):.2f} [{ci[0]:.2f},{ci[1]:.2f}] p={m.pvalues['z']:.2g}")

INK, MUTED, GRID, SURF, UP = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))

# (a) one dot per completed level: level-up turn vs median of the other turns on that level
pairs = []
for (run, game, lv), y in d.groupby(["run", "game", "level"]):
    if y.level_up.sum() == 1 and len(y) >= 2:
        pairs.append(dict(run=run, up=y[y.level_up == 1].think.iloc[0] + 1, other=y[y.level_up == 0].think.median() + 1))
p = pd.DataFrame(pairs)
for run, (lab, c, _) in RUNS.items():
    q = p[p.run == run]
    ax[0].scatter(q.other, q.up, s=36, color=c, edgecolor=SURF, linewidth=1, alpha=.85, label=f"{lab} ({len(q)} levels)")
ax[0].plot([30, 2e4], [30, 2e4], color=MUTED, lw=1, ls="--")
ax[0].text(9e3, 4.5e3, "equal", color=MUTED, ha="left", va="top", fontsize=8.5)
below = (p.up < p.other).mean()
w = stats.wilcoxon(np.log(p.up) - np.log(p.other))
print(f"Paired levels: {len(p)}, level-up turn below the level's median in {below:.0%}, Wilcoxon p={w.pvalue:.2g}")
ax[0].set(xscale="log", yscale="log", xlim=(30, 2e4), ylim=(1, 3e4),
          xlabel="Median thinking tokens of the level's other turns", ylabel="Thinking tokens on the level-up turn",
          title=f"(a) Same level, paired: {below:.0%} of {len(p)} levels fall below\nthe diagonal (Wilcoxon p = {w.pvalue:.0e})")
ax[0].legend(frameon=False, fontsize=8.5, loc="upper left")

# (b) level-up rate by thinking rank within the same level, levels with >= 5 turns, runs pooled
x = d[d.groupby(["run", "game", "level"]).think.transform("size") >= 5].copy()
x["b"] = pd.cut(x.groupby(["run", "game", "level"]).think.rank(pct=True, method="first"), [0, .2, .4, .6, .8, 1], labels=False)
r = x.groupby("b").level_up.agg(["mean", "count", "sum"])
ci = np.array([stats.binomtest(int(k), int(n)).proportion_ci() for k, n in zip(r["sum"], r["count"])])
ax[1].bar(r.index, r["mean"], width=.62, color=UP, edgecolor=SURF, linewidth=2)
ax[1].errorbar(r.index, r["mean"], yerr=[r["mean"] - ci[:, 0], ci[:, 1] - r["mean"]], fmt="none", ecolor=INK, elinewidth=1)
for i, row in r.iterrows():
    ax[1].text(i, ci[int(i), 1] + .006, f"{row['mean']:.0%}\n{int(row['sum'])}/{int(row['count'])}", ha="center", va="bottom", fontsize=8.5)
base = x.level_up.mean()
ax[1].axhline(base, color=MUTED, lw=1, ls="--")
ax[1].text(3, base + .004, f"base rate {base:.0%}", ha="center", va="bottom", color=MUTED, fontsize=8.5)
nlev = x.groupby(["run", "game", "level"]).ngroups
ax[1].set(xticks=range(5), xticklabels=["least\n20%", "20–40%", "40–60%", "60–80%", "most\n20%"], ylim=(0, ci[:, 1].max() + .06),
          xlabel="Turn's thinking, ranked among turns on the same level", ylabel="Share of turns that level up",
          title=f"(b) Chance the turn clears the level\n({nlev} levels with ≥5 turns, 3 runs)")
ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
print("Level-up rate by within-level thinking quintile:", r["mean"].round(3).tolist())

# (c) within-game z of log thinking around each level-up
rows = []
for _, y in d.groupby(["run", "game"]):
    y = y.reset_index(drop=True)
    for i in y.index[y.level_up == 1]:
        for k in range(-6, 4):
            if 0 <= i + k < len(y):
                rows.append(dict(k=k, z=y.z[i + k]))
e = pd.DataFrame(rows).groupby("k").z.agg(["mean", "sem"])
ax[2].axhline(0, color=MUTED, lw=1)
ax[2].axvspan(.5, 3.5, color=GRID, alpha=.6, lw=0)
ax[2].text(2, e["mean"].max() + .12, "next level", ha="center", color=MUTED, fontsize=8.5)
ax[2].fill_between(e.index, e["mean"] - 1.96 * e["sem"], e["mean"] + 1.96 * e["sem"], color=UP, alpha=.18, lw=0)
ax[2].plot(e.index, e["mean"], color=UP, lw=2, marker="o", ms=7)
ax[2].annotate("level-up turn", (0, e.loc[0, "mean"]), (-3.6, -.45), fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=.8))
ax[2].set(xticks=range(-6, 4), xlabel="Turns relative to the level-up turn", ylabel="Thinking vs that game's average (SD)",
          title="(c) Thinking dips on the level-up turn,\nspikes on the new level")
print("Event-aligned mean z, k=-6..3:", e["mean"].round(2).tolist())
for a in ax:
    a.grid(color=GRID, lw=.6)
    a.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG, dpi=150)
