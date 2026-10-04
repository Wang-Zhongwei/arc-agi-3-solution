"""Per-turn thinking tokens over a game, with level regions shaded.

A turn is the model calls between two action emissions: exploratory python
calls, then the call that emits the action (the turn's last call). Plots both
the turn's total thinking and the last call's thinking, from exact
`usage.reasoning_tokens` in a run's request logs. Levels and level-ups come
from data/games_turns.csv (written by thinking_vs_level_up_games.py).

Needs `*_p0_requests.jsonl` under ARC3-Inference/runs/kaggle (or ARC3_RUNS_DIR);
the duck-harness copy of the run has them.

    python3 docs/reasoning/scripts/thinking_trace.py
"""
import json
import os
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORT = Path(__file__).resolve().parents[1]
K = Path(os.environ.get("ARC3_RUNS_DIR", REPORT.parents[1] / "ARC3-Inference/runs")) / "kaggle"
RUN, RUN_DIR = "daniel_v05", K / "dangthuygiang/arc3-v207-danibase-base-dfranzen/v05"
GAMES = ["ar25", "lp85", "vc33", "ls20", "su15", "lf52"]
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TURN, LAST = "#9fc3ee", "#2a78d6"
SHADES = ["#fcfcfb", "#efeeea"]


def last_call_thinking(game):
    """Reasoning tokens of each step's last response (the action-emitting call)."""
    path = next(RUN_DIR.glob(f"{game}-*_p0_requests.jsonl"))
    last = {}
    for line in open(path):
        r = json.loads(line)
        if r["event"] == "response":
            last[r["analysis_step"]] = (r.get("usage") or {}).get("reasoning_tokens") or 0
    return pd.Series(last, name="last")


plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
d = pd.read_csv(REPORT / "data/games_turns.csv")
d = d[d.run == RUN]
fig, axes = plt.subplots(len(GAMES), 1, figsize=(14, 2.3 * len(GAMES)), sharey=True)
for ax, game in zip(axes, GAMES):
    y = d[d.game == game].set_index("step").join(last_call_thinking(game)).reset_index()
    y["turn"] = y.index + 1
    for i, (lv, seg) in enumerate(y.groupby("level")):
        lo, hi = seg.turn.min() - .5, seg.turn.max() + .5
        ax.axvspan(lo, hi, color=SHADES[i % 2], lw=0)
        ax.text((lo + hi) / 2, 2.2e4, f"L{lv}", ha="center", va="top", color=MUTED, fontsize=8.5)
    ax.plot(y.turn, y.think.clip(lower=1), color=TURN, lw=1.2, marker="o", ms=3, label="whole turn")
    ax.plot(y.turn, y["last"].clip(lower=1), color=LAST, lw=1.2, marker="o", ms=3.5, label="last call (emits the action)")
    up = y[y.level_up == 1]
    ax.scatter(up.turn, up["last"].clip(lower=1), s=60, facecolor="none", edgecolor=INK, lw=1.4, zorder=4,
               label="level-up turn")
    ax.axhline(2048, color=MUTED, lw=1, ls="--")
    ax.set(yscale="log", ylim=(5, 3e4), xlim=(0.5, len(y) + .5))
    ax.set_title(f"{game}: {len(y)} turns, {int(y.level_up.sum())} levels cleared; median last call "
                 f"{y['last'].median():,.0f}, whole turn {y.think.median():,.0f} tokens", loc="left", fontsize=10)
    ax.grid(axis="y", color=GRID, lw=.6)
axes[0].legend(frameon=False, fontsize=8.5, loc="lower right", ncol=3)
axes[0].text(0.6, 2048 * 1.15, "2048", color=MUTED, fontsize=8)
axes[-1].set_xlabel("Turn (one action emission each)")
fig.supylabel("Thinking tokens (log scale)", color=INK)
fig.tight_layout()
fig.savefig(REPORT / "figures/thinking_trace_daniel_v05.png", dpi=150)

y = d.set_index(["game", "step"])
calls = pd.concat({g: last_call_thinking(g) for g in d.game.unique()}).rename_axis(["game", "step"])
both = y.join(calls)
print(f"{RUN}: last call is {both['last'].sum() / both.think.sum():.0%} of all thinking; "
      f"median last call {both['last'].median():.0f} vs whole turn {both.think.median():.0f} tokens")
