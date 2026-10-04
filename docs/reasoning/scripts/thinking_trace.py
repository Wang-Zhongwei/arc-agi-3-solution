"""Per-turn thinking tokens over a game, with level regions shaded.

Reads data/games_turns.csv (exact `usage.reasoning_tokens`, written by
thinking_vs_level_up_games.py) and plots a few Daniel v05 games.

    python3 docs/reasoning/scripts/thinking_trace.py
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORT = Path(__file__).resolve().parents[1]
RUN = "daniel_v05"
GAMES = ["ar25", "lp85", "vc33", "ls20", "su15", "lf52"]
INK, MUTED, GRID, SURF, LINE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"
SHADES = ["#fcfcfb", "#efeeea"]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
d = pd.read_csv(REPORT / "data/games_turns.csv")
d = d[d.run == RUN]
fig, axes = plt.subplots(len(GAMES), 1, figsize=(14, 2.3 * len(GAMES)), sharey=True)
for ax, game in zip(axes, GAMES):
    y = d[d.game == game].reset_index(drop=True)
    y["turn"] = y.index + 1
    for i, (lv, seg) in enumerate(y.groupby("level")):
        lo, hi = seg.turn.min() - .5, seg.turn.max() + .5
        ax.axvspan(lo, hi, color=SHADES[i % 2], lw=0)
        ax.text((lo + hi) / 2, 2.2e4, f"L{lv}", ha="center", va="top", color=MUTED, fontsize=8.5)
    ax.plot(y.turn, y.think.clip(lower=1), color=LINE, lw=1.2)
    ax.scatter(y.turn, y.think.clip(lower=1), s=14, color=LINE, zorder=3)
    up = y[y.level_up == 1]
    ax.scatter(up.turn, up.think.clip(lower=1), s=60, facecolor="none", edgecolor=INK, lw=1.4, zorder=4,
               label="level-up turn")
    ax.axhline(2048, color=MUTED, lw=1, ls="--")
    ax.set(yscale="log", ylim=(5, 3e4), xlim=(0.5, len(y) + .5))
    ax.set_title(f"{game}: {len(y)} turns, {int(y.level_up.sum())} levels cleared, "
                 f"median {y.think.median():,.0f} tokens", loc="left", fontsize=10)
    ax.grid(axis="y", color=GRID, lw=.6)
axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
axes[0].text(0.6, 2048 * 1.15, "2048", color=MUTED, fontsize=8)
axes[-1].set_xlabel("Turn")
fig.supylabel("Thinking tokens in the turn (log scale)", color=INK)
fig.tight_layout()
fig.savefig(REPORT / "figures/thinking_trace_daniel_v05.png", dpi=150)
