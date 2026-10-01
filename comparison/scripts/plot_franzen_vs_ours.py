"""Score-vs-tokens / score-vs-actions for Franzen v01 vs our v30 and v11, restricted to Franzen's 10 games.

Pooling mirrors viewer/curves.html `pool()`: every lane (one trial of one game) is
weighted 1/passes of its game; at each knot t the x value is the weighted mean of
min(t, lane_end). The mean row is the weighted mean step value over all lanes; the
median row is the median across games of each game's weighted score at that cost.
"""
import json
import statistics
import sys
from bisect import bisect_right
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/wang/Documents/arc3/duck-harness/ARC3-Inference/runs/kaggle")
# (label, evaluation.json, colour, leaderboard score)
RUNS = [
    ("Franzen v01", ROOT / "dfranzen/arc-agi-3-milestone-2-solution/v01/evaluation.json", "#2f6fdb", 27.89),
    ("v30", ROOT / "jonathanwang2022/adaptive-reasoning-ucb/v30/evaluation.json", "#e4572e", 9.8),
    ("v11", ROOT / "jonathanwang2022/adaptive-reasoning-ucb/v11/evaluation.json", "#1a9e5f", 11.64),
]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "figures" / "franzen_vs_ours_10games.png"))

BG_PAGE, BG_CARD, GRID, INK, QUIET = "#ece9df", "#fbf8ee", "#e6e1d3", "#1d1d1b", "#6b6b66"


def step_value(points, x):
    i = bisect_right([p[0] for p in points], x + 1e-9) - 1
    return points[i][1] if i >= 0 else 0.0


def pool(lanes):
    knots = sorted({0.0} | {p[0] for lane in lanes for p in lane["points"]})
    total = sum(lane["weight"] for lane in lanes)
    games = list(dict.fromkeys(lane["game"] for lane in lanes))
    mean, median = [], []
    for t in knots:
        cost = score = 0.0
        per_game = dict.fromkeys(games, 0.0)
        for lane in lanes:
            end = lane["points"][-1][0]
            tg = min(t, end)
            value = step_value(lane["points"], tg)
            cost += tg * lane["weight"]
            score += value * lane["weight"]
            per_game[lane["game"]] += value * lane["weight"]
        x = cost / total
        mean.append((x, score / total))
        median.append((x, statistics.median(per_game.values())))
    return {"mean": mean, "median": median}


def run_curves(evaluation, axis, games):
    lane_map = evaluation["progression_lanes"][axis]
    lanes = []
    for game in games:
        trials = list(evaluation["games"][game]["trial_scores"])
        for trial in trials:
            lanes.append({"points": lane_map[game][trial], "game": game, "weight": 1 / len(trials)})
    return pool(lanes)


evaluations = [(name, json.loads(path.read_text()), color, lb) for name, path, color, lb in RUNS]
games = sorted(evaluations[0][1]["games"])  # Franzen's 10 games
for name, evaluation, _, _ in evaluations[1:]:
    missing = set(games) - set(evaluation["games"])
    assert not missing, f"{name} lacks {missing}"

curves = {(name, axis): run_curves(evaluation, axis, games)
          for name, evaluation, _, _ in evaluations for axis in ("tokens", "actions")}

fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=BG_PAGE)
columns = [("tokens", "GENERATED TOKENS", "Mean generated tokens / game trial"),
           ("actions", "ACTIONS", "Mean actions / game trial")]
rows = [("mean", "Mean"), ("median", "Median")]
for row, (stat, stat_label) in enumerate(rows):
    for col, (axis, axis_title, xlabel) in enumerate(columns):
        ax = axes[row][col]
        ax.set_facecolor(BG_CARD)
        drawn = []
        for name, _, color, lb in evaluations:
            curve = curves[(name, axis)][stat]
            xs, ys = zip(*curve)
            ax.step(xs, ys, where="post", color=color, linewidth=2.0)
            ax.plot(xs[-1], ys[-1], "o", color=color, markersize=5)
            drawn.append((name, color, lb, curve))
        xmax = max(c[-1][0] for *_, c in drawn) * 1.04
        ymax = max(max(y for _, y in c) for *_, c in drawn)
        # endpoint labels in the right margin, as on the viewer
        label_x = xmax * 1.02
        last_y = None
        for name, color, lb, curve in sorted(drawn, key=lambda item: -item[3][-1][1]):
            ex, ey = curve[-1]
            gap = ymax * 0.11  # two-line labels
            ly = ey if last_y is None or last_y - ey > gap else last_y - gap
            last_y = ly
            ax.plot([ex, label_x], [ey, ly], color=color, linewidth=0.8, alpha=0.7, clip_on=False)
            ax.text(label_x * 1.01, ly, f"{ey:.1f} {name}", color=INK, fontsize=10,
                    fontweight="bold", va="bottom", clip_on=False)
            ax.text(label_x * 1.01, ly, f"LB {lb:g}", color=color, fontsize=9,
                    va="top", clip_on=False)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax * 1.12)
        ax.grid(axis="y", color=GRID, linewidth=1)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#b9b5a8")
        ax.tick_params(colors=QUIET, labelsize=9, length=0)
        if axis == "tokens":
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: "0" if v == 0 else f"{v / 1000:.0f}k"))
        ax.set_xlabel(xlabel, color=QUIET, fontsize=10)
        ax.set_title(f"{stat_label.upper()} SCORE VS {axis_title}", loc="left", fontsize=13,
                     fontweight="bold", color=INK, pad=24)
        ax.text(0, 1.035, f"{stat_label} game score over Franzen's {len(games)} games (1 trial each).",
                transform=ax.transAxes, color=QUIET, fontsize=10)

fig.subplots_adjust(left=0.04, right=0.87, wspace=0.42, hspace=0.42, top=0.93, bottom=0.08)
fig.text(0.04, 0.012, "Games: " + ", ".join(g.split("-")[0] for g in games)
         + ".  Franzen v01 = dfranzen/arc-agi-3-milestone-2-solution v1 (25 min public run);"
         " v30, v11 = jonathanwang2022/adaptive-reasoning-ucb.\nLB = Kaggle leaderboard score.",
         color=QUIET, fontsize=9)
fig.savefig(OUT, dpi=160, facecolor=BG_PAGE)
print("saved", OUT)

# Text summary: each baseline read at Franzen's endpoint, per statistic and axis.
franzen = evaluations[0][0]
for name, *_ in evaluations[1:]:
    for stat, _ in rows:
        for axis, *_ in columns:
            f_curve, b_curve = curves[(franzen, axis)][stat], curves[(name, axis)][stat]
            fx, fy = f_curve[-1]
            i = bisect_right([p[0] for p in b_curve], fx) - 1
            print(f"{name} {stat} {axis}: Franzen end x={fx:,.0f} score={fy:.2f}; "
                  f"{name} at same x={b_curve[i][1]:.2f}; end x={b_curve[-1][0]:,.0f} score={b_curve[-1][1]:.2f}")
