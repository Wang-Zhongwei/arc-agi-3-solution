"""Generated tokens our runs spent on levels 1-2, per game, against a GRPO rollout budget.

Budget per game = min(CAP, PER_HUMAN_ACTION x (human actions of L1 + L2)). An attempt is one game
of one run. A cleared attempt is counted at the L2 clear; an attempt that never cleared L2 is
counted at the end of its game (what it spent, a lower bound on what clearing would have taken).

Reads evaluation.json (per-level progression) and benchmark.json (human action counts) of the
25-game Kaggle runs under RUNS_DIR. Writes figures into docs/RL/images/ and prints the tables.

    python docs/RL/scripts/l1_l2_token_budget.py [RUNS_DIR]
"""

import collections
import glob
import json
import os
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUNS_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "ARC3-Inference", "runs", "kaggle")
IMAGES = os.path.join(REPO, "docs", "RL", "images")
CAP = 100_000
PER_HUMAN_ACTION = 2_000
MIN_RUN_SCORE = 30  # drops broken runs (crashed servers, 5-minute smoke runs)

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CLEARED, NOT_CLEARED = "#2a78d6", "#eb6834"


def load_runs():
    runs = []
    for path in sorted(glob.glob(os.path.join(RUNS_DIR, "*", "*", "v*", "evaluation.json"))):
        run_dir = os.path.dirname(path)
        evaluation = json.load(open(path))
        if len(evaluation.get("games", {})) != 25 or evaluation["score"] < MIN_RUN_SCORE:
            continue
        if "/rft-model/" in run_dir:  # a different model
            continue
        runs.append((os.path.relpath(run_dir, RUNS_DIR), evaluation))
    return runs


def human_actions():
    benchmark = json.load(open(os.path.join(RUNS_DIR, "ticbudja", "reset-daniel-context", "v01", "benchmark.json")))
    return {g["game_id"][:4]: g["base_actions_per_level"] for g in benchmark["game_runs"]}


def attempts(runs):
    """game -> list of dicts: cleared (L1+L2), tokens, tokens at L1 clear, tokens on L2, levels."""
    out = collections.defaultdict(list)
    per_model_action = []
    for _, evaluation in runs:
        totals = evaluation["usage"]["totals"]
        per_model_action.append(totals["generated_tokens"] / totals["actions"])
        lanes = evaluation["progression_lanes"]["tokens"]
        for game, record in evaluation["games"].items():
            tokens = [point[0] for point in next(iter(lanes[game].values()))]  # cumulative, per level clear + end
            levels = int(record["levels_completed_mean"])
            out[game[:4]].append({
                "cleared": levels >= 2,
                "tokens": tokens[2] if levels >= 2 else tokens[-1],
                "l1": tokens[1] if levels >= 1 else None,
                "l1_spent": tokens[1] if levels >= 1 else tokens[-1],
                "l2_spent": (tokens[2] if levels >= 2 else tokens[-1]) - tokens[1] if levels >= 1 else None,
                "levels": levels,
            })
    return out, per_model_action


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def strip(ax, order, values, ticks, xmax, title, note=True, label_rows=True):
    """One row per game; filled dots = cleared, crosses = not cleared; dashed cap; per-game budget ticks.

    ticks: list of (game -> tokens, color, half-height). A tick past xmax is drawn at the edge as a
    triangle labelled with its value.
    """
    for row, game in enumerate(order):
        y = len(order) - 1 - row
        jitter = [((i * 7919) % 11 - 5) / 22 for i in range(len(values[game]))]
        cleared = [(v, y + j) for (v, ok), j in zip(values[game], jitter) if ok and v is not None]
        missed = [(v, y + j) for (v, ok), j in zip(values[game], jitter) if not ok and v is not None]
        if cleared:
            ax.scatter([min(v, xmax) for v, _ in cleared], [b for _, b in cleared], s=16, color=CLEARED,
                       edgecolors=SURFACE, linewidths=0.6, zorder=3)
        if missed:
            ax.scatter([min(v, xmax) for v, _ in missed], [b for _, b in missed], s=22, color=NOT_CLEARED,
                       marker="x", linewidths=1.4, zorder=3)
        for k, (tokens, color, half) in enumerate(ticks or []):
            value = tokens[game]
            if value <= xmax:
                ax.plot([value] * 2, [y - half, y + half], color=color, linewidth=2.4, zorder=4,
                        solid_capstyle="butt")
            else:
                ax.scatter([xmax], [y + (0.22 if k == 0 else -0.22)], marker=">", s=26, color=color, zorder=4,
                           clip_on=False)
                ax.text(xmax * 0.985, y + (0.22 if k == 0 else -0.22), f"{value / 1000:.0f}K", color=color,
                        fontsize=7.5, ha="right", va="center", zorder=4)
    ax.axvline(CAP, color=INK, linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
    if note:
        ax.text(CAP, len(order) - 0.3, " 100K cap", color=INK, fontsize=9, va="bottom")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(list(reversed(order)), fontsize=9, color=INK)
    ax.tick_params(axis="y", labelleft=label_rows)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlim(0, xmax * 1.02)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=14)
    style(ax)


TICK_1000, TICK_2000 = INK, "#9a988f"


def legend(fig, ticks=True):
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=CLEARED, markersize=6, label="cleared"),
        plt.Line2D([], [], marker="x", linestyle="", color=NOT_CLEARED, markersize=7, markeredgewidth=1.4,
                   label="not cleared (spent until the game ended)"),
        plt.Line2D([], [], linestyle=(0, (4, 3)), color=INK, label="100K cap"),
    ]
    if ticks:
        handles += [
            plt.Line2D([], [], marker="|", linestyle="", color=TICK_1000, markersize=11, markeredgewidth=2.4,
                       label="1,000 × human actions"),
            plt.Line2D([], [], marker="|", linestyle="", color=TICK_2000, markersize=7, markeredgewidth=2.4,
                       label="2,000 × human actions"),
        ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.995), ncol=len(handles), frameon=False,
               fontsize=9, labelcolor=MUTED, handletextpad=0.3, columnspacing=1.4)


def main():
    runs = load_runs()
    humans = human_actions()
    data, per_model_action = attempts(runs)
    games = sorted(data)
    h12 = {g: humans[g][0] + humans[g][1] for g in games}
    rules = {  # name -> per-game budget
        "1,000·h": {g: 1_000 * h12[g] for g in games},
        "min(100K, 1,000·h)": {g: min(CAP, 1_000 * h12[g]) for g in games},
        "min(100K, 2,000·h)": {g: min(CAP, PER_HUMAN_ACTION * h12[g]) for g in games},
    }
    within = {name: {g: sum(a["cleared"] and a["tokens"] <= budget[g] for a in data[g]) / len(data[g])
                     for g in games} for name, budget in rules.items()}
    order = sorted(games, key=lambda g: st.median(a["tokens"] for a in data[g]))
    os.makedirs(IMAGES, exist_ok=True)
    xmax = 300_000

    # Figure 1: tokens to clear L1 + L2, with both budget ticks for every game
    fig, ax = plt.subplots(figsize=(12, 8.8), facecolor=SURFACE)
    ticks = [({g: 1_000 * h12[g] for g in games}, TICK_1000, 0.42),
             ({g: PER_HUMAN_ACTION * h12[g] for g in games}, TICK_2000, 0.26)]
    strip(ax, order, {g: [(a["tokens"], a["cleared"]) for a in data[g]] for g in games}, ticks, xmax,
          f"Generated tokens to clear levels 1-2, per game ({len(runs)} runs; values past 300K drawn at 300K)")
    for col, name in enumerate(rules):
        x = xmax * (1.05 + 0.115 * col)
        ax.text(x, len(order) - 0.3, name.replace("min(100K, ", "min(100K,\n"), fontsize=7.5, color=MUTED,
                va="bottom", ha="center")
        for row, game in enumerate(order):
            ax.text(x, len(order) - 1 - row, f"{within[name][game]:.0%}", fontsize=9, color=MUTED, va="center",
                    ha="center")
    ax.text(xmax * 1.165, len(order) + 1.15, "within budget", fontsize=8, color=MUTED, ha="center")
    legend(fig)
    fig.tight_layout(rect=(0, 0, 0.83, 0.96))
    fig.savefig(os.path.join(IMAGES, "l1_l2_tokens_per_game.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Figure 2: L1 and L2 separately (L2 only for attempts that reached it)
    fig, axes = plt.subplots(1, 2, figsize=(12, 8.6), facecolor=SURFACE)
    strip(axes[0], order, {g: [(a["l1_spent"], a["levels"] >= 1) for a in data[g]] for g in games}, None, 200_000,
          "Level 1", note=True)
    strip(axes[1], order, {g: [(a["l2_spent"], a["cleared"]) for a in data[g] if a["l2_spent"] is not None]
                           for g in games}, None, 200_000, "Level 2 (attempts that reached it)", note=True,
          label_rows=False)
    legend(fig, ticks=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(IMAGES, "l1_l2_tokens_by_level.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Figure 3: tokens per human action for cleared L1 + L2
    per_human = {g: [a["tokens"] / h12[g] for a in data[g] if a["cleared"]] for g in games}
    order3 = sorted(games, key=lambda g: st.median(per_human[g]) if per_human[g] else 0)
    fig, ax = plt.subplots(figsize=(10, 8.6), facecolor=SURFACE)
    strip(ax, order3, {g: [(v, True) for v in per_human[g]] for g in games}, None, 4_000,
          "Generated tokens per human action, cleared levels 1-2 (values past 4,000 drawn at 4,000)", note=False)
    for ref in (1_000, 2_000):
        ax.axvline(ref, color=INK, linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
        ax.text(ref, len(order3) - 0.3, f" {ref:,}", color=INK, fontsize=9, va="bottom")
    ax.axvline(CAP, color=SURFACE, linewidth=0)  # the cap line is off-scale here
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(IMAGES, "l1_l2_tokens_per_human_action.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Tables
    print(f"runs: {len(runs)}; attempts: {sum(len(data[g]) for g in games)}")
    print(f"generated tokens per model action, run totals: median {st.median(per_model_action):,.0f}, "
          f"range {min(per_model_action):,.0f}-{max(per_model_action):,.0f}")
    pooled = [v for g in games for v in per_human[g]]
    q = st.quantiles(pooled, n=10)
    print(f"tokens per human action, cleared L1+L2, pooled: median {st.median(pooled):,.0f}, p10 {q[0]:,.0f}, "
          f"p90 {q[8]:,.0f}; share in 1,000-2,000: {sum(1000 <= v <= 2000 for v in pooled) / len(pooled):.0%}, "
          f"below 1,000: {sum(v < 1000 for v in pooled) / len(pooled):.0%}")
    total = sum(len(data[g]) for g in games)
    print(f"cleared L1+L2: {sum(a['cleared'] for g in games for a in data[g]) / total:.0%}")
    for name, budget in rules.items():
        inside = sum(a["cleared"] and a["tokens"] <= budget[g] for g in games for a in data[g]) / total
        mixed = [g for g in games if 0.2 <= within[name][g] <= 0.8]
        print(f"{name}: within {inside:.0%}; games 20-80%: {len(mixed)} {mixed}; >=95%: "
              f"{sum(within[name][g] >= 0.95 for g in games)}; <=5%: {sum(within[name][g] <= 0.05 for g in games)}")
    print()
    names = list(rules)
    print("| Game | Human actions L1 + L2 | 1,000·h | " + " | ".join(f"Within {n}" for n in names)
          + " | Cleared at all | Median tokens (cleared) | p90 tokens (cleared) | Tokens per human action (median) |")
    print("|---|---:|---:|" + "---:|" * len(names) + "---:|---:|---:|---:|")
    for g in sorted(games, key=lambda g: -within[names[0]][g]):
        cleared = [a["tokens"] for a in data[g] if a["cleared"]]
        p90 = st.quantiles(cleared, n=10)[8] if len(cleared) >= 10 else max(cleared)
        print(f"| {g} | {humans[g][0]} + {humans[g][1]} | {1000 * h12[g] / 1000:.0f}K | "
              + " | ".join(f"{within[n][g]:.0%}" for n in names)
              + f" | {len(cleared) / len(data[g]):.0%} | {st.median(cleared) / 1000:.0f}K | {p90 / 1000:.0f}K | "
              f"{st.median(per_human[g]):,.0f} |")

if __name__ == "__main__":
    main()
