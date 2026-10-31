"""Context tokens our runs reached on levels 1-2, per game, against a GRPO rollout budget.

GRPO rollouts on levels 1-2 never trim context, so the budget caps the whole sequence: system
prompt, observations, tool results, images and generated tokens. Budget per game =
min(CAP, PER_HUMAN_ACTION x (human actions of L1 + L2)) context tokens. An attempt is one game of
one run. A cleared attempt is counted at the L2 clear; an attempt that never cleared L2 is counted
at the end of its game (a lower bound on what clearing would have taken).

Reads docs/RL/data/l1_l2_context.json (untrimmed context per attempt, written by
l1_l2_context_tokens.py from the runs' transcripts), benchmark.json (human action counts) and
evaluation.json (generated tokens, for the earlier generated-token rule) of the 25-game Kaggle runs
under RUNS_DIR. Writes figures into docs/RL/images/ and prints the tables.

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
CONTEXT = os.path.join(REPO, "docs", "RL", "data", "l1_l2_context.json")
CAP = 130_000  # context tokens; the trainer fits ~133K on one H200
PER_HUMAN_ACTION = 1_750  # chosen: most games with mixed outcomes under the cap
ALTERNATIVE = 1_500
OLD_CAP, OLD_PER_HUMAN_ACTION = 100_000, 1_000  # the earlier rule, on generated tokens
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
    """game -> list of dicts: cleared (L2), tokens (context at the L2 clear, else at the game's end),
    l1_spent (context at the L1 clear, else at the end), l2_spent (context added on L2), levels, and
    generated (generated tokens to the L2 clear or the end, from evaluation.json)."""
    generated = {}
    for run, evaluation in runs:
        lanes = evaluation["progression_lanes"]["tokens"]
        for game, record in evaluation["games"].items():
            tokens = [point[0] for point in next(iter(lanes[game].values()))]  # cumulative, per level clear + end
            generated[(run, game[:4])] = tokens[2] if int(record["levels_completed_mean"]) >= 2 else tokens[-1]
    out = collections.defaultdict(list)
    for row in json.load(open(CONTEXT)):
        cleared = row["levels"] >= 2
        context = row["l2"] if cleared else row["end"]
        out[row["game"]].append({
            "cleared": cleared,
            "tokens": context,
            "l1_spent": row["l1"] if row["levels"] >= 1 else row["end"],
            "l2_spent": context - row["l1"] if row["levels"] >= 1 else None,
            "levels": row["levels"],
            "generated": generated[(row["run"], row["game"])],
        })
    return out


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
        ax.text(CAP, len(order) - 0.3, f" {CAP // 1000}K cap", color=INK, fontsize=9, va="bottom")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(list(reversed(order)), fontsize=9, color=INK)
    ax.tick_params(axis="y", labelleft=label_rows)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlim(0, xmax * 1.02)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=14)
    style(ax)


TICK_CHOSEN, TICK_ALTERNATIVE = INK, "#9a988f"


def legend(fig, ticks=True):
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=CLEARED, markersize=6, label="cleared"),
        plt.Line2D([], [], marker="x", linestyle="", color=NOT_CLEARED, markersize=7, markeredgewidth=1.4,
                   label="not cleared (spent until the game ended)"),
        plt.Line2D([], [], linestyle=(0, (4, 3)), color=INK, label=f"{CAP // 1000}K cap"),
    ]
    if ticks:
        handles += [
            plt.Line2D([], [], marker="|", linestyle="", color=TICK_CHOSEN, markersize=11, markeredgewidth=2.4,
                       label=f"{PER_HUMAN_ACTION:,} × human actions"),
            plt.Line2D([], [], marker="|", linestyle="", color=TICK_ALTERNATIVE, markersize=7, markeredgewidth=2.4,
                       label=f"{ALTERNATIVE:,} × human actions"),
        ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.995), ncol=len(handles), frameon=False,
               fontsize=9, labelcolor=MUTED, handletextpad=0.3, columnspacing=1.4)


def main():
    runs = load_runs()
    humans = human_actions()
    data = attempts(runs)
    games = sorted(data)
    h12 = {g: humans[g][0] + humans[g][1] for g in games}
    chosen = f"min({CAP // 1000}K, {PER_HUMAN_ACTION:,}·h)"
    rules = {  # name -> (per-game budget, measure)
        chosen: ({g: min(CAP, PER_HUMAN_ACTION * h12[g]) for g in games}, "tokens"),
        f"min({CAP // 1000}K, {ALTERNATIVE:,}·h)": ({g: min(CAP, ALTERNATIVE * h12[g]) for g in games}, "tokens"),
        f"{PER_HUMAN_ACTION:,}·h, uncapped": ({g: PER_HUMAN_ACTION * h12[g] for g in games}, "tokens"),
        f"earlier: min({OLD_CAP // 1000}K, {OLD_PER_HUMAN_ACTION:,}·h) generated":
            ({g: min(OLD_CAP, OLD_PER_HUMAN_ACTION * h12[g]) for g in games}, "generated"),
    }
    within = {name: {g: sum(a["cleared"] and a[key] <= budget[g] for a in data[g]) / len(data[g]) for g in games}
              for name, (budget, key) in rules.items()}
    order = sorted(games, key=lambda g: st.median(a["tokens"] for a in data[g]))
    os.makedirs(IMAGES, exist_ok=True)
    xmax = 300_000
    n_runs = len({row["run"] for row in json.load(open(CONTEXT))})  # runs with transcripts

    # Figure 1: context at the L2 clear, with both budget ticks for every game
    fig, ax = plt.subplots(figsize=(13.4, 9.0), facecolor=SURFACE)
    ticks = [({g: PER_HUMAN_ACTION * h12[g] for g in games}, TICK_CHOSEN, 0.42),
             ({g: ALTERNATIVE * h12[g] for g in games}, TICK_ALTERNATIVE, 0.26)]
    title = f"Context tokens at the level-2 clear, per game ({n_runs} runs, no trimming; values past 300K drawn at 300K)"
    strip(ax, order, {g: [(a["tokens"], a["cleared"]) for a in data[g]] for g in games}, ticks, xmax, title)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=44)
    labels = [f"min({CAP // 1000}K,\n{PER_HUMAN_ACTION:,}·h)", f"min({CAP // 1000}K,\n{ALTERNATIVE:,}·h)",
              f"{PER_HUMAN_ACTION:,}·h,\nuncapped"]  # context rules only; the earlier rule is in the tables
    for col, (name, label) in enumerate(zip(rules, labels)):
        x = xmax * (1.07 + 0.125 * col)
        ax.text(x, len(order) - 0.3, label, fontsize=7.5, color=MUTED, va="bottom", ha="center")
        for row, game in enumerate(order):
            ax.text(x, len(order) - 1 - row, f"{within[name][game]:.0%}", fontsize=9,
                    color=INK if col == 0 else MUTED, va="center", ha="center")
    ax.text(xmax * 1.195, len(order) + 1.0, "within budget", fontsize=8, color=MUTED, ha="center")
    legend(fig)
    fig.tight_layout(rect=(0, 0, 0.8, 0.96))
    fig.savefig(os.path.join(IMAGES, "l1_l2_tokens_per_game.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Figure 2: L1 and L2 separately (L2 only for attempts that reached it)
    fig, axes = plt.subplots(1, 2, figsize=(12, 8.6), facecolor=SURFACE)
    strip(axes[0], order, {g: [(a["l1_spent"], a["levels"] >= 1) for a in data[g]] for g in games}, None, 200_000,
          "Level 1: context at the clear", note=True)
    strip(axes[1], order, {g: [(a["l2_spent"], a["cleared"]) for a in data[g] if a["l2_spent"] is not None]
                           for g in games}, None, 200_000, "Level 2: context added (attempts that reached it)",
          note=True, label_rows=False)
    legend(fig, ticks=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(IMAGES, "l1_l2_tokens_by_level.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Figure 3: context per human action for cleared L1 + L2
    per_human = {g: [a["tokens"] / h12[g] for a in data[g] if a["cleared"]] for g in games}
    order3 = sorted(games, key=lambda g: st.median(per_human[g]) if per_human[g] else 0)
    fig, ax = plt.subplots(figsize=(10, 8.6), facecolor=SURFACE)
    strip(ax, order3, {g: [(v, True) for v in per_human[g]] for g in games}, None, 6_000,
          "Context tokens per human action, cleared levels 1-2 (values past 6,000 drawn at 6,000)", note=False)
    for ref in (ALTERNATIVE, PER_HUMAN_ACTION):
        ax.axvline(ref, color=INK if ref == PER_HUMAN_ACTION else TICK_ALTERNATIVE, linestyle=(0, (4, 3)),
                   linewidth=1.2, zorder=2)
        ax.text(ref, len(order3) - 0.3, f" {ref:,}", color=INK, fontsize=9, va="bottom",
                ha="left" if ref == PER_HUMAN_ACTION else "right")
    ax.axvline(CAP, color=SURFACE, linewidth=0)  # the cap line is off-scale here
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(IMAGES, "l1_l2_tokens_per_human_action.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Tables
    total = sum(len(data[g]) for g in games)
    cleared_all = [a for g in games for a in data[g] if a["cleared"]]
    print(f"runs: {n_runs}; attempts: {total}")
    ratio = [a["tokens"] / a["generated"] for a in cleared_all]
    q = st.quantiles(ratio, n=10)
    print(f"context / generated at the L2 clear: median {st.median(ratio):.2f}, p10 {q[0]:.2f}, p90 {q[8]:.2f}")
    pooled = [v for g in games for v in per_human[g]]
    q = st.quantiles(pooled, n=10)
    print(f"context per human action, cleared L1+L2, pooled: median {st.median(pooled):,.0f}, p10 {q[0]:,.0f}, "
          f"p90 {q[8]:,.0f}")
    print(f"cleared L1+L2: {len(cleared_all) / total:.0%}; cleared with context > {CAP // 1000}K: "
          f"{sum(a['tokens'] > CAP for a in cleared_all) / len(cleared_all):.0%}")
    for name, (budget, key) in rules.items():
        inside = sum(a["cleared"] and a[key] <= budget[g] for g in games for a in data[g]) / total
        mixed = [g for g in games if 0.2 <= within[name][g] <= 0.8]
        print(f"{name}: within {inside:.0%}; mean budget {st.mean(budget.values()) / 1000:.0f}K; games 20-80%: "
              f"{len(mixed)} {mixed}; >=95%: {[g for g in games if within[name][g] >= 0.95]}; "
              f"<=5%: {[g for g in games if within[name][g] <= 0.05]}")
    print("cap binds on:", [g for g in games if PER_HUMAN_ACTION * h12[g] > CAP])
    for k in (1_000, 1_250, 1_500, 1_750, 2_000, 2_500):
        b = {g: min(CAP, k * h12[g]) for g in games}
        w = {g: sum(a["cleared"] and a["tokens"] <= b[g] for a in data[g]) / len(data[g]) for g in games}
        inside = sum(a["cleared"] and a["tokens"] <= b[g] for g in games for a in data[g]) / total
        print(f"| min({CAP // 1000}K, {k:,}·h) | {inside:.0%} | {sum(0.2 <= w[g] <= 0.8 for g in games)} | "
              f"{sum(w[g] >= 0.95 for g in games)} | {sum(w[g] <= 0.05 for g in games)} | "
              f"{sum(k * h12[g] > CAP for g in games)} |")
    print()
    names = list(rules)
    print(f"| Game | Human actions L1 + L2 | Budget | " + " | ".join(f"Within {n}" for n in names)
          + " | Cleared at all | Median context (cleared) | p90 context (cleared) "
            "| Context per human action (median) | Context / generated (median) |")
    print("|---|---:|---:|" + "---:|" * len(names) + "---:|---:|---:|---:|---:|")
    for g in sorted(games, key=lambda g: -within[names[0]][g]):
        cleared = [a for a in data[g] if a["cleared"]]
        values = [a["tokens"] for a in cleared]
        p90 = st.quantiles(values, n=10)[8] if len(values) >= 10 else max(values)
        print(f"| {g} | {humans[g][0]} + {humans[g][1]} | {rules[chosen][0][g] / 1000:.0f}K | "
              + " | ".join(f"{within[n][g]:.0%}" for n in names)
              + f" | {len(cleared) / len(data[g]):.0%} | {st.median(values) / 1000:.0f}K | {p90 / 1000:.0f}K | "
              f"{st.median(per_human[g]):,.0f} | {st.median(a['tokens'] / a['generated'] for a in cleared):.2f} |")


if __name__ == "__main__":
    main()
