"""Thinking-token statistics for Daniel's base notebook (v207 danibase, v05).

Reads usage.reasoning_tokens from every model response in the per-game request
logs and writes thinking_budget.csv, the per-call histogram, and the two scatter plots used in
thinking-budget.md.
"""

import csv
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT = Path(__file__).resolve().parents[1]
FIGURES, DATA = REPORT / "figures", REPORT / "data"
RUN = (REPORT.parents[2] / "duck-harness/ARC3-Inference/runs/kaggle/dangthuygiang"
       / "arc3-v207-danibase-base-dfranzen/v05")

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
# Label offsets (points) for games whose markers sit close together.
LABEL_OFFSETS = {"sb26": (-6, -13), "lp85": (5, 4), "wa30": (-30, -13), "cn04": (5, 4),
                 "ar25": (-6, -13)}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def summary_scores():
    games = {}
    for line in (RUN / "summary.txt").read_text().splitlines():
        if "score=" not in line:
            continue
        game, rest = line.strip().split(":", 1)
        kv = dict(part.split("=") for part in rest.replace(" ", "").split(","))
        done, total = kv["levels"].split("/")
        games[game] = dict(score=float(kv["score"]), levels=float(done),
                           total_levels=float(total), actions=int(kv["actions"]))
    return games


def call_tokens():
    tokens = []
    for path in sorted(glob.glob(str(RUN / "*_p0_requests.jsonl"))):
        for line in open(path):
            record = json.loads(line)
            if record["event"] == "response":
                tokens.append(record["usage"].get("reasoning_tokens") or 0)
    return tokens


def call_actions():
    """(thinking tokens, actions executed) for every model call.

    A call's actions are the jump in the `action` counter before the next
    request. The last call of a game gets whatever remains of the game's total.
    """
    totals = {game: s["actions"] for game, s in summary_scores().items()}
    pairs = []
    for path in sorted(glob.glob(str(RUN / "*_p0_requests.jsonl"))):
        game = Path(path).name.split("_p0")[0]
        records = [json.loads(line) for line in open(path)]
        calls = []
        for i, record in enumerate(records):
            if record["event"] != "response":
                continue
            following = next((r["action"] for r in records[i + 1:] if r["event"] == "request"), None)
            calls.append([record["usage"].get("reasoning_tokens") or 0,
                          None if following is None else following - record["action"]])
        if calls[-1][1] is None:
            calls[-1][1] = totals[game] - sum(a for _, a in calls[:-1])
        assert sum(a for _, a in calls) == totals[game], game
        pairs += calls
    return np.array(pairs)


def transcript_pairs(name):
    """(thinking tokens, actions) per call from data/<name>_calls.csv.

    Written by extract_transcript_calls.py for runs without request logs.
    """
    with open(DATA / f"{name}_calls.csv") as f:
        return np.array([[int(r["thinking_tokens"]), int(r["actions"])] for r in csv.DictReader(f)])


def actions_panel(ax, pairs, title):
    tokens, actions = pairs[:, 0], pairs[:, 1]
    r = np.corrcoef(tokens, actions)[0, 1]
    jitter = np.random.default_rng(0).uniform(-0.25, 0.25, len(actions))

    edges = np.logspace(np.log10(5), np.log10(20000), 14)
    centers, means = [], []
    for lo, hi in zip(edges, edges[1:]):
        inside = (tokens >= lo) & (tokens < hi)
        if inside.sum() >= 10:
            centers.append(np.sqrt(lo * hi))
            means.append(actions[inside].mean())

    ax.scatter(tokens, actions + jitter, s=8, color=BLUE, alpha=0.35, linewidth=0,
               zorder=3, label="One model call")
    ax.plot(centers, means, color=ORANGE, linewidth=2, marker="o", markersize=8,
            markeredgecolor="white", markeredgewidth=2, zorder=4,
            label="Mean actions per thinking bin")
    ax.set_xscale("log")
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["10", "100", "1k", "10k"])
    ax.set_yscale("symlog", linthresh=10)
    ax.set_yticks([0, 1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["0", "1", "2", "5", "10", "20", "50"])
    ax.set_xlabel("Thinking tokens in the call (log scale)", color=INK)
    ax.set_title(f"{title}\n{len(tokens):,} calls, Pearson r = {r:+.2f}",
                 color=INK, loc="left", fontsize=11)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    return r


def actions_scatter(panels, out):
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 5), dpi=150, sharex=True, sharey=True)
    top = max(pairs[:, 1].max() for _, pairs in panels)
    rs = [actions_panel(ax, pairs, title) for ax, (title, pairs) in zip(axes, panels)]
    axes[0].set_ylim(-0.5, top * 1.15)
    axes[0].set_ylabel("Actions executed right after", color=INK)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / out)
    plt.close(fig)
    return rs


def histogram(tokens, out):
    median, p90 = st.median(tokens), pct(tokens, .9)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    bins = np.logspace(np.log10(5), np.log10(13000), 41)
    ax.hist(tokens, bins=bins, color=BLUE, edgecolor="white", linewidth=1, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["10", "100", "1k", "10k"])
    for x, label in ((median, f"median {median:,.0f}"), (p90, f"p90 {p90:,}"),
                     (12288, "12.3k cap")):
        ax.axvline(x, color=MUTED, linewidth=1.2, linestyle="--", zorder=4)
        ax.annotate(label, (x, 1), xycoords=("data", "axes fraction"), xytext=(-4, -12),
                    textcoords="offset points", ha="right", fontsize=8, color=MUTED)
    ax.set_xlabel("Thinking tokens per model call (log scale)", color=INK)
    ax.set_ylabel("Model calls", color=INK)
    ax.set_title(f"Thinking tokens per call, all {len(tokens):,} calls", color=INK, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(FIGURES / out)
    plt.close(fig)


def game_rows():
    scores = summary_scores()
    rows = []
    for path in sorted(glob.glob(str(RUN / "*_p0_requests.jsonl"))):
        game = Path(path).name.split("_p0")[0]
        responses = [json.loads(line) for line in open(path)]
        responses = [r for r in responses if r["event"] == "response"]
        reasoning = [r["usage"].get("reasoning_tokens") or 0 for r in responses]
        completion = sum(r["usage"].get("completion_tokens") or 0 for r in responses)
        s = scores[game]
        rows.append(dict(
            game=game.split("-")[0], calls=len(reasoning), total=sum(reasoning),
            mean=st.mean(reasoning), median=st.median(reasoning),
            p90=pct(reasoning, .9), max=max(reasoning),
            capped=sum(r["finish_reason"] == "length" for r in responses),
            share=100 * sum(reasoning) / completion,
            score=s["score"], levels=s["levels"], total_levels=s["total_levels"],
            actions=s["actions"], per_action=sum(reasoning) / s["actions"]))
    return sorted(rows, key=lambda r: -r["total"])


def scatter(rows, y, ylabel, title, out):
    x = np.array([r["mean"] for r in rows])
    yv = np.array([r[y] for r in rows])
    r = np.corrcoef(x, yv)[0, 1]
    slope, intercept = np.polyfit(x, yv, 1)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    ax.scatter(x, yv, s=60, color=BLUE, edgecolor="white", linewidth=2, zorder=3)
    xs = np.linspace(x.min(), x.max(), 2)
    ax.plot(xs, slope * xs + intercept, color=MUTED, linewidth=1.5,
            linestyle="--", zorder=2, label=f"Linear fit (Pearson r = {r:+.2f})")
    for row in rows:
        ax.annotate(row["game"], (row["mean"], row[y]),
                    xytext=LABEL_OFFSETS.get(row["game"], (5, 4)),
                    textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xlabel("Average thinking tokens per model call", color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, color=INK, loc="left")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / out)
    plt.close(fig)
    return r


def main():
    histogram(call_tokens(), "thinking_per_call_hist.png")
    r_daniel, r_v30 = actions_scatter(
        [("(a) Daniel base, v207 danibase v05", call_actions()),
         ("(b) Our v30, adaptive-reasoning-ucb v30", transcript_pairs("v30"))],
        "thinking_vs_actions_per_call.png")
    print(f"r(thinking, actions per call): Daniel {r_daniel:+.2f}, v30 {r_v30:+.2f}")
    rows = game_rows()
    with open(DATA / "thinking_budget.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    r_score = scatter(rows, "score", "Game score",
                      "Average thinking tokens vs score, per game",
                      "thinking_vs_score.png")
    r_levels = scatter(rows, "levels", "Levels completed",
                       "Average thinking tokens vs levels completed, per game",
                       "thinking_vs_levels.png")
    print(f"r(mean thinking, score) = {r_score:+.2f}")
    print(f"r(mean thinking, levels) = {r_levels:+.2f}")
    for row in rows:
        print("| {game} | {calls} | {total:,} | {mean:,.0f} | {median:,.0f} | {p90:,} | "
              "{max:,} | {capped} | {share:.0f}% | {score:.2f} | {levels:.0f}/{total_levels:.0f} | "
              "{actions} | {per_action:,.0f} |".format(**row))


if __name__ == "__main__":
    main()
