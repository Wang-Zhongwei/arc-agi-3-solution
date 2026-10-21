"""Build the run atlas: aggregate every ARC-AGI-3 harness run we have outputs for,
write tidy CSVs, and render the charts the docs reference.

Inputs (all optional; whatever exists is used):
  kaggle-outputs/fz-*                 Franzen-era runs (summary.txt, benchmark.json, serve.log)
  kaggle-outputs/b1-thinking-real, final-c1-ledgerfix   Wang-era runs fetched earlier
  kaggle-outputs/all/<kernel>/        summaries/benchmarks fetched for every kernel
  external/dfranzen/run-output        Franzen's own public reference run
  franzen/analysis/levels.csv         per-level attempts (scheduler dataset)
  reports/run-atlas/data/leaderboard.csv

Usage (from arc-agi-3-v3/):
  <python with matplotlib> reports/run-atlas/build_atlas.py

Every score is recomputed from per-game scores with the same formulas, so
runs from different eras are comparable on: mean, trimmed mean (drop the top
2 games), median, games > 0, games >= 10, games won.
"""
from __future__ import annotations

import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]          # arc-agi-3-v3/
ATLAS = Path(__file__).resolve().parent
DATA, CHARTS = ATLAS / "data", ATLAS / "charts"
OUT = ROOT / "kaggle-outputs"

# Reference palette (dataviz skill, light mode), fixed slot order.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ERA_COLOR = {"wang": SLOT[0], "franzen": SLOT[1], "tufa-duck": SLOT[2]}

# ---------------------------------------------------------------- registry
# What each version is and what it changed. Era = harness lineage.
#   wang:    Wang fork of Tufa's Duck harness, vLLM W4A16, 49k ctx, UCB allocation.
#   franzen: Franzen's Milestone-2 harness, SGLang Pennyroyal, 128k ctx, FP8 KV.
# budget_min = per-game budget in minutes as configured; games = offline games.
REGISTRY = [
    # --- Wang era (25 games x 132 min unless noted) ---
    ("taaf-wang-intel-w4a16-0922", "wang", "2026-09-22", "0922 baseline",
     "Wang fork as deployed on the Intel W4A16 serving bundle (vLLM, 49k context).", None),
    ("taaf-wang-intel-w4a16-explore", "wang", "2026-09-23", "explore 0.25 + gpu 0.96",
     "UCB exploration floor 0.25 and gpu_memory_utilization 0.96.", "taaf-wang-intel-w4a16-0922"),
    ("taaf-wang-intel-w4a16-memory", "wang", "2026-09-23", "memory (pre-fix)",
     "Memory ledgers before the starvation fix.", "taaf-wang-intel-w4a16-0922"),
    ("taaf-wang-intel-memory-fixed", "wang", "2026-09-23", "memory-fixed",
     "Memory ledger (world model, confirmed/refuted hypotheses, level notes) fed every turn.", "taaf-wang-intel-w4a16-0922"),
    ("taaf-wang-explore-high", "wang", "2026-09-24", "explore 0.5",
     "UCB exploration floor raised to 0.5.", "taaf-wang-intel-memory-fixed"),
    ("taaf-wang-spend-actions", "wang", "2026-09-24", "spend-actions prompt",
     "Prompt: spend actions to finish levels; batch when the path is known.", "taaf-wang-intel-memory-fixed"),
    ("taaf-wang-ucb-v5-exact", "wang", "2026-09-23", "Wang v5 (his stack)",
     "Wang's own v5 configuration replicated.", None),
    ("taaf-wang-v5-fixed", "wang", "2026-09-23", "Wang v5 fixed", "Wang v5 with deploy fixes.", None),
    ("taaf-wang-v5-headroom", "wang", "2026-09-23", "Wang v5 headroom", "Wang v5 with more GPU headroom.", None),
    ("taaf-a1-marginal", "wang", "2026-09-24", "a1 marginal alloc (30m)",
     "Marginal-gain allocation (silently ran uniform; 30-min budget).", "taaf-wang-intel-memory-fixed"),
    ("taaf-a2-knapsack", "wang", "2026-09-24", "a2 knapsack alloc (30m)",
     "Knapsack allocation (silently ran uniform; 30-min budget).", "taaf-wang-intel-memory-fixed"),
    ("taaf-a3-thinking", "wang", "2026-09-24", "a3 thinking guide (30m)",
     "Thinking guide (diff, rule, goal, plan); 30-min budget.", "taaf-wang-intel-memory-fixed"),
    ("taaf-a4-scratchpad", "wang", "2026-09-24", "a4 scratchpad (30m)",
     "Free-form scratchpad tag; model wrote zero tags. 30-min budget.", "taaf-wang-intel-memory-fixed"),
    ("taaf-b1-thinking-real", "wang", "2026-09-26", "b1 thinking (inert) + introspection",
     "Thinking guide (inert in practice) with logprob introspection on; effectively a control.", "taaf-wang-intel-memory-fixed"),
    ("taaf-b2-prior-run", "wang", "2026-09-26", "b2 prior-run notes",
     "Notes carried over from a prior run.", "taaf-b1-thinking-real"),
    ("taaf-b3-label-diff", "wang", "2026-09-26", "b3 labeled diff image",
     "Changed-cells diff image with labels.", "taaf-b1-thinking-real"),
    ("taaf-b4-scoring-guide", "wang", "2026-09-26", "b4 scoring guide",
     "Prompt stating the RHAE scoring rule; longer batches.", "taaf-b1-thinking-real"),
    ("taaf-c1-upscale8", "wang", "2026-09-26", "c1 image upscale 8",
     "Board image upscaled 8x (from 4x).", "taaf-b1-thinking-real"),
    ("taaf-c2-upscale12-ascii", "wang", "2026-09-26", "c2 upscale 12 + ascii",
     "Upscale 12x plus ASCII overlay.", "taaf-b1-thinking-real"),
    ("taaf-d1-thinkecho", "wang", "2026-09-26", "d1 think echo (30m)",
     "Echo the model's reasoning back; 30-min budget.", "taaf-b1-thinking-real"),
    ("taaf-e0-uniform-base", "wang", "2026-09-26", "e0 uniform allocation",
     "Uniform time per game instead of UCB.", "taaf-b1-thinking-real"),
    ("taaf-e1-predict-uniform", "wang", "2026-09-26", "e1 predict + uniform",
     "Prediction line requested, uniform allocation.", "taaf-e0-uniform-base"),
    ("taaf-f1-score-cap35", "wang", "2026-09-26", "f1 score cap 35",
     "Stop investing in a game after score 35.", "taaf-b1-thinking-real"),
    ("taaf-g1-scratchpad2h", "wang", "2026-09-26", "g1 scratchpad (2h)",
     "Scratchpad tag at the full budget.", "taaf-b1-thinking-real"),
    ("taaf-h1-ascii-only", "wang", "2026-09-26", "h1 ascii only",
     "Text-only board (no image).", "taaf-b1-thinking-real"),
    ("taaf-h2-ascii-plus-image", "wang", "2026-09-26", "h2 ascii + image",
     "ASCII board plus image.", "taaf-b1-thinking-real"),
    ("taaf-final-c1-ledgerfix", "wang", "2026-09-27", "final: c1 + ledger fix",
     "c1-upscale8 plus the fix feeding the ledger from reasoning-only replies.", "taaf-c1-upscale8"),
    # --- Franzen era (25 games x 121 min unless noted) ---
    ("dfranzen-reference", "franzen", "2026-09-30", "Franzen reference (10 g x 25 min)",
     "Franzen's own public run: 10 demo games, 25 min (not comparable to 25 x 121).", None),
    ("fz-arms-v3-smoke", "franzen", "2026-10-03", "smoke (10 g x 5 min)",
     "Startup check after the cold-start hang; 5-minute games.", None),
    ("fz-arms-v4-control", "franzen", "2026-10-03", "control",
     "Franzen's Milestone-2 configuration, unchanged (plus an inert retry fix).", None),
    ("fz-arms-v5-temp06", "franzen", "2026-10-03", "temperature 0.6",
     "Sampling temperature 0.7 -> 0.6.", "fz-arms-v4-control"),
    ("fz-arms-v6-audit60k", "franzen", "2026-10-03", "strategy audit 60k",
     "lordhansolo's strategy audit every 60k tokens on an uncleared level.", "fz-arms-v4-control"),
    ("fz-streams14", "franzen", "2026-10-03", "14 streams",
     "14 concurrent streams; context 128k -> 96k (drain 42k) to fit the KV pool.", "fz-arms-v4-control"),
    ("fz-ledger-on-trim", "franzen", "2026-10-03", "ledger on trim",
     "Memory ledger placed at the head of history right after each trim.", "fz-arms-v4-control"),
    ("fz-ledger-streams12", "franzen", "2026-10-03", "ledger + 12 streams",
     "Ledger on trim with 12 streams; context 110k, drain 49k.", "fz-ledger-on-trim"),
    ("fz-ucb", "franzen", "2026-10-03", "UCB scheduler",
     "Gate ranks games by a reward-rate UCB index instead of Franzen's P=(A+B)C.", "fz-arms-v4-control"),
    ("fz-tput-all-v7", "franzen", "2026-10-03", "throughput: MXFP8 + greedy (10 g x 25 min)",
     "Online MXFP8 activations and temperature 0; throughput test only.", "dfranzen-reference"),
]
REG = {r[0]: r for r in REGISTRY}
LOCAL_DIRS = {
    "fz-arms-v3-smoke": OUT / "fz-arms-v3-smoke",
    "fz-arms-v4-control": OUT / "fz-arms-v4-control",
    "fz-arms-v5-temp06": OUT / "fz-arms-v5-temp06",
    "fz-arms-v6-audit60k": OUT / "fz-arms-v6-audit60k",
    "fz-streams14": OUT / "fz-streams14",
    "fz-ledger-on-trim": OUT / "fz-ledger-on-trim",
    "fz-ledger-streams12": OUT / "fz-ledger-streams12",
    "fz-ucb": OUT / "fz-ucb",
    "fz-tput-all-v7": OUT / "fz-tput-all-v7",
    "dfranzen-reference": ROOT / "external/dfranzen/run-output",
    "taaf-b1-thinking-real": OUT / "b1-thinking-real",
    "taaf-final-c1-ledgerfix": OUT / "final-c1-ledgerfix",
}


# ---------------------------------------------------------------- parsing
GAME_RE = re.compile(r"^\s+(\S+): score=([\d.]+), levels=([\d.]+)/(\d+), actions=(\d+), tokens=(\d+)", re.M)


def parse_run(run_id: str, path: Path) -> dict | None:
    """Per-game rows and run-level facts from summary.txt, else benchmark.json."""
    games, minutes = {}, None
    summary = path / "summary.txt"
    if summary.exists():
        text = summary.read_text(errors="replace")
        for g, sc, lv, n, a, t in GAME_RE.findall(text):
            games[g[:4]] = dict(score=float(sc), levels=float(lv), n=int(n), actions=int(a), tokens=int(t))
        if (m := re.search(r"duration:\s+(?:(\d+)h )?(\d+)m", text)):
            minutes = 60 * int(m[1] or 0) + int(m[2])
    bench = path / "benchmark.json"
    if not games and bench.exists():
        b = json.loads(bench.read_text())
        for g in b.get("game_runs", []):
            tok = re.search(r"tokens=(\d+)", g.get("solver_note") or "")
            games[g["game_id"][:4]] = dict(
                score=float(g.get("final_score") or 0), levels=float(g["levels_completed"]),
                n=int(g["number_of_levels"]), actions=len(g.get("history") or []),
                tokens=int(tok[1]) if tok else sum(int(h.get("generated_tokens") or 0) for h in g.get("history") or []),
            )
    if not games:
        return None
    start = ""
    if (bench := path / "benchmark.json").exists():
        start = json.loads(bench.read_text()).get("start_time", "")[:10]
    return {"run": run_id, "games": games, "minutes": minutes, "dir": str(path), "start": start}


def metrics(games: dict) -> dict:
    """Run metrics recomputed from per-game scores."""
    s = sorted(g["score"] for g in games.values())
    trimmed = s[:-2] if len(s) > 4 else s
    cleared = sum(g["levels"] for g in games.values())
    tokens = sum(g["tokens"] for g in games.values())
    return dict(
        games=len(s), mean=st.mean(s), trimmed=st.mean(trimmed), median=st.median(s),
        gt0=sum(x > 0 for x in s), ge10=sum(x >= 10 for x in s),
        won=sum(g["levels"] >= g["n"] for g in games.values()),
        levels=cleared, tokens=tokens, tokens_per_level=tokens / cleared if cleared else float("nan"),
    )


DECODE = re.compile(r"Decode batch, #running-req: (\d+).*?full token usage: ([\d.]+).*?accept len: ([\d.]+)"
                    r".*?gen throughput \(token/s\): ([\d.]+)")


def serve_metrics(path: Path) -> dict | None:
    """Decode-speed stats from serve.log (Franzen era only)."""
    log = path / "serve.log"
    if not log.exists():
        return None
    rows = [m.groups() for m in map(DECODE.search, log.read_text(errors="replace").splitlines()) if m]
    if not rows:
        return None
    speeds = sorted(float(r[3]) for r in rows)
    running = [int(r[0]) for r in rows]
    top = max(running)
    full = [float(r[3]) for r in rows if int(r[0]) == top]
    return dict(decode_median=st.median(speeds), decode_p90=speeds[int(0.9 * len(speeds))],
                decode_full=st.median(full), batch_mean=st.mean(running), batch_max=top,
                accept=st.mean(float(r[2]) for r in rows))


# ---------------------------------------------------------------- charts
def style(ax, title, xlabel=None, ylabel=None):
    """Recessive grid/axes, ink-colored text, title as a left-aligned heading."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(CHARTS / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_era_scores(rows):
    """Comparable full runs (25 games, >=120 min), mean and trimmed mean, by era."""
    rows = [r for r in rows if r["era"] != "tufa-duck" and r["games"] == 25 and (r["budget_min"] or 0) >= 115]
    rows.sort(key=lambda r: (r["era"] != "wang", r["mean"]))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(rows) + 1)))
    y = range(len(rows))
    for i, r in zip(y, rows):
        c = ERA_COLOR[r["era"]]
        ax.plot([r["trimmed"], r["mean"]], [i, i], color=c, linewidth=2, alpha=0.5)
        ax.scatter(r["mean"], i, s=40, color=c, zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.scatter(r["trimmed"], i, s=40, facecolor=SURFACE, edgecolor=c, linewidth=2, zorder=3)
        ax.text(max(r["mean"], r["trimmed"]) + 0.8, i, f"{r['mean']:.1f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8, color=INK)
    style(ax, "Offline score of every comparable full run (25 games, full budget)",
          "score (filled = mean, hollow = trimmed mean, top-2 games dropped)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=ERA_COLOR[e], label=lab)
               for e, lab in (("wang", "Wang fork (vLLM, 49k ctx)"), ("franzen", "Franzen base (SGLang, 128k ctx)"))]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    save(fig, "01_scores_all_runs.png")


def chart_leaderboard(lb):
    """Public leaderboard submissions over time, coloured by harness era."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    rows = [r for r in lb if r["status"] == "COMPLETE" and r["harness_era"] in ERA_COLOR]
    import datetime as dt
    for era in ("tufa-duck", "wang", "franzen"):
        pts = [(dt.datetime.strptime(r["date"], "%Y-%m-%d %H:%M"), float(r["score"])) for r in rows if r["harness_era"] == era]
        if pts:
            ax.scatter(*zip(*pts), s=50, color=ERA_COLOR[era], edgecolor=SURFACE, linewidth=2, zorder=3,
                       label={"tufa-duck": "Tufa Duck (Sep 7-19)", "wang": "Wang fork", "franzen": "Franzen base"}[era])
    best = max(rows, key=lambda r: float(r["score"]))
    ax.annotate(f"{float(best['score']):.2f}  {best['notebook']}", (dt.datetime.strptime(best["date"], "%Y-%m-%d %H:%M"), float(best["score"])),
                xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=8, color=INK2)
    ax.axhline(25.77, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(ax.get_xlim()[0], 26.3, "  Franzen copies: mean 25.77 (sd 3.93, community snapshot)", fontsize=8, color=INK2)
    style(ax, "Public leaderboard submissions", None, "public score")
    ax.set_ylim(-1, 29)
    ax.legend(frameon=False, fontsize=9, loc="center left")
    fig.autofmt_xdate()
    save(fig, "02_leaderboard_history.png")


def chart_franzen_arms(rows, control_id="fz-arms-v4-control"):
    """Franzen-era arms vs control: mean and trimmed mean."""
    order = ["fz-arms-v4-control", "fz-arms-v5-temp06", "fz-arms-v6-audit60k", "fz-ucb",
             "fz-streams14", "fz-ledger-streams12", "fz-ledger-on-trim"]
    rows = [r for o in order for r in rows if r["run"] == o]
    ctrl = next(r for r in rows if r["run"] == control_id)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = range(len(rows))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [r["mean"] for r in rows], width=w - 0.04, color=SLOT[1], label="mean")
    ax.bar([i + w / 2 for i in x], [r["trimmed"] for r in rows], width=w - 0.04, color=SLOT[0], label="trimmed mean")
    for i, r in zip(x, rows):
        ax.text(i - w / 2, r["mean"] + 0.6, f"{r['mean']:.1f}", ha="center", fontsize=8, color=INK2)
    ax.axhline(ctrl["mean"], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["label"] for r in rows], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(30, max(r["mean"] for r in rows) + 6)
    style(ax, "Franzen-era arms (25 games x 121 min); dashed line = control mean", None, "score")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "03_franzen_arms.png")


def chart_heatmap(rows, per_game):
    """Per-game score for the comparable Franzen-era runs plus the best Wang runs."""
    keep = ["taaf-b1-thinking-real", "taaf-c1-upscale8", "fz-arms-v4-control", "fz-arms-v5-temp06",
            "fz-arms-v6-audit60k", "fz-ucb", "fz-streams14", "fz-ledger-streams12", "fz-ledger-on-trim"]
    runs = [r for k in keep for r in rows if r["run"] == k]
    games = sorted({g for r in runs for g in per_game[r["run"]]})
    grid = [[per_game[r["run"]].get(g, {}).get("score", float("nan")) for r in runs] for g in games]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f3f0"] + SEQ)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([r["label"] for r in runs], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(games)))
    ax.set_yticklabels(games, fontsize=8)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if v == v:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                        color=SURFACE if v > 55 else INK2)
    style(ax, "Per-game score (0-100) across runs: wide run-to-run swings on the same game")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    save(fig, "04_per_game_heatmap.png")


def chart_throughput(serve_rows):
    """Serving speed per Franzen-era run: P90 decode and harness tok/s (two panels, one axis each)."""
    order = ["dfranzen-reference", "fz-arms-v4-control", "fz-arms-v5-temp06", "fz-arms-v6-audit60k", "fz-ucb",
             "fz-ledger-on-trim", "fz-ledger-streams12", "fz-streams14", "fz-tput-all-v7"]
    rows = [r for o in order for r in serve_rows if r["run"] == o]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    labels = [r["label"] for r in rows]
    y = range(len(rows))
    for ax, key, title, color in ((axes[0], "decode_p90", "P90 decode tok/s (server)", SLOT[0]),
                                  (axes[1], "harness_tps", "Harness tok/s (generated / wallclock)", SLOT[1])):
        vals = [r.get(key) or 0 for r in rows]
        ax.barh(list(y), vals, height=0.6, color=color)
        for i, v in zip(y, vals):
            ax.text(v + 8, i, f"{v:.0f}", va="center", fontsize=8, color=INK2)
        style(ax, title)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    save(fig, "05_throughput.png")


def chart_tokens_vs_levels(per_game):
    """Wang b1 vs Franzen control: generated tokens per game vs levels cleared."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for run, color, label in (("taaf-b1-thinking-real", SLOT[0], "Wang b1 (294 tok/s)"),
                              ("fz-arms-v4-control", SLOT[1], "Franzen control (611 tok/s)")):
        g = per_game.get(run, {})
        ax.scatter([v["tokens"] / 1000 for v in g.values()], [v["levels"] for v in g.values()],
                   s=45, color=color, edgecolor=SURFACE, linewidth=2, label=label, zorder=3)
    style(ax, "Same ~42k tokens per cleared level; Franzen just gets ~2x the tokens",
          "generated tokens per game (thousands)", "levels cleared")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "06_tokens_vs_levels.png")


def chart_completion_curve(levels_csv):
    """P(clear within the next 30k tokens) vs tokens already spent: data vs Franzen's C."""
    sys.path.insert(0, str(ROOT / "franzen/analysis"))
    from fit_levels import kaplan_meier, survival_at
    rows = list(csv.DictReader(open(levels_csv)))
    curve = kaplan_meier([(int(r["tokens"]), int(r["cleared"])) for r in rows])
    xs = list(range(0, 151, 5))
    data = [1 - survival_at(curve, (x + 30) * 1000) / max(1e-9, survival_at(curve, x * 1000)) for x in xs]
    franzen = [0.53 * (0.25 + 0.75 * 2 ** -((x * 1000 / 62000) ** 2)) for x in xs]  # C scaled to the data's start
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(xs, data, color=SLOT[0], linewidth=2, label="measured (869 level attempts, 7 runs)")
    ax.plot(xs, franzen, color=SLOT[1], linewidth=2, linestyle=(0, (5, 3)),
            label="Franzen's C (token part + 0.25 floor), scaled to the same start")
    ax.axvspan(80, 150, color=GRID, alpha=0.5, linewidth=0)
    ax.text(82, 0.52, "thin data: 118 attempts reach 80k,\n31 reach 120k", fontsize=8, color=INK2, va="top")
    style(ax, "Measured clear rate vs Franzen's C: close until ~70k tokens, apart only where data is thin",
          "tokens already spent on the level (thousands)", "P(clear within next 30k tokens)")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "07_completion_curve.png")


def chart_tokens_per_level(rows):
    """Generated tokens per cleared level, every comparable full run."""
    rows = [r for r in rows if r["era"] != "tufa-duck" and r["games"] == 25 and (r["budget_min"] or 0) >= 115 and r["levels"]]
    rows.sort(key=lambda r: r["tokens_per_level"])
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(rows) + 1)))
    y = range(len(rows))
    ax.barh(list(y), [r["tokens_per_level"] / 1000 for r in rows], height=0.6,
            color=[ERA_COLOR[r["era"]] for r in rows])
    for i, r in zip(y, rows):
        ax.text(r["tokens_per_level"] / 1000 + 0.5, i, f"{r['tokens_per_level'] / 1000:.1f}k", va="center", fontsize=8, color=INK2)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8)
    style(ax, "Tokens per cleared level: the best arm of each harness lands near 40k", "thousands of tokens")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ERA_COLOR[e], label=n) for e, n in (("wang", "Wang fork"), ("franzen", "Franzen base"))],
              frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    save(fig, "08_tokens_per_level.png")


# One-line verdicts for the runs where we drew a conclusion (else blank).
VERDICT = {
    "taaf-wang-intel-memory-fixed": "memory ledger helped (first games ever won)",
    "taaf-wang-spend-actions": "best robust metrics of its batch; reverted too early",
    "taaf-b1-thinking-real": "best Wang-era LB (7.07) but the same notebook also scored 4.21 and 3.95",
    "taaf-c1-upscale8": "best Wang-era offline mean",
    "taaf-c2-upscale12-ascii": "worse: too much image/text",
    "taaf-h1-ascii-only": "worse: the image matters",
    "taaf-final-c1-ledgerfix": "final Wang build; below c1 in its single run",
    "fz-arms-v4-control": "baseline for every Franzen-era arm",
    "fz-arms-v5-temp06": "no effect",
    "fz-arms-v6-audit60k": "no effect (slightly lower)",
    "fz-streams14": "+15% tok/s, flat score (less context per game)",
    "fz-ledger-on-trim": "best so far: +6.3 mean / +6.9 trimmed over control",
    "fz-ledger-streams12": "+9% tok/s, but below ledger alone",
    "fz-ucb": "worse: sent fewer tokens to games that progressed (61% vs 74%)",
    "fz-tput-all-v7": "slower: MXFP8 cut decode ~27% (see throughput)",
    "fz-arms-v3-smoke": "startup check only",
    "dfranzen-reference": "reference for throughput tests",
}


def write_versions_md(rows, serve_rows):
    """versions.md: every run grouped by era, with what it changed and how it did."""
    serve = {r["run"]: r for r in serve_rows}
    eras = (("tufa-duck", "Tufa Duck (Sep 7-21): Tufa's Milestone-1 harness, our early patches; budgets vary, not comparable"),
            ("wang", "Wang fork (Sep 22-27): vLLM, Intel W4A16, 49k context, UCB allocation"),
            ("franzen", "Franzen base (Oct 1-): SGLang Pennyroyal, 128k context, FP8 KV, priority gate"))
    lines = ["# Versions", "",
             "Generated by `build_atlas.py` from the run outputs; do not edit by hand.",
             "Scores are recomputed per game: **mean**, **trimmed** (top 2 games dropped), games > 0, games won.",
             "`budget` is the run's wall-clock duration in minutes; only 25-game runs at the full budget are comparable.",
             ""]
    for era, title in eras:
        lines += [f"## {title}", "",
                  "| version | date | what changed | parent | games | budget | mean | trimmed | >0 | won | tok/level | verdict |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted((r for r in rows if r["era"] == era), key=lambda r: (r["date"], r["run"])):
            parent = REG[r["parent"]][3] if r["parent"] in REG else ""
            lines.append(
                f"| **{r['label']}**<br>`{r['run']}` | {r['date']} | {r['change']} | {parent} | {r['games']} | "
                f"{r['budget_min'] or '?'} | {r['mean']:.2f} | {r['trimmed']:.2f} | {r['gt0']} | {r['won']} | "
                f"{r['tokens_per_level'] / 1000:.1f}k | {VERDICT.get(r['run'], '')} |")
        lines.append("")
    lines += ["## Serving (Franzen era, from serve.log)", "",
              "| version | harness tok/s | decode median | decode P90 | decode at full batch | batch mean / max | spec accept |",
              "|---|---|---|---|---|---|---|"]
    for r in serve_rows:
        tps = f"{r['harness_tps']:.0f}" if r.get("harness_tps") else "-"
        lines.append(f"| {r['label']} | {tps} | {r['decode_median']:.0f} | {r['decode_p90']:.0f} | "
                     f"{r['decode_full']:.0f} | {r['batch_mean']:.1f} / {r['batch_max']} | {r['accept']:.2f} |")
    (ATLAS / "versions.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- main
def main():
    DATA.mkdir(exist_ok=True)
    CHARTS.mkdir(exist_ok=True)
    found = {}
    for run_id, path in LOCAL_DIRS.items():
        if (r := parse_run(run_id, path)):
            found[run_id] = r
    for d in sorted((OUT / "all").glob("*")):
        # fz-* kernels are already read from their per-version local copies
        if d.name.startswith("fz-") or d.name in found:
            continue
        if (r := parse_run(d.name, d)):
            found[d.name] = r
    rows, per_game, serve_rows = [], {}, []
    for run_id, r in found.items():
        if run_id in REG:
            era, date, label, change, parent = REG[run_id][1:]
        else:
            # Unregistered kernels: fz-* is Franzen, duck-*/arc3-* the original Tufa Duck harness.
            era = ("franzen" if run_id.startswith("fz-")
                   else "tufa-duck" if run_id.startswith(("duck-", "arc3-")) else "wang")
            date, label, change, parent = r["start"], run_id, "(no description recorded)", None
        m = metrics(r["games"])
        budget = r["minutes"]
        rows.append(dict(run=run_id, era=era, date=date, label=label, change=change, parent=parent or "",
                         budget_min=budget, **m))
        per_game[run_id] = r["games"]
        if (s := serve_metrics(Path(r["dir"]))):
            tps = None
            summary = Path(r["dir"]) / "summary.txt"
            if summary.exists() and (mm := re.search(r"generated tokens/sec: ([\d.]+)", summary.read_text())):
                tps = float(mm[1])
            serve_rows.append(dict(run=run_id, label=label, harness_tps=tps, **s))
    rows.sort(key=lambda r: (r["era"], r["date"], r["run"]))
    with open(DATA / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    with open(DATA / "per_game.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "game", "score", "levels", "levels_total", "actions", "tokens"])
        for run, games in per_game.items():
            for g, v in sorted(games.items()):
                w.writerow([run, g, v["score"], v["levels"], v["n"], v["actions"], v["tokens"]])
    with open(DATA / "serving.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(serve_rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in serve_rows)
    lb = list(csv.DictReader(open(DATA / "leaderboard.csv")))
    levels_csv = ROOT / "franzen/analysis/levels.csv"
    chart_era_scores(rows)
    chart_leaderboard(lb)
    chart_franzen_arms(rows)
    chart_heatmap(rows, per_game)
    chart_throughput(serve_rows)
    chart_tokens_vs_levels(per_game)
    if levels_csv.exists():
        chart_completion_curve(levels_csv)
    chart_tokens_per_level(rows)
    write_versions_md(rows, serve_rows)
    print(f"{len(rows)} runs, {sum(len(g) for g in per_game.values())} game-runs, {len(serve_rows)} with serve.log")
    for r in rows:
        print(f"  {r['era']:8} {r['run']:34} games {r['games']:2} {str(r['budget_min'] or '?'):>4}m  "
              f"mean {r['mean']:6.2f}  trim {r['trimmed']:6.2f}  >0 {r['gt0']:2}  won {r['won']}")


if __name__ == "__main__":
    main()
