"""Generated tokens and context on levels 1-2 of the later 15 public games.

The 25 public games split into the 10 that early 10-game runs used and the 15
the 15-game runs used; the 15 are harder. This pools every saved Kaggle run with
the 118,272-token context budget and a full-length public run (>= 90 minutes),
skipping the early-level-recovery treatment arms.

Sources:
- benchmark.json: per-action generated tokens and actions per level, so tokens
  spent on each level, and whether it was solved. An unsolved last level also
  gets the tokens generated after the last action.
- *_requests.jsonl (five runs): server prompt_tokens per request, to locate the
  context drops relative to tokens spent on a level, and whether the level's own
  start was still in context.

Run from ARC3-Inference/ in the main checkout (runs/ is a Drive symlink):
    python ../docs/harness/scripts/low_level_tokens.py [OUT_DIR]
Needs pandas, numpy, scipy, statsmodels, matplotlib.
"""

import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

RUNS_DIR = os.environ.get("RUNS_DIR", "runs/kaggle")
LATER15 = set("bp35 cd82 cn04 dc22 g50t ka59 lf52 ls20 m0r0 s5i5 sk48 sp80 su15 tn36 wa30".split())
SKIP = ("early-level-recovery/",)
REQUEST_RUNS = [
    "matijaludvig/arc3-vcintel-25x1/v01",
    "matijaludvig/arc3-vcintel-lorddraft-25x1/v01",
    "ticbudja/thinking-budget-and-user-prompt-trim/v02",
    "jonathanwang2022/lorddraft-15-games-14-seqs/v01",
    "jonathanwang2022/lorddraft-retain-decode-checkpoints/v01",
]
DROP = 25_000
BIN = 10_000
SOLVED, UNSOLVED, MUTED, INK = "#2a78d6", "#eb6834", "#8a8984", "#52514e"


def context_budget(run_dir):
    for path in glob.glob(f"{run_dir}/transcripts/*.txt")[:1]:
        with open(path, errors="ignore") as fh:
            m = re.search(r"context_budget_tokens: (\d+)", fh.read(200_000))
            return int(m.group(1)) if m else None
    return None


def load_levels():
    rows = []
    for path in sorted(glob.glob(f"{RUNS_DIR}/*/*/v*/benchmark.json")):
        run_dir = os.path.dirname(path)
        name = run_dir[len(RUNS_DIR) + 1:]
        if any(s in name + "/" for s in SKIP) or context_budget(run_dir) != 118272:
            continue
        bench = json.load(open(path))
        game_runs = bench.get("game_runs") or []
        minutes = max((g.get("final_wallclock_seconds") or 0) for g in game_runs) / 60 if game_runs else 0
        if minutes < 90:
            continue
        for gr in game_runs:
            game = gr["game_id"].split("-")[0]
            hist = gr.get("history") or []
            if game not in LATER15 or not hist:
                continue
            m = re.search(r"tokens=(\d+)", gr.get("solver_note") or "")
            total = int(m.group(1)) if m else sum(h["generated_tokens"] for h in hist)
            tail = max(0, total - sum(h["generated_tokens"] for h in hist))
            start = 0
            for level, count in enumerate(gr["actions_per_level"], start=1):
                solved = level <= gr["levels_completed"]
                acts = hist[start:start + count]
                start += count
                rows.append(dict(run=name, game=game, level=level, solved=solved, actions=count,
                                 tokens=sum(h["generated_tokens"] for h in acts) + (0 if solved else tail)))
                if not solved:
                    break
    return pd.DataFrame(rows)


def load_requests():
    rows = []
    for run in REQUEST_RUNS:
        run_dir = f"{RUNS_DIR}/{run}"
        for path in sorted(glob.glob(f"{run_dir}/*_requests.jsonl")):
            stem = os.path.basename(path).removesuffix("_requests.jsonl")
            game = stem.split("-")[0]
            if game not in LATER15:
                continue
            ups, best = [], 0
            for line in open(f"{run_dir}/artifacts/{stem}_events.jsonl"):
                event = json.loads(line)
                if event.get("type") == "action":
                    score = int(event.get("score") or 0)
                    if score > best:
                        ups.append(int(event["action_num"]))
                        best = score
            reqs = []
            for line in open(path):
                event = json.loads(line)
                if event.get("event") == "response":
                    usage = event.get("usage") or {}
                    reqs.append(dict(action=event["action"], ptok=usage.get("prompt_tokens", 0),
                                     ctok=usage.get("completion_tokens", 0), up=0))
            if not reqs:
                continue
            pointers = [r["action"] for r in reqs]
            for action in ups:
                owner = max((i for i, p in enumerate(pointers) if p <= action), default=None)
                if owner is not None:
                    reqs[owner]["up"] += 1
            # Context this level added, tracked through drains: a drain keeps the
            # newest prompt_tokens - base, so once the level's share exceeds that,
            # its first turns are gone.
            base = reqs[0]["ptok"]
            level, generated, level_ctx, drains, start_in_ctx, prev = 1, 0, 0, 0, True, None
            for r in reqs:
                if prev is not None:
                    if r["ptok"] < prev - DROP:
                        drains += 1
                        retained = max(0, r["ptok"] - base)
                        if level_ctx > retained:
                            start_in_ctx, level_ctx = False, retained
                    else:
                        level_ctx += max(0, r["ptok"] - prev)
                rows.append(dict(gid=f"{run}|{game}|{level}", game=game, level=level, ptok=r["ptok"],
                                 generated=generated, drains=drains, start_in_ctx=start_in_ctx,
                                 up=int(r["up"] > 0)))
                generated += r["ctok"]
                prev = r["ptok"]
                if r["up"]:
                    level += r["up"]
                    generated, level_ctx, drains, start_in_ctx = 0, 0, 0, True
    return pd.DataFrame(rows)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def hazard_table(levels, level, upto=200_000):
    d = levels[levels.level == level]
    out = []
    for lo in range(0, upto, BIN):
        at_risk = int((d.tokens >= lo).sum())
        events = int((d.solved & (d.tokens >= lo) & (d.tokens < lo + BIN)).sum())
        out.append(dict(lo=lo, at_risk=at_risk, events=events))
    t = pd.DataFrame(out)
    t["hazard"] = t.events / t.at_risk.where(t.at_risk > 0)
    t["survival"] = (1 - t.hazard.fillna(0)).cumprod()
    return t


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)


def mark_drops(ax, first_drop, start_lost):
    for x, label, y in ((first_drop, "first context drop", 0.98), (start_lost, "level start dropped", 0.84)):
        ax.axvline(x / 1000, color=MUTED, linestyle="--", linewidth=1)
        ax.text(x / 1000, y, f" {label} (median {x / 1000:.0f}k)", color=INK, fontsize=8, va="top", ha="left",
                transform=ax.get_xaxis_transform())


def within_game_bands(levels, level, edges=(0, 20, 40, 60, 80, 100, 120)):
    """Solve chance per 20k band given unsolved at its start, odds ratio vs
    20-40k with game fixed effects. Without them the late bands mostly measure
    which games are still stuck."""
    labels = [f"{lo}-{hi}k" for lo, hi in zip(edges[:-1], edges[1:])]
    rows = []
    for game, solved, tokens in levels.loc[levels.level == level, ["game", "solved", "tokens"]].itertuples(index=False):
        k = tokens / 1000
        for lo, hi, label in zip(edges[:-1], edges[1:], labels):
            if k < lo:
                break
            rows.append(dict(game=game, band=label, up=int(solved and k < hi)))
    pp = pd.DataFrame(rows)
    fit = smf.logit(f"up ~ C(band, Treatment('{labels[1]}')) + C(game)", pp).fit(disp=0, maxiter=200)
    print(f"level {level}: solved within band | unsolved at band start; within-game OR vs {labels[1]}")
    for label in labels:
        d = pp[pp.band == label]
        key = f"C(band, Treatment('{labels[1]}'))[T.{label}]"
        orr, (lo, hi) = (np.exp(fit.params[key]), np.exp(fit.conf_int().loc[key])) if key in fit.params else (1.0, (1.0, 1.0))
        print(f"  {label:9s} {d.up.sum():3d}/{len(d):3d} = {d.up.mean():.2f}   OR {orr:.2f} ({lo:.2f}-{hi:.2f})")


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    levels = load_levels()
    reqs = load_requests()
    print(f"{levels.groupby(['run', 'game']).ngroups} game runs from {levels.run.nunique()} runs")

    low = reqs[reqs.level <= 2]
    first_drop = low[low.drains >= 1].groupby("gid").generated.min()
    start_lost = low[~low.start_in_ctx].groupby("gid").generated.min()
    print(f"generated tokens on level at first drop: median {first_drop.median():.0f} "
          f"(IQR {first_drop.quantile(.25):.0f}-{first_drop.quantile(.75):.0f}, n={len(first_drop)})")
    print(f"generated tokens on level when its start left context: median {start_lost.median():.0f} "
          f"(IQR {start_lost.quantile(.25):.0f}-{start_lost.quantile(.75):.0f}, n={len(start_lost)})")

    for level in (1, 2, 3):
        d = levels[levels.level == level]
        print(f"level {level}: {d.solved.sum()}/{len(d)} solved; solved median {d[d.solved].tokens.median():.0f}, "
              f"p90 {d[d.solved].tokens.quantile(.9):.0f}; unsolved median {d[~d.solved].tokens.median():.0f}")
    within_game_bands(levels, 1)
    within_game_bands(levels, 2)
    unsolved = levels[~levels.solved]
    total = levels.tokens.sum()
    print(f"tokens on levels never solved: {unsolved.tokens.sum() / total:.0%} of later-15 tokens; "
          f"levels 1-2 alone {unsolved[unsolved.level <= 2].tokens.sum() / total:.0%}")

    # Figure 1: tokens spent on level, solved vs never solved
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    bins = np.arange(0, 300_001, BIN) / 1000
    for ax, level in zip(axes, (1, 2)):
        d = levels[levels.level == level]
        ax.hist([np.clip(d[d.solved].tokens / 1000, 0, 299), np.clip(d[~d.solved].tokens / 1000, 0, 299)],
                bins=bins, stacked=True, color=[SOLVED, UNSOLVED], rwidth=0.9, edgecolor="#fcfcfb", linewidth=0.5,
                label=[f"solved (n={d.solved.sum()})", f"never solved (n={(~d.solved).sum()})"])
        style(ax)
        ax.set_title(f"Level {level}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("generated tokens on the level (k; last bin = 290k+)", color=INK, fontsize=9)
        ax.set_ylabel("level attempts", color=INK, fontsize=9)
        mark_drops(ax, first_drop.median(), start_lost.median())
        ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.suptitle("Later 15 games: generated tokens spent on a level", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/tokens_on_level.png", dpi=150, facecolor="#fcfcfb")

    # Figure 2: still-unsolved fraction and solve rate per 10k tokens
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for level, color in ((1, SOLVED), (2, UNSOLVED)):
        t = hazard_table(levels, level)
        x = (t.lo + BIN) / 1000
        top.step(np.r_[0, x], np.r_[1, t.survival], where="post", color=color, linewidth=2, label=f"level {level}")
        keep = t.at_risk >= 15
        ci = np.array([wilson(e, n) for e, n in zip(t.events, t.at_risk)])
        centre = (t.lo + BIN / 2) / 1000 + (-1.2 if level == 1 else 1.2)
        bottom.errorbar(centre[keep], t.hazard[keep], yerr=[t.hazard[keep] - ci[keep, 0], ci[keep, 1] - t.hazard[keep]],
                        fmt="o", color=color, markersize=4, linewidth=1, capsize=0, label=f"level {level}")
    for ax in (top, bottom):
        style(ax)
    top.set_ylabel("share still unsolved", color=INK, fontsize=9)
    bottom.set_ylabel("solved in the next 10k tokens\n(given unsolved so far)", color=INK, fontsize=9)
    bottom.set_xlabel("generated tokens already spent on the level (k); bins with >= 15 attempts at risk", color=INK, fontsize=9)
    top.legend(frameon=False, fontsize=9)
    mark_drops(top, first_drop.median(), start_lost.median())
    bottom.axvline(first_drop.median() / 1000, color=MUTED, linestyle="--", linewidth=1)
    bottom.axvline(start_lost.median() / 1000, color=MUTED, linestyle="--", linewidth=1)
    fig.suptitle("Later 15 games: does the chance of solving fall as tokens on the level grow?",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/solve_rate_vs_tokens.png", dpi=150, facecolor="#fcfcfb")

    # Figure 2b: the same solve rate for every level reached often enough
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    for ax, level in zip(axes.flat, range(1, 7)):
        t = hazard_table(levels, level, upto=150_000)
        keep = t.at_risk >= 15
        ci = np.array([wilson(e, n) for e, n in zip(t.events, t.at_risk)])
        x = (t.lo + BIN / 2) / 1000
        ax.errorbar(x[keep], t.hazard[keep], yerr=[t.hazard[keep] - ci[keep, 0], ci[keep, 1] - t.hazard[keep]],
                    fmt="o", color=SOLVED, markersize=4, linewidth=1, capsize=0)
        d = levels[levels.level == level]
        ax.set_title(f"Level {level}: {d.solved.sum()}/{len(d)} solved", color=INK, fontsize=10, loc="left")
        ax.axvline(first_drop.median() / 1000, color=MUTED, linestyle="--", linewidth=1)
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel("generated tokens already spent on the level (k)", color=INK, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("solved in the next 10k tokens\n(given unsolved so far)", color=INK, fontsize=9)
    fig.suptitle("Later 15 games: solve rate per 10k tokens by level (bins with >= 15 attempts at risk; "
                 "dashed = median first context drop)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/solve_rate_by_level.png", dpi=150, facecolor="#fcfcfb")
    for level in (3, 4, 5):
        within_game_bands(levels, level, edges=(0, 20, 40, 60, 80, 100))

    # Figure 3: where each game ends
    last = levels.sort_values("level").groupby(["run", "game"]).tail(1)
    ends = last.assign(end=np.where(last.solved, "won", np.where(last.level == 1, "stuck on 1",
                                    np.where(last.level == 2, "stuck on 2", "stuck on 3+"))))
    share = ends.groupby("game").end.value_counts(normalize=True).unstack(fill_value=0)
    order = ["stuck on 1", "stuck on 2", "stuck on 3+", "won"]
    share = share.reindex(columns=order, fill_value=0).sort_values(["stuck on 1", "stuck on 2"])
    fig, ax = plt.subplots(figsize=(9, 5))
    left = np.zeros(len(share))
    colors = {"stuck on 1": UNSOLVED, "stuck on 2": "#eda100", "stuck on 3+": "#9ec5f4", "won": SOLVED}
    for col in order:
        ax.barh(share.index, share[col], left=left, color=colors[col], edgecolor="#fcfcfb", linewidth=2, label=col)
        left += share[col].values
    style(ax)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    sizes = ends.groupby("game").size()
    span = f"{sizes.min()}" if sizes.min() == sizes.max() else f"{sizes.min()}-{sizes.max()}"
    ax.set_xlabel(f"share of runs ({span} runs per game)", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.suptitle("Later 15 games: where each run ended", color=INK, fontsize=12, x=0.01, ha="left", y=0.99)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/where_runs_end.png", dpi=150, facecolor="#fcfcfb")
    print(share.round(2).to_string())

    # Figure 4 / test: solve rate per request before vs after the level's start left context
    rates = low.groupby(["level", "start_in_ctx"]).up.agg(["sum", "size"]).reset_index()
    print(rates.to_string())
    # Requests with the level's start dropped have spent more tokens on the
    # level, so compare within 20k token bands; a log-tokens control assumes the
    # rate keeps rising and pushes the difference onto the dropped state.
    model = low.assign(lost=(~low.start_in_ctx).astype(int),
                       band=pd.cut(low.generated / 1000, [0, 20, 40, 60, 90, 1e9], right=False))
    print(model.groupby(["band", "lost"], observed=True).up.agg(["sum", "size", "mean"]).round(3).to_string())
    fit = smf.logit("up ~ lost + C(band) + C(level)", model).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(model.game)[0]})
    lo, hi = np.exp(fit.conf_int().loc["lost"])
    print(f"odds ratio per request, level start dropped vs kept (same token band and level, clustered by game): "
          f"{np.exp(fit.params.lost):.2f} (95% CI {lo:.2f}-{hi:.2f}, p={fit.pvalues.lost:.2g})")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, (level, kept) in enumerate([(1, True), (1, False), (2, True), (2, False)]):
        row = rates[(rates.level == level) & (rates.start_in_ctx == kept)].iloc[0]
        p = row["sum"] / row["size"]
        lo_ci, hi_ci = wilson(row["sum"], row["size"])
        color = SOLVED if kept else UNSOLVED
        ax.errorbar(i, p, yerr=[[p - lo_ci], [hi_ci - p]], fmt="o", color=color, markersize=7, linewidth=1.5)
        ax.text(i + 0.08, p, f"{int(row['sum'])}/{int(row['size'])}", color=INK, fontsize=8, va="center")
    ax.set_xticks(range(4), ["L1 start\nin context", "L1 start\ndropped", "L2 start\nin context", "L2 start\ndropped"])
    style(ax)
    ax.set_ylabel("level-ups per request", color=INK, fontsize=9)
    fig.suptitle("Later 15 games, 5 runs with request logs (unadjusted; same token band: "
                 f"OR {np.exp(fit.params.lost):.2f}, 95% CI {lo:.2f}-{hi:.2f})", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/level_start_dropped.png", dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "low-level-tokens")
