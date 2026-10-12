"""Does keeping the level-up turn in context help solve the next level?

Per turn on levels 2-3: is the turn that completed the previous level still in
the (simulated or real) trimmed context? After a trim the context keeps roughly
the newest (current - base) tokens, so the level-up is retained iff the
untrimmed context added since it is <= current - base.

Run from the repo root:
    RUNS_DIR=<main checkout>/ARC3-Inference/runs/kaggle python3 docs/harness/scripts/levelup_in_context.py
"""
import glob, json, os, importlib.util, io, contextlib
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2

spec = importlib.util.spec_from_file_location("s", os.path.join(os.path.dirname(os.path.abspath(__file__)), "solve_rate_vs_context.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with contextlib.redirect_stdout(io.StringIO()):
    coef = m.calibrate()
BASE = coef[0]


def build(real):
    rows = []
    for run_dir in sorted(glob.glob(f"{m.RUNS_DIR}/*/*/v*")):
        if m.budget(run_dir) != m.BUDGET or not m.daniel_base(run_dir):
            continue
        for t in glob.glob(f"{run_dir}/transcripts/*.txt"):
            stem = os.path.basename(t)[:-4]
            game = stem.split("-")[0]
            ev = f"{run_dir}/artifacts/{stem}_events.jsonl"
            rq = f"{run_dir}/{stem}_requests.jsonl"
            if game not in m.LATER15 or not os.path.exists(ev) or (real and not os.path.exists(rq)):
                continue
            steps = m.transcript_steps(t)
            if len(steps) < 2:
                continue
            ends = BASE + np.cumsum([c for _, c, _ in steps]) * coef[1] + np.cumsum([i for _, _, i in steps]) * coef[2]
            starts = np.r_[BASE, ends[:-1]]
            if real:
                first = {}
                for line in open(rq):
                    e = json.loads(line)
                    if e.get("event") == "response":
                        first.setdefault(e["analysis_step"], e["usage"]["prompt_tokens"])
                nums = [int(x.group(1)) for x in m.HEADER.finditer(open(t, errors="ignore").read())]
                current = np.array([first.get(s, np.nan) for s in nums], dtype=float)
            else:
                current = m.simulate_trimming(starts)
            ptr = [a for a, _, _ in steps]
            solve = {}
            for lvl, a in enumerate(m.level_up_actions(ev), start=1):
                solve.setdefault(max((k for k, p in enumerate(ptr) if p <= a), default=0), lvl)
            level, lvl_start = 1, BASE
            for k in range(len(steps)):
                if level in (2, 3) and not np.isnan(current[k]):
                    since = starts[k] - lvl_start
                    rows.append(dict(attempt=f"{run_dir}|{game}", game=game, level=level, since=since,
                                     added=ends[k] - starts[k], up=int(k in solve),
                                     kept=int(since <= current[k] - BASE)))
                if k in solve:
                    level, lvl_start = solve[k] + 1, ends[k]
    return pd.DataFrame(rows)


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "levelup-in-context.png")


def adjusted(d):
    """Kept vs trimmed rate ratio within the same 20k band, game and level."""
    fit = smf.glm("up ~ kept + C(band) + C(game) + C(level)", d, family=sm.families.Poisson(),
                  offset=np.log(d.added / 1e4)).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.attempt)[0]})
    lo, hi = np.exp(fit.conf_int().loc["kept"])
    return np.exp(fit.params.kept), lo, hi, fit.pvalues.kept


def plot(d):
    """Columns per level: level-ups per 10k added context (~ chance of leveling
    up in the next 10k) with the last level-up kept vs trimmed away, over the
    40-100k since the level-up where both cases occur."""
    d = d[(d.since >= 40_000) & (d.since < 100_000)]
    groups = [("Level 2", d[d.level == 2]), ("Level 3", d[d.level == 3]), ("Levels 2-3", d)]
    ticks = []
    fig, ax = plt.subplots(figsize=(8, 4.4))
    width = 0.36
    for i, (name, g) in enumerate(groups):
        for kept, color, dx, label in ((1, m.COLORS[1], -width / 2, "last level-up kept"),
                                       (0, m.COLORS[2], width / 2, "last level-up trimmed away")):
            x = g[g.kept == kept]
            exp, k = x.added.sum() / 1e4, int(x.up.sum())
            r = k / exp
            ci = (chi2.ppf(0.025, 2 * k) / 2 / exp if k else 0.0, chi2.ppf(0.975, 2 * k + 2) / 2 / exp)
            ax.bar(i + dx, r, width=width * 0.92, color=color, label=label if i == 0 else None)
            ax.errorbar(i + dx, r, yerr=[[r - ci[0]], [ci[1] - r]], fmt="none", ecolor=m.INK, elinewidth=1)
            ax.text(i + dx, ci[1] + 0.004, f"{k} ups", color=m.INK, fontsize=8, ha="center")
        rr, lo, hi, p = adjusted(g)
        print(f"{name}, 40-100k: kept vs trimmed {rr:.2f} (95% CI {lo:.2f}-{hi:.2f}, p={p:.2g}); level-ups {int(g.up.sum())}")
        ticks.append(f"{name}\nkept/trimmed {rr:.2f}x\n95% CI {lo:.2f}-{hi:.2f}, p={p:.2f}")
    ax.set_xticks(range(len(groups)), ticks)
    ax.set_ylim(bottom=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=m.INK, labelsize=9)
    ax.set_ylabel("level-ups per 10k of added context\n(chance of leveling up in the next 10k)", color=m.INK, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Later 15 games, 30 runs: last level-up kept vs trimmed away (40-100k since the level-up)",
                 color=m.INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, facecolor="#fcfcfb")


for name, real in (("simulated, 30 runs", False), ("real prompt_tokens, request-log runs", True)):
    d = build(real)
    d = d[d.added > 0]
    d["band"] = pd.cut(d.since / 1000, [0, 20, 40, 60, 80, 100, 130, 1e9], right=False).astype(str)
    print(f"\n== {name}: {len(d)} turns, {d.up.sum()} level-ups")
    g = d.groupby(["band", "kept"]).agg(turns=("up", "size"), ups=("up", "sum"), exp=("added", lambda x: x.sum() / 1e4))
    g["per10k"] = g.ups / g.exp
    print(g.round(3).to_string())
    overlap = d[d.groupby("band").kept.transform("nunique") == 2]
    for label, sub in (("levels 2-3", overlap), ("level 2", overlap[overlap.level == 2]), ("level 3", overlap[overlap.level == 3])):
        rr, lo, hi, p = adjusted(sub)
        print(f"{label}: rate ratio, level-up still in context vs trimmed away (same band, game, level): "
              f"{rr:.2f} (95% CI {lo:.2f}-{hi:.2f}, p={p:.2g}); turns {len(sub)}, level-ups {sub.up.sum()}")
    if not real:
        plot(d)
