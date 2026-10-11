"""Level 1-3 solve rate vs untrimmed context, later 15 public games.

x is the context a game would have if history were never trimmed: everything
that entered the conversation since the game started. It is rebuilt from the
transcript (user prompts, thinking, replies, tool calls and results, plus a
per-image cost), excluding the per-turn system prompt copy and status blocks,
and calibrated against server prompt_tokens before the first trim in the runs
that saved request logs.

For each level and each 10k window of x: among attempts on that level that
started before the window and were unsolved at its start, the share solved
inside it.

Run from the repo root:
    RUNS_DIR=<main checkout>/ARC3-Inference/runs/kaggle python docs/harness/scripts/solve_rate_vs_context.py
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
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

RUNS_DIR = os.environ.get("RUNS_DIR", "ARC3-Inference/runs/kaggle")
OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/harness/images/solve-rate-vs-context.png"
OUT_CURRENT = os.path.join(os.path.dirname(OUT), "solve-rate-vs-current-context.png")
LATER15 = set("bp35 cd82 cn04 dc22 g50t ka59 lf52 ls20 m0r0 s5i5 sk48 sp80 su15 tn36 wa30".split())
BUDGET = 118272
KEEP = ("[USER PROMPT]", "[THINKING]", "[ASSISTANT]", "[TOOL CALL", "[TOOL RESULT")
HEADER = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \|", re.M)
WINDOW = 10_000
TRIM_AT, TRIM_TO = 118_000, 58_000  # fitted; = budget and budget - 58*1024 drain
MIN_AT_RISK = 20
COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
INK, MUTED = "#52514e", "#8a8984"


def transcript_steps(path):
    """(action pointer, context chars, images) per turn header, in order."""
    text = open(path, errors="ignore").read()
    marks = list(HEADER.finditer(text))
    out = []
    for i, m in enumerate(marks):
        block = text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        chars = images = 0
        keep = False
        for line in block.split("\n"):
            if line.startswith("[") and line.rstrip().endswith("]") and len(line) < 40:
                if line.strip() == "[image attached]":
                    images += keep
                else:
                    keep = line.strip().startswith(KEEP)
                continue
            if keep:
                chars += len(line) + 1
        out.append((int(m.group(2)), chars, images))
    return out


def level_up_actions(path):
    ups, best = [], 0
    for line in open(path):
        e = json.loads(line)
        if e.get("type") == "action":
            score = int(e.get("score") or 0)
            if score > best:
                ups.extend([int(e["action_num"])] * (score - best))
                best = score
    return ups


def daniel_base(run_dir):
    """Daniel's harness settings: 3072-token tool output, no sampled-animation or
    intermediate-frame images. Drops main's harness runs and the bp35-lordhansolo
    series."""
    if "/bp35-lordhansolo/" in run_dir + "/":
        return False
    head = ""
    for path in glob.glob(f"{run_dir}/transcripts/*.txt")[:1]:
        head = open(path, errors="ignore").read(200_000)
    setup = ""
    for name in ("setup_commands.json", "taaf_setup_env.json"):
        if os.path.exists(f"{run_dir}/{name}"):
            setup += open(f"{run_dir}/{name}", errors="ignore").read()
    if "tool_output_tokens: 3072" not in head:
        return False
    if re.search(r"ARC3_ANIMATION_IMAGE\W{1,6}sampled", setup) or re.search(r"MULTIMODAL_INTERMEDIATE_FRAMES\W{1,6}true", setup):
        return False
    return True


def budget(run_dir):
    for path in glob.glob(f"{run_dir}/transcripts/*.txt")[:1]:
        m = re.search(r"context_budget_tokens: (\d+)", open(path, errors="ignore").read(200_000))
        return int(m.group(1)) if m else None


def calibrate():
    """Least squares of server prompt_tokens on (1, cumulative chars, images)
    at each turn's first request, before the game's first trim."""
    rows = []
    for req in glob.glob(f"{RUNS_DIR}/*/*/v*/*_requests.jsonl"):
        run_dir, stem = os.path.dirname(req), os.path.basename(req).removesuffix("_requests.jsonl")
        transcript = f"{run_dir}/transcripts/{stem}.txt"
        if not os.path.exists(transcript) or budget(run_dir) != BUDGET:
            continue
        first, peak, trimmed = {}, 0, set()
        for line in open(req):
            e = json.loads(line)
            if e.get("event") == "response":
                p = e["usage"]["prompt_tokens"]
                if p < peak - 25_000:
                    trimmed.add(e["analysis_step"])
                peak = max(peak, p)
                first.setdefault(e["analysis_step"], (p, bool(trimmed)))
        text = open(transcript, errors="ignore").read()
        steps = [int(m.group(1)) for m in HEADER.finditer(text)]
        chars = images = 0
        seen = set()
        for step, (_, c, i) in zip(steps, transcript_steps(transcript)):
            if step in first and step not in seen and step > 1 and not first[step][1]:
                rows.append((chars, images, first[step][0]))
            seen.add(step)
            chars += c
            images += i
    a = np.array(rows, dtype=float)
    coef, *_ = np.linalg.lstsq(np.c_[np.ones(len(a)), a[:, 0], a[:, 1]], a[:, 2], rcond=None)
    err = np.abs(np.c_[np.ones(len(a)), a[:, 0], a[:, 1]] @ coef / a[:, 2] - 1)
    print(f"calibration: n={len(a)} base={coef[0]:.0f} chars/token={1 / coef[1]:.2f} "
          f"tokens/image={coef[2]:.0f}; abs error median {np.median(err):.1%}, p90 {np.quantile(err, .9):.1%}")
    return coef


def attempts(coef):
    rows, runs = [], set()
    for run_dir in sorted(glob.glob(f"{RUNS_DIR}/*/*/v*")):
        if budget(run_dir) != BUDGET or not daniel_base(run_dir):
            continue
        for transcript in glob.glob(f"{run_dir}/transcripts/*.txt"):
            stem = os.path.basename(transcript)[:-4]
            game = stem.split("-")[0]
            events = f"{run_dir}/artifacts/{stem}_events.jsonl"
            if game not in LATER15 or not os.path.exists(events):
                continue
            steps = transcript_steps(transcript)
            if not steps:
                continue
            runs.add(run_dir)
            # untrimmed context at the end of each turn
            ends = coef[0] + np.cumsum([c for _, c, _ in steps]) * coef[1] + np.cumsum([i for _, _, i in steps]) * coef[2]
            pointers = [a for a, _, _ in steps]
            start, level = coef[0], 1
            for action in level_up_actions(events):
                turn = max((k for k, p in enumerate(pointers) if p <= action), default=0)
                rows.append(dict(run=run_dir, game=game, level=level, start=start, end=ends[turn], solved=True))
                start, level = ends[turn], level + 1
            rows.append(dict(run=run_dir, game=game, level=level, start=start, end=ends[-1], solved=False))
    print("runs:", " ".join(sorted(r[len(RUNS_DIR) + 1:] for r in runs)))
    print(f"{len(runs)} runs, {pd.DataFrame(rows).groupby(['run', 'game']).ngroups} game runs")
    return pd.DataFrame(rows)


def simulate_trimming(starts, hi=TRIM_AT, lo=TRIM_TO):
    """Current context per turn: grows with the untrimmed estimate, and once it
    passes `hi` the drain drops it to `lo`. Fit against server prompt_tokens:
    median error 3k tokens, correlation 0.60 inside the 50-118k band."""
    current, c = np.empty_like(starts), starts[0]
    for k in range(len(starts)):
        if k:
            c += starts[k] - starts[k - 1]
        if c > hi:
            c = lo
        current[k] = c
    return current


def turns(coef, real=False):
    """One row per turn: level, current context at its start, the context it
    added, whether it solved the level, and whether the game had trimmed yet.
    real=True uses server prompt_tokens instead (runs with request logs only)."""
    rows = []
    for run_dir in sorted(glob.glob(f"{RUNS_DIR}/*/*/v*")):
        if budget(run_dir) != BUDGET or not daniel_base(run_dir):
            continue
        for transcript in glob.glob(f"{run_dir}/transcripts/*.txt"):
            stem = os.path.basename(transcript)[:-4]
            game = stem.split("-")[0]
            events = f"{run_dir}/artifacts/{stem}_events.jsonl"
            requests = f"{run_dir}/{stem}_requests.jsonl"
            if game not in LATER15 or not os.path.exists(events) or (real and not os.path.exists(requests)):
                continue
            steps = transcript_steps(transcript)
            if len(steps) < 2:
                continue
            ends = coef[0] + np.cumsum([c for _, c, _ in steps]) * coef[1] + np.cumsum([i for _, _, i in steps]) * coef[2]
            starts = np.r_[coef[0], ends[:-1]]
            if real:
                first = {}
                for line in open(requests):
                    e = json.loads(line)
                    if e.get("event") == "response":
                        first.setdefault(e["analysis_step"], e["usage"]["prompt_tokens"])
                numbers = [int(m.group(1)) for m in HEADER.finditer(open(transcript, errors="ignore").read())]
                current = np.array([first.get(s, np.nan) for s in numbers], dtype=float)
                peak = np.fmax.accumulate(np.nan_to_num(current))
                trimmed = np.r_[False, np.logical_or.accumulate(current[1:] < peak[:-1] - 25_000)]
            else:
                current = simulate_trimming(starts)
                trimmed = np.logical_or.accumulate(np.r_[False, np.diff(current) < 0])
            pointers = [a for a, _, _ in steps]
            solve_turn = {}
            for level, action in enumerate(level_up_actions(events), start=1):
                solve_turn.setdefault(max((k for k, p in enumerate(pointers) if p <= action), default=0), level)
            level = 1
            for k in range(len(steps)):
                up = k in solve_turn
                rows.append(dict(attempt=f"{run_dir}|{game}", game=game, level=level, current=current[k],
                                 added=ends[k] - starts[k], up=int(up), trimmed=bool(trimmed[k])))
                if up:
                    level = solve_turn[k] + 1
    return pd.DataFrame(rows).dropna(subset=["current"])


def rate_by_current(d, lo=50_000, hi=120_000):
    """Level-ups per 10k of added context, by current-context window."""
    out = []
    for a in range(lo, hi, WINDOW):
        w = d[(d.current >= a) & (d.current < a + WINDOW)]
        exposure = w.added.sum() / WINDOW
        events = int(w.up.sum())
        lo_ci = chi2.ppf(0.025, 2 * events) / 2 / exposure if events else 0.0
        hi_ci = chi2.ppf(0.975, 2 * events + 2) / 2 / exposure if exposure else np.nan
        out.append((a, events, exposure, events / exposure if exposure else np.nan, lo_ci, hi_ci))
    return pd.DataFrame(out, columns=["lo", "events", "exposure", "rate", "ci_lo", "ci_hi"])


def current_context_trend(d):
    """Poisson rate of level-ups per added context vs current context, within
    game and level, errors clustered by game run."""
    d = d[d.added > 0].assign(x=lambda t: t.current / WINDOW)
    fit = smf.glm("up ~ x + C(game) + C(level)", d, family=sm.families.Poisson(),
                  offset=np.log(d.added / WINDOW)).fit(cov_type="cluster",
                                                       cov_kwds={"groups": pd.factorize(d.attempt)[0]})
    lo, hi = np.exp(fit.conf_int().loc["x"])
    return np.exp(fit.params.x), lo, hi, fit.pvalues.x, len(d), int(d.up.sum())


def hazard(d, upto=400_000):
    out = []
    for lo in range(0, upto, WINDOW):
        at_risk = d[(d.start <= lo) & (d.end >= lo)]
        events = int((at_risk.solved & (at_risk.end < lo + WINDOW)).sum())
        out.append((lo, len(at_risk), events))
    t = pd.DataFrame(out, columns=["lo", "at_risk", "events"])
    t["rate"] = t.events / t.at_risk.where(t.at_risk > 0)
    return t


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main():
    coef = calibrate()
    df = attempts(coef)
    # x = untrimmed context added since the level started; level 1 counts from
    # game start, so its base prompt is included and the budget line applies
    later = df.level > 1
    df.loc[later, "end"] = df.loc[later, "end"] - df.loc[later, "start"]
    df.loc[later, "start"] = 0.0
    for level in (1, 2, 3):
        d = df[df.level == level]
        print(f"level {level}: {d.solved.sum()}/{len(d)} solved; "
              f"solve point median {d[d.solved].end.median() / 1000:.0f}k")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for level, shift in ((1, -2.5), (2, 0), (3, 2.5)):
        t = hazard(df[df.level == level])
        t = t[t.at_risk >= MIN_AT_RISK]
        ci = np.array([wilson(e, n) for e, n in zip(t.events, t.at_risk)])
        x = (t.lo + WINDOW / 2) / 1000 + shift
        ax.errorbar(x, t.rate, yerr=[t.rate - ci[:, 0], ci[:, 1] - t.rate], fmt="o-", color=COLORS[level],
                    markersize=4, linewidth=1.2, elinewidth=0.8, label=f"level {level}")
    ax.axvline(BUDGET / 1000, color=MUTED, linestyle="--", linewidth=1)
    ax.text(BUDGET / 1000, 0.98, f" context budget {BUDGET / 1000:.0f}k: level 1 trimming starts", color=INK, fontsize=8,
            va="top", transform=ax.get_xaxis_transform())
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)
    ax.set_xlabel("untrimmed context since the level started (k tokens; level 1: since game start)", color=INK, fontsize=9)
    ax.set_ylabel("solved in the next 10k\n(given unsolved so far)", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title(f"Later 15 games: solve rate vs untrimmed context on the level (windows with >= {MIN_AT_RISK} attempts at risk)",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor="#fcfcfb")
    for level in (1, 2, 3):
        t = hazard(df[df.level == level])
        t = t[t.at_risk >= MIN_AT_RISK]
        print(f"level {level}: " + " ".join(f"{lo // 1000}k:{r:.2f}({n})" for lo, r, n in zip(t.lo, t.rate, t.at_risk)))
    plot_current(coef)


def plot_current(coef):
    """Levels 2-3 against the current (trimmed, oscillating) context, after the
    game's first trim so x is the sawtooth rather than time on the level."""
    sim = turns(coef)
    real = turns(coef, real=True)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for level, shift in ((2, -1.5), (3, 1.5)):
        d = sim[(sim.level == level) & sim.trimmed]
        t = rate_by_current(d)
        t = t[t.exposure >= 20]
        x = (t.lo + WINDOW / 2) / 1000 + shift
        ax.errorbar(x, t.rate, yerr=[t.rate - t.ci_lo, t.ci_hi - t.rate], fmt="o-", color=COLORS[level],
                    markersize=4, linewidth=1.2, elinewidth=0.8, label=f"level {level}")
        print(f"level {level} by current context: " + " ".join(
            f"{a // 1000}k:{r:.3f}({e}/{x_:.0f})" for a, r, e, x_ in zip(t.lo, t.rate, t.events, t.exposure)))
    for name, d in (("simulated, 30 runs", sim), ("real prompt_tokens, request-log runs", real)):
        d = d[d.level.isin([2, 3]) & d.trimmed]
        rr, lo, hi, p, n, k = current_context_trend(d)
        print(f"levels 2-3 after first trim, {name}: rate ratio per +10k current context {rr:.3f} "
              f"(95% CI {lo:.3f}-{hi:.3f}, p={p:.2g}); turns {n}, level-ups {k}")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)
    ax.set_xlabel("current context at the start of the turn (k tokens; simulated trimming, after the game's first trim)",
                  color=INK, fontsize=9)
    ax.set_ylabel("level-ups per 10k\nof added context", color=INK, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Later 15 games: levels 2-3 vs current context (windows with >= 20 x 10k added)",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(OUT_CURRENT, dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
