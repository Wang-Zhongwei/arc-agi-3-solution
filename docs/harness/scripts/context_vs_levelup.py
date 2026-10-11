"""Level-up probability vs prompt size, from saved request logs.

Each level-up (score increase in the events sidecar) is attributed to the
request whose tool call executed it: the last request whose `action` pointer
(next action to run) is <= the level-completing action number. Prompt size is
the server-reported `usage.prompt_tokens` of that request.

"Steady state" keeps requests after a game's first drain (prompt drops by more
than 25k), i.e. inside the ~53k-118k sawtooth that ARC3_CONTEXT_DRAIN_TOKENS
produces with the 118,272-token budget. Before that, small prompts coincide
with the first, usually easiest, level of every game.

Run from ARC3-Inference/ in the main checkout (runs/ is a Drive symlink):
    python ../docs/harness/scripts/context_vs_levelup.py
Needs pandas, numpy and statsmodels.
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit

RUNS = [
    "matijaludvig/arc3-vcintel-25x1/v01",
    "matijaludvig/arc3-vcintel-lorddraft-25x1/v01",
    "ticbudja/thinking-budget-and-user-prompt-trim/v02",
    "jonathanwang2022/lorddraft-15-games-14-seqs/v01",
    "jonathanwang2022/lorddraft-retain-decode-checkpoints/v01",
]
BINS = [0, 20, 40, 53, 60, 70, 80, 90, 100, 110, 125]
DRAIN_DROP = 25_000


def level_up_actions(events_path):
    ups, best = [], 0
    for line in open(events_path):
        event = json.loads(line)
        if event.get("type") != "action":
            continue
        score = int(event.get("score") or 0)
        if score > best:
            ups.append(int(event["action_num"]))
            best = score
    return ups


def load_requests(run_dir):
    rows = []
    for path in sorted(glob.glob(f"{run_dir}/*_requests.jsonl")):
        game = os.path.basename(path).removesuffix("_requests.jsonl")
        reqs = []
        for line in open(path):
            event = json.loads(line)
            if event.get("event") != "response":
                continue
            usage = event.get("usage") or {}
            reqs.append(dict(
                run=run_dir, game=game, step=event["analysis_step"],
                action=event["action"], ptok=usage.get("prompt_tokens", 0),
                ctok=usage.get("completion_tokens", 0), up=0,
            ))
        pointers = [r["action"] for r in reqs]
        for action in level_up_actions(f"{run_dir}/artifacts/{game}_events.jsonl"):
            owner = max((i for i, a in enumerate(pointers) if a <= action), default=None)
            if owner is not None:
                reqs[owner]["up"] += 1
        peak, drained, since = 0, False, 0
        for r in reqs:
            dropped = r["ptok"] < peak - DRAIN_DROP
            drained |= dropped
            peak = r["ptok"] if dropped else max(peak, r["ptok"])
            r["steady"], r["req_in_level"] = drained, since
            since = 0 if r["up"] else since + 1
        rows += reqs
    return rows


def turn_start_seconds(run_dir):
    """First transcript header time of each analysis step, per game."""
    starts = {}
    for path in glob.glob(f"{run_dir}/transcripts/*.txt"):
        game, previous, offset = os.path.basename(path)[:-4], None, 0
        pattern = r"^--- analysis_step=(\d+) \| action=\d+ \| (\d\d):(\d\d):(\d\d)"
        for m in re.finditer(pattern, open(path).read(), re.M):
            sec = int(m[2]) * 3600 + int(m[3]) * 60 + int(m[4]) + offset
            if previous is not None and sec < previous - 3600:
                offset += 86400
                sec += 86400
            previous = sec
            starts.setdefault((game, int(m[1])), sec)
    return starts


def rate_table(df, unit):
    df = df.assign(bin=pd.cut(df.ptok / 1000, BINS, right=False))
    table = df.groupby("bin", observed=True).agg(n=("u", "size"), ups=("u", "sum"))
    table["p"] = table.ups / table.n
    print(f"\n{unit}: n={len(df)} level-ups={df.u.sum()}")
    print(table.round(4).to_string())


def within_game_or(df, extra=()):
    df = df[df.groupby("gid").u.transform("sum") > 0]
    x = df[["k", *extra]]
    fit = ConditionalLogit(df.u, x, groups=df.gid).fit(disp=0)
    lo, hi = fit.conf_int().loc["k"]
    return (f"OR per +10k = {np.exp(10 * fit.params.k):.3f} "
            f"(95% CI {np.exp(10 * lo):.3f}-{np.exp(10 * hi):.3f}, p={fit.pvalues.k:.3g})")


def main():
    rows = []
    for run in RUNS:
        rows += load_requests(f"runs/kaggle/{run}")
    req = pd.DataFrame(rows)
    req["u"] = (req.up > 0).astype(int)
    req["k"] = req.ptok / 1000
    req["gid"] = req.run + "|" + req.game
    req["log_req_in_level"] = np.log1p(req.req_in_level)
    print(req.groupby("run").agg(requests=("u", "size"), level_ups=("up", "sum")))

    steady = req[req.steady]
    rate_table(req, "All requests")
    rate_table(steady, "Steady-state requests")
    print("\nWithin-game (conditional logit, grouped by run x game):")
    print("  all requests:            ", within_game_or(req))
    print("  all, + log(req in level):", within_game_or(req, ["log_req_in_level"]))
    print("  steady state:            ", within_game_or(steady))

    band = steady[steady.ptok >= 53_000].assign(hi=lambda d: d.ptok >= 85_000)
    counts = band.groupby(["gid", "hi"]).u.agg(["sum", "size"]).unstack(fill_value=0)
    ratio = lambda c: (c[("sum", True)].sum() / c[("size", True)].sum()) / (
        c[("sum", False)].sum() / c[("size", False)].sum())
    rng = np.random.default_rng(0)
    boot = [ratio(counts.loc[rng.choice(counts.index, len(counts))]) for _ in range(5000)]
    print(f"\nSteady 85k+ vs 53-85k per-request rate ratio {ratio(counts):.2f} "
          f"(game bootstrap 95% CI {np.percentile(boot, 2.5):.2f}-{np.percentile(boot, 97.5):.2f})")

    turns = req.groupby(["run", "game", "step"], sort=False).agg(
        ptok=("ptok", "median"), pmax=("ptok", "max"), nreq=("u", "size"),
        ups=("up", "sum"), ctok=("ctok", "sum"), steady=("steady", "first"),
    ).reset_index()
    starts = {}
    for run in RUNS:
        for (game, step), sec in turn_start_seconds(f"runs/kaggle/{run}").items():
            starts[(f"runs/kaggle/{run}", game, step)] = sec
    turns["t0"] = [starts.get(key) for key in zip(turns.run, turns.game, turns.step)]
    turns["dur"] = turns.groupby(["run", "game"]).t0.shift(-1) - turns.t0
    turns["u"] = (turns.ups > 0).astype(int)
    turns["k"] = turns.ptok / 1000
    turns["gid"] = turns.run + "|" + turns.game
    turns["post_drain"] = turns.ptok < turns.groupby(["run", "game"]).pmax.shift(1) - DRAIN_DROP
    rate_table(turns[turns.steady], "Steady-state turns")
    print("  steady turns, within game:", within_game_or(turns[turns.steady]))

    timed = turns.dropna(subset=["dur"])
    print("\nTurns right after a drain vs other turns:")
    print(timed.groupby("post_drain").agg(
        turns=("u", "size"), level_up_rate=("u", "mean"), requests=("nreq", "mean"),
        gen_tokens_median=("ctok", "median"), sec_median=("dur", "median"),
        sec_mean=("dur", "mean")).round(3).to_string())
    share = timed[timed.post_drain].dur.sum() / timed.dur.sum()
    print(f"  share of turn wall time in post-drain turns: {share:.2f}")


if __name__ == "__main__":
    main()
