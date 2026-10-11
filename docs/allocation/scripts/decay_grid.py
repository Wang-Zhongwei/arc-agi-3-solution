"""Grid search over the stall falloff C (action weight w, action half-life h_a, token half-life h).

R = v · C with C = w·2^-(a/h_a)^2 + (1 - w)·2^-(s/h)^2 (B = 0, no efficiency), replayed with
K = 15 slots on G = 110-game pools drawn from the 275 recorded trajectories, so K/G matches
15 streams on the ~110-game private pool. Notation follows ../priority-replay.md. Writes
data/decay_grid.md:

  python docs/allocation/scripts/decay_grid.py
"""
from __future__ import annotations

import json
import math
import random
import statistics as st

from replay import DATA, GRANT, Traj, replay, table, value

BUDGET = 150_000   # tokens per game
SLOTS = 15         # ARC3_MAX_ACTIVE_STREAMS
GAMES = 110        # private pool size
POOLS = 100        # 110-game pools for the grid
JACK_POOLS = 30    # pools per leave-one-run-out replicate
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
ACTION_HALF_LIVES = (43, 58, 72, 86, 100, 115, 144, 173, 230, 345)
TOKEN_HALF_LIVES = (31_000, 46_000, 55_000, 62_000, 70_000, 78_000, 86_000, 100_000, 124_000, 186_000)
CONTROL = (0.25, 115, 62_000)  # ARC3_PRIORITY_ACTION_WEIGHT / ACTION_SCALE / TOKEN_SCALE defaults


def ranking(w: float, ha: float | None, h: float | None):
    def rank(tr: Traj, t: float) -> float:
        L, s, a = tr.state(t)
        act = 0.5 ** ((a / ha) ** 2) if w > 0 else 0.0
        tok = 0.5 ** ((s / h) ** 2) if w < 1 else 0.0
        return value(tr, L) * (w * act + (1 - w) * tok)
    return rank


CONFIGS = ([(0.0, None, h) for h in TOKEN_HALF_LIVES] + [(1.0, ha, None) for ha in ACTION_HALF_LIVES]
           + [(w, ha, h) for w in WEIGHTS[1:-1] for ha in ACTION_HALF_LIVES for h in TOKEN_HALF_LIVES])


def label(c) -> str:
    w, ha, h = c
    return f"w = {w:g}" + (f", h_a = {ha}" if ha else "") + (f", h = {h // 1000}k" if h else "")


def pools(trajs: list[Traj], n: int, seed: int) -> list[list[Traj]]:
    rng = random.Random(seed)
    return [rng.sample(trajs, GAMES) for _ in range(n)]


def scores(rank, ps: list[list[Traj]]) -> list[float]:
    return [replay(p, rank, BUDGET, GRANT, SLOTS) for p in ps]


def main() -> None:
    raw = json.loads((DATA / "trajectories.json").read_text())
    runs = [[Traj(rec) for rec in games.values()] for games in raw.values()]
    ps = pools([t for r in runs for t in r], POOLS, 0)
    S = {c: scores(ranking(*c), ps) for c in CONFIGS}
    uni = st.mean(scores(lambda tr, t: -t, ps))
    mean = {c: st.mean(v) for c, v in S.items()}
    ctrl = mean[CONTROL]
    order = sorted(CONFIGS, key=mean.get, reverse=True)
    text = (f"{len(CONFIGS)} settings, {POOLS} pools of {GAMES} from {sum(map(len, runs))} trajectories, "
            f"K = {SLOTS}, β = {BUDGET // 1000}k, γ = {GRANT // 1000}k. Uniform {uni:.2f}; control "
            f"{label(CONTROL)}: {ctrl:.2f} (rank {order.index(CONTROL) + 1}).\n")
    rows = []
    for w in WEIGHTS:
        cs = [c for c in CONFIGS if c[0] == w]
        best = max(cs, key=mean.get)
        rows.append([f"{w:g}", label(best), f"{mean[best]:.2f}", f"{mean[best] - ctrl:+.2f}"])
    text += "\nBest setting per action weight, mean over pools, difference vs control\n"
    text += table(["w", "best", "score", "vs control"], rows) + "\n"
    text += "\nSingle-term profiles, difference vs control\n" + table(
        ["term"] + [f"{h // 1000}k" for h in TOKEN_HALF_LIVES],
        [["tokens (w = 0), h"] + [f"{mean[(0.0, None, h)] - ctrl:+.2f}" for h in TOKEN_HALF_LIVES]]) + "\n"
    text += table(["term"] + [str(ha) for ha in ACTION_HALF_LIVES],
                  [["actions (w = 1), h_a"] + [f"{mean[(1.0, ha, None)] - ctrl:+.2f}" for ha in ACTION_HALF_LIVES]]) + "\n"
    for w in WEIGHTS[1:-1]:
        text += f"\nw = {w:g}: difference vs control, rows h_a, columns h\n" + table(
            ["h_a \\ h"] + [f"{h // 1000}k" for h in TOKEN_HALF_LIVES],
            [[str(ha)] + [f"{mean[(w, ha, h)] - ctrl:+.2f}" for h in TOKEN_HALF_LIVES] for ha in ACTION_HALF_LIVES]) + "\n"

    # Leave-one-run-out jackknife: pools share trajectories, so the spread over pools understates
    # the error; runs are the independent unit.
    cands = {"uniform": lambda tr, t: -t, "best of grid": ranking(*order[0])}
    for w in (0.0, 1.0):
        best = max((c for c in CONFIGS if c[0] == w), key=mean.get)
        cands[label(best)] = ranking(*best)
    cands[label((0.0, None, 70_000))] = ranking(0.0, None, 70_000)
    theta = {k: [] for k in cands}
    for r in range(len(runs)):
        jp = pools([t for i, rr in enumerate(runs) if i != r for t in rr], JACK_POOLS, 100 + r)
        base = st.mean(scores(ranking(*CONTROL), jp))
        for k, f in cands.items():
            theta[k].append(st.mean(scores(f, jp)) - base)
    n, rows = len(runs), []
    for k, v in theta.items():
        m = st.mean(v)
        rows.append([k + (f" ({label(order[0])})" if k == "best of grid" else ""),
                     f"{m:+.2f} ± {math.sqrt((n - 1) / n * sum((x - m) ** 2 for x in v)):.2f}"])
    text += "\nLeave-one-run-out jackknife, difference vs control ± SE\n" + table(["ranking", "vs control"], rows) + "\n"

    # Two-fold selection check: pick the best setting on pools from one half of the runs,
    # score it on pools from the other half.
    halves = [[t for r in runs[:6] for t in r], [t for r in runs[6:] for t in r]]
    F = [{c: st.mean(scores(ranking(*c), pools(half, 30, 1 + i))) for c in CONFIGS} for i, half in enumerate(halves)]
    rows = []
    for i in (0, 1):
        best = max(CONFIGS, key=F[i].get)
        rows.append([f"runs {'1-6' if i == 0 else '7-11'}", label(best), f"{F[i][best] - F[i][CONTROL]:+.2f}",
                     f"{F[1 - i][best] - F[1 - i][CONTROL]:+.2f}"])
    text += "\nSelection check: best on one half of the runs, scored on the other half (vs control)\n" + table(
        ["selected on", "best", "in-sample", "held-out"], rows) + "\n"
    (DATA / "decay_grid.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
