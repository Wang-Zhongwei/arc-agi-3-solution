"""Offline replay of token allocation policies on recorded game trajectories.

Notation follows ../priority-replay.md. Prints both tables of that report and
writes them to data/results.md:

  python docs/allocation/scripts/replay.py
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from bisect import bisect_right
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
sys.path.insert(0, str(HERE.parents[2] / "ARC3-Inference"))
from inference.agent.scheduling.priority import PrioritySnapshot, priority_value  # noqa: E402

SLOTS = 14                    # ARC3_MAX_ACTIVE_STREAMS
BUDGETS = (100_000, 150_000)  # tokens per game
GRANTS = (35_000, 15_000)     # tokens per slot grant


class Traj:
    """Recorded score as a step function of the game's own generated tokens."""

    def __init__(self, rec: dict):
        self.n, self.won = rec["n"], rec["won"]
        self.tx = [p[0] for p in rec["tokens"]]
        self.y = [p[1] for p in rec["tokens"]]
        self.ax = [p[0] for p in rec["actions"]]
        self.end = self.tx[-1]
        up = [i for i in range(1, len(self.y)) if self.y[i] > self.y[i - 1] + 1e-9]
        self.ups = [self.tx[i] for i in up]
        self.ups_a = [self.ax[i] for i in up]

    def score(self, t: float) -> float:
        i = bisect_right(self.tx, min(t, self.end) + 1e-9) - 1
        return self.y[i] if i >= 0 else 0.0

    def finished(self, t: float) -> bool:
        return self.won and bisect_right(self.ups, t + 1e-9) == len(self.ups)

    def actions(self, t: float) -> float:
        t = min(t, self.end)
        i = max(0, bisect_right(self.tx, t) - 1)
        if i >= len(self.tx) - 1:
            return self.ax[-1]
        x0, x1, a0, a1 = self.tx[i], self.tx[i + 1], self.ax[i], self.ax[i + 1]
        return a0 + (a1 - a0) * ((t - x0) / (x1 - x0) if x1 > x0 else 0.0)

    def state(self, t: float) -> tuple[int, float, float]:
        """(L levels completed, s tokens since last level-up, a actions since last level-up)."""
        L = bisect_right(self.ups, t + 1e-9)
        return L, t - (self.ups[L - 1] if L else 0.0), self.actions(t) - (self.ups_a[L - 1] if L else 0.0)


def value(tr: Traj, L: int) -> float:
    return 100.0 * (L + 1) / (tr.n * (tr.n + 1) / 2) if L < tr.n else 0.0


def hazard(s: float, a: float) -> float:
    return 0.25 * 0.5 ** ((a / 115) ** 2) + 0.75 * 0.5 ** ((s / 62_000) ** 2)


def current(tr: Traj, t: float) -> int:
    L, s, a = tr.state(t)
    return priority_value(PrioritySnapshot(L + 1, int(a), s, 1.0, tr.n, None),
                          human_actions=25, action_scale=115, token_scale=62_000, action_weight=0.25,
                          tail_base=2, tail_cap=4, normalize_score=True, tail_lookup="remaining")


def b0(tr: Traj, t: float) -> float:
    L, s, a = tr.state(t)
    return value(tr, L) * (25 / (25 + a)) ** 2 * hazard(s, a)


def b0_no_eff(tr: Traj, t: float) -> float:
    L, s, a = tr.state(t)
    return value(tr, L) * hazard(s, a)


POLICIES = {
    "uniform": lambda tr, t: -t,
    "current: (A+B)·C": current,
    "B = 0: v·e·C": b0,
    "B = 0, no efficiency: v·C": b0_no_eff,
}


def replay(trajs: list[Traj], rank, budget: float, grant: float) -> float:
    """Each round the SLOTS best-ranked live games get `grant` tokens; unstarted games first."""
    pos = {id(t): 0.0 for t in trajs}
    total, spent = budget * len(trajs), 0.0
    while spent < total:
        live = [t for t in trajs if not t.finished(pos[id(t)]) and pos[id(t)] < t.end - 1]
        if not live:
            break
        live.sort(key=lambda t: (1, 0.0) if pos[id(t)] == 0 else (0, rank(t, pos[id(t)])), reverse=True)
        for t in live[:SLOTS]:
            g = min(grant, total - spent)
            if g <= 0:
                break
            pos[id(t)] += g
            spent += g
    return st.mean(t.score(pos[id(t)]) for t in trajs)


def paired(x: list[float], y: list[float]) -> str:
    d = [a - b for a, b in zip(x, y)]
    return f"{st.mean(d):+.2f} ± {st.stdev(d) / math.sqrt(len(d)):.2f} ({sum(v > 1e-9 for v in d)}/{len(d)})"


def table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def main() -> None:
    raw = json.loads((DATA / "trajectories.json").read_text())
    runs = {run: [Traj(rec) for rec in games.values()] for run, games in raw.items()}
    S = {(p, b, g): [replay(trs, f, b, g) for trs in runs.values()]
         for p, f in POLICIES.items() for b in BUDGETS for g in GRANTS}
    names = [p for p in POLICIES if p != "uniform"]

    t1 = [[p] + [f"{st.mean(S[p, b, 35_000]):.1f}, {paired(S[p, b, 35_000], S['uniform', b, 35_000])}"
                 for b in BUDGETS] for p in names]
    t2 = [[p] + [c for b in BUDGETS for c in (f"{st.mean(S[p, b, 35_000]):.1f}", f"{st.mean(S[p, b, 15_000]):.1f}",
                                              paired(S[p, b, 15_000], S[p, b, 35_000]))] for p in names]
    uni = ", ".join(f"β={b // 1000}k: {st.mean(S['uniform', b, g]):.1f} (γ={g // 1000}k)"
                    for b in BUDGETS for g in GRANTS)
    text = "\n".join([
        f"{sum(map(len, runs.values()))} trajectories, {len(runs)} runs, {SLOTS} slots. Uniform: {uni}.",
        "\n## Table 1. γ = 35k: mean score, difference vs uniform ± SE (runs better)",
        table(["policy"] + [f"β = {b // 1000}k" for b in BUDGETS], t1),
        "\n## Table 2. Grant size: mean score at γ = 35k and 15k, difference 15k − 35k ± SE (runs better)",
        table(["policy"] + [h for b in BUDGETS for h in (f"β={b // 1000}k, 35k", f"β={b // 1000}k, 15k", f"β={b // 1000}k, Δ")], t2),
    ]) + "\n"
    (DATA / "results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
