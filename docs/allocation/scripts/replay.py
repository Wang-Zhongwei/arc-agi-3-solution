"""Offline replay of token allocation policies on recorded game trajectories.

Notation follows ../priority-replay.md. Prints the table of that report and
writes it to data/results.md:

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
from inference.agent.scheduling.priority import ProgressPace, PrioritySnapshot, priority_value  # noqa: E402
from inference.agent.scheduling.ucb import (  # noqa: E402
    UCB_DEFAULT_C, YieldProgress, YieldSnapshot, ucb_value, weighted_levels,
)

PACE_REF = json.loads((HERE.parents[2] / "ARC3-Inference/inference/agent/scheduling/priority_pace_reference.json").read_text())

SLOTS = 14                    # ARC3_MAX_ACTIVE_STREAMS
BUDGETS = (100_000, 150_000)  # tokens per game
GRANT = 35_000               # tokens per slot grant (median between context trims)
TOKENS_PER_MINUTE = 52_000   # v07: 25 games x 218k generated tokens in ~105 min; the UCB clock
UCB_SCALES = (0.25, 1, 4, 16)  # multiples of UCB_DEFAULT_C
CONCURRENCY = (1, 2, 3, 5, 7, 10, 14, 20, 25)  # slots K for the concurrency study (25 = every game)
CLOCK = {"spent": 0.0}       # tokens spent so far in the replayed run


class Traj:
    """Recorded score as a step function of the game's own generated tokens."""

    def __init__(self, rec: dict, thinking: list | None = None):
        self.n, self.won = rec["n"], rec["won"]
        self.think_x = [x for x, _ in thinking or []]
        self.think_t = [r for _, r in thinking or []]
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

    def pace(self, t: float) -> float:
        """ARC3_PRIORITY_PACE multiplier from the levels completed by t (tool_agent.py record_completion)."""
        pace, prev = ProgressPace(), 0.0
        for j, u in enumerate(self.ups[:bisect_right(self.ups, t + 1e-9)], start=1):
            pace.record_completion(u - prev, PACE_REF["by_level"].get(str(j), PACE_REF["fallback_tokens"]))
            prev = u
        return min(2.0, max(0.5, pace.cost_multiplier()))

    def thinking(self, t: float) -> float | None:
        """Reasoning tokens of the last call of the latest turn finished by t (None before the first)."""
        i = bisect_right(self.think_x, t + 1e-9) - 1
        return self.think_t[i] if i >= 0 else None

    def state(self, t: float) -> tuple[int, float, float]:
        """(L levels completed, s tokens since last level-up, a actions since last level-up)."""
        L = bisect_right(self.ups, t + 1e-9)
        return L, t - (self.ups[L - 1] if L else 0.0), self.actions(t) - (self.ups_a[L - 1] if L else 0.0)


def value(tr: Traj, L: int) -> float:
    return 100.0 * (L + 1) / (tr.n * (tr.n + 1) / 2) if L < tr.n else 0.0


def hazard(s: float, a: float, p: float = 1.0) -> float:
    return 0.25 * 0.5 ** ((a / 115) ** 2) + 0.75 * 0.5 ** ((s / (62_000 * p)) ** 2)


def current(tr: Traj, t: float, pace: bool = False) -> int:
    L, s, a = tr.state(t)
    return priority_value(PrioritySnapshot(L + 1, int(a), s, tr.pace(t) if pace else 1.0, tr.n, None),
                          human_actions=25, action_scale=115, token_scale=62_000, action_weight=0.25,
                          tail_base=2, tail_cap=4, normalize_score=True, tail_lookup="remaining")


def b0(tr: Traj, t: float, pace: bool = False) -> float:
    L, s, a = tr.state(t)
    p = tr.pace(t) if pace else 1.0
    return value(tr, L) * (25 / (25 + a)) ** 2 * hazard(s, a, p) / p


def b0_no_eff(tr: Traj, t: float, pace: bool = False) -> float:
    L, s, a = tr.state(t)
    p = tr.pace(t) if pace else 1.0
    return value(tr, L) * hazard(s, a, p) / p


def thinking_multiplier(t: float | None, ref: float = 2048.0) -> float:
    """m = clip((ref / t)^(1/2), 1/2, 2); 1 before the first call (ARC3_PRIORITY_THINKING)."""
    return 1.0 if t is None else min(2.0, max(0.5, (ref / max(1.0, t)) ** 0.5))


def ucb(tr: Traj, t: float, denominator: str, c: float) -> float:
    """Yield-UCB (scheduling/ucb.py): observed weighted-level yield + exploration bonus.
    Elapsed minutes come from the run's tokens spent at TOKENS_PER_MINUTE."""
    progress = YieldProgress()
    progress.observe(0, 0.0)
    for L, u in enumerate(tr.ups[:bisect_right(tr.ups, t + 1e-9)], start=1):
        progress.observe(int(u), weighted_levels(L))
    count = int(t) if denominator == "tokens" else int(round(tr.actions(t)))
    snap = YieldSnapshot(rate=progress.rate(int(t), 0.5), count=count, c=c)
    return ucb_value(snap, CLOCK["spent"] / TOKENS_PER_MINUTE)


UCB_POLICIES = {"yield only (c = 0)": lambda tr, t: ucb(tr, t, "actions", 0.0)}
for _den in ("actions", "tokens"):
    for _k in UCB_SCALES:
        _c = UCB_DEFAULT_C[_den] * _k
        UCB_POLICIES[f"{_den}, c = {_c:g} ({_k:g}x default)"] = (
            lambda tr, t, d=_den, c=_c: ucb(tr, t, d, c))


POLICIES = {
    "uniform": lambda tr, t: -t,
    "current: (A+B)·C": current,
    "B = 0: v·e·C": b0,
    "B = 0, no efficiency: v·C": b0_no_eff,
    "B = 0 + pace: v·e·C(p)/p": lambda tr, t: b0(tr, t, pace=True),
    "B = 0 + thinking: v·e·C·m": lambda tr, t: b0(tr, t) * thinking_multiplier(tr.thinking(t)),
}
EXTRA = {  # printed for reference, not in the report table
    "current + pace": lambda tr, t: current(tr, t, pace=True),
    "B = 0, no efficiency + pace": lambda tr, t: b0_no_eff(tr, t, pace=True),
    "B = 0 + thinking, ref = data median": lambda tr, t: b0(tr, t) * thinking_multiplier(tr.thinking(t), THINK_MEDIAN),
    "current + thinking": lambda tr, t: current(tr, t) * thinking_multiplier(tr.thinking(t)),
}


def replay(trajs: list[Traj], rank, budget: float, grant: float, slots: int = SLOTS) -> float:
    """Each round the SLOTS best-ranked live games get `grant` tokens; unstarted games first."""
    pos = {id(t): 0.0 for t in trajs}
    total, spent = budget * len(trajs), 0.0
    while spent < total:
        live = [t for t in trajs if not t.finished(pos[id(t)]) and pos[id(t)] < t.end - 1]
        if not live:
            break
        CLOCK["spent"] = spent
        live.sort(key=lambda t: (1, 0.0) if pos[id(t)] == 0 else (0, rank(t, pos[id(t)])), reverse=True)
        for t in live[:slots]:
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
    global THINK_MEDIAN
    raw = json.loads((DATA / "trajectories.json").read_text())
    think = json.loads((DATA / "thinking.json").read_text())
    runs = {run: [Traj(rec, think[run][game]) for game, rec in games.items()] for run, games in raw.items()}
    all_t = sorted(r for g in think.values() for s in g.values() for _, r in s)
    THINK_MEDIAN = st.median(all_t)
    print(f"last-call thinking: median {THINK_MEDIAN:.0f} tokens; <=512: {sum(x <= 512 for x in all_t) / len(all_t):.0%}, "
          f">=8192: {sum(x >= 8192 for x in all_t) / len(all_t):.0%}")
    S = {(p, b): [replay(trs, f, b, GRANT) for trs in runs.values()]
         for p, f in POLICIES.items() for b in BUDGETS}
    rows = [["uniform (baseline)"] + [f"{st.mean(S['uniform', b]):.1f}" for b in BUDGETS]]
    rows += [[p] + [f"{st.mean(S[p, b]):.1f}, {paired(S[p, b], S['uniform', b])}" for b in BUDGETS]
             for p in POLICIES if p != "uniform"]
    uni = ", ".join(f"β={b // 1000}k: {st.mean(S['uniform', b]):.1f}" for b in BUDGETS)
    text = "\n".join([
        f"{sum(map(len, runs.values()))} trajectories, {len(runs)} runs, {SLOTS} slots, γ = {GRANT // 1000}k. "
        f"Uniform: {uni}.",
        "\nMean score, difference vs uniform ± SE (runs better)",
        table(["policy"] + [f"β = {b // 1000}k, γ = {GRANT // 1000}k" for b in BUDGETS], rows),
    ]) + "\n"
    K = {(k, b): [replay(trs, b0_no_eff, b, GRANT, k) for trs in runs.values()]
         for k in CONCURRENCY for b in BUDGETS}
    rows = [[f"K = {k}" + (" (control)" if k == SLOTS else "")]
            + [f"{st.mean(K[k, b]):.1f}" + ("" if k == SLOTS else f", {paired(K[k, b], K[SLOTS, b])}")
               for b in BUDGETS] for k in CONCURRENCY]
    text += "\n".join([
        "\nConcurrency, B = 0 with efficiency ablated: mean score, difference vs K = 14 ± SE (runs better)",
        table(["slots"] + [f"β = {b // 1000}k, γ = {GRANT // 1000}k" for b in BUDGETS], rows),
    ]) + "\n"
    U = {(p, b): [replay(trs, f, b, GRANT) for trs in runs.values()]
         for p, f in UCB_POLICIES.items() for b in BUDGETS}
    rows = [["uniform (baseline)"] + [f"{st.mean(S['uniform', b]):.1f}" for b in BUDGETS]]
    rows += [[p] + [f"{st.mean(U[p, b]):.1f}, {paired(U[p, b], S['uniform', b])}" for b in BUDGETS]
             for p in UCB_POLICIES]
    text += "\n".join([
        f"\nYield-UCB, elapsed minutes = tokens spent / {TOKENS_PER_MINUTE}. "
        "Mean score, difference vs uniform ± SE (runs better)",
        table(["policy"] + [f"β = {b // 1000}k, γ = {GRANT // 1000}k" for b in BUDGETS], rows),
    ]) + "\n"
    (DATA / "results.md").write_text(text)
    print(text)
    for p, f in EXTRA.items():
        sc = {b: [replay(trs, f, b, GRANT) for trs in runs.values()] for b in BUDGETS}
        print(f"{p}: " + " | ".join(f"{st.mean(sc[b]):.1f}, {paired(sc[b], S['uniform', b])}" for b in BUDGETS))


if __name__ == "__main__":
    main()
