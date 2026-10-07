"""Offline replay of token allocation policies on recorded game trajectories.

Notation follows ../priority-replay.md. Every table in that report is printed by
this script and written to data/results.md:

  python docs/allocation/scripts/replay.py
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
sys.path.insert(0, str(HERE.parents[2] / "ARC3-Inference"))
from inference.agent.scheduling.priority import (  # noqa: E402
    TAIL_LOOKUP_REMAINING, PrioritySnapshot, parse_total_levels, priority_value,
)

SLOTS = 14                    # ARC3_MAX_ACTIVE_STREAMS
HORIZON = 35_000              # D: window of the fitted level-up probability
BUDGETS = (100_000, 150_000)  # tokens per game; 200k exceeds recorded data
STALL_BINS = [0, 20e3, 40e3, 60e3, 80e3, 100e3, 130e3, 160e3, math.inf]


# ----------------------------------------------------------------- trajectories
class Traj:
    """Recorded score as a step function of the game's own generated tokens."""

    def __init__(self, run: str, game: str, rec: dict):
        self.run, self.game, self.n, self.won = run, game.split("#")[0], rec["n"], rec["won"]
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

    def levels(self, t: float) -> int:
        return bisect_right(self.ups, t + 1e-9)

    def finished(self, t: float) -> bool:
        return self.won and self.levels(t) == len(self.ups)

    def actions(self, t: float) -> float:
        t = min(t, self.end)
        i = max(0, bisect_right(self.tx, t) - 1)
        if i >= len(self.tx) - 1:
            return self.ax[-1]
        x0, x1, a0, a1 = self.tx[i], self.tx[i + 1], self.ax[i], self.ax[i + 1]
        return a0 + (a1 - a0) * ((t - x0) / (x1 - x0) if x1 > x0 else 0.0)

    def state(self, t: float) -> tuple[int, float, float]:
        """(L levels completed, s tokens since last level-up, a actions since last level-up)."""
        L = self.levels(t)
        return L, t - (self.ups[L - 1] if L else 0.0), self.actions(t) - (self.ups_a[L - 1] if L else 0.0)


def load() -> tuple[list[Traj], dict]:
    raw = json.loads((DATA / "trajectories.json").read_text())
    trajs = [Traj(run, game, rec) for run, games in raw.items() for game, rec in games.items()]
    return trajs, json.loads((DATA / "baselines.json").read_text())


# ----------------------------------------------------------------- index terms
def value(tr: Traj, t: float) -> float:
    """v: game score points for clearing the current level k = L+1 at full efficiency."""
    L = tr.levels(t)
    return 100.0 * (L + 1) / (tr.n * (tr.n + 1) / 2) if L < tr.n else 0.0


def eff_current(tr: Traj, t: float) -> float:
    return (25.0 / (25.0 + tr.state(t)[2])) ** 2


def hazard_c(tr: Traj, t: float) -> float:
    _, s, a = tr.state(t)
    return 0.25 * 0.5 ** ((a / 115) ** 2) + 0.75 * 0.5 ** ((s / 62_000) ** 2)


def shipped(tr: Traj, t: float, **over) -> int:
    L, s, a = tr.state(t)
    kw = dict(human_actions=25, action_scale=115, token_scale=62_000, action_weight=0.25,
              tail_base=2, tail_cap=4, normalize_score=True, tail_lookup="remaining")
    kw.update(over)
    return priority_value(PrioritySnapshot(L + 1, int(a), s, 1.0, tr.n, None), **kw)


def tail_default(tr: Traj, t: float) -> float:
    """(A + B) C with lookup off: B = min(4, 2 + k - 1)."""
    L, _, _ = tr.state(t)
    k = L + 1
    A = k * eff_current(tr, t) * 55 / (tr.n * (tr.n + 1) / 2)
    return (A + min(4.0, 2.0 + k - 1)) * hazard_c(tr, t)


class FittedP:
    """P(level-up within the next D tokens | stall bin of s, L = 0 or L >= 1), Laplace-smoothed."""

    def __init__(self, trajs: list[Traj], D: float = HORIZON, step: float = 5e3):
        hit, cnt = defaultdict(float), defaultdict(float)
        for tr in trajs:
            t = 0.0
            while t < tr.end and not tr.finished(t):
                L, s, _ = tr.state(t)
                key = (self.bin(s), min(L, 1))
                nxt = bisect_right(tr.ups, t + 1e-9)
                if nxt < len(tr.ups) and tr.ups[nxt] <= t + D:
                    hit[key] += 1; cnt[key] += 1
                elif t + D <= tr.end:
                    cnt[key] += 1
                t += step
        self.p = {k: (hit[k] + 0.5) / (cnt[k] + 2.0)
                  for k in [(b, l) for b in range(len(STALL_BINS) - 1) for l in (0, 1)]}
        self.n = dict(cnt)

    @staticmethod
    def bin(s: float) -> int:
        return bisect_right(STALL_BINS, s) - 1

    def __call__(self, tr: Traj, t: float) -> float:
        L, s, _ = tr.state(t)
        return self.p[(self.bin(s), min(L, 1))]


def projected_baseline(baselines: dict, exclude: str, pool_from: int = 7) -> dict[int, float]:
    """p_hat(l): median human baseline at level l over the other games; levels >= pool_from pooled."""
    others = [b for g, b in baselines.items() if g != exclude]
    proj = {l: st.median(b[l - 1] for b in others if len(b) >= l) for l in range(1, pool_from)}
    tail = st.median(x for b in others for x in b[pool_from - 1:])
    proj.update({l: tail for l in range(pool_from, 16)})
    return proj


# ----------------------------------------------------------------- replay
def replay(trajs: list[Traj], rank, budget: float, grant: float, slots: int = SLOTS) -> float:
    """Each round the `slots` best-ranked live games receive `grant` tokens; unstarted games first.
    Returns the mean recorded score at the token positions reached."""
    pos = {id(t): 0.0 for t in trajs}
    total, spent = budget * len(trajs), 0.0
    while spent < total:
        live = [t for t in trajs if not t.finished(pos[id(t)]) and pos[id(t)] < t.end - 1]
        if not live:
            break
        live.sort(key=lambda t: (1, 0.0) if pos[id(t)] == 0 else (0, rank(t, pos[id(t)])), reverse=True)
        for t in live[:slots]:
            g = min(grant, total - spent)
            if g <= 0:
                break
            pos[id(t)] += g
            spent += g
    return st.mean(t.score(pos[id(t)]) for t in trajs)


def policies(fitted: FittedP, baselines: dict) -> dict:
    def v_e_p(tr, t, kind):
        L, _, a = tr.state(t)
        if kind == "proj-min":
            p = projected_baseline_cache[tr.game][min(L + 1, 15)]
            e = 1.0 if a <= 0 else min(1.0, (p / a) ** 2)
        elif kind == "proj-offset":
            p = projected_baseline_cache[tr.game][min(L + 1, 15)]
            e = (p / (p + a)) ** 2
        elif kind == "true-min":
            b = baselines[tr.game]
            e = 1.0 if a <= 0 else min(1.0, (b[min(L, len(b) - 1)] / a) ** 2)
        return value(tr, t) * e * hazard_c(tr, t)

    projected_baseline_cache = {g: projected_baseline(baselines, g) for g in baselines}
    return {
        "uniform": lambda tr, t: -t,
        "shipped": lambda tr, t: shipped(tr, t),
        "shipped, TOKEN_SCALE=150k, action weight 0": lambda tr, t: shipped(tr, t, token_scale=150_000, action_weight=0.0),
        "lookup off, B=min(4,k+1)": tail_default,
        "B=0: v·e·C": lambda tr, t: value(tr, t) * eff_current(tr, t) * hazard_c(tr, t),
        "B=0, no efficiency: v·C": lambda tr, t: value(tr, t) * hazard_c(tr, t),
        "C": hazard_c,
        "P": fitted,
        "v·P": lambda tr, t: value(tr, t) * fitted(tr, t),
        "v·e·P": lambda tr, t: value(tr, t) * eff_current(tr, t) * fitted(tr, t),
        "stall: −s": lambda tr, t: -tr.state(t)[1],
        "e = (p̂/(p̂+a))²": lambda tr, t: v_e_p(tr, t, "proj-offset"),
        "e = min(1,(p̂/a)²)": lambda tr, t: v_e_p(tr, t, "proj-min"),
        "e = min(1,(b/a)²), true b": lambda tr, t: v_e_p(tr, t, "true-min"),
    }


def lookahead(grant):
    return lambda tr, t: tr.score(t + grant) - tr.score(t)


# ----------------------------------------------------------------- tables
def paired(x: list[float], y: list[float]) -> str:
    d = [a - b for a, b in zip(x, y)]
    se = st.stdev(d) / math.sqrt(len(d))
    return f"{st.mean(d):+.2f} ± {se:.2f} ({sum(v > 1e-9 for v in d)}/{len(d)})"


class Runner:
    def __init__(self, trajs, baselines):
        self.runs = sorted({t.run for t in trajs})
        self.by_run = {r: [t for t in trajs if t.run == r] for r in self.runs}
        self.pol = {r: policies(FittedP([t for t in trajs if t.run != r]), baselines) for r in self.runs}
        self.cache = {}

    def scores(self, name: str, budget: float, grant: float, slots: int = SLOTS) -> list[float]:
        key = (name, budget, grant, slots)
        if key not in self.cache:
            self.cache[key] = [
                replay(self.by_run[r], lookahead(grant) if name == "lookahead" else self.pol[r][name],
                       budget, grant, slots)
                for r in self.runs]
        return self.cache[key]


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    trajs, baselines = load()
    R = Runner(trajs, baselines)
    out: list[str] = [f"Generated by `scripts/replay.py`: {len(trajs)} trajectories, {len(R.runs)} runs, "
                      f"{SLOTS} slots, D = {HORIZON // 1000}k. Cells: paired difference ± SE (runs better / runs)."]

    # T1 coverage
    rows = []
    for b in (100e3, 150e3, 200e3):
        cov = st.mean(sum(t.end for t in R.by_run[r]) / (len(R.by_run[r]) * b) for r in R.runs)
        rows.append([f"{b/1e3:.0f}k", f"{cov:.2f}"])
    out += ["\n## T1. Recorded tokens / replay budget", md_table(["β (tokens/game)", "Σ T_g / (G β)"], rows)]

    # T2 main policies, grant 35k
    main_pols = ["shipped", "lookup off, B=min(4,k+1)", "B=0: v·e·C", "B=0, no efficiency: v·C", "v·P",
                 "stall: −s", "shipped, TOKEN_SCALE=150k, action weight 0", "lookahead"]
    for slots, label in ((SLOTS, f"{SLOTS} slots"), (1, "free choice (1 slot per round)")):
        rows = []
        uni = {b: R.scores("uniform", b, 35e3, slots) for b in BUDGETS}
        rows.append(["uniform"] + [f"{st.mean(uni[b]):.1f}" for b in BUDGETS])
        for p in main_pols:
            rows.append([p] + [f"{st.mean(R.scores(p, b, 35e3, slots)):.1f}  {paired(R.scores(p, b, 35e3, slots), uni[b])}"
                               for b in BUDGETS])
        out += [f"\n## T2. Policies vs uniform, grant 35k, {label}",
                md_table(["policy"] + [f"β = {b/1e3:.0f}k" for b in BUDGETS], rows)]

    # T3 decomposition
    rows = []
    uni = {b: R.scores("uniform", b, 35e3) for b in BUDGETS}
    for p in ["C", "P", "B=0: v·e·C", "B=0, no efficiency: v·C", "v·P", "v·e·P"]:
        rows.append([p] + [paired(R.scores(p, b, 35e3), uni[b]) for b in BUDGETS])
    out += ["\n## T3. Decomposition (vs uniform, grant 35k)",
            md_table(["index"] + [f"β = {b/1e3:.0f}k" for b in BUDGETS], rows)]

    # T4 efficiency variants, vs current efficiency
    effs = ["B=0, no efficiency: v·C", "e = (p̂/(p̂+a))²", "e = min(1,(p̂/a)²)", "e = min(1,(b/a)²), true b"]
    rows = []
    for p in effs:
        cells = []
        for g in (35e3, 15e3):
            for b in BUDGETS:
                cells.append(paired(R.scores(p, b, g), R.scores("B=0: v·e·C", b, g)))
        rows.append([p] + cells)
    out += ["\n## T4. Efficiency term inside v·e·C, difference vs e = (25/(25+a))²",
            md_table(["efficiency"] + [f"γ={g/1e3:.0f}k, β={b/1e3:.0f}k" for g in (35e3, 15e3) for b in BUDGETS], rows)]

    # T5 grant size
    rows = []
    for b in BUDGETS:
        for p in ["uniform", "shipped", "B=0: v·e·C", "B=0, no efficiency: v·C", "lookahead"]:
            rows.append([f"{b/1e3:.0f}k", p] + [f"{st.mean(R.scores(p, b, g)):.1f}" for g in (5e3, 15e3, 35e3, 70e3)])
    out += ["\n## T5. Mean score by grant size γ",
            md_table(["β", "policy", "γ=5k", "γ=15k", "γ=35k", "γ=70k"], rows)]

    # T6 life table vs C
    for axis, w, nb, scale in (("tokens", 15e3, 10, 62_000), ("actions", 25, 9, 115)):
        att = []
        for tr in trajs:
            ups = tr.ups if axis == "tokens" else tr.ups_a
            end = tr.end if axis == "tokens" else tr.ax[-1]
            prev = 0.0
            for u in ups:
                att.append((u - prev, True)); prev = u
            if not tr.won:
                att.append((end - prev, False))
        rows, h0 = [], None
        for i in range(nb):
            lo, hi = i * w, (i + 1) * w
            n = sum(1 for x, _ in att if x >= lo)
            d = sum(1 for x, c in att if c and lo <= x < hi)
            cz = sum(1 for x, c in att if not c and lo <= x < hi)
            h = d / (n - cz / 2)
            h0 = h if h0 is None else h0
            curve = h0 * 0.5 ** ((((lo + hi) / 2) / scale) ** 2)
            rows.append([f"{lo:g}–{hi:g}", str(n), str(d), str(cz), f"{h:.1%}", f"{curve:.1%}"])
        out += [f"\n## T6. Completion rate per window, {axis} ({len(att)} level attempts)",
                md_table(["spent on level", "at risk", "completed", "censored", "ĥ", "h₀·2^(−(x/σ)²)"], rows)]

    # T7 fitted P on all data
    fp = FittedP(trajs)
    rows = []
    for l, name in ((0, "L = 0"), (1, "L ≥ 1")):
        rows.append([name] + [f"{fp.p[(i, l)]:.2f}" for i in range(len(STALL_BINS) - 1)])
    out += ["\n## T7. Fitted P(level-up within 35k | s), all runs",
            md_table(["", *[f"{STALL_BINS[i]/1e3:.0f}–{STALL_BINS[i+1]/1e3:.0f}k" if STALL_BINS[i + 1] < math.inf
                            else f"{STALL_BINS[i]/1e3:.0f}k+" for i in range(len(STALL_BINS) - 1)]], rows)]

    # T8 baselines by level
    rows = []
    for l in range(1, 11):
        v = [b[l - 1] for b in baselines.values() if len(b) >= l]
        rows.append([str(l), f"{st.median(v):.0f}", f"{st.mean(v):.0f}", str(len(v))])
    out += ["\n## T8. Human baseline actions by level (25 public games)",
            md_table(["level", "median", "mean", "games"], rows)]

    # T9 marginal return by own-token bin
    bins = [0, 50e3, 100e3, 150e3, 200e3, 300e3]
    rows = []
    for name, sel in (("all runs", trajs), ("bp35-lordhansolo", [t for t in trajs if "bp35-lordhansolo" in t.run])):
        cells = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            gain = cost = 0.0
            for tr in sel:
                top = min(hi, tr.end)
                if top > lo:
                    gain += tr.score(top) - tr.score(lo)
                    cost += top - lo
            cells.append(f"{gain / cost * 1e5:.1f} ({cost / (hi - lo):.0f})")
        rows.append([name] + cells)
    out += ["\n## T9. Score points per 100k tokens by tokens already generated (game-equivalents in bin)",
            md_table(["runs", *[f"{lo/1e3:.0f}–{hi/1e3:.0f}k" for lo, hi in zip(bins[:-1], bins[1:])]], rows)]

    text = "\n".join(out) + "\n"
    (DATA / "results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
