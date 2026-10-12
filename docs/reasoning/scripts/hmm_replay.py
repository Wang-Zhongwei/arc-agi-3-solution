"""Replay a 'hmm' multiplier on top of the deployed ranking R = v·C.

Calls (docs/reasoning/data/hmm_calls.csv.gz) are placed on each trajectory's own token axis by
piecewise-linear interpolation between level-up knots. The allocation budget stays in generated
tokens; the hmm signal uses thinking only. At token t, H and T are the hmms and thinking tokens of
the calls on the current level within the last W thinking tokens, and d = H / T is the density
(hmms per thinking token). The ranking is

    R = v · C · u,   u = clip((mu / d) ** alpha, lo, hi),

with u = 1 while T < T_min (too little thinking on the level to judge) and u = hi when H = 0.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import random
import statistics as st
import sys
from bisect import bisect_left, bisect_right
from itertools import accumulate
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "docs/allocation/scripts"))
from replay import DATA, GRANT, Traj, hazard, paired, replay, table, value  # noqa: E402

CALLS = ROOT / "docs/reasoning/data/hmm_calls.csv.gz"
OUT = ROOT / "docs/reasoning/data/hmm_replay.md"
MU = 5.8e-3     # pooled hmm density, hmms per thinking token (5.8 per 1k)


class HTraj(Traj):
    def attach(self, rows: list[dict]) -> None:
        end, pos = 0, []
        for r in rows:
            end += int(r["out_tok"])
            pos.append(end)
        mine = [p for p, r in zip(pos, rows) if int(r["levelups"]) > 0]
        k = min(len(mine), len(self.ups))
        xs, ys = [0.0] + mine[:k], [0.0] + self.ups[:k]
        if pos[-1] > xs[-1] and self.end > ys[-1]:
            xs.append(pos[-1]); ys.append(self.end)
        # strictly increasing knots
        kx, ky = [xs[0]], [ys[0]]
        for x, y in zip(xs[1:], ys[1:]):
            if x > kx[-1] and y > ky[-1]:
                kx.append(x); ky.append(y)
        slope = (ky[-1] / kx[-1]) if kx[-1] > 0 else 1.0

        def to_traj(p: float) -> float:
            i = bisect_right(kx, p) - 1
            if i >= len(kx) - 1:
                return ky[-1] + (p - kx[-1]) * slope
            return ky[i] + (p - kx[i]) * (ky[i + 1] - ky[i]) / (kx[i + 1] - kx[i])

        self.cx = [to_traj(p) for p in pos]
        hmm = [int(r["hmm"]) for r in rows]
        think = [int(r["think_tok"]) for r in rows]
        self.cum_h = [0, *accumulate(hmm)]
        self.cum_t = [0, *accumulate(think)]  # cum_t[j] is call j's thinking start, cum_t[j + 1] its end

    def hmm(self, t: float, window: float) -> tuple[int, int]:
        """(hmms, thinking tokens) of the calls on the current level that lie within the last
        `window` thinking tokens of the latest call ended by t; the latest call always counts."""
        L, _, _ = self.state(t)
        first = bisect_right(self.cx, (self.ups[L - 1] if L else 0.0) + 1e-9)
        i = bisect_right(self.cx, t + 1e-9) - 1
        if i < first:
            return 0, 0
        j = min(i, max(first, bisect_left(self.cum_t, self.cum_t[i + 1] - window, 0, i + 1)))
        return self.cum_h[i + 1] - self.cum_h[j], self.cum_t[i + 1] - self.cum_t[j]


def vc(tr, t):
    L, s, a = tr.state(t)
    return value(tr, L) * hazard(s, a)


def hmm_rank(alpha: float, window: float = 10_000, t_min: float = 1_000, lo: float = 0.5, hi: float = 2.0):
    def rank(tr: HTraj, t: float) -> float:
        h, th = tr.hmm(t, window)
        base = vc(tr, t)
        if th < t_min or alpha == 0:
            return base
        if h == 0:
            return base * hi
        return base * min(hi, max(lo, (MU / (h / th)) ** alpha))
    return rank


def shrunk_rank(alpha: float, window: float = 10_000, n0: float = 3_000):
    """Earlier form, for comparison: d shrunk toward mu as (H + N0·mu) / (T + N0), no T_min."""
    def rank(tr: HTraj, t: float) -> float:
        h, th = tr.hmm(t, window)
        base = vc(tr, t)
        if th == 0:
            return base
        return base * min(2.0, max(0.5, (MU * (th + n0) / (h + n0 * MU)) ** alpha))
    return rank


def think_rank(alpha: float, window: float = 10_000):
    """Control: same window, thinking volume instead of hmm density (relative to 4k tokens)."""
    def rank(tr: HTraj, t: float) -> float:
        _, th = tr.hmm(t, window)
        base = vc(tr, t)
        return base if th == 0 else base * min(2.0, max(0.5, (max(th, 1) / 4000) ** -alpha))
    return rank


def load() -> dict[str, list[HTraj]]:
    raw = json.loads((DATA / "trajectories.json").read_text())
    by_gid: dict[str, list[dict]] = {}
    with gzip.open(CALLS, "rt") as f:
        for r in csv.DictReader(f):
            by_gid.setdefault(r["gid"], []).append(r)
    runs = {}
    for run, games in raw.items():
        trs = []
        for game, rec in games.items():
            tr = HTraj(rec)
            rows = sorted(by_gid.get(f"{run}/{game}_p0", []), key=lambda r: int(r["call"]))
            if not rows:
                raise SystemExit(f"no calls for {run}/{game}")
            tr.attach(rows)
            trs.append(tr)
        runs[run] = trs
    return runs


def pooled(runs, rank, n=40, games=110, slots=15, budget=150_000, seed=0, skip=None):
    trs = [t for i, r in enumerate(runs.values()) if i != skip for t in r]
    rng = random.Random(seed)
    return st.mean(replay(rng.sample(trs, games), rank, budget, GRANT, slots) for _ in range(n))


def main() -> None:
    runs = load()
    budgets = (100_000, 150_000)
    alphas = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
    S = lambda f, b: [replay(trs, f, b, GRANT) for trs in runs.values()]
    ctrl = {b: S(vc, b) for b in budgets}
    uni = {b: S(lambda tr, t: -t, b) for b in budgets}
    out = [f"{sum(map(len, runs.values()))} trajectories, {len(runs)} runs, K = 14, γ = {GRANT // 1000}k. "
           f"Control R = v·C: " + ", ".join(f"β={b // 1000}k {st.mean(ctrl[b]):.2f}" for b in budgets)
           + "; uniform: " + ", ".join(f"{st.mean(uni[b]):.2f}" for b in budgets) + "."]

    rows, grid = [], {}
    for w in (5_000, 10_000, 20_000):
        for t_min in (1_000, 3_000):
            for a in alphas:
                sc = {b: S(hmm_rank(a, w, t_min), b) for b in budgets}
                grid[(w, t_min, a)] = sc
                rows.append([f"{w // 1000}k", f"{t_min // 1000}k", f"{a:g}"]
                            + [f"{st.mean(sc[b]):.2f}, {paired(sc[b], ctrl[b])}" for b in budgets])
    out += ["\nHmm multiplier u = clip((mu/d)^alpha, 1/2, 2), W and T_min in thinking tokens: mean score, difference vs R = v·C ± SE (runs better)",
            table(["W", "T_min", "alpha"] + [f"β = {b // 1000}k" for b in budgets], rows)]

    rows = []
    for a in (0.25, 0.5, 1.0):
        sc = {b: S(think_rank(a), b) for b in budgets}
        rows.append([f"thinking volume, alpha = {a:g}"] + [f"{st.mean(sc[b]):.2f}, {paired(sc[b], ctrl[b])}" for b in budgets])
    for lo, hi in ((0.25, 4.0), (0.125, 8.0)):
        sc = {b: S(hmm_rank(1.0, 10_000, 1_000, lo, hi), b) for b in budgets}
        rows.append([f"hmm alpha = 1, clip [{lo:g}, {hi:g}]"] + [f"{st.mean(sc[b]):.2f}, {paired(sc[b], ctrl[b])}" for b in budgets])
    sc = {b: S(shrunk_rank(0.5), b) for b in budgets}
    rows.append(["earlier shrinkage form, alpha = 0.5, N0 = 3k"] + [f"{st.mean(sc[b]):.2f}, {paired(sc[b], ctrl[b])}" for b in budgets])
    out += ["\nControls and clip variants (W = 10k, T_min = 1k)",
            table(["ranking"] + [f"β = {b // 1000}k" for b in budgets], rows)]

    # Shorter grants let the ranking react before a short-horizon signal goes stale.
    rows = []
    for g in (10_000, 20_000):
        base = {b: [replay(trs, vc, b, g) for trs in runs.values()] for b in budgets}
        rows.append([f"{g // 1000}k", "R = v·C"] + [f"{st.mean(base[b]):.2f}" for b in budgets])
        for a in (0.5, 1.0, 2.0):
            sc = {b: [replay(trs, hmm_rank(a, 10_000, 1_000), b, g) for trs in runs.values()] for b in budgets}
            rows.append([f"{g // 1000}k", f"alpha = {a:g}"] + [paired(sc[b], base[b]) for b in budgets])
    out += ["\nShorter grants (W = 10k, T_min = 1k): R = v·C at that grant, then difference vs it ± SE (runs better)",
            table(["γ", "ranking"] + [f"β = {b // 1000}k" for b in budgets], rows)]

    # Private-pool shape: K = 15 on 110-game pools, beta = 150k, leave-one-run-out jackknife.
    best = max(grid, key=lambda k: st.mean(grid[k][100_000]) + st.mean(grid[k][150_000]))
    cands = {f"best of grid (W = {best[0] // 1000}k, T_min = {best[1] // 1000}k, alpha = {best[2]:g})": hmm_rank(best[2], best[0], best[1]),
             "earlier shrinkage form, alpha = 0.5, W = 10k, N0 = 3k": shrunk_rank(0.5, 10_000, 3_000),
             "uniform": lambda tr, t: -t}
    n = len(runs)
    rows = []
    for name, f in cands.items():
        th = [pooled(runs, f, n=20, seed=100 + r, skip=r) - pooled(runs, vc, n=20, seed=100 + r, skip=r) for r in range(n)]
        m = st.mean(th)
        rows.append([name, f"{m:+.2f} ± {math.sqrt((n - 1) / n * sum((x - m) ** 2 for x in th)):.2f}"])
    out += ["\nK = 15 on 110-game pools, β = 150k: difference vs R = v·C, leave-one-run-out jackknife ± SE",
            table(["ranking", "vs v·C"], rows)]
    out += alpha_sweep(runs, S, ctrl, budgets)
    text = "\n".join(out) + "\n"
    OUT.write_text(text)
    print(text)


SWEEP = (0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0)


def alpha_sweep(runs, S, ctrl, budgets) -> list[str]:
    """alpha on the deployed score R = v·C·u (W = 10k, T_min = 1k); alpha = 0 is R = v·C.
    The private-pool shape reuses the same pools for every alpha, so differences are paired."""
    run_list = list(runs.values())

    def pool_means(trs, n, seed):
        rng = random.Random(seed)
        ps = [rng.sample(trs, 110) for _ in range(n)]
        return {a: st.mean(replay(p, hmm_rank(a), 150_000, GRANT, 15) for p in ps) for a in SWEEP}

    full = pool_means([t for r in run_list for t in r], 200, 0)
    jack = [pool_means([t for i, r in enumerate(run_list) if i != k for t in r], 40, 100 + k)
            for k in range(len(run_list))]
    n = len(jack)
    rows = []
    for a in SWEEP:
        th = [j[a] - j[0.0] for j in jack]
        m = st.mean(th)
        se = math.sqrt((n - 1) / n * sum((v - m) ** 2 for v in th))
        k14 = [f"{st.mean(ctrl[b]):.2f}" if a == 0 else paired(S(hmm_rank(a), b), ctrl[b]) for b in budgets]
        rows.append([f"{a:g}"] + k14 + [f"{full[a]:.2f}" if a == 0 else f"{full[a] - full[0.0]:+.2f} ± {se:.2f}"])
    best = max(SWEEP, key=full.get)
    halves = [[t for r in run_list[:6] for t in r], [t for r in run_list[6:] for t in r]]
    H = [pool_means(h, 40, 1 + i) for i, h in enumerate(halves)]
    sel = []
    for i in (0, 1):
        a = max(SWEEP, key=H[i].get)
        sel.append(f"- best on runs {'1-6' if i == 0 else '7-11'}: alpha = {a:g}, in-sample "
                   f"{H[i][a] - H[i][0.0]:+.2f}, held-out {H[1 - i][a] - H[1 - i][0.0]:+.2f}")
    return ["\nAlpha sweep on R = v·C·u, W = 10k, T_min = 1k, clip [1/2, 2]. K = 14 columns: paired difference vs "
            "alpha = 0 over the 11 runs. Pool column: K = 15 on 200 pools of 110 games, β = 150k, difference vs "
            "alpha = 0 ± leave-one-run-out jackknife SE (40 pools per replicate).",
            table(["alpha"] + [f"K = 14, β = {b // 1000}k" for b in budgets] + ["K = 15 pools, β = 150k"], rows),
            f"\nBest alpha on the pools: {best:g} ({full[best] - full[0.0]:+.2f}). Selection check on halves of the runs:",
            *sel]


if __name__ == "__main__":
    main()
