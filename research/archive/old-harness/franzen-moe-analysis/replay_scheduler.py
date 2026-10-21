"""Replay real level costs under different slot-scheduling policies.

Offline runs put 25 games on 10 slots; the scored rerun puts 110 games on 10,
so scheduling matters ~4x more there than our offline scores can show. This
simulator replays per-level token costs observed in our runs (levels.csv) at
the hidden-set ratio and scores each policy with RHAE.

Model (deliberately simple):
- A game is a sequence of levels with token costs taken from one observed
  (run, game) trajectory. The level it was stuck on when its run ended is
  right-censored: its remaining cost is drawn from the pooled Kaplan-Meier
  tail conditional on the tokens it had already used (or never, past the data).
- `slots` games generate at once, each at `rate` tokens/s, for `minutes`.
- A game holds its slot for `quantum` generated tokens (a context trim), then
  every waiting game is ranked by the policy and the best takes the slot.
- With `stall_swap`, a game whose index falls below `stall_threshold` (on a
  level, checked every `check` tokens) releases early, paying `switch_cost`
  tokens of lost time (re-prefilling its history).
- Unstarted games go first, as in Franzen's gate.

Usage:
    python analysis/replay_scheduler.py [--games 110] [--reps 200]
"""
from __future__ import annotations

import argparse
import bisect
import collections
import csv
import math
import random
import statistics as st
import sys
from dataclasses import dataclass, field

sys.path.insert(0, "ARC3-Inference")
from inference.agent.priority_scheduler import PrioritySnapshot, priority_value  # noqa: E402

from fit_levels import kaplan_meier, survival_at  # noqa: E402


@dataclass
class Level:
    cost: float          # generated tokens needed to clear (inf = never)
    efficiency: float    # RHAE efficiency once cleared (min(1.15, (h/a)^2))
    actions: float       # actions it took (for Franzen's A/C)


@dataclass
class Game:
    levels: list[Level]
    total_levels: int
    level: int = 0                 # index of the current level (0-based)
    spent_on_level: float = 0.0
    tokens: float = 0.0
    started: bool = False
    score_parts: list[float] = field(default_factory=list)

    def score(self) -> float:
        """RHAE game score: level-weighted efficiencies, capped at the weighted
        fraction of levels cleared (efficiency above 1 cannot exceed it)."""
        n = self.total_levels
        cleared_weight = sum(range(1, self.level + 1)) if self.level <= len(self.levels) else 0
        return 100 * min(sum(self.score_parts), cleared_weight) / (n * (n + 1) / 2)


def load_trajectories(path: str):
    """{(run, game): [rows...]} ordered by level."""
    trajectories = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        trajectories[(r["run"], r["game"])].append(r)
    for rows in trajectories.values():
        rows.sort(key=lambda r: int(r["level"]))
    return list(trajectories.values())


def make_game(rows, curve, eff_pool, rng, never: float = 0.0) -> Game:
    """Build a Game from one trajectory, completing censored/unplayed levels."""
    n = int(rows[0]["levels_total"])
    levels = []
    for r in rows:
        if r["cleared"] == "1":
            levels.append(Level(float(r["tokens"]), float(r["efficiency"]), float(r["actions"])))
        else:
            # censored: with probability `never` the level is unclearable (the KM
            # tail is unidentified past the last observed clear); otherwise draw
            # its remaining cost from the KM tail given the tokens already used
            used = float(r["tokens"])
            if rng.random() < never:
                levels.append(Level(math.inf, 0.0, float(r["actions"])))
                break
            s_used = survival_at(curve, used)
            u = rng.random() * s_used
            later = [t for t, s in curve if t > used and s <= u]
            levels.append(Level(later[0] if later else math.inf, rng.choice(eff_pool), float(r["actions"])))
            break
    # levels never reached in the source run: resample cleared costs (rare)
    while len(levels) < n and not math.isinf(levels[-1].cost):
        levels.append(Level(rng.choice([lv.cost for lv in levels]), rng.choice(eff_pool), 50.0))
    return Game(levels=levels, total_levels=n)


# ---- policies: value of giving this game the next slot ----

def franzen_index(g: Game, _ctx) -> float:
    if g.level >= len(g.levels):
        return -1.0
    lv = g.levels[g.level]
    actions = lv.actions * min(1.0, g.spent_on_level / lv.cost) if not math.isinf(lv.cost) else 50.0
    snap = PrioritySnapshot(g.level + 1, int(actions), g.spent_on_level, 1.0, g.total_levels)
    return float(priority_value(snap, human_actions=25.0, action_scale=115.0, token_scale=62000.0,
                                action_weight=0.25, tail_base=2.0, tail_cap=4.0,
                                normalize_score=True, tail_lookup="remaining"))


def make_fitted_index(curve, p_next, window=30_000.0, eff=0.9):
    """Reward rate over the next `window` tokens: P(clear soon) * value / window.

    value = this level's weight (level k of N) plus the expected weight of the
    levels after it, chaining the fitted P(clear next | cleared this), all
    normalised by N(N+1)/2 so games of different length compare fairly.
    """
    def continuation(k, n):
        # expected extra weight from levels k+1..n after clearing level k
        total, reach = 0.0, 1.0
        for j in range(k + 1, n + 1):
            reach *= p_next(n - (j - 1))
            total += reach * j * eff
        return total

    def index(g: Game, _ctx) -> float:
        if g.level >= len(g.levels):
            return -1.0
        k, n = g.level + 1, g.total_levels
        s_now = max(1e-9, survival_at(curve, g.spent_on_level))
        p_soon = 1 - survival_at(curve, g.spent_on_level + window) / s_now
        value = (k * eff + continuation(k, n)) / (n * (n + 1) / 2)
        return p_soon * value
    return index


def make_gittins_index(curve, p_next, eff=0.9, step=5_000.0, horizon=150_000.0):
    """Gittins-style index: best (expected value gained) / (expected tokens spent)
    over lookahead windows up to `horizon`, using the fitted survival curve and
    the same chained continuation value as the fitted index."""
    fitted_value = make_fitted_index(curve, p_next, eff=eff)

    def continuation(k, n):
        total, reach = 0.0, 1.0
        for j in range(k + 1, n + 1):
            reach *= p_next(n - (j - 1))
            total += reach * j * eff
        return total

    def index(g: Game, _ctx) -> float:
        if g.level >= len(g.levels):
            return -1.0
        k, n = g.level + 1, g.total_levels
        value = (k * eff + continuation(k, n)) / (n * (n + 1) / 2)
        s0 = g.spent_on_level
        s_now = max(1e-9, survival_at(curve, s0))
        best, spent, tau = 0.0, 0.0, step
        while tau <= horizon:
            # expected tokens spent within the window: integral of survival
            spent += step * survival_at(curve, s0 + tau - step) / s_now
            p_clear = 1 - survival_at(curve, s0 + tau) / s_now
            best = max(best, p_clear * value / max(spent, 1.0))
            tau += step
        return best * 1e5
    _ = fitted_value
    return index


def round_robin(g: Game, _ctx) -> float:
    """No preference: every unfinished game is equal (first waiting goes)."""
    return 1.0 if g.level < len(g.levels) else -1.0


def simulate(games, policy, *, slots, rate, minutes, quantum, stall_swap=False,
             stall_threshold=0.0, check=5_000.0, switch_cost=600.0):
    """Run the slot scheduler; returns mean RHAE score over games."""
    budget_tokens_per_slot = rate * minutes * 60
    clocks = [0.0] * slots          # tokens of work done in each slot
    holder = [None] * slots
    waiting = list(range(len(games)))
    def pick():
        if not waiting:
            return None
        fresh = [i for i in waiting if not games[i].started]
        if fresh:
            i = fresh[0]
        else:
            i = max(waiting, key=lambda i: policy(games[i], None))
            if policy(games[i], None) < 0:
                return None
        waiting.remove(i)
        games[i].started = True
        return i
    for s in range(slots):
        holder[s] = pick()
    while True:
        # advance the slot with the least work done (round-robin in time)
        s = min(range(slots), key=lambda s: clocks[s] if holder[s] is not None else math.inf)
        if holder[s] is None or clocks[s] >= budget_tokens_per_slot:
            break
        g = games[holder[s]]
        run = 0.0
        released = False
        while run < quantum and clocks[s] < budget_tokens_per_slot and g.level < len(g.levels):
            lv = g.levels[g.level]
            step = min(check, quantum - run, budget_tokens_per_slot - clocks[s])
            need = lv.cost - g.spent_on_level
            if need <= step:
                step = need
                g.score_parts.append((g.level + 1) * lv.efficiency)
                g.level += 1
                g.spent_on_level = 0.0
            else:
                g.spent_on_level += step
            g.tokens += step
            run += step
            clocks[s] += step
            if stall_swap and g.level < len(g.levels) and policy(g, None) < stall_threshold:
                clocks[s] += switch_cost
                released = True
                break
        # handover: game re-enters the queue and competes for the slot
        if g.level < len(g.levels):
            waiting.append(holder[s])
        holder[s] = pick()
        if holder[s] is None and not released:
            clocks[s] = math.inf
    return st.mean(g.score() for g in games)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="analysis/levels.csv")
    ap.add_argument("--games", type=int, default=110)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--minutes", type=float, default=532.0)
    ap.add_argument("--rate", type=float, default=61.0, help="tokens/s per active stream")
    ap.add_argument("--slots", type=int, default=10)
    ap.add_argument("--quantum", type=float, default=25_000.0, help="generated tokens per slot hold")
    ap.add_argument("--never", type=float, default=0.0,
                    help="P(a level stuck at run end is unclearable); calibrate with --validate")
    ap.add_argument("--stall", type=float, nargs="*", default=[], help="gittins index thresholds for early swap")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    curve = kaplan_meier([(int(r["tokens"]), int(r["cleared"])) for r in rows])
    eff_pool = [float(r["efficiency"]) for r in rows if r["cleared"] == "1"]
    trajectories = load_trajectories(args.csv)
    p_table = {1: 0.73, 2: 0.77, 3: 0.83}
    p_next = lambda left: p_table.get(left, 0.86)  # noqa: E731
    fitted = make_fitted_index(curve, p_next)

    gittins = make_gittins_index(curve, p_next)
    policies = {
        "round-robin": (round_robin, {}),
        "franzen": (franzen_index, {}),
        "fitted": (fitted, {}),
        "gittins": (gittins, {}),
    }
    for thr in args.stall:
        policies[f"gittins+stall{thr:g}"] = (gittins, {"stall_swap": True, "stall_threshold": thr})
    results = collections.defaultdict(list)
    for rep in range(args.reps):
        rng = random.Random(rep)
        sample = [rng.choice(trajectories) for _ in range(args.games)]
        for name, (policy, extra) in policies.items():
            games = [make_game(t, curve, eff_pool, random.Random(rep * 1000 + i), args.never)
                     for i, t in enumerate(sample)]
            results[name].append(simulate(games, policy, slots=args.slots, rate=args.rate,
                                          minutes=args.minutes, quantum=args.quantum, **extra))
    base = results["franzen"]
    for name, vals in results.items():
        diffs = [v - b for v, b in zip(vals, base)]
        print(f"{name:14} mean {st.mean(vals):6.2f}  vs franzen {st.mean(diffs):+5.2f} "
              f"(paired sd {st.pstdev(diffs):.2f}, better in {sum(d > 0 for d in diffs)}/{len(diffs)})")


if __name__ == "__main__":
    main()
