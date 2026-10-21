"""Fit level-completion statistics from analysis/levels.csv (see level_dataset.py).

Prints the Kaplan-Meier curve of clearing a level vs generated tokens spent on
it, the conditional chance of eventually clearing given tokens already spent
(the quantity Franzen's C term approximates), the chance the next level is
cleared given this one was, and the token cost of cleared levels by index.
"""
from __future__ import annotations

import collections
import csv
import statistics as st
import sys


def kaplan_meier(attempts):
    """[(t, S(t))] for (tokens, cleared) pairs; cleared=0 is right-censored."""
    events = sorted(attempts)
    at_risk = len(events)
    survival, curve, i = 1.0, [], 0
    while i < len(events):
        t, cleared, censored = events[i][0], 0, 0
        while i < len(events) and events[i][0] == t:
            cleared += events[i][1]
            censored += 1 - events[i][1]
            i += 1
        if cleared:
            survival *= 1 - cleared / at_risk
        at_risk -= cleared + censored
        curve.append((t, survival))
    return curve


def survival_at(curve, t):
    """S(t): probability the level is still uncleared after t tokens."""
    value = 1.0
    for x, s in curve:
        if x > t:
            break
        value = s
    return value


def main(path="analysis/levels.csv"):
    rows = list(csv.DictReader(open(path)))
    attempts = [(int(r["tokens"]), int(r["cleared"])) for r in rows]
    curve = kaplan_meier(attempts)
    grid = (10, 20, 30, 40, 60, 80, 100, 150, 200)
    print("S(t) uncleared after t k-tokens:", {k: round(survival_at(curve, k * 1000), 3) for k in grid})
    # P(clear eventually | still uncleared at t) = 1 - S(inf)/S(t), with S(inf) = curve end
    s_end = curve[-1][1]
    print("KM floor S(inf) =", round(s_end, 3))
    print("P(clear eventually | uncleared at t):",
          {k: round(1 - s_end / max(1e-9, survival_at(curve, k * 1000)), 2) for k in grid})
    print("P(clear within next 30k | uncleared at t):",
          {k: round(1 - survival_at(curve, (k + 30) * 1000) / max(1e-9, survival_at(curve, k * 1000)), 2) for k in grid})
    index = {(r["run"], r["game"], int(r["level"])): r for r in rows}
    nxt = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        k, n = int(r["level"]), int(r["levels_total"])
        if r["cleared"] == "1" and k < n:
            following = index.get((r["run"], r["game"], k + 1))
            bucket = nxt[min(n - k, 4)]
            bucket[0] += 1
            bucket[1] += int(following is not None and following["cleared"] == "1")
    print("P(clear next | cleared this), by levels left after this (4 = 4+):",
          {k: f"{v[1]}/{v[0]}={v[1] / v[0]:.2f}" for k, v in sorted(nxt.items())})
    cost = collections.defaultdict(list)
    for r in rows:
        if r["cleared"] == "1":
            cost[min(int(r["level"]), 8)].append(int(r["tokens"]))
    print("median k-tokens per cleared level by index:", {k: round(st.median(v) / 1000, 1) for k, v in sorted(cost.items())})


if __name__ == "__main__":
    main(*sys.argv[1:])
