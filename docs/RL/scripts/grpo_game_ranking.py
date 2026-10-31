"""Rank the 25 public games for GRPO by how much signal a group of 8 rollouts gives.

A group gives a gradient only when its rewards differ, and the advantage R - mean is larger when they spread more,
so a game's expected signal per group is the variance of its reward over attempts. The reward is the decided one
(docs/RL/decisions.md, Reward) without the thinking factor:
    R = (1 * [L1 within budget] * min(1, (h1/a1)^2) + 2 * [L2 within budget] * min(1, (h2/a2)^2)) / 3
with the budget min(1,750 x human actions of L1 + L2, 130K) context tokens. "Level" columns use the level reward
alone (0, 1/3 or 1): averaged over the 8 round-1 games it gives decisions.md's 0.092 (0.125 with dc22 + m0r0).

Cost: a rollout ends at the L2 clear or at the budget, so its length is the context at the L2 clear if that is within
the budget, else the budget. The wave lasts about as long as the longest rollouts (the budget); trainer time goes with
the mean length.

Reads docs/RL/data/l1_l2_context.json (43 runs, 1,075 attempts), and benchmark.json (human actions) and each run's
evaluation.json (actions at each level clear) under RUNS_DIR. Writes docs/RL/data/grpo_game_ranking.json and prints
a markdown table.

    python3 docs/RL/scripts/grpo_game_ranking.py [RUNS_DIR]
"""

import collections
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUNS_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "ARC3-Inference", "runs", "kaggle")
CONTEXT = os.path.join(REPO, "docs", "RL", "data", "l1_l2_context.json")
OUT = os.path.join(REPO, "docs", "RL", "data", "grpo_game_ranking.json")
CAP, PER_HUMAN_ACTION = 130_000, 1_750  # decided 2026-10-23
GROUP = 8
ROUND_1 = {"bp35", "cd82", "g50t", "s5i5", "sk48", "sp80", "tn36", "wa30"}  # decided 2026-10-26
GUARDS = {"ft09", "lp85", "re86", "tr87"}  # kept untrained (decisions.md, When to stop)
ADDED = ("su15", "lf52")  # the B200 run's 2 games on top of round 1 (decided 2026-10-27)


def variance(values):
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def main():
    benchmark = json.load(open(os.path.join(RUNS_DIR, "ticbudja", "reset-daniel-context", "v01", "benchmark.json")))
    humans = {g["game_id"][:4]: g["base_actions_per_level"] for g in benchmark["game_runs"]}
    lanes = {}  # run -> game -> cumulative actions at each level clear
    rows = json.load(open(CONTEXT))
    for run in sorted({row["run"] for row in rows}):
        evaluation = json.load(open(os.path.join(RUNS_DIR, run, "evaluation.json")))
        lanes[run] = {game[:4]: [p[0] for p in next(iter(lane.values()))]
                      for game, lane in evaluation["progression_lanes"]["actions"].items()}

    attempts = collections.defaultdict(list)
    for row in rows:
        game = row["game"]
        h1, h2 = humans[game][:2]
        budget = min(CAP, PER_HUMAN_ACTION * (h1 + h2))
        actions = lanes[row["run"]][game]
        l1 = row["levels"] >= 1 and row["l1"] <= budget
        l2 = row["levels"] >= 2 and row["l2"] <= budget
        e1 = min(1.0, (h1 / max(1.0, actions[1])) ** 2) if l1 else 0.0
        e2 = min(1.0, (h2 / max(1.0, actions[2] - actions[1])) ** 2) if l2 else 0.0
        attempts[game].append(dict(budget=budget, level=(l1 + 2 * l2) / 3, reward=(e1 + 2 * e2) / 3, l1=l1, l2=l2,
                                   tokens=row["l2"] if l2 else budget))

    table = []
    for game, items in attempts.items():
        n = len(items)
        p0 = sum(not a["l1"] for a in items) / n
        p2 = sum(a["l2"] for a in items) / n
        p1 = 1 - p0 - p2
        rewards = [a["reward"] for a in items]
        tokens = sum(a["tokens"] for a in items) / n
        var_reward = variance(rewards)
        table.append(dict(
            game=game, attempts=n, budget=items[0]["budget"], l1=round(1 - p0, 3), l1_l2=round(p2, 3),
            var_level=round(variance([a["level"] for a in items]), 4), var_reward=round(var_reward, 4),
            mean_reward=round(sum(rewards) / n, 3),
            # chance that all 8 rollouts land in the same outcome class (no L1 / L1 only / L1 + L2): no level signal
            p_flat_group=round(p0 ** GROUP + p1 ** GROUP + p2 ** GROUP, 3),
            mean_tokens=round(tokens), var_per_100k_tokens=round(var_reward / tokens * 1e5, 4),
            round_1=game in ROUND_1, guard=game in GUARDS))
    table.sort(key=lambda r: -r["var_reward"])
    for i, r in enumerate(table, 1):
        r["rank"] = i
    json.dump(table, open(OUT, "w"), indent=1)

    print("| Rank | Game | Attempts | Budget | L1 | L1 + L2 | Var(level) | Var(reward) | Mean reward | "
          "P(flat group) | Mean tokens | Var per 100K tokens | Note |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in table:
        note = ", ".join(n for n, on in (("round 1", r["round_1"]), ("guard", r["guard"])) if on)
        print(f"| {r['rank']} | {r['game']} | {r['attempts']} | {r['budget'] / 1000:.0f}K | {r['l1']:.0%} | "
              f"{r['l1_l2']:.0%} | {r['var_level']:.3f} | {r['var_reward']:.3f} | {r['mean_reward']:.2f} | "
              f"{r['p_flat_group']:.2f} | {r['mean_tokens'] / 1000:.0f}K | {r['var_per_100k_tokens']:.3f} | {note} |")
    r1 = [r for r in table if r["round_1"]]
    added = [r for r in table if r["game"] in ADDED]
    for name, chosen in (("round 1", r1), (f"round 1 + {' '.join(r['game'] for r in added)}", r1 + added)):
        print(f"{name} ({len(chosen)} games): mean Var(reward) {sum(r['var_reward'] for r in chosen) / len(chosen):.3f}, "
              f"Var(level) {sum(r['var_level'] for r in chosen) / len(chosen):.3f}; wave {len(chosen) * GROUP} rollouts, "
              f"{sum(r['mean_tokens'] for r in chosen) * GROUP / 1e6:.1f}M context tokens")


if __name__ == "__main__":
    main()
