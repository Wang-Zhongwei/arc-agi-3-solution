"""Rewards, advantages and which rollouts each trainer gets (decisions.md: Reward; grpo-pipeline.md).

- Reward of an episode: 1/3 x sum over the levels cleared (an episode ends at its budget, so every clear is within
  it) of w_l x min(1, (h_l / a_l)^2) x (1 + beta x clip((1 - rho_l) / saturation, -1, 1)), w = (1, 2), beta 0.2,
  saturation 0.5. rho_l = (thinking tokens on the level / a_l) / tau, tau the game's base-model median thinking
  tokens per action; without a tau the factor is 1.
- Advantage: reward minus the mean of its group (one game's rollouts of one wave), no division by the spread.
- Groups whose rewards are all equal are skipped (no gradient).
- Selection: the `keep` rollouts of a group whose rewards spread the most: the top i and bottom keep - i for the i
  with the largest variance (max-variance down-sampling, as in PODS).
- Minibatches: the selected rollouts are split into n minibatches (4, decided 2026-10-26), one optimizer step each,
  every minibatch mixing the games and each game's high and low advantages.
- Balance: within a minibatch, rollouts go to trainers longest first, each to the trainer with fewer tokens so far.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

LEVEL_WEIGHTS = (1.0, 2.0)
THINKING_BETA = 0.2
THINKING_SATURATION = 0.5
TAU_PATH = Path(__file__).resolve().parents[3] / "docs" / "RL" / "data" / "thinking_per_action.json"


def load_tau(path: Path = TAU_PATH) -> dict[str, float]:
    """game (4 letters) -> base-model median thinking tokens per action on L1-L2."""
    if not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text())
    return {game: float(row["tau"]) for game, row in data.get("games", {}).items() if row.get("tau")}


def thinking_factor(thinking: float, actions: int, tau: float | None, beta: float = THINKING_BETA,
                    saturation: float = THINKING_SATURATION) -> float:
    if not tau or actions <= 0:
        return 1.0
    rho = thinking / actions / tau
    return 1.0 + beta * max(-1.0, min(1.0, (1.0 - rho) / saturation))


def episode_reward(levels_completed: int, actions_per_level: list[int], base_actions_per_level: list[int],
                   thinking_per_level: dict[int, int], tau: float | None, weights=LEVEL_WEIGHTS,
                   beta: float = THINKING_BETA, saturation: float = THINKING_SATURATION) -> dict[str, Any]:
    """The reward and its parts. thinking_per_level is keyed by the 1-based level."""
    total, parts = 0.0, []
    for i, w in enumerate(weights):
        if levels_completed <= i:
            break
        actions, human = int(actions_per_level[i]), int(base_actions_per_level[i])
        if actions <= 0:
            continue
        efficiency = min(1.0, (human / actions) ** 2)
        factor = thinking_factor(thinking_per_level.get(i + 1, 0), actions, tau, beta, saturation)
        total += w * efficiency * factor
        parts.append(dict(level=i + 1, actions=actions, human=human, efficiency=round(efficiency, 4),
                          thinking=thinking_per_level.get(i + 1, 0), thinking_factor=round(factor, 4)))
    return dict(reward=total / sum(weights), levels=parts)


def max_variance_subset(rewards: list[float], keep: int) -> list[int]:
    """Indices of the `keep` rewards with the largest variance: the top i and bottom keep - i for the best i."""
    order = sorted(range(len(rewards)), key=lambda j: (rewards[j], j))
    if keep >= len(rewards):
        return order
    best, best_var = None, -1.0
    for top in range(keep + 1):
        chosen = order[:keep - top] + (order[len(order) - top:] if top else [])
        var = statistics.pvariance([rewards[j] for j in chosen])
        if var > best_var + 1e-12:
            best, best_var = chosen, var
    return sorted(best)


def advantages(groups: dict[str, list[float]], keep: int) -> dict[str, list[tuple[int, float]]]:
    """group -> [(index in the group, advantage)] for the rollouts to train; equal-reward groups drop out."""
    out = {}
    for name, rewards in groups.items():
        if len(rewards) < 2 or max(rewards) - min(rewards) < 1e-9:
            continue
        mean = sum(rewards) / len(rewards)
        out[name] = [(j, rewards[j] - mean) for j in max_variance_subset(rewards, keep)]
    return out


def split_minibatches(items: list[tuple[str, Any, float]], n: int) -> list[list[tuple[str, Any, float]]]:
    """(group, key, advantage) items into n minibatches, one optimizer step each. Each group's items, sorted by
    advantage, are dealt round-robin, continuing where the previous group stopped: a minibatch gets a group's high
    and low rollouts where the group has enough, and sizes differ by at most one. Empty minibatches are dropped."""
    out: list[list[tuple[str, Any, float]]] = [[] for _ in range(max(1, n))]
    by_group: dict[str, list[tuple[str, Any, float]]] = defaultdict(list)
    for item in items:
        by_group[item[0]].append(item)
    cursor = 0
    for group in sorted(by_group):
        for item in sorted(by_group[group], key=lambda it: (-it[2], str(it[1]))):
            out[cursor % len(out)].append(item)
            cursor += 1
    return [m for m in out if m]


def assign(lengths: dict[Any, int], trainers: int) -> list[list[Any]]:
    """Items to trainers, longest first, each to the least-loaded trainer."""
    loads = [0] * trainers
    out: list[list[Any]] = [[] for _ in range(trainers)]
    for key, n in sorted(lengths.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        t = min(range(trainers), key=lambda i: (loads[i], i))
        out[t].append(key)
        loads[t] += n
    return out


def group_by_game(names: list[str]) -> dict[str, list[int]]:
    """Stems like ls20-9607627b_p3 -> {ls20: [indices]}."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(names):
        groups[name[:4]].append(i)
    return dict(groups)
