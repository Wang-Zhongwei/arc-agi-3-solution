"""GRPO's reward, advantages, selection and trainer balance (inference/grpo/reward.py)."""

import itertools
import random
import statistics

import pytest

from inference.grpo import reward


def test_reward_is_the_score_formula_on_l1_l2_with_the_thinking_factor():
    # L1 cleared in 2x the human actions, L2 not: 1 x (1/2)^2 / 3
    r = reward.episode_reward(1, [44, 30], [22, 50], {1: 0, 2: 0}, tau=None)
    assert r["reward"] == pytest.approx(0.25 / 3)
    # both cleared at the human count: (1 + 2) / 3
    assert reward.episode_reward(2, [22, 50], [22, 50], {}, tau=None)["reward"] == pytest.approx(1.0)
    # faster than humans is capped at 1
    assert reward.episode_reward(1, [10, 0], [22, 50], {}, tau=None)["reward"] == pytest.approx(1 / 3)
    # nothing cleared
    assert reward.episode_reward(0, [80, 0], [22, 50], {1: 9999}, tau=100.0)["reward"] == 0.0


@pytest.mark.parametrize("per_action,factor", [(50, 1.2), (25, 1.2), (100, 1.0), (125, 0.9), (150, 0.8), (400, 0.8)])
def test_thinking_factor_runs_from_1_2_to_0_8(per_action, factor):
    # tau 100 thinking tokens per action; 0.5x or less -> 1.2, 1.5x or more -> 0.8, linear between
    assert reward.thinking_factor(per_action * 10, 10, 100.0) == pytest.approx(factor)
    r = reward.episode_reward(1, [22, 0], [22, 50], {1: per_action * 22}, tau=100.0)
    assert r["reward"] == pytest.approx(factor / 3)


def test_max_variance_subset_is_the_best_subset():
    rng = random.Random(0)
    for _ in range(200):
        n, keep = 8, rng.choice([4, 6])
        rewards = [round(rng.random() * rng.choice([0, 1, 0.3]), 2) for _ in range(n)]
        chosen = reward.max_variance_subset(rewards, keep)
        assert len(chosen) == keep and len(set(chosen)) == keep
        best = max(statistics.pvariance([rewards[j] for j in c]) for c in itertools.combinations(range(n), keep))
        assert statistics.pvariance([rewards[j] for j in chosen]) == pytest.approx(best)


def test_advantages_center_on_the_group_and_skip_equal_groups():
    groups = {"ls20": [0.0, 0.5, 1.0, 0.5], "bp35": [0.2, 0.2, 0.2, 0.2]}
    out = reward.advantages(groups, keep=2)
    assert "bp35" not in out
    picked = dict(out["ls20"])
    assert picked == {0: pytest.approx(-0.5), 2: pytest.approx(0.5)}


def test_assign_balances_tokens():
    lengths = {f"e{i}": n for i, n in enumerate([130, 120, 100, 90, 60, 40, 30, 20])}
    ranks = reward.assign(lengths, 2)
    loads = [sum(lengths[k] for k in r) for r in ranks]
    assert sorted(k for r in ranks for k in r) == sorted(lengths) and abs(loads[0] - loads[1]) <= 20
