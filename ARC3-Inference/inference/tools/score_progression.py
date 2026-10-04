"""Compact score curves (generated tokens and actions) for evaluated public runs."""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from math import isclose
from statistics import median
from typing import Any

# Per-trial curve attribute on eval.GameTrialResult, and its aggregate x metric.
_AXES = {
    "tokens": ("score_progression", "mean_generated_tokens_per_trial"),
    "actions": ("action_progression", "mean_actions_per_trial"),
}


def _action_count(run: Any) -> float:
    return float(len(run.history))


def _game_progression(run: Any, score: float, axis: str) -> list[tuple[float, float]] | None:
    from taaf.diagnostics import _per_game_partial_curve, _total_tokens

    # TAAF's partial curve steps one unit per action for any x_fn other than
    # tokens/wallclock, so the action-count total yields a score-vs-actions curve.
    x_fn = _total_tokens if axis == "tokens" else _action_count
    if not run.history and score == 0:
        return [(0.0, 0.0), (float(x_fn(run)), 0.0)]
    # Never turn missing intermediate states into an invented final-step curve.
    if (run.base_actions_per_level is None
            or len(run.intermediate_states) < len(run.history) + 1):
        return None
    points = _per_game_partial_curve(run, x_fn)
    if not isclose(points[-1][1], score, abs_tol=1e-7):
        return None
    return points


def game_score_progression(run: Any, score: float) -> list[tuple[float, float]] | None:
    return _game_progression(run, score, "tokens")


def game_action_progression(run: Any, score: float) -> list[tuple[float, float]] | None:
    return _game_progression(run, score, "actions")


def _build_progression(games: list[Any], axis: str) -> dict[str, Any]:
    attribute, x_metric = _AXES[axis]
    lanes = [getattr(game, attribute) for game in games]
    available = [points for points in lanes if points is not None]
    result: dict[str, Any] = {
        "version": 2,
        "x_metric": x_metric,
        "y_metric": "public_score",
        "aggregation": "equal_game_mean_of_trial_means",
        "trials_total": len(games),
        "trials_with_history": len(available),
        "points": [],
        "median_points": [],
        "median_aggregation": "median_of_game_trial_means",
    }
    if not games or len(available) != len(games):
        result["status"] = "unavailable"
        return result
    from taaf.diagnostics import _pool_lane_events

    counts = Counter(game.game_id for game in games)
    xs, ys = _pool_lane_events(lanes, weights=[1.0 / counts[game.game_id] for game in games])
    # Use the same budget knots and mean-cost axis for both statistics.
    knots = sorted({x for points in lanes for x, _ in points} | {0.0})
    lane_costs = [[x for x, _ in points] for points in lanes]
    medians = []
    for budget in knots:
        game_scores: dict[str, float] = defaultdict(float)
        for game, points, costs in zip(games, lanes, lane_costs):
            index = bisect_right(costs, min(budget, costs[-1]) + 1e-9) - 1
            score = points[index][1] if index >= 0 else 0.0
            game_scores[game.game_id] += score / counts[game.game_id]
        medians.append(float(median(game_scores.values())))
    result.update(
        status="complete",
        points=[[float(x), float(y)] for x, y in zip(xs, ys)],
        median_points=[[float(x), y] for x, y in zip(xs, medians)],
    )
    return result


def build_score_progression(games: list[Any]) -> dict[str, Any]:
    return _build_progression(games, "tokens")


def build_action_progression(games: list[Any]) -> dict[str, Any]:
    return _build_progression(games, "actions")


def build_progression_lanes(games: list[Any]) -> dict[str, Any]:
    """Per-trial step curves keyed by axis, game id, then trial run name.

    The curves page pools these in the browser so any subset of games can be
    re-aggregated exactly like ``_build_progression``. Trials without a
    reconstructable history are omitted; a game's evaluated trials are listed
    in the evaluation's ``games[*].trial_scores``.
    """
    lanes: dict[str, Any] = {"version": 1}
    for axis, (attribute, _) in _AXES.items():
        by_game: dict[str, dict[str, list[list[float]]]] = {}
        for game in games:
            points = getattr(game, attribute)
            if points is None:
                continue
            by_game.setdefault(game.game_id, {})[game.run_name] = [
                [float(x), float(y)] for x, y in points
            ]
        lanes[axis] = by_game
    return lanes
