"""Derived run-table values and separate, editable leaderboard annotations."""
from __future__ import annotations

import json
import math
import statistics
import threading
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

_ANNOTATION_LOCK = threading.Lock()
ANNOTATIONS_FILE = "comparison_annotations.json"


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    a, b = _number(numerator), _number(denominator)
    return a / b if a is not None and b is not None and b > 0 else None


def run_summary(run_dir: Path, score: dict) -> dict:
    games = score.get("games") or {}
    totals = (score.get("usage") or {}).get("totals") or {}
    scores = [value for game in games.values() if (value := _number(game.get("score"))) is not None]
    levels = possible = progressed = won = trials = 0
    for game in games.values():
        completed = game.get("levels_completed") or {}
        states = game.get("states") or {}
        for trial, count in completed.items():
            count, total = _number(count), _number(game.get("total_levels"))
            if count is None:
                continue
            trials += 1
            levels += count
            progressed += count > 0
            won += states.get(trial) == "won" or (total is not None and total > 0 and count >= total)
            if total is not None:
                possible += total
    generated = totals.get("generated_tokens")
    actions, turns = totals.get("actions"), totals.get("turns")
    result = {
        "game_count": len(games),
        "mean_score": score.get("score"),
        "median_score": statistics.median(scores) if scores else None,
        "generated_tokens": generated, "actions": actions, "turns": turns,
        "levels_completed": levels if trials else None,
        "won_trials": won if trials else None, "trial_count": trials,
        "coverage_pct": 100 * progressed / trials if trials else None,
        "depth_pct": 100 * levels / possible if possible and all(
            _number(game.get("total_levels")) is not None for game in games.values()
        ) else None,
        "tokens_per_turn": _ratio(generated, turns),
        "actions_per_turn": _ratio(actions, turns),
        "tokens_per_level": _ratio(generated, levels),
        "tokens_per_score": _ratio(generated, score.get("score")),
    }
    try:
        benchmark = json.loads((run_dir / "benchmark.json").read_text())
        result["started_at"] = benchmark.get("start_time")
        start = datetime.fromisoformat(benchmark["start_time"])
        end = datetime.fromisoformat(benchmark["end_time"])
        result["runtime_seconds"] = max(0, (end - start).total_seconds())
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return result


def filter_score(score: dict, selected: set[str] | None = None, passes: set[str] | None = None) -> dict:
    """Recompute game-dependent statistics without changing the saved evaluation."""
    games = {key: game for key, game in (score.get("games") or {}).items()
             if not selected or key in selected or key.split("-")[0] in selected}
    if passes:
        filtered = {}
        trial_order = (score.get("metadata") or {}).get("trial_labels") or []
        for key, original in games.items():
            labels = list((original.get("trial_scores") or {}).keys())
            order = [label for label in trial_order if label in labels] or labels
            chosen = {label for index, label in enumerate(order)
                      if label in passes or str(index) in passes or f"p{index}" in passes}
            if not chosen:
                continue
            game = dict(original)
            # Filter every per-trial field and recompute its per-game mean.
            for field, mapping in original.items():
                if not isinstance(mapping, dict) or not set(mapping).intersection(labels):
                    continue
                game[field] = {label: value for label, value in mapping.items() if label in chosen}
                numbers = [value for raw in game[field].values() if (value := _number(raw)) is not None]
                if numbers:
                    game[field + "_mean"] = statistics.fmean(numbers)
            game["score"] = statistics.fmean(game["trial_scores"].values())
            game["trial_count"] = len(chosen)
            filtered[key] = game
        games = filtered
    result = {**score, "games": games}
    game_scores = [game["score"] for game in games.values() if _number(game.get("score")) is not None]
    result["score"] = statistics.fmean(game_scores) if game_scores else None
    keys = ("actions", "turns", "generated_tokens", "total_tokens", "uncached_input_tokens")
    totals = {}
    means = {}
    for key in keys:
        values = [value for game in games.values() for raw in (game.get(key) or {}).values()
                  if (value := _number(raw)) is not None]
        totals[key] = sum(values) if values else None
        per_game = [value for game in games.values() if (value := _number(game.get(key + "_mean"))) is not None]
        means[key] = statistics.fmean(per_game) if per_game else None
    for key in ("tokens_per_action", "tokens_per_turn"):
        values = [value for game in games.values() if (value := _number(game.get(key + "_mean"))) is not None]
        means[key] = statistics.fmean(values) if values else None
    result["usage"] = {"totals": totals, "mean": means}
    lane_map = (score.get("progression_lanes") or {}).get("tokens", {})
    selected_lanes = {game: {trial: points for trial, points in lane_map.get(game, {}).items()
                             if trial in (games[game].get("trial_scores") or {})}
                      for game in games}
    lanes = [(game, points, 1 / len(by_trial)) for game, by_trial in selected_lanes.items()
             for points in by_trial.values() if points]
    if len({game for game, _, _ in lanes}) != len(games) or not games:
        result["score_progression"] = {"status": "unavailable"}
        return result
    knots = sorted({0, *(point[0] for _, points, _ in lanes for point in points)})
    prepared = [(game, points, [point[0] for point in points], weight) for game, points, weight in lanes]
    mean_points, median_points = [], []
    for knot in knots:
        cost = 0
        per_game = {game: 0 for game in games}
        for game, points, xs, weight in prepared:
            budget = min(knot, xs[-1])
            index = bisect_right(xs, budget) - 1
            per_game[game] += (points[index][1] if index >= 0 else 0) * weight
            cost += budget * weight
        x = cost / len(games)
        mean_points.append([x, statistics.fmean(per_game.values())])
        median_points.append([x, statistics.median(per_game.values())])
    result["score_progression"] = {"status": "complete", "points": mean_points, "median_points": median_points}
    return result


def load_annotations(runs_dir: Path) -> dict:
    try:
        value = json.loads((runs_dir / ANNOTATIONS_FILE).read_text())
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Run annotations must be a JSON object keyed by run ID.")
    return value


def save_annotation(runs_dir: Path, run_id: str, annotation: dict, *, valid_runs: set[str]) -> dict:
    if run_id not in valid_runs:
        raise ValueError("Select an existing evaluated run.")
    if not isinstance(annotation, dict) or set(annotation) - {"leaderboard_score", "notes"}:
        raise ValueError("Only leaderboard_score and notes can be edited.")
    score = annotation.get("leaderboard_score")
    if score is not None and (_number(score) is None or not 0 <= score <= 100):
        raise ValueError("Leaderboard score must be between 0 and 100, or null.")
    notes = annotation.get("notes", "")
    if not isinstance(notes, str) or len(notes) > 20000:
        raise ValueError("Notes must be text of at most 20,000 characters.")
    saved = {"leaderboard_score": score, "notes": notes,
             "updated_at": datetime.now(timezone.utc).isoformat()}
    with _ANNOTATION_LOCK:
        records = load_annotations(runs_dir)
        records[run_id] = saved
        with NamedTemporaryFile(mode="w", dir=runs_dir, suffix=".tmp", delete=False) as file:
            temp = Path(file.name)
            try:
                json.dump(records, file, indent=2)
                file.write("\n")
            except Exception:
                temp.unlink(missing_ok=True)
                raise
        try:
            temp.replace(runs_dir / ANNOTATIONS_FILE)
        finally:
            temp.unlink(missing_ok=True)
    return saved
