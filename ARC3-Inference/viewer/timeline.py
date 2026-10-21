"""Score, levels and tokens over wall-clock time, rebuilt from a run's benchmark.json.

Time axis: wall-clock seconds since the *run* started
(``game.started_at - benchmark.start_time + action.wallclock_seconds``),
the same axis taaf's diagnostics plots use. Compare runs on this clock, not
on any harness-internal runtime clock.

Score: the official ARC-AGI-3 formula (mirrors ``taaf.game.GameRun._compute_final_score``):
level score = min(115, (human_actions / agent_actions)^2 * 100) for a completed level,
game score = sum(level_score_i * i) / sum(i) over levels i = 1..n, capped at
(sum of i over scored levels) / sum(i) * 100. The score "at time t" is what the
game would score if it stopped right then; the run score is the mean over games.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from inference.agent.action_names import ENGINE_TO_MODEL_ACTION


MAX_RUN_POINTS = 1200


def level_score(baseline_actions: float, agent_actions: int) -> float:
    if agent_actions <= 0 or baseline_actions <= 0:
        return 0.0
    return min(115.0, (baseline_actions / agent_actions) ** 2 * 100.0)


def game_score(
    base_actions_per_level: list[float] | None,
    actions_per_level: list[int],
    levels_completed: int,
    number_of_levels: int | None = None,
) -> float:
    """Official game score for ``levels_completed`` levels taking ``actions_per_level`` actions."""
    if not base_actions_per_level:
        return 0.0
    n_levels = int(number_of_levels or len(base_actions_per_level))
    total_score = 0.0
    total_weights = 0
    max_weights = 0
    for index in range(n_levels):
        weight = index + 1
        total_weights += weight
        actions = actions_per_level[index] if index < len(actions_per_level) else 0
        baseline = base_actions_per_level[index] if index < len(base_actions_per_level) else 0
        score = level_score(baseline, actions) if index < levels_completed else 0.0
        if score > 0:
            max_weights += weight
        total_score += score * weight
    if total_weights == 0:
        return 0.0
    return min(total_score / total_weights, max_weights / total_weights * 100.0)


def parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def format_action(action: dict[str, Any] | None) -> str:
    """Model-facing label of a benchmark history action (``RIGHT``, ``MOUSE(row=3, col=7)``)."""
    action = action or {}
    name = str(action.get("id") or "?").strip().upper()
    if name == "ACTION6":
        data = action.get("data") or {}
        if "x" in data and "y" in data:
            return f"MOUSE(row={data.get('y')}, col={data.get('x')})"
    return ENGINE_TO_MODEL_ACTION.get(name, name)


def game_series(game_run: dict[str, Any], run_start: datetime | None) -> dict[str, Any]:
    """Per-action series for one benchmark game run.

    Index 0 is the start of the game (no actions); index n is the state right
    after action n. ``level`` is the level being played after the action (1-based),
    ``levels`` the levels completed so far.
    """
    history = [entry for entry in game_run.get("history") or [] if isinstance(entry, dict)]
    base = [float(value) for value in game_run.get("base_actions_per_level") or []]
    n_levels = int(game_run.get("number_of_levels") or len(base) or 0)
    final_levels = int(game_run.get("levels_completed") or 0)
    per_level_final = [int(value) for value in game_run.get("actions_per_level") or []]

    # Level boundaries: level i (0-based) took per_level_final[i] actions, so
    # it was completed by action cum[i] (1-based) when i < levels_completed.
    completion_action: list[int] = []
    running = 0
    for index in range(final_levels):
        running += per_level_final[index] if index < len(per_level_final) else 0
        completion_action.append(running)

    started = parse_time(game_run.get("started_at"))
    offset = (started - run_start).total_seconds() if started and run_start else 0.0

    times = [round(offset, 2)]
    scores = [0.0]
    levels = [0]
    level_now = [1 if n_levels else 0]
    tokens = [0]
    labels = ["START"]
    cumulative_tokens = 0
    done = 0
    for number, entry in enumerate(history, start=1):
        cumulative_tokens += int(entry.get("generated_tokens") or 0)
        while done < len(completion_action) and completion_action[done] <= number:
            done += 1
        times.append(round(offset + float(entry.get("wallclock_seconds") or 0.0), 2))
        tokens.append(cumulative_tokens)
        levels.append(done)
        level_now.append(min(done + 1, n_levels) if n_levels else done + 1)
        scores.append(round(game_score(base, per_level_final, done, n_levels), 3))
        labels.append(format_action(entry.get("action")))

    final_seconds = max(
        float(history[-1].get("wallclock_seconds") or 0.0) if history else 0.0,
        float(game_run.get("final_wallclock_seconds") or 0.0),
    )
    end_time = round(offset + final_seconds, 2)
    end_tokens = cumulative_tokens + int(game_run.get("final_generated_tokens") or 0)
    final_score = game_run.get("final_score")
    return {
        "game_id": str(game_run.get("game_id") or ""),
        "number_of_levels": n_levels,
        "levels_completed": final_levels,
        "state": str(game_run.get("state") or ""),
        "final_score": round(float(final_score), 3) if final_score is not None else scores[-1],
        "offset": round(offset, 2),
        "end": end_time,
        "end_tokens": end_tokens,
        "t": times,
        "score": scores,
        "levels": levels,
        "level": level_now,
        "tokens": tokens,
        "action": labels,
    }


def pass_keys(game_runs: list[dict[str, Any]]) -> list[str]:
    """``<game_id>_p<pass>`` for each game run; the pass is the game's occurrence index."""
    seen: dict[str, int] = {}
    keys = []
    for run in game_runs:
        game_id = str(run.get("game_id") or "")
        pass_index = seen.get(game_id, 0)
        seen[game_id] = pass_index + 1
        keys.append(f"{game_id}_p{pass_index}")
    return keys


def run_timeline(benchmark: dict[str, Any]) -> dict[str, Any]:
    """Run-level series (mean score, total levels, total tokens) plus per-game series."""
    game_runs = [run for run in benchmark.get("game_runs") or [] if isinstance(run, dict)]
    run_start = parse_time(benchmark.get("start_time"))
    run_end = parse_time(benchmark.get("end_time"))
    keys = pass_keys(game_runs)
    games = {key: game_series(run, run_start) for key, run in zip(keys, game_runs)}

    # Merge every game's change points onto one clock. Each game contributes
    # deltas (score, levels, tokens) at its action times; the run series is the
    # running sum, with score averaged over all games in the run.
    deltas: list[tuple[float, float, int, int]] = []
    for series in games.values():
        previous_score, previous_levels, previous_tokens = 0.0, 0, 0
        for index in range(1, len(series["t"])):
            score, levels, tokens = series["score"][index], series["levels"][index], series["tokens"][index]
            deltas.append(
                (series["t"][index], score - previous_score, levels - previous_levels, tokens - previous_tokens)
            )
            previous_score, previous_levels, previous_tokens = score, levels, tokens
        extra_tokens = series["end_tokens"] - previous_tokens
        if extra_tokens:
            deltas.append((series["end"], 0.0, 0, extra_tokens))
    deltas.sort(key=lambda item: item[0])

    count = max(1, len(games))
    run_t, run_score, run_levels, run_tokens = [0.0], [0.0], [0], [0]
    score_sum, level_sum, token_sum = 0.0, 0, 0
    for time, d_score, d_levels, d_tokens in deltas:
        score_sum += d_score
        level_sum += d_levels
        token_sum += d_tokens
        if run_t and abs(run_t[-1] - time) < 1e-9:
            run_score[-1] = round(score_sum / count, 3)
            run_levels[-1] = level_sum
            run_tokens[-1] = token_sum
            continue
        run_t.append(round(time, 2))
        run_score.append(round(score_sum / count, 3))
        run_levels.append(level_sum)
        run_tokens.append(token_sum)

    duration = (run_end - run_start).total_seconds() if run_end and run_start else None
    run_series = _downsample(
        {"t": run_t, "score": run_score, "levels": run_levels, "tokens": run_tokens},
        MAX_RUN_POINTS,
    )
    return {
        "start_time": benchmark.get("start_time"),
        "end_time": benchmark.get("end_time"),
        "duration": round(duration, 2) if duration is not None else None,
        "game_count": len(games),
        "total_levels": sum(series["number_of_levels"] for series in games.values()),
        "final_mean_score": round(
            sum(series["final_score"] for series in games.values()) / count, 3
        ) if games else 0.0,
        "run": run_series,
        "games": games,
    }


def _downsample(series: dict[str, list[Any]], max_points: int) -> dict[str, list[Any]]:
    """Thin a step series to ~max_points, always keeping points where score or levels change."""
    length = len(series["t"])
    if length <= max_points:
        return series
    stride = max(1, length // max_points)
    keep = []
    for index in range(length):
        changed = index == 0 or index == length - 1 or (
            series["score"][index] != series["score"][index - 1]
            or series["levels"][index] != series["levels"][index - 1]
        )
        if changed or index % stride == 0:
            keep.append(index)
    return {key: [values[index] for index in keep] for key, values in series.items()}
