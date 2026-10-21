"""Run outputs in the shapes the repo's tooling already reads.

- `benchmark.json` with `game_runs[]` in the taaf shape (scripts/summarize_run.py reads
  game_id, number_of_levels, base_actions_per_level, state, actions_per_level, levels_completed,
  final_score, history[{action,generated_tokens,wallclock_seconds}], solver_note, started_at,
  final_wallclock_seconds, final_generated_tokens; benchmark start_time/end_time/label/n_passes).
- The `[finished] ...` line (taaf/game.py:646 format) and the `LEVEL_START_RESULT ...` line
  (experiments/level_start) that scripts/gb10_level.sh and scripts/level_results.sh grep.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def taaf_score(baselines: list[int] | None, actions_per_level: list[int], levels_completed: int) -> float:
    """Mirror of taaf.game.GameRun._compute_final_score (== arc_agi.scorecard)."""
    if not baselines:
        return 0.0
    n = len(baselines)
    total_score = 0.0
    total_weights = 0
    max_weights = 0
    for i in range(n):
        w = i + 1
        total_weights += w
        completed = i < levels_completed
        actions = actions_per_level[i] if i < len(actions_per_level) else 0
        if completed and actions > 0:
            s = min(115.0, (baselines[i] / actions) ** 2 * 100)
        else:
            s = 0.0
        if s > 0:
            max_weights += w
        total_score += s * w
    if total_weights == 0:
        return 0.0
    return min(total_score / total_weights, max_weights / total_weights * 100)


def game_run_dict(rec: Any, *, started_at: datetime, history: list[dict], generated_tokens: int,
                  solver_note: str = "", cancelled: bool = False) -> dict[str, Any]:
    """Convert an EnvRunRecord (vendor/tycho/harness.py) into a taaf-shaped game run."""
    n = int(rec.n_levels)
    actions_per_level = [0] * n
    for lr in rec.level_results:
        idx = int(lr["level_index"]) - 1
        if 0 <= idx < n:
            actions_per_level[idx] += int(lr["actions_taken"])
    if rec.unfinished_level_index is not None:
        idx = int(rec.unfinished_level_index) - 1
        if 0 <= idx < n:
            actions_per_level[idx] += int(rec.unfinished_level_actions)
    levels_completed = int(rec.levels_completed)
    baselines = list(rec.baselines) if rec.baselines and any(rec.baselines) else None
    if rec.final_state == "GameState.WIN":
        state = "won"
    elif rec.error:
        state = "crashed"
    elif cancelled or rec.stop_reason == "deadline":
        state = "cancelled"
    else:
        state = "gave_up"
    return {
        "game_id": rec.game_id,
        "number_of_levels": n,
        "base_actions_per_level": baselines,
        "hint": None,
        "state": state,
        "history": history,
        "record_intermediate_states": False,
        "actions_per_level": actions_per_level,
        "levels_completed": levels_completed,
        "final_score": taaf_score(baselines, actions_per_level, levels_completed),
        "solver_note": solver_note or (f"error: {rec.error}" if rec.error else f"tokens={generated_tokens} stop={rec.stop_reason}"),
        "solver_analysis_html": None,
        "final_generated_tokens": 0,
        "final_uncached_input_tokens": 0,
        "final_wallclock_seconds": float(rec.wall_clock_s),
        "started_at": started_at.replace(tzinfo=None).isoformat(),
        # v9 extras (ignored by summarize_run.py)
        "v9": {"start_level": rec.start_level, "stop_reason": rec.stop_reason, "resets": rec.resets,
               "noop_actions": rec.noop_actions, "revisits": rec.revisits, "env_score_rhae": rec.env_score,
               "truncated_levels": rec.truncated_levels, "error": rec.error},
    }


def finished_line(run: dict[str, Any]) -> str:
    """Exactly the taaf per-game finish line, so existing greps and summaries keep working."""
    actions = sum(run["actions_per_level"])
    tokens = sum(int(h.get("generated_tokens") or 0) for h in run["history"]) + int(run.get("final_generated_tokens") or 0)
    bl = run.get("base_actions_per_level")
    per_level = ",".join(
        f"{run['actions_per_level'][i]}/{bl[i] if bl else '?'}" for i in range(run["number_of_levels"])
    )
    note = f' note="{run["solver_note"]}"' if run.get("solver_note") else ""
    return (f"[finished] {run['game_id']} state={run['state']} "
            f"level={run['levels_completed']}/{run['number_of_levels']} "
            f"score={run['final_score']:.2f} actions={actions} tokens={tokens} per-level={per_level}{note}")


def level_start_line(run: dict[str, Any]) -> str:
    start = int(run.get("v9", {}).get("start_level", 1))
    final = int(run["levels_completed"])
    return (f"LEVEL_START_RESULT game={run['game_id']} start_level={start} "
            f"levels_completed={final} gained={final - (start - 1)} actions={sum(run['actions_per_level'])}")


def write_benchmark(out_dir: Path, *, label: str, runs: list[dict[str, Any]], start_time: datetime,
                    end_time: datetime, n_passes: int, extra: dict[str, Any] | None = None) -> Path:
    payload = {
        "label": label,
        "start_time": start_time.replace(tzinfo=None).isoformat(),
        "end_time": end_time.replace(tzinfo=None).isoformat(),
        "n_passes": n_passes,
        "game_runs": runs,
        "v9": extra or {},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "benchmark.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)
    return path


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
