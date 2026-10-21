"""Per-level attempt dataset from our Kaggle run outputs, for scheduler fitting.

Each row is one level attempt in one game of one run: how many generated
tokens and actions it consumed, whether it was cleared, and the context needed
to value it (level index, level count, human baseline actions). Attempts on
the level a game was still playing when the run ended are right-censored.

Tokens come from the benchmark history, where each action carries the
generated tokens spent since the previous action. Tokens generated after a
game's last action (reasoning that never led to another action) are the gap
between the game's `tokens=` total and the summed history, and are charged to
the level it ended on.

Usage:
    python analysis/level_dataset.py <run_dir> [<run_dir> ...] > levels.csv
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

FIELDS = [
    "run", "game", "level", "levels_total", "tokens", "actions", "cleared",
    "human_actions", "efficiency", "game_won",
]


def level_rows(run_dir: Path):
    """Yield one dict per level attempt in one run's benchmark.json."""
    bench = json.loads((run_dir / "benchmark.json").read_text())
    for game in bench["game_runs"]:
        total = int(re.search(r"tokens=(\d+)", game.get("solver_note") or "tokens=0").group(1))
        history = game["history"]
        per_level = game.get("actions_per_level") or []
        baselines = game.get("base_actions_per_level") or []
        done = int(game["levels_completed"])
        n = int(game["number_of_levels"])
        cursor = 0
        charged = 0
        # attempted levels: every cleared one plus the one in progress (if any)
        attempted = done + (1 if done < n else 0)
        for level in range(1, attempted + 1):
            actions = int(per_level[level - 1]) if level - 1 < len(per_level) else 0
            chunk = history[cursor:cursor + actions]
            tokens = sum(int(step.get("generated_tokens") or 0) for step in chunk)
            cursor += actions
            charged += tokens
            cleared = level <= done
            if not cleared:
                # everything not charged to a cleared level belongs to the unfinished one,
                # including reasoning after the game's last action
                tokens = max(0, total - (charged - tokens))
            human = float(baselines[level - 1]) if level - 1 < len(baselines) else float("nan")
            efficiency = min(1.15, (human / actions) ** 2) if cleared and actions else 0.0
            yield {
                "run": run_dir.name, "game": game["game_id"][:4], "level": level,
                "levels_total": n, "tokens": tokens, "actions": actions,
                "cleared": int(cleared), "human_actions": human,
                "efficiency": round(efficiency, 4), "game_won": int(done == n),
            }


def main() -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    for arg in sys.argv[1:]:
        for row in level_rows(Path(arg)):
            writer.writerow(row)


if __name__ == "__main__":
    main()
