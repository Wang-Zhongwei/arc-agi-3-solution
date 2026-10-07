"""Extract per-game score trajectories from evaluated runs.

Reads `<runs-root>/kaggle/<run>/evaluation.json` (progression lanes). Writes:

  data/trajectories.json  {run: {game: {n, won, tokens: [[x, score]...], actions: [[x, score]...]}}}

Usage (from the repository root of the main checkout, where runs/ exists):
  python docs/allocation/scripts/extract_trajectories.py --runs-root ARC3-Inference/runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = [
    "jonathanwang2022/bp35-lordhansolo/v02",
    "jonathanwang2022/bp35-lordhansolo/v03",
    "jonathanwang2022/bp35-lordhansolo/v07",
    "jonathanwang2022/bp35-lordhansolo/v08",
    "jonathanwang2022/bp35-lordhansolo/v09",
    "jonathanwang2022/persistent-animations/v01",
    "ticbudja/thinking-budget-and-user-prompt-trim/v01",
    "ticbudja/thinking-budget-and-user-prompt-trim/v02",
    "ticbudja/thinking-budget-and-user-prompt-trim/v03",
    "ticbudja/yield-ucb-scheduling-1001/v01",
    "dangthuygiang/arc3-v207-danibase-base-dfranzen/v05",
]

DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=Path("ARC3-Inference/runs"))
    args = ap.parse_args()

    trajectories: dict = {}
    for run in RUNS:
        folder = args.runs_root / "kaggle" / run
        ev = json.loads((folder / "evaluation.json").read_text())
        tok, act = ev["progression_lanes"]["tokens"], ev["progression_lanes"]["actions"]
        games = {}
        for game, trials in tok.items():
            meta = ev["games"][game]
            for trial, points in trials.items():
                if [round(y, 9) for _, y in points] != [round(y, 9) for _, y in act[game][trial]]:
                    raise SystemExit(f"{run} {game}: token and action knots do not align")
                games[game if len(trials) == 1 else f"{game}#{trial}"] = {
                    "n": int(meta["total_levels"]),
                    "won": meta["states"].get(trial) == "won",
                    "tokens": points,
                    "actions": act[game][trial],
                }
        trajectories[run] = games

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "trajectories.json").write_text(json.dumps(trajectories, separators=(",", ":")))
    n = sum(len(g) for g in trajectories.values())
    print(f"{len(trajectories)} runs, {n} trajectories -> {DATA}")


if __name__ == "__main__":
    main()
