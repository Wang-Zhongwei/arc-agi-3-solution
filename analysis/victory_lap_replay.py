"""Replay saved wins through the victory-lap code, offline.

For every won game of each run's ``benchmark.json``: build the game from the
competition's environment files, replay the first play's recorded actions
(RESETs included, ONLY_RESET_LEVELS=true as in our runs) while recording each
transition exactly as the harness does (``inference.framework.victory_lap``),
then plan the second play and run it on the same engine with every step
checked. The defaults are the harness's (``--mode last --border 0``); ``--mode
bfs`` and ``--border 4`` are the alternatives research/reports/probe-lap-design.md
compares. Prints one JSON line per game and a per-run summary:
play-1 score (the framework's ``final_score``), the lap's play-2 score, the
official game score max(play 1, play 2), the per-level route lengths and the
lap's status or abort reason.

    nice -n 15 ARC3-Inference/.venv/bin/python analysis/victory_lap_replay.py \
        --env-files /path/to/environment_files \
        runs/kaggle/v4-user-combo-10x/v1/benchmark.json.xz ...

Needs arcengine (the ARC3-Inference venv) and the environment files, which are
not in this repo (``ARC3_ENV_FILES`` or ``--env-files``).
"""
from __future__ import annotations

import argparse
import collections
import json
import lzma
import os
import sys
from pathlib import Path

os.environ["ONLY_RESET_LEVELS"] = "true"  # as every run sets it (taaf game_api.py)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ARC3-Inference"))

from arcengine import ARCBaseGame, ActionInput, GameAction, GameState  # noqa: E402

from inference.framework import victory_lap as vl  # noqa: E402

DEFAULT_ENV_FILES = "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-3/data/environment_files"


def load_game(env_files: Path, game_id: str):
    name, version = game_id.split("-", 1)
    folder = env_files / name / version
    source = (folder / f"{name}.py").read_text()
    module = type(sys)(f"env_{name}")
    exec(source, module.__dict__)  # noqa: S102 - the competition's own environment files
    cls = next(v for v in module.__dict__.values()
               if isinstance(v, type) and issubclass(v, ARCBaseGame) and v is not ARCBaseGame)
    try:
        return cls(seed=0)
    except TypeError:
        return cls()


def _act(game, name: str, data: dict):
    return game.perform_action(ActionInput(id=GameAction[name], data=data or {}), raw=True)


def replay_play1(game, history: list[dict], border: int = 0):
    """Play 1 as recorded; returns (transitions, start_hash, final_state)."""
    frame = _act(game, "RESET", {})
    current = vl.state_hash(frame.frame[-1], border)
    start = current
    levels = int(frame.levels_completed)
    transitions = []
    for record in history:
        name, data = record["id"], dict(record.get("data") or {})
        frame = _act(game, name, data)
        if not frame.frame:
            raise RuntimeError(f"empty frame replaying {name}")
        nxt = vl.state_hash(frame.frame[-1], border)
        transitions.append(vl.Transition.make(
            prev=current, action=name, data=data, next=nxt, levels_before=levels,
            levels_after=int(frame.levels_completed), state=frame.state.name))
        current, levels = nxt, int(frame.levels_completed)
    return transitions, start, frame.state


def lap_on_engine(game, plan: vl.RoutePlan, border: int = 0) -> vl.LapResult:
    def step(name: str, data: dict):
        frame = _act(game, name, data)
        if not frame.frame:
            raise RuntimeError(f"empty frame for {name}")
        return vl.state_hash(frame.frame[-1], border), int(frame.levels_completed), frame.state.name

    return vl.run_lap(plan, step, should_stop=lambda: False)


def analyse_game(env_files: Path, run: dict, *, border: int = 0, mode: str = "bfs") -> dict:
    game = load_game(env_files, run["game_id"])
    history = [h["action"] for h in run["history"]]
    transitions, start, final = replay_play1(game, history, border)
    out = {"game": run["game_id"][:4], "levels": run["number_of_levels"]}
    if final != GameState.WIN:
        out.update(status="replay-diverged", reason=f"play 1 replay ended {final.name}")
        return out
    base = run["base_actions_per_level"]
    s1 = vl.game_score(base, run["actions_per_level"], run["levels_completed"])
    plan = vl.plan_routes(transitions, start_hash=start, number_of_levels=run["number_of_levels"], mode=mode)
    out.update(s1=round(s1, 2), apl1=run["actions_per_level"], base=base)
    if plan.skip_reason:
        out.update(status="skipped", reason=plan.skip_reason, official=round(s1, 2))
        return out
    out["route"] = plan.lengths
    expected = vl.game_score(base, plan.lengths, len(plan.lengths))
    out["expected_s2"] = round(expected, 2)
    result = lap_on_engine(game, plan, border)
    s2 = vl.game_score(base, result.actions_per_level, result.levels_completed)
    out.update(status=result.status, reason=result.reason, s2=round(s2, 2),
               official=round(max(s1, s2), 2), gain=round(max(s1, s2) - s1, 2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmarks", nargs="+")
    parser.add_argument("--env-files", default=os.environ.get("ARC3_ENV_FILES", DEFAULT_ENV_FILES))
    parser.add_argument("--border", type=int, default=0,
                        help="HUD frame width left out of the state hash; 0 (the harness) hashes every cell")
    parser.add_argument("--mode", choices=("last", "bfs"), default="last",
                        help="last: each level's last attempt verbatim (the harness); bfs: shortest observed route")
    args = parser.parse_args()
    env_files = Path(args.env_files)
    for path in args.benchmarks:
        opener = lzma.open if path.endswith(".xz") else open
        with opener(path, "rt") as handle:
            data = json.load(handle)
        runs = data["game_runs"]
        tally = collections.Counter()
        gain = 0.0
        for run in runs:
            if run["levels_completed"] != run["number_of_levels"]:
                continue
            try:
                row = analyse_game(env_files, run, border=args.border, mode=args.mode)
            except Exception as exc:  # noqa: BLE001
                row = {"game": run["game_id"][:4], "status": "error", "reason": repr(exc)[:200]}
            row["run"] = path
            tally[row["status"]] += 1
            gain += float(row.get("gain") or 0.0)
            print(json.dumps(row), flush=True)
        print(json.dumps({"run": path, "summary": dict(tally), "points_added_per_game_mean": round(gain / len(runs), 3)}))


if __name__ == "__main__":
    main()
