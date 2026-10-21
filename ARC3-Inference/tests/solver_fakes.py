"""A scripted stand-in for taaf's Game, enough to drive step_env."""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import arcengine

from inference.framework import solver as solver_module

Grid = tuple[tuple[int, ...], ...]
SIZE = 16


def blank(fill: int = 0) -> list[list[int]]:
    return [[fill] * SIZE for _ in range(SIZE)]


def frozen(grid: list[list[int]]) -> Grid:
    return tuple(tuple(row) for row in grid)


class _State:
    def __init__(self, frames: list[Grid], *, levels_completed: int = 0,
                 raw_state=arcengine.GameState.NOT_FINISHED) -> None:
        self.frame = SimpleNamespace(data=frames[-1])
        self.raw = SimpleNamespace(state=raw_state, frame=list(frames))
        self.levels_completed = levels_completed
        self.won = raw_state == arcengine.GameState.WIN
        self.just_won_level = False
        self.available_actions = [1, 2, 3, 4, 5, 6]


class FakeGame:
    """``script(action_name, data, board) -> (frames, flags)`` drives each step."""

    number_of_levels = 3

    def __init__(self, start: Grid, script: Callable) -> None:
        self.game_run = SimpleNamespace(state="playing", history=[])
        self.current_state = _State([start])
        self._script = script

    def execute_action(self, action, generated_tokens=0, uncached_input_tokens=0):
        board = self.current_state.frame.data
        frames, flags = self._script(action.id.name, dict(action.data), board)
        self.game_run.history.append(action)
        raw_state = arcengine.GameState.GAME_OVER if flags.get("game_over") else (
            arcengine.GameState.NOT_FINISHED
        )
        completed = self.current_state.levels_completed + (1 if flags.get("level_up") else 0)
        self.current_state = _State(frames, levels_completed=completed, raw_state=raw_state)
        self.current_state.just_won_level = bool(flags.get("level_up"))
        return self.current_state


def make_session(tmp_path: Path, game: FakeGame):
    solver = SimpleNamespace(
        max_runtime_s_per_game=None,
        max_actions_per_game=None,
        max_generated_tokens_per_game=None,
        job_dir=None,
    )
    return solver_module._HarnessGameSession(
        solver=solver,
        game=game,
        analyzer=None,
        game_index=0,
        pass_index=0,
        state_path=tmp_path / "state.json",
        transcript_path=tmp_path / "transcript.txt",
        analysis_html_relpath="",
        stop_event=threading.Event(),
        viewer_data_path=tmp_path / "viewer.json",
    )


def flash_script(action_name: str, data: dict, board: Grid):
    """ACTION5 (SPACE) flashes a 2x2 block and reverts: a transient no-op.
    ACTION6 (MOUSE) flashes at the clicked cell. ACTION4 (RIGHT) moves a dot
    one cell with no animation. ACTION1 (UP) changes nothing at all. ACTION3
    (LEFT) completes the level; ACTION2 (DOWN) flashes a 3x1 bar and ends the
    attempt."""
    if action_name == "ACTION3":
        return [board], {"level_up": True}
    if action_name == "ACTION2":
        flash = [list(line) for line in board]
        for row in (6, 7, 8):
            flash[row][10] = 12
        dead = [list(line) for line in board]
        dead[9][9] = 12
        return [frozen(flash), frozen(dead)], {"game_over": True}
    if action_name == "ACTION5":
        flash = [list(line) for line in board]
        for row in (7, 8):
            for col in (7, 8):
                flash[row][col] = 8
        return [frozen(flash), board], {}
    if action_name == "ACTION6" and int(data.get("y", 0)) == 12:
        # a sweep: a 1x2 highlight steps along row 12, then everything reverts
        frames = []
        for left in (4, 6, 8, 10, 12):
            sweep = [list(line) for line in board]
            sweep[12][left] = sweep[12][left + 1] = 14
            frames.append(frozen(sweep))
        return [*frames, board], {}
    if action_name == "ACTION6":
        row, col = int(data.get("y", 0)), int(data.get("x", 0))
        flash = [list(line) for line in board]
        flash[row][col] = 8
        return [frozen(flash), board], {}
    if action_name == "ACTION4":
        moved = [list(line) for line in board]
        for row in range(SIZE):
            for col in range(SIZE - 1, 0, -1):
                if moved[row][col - 1] == 9 and col < SIZE - 4:
                    moved[row][col], moved[row][col - 1] = 9, 0
        return [frozen(moved)], {}
    return [board], {}
