"""An action that only animated inside the board area is not a no-op for the guards."""

import time
from types import SimpleNamespace

import arcengine
import pytest

from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.framework import solver as solver_module
from inference.framework.solver import _HarnessGameSession
from inference.utils.grid_utils import format_grid_ascii

SIZE = 16
BOARD = tuple((0,) * SIZE for _ in range(SIZE))


def _flash(row):
    return tuple(((5,) * SIZE if r == row else (0,) * SIZE) for r in range(SIZE))


class _Game:
    """ACTION1 flashes an interior row, ACTION3 a border row; both settle back."""

    number_of_levels = 2

    def __init__(self):
        self.game_run = SimpleNamespace(history=[])
        self.current_state = self._state([BOARD])

    @staticmethod
    def _state(frames):
        return SimpleNamespace(
            frame=SimpleNamespace(data=frames[-1]),
            raw=SimpleNamespace(state=arcengine.GameState.NOT_FINISHED, frame=list(frames)),
            levels_completed=0, won=False, just_won_level=False, available_actions=[1, 2, 3],
        )

    def execute_action(self, action, **_):
        flashes = {arcengine.GameAction.ACTION1: [_flash(8), BOARD], arcengine.GameAction.ACTION3: [_flash(0), BOARD]}
        self.current_state = self._state(flashes.get(action.id, [BOARD]))
        self.game_run.history.append(action)
        return self.current_state


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(solver_module, "BATCH_NOOP_BLOCK", True)
    monkeypatch.setattr(solver_module, "NOOP_GUARD_BORDER_WIDTH", 4)
    session = object.__new__(_HarnessGameSession)
    session.solver = SimpleNamespace(max_runtime_s_per_game=None)
    session.game = _Game()
    session.analyzer = None
    session.should_stop = lambda: False
    session.started_at = time.monotonic()
    session.history_entries = []
    session.animation_record = None
    session.token_baseline = 0
    session.write_runtime_state = lambda: None
    session._append_action_viewer_event = lambda *_: None
    session.write_viewer_payload = lambda: None
    return session


def test_an_animated_action_does_not_stop_the_batch(session):
    result = session.step_env({"actions": [{"action": "ACTION1"}, {"action": "ACTION2"}, {"action": "ACTION2"}]})
    # the flash runs on; the inert action after it is the real no-op that cuts the batch
    assert result["executed_count"] == 2 and result["stop_reason"] == "no_op_action"
    assert result["gameplay_changed_per_action"] == [False, False]
    assert result["no_op_per_action"] == [False, True]
    assert "1: UP -> effect (animation only; the board area ended unchanged)" in result["action_trace"]


def test_a_flash_only_in_the_border_is_still_a_no_op(session):
    result = session.step_env({"actions": [{"action": "ACTION3"}, {"action": "ACTION2"}]})
    assert result["executed_count"] == 1 and result["no_op_per_action"] == [True]


@pytest.mark.parametrize("no_op, expected_stale", [(False, None), (True, "A")])
def test_only_a_true_no_op_arms_the_stale_state_flag(no_op, expected_stale):
    grid = [[0, 0], [0, 0]]
    frame = {"ascii": format_grid_ascii(grid), "step": 0, "level": 2, "shape": [2, 2], "grid": grid}
    state = {"current_frame": frame, "history": [{"action": "", "frame": frame}], "valid_actions": ["A", "B"]}
    seen = []

    def handler(actions, stale_after=None):
        seen.append(stale_after)
        name = actions[0].get("action") if actions else "?"
        return {"action_result": {"executed": True, "action_display": name, "executed_actions": [name],
                                  "gameplay_changed": False, "no_op": no_op},
                "state": state}

    out = run_sandboxed_python(code="action(['A'])\naction(['B'])\nresult = 1", timeout_seconds=20,
                               initial_state=state, action_handler=handler)
    assert out.get("result") == 1, out
    assert seen == [None, expected_stale]
