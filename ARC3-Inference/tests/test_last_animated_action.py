"""The call's last animated action is reported even when later actions are inert."""

import time
from types import SimpleNamespace

import arcengine
import pytest

from inference.agent import tool_agent
from inference.agent.runtime_state import Frame, HistoryEntry
from inference.framework import solver as solver_module
from inference.framework.solver import _HarnessGameSession

BOARD = tuple((0,) * 8 for _ in range(8))
FLASH = tuple(((5,) * 8 if row == 3 else (0,) * 8) for row in range(8))


class _Game:
    """ACTION1 flashes row 3 and settles back; every other action is inert."""

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
        animated = action.id == arcengine.GameAction.ACTION1
        self.current_state = self._state([FLASH, BOARD] if animated else [BOARD])
        self.game_run.history.append(action)
        return self.current_state


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(solver_module, "ANIMATION_ENABLED", True)
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


def _call(session, *names):
    return session.step_env({"actions": [{"action": name} for name in names]})


def test_an_animated_action_followed_by_an_inert_one_is_reported(session):
    result = _call(session, "ACTION1", "ACTION2")
    assert result["executed_count"] == 2
    assert result["animation"]["action"] == "UP" and result["animation"]["position"] == 1
    assert session.animation_record["action_display"] == "UP"
    # the next call starts clean, so an old animation is never reported as new
    _call(session, "ACTION2")
    assert session.animation_record is None


def test_the_final_animated_action_keeps_its_position(session):
    result = _call(session, "ACTION2", "ACTION1")
    assert result["animation"]["position"] == 2 and result["animation"]["action"] == "UP"


def _opener(monkeypatch, tmp_path, summary):
    monkeypatch.setenv("ARC3_ANIMATION", "1")
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", "off")
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    agent._ensure_session(tmp_path / "state.json")
    current = Frame(grid=tuple((0,) * 64 for _ in range(64)), step=9, level=2)
    entries = [HistoryEntry("INIT", Frame(grid=current.grid, step=7, level=2)), HistoryEntry("DOWN", current)]
    agent._auto_diff_prev_frame = entries[0].frame
    agent._last_action_gameplay_changed = False
    return agent._build_user_prompt(current.step, valid_actions=["UP"], current_frame=current,
                                    history_entries=entries, previous_step_summary=summary)


ANIMATION = {"frames": 2, "transient_pixels": 8, "transient_bbox": [3, 0, 3, 7]}


def test_opener_names_an_earlier_animated_action(monkeypatch, tmp_path):
    last_call = {"executed": True, "executed_count": 2, "action_num": 9,
                 "executed_actions": ["UP", "DOWN"], "gameplay_changed_per_action": [True, False],
                 "animation": {"action": "UP", "position": 1, **ANIMATION}}
    summary = tool_agent.ToolAgent._summarize_step_sequence(object.__new__(tool_agent.ToolAgent), [last_call])
    prompt = _opener(monkeypatch, tmp_path, summary)
    assert "Action 1 of 2 (`UP`) animated over 2 frames. 8 cells changed and changed back" in prompt


def test_opener_keeps_the_original_line_when_the_final_action_animated(monkeypatch, tmp_path):
    last_call = {"executed": True, "action_num": 9, "executed_actions": ["UP"],
                 "animation": {"action": "UP", "position": 1, **ANIMATION}}
    summary = tool_agent.ToolAgent._summarize_step_sequence(object.__new__(tool_agent.ToolAgent), [last_call])
    prompt = _opener(monkeypatch, tmp_path, summary)
    assert "The last executed action (`UP`) animated over 2 frames" in prompt
