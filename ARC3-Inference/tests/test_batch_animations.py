"""Animations of any action in a batch, not only the final one."""

import time
from types import SimpleNamespace

import arcengine
import pytest

from inference.agent import tool_agent
from inference.agent.runtime_state import Frame, HistoryEntry
from inference.framework import solver as solver_module
from inference.framework.solver import _HarnessGameSession
from inference.utils.animation import describe_animations

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
            levels_completed=0, won=False, just_won_level=False,
            available_actions=[1, 2, 3],
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


def _run(session, *ids):
    return [
        session._execute_action(
            arcengine.ActionInput(id=action_id), batch_index=index,
            batch_size=len(ids), generated_tokens=0, flush_viewer_payload=False,
        )
        for index, action_id in enumerate(ids, start=1)
    ]


def test_record_keeps_an_animated_action_followed_by_inert_ones(session):
    A = arcengine.GameAction
    payloads = _run(session, A.ACTION1, A.ACTION2)
    assert payloads[0].get("animation") and not payloads[1].get("animation")
    assert session.animation_record["action_display"] == payloads[0]["action_display"]
    # the next call starts clean, so an old animation is never reported as new
    _run(session, A.ACTION2)
    assert session.animation_record is None


def test_batch_result_lists_every_animated_action(session):
    result = session.step_env({"actions": [{"action": "ACTION2"}, {"action": "ACTION1"},
                                           {"action": "ACTION2"}]})
    assert result["executed_count"] == 3 and result.get("animation") is None
    assert [(item["position"], item["action"]) for item in result["animated_actions"]] == [(2, "UP")]


def _entry(position, action, transient=12, gameplay_changed=True):
    return {"position": position, "action": action, "gameplay_changed": gameplay_changed,
            "frames": 2, "transient_pixels": transient, "transient_bbox": [3, 0, 3, 7]}


def test_one_animated_action_is_named_by_position():
    line = describe_animations([_entry(2, "MOUSE(row=58, col=25)")], executed_count=5)
    assert line.startswith("Action 2 of 5 (`MOUSE(row=58, col=25)`) animated over 2 frames.")
    assert "`last_animation_timeline` shows them." in line


def test_the_final_action_keeps_the_original_wording():
    line = describe_animations([_entry(3, "UP")], executed_count=3)
    assert line.startswith("The last executed action (`UP`) animated over 2 frames.")


def test_several_animated_actions_say_only_the_last_is_kept():
    line = describe_animations([_entry(2, "A"), _entry(4, "B", transient=1)], executed_count=5)
    assert "Action 2 of 5 (`A`) animated over 2 frames: 12 cells" in line
    assert "Action 4 of 5 (`B`) animated over 2 frames: 1 cell changed and changed back" in line
    assert line.endswith("`last_animation_timeline` shows action 4 only; the earlier animations are not kept.")


def test_frames_accessor_when_timeline_is_off():
    line = describe_animations([_entry(1, "A"), _entry(2, "B")], executed_count=2, timeline=False)
    assert "`last_animation_frames` shows action 2 only" in line


def test_opener_names_the_animated_middle_action(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC3_ANIMATION", "1")
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", "off")
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    agent._ensure_session(tmp_path / "state.json")
    current = Frame(grid=tuple((0,) * 64 for _ in range(64)), step=9, level=2)
    entries = [HistoryEntry("INIT", Frame(grid=current.grid, step=4, level=2)), HistoryEntry("RIGHT", current)]
    last_call = {
        "executed": True, "executed_count": 3, "action_num": 9,
        "executed_actions": ["RESET", "MOUSE(row=58, col=25)", "RIGHT"],
        "animated_actions": [_entry(2, "MOUSE(row=58, col=25)")],
    }
    earlier_call = {"executed": True, "executed_count": 2, "action_num": 6,
                    "executed_actions": ["UP", "UP"]}
    summary = agent._summarize_step_sequence([earlier_call, last_call])
    assert [item["position"] for item in summary["animated_actions"]] == [4]
    agent._auto_diff_prev_frame = entries[0].frame
    prompt = agent._build_user_prompt(current.step, valid_actions=["RIGHT"], current_frame=current,
                                      history_entries=entries, previous_step_summary=summary)
    assert "Action 4 of 5 (`MOUSE(row=58, col=25)`) animated over 2 frames." in prompt


def test_executed_actions_group_repeats_and_are_never_cut():
    names = ["RIGHT"] * 4 + ["UP"] + ["MOUSE(row=1, col=2)"] * 2 + ["RIGHT"] + [f"A{i}" for i in range(12)]
    text = tool_agent._grouped_actions(names)
    assert text.startswith("RIGHTx4, UP, MOUSE(row=1, col=2)x2, RIGHT, A0, ")
    assert text.endswith("A11")
