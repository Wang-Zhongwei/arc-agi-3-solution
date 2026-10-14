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


# ARC3_ANIMATION_FILTER=motion

from inference.utils.animation import plain_motion


def _board(size=16, sprites=(), flash=()):
    grid = [[0] * size for _ in range(size)]
    for row, col, color in sprites:
        for dr in range(2):
            for dc in range(2):
                grid[row + dr][col + dc] = color
    for row, col in flash:
        grid[row][col] = 9
    return tuple(tuple(line) for line in grid)


def _slide(start=2, end=6, row=7, extra=lambda col: {}):
    return [_board(sprites=[(row, col, 3)], **extra(col)) for col in range(start, end + 1)]


def test_a_sprite_sliding_to_its_end_is_plain_motion():
    assert plain_motion(_slide(), border=0)


def test_a_flash_away_from_what_moved_is_reported():
    chain = _slide(extra=lambda col: {"flash": [(1, 14)]} if col == 4 else {})
    assert not plain_motion(chain, border=0)


def test_a_flash_with_nothing_moved_is_reported():
    still = _board(sprites=[(7, 2, 3)])
    assert not plain_motion([still, _board(sprites=[(7, 2, 3)], flash=[(3, 3)]), still], border=0)


def test_a_full_board_flash_is_reported():
    full = tuple(tuple(9 for _ in range(16)) for _ in range(16))
    chain = _slide()
    assert not plain_motion([chain[0], full, chain[-1]], border=0, max_cells=100)


def test_two_sprites_moving_apart_are_reported():
    chain = [_board(sprites=[(7, 6 - k, 3), (7, 8 + k, 4)]) for k in range(4)]
    assert not plain_motion(chain, border=0)


def test_solver_tags_plain_motion_only_when_the_filter_is_on(session, monkeypatch):
    slide = _slide(start=4, end=8)

    class _SlidingGame(_Game):
        def __init__(self):
            super().__init__()
            self.current_state = self._state([slide[0]])

        def execute_action(self, action, **_):
            frames = slide[1:] if action.id == arcengine.GameAction.ACTION4 else [slide[0]]
            self.current_state = self._state(frames)
            return self.current_state

    monkeypatch.setattr(solver_module, "NOOP_GUARD_BORDER_WIDTH", 0)
    for setting, tagged in (("off", None), ("motion", True)):
        monkeypatch.setattr(solver_module, "ANIMATION_FILTER", setting)
        session.game = _SlidingGame()
        payload, = _run(session, arcengine.GameAction.ACTION4)
        # cols 6-7 are empty, then sprite, then empty: transient, so animated
        assert payload["animation"]["transient_pixels"] > 0
        assert payload["animation"].get("plain_motion") is tagged


def test_opener_leaves_out_plain_motion(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC3_ANIMATION", "1")
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", "off")
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    agent._ensure_session(tmp_path / "state.json")
    current = Frame(grid=tuple((0,) * 64 for _ in range(64)), step=9, level=2)
    entries = [HistoryEntry("INIT", Frame(grid=current.grid, step=6, level=2)), HistoryEntry("RIGHT", current)]
    agent._auto_diff_prev_frame = entries[0].frame

    def opener(summary):
        return agent._build_user_prompt(current.step, valid_actions=["RIGHT"], current_frame=current,
                                        history_entries=entries, previous_step_summary=summary)

    moved = {**_entry(1, "RIGHT"), "plain_motion": True}
    batch = {"executed_count": 3, "executed_actions": ["RIGHT", "MOUSE(row=5, col=5)", "RIGHT"],
             "animated_actions": [moved, _entry(2, "MOUSE(row=5, col=5)")]}
    text = opener(batch)
    assert "Action 2 of 3 (`MOUSE(row=5, col=5)`) animated" in text
    assert "Action 1 of 3" not in text
    single = {"executed_count": 1, "executed_actions": ["RIGHT"],
              "animation": {"frames": 3, "transient_pixels": 4, "transient_bbox": [1, 1, 1, 4],
                            "plain_motion": True}}
    assert "animated over" not in opener(single)
