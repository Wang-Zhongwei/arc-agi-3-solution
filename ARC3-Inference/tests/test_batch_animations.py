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
    session.animation_records = []
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
    assert [(r["position"], r["action_display"]) for r in session.animation_records] == [
        (1, payloads[0]["action_display"])]
    # the next call starts clean, so an old animation is never reported as new
    _run(session, A.ACTION2)
    assert session.animation_records == []


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
    assert "`call_animations` shows them." in line


def test_the_final_action_keeps_the_original_wording():
    line = describe_animations([_entry(3, "UP")], executed_count=3)
    assert line.startswith("The last executed action (`UP`) animated over 2 frames.")


def test_earlier_animations_get_one_clause_and_the_last_the_full_sentence():
    entries = [_entry(2, "A"), _entry(3, "B"), _entry(5, "C", transient=1)]
    for item, frames in zip(entries, (3, 5, 5)):
        item["frames"] = frames
    line = describe_animations(entries, executed_count=8)
    assert line == (
        "Actions 2, 3 animated over 3, 5 frames. Action 5 of 8 (`C`) animated over 5 frames. "
        "1 cell changed and changed back in row 3, cols 0-7. `call_animations` shows them."
    )


def test_one_earlier_animation_and_the_final_action():
    line = describe_animations([_entry(1, "A"), _entry(2, "B")], executed_count=2)
    assert line.startswith("Action 1 animated over 2 frames. The last executed action (`B`) animated over 2 frames.")
    assert line.endswith("`call_animations` shows them.")


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


def _slide(start=2, end=6, row=7, extra=lambda col: {}, step=1):
    return [_board(sprites=[(row, col, 3)], **extra(col)) for col in range(start, end + 1, step)]


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


@pytest.mark.parametrize("step, frames, dropped", [(2, 2, True), (1, 4, False)])
def test_two_frame_slides_count_as_no_animation_when_the_filter_is_on(session, monkeypatch, step, frames, dropped):
    # a 2x2 sprite jumping 2 columns per frame: 4 -> 6 -> 8 is one hidden frame
    slide = _slide(start=4, end=8, step=step)

    class _SlidingGame(_Game):
        def __init__(self):
            super().__init__()
            self.current_state = self._state([slide[0]])

        def execute_action(self, action, **_):
            frames = slide[1:] if action.id == arcengine.GameAction.ACTION4 else [slide[0]]
            self.current_state = self._state(frames)
            return self.current_state

    monkeypatch.setattr(solver_module, "NOOP_GUARD_BORDER_WIDTH", 0)
    for setting in ("off", "motion"):
        monkeypatch.setattr(solver_module, "ANIMATION_FILTER", setting)
        session.game = _SlidingGame()
        payload, = _run(session, arcengine.GameAction.ACTION4)
        if setting == "motion" and dropped:
            assert "animation" not in payload and session.animation_records == []
        else:
            # cols 6-7 are empty, then sprite, then empty: transient, so animated
            assert payload["animation"]["frames"] == frames
            assert session.animation_records[-1]["summary"] is payload["animation"]


def test_a_dropped_slide_keeps_the_earlier_animation_in_the_record(session, monkeypatch):
    slide = _slide(start=4, end=8, step=2)

    class _Mixed(_Game):
        def execute_action(self, action, **_):
            if action.id == arcengine.GameAction.ACTION4:
                frames = slide[1:]
            else:  # ACTION1: a flash that settles back on the starting board
                frames = [_board(sprites=[(7, 4, 3)], flash=[(1, 1)]), slide[0]]
            self.current_state = self._state(frames)
            return self.current_state

    monkeypatch.setattr(solver_module, "NOOP_GUARD_BORDER_WIDTH", 0)
    monkeypatch.setattr(solver_module, "ANIMATION_FILTER", "motion")
    session.game = _Mixed()
    session.game.current_state = session.game._state([slide[0]])
    A = arcengine.GameAction
    flash, moved = _run(session, A.ACTION1, A.ACTION4)
    assert flash.get("animation") and "animation" not in moved
    assert [r["action_display"] for r in session.animation_records] == [flash["action_display"]]


def test_sandbox_exposes_every_animation_of_the_call():
    from inference.agent.python_tool_sandbox import run_sandboxed_python
    from inference.utils.animation import build_animation_frames, build_animation_view
    from inference.utils.grid_utils import format_grid_ascii

    records = [
        {"position": 2, "action_display": "UP", "chain": [BOARD, FLASH, BOARD], "step": 5, "level": 1},
        {"position": 4, "action_display": "LEFT", "chain": [BOARD, FLASH, FLASH[::-1], BOARD], "step": 7, "level": 1},
    ]

    def host(_request):
        return [{"position": r["position"], "action": r["action_display"],
                 "frames": build_animation_frames(r), "timeline": build_animation_view(r)} for r in records]

    grid = [list(row) for row in BOARD]
    frame = {"ascii": format_grid_ascii(grid), "step": 7, "level": 1, "shape": [8, 8], "grid": grid}
    out = run_sandboxed_python(
        code=(
            "result = [(e['position'], e['action'], len(e['frames']), e['frames'][0].ascii == current_frame.ascii,\n"
            "           e['timeline']['frames']) for e in call_animations]\n"
            "gone = []\n"
            "try:\n    last_animation_frames\nexcept NameError:\n    gone.append('frames')\n"
            "try:\n    last_animation_timeline\nexcept NameError:\n    gone.append('timeline')\n"
            "result.append(gone)\n"
        ),
        timeout_seconds=20,
        initial_state={"current_frame": frame, "history": [{"action": "", "frame": frame}]},
        action_handler=lambda *a, **k: {},
        animation_handler=host,
    )
    assert out.get("result") == [[2, "UP", 2, False, 2], [4, "LEFT", 3, False, 3], ["frames", "timeline"]], out
