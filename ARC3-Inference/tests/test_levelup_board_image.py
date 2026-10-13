"""ARC3_LEVELUP_BOARD_IMAGE: the opener after a level-up shows the solved board."""

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from arcengine import ARCBaseGame, ActionInput, GameAction, Level, Sprite
from PIL import Image

from inference.agent import tool_agent, vision_context as vision
from inference.agent.runtime_state import Frame, HistoryEntry
from inference.framework.solver import _level_end_grid


class _OneStepLevels(ARCBaseGame):
    """RIGHT moves the only sprite; reaching x=2 completes the level."""

    def step(self):
        if self.action.id == GameAction.ACTION4:
            player = self.current_level.get_sprites_by_name("p")[0]
            player.move(1, 0)
            if player.x >= 2:
                self.next_level()
        self.complete_action()


def _level(color):
    return Level(sprites=[Sprite([[color]], name="p")], grid_size=(4, 4))


def test_engine_returns_solved_board_before_next_level():
    game = _OneStepLevels("toy", [_level(3), _level(4)])
    game.perform_action(ActionInput(id=GameAction.RESET))
    game.perform_action(ActionInput(id=GameAction.ACTION4), raw=True)
    raw = game.perform_action(ActionInput(id=GameAction.ACTION4), raw=True)
    assert raw.levels_completed == 1
    solved = _level_end_grid(SimpleNamespace(raw=raw))
    colors = {cell for row in solved for cell in row}
    assert 3 in colors and 4 not in colors
    assert solved != tuple(tuple(row) for row in np.asarray(raw.frame[-1]).tolist())


def test_taaf_placeholder_and_single_frame_are_not_boards():
    board = [[0, 3], [5, 5]]
    green = [[14, 14], [14, 14]]
    assert _level_end_grid(SimpleNamespace(raw=SimpleNamespace(frame=[green, board]))) is None
    assert _level_end_grid(SimpleNamespace(raw=SimpleNamespace(frame=[board]))) is None
    assert _level_end_grid(None) is None


def _opener(monkeypatch, tmp_path, enabled, last_shown=None):
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "current_grid")
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", "off")
    monkeypatch.setenv("ARC3_LEVELUP_BOARD_IMAGE", "1" if enabled else "0")
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    agent._ensure_session(tmp_path / "state.json")
    agent._level_end_grid = tuple((3,) * 64 for _ in range(64))
    agent._last_shown_grid = last_shown
    current = Frame(grid=tuple((4,) * 64 for _ in range(64)), step=8, level=2)
    entries = [HistoryEntry("INIT", Frame(grid=current.grid, step=7, level=1)), HistoryEntry("RIGHT", current)]
    summary = {"executed_count": 1, "executed_actions": ["RIGHT"], "level_transition": True}
    agent._auto_diff_prev_frame = entries[0].frame
    prompt = agent._build_user_prompt(current.step, valid_actions=["RIGHT"], current_frame=current,
                                      history_entries=entries, previous_step_summary=summary)
    return agent, agent._build_user_message(prompt, current)["content"]


def _color(part):
    url = part["image_url"]["url"].split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(url))).convert("RGB").getpixel((0, 0))


@pytest.mark.parametrize("enabled", [True, False])
def test_levelup_opener_shows_solved_board_before_current(monkeypatch, tmp_path, enabled):
    agent, parts = _opener(monkeypatch, tmp_path, enabled)
    images = [part for part in parts if part["type"] == "image_url"]
    assert _color(images[-1]) == vision.ARC_COLOR_MAP[4]
    if enabled:
        assert len(images) == 2
        assert parts[0]["text"].endswith("Last level-complete image:")
        assert "You have progressed to a new level!" not in parts[0]["text"]
        assert _color(parts[1]) == vision.ARC_COLOR_MAP[3]
        assert parts[2]["text"].startswith("You have progressed to a new level!")
        assert parts[2]["text"].endswith("\n\nCurrent grid image:")
    else:
        assert len(images) == 1
        assert "Last level-complete image:" not in parts[0]["text"]
    assert agent._level_end_grid is None


def test_levelup_image_skipped_when_interior_matches_last_turn(monkeypatch, tmp_path):
    fringe_only = tuple((5,) * 64 if r < 4 else (3,) * 64 for r in range(64))
    agent, parts = _opener(monkeypatch, tmp_path, True, last_shown=fringe_only)
    assert sum(part["type"] == "image_url" for part in parts) == 1
    assert "Last level-complete image:" not in parts[0]["text"]
    moved = tuple((3,) * 64 if r != 30 else (5,) * 64 for r in range(64))
    agent, parts = _opener(monkeypatch, tmp_path, True, last_shown=moved)
    assert sum(part["type"] == "image_url" for part in parts) == 2
