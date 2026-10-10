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


def _opener(monkeypatch, tmp_path, enabled):
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "current_grid")
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", "off")
    monkeypatch.setenv("ARC3_LEVELUP_BOARD_IMAGE", "1" if enabled else "0")
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    agent._ensure_session(tmp_path / "state.json")
    agent._level_end_grid = tuple((3,) * 64 for _ in range(64))
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
def test_levelup_opener_attaches_solved_board_after_current(monkeypatch, tmp_path, enabled):
    agent, parts = _opener(monkeypatch, tmp_path, enabled)
    images = [part for part in parts if part["type"] == "image_url"]
    assert _color(images[0]) == vision.ARC_COLOR_MAP[4]
    labels = [i for i, part in enumerate(parts) if part.get("text", "").startswith("Level-complete image:")]
    if enabled:
        assert len(images) == 2 and len(labels) == 1
        assert _color(parts[labels[0] + 1]) == vision.ARC_COLOR_MAP[3]
    else:
        assert len(images) == 1 and labels == []
    assert agent._level_end_grid is None
