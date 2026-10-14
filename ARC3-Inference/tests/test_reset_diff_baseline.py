"""After a deliberate RESET the diff image starts from the board RESET returned."""

import pytest

from inference.agent import tool_agent, vision_context as vision
from inference.agent.runtime_state import Frame, HistoryEntry


def _grid(value, row=10):
    return tuple(tuple(value if r == row else 0 for _ in range(64)) for r in range(64))


BEFORE, LEVEL_START, AFTER = _grid(3), _grid(5, row=20), _grid(5, row=30)


def _opener(monkeypatch, tmp_path, actions, frames):
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "current_grid")
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", "off")
    monkeypatch.setenv("EXPOSE_RESET", "on")
    monkeypatch.setattr(tool_agent, "_DIFF_IMAGE", True)
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    agent._ensure_session(tmp_path / "state.json")
    previous = Frame(grid=BEFORE, step=4, level=1)
    entries = [HistoryEntry("RIGHT", previous)]
    for step, (action, grid) in enumerate(zip(actions, frames), start=5):
        name = "RESET" if action == "RESET" else "ACTION6"
        entries.append(HistoryEntry(action, Frame(grid=grid, step=step, level=1),
                                    result={"action_name": name, "automatic": False}))
    current = entries[-1].frame
    agent._auto_diff_prev_frame = previous
    summary = {"executed_count": len(actions), "executed_actions": list(actions)}
    agent._build_user_prompt(current.step, valid_actions=["MOUSE", "RESET"], current_frame=current,
                             history_entries=entries, previous_step_summary=summary)
    return agent


def test_reset_then_actions_diffs_from_the_reset_board(monkeypatch, tmp_path):
    agent = _opener(monkeypatch, tmp_path, ["RESET", "MOUSE(row=30, col=1)"], [LEVEL_START, AFTER])
    expected = vision.diff_image_part(Frame(grid=LEVEL_START, step=5, level=1), Frame(grid=AFTER, step=6, level=1))
    assert agent._pending_diff_image == expected
    assert "since your RESET" in agent._pending_diff_image_label
    assert "step 5 -> 6" in agent._pending_diff_image_label


def test_reset_alone_has_no_diff_image(monkeypatch, tmp_path):
    agent = _opener(monkeypatch, tmp_path, ["RESET"], [LEVEL_START])
    assert agent._pending_diff_image is None


def test_without_reset_the_diff_spans_the_whole_turn(monkeypatch, tmp_path):
    agent = _opener(monkeypatch, tmp_path, ["MOUSE(row=30, col=1)"], [AFTER])
    expected = vision.diff_image_part(Frame(grid=BEFORE, step=4, level=1), Frame(grid=AFTER, step=5, level=1))
    assert agent._pending_diff_image == expected
    assert "since your previous analyzer turn" in agent._pending_diff_image_label
