"""No diff image after a turn in which the model issued RESET."""

import pytest

from inference.agent import tool_agent
from inference.agent.runtime_state import Frame, HistoryEntry


def _grid(value, row):
    return tuple(tuple(value if r == row else 0 for _ in range(64)) for r in range(64))


@pytest.mark.parametrize("actions, expect_diff", [
    (["RESET", "MOUSE(row=30, col=1)"], False),
    (["RESET"], False),
    (["MOUSE(row=30, col=1)"], True),
])
def test_diff_image_only_without_an_issued_reset(monkeypatch, tmp_path, actions, expect_diff):
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "current_grid")
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", "off")
    monkeypatch.setattr(tool_agent, "_DIFF_IMAGE", True)
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    agent._ensure_session(tmp_path / "state.json")
    previous = Frame(grid=_grid(3, 10), step=4, level=1)
    current = Frame(grid=_grid(5, 30), step=4 + len(actions), level=1)
    entries = [HistoryEntry("RIGHT", previous), HistoryEntry(actions[-1], current)]
    agent._auto_diff_prev_frame = previous
    summary = {"executed_count": len(actions), "executed_actions": actions}
    agent._build_user_prompt(current.step, valid_actions=["MOUSE", "RESET"], current_frame=current,
                             history_entries=entries, previous_step_summary=summary)
    assert (agent._pending_diff_image is not None) is expect_diff
