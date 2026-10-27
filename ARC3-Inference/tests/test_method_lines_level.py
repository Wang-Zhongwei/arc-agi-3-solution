"""ARC3_METHOD_LINES_UNTIL_LEVEL: the opener's two methodology lines stop after a level."""

import pytest

from inference.agent import tool_agent
from inference.agent.runtime_state import Frame, HistoryEntry

INSPECT = "Use Python to inspect the evidence from the newest history"
FOCUS = "Focus on what changed most recently in `history`"


def _prompt(monkeypatch, tmp_path, level, until):
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", "off")
    if until is None:
        monkeypatch.delenv("ARC3_METHOD_LINES_UNTIL_LEVEL", raising=False)
    else:
        monkeypatch.setenv("ARC3_METHOD_LINES_UNTIL_LEVEL", until)
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    agent._ensure_session(tmp_path / "state.json")
    grid = tuple((0,) * 8 for _ in range(8))
    current = Frame(grid=grid, step=8, level=level)
    entries = [HistoryEntry("INIT", Frame(grid=grid, step=7, level=level)), HistoryEntry("RIGHT", current)]
    summary = {"executed_count": 1, "executed_actions": ["RIGHT"]}
    return agent._build_user_prompt(current.step, valid_actions=["RIGHT"], current_frame=current,
                                    history_entries=entries, previous_step_summary=summary)


@pytest.mark.parametrize("level,until,shown", [
    (1, None, True), (5, None, True), (5, "0", True),
    (1, "2", True), (2, "2", True), (3, "2", False), (7, "2", False),
])
def test_method_lines_follow_the_level_ceiling(monkeypatch, tmp_path, level, until, shown):
    prompt = _prompt(monkeypatch, tmp_path, level, until)

    assert (INSPECT in prompt) is shown
    assert (FOCUS in prompt) is shown
    assert f"level {level}" in prompt
