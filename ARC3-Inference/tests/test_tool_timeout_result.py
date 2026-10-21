"""What the model is shown when a python call times out, end to end through ToolAgent (#4)."""
from __future__ import annotations

import json

import pytest

from inference.agent import tool_agent as tool_agent_module
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.agent.tool_agent import ToolAgent


@pytest.fixture
def state_path(tmp_path):
    path = tmp_path / "artifacts" / "g_p0_tool_runtime_state.json"
    frame = Frame(grid=((0, 1), (2, 3)), step=0, level=1)
    write_runtime_state(path, current_frame=frame, history=[HistoryEntry(action="", frame=frame)])
    return path


@pytest.fixture
def agent(state_path):
    instance = ToolAgent(model="test", base_url="http://127.0.0.1:9/v1", provider="vllm")
    steps = []

    def step_env(arguments):
        steps.append(arguments)
        return {
            "executed": True,
            "executed_count": 1,
            "executed_actions": ["UP"],
            "action_num": len(steps),
            "level": 1,
            "gameplay_changed": True,
            "valid_actions": ["UP"],
        }

    instance._step_env_callback = step_env
    instance._current_valid_actions = ["UP"]
    instance._ensure_session(state_path)
    instance._python_timeout = 1
    return instance


def _run(agent, state_path, code):
    return json.loads(agent._run_python_tool(state_path, {"code": code}).content)


def test_timed_out_call_shows_its_output_and_executed_actions(agent, state_path):
    result = _run(
        agent,
        state_path,
        'print("scanning")\naction("UP")\naction("UP")\nprint("stuck")\nwhile True:\n    pass',
    )
    assert result["tool"] == "python"
    assert result["error"] == "Tool timed out after 1s (2 actions already executed)"
    assert result["stdout"] == "scanning\nstuck\n"
    assert "returncode" not in result
    # the next prompt's executed-action line agrees with the error
    assert agent._last_step_summary["executed_count"] == 2


def test_wall_cap_factor_comes_from_the_environment(agent, state_path, monkeypatch):
    seen = []

    def fake_sandbox(**kwargs):
        seen.append(kwargs)
        return {"stdout": "", "result": None, "error": "", "action_results": [],
                "keepable_functions": []}

    monkeypatch.setattr(tool_agent_module, "run_sandboxed_python", fake_sandbox)
    agent._python_timeout = 30
    _run(agent, state_path, "1")
    monkeypatch.setenv("ARC3_TOOL_WALL_CAP_FACTOR", "1")
    _run(agent, state_path, "1")
    monkeypatch.setenv("ARC3_TOOL_WALL_CAP_FACTOR", "0.5")
    _run(agent, state_path, "1")
    assert [call["wall_cap_seconds"] for call in seen] == [90, 30, 30]
    assert all(call["timeout_seconds"] == 30 for call in seen)
