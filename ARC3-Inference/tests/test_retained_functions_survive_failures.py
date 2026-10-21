"""A python call the sandbox child never answers must not wipe the retained pool (#5).

Timeouts, crashes, unreadable or unknown replies and a failed start all come
back without 'keepable_functions'. They say nothing about the pool, so the
earlier helpers stay and no retention notes are added. A reply the child does
send (final or error) still decides the pool exactly as before.
"""
from __future__ import annotations

import json
import subprocess

import pytest

from inference.agent import python_tool_sandbox as sandbox_module
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.agent.tool_agent import ToolAgent

HELPERS = {
    "shift": "def shift(x):\n    return x + 1",
    "double": "def double(x):\n    return x * 2",
}

# Stand-ins for the sandbox child: each reads the request, then misbehaves.
_CRASHING_CHILD = "import sys\nsys.stdin.readline()\nsys.stderr.write('boom')\nsys.exit(1)\n"
_GARBLED_CHILD = "import sys\nsys.stdin.readline()\nprint('not json', flush=True)\n"
_UNKNOWN_CHILD = "import sys\nsys.stdin.readline()\nprint('{\"type\": \"mystery\"}', flush=True)\n"


@pytest.fixture(autouse=True)
def persistent_functions(monkeypatch):
    monkeypatch.setenv("ARC3_PERSISTENT_FUNCTIONS", "1")
    monkeypatch.setenv("ARC3_PERSISTENT_FUNCTIONS_SCOPE", "game")


@pytest.fixture
def state_path(tmp_path):
    path = tmp_path / "artifacts" / "g_p0_tool_runtime_state.json"
    frame = Frame(grid=((0, 1), (2, 3)), step=0, level=1)
    write_runtime_state(path, current_frame=frame, history=[HistoryEntry(action="", frame=frame)])
    return path


@pytest.fixture
def agent(state_path):
    instance = ToolAgent(model="test", base_url="http://127.0.0.1:9/v1", provider="vllm")
    instance._step_env_callback = lambda arguments: {}
    instance._current_valid_actions = ["UP"]
    instance._ensure_session(state_path)
    instance._kept_functions = dict(HELPERS)
    return instance


def _run(agent, state_path, code: str) -> dict:
    return json.loads(agent._run_python_tool(state_path, {"code": code}).content)


def _assert_pool_kept(agent, result: dict) -> None:
    assert agent._kept_functions == HELPERS
    assert "function_retention" not in result
    context = agent._retained_function_context()
    assert "shift(x)" in context and "double(x)" in context
    assert "no longer retained" not in context


def test_timeout_keeps_previous_pool(agent, state_path):
    agent._python_timeout = 1
    result = _run(agent, state_path, "def fresh(y):\n    return y\nwhile True:\n    pass")
    assert result["error"] == "Tool timed out after 1s"
    _assert_pool_kept(agent, result)  # `fresh` never reported, so it is not kept


@pytest.mark.parametrize("child, error", [
    (_CRASHING_CHILD, "Sandbox process exited unexpectedly."),
    (_GARBLED_CHILD, "Sandbox process returned an invalid response."),
    (_UNKNOWN_CHILD, "Sandbox process returned an unknown message type."),
], ids=["crash", "invalid-json", "unknown-type"])
def test_child_that_never_reports_keeps_previous_pool(agent, state_path, monkeypatch, child, error):
    monkeypatch.setattr(sandbox_module, "_SANDBOX_BOOTSTRAP", child)
    result = _run(agent, state_path, "print(shift(1))")
    assert result["error"] == error
    _assert_pool_kept(agent, result)


def test_failed_start_keeps_previous_pool(agent, state_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("no processes left")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    result = _run(agent, state_path, "print(shift(1))")
    assert result["error"] == "Sandbox process could not start."
    _assert_pool_kept(agent, result)


def test_reported_error_still_records_pool(agent, state_path):
    result = _run(agent, state_path, "def fresh(y):\n    return double(y)\n1 / 0")
    assert "ZeroDivisionError" in result["error"]
    assert agent._kept_functions == {**HELPERS, "fresh": "def fresh(y):\n    return double(y)"}
    assert "Retained your function fresh(y)" in result["function_retention"]


def test_reported_empty_list_still_clears_pool(agent):
    payload = {}
    agent._record_retained_functions({"keepable_functions": [], "retention_rejected": []}, payload)
    assert agent._kept_functions == {}
    notes = payload["function_retention"]
    assert "Your previous function shift is no longer retained" in notes
    assert "Your previous function double is no longer retained" in notes


def test_snippet_that_removes_helpers_still_clears_pool(agent, state_path):
    result = _run(agent, state_path, "del shift\ndel double\nprint(1)")
    assert result["stdout"] == "1\n"
    assert agent._kept_functions == {}
    assert "Your function shift was not retained" in result["function_retention"]
    assert "Your function double was not retained" in result["function_retention"]
