"""ARC3_FRAME_DIFF (default on) controls the sandbox `frame_diff` helper and every
prompt mention of it; off is the ablation arm in docs/harness/frame-diff-fields.md."""

import pytest

from inference.agent import tool_agent
from inference.agent.prompts import FRAME_DIFF_HINT_ADDENDUM
from inference.agent.python_tool_sandbox import frame_diff_enabled, run_sandboxed_python
from inference.utils.grid_utils import format_grid_ascii

GRID_A = [[0, 0, 0], [0, 9, 0], [0, 0, 0]]
GRID_B = [[0, 0, 0], [0, 0, 9], [0, 0, 0]]


def _frame(grid):
    return {"ascii": format_grid_ascii(grid), "step": 0, "level": 1,
            "shape": [len(grid), len(grid[0])], "grid": grid}


def _run(code):
    return run_sandboxed_python(
        code=code,
        timeout_seconds=20,
        initial_state={
            "current_frame": _frame(GRID_B),
            "history": [{"action": "", "frame": _frame(GRID_A)},
                        {"action": "ACTION4", "frame": _frame(GRID_B)}],
        },
        action_handler=lambda *a, **k: {},
    )


@pytest.mark.parametrize("value, expected", [
    (None, True), ("1", True), ("on", True), ("0", False), ("off", False), ("false", False), ("no", False),
])
def test_flag_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("ARC3_FRAME_DIFF", raising=False)
    else:
        monkeypatch.setenv("ARC3_FRAME_DIFF", value)
    assert frame_diff_enabled() is expected


def test_sandbox_exposes_frame_diff_by_default(monkeypatch):
    monkeypatch.delenv("ARC3_FRAME_DIFF", raising=False)
    out = _run("result = frame_diff()['changed_cell_count']\n")
    assert out.get("error") == "", out
    assert out["result"] == 2


def test_sandbox_hides_frame_diff_when_off(monkeypatch):
    monkeypatch.setenv("ARC3_FRAME_DIFF", "0")
    out = _run("try:\n    frame_diff\n    result = 'defined'\nexcept NameError:\n    result = 'missing'\n")
    assert out.get("error") == "", out
    assert out["result"] == "missing"


def _system_prompt(monkeypatch, flag):
    monkeypatch.setenv("ARC3_FRAME_DIFF_HINT", "1")
    monkeypatch.setattr(tool_agent, "_AUTO_FRAME_DIFF", True)
    if flag is None:
        monkeypatch.delenv("ARC3_FRAME_DIFF", raising=False)
    else:
        monkeypatch.setenv("ARC3_FRAME_DIFF", flag)
    return tool_agent._build_system_prompt(tool_output_tokens=1024)


def test_system_prompt_documents_frame_diff_by_default(monkeypatch):
    prompt = _system_prompt(monkeypatch, None)
    assert FRAME_DIFF_HINT_ADDENDUM in prompt
    assert "consult it before acting, and call `frame_diff()` in python" in prompt


def test_system_prompt_never_mentions_frame_diff_when_off(monkeypatch):
    prompt = _system_prompt(monkeypatch, "0")
    assert "frame_diff" not in prompt
    assert "consult it before acting." in prompt


@pytest.mark.parametrize("budget", [10_000, 40, 1])
def test_auto_frame_diff_drops_pointer_when_off(monkeypatch, budget):
    diff = {"changed_cell_count": 40, "moved": [], "rotated": [], "changed_color": [], "resized": [],
            "appeared": [{"color": "b", "pixels": 1, "at": [r, 0], "bbox": [r, 0, r, 0]} for r in range(20)],
            "disappeared": []}
    monkeypatch.delenv("ARC3_FRAME_DIFF", raising=False)
    on = "\n".join(tool_agent._render_auto_frame_diff(diff, budget))
    monkeypatch.setenv("ARC3_FRAME_DIFF", "0")
    off = "\n".join(tool_agent._render_auto_frame_diff(diff, budget))
    assert "frame_diff" not in off
    if budget < 10_000:  # the full listing has no pointer to begin with
        assert "frame_diff" in on
