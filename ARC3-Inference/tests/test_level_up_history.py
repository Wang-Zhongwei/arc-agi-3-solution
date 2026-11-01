"""ARC3_LEVEL_UP_HISTORY: the sandbox's `level_up_history` holds each level's solved board."""

import json

import pytest

from inference.agent import tool_agent
from inference.agent.runtime_state import Frame, HistoryEntry, discard_runtime_state, write_runtime_state
from inference.utils.grid_utils import format_grid_ascii


def board(value, level, step):
    return Frame(grid=tuple(tuple((value + r) % 16 for _ in range(8)) for r in range(8)), step=step, level=level)


SOLVED = {2: board(9, 1, 2).grid, 4: board(11, 2, 4).grid}  # step -> solved board; step 6: placeholder


def make_agent(path, monkeypatch, enabled):
    monkeypatch.setenv("ARC3_LEVEL_UP_HISTORY", "1" if enabled else "0")
    entries = [HistoryEntry("", board(0, 1, 0))]
    write_runtime_state(path, current_frame=entries[-1].frame, history=entries)

    def step_env(arguments):
        payload = {}
        for _ in arguments.get("actions") or []:
            step = entries[-1].frame.step + 1
            level = entries[-1].frame.level
            completed = step in (2, 4, 6)
            level += int(completed)
            result = {"action_name": "ACTION4", "action_num": step, "level": level,
                      "level_completed": completed, "state": "NOT_FINISHED"}
            entries.append(HistoryEntry("RIGHT", board(step, level, step), result=result))
            payload = {"executed": True, "executed_count": 1, "action_display": "RIGHT",
                       "valid_actions": ["RIGHT"], "gameplay_changed": True, "board_changed": True,
                       **result}
            if step in SOLVED:
                payload["level_end_grid"] = SOLVED[step]
            if completed:
                break  # the solver stops a batch at a level-up
        write_runtime_state(path, current_frame=entries[-1].frame, history=entries)
        return payload

    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    agent._current_valid_actions = ["RIGHT"]
    agent._step_env_callback = step_env
    return agent


def run(agent, path, code):
    payload = json.loads(agent._run_python_tool(path, {"code": code}).content)
    assert "error" not in payload, payload
    return payload["result"]


SNAP = (
    "def snap():\n"
    "    return [(e.action, e.step, e.level, e.frame.ascii if e.frame else None, e.result['level_completed'])\n"
    "            for e in level_up_history]\n"
)


def expected(step, level):
    grid = [list(row) for row in SOLVED[step]] if step in SOLVED else None
    return ["RIGHT", step if grid else None, level if grid else None,
            format_grid_ascii(grid) if grid else None, True]


def test_level_up_history_holds_solved_boards(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    agent = make_agent(path, monkeypatch, True)
    out = run(agent, path, SNAP + (
        "out = [snap()]\n"
        "action(['RIGHT', 'RIGHT', 'RIGHT'])\n"  # stops at the level-up on step 2
        "out.append(snap())\n"
        "out.append([history[-1].frame.level, history[-1].frame.ascii == level_up_history[0].frame.ascii])\n"
        "result = out\n"
    ))
    assert out == [[], [expected(2, 1)], [2, False]]
    # a new call gets the whole list; the next level-ups arrive as tails
    # (the harness refuses further actions in a snippet after a level-up)
    out = run(agent, path, SNAP + (
        "out = [snap()]\n"
        "level_up_history.append('junk')\n"
        "action(['RIGHT', 'RIGHT'])\n"
        "out.append(snap())\n"
        "result = out\n"
    ))
    assert out == [[expected(2, 1)], [expected(2, 1), expected(4, 2)]]
    out = run(agent, path, SNAP + "action(['RIGHT', 'RIGHT'])\nresult = snap()\n")
    assert out == [expected(2, 1), expected(4, 2), expected(6, 3)]  # step 6: no solved board
    discard_runtime_state(path)


def test_level_up_history_off_by_default(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    agent = make_agent(path, monkeypatch, False)
    out = run(agent, path, "action(['RIGHT', 'RIGHT'])\ntry:\n    level_up_history\n    result = True\nexcept NameError:\n    result = False\n")
    assert out is False
    discard_runtime_state(path)


@pytest.mark.parametrize("enabled", [True, False])
def test_prompt_line_follows_the_flag(monkeypatch, enabled):
    monkeypatch.setenv("ARC3_LEVEL_UP_HISTORY", "1" if enabled else "0")
    prompt = tool_agent._build_system_prompt(tool_output_tokens=3072)
    assert ("`level_up_history` is a Python list" in prompt) is enabled
