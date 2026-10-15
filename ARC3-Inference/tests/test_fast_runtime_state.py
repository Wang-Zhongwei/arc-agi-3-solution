"""Runtime state served from memory, history sent to the sandbox once then in tails."""

import json

from inference.agent import runtime_state, tool_agent
from inference.agent.runtime_state import (
    Frame, HistoryEntry, discard_runtime_state, load_runtime_state, write_runtime_state,
)
from inference.utils.grid_utils import format_grid_ascii


def frame(step, value=None):
    value = step % 16 if value is None else value
    return Frame(grid=tuple(tuple((value + r + c) % 16 for c in range(8)) for r in range(8)), step=step, level=1)


def test_reads_see_every_write_and_the_file_is_rewritten_only_when_due(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    entries = [HistoryEntry("", frame(0))]
    write_runtime_state(path, current_frame=entries[-1].frame, history=entries)
    assert json.loads(path.read_text())["current_frame"]["step"] == 0  # first write lands on disk
    entries.append(HistoryEntry("RIGHT", frame(1)))
    write_runtime_state(path, current_frame=entries[-1].frame, history=entries)
    current, history = load_runtime_state(path)
    assert current.step == 1 and [e.frame.step for e in history] == [0, 1]
    assert json.loads(path.read_text())["current_frame"]["step"] == 0  # not due yet
    history.append("junk")  # a reader's copy, not the live list
    assert len(load_runtime_state(path)[1]) == 2
    write_runtime_state(path, current_frame=entries[-1].frame, history=entries, flush=True)
    assert json.loads(path.read_text())["current_frame"]["step"] == 1
    monkeypatch.setattr(runtime_state, "RUNTIME_STATE_FILE_INTERVAL_SECONDS", 0.0)
    entries.append(HistoryEntry("RIGHT", frame(2)))
    write_runtime_state(path, current_frame=entries[-1].frame, history=entries)
    assert json.loads(path.read_text())["current_frame"]["step"] == 2  # due again
    discard_runtime_state(path)
    assert not path.exists() and load_runtime_state(path) == (None, [])


def test_sandbox_sees_the_same_history_as_a_full_refresh(tmp_path):
    path = tmp_path / "state.json"
    entries = [HistoryEntry("", frame(0))] + [HistoryEntry("RIGHT", frame(step)) for step in range(1, 6)]
    write_runtime_state(path, current_frame=entries[-1].frame, history=entries)

    def step_env(arguments):
        actions = arguments.get("actions") or []
        for _ in actions:
            step = entries[-1].frame.step + 1
            entries.append(HistoryEntry("RIGHT", frame(step), result={"action_name": "ACTION4"}))
        write_runtime_state(path, current_frame=entries[-1].frame, history=entries)
        return {"executed": True, "executed_count": len(actions), "action_display": "RIGHT",
                "action_name": "ACTION4", "valid_actions": ["RIGHT"], "gameplay_changed": True,
                "board_changed": True, "level": 1, "state": "NOT_FINISHED"}

    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    agent._current_valid_actions = ["RIGHT"]
    agent._step_env_callback = step_env
    code = (
        "def snap():\n"
        "    return [len(history), history[-1].frame.step, previous_frame.step, len(transitions),\n"
        "            last_transition.before_frame.step, current_frame.step, history[2].frame.ascii]\n"
        "out = [snap()]\n"
        "action(['RIGHT'])\n"
        "out.append(snap())\n"
        "history.append('junk')\n"
        "action(['RIGHT', 'RIGHT'])\n"
        "out.append(snap())\n"
        "action(['RIGHT'])\n"
        "out.append(snap())\n"
        "result = out\n"
    )
    result = agent._run_python_tool(path, {"code": code})
    payload = json.loads(result.content) if result.content.lstrip().startswith("{") else {}
    snaps = payload.get("result") or json.loads(json.dumps(payload)).get("result")
    ascii2 = format_grid_ascii([list(row) for row in frame(2).grid])
    assert snaps == [
        [6, 5, 4, 5, 4, 5, ascii2],
        [7, 6, 5, 6, 5, 6, ascii2],
        [9, 8, 7, 8, 7, 8, ascii2],
        [10, 9, 8, 9, 8, 9, ascii2],
    ], payload
