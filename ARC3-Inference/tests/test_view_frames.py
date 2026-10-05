"""view_frames: past boards by history index, attached as images after the call."""

import base64
import io

from PIL import Image

from inference.agent import tool_agent
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state


def frame(level, step, colour):
    return Frame(grid=tuple(tuple([colour] * 8) for _ in range(8)), step=step, level=level)


# start, a level-up (action 1), a death (action 2), its automatic reset (action 3)
HISTORY = [
    HistoryEntry("RESET", frame(1, 0, 0), {}),
    HistoryEntry("LEFT", frame(2, 1, 1), {"action_num": 1, "level": 2, "level_completed": True}),
    HistoryEntry("UP", frame(2, 2, 2), {"action_num": 2, "level": 2, "game_over": True}),
    HistoryEntry("RESET", frame(2, 3, 3), {"action_num": 3, "level": 2, "automatic": True}),
]


def run(code, limit):
    return run_sandboxed_python(
        code=code,
        timeout_seconds=20,
        initial_state={
            "current_frame": tool_agent._ascii_frame_view_payload(HISTORY[-1].frame),
            "history": tool_agent._ascii_history_view_payload(HISTORY),
            "valid_actions": [],
            "last_action_call_result": {},
        },
        action_handler=lambda actions, **_: {},
        view_frames_limit=limit,
    )


def test_labels_and_indices_follow_history():
    result = run("view_frames([1, 2, -1])", limit=4)
    assert result["error"] == ""
    assert result["view_frames"] == [1, 2, 3]
    stdout = result["stdout"]
    assert "history[1]: board after LEFT, level 2, step 1 (level_completed)" in stdout
    assert "history[2]: board after UP, level 2, step 2 (game_over)" in stdout
    assert "history[3]: board after RESET, level 2, step 3" in stdout


def test_per_call_limit_and_duplicates():
    result = run("view_frames([0, 0, 1])\nview_frames([1, 2, 3])", limit=2)
    assert result["view_frames"] == [0, 1]
    assert "skipped history[2, 3]: at most 2 images per python call." in result["stdout"]


def test_out_of_range_index_is_an_error():
    result = run("view_frames([9])", limit=4)
    assert "out of range (0..3)" in result["error"]
    assert result["view_frames"] == []


def test_hidden_when_disabled():
    result = run("view_frames([0])", limit=0)
    assert "view_frames" in result["error"]
    assert result["view_frames"] == []


def test_host_renders_requested_boards(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC3_VIEW_FRAMES", "4")
    monkeypatch.setenv("ARC3_VIEW_FRAMES_UPSCALE", "4")
    state = tmp_path / "state.json"
    write_runtime_state(state, current_frame=HISTORY[-1].frame, history=HISTORY)
    images = tool_agent.ToolAgent._view_frames_images(None, state, [2, 7])
    assert [label for label, _ in images] == ["history[2]: board after UP, level 2, step 2"]
    url = images[0][1]["image_url"]["url"]
    image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert image.size == (32, 32)


def test_prompt_lines_are_gated(monkeypatch):
    base = tool_agent._build_system_prompt(tool_output_tokens=1024)
    assert "view_frames" not in base and "conversation trimming" not in base
    monkeypatch.setenv("ARC3_HISTORY_REVISIT_HINT", "1")
    monkeypatch.setenv("ARC3_VIEW_FRAMES", "4")
    prompt = tool_agent._build_system_prompt(tool_output_tokens=1024)
    assert "RETAIN earlier actions and boards even after conversation trimming" in prompt
    assert "game over.\n- `view_frames(indices)`" in prompt
    assert "`view_frames(indices)`" in prompt and "At most 4 images per call." in prompt
