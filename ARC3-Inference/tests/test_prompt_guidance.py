from unittest.mock import Mock

import pytest

from inference.agent import tool_agent
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state


@pytest.mark.parametrize("memory,retention", [("off", False), ("slim", True), ("full", False)])
def test_no_tool_call_followup_respects_prompt_options(tmp_path, monkeypatch, memory, retention):
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", memory)
    monkeypatch.setenv("ARC3_PERSISTENT_FUNCTIONS", str(retention).lower())
    monkeypatch.setenv("ARC3_API_SIGNATURES", "true")
    monkeypatch.setenv("ARC3_FRAME_DIFF_HINT", "true")
    monkeypatch.setattr(tool_agent, "_LOCAL_ANALYZER_TOOL_TIMEOUT", 7)
    monkeypatch.setattr(tool_agent, "current_grid_image_part", lambda _: None)
    agent = tool_agent.ToolAgent(model="test-prompts")
    monkeypatch.setattr(agent, "_maybe_append_rolling_summary", lambda *args, **kwargs: None)
    response = tool_agent._ChatCompletionResult(
        message={"role": "assistant", "content": "I will inspect the objects."},
        finish_reason="stop",
    )
    request = Mock(return_value=response)
    monkeypatch.setattr(agent, "_chat_completion", request)
    current = Frame(grid=((0, 4), (4, 0)), step=0, level=1)
    path = tmp_path / "runtime.json"
    write_runtime_state(path, current_frame=current, history=[HistoryEntry("", current)])
    transcript = tmp_path / "transcript.txt"
    action = Mock(side_effect=AssertionError("Prompt checks must not act"))

    def stop_after_followup():
        return transcript.exists() and "You have not acted yet." in transcript.read_text()

    agent.analyze(path, 0, valid_actions=["LEFT"], step_env=action,
                  transcript_path=transcript, should_stop=stop_after_followup)
    request.assert_called_once()
    messages = request.call_args.args[0]
    system = messages[0]["content"]
    assert "Current state: step 0, level 1." in messages[1]["content"]
    tool = request.call_args.kwargs["tools"][0]["function"]
    text = transcript.read_text()
    assert ".ascii" not in text and "ASCII" not in tool["description"]
    assert ".ascii" not in tool["description"]
    assert "frame[r0:r1, c0:c1]" in tool["description"]
    assert "History entries expose only `.action` and `.frame`" in tool["description"]
    assert "`transitions[i].result` or `last_transition.result`" in tool["description"]
    assert "History entries and frames have no `.result`" in system
    assert "history[i].result" not in system and "history/transition" not in system
    assert "class HistoryEntryView:\n    action: str\n    frame: FrameView\n\n" in system
    assert "hard time limit of 7 seconds" in system
    assert ("eligible functions are retained" in system) == retention
    assert ("world model" in system.lower()) == (memory != "off")
    followup = text.split("You have not acted yet.", 1)[1]
    assert ("memory update lines" in followup) == (memory != "off")
    assert ("`Recent findings:`" in followup) == (memory == "full")
    action.assert_not_called()
