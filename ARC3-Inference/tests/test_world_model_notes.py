"""ARC3_WORLD_MODEL_NOTES: `<world_model>` blocks logged per game, read from Python."""

import pytest

from inference.agent import tool_agent
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.agent.world_model_notes import extract_world_model
from inference.utils.grid_utils import format_grid_ascii


def test_last_complete_block_wins_and_reply_beats_thinking():
    thinking = "<world_model>\ndraft\n</world_model>\nno wait\n<world_model>\nrevised\n</world_model>"
    assert extract_world_model(thinking, "") == "revised"
    assert extract_world_model(thinking, "<world_model>\nfrom reply\n</world_model>") == "from reply"


@pytest.mark.parametrize("text", [
    "<world_model>\ncut off by the output cap",  # unclosed
    "I should update my <world_model> later",  # mention, not at line start
    "use `<world_model>` tags",  # backticked mention
    "<world_model>\n   \n</world_model>",  # empty
    "<tool_call>\n<function=python>\n<parameter=code>\n<world_model>\nx\n</world_model>\n"
    "</parameter>\n</function>\n</tool_call>",  # inside a tool call
])
def test_non_blocks_are_ignored(text):
    assert extract_world_model(text, "") == ""


def test_unclosed_draft_before_a_complete_block_is_dropped():
    text = "<world_model>\nabandoned draft\n<world_model>\nkept\n</world_model>"
    assert extract_world_model(text, "") == "kept"


def test_indented_block_and_inline_close():
    assert extract_world_model("  <world_model> ACTION1 moves up </world_model>", "") == "ACTION1 moves up"


@pytest.mark.parametrize("enabled", [True, False])
def test_system_prompt_section_follows_flag(monkeypatch, enabled):
    monkeypatch.setenv("ARC3_WORLD_MODEL_NOTES", "1" if enabled else "0")
    prompt = tool_agent._build_system_prompt(tool_output_tokens=2000)
    assert ("World-model notes:" in prompt) is enabled
    assert ("`world_model_notes`" in prompt) is enabled


def _agent(tmp_path, monkeypatch, *, level, step):
    monkeypatch.setenv("ARC3_WORLD_MODEL_NOTES", "1")
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    state_path = tmp_path / "state.json"
    agent._ensure_session(state_path)
    _set_frame(state_path, level=level, step=step)
    return agent, state_path


def _set_frame(state_path, *, level, step):
    frame = Frame(grid=((0, 0), (0, 9)), step=step, level=level)
    write_runtime_state(state_path, current_frame=frame, history=[HistoryEntry("", frame)], flush=True)


def test_notes_are_tagged_deduplicated_and_mirrored(tmp_path, monkeypatch):
    agent, state_path = _agent(tmp_path, monkeypatch, level=1, step=1)
    agent._record_world_model_note(state_path, "<world_model>\nACTION1 moves the block up\n</world_model>", "")
    agent._record_world_model_note(state_path, "no block this time", "")
    _set_frame(state_path, level=2, step=57)
    agent._record_world_model_note(state_path, "", "<world_model>\nACTION1 moves the block up\n</world_model>")
    agent._record_world_model_note(state_path, "", "<world_model>\nwalls now push back\n</world_model>")
    assert agent._world_model_notes == [
        "[L1 s1] ACTION1 moves the block up",
        "[L2 s57] walls now push back",
    ]
    mirrored = (tmp_path / "state_world_model.md").read_text()
    assert mirrored == "[L1 s1] ACTION1 moves the block up\n\n[L2 s57] walls now push back\n"


def test_notes_reset_on_a_new_game(tmp_path, monkeypatch):
    agent, state_path = _agent(tmp_path, monkeypatch, level=1, step=3)
    agent._record_world_model_note(state_path, "<world_model>\nx\n</world_model>", "")
    assert agent._world_model_notes
    other = tmp_path / "next-game" / "state.json"
    other.parent.mkdir()
    agent._ensure_session(other)
    assert agent._world_model_notes == []


def _run_sandbox(code, **state):
    grid = [[0, 0], [0, 9]]
    frame = {"ascii": format_grid_ascii(grid), "step": 0, "level": 1, "shape": [2, 2], "grid": grid}
    return run_sandboxed_python(
        code=code, timeout_seconds=20,
        initial_state={"current_frame": frame, "history": [{"action": "", "frame": frame}], **state},
        action_handler=lambda *a, **k: {},
    )


def test_sandbox_exposes_notes_only_when_sent():
    out = _run_sandbox("result = world_model_notes", world_model_notes="[L1 s1] a")
    assert out.get("result") == "[L1 s1] a", out
    out = _run_sandbox("result = 'world_model_notes' in globals()")
    assert "error" in out or out.get("result") is False, out


def test_mirror_is_named_after_the_game_run(tmp_path, monkeypatch):
    agent, _ = _agent(tmp_path, monkeypatch, level=1, step=0)
    state_path = tmp_path / "ar25-0c556536_p0_tool_runtime_state.json"
    agent._ensure_session(state_path)
    _set_frame(state_path, level=1, step=4)
    agent._record_world_model_note(state_path, "<world_model>\nx\n</world_model>", "")
    assert (tmp_path / "ar25-0c556536_p0_world_model.md").read_text() == "[L1 s4] x\n"
