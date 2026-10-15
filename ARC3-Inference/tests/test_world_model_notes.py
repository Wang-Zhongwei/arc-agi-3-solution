"""ARC3_WORLD_MODEL_NOTES: `<world_model>` updates kept per game in world_model.md."""

import pytest

from inference.agent import tool_agent
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.agent.world_model_notes import extract_world_model, render_notes
from inference.utils.grid_utils import format_grid_ascii


def test_last_complete_block_wins_and_reply_beats_thinking():
    thinking = "<world_model>\ndraft\n</world_model>\nno wait\n<world_model>\nrevised\n</world_model>"
    assert extract_world_model(thinking, "") == "revised"
    assert extract_world_model(thinking, "<world_model>\nfrom reply\n</world_model>") == "from reply"


def test_inline_block_counts():
    assert extract_world_model("so: <world_model>ACTION1 moves up</world_model> ok", "") == "ACTION1 moves up"


@pytest.mark.parametrize("text", [
    "<world_model>\ncut off by the output cap",  # unclosed
    "I should update my <world_model> later",  # mention without a block
    "use `<world_model>` and `</world_model>` tags",  # backticked mention
    "<world_model>\n   \n</world_model>",  # empty
    "<tool_call>\n<function=python>\n<parameter=code>\n<world_model>\nx\n</world_model>\n"
    "</parameter>\n</function>\n</tool_call>",  # inside a tool call
])
def test_non_blocks_are_ignored(text):
    assert extract_world_model(text, "") == ""


def test_a_mention_or_unclosed_draft_before_a_block_is_dropped():
    assert extract_world_model("my <world_model> so far\n<world_model>\nkept\n</world_model>", "") == "kept"


def test_every_line_is_tagged():
    assert render_notes([(1, 3, "a\n\n- b"), (2, 57, "c")]) == "[L1 s3] a\n[L1 s3] - b\n[L2 s57] c\n"
    assert render_notes([]) == ""


@pytest.mark.parametrize("enabled", [True, False])
def test_system_prompt_section_follows_flag(monkeypatch, enabled):
    monkeypatch.setenv("ARC3_WORLD_MODEL_NOTES", "1" if enabled else "0")
    prompt = tool_agent._build_system_prompt(tool_output_tokens=2000)
    assert ("<world_model>" in prompt) is enabled
    assert ("world_model.md" in prompt) is enabled


def _set_frame(state_path, *, level, step):
    frame = Frame(grid=((0, 0), (0, 9)), step=step, level=level)
    write_runtime_state(state_path, current_frame=frame, history=[HistoryEntry("", frame)], flush=True)


def _agent(tmp_path, monkeypatch, state_name="ar25-0c556536_p0_tool_runtime_state.json"):
    monkeypatch.setenv("ARC3_WORLD_MODEL_NOTES", "1")
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    state_path = tmp_path / state_name
    agent._ensure_session(state_path)
    return agent, state_path


def test_updates_are_tagged_deduplicated_and_written_per_game(tmp_path, monkeypatch):
    agent, state_path = _agent(tmp_path, monkeypatch)
    _set_frame(state_path, level=1, step=1)
    agent._record_world_model_note(state_path, "<world_model>\nACTION1 moves the block up\n</world_model>", "")
    agent._record_world_model_note(state_path, "no block this time", "")
    _set_frame(state_path, level=2, step=57)
    agent._record_world_model_note(state_path, "", "<world_model>\nACTION1 moves the block up\n</world_model>")
    agent._record_world_model_note(state_path, "", "<world_model>\nwalls push back\ngoal: fill the frame\n</world_model>")
    expected = "[L1 s1] ACTION1 moves the block up\n[L2 s57] walls push back\n[L2 s57] goal: fill the frame\n"
    assert render_notes(agent._world_model_notes) == expected
    assert (tmp_path / "ar25-0c556536_p0_world_model.md").read_text() == expected


def test_updates_reset_on_a_new_game(tmp_path, monkeypatch):
    agent, state_path = _agent(tmp_path, monkeypatch)
    _set_frame(state_path, level=1, step=3)
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


NOTES = "[L1 s3] ACTION3 rotates the piece\n[L2 s57] walls push back\n[L2 s60] ACTION3 only near walls\n"


@pytest.mark.parametrize("code, expected", [
    ("result = open('world_model.md').read()", NOTES),
    ("result = [l.rstrip() for l in open('world_model.md') if l.startswith('[L2 ')]",
     ["[L2 s57] walls push back", "[L2 s60] ACTION3 only near walls"]),
    ("import re\nresult = re.findall(r'^.*ACTION3.*$', open('world_model.md').read(), re.M)",
     ["[L1 s3] ACTION3 rotates the piece", "[L2 s60] ACTION3 only near walls"]),
    ("with open('./world_model.md') as f:\n    result = len(f.readlines())", 3),
])
def test_sandbox_reads_and_filters_the_file(code, expected):
    out = _run_sandbox(code, world_model_md=NOTES)
    assert out.get("result") == expected, out


@pytest.mark.parametrize("code, message", [
    ("open('world_model.md', 'w')", "read-only"),
    ("open('/etc/passwd')", "the only readable file"),
])
def test_sandbox_open_serves_nothing_else(code, message):
    out = _run_sandbox(code, world_model_md=NOTES)
    assert message in str(out.get("error") or out), out


def test_sandbox_has_no_open_when_off():
    out = _run_sandbox("open('world_model.md')")
    assert "open" in str(out.get("error") or out), out


def test_prompt_examples_run_as_written():
    # the filter examples the prompt shows must run verbatim in the sandbox
    import re

    from inference.agent.prompts import WORLD_MODEL_NOTES_ADDENDUM

    examples = re.findall(r"`(\[l for l.*?|import re;.*?)`", WORLD_MODEL_NOTES_ADDENDUM)
    assert len(examples) == 2, examples
    for example in examples:
        out = _run_sandbox(f"result = {example}" if example.startswith("[") else
                           example.replace("import re; ", "import re\nresult = "), world_model_md=NOTES)
        assert out.get("result"), (example, out)
