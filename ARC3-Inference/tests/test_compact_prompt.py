"""ARC3_COMPACT_PROMPT: the compact system prompt, tool description and turn messages."""

from inference.agent import tool_agent
from inference.agent.compact_prompt import COMPACT_PYTHON_TOOL_DESCRIPTION, run_length_actions
from inference.agent.runtime_state import Frame, HistoryEntry


def _system(monkeypatch, compact, **env):
    monkeypatch.setenv("ARC3_COMPACT_PROMPT", "1" if compact else "0")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return tool_agent._build_system_prompt(tool_output_tokens=3072)


def test_off_by_default_keeps_the_original_prompt(monkeypatch):
    monkeypatch.delenv("ARC3_COMPACT_PROMPT", raising=False)
    prompt = tool_agent._build_system_prompt(tool_output_tokens=3072)
    assert "Game overview:" in prompt and "Loop: each turn" not in prompt


def test_compact_prompt_is_about_half(monkeypatch):
    original = _system(monkeypatch, False, EXPOSE_RESET="on")
    compact = _system(monkeypatch, True, EXPOSE_RESET="on")
    assert compact.startswith("You are a coding agent solving a grid-based puzzle game.\n\nLoop: each turn")
    assert len(compact) < 0.7 * len(original)
    assert "note(" not in compact and "NOTES" not in compact


def test_compact_prompt_gates_reset_and_undo(monkeypatch):
    on = _system(monkeypatch, True, EXPOSE_RESET="on", EXPOSE_UNDO="on")
    assert "- `RESET`: restores the current level" in on
    assert "- `UNDO`: reverses a previous action" in on
    off = _system(monkeypatch, True, EXPOSE_RESET="off", EXPOSE_UNDO="off")
    assert "`RESET`" not in off and "`UNDO`" not in off


def test_compact_prompt_keeps_the_prefix(monkeypatch):
    prompt = _system(monkeypatch, True, ARC3_SYSTEM_PROMPT_PREFIX="/think")
    assert prompt.startswith("/think\n\nYou are a coding agent")


def test_run_length_actions():
    assert run_length_actions(["A", "A", "A", "B"]) == "Ax3, B"
    assert run_length_actions([str(i) for i in range(13)]) == ", ".join(map(str, range(12))) + " (+1 more, omitted)"
    assert run_length_actions(["A"] * 5 + ["B", "C"], limit=2) == "Ax5, B (+1 more, omitted)"


def _agent(monkeypatch, tmp_path, compact):
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "current_grid")
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", "off")
    monkeypatch.setenv("ARC3_LEVEL_TRANSFER_GUIDANCE", "1")
    monkeypatch.setenv("ARC3_LEVELUP_BOARD_IMAGE", "1")
    monkeypatch.setenv("ARC3_COMPACT_PROMPT", "1" if compact else "0")
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    agent._ensure_session(tmp_path / "state.json")
    return agent


def _turn(agent, actions, level_transition=False):
    current = Frame(grid=tuple((4,) * 64 for _ in range(64)), step=8, level=2)
    entries = [HistoryEntry("INIT", Frame(grid=current.grid, step=7, level=1)), HistoryEntry("RIGHT", current)]
    summary = {"executed_count": len(actions), "executed_actions": actions, "level_transition": level_transition}
    agent._auto_diff_prev_frame = entries[0].frame
    prompt = agent._build_user_prompt(current.step, valid_actions=["RIGHT"], current_frame=current,
                                      history_entries=entries, previous_step_summary=summary)
    return prompt, current


def test_compact_turn_message(monkeypatch, tmp_path):
    prompt, _ = _turn(_agent(monkeypatch, tmp_path, True), ["RIGHT", "RIGHT", "UP"])
    assert "Executed actions: RIGHTx2, UP." in prompt
    assert "Use Python to inspect" not in prompt and "Focus on what changed" not in prompt
    prompt, _ = _turn(_agent(monkeypatch, tmp_path, False), ["RIGHT", "RIGHT", "UP"])
    assert "Executed actions: RIGHT, RIGHT, UP." in prompt
    assert "Use Python to inspect" in prompt and "Focus on what changed" in prompt


def test_compact_level_start_keeps_the_levelup_board(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path, True)
    agent._level_end_grid = tuple((3,) * 64 for _ in range(64))
    prompt, current = _turn(agent, ["RIGHT"], level_transition=True)
    parts = agent._build_user_message(prompt, current)["content"]
    assert sum(part["type"] == "image_url" for part in parts) == 2
    assert parts[0]["text"].endswith("Last level-complete image:")
    assert parts[2]["text"].startswith("You cleared the level. `current_frame` is the NEW level's starting board.")


def test_compact_tool_description(monkeypatch, tmp_path):
    tools = _agent(monkeypatch, tmp_path, True)._tools(tmp_path / "state.json")
    assert tools[0]["function"]["description"] == COMPACT_PYTHON_TOOL_DESCRIPTION
