from inference.agent import tool_agent
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state

TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "python", "arguments": '{"code": "print(1)"}'},
}


def run_turn(monkeypatch, tmp_path, *, yield_tokens, completion_tokens, executed=False, tool_steps=None):
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    agent._yield_tokens = yield_tokens
    agent._tool_steps = tool_steps
    frame = Frame(grid=tuple((0,) * 64 for _ in range(64)), step=1, level=1)
    state = tmp_path / "state.json"
    write_runtime_state(state, current_frame=frame, history=[HistoryEntry("UP", frame)])

    def chat(messages, **kwargs):
        return tool_agent._ChatCompletionResult(
            message={"role": "assistant", "content": "", "tool_calls": [TOOL_CALL]},
            finish_reason="tool_calls",
            usage={"completion_tokens": completion_tokens},
        )

    monkeypatch.setattr(agent, "_chat_completion", chat)
    monkeypatch.setattr(
        agent,
        "_dispatch_tool",
        lambda *args: tool_agent._ToolDispatchResult(content="1", step_executed=executed),
    )
    result = agent.analyze(state, 1, valid_actions=["UP"], transcript_path=tmp_path / "t.txt")
    tool_messages = [m for m in agent._history_messages if m.get("role") == "tool"]
    return result, tool_messages, (tmp_path / "t.txt").read_text()


def test_token_yield_says_so_in_the_tool_result(monkeypatch, tmp_path):
    result, tool_messages, transcript = run_turn(
        monkeypatch, tmp_path, yield_tokens=100, completion_tokens=150
    )
    assert result.yielded_control and not result.step_executed
    assert tool_messages[-1]["content"].startswith("1\n\n[Turn token limit reached")
    assert "about 150 tokens (limit 100)" in tool_messages[-1]["content"]
    assert "Yielded control to solver: turn_token_budget." in transcript


def test_no_note_below_the_limit_or_when_an_action_ran(monkeypatch, tmp_path):
    # one exchange of 150 tokens, then the tool-step budget ends the turn
    _, below, _ = run_turn(
        monkeypatch, tmp_path / "below", yield_tokens=1000, completion_tokens=150, tool_steps=1
    )
    _, acted, _ = run_turn(
        monkeypatch, tmp_path / "acted", yield_tokens=100, completion_tokens=150, executed=True
    )
    for messages in (below, acted):
        assert all("Turn token limit" not in m["content"] for m in messages)
