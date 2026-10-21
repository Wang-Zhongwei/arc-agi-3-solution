"""History keeps each reply as the model generated it, so the next request reuses its cache (#3)."""
from __future__ import annotations

import json

import pytest

from inference.agent import tool_agent as tool_agent_module
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.agent.tool_agent import ToolAgent, _ChatCompletionResult

# Paragraph breaks, a whitespace-only line, and the outer whitespace the
# template trims.
REASONING = "\nI look at the board first.\n\nThe player is at (3, 4).\n   \nSo I probe UP.\n"
CONTENT = "Plan:\n\n1. probe\n\n2. act\n"


@pytest.fixture
def state_path(tmp_path):
    path = tmp_path / "artifacts" / "g_p0_tool_runtime_state.json"
    frame = Frame(grid=((0, 1), (2, 3)), step=0, level=1)
    write_runtime_state(path, current_frame=frame, history=[HistoryEntry(action="", frame=frame)])
    return path


@pytest.fixture
def agent(state_path, monkeypatch):
    monkeypatch.setenv("ARC3_REASONING_HISTORY_KEY", "reasoning_content")
    instance = ToolAgent(model="test", base_url="http://127.0.0.1:9/v1", provider="vllm")
    instance._tool_steps = 2
    return instance


@pytest.fixture
def sandbox(monkeypatch):
    """The code the python tool runs, without starting a sandbox."""
    ran: list[str] = []

    def fake_sandbox(**kwargs):
        ran.append(kwargs["code"])
        return {"stdout": "ok\n", "result": None, "error": "", "action_results": [],
                "keepable_functions": []}

    monkeypatch.setattr(tool_agent_module, "run_sandboxed_python", fake_sandbox)
    return ran


def _reply(reasoning: str, content: str = "", code: str | None = None) -> _ChatCompletionResult:
    message = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if code is not None:
        message["tool_calls"] = [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "python", "arguments": json.dumps({"code": code}, ensure_ascii=False)},
        }]
    return _ChatCompletionResult(
        message=message,
        finish_reason="tool_calls" if code is not None else "stop",
        usage={"prompt_tokens": 1000, "completion_tokens": 100},
    )


def _run_turn(agent, state_path, monkeypatch, replies):
    """Run one analyzer turn against scripted replies; return each request's messages."""
    sent: list[list[dict]] = []
    queue = list(replies)

    def fake_chat_completion(messages, **kwargs):
        sent.append(json.loads(json.dumps(messages)))
        return queue.pop(0)

    monkeypatch.setattr(agent, "_chat_completion", fake_chat_completion)
    agent.analyze(state_path, action_num=1, valid_actions=["UP"], step_env=lambda arguments: {})
    return sent


def _assistant_messages(messages):
    return [message for message in messages if message.get("role") == "assistant"]


def test_reasoning_with_paragraph_breaks_round_trips_into_history(
    agent, state_path, monkeypatch, sandbox
):
    monkeypatch.setenv("ARC3_VERBATIM_HISTORY", "1")
    sent = _run_turn(agent, state_path, monkeypatch, [
        _reply(REASONING, code="print('probe')\n"),
        _reply("done", code="print(1)"),
    ])

    stored = _assistant_messages(sent[1])[0]
    assert stored["reasoning_content"] == REASONING.strip()
    assert "\n\n" in stored["reasoning_content"]
    # and it stays that way in the history the next turn starts from
    assert _assistant_messages(agent._history_messages)[0]["reasoning_content"] == REASONING.strip()


def test_content_with_paragraph_breaks_round_trips_into_history(
    agent, state_path, monkeypatch, sandbox
):
    monkeypatch.setenv("ARC3_VERBATIM_HISTORY", "1")
    sent = _run_turn(agent, state_path, monkeypatch, [
        _reply(REASONING, content=CONTENT, code="print('probe')"),
        _reply("done", code="print(1)"),
    ])

    stored = _assistant_messages(sent[1])[0]
    assert stored["content"] == CONTENT.strip()


def test_no_tool_call_reply_round_trips_into_history(agent, state_path, monkeypatch, sandbox):
    monkeypatch.setenv("ARC3_VERBATIM_HISTORY", "1")
    sent = _run_turn(agent, state_path, monkeypatch, [
        _reply(REASONING, content=CONTENT),
        _reply("done", code="print(1)"),
    ])

    stored = _assistant_messages(sent[1])[0]
    assert stored["reasoning_content"] == REASONING.strip()
    assert stored["content"] == CONTENT.strip()
    assert "tool_calls" not in stored


def test_transcript_keeps_the_normalised_text(agent, state_path, monkeypatch, sandbox, tmp_path):
    monkeypatch.setenv("ARC3_VERBATIM_HISTORY", "1")
    queue = [_reply(REASONING, code="print('probe')"), _reply("done", code="print(1)")]
    monkeypatch.setattr(agent, "_chat_completion", lambda messages, **kw: queue.pop(0))
    transcript = tmp_path / "transcript.txt"
    result = agent.analyze(
        state_path, action_num=1, valid_actions=["UP"], step_env=lambda arguments: {},
        transcript_path=transcript,
    )

    normalised = "I look at the board first.\nThe player is at (3, 4).\nSo I probe UP."
    assert normalised in transcript.read_text(encoding="utf-8")
    assert result.reasoning == "done"


def test_default_still_collapses_blank_lines(agent, state_path, monkeypatch, sandbox):
    monkeypatch.delenv("ARC3_VERBATIM_HISTORY", raising=False)
    sent = _run_turn(agent, state_path, monkeypatch, [
        _reply(REASONING, content=CONTENT, code="print('probe')"),
        _reply("done", code="print(1)"),
    ])

    stored = _assistant_messages(sent[1])[0]
    assert stored["reasoning_content"] == (
        "I look at the board first.\nThe player is at (3, 4).\nSo I probe UP."
    )
    assert stored["content"] == "Plan:\n1. probe\n2. act"


# Trailing newlines, a blank line inside, and an indented first line: the
# python tool runs code.rstrip(), and none of that may reach history.
CODE = "\n  # probe\nx = 1\n\nprint(x)\n\n\n"


@pytest.mark.parametrize("verbatim", ["0", "1"])
def test_tool_call_arguments_in_history_equal_what_the_model_sent(
    agent, state_path, monkeypatch, sandbox, verbatim
):
    monkeypatch.setenv("ARC3_VERBATIM_HISTORY", verbatim)
    first = _reply(REASONING, code=CODE)
    sent_arguments = first.message["tool_calls"][0]["function"]["arguments"]
    sent = _run_turn(agent, state_path, monkeypatch, [first, _reply("done", code="print(1)")])

    stored = _assistant_messages(sent[1])[0]["tool_calls"][0]["function"]
    assert stored["arguments"] == sent_arguments
    assert json.loads(stored["arguments"])["code"] == CODE
    kept = _assistant_messages(agent._history_messages)[0]["tool_calls"][0]["function"]
    assert kept["arguments"] == sent_arguments
    # what ran is the stripped copy
    assert sandbox[0] == CODE.rstrip()


SUMMARY = "\nLevel 1: the key opens the door.\n\nTried: UP twice.\n\n- next: LEFT\n"


@pytest.mark.parametrize(
    ("verbatim", "stored"),
    [("1", SUMMARY.strip()), ("0", "Level 1: the key opens the door.\nTried: UP twice.\n- next: LEFT")],
)
def test_rolling_summary_reply_is_stored_verbatim(agent, monkeypatch, verbatim, stored):
    monkeypatch.setenv("ARC3_VERBATIM_HISTORY", verbatim)
    monkeypatch.setenv("ARC3_SUMMARY_INTERVAL_TOKENS", "1")
    agent._history_messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "reasoning_content": "r"},
    ]
    agent._summary_context_tokens_since_attempt = 10
    monkeypatch.setattr(
        agent, "_chat_completion",
        lambda messages, **kwargs: _ChatCompletionResult(
            message={"role": "assistant", "content": SUMMARY}, finish_reason="stop",
        ),
    )
    agent._maybe_append_rolling_summary(lambda label, text: None, tools=None)
    assert agent._history_messages[-1]["content"] == stored
