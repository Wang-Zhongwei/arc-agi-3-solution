"""LOCAL_ANALYZER_THINKING_TOKEN_BUDGET: vLLM request field or sglang continuation."""

import pytest

from inference.agent import tool_agent
from inference.agent.tool_agent import ToolAgent, _ChatCompletionResult

BUDGET = 4096
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
TOOL_CALL = [{"id": "c1", "type": "function", "function": {"name": "python", "arguments": "{}"}}]


def make_agent(monkeypatch, *, budget=BUDGET, mode="continue", max_output=None, replies=()):
    monkeypatch.setattr(tool_agent, "_LOCAL_ANALYZER_THINKING_TOKEN_BUDGET", budget)
    monkeypatch.setattr(tool_agent, "_LOCAL_ANALYZER_ENABLE_THINKING", True)
    monkeypatch.setenv("ARC3_THINKING_BUDGET_MODE", mode)
    agent = object.__new__(ToolAgent)
    agent._max_output_tokens = max_output
    agent.sent = []
    pending = list(replies)

    def once(messages, **kwargs):
        agent.sent.append({"messages": messages, **kwargs})
        return pending.pop(0)

    agent._chat_completion_once = once
    return agent


def reply(finish, *, reasoning="", content="", tool_calls=None, completion=0, reasoning_tokens=0):
    message = {"role": "assistant", "content": content, "reasoning_content": reasoning or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = {"prompt_tokens": 1000, "completion_tokens": completion, "reasoning_tokens": reasoning_tokens}
    return _ChatCompletionResult(message=message, finish_reason=finish, usage=usage)


def test_disabled_budget_sends_one_unchanged_request(monkeypatch):
    agent = make_agent(monkeypatch, budget=0, replies=[reply("tool_calls", tool_calls=TOOL_CALL)])
    agent._chat_completion(MESSAGES, tools=None)
    assert len(agent.sent) == 1
    assert agent.sent[0]["max_tokens"] is None
    assert "extra_payload" not in agent.sent[0]


def test_reply_within_budget_is_returned_as_is(monkeypatch):
    first = reply("tool_calls", reasoning="short", tool_calls=TOOL_CALL)
    agent = make_agent(monkeypatch, replies=[first])
    assert agent._chat_completion(MESSAGES, tools=None) is first
    assert agent.sent[0]["max_tokens"] == BUDGET


def test_reply_cut_at_budget_is_resumed_after_closing_think(monkeypatch):
    agent = make_agent(
        monkeypatch,
        max_output=12288,
        replies=[
            reply("length", reasoning="long thought ", completion=BUDGET, reasoning_tokens=BUDGET),
            reply("tool_calls", content="", tool_calls=TOOL_CALL, completion=80),
        ],
    )
    result = agent._chat_completion(MESSAGES, tools=None)

    resumed = agent.sent[1]
    assert resumed["messages"][:-1] == MESSAGES
    assert resumed["messages"][-1] == {"role": "assistant", "content": "long thought\n</think>\n\n"}
    assert resumed["extra_payload"] == {"continue_final_message": True, "separate_reasoning": False}
    assert resumed["max_tokens"] == 12288 - BUDGET

    assert result.finish_reason == "tool_calls"
    assert result.message["tool_calls"] == TOOL_CALL
    assert result.message["reasoning_content"] == "long thought"
    assert result.usage == {
        "prompt_tokens": 1000,
        "completion_tokens": BUDGET + 80,
        "reasoning_tokens": BUDGET,
        "total_tokens": 1000 + BUDGET + 80,
    }


def test_cut_without_reasoning_is_not_resumed(monkeypatch):
    first = reply("length", content="answer cut off")
    agent = make_agent(monkeypatch, replies=[first])
    assert agent._chat_completion(MESSAGES, tools=None) is first
    assert len(agent.sent) == 1


def test_server_mode_sends_vllm_field(monkeypatch):
    agent = make_agent(monkeypatch, mode="server", replies=[reply("tool_calls", tool_calls=TOOL_CALL)])
    agent._chat_completion(MESSAGES, tools=None)
    assert agent.sent[0]["extra_payload"] == {"thinking_token_budget": BUDGET}
    assert agent.sent[0]["max_tokens"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": 8192},  # the rolling summary keeps its own limit
        {"enable_thinking": False},
    ],
)
def test_side_requests_are_not_budgeted(monkeypatch, kwargs):
    agent = make_agent(monkeypatch, replies=[reply("stop", content="x")])
    agent._chat_completion(MESSAGES, tools=None, **kwargs)
    assert len(agent.sent) == 1
    assert "extra_payload" not in agent.sent[0]


def test_output_cap_at_or_below_budget_leaves_budget_off(monkeypatch):
    agent = make_agent(monkeypatch, max_output=BUDGET, replies=[reply("stop", content="x")])
    agent._chat_completion(MESSAGES, tools=None)
    assert agent.sent[0]["max_tokens"] is None


@pytest.mark.parametrize(
    ("owned_by", "expected"), [("vllm", "server"), ("sglang", "continue"), (None, "continue")]
)
def test_auto_mode_detects_server_from_models(monkeypatch, owned_by, expected):
    agent = make_agent(monkeypatch, mode="auto")
    agent._model = type("M", (), {"base_url": "http://server/v1"})()
    agent._headers = lambda: {}

    class Response:
        def json(self):
            if owned_by is None:
                raise ValueError("not json")
            return {"data": [{"id": "flashnext", "owned_by": owned_by}]}

    calls = []
    monkeypatch.setattr(tool_agent.requests, "get", lambda url, **kw: calls.append(url) or Response())
    assert agent._thinking_budget_enforcement() == expected
    assert agent._thinking_budget_enforcement() == expected
    assert calls == ["http://server/v1/models"]  # resolved once per agent


def test_extra_payload_reaches_the_request(monkeypatch):
    agent = object.__new__(ToolAgent)
    agent._model = type("M", (), {"provider": "vllm", "model_id": "m", "base_url": "http://server/v1"})()
    agent._max_output_tokens = None
    agent._timeout = 10
    agent._http_initial_grace_used = True
    agent._headers = lambda: {}
    agent._harness_template_kwargs = lambda: {}
    posted = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(tool_agent.requests, "post", lambda url, json, **kw: posted.append(json) or Response())
    agent._chat_completion_once(
        MESSAGES, tools=None, extra_payload={"continue_final_message": True, "separate_reasoning": False}
    )
    assert posted[0]["continue_final_message"] is True
    assert posted[0]["separate_reasoning"] is False
