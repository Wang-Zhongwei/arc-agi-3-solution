"""LOCAL_ANALYZER_THINKING_TOKEN_BUDGET: one request field, enforced by the server."""

import logging

import pytest

from inference.agent import tool_agent
from inference.agent.tool_agent import ToolAgent

BUDGET = 4096
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


class Response:
    status_code = 200

    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


def make_agent(monkeypatch, *, budget=BUDGET, mode="sglang", max_output=None, gets=None):
    monkeypatch.setattr(tool_agent, "_LOCAL_ANALYZER_THINKING_TOKEN_BUDGET", budget)
    monkeypatch.setattr(tool_agent, "_LOCAL_ANALYZER_ENABLE_THINKING", True)
    monkeypatch.setenv("ARC3_THINKING_BUDGET_MODE", mode)
    agent = object.__new__(ToolAgent)
    agent._model = type("M", (), {"provider": "vllm", "model_id": "m", "base_url": "http://server/v1"})()
    agent._max_output_tokens = max_output
    agent._timeout = 10
    agent._http_initial_grace_used = True
    agent._headers = lambda: {}
    agent._harness_template_kwargs = lambda: {}
    agent.posted, agent.got = [], []
    reply = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
    monkeypatch.setattr(
        tool_agent.requests, "post", lambda url, json, **kw: agent.posted.append(json) or Response(reply)
    )
    routes = gets or {"http://server/server_info": {"enable_strict_thinking": True}}

    def get(url, **kw):
        agent.got.append(url)
        return Response(routes[url])

    monkeypatch.setattr(tool_agent.requests, "get", get)
    return agent


def test_disabled_budget_adds_nothing(monkeypatch):
    agent = make_agent(monkeypatch, budget=0)
    agent._chat_completion(MESSAGES, tools=None)
    assert "custom_params" not in agent.posted[0]
    assert "thinking_token_budget" not in agent.posted[0]
    assert agent.got == []


def test_sglang_gets_custom_params(monkeypatch):
    agent = make_agent(monkeypatch, mode="sglang", max_output=12288)
    agent._chat_completion(MESSAGES, tools=None)
    assert agent.posted[0]["custom_params"] == {"thinking_budget": BUDGET}
    assert agent.posted[0]["max_tokens"] == 12288  # the output cap is unchanged
    assert "thinking_token_budget" not in agent.posted[0]


def test_vllm_gets_thinking_token_budget(monkeypatch):
    agent = make_agent(monkeypatch, mode="vllm")
    agent._chat_completion(MESSAGES, tools=None)
    assert agent.posted[0]["thinking_token_budget"] == BUDGET
    assert "custom_params" not in agent.posted[0]
    assert agent.got == []  # no strict-thinking check on vLLM


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": 8192},  # the rolling summary keeps its own limit
        {"enable_thinking": False},
    ],
)
def test_side_requests_are_not_budgeted(monkeypatch, kwargs):
    agent = make_agent(monkeypatch)
    agent._chat_completion(MESSAGES, tools=None, **kwargs)
    assert "custom_params" not in agent.posted[0]


def test_output_cap_within_budget_leaves_budget_off(monkeypatch):
    agent = make_agent(monkeypatch, max_output=BUDGET)
    agent._chat_completion(MESSAGES, tools=None)
    assert "custom_params" not in agent.posted[0]


@pytest.mark.parametrize(
    ("owned_by", "field"),
    [("vllm", "thinking_token_budget"), ("sglang", "custom_params"), ("someone", None)],
)
def test_auto_mode_reads_models_once(monkeypatch, owned_by, field):
    agent = make_agent(
        monkeypatch,
        mode="auto",
        gets={
            "http://server/v1/models": {"data": [{"id": "flashnext", "owned_by": owned_by}]},
            "http://server/server_info": {"enable_strict_thinking": True},
        },
    )
    agent._chat_completion(MESSAGES, tools=None)
    agent._chat_completion(MESSAGES, tools=None)
    for payload in agent.posted:
        sent = {key for key in ("thinking_token_budget", "custom_params") if key in payload}
        assert sent == ({field} if field else set())
    assert agent.got.count("http://server/v1/models") == 1


def test_auto_mode_retries_when_models_is_unreadable(monkeypatch):
    agent = make_agent(monkeypatch, mode="auto", gets={"http://server/v1/models": ValueError("down")})
    agent._chat_completion(MESSAGES, tools=None)
    agent._chat_completion(MESSAGES, tools=None)
    assert "custom_params" not in agent.posted[0]
    assert agent.got == ["http://server/v1/models"] * 2


def test_sglang_without_strict_thinking_is_reported(monkeypatch, caplog):
    agent = make_agent(monkeypatch, gets={"http://server/server_info": {"enable_strict_thinking": False}})
    with caplog.at_level(logging.ERROR):
        agent._chat_completion(MESSAGES, tools=None)
    assert "--enable-strict-thinking" in caplog.text
    assert agent.posted[0]["custom_params"] == {"thinking_budget": BUDGET}
