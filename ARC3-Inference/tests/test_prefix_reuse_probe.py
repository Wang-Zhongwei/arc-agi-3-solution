"""The ARC3_PREFIX_REUSE_DIAG probe, and the template rendering it relies on (#3).

fixtures/qwen3.8-flash-next.chat_template.jinja is the served model's
chat_template.jinja (huggingface.co/Qwen/Qwen3.8-Flash-Next, identical in the
Intel W4A16 AutoRound checkpoint), unchanged.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import requests

from inference.agent import prefix_reuse
from inference.agent import tool_agent as tool_agent_module
from inference.agent.prefix_reuse import (
    PrefixReuseProbe,
    cache_outcome,
    render_history_message,
    render_reply,
)
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.agent.tool_agent import ToolAgent, _ChatCompletionResult

TEMPLATE = Path(__file__).parent / "fixtures" / "qwen3.8-flash-next.chat_template.jinja"
REASONING = "\nI look at the board first.\n\nThe player is at (3, 4).\n   \nSo I probe UP.\n"
NORMALISED = "I look at the board first.\nThe player is at (3, 4).\nSo I probe UP."
CODE = "\n  # probe\nx = 1\n\nprint(x)\n\n\n"


def _call(arguments: dict, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "python", "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def _render_with_template(messages: list[dict]) -> str:
    """Render as transformers' apply_chat_template does, which SGLang calls."""
    jinja2 = pytest.importorskip("jinja2")
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(TEMPLATE.read_text(encoding="utf-8"))
    return template.render(messages=messages, preserve_thinking=True, add_generation_prompt=False)


def _as_served(message: dict) -> dict:
    """The message as SGLang hands it to the template: arguments parsed to a dict."""
    served = json.loads(json.dumps(message))
    for call in served.get("tool_calls") or []:
        call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    return served


TEMPLATE_CASES = {
    "paragraphs and a call": {
        "role": "assistant", "reasoning_content": REASONING, "content": "",
        "tool_calls": [_call({"code": CODE})],
    },
    "content before the call": {
        "role": "assistant", "reasoning_content": REASONING, "content": "\nLet me check.\n\n",
        "tool_calls": [_call({"code": "print(1)"})],
    },
    "two calls": {
        "role": "assistant", "reasoning_content": "r", "content": None,
        "tool_calls": [_call({"code": "a"}), _call({"code": "b"}, "call_2")],
    },
    "no call": {"role": "assistant", "reasoning_content": REASONING, "content": "Answer.\n\nMore."},
    "non-string parameter": {
        "role": "assistant", "reasoning_content": "r", "content": "",
        "tool_calls": [_call({"code": "x", "repeat": 3, "flags": {"a": [1, 2]}})],
    },
}


@pytest.mark.parametrize("message", TEMPLATE_CASES.values(), ids=TEMPLATE_CASES.keys())
def test_render_assistant_turn_matches_the_chat_template(message):
    rendered = _render_with_template([{"role": "user", "content": "go"}, _as_served(message)])
    assert rendered.endswith("<|im_start|>assistant\n" + render_history_message(message) + "<|im_end|>\n")


def test_only_the_outer_trim_leaves_the_rendered_reply_unchanged():
    reply = {"role": "assistant", "reasoning_content": REASONING, "content": "",
             "tool_calls": [_call({"code": CODE})]}
    verbatim = {**reply, "reasoning_content": REASONING.strip()}
    normalised = {**reply, "reasoning_content": NORMALISED}
    assert render_history_message(verbatim) == render_reply(reply)
    assert render_history_message(normalised) != render_reply(reply)


def test_template_reads_reasoning_only_from_reasoning_content():
    reply = {"role": "assistant", "reasoning": "thought", "content": "",
             "tool_calls": [_call({"code": "a"})]}
    assert render_history_message(reply) != render_reply(reply)
    assert "thought" not in _render_with_template([{"role": "user", "content": "go"}, _as_served(reply)])


@pytest.mark.parametrize(
    ("cached", "outcome"),
    [(1152, "reply_end"), (1216, "reply_end"), (960, "prompt_end"), (1024, "partial"), (512, "below_prompt"),
     (None, "unknown")],
)
def test_cache_outcome_is_measured_in_pages(cached, outcome):
    # prompt 1000 -> its last full page ends at 960; reply 200 -> 1152
    assert cache_outcome(1000, 200, cached) == outcome


def _usage(prompt: int, output: int, cached: int | None = None) -> dict:
    usage = {"prompt_tokens": prompt, "completion_tokens": output}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return usage


def _exchange(stored_reasoning: str):
    """One request, its reply stored with `stored_reasoning`, and the next request."""
    probe = PrefixReuseProbe("g00")
    system, user = {"role": "system", "content": "s"}, {"role": "user", "content": "go"}
    first = probe.check([system, user], now=10.0)
    reply = {"role": "assistant", "reasoning_content": REASONING, "content": "",
             "tool_calls": [_call({"code": CODE})]}
    stored = {"role": "assistant", "reasoning_content": stored_reasoning,
              "tool_calls": json.loads(json.dumps(reply["tool_calls"]))}
    probe.note_reply(reply=reply, stored=stored, usage=_usage(1000, 200), now=11.0)
    tool = {"role": "tool", "tool_call_id": "call_1", "content": "ok"}
    return probe, first, [system, user, stored, tool]


def test_probe_reports_a_reused_reply():
    probe, first, messages = _exchange(REASONING.strip())
    assert first == {"history": "none"}
    fields = probe.check(messages, now=11.5)
    assert fields["history"] == "match"
    assert fields["prefix"] == "same"
    assert fields["gap_s"] == "0.50"
    line = probe.report(fields, _usage(1300, 50, cached=1152), action=3, request=2)[0]
    assert line.startswith("PREFIXDIAG g00 action=3 request=2 history=match prefix=same cache=reply_end")
    assert line.endswith("lost=0")


def test_probe_reports_where_a_normalised_reply_diverges():
    probe, _, messages = _exchange(NORMALISED)
    fields = probe.check(messages, now=11.5)
    assert fields["history"] == "mismatch"
    at = int(fields["diverge_at"].split("/")[0])
    # the first blank line the normaliser dropped
    assert at == len("<think>\n") + REASONING.strip().index("\n\n") + 1
    line = probe.report(fields, _usage(1300, 50, cached=960))[0]
    assert "cache=prompt_end" in line
    assert " lost=192 diverge_at=" in line


def test_probe_reports_a_stub_a_missing_reply_and_a_changed_prefix():
    probe, _, messages = _exchange(REASONING.strip())
    assert probe.check(messages[:2] + messages[3:], now=12.0)["history"] == "absent"

    probe, _, messages = _exchange(REASONING.strip())
    probe._reply.replaced = True
    assert probe.check(messages, now=12.0)["history"] == "replaced"

    probe, _, messages = _exchange(REASONING.strip())
    fields = probe.check([{"role": "system", "content": "other"}, *messages[1:]], now=12.0)
    assert fields["prefix"] == "changed"
    # a changed prefix explains the miss, so no reply tokens are charged to it
    assert probe.report(fields, _usage(1300, 50, cached=0))[0].endswith("lost=0")


def test_probe_checks_a_retried_request_against_the_same_reply():
    probe, _, messages = _exchange(REASONING.strip())
    # the request fails before a response, so nothing is reported
    probe.check(messages, now=12.0)
    retried = probe.check(messages[:1] + messages[2:], now=13.0)
    assert retried["history"] == "match"
    assert retried["prefix"] == "changed"
    assert retried["gap_s"] == "2.00"
    probe.report(retried, _usage(1250, 50, cached=0))
    # reported: the next request follows a reply not yet noted
    assert probe.check(messages, now=14.0) == {"history": "none"}


def test_probe_prints_running_totals(monkeypatch):
    monkeypatch.setattr(prefix_reuse, "_TOTALS", {})
    monkeypatch.setattr(prefix_reuse, "_TOTALS_EVERY", 2)
    probe, _, messages = _exchange(NORMALISED)
    assert len(probe.report(probe.check(messages, now=12.0), _usage(1300, 50, cached=960))) == 1
    lines = probe.report(probe.check(messages, now=13.0), _usage(1300, 50, cached=1216))
    assert lines[1] == (
        "PREFIXDIAG totals cache_first=1 cache_prompt_end=1 history_mismatch=1 history_none=1 "
        "lost_tokens=192 lost_tokens_mismatch=192 lost_tokens_none=0 mismatch_prompt_end=1 "
        "none_first=1 requests=2"
    )


# End to end through ToolAgent.


@pytest.fixture
def state_path(tmp_path):
    path = tmp_path / "artifacts" / "g_p0_tool_runtime_state.json"
    frame = Frame(grid=((0, 1), (2, 3)), step=0, level=1)
    write_runtime_state(path, current_frame=frame, history=[HistoryEntry(action="", frame=frame)])
    return path


@pytest.fixture
def agent(state_path, monkeypatch):
    monkeypatch.setenv("ARC3_REASONING_HISTORY_KEY", "reasoning_content")
    monkeypatch.setenv("ARC3_PREFIX_REUSE_DIAG", "1")
    monkeypatch.setattr(
        tool_agent_module, "run_sandboxed_python",
        lambda **kwargs: {"stdout": "", "result": None, "error": "", "action_results": [],
                          "keepable_functions": []},
    )
    instance = ToolAgent(model="test", base_url="http://127.0.0.1:9/v1", provider="vllm")
    instance._tool_steps = 2
    return instance


def _turn_log(agent, state_path, monkeypatch, caplog) -> list[str]:
    replies = [
        _ChatCompletionResult(
            message={"role": "assistant", "content": "", "reasoning_content": REASONING,
                     "tool_calls": [_call({"code": CODE})]},
            usage=_usage(1000, 200, cached=0),
        ),
        _ChatCompletionResult(
            message={"role": "assistant", "content": "", "reasoning_content": "done",
                     "tool_calls": [_call({"code": "print(1)"})]},
            usage=_usage(1300, 50, cached=1152),
        ),
    ]
    monkeypatch.setattr(agent, "_chat_completion", lambda messages, **kwargs: replies.pop(0))
    with caplog.at_level(logging.WARNING, logger=tool_agent_module.log.name):
        agent.analyze(state_path, action_num=1, valid_actions=["UP"], step_env=lambda arguments: {})
    return [record.getMessage() for record in caplog.records if record.getMessage().startswith("PREFIXDIAG g")]


@pytest.mark.parametrize(("verbatim", "history"), [("1", "match"), ("0", "mismatch")])
def test_tool_agent_logs_one_line_per_request(agent, state_path, monkeypatch, caplog, verbatim, history):
    monkeypatch.setenv("ARC3_VERBATIM_HISTORY", verbatim)
    lines = _turn_log(agent, state_path, monkeypatch, caplog)
    assert len(lines) == 2
    assert "history=none" in lines[0] and "cache=first" in lines[0]
    assert f"history={history} prefix=same cache=reply_end cached=1152" in lines[1]


def test_tool_agent_checks_a_context_overflow_retry_against_the_previous_reply(
    agent, state_path, monkeypatch, caplog
):
    monkeypatch.setenv("ARC3_VERBATIM_HISTORY", "1")
    replies = [
        _ChatCompletionResult(
            message={"role": "assistant", "content": "", "reasoning_content": text,
                     "tool_calls": [_call({"code": f"print({index})"}, f"call_{index}")]},
            usage=_usage(1000 + 300 * index, 100, cached=0),
        )
        for index, text in enumerate(["first", "second", "third"])
    ]
    replies.insert(2, requests.RequestException("This model's maximum context length is 8 tokens"))

    def chat_completion(messages, **kwargs):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(agent, "_chat_completion", chat_completion)
    with caplog.at_level(logging.WARNING, logger=tool_agent_module.log.name):
        # two one-request turns, then a turn whose request overflows and is
        # retried after the first turn's exchange is dropped
        for action_num, steps in ((1, 1), (2, 1), (3, 2)):
            agent._tool_steps = steps
            agent.analyze(state_path, action_num=action_num, valid_actions=["UP"],
                          step_env=lambda arguments: {})
    lines = [record.getMessage() for record in caplog.records if record.getMessage().startswith("PREFIXDIAG g")]
    assert not replies
    assert len(lines) == 3
    assert "history=match prefix=changed" in lines[2]
    assert "prev_prompt=1300 prev_output=100" in lines[2]


def test_tool_agent_logs_nothing_with_the_flag_off(agent, state_path, monkeypatch, caplog):
    monkeypatch.delenv("ARC3_PREFIX_REUSE_DIAG")
    assert _turn_log(agent, state_path, monkeypatch, caplog) == []


def test_a_failing_probe_does_not_end_the_turn(agent, state_path, monkeypatch, caplog):
    def broken(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(PrefixReuseProbe, "check", broken)
    _turn_log(agent, state_path, monkeypatch, caplog)
    assert "PREFIXDIAG check failed: boom" in caplog.text
    assert len([m for m in agent._history_messages if m.get("role") == "assistant"]) == 2
