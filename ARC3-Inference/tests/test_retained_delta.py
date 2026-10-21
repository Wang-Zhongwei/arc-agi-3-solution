"""ARC3_RETAINED_DELTA (arm/mem-repair-rules): the retained-functions list.

stall-fix-10x's turn messages: the list was 34% of a turn message's text
(median 117 tokens, p90 464, max 2,048), identical to the previous turn
message's 69% of the time, and long default values were much of it (m0r0's
`blkplan(..., FREE=((0, 2), (0, 7), ...))`). The game loop is in
tests/test_mem_repair_rules_loop.py.
"""
from __future__ import annotations

import pytest

from inference.agent.tool_agent import (
    ToolAgent, _default_summary, _retained_list_unchanged, _short_function_signature, _with_retained_delta,
)

M0R0_BLKPLAN = (
    "def blkplan(L0, R0, walls, FREE=((0, 2), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (1, 1), (1, 2)), "
    "P='..######.\\n.###.####\\n###...###\\n##.....##', NM='U:UP,D:DOWN', k=3):\n    return L0\n"
)
FULL = ("Your retained functions from earlier Python calls in this game: crop(f, r0, r1, c0, c1), tipinfo(f). "
        "These are your previous definitions, available to call directly. Their presence does not mean they are "
        "correct; revise them if new evidence contradicts their assumptions.")
OTHER = FULL.replace("tipinfo(f)", "tipinfo(f), build(f)")


def _user(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {}}]}


def _tool(text: str) -> dict:
    return {"role": "tool", "content": text}


def test_long_defaults_become_a_type_summary():
    assert _short_function_signature(M0R0_BLKPLAN) == (
        "blkplan(L0, R0, walls, FREE=<tuple, 8 items>, P=<str, 39 chars>, NM='U:UP,D:DOWN', k=3)")
    assert _short_function_signature("def crop(f, r0, r1, c0, c1):\n    pass\n") == "crop(f, r0, r1, c0, c1)"
    assert _short_function_signature("def g(a, *, b={'k': 1, 'kkkkkkkkkkkkkkkkkkkkkkkkkkk': 2}):\n    pass\n") == (
        "g(a, *, b=<dict, 2 items>)")


@pytest.mark.parametrize("literal, summary", [
    ("[1, 2, 3]", "<list, 3 items>"), ("'abc'", "<str, 3 chars>"), ("12345", "<int>"), ("{1, 2}", "<set, 2 items>"),
])
def test_default_summary(literal, summary):
    import ast

    assert _default_summary(ast.parse(literal, mode="eval").body) == summary


@pytest.mark.parametrize("retained, unchanged", [
    ([_user("turn 1\n\n" + FULL)], True),
    # later "unchanged" lines and tool results do not hide the full list
    ([_user(FULL), _tool('{"stdout": "x"}'), _user("Your retained functions are unchanged (2 functions).")], True),
    ([_user(OTHER)], False),
    # the newest list counts, not any identical older one
    ([_user(FULL), _user(OTHER)], False),
    # a clear notice after the list
    ([_user(FULL), _tool('{"function_retention": "Your retained functions were cleared: boom."}')], False),
    ([_user(FULL), _tool('{"function_retention": "Your previous function tipinfo is no longer retained; '
                         'redefine it if needed."}')], False),
    ([_user("Your retained functions were cleared at this level transition.\n" + FULL)], True),
    ([_tool("no list")], False),
    ([], False),
])
def test_unchanged_against_the_newest_list_kept(retained, unchanged):
    assert _retained_list_unchanged(FULL, retained) is unchanged


def test_the_turn_message_gets_the_unchanged_line():
    message = _user("Valid actions right now: UP.\n\n" + FULL + "\n\nCurrent grid image:")
    short = _with_retained_delta(message, FULL, 2)
    assert short["content"][0]["text"] == ("Valid actions right now: UP.\n\nYour retained functions are unchanged "
                                           "(2 functions).\n\nCurrent grid image:")
    assert short["content"][1] is message["content"][1] and message["content"][0]["text"].count(FULL) == 1
    assert _with_retained_delta({"role": "user", "content": FULL}, FULL, 1)["content"] == (
        "Your retained functions are unchanged (1 function).")
    assert _with_retained_delta(_user("no list"), FULL, 2) == _user("no list")


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setenv("ARC3_PERSISTENT_FUNCTIONS", "1")
    monkeypatch.setenv("ARC3_PERSISTENT_FUNCTIONS_SCOPE", "game")
    instance = ToolAgent(model="test", base_url="http://127.0.0.1:9/v1", provider="vllm")
    instance._kept_functions = {"blkplan": M0R0_BLKPLAN}
    return instance


def test_flag_off_the_list_is_as_before(agent, monkeypatch):
    monkeypatch.delenv("ARC3_RETAINED_DELTA", raising=False)
    text = agent._retained_function_context()
    assert "FREE=((0, 2), (0, 7)," in text and agent._retained_sentence is None


def test_flag_on_the_list_is_shortened_and_remembered(agent, monkeypatch):
    monkeypatch.setenv("ARC3_RETAINED_DELTA", "1")
    text = agent._retained_function_context()
    sentence, count = agent._retained_sentence
    assert count == 1 and sentence in text and "FREE=<tuple, 8 items>" in sentence
    agent._kept_functions = {}
    assert agent._retained_function_context() == "" and agent._retained_sentence is None
