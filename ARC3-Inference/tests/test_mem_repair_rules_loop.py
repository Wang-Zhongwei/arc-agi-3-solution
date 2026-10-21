"""arm/mem-repair-rules' three flags through the real game loop.

tests/anim_loop_sim.py (the real solver session, ToolAgent.analyze(), the
python sandbox, the Qwen3.8 chat template) with stall-fix-10x's env, the rule
scene and the legibility replies (ANIM_SIM_LEGIBILITY=1: a retained helper,
`twice(x)`), with ARC3_RULES_FILTER_V2, ARC3_RULES_ACTION_TAG and
ARC3_RETAINED_DELTA off and on, in the 98k and the 30k window. The scene's
rules all pass the v2 filter, so on, only two things change: each quoted rule
ends with its action, and a turn message whose retained-functions list equals
the last full list still in the context says so in one line. Flags off, every
request is still pf-keep-v2's (tests/test_mem_unified_loop.py).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mem_unified_golden import SCRIPT
from test_anim_loop import TIGHT, run_sim, texts

HERE = Path(__file__).resolve().parent
ARMS = json.loads((HERE.parents[1] / "kaggle" / "arms.json").read_text())["arms"]
FLAGS = {"ARC3_RULES_FILTER_V2": "1", "ARC3_RULES_ACTION_TAG": "1", "ARC3_RETAINED_DELTA": "1"}
FULL = ("Your retained functions from earlier Python calls in this game: twice(x). These are your previous "
        "definitions, available to call directly. Their presence does not mean they are correct; revise them if "
        "new evidence contradicts their assumptions.")
SHORT = "Your retained functions are unchanged (1 function)."
TAG = re.compile(r'" \(action (\d+)\)')


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mem-repair-rules")
    env = {**ARMS["stall-fix-10x"]["env"], **SCRIPT, "ANIM_SIM_LEGIBILITY": "1"}
    configs = {"off": env, "on": {**env, **FLAGS}, "off-tight": {**env, **TIGHT},
               "on-tight": {**env, **FLAGS, **TIGHT}}
    return {name: run_sim(tmp, name, config) for name, config in configs.items()}


def _text(message) -> str:
    return "\n".join(texts(message))


def _normal(text: str) -> str:
    return TAG.sub('"', text).replace(SHORT, FULL)


@pytest.mark.parametrize("window", ["", "-tight"])
def test_same_game_and_system(runs, window):
    off, on = runs["off" + window], runs["on" + window]
    assert on["levels_completed"] == off["levels_completed"] == 1
    assert on["engine_actions"] == off["engine_actions"]
    assert len(on["requests"]) == len(off["requests"])
    assert on["requests"][0]["messages"][0] == off["requests"][0]["messages"][0]


def test_only_the_tags_and_the_retained_line_change(runs):
    off, on = runs["off"], runs["on"]
    shortened = tagged = 0
    for a, b in zip(off["requests"], on["requests"]):
        assert a["evictions"] == b["evictions"] == 0
        assert len(a["messages"]) == len(b["messages"])
        for ma, mb in zip(a["messages"], b["messages"]):
            ta, tb = _text(ma), _text(mb)
            assert _normal(tb) == ta
            shortened += SHORT in tb
            tagged += bool(TAG.search(tb))
    assert shortened and tagged


@pytest.mark.parametrize("name", ["on", "on-tight"])
def test_each_request_extends_the_previous_between_trims(runs, name):
    reqs = runs[name]["requests"]
    for i in range(1, len(reqs)):
        a, b = reqs[i - 1], reqs[i]
        if b["evictions"] != a["evictions"]:
            continue
        assert b["prompt"].startswith(a["prompt"]), f"request {i}: prompt does not extend request {i - 1}"
        assert b["messages"][:len(a["messages"])] == a["messages"], f"request {i}: an earlier message changed"


@pytest.mark.parametrize("name", ["on", "on-tight"])
def test_unchanged_only_while_the_full_list_is_kept(runs, name):
    sent_full = 0
    for request in runs[name]["requests"]:
        newest = _text(request["messages"][-1])
        kept_full = any(FULL in _text(m) for m in request["messages"][1:-1])
        if SHORT in newest:
            assert kept_full
        if FULL in newest:
            assert not kept_full
            sent_full += 1
    # the 30k window trims the first full list away: the list is sent again
    assert sent_full == (2 if name == "on-tight" else 1)


def test_tags_are_actions_oldest_first(runs):
    lines = [line for request in runs["on"]["requests"] for line in _text(request["messages"][-1]).split("\n")
             if "Rules you confirmed" in line]
    assert lines
    for line in lines:
        quotes = line.count(" | ") + 1
        numbers = [int(n) for n in TAG.findall(line)]
        assert len(numbers) == quotes and numbers == sorted(numbers)
