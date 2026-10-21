"""arm/mem-repair-facts's three flags through the real game loop.

tests/anim_loop_sim.py (the real solver session, ToolAgent.analyze(), the
python sandbox, the Qwen3.8 chat template) with stall-fix-10x's env and the
object scene, in a 26k window, run with ARC3_MOVED_PERSIST,
ARC3_CLEARED_BOARDS and ARC3_THIS_LEVEL_AFTER_TRIM off and on. In that
window the trim at request 5 (a reply that executed no actions) drops every
turn message that held `<this_level>`. The sim's level-up action returns the
next level's board alone, so the clear line is the "never shown" one; the
scene has no corner frames, so ARC3_MOVED_PERSIST adds nothing. Flags off,
every request is pf-keep-v2's (tests/test_mem_unified_loop.py).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mem_unified_golden import SCRIPT
from test_anim_loop import check_static_system, run_sim, texts

HERE = Path(__file__).resolve().parent
ARMS = json.loads((HERE.parents[1] / "kaggle" / "arms.json").read_text())["arms"]
FLAGS = {"ARC3_MOVED_PERSIST": "1", "ARC3_CLEARED_BOARDS": "1", "ARC3_THIS_LEVEL_AFTER_TRIM": "1"}
WINDOW = {"LOCAL_ANALYZER_CONTEXT_WINDOW": "26000"}
BLOCK = re.compile(r"\n\n<this_level>\nLevel (\d+) so far.*?\n</this_level>", re.S)
STATE = re.compile(r"Current state: step \d+, level (\d+)")
CLEAR = ("Level 1 was cleared by action 20, LEFT. The board it produced was never shown; history's last "
         "level-1 frame (step 19) is one action short of the clear.")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mem-repair-facts")
    env = {**ARMS["stall-fix-10x"]["env"], **SCRIPT, **WINDOW}
    return {"off": run_sim(tmp, "off", env), "on": run_sim(tmp, "on", {**env, **FLAGS})}


def _text(message) -> str:
    return "\n".join(texts(message))


def _levels(request) -> tuple[int | None, set[int]]:
    """The current level (newest `Current state` line) and the levels of the
    `<this_level>` blocks in the request's user messages."""
    users = [_text(m) for m in request["messages"] if m["role"] == "user"]
    current = next((int(m.group(1)) for t in reversed(users) for m in [STATE.search(t)] if m), None)
    return current, {int(level) for t in users for level in BLOCK.findall(t)}


def test_same_game_and_system(runs):
    off, on = runs["off"], runs["on"]
    assert on["levels_completed"] == off["levels_completed"] == 1
    assert on["engine_actions"] == off["engine_actions"]
    assert len(on["requests"]) == len(off["requests"])
    assert check_static_system(on) == []
    assert on["requests"][0]["messages"][0] == off["requests"][0]["messages"][0]


def test_a_trim_no_longer_leaves_the_level_without_this_level(runs):
    off, on = runs["off"], runs["on"]
    missing_off = [i for i, r in enumerate(off["requests"]) if _levels(r)[0] not in _levels(r)[1]
                   and _levels(r)[0] is not None]
    missing_on = [i for i, r in enumerate(on["requests"]) if _levels(r)[0] not in _levels(r)[1]
                  and _levels(r)[0] is not None]
    assert missing_off and off["requests"][missing_off[0]]["messages"][-1]["role"] == "user"
    assert missing_on == []
    added = on["requests"][missing_off[0]]["messages"][-1]
    assert _text(added).startswith("Your last python snippet executed NO actions")
    assert texts(added)[-1].startswith("\n\n<this_level>\nLevel 1 so far")


def test_only_the_facts_differ(runs):
    off, on = runs["off"], runs["on"]
    clear_seen = added = 0
    for i, (a, b) in enumerate(zip(off["requests"], on["requests"])):
        assert len(a["messages"]) == len(b["messages"]), i
        for ma, mb in zip(a["messages"], b["messages"]):
            pa, pb = texts(ma), texts(mb)
            if pa == pb:
                continue
            stripped = [t.replace(f"\n{CLEAR}", "").replace(f"  {CLEAR}\n", "") for t in pb]
            clear_seen += stripped != pb
            if stripped != pa:
                # the only other change: a <this_level> part on a message that had none
                assert not any(BLOCK.search(t) for t in pa), i
                assert stripped[:-1] == pa and BLOCK.fullmatch(stripped[-1]), i
                added += 1
    assert clear_seen > 0 and added > 0


def test_the_clear_line_follows_the_level_start_text_and_rides_in_the_entry(runs):
    on = runs["on"]
    newest = [_text(r["messages"][-1]) for r in on["requests"] if r["messages"][-1]["role"] == "user"]
    turn = [t for t in newest if CLEAR in t and "<cleared_level>" not in t.split(CLEAR)[0]]
    assert turn and all(t.index("You have completed the previous level.") < t.index(CLEAR)
                        < t.index("Current state:") for t in turn)
    every = "\n".join(_text(m) for r in on["requests"] for m in r["messages"] if m["role"] == "user")
    entries = re.findall(r"<(?:cleared_level|previous_levels)>\n(Level 1: .*?)\n</", every, re.S)
    assert entries and all(entry.split("\n")[1] == f"  {CLEAR}" for entry in entries)


def test_each_request_extends_the_previous_between_trims(runs):
    reqs = runs["on"]["requests"]
    for i in range(1, len(reqs)):
        a, b = reqs[i - 1], reqs[i]
        if b["evictions"] != a["evictions"]:
            continue
        assert b["prompt"].startswith(a["prompt"]), f"request {i}: prompt does not extend request {i - 1}"
        assert b["messages"][:len(a["messages"])] == a["messages"], f"request {i}: an earlier message changed"
