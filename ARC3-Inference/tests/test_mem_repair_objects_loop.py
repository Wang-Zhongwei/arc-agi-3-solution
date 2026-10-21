"""arm/mem-repair-objects' four flags through the real game loop.

tests/anim_loop_sim.py (the real solver session, ToolAgent.analyze(), the
python sandbox, the Qwen3.8 chat template) with stall-fix-10x's env and the
object scene, run with ARC3_MEM_CONTACT_ROW, ARC3_CLICK_MISS,
ARC3_MEM_CLICKED_ALSO and ARC3_EFFECT_MOVED_FIX off and on. The same game is
played; the system prompt is the same; in the turn messages only the
`<this_level>` rows (the contact row in place of the colours row, the
clicked-also row, the effect counts) and the click-miss lines change; every
request still extends the one before it between trims. Flags off, every
request is still stall-fix-10x's (tests/test_mem_unified_loop.py and
tests/test_stall_fix_tool_loop.py hold the goldens).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mem_unified_golden import SCRIPT
from test_anim_loop import run_sim, texts

HERE = Path(__file__).resolve().parent
ARMS = json.loads((HERE.parents[1] / "kaggle" / "arms.json").read_text())["arms"]
FLAGS = {"ARC3_MEM_CONTACT_ROW": "1", "ARC3_CLICK_MISS": "1", "ARC3_MEM_CLICKED_ALSO": "1",
         "ARC3_EFFECT_MOVED_FIX": "1"}
COLOURS = "Colours on the board not clicked on this level:"
CONTACT = "Not touched yet on this level (never clicked, entered or bumped)"
BLOCK = re.compile(r"<this_level>\n.*?\n</this_level>", re.S)
KEY_ROW = re.compile(r"^(UP|DOWN|LEFT|RIGHT|SPACE) \(\d+ press")
CLICK_ROW = re.compile(r"^Clicks on .* cells \(")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mem-repair-objects")
    env = {**ARMS["stall-fix-10x"]["env"], **SCRIPT}
    return {"off": run_sim(tmp, "off", env), "on": run_sim(tmp, "on", {**env, **FLAGS})}


def _changeable(line: str) -> bool:
    return (line.startswith((COLOURS, CONTACT, "Also clicked on this level:"))
            or bool(KEY_ROW.match(line)) or bool(CLICK_ROW.match(line))
            or (line.startswith("Action ") and " hit background (" in line))


def _strip(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if not _changeable(line))


def test_same_game_and_system(runs):
    off, on = runs["off"], runs["on"]
    assert on["levels_completed"] == off["levels_completed"]
    assert len(on["requests"]) == len(off["requests"])
    assert on["requests"][0]["messages"][0] == off["requests"][0]["messages"][0]


def test_only_the_object_rows_change(runs):
    off, on = runs["off"], runs["on"]
    contact = 0
    for a, b in zip(off["requests"], on["requests"]):
        assert len(a["messages"]) == len(b["messages"])
        for ma, mb in zip(a["messages"], b["messages"]):
            ta, tb = "\n".join(texts(ma)), "\n".join(texts(mb))
            if ta != tb:
                assert _strip(ta) == _strip(tb)
            for block in BLOCK.findall(tb):
                lines = block.split("\n")
                assert not any(line.startswith(COLOURS) for line in lines)
                at = [i for i, line in enumerate(lines) if line.startswith(CONTACT)]
                assert at in ([], [2])
                contact += bool(at)
    assert contact > 0  # the scene has objects, so the row is there


def test_requests_extend_the_one_before_between_trims(runs):
    for name in ("off", "on"):
        requests = runs[name]["requests"]
        broken = 0
        for before, after in zip(requests, requests[1:]):
            m0, m1 = before["messages"], after["messages"]
            if len(m1) < len(m0) or m1[: len(m0)] != m0:
                broken += 1
        # the flags add no new prefix breaks: trims break it as often in both runs
        if name == "off":
            off_broken = broken
    assert broken == off_broken
