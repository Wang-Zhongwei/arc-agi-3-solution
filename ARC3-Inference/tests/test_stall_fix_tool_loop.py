"""arm/stall-fix-tool's three flags through the real game loop.

tests/anim_loop_sim.py (the real solver session, ToolAgent.analyze(), the
python sandbox, the Qwen3.8 chat template) with mem-unified-10x's env and the
object scene, run with ARC3_ACTION_ERROR_RAISE, ARC3_ACTION_ALIASES and
ARC3_MEM_COLOURS_FIRST off and on. The scripted replies use only valid
action names, so the only change on is where `<this_level>` puts the
not-clicked colours row: first after the summary line. Flags off, every
request is still pf-keep-v2's (tests/test_mem_unified_loop.py).
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
FLAGS = {"ARC3_ACTION_ERROR_RAISE": "1", "ARC3_ACTION_ALIASES": "1", "ARC3_MEM_COLOURS_FIRST": "1"}
COLOURS = "Colours on the board not clicked on this level:"
BLOCK = re.compile(r"<this_level>\n.*?\n</this_level>", re.S)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stall-fix-tool")
    env = {**ARMS["mem-unified-10x"]["env"], **SCRIPT}
    return {"off": run_sim(tmp, "off", env), "on": run_sim(tmp, "on", {**env, **FLAGS})}


def _strip_colours(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if not line.startswith(COLOURS))


def _names(line: str) -> set[str]:
    body = re.sub(r" \(new on this level; [^)]*\)", "", line[len(COLOURS):]).strip().rstrip(".")
    return {name.strip() for name in body.split(",") if name.strip()}


def test_same_game_and_system(runs):
    off, on = runs["off"], runs["on"]
    assert on["levels_completed"] == off["levels_completed"]
    assert len(on["requests"]) == len(off["requests"])
    assert on["requests"][0]["messages"][0] == off["requests"][0]["messages"][0]


def test_only_the_colours_row_moves(runs):
    off, on = runs["off"], runs["on"]
    moved = 0
    for a, b in zip(off["requests"], on["requests"]):
        assert len(a["messages"]) == len(b["messages"])
        for ma, mb in zip(a["messages"], b["messages"]):
            ta, tb = "\n".join(texts(ma)), "\n".join(texts(mb))
            if ta == tb:
                continue
            assert _strip_colours(ta) == _strip_colours(tb)
            for block_a, block_b in zip(BLOCK.findall(ta), BLOCK.findall(tb)):
                old = [line for line in block_a.split("\n") if line.startswith(COLOURS)]
                new_lines = block_b.split("\n")
                if old:
                    assert new_lines[2].startswith(COLOURS)
                    assert _names(new_lines[2]) == _names(old[0])
                    moved += 1
    assert moved > 0  # the scene has clicks and colours, so the row is there to move


def test_every_colours_row_is_first(runs):
    seen = 0
    for request in runs["on"]["requests"]:
        text = "\n".join(texts(request["messages"][-1]))
        for block in BLOCK.findall(text):
            lines = block.split("\n")
            at = [i for i, line in enumerate(lines) if line.startswith(COLOURS)]
            if at:
                assert at == [2]
                seen += 1
    assert seen > 0
