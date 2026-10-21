"""Memory text the harness writes states facts, never advice or verdicts.

Every model-visible string a memory flag adds is checked here against words
that judge or instruct. The model's own notes are not harness text and are
not checked.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import memory_sim  # noqa: E402
from test_level_path import CLEAR, DEATH, Walk  # noqa: E402

ADVICE = re.compile(
    r"\b(inert|useless|impossible|pointless|waste|should|must|avoid|try|better|best|optimal|don't|instead)\b",
    re.I,
)


def _route_strings() -> list[str]:
    """ARC3_MEM_PATH: system and tool text, echo suffixes, turn lines, solved lines."""
    from inference.agent import tool_agent as ta
    from inference.agent.memory_facts import route_change_text, solved_head
    from inference.agent.prompts import MEMORY_SOLVED_ROUTE_LINE, memory_path_line

    out = [memory_path_line(4), memory_path_line(4, 8000, 5000), MEMORY_SOLVED_ROUTE_LINE, ta._LEVEL_PATH_TOOL_TEXT,
           solved_head(1, ["DOWN", "RIGHT"], 6, 0, winning=4)]
    walks = [
        Walk().go("RIGHT", "A").go("RIGHT", "B").go("DOWN", "C").go("UP", "B").reset().go("DOWN", "C"),
        Walk().go("RIGHT", "A").go("SPACE").go("UP", "X", DEATH).reset(manual=False),
        Walk().go("RIGHT", "A").go("RIGHT", "B").go("LEFT", "A").go("RIGHT", "D").go("UNDO", "E"),
        Walk().go("MOUSE(4,7)", "A").go("DOWN", "W", CLEAR).go("UP", "P"),
    ]
    for walk in walks:
        rows = walk.rec.route_rows
        out += [route_change_text(row) for row in rows]
        out += [walk.suffix(i) for i in range(len(rows))]
        out += [walk.rec.route_line(i) for i in range(len(rows))]
    return [text for text in out if text]


def test_route_strings_are_facts():
    strings = _route_strings()
    assert len(strings) >= 40
    bad = [text for text in strings if ADVICE.search(text)]
    assert not bad, bad


def test_route_lines_of_a_whole_game_are_facts(monkeypatch):
    env = memory_sim.scenario_env("kaggle-memv2-both")
    env["ARC3_MEM_PATH"] = "1"
    proc = subprocess.run(
        [sys.executable, "tests/memory_sim.py", "dump"], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=300,
        preexec_fn=lambda: os.nice(10),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    dump = json.loads(proc.stdout.strip().splitlines()[-1])
    lines = {
        line
        for request in dump["requests"] for message in request
        for line in message["text"].splitlines()
        if line.startswith("Shortest route on record") or "[route] route" in line
    }
    assert len(lines) >= 20
    bad = [line for line in lines if ADVICE.search(line)]
    assert not bad, bad


@pytest.mark.parametrize("word", ["should", "Instead", "best", "try", "don't"])
def test_the_check_catches_advice(word):
    assert ADVICE.search(f"the route {word} go left")
