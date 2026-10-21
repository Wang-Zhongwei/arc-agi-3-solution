"""ARC3_PROMPT_HYGIENE with the memory-main memory sections (arm/pf-mem).

ARC3_PROMPT_HYGIENE moves the system prompt's memory section from between
the Python guidance bullets to after the last of them. With memory-main's
flags the section is no longer only the ledger's: it may be memory-v2's
(ARC3_MEM_SOLVED and friends), the memory layout's (ARC3_MEM_EVERY_CALL)
or the route line alone (ARC3_MEM_PATH). Whichever it is, hygiene moves the
whole section, unchanged; without hygiene it stays where memory-main put it.
Everything else is exactly ARC3_PROMPT_HYGIENE's wording fix.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from inference.agent import prompts
from inference.agent import tool_agent as ta
from notebook_env import notebook_env

ROOT = Path(__file__).resolve().parents[1]
ARMS = json.loads((ROOT.parent / "kaggle" / "arms.json").read_text(encoding="utf-8"))["arms"]
TAIL_START = prompts.PYTHON_ADDENDUM_TAIL.splitlines()[0]
TOOL_SESSION_HEADER = "\n\nTool session rules:\n"

CONFIGS = {
    "ledger": ("user-combo-10x", {}),
    "solved": ("mem-solved-10x", {}),
    "keep": ("mem-recap-keep-10x", {}),
    "path": ("user-combo-10x", {"ARC3_LEDGER_ON_TRIM": "0", "ARC3_MEM_PATH": "1"}),
}


def _render(monkeypatch, arm, extra, hygiene):
    for key in list(os.environ):
        if key.startswith(("ARC3_", "LOCAL_ANALYZER_", "MULTIMODAL_", "EXPOSE_")):
            monkeypatch.delenv(key)
    env = {**notebook_env(), **ARMS[arm]["env"], **extra, "ARC3_PROMPT_HYGIENE": "1" if hygiene else "0"}
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return ta._build_system_prompt(tool_output_tokens=3072), ta._memory_system_section()


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_hygiene_moves_the_whole_memory_section(monkeypatch, name):
    arm, extra = CONFIGS[name]
    off, section = _render(monkeypatch, arm, extra, hygiene=False)
    on, section_on = _render(monkeypatch, arm, extra, hygiene=True)
    assert section and section == section_on
    # off: memory-main's place, ahead of the Python guidance tail
    assert off.count(section) == 1 and off.index(section) < off.index(TAIL_START)
    # on: after the last Python guidance bullet, right before the tool rules
    assert on.count(section) == 1
    assert on.index(section) > on.index(TAIL_START)
    assert on.index(section) > on.index("`frame_diff(before=None")
    assert on.split(section, 1)[1].startswith(TOOL_SESSION_HEADER)
    # nothing else moves: the rest is hygiene's wording fix of the rest
    assert on.replace(section, "") == prompts.apply_prompt_hygiene(
        off.replace(section, ""), persistent_functions=ta._persistent_functions())


def test_memory_section_kinds(monkeypatch):
    assert _render(monkeypatch, "user-combo-10x", {}, False)[1] == prompts.LEDGER_ON_TRIM_ADDENDUM
    solved = _render(monkeypatch, "mem-solved-10x", {}, False)[1]
    keep = _render(monkeypatch, "mem-recap-keep-10x", {}, False)[1]
    assert "`<solved>` lines" in solved and "`<previous_levels>`" in keep
    assert _render(monkeypatch, "user-combo-10x", {"ARC3_LEDGER_ON_TRIM": "0"}, True)[1] == ""
