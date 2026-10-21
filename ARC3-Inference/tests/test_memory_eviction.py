"""Memory text must never give a game's GPU slot away or break its prefix unpriced (#58).

A game hands its slot over only when ``_note_history_evicted`` fires, which is
meant to happen only when already-sent messages were really dropped. Each config
below plays the scripted game of ``tests/eviction_sim.py`` through the real
``ToolAgent.analyze()`` - multi-request turns, yields and resume openers, a
context-length rejection and a rolled-back turn included - and checks that

  1. every handover follows a real drop,
  2. every prefix break has a handover,
  3. no two consecutive requests are both full re-prefills (none with the flags
     off either, so this is "no more than with the flags off"),
  4. no request holds two ``<memory>`` blocks or two copies of a pinned message.

The ``neg-*`` configs install the traps #58 measured (a delta stripped from older
openers, counted or not; Fable's content-deduped pin on top of ledger-on-trim)
and must fail, which shows the checks can see them. The memory-v2 placements are
skipped until that arm's flags exist on the branch under test.

Each config runs in its own interpreter, at low priority, because several knobs
are read at import time; each takes about a second.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import eviction_sim  # noqa: E402

AGENT_SOURCE = "".join(p.read_text(encoding="utf-8") for p in sorted((ROOT / "inference" / "agent").glob("*.py")))
GUARDED = [name for name, cfg in eviction_sim.CONFIGS.items() if not cfg.get("expect_fail")]
TRAPS = [name for name, cfg in eviction_sim.CONFIGS.items() if cfg.get("expect_fail")]
NO_VIOLATIONS = {"1": [], "2": [], "3": [], "4": []}


def _lower_priority() -> None:
    try:
        os.nice(19)
    except OSError:
        pass


def _run(name: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "tests/eviction_sim.py", name],
        cwd=ROOT, env=eviction_sim.config_env(name), capture_output=True, text=True,
        timeout=300, preexec_fn=_lower_priority,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("name", GUARDED)
def test_only_real_drops_hand_the_slot_over(name):
    needs = eviction_sim.CONFIGS[name].get("needs")
    if needs and needs not in AGENT_SOURCE:
        pytest.skip(f"{needs} is not implemented on this branch")
    got = _run(name)
    assert got["violations"] == NO_VIOLATIONS, got
    # A run that never reaches these paths would pass without proving anything.
    assert got["handovers"] >= 5
    assert got["later_in_turn_requests"] >= 4
    assert got["later_in_turn_handovers"], "the force-drain retry never handed over"
    assert got["resumes"] >= 2
    assert got["rolled_back_requests"]
    # the route configs reach their own placements; the others send no route text
    if eviction_sim.CONFIGS[name]["env"].get("ARC3_MEM_PATH") == "1":
        assert got["route_turn_lines"] >= 10 and got["route_echo_lines"] >= 10, got
    else:
        assert got["route_turn_lines"] == got["route_echo_lines"] == 0, got


@pytest.mark.parametrize("name", TRAPS)
def test_the_checks_catch_each_known_trap(name):
    got = _run(name)
    for number in sorted(eviction_sim.CONFIGS[name]["expect_fail"]):
        assert got["violations"][str(number)], f"assertion {number} missed the {name} trap: {got}"
