"""With every memory-v2 flag off, the model is sent exactly what stack-v2 sent;
with memory-v2 on and every later flag off, exactly what arm/memory-v2 sent.

The first golden hashes were produced on arm/stack-v2 by ``tests/memory_sim.py``:
the system prompt and every request of a scripted 31-turn game (trims, deaths,
level changes), under the default env, the Kaggle env, and the Kaggle env with
the ledger (and death ledger) on. The second (``memory_v2_on_golden.json``) ran
this branch's ``memory_sim.py`` on arm/memory-v2 under the memory-v2 arms' env,
and adds the sha of the tool schema sent with each request. Each scenario runs
in its own interpreter because several knobs are read at import time.
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

import memory_sim  # noqa: E402

GOLDEN = json.loads((ROOT / "tests" / "fixtures" / "memory_flags_off_golden.json").read_text())
V2_GOLDEN = json.loads((ROOT / "tests" / "fixtures" / "memory_v2_on_golden.json").read_text())


def _run_scenario(name: str, flags=memory_sim.MEMORY_V2_FLAGS) -> dict:
    env = memory_sim.scenario_env(name)
    for flag in flags:
        env.pop(flag, None)
    proc = subprocess.run(
        [sys.executable, "tests/memory_sim.py", name],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("name", sorted(memory_sim.SCENARIOS))
def test_requests_match_stack_v2(name):
    got = _run_scenario(name)
    want = GOLDEN["scenarios"][name]
    assert got["system_prompt"] == want["system_prompt"]
    assert got["request_count"] == want["request_count"]
    assert got["trims"] == want["trims"]
    assert got["ledger_render"] == want["ledger_render"]
    mismatched = [i for i, (a, b) in enumerate(zip(got["requests"], want["requests"])) if a != b]
    assert not mismatched, f"first differing request: {mismatched[0]}"


@pytest.mark.parametrize("name", sorted(memory_sim.MEMORY_V2_ON_SCENARIOS))
def test_memory_v2_requests_match_memory_v2_with_later_flags_off(name):
    got = _run_scenario(name, flags=memory_sim.MEMORY_V3_FLAGS)
    want = V2_GOLDEN["scenarios"][name]
    assert got["system_prompt"] == want["system_prompt"]
    assert got["tools_sha"] == want["tools_sha"] and len(got["tools_sha"]) == 1
    assert got["request_count"] == want["request_count"]
    assert got["trims"] == want["trims"]
    assert got["ledger_render"] == want["ledger_render"]
    mismatched = [i for i, (a, b) in enumerate(zip(got["requests"], want["requests"])) if a != b]
    assert not mismatched, f"first differing request: {mismatched[0]}"


def test_replayed_transcripts_give_the_same_ledger_blocks():
    got = memory_sim.replay_ledger_blocks(GOLDEN["replay"]["run"])
    if got is None:
        pytest.skip("fz-ledger-on-trim-r2 transcripts are not on this machine")
    assert got == GOLDEN["replay"]["render_hashes"]


def test_scenarios_exercise_trims_and_the_ledger():
    # a golden that never trims or never shows the ledger would prove nothing
    assert GOLDEN["scenarios"]["kaggle-ledger"]["trims"] >= 3
    assert GOLDEN["scenarios"]["kaggle-ledger"]["ledger_render"] is not None
    assert os.path.exists(ROOT / "tests" / "memory_sim.py")
