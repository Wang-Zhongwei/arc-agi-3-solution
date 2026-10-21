"""ARC3_BUDGET_TRUTH through the real analyze() loop (tests/budget_loop_sim.py).

The sim runs in a subprocess under the Kaggle notebook's own setup_env, read
from kaggle/template.ipynb, so import-time knobs match a real run. The python
tool is the real one; only the HTTP post and the environment are faked.
BUDGET_SIM_BASELINE_ROOT=<path to another checkout's ARC3-Inference> also
compares flags-off requests with that checkout's (e.g. main) byte for byte.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from notebook_env import notebook_env

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLOCK_KEYS = ("run_elapsed_seconds", "time_remaining_seconds")
REMOVED = ("do not get distracted by it", "DON'T DO THIS!", "HUD elements such as countdown bars",
           "HUD/timer state", "ephemeral")
BAR_LINE = ("- In many games, a long horizontal or vertical line near an edge is a step bar. It "
            "usually counts down a per-level action budget: it shrinks as actions are taken, not "
            "with wall-clock time.\n")


def run_sim(tmp_path: Path, name: str, extra: dict[str, str], root: Path = ROOT) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ARC3_", "LOCAL_ANALYZER_"))}
    env.update(notebook_env())
    env.update(extra)
    env["PYTHONPATH"] = str(root)
    out = tmp_path / f"{name}.json"
    subprocess.run([sys.executable, str(HERE / "budget_loop_sim.py"), str(out), "--inference-root", str(root)],
                   env=env, check=True, cwd=root, timeout=300)
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("budget")
    return {"on": run_sim(tmp, "on", {"ARC3_BUDGET_TRUTH": "1"}), "off": run_sim(tmp, "off", {})}


def clock_strings(run: dict) -> list[str]:
    """Every way the run clock could show up in a request."""
    return list(CLOCK_KEYS) + [repr(v) for v in run["clock_values"]]


def test_game_plays_through(runs):
    for run in runs.values():
        assert run["won"], "the scripted game must finish under every config"
        assert run["actions"] == 7
        # a turn with an inspect and an act request, and the 1-request yield turn
        assert run["requests_per_turn"] == [2, 2, 2, 1, 2]


def test_flags_off_shows_the_clock(runs):
    """Proves the sim carries the clock to the model, so the flag-on check can fail."""
    off = runs["off"]
    tool_texts = off["requests"][-1]["tool_texts"]
    joined = "\n".join(tool_texts)
    for key in CLOCK_KEYS:
        assert key in joined
    # every route: auto-printed result (turn 0), last_action_call_result
    # inspection (turn 1), printed batch result (turn 1), printed single
    # result (turn 2), the persisted result after the turn (turn 3)
    assert sum("time_remaining_seconds" in text for text in tool_texts) == 6
    first_clock = repr(off["clock_values"][0])
    assert first_clock in joined
    assert "time_remaining_seconds" in off["requests"][-1]["prompt"]


def test_flag_on_hides_the_clock_everywhere(runs):
    on = runs["on"]
    needles = clock_strings(on)
    for request in on["requests"]:
        for needle in needles:
            assert needle not in request["prompt"], (request["turn"], request["index"], needle)
    assert len(on["requests"][-1]["tool_texts"]) == 8, "every tool result reached a request"


def test_flag_on_keeps_the_clock_for_scheduling(runs):
    """The clock still sets the agent's elapsed time (endgame phase, etc.)."""
    for run in runs.values():
        assert run["game_elapsed_seconds"] == run["clock_values"][-2]


def test_tool_results_otherwise_unchanged(runs):
    """On differs from off only by the clock entries in each tool result."""
    clock = re.compile(r", '(?:run_elapsed_seconds|time_remaining_seconds)': [0-9.]+"
                       r"|'(?:run_elapsed_seconds|time_remaining_seconds)', ")
    on_texts = runs["on"]["requests"][-1]["tool_texts"]
    off_texts = runs["off"]["requests"][-1]["tool_texts"]
    assert len(on_texts) == len(off_texts)
    for on_text, off_text in zip(on_texts, off_texts):
        assert clock.sub("", off_text) == on_text


def test_flag_on_bar_wording(runs):
    for request in runs["on"]["requests"]:
        assert BAR_LINE in request["system"]
        for phrase in REMOVED:
            assert phrase not in request["prompt"], phrase
    off_system = runs["off"]["requests"][0]["system"]
    assert BAR_LINE not in off_system
    for phrase in REMOVED[:-1]:
        assert phrase in off_system
    off_prompt = runs["off"]["requests"][0]["prompt"]
    assert off_prompt.count("ephemeral") == 3  # system line + tool description + code parameter
    assert len(runs["on"]["requests"][0]["system"]) < len(off_system)


@pytest.mark.parametrize("config", ["on", "off"])
def test_system_prompt_and_tools_static(runs, config):
    requests = runs[config]["requests"]
    assert len({r["system"] for r in requests}) == 1
    assert len({json.dumps(r["tools"], sort_keys=True) for r in requests}) == 1


@pytest.mark.parametrize("config", ["on", "off"])
def test_each_prompt_extends_the_previous_one(runs, config):
    """The prefix-cache property: request k+1's prompt starts with request k's
    whole prompt, then the reasoning the model generated."""
    requests = runs[config]["requests"]
    for before, after in zip(requests, requests[1:]):
        assert after["prompt"].startswith(before["prompt"]), (config, before["turn"], before["index"])
        rest = after["prompt"][len(before["prompt"]):]
        assert rest.startswith(before["generated_reasoning"].strip()) or rest.startswith(
            before["generated_reasoning"]), (config, rest[:80])


def test_flags_off_matches_baseline_checkout(runs, tmp_path):
    baseline = os.environ.get("BUDGET_SIM_BASELINE_ROOT", "")
    if not baseline:
        pytest.skip("set BUDGET_SIM_BASELINE_ROOT to compare with another checkout")
    base = run_sim(tmp_path, "baseline", {}, root=Path(baseline))
    strip = lambda run: [{k: r[k] for k in ("prompt", "system", "tools", "last_message")}
                         for r in run["requests"]]
    assert strip(base) == strip(runs["off"])
    assert base["clock_values"] == runs["off"]["clock_values"]
