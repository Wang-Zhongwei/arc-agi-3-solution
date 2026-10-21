"""EXPOSE_RESET: the zero-action RESET guard, its prompt text, and the loop.

Unit tests cover the counter (inference.agent.runtime_state.actions_since_reset)
and the refusal inside the sandboxed ``action(...)``. The loop tests run
tests/reset_loop_sim.py - the real solver session, ToolAgent and sandbox over a
fake server and a small game that restarts the WHOLE game on a zero-action
RESET, as the ARC API documents - in a subprocess under the Kaggle notebook's
own setup_env (kaggle/template.ipynb), so import-time knobs match a real run.

RESET_SIM_BASELINE_ROOT=<another checkout's ARC3-Inference> (e.g. main, from
``git archive origin/main ARC3-Inference``) also checks that with EXPOSE_RESET
unset every request body matches that checkout's byte for byte.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from inference.agent import tool_agent as ta
from inference.agent.prompts import RESET_INFO_ADDENDUM, UNDO_COST_ADDENDUM, UNDO_INFO_ADDENDUM
from inference.agent.runtime_state import Frame, HistoryEntry, actions_since_reset, write_runtime_state

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEMPLATE_NB = ROOT.parent / "kaggle" / "template.ipynb"
GRID = ((0, 0), (0, 0))


def entry(action: str, level: int = 1, **result) -> HistoryEntry:
    name = {"UP": "ACTION1", "RIGHT": "ACTION4", "UNDO": "ACTION7"}.get(action, action)
    payload = {"action_name": name, **result} if action else {}
    return HistoryEntry(action=action, frame=Frame(grid=GRID, step=0, level=level), result=payload)


SEED = entry("")


# --- the counter -------------------------------------------------------------

def test_counter_is_zero_at_game_start():
    assert actions_since_reset([]) == 0
    assert actions_since_reset([SEED]) == 0


def test_counter_counts_every_non_reset_action_including_undo():
    assert actions_since_reset([SEED, entry("RIGHT")]) == 1
    assert actions_since_reset([SEED, entry("RIGHT"), entry("UNDO"), entry("UP")]) == 3


def test_counter_restarts_at_a_model_reset():
    history = [SEED, entry("RIGHT"), entry("RIGHT"), entry("RESET")]
    assert actions_since_reset(history) == 0
    assert actions_since_reset(history + [entry("UP")]) == 1


def test_counter_restarts_at_the_automatic_reset_after_game_over():
    history = [SEED, entry("RIGHT"), entry("UP", game_over=True), entry("RESET", automatic=True)]
    assert actions_since_reset(history) == 0
    assert actions_since_reset(history + [entry("RIGHT")]) == 1


def test_counter_restarts_at_a_level_change():
    completed = entry("RIGHT", level=2, level_completed=True)
    assert actions_since_reset([SEED, entry("RIGHT"), completed]) == 0
    assert actions_since_reset([SEED, entry("RIGHT"), completed, entry("UP", level=2)]) == 1
    # a level change with no flag (an older history) is still a boundary
    unflagged = [SEED, entry("RIGHT"), entry("RIGHT", level=2), entry("UP", level=2)]
    assert actions_since_reset(unflagged) == 1


def test_counter_reads_entries_without_results():
    bare = [HistoryEntry(action=a, frame=Frame(grid=GRID, step=0, level=1))
            for a in ("", "RIGHT", "RESET", "UP", "UP")]
    assert actions_since_reset(bare) == 2
    assert actions_since_reset(bare[:3]) == 0


# --- the refusal inside action(...) ------------------------------------------

def run_snippet(tmp_path, monkeypatch, history, code, *, exposed=True):
    """Run one python tool call against a fake step_env; return (payload, engine calls)."""
    if exposed:
        monkeypatch.setenv("EXPOSE_RESET", "on")
    else:
        monkeypatch.delenv("EXPOSE_RESET", raising=False)
    monkeypatch.setenv("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:9/v1")
    state = tmp_path / "state.json"
    write_runtime_state(state, current_frame=history[-1].frame, history=history)
    agent = ta.ToolAgent(model="test-model")
    agent._ensure_session(state)
    calls = []

    def step_env(arguments):
        if arguments.get("query"):
            return {"executed": False, "query": arguments["query"], "record": None}
        calls.append([a["action"] for a in arguments["actions"]])
        return {"executed": True, "executed_count": len(arguments["actions"]), "action_num": len(history),
                "level": 1, "score": 0, "state": "NOT_FINISHED", "valid_actions": ["RESET", "RIGHT"],
                "board_changed": True, "gameplay_changed": True,
                "action_display": arguments["actions"][-1]["action"]}

    agent._step_env_callback = step_env
    agent._current_valid_actions = ["RESET", "RIGHT"]
    result = agent._run_python_tool(state, {"code": code})
    return json.loads(result.content), calls, result, agent


def _rendered(payload) -> str:
    return json.dumps(payload)


@pytest.mark.parametrize("history", [
    [SEED],
    [SEED, entry("RIGHT"), entry("RESET")],
    [SEED, entry("RIGHT"), entry("UP", game_over=True), entry("RESET", automatic=True)],
    [SEED, entry("RIGHT"), entry("RIGHT", level=2, level_completed=True)],
], ids=["game-start", "after-reset", "after-game-over", "after-level-change"])
def test_reset_refused_with_no_action_since_reset(tmp_path, monkeypatch, history):
    payload, calls, result, agent = run_snippet(tmp_path, monkeypatch, history, "action(['RESET'])")
    assert calls == [], "the RESET must not reach the engine"
    assert "RESET was refused and NOT executed" in _rendered(payload)
    assert not result.step_executed
    assert agent._last_action_call_result["stop_reason"] == "reset_no_actions"
    assert agent._last_action_call_result["executed"] is False


@pytest.mark.parametrize("history", [
    [SEED, entry("RIGHT")],
    [SEED, entry("RIGHT"), entry("RESET"), entry("UNDO")],
    [SEED, entry("UP", game_over=True), entry("RESET", automatic=True), entry("RIGHT")],
    [SEED, entry("RIGHT", level=2, level_completed=True), entry("UP", level=2)],
], ids=["one-action", "undo-counts", "after-game-over", "after-level-change"])
def test_reset_allowed_after_an_action(tmp_path, monkeypatch, history):
    payload, calls, result, _ = run_snippet(tmp_path, monkeypatch, history, "action(['RESET'])")
    assert calls == [["RESET"]]
    assert result.step_executed
    assert "refused" not in _rendered(payload)


def test_reset_not_first_still_wins_inside_a_snippet(tmp_path, monkeypatch):
    payload, calls, _, _ = run_snippet(tmp_path, monkeypatch, [SEED], "action(['RIGHT', 'RESET'])")
    assert calls == []
    assert "RESET must be the first executed game action" in _rendered(payload)


def test_guard_is_off_without_expose_reset(tmp_path, monkeypatch):
    # legacy handling: no RESET guards at all, the solver decides
    payload, calls, _, _ = run_snippet(tmp_path, monkeypatch, [SEED], "action(['RESET'])", exposed=False)
    assert calls == [["RESET"]]
    assert "refused" not in _rendered(payload)


# --- the system prompt -------------------------------------------------------

def test_prompt_states_refusal_and_undo_cost_only_when_exposed(monkeypatch):
    monkeypatch.setenv("ARC3_ACTION_INFO", "1")
    monkeypatch.setenv("EXPOSE_UNDO", "on")
    monkeypatch.delenv("EXPOSE_RESET", raising=False)
    off = ta._build_system_prompt(tool_output_tokens=3072)
    monkeypatch.setenv("EXPOSE_RESET", "on")
    on = ta._build_system_prompt(tool_output_tokens=3072)
    assert UNDO_INFO_ADDENDUM in off and RESET_INFO_ADDENDUM not in off
    assert UNDO_COST_ADDENDUM.strip() not in off
    assert RESET_INFO_ADDENDUM in on
    assert "RESET is refused, and spends nothing, when no action has been executed" in on
    assert UNDO_INFO_ADDENDUM.rstrip("\n") + UNDO_COST_ADDENDUM in on


# --- the loop ----------------------------------------------------------------

def notebook_env() -> dict[str, str]:
    """setup_env exactly as the Kaggle template builds it (priority scheduling on)."""
    cells = json.loads(TEMPLATE_NB.read_text())["cells"]
    source = next("".join(c["source"]) for c in cells if "setup_env = {" in "".join(c["source"]))
    start = source.index("setup_env = {")
    end = source.index("os.environ.update({k: str(v) for k,v in setup_env.items()})")
    namespace = {"SERVER_BASE_URL": "http://127.0.0.1:9/v1", "SERVED_MODEL_NAME": "flashnext",
                 "USE_PRIORITY_SCHEDULING": True}
    exec(source[start:end], namespace)  # noqa: S102 - our own committed template
    return {k: str(v) for k, v in namespace["setup_env"].items()}


def run_sim(tmp_path: Path, name: str, extra: dict[str, str], root: Path = ROOT) -> dict:
    drop = ("ARC3_", "LOCAL_ANALYZER_", "EXPOSE_", "ONLY_RESET_LEVELS")
    env = {k: v for k, v in os.environ.items() if not k.startswith(drop)}
    env.update(notebook_env())
    env.update(extra)
    env["PYTHONPATH"] = str(root)
    out = tmp_path / f"{name}.json"
    subprocess.run([sys.executable, str(HERE / "reset_loop_sim.py"), str(out), "--inference-root", str(root)],
                   env=env, check=True, cwd=root, timeout=300)
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("reset-sim")
    return {"on": run_sim(tmp, "on", {"EXPOSE_RESET": "on"}), "off": run_sim(tmp, "off", {})}


def _text(message) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if part.get("type") == "text")
    return str(content or "")


def test_loop_reset_round_trip(runs):
    run = runs["on"]
    assert run["won"] and run["script_left"] == 0
    # one model RESET and one automatic RESET reached the engine; the four
    # zero-action RESETs and the not-first one never did, so the game never
    # restarted from level 1
    assert run["engine_actions"].count("RESET") == 2
    assert run["full_resets"] == 0 and run["level_resets"] == 2
    assert run["history_actions"][:5] == ["", "RIGHT", "RESET", "UP", "RESET"]
    requests = run["requests"]
    system = _text(requests[0]["body"]["messages"][0])
    assert RESET_INFO_ADDENDUM in system
    assert UNDO_INFO_ADDENDUM.rstrip("\n") + UNDO_COST_ADDENDUM in system
    first_user = _text(requests[0]["body"]["messages"][1])
    assert "RESET" in first_user.split("Valid actions", 1)[1].split("\n", 1)[0]
    # every zero-action RESET came back as a refusal in the following request
    # requests 1, 5, 7 and 10 send a zero-action RESET (4 sends the allowed
    # one): each refusal is the tool result the next request carries
    refused = [r["index"] for r in requests
               if "RESET was refused and NOT executed" in _text(r["body"]["messages"][-1])]
    assert refused == [2, 6, 8, 11]
    assert [requests[i - 1]["code"] for i in refused] == ["action(['RESET'])"] * 4
    # the deliberate RESET is named in the next turn's opener
    assert requests[5]["engine_actions"][-1] == "RESET"
    assert "You deliberately reset the current level." in _text(requests[5]["body"]["messages"][-1])


@pytest.mark.parametrize("config", ["on", "off"])
def test_loop_each_prompt_extends_the_previous_one(runs, config):
    """The prefix-cache property: request k+1's prompt starts with request k's."""
    requests = runs[config]["requests"]
    # flags off, the unguarded RESETs restart the game, so the script runs
    # out before the win and one more request ends the game
    assert len(requests) == (12 if config == "on" else 13)
    for before, after in zip(requests, requests[1:]):
        assert after["prompt"].startswith(before["prompt"]), (config, before["index"])


def test_loop_flags_off_is_legacy(runs):
    run = runs["off"]
    system = _text(run["requests"][0]["body"]["messages"][0])
    assert RESET_INFO_ADDENDUM not in system and UNDO_COST_ADDENDUM.strip() not in system
    assert "RESET" not in _text(run["requests"][0]["body"]["messages"][1]).split("Valid actions", 1)[1].split("\n", 1)[0]
    assert all("RESET was refused" not in json.dumps(r["body"]) for r in run["requests"])
    # unguarded, the first zero-action RESET reaches the engine, which under
    # the documented rule restarts the whole game
    assert run["full_resets"] >= 1


def test_loop_flags_off_matches_baseline(runs, tmp_path):
    baseline = os.environ.get("RESET_SIM_BASELINE_ROOT")
    if not baseline:
        pytest.skip("set RESET_SIM_BASELINE_ROOT to another checkout's ARC3-Inference")
    other = run_sim(tmp_path, "baseline", {}, root=Path(baseline))
    ours = runs["off"]
    assert [r["body"] for r in other["requests"]] == [r["body"] for r in ours["requests"]]
    assert [r["prompt"] for r in other["requests"]] == [r["prompt"] for r in ours["requests"]]
    assert other["engine_actions"] == ours["engine_actions"]
    clock = re.compile(r"\b\d\d:\d\d:\d\d\b")  # the transcript's wall-clock stamps
    assert clock.sub("T", other["transcript"]) == clock.sub("T", ours["transcript"])
