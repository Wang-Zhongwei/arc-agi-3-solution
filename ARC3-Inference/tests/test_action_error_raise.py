"""ARC3_ACTION_ERROR_RAISE: a call the host refuses as invalid input raises (arm/stall-fix-tool).

Run U, m0r0 L3: a BFS executor called action(['L']); the host answered
{'executed': False, 'error': "Unknown action at index 1: 'L'"}, nothing
raised or printed, and the model read its own verifier's MISMATCH as a game
rule for ~73 minutes. With the flag on the solver tags such refusals with
stop_reason "invalid_input" and the sandbox echoes them and raises
InvalidActionError (a BaseException, like the other refusals). Off, every
payload and every snippet behaves as before.
"""
from __future__ import annotations

import json

import pytest

from inference.agent.python_tool_sandbox import run_sandboxed_python

from test_animation_layers import _harness, _python

VALID = ["RESET", "UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE"]


@pytest.fixture
def raise_on(monkeypatch):
    monkeypatch.setenv("ARC3_ACTION_ERROR_RAISE", "1")
    monkeypatch.delenv("ARC3_ACTION_ALIASES", raising=False)
    return monkeypatch


@pytest.fixture
def raise_off(monkeypatch):
    monkeypatch.delenv("ARC3_ACTION_ERROR_RAISE", raising=False)
    monkeypatch.delenv("ARC3_ACTION_ALIASES", raising=False)
    return monkeypatch


# ---------------------------------------------------------------- the solver's payloads

def test_flag_off_payloads_are_unchanged(tmp_path, raise_off):
    session, _ = _harness(tmp_path, raise_off)
    raw = session.step_env({"actions": [{"action": "L"}]})
    assert raw["executed"] is False
    assert raw["error"] == "Unknown action at index 1: 'L'"
    assert "stop_reason" not in raw
    raw = session.step_env({"actions": [{"action": "MOUSE", "row": None, "col": 3}]})
    assert raw["error"] == "MOUSE action at index 1 requires integer row and col arguments."
    assert "stop_reason" not in raw


@pytest.mark.parametrize("actions, error", [
    ([{"action": "L"}], "Unknown action at index 1: 'L'"),
    ([{"action": "UP"}, {"action": "Q"}], "Unknown action at index 2: 'Q'"),
    ([{"action": "MOUSE", "row": "x", "col": 3}],
     "MOUSE action at index 1 requires integer row and col arguments."),
    ([], "`actions` must contain at least one action."),
])
def test_invalid_input_is_tagged(tmp_path, raise_on, actions, error):
    session, _ = _harness(tmp_path, raise_on)
    raw = session.step_env({"actions": actions})
    assert raw["executed"] is False
    assert raw["error"] == error
    assert raw["stop_reason"] == "invalid_input"
    assert session.action_count == 0


def test_unavailable_first_action_is_tagged_but_a_partial_batch_is_not(tmp_path, raise_on):
    session, _ = _harness(tmp_path, raise_on)
    game = session.game
    step = game.execute_action

    def keyboard_only(*args, **kwargs):
        state = step(*args, **kwargs)
        state.available_actions = [1, 2, 3, 4]
        return state

    game.execute_action = keyboard_only
    game.current_state.available_actions = [1, 2, 3, 4]  # no SPACE, no MOUSE
    raw = session.step_env({"actions": [{"action": "SPACE"}]})
    assert raw["error"] == "SPACE is not valid right now."
    assert raw["stop_reason"] == "invalid_input"
    # after an executed action the batch stops as before: executed, invalid_action
    raw = session.step_env({"actions": [{"action": "RIGHT"}, {"action": "SPACE"}]})
    assert raw["executed"] is True
    assert raw["stop_reason"] == "invalid_action"


def test_engine_errors_and_terminal_payloads_keep_their_shape(tmp_path, raise_on):
    session, _ = _harness(tmp_path, raise_on)

    def boom(*args, **kwargs):
        raise KeyError("engine")

    session.game.execute_action = boom
    raw = session.step_env({"actions": [{"action": "UP"}]})
    # the step may have happened, so it is not called invalid input
    assert raw["executed"] is False and raw["error"].startswith("KeyError")
    assert "stop_reason" not in raw


# ---------------------------------------------------------------- the sandbox

def _handler(result, calls=None):
    def handler(actions, stale_after=None):
        if calls is not None:
            calls.append(list(actions))
        return {"action_result": dict(result), "state": {}}

    return handler


REFUSAL = {
    "executed": False, "valid_actions": VALID, "board_changed": False,
    "error": "Unknown action at index 1: 'L'", "stop_reason": "invalid_input",
}


def _run(code, result, *, flag=True, calls=None):
    return run_sandboxed_python(
        code=code, timeout_seconds=5, initial_state={},
        action_handler=_handler(result, calls), action_error_raise=flag,
    )


def test_refusal_echoes_and_raises_through_except_exception():
    calls = []
    out = _run(
        "try:\n    action(['L'])\nexcept Exception:\n    print('swallowed')\nprint('after')\n",
        REFUSAL, calls=calls,
    )
    assert out["stdout"] == "[action] NOT EXECUTED: Unknown action at index 1: 'L'\n"
    assert out["error"].splitlines()[-1] == (
        "InvalidActionError: Unknown action at index 1: 'L'. Nothing in this call was executed. "
        "Valid action names: RESET, UP, DOWN, LEFT, RIGHT, SPACE, MOUSE."
    )
    assert len(calls) == 1


def test_invalid_action_error_is_exposed_only_with_the_flag():
    code = "print('InvalidActionError' in dir())"
    assert _run(code, REFUSAL, flag=True)["stdout"] == "True\n"
    assert _run(code, REFUSAL, flag=False)["stdout"] == "False\n"
    # catchable by name for a snippet that wants to
    out = _run(
        "try:\n    action(['L'])\nexcept InvalidActionError as e:\n    print('caught', str(e)[:14])\n",
        REFUSAL,
    )
    assert out["error"] == ""
    assert out["stdout"].splitlines()[-1] == "caught Unknown action"


def test_untagged_refusal_stays_silent():
    # flags off the solver never tags it: the old payload goes back as it was
    old = {key: value for key, value in REFUSAL.items() if key != "stop_reason"}
    out = _run("r = action(['L'])\nprint(r['executed'], r['error'])\n", old)
    assert out["error"] == ""
    assert out["stdout"] == "False Unknown action at index 1: 'L'\n"


@pytest.mark.parametrize("stop_reason, exc", [
    ("known_noop", "KnownNoOpActionError"),
    ("known_death", "KnownDeathActionError"),
    ("repeated_action_in_state", "RepeatedActionInStateError"),
    ("reset_not_first", "RuntimeError"),
])
def test_guard_refusals_keep_their_own_exceptions(stop_reason, exc):
    result = {"executed": False, "error": "detail", "stop_reason": stop_reason, "stop_detail": "detail"}
    out = _run("action(['UP'])\n", result)
    assert out["error"].splitlines()[-1].startswith(exc + ":")
    assert "NOT EXECUTED" not in out["stdout"]


def test_terminal_payload_is_not_an_invalid_input():
    result = {"executed": False, "error": "No action was executed because the current game state is "
              "terminal or stopping.", "stop_reason": "game_over", "game_over": True}
    out = _run("r = action(['UP'])\nprint(r['stop_reason'])\n", result)
    assert out["error"] == ""
    assert out["stdout"] == "game_over\n"


# ---------------------------------------------------------------- end to end

def test_m0r0_shape_end_to_end(tmp_path, raise_on):
    # the U m0r0 L3 executor: a click that runs, then a BFS letter
    harness = _harness(tmp_path, raise_on)
    session, agent = harness
    code = (
        "action([{'action': 'MOUSE', 'row': 6, 'col': 6}])\n"
        "for step in ['L', 'L', 'U']:\n"
        "    try:\n"
        "        r = action([step])\n"
        "    except Exception as e:\n"
        "        print('err', e)\n"
        "print('done')\n"
    )
    dispatch = agent._run_python_tool(session.state_path, {"code": code})
    payload = json.loads(dispatch.content)
    assert payload["stdout"] == "[action] NOT EXECUTED: Unknown action at index 1: 'L'\n"
    assert "InvalidActionError: Unknown action at index 1: 'L'. Nothing in this call was executed." in payload["error"]
    # the click ran and is the only action on record; the turn's summary says so
    assert session.action_count == 1
    assert agent._last_step_summary["executed_count"] == 1
    assert agent._last_step_summary["executed_actions"] == ["MOUSE(row=6, col=6)"]


def test_flag_off_end_to_end_is_silent(tmp_path, raise_off):
    harness = _harness(tmp_path, raise_off)
    out = _python(harness, "r = action(['L'])\nprint(r['executed'])\n")
    assert out == "False\n"


# ---------------------------------------------------------------- the next turn message

def test_turn_message_after_a_refused_snippet_says_it_tried(tmp_path, raise_on):
    from test_animation_layers import _opener

    harness = _harness(tmp_path, raise_on)
    _python(harness, "action(['RIGHT'])")
    assert "The code executed 1 action in the previous sequence." in _opener(harness)
    _python(harness, "action(['L'])")
    opener = _opener(harness)
    assert ("Your last python snippet executed NO actions: its action() call was refused as invalid "
            "input (Unknown action at index 1: 'L'), so nothing in that call ran.") in opener
    assert "it only computed or inspected state" not in opener
    assert "For reference, the most recent executed sequence earlier this turn was: RIGHT." in opener
    # a compute-only snippet afterwards is described as before
    _python(harness, "print(1)")
    opener = _opener(harness)
    assert "Your last python snippet executed NO actions (it only computed or inspected state" in opener
    assert "refused" not in opener


def test_turn_message_after_a_refused_snippet_flag_off(tmp_path, raise_off):
    from test_animation_layers import _opener

    harness = _harness(tmp_path, raise_off)
    _python(harness, "action(['RIGHT'])")
    _python(harness, "action(['L'])")
    assert "Your last python snippet executed NO actions (it only computed or inspected state" in _opener(harness)
