import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import arcengine
import pytest

from inference.agent import tool_agent
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.runtime_state import (
    Frame,
    frame_to_payload,
    load_runtime_state,
    write_runtime_state,
)
from inference.framework import solver


@pytest.fixture
def session(tmp_path, monkeypatch):
    """Use the real harness action/serialization path with a small deterministic game."""
    monkeypatch.setattr(solver, "ANIMATION_ENABLED", False)
    game = SimpleNamespace(number_of_levels=2, game_run=SimpleNamespace(history=[]))

    def state(cell, terminal=arcengine.GameState.NOT_FINISHED):
        return SimpleNamespace(
            frame=SimpleNamespace(data=[[cell]]), levels_completed=0, won=False,
            just_won_level=False, raw=SimpleNamespace(state=terminal),
            available_actions=[1, 2, 3, 4, 5],
        )

    game.current_state = state(0)

    def execute(action, **kwargs):
        game.game_run.history.append(action)
        game.current_state = state(
            0 if action.id == arcengine.GameAction.RESET else len(game.game_run.history),
            arcengine.GameState.GAME_OVER if action.id == arcengine.GameAction.ACTION5
            else arcengine.GameState.NOT_FINISHED,
        )
        return game.current_state

    game.execute_action = execute
    session = solver._HarnessGameSession(
        solver=SimpleNamespace(max_runtime_s_per_game=None), game=game, analyzer=None,
        game_index=0, pass_index=0, state_path=tmp_path / "runtime.json",
        transcript_path=tmp_path / "transcript.txt", analysis_html_relpath="",
        stop_event=threading.Event(), viewer_data_path=tmp_path / "viewer.json",
    )
    monkeypatch.setattr(session, "should_stop", lambda: False)
    monkeypatch.setattr(session, "_append_action_viewer_event", Mock())
    monkeypatch.setattr(session, "write_viewer_payload", Mock())
    session.seed_initial_history()
    session.write_runtime_state()
    return session


def sandbox_state(session, call_result=None):
    current, history = load_runtime_state(session.state_path)
    return {
        "current_frame": tool_agent._ascii_frame_view_payload(current),
        "history": tool_agent._ascii_history_view_payload(history),
        "last_action_call_result": call_result or {},
    }


def run(session, code, *, call_result=None):
    def action_handler(actions, **kwargs):
        result = session.step_env({"actions": actions, **kwargs})
        return {"action_result": result, "state": sandbox_state(session, result)}

    response = run_sandboxed_python(
        code=code, timeout_seconds=10, initial_state=sandbox_state(session, call_result),
        action_handler=action_handler,
    )
    assert response["error"] == "", response
    return response["result"]


def test_batch_results_belong_to_each_transition_and_survive_inspection(session):
    call_result = run(session, """
assert len(history) == 1 and not transitions and last_transition is None
call_result = action(['UP', 'RIGHT'])
assert call_result == last_action_call_result
assert [t.action for t in transitions] == ['UP', 'RIGHT']
assert [t.result['action_num'] for t in transitions] == [1, 2]
assert [t.result['action_name'] for t in transitions] == ['ACTION1', 'ACTION4']
assert all(t.result['automatic'] is False for t in transitions)
assert all('requested_count' not in t.result for t in transitions)
assert all('animation_chain' not in t.result for t in transitions)
assert [(t.before_frame.step, t.after_frame.step) for t in transitions] == [(0, 1), (1, 2)]
assert all(t.frame is t.after_frame for t in transitions)
assert last_transition is transitions[-1]
assert previous_frame is last_transition.before_frame
assert all(not hasattr(h, 'result') for h in history)
assert all(not hasattr(f, 'result') for f in [current_frame, previous_frame, latest_frame, last_action_frame])
assert not hasattr(current_frame[:, :], 'result')
result = call_result
""")
    assert call_result["executed_count"] == call_result["requested_count"] == 2
    assert run(session, """
assert [t.result['action_num'] for t in transitions] == [1, 2]
assert last_action_call_result['executed_count'] == 2
assert not hasattr(history[-1], 'result')
result = last_transition.result['action_name']
""", call_result=call_result) == "ACTION4"
    assert all(not hasattr(h, "result") for h in session.history_entries)
    assert all(not hasattr(h, "result") for h in
               tool_agent._to_ascii_history_views(session.history_entries))


def test_refused_action_preserves_transition_results(session):
    session.step_env({"action": "UP"})
    session.action_guard_hook = lambda *args: ("known_noop", "already proven inert")
    assert run(session, """
before = last_transition.result.copy()
try:
    action(['RIGHT'])
except KnownNoOpActionError:
    pass
assert len(transitions) == 1 and len(history) == 2
assert last_transition.result == before
assert not hasattr(history[-1], 'result')
assert last_action_call_result['executed_count'] == 0
result = last_action_call_result['stop_reason']
""") == "known_noop"


def test_automatic_reset_has_own_result_and_preserves_fatal_call(session):
    fatal = session.step_env({"actions": [{"action": "UP"}, {"action": "SPACE"}, {"action": "RIGHT"}]})
    assert fatal["executed_count"] == 2 and fatal["skipped_actions"] == ["RIGHT"]
    session._execute_auto_reset()
    assert run(session, """
assert [t.action for t in transitions] == ['UP', 'SPACE', 'RESET']
assert [t.result['automatic'] for t in transitions] == [False, False, True]
assert transitions[-2].result['game_over'] is True
assert last_transition.result['game_over'] is False
assert current_frame.step == 3 and previous_frame.step == 2
assert all(not hasattr(h, 'result') for h in history)
assert last_action_call_result['fatal_action'] == 'SPACE'
result = last_action_call_result['stop_reason']
""", call_result=fatal) == "game_over"


@pytest.mark.parametrize("key", ["transition_result", "result"])
def test_saved_snapshot_results_roundtrip_without_public_history_result(tmp_path, key):
    frame = Frame(grid=((1,),), step=1, level=1)
    outcome = {"executed": True, "action_num": 1, "automatic": False}
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({
        "current_frame": frame_to_payload(frame),
        "history": [{"action": "UP", "frame": frame_to_payload(frame), key: outcome}],
    }))
    current, history = load_runtime_state(path)
    assert not hasattr(history[0], "result")
    write_runtime_state(path, current_frame=current, history=history)
    raw = json.loads(path.read_text())["history"][0]
    assert "result" not in raw and raw["transition_result"] == outcome
    response = run_sandboxed_python(
        code="assert not hasattr(history[0], 'result')\nresult = last_transition.result",
        timeout_seconds=10, initial_state={
            "current_frame": tool_agent._ascii_frame_view_payload(current),
            "history": tool_agent._ascii_history_view_payload(history),
        }, action_handler=Mock(side_effect=AssertionError("read only")),
    )
    assert response["error"] == "" and response["result"] == outcome


def test_sandbox_reads_legacy_results_without_exposing_them_on_history():
    frame = tool_agent._ascii_frame_view_payload(Frame(grid=((1,),), step=1, level=1))
    response = run_sandboxed_python(
        code="assert not hasattr(history[0], 'result')\nresult = last_transition.result",
        timeout_seconds=10, initial_state={"current_frame": frame, "history": [
            None, {"action": "UP", "frame": frame, "result": {"action_num": 1}},
        ]}, action_handler=Mock(side_effect=AssertionError("read only")),
    )
    assert response["error"] == "" and response["result"] == {"action_num": 1}


def test_hud_reads_transition_result_after_state_reload(session, monkeypatch):
    session.step_env({"action": "RIGHT"})
    current, history = load_runtime_state(session.state_path)
    monkeypatch.setenv("ARC3_HUD_DISPLAY", "true")
    monkeypatch.setenv("ARC3_PERSISTENT_FUNCTIONS", "true")
    agent = tool_agent.ToolAgent(model="test-transition-hud")
    agent._kept_functions = {"hud": """def hud():
    return last_transition.result['action_name'], hasattr(history[-1], 'result')
"""}
    assert json.loads(agent._hud_display_context(current, history).split(": ", 1)[1]) == ["ACTION4", False]
