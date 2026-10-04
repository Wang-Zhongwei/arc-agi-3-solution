"""ARC3_GAME_SOFT_STOP_SECONDS: a game's last turn finishes instead of being cut."""

import threading
import time
from types import SimpleNamespace

import pytest

from inference.agent import tool_agent
from inference.agent.tool_agent import ToolAgent
from inference.framework import solver as solver_module
from inference.framework.solver import _HarnessGameSession

CAP = 100.0
GRACE = 30.0


def make_session(monkeypatch, tmp_path, *, elapsed, results, on_call=None):
    """A session ``elapsed`` seconds into a CAP-second game whose analyzer
    returns ``results`` in order; ``on_call(session, n)`` runs before each."""
    monkeypatch.setattr(solver_module, "_is_run_complete", lambda game: False)
    monkeypatch.setattr(solver_module, "_is_engine_game_over", lambda game: False)
    monkeypatch.setattr(solver_module, "_engine_action_names", lambda game: [])
    monkeypatch.setattr(solver_module, "_analyzer_reported_tokens", lambda analyzer: 0)
    session = object.__new__(_HarnessGameSession)
    session.game = SimpleNamespace(
        game_run=SimpleNamespace(state="playing", history=[], game_id="g", solver_note=None, final_score=0),
        finish_game=lambda: None,
    )
    session.solver = SimpleNamespace(
        max_runtime_s_per_game=CAP,
        max_actions_per_game=None,
        max_generated_tokens_per_game=None,
        _claim_warmup_slot=lambda: False,
        soft_time_remaining_seconds=lambda: None,
    )
    session.stop_event = threading.Event()
    session.started_at = time.monotonic() - elapsed
    session.analysis_step = 0
    session.last_engine_action = None
    session.state_path = tmp_path / "state.json"
    session.transcript_path = tmp_path / "transcript.txt"
    session.analyzer = SimpleNamespace(_timeout=900)
    for name in ("write_runtime_state", "write_viewer_payload", "_write_analysis_html"):
        setattr(session, name, lambda: None)
    session._read_transcript_bytes = lambda: b""
    session._transcript_delta_since = lambda before: ""
    session._append_analysis_viewer_event = lambda step, delta: None
    session.calls = []
    pending = list(results)

    def analyze(*args, should_stop, **kwargs):
        session.calls.append(should_stop())
        if on_call is not None:
            on_call(session, len(session.calls))
        return pending.pop(0)

    session.analyzer.analyze = analyze
    return session


def turn(*, yielded=False, executed=True):
    return SimpleNamespace(retryable_failure=False, yielded_control=yielded, step_executed=executed)


def age(session, seconds):
    session.started_at -= seconds


@pytest.fixture
def grace(monkeypatch):
    monkeypatch.setenv("ARC3_GAME_SOFT_STOP_SECONDS", str(GRACE))


def test_no_new_turn_inside_grace(grace, monkeypatch, tmp_path):
    session = make_session(monkeypatch, tmp_path, elapsed=CAP - GRACE + 5, results=[turn()])
    session._play_inner()
    assert session.calls == []


def test_turn_started_before_grace_finishes_its_thinking(grace, monkeypatch, tmp_path):
    # The first invocation runs into the grace and yields before acting; the
    # continuation still runs, the agent's own stop check stays open, and the
    # game ends once the turn has acted.
    session = make_session(
        monkeypatch, tmp_path, elapsed=CAP - GRACE - 5,
        results=[turn(yielded=True, executed=False), turn()],
        on_call=lambda s, n: age(s, 15) if n == 1 else None,
    )
    session._play_inner()
    assert session.calls == [False, False]


def test_turn_that_already_acted_does_not_start_another_in_grace(grace, monkeypatch, tmp_path):
    session = make_session(
        monkeypatch, tmp_path, elapsed=CAP - GRACE - 5,
        results=[turn(yielded=True, executed=True), turn()],
        on_call=lambda s, n: age(s, 15),
    )
    session._play_inner()
    assert len(session.calls) == 1


def test_hard_cap_still_stops_a_continuation(grace, monkeypatch, tmp_path):
    session = make_session(
        monkeypatch, tmp_path, elapsed=CAP - GRACE - 5,
        results=[turn(yielded=True, executed=False), turn()],
        on_call=lambda s, n: age(s, GRACE + 10),
    )
    session._play_inner()
    assert session.calls == [False]


def test_without_grace_the_cap_is_where_turns_stop(monkeypatch, tmp_path):
    monkeypatch.delenv("ARC3_GAME_SOFT_STOP_SECONDS", raising=False)
    session = make_session(monkeypatch, tmp_path, elapsed=CAP - 5, results=[turn(), turn()],
                           on_call=lambda s, n: age(s, 10))
    session._play_inner()
    assert len(session.calls) == 1
    assert session.runtime_limit_reached(soft=True) and session.runtime_limit_reached()


def make_trimmed_agent(monkeypatch, *, seconds_left):
    agent = object.__new__(ToolAgent)
    agent._context_was_trimmed = True
    agent._pending_trim_reasons = ["context"]
    agent._last_step_summary = {}
    agent._game_deadline = time.monotonic() + seconds_left
    agent.handed_over = False

    def scheduler():
        agent.handed_over = True
        raise StopIteration  # stop before the gate; reaching here is the point

    monkeypatch.setattr(tool_agent, "_scheduler", scheduler)
    return agent


def test_handover_skipped_inside_grace(grace, monkeypatch):
    agent = make_trimmed_agent(monkeypatch, seconds_left=GRACE - 5)
    agent._maybe_handover()
    assert not agent.handed_over
    assert agent._context_was_trimmed is False


def test_handover_runs_before_grace(grace, monkeypatch):
    agent = make_trimmed_agent(monkeypatch, seconds_left=GRACE + 60)
    with pytest.raises(StopIteration):
        agent._maybe_handover()
    assert agent.handed_over


def test_handover_runs_without_grace_setting(monkeypatch):
    monkeypatch.delenv("ARC3_GAME_SOFT_STOP_SECONDS", raising=False)
    agent = make_trimmed_agent(monkeypatch, seconds_left=1)
    with pytest.raises(StopIteration):
        agent._maybe_handover()
    assert agent.handed_over
