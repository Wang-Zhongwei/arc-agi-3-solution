"""ARC3_VICTORY_LAP: route planning, the checked replay, and the solver hook.

The planner and executor are tested on hand-built transitions; the solver hook
on a small deterministic engine (a counter that must reach 3 on each level,
with clicks, deaths and RESETs) driven through the real
``_HarnessGameSession._execute_action``. The official-scorer end-to-end check
lives in test_victory_lap_e2e.py.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import arcengine
import pytest

from inference.framework import solver as solver_module
from inference.framework import victory_lap as vl

import taaf.game

# ---------------------------------------------------------------------------- planner


def T(prev, action, nxt, before=0, after=None, state="NOT_FINISHED", data=None):
    return vl.Transition.make(prev=prev, action=action, data=data, next=nxt, levels_before=before,
                              levels_after=before if after is None else after, state=state)


def test_state_hash_is_stable_and_border_crop_ignores_the_hud():
    a = [[0] * 10 for _ in range(10)]
    b = [row[:] for row in a]
    b[0][5] = 7  # a HUD cell
    assert vl.state_hash(a) == vl.state_hash([row[:] for row in a])
    assert vl.state_hash(a) != vl.state_hash(b)
    assert vl.state_hash(a, border=2) == vl.state_hash(b, border=2)
    b[5][5] = 7  # a board cell
    assert vl.state_hash(a, border=2) != vl.state_hash(b, border=2)


def test_last_attempt_drops_failed_attempts_and_keeps_clicks():
    ts = [
        T("s0", "ACTION1", "a"), T("a", "ACTION2", "dead", state="GAME_OVER"),
        T("dead", "RESET", "s0"),
        T("s0", "ACTION6", "b", data={"x": 3, "y": 4}), T("b", "ACTION1", "s1", after=1),
        T("s1", "ACTION1", "c", before=1), T("c", "RESET", "s1", before=1),
        T("s1", "ACTION6", "d", before=1, data={"y": 9, "x": 1}),
        T("d", "ACTION6", "win", before=1, after=2, state="WIN", data={"x": 2, "y": 2}),
    ]
    plan = vl.plan_routes(ts, start_hash="s0", number_of_levels=2)
    assert plan.skip_reason is None
    assert plan.lengths == [2, 2]
    first, second = plan.routes
    assert [(s.action, s.data_dict(), s.expect_hash, s.expect_levels) for s in first] == [
        ("ACTION6", {"x": 3, "y": 4}, "b", 0), ("ACTION1", {}, "s1", 1)]
    assert [(s.action, s.data_dict(), s.expect_hash) for s in second] == [
        ("ACTION6", {"x": 1, "y": 9}, "d"), ("ACTION6", {"x": 2, "y": 2}, "win")]


def test_last_attempt_keeps_a_loop_inside_the_winning_attempt():
    ts = [T("s0", "ACTION1", "a"), T("a", "ACTION2", "s0"), T("s0", "ACTION1", "a"),
          T("a", "ACTION3", "w", after=1, state="WIN")]
    assert vl.plan_routes(ts, start_hash="s0", number_of_levels=1).lengths == [4]


def test_bfs_cuts_loops_and_failed_attempts():
    ts = [T("s0", "ACTION1", "a"), T("a", "ACTION2", "s0"), T("s0", "ACTION1", "a"),
          T("a", "ACTION4", "b"), T("b", "ACTION6", "a", data={"x": 1, "y": 1}),
          T("a", "ACTION3", "w", after=1, state="WIN")]
    plan = vl.plan_routes(ts, start_hash="s0", number_of_levels=1, mode="bfs")
    assert [s.action for s in plan.routes[0]] == ["ACTION1", "ACTION3"]


def test_bfs_drops_ambiguous_moves():
    ts = [T("s0", "ACTION1", "a"), T("a", "RESET", "s0"), T("s0", "ACTION1", "x"),  # same move, other result
          T("x", "ACTION2", "s0"), T("s0", "ACTION4", "b"), T("b", "ACTION4", "a"),
          T("a", "ACTION3", "w", after=1, state="WIN")]
    plan = vl.plan_routes(ts, start_hash="s0", number_of_levels=1, mode="bfs")
    assert [s.action for s in plan.routes[0]] == ["ACTION4", "ACTION4", "ACTION3"]


def test_last_attempt_must_start_at_the_level_start():
    ts = [T("s0", "ACTION1", "a"), T("a", "RESET", "s0-but-hud-differs"),
          T("s0-but-hud-differs", "ACTION3", "w", after=1, state="WIN")]
    plan = vl.plan_routes(ts, start_hash="s0", number_of_levels=1)
    assert plan.skip_reason and "does not start at the level start" in plan.skip_reason


def test_unwon_level_has_no_route():
    ts = [T("s0", "ACTION1", "s1", after=1), T("s1", "ACTION1", "c", before=1)]
    plan = vl.plan_routes(ts, start_hash="s0", number_of_levels=2)
    assert plan.skip_reason == "level 2: no recorded level-up"


def test_game_score_matches_the_framework():
    base = [10, 20, 30]
    for apl, done in (([10, 20, 30], 3), ([20, 10, 31], 3), ([5, 40, 0], 2), ([12, 0, 0], 1)):
        run = taaf.game.GameRun(game_id="x", number_of_levels=3, base_actions_per_level=base,
                                actions_per_level=list(apl), levels_completed=done)
        assert vl.game_score(base, apl, done) == pytest.approx(run._compute_final_score())
    assert vl.game_score(None, [1], 1) is None


# ---------------------------------------------------------------------------- executor


def scripted_step(outcomes):
    calls = []

    def step(name, data):
        calls.append((name, data))
        out = outcomes[len(calls) - 1]
        if isinstance(out, Exception):
            raise out
        return out

    return step, calls


def two_level_plan():
    return vl.RoutePlan(start_hash="s0", routes=[
        [vl.Step("ACTION1", (), "a", 0), vl.Step("ACTION6", (("x", 1), ("y", 2)), "s1", 1)],
        [vl.Step("ACTION2", (), "w", 2)],
    ])


def test_lap_runs_reset_then_every_route():
    step, calls = scripted_step([("s0", 0, "NOT_FINISHED"), ("a", 0, "NOT_FINISHED"),
                                 ("s1", 1, "NOT_FINISHED"), ("w", 2, "WIN")])
    result = vl.run_lap(two_level_plan(), step, should_stop=lambda: False)
    assert result.status == "won" and result.reason == ""
    assert result.actions_per_level == [2, 1] and result.levels_completed == 2
    assert calls == [("RESET", {}), ("ACTION1", {}), ("ACTION6", {"x": 1, "y": 2}), ("ACTION2", {})]


@pytest.mark.parametrize("outcomes,reason", [
    ([("other", 0, "NOT_FINISHED")], "RESET did not restore"),
    ([("s0", 1, "NOT_FINISHED")], "RESET did not restore"),
    ([("s0", 0, "NOT_FINISHED"), ("zzz", 0, "NOT_FINISHED")], "mismatch at level 1 step 1/2"),
    ([("s0", 0, "NOT_FINISHED"), ("a", 0, "NOT_FINISHED"), ("s1", 0, "NOT_FINISHED")],
     "mismatch at level 1 step 2/2"),
    ([("s0", 0, "NOT_FINISHED"), ("a", 0, "GAME_OVER")], "GAME_OVER at level 1 step 1"),
    ([("s0", 0, "NOT_FINISHED"), RuntimeError("engine refused")], "exception at level 1 step 1"),
    ([RuntimeError("down")], "exception on RESET"),
    ([("s0", 0, "NOT_FINISHED"), ("a", 0, "NOT_FINISHED"), ("s1", 1, "NOT_FINISHED"),
      ("w", 2, "NOT_FINISHED")], "routes done but the engine state is NOT_FINISHED"),
])
def test_lap_stops_at_the_first_surprise(outcomes, reason):
    step, calls = scripted_step(outcomes)
    result = vl.run_lap(two_level_plan(), step, should_stop=lambda: False)
    assert result.status == "aborted"
    assert result.reason.startswith(reason), result.reason
    assert len(calls) == len(outcomes), "nothing runs after the surprise"


def test_lap_honours_stop_requests():
    step, calls = scripted_step([])
    assert vl.run_lap(two_level_plan(), step, should_stop=lambda: True).status == "skipped"
    assert calls == []
    flags = iter([False, False, True])
    step, calls = scripted_step([("s0", 0, "NOT_FINISHED"), ("a", 0, "NOT_FINISHED")])
    result = vl.run_lap(two_level_plan(), step, should_stop=lambda: next(flags))
    assert result.status == "aborted" and result.reason.startswith("stop requested at level 1 step 2")
    assert len(calls) == 2


def test_skipped_plan_sends_nothing():
    step, calls = scripted_step([])
    result = vl.run_lap(vl.RoutePlan(skip_reason="level 1: no recorded level-up"), step,
                        should_stop=lambda: False)
    assert result.status == "skipped" and not result.reset_sent and calls == []


# ---------------------------------------------------------------------------- solver hook

SIZE = 8


def grid(level: int, pos: int):
    rows = [[0] * SIZE for _ in range(SIZE)]
    rows[1][level] = 3
    rows[4][pos] = 9
    return tuple(tuple(row) for row in rows)


class CounterGame:
    """Reach pos 3 to clear a level. ACTION1 +1, ACTION2 -1, ACTION6 at x=k jumps
    to k (k <= 2), ACTION5 at pos 2 dies. RESET restarts the level (the whole
    game when WON). Two levels; baselines 3 and 3."""

    number_of_levels = 2

    def __init__(self) -> None:
        self.level = self.pos = 0
        self.won = False
        self.game_run = SimpleNamespace(
            game_id="ct01-0000", state="playing", history=[], actions_per_level=[0, 0],
            levels_completed=0, number_of_levels=2, base_actions_per_level=[3, 3],
            final_score=None, solver_note=None, solver_analysis_html="",
        )
        self.current_state = self._state("NOT_FINISHED")
        self.engine_calls: list[str] = []

    def _state(self, name):
        raw_state = getattr(arcengine.GameState, name)
        from solver_fakes import _State
        state = _State([grid(self.level, self.pos)], levels_completed=self.level, raw_state=raw_state)
        state.available_actions = [0, 1, 2, 5, 6]
        return state

    def _execute_action(self, action):
        name = action.id.name
        self.engine_calls.append(name)
        if name == "RESET":
            if self.won:
                self.level, self.won = 0, False
            self.pos = 0
            return self._state("NOT_FINISHED")
        if self.won:
            raise RuntimeError("empty frame after WIN")
        state_name = "NOT_FINISHED"
        if name == "ACTION1":
            self.pos += 1
        elif name == "ACTION2":
            self.pos = max(0, self.pos - 1)
        elif name == "ACTION6":
            self.pos = min(2, int(action.data["x"]))
        elif name == "ACTION5" and self.pos == 2:
            state_name = "GAME_OVER"
        if self.pos >= 3:
            self.level += 1
            self.pos = 0
            if self.level == self.number_of_levels:
                self.won, state_name = True, "WIN"
                self.pos = 3
        return self._state(state_name)

    def finish_game(self):
        self.game_run.final_score = 0.0

    def execute_action(self, action, generated_tokens=0, uncached_input_tokens=0):
        before = self.level
        state = self._execute_action(action)
        self.game_run.history.append(action)
        self.game_run.actions_per_level[min(before, 1)] += 1
        self.game_run.levels_completed = self.level
        if self.won:
            self.game_run.state = "won"
        self.current_state = state
        return state


class Gate:
    def __init__(self):
        self.released = 0

    def release(self):
        self.released += 1


def A(name, **data):
    return arcengine.ActionInput(id=arcengine.GameAction[name], data=data)


@pytest.fixture
def lap_on(monkeypatch):
    monkeypatch.setattr(solver_module, "VICTORY_LAP", True)


def play(tmp_path, actions, *, job_dir=None):
    game = CounterGame()
    solver = SimpleNamespace(max_runtime_s_per_game=None, max_actions_per_game=None,
                             max_generated_tokens_per_game=None, job_dir=job_dir,
                             soft_time_remaining_seconds=lambda: None, _claim_warmup_slot=lambda: False,
                             label="test")
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
    session = solver_module._HarnessGameSession(
        solver=solver, game=game, analyzer=None, game_index=0, pass_index=0,
        state_path=tmp_path / "artifacts" / "ct01-0000_p0_tool_runtime_state.json",
        transcript_path=tmp_path / "t.txt", analysis_html_relpath="",
        stop_event=threading.Event(), viewer_data_path=tmp_path / "viewer.json",
    )
    for action in actions:
        if solver_module._is_engine_game_over(game):
            session._execute_auto_reset()
        session._execute_action(action, batch_index=1, batch_size=1, generated_tokens=0)
    return session, game


# level 1: a click, two steps back, a death, then the clean attempt; level 2: straight
WIN_WITH_DETOURS = [A("ACTION6", x=1, y=0), A("ACTION2"), A("ACTION1"), A("ACTION1"), A("ACTION5"),
                    A("ACTION6", x=2, y=5), A("ACTION1"),
                    A("ACTION1"), A("ACTION1"), A("ACTION1")]


def test_flag_off_records_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(solver_module, "VICTORY_LAP", False)
    session, game = play(tmp_path, WIN_WITH_DETOURS)
    assert game.game_run.state == "won"
    assert session.lap_transitions == [] and session.lap_start_hash is None


def test_hook_replays_the_winning_attempts_and_reports(tmp_path, lap_on, caplog):
    session, game = play(tmp_path, WIN_WITH_DETOURS, job_dir=tmp_path)
    assert game.game_run.actions_per_level == [8, 3]
    gate = Gate()
    session._held_gate = gate
    before = (list(game.game_run.history), list(game.game_run.actions_per_level), game.game_run.state)
    with caplog.at_level("WARNING"):
        session._victory_lap()
    assert gate.released == 1, "the slot goes back before the lap"
    session._release_gate_slot()
    assert gate.released == 1, "and only once"
    record = session.lap_record
    assert record["status"] == "won", record
    assert record["route_lengths"] == [2, 3]  # click to 2 then +1; three steps on level 2
    assert record["play2_actions_per_level"] == [2, 3]
    assert record["play1_score"] == pytest.approx(vl.game_score([3, 3], [8, 3], 2))
    assert record["play2_score"] == pytest.approx(vl.game_score([3, 3], [2, 3], 2))
    assert record["official_score"] == pytest.approx(max(record["play1_score"], record["play2_score"]))
    assert game.engine_calls[-6:] == ["RESET", "ACTION6", "ACTION1", "ACTION1", "ACTION1", "ACTION1"]
    # the first play's bookkeeping is untouched
    assert (game.game_run.history, game.game_run.actions_per_level, game.game_run.state) == before
    lines = (tmp_path / "ct01-0000_p0_victory_lap.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["status"] == "won"
    assert any(r.getMessage().startswith("[VICTORY_LAP] ct01-0000 won") for r in caplog.records)
    assert solver_module.victory_lap_note(record).startswith(" victory_lap=won play2=")


def test_hook_skips_when_no_level_gets_shorter(tmp_path, lap_on):
    session, game = play(tmp_path, [A("ACTION1")] * 6)
    session._victory_lap()
    assert session.lap_record["status"] == "skipped"
    assert session.lap_record["reason"] == "no level gets shorter"
    assert "RESET" not in game.engine_calls


def test_hook_honours_the_stop_event(tmp_path, lap_on):
    session, game = play(tmp_path, WIN_WITH_DETOURS)
    session.stop_event.set()
    session._victory_lap()
    assert session.lap_record["status"] == "skipped"
    assert "RESET" not in game.engine_calls[-1:]


def test_hook_skips_near_the_deadline(tmp_path, lap_on):
    session, game = play(tmp_path, WIN_WITH_DETOURS)
    session.solver.soft_time_remaining_seconds = lambda: 30.0
    session._victory_lap()
    assert session.lap_record["status"] == "skipped" and "left" in session.lap_record["reason"]


def test_hook_stops_when_the_deadline_passes_mid_lap(tmp_path, lap_on):
    session, game = play(tmp_path, WIN_WITH_DETOURS)
    clock = iter([120.0, 120.0] + [0.0] * 50)
    session.solver.soft_time_remaining_seconds = lambda: next(clock)
    session._victory_lap()
    assert session.lap_record["status"] == "aborted"
    assert session.lap_record["reason"].startswith("stop requested")


def test_hook_aborts_on_a_divergent_engine(tmp_path, lap_on):
    session, game = play(tmp_path, WIN_WITH_DETOURS)
    real = game._execute_action

    def drifting(action):
        state = real(action)
        if action.id.name == "ACTION6":
            game.pos = 1  # the engine no longer does what play 1 saw
            state = game._state("NOT_FINISHED")
        return state

    game._execute_action = drifting
    session._victory_lap()
    record = session.lap_record
    assert record["status"] == "aborted" and record["reason"].startswith("mismatch at level 1 step 1/2")
    assert record["official_score"] == record["play1_score"], "a partial play never counts"


def test_hook_survives_an_engine_exception(tmp_path, lap_on):
    session, game = play(tmp_path, WIN_WITH_DETOURS)

    def broken(action):
        raise RuntimeError("gateway gone")

    game._execute_action = broken
    session._victory_lap()
    assert session.lap_record["status"] == "aborted"
    assert "gateway gone" in session.lap_record["reason"]


def test_play_inner_runs_the_lap_only_after_a_win(tmp_path, lap_on, monkeypatch):
    calls = []
    monkeypatch.setattr(solver_module._HarnessGameSession, "_victory_lap", lambda self: calls.append(1))
    session, game = play(tmp_path, WIN_WITH_DETOURS)
    session.analyzer = SimpleNamespace(generated_tokens=0)
    session._play_inner()
    assert calls == [1]
    session, game = play(tmp_path / "b", [A("ACTION1")])
    session.analyzer = SimpleNamespace(generated_tokens=0)
    session.stop_event.set()
    session._play_inner()
    assert calls == [1]


# ------------------------------------------------------------ kernel-kill guard (rerun only)


def test_kill_guard_is_off_outside_a_rerun(monkeypatch):
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    monkeypatch.delenv("ARC3_VICTORY_LAP_KILL_LIMIT_S", raising=False)
    monkeypatch.setattr(solver_module, "_process_age_seconds", lambda: 32_000.0)
    assert solver_module._kernel_kill_seconds_left() is None


def test_kill_guard_counts_down_from_the_process_start_in_a_rerun(monkeypatch):
    monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "True")
    monkeypatch.delenv("ARC3_VICTORY_LAP_KILL_LIMIT_S", raising=False)
    monkeypatch.delenv("ARC3_VICTORY_LAP_KILL_MARGIN_S", raising=False)
    monkeypatch.setattr(solver_module, "_process_age_seconds", lambda: 31_000.0)
    assert solver_module._kernel_kill_seconds_left() == pytest.approx(32_400 - 300 - 31_000)


def test_kill_guard_gives_up_without_a_process_age(monkeypatch):
    monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "1")
    monkeypatch.setattr(solver_module, "_process_age_seconds", lambda: None)
    assert solver_module._kernel_kill_seconds_left() is None


def test_hook_skips_a_late_win_before_the_kernel_kill(tmp_path, lap_on, monkeypatch):
    # The game clock still has plenty of time (the 532-min deadline ends after the
    # 9 h kill in a rerun), but the kernel limit is 40 s away, under the 60 s a lap
    # needs to start: no lap may start.
    session, game = play(tmp_path, WIN_WITH_DETOURS)
    session.solver.soft_time_remaining_seconds = lambda: 3_600.0
    monkeypatch.setenv("ARC3_VICTORY_LAP_KILL_LIMIT_S", "1000")
    monkeypatch.setenv("ARC3_VICTORY_LAP_KILL_MARGIN_S", "0")
    monkeypatch.setattr(solver_module, "_process_age_seconds", lambda: 960.0)
    session._victory_lap()
    assert session.lap_record["status"] == "skipped" and "left" in session.lap_record["reason"]
    assert "RESET" not in game.engine_calls[-1:]


def test_process_age_is_a_small_non_negative_number_or_none():
    age = solver_module._process_age_seconds()
    assert age is None or 0.0 <= age < 7 * 24 * 3600
