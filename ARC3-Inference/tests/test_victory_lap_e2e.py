"""ARC3_VICTORY_LAP against the official scorer, end to end.

Logged wins from user-combo-10x (tests/fixtures/victory_lap_wins.json.gz:
every action of the first play, clicks included) are replayed through the
harness's own ``_HarnessGameSession._execute_action`` into a real taaf
``GameAPI``, then ``_victory_lap()`` runs exactly as after a WIN on Kaggle.
The scorecard is then closed and read back from arc_agi itself:

- competition mode: the official arc_agi REST app in competition mode
  (taaf ``CompetitionArcadeServer``, what a submission's gateway runs), which
  hides the baselines from the client;
- offline mode: the local Arcade the Kaggle offline run plays, where the
  harness also reads the scorecard before it closes (``official_scorecard``).

Each game must score exactly max(play 1, play 2) under the official rules,
with play 1 equal to the framework's own final score.

Needs the competition's environment files (not in this repo): set
ARC3_ENV_FILES, or the test is skipped.
"""
from __future__ import annotations

import gzip
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import arc_agi
import arcengine
import pytest

import taaf.game
import taaf.game_api
from taaf.competition_arcade import DEFAULT_API_KEY, CompetitionArcadeServer

from inference.framework import solver as solver_module
from inference.framework import victory_lap as vl

HERE = Path(__file__).resolve().parent
DEFAULT_ENV_FILES = "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-3/data/environment_files"
ENV_FILES = Path(os.environ.get("ARC3_ENV_FILES", DEFAULT_ENV_FILES))
WINS = json.loads(gzip.open(HERE / "fixtures" / "victory_lap_wins.json.gz", "rt").read())

pytestmark = pytest.mark.skipif(not ENV_FILES.is_dir(), reason="set ARC3_ENV_FILES to the environment files")


def _session(tmp_path: Path, game) -> solver_module._HarnessGameSession:
    solver = SimpleNamespace(max_runtime_s_per_game=None, max_actions_per_game=None,
                             max_generated_tokens_per_game=None, job_dir=tmp_path, label="e2e",
                             soft_time_remaining_seconds=lambda: None)
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
    return solver_module._HarnessGameSession(
        solver=solver, game=game, analyzer=None, game_index=0, pass_index=0,
        state_path=tmp_path / "artifacts" / f"{game.game_id}_p0_tool_runtime_state.json",
        transcript_path=tmp_path / "t.txt", analysis_html_relpath="",
        stop_event=threading.Event(), viewer_data_path=tmp_path / "viewer.json",
    )


def _play(tmp_path: Path, win: dict, spec: taaf.game_api.ArcadeSpec):
    game = taaf.game_api.GameAPI(env_name=win["game_id"], arcade_spec=spec)
    game.start_game(taaf.game.RunSession(record_intermediate_states=False))
    session = _session(tmp_path, game)
    for action in win["actions"]:
        session._execute_action(
            arcengine.ActionInput(id=arcengine.GameAction[action["id"]], data=dict(action["data"])),
            batch_index=1, batch_size=1, generated_tokens=0,
        )
    run = game.game_run
    assert run.state == "won", "the logged first play must replay to a WIN"
    assert run.actions_per_level == win["actions_per_level"]
    session._victory_lap()
    return game, session


def _official(game, *, allow_missing: bool = False):
    scorecard = game._arcade.close_scorecard(game._scorecard_id)
    plays = scorecard.find_environment(game.env.environment_info.game_id) if scorecard else None
    assert plays is not None or allow_missing
    return plays


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(solver_module, "VICTORY_LAP", True)
    monkeypatch.setenv("ONLY_RESET_LEVELS", "true")  # as the notebook sets it before anything runs
    monkeypatch.delenv("OPERATION_MODE", raising=False)
    monkeypatch.setenv("ARC_API_KEY", DEFAULT_API_KEY)


def _check(win: dict, game, session, plays) -> dict:
    base = win["base_actions_per_level"]
    record = session.lap_record
    play1 = vl.game_score(base, win["actions_per_level"], win["levels_completed"])
    assert play1 == pytest.approx(win["final_score"])
    assert game.game_run._compute_final_score() == pytest.approx(play1) or game.game_run.base_actions_per_level is None
    if record["status"] == "skipped":
        assert len(plays.runs) == 1
        assert plays.score == pytest.approx(play1)
        return record
    assert record["status"] == "won", record
    play2 = vl.game_score(base, record["play2_actions_per_level"], record["play2_levels_completed"])
    assert len(plays.runs) == 2, "RESET after WIN must open a second play"
    assert plays.runs[0].score == pytest.approx(play1)
    assert plays.runs[1].score == pytest.approx(play2)
    assert plays.score == pytest.approx(max(play1, play2))
    return record


@pytest.mark.parametrize("win", WINS, ids=[w["game_id"][:4] + "@" + w["source"].split("/")[3] for w in WINS])
def test_competition_server_scores_the_best_play(tmp_path, env, win):
    server = CompetitionArcadeServer(game_ids=[win["game_id"]], environments_dir=str(ENV_FILES)).start()
    try:
        spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.COMPETITION,
                                        arc_base_url=server.base_url, environments_dir="")
        game, session = _play(tmp_path, win, spec)
        assert game.game_run.base_actions_per_level is None, "competition hides the baselines"
        record = _check(win, game, session, _official(game))
        # without baselines the lap cannot price itself: it runs whenever a level gets shorter
        assert record["status"] == ("won" if any(
            n < o for n, o in zip(record["route_lengths"] or [], win["actions_per_level"])) else "skipped")
    finally:
        server.stop()


@pytest.mark.parametrize("before_make", [True, False], ids=["notebook-env", "taaf-default"])
@pytest.mark.parametrize("win", WINS, ids=[w["game_id"][:4] + "@" + w["source"].split("/")[3] for w in WINS])
def test_offline_arcade_scores_the_best_play(tmp_path, env, monkeypatch, win, before_make):
    """The Kaggle offline run's Arcade. The notebook sets ONLY_RESET_LEVELS
    before the game is made, which keeps play 1 off the local scorecard (the
    make-time RESET becomes a level reset, not a new play); taaf alone sets it
    after make, and then both plays are there. Either way the harness's
    official_score is max(play 1, play 2) and its play 2 is the scorecard's."""
    if not before_make:
        monkeypatch.delenv("ONLY_RESET_LEVELS", raising=False)
    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_FILES))
    game, session = _play(tmp_path, win, spec)
    record = session.lap_record
    # with the notebook's env and no lap, the local scorecard never saw a play at all
    plays = _official(game, allow_missing=before_make and record["status"] == "skipped")
    base = win["base_actions_per_level"]
    play1 = vl.game_score(base, win["actions_per_level"], win["levels_completed"])
    assert record["play1_score"] == pytest.approx(play1)
    if record["status"] == "skipped":
        assert record["official_score"] == pytest.approx(play1)
        assert "scorecard_plays" not in record
        if not before_make:
            assert [run.score for run in plays.runs] == pytest.approx([play1])
        return
    assert record["status"] == "won", record
    play2 = vl.game_score(base, record["play2_actions_per_level"], record["play2_levels_completed"])
    assert record["play2_score"] == pytest.approx(play2)
    assert record["official_score"] == pytest.approx(max(play1, play2))
    assert f"official={max(play1, play2):.2f}" in solver_module.victory_lap_note(record)
    assert record["scorecard_plays"][-1] == pytest.approx(play2)
    if before_make:
        assert [run.score for run in plays.runs] == pytest.approx([play2])
    else:
        assert [run.score for run in plays.runs] == pytest.approx([play1, play2])
        assert plays.score == pytest.approx(max(play1, play2))


def test_fixture_covers_clicks_and_gains():
    assert len(WINS) >= 3
    assert all(any(a["id"] == "ACTION6" for a in w["actions"]) for w in WINS[:3])
    gains = [vl.game_score(w["base_actions_per_level"], w["actions_per_level"], w["levels_completed"]) < 100
             for w in WINS]
    assert sum(gains) >= 3, "at least three wins the lap can improve"
