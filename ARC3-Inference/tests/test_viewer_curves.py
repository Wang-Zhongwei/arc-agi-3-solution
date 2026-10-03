import json
from pathlib import Path

import pytest

from viewer.server import _load_comparison_payload, _load_curves_payload, _load_evaluation


def _write_evaluation(run_dir: Path, **extra) -> None:
    run_dir.mkdir(parents=True)
    evaluation = {
        "score": 30.0,
        "metadata": {"created_at": "2026-09-23T00:00:00Z"},
        "games": {
            "aa01-1": {
                "score": 30.0,
                "trial_scores": {"v1": 30.0},
                "generated_tokens_mean": 10.0,
                "actions_mean": 2.0,
                "levels_completed_mean": 1.0,
                "total_levels": 3,
                "states": {"v1": "won"},
            },
        },
        **extra,
    }
    (run_dir / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")


def test_curves_payload_sends_lanes_and_game_means_for_every_evaluated_run(tmp_path: Path) -> None:
    lanes = {"version": 1, "tokens": {"aa01-1": {"v1": [[0, 0], [10, 30]]}},
             "actions": {"aa01-1": {"v1": [[0, 0], [2, 30]]}}}
    _write_evaluation(tmp_path / "kaggle" / "owner" / "slug" / "v02",
                      score_progression={"status": "complete"}, action_progression={"status": "complete"},
                      progression_lanes=lanes)
    _write_evaluation(tmp_path / "kaggle" / "owner" / "slug" / "v01")
    (tmp_path / "kaggle" / "owner" / "slug" / "v03").mkdir()

    runs = _load_curves_payload(tmp_path)["runs"]

    assert [run["run_id"] for run in runs] == ["kaggle/owner/slug/v02", "kaggle/owner/slug/v01"]
    newest, oldest = runs
    assert newest["lanes"] == lanes
    assert newest["curves"]["actions"] == {"status": "complete"}
    assert newest["games"]["aa01-1"] == {
        "score": 30.0,
        "trial_scores": {"v1": 30.0},
        "generated_tokens_mean": 10.0,
        "actions_mean": 2.0,
        "levels_completed_mean": 1.0,
        "total_levels": 3,
        "levels_completed": None, "states": {"v1": "won"},
        "generated_tokens": None, "actions": None, "turns": None,
    }
    assert oldest["lanes"] is None and oldest["curves"]["tokens"] is None


@pytest.mark.parametrize("loader", [_load_curves_payload, _load_comparison_payload])
@pytest.mark.parametrize("games", [
    "aa01-1",
    {"aa01-1": []},
    {"aa01-1": None},
])
def test_payload_skips_incompatible_game_shapes_without_losing_valid_runs(
    tmp_path: Path, caplog, loader, games,
) -> None:
    valid = tmp_path / "kaggle" / "owner" / "slug" / "v01"
    invalid = valid.parent / "v02"
    _write_evaluation(valid)
    _write_evaluation(invalid, games=games)
    invalid_path = invalid / "evaluation.json"
    original = invalid_path.read_text()

    payload = loader(tmp_path)

    runs = payload["runs"]
    run_ids = list(runs) if isinstance(runs, dict) else [run["run_id"] for run in runs]
    assert run_ids == ["kaggle/owner/slug/v01"]
    assert str(invalid_path) in caplog.text
    assert "games" in caplog.text
    assert invalid_path.read_text() == original


def test_curves_page_is_linked_from_viewer_pages() -> None:
    viewer = Path(__file__).parents[1] / "viewer"
    assert 'href="/curves"' in (viewer / "index.html").read_text(encoding="utf-8")
    assert 'href="/curves"' in (viewer / "comparison.html").read_text(encoding="utf-8")
    html = (viewer / "curves.html").read_text(encoding="utf-8")
    assert 'fetch("/api/curves"' in html
    assert "function pool(lanes)" in html


@pytest.mark.parametrize("loader", [_load_curves_payload, _load_comparison_payload])
@pytest.mark.parametrize("games", [[], [{"game_id": "aa01-1", "score": 30.0,
                                        "levels_completed": 1.5, "states": "won"}]])
def test_payload_accepts_list_games_and_scalar_trial_summaries(tmp_path, loader, games):
    run = tmp_path / "kaggle" / "owner" / "slug" / "v01"
    _write_evaluation(run, games=games)
    path = run / "evaluation.json"
    original = path.read_text()
    runs = loader(tmp_path)["runs"]
    assert len(runs) == 1
    score = _load_evaluation(path)
    assert set(score["games"]) == ({"aa01-1"} if games else set())
    assert path.read_text() == original
