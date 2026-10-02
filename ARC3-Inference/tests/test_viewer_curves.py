import json
from pathlib import Path

from viewer.server import _load_curves_payload


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


def test_curves_page_is_linked_from_viewer_pages() -> None:
    viewer = Path(__file__).parents[1] / "viewer"
    assert 'href="/curves"' in (viewer / "index.html").read_text(encoding="utf-8")
    assert 'href="/curves"' in (viewer / "comparison.html").read_text(encoding="utf-8")
    html = (viewer / "curves.html").read_text(encoding="utf-8")
    assert 'fetch("/api/curves"' in html
    assert "function pool(lanes)" in html
