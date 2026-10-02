import json
from pathlib import Path

import pytest

from viewer.comparison_records import filter_score, load_annotations, run_summary, save_annotation


def example_score():
    return {
        "score": 25,
        "metadata": {"trial_labels": ["p0", "p1"]},
        "games": {
            "aa01-test": {"score": 50, "trial_scores": {"p0": 100, "p1": 0},
                          "levels_completed": {"p0": 2, "p1": 0}, "total_levels": 2,
                          "generated_tokens": {"p0": 100, "p1": 300}, "generated_tokens_mean": 200,
                          "actions": {"p0": 10, "p1": 30}, "actions_mean": 20,
                          "turns": {"p0": 2, "p1": 6}, "turns_mean": 4},
            "bb02-test": {"score": 0, "trial_scores": {"p0": 0, "p1": 0},
                          "levels_completed": {"p0": 0, "p1": 0}, "total_levels": 2,
                          "generated_tokens": {"p0": 50, "p1": 50}, "generated_tokens_mean": 50,
                          "actions": {"p0": 5, "p1": 5}, "actions_mean": 5,
                          "turns": {"p0": 1, "p1": 1}, "turns_mean": 1},
        },
        "progression_lanes": {"tokens": {
            "aa01-test": {"p0": [[0, 0], [100, 100]], "p1": [[0, 0], [300, 0]]},
            "bb02-test": {"p0": [[0, 0], [50, 0]], "p1": [[0, 0], [50, 0]]},
        }},
    }


def test_game_filter_updates_scores_totals_and_curves_without_mutation(tmp_path: Path):
    original = example_score()
    snapshot = json.dumps(original)
    result = filter_score(original, {"aa01"})
    assert result["score"] == 50
    assert result["usage"]["totals"]["generated_tokens"] == 400
    assert result["score_progression"]["points"][-1] == [200, 50]
    summary = run_summary(tmp_path, result)
    assert summary["coverage_pct"] == 50
    assert summary["depth_pct"] == 50
    assert summary["won_trials"] == 1
    assert json.dumps(original) == snapshot


def test_pass_filter_updates_every_trial_dependent_field(tmp_path: Path):
    result = filter_score(example_score(), passes={"0"})
    assert result["score"] == 50
    assert result["usage"]["totals"]["generated_tokens"] == 150
    assert result["usage"]["mean"]["actions"] == 7.5
    assert result["score_progression"]["points"][-1] == [75, 50]
    summary = run_summary(tmp_path, result)
    assert summary["levels_completed"] == 2
    assert summary["tokens_per_turn"] == 50
    assert summary["actions_per_turn"] == 5
    assert summary["tokens_per_level"] == 75
    assert summary["tokens_per_score"] == 3
    assert summary["trial_count"] == 2
    assert summary["coverage_pct"] == 50


def test_combined_filter_and_empty_selection(tmp_path: Path):
    result = filter_score(example_score(), {"aa01-test"}, {"p1"})
    assert result["score"] == 0
    assert result["usage"]["totals"]["generated_tokens"] == 300
    assert result["score_progression"]["points"][-1] == [300, 0]
    assert run_summary(tmp_path, result)["tokens_per_level"] is None
    empty = filter_score(example_score(), {"missing"}, {"0"})
    assert empty["score"] is None
    assert empty["score_progression"]["status"] == "unavailable"


def test_annotations_persist_overwrite_clear_and_preserve_other_runs(tmp_path: Path):
    valid = {"run-a", "run-b"}
    save_annotation(tmp_path, "run-a", {"leaderboard_score": 27.89, "notes": "first"}, valid_runs=valid)
    save_annotation(tmp_path, "run-b", {"leaderboard_score": 23, "notes": "other"}, valid_runs=valid)
    save_annotation(tmp_path, "run-a", {"leaderboard_score": None, "notes": "revised"}, valid_runs=valid)
    records = load_annotations(tmp_path)
    assert records["run-a"]["leaderboard_score"] is None
    assert records["run-a"]["notes"] == "revised"
    assert records["run-b"]["leaderboard_score"] == 23


@pytest.mark.parametrize("annotation", [
    {"leaderboard_score": float("nan")}, {"leaderboard_score": 101},
    {"leaderboard_score": True}, {"notes": 7}, {"mean_score": 100},
])
def test_rejects_invalid_annotation_without_writing(tmp_path: Path, annotation):
    with pytest.raises(ValueError):
        save_annotation(tmp_path, "run-a", annotation, valid_runs={"run-a"})
    assert load_annotations(tmp_path) == {}
