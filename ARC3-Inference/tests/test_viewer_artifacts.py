from pathlib import Path

from inference.utils.viewer_artifacts import (
    append_raw_events_sidecar,
    load_raw_events,
    raw_events_jsonl_sidecar_path,
    reset_raw_events_sidecar,
)


def test_jsonl_append_load_and_reset(tmp_path: Path) -> None:
    viewer = tmp_path / "game_p0_viewer_data.json"
    assert load_raw_events({}, viewer_data_path=viewer) == []
    append_raw_events_sidecar(viewer, [{"type": "initial", "score": 0}])
    append_raw_events_sidecar(viewer, [{"type": "action", "score": 5}])
    assert load_raw_events({}, viewer_data_path=viewer) == [
        {"type": "initial", "score": 0}, {"type": "action", "score": 5},
    ]
    reset_raw_events_sidecar(viewer)
    assert load_raw_events({}, viewer_data_path=viewer) == []


def test_jsonl_can_be_read_during_partial_append(tmp_path: Path) -> None:
    viewer = tmp_path / "game_p0_viewer_data.json"
    raw_events_jsonl_sidecar_path(viewer).write_text('{"score":0}\n{"score":')
    assert load_raw_events({}, viewer_data_path=viewer) == [{"score": 0}]


def test_embedded_events_take_precedence(tmp_path: Path) -> None:
    viewer = tmp_path / "game_p0_viewer_data.json"
    append_raw_events_sidecar(viewer, [{"score": 99}])
    assert load_raw_events({"events": [{"score": 0}, None]}, viewer_data_path=viewer) == [{"score": 0}]
