import io
import json
from pathlib import Path

import pytest

from viewer import thumbnails
from inference.utils.viewer_artifacts import write_raw_events_sidecar

VIEWER_DIR = Path(__file__).parents[1] / "viewer"


def test_base_game_id_and_play_url():
    assert thumbnails.base_game_id("ls20-9607627b") == "ls20"
    assert thumbnails.base_game_id("LS20") == "ls20"
    assert thumbnails.base_game_id("../etc") is None
    assert thumbnails.base_game_id("") is None
    assert thumbnails.play_url("ls20-9607627b") == "https://arcprize.org/tasks/ls20"


def test_render_grid_png_scales_to_thumbnail_size():
    Image = pytest.importorskip("PIL.Image")
    import io

    png = thumbnails.render_grid_png([[0, 9], [12, 5]], size=64)
    assert png.startswith(b"\x89PNG")
    image = Image.open(io.BytesIO(png)).convert("RGB")
    assert image.size == (64, 64)
    assert image.getpixel((0, 0)) == (255, 255, 255)
    assert image.getpixel((63, 0)) == (30, 147, 255)
    assert image.getpixel((0, 63)) == (255, 133, 27)
    assert image.getpixel((63, 63)) == (0, 0, 0)


def test_game_thumbnail_png_rejects_invalid_ids(tmp_path):
    assert thumbnails.game_thumbnail_png("../etc", environments_dir=tmp_path) is None
    assert thumbnails.game_thumbnail_png("zzzz", environments_dir=tmp_path) is None
    assert thumbnails.game_thumbnail_png("ls20", environments_dir=tmp_path / "missing") is None


def _saved_game(runs_dir, game_id, *, events=None):
    artifacts = runs_dir / "kaggle" / "owner" / "notebook" / "v01" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    path = artifacts / f"{game_id}_p0_viewer_data.json"
    payload = {"game_id": game_id, "viewer_steps": []}
    if events is not None:
        payload["events"] = events
    path.write_text(json.dumps(payload))
    return path


def test_thumbnail_uses_saved_initial_frame_without_offline_environments(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    path = _saved_game(tmp_path, "ls20-9607627b")
    path.with_name("ls20-9607627b_p0_events.jsonl").write_text(
        json.dumps({"type": "initial", "board": [[9, 5], [12, 0]]}) + "\n{truncated later event",
    )
    monkeypatch.setattr(thumbnails, "default_environments_dir", lambda: None)
    monkeypatch.setattr(thumbnails, "_initial_grid", lambda *_: pytest.fail("Saved frames must not instantiate a game"))

    png = thumbnails.game_thumbnail_png("LS20-9607627B", runs_dir=tmp_path)

    image = Image.open(io.BytesIO(png)).convert("RGB")
    assert image.getpixel((0, 0)) == (30, 147, 255)
    assert image.getpixel((255, 255)) == (255, 255, 255)
    assert (tmp_path / "thumbnails" / "ls20-9607627b.png").read_bytes() == png
    # Subsequent hovers need not reopen the trace.
    path.with_name("ls20-9607627b_p0_events.jsonl").unlink()
    assert thumbnails.game_thumbnail_png("ls20-9607627b", runs_dir=tmp_path) == png


def test_synced_thumbnail_loads_without_trace_or_game_dependencies(tmp_path, monkeypatch):
    pytest.importorskip("PIL.Image")
    expected = thumbnails.render_grid_png([[9, 5], [12, 0]])
    shared = tmp_path / "thumbnails"
    shared.mkdir()
    (shared / "ls20-9607627b.png").write_bytes(expected)
    monkeypatch.setattr(thumbnails, "_MEMORY_CACHE", {})
    monkeypatch.setattr(thumbnails, "_saved_initial_grid", lambda *_: pytest.fail("Synced PNGs must load directly"))
    monkeypatch.setattr(thumbnails, "render_grid_png", lambda *_: pytest.fail("No renderer needed for synced PNGs"))
    monkeypatch.setattr(thumbnails, "default_environments_dir", lambda: pytest.fail("No game download needed"))

    assert thumbnails.game_thumbnail_png("ls20-9607627b", runs_dir=tmp_path) == expected


def test_thumbnail_cache_rejects_path_traversal(tmp_path):
    assert thumbnails.game_thumbnail_png("ls20-../../outside", runs_dir=tmp_path) is None
    assert not (tmp_path / "thumbnails").exists()


@pytest.mark.parametrize("storage", ["inline", "gzip"])
def test_saved_thumbnails_keep_game_versions_separate(tmp_path, monkeypatch, storage):
    pytest.importorskip("PIL.Image")
    for version, color in (("old", 9), ("new", 12)):
        events = [{"type": "initial", "board": [[color]]}]
        path = _saved_game(tmp_path, f"ls20-{version}", events=events if storage == "inline" else None)
        if storage == "gzip":
            write_raw_events_sidecar(path, events)
    monkeypatch.setattr(thumbnails, "default_environments_dir", lambda: None)

    old = thumbnails.game_thumbnail_png("ls20-old", runs_dir=tmp_path)
    new = thumbnails.game_thumbnail_png("ls20-new", runs_dir=tmp_path)

    assert old is not None and new is not None and old != new
    assert (tmp_path / "thumbnails" / "ls20-old.png").read_bytes() == old
    assert (tmp_path / "thumbnails" / "ls20-new.png").read_bytes() == new
    assert thumbnails.game_thumbnail_png("ls20-unknown", runs_dir=tmp_path) is None


def test_saved_thumbnail_does_not_present_final_board_as_initial(tmp_path, monkeypatch):
    _saved_game(tmp_path, "ls20-test", events=[{"type": "action", "board": [[9]], "action_num": 10}])
    monkeypatch.setattr(thumbnails, "default_environments_dir", lambda: None)

    assert thumbnails.game_thumbnail_png("ls20-test", runs_dir=tmp_path) is None


def test_saved_thumbnail_uses_initial_board_in_compact_viewer_steps(tmp_path, monkeypatch):
    pytest.importorskip("PIL.Image")
    path = _saved_game(tmp_path, "ls20-test")
    payload = json.loads(path.read_text())
    payload["viewer_steps"] = [{"stepKind": "initial", "boardEvent": {"type": "initial", "board": [[9]]}}]
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(thumbnails, "default_environments_dir", lambda: None)

    assert thumbnails.game_thumbnail_png("ls20-test", runs_dir=tmp_path).startswith(b"\x89PNG")


def test_pages_include_game_peek_hover():
    for name in ("index.html", "comparison.html", "curves.html"):
        html = (VIEWER_DIR / name).read_text(encoding="utf-8")
        assert '<script src="/game-peek.js"></script>' in html, name
        assert "data-game-peek" in html, name
    js = (VIEWER_DIR / "game_peek.js").read_text(encoding="utf-8")
    assert "/api/thumbnail?game=" in js
    assert "https://arcprize.org/tasks/" in js
