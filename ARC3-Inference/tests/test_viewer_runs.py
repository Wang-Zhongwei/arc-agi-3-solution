"""Viewer over saved runs: transparent .xz, run discovery, benchmark/transcript fallback, timeline."""
from __future__ import annotations

import gzip
import json
import lzma
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from viewer import data, runfiles, server, timeline


GAME = "ab12-00000000"
KEY = f"{GAME}_p0"
BOARD_0 = [[0, 1], [2, 3]]
BOARD_1 = [[4, 5], [6, 15]]
BOARD_2 = [[9, 9], [9, 9]]


def _write(path: Path, text: str, *, xz: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if xz:
        path.with_name(path.name + ".xz").write_bytes(lzma.compress(text.encode("utf-8")))
    else:
        path.write_text(text, encoding="utf-8")


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


def _transcript() -> str:
    return (
        "--- analysis_step=1 | action=1 | 10:00:00 | tool-agent ---\n"
        "[SYSTEM PROMPT]\nYou are a coding agent.\n"
        "[USER PROMPT]\nTurn one prompt.\n"
        "[THINKING]\nFirst I will move right.\n"
        "--- analysis_step=2 | action=2 | 10:00:30 | tool-agent ---\n"
        "[SYSTEM PROMPT]\nYou are a coding agent.\n"
        "[USER PROMPT]\nTurn two prompt.\n"
        "[THINKING]\nNow press up twice.\n"
        "[TOOL CALL: python]\naction(['UP', 'UP'])\n"
    )


def _benchmark(*, start: str = "2026-10-04T10:00:00") -> dict:
    return {
        "label": "duck-harness-kaggle",
        "start_time": start,
        "end_time": "2026-10-04T10:10:00",
        "game_runs": [
            {
                "game_id": GAME,
                "number_of_levels": 2,
                "base_actions_per_level": [1, 4],
                "actions_per_level": [1, 2],
                "levels_completed": 1,
                "state": "gave_up",
                "final_score": timeline.game_score([1, 4], [1, 2], 1, 2),
                "started_at": "2026-10-04T10:01:00",
                "final_wallclock_seconds": 200.0,
                "final_generated_tokens": 5,
                "history": [
                    {"action": {"id": "ACTION4", "data": {}}, "generated_tokens": 100, "wallclock_seconds": 30.0},
                    {"action": {"id": "ACTION1", "data": {}}, "generated_tokens": 50, "wallclock_seconds": 60.0},
                    {"action": {"id": "ACTION6", "data": {"x": 3, "y": 7}}, "generated_tokens": 0, "wallclock_seconds": 61.0},
                ],
            }
        ],
    }


def _events() -> list[dict]:
    return [
        {"type": "initial", "title": "Initial State", "action_num": 0, "level": 1, "score": 0, "state": "NOT_FINISHED", "board": BOARD_0},
        {"type": "action", "action_num": 1, "analysis_step": 1, "action_name": "ACTION4", "action_display": "RIGHT", "level": 2, "score": 1, "state": "NOT_FINISHED", "board": BOARD_1},
        {"type": "analysis", "action_num": 1, "analysis_step": 1, "transcript": _transcript().split("--- analysis_step=2")[0], "level": 2, "score": 1, "board": BOARD_1},
        {"type": "action", "action_num": 2, "analysis_step": 2, "action_name": "ACTION1", "action_display": "UP", "level": 2, "score": 1, "state": "NOT_FINISHED", "board": BOARD_2},
        {"type": "action", "action_num": 3, "analysis_step": 2, "action_name": "ACTION6", "action_display": "MOUSE(row=7, col=3)", "level": 2, "score": 1, "state": "NOT_FINISHED", "board": BOARD_2},
        {"type": "analysis", "action_num": 3, "analysis_step": 2, "transcript": "--- analysis_step=2" + _transcript().split("--- analysis_step=2")[1], "level": 2, "score": 1, "board": BOARD_2},
    ]


@pytest.fixture()
def runs_root(tmp_path: Path) -> Path:
    """runs/kaggle-style tree: full/v1 (viewer data, all .xz) and panel/v1 (benchmark + transcripts only)."""
    root = tmp_path / "kaggle"
    full = root / "full" / "v1"
    viewer_data = {
        "game_id": GAME, "status": "gave_up", "viewer_steps": [], "eventCount": 6,
        "levels_completed": 1, "total_levels": 2, "final_score": 1.0,
    }
    _write(full / "artifacts" / f"{KEY}_viewer_data.json", json.dumps(viewer_data), xz=True)
    _write(full / "artifacts" / f"{KEY}_events.jsonl", _jsonl(_events()), xz=True)
    requests = [
        {
            "analysis_step": 2, "action": 2, "request_index_within_turn": 1,
            "messages": [{"role": "user", "content": "Turn two prompt."}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 40},
        },
        {
            "analysis_step": 2, "action": 2, "request_index_within_turn": 2,
            "messages": [
                {"role": "user", "content": "Turn two prompt."},
                {"role": "assistant", "content": "LATEST WINDOW"},
            ],
            "usage": {"prompt_tokens": 1100, "completion_tokens": 60},
        },
    ]
    _write(full / f"{KEY}_requests.jsonl", _jsonl(requests), xz=True)
    _write(full / "benchmark.json", json.dumps(_benchmark()), xz=True)
    _write(full / "summary.txt", "benchmark: x\nstarted:   2026-10-04 10:00:00\nmean score:    12.50\ngames:     1\nduration:  10m 0s\n", xz=False)
    _write(full / "transcripts" / f"{KEY}.txt", _transcript(), xz=True)

    panel = root / "panel" / "v1"
    _write(panel / "benchmark.json", json.dumps(_benchmark(start="2026-10-07T10:00:00")), xz=True)
    _write(panel / "transcripts" / f"{KEY}.txt", _transcript(), xz=True)
    _write(panel / "transcripts" / "zz99-11111111_p0.txt", "--- analysis_step=1 | action=1 | x ---\n[THINKING]\nonly a transcript\n", xz=False)

    (root / "logs-only" / "v1").mkdir(parents=True)
    (root / "logs-only" / "v1" / "logs-only.log").write_text("hello")
    return root


# --- transparent .xz -------------------------------------------------------------------------

def test_resolve_prefers_plain_and_falls_back_to_xz(tmp_path: Path) -> None:
    _write(tmp_path / "a.json", '{"v": "xz"}', xz=True)
    assert runfiles.resolve(tmp_path / "a.json") == tmp_path / "a.json.xz"
    assert runfiles.read_json(tmp_path / "a.json") == {"v": "xz"}
    (tmp_path / "a.json").write_text('{"v": "plain"}')
    assert runfiles.read_json(tmp_path / "a.json") == {"v": "plain"}
    assert runfiles.resolve(tmp_path / "missing.json") is None
    with pytest.raises(FileNotFoundError):
        runfiles.read_bytes(tmp_path / "missing.json")


def test_glob_logical_dedupes_plain_and_xz(tmp_path: Path) -> None:
    _write(tmp_path / "x_viewer_data.json", "{}", xz=True)
    _write(tmp_path / "y_viewer_data.json", "{}", xz=False)
    _write(tmp_path / "y_viewer_data.json", "{}", xz=True)
    names = [path.name for path in runfiles.glob_logical(tmp_path, "*viewer_data.json")]
    assert names == ["x_viewer_data.json", "y_viewer_data.json"]


def test_iter_jsonl_skips_torn_last_line(tmp_path: Path) -> None:
    _write(tmp_path / "e.jsonl", '{"a": 1}\n{"a": 2}\n{"a": ', xz=True)
    assert [row["a"] for row in runfiles.iter_jsonl(tmp_path / "e.jsonl")] == [1, 2]


# --- discovery -------------------------------------------------------------------------------

def test_list_runs_finds_nested_runs_newest_first(runs_root: Path) -> None:
    runs = runfiles.list_runs(runs_root)
    assert [meta.run_id for meta in runs] == ["panel/v1", "full/v1"]
    full = runs[1]
    assert full.mean_score == 12.5  # from summary.txt
    assert full.has_viewer_data and full.has_benchmark
    panel = runs[0]
    assert not panel.has_viewer_data and panel.has_transcripts
    assert panel.mean_score == pytest.approx(round(_benchmark()["game_runs"][0]["final_score"], 2))
    assert panel.started == "2026-10-07 10:00:00"


def test_parse_summary_txt_reads_header_fields() -> None:
    fields = runfiles.parse_summary_txt("benchmark: b\nmean score:    49.14\nstarted:   2026-10-04 21:01:24\n\nper-game:\n  x: score=1\n")
    assert fields["mean score"] == "49.14"
    assert fields["started"] == "2026-10-04 21:01:24"
    assert "x" not in fields


# --- runs with viewer data stored as .xz ----------------------------------------------------

def test_viewer_data_run_loads_from_xz(runs_root: Path) -> None:
    run_dir = runs_root / "full" / "v1"
    summary = data.load_run_summary(runs_dir=runs_root, run_dir=run_dir)
    assert summary["selected_run"] == "full/v1"
    assert [run["id"] for run in summary["runs"]] == ["panel/v1", "full/v1"]
    [game] = summary["games"]
    assert game["key"] == KEY and game["source"] == "viewer_data"
    assert game["actions"] == 3 and game["tokens"] == 155

    shell = data.load_game_shell_payload(runs_dir=runs_root, run_dir=run_dir, game_index=0)
    actions = shell["actions"]
    assert actions["n"] == [0, 1, 2, 3]
    assert actions["label"] == ["START", "RIGHT", "UP", "MOUSE(row=7, col=3)"]
    assert actions["step"] == [None, 1, 2, 2]
    # wall clock since run start: game started 60 s after the run
    assert actions["t"] == [60.0, 90.0, 120.0, 121.0]
    assert actions["tokens"] == [0, 100, 150, 150]
    # level 1 done in 1 action (baseline 1) = 100, weighted 1 of 1+2 -> 33.33
    assert actions["score"][0] == 0.0 and actions["score"][1] == pytest.approx(100 / 3, abs=1e-3)
    assert shell["stepCount"] >= 2

    lite = data.strip_step_boards(shell)
    assert all("board" not in (step.get("boardEvent") or {}) for step in lite["viewer_steps"])
    assert any("board" in (step.get("boardEvent") or {}) for step in shell["viewer_steps"])  # cache untouched

    frames = data.load_game_frames_payload(runs_dir=runs_root, run_dir=run_dir, game_index=0)
    assert frames["frames"] == ["01|23", "45|6f", "99|99", "99|99"]


def test_step_hydration_uses_latest_xz_request_snapshot(runs_root: Path) -> None:
    run_dir = runs_root / "full" / "v1"
    shell = data.load_game_shell_payload(runs_dir=runs_root, run_dir=run_dir, game_index=0)
    index = next(i for i, step in enumerate(shell["viewer_steps"]) if step.get("analysisStep") == 2 and step["stepKind"] == "turn")
    assert shell["viewer_steps"][index]["usage"]["completionTokens"] == 100
    payload = data.load_game_step_payload(runs_dir=runs_root, run_dir=run_dir, game_index=0, step_index=index)
    sections = payload["step"]["context"]["sections"]
    assert any(section["content"] == "LATEST WINDOW" for section in sections)
    local = [section["label"] for section in payload["step"]["localContext"]["sections"]]
    assert "THINKING" in local and "TOOL CALL: python" in local
    lite = data.strip_step_context(payload)
    assert "sections" not in lite["step"]["context"]
    assert lite["step"]["contextSectionCount"] == len(sections)
    assert lite["step"]["localContext"]["sections"]


# --- runs without viewer data ----------------------------------------------------------------

def test_benchmark_only_run_falls_back(runs_root: Path) -> None:
    run_dir = runs_root / "panel" / "v1"
    summary = data.load_run_summary(runs_dir=runs_root, run_dir=run_dir)
    assert summary["source"] == "benchmark"
    keys = [game["key"] for game in summary["games"]]
    assert keys == [KEY, "zz99-11111111_p0"]

    shell = data.load_game_shell_payload(runs_dir=runs_root, run_dir=run_dir, game_index=0)
    assert shell["source"] == "benchmark"
    actions = shell["actions"]
    assert actions["label"] == ["START", "RIGHT", "UP", "MOUSE(row=7, col=3)"]
    assert actions["step"] == [None, 1, 2, 2]  # transcript block K owns actions from its action=M on
    assert actions["level"] == [1, 2, 2, 2]
    assert actions["levels"] == [0, 1, 1, 1]
    assert actions["hasBoards"] is False
    turn_steps = [step for step in shell["viewer_steps"] if step["stepKind"] == "turn"]
    assert [step["analysisStep"] for step in turn_steps] == [1, 2]

    index = shell["viewer_steps"].index(turn_steps[1])
    payload = data.load_game_step_payload(runs_dir=runs_root, run_dir=run_dir, game_index=0, step_index=index)
    contents = [section["content"] for section in payload["step"]["localContext"]["sections"]]
    assert "Now press up twice." in contents

    frames = data.load_game_frames_payload(runs_dir=runs_root, run_dir=run_dir, game_index=0)
    assert frames["frames"] == [None, None, None, None]

    transcript_only = data.load_game_shell_payload(runs_dir=runs_root, run_dir=run_dir, game_index=1)
    assert transcript_only["actions"]["n"] == [0]
    assert transcript_only["viewer_steps"]


# --- score timeline --------------------------------------------------------------------------

def test_game_score_matches_official_formula() -> None:
    # level 1: (10/10)^2*100 = 100, level 2: (10/20)^2*100 = 25 -> (100*1 + 25*2) / 3
    assert timeline.game_score([10, 10], [10, 20], 2) == pytest.approx(50.0)
    assert timeline.game_score([10, 10], [5, 20], 1) == pytest.approx(100 / 3)  # capped at 115, then at 1/3*100
    assert timeline.game_score([10, 10], [10, 20], 0) == 0.0


def test_run_timeline_on_wall_clock(runs_root: Path) -> None:
    payload = data.load_run_timeline_payload(runs_dir=runs_root, run_dir=runs_root / "full" / "v1", game_key=KEY)
    assert payload["available"]
    game = payload["game"]
    assert game["t"] == [60.0, 90.0, 120.0, 121.0]
    assert game["score"][-1] == pytest.approx(game["final_score"])
    assert game["end"] == 260.0 and game["end_tokens"] == 155
    run = payload["run"]
    assert run["score"][-1] == pytest.approx(game["final_score"])
    assert run["tokens"][-1] == 155
    assert payload["duration"] == 600.0


# --- server ----------------------------------------------------------------------------------

def test_requested_run_dir_stays_inside_runs_dir(runs_root: Path) -> None:
    assert server._requested_run_dir(runs_dir=runs_root, default_run_dir=None, requested_run="full/v1") == runs_root / "full" / "v1"
    for bad in ("../..", "/etc", "full/../../x"):
        with pytest.raises(FileNotFoundError):
            server._requested_run_dir(runs_dir=runs_root, default_run_dir=None, requested_run=bad)


def test_server_end_to_end(runs_root: Path) -> None:
    handler = server.build_handler(runs_dir=runs_root, run_dir=None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def get(path: str) -> tuple[int, dict | str]:
        request = urllib.request.Request(base + path, headers={"Accept-Encoding": "gzip"})
        try:
            with urllib.request.urlopen(request) as response:
                body = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                kind = response.headers.get("Content-Type", "")
                return response.status, json.loads(body) if "json" in kind else body.decode()
        except urllib.error.HTTPError as error:
            return error.code, {}

    try:
        status, html = get("/")
        assert status == 200 and "ARC3 Run Viewer" in html
        status, runs = get("/api/runs")
        assert [run["id"] for run in runs["runs"]] == ["panel/v1", "full/v1"]
        status, run = get("/api/run")  # newest run by default
        assert run["selected_run"] == "panel/v1"
        status, shell = get("/api/game?run=full/v1&index=0&lite=1")
        assert status == 200 and shell["actions"]["n"] == [0, 1, 2, 3]
        status, frames = get("/api/frames?run=full/v1&index=0")
        assert frames["frames"][1] == "45|6f"
        status, step = get("/api/game-step?run=full/v1&index=0&step=0&lite=1")
        assert status == 200 and "step" in step
        status, tl = get(f"/api/timeline?run=panel/v1&game={KEY}")
        assert tl["available"] and tl["game"]["key"] == KEY
        status, text = get(f"/api/file?run=full/v1&path=transcripts/{KEY}.txt")
        assert status == 200 and "Now press up twice." in text
        assert get("/api/file?run=full/v1&path=../../panel/v1/benchmark.json")[0] == 404
        assert get("/api/run?run=../../")[0] == 404
        assert get("/api/game?run=full/v1&index=9")[0] == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
