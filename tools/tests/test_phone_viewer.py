"""tools/phone_viewer.py against a scratch run folder, served on 127.0.0.1.

Run from the repo root:  python -m pytest -q tools/tests/test_phone_viewer.py
"""
import importlib.util
import json
import lzma
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("phone_viewer", REPO / "tools" / "phone_viewer.py")
pv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pv)

TRANSCRIPT = (
    "preamble\n"
    "--- analysis_step=1 | action=1 ---\n[THINKING]\nlook at the board\n[TOOL CALL: python]\nprint(1)\n"
    "--- analysis_step=2 | action=2 ---\n[THINKING]\nmove right\n"
)


F001_SNAPSHOT = (
    "LATEST MODEL CALL SNAPSHOT\nanalysis_step: 20\n\n[AVAILABLE TOOLS]\n- python: ...\n\n[MODEL INPUT]\n"
    "[SYSTEM]\nYou are a coding agent.\n[USER]\nthe board\n[REASONING]\nthe tray is full\n"
    "[ASSISTANT TOOL CALL: python]\naction(['ACTION1'])\n"
    "--- analysis_step=20 | action=37 | 14:13:46 | tool-agent ---\n"
)
TYCHO = {"game_id": "bp35-0a0ad940", "trace": [
    {"turn": 1, "action": "ACTION3", "x": None, "y": None, "reasoning": {"level": 0, "llm_calls": [
        {"prompt": "=== Level 0, turn 0 ===", "reasoning_text": "the player is a blue blob",
         "response": "[tool] run_python({})", "tokens_in": 9444, "tokens_out": 756, "cache_read": 1536},
        {"prompt": "[run_python result]", "reasoning_text": "move left", "response": "[tool] take_action({})"},
    ]}},
    {"turn": 2, "action": "ACTION6", "x": 3, "y": 4, "reasoning": None},
]}


@pytest.fixture
def root(tmp_path):
    run = tmp_path / "k" / "v1"
    (run / "transcripts").mkdir(parents=True)
    good = lzma.compress(TRANSCRIPT.encode() * 50)
    (run / "transcripts" / "good_p0.txt.xz").write_bytes(good)
    (run / "transcripts" / "trunc_p0.txt.xz").write_bytes(good[: len(good) // 2])
    (run / "transcripts" / "corrupt_p0.txt.xz").write_bytes(b"\xfd7zXZ\x00" + b"\x00" * 200)
    (run / "k.log.xz").write_bytes(good[:40])

    f001 = tmp_path / "v4-fable-f001" / "v2"
    (f001 / "prompts").mkdir(parents=True)
    (f001 / "prompts" / "sk48-d8078629_p0.log.xz").write_bytes(lzma.compress(F001_SNAPSHOT.encode()))
    tycho = tmp_path / "v4-tycho-port" / "v1"
    (tycho / "tycho-results").mkdir(parents=True)
    (tycho / "tycho-results" / "game_bp35.json.xz").write_bytes(lzma.compress(json.dumps(TYCHO).encode()))
    (tycho / "tycho-results" / "manifest.json").write_text("{}")
    (tmp_path / "v4-fable-f001" / "v1").mkdir()
    (tmp_path / "v4-fable-f001" / "v1" / "v4-fable-f001.log").write_text("server failed\n")
    return tmp_path


@pytest.fixture
def get(root):
    server = ThreadingHTTPServer(("127.0.0.1", 0), pv.make_handler(pv.Runs([root])))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def fetch(path):
        try:
            with urllib.request.urlopen(base + path, timeout=10) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode()

    yield fetch
    server.shutdown()


def test_a_good_transcript_renders(get):
    status, body = get("/game?r=k/v1&g=good_p0")
    assert status == 200 and "look at the board" in body


@pytest.mark.parametrize("path", ["/game?r=k/v1&g=trunc_p0", "/game?r=k/v1&g=corrupt_p0", "/file?r=k/v1&f=k.log.xz"])
def test_a_truncated_or_corrupt_xz_gets_an_error_page(get, path):
    status, body = get(path)  # used to be an empty reply (curl exit 52)
    assert status == 500
    assert "truncated or corrupt" in body


def test_paths_outside_the_run_stay_refused(get):
    assert get("/file?r=k/v1&f=../../secret.txt")[0] == 404


def test_every_run_version_is_listed(get):
    status, body = get("/")
    for label in ("k/v1", "v4-fable-f001/v2", "v4-fable-f001/v1", "v4-tycho-port/v1"):
        assert f">{label}<" in body


def test_a_run_without_model_calls_says_so(get):
    status, body = get("/run?r=v4-fable-f001/v1")
    assert status == 200 and "No per-game model calls saved" in body and "v4-fable-f001.log" in body


def test_f001_prompt_logs_render_the_latest_call(get):
    status, body = get("/run?r=v4-fable-f001/v2")
    assert "sk48-d8078629_p0" in body
    status, body = get("/game?r=v4-fable-f001/v2&g=sk48-d8078629_p0")
    assert status == 200 and "1 model calls" in body and "latest model call only" in body
    assert "<summary>REASONING" in body and "the tray is full" in body
    assert "<details class='sec tool' open><summary>ASSISTANT TOOL CALL: python" in body


def test_tycho_results_render_each_model_call(get):
    status, body = get("/run?r=v4-tycho-port/v1")
    assert ">bp35<" in body and "manifest" not in body.split("<h2>Summary")[0]
    status, body = get("/game?r=v4-tycho-port/v1&g=bp35")
    assert status == 200 and "2 model calls" in body
    assert "turn=1 | level=0 | call 1/2 | then ACTION3" in body and "the player is a blue blob" in body
    assert "move left" in body
