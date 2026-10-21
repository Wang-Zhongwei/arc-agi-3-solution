"""Tests for the notebook's score-timeline logger (kaggle/score_logger.py) and the cell
build_notebook.py adds for it.

Run from the repo root: python -m pytest kaggle/tests -q
With the ARC3-Inference venv (PYTHONPATH=tufa-arc-agi-framework/src) the stubs are real
taaf GameRun / ActionRecord objects, so the logger is checked against the framework's own
score; with plain python a look-alike stub is used.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import build_notebook as bn  # noqa: E402
import score_logger as sl  # noqa: E402

try:
    import taaf.game as taaf_game
except Exception:  # noqa: BLE001 - taaf needs arcengine; fall back to a stub
    taaf_game = None


def game_run(game_id, n_levels, base, actions_per_level, levels, *, state="playing", final=None, tokens=100):
    history_len = sum(actions_per_level)
    if taaf_game is not None:
        run = taaf_game.GameRun(game_id=game_id, number_of_levels=n_levels, base_actions_per_level=base,
                                state=state, actions_per_level=list(actions_per_level), levels_completed=levels,
                                final_score=final)
        run.history.extend(taaf_game.ActionRecord(action=None, generated_tokens=tokens, uncached_input_tokens=0,
                                                  wallclock_seconds=float(i)) for i in range(history_len))
        return run
    return SimpleNamespace(game_id=game_id, number_of_levels=n_levels, base_actions_per_level=base, state=state,
                           actions_per_level=list(actions_per_level), levels_completed=levels, final_score=final,
                           history=[SimpleNamespace(generated_tokens=tokens) for _ in range(history_len)])


def framework_score(run):
    if taaf_game is not None:
        return run._compute_final_score()
    return sl._official_score(run.number_of_levels, run.base_actions_per_level, run.levels_completed,
                              run.actions_per_level)


class BrokenRun:
    game_id = "zz99-broken"

    @property
    def levels_completed(self):
        raise RuntimeError("half-built")


def bench(runs, *, minutes_in=37.0, ended=False):
    start = datetime.now() - timedelta(minutes=minutes_in)
    return SimpleNamespace(game_runs=runs, start_time=start, end_time=datetime.now() if ended else None, n_passes=1)


def test_snapshot_rows_and_official_score():
    runs = [
        game_run("ft09-0d8bbf25", 6, [43, 12, 23, 28, 65, 37], [4, 7, 15, 3, 0, 0], 3),
        game_run("sk48-d8078629", 8, [61, 177, 101, 103, 230, 181, 125, 92], [138] + [0] * 7, 0,
                 state="gave_up", final=0.0, tokens=7),
        game_run("tu93-0768757b", 3, [19, 16, 34], [40, 15, 2], 2),  # level 1 at 2x baseline
    ]
    now = time.time()
    rows = sl.score_snapshot(bench(runs), notebook_start=now - 49 * 60, server_ready=now - 38 * 60, now=now)
    games, run = rows[:-1], rows[-1]
    assert [r["game"] for r in games] == ["ft09-0d8bbf25", "sk48-d8078629", "tu93-0768757b"]
    for r, gr in zip(games, runs):
        assert r["score"] == pytest.approx(framework_score(gr), abs=1e-4)
        assert r["actions"] == len(gr.history)
    assert games[0]["levels"] == 3 and games[0]["tokens"] == 2900 and games[0]["state"] == "playing"
    assert games[1]["state"] == "finished" and games[1]["run_state"] == "gave_up" and games[1]["tokens"] == 7 * 138
    assert games[0]["t_notebook_s"] == pytest.approx(49 * 60, abs=1)
    assert games[0]["t_ready_s"] == pytest.approx(38 * 60, abs=1)
    assert games[0]["t_bench_s"] == pytest.approx(37 * 60, abs=2)
    assert run["kind"] == "run" and run["n_games"] == 3 and run["games_finished"] == 1
    assert run["mean_score"] == pytest.approx(sum(r["score"] for r in games) / 3, abs=1e-3)
    line = sl.format_score_line(rows)
    assert line.startswith("[score] t=37m nb=49m mean=")
    assert f"| ft09 3L {games[0]['score']:.1f} |" in line
    assert "sk48 0L 0.0*" in line


def test_hidden_baselines_give_no_score():
    rows = sl.score_snapshot(bench([game_run("ar25-x", 2, None, [3, 0], 1)]))
    assert rows[0]["score"] is None and rows[-1]["mean_score"] == 0.0
    assert "ar25 1L ?" in sl.format_score_line(rows)


def test_snapshot_survives_a_broken_game():
    rows = sl.score_snapshot(bench([game_run("ft09-x", 2, [5, 5], [5, 1], 1), BrokenRun()]))
    assert "error" in rows[1] and "half-built" in rows[1]["error"]
    assert rows[-1]["unreadable"] == 1 and rows[-1]["levels"] == 1
    assert "zz99 ?" in sl.format_score_line(rows)


def test_token_counter_reads_only_new_records():
    gr = game_run("ft09-x", 2, [5, 5], [3, 0], 0, tokens=10)
    counter = sl._TokenCounter()
    assert counter.tokens(gr, len(gr.history)) == 30
    gr.history.append(SimpleNamespace(generated_tokens=5) if taaf_game is None else
                      taaf_game.ActionRecord(action=None, generated_tokens=5, uncached_input_tokens=0,
                                             wallclock_seconds=9.0))
    assert counter.tokens(gr, len(gr.history)) == 35


def _wait(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_logger_thread_writes_prints_and_stops(tmp_path):
    path = tmp_path / "score_timeline.jsonl"
    holder = {"bm": None}
    printed = []
    stop = sl.start_score_logger(lambda: holder["bm"], path, notebook_start=time.time() - 600,
                                 get_server_ready=lambda: None, interval_s=0.05, print_every_s=0.0,
                                 out=lambda *a, **k: printed.append(" ".join(map(str, a))))
    time.sleep(0.2)
    assert not path.exists()  # no benchmark yet: nothing written
    bm = bench([game_run("ft09-x", 2, [5, 5], [5, 1], 1), game_run("vc33-y", 2, [3, 3], [1, 0], 0)])
    holder["bm"] = bm
    assert _wait(lambda: path.exists() and len(path.read_text().splitlines()) >= 6)
    rows = [json.loads(line) for line in path.read_text().split("\n")[:-1]]  # complete lines only
    assert {r["kind"] for r in rows} == {"game", "run"}
    assert all(r["t_ready_s"] is None for r in rows)
    assert any(p.startswith("[score] t=37m nb=") for p in printed)
    bm.end_time = datetime.now()
    assert _wait(lambda: not stop.thread.is_alive())  # the final snapshot ends the thread
    assert json.loads(path.read_text().splitlines()[-1])["kind"] == "run"


def test_logger_thread_survives_errors(tmp_path):
    class Exploding:
        start_time = datetime.now()
        end_time = None

        @property
        def game_runs(self):
            raise OSError("boom")

    printed = []
    stop = sl.start_score_logger(lambda: Exploding(), tmp_path / "x.jsonl", interval_s=0.02,
                                 out=lambda *a, **k: printed.append(" ".join(map(str, a))))
    time.sleep(0.3)
    assert stop.thread.is_alive(), "the logger thread died on an error"
    assert 1 <= len(printed) <= 3 and "boom" in printed[0]
    # Unwritable path: errors are reported (at most 3 times), the thread keeps going.
    stop.set()
    printed.clear()
    stop2 = sl.start_score_logger(lambda: bench([game_run("ft09-x", 2, [5, 5], [5, 1], 1)]),
                                  tmp_path / "missing-dir" / "x.jsonl", interval_s=0.02,
                                  out=lambda *a, **k: printed.append(" ".join(map(str, a))))
    time.sleep(0.3)
    assert 1 <= len(printed) <= 3 and "No such file" in printed[0]
    assert stop2.thread.is_alive()
    stop2.set()


def test_snapshot_cost_is_negligible():
    runs = [game_run(f"g{i:03d}-x", 8, [50] * 8, [60, 60, 60, 60, 60, 60, 60, 80], 7) for i in range(110)]
    bm = bench(runs)
    counter = sl._TokenCounter()
    sl.score_snapshot(bm, counter=counter)  # first tick counts every record once
    t0 = time.perf_counter()
    for _ in range(10):
        sl.score_snapshot(bm, counter=counter)
    per_tick = (time.perf_counter() - t0) / 10
    assert per_tick < 0.05, per_tick  # once a minute, so < 0.1% of one core


def _template():
    return json.loads((HERE.parent / "template.ipynb").read_text(encoding="utf-8"))


def test_builder_adds_logger_cell_after_monitor():
    nb = bn.build_notebook(_template(), "diff --git a/x b/x\n", env={}, demo_games="balanced", demo_minutes=20)
    texts = [bn._cell_text(c) for c in nb["cells"]]
    monitor = next(i for i, t in enumerate(texts) if bn.MONITOR_ANCHOR in t)
    assert nb["cells"][monitor + 1]["cell_type"] == "markdown" and "score timeline" in texts[monitor + 1]
    code = texts[monitor + 2]
    assert nb["cells"][monitor + 2]["cell_type"] == "code" and "def start_score_logger(" in code
    assert sum("def start_score_logger(" in t for t in texts) == 1
    compile(code, "score-logger-cell", "exec")
    serve = next(t for t in texts if bn.SERVE_ANCHOR in t)
    assert serve.count("SERVER_READY_TIME = time.time()") == 2  # wait loop + post-deadline streamer
    assert texts.index(code) < next(i for i, t in enumerate(texts) if "await bm.run(" in t)
    assert len(nb["cells"]) == len(_template()["cells"]) + 2


def test_built_cell_runs_against_a_stub(tmp_path):
    nb = bn.build_notebook(_template(), "diff --git a/x b/x\n", env={}, demo_games="balanced", demo_minutes=20)
    code = next(bn._cell_text(c) for c in nb["cells"] if "def start_score_logger(" in bn._cell_text(c))
    bm = bench([game_run("ft09-x", 2, [5, 5], [5, 1], 1)])
    ns = {"TRUE_SUBMISSION": False, "WORKING_DIR": tmp_path, "NOTEBOOK_START_TIME": time.time() - 700,
          "SERVER_READY_TIME": time.time() - 100, "bm": bm, "__name__": "__main__"}
    exec(compile(code, "score-logger-cell", "exec"), ns)
    path = tmp_path / "score_timeline.jsonl"
    # Wait for the whole first tick (one game line and the run line), not just the file.
    assert _wait(lambda: path.exists() and path.read_text().count("\n") >= 2)
    row = json.loads(path.read_text().splitlines()[0])
    assert row["game"] == "ft09-x" and row["t_ready_s"] == pytest.approx(100, abs=2)
    ns["_stop_score_logger"].set()
    # In a competition rerun the cell starts nothing.
    ns2 = {**ns, "TRUE_SUBMISSION": True, "WORKING_DIR": tmp_path / "rerun"}
    ns2.pop("_stop_score_logger")
    exec(compile(code, "score-logger-cell", "exec"), ns2)
    assert "_stop_score_logger" not in ns2
