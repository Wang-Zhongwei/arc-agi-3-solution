"""Tests for analysis/score_timeline.py: a synthetic run plus a real-run cross-check.

Run from the repo root: python -m pytest analysis/tests -q
(the ARC3-Inference venv has pytest and taaf: ARC3-Inference/.venv/bin/python -m pytest analysis/tests -q)
"""
from __future__ import annotations

import json
import lzma
import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import score_timeline as st  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REAL_RUN = REPO / "runs" / "kaggle" / "fz-arms" / "v4"


def _history(times, tokens=100):
    return [{"action": {"id": "ACTION1", "data": {}}, "generated_tokens": tokens,
             "uncached_input_tokens": 0, "wallclock_seconds": t} for t in times]


def make_run(tmp: Path, *, summary_override: dict | None = None, xz: bool = True) -> Path:
    """Two games. aa01 clears level 1 with action 2 (t=70 s) and level 2 with action 5
    (t=200 s) and wins; bb02 starts 30 s late and clears level 1 with its 4th action
    (t=30+150 s), then gives up at t=30+600 s."""
    run = tmp / "runs" / "kaggle" / "synth" / "v1"
    run.mkdir(parents=True)
    games = [
        {"game_id": "aa01-x", "number_of_levels": 2, "base_actions_per_level": [2, 6], "hint": None,
         "state": "won", "history": _history([10, 70, 100, 150, 200]), "actions_per_level": [2, 3],
         "levels_completed": 2, "final_score": None, "final_generated_tokens": 0,
         "final_wallclock_seconds": 201.0, "started_at": "2026-10-01T10:00:00"},
        {"game_id": "bb02-y", "number_of_levels": 3, "base_actions_per_level": [8, 5, 5], "hint": None,
         "state": "gave_up", "history": _history([20, 40, 90, 150, 300], tokens=50),
         "actions_per_level": [4, 1, 0], "levels_completed": 1, "final_score": None,
         "final_generated_tokens": 7, "final_wallclock_seconds": 600.0, "started_at": "2026-10-01T10:00:30"},
    ]
    for g in games:
        g["final_score"] = st.official_score(g["number_of_levels"], g["base_actions_per_level"],
                                             g["levels_completed"], g["actions_per_level"])
    bench = {"label": "synth", "n_passes": 1, "start_time": "2026-10-01T10:00:00",
             "end_time": "2026-10-01T10:10:40", "game_runs": games}
    text = json.dumps(bench)
    if xz:
        with lzma.open(run / "benchmark.json.xz", "wt") as f:
            f.write(text)
    else:
        (run / "benchmark.json").write_text(text)
    # aa01: level 1 2 actions vs baseline 2 -> 100; level 2 3 vs 6 -> min(115, 400) = 115.
    # Weighted (100*1 + 115*2)/3 = 110, capped at 100. bb02: (100*1)/6 = 16.67, cap 1/6*100.
    per = {"aa01-x": (100.0, 2, 5, 500), "bb02-y": (100 / 6, 1, 5, 250)}
    if summary_override:
        per.update(summary_override)
    lines = ["benchmark: synth", "games:     2", f"mean score:    {(per['aa01-x'][0] + per['bb02-y'][0]) / 2:.2f}",
             "", "per-game (mean across passes):"]
    n_levels = {"aa01-x": 2, "bb02-y": 3}
    for gid, (score, lv, acts, tok) in per.items():
        lines.append(f"  {gid}: score={score:.2f}, levels={lv:.1f}/{n_levels[gid]}, actions={acts}, tokens={tok}")
    (run / "summary.txt").write_text("\n".join(lines) + "\n")
    # Notebook log: benchmark starts at notebook second 600, server ready at 590.
    log = [
        {"stream_name": "stdout", "time": 590.0, "data": "\nREADY after 500s -> http://127.0.0.1:8001/v1\n"},
        {"stream_name": "stdout", "time": 600.5, "data": "Starting benchmark...\n"},
        {"stream_name": "stdout", "time": 801.0, "data": "[finished] aa01-x state=won level=2/2 score=100.00\n"},
        {"stream_name": "stdout", "time": 1230.0, "data": "[finished] bb02-y state=gave_up level=1/3\n"},
    ]
    (run / "synth.log").write_text(json.dumps(log))
    (run / "artifacts").mkdir()
    events = [{"type": "initial", "score": 0, "action_num": 0}]
    events += [{"type": "action", "action_num": i, "score": (1 if i >= 2 else 0) + (1 if i >= 5 else 0)}
               for i in range(1, 6)]
    (run / "artifacts" / "aa01-x_p0_events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return run


def test_official_score_cap_and_weights():
    # Every level cleared at baseline or better scores 100 (the cap), not the 115 ceiling.
    assert st.official_score(2, [2, 6], 2, [2, 3]) == 100.0
    # One of three levels at baseline: 100 * 1 / 6.
    assert st.official_score(3, [8, 5, 5], 1, [4, 1, 0]) == pytest.approx(100 / 6)
    # Twice the baseline: (1/2)^2 * 100 = 25 for level 1 only.
    assert st.official_score(2, [10, 10], 1, [20, 4]) == pytest.approx(25 / 3)
    assert st.official_score(2, None, 2, [1, 1]) == 0.0
    assert st.official_score(2, [3, 3], 0, [9, 0]) == 0.0


def test_official_score_matches_framework():
    taaf_game = pytest.importorskip("taaf.game")
    rng = random.Random(7)
    for _ in range(500):
        n = rng.randint(1, 10)
        base = [rng.randint(1, 200) for _ in range(n)]
        done = rng.randint(0, n)
        apl = [rng.randint(1, 400) if i <= done else 0 for i in range(n)]
        if rng.random() < 0.1 and done:
            apl[rng.randrange(done)] = 0  # a cleared level with no actions scores 0
        run = taaf_game.GameRun(game_id="g", number_of_levels=n, base_actions_per_level=base,
                                actions_per_level=apl, levels_completed=done)
        assert st.official_score(n, base, done, apl) == run._compute_final_score()


def test_synthetic_timeline(tmp_path):
    run = st.load_run(make_run(tmp_path))
    a, b = run.games
    assert b.offset_s == 30.0
    assert a.level_actions == [2, 5] and b.level_actions == [4]
    # Before any clear.
    assert run.at(69)["mean_score"] == 0.0
    # aa01 cleared level 1 at t=70: 100*1/3, cap 1/3*100.
    s = run.at(70)
    assert s["per_game"]["aa01-x"]["levels"] == 1
    assert s["per_game"]["aa01-x"]["score"] == pytest.approx(100 / 3)
    assert s["mean_score"] == pytest.approx(100 / 6)
    # bb02's 4th action is at 30+150=180 s.
    assert run.at(179.9)["per_game"]["bb02-y"]["levels"] == 0
    s = run.at(180)
    assert s["per_game"]["bb02-y"]["levels"] == 1
    assert s["per_game"]["bb02-y"]["actions"] == 4 and s["per_game"]["bb02-y"]["tokens"] == 200
    s = run.at(200)
    assert s["per_game"]["aa01-x"]["score"] == 100.0
    assert s["mean_score"] == pytest.approx((100 + 100 / 6) / 2)
    assert s["games_finished"] == 0 and run.at(201)["games_finished"] == 1
    assert run.at(40)["per_game"]["bb02-y"]["state"] == "playing"
    assert run.at(10)["per_game"]["bb02-y"]["state"] == "waiting"
    assert run.at(700)["per_game"]["bb02-y"]["state"] == "gave_up"
    # Final values equal summary.txt and final_score.
    assert st.cross_check(run) == []
    assert st.check_events(run) == []
    # Notebook clock: [finished] lines at 801 and 1230 for finals at 201 and 630 -> 600.
    assert run.bench_nb_s == pytest.approx(600.0)
    assert run.ready_nb_s == 590.0
    assert run.axis_shift("ready") == pytest.approx(10.0)
    rows = st.at_minutes([run], [1, 3.5, 20], "bench")
    assert rows[0]["mean_score"] == pytest.approx(0.0)
    assert rows[1]["mean_score"] == pytest.approx((100 + 100 / 6) / 2)
    assert rows[2]["past_end"] is True
    # On the notebook clock, minute 10 is benchmark second 0.
    assert st.at_minutes([run], [10], "notebook")[0]["mean_score"] == 0.0


def test_plain_json_and_mismatch_reported(tmp_path):
    run = st.load_run(make_run(tmp_path, xz=False, summary_override={"aa01-x": (90.0, 2, 5, 500)}))
    problems = st.cross_check(run)
    assert any("aa01-x: score" in p for p in problems)
    assert any("mean" in p for p in problems)


def test_cli_outputs(tmp_path, capsys):
    run_dir = make_run(tmp_path)
    out = tmp_path / "out"
    html = tmp_path / "chart.html"
    code = st.main(["--runs", str(run_dir), "--at", "1,4", "--out", str(out), "--html", str(html),
                    "--step", "30", "--check-events", "--strict"])
    assert code == 0
    text = capsys.readouterr().out
    assert "mean score at wall-clock minute" in text and "match summary.txt" in text
    games = (out / "synth_v1_games.csv").read_text().splitlines()
    assert games[0].startswith("run,t_bench_min,t_notebook_min,t_ready_min,game,levels")
    assert len(games) == 1 + 2 * (math.ceil(640 / 30) + 1)
    mean = (out / "synth_v1_mean.csv").read_text().splitlines()
    assert mean[-1].split(",")[4] == f"{(100 + 100 / 6) / 2:.4f}"
    levels = (out / "synth_v1_levels.csv").read_text().splitlines()
    assert len(levels) == 1 + 3
    assert (out / "at_minutes.csv").exists()
    page = html.read_text()
    assert "<svg" in page and '"synth/v1"' in page and "__DATA__" not in page


@pytest.mark.skipif(not REAL_RUN.exists(), reason="runs/kaggle/fz-arms/v4 not present")
def test_real_run_matches_summary():
    run = st.load_run(REAL_RUN)
    assert len(run.games) == 25
    assert st.cross_check(run) == []
    assert st.check_events(run) == []
    assert run.at(math.inf)["mean_score"] == pytest.approx(46.25, abs=0.005)
    # ft09 was won at about benchmark minute 48.7 with every level cleared.
    ft09 = next(g for g in run.games if g.game_id.startswith("ft09"))
    assert ft09.final()["levels"] == 6 and ft09.final()["score"] == 100.0
    assert ft09.at(ft09.action_t[-1] - 1e-6)["levels"] == 5
    assert 48 < ft09.action_t[-1] / 60 < 50
    # The notebook log ties the clocks: READY and the benchmark start about 11 min in.
    assert 10 < run.bench_nb_s / 60 < 12 and 10 < run.ready_nb_s / 60 < 12
    taaf_game = pytest.importorskip("taaf.game")
    with lzma.open(REAL_RUN / "benchmark.json.xz", "rt") as f:
        raw = json.load(f)["game_runs"]
    for r in raw:
        g = taaf_game.GameRun(game_id=r["game_id"], number_of_levels=r["number_of_levels"],
                              base_actions_per_level=r["base_actions_per_level"],
                              actions_per_level=r["actions_per_level"], levels_completed=r["levels_completed"])
        mine = next(x for x in run.games if x.game_id == r["game_id"])
        assert mine.at(math.inf)["score"] == g._compute_final_score()


def test_ready_time_from_live_log(tmp_path):
    run_dir = make_run(tmp_path)
    log = json.loads((run_dir / "synth.log").read_text())
    (run_dir / "synth.log").write_text(json.dumps([e for e in log if "READY" not in e["data"]]))
    assert st.load_run(run_dir).ready_nb_s is None
    row = {"kind": "game", "t_notebook_s": 700.0, "t_ready_s": 120.0, "t_bench_s": 100.0}
    (run_dir / "score_timeline.jsonl").write_text(json.dumps(row) + "\n")
    assert st.load_run(run_dir).ready_nb_s == 580.0
