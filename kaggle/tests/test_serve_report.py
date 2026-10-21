"""serve_report.py weights decode lines by the wall time they cover (issue #121).

SGLang writes one "Decode batch" line per 40 decode iterations, so a plain
mean over lines over-weights small, fast batches. These tests feed a
synthetic serve.log in the real line format.

Run from ARC3-Inference:  python -m pytest -q ../kaggle/tests/test_serve_report.py
"""
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("serve_report", REPO / "kaggle" / "serve_report.py")
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)


def decode_line(running, tps, accept=1.0, usage=0.10, queue=0):
    return (f"[2026-10-08 15:22:44] Decode batch, #running-req: {running}, #full token: 30336, "
            f"full token usage: {usage:.2f}, mamba num: 40, mamba usage: 0.67, accept len: {accept:.2f}, "
            f"accept rate: 0.67, cuda graph: True, gen throughput (token/s): {tps:.2f}, "
            f"#queue-req: {queue}")


def write_run(tmp_path, lines):
    run = tmp_path / "run"
    run.mkdir()
    (run / "serve.log").write_text("\n".join(lines) + "\n")
    return run


def test_window_is_the_tokens_a_line_counted_over_its_speed():
    # 40 iterations x 10 requests x 2.5 tokens accepted = 1000 tokens at 500 tok/s
    assert sr.window_seconds(10, 2.5, 500.0) == pytest.approx(2.0)
    assert sr.window_seconds(10, 0.0, 400.0) == pytest.approx(1.0)  # no spec: one token a step
    assert sr.window_seconds(10, 2.5, 0.0) == 0.0


def test_decode_rows_are_weighted_by_wall_time_not_by_line(tmp_path):
    # one 4 s window at batch 10 (100 tok/s), then four 1 s windows at batch 2
    # (80 tok/s): 4 s of each, so time-weighted means sit halfway, while a
    # mean over lines counts the small batch four times
    lines = [decode_line(10, 100.0, usage=0.97, queue=1)] + [decode_line(2, 80.0)] * 4
    out = sr.report(write_run(tmp_path, lines))
    assert out["decode_minutes"] == pytest.approx(8 / 60)
    assert out["batch_mean"] == pytest.approx(6.0)
    assert out["batch_mean_logs"] == pytest.approx(3.6)
    assert out["batch_full"] == pytest.approx(0.5)
    assert out["batch_full_logs"] == pytest.approx(0.2)
    assert out["kv_full"] == pytest.approx(0.5) and out["kv_full_logs"] == pytest.approx(0.2)
    assert out["queued"] == pytest.approx(0.5) and out["queued_logs"] == pytest.approx(0.2)
    # 400 + 320 tokens over 8 s
    assert out["decode_tps"] == pytest.approx(90.0)
    assert out["decode_median"] == 80.0 and out["decode_p90"] == 100.0
    assert out["decode_median_logs"] == 80.0
    assert out["decode_full_batch"] == 100.0 and out["batch_max"] == 10


def test_report_prints_harness_first_and_labels_the_per_line_rows(tmp_path, capsys, monkeypatch):
    run = write_run(tmp_path, [decode_line(10, 500.0, accept=2.5)])
    (run / "summary.txt").write_text("mean score:    50.62\ngenerated tokens/sec: 601.99 (job wallclock)\n")
    monkeypatch.setattr(sr.sys, "argv", ["serve_report.py", str(run)])
    sr.main()
    printed = capsys.readouterr().out.splitlines()
    assert printed[1].startswith("harness tok/s") and printed[1].split()[-1] == "602"
    assert "-- time-weighted" in printed and "-- per decode log (diagnostic)" in printed
    assert printed.index("-- time-weighted") < printed.index("-- per decode log (diagnostic)")
