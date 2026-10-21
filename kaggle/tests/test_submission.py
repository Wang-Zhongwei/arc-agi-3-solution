"""--submission pushes the notebook an arm already ran, without touching the batch (#119 item 2).

main() runs against a copy of kaggle/ in a temp folder; the `kaggle kernels push` call is
faked, git runs for real on this checkout.

Run from the repo root:  python -m pytest -q kaggle/tests/test_submission.py
"""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bn = _load("build_notebook_submission", REPO / "kaggle" / "build_notebook.py")


@pytest.fixture
def kaggle_dir(tmp_path, monkeypatch):
    for name in ("arms.json", "template.ipynb", "template-metadata.json"):
        shutil.copy(REPO / "kaggle" / name, tmp_path / name)
    (tmp_path / "runs.json").write_text("[]\n")
    monkeypatch.setattr(bn, "HERE", tmp_path)
    monkeypatch.setattr(bn, "_branch_runs", lambda: {})  # other branches' records: test_cross_branch.py
    versions = iter(range(1, 100))
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if cmd[0] == "kaggle":
            return subprocess.CompletedProcess(cmd, 0, f"Kernel version {next(versions)} successfully pushed.", "")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(bn.subprocess, "run", fake_run)
    return tmp_path


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["build_notebook.py", *argv])
    bn.main()


def _runs(kaggle_dir):
    return json.loads((kaggle_dir / "runs.json").read_text())


def test_an_arm_that_ran_can_be_submitted_and_the_batch_stays_open(kaggle_dir, monkeypatch, capsys):
    _main(monkeypatch, "--arm", "control", "--push", "--new-batch")
    _main(monkeypatch, "--arm", "control", "--push", "--submission")
    assert "submitting the notebook 'control' ran as mellamoa/v4-control version 1" in capsys.readouterr().out
    control, submission = _runs(kaggle_dir)
    assert control["batch_open"] is True
    assert submission["kernel"] == "mellamoa/v4-submission" and submission["submission"] is True
    assert submission["batch_open"] is False
    assert submission["fingerprint"] == control["fingerprint"]
    # a resubmission is allowed too, and the experiment itself is still a duplicate
    _main(monkeypatch, "--arm", "control", "--push", "--submission")
    with pytest.raises(SystemExit, match="not building a duplicate"):
        _main(monkeypatch, "--arm", "control", "--push")


def test_a_submission_does_not_count_as_a_run_or_a_batch(kaggle_dir, monkeypatch):
    _main(monkeypatch, "--arm", "control", "--push", "--submission")
    _main(monkeypatch, "--arm", "control", "--push")  # not a duplicate of the submission
    assert [run.get("submission", False) for run in _runs(kaggle_dir)] == [True, False]


def test_submission_and_new_batch_are_exclusive(kaggle_dir, monkeypatch):
    with pytest.raises(SystemExit):
        _main(monkeypatch, "--arm", "control", "--push", "--submission", "--new-batch")
    assert _runs(kaggle_dir) == []


def test_old_submission_records_are_recognised_by_kernel():
    assert bn._is_submission({"kernel": "mellamoa/v4-submission"})
    assert not bn._is_submission({"kernel": "mellamoa/v4-control"})
