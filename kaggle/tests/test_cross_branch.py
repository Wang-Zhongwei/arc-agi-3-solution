"""The builder's rules see every branch's kaggle/runs.json, not only the checked-out one (#119 item 7).

Run from the repo root:  python -m pytest -q kaggle/tests/test_cross_branch.py
"""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bn = _load("build_notebook_cross_branch", REPO / "kaggle" / "build_notebook.py")


def _run(arm, version=1, *, commit="c1", code="code1", fingerprint=None, kernel=None, **extra):
    return {"arm": arm, "kernel": kernel or f"mellamoa/v4-{arm}", "version": version, "commit": commit,
            "code": code, "runtime_code": code, "fingerprint": fingerprint or f"f-{arm}-{version}", **extra}


def _sh(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
                   cwd=repo, check=True, capture_output=True)


def test_branch_runs_reads_every_local_and_remote_branch(tmp_path, monkeypatch):
    (tmp_path / "kaggle").mkdir()
    runs = tmp_path / "kaggle" / "runs.json"
    runs.write_text("[]\n")
    _sh(tmp_path, "init", "-q", "-b", "main")
    _sh(tmp_path, "add", "-A")
    _sh(tmp_path, "commit", "-qm", "empty record")
    _sh(tmp_path, "checkout", "-qb", "arm/a")
    runs.write_text(json.dumps([_run("control")]))
    _sh(tmp_path, "commit", "-qam", "runs: control")
    _sh(tmp_path, "update-ref", "refs/remotes/origin/arm/a", "HEAD")
    _sh(tmp_path, "checkout", "-qb", "arm/b", "main")
    (tmp_path / "kaggle" / "runs.json").unlink()
    _sh(tmp_path, "commit", "-qam", "no record file")
    monkeypatch.setattr(bn, "REPO", tmp_path)
    found = bn._branch_runs()
    assert set(found) == {"arm/a", "origin/arm/a"}  # empty and missing files are left out
    assert found["arm/a"][0]["arm"] == "control"


def test_other_branch_runs_merges_copies_and_skips_local_and_foreign_kernels():
    local = [_run("control")]
    branches = {
        "arm/x": [_run("control"), _run("tput", batch_open=True), _run("old", fingerprint="f-old")],
        "origin/arm/x": [_run("tput", batch_open=False), _run("old", status="failed")],
        "oldfz/base": [_run("fz", kernel="mellamoa/fz-arms")],
    }
    other = {run["arm"]: run for run in bn._other_branch_runs(local, branches)}
    assert set(other) == {"tput", "old"}
    assert other["tput"]["batch_open"] is False  # closed in one copy: closed
    assert other["old"]["status"] == "failed"
    assert other["tput"]["branches"] == ["arm/x", "origin/arm/x"]


def test_a_duplicate_on_another_branch_is_refused():
    other = [{**_run("control", fingerprint="f-same"), "branches": ["arm/throughput-stack"]}]
    with pytest.raises(SystemExit, match="on arm/throughput-stack\\); not building a duplicate"):
        bn._check_unique("control", "f-same", "code1", [], new_batch=True, other_runs=other)


def test_a_branch_without_a_batch_may_not_build_against_another_branchs_batch():
    """The audit's case: ledger-on-trim built on its own branch while control@519c4f5 runs."""
    other = [{**_run("control", commit="519c4f5", code="batch-code"), "branches": ["arm/throughput-stack"]}]
    with pytest.raises(SystemExit, match="control @519c4f5 on arm/throughput-stack"):
        bn._check_unique("ledger-on-trim", "f-new", "other-code", [], new_batch=False, other_runs=other)
    bn._check_unique("ledger-on-trim", "f-new", "other-code", [], new_batch=True, other_runs=other)
    bn._check_unique("tput-safe", "f-new", "batch-code", [], new_batch=False, other_runs=other)


def test_a_branch_with_its_own_batch_follows_only_that_batch():
    local = [_run("control", code="mine")]
    other = [{**_run("x", code="theirs"), "branches": ["arm/y"]}]
    bn._check_unique("arm", "f-new", "mine", local, new_batch=False, other_runs=other)
    with pytest.raises(SystemExit, match="the open batch ran different code"):
        bn._check_unique("arm", "f-new", "theirs", local, new_batch=False, other_runs=other)


def test_closed_failed_and_submission_runs_elsewhere_do_not_block():
    other = [
        {**_run("a", code="x", batch_open=False), "branches": ["b1"]},
        {**_run("b", code="x", status="failed"), "branches": ["b2"]},
        {**_run("c", code="x", kernel="mellamoa/v4-submission", submission=True), "branches": ["b3"]},
    ]
    bn._check_unique("arm", "f-new", "mine", [], new_batch=False, other_runs=other)
