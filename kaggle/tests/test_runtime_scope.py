"""What kaggle/build_notebook.py embeds and hashes: only the harness's runtime paths (#122).

Docs, tests, the viewer and inference/tools never run on Kaggle, so a commit touching
only them must leave the patch, the code hash and the notebook fingerprint unchanged.

Run from the repo root:  python -m pytest -q kaggle/tests/test_runtime_scope.py
"""
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bn = _load("build_notebook", REPO / "kaggle" / "build_notebook.py")


def _is_runtime(path: str) -> bool:
    return (
        (path.startswith("ARC3-Inference/inference/") and not path.startswith("ARC3-Inference/inference/tools/"))
        or path.startswith("ARC3-Inference/configs/")
        or path in ("ARC3-Inference/pyproject.toml", "ARC3-Inference/uv.lock")
    )


def _blocks(patch: str) -> dict[str, str]:
    parts = re.split(r"(?m)^(?=diff --git )", patch)
    return {re.match(r"diff --git a/(\S+)", part).group(1): part for part in parts if part.strip()}


def test_patch_is_the_old_patch_without_the_non_runtime_files():
    """Same hunks for every runtime file, byte for byte; nothing else."""
    legacy = _blocks(bn._diff(bn.LEGACY_PATCH_PATHS))
    runtime = _blocks(bn._diff(bn.PATCH_PATHS))
    assert runtime == {path: block for path, block in legacy.items() if _is_runtime(path)}
    assert set(legacy) - set(runtime), "expected docs/tests/viewer/tools in the legacy patch at HEAD"


def test_nothing_on_kaggle_imports_an_excluded_path():
    root = REPO / "ARC3-Inference" / "inference"
    for path in root.rglob("*.py"):
        if (root / "tools") in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+(inference\.tools|viewer|tests)\b", text, re.M), path


def test_notebooks_differ_only_in_the_patch_cell():
    template = json.loads((REPO / "kaggle" / "template.ipynb").read_text(encoding="utf-8"))
    config = bn._arm_config("control")

    def build(patch):
        return bn.build_notebook(
            template, patch, env=config["env"], serve=config["serve"], serve_env=config["serve_env"],
            demo_games=config["demo_games"], demo_minutes=config["demo_minutes"],
        )

    new, old = build(bn._diff(bn.PATCH_PATHS)), build(bn._diff(bn.LEGACY_PATCH_PATHS))
    changed = [i for i, (a, b) in enumerate(zip(new["cells"], old["cells"])) if a != b]
    assert len(changed) == 1
    assert bn._cell_text(new["cells"][changed[0]]).startswith("# Harness patch")


# A throwaway repo with a bundle-base tag, to commit and dirty files in.

def _sh(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
                   cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for rel in ("ARC3-Inference/inference/agent/a.py", "ARC3-Inference/inference/tools/eval.py",
                "ARC3-Inference/README.md", "ARC3-Inference/tests/test_a.py", "ARC3-Inference/viewer/data.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("v0\n")
    _sh(tmp_path, "init", "-q")
    _sh(tmp_path, "add", "-A")
    _sh(tmp_path, "commit", "-qm", "bundle")
    _sh(tmp_path, "tag", "bundle-base")
    (tmp_path / "ARC3-Inference/inference/agent/a.py").write_text("v1\n")
    _sh(tmp_path, "commit", "-qam", "harness change")
    monkeypatch.setattr(bn, "REPO", tmp_path)
    return tmp_path


def _head(repo):
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.mark.parametrize("rel", ["ARC3-Inference/README.md", "ARC3-Inference/tests/test_a.py",
                                 "ARC3-Inference/viewer/data.py", "ARC3-Inference/inference/tools/eval.py"])
def test_a_non_runtime_commit_keeps_the_patch_and_code_hash(repo, rel):
    before, legacy_before = bn.build_patch(), bn.build_legacy_patch()
    (repo / rel).write_text("edited\n")
    _sh(repo, "commit", "-qam", "docs only")
    assert bn.build_patch() == before
    assert bn.build_legacy_patch() != legacy_before  # the old builder would have called this new code


@pytest.mark.parametrize("rel", ["ARC3-Inference/README.md", "ARC3-Inference/inference/tools/eval.py"])
def test_an_uncommitted_non_runtime_edit_does_not_block_a_build(repo, rel):
    before = bn.build_patch()
    (repo / rel).write_text("dirty\n")
    assert bn.build_patch() == before  # it would not reach the notebook either way


def test_an_uncommitted_runtime_edit_is_refused(repo):
    (repo / "ARC3-Inference/inference/agent/a.py").write_text("dirty\n")
    with pytest.raises(SystemExit, match="Uncommitted changes"):
        bn.build_patch()


def _record(commit, code, **extra):
    return {"arm": "control", "version": 1, "fingerprint": "f-control", "commit": commit, "code": code, **extra}


def test_an_old_batch_accepts_a_later_doc_only_commit(repo):
    """The live-batch case: the control was recorded with the whole-ARC3-Inference hash."""
    control_commit = _head(repo)
    control = _record(control_commit, bn._code_hash(bn.build_legacy_patch()))
    (repo / "ARC3-Inference/README.md").write_text("doc fix\n")
    _sh(repo, "commit", "-qam", "doc fix")
    code, legacy = bn._code_hash(bn.build_patch()), bn._code_hash(bn.build_legacy_patch())
    assert legacy != control["code"]
    with pytest.raises(SystemExit, match="open batch ran different code"):  # what the old rule did
        bn._check_unique("tput-safe", "f-new", code, [control], new_batch=False, legacy_code=legacy)
    bn._check_unique("tput-safe", "f-new", code, bn._with_runtime_code([control]),
                     new_batch=False, legacy_code=legacy)


def test_an_old_batch_still_refuses_a_runtime_change(repo):
    control = _record(_head(repo), bn._code_hash(bn.build_legacy_patch()))
    (repo / "ARC3-Inference/inference/agent/a.py").write_text("v2\n")
    _sh(repo, "commit", "-qam", "harness change 2")
    code, legacy = bn._code_hash(bn.build_patch()), bn._code_hash(bn.build_legacy_patch())
    with pytest.raises(SystemExit, match="open batch ran different code"):
        bn._check_unique("tput-safe", "f-new", code, bn._with_runtime_code([control]),
                         new_batch=False, legacy_code=legacy)


def test_an_old_record_from_an_unknown_commit_falls_back_to_the_old_hash(repo):
    legacy = bn._code_hash(bn.build_legacy_patch())
    control = _record("0000000", legacy)
    assert bn._with_runtime_code([control]) == [control]
    bn._check_unique("arm", "f-new", "runtime", bn._with_runtime_code([control]), new_batch=False, legacy_code=legacy)
    with pytest.raises(SystemExit, match="open batch ran different code"):
        bn._check_unique("arm", "f-new", "runtime", [control], new_batch=False, legacy_code="other")


def test_a_duplicate_of_an_old_record_is_caught_by_its_legacy_fingerprint():
    old = _record("abc", "c")
    with pytest.raises(SystemExit, match="not building a duplicate"):
        bn._check_unique("control", "f-runtime-only", "c", [old], new_batch=False, legacy_fingerprint="f-control")


def test_new_records_compare_by_runtime_code():
    run = _record("abc", "whole-tree", runtime_code="rt")
    bn._check_unique("arm", "f-new", "rt", [run], new_batch=False, legacy_code="changed-docs")
    with pytest.raises(SystemExit, match="open batch ran different code"):
        bn._check_unique("arm", "f-new", "rt2", [run], new_batch=False, legacy_code="whole-tree")
