"""Tests for the serving cell kaggle/build_notebook.py builds (issue #6).

Dry builds only: nothing is pushed and kaggle/runs.json is not read.

Run from ARC3-Inference:  python -m pytest -q ../kaggle/tests/test_build_serving.py
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bn = _load("build_notebook", REPO / "kaggle" / "build_notebook.py")
jc = _load("jit_cache_source", REPO / "serving" / "kaggle" / "jit_cache.py")
TEMPLATE = json.loads((REPO / "kaggle" / "template.ipynb").read_text(encoding="utf-8"))
PATCH = "diff --git a/ARC3-Inference/x b/ARC3-Inference/x\n"  # stands in for the harness diff


def test_embedded_jit_cache_is_the_module_source():
    namespace = {}
    exec(bn._jit_cache_block(), namespace)
    embedded = namespace["jit_cache"]
    assert embedded.__name__ == "jit_cache"
    public = sorted(n for n in vars(jc) if not n.startswith("__"))
    assert sorted(n for n in vars(embedded) if not n.startswith("__")) == public
    assert embedded.RUNTIME_PROBE == jc.RUNTIME_PROBE
    assert embedded.candidate_dirs("/kaggle/input/datasets/o/s") == jc.candidate_dirs("/kaggle/input/datasets/o/s")


def test_launcher_embeds_the_module_once():
    text = bn._serving_launcher()
    assert bn.JIT_CACHE_ANCHOR not in text
    assert text.count('jit_cache = types.ModuleType("jit_cache")') == 1
    compile(text, "launch_sglang.py", "exec")


# ---------------------------------------------------------------- arm runtime patches

def _embedded_patches(launcher):
    """The SGLANG_RUNTIME_PATCHES table the launcher text defines (runs only that statement)."""
    start = launcher.index("SGLANG_RUNTIME_PATCHES = ")
    end = launcher.index("\n", launcher.index('""")))', start))
    namespace = {}
    exec("import base64, gzip, json\n" + launcher[start:end], namespace)
    return namespace["SGLANG_RUNTIME_PATCHES"]


def test_no_arm_patches_leave_the_launcher_unchanged():
    assert bn._serving_launcher() == bn._serving_launcher(None) == bn._serving_launcher({})


def test_arm_patches_join_the_shared_ones(tmp_path, monkeypatch):
    (tmp_path / "10-shared.py").write_text("print('shared')\n")
    monkeypatch.setattr(bn, "RUNTIME_PATCHES", tmp_path)
    arm = bn._bench_patches(["m64.py", "nosplitk.py"], "arm 'x'")
    launcher = bn._serving_launcher(arm)
    table = _embedded_patches(launcher)
    assert list(sorted(table)) == ["10-shared.py", "m64.py", "nosplitk.py"]
    assert table["m64.py"] == (bn.BENCH_PATCHES / "m64.py").read_text(encoding="utf-8")
    assert "embedded from serving/runtime_patches/ and the arm's serving/bench_patches/" in launcher
    compile(launcher, "launch_sglang.py", "exec")


def test_arm_patches_alone_are_embedded(tmp_path, monkeypatch):
    # an empty shared folder: the arm's patch is the whole table
    monkeypatch.setattr(bn, "RUNTIME_PATCHES", tmp_path)
    table = _embedded_patches(bn._serving_launcher(bn._bench_patches(["m64.py"], "arm 'x'")))
    assert set(table) == {"m64.py"}


def test_arm_patch_that_would_run_before_a_shared_one_is_refused(tmp_path, monkeypatch):
    (tmp_path / "zz-late.py").write_text("print('late')\n")
    monkeypatch.setattr(bn, "RUNTIME_PATCHES", tmp_path)
    with pytest.raises(SystemExit, match="would apply before shared"):
        bn._serving_launcher({"m64.py": "x"})


def test_arm_patch_reusing_a_shared_name_is_refused(tmp_path, monkeypatch):
    (tmp_path / "m64.py").write_text("print('shared')\n")
    monkeypatch.setattr(bn, "RUNTIME_PATCHES", tmp_path)
    with pytest.raises(SystemExit, match="reuse shared patch names"):
        bn._serving_launcher({"m64.py": "x"})


def test_missing_arm_patch_is_refused():
    with pytest.raises(SystemExit, match="arm 'x': runtime patch"):
        bn._bench_patches(["no-such-patch.py"], "arm 'x'")


def test_arm_config_reads_runtime_patches(tmp_path, monkeypatch):
    (tmp_path / "arms.json").write_text(json.dumps({
        "defaults": {}, "arms": {"plain": {}, "lowm": {"runtime_patches": ["m64.py", "nosplitk.py"]}}}))
    monkeypatch.setattr(bn, "HERE", tmp_path)
    assert bn._arm_config("plain")["runtime_patches"] == []
    assert bn._arm_config("lowm")["runtime_patches"] == ["m64.py", "nosplitk.py"]


# ---------------------------------------------------------------- arm datasets

META = json.loads((REPO / "kaggle" / "template-metadata.json").read_text(encoding="utf-8"))


def test_metadata_without_datasets_keeps_the_template_sources():
    meta = bn.build_metadata(META, "v4-x", "v4-x", "v4-x.ipynb")
    assert meta["dataset_sources"] == META["dataset_sources"]
    assert bn.build_metadata(META, "v4-x", "v4-x", "v4-x.ipynb", datasets=[]) == meta


def test_arm_datasets_are_appended_once():
    meta = bn.build_metadata(META, "v4-x", "v4-x", "v4-x.ipynb",
                             datasets=["mellamoa/sgl-jit-cache", META["dataset_sources"][0]])
    assert meta["dataset_sources"] == META["dataset_sources"] + ["mellamoa/sgl-jit-cache"]
    assert META["dataset_sources"] == json.loads(
        (REPO / "kaggle" / "template-metadata.json").read_text(encoding="utf-8"))["dataset_sources"]


def test_arm_datasets_must_be_owner_slug():
    with pytest.raises(SystemExit):
        bn.build_metadata(META, "v4-x", "v4-x", "v4-x.ipynb", datasets=["sgl-jit-cache"])


def test_arm_config_reads_datasets(tmp_path, monkeypatch):
    (tmp_path / "arms.json").write_text(json.dumps({
        "defaults": {}, "arms": {"plain": {}, "cached": {"datasets": ["mellamoa/sgl-jit-cache"]}}}))
    monkeypatch.setattr(bn, "HERE", tmp_path)
    assert bn._arm_config("plain")["datasets"] == []
    assert bn._arm_config("cached")["datasets"] == ["mellamoa/sgl-jit-cache"]


def test_fingerprint_counts_extra_datasets_only_when_present():
    notebook = {"cells": [{"source": "x"}]}
    plain = hashlib.sha256(json.dumps(notebook, sort_keys=True).encode()).hexdigest()[:16]
    assert bn._fingerprint(notebook) == bn._fingerprint(notebook, []) == plain
    assert bn._fingerprint(notebook, ["mellamoa/sgl-jit-cache"]) != plain


# ---------------------------------------------------------------- dry builds of the arms

ARMS = json.loads((REPO / "kaggle" / "arms.json").read_text(encoding="utf-8"))["arms"]


def _serving_cell(arm):
    config = bn._arm_config(arm)
    nb = bn.build_notebook(TEMPLATE, PATCH, env={**config["env"], "TAAF_ARM": arm}, serve=config["serve"],
                           serve_env=config["serve_env"],
                           runtime_patches=bn._bench_patches(config["runtime_patches"], f"arm {arm!r}"),
                           demo_games=config["demo_games"], demo_minutes=config["demo_minutes"])
    cells = [bn._cell_text(c) for c in nb["cells"] if c["cell_type"] == "code" and bn.SERVE_ANCHOR in bn._cell_text(c)]
    assert len(cells) == 1
    return cells[0]


def _cfg(cell):
    """CFG as the serving cell leaves it after the arm's overrides (runs only the top of the cell)."""
    head = cell[:cell.index(bn.SERVE_ANCHOR)]
    namespace = {"WORKING_DIR": "/kaggle/working", "SERVED_MODEL_NAME": "flashnext"}
    exec(compile(head, "serving cell", "exec"), namespace)
    return namespace["CFG"]


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_every_arm_builds_a_serving_cell_that_compiles(arm):
    cell = _serving_cell(arm)
    compile(cell, f"serving cell ({arm})", "exec")
    cfg = _cfg(cell)
    for key, value in ARMS[arm].get("serve", {}).items():
        assert cfg[key] == value


def test_new_cfg_keys_default_off():
    cfg = _cfg(_serving_cell("control"))
    assert cfg["JIT_CACHE_SAVE"] is False and cfg["JIT_CACHE_DIR"] is None
    assert cfg["AUTOTUNE"] is True


@pytest.mark.parametrize("arm, expected", [
    ("coldstart-save", {"JIT_CACHE_SAVE": True, "JIT_CACHE_DIR": None, "AUTOTUNE": True}),
])
def test_cold_start_arms(arm, expected):
    cfg = _cfg(_serving_cell(arm))
    assert {k: cfg[k] for k in expected} == expected
    assert bn._arm_config(arm)["env"] == {"ARC3_MAX_ACTIVE_STREAMS": "4"}
    assert bn._arm_config(arm)["demo_games"] == "balanced"

