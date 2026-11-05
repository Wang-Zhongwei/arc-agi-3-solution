"""int4 KV on the Lord runtime: the setup installs the vendored kernels and patches the stock QSA owner."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from inference.framework.lordhansolo import kaggle as lord

DIST = "dist/"
INT4 = "int4_per_token_head"
PROJECT = Path(lord.__file__).resolve().parent / "int4_kv"


def _namespace(root, dtype=INT4, draft_dtype=""):
    script = lord._DUCK_VLLM_SETUP_SCRIPT
    start = script.index("INT4_KV_DTYPE = ")
    end = script.index("# Where the e975732 rc3 runtime picks the FP8 PLE")
    source = script[start:end].replace("__PROJECT_DIR_NAME__", "'project'")
    namespace = {
        "Path": Path, "json": json, "os": os, "shutil": shutil, "subprocess": subprocess, "sys": sys,
        "IMAGE_RUNTIME_ROOT": root, "VLLM_KV_CACHE_DTYPE": dtype, "VLLM_DRAFT_KV_CACHE_DTYPE": draft_dtype,
    }
    exec(source, namespace)
    namespace["INT4_KV_DIR"] = PROJECT
    return namespace


def _patches():
    return json.loads((PROJECT / "runtime_patches.json").read_text())


def _stand_in_runtime(root):
    (root / DIST / "vllm").mkdir(parents=True)
    for patch in _patches():
        path = root / DIST / patch["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# filler\n".join(["", *(edit["anchor"] for edit in patch["replacements"]), ""]))


def test_install_copies_the_kernels_and_patches_each_anchor(tmp_path):
    namespace = _namespace(tmp_path)
    _stand_in_runtime(tmp_path)
    namespace["install_int4_kv"]({"dist_packages": DIST})
    assert (tmp_path / DIST / "vllm" / "taaf_qsa_kv8.py").is_file()
    for patch in _patches():
        text = (tmp_path / DIST / patch["path"]).read_text()
        for edit in patch["replacements"]:
            assert edit["replacement"] in text


def test_other_dtypes_leave_the_runtime_alone(tmp_path):
    namespace = _namespace(tmp_path, dtype="fp8_e4m3")
    namespace["install_int4_kv"]({"dist_packages": DIST})
    namespace["check_int4_kv_kernels"]({})
    assert not (tmp_path / DIST).exists()


def test_a_missing_anchor_fails_before_the_load(tmp_path):
    namespace = _namespace(tmp_path)
    _stand_in_runtime(tmp_path)
    (tmp_path / DIST / _patches()[0]["path"]).write_text("")
    with pytest.raises(RuntimeError, match="int4 KV anchor appears 0 times"):
        namespace["install_int4_kv"]({"dist_packages": DIST})


def test_a_draft_dtype_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="inherit int4"):
        _namespace(tmp_path, draft_dtype="fp8_e4m3")["install_int4_kv"]({"dist_packages": DIST})


def test_a_failed_preflight_stops_the_setup(tmp_path, monkeypatch):
    namespace = _namespace(tmp_path)
    monkeypatch.setitem(namespace, "INT4_KV_DIR", tmp_path)
    (tmp_path / "preflight.py").write_text("raise SystemExit(3)\n")
    with pytest.raises(RuntimeError, match="exit code 3"):
        namespace["check_int4_kv_kernels"](dict(os.environ))


@pytest.mark.skipif(not os.environ.get("ARC3_LORD_RUNTIME_TREE"), reason="needs the e975732 vllm/ tree")
def test_anchors_match_the_real_runtime(tmp_path):
    namespace = _namespace(tmp_path)
    tree = Path(os.environ["ARC3_LORD_RUNTIME_TREE"])
    (tmp_path / DIST / "vllm").mkdir(parents=True)
    for patch in _patches():
        target = tmp_path / DIST / patch["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tree.parent / patch["path"], target)
    namespace["install_int4_kv"]({"dist_packages": DIST})
    for patch in _patches():
        compile((tmp_path / DIST / patch["path"]).read_text(), patch["path"], "exec")


def test_the_config_keeps_the_drafter_on_int4_and_fits_the_measured_block_size():
    config = json.loads((Path(lord.__file__).parents[3] / "configs" / "int4-kv" / "v3.json").read_text())
    vllm = config["kaggle"]["vllm"]
    assert vllm["kv_cache_dtype"] == INT4
    assert not vllm.get("draft_kv_cache_dtype")
    # int4-kv v1 logged group block sizes of 1664: alignment sizes the block from the non-QSA 1024 B page.
    assert 1664 % vllm["prefix_match_unit"] == 0
    assert vllm["max_num_seqs"] * (1 + vllm["mtp_tokens"]) in vllm["cudagraph_capture_sizes"]


def test_render_runs_install_before_the_load_and_preflight_before_the_server(monkeypatch):
    monkeypatch.setenv("LOCAL_ANALYZER_PROVIDER", "vllm")
    script = lord.duck_kaggle_setup_command(lord.DuckKaggleVllmConfig(kv_cache_dtype=INT4))
    assert script.index("    install_int4_kv(manifest)\n") < script.index("    cmd = build_vllm_server_command(")
    assert script.index("    check_int4_kv_kernels(server_env)\n") < script.index("    VLLM_SERVER_LOG.parent.mkdir(")
    # The setup command is one /bin/sh argument, which Linux limits to 128 KiB.
    assert len(script.encode()) < 120 * 1024
