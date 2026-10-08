"""Offline compatibility checks; numerical kernels still require a GPU run."""

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from inference.framework import kaggle, pennyroyal_patch

ROOT = Path(__file__).resolve().parents[1]


def test_patch_is_idempotent_and_fails_before_partial_write(tmp_path):
    model = tmp_path / "srt/models/qwen4_exp.py"
    gemm = tmp_path / "kernels/ops/gemm/sm120_lowm_bf16_gemm.py"
    model.parent.mkdir(parents=True)
    gemm.parent.mkdir(parents=True)
    original_model = (
        "class Qwen4ExpModel:\n" + pennyroyal_patch.BUILD_ANCHOR + "\n            10, 20)\n"
        "    def __init__(self, config, quant_config, prefix, is_nextn):\n"
        + pennyroyal_patch.INIT_ANCHOR
    )
    model.write_text(original_model)
    gemm.write_text("# unexpected kernel\n")
    with pytest.raises(ValueError, match="M <= 32"):
        pennyroyal_patch.install(tmp_path, embedding=True, wide_gemm=True)
    assert model.read_text() == original_model
    gemm.write_text("_MAX_M = 32\ndef use_sm120_lowm_bf16_gemm(m,n,k): return m <= 32\n"
                    "def sm120_lowm_bf16_gemm(x,w): return 'baseline'\n")
    pennyroyal_patch.install(tmp_path, embedding=True, wide_gemm=True)
    first = {p: p.read_bytes() for p in tmp_path.rglob("*.py")}
    pennyroyal_patch.install(tmp_path, embedding=True, wide_gemm=True)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.py")} == first
    ns = {}
    exec(gemm.read_text(), ns)
    assert ns["use_sm120_lowm_bf16_gemm"](32, 512, 2560)
    assert not ns["use_sm120_lowm_bf16_gemm"](56, 512, 2560)
    assert ns["sm120_lowm_bf16_gemm"](None, None) == "baseline"


def test_wide_hook_keeps_small_and_unknown_shapes(monkeypatch):
    import types
    module = types.ModuleType("sglang.kernels.ops.gemm.arc3_wide_bf16_gemm")
    module.eligible = lambda m, n, k: m in (56, 64) and (n, k) == (512, 2560)
    module.gemm = lambda x, w: "wide"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("ARC3_WIDE_BF16_GEMM", "1")
    ns = {"use_sm120_lowm_bf16_gemm": lambda m, n, k: m <= 32,
          "sm120_lowm_bf16_gemm": lambda x, w: "small"}
    exec(pennyroyal_patch.WIDE_HOOK, ns)
    for m, expected in [(16, True), (32, True), (56, True), (64, True), (68, False)]:
        assert ns["use_sm120_lowm_bf16_gemm"](m, 512, 2560) == expected
    assert not ns["use_sm120_lowm_bf16_gemm"](56, 100, 100)
    for m, expected in [(16, "small"), (56, "wide"), (64, "wide")]:
        x = types.SimpleNamespace(shape=(m, 2560), numel=lambda: m * 2560)
        assert ns["sm120_lowm_bf16_gemm"](x, types.SimpleNamespace(shape=(512, 2560))) == expected


def test_new_baseline_v3_differs_from_v2_only_in_startup_and_offload(monkeypatch):
    v2 = json.loads((ROOT / "configs/new-baseline/v2.json").read_text())
    v3 = json.loads((ROOT / "configs/new-baseline/v3.json").read_text())
    assert v3["kaggle"].pop("sglang") == {"OFFLOAD_EMBED_TOKENS": True}
    assert (v2["kaggle"]["server_startup_timeout_s"], v3["kaggle"]["server_startup_timeout_s"]) == (720, 1200)
    v3["kaggle"]["server_startup_timeout_s"] = 720
    assert v3 == v2
    v3["kaggle"]["sglang"] = {"OFFLOAD_EMBED_TOKENS": True}
    monkeypatch.setenv("ARC3_KAGGLE_JSON", json.dumps(v3["kaggle"]))
    monkeypatch.setenv("ARC3_HARNESS_ENV_JSON", json.dumps(v3["harness_env"]))
    command = kaggle.pennyroyal_setup_command()
    spec = json.loads(command.split("ARC3_PENNYROYAL_SPEC=", 1)[1].split(" \"$PYTHON\"", 1)[0].strip("'"))
    assert spec["sglang"] == {"OFFLOAD_EMBED_TOKENS": True}


@pytest.mark.parametrize("config,threads,early,prefetch", [
    ("new-baseline/v3", 3, False, None),
    ("startup-read-threads/v1", 12, True, None),
    ("startup-read-threads/v2", 6, False, "8"),
    ("startup-read-threads/v3", 0, False, "16"),
])
def test_model_read_threads_reach_the_launcher(config, threads, early, prefetch):
    sglang = json.loads((ROOT / f"configs/{config}.json").read_text())["kaggle"].get("sglang", {})
    launcher = (ROOT / "inference/framework/pennyroyal_launch.py").read_text()
    ns = {"SPEC": {"sglang": sglang}, "DRAFT_MODEL_DIR": "draft", "SERVED_MODEL_NAME": "flashnext"}
    exec(launcher[launcher.index("_READ = "):launcher.index("\n# Daemon:")], ns)
    exec(launcher[launcher.index("CFG = dict("):launcher.index("\nVENV = ")], ns)
    assert (ns["MODEL_PRECACHE_THREADS"], ns["MODEL_PRECACHE_EARLY"]) == (threads, early)
    line = next(l for l in launcher.splitlines() if "--weight-loader-prefetch-num-threads" in l)
    ns["args"] = []
    exec(line, ns)
    assert ns["args"] == ([] if prefetch is None else ["--weight-loader-prefetch-num-threads", prefetch])


def test_quick_read_runs_differ_from_v3_only_in_read_and_runtime():
    v3 = json.loads((ROOT / "configs/new-baseline/v3.json").read_text())
    for name in ("v1", "v2", "v3"):
        quick = json.loads((ROOT / f"configs/startup-read-threads/{name}.json").read_text())
        assert quick["kaggle"]["public_runtime_minutes"] == 5
        for key in ("kernel_slug", "kernel_title", "dataset_slug", "public_runtime_minutes", "sglang"):
            quick["kaggle"][key] = v3["kaggle"][key]
        assert quick == v3
