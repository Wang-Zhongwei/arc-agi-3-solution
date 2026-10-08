"""Offline compatibility checks; numerical kernels still require a GPU run."""

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from inference.framework import kaggle, pennyroyal_patch

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/adaptive-context"


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


@pytest.mark.parametrize("name,streams,adaptive", [("control-10", 10, False),
                                                  ("adaptive-10", 10, True),
                                                  ("adaptive-16", 16, True)])
def test_config_export_and_pennyroyal_handoff(monkeypatch, name, streams, adaptive):
    path = CONFIGS / f"{name}.json"
    cfg = json.loads(path.read_text())
    # Run Make's actual config export, without a dataset upload or packaging.
    recipe = "print-test-env:\n\t@$(PYTHON) -c 'import json,os; print(json.dumps({k: os.environ[k] for k in [\"ARC3_KAGGLE_JSON\", \"ARC3_HARNESS_ENV_JSON\"]}))'\n"
    result = subprocess.run(["make", "--no-print-directory", "-s", "-f", "Makefile", "-f", "-",
                             f"CONFIG_PATH={path}", f"PYTHON={sys.executable}", "print-test-env"],
                            input=recipe, cwd=ROOT, capture_output=True, text=True, check=True)
    for key, value in json.loads(result.stdout).items():
        monkeypatch.setenv(key, value)
    command = kaggle.pennyroyal_setup_command()
    assert "pennyroyal_launch.py" in command
    env = kaggle.harness_env()
    assert int(env["ARC3_MAX_ACTIVE_STREAMS"]) == streams
    assert env["ARC3_ADAPTIVE_CONTEXT"] == str(int(adaptive))
    assert int(env["LOCAL_ANALYZER_THINKING_TOKEN_BUDGET"]) == 10240
    assert int(env["LOCAL_ANALYZER_MAX_OUTPUT"]) == 12288
    assert cfg["kaggle"]["sglang"]["MAXREQ"] == streams
    assert cfg["kaggle"]["sglang"]["CUDAGRAPH_MAXBS"] == streams
    assert cfg["kaggle"]["wheelhouse_dataset_source"] == "dfranzen/pennyroyal-v253"
    assert not cfg["kaggle"]["sglang"]["OFFLOAD_EMBED_TOKENS"]
    assert not cfg["kaggle"]["sglang"]["WIDE_BF16_GEMM"]
    assert len(cfg["kaggle"]["public_games"]) == 25
    assert cfg["environment"]["n_passes"] == 1
    if adaptive:
        assert int(env["ARC3_CONTEXT_EXPANDED_HIGH"]) + int(env["LOCAL_ANALYZER_MAX_OUTPUT"]) + 512 <= int(env["LOCAL_ANALYZER_CONTEXT_WINDOW"])


@pytest.mark.parametrize("change,error", [
    ({"MAXREQ": 10}, "active streams"),
    ({"CUDAGRAPH_MAXBS": 10}, "CUDAGRAPH"),
    ({"CTX": 131072}, "context window"),
])
def test_invalid_capacity_is_rejected_before_packaging(monkeypatch, change, error):
    cfg = json.loads((CONFIGS / "adaptive-16.json").read_text())
    cfg["kaggle"]["sglang"].update(change)
    monkeypatch.setenv("ARC3_KAGGLE_JSON", json.dumps(cfg["kaggle"]))
    monkeypatch.setenv("ARC3_HARNESS_ENV_JSON", json.dumps(cfg["harness_env"]))
    with pytest.raises(ValueError, match=error):
        kaggle.pennyroyal_setup_command()


def test_bad_input_headroom_is_rejected_before_packaging(monkeypatch):
    cfg = json.loads((CONFIGS / "adaptive-16.json").read_text())
    cfg["harness_env"]["LOCAL_ANALYZER_CONTEXT_WINDOW"] = 131072
    monkeypatch.setenv("ARC3_KAGGLE_JSON", json.dumps(cfg["kaggle"]))
    monkeypatch.setenv("ARC3_HARNESS_ENV_JSON", json.dumps(cfg["harness_env"]))
    with pytest.raises(ValueError, match="output"):
        kaggle.pennyroyal_setup_command()


def test_comparison_workload_and_sampling_are_identical():
    configs = [json.loads((CONFIGS / f"{name}.json").read_text())
               for name in ("control-10", "adaptive-10", "adaptive-16")]
    for section, keys in (("kaggle", ("public_games", "public_runtime_minutes", "private_runtime_minutes")),
                          ("harness_env", ("LOCAL_ANALYZER_TEMPERATURE", "LOCAL_ANALYZER_MAX_OUTPUT", "ARC3_SCHEDULER"))):
        for key in keys:
            assert all(cfg[section][key] == configs[0][section][key] for cfg in configs)


def test_graphs_include_16_streams_and_preserve_original_sizes():
    source = (ROOT / "inference/framework/pennyroyal_launch.py").read_text()
    tree = ast.parse(source)
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "graph_bs" for t in node.targets)]
    for capacity in (10, 16):
        ns = {"CFG": {"MAXREQ": capacity, "CUDAGRAPH_MAXBS": capacity}}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), "launcher", "exec"), ns)
        assert ns["graph_bs"] == [1, 2, 4, 7, 8, 9, 10] + ([16] if capacity == 16 else [])
