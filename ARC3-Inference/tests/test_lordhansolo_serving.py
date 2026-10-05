"""Offline checks for the serving swap and private-wave notebook path."""

import ast
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference.framework import kaggle
from inference.framework.solver import HarnessSolver


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/1006-prompt-cleaning-engine-swap/lordhansolo.json"
NOTEBOOK = ROOT.parent / "tufa-arc-agi-framework/src/taaf/kaggle/taaf_kaggle_run.ipynb"


@pytest.fixture
def rendered(monkeypatch, tmp_path):
    config = json.loads(CONFIG.read_text())
    monkeypatch.setenv("ARC3_KAGGLE_JSON", json.dumps(config["kaggle"]))
    monkeypatch.setenv("ARC3_HARNESS_ENV_JSON", json.dumps(config["harness_env"]))
    for key, value in config["harness_env"].items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setenv("TAAF_KAGGLE_WORKING_DIR", str(tmp_path))
    monkeypatch.setenv("TAAF_KAGGLE_BUNDLE_DIR", str(tmp_path))
    monkeypatch.setenv("TAAF_KAGGLE_SETUP_ENV", str(tmp_path / "setup_env.json"))
    vocab = tmp_path / "src/ARC3-Inference/configs/draft_vocab_32k.json"
    vocab.parent.mkdir(parents=True)
    vocab.write_bytes((ROOT / "configs/draft_vocab_32k.json").read_bytes())
    command = kaggle.lordhansolo_setup_command()
    assert len(command.encode()) < 128 * 1024
    source = command.split("\n", 1)[1].rsplit("\nPYSETUP", 1)[0]
    ast.parse(source)
    namespace = {}
    # Define the launcher without unpacking its runtime or starting a server.
    exec(source.split("print(f'vLLM runtime dataset path:", 1)[0], namespace)
    return config, source, namespace


def test_server_uses_lord_runtime_and_preserves_request_settings(rendered):
    config, source, namespace = rendered
    command = namespace["build_vllm_server_command"]()
    assert command[command.index("--max-num-seqs") + 1] == "14"
    assert command[command.index("--max-model-len") + 1] == "147072"
    assert command[command.index("--quantization") + 1] == "compressed-tensors"
    assert command[command.index("--prefix-match-unit") + 1] == "128"
    assert command[command.index("--cpu-offload-params") + 1] == "embed_tokens"
    assert json.loads(command[command.index("--speculative-config") + 1]) == {
        "method": "mtp", "num_speculative_tokens": 3, "use_local_argmax_reduction": True,
    }
    assert json.loads(command[command.index("--default-chat-template-kwargs") + 1]) == {
        "preserve_thinking": True,
    }
    assert namespace["VLLM_IMAGE_MAX_PIXELS"] == 640 * 640
    exec(source[source.index("\nsetup_env = {") + 1:], namespace)
    handoff = json.loads(Path(namespace["setup_env_path"]).read_text())
    for key, value in config["harness_env"].items():
        assert handoff[key] == str(value)
    assert handoff["LOCAL_ANALYZER_MODEL_ID"] == "flashnext"
    assert handoff["ARC3_KAGGLE_PRIVATE_WAVES"] == "2"
    assert "LOCAL_ANALYZER_REASONING_EFFORT" not in handoff
    assert "LOCAL_ANALYZER_TARGET_CONTEXT" not in handoff


def test_solver_declares_only_lord_inputs(rendered):
    config, _, _ = rendered
    solver = HarnessSolver(kaggle_enable_vllm=True)
    assert solver.kaggle_dataset_sources == [config["kaggle"]["wheelhouse_dataset_source"]]
    assert solver.kaggle_model_sources == [config["kaggle"]["model_source"]]
    assert len(solver.kaggle_setup_commands) == 1
    assert "vllm-watchdog.pid" in solver.kaggle_teardown_commands[0]
    assert len(config["kaggle"]["public_games"]) == 25
    assert config["environment"]["n_passes"] == 1


def test_insufficient_server_context_fails_at_preparation(rendered, monkeypatch):
    config, _, _ = rendered
    config["kaggle"]["vllm"]["max_model_len"] = 131072
    monkeypatch.setenv("ARC3_KAGGLE_JSON", json.dumps(config["kaggle"]))
    with pytest.raises(ValueError, match="context window plus output"):
        kaggle.lordhansolo_setup_command()


def wave_helpers():
    notebook = json.loads(NOTEBOOK.read_text())
    tree = ast.parse("".join(notebook["cells"][6]["source"]))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_run_in_one_wave", "_configure_private_waves"}]
    import os
    namespace = {"os": os, "time": time, "NOTEBOOK_START_EPOCH": time.time() - 60}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("games, lanes", [(110, 55), (111, 56)])
def test_private_waves_split_live_games_and_remaining_budget(monkeypatch, games, lanes):
    monkeypatch.setenv("ARC3_KAGGLE_PRIVATE_WAVES", "2")
    monkeypatch.setenv("ARC3_KAGGLE_PRIVATE_RUNTIME_MINUTES", "532")
    monkeypatch.setenv("ARC3_SCHEDULER", "priority")
    helpers = wave_helpers()
    benchmark = SimpleNamespace(games=list(range(games)), n_passes=1,
                                solver=SimpleNamespace(concurrency=25, max_runtime_s_per_game=532 * 60))
    helpers["_configure_private_waves"](benchmark)
    assert benchmark.solver.concurrency == lanes
    assert benchmark.solver.max_runtime_s_per_game == pytest.approx((532 * 60 - 60) / 2, abs=1)


def test_public_stays_one_wave_and_ucb_rejects_private_waves(monkeypatch):
    monkeypatch.setenv("ARC3_KAGGLE_PRIVATE_WAVES", "2")
    monkeypatch.setenv("ARC3_SCHEDULER", "ucb")
    helpers = wave_helpers()
    benchmark = SimpleNamespace(games=list(range(25)), n_passes=1,
                                solver=SimpleNamespace(concurrency=10, max_runtime_s_per_game=532 * 60))
    helpers["_run_in_one_wave"](benchmark)
    assert benchmark.solver.concurrency == 25
    assert benchmark.solver.max_runtime_s_per_game == 532 * 60
    with pytest.raises(ValueError, match="priority scheduler"):
        helpers["_configure_private_waves"](benchmark)
    # Existing UCB configurations retain their one-wave path.
    monkeypatch.setenv("ARC3_KAGGLE_PRIVATE_WAVES", "1")
    helpers["_configure_private_waves"](benchmark)
    assert benchmark.solver.concurrency == 25
