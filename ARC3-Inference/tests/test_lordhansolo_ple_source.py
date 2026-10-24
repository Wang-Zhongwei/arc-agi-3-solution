"""The Lord setup script serves a partial target (an RFT merge published without its PLE table) with
the PLE files of the base model it was trained from."""
import json
import shutil
from pathlib import Path

import pytest

from inference.framework.lordhansolo import kaggle as lord

TARGET_REF = "jonathanwang2022/qwen3-8-flash-next-w4a16-rft-v1/Transformers/default/1"
BASE_REF = "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1"


def _assemble(tmp_path, weight_map: dict, target_files: list[str], base_files: list[str]):
    """Run the rendered assemble_target_model against stand-in checkpoints."""
    script = lord._DUCK_VLLM_SETUP_SCRIPT
    start = script.index("def assemble_target_model")
    end = script.index("\n\n\n", start)
    target, base = tmp_path / "rft", tmp_path / "base"
    for folder, files in ((target, target_files), (base, base_files)):
        folder.mkdir()
        (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        for name in files:
            (folder / name).write_text(folder.name)
    namespace = {
        "json": json, "Path": Path, "shutil": shutil,
        "PLE_MODEL_SOURCE_PARTS": tuple(BASE_REF.split("/")),
        "MOUNTED_MODEL_PATH": target,
        "TARGET_MODEL_DIR": tmp_path / "served",
        "resolve_kaggle_model_path": lambda parts: base,
        "describe_kaggle_input_tree": lambda: "",
    }
    exec(script[start:end], namespace)
    return namespace["assemble_target_model"], tmp_path / "served"


WEIGHT_MAP = {
    "model.language_model.layers.0.self_attn.q_proj.weight": "model-00001-of-00017.safetensors",
    "model.language_model.layers.0.ngram_embedding.0.weight": "model-ple-00001-of-00022.safetensors",
    "model.language_model.layers.1.ngram_embedding.0.weight": "model-ple-00002-of-00022.safetensors",
}


def test_merged_weights_come_from_the_target_and_ple_from_the_base(tmp_path):
    assemble, served = _assemble(
        tmp_path, WEIGHT_MAP,
        target_files=["config.json", "model-00001-of-00017.safetensors"],
        base_files=["config.json", "model-00001-of-00017.safetensors",
                    "model-ple-00001-of-00022.safetensors", "model-ple-00002-of-00022.safetensors"],
    )
    assemble()
    assert (served / "model-00001-of-00017.safetensors").read_text() == "rft"
    assert (served / "config.json").read_text() == "rft"
    assert (served / "model-ple-00001-of-00022.safetensors").read_text() == "base"
    assert (served / "model-ple-00002-of-00022.safetensors").read_text() == "base"
    assert json.loads((served / "model.safetensors.index.json").read_text())["weight_map"] == WEIGHT_MAP


def test_a_missing_weight_shard_is_not_taken_from_the_base(tmp_path):
    assemble, _ = _assemble(
        tmp_path, WEIGHT_MAP,
        target_files=["config.json"],
        base_files=["model-00001-of-00017.safetensors",
                    "model-ple-00001-of-00022.safetensors", "model-ple-00002-of-00022.safetensors"],
    )
    with pytest.raises(FileNotFoundError, match="model-00001-of-00017"):
        assemble()


def test_render_mounts_the_ple_model_and_serves_the_assembled_dir(monkeypatch):
    monkeypatch.setenv("LOCAL_ANALYZER_PROVIDER", "vllm")
    config = lord.DuckKaggleVllmConfig(model_source=TARGET_REF, ple_model_source=BASE_REF)
    script = lord.duck_kaggle_setup_command(config)
    assert f"PLE_MODEL_SOURCE_PARTS = {tuple(BASE_REF.split('/'))!r}" in script
    assert "elif PLE_MODEL_SOURCE_PARTS:\n    MODEL_PATH = TARGET_MODEL_DIR" in script
    assert script.index("assemble_target_model()\nstart_vllm_server()") > 0
    assert lord.duck_kaggle_model_sources(config) == [TARGET_REF, BASE_REF]
    without = lord.duck_kaggle_setup_command(lord.DuckKaggleVllmConfig(model_source=BASE_REF))
    assert "PLE_MODEL_SOURCE_PARTS = None" in without
