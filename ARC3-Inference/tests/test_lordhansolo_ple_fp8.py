"""The Lord setup script can serve the target with another checkpoint's FP8 PLE table, as
dfranzen/albucino-qwen3.8-flash-next-w4a16-fp8ple carries RadixArk's table next to Intel's body."""
import json
import shutil
from pathlib import Path

import pytest

from inference.framework.lordhansolo import kaggle as lord

TARGET_REF = "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1"
PLE_FP8_REF = "dfranzen/albucino-qwen3.8-flash-next-w4a16-fp8ple/Transformers/default/1"
TABLE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
ROWS = [f"{TABLE}shard_{index}.weight" for index in range(4)]
BODY = {
    "model.language_model.layers.0.mlp.experts.0.w1.qweight": "model-00001-of-00017.safetensors",
    "model.language_model.layers.1.ple.key_proj.weight": "model-00002-of-00017.safetensors",
}


def _checkpoint(path: Path, weight_map: dict, text_config: dict) -> Path:
    path.mkdir()
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (path / "config.json").write_text(json.dumps({"text_config": text_config, "quantization_config": {"quant_method": "auto-round"}}))
    (path / "chat_template.jinja").write_text("")
    for shard in set(weight_map.values()):
        (path / shard).write_text("")
    return path


def _target_map() -> dict:
    return {**BODY, **{name: f"model-ple-{index // 2 + 1:05d}-of-00002.safetensors" for index, name in enumerate(ROWS)}}


def _source_map(rows=ROWS, scale=True) -> dict:
    weight_map = {**BODY, **{name: f"model-plefp8-{index // 2:05d}.safetensors" for index, name in enumerate(rows)}}
    if scale:
        weight_map[TABLE + "weight_scale"] = "model-plefp8-00001.safetensors"
    return weight_map


def _assemble(tmp_path, source_map: dict, source_dtype="float8_e4m3fn", target_map=None):
    """Run the rendered assemble_ple_fp8_target against stand-in checkpoints."""
    script = lord._DUCK_VLLM_SETUP_SCRIPT
    start = script.index("# The tensors of the PLE n-gram table")
    end = script.index("\n\n\n", script.index("def assemble_ple_fp8_target"))
    target = _checkpoint(tmp_path / "intel", target_map or _target_map(), {"ple_embed_dim": 2560})
    source = _checkpoint(tmp_path / "albucino", source_map, {"ple_embed_dim": 2560, "ple_embedding_dtype": source_dtype})
    namespace = {
        "json": json, "Path": Path, "shutil": shutil,
        "PLE_FP8_MODEL_SOURCE_PARTS": tuple(PLE_FP8_REF.split("/")),
        "PLE_FP8_TARGET_DIR": tmp_path / "served",
        "MOUNTED_MODEL_PATH": target,
        "resolve_kaggle_model_path": lambda parts: source,
        "describe_kaggle_input_tree": lambda: "",
    }
    exec(script[start:end], namespace)
    return namespace["assemble_ple_fp8_target"], target, source


def test_served_dir_swaps_only_the_ple_table(tmp_path):
    assemble, target, source = _assemble(tmp_path, _source_map())
    assemble()
    served = tmp_path / "served"
    index = json.loads((served / "model.safetensors.index.json").read_text())["weight_map"]
    assert index == {**BODY, **{name: shard for name, shard in _source_map().items() if TABLE in name}}
    assert sorted(path.name for path in served.glob("*.safetensors")) == [
        "model-00001-of-00017.safetensors", "model-00002-of-00017.safetensors",
        "model-plefp8-00000.safetensors", "model-plefp8-00001.safetensors",
    ]
    assert (served / "model-00001-of-00017.safetensors").resolve() == (target / "model-00001-of-00017.safetensors").resolve()
    assert (served / "model-plefp8-00000.safetensors").resolve() == (source / "model-plefp8-00000.safetensors").resolve()
    assert (served / "chat_template.jinja").resolve() == (target / "chat_template.jinja").resolve()
    config = json.loads((served / "config.json").read_text())
    assert config["text_config"] == {"ple_embed_dim": 2560, "ple_embedding_dtype": "float8_e4m3fn"}
    assert config["quantization_config"] == {"quant_method": "auto-round"}


@pytest.mark.parametrize("source_map, dtype, match", [
    (_source_map(rows=ROWS[:3]), "float8_e4m3fn", "cannot replace"),
    (_source_map(scale=False), "float8_e4m3fn", "cannot replace"),
    (_source_map(), None, "not float8_e4m3fn"),
    ({**_source_map(), "lm_head.weight": "model-plefp8-00000.safetensors"}, "float8_e4m3fn", "also hold other tensors"),
])
def test_unusable_fp8_table_fails(tmp_path, source_map, dtype, match):
    assemble, _, _ = _assemble(tmp_path, source_map, source_dtype=dtype)
    with pytest.raises(RuntimeError, match=match):
        assemble()


def test_runtime_check_needs_the_dtype_switch(tmp_path):
    script = lord._DUCK_VLLM_SETUP_SCRIPT
    start = script.index("# Where the e975732 rc3 runtime picks the FP8 PLE method")
    end = script.index("\n\n\n", script.index("def check_runtime_reads_fp8_ple"))
    ple_py = tmp_path / "dist" / "vllm" / "models" / "qwen4_exp" / "nvidia" / "ngram_embedding.py"
    ple_py.parent.mkdir(parents=True)
    namespace = {"IMAGE_RUNTIME_ROOT": tmp_path, "PLE_FP8_MODEL_SOURCE_PARTS": tuple(PLE_FP8_REF.split("/"))}
    exec(script[start:end], namespace)
    check = namespace["check_runtime_reads_fp8_ple"]
    ple_py.write_text(
        '        if embedding_dtype == "float8_e4m3fn":\n'
        "            return Qwen4ExpPLEFp8EmbeddingMethod()\n"
        '            getattr(config, "ple_embedding_dtype", None),\n'
    )
    check({"dist_packages": "dist"})
    ple_py.write_text("        if quant_config is None:\n")
    with pytest.raises(RuntimeError, match="does not select the FP8 PLE method"):
        check({"dist_packages": "dist"})


def test_render_mounts_the_fp8_source_and_serves_the_swapped_dir(monkeypatch):
    monkeypatch.setenv("LOCAL_ANALYZER_PROVIDER", "vllm")
    config = lord.DuckKaggleVllmConfig(model_source=TARGET_REF, ple_fp8_model_source=PLE_FP8_REF)
    script = lord.duck_kaggle_setup_command(config)
    assert f"PLE_FP8_MODEL_SOURCE_PARTS = {tuple(PLE_FP8_REF.split('/'))!r}" in script
    assert "if PLE_FP8_MODEL_SOURCE_PARTS:\n    MODEL_PATH = PLE_FP8_TARGET_DIR" in script
    assert script.index("\nassemble_ple_fp8_target()\n") < script.index("\nstart_vllm_server()\n")
    assert lord.duck_kaggle_model_sources(config)[-1] == PLE_FP8_REF
    without = lord.duck_kaggle_setup_command(lord.DuckKaggleVllmConfig(model_source=TARGET_REF))
    assert "PLE_FP8_MODEL_SOURCE_PARTS = None" in without
