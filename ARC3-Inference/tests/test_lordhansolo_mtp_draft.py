"""The Lord setup script can draft with another checkpoint's MTP head, as
matijaludvig/arc3-vcintel-lorddraft-25x1 drafts for Intel's W4A16 target with
Lord's NVFP4 head."""
import json
import re
import shutil
from pathlib import Path

import pytest

from inference.framework.lordhansolo import kaggle as lord

DRAFT_REF = "lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8/PyTorch/hf-mixed-mtp-nvfp4/1"
TARGET_REF = "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1"


def _assemble(tmp_path, weight_map: dict):
    """Run the rendered assemble_mtp_draft against a stand-in checkpoint."""
    script = lord._DUCK_VLLM_SETUP_SCRIPT
    start = script.index("DRAFT_SHARED_TENSORS = (")
    end = script.index("\n\n\n", script.index("def assemble_mtp_draft"))
    source = tmp_path / "lord"
    source.mkdir()
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (source / "config.json").write_text("{}")
    for shard in set(weight_map.values()):
        (source / shard).write_text("")
    namespace = {
        "json": json, "Path": Path, "shutil": shutil,
        "DRAFT_MODEL_SOURCE_PARTS": tuple(DRAFT_REF.split("/")),
        "DRAFT_MODEL_DIR": tmp_path / "draft",
        "resolve_kaggle_model_path": lambda parts: source,
        "describe_kaggle_input_tree": lambda: "",
    }
    exec(script[start:end], namespace)
    return namespace["assemble_mtp_draft"], source


def test_draft_dir_holds_only_the_mtp_head(tmp_path):
    weight_map = {
        "mtp.layers.0.mlp.experts.0.w1": "mtp-experts-nvfp4.safetensors",
        "mtp.fc.weight": "carry-model-bf16-00012.safetensors",
        "lm_head.weight": "carry-model-bf16-00011.safetensors",
        "model.language_model.embed_tokens.weight": "carry-model-bf16-00010.safetensors",
        "model.language_model.layers.0.mlp.experts.0.w1": "model-00001.safetensors",
    }
    assemble, source = _assemble(tmp_path, weight_map)
    draft = assemble()
    assert draft == tmp_path / "draft"
    index = json.loads((draft / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(index) == set(weight_map) - {"model.language_model.layers.0.mlp.experts.0.w1"}
    assert sorted(p.name for p in draft.glob("*.safetensors")) == [
        "carry-model-bf16-00010.safetensors", "carry-model-bf16-00011.safetensors",
        "carry-model-bf16-00012.safetensors", "mtp-experts-nvfp4.safetensors",
    ]
    assert (draft / "config.json").resolve() == (source / "config.json").resolve()


def test_checkpoint_without_mtp_head_fails(tmp_path):
    assemble, _ = _assemble(tmp_path, {"lm_head.weight": "a.safetensors",
                                       "model.language_model.embed_tokens.weight": "a.safetensors"})
    with pytest.raises(RuntimeError, match="cannot draft"):
        assemble()


def test_render_passes_the_draft_dir_to_the_mtp_speculative_config(monkeypatch):
    monkeypatch.setenv("LOCAL_ANALYZER_PROVIDER", "vllm")
    with_draft = lord.duck_kaggle_setup_command(lord.DuckKaggleVllmConfig(
        model_source=TARGET_REF, draft_model_source=DRAFT_REF))
    assert f"DRAFT_MODEL_SOURCE_PARTS = {tuple(DRAFT_REF.split('/'))!r}" in with_draft
    assert re.search(r"draft_model_dir = assemble_mtp_draft\(\)\n.*\n\s+cmd = build_vllm_server_command\(draft_model_dir\)", with_draft)
    assert "speculative_config['model'] = str(draft_model_dir)" in with_draft
    without = lord.duck_kaggle_setup_command(lord.DuckKaggleVllmConfig(model_source=TARGET_REF))
    assert "DRAFT_MODEL_SOURCE_PARTS = None" in without
    assert lord.duck_kaggle_model_sources(lord.DuckKaggleVllmConfig(
        model_source=TARGET_REF, draft_model_source=DRAFT_REF)) == [TARGET_REF, DRAFT_REF]


@pytest.mark.parametrize("requested", [True, False])
def test_cache_diagnostics_follow_the_config(monkeypatch, requested):
    monkeypatch.setenv("LOCAL_ANALYZER_PROVIDER", "vllm")
    script = lord.duck_kaggle_setup_command(lord.DuckKaggleVllmConfig(
        model_source=TARGET_REF, cache_diagnostics=requested))
    assert f"VLLM_CACHE_DIAGNOSTICS_REQUESTED = {requested!r}" in script
    assert "enabled = VLLM_CACHE_DIAGNOSTICS_REQUESTED and namespace['is_test_run']()" in script
