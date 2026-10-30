"""The Lord setup script can serve a DFlash block drafter (incoai/Qwen3.8-Flash-Next-DFlash2) for
the Qwen4Exp target by patching the unpacked e975732 rc3 runtime."""
import json
import os
import re
import shutil
from pathlib import Path

import pytest

from inference.framework.lordhansolo import kaggle as lord

TARGET_REF = "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1"
DFLASH_REF = "jonathanwang2022/qwen3-8-flash-next-dflash2/Transformers/default/1"
DIST = "usr/local/lib/python3.12/dist-packages"


def _patch_namespace(method: str, runtime_root: Path) -> dict:
    """Execute the rendered DFlash runtime patcher in a scratch namespace."""
    script = lord._DUCK_VLLM_SETUP_SCRIPT
    start = script.index("DFLASH_RUNTIME_PATCHES = (")
    end = script.index("\n\n\n", script.index("def patch_runtime_for_dflash"))
    namespace = {"Path": Path, "VLLM_DRAFT_METHOD": method, "IMAGE_RUNTIME_ROOT": runtime_root}
    exec(script[start:end], namespace)
    return namespace


def _stand_in_runtime(tmp_path: Path, patches) -> Path:
    """Write each patched file as its anchors separated by filler lines."""
    files: dict[str, list[str]] = {}
    for relative_path, anchor, _ in patches:
        files.setdefault(relative_path, []).append(anchor)
    for relative_path, anchors in files.items():
        path = tmp_path / DIST / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# filler\n".join(["", *anchors, ""]))
    return tmp_path


def test_setup_patches_the_runtime_before_the_server_command():
    script = lord._DUCK_VLLM_SETUP_SCRIPT
    assert re.search(
        r"check_runtime_reads_fp8_ple\(manifest\)\n\s+patch_runtime_for_dflash\(manifest\)\n"
        r"\s+draft_model_dir = resolve_dflash_draft\(\) if VLLM_DRAFT_METHOD == 'dflash' else assemble_mtp_draft\(\)",
        script,
    )


def test_every_anchor_is_replaced_once(tmp_path):
    patches = _patch_namespace("dflash", tmp_path)["DFLASH_RUNTIME_PATCHES"]
    root = _stand_in_runtime(tmp_path, patches)
    _patch_namespace("dflash", root)["patch_runtime_for_dflash"]({"dist_packages": DIST})
    for relative_path, anchor, replacement in patches:
        text = (root / DIST / relative_path).read_text()
        assert text.count(replacement) == 1
    model = (root / DIST / "vllm/models/qwen4_exp/nvidia/model.py").read_text()
    assert "return sample_hidden_states, aux_hidden_states" in model
    assert "supports_eagle3 = True" in model
    assert '"dflash",' in (root / DIST / "vllm/model_executor/models/config.py").read_text()


def test_missing_anchor_fails_before_writing(tmp_path):
    patches = _patch_namespace("dflash", tmp_path)["DFLASH_RUNTIME_PATCHES"]
    root = _stand_in_runtime(tmp_path, patches[:-1])
    before = {p: p.read_text() for p in (root / DIST).rglob("*.py")}
    with pytest.raises(RuntimeError, match="anchor appears 0 times"):
        _patch_namespace("dflash", root)["patch_runtime_for_dflash"]({"dist_packages": DIST})
    assert {p: p.read_text() for p in before} == before


def test_mtp_runs_leave_the_runtime_alone(tmp_path):
    patches = _patch_namespace("mtp", tmp_path)["DFLASH_RUNTIME_PATCHES"]
    root = _stand_in_runtime(tmp_path, patches)
    before = {p: p.read_text() for p in (root / DIST).rglob("*.py")}
    _patch_namespace("mtp", root)["patch_runtime_for_dflash"]({"dist_packages": DIST})
    assert {p: p.read_text() for p in before} == before


@pytest.mark.skipif(not os.environ.get("ARC3_LORD_RUNTIME_TREE"), reason="needs an unpacked e975732 rc3 vllm/ tree")
def test_anchors_match_the_real_runtime(tmp_path):
    shutil.copytree(Path(os.environ["ARC3_LORD_RUNTIME_TREE"]), tmp_path / DIST / "vllm")
    _patch_namespace("dflash", tmp_path)["patch_runtime_for_dflash"]({"dist_packages": DIST})
    for relative_path in {p for p, _, _ in _patch_namespace("dflash", tmp_path)["DFLASH_RUNTIME_PATCHES"]}:
        compile((tmp_path / DIST / relative_path).read_text(), relative_path, "exec")


def test_render_serves_the_mounted_checkpoint_as_a_dflash_draft(monkeypatch):
    monkeypatch.setenv("LOCAL_ANALYZER_PROVIDER", "vllm")
    script = lord.duck_kaggle_setup_command(lord.DuckKaggleVllmConfig(
        model_source=TARGET_REF, draft_model_source=DFLASH_REF, draft_method="dflash"))
    assert "VLLM_DRAFT_METHOD = 'dflash'" in script
    assert "speculative_config = {'method': VLLM_DRAFT_METHOD, 'num_speculative_tokens': VLLM_MTP_TOKENS}" in script
    assert "if VLLM_DRAFT_VOCAB and VLLM_DRAFT_METHOD == 'mtp':" in script
    assert lord.duck_kaggle_model_sources(lord.DuckKaggleVllmConfig(
        model_source=TARGET_REF, draft_model_source=DFLASH_REF, draft_method="dflash")) == [TARGET_REF, DFLASH_REF]
    assert "VLLM_DRAFT_ATTENTION_BACKEND = ''" in script
    triton = lord.duck_kaggle_setup_command(lord.DuckKaggleVllmConfig(
        model_source=TARGET_REF, draft_model_source=DFLASH_REF, draft_method="dflash",
        draft_attention_backend="TRITON_ATTN"))
    assert "VLLM_DRAFT_ATTENTION_BACKEND = 'TRITON_ATTN'" in triton
    assert "speculative_config['attention_backend'] = VLLM_DRAFT_ATTENTION_BACKEND" in triton
    mtp = lord.duck_kaggle_setup_command(lord.DuckKaggleVllmConfig(model_source=TARGET_REF))
    assert "VLLM_DRAFT_METHOD = 'mtp'" in mtp


@pytest.mark.parametrize("overrides", [
    {"draft_method": "eagle"},
    {"draft_method": "dflash"},
    {"draft_method": "dflash", "draft_model_source": DFLASH_REF, "draft_vocab": "configs/draft_vocab_32k.json"},
    {"draft_method": "dflash", "draft_model_source": DFLASH_REF, "mtp_tokens": 0},
])
def test_render_rejects_an_unservable_draft(monkeypatch, overrides):
    monkeypatch.setenv("LOCAL_ANALYZER_PROVIDER", "vllm")
    with pytest.raises(ValueError, match="draft_method"):
        lord.duck_kaggle_setup_command(lord.DuckKaggleVllmConfig(model_source=TARGET_REF, **overrides))


def test_resolve_dflash_draft_reads_the_mounted_config(tmp_path, capsys):
    script = lord._DUCK_VLLM_SETUP_SCRIPT
    start = script.index("def resolve_dflash_draft")
    end = script.index("\n\n\n", start)
    source = tmp_path / "dflash2"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "architectures": ["DFlash2DraftModel"],
        "dflash_config": {"block_size": 8, "target_layer_ids": [5, 15, 25, 35, 45]},
    }))
    namespace = {
        "json": json, "Path": Path, "VLLM_MTP_TOKENS": 3,
        "DRAFT_MODEL_SOURCE_PARTS": tuple(DFLASH_REF.split("/")),
        "resolve_kaggle_model_path": lambda parts: source,
        "describe_kaggle_input_tree": lambda: "",
    }
    exec(script[start:end], namespace)
    assert namespace["resolve_dflash_draft"]() == source
    assert "target layers [5, 15, 25, 35, 45], block 8, 3 draft tokens" in capsys.readouterr().out
    namespace["resolve_kaggle_model_path"] = lambda parts: tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="DFlash draft checkpoint not mounted"):
        namespace["resolve_dflash_draft"]()
