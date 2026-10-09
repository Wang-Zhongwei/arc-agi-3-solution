"""The Lord setup script routes an AutoRound (INC) checkpoint's PLE embedding to the
unquantized method, so Daniel's W4A16 weights load on Lord's vLLM runtime."""
import json
import re
from pathlib import Path

import pytest

from inference.framework.lordhansolo import kaggle as lord


def _route_function():
    """Execute the INC route from the rendered setup script in a scratch namespace."""
    script = lord._DUCK_VLLM_SETUP_SCRIPT
    start = script.index("INC_PLE_ANCHOR = (")
    end = script.index("\n\n\n", script.index("def route_inc_ple_to_unquantized"))
    namespace = {"json": json, "Path": Path}
    exec(script[start:end], namespace)
    return namespace


def _runtime(tmp_path, anchor_present: bool, quant_method: str):
    ns = _route_function()
    ple = tmp_path / "runtime" / "dist" / "vllm" / "models" / "qwen4_exp" / "nvidia" / "ngram_embedding.py"
    ple.parent.mkdir(parents=True)
    body = ns["INC_PLE_ANCHOR"] + "type(quant_config))\n" if anchor_present else "pass\n"
    ple.write_text("def get_method(quant_config):\n" + body)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": quant_method}}))
    ns.update(IMAGE_RUNTIME_ROOT=tmp_path / "runtime", MODEL_PATH=model)
    return ns, ple


def test_setup_calls_the_route_after_unpacking_the_runtime():
    script = lord._DUCK_VLLM_SETUP_SCRIPT
    assert re.search(r"manifest = install_image_runtime\(\)\n\s+route_inc_ple_to_unquantized\(manifest\)", script)


@pytest.mark.parametrize("method", ["auto-round", "compressed-tensors"])
def test_anchor_gets_the_inc_branch_for_any_checkpoint(tmp_path, method):
    ns, ple = _runtime(tmp_path, anchor_present=True, quant_method=method)
    ns["route_inc_ple_to_unquantized"]({"dist_packages": "dist"})
    text = ple.read_text()
    assert 'if type(quant_config).__name__ == "INCConfig":' in text
    assert text.index("INCConfig") < text.index("isinstance(quant_config, Fp8Config)")


def test_missing_anchor_fails_only_for_autoround(tmp_path):
    ns, _ = _runtime(tmp_path, anchor_present=False, quant_method="auto-round")
    with pytest.raises(RuntimeError, match="INC PLE route"):
        ns["route_inc_ple_to_unquantized"]({"dist_packages": "dist"})
    ns, ple = _runtime(tmp_path / "lord", anchor_present=False, quant_method="compressed-tensors")
    ns["route_inc_ple_to_unquantized"]({"dist_packages": "dist"})
    assert ple.read_text() == "def get_method(quant_config):\npass\n"
