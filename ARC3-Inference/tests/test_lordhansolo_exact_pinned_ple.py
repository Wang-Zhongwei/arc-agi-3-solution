"""The Lord setup script pins the host PLE table at its own size: torch's pinned allocator rounded
the 95.4 GiB BF16 table up to 128 GiB (porting-advaith v3)."""
import json
import re
import types
from pathlib import Path

import pytest

from inference.framework.lordhansolo import kaggle as lord

# The runtime's allocator as e975732 ships it, inside enough of its class to compile.
RUNTIME_SOURCE = '''import torch


class Qwen4ExpPLEPinnedHostEmbedding:
    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate the complete PLE weight directly in pinned CPU memory."""
        return torch.empty(
            num_embeddings,
            embedding_dim,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )


__all__ = [
    "Qwen4ExpNGramEmbedding",
]
'''


def _patch_namespace():
    """Execute the exact-size patch from the rendered setup script in a scratch namespace."""
    script = lord._DUCK_VLLM_SETUP_SCRIPT
    start = script.index("# Qwen4ExpPLEPinnedHostEmbedding.allocate_embedding_weight")
    end = script.index("\n\n\n", script.index("def pin_ple_table_at_exact_size"))
    namespace = {"json": json, "Path": Path}
    exec(script[start:end], namespace)
    return namespace


def _runtime(tmp_path, source: str):
    ns = _patch_namespace()
    ple = tmp_path / "runtime" / "dist" / "vllm" / "models" / "qwen4_exp" / "nvidia" / "ngram_embedding.py"
    ple.parent.mkdir(parents=True)
    ple.write_text(source)
    ns["IMAGE_RUNTIME_ROOT"] = tmp_path / "runtime"
    return ns, ple


def test_setup_patches_after_the_inc_route():
    script = lord._DUCK_VLLM_SETUP_SCRIPT
    assert re.search(
        r"route_inc_ple_to_unquantized\(manifest\)\n\s+pin_ple_table_at_exact_size\(manifest\)\n", script
    )


def test_anchor_matches_the_runtime_allocator_once():
    assert RUNTIME_SOURCE.count(_patch_namespace()["EXACT_PINNED_PLE_ANCHOR"]) == 1


def test_patched_allocator_calls_the_helper_and_compiles(tmp_path):
    ns, ple = _runtime(tmp_path, RUNTIME_SOURCE)
    ns["pin_ple_table_at_exact_size"]({"dist_packages": "dist"})
    text = ple.read_text()
    assert "pin_memory=True,\n        )" not in text
    assert "return _allocate_exact_pinned_cpu_weight(num_embeddings, embedding_dim, dtype)" in text
    compile(text, str(ple), "exec")


def test_missing_anchor_leaves_the_runtime_unchanged(tmp_path, capsys):
    ns, ple = _runtime(tmp_path, "pass\n")
    ns["pin_ple_table_at_exact_size"]({"dist_packages": "dist"})
    assert ple.read_text() == "pass\n"
    assert "keeps the power-of-two pinned allocator" in capsys.readouterr().out


class _FakeTensor:
    def __init__(self, rows, cols, pinned):
        self.shape, self.pinned = (rows, cols), pinned

    def numel(self):
        return self.shape[0] * self.shape[1]

    def element_size(self):
        return 2

    def data_ptr(self):
        return 4096


def _helpers(register_result, cleared=True):
    """The injected helpers with a stand-in torch that records the cudaHostRegister call."""
    calls = []

    def empty(rows, cols, *, dtype, device, pin_memory=False):
        return _FakeTensor(rows, cols, pin_memory)

    def register(ptr, size, flags):
        calls.append((ptr, size, flags))
        return register_result

    torch = types.SimpleNamespace(
        empty=empty, cuda=types.SimpleNamespace(cudart=lambda: types.SimpleNamespace(cudaHostRegister=register))
    )
    logger = types.SimpleNamespace(info=lambda *a: None, warning=lambda *a: calls.append("warning"))
    ns = {"torch": torch, "logger": logger}
    exec(_patch_namespace()["EXACT_PINNED_PLE_HELPERS"], ns)
    ns["_clear_cuda_runtime_error"] = lambda: cleared
    return ns["_allocate_exact_pinned_cpu_weight"], calls


def test_registers_exactly_the_table_bytes():
    allocate, calls = _helpers(0)
    weight = allocate(1000, 256, "bf16")
    assert not weight.pinned
    assert calls == [(4096, 1000 * 256 * 2, 3)]


def test_failed_registration_falls_back_to_pin_memory():
    allocate, calls = _helpers(2)
    assert allocate(1000, 256, "bf16").pinned
    assert calls[-1] == "warning"


def test_uncleared_registration_error_fails():
    allocate, _ = _helpers(2, cleared=False)
    with pytest.raises(RuntimeError, match="could not be cleared"):
        allocate(1000, 256, "bf16")
