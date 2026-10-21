"""Run the launcher's JIT cache blocks (issue #6) with the GPU, SGLang and /health stubbed.

The blocks are cut from the serving cell the builder produces, so this checks
the wiring that ships: the runtime probe feeds the manifest, the save waits
for /health, and a restore installs the snapshot only when it matches.

Run from ARC3-Inference:  python -m pytest -q ../kaggle/tests/test_launcher_jit_cache.py
"""
import importlib.util
import json
import os
import threading
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("build_notebook", REPO / "kaggle" / "build_notebook.py")
bn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bn)

LAUNCHER = bn._serving_launcher()
RESTORE_BLOCK = LAUNCHER[LAUNCHER.index("# ---- JIT caches (issue #6): describe"):
                         LAUNCHER.index("# ---- validate target")]
SAVE_BLOCK = LAUNCHER[LAUNCHER.index("# ---- JIT caches (issue #6): snapshot"):]
RUNTIME = {"python": "3.12.12", "torch": "2.9.1+cu130", "torch_cuda": "13.0", "sglang": "2.5.3",
           "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "capability": "12.0", "driver": "580"}
CFG = {"MAXREQ": 10, "AUTOTUNE": True, "JIT_CACHE_SAVE": False, "JIT_CACHE_DIR": None}


def _jit_cache_module():
    namespace = {}
    exec(bn._jit_cache_block(), namespace)
    return namespace["jit_cache"]


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Health:
    """Stands in for urllib: /health fails ``failures`` times, then answers 200."""

    def __init__(self, failures):
        self.failures = failures
        self.request = types.SimpleNamespace(urlopen=self.urlopen)

    def urlopen(self, url, timeout):
        if self.failures > 0:
            self.failures -= 1
            raise OSError("connection refused")
        return _Response()


def _machine(tmp_path, name, cfg, runtime=RUNTIME):
    """The serving cell's globals just before the JIT cache blocks, on one 'machine'."""
    prefix = tmp_path / name / "sgl-intel"
    cache = prefix / "cache"
    for sub in ("huggingface", "torch", "torchinductor", "triton", "flashinfer", "cuda", "sglang/jit"):
        (cache / sub).mkdir(parents=True, exist_ok=True)
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text('{"model_type": "qwen3_next"}')
    working = tmp_path / name / "working"
    working.mkdir(parents=True)
    calls = []

    def run(cmd, *, check=True, env=None):
        calls.append([str(c) for c in cmd])
        assert Path(cmd[1]).read_text() == namespace["jit_cache"].RUNTIME_PROBE
        return types.SimpleNamespace(stdout="noise\nJIT_CACHE_RUNTIME " + json.dumps(runtime) + "\n")

    namespace = {
        "jit_cache": _jit_cache_module(), "CFG": dict(CFG, **cfg), "cache": str(cache), "PREFIX": str(prefix),
        "PYTHON": "/venv/bin/python", "MODEL_DIR": str(model), "WORKING_DIR": working, "run": run,
        "env": {"HOME": str(tmp_path / name / "home"), "SGLANG_SM120_ONLINE_MXFP8": "0", "PATH": "/usr/bin"},
        "os": os, "time": time, "Path": Path, "url": "http://127.0.0.1:1/health",
    }
    return namespace, calls


def _proc(alive=True):
    return types.SimpleNamespace(poll=lambda: None if alive else 1)


def _fill(namespace):
    triton = Path(namespace["cache"], "triton", "ab12")
    triton.mkdir(parents=True)
    (triton / "_fwd_kernel.cubin").write_bytes(b"cubin")
    tilelang = Path(namespace["env"]["HOME"], ".tilelang", "cache")
    tilelang.mkdir(parents=True)
    (tilelang / "k.so").write_bytes(b"so")
    (Path(namespace["cache"], "torch") / "hub.bin").write_bytes(b"12345")


def _save(tmp_path, capsys, failures=0):
    namespace, calls = _machine(tmp_path, "save", {"JIT_CACHE_SAVE": True})
    exec(RESTORE_BLOCK, namespace)
    assert len(calls) == 1 and namespace["JIT_CACHE_MANIFEST"]["runtime"]["gpu"] == RUNTIME["gpu"]
    _fill(namespace)
    namespace.update(urllib=_Health(failures), proc=_proc(),
                     time=types.SimpleNamespace(time=time.time, sleep=lambda s: None))
    exec(SAVE_BLOCK, namespace)
    for thread in threading.enumerate():  # a server still loading is saved from a thread
        if "_save_jit_cache_when_ready" in thread.name:
            thread.join(timeout=10)
    return namespace, capsys.readouterr().out


def test_save_runs_when_the_server_is_ready(tmp_path, capsys):
    namespace, out = _save(tmp_path, capsys)
    assert "JIT cache saved to" in out and "2 files" in out
    assert "not in the snapshot: " + str(Path(namespace["cache"], "torch")) in out
    manifest = json.loads((Path(namespace["WORKING_DIR"]) / "sgl-jit-cache.manifest.json").read_text())
    assert manifest["files_per_root"]["cache/triton"] == 1 and manifest["files_per_root"]["tilelang"] == 1


def test_save_waits_for_a_server_still_loading(tmp_path, capsys):
    namespace, out = _save(tmp_path, capsys, failures=3)
    assert (Path(namespace["WORKING_DIR"]) / "sgl-jit-cache.tar").exists()
    assert "JIT cache saved to" in out


def test_restore_installs_a_matching_snapshot(tmp_path, capsys):
    saved, _ = _save(tmp_path, capsys)
    namespace, _ = _machine(tmp_path, "load", {"JIT_CACHE_DIR": str(saved["WORKING_DIR"])})
    exec(RESTORE_BLOCK, namespace)
    out = capsys.readouterr().out
    assert "JIT cache restored in" in out, out
    assert Path(namespace["cache"], "triton", "ab12", "_fwd_kernel.cubin").read_bytes() == b"cubin"
    assert Path(namespace["env"]["HOME"], ".tilelang", "cache", "k.so").read_bytes() == b"so"


def test_restore_starts_cold_on_a_different_runtime(tmp_path, capsys):
    saved, _ = _save(tmp_path, capsys)
    namespace, _ = _machine(tmp_path, "load", {"JIT_CACHE_DIR": str(saved["WORKING_DIR"])},
                            runtime=dict(RUNTIME, sglang="2.5.4"))
    exec(RESTORE_BLOCK, namespace)
    out = capsys.readouterr().out
    assert "JIT cache NOT restored, starting cold" in out and "sglang: saved '2.5.3', here '2.5.4'" in out
    assert not list(Path(namespace["cache"], "triton").iterdir())


def test_a_failing_probe_disables_the_cache_without_failing_the_launch(tmp_path, capsys):
    namespace, _ = _machine(tmp_path, "load", {"JIT_CACHE_DIR": str(tmp_path / "anywhere")})

    def broken_run(cmd, *, check=True, env=None):
        raise RuntimeError("Command failed (1)")

    namespace["run"] = broken_run
    exec(RESTORE_BLOCK, namespace)
    assert namespace["JIT_CACHE_MANIFEST"] is None
    assert "JIT cache disabled, the runtime probe failed" in capsys.readouterr().out


def test_blocks_do_nothing_with_the_flags_off(tmp_path, capsys):
    namespace, calls = _machine(tmp_path, "control", {})
    exec(RESTORE_BLOCK, namespace)
    namespace.update(urllib=None, proc=None)  # any use would raise
    exec(SAVE_BLOCK, namespace)
    assert calls == [] and namespace["JIT_CACHE_MANIFEST"] is None
    assert capsys.readouterr().out == ""
    assert not list(Path(namespace["WORKING_DIR"]).iterdir())
