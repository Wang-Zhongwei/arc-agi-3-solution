"""Tests for serving/kaggle/jit_cache.py: manifest matching, save and restore (issue #6).

Pure Python, no GPU: the runtime probe runs against a stub torch module.

Run from ARC3-Inference:  python -m pytest -q ../kaggle/tests/test_jit_cache.py
"""
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("jit_cache", REPO / "serving" / "kaggle" / "jit_cache.py")
jc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jc)

RUNTIME = {
    "python": "3.12.12", "torch": "2.9.1+cu130", "torch_cuda": "13.0", "nvcc": "Cuda compilation tools, release 13.0, V13.0.88",
    "sglang": "2.5.3", "sgl-kernel": "0.4.0", "triton": "3.5.1", "flashinfer-python": "0.5.3",
    "flashinfer-cubin": None, "flashinfer-jit-cache": None, "tilelang": "0.1.7", "apache-tvm-ffi": "0.1.4",
    "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "capability": "12.0", "driver": "580.95.05",
}
CFG = {"CTX": 136 * 1024, "MAXREQ": 10, "AUTOTUNE": True, "JIT_CACHE_SAVE": False, "JIT_CACHE_DIR": None,
       "PREFETCH_CHECKPOINTS": True, "DROP_CACHE_AFTER_LOAD": True}
ENV = {"SGLANG_SM120_ONLINE_MXFP8": "0", "TRITON_CACHE_DIR": "/tmp/sgl-intel/cache/triton",
       "TORCH_CUDA_ARCH_LIST": "12.0", "TAAF_ARM": "control", "PATH": "/usr/bin", "HOME": "/root"}


def manifest(roots, runtime=RUNTIME, cfg=CFG, env=ENV):
    return jc.make_manifest(runtime, jc.config_snapshot(cfg, env), roots, label="test")


def roots_under(base):
    return jc.cache_roots(base / "cache", base / "home")


def fill(roots):
    """A small cache: files in several roots plus things a snapshot must skip."""
    files = {
        "cache/triton/ab12/_fwd_kernel.cubin": b"cubin",
        "cache/triton/ab12/__grp___fwd_kernel.json": b"{}",
        "cache/flashinfer/.cache/flashinfer/0.5.3/120/cached_ops/gdn/gdn.so": b"\x7fELF so",
        "cache/sglang/jit/k1/k1.so": b"\x7fELF k1",
        "tilelang/cache/k/kernel.cu": b"// tilelang",
    }
    for arcname, data in files.items():
        key = next(k for k in roots if arcname.startswith(k + "/"))
        path = roots[key] / arcname[len(key) + 1:]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        os.utime(path, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    triton = roots["cache/triton"]
    (triton / "ab12" / "lock").write_text("")
    (triton / "ab12" / "x.lock").write_text("")
    (triton / "tmp.pid_7_123").mkdir()
    (triton / "tmp.pid_7_123" / "half.cubin").write_bytes(b"partial")
    (triton / "ab12" / "link.cubin").symlink_to(triton / "ab12" / "_fwd_kernel.cubin")
    staging = roots["cache/sglang/jit"] / ".staging-0f1e"  # an SGLang JIT build in progress
    staging.mkdir()
    (staging / "k2.so").write_bytes(b"half-built")
    roots["cache/torchinductor"].mkdir(parents=True, exist_ok=True)  # empty root
    return files


def fresh_roots(base):
    """Roots as the launcher leaves them before launch: the cache dirs exist, empty."""
    roots = roots_under(base)
    for key, path in roots.items():
        if key.startswith("cache/"):
            path.mkdir(parents=True, exist_ok=True)
    return roots


def read_tree(roots):
    out = {}
    for key, root in roots.items():
        for path in sorted(Path(root).rglob("*")) if Path(root).is_dir() else []:
            if path.is_file():
                out[f"{key}/{path.relative_to(root).as_posix()}"] = path.read_bytes()
    return out


# ---------------------------------------------------------------- manifest


def test_config_hash_ignores_startup_and_cache_keys_and_unrelated_env():
    base = jc.config_hash(jc.config_snapshot(CFG, ENV))
    same = dict(CFG, JIT_CACHE_SAVE=True, JIT_CACHE_DIR="/kaggle/input/x",
                PREFETCH_CHECKPOINTS=False, DROP_CACHE_AFTER_LOAD=False,
                SPEC_ACCEPT_SINGLE=0.5, SPEC_ACCEPT_ACC=0.5)
    assert jc.config_hash(jc.config_snapshot(same, dict(ENV, TAAF_ARM="other", PATH="/x"))) == base
    assert jc.config_hash(jc.config_snapshot(dict(CFG, MAXREQ=11), ENV)) != base
    assert jc.config_hash(jc.config_snapshot(dict(CFG, AUTOTUNE=False), ENV)) != base
    assert jc.config_hash(jc.config_snapshot(CFG, dict(ENV, SGLANG_SM120_ONLINE_MXFP8="1"))) != base
    assert jc.config_snapshot(CFG, ENV)["env"] == {
        "SGLANG_SM120_ONLINE_MXFP8": "0", "TORCH_CUDA_ARCH_LIST": "12.0", "TRITON_CACHE_DIR": "/tmp/sgl-intel/cache/triton"}


def test_config_hash_covers_the_model_config():
    a = jc.config_snapshot(CFG, ENV, b'{"model_type": "qwen3_next"}')
    assert jc.config_hash(a) == jc.config_hash(jc.config_snapshot(CFG, ENV, b'{"model_type": "qwen3_next"}'))
    assert jc.config_hash(a) != jc.config_hash(jc.config_snapshot(CFG, ENV, b'{"model_type": "other"}'))
    saved = jc.make_manifest(RUNTIME, a, {})
    here = jc.make_manifest(RUNTIME, jc.config_snapshot(CFG, ENV, b"{}"), {})
    assert jc.mismatches(saved, here)[0].endswith("(model.config.json sha256)")


def test_matching_manifest_has_no_mismatches(tmp_path):
    roots = roots_under(tmp_path)
    saved = manifest(roots)
    here = manifest(roots, runtime=dict(RUNTIME, driver="590.1"), cfg=dict(CFG, JIT_CACHE_DIR="/kaggle/input/c"))
    assert jc.mismatches(saved, here) == []  # the driver is informational only


@pytest.mark.parametrize("key, value", [
    ("gpu", "NVIDIA H100"), ("capability", "9.0"), ("torch", "2.10.0"), ("torch_cuda", "12.9"),
    ("nvcc", "release 13.1"), ("sglang", "2.5.4"), ("triton", "3.6.0"), ("flashinfer-python", "0.6.0"),
    ("python", "3.11.9"),
])
def test_runtime_differences_are_reported(tmp_path, key, value):
    roots = roots_under(tmp_path)
    problems = jc.mismatches(manifest(roots), manifest(roots, runtime=dict(RUNTIME, **{key: value})))
    assert len(problems) == 1 and problems[0].startswith(f"{key}:")


def test_missing_runtime_field_counts_as_a_mismatch(tmp_path):
    roots = roots_under(tmp_path)
    here = {k: v for k, v in RUNTIME.items() if k != "tilelang"}
    assert [p.split(":")[0] for p in jc.mismatches(manifest(roots), manifest(roots, runtime=here))] == ["tilelang"]


def test_config_mismatch_names_the_differing_keys(tmp_path):
    roots = roots_under(tmp_path)
    problems = jc.mismatches(manifest(roots), manifest(roots, cfg=dict(CFG, MAXREQ=12)))
    assert len(problems) == 1 and problems[0].startswith("config_hash:") and "(cfg.MAXREQ)" in problems[0]


def test_kind_or_format_mismatch_stops_early(tmp_path):
    roots = roots_under(tmp_path)
    saved = dict(manifest(roots), format=jc.FORMAT + 1, runtime={})
    assert [p.split(":")[0] for p in jc.mismatches(saved, manifest(roots))] == ["format"]


def test_runtime_probe_runs_against_a_stub_torch(tmp_path):
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "torch.py").write_text(
        "class version:\n    cuda = '13.0'\n"
        "class cuda:\n"
        "    get_device_name = staticmethod(lambda i: 'NVIDIA RTX PRO 6000 Blackwell Server Edition')\n"
        "    get_device_capability = staticmethod(lambda i: (12, 0))\n"
    )
    env = dict(os.environ, PYTHONPATH=str(stub), CUDACXX=str(tmp_path / "no-nvcc"))
    out = subprocess.run([sys.executable, "-c", jc.RUNTIME_PROBE], env=env, capture_output=True, text=True, check=True)
    info = jc.parse_probe("noise\n" + out.stdout)
    assert info["gpu"] == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    assert info["capability"] == "12.0" and info["torch_cuda"] == "13.0"
    assert info["nvcc"] is None and "sglang" in info and "driver" in info


def test_parse_probe_without_marker_raises():
    with pytest.raises(ValueError):
        jc.parse_probe("no runtime here\n")


def test_candidate_dirs_cover_both_kaggle_mount_layouts():
    assert [str(p) for p in jc.candidate_dirs("/kaggle/input/datasets/mellamoa/sgl-jit-cache")] == [
        "/kaggle/input/datasets/mellamoa/sgl-jit-cache", "/kaggle/input/sgl-jit-cache"]
    assert [str(p) for p in jc.candidate_dirs("/kaggle/input/datasets/mellamoa/sgl-jit-cache/v2")] == [
        "/kaggle/input/datasets/mellamoa/sgl-jit-cache/v2", "/kaggle/input/sgl-jit-cache/v2"]
    assert [str(p) for p in jc.candidate_dirs("/kaggle/input/sgl-jit-cache")] == ["/kaggle/input/sgl-jit-cache"]
    assert [str(p) for p in jc.candidate_dirs("/data/cache")] == ["/data/cache"]


# ---------------------------------------------------------------- save


def test_save_writes_tar_and_manifest_skipping_locks_temps_and_symlinks(tmp_path):
    roots = roots_under(tmp_path / "src")
    files = fill(roots)
    out = tmp_path / "working"
    out.mkdir()
    written = jc.save(out / jc.TAR_NAME, roots, manifest(roots))
    with tarfile.open(out / jc.TAR_NAME) as tar:
        names = tar.getnames()
        mtimes = {m.name: m.mtime for m in tar.getmembers()}
        inner = json.loads(tar.extractfile(jc.MANIFEST_NAME).read())
    assert names[-1] == jc.MANIFEST_NAME
    assert sorted(names[:-1]) == sorted(files)
    assert all(mtimes[name] == 1_700_000_000 for name in files)
    sidecar = json.loads((out / jc.MANIFEST_NAME).read_text())
    assert sidecar == written and sidecar["tar_bytes"] == (out / jc.TAR_NAME).stat().st_size
    assert inner == {k: v for k, v in written.items() if k != "tar_bytes"}
    assert written["files"] == len(files) and written["content_bytes"] == sum(map(len, files.values()))
    assert written["files_per_root"] == {
        "cache/cuda": 0, "cache/flashinfer": 1, "cache/sglang/jit": 1, "cache/torchinductor": 0,
        "cache/triton": 2, "tilelang": 1}
    assert not list(out.glob("*.partial"))


def test_save_leaves_out_a_file_removed_after_the_listing(tmp_path, monkeypatch):
    roots = roots_under(tmp_path / "src")
    files = fill(roots)
    listed = list(jc.iter_cache_files(roots))
    gone = roots["cache/triton"] / "ab12" / "_fwd_kernel.cubin"
    gone.unlink()
    monkeypatch.setattr(jc, "iter_cache_files", lambda roots: iter(listed))
    written = jc.save(tmp_path / jc.TAR_NAME, roots, manifest(roots))
    with tarfile.open(tmp_path / jc.TAR_NAME) as tar:
        names = tar.getnames()
        inner = json.loads(tar.extractfile(jc.MANIFEST_NAME).read())
    expected = sorted(n for n in files if n != "cache/triton/ab12/_fwd_kernel.cubin")
    assert sorted(names[:-1]) == expected
    assert inner["files"] == written["files"] == len(expected)
    assert written["files_per_root"]["cache/triton"] == 1
    assert written["content_bytes"] == sum(len(files[n]) for n in expected)


def test_a_failed_save_leaves_no_partial_tar(tmp_path, monkeypatch):
    roots = roots_under(tmp_path / "src")
    fill(roots)
    out = tmp_path / "working"
    out.mkdir()
    real_add, calls = tarfile.TarFile.add, []

    def add(self, *args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            raise OSError("disk went away")
        return real_add(self, *args, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "add", add)
    with pytest.raises(OSError, match="disk went away"):
        jc.save(out / jc.TAR_NAME, roots, manifest(roots))
    assert list(out.iterdir()) == []


def test_save_refuses_when_the_disk_would_fill(tmp_path):
    roots = roots_under(tmp_path / "src")
    fill(roots)
    with pytest.raises(OSError):
        jc.save(tmp_path / jc.TAR_NAME, roots, manifest(roots), reserve=1 << 62)
    assert not (tmp_path / jc.TAR_NAME).exists()


def test_unsaved_dirs_reports_only_what_the_snapshot_leaves_out(tmp_path):
    roots = roots_under(tmp_path)
    fill(roots)
    cache = tmp_path / "cache"
    (cache / "huggingface").mkdir()
    (cache / "tvm-ffi" / "x").mkdir(parents=True)
    (cache / "tvm-ffi" / "x" / "k.so").write_bytes(b"12345")
    (cache / "sglang" / "other.json").write_bytes(b"123")
    report = dict(jc.unsaved_dirs(cache, roots))
    assert report == {str(cache / "sglang"): 3, str(cache / "tvm-ffi"): 5}


# ---------------------------------------------------------------- restore


@pytest.fixture
def saved(tmp_path):
    roots = roots_under(tmp_path / "src")
    files = fill(roots)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    jc.save(dataset / jc.TAR_NAME, roots, manifest(roots))
    return dataset, files


def test_restore_round_trip_preserves_files_and_mtimes(tmp_path, saved):
    dataset, expected = saved
    roots = fresh_roots(tmp_path / "dst")
    ok, message = jc.restore(dataset, roots, manifest(roots), tmp_path / "staging")
    assert ok, message
    assert "restored 5 files" in message
    assert read_tree(roots) == expected
    cubin = roots["cache/triton"] / "ab12" / "_fwd_kernel.cubin"
    assert cubin.stat().st_mtime == 1_700_000_000
    assert not (tmp_path / "staging").exists()


def test_restore_finds_the_tar_in_a_subfolder_without_its_sidecar(tmp_path, saved):
    dataset, expected = saved
    (dataset / jc.MANIFEST_NAME).unlink()
    nested = dataset / "v1"
    nested.mkdir()
    (dataset / jc.TAR_NAME).rename(nested / jc.TAR_NAME)
    roots = fresh_roots(tmp_path / "dst")
    ok, message = jc.restore(dataset, roots, manifest(roots), tmp_path / "staging")
    assert ok, message
    assert read_tree(roots) == expected


def test_restore_from_a_tar_kaggle_unpacked_on_upload(tmp_path, saved):
    dataset, expected = saved
    unpacked = tmp_path / "unpacked"
    (unpacked / "sgl-jit-cache").mkdir(parents=True)
    with tarfile.open(dataset / jc.TAR_NAME) as tar:
        tar.extractall(unpacked / "sgl-jit-cache", filter="data")
    (unpacked / jc.MANIFEST_NAME).write_bytes((dataset / jc.MANIFEST_NAME).read_bytes())  # sidecar, no tar
    roots = fresh_roots(tmp_path / "dst")
    ok, message = jc.restore(unpacked, roots, manifest(roots), tmp_path / "staging")
    assert ok and "(tree," in message, message
    assert read_tree(roots) == expected


def test_restore_merges_into_a_non_empty_cache(tmp_path, saved):
    dataset, expected = saved
    roots = fresh_roots(tmp_path / "dst")
    (roots["cache/triton"] / "old").mkdir()
    (roots["cache/triton"] / "old" / "keep.cubin").write_bytes(b"old")
    ok, message = jc.restore(dataset, roots, manifest(roots), tmp_path / "staging")
    assert ok, message
    assert read_tree(roots) == {**expected, "cache/triton/old/keep.cubin": b"old"}


def test_restore_refuses_a_mismatched_runtime_and_installs_nothing(tmp_path, saved):
    dataset, _ = saved
    roots = fresh_roots(tmp_path / "dst")
    here = manifest(roots, runtime=dict(RUNTIME, sglang="2.5.4"))
    ok, message = jc.restore(dataset, roots, here, tmp_path / "staging")
    assert not ok and "sglang: saved '2.5.3', here '2.5.4'" in message
    assert read_tree(roots) == {}


def test_restore_without_a_cache_reports_where_it_looked(tmp_path):
    roots = fresh_roots(tmp_path / "dst")
    ok, message = jc.restore(tmp_path / "missing", roots, manifest(roots), tmp_path / "staging")
    assert not ok and str(tmp_path / "missing") in message


def test_restore_refuses_a_truncated_tar(tmp_path, saved):
    dataset, _ = saved
    tar = dataset / jc.TAR_NAME
    tar.write_bytes(tar.read_bytes()[:-1024])
    roots = fresh_roots(tmp_path / "dst")
    ok, message = jc.restore(dataset, roots, manifest(roots), tmp_path / "staging")
    assert not ok and "manifest says" in message
    assert read_tree(roots) == {}


@pytest.mark.parametrize("name", ["../escape.so", "cache/huggingface/model.bin", "/abs/path.so"])
def test_restore_rejects_members_outside_the_cache_roots(tmp_path, name):
    roots = fresh_roots(tmp_path / "dst")
    good = manifest(roots)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    with tarfile.open(dataset / jc.TAR_NAME, "w") as tar:
        for arcname, data in ((jc.MANIFEST_NAME, json.dumps(good).encode()), ("cache/triton/a.cubin", b"a"),
                              (name, b"bad")):
            info = tarfile.TarInfo(arcname)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError):
        jc.restore(dataset, roots, good, tmp_path / "staging")
    assert read_tree(roots) == {}
    assert not (tmp_path / "staging").exists()
