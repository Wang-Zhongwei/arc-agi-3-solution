"""Save and restore SGLang's JIT caches across Kaggle containers (issue #6).

Every Kaggle container starts with empty Triton / FlashInfer / SGLang-JIT /
TorchInductor / CUDA / TileLang caches, so the first CUDA-graph capture compiles
from scratch (~145 s). The launcher (serving/kaggle/launch_sglang.py) uses this
module to snapshot those caches into one tar once the server is ready
(CFG JIT_CACHE_SAVE) and to restore a snapshot from a mounted dataset before
launch (CFG JIT_CACHE_DIR). A snapshot is restored only when its manifest
matches this runtime (GPU, CUDA, torch, SGLang and kernel package versions)
and this serving config; otherwise the launch stays cold.

kaggle/build_notebook.py embeds this file into the launcher cell as the module
``jit_cache``, so it must stay plain Python with no imports from the repo (and
no ``from __future__`` line, which is only legal at the top of a file).
Nothing here needs a GPU: the runtime probe is a code string the launcher runs
in the SGLang venv.
"""
import datetime
import hashlib
import io
import json
import os
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath

KIND = "sgl-jit-cache"
FORMAT = 1
TAR_NAME = "sgl-jit-cache.tar"
# Written next to the tar and also stored inside it, so a tar that Kaggle
# unpacked on upload still carries its manifest at the root of the tree.
MANIFEST_NAME = "sgl-jit-cache.manifest.json"
# Directories under the launcher's cache root (PREFIX/cache) that hold compiled kernels.
CACHE_SUBDIRS = ("triton", "flashinfer", "sglang/jit", "torchinductor", "cuda")
TILELANG_KEY = "tilelang"  # ~/.tilelang
# Runtime fields recorded for information only; every other field must match.
INFO_ONLY = ("driver",)
# CFG keys that cannot change a compiled kernel; they are left out of the
# config hash so a cache saved by one startup variant serves the others. The
# MTP acceptance thresholds are plain float arguments of sgl-kernel's prebuilt
# tree_speculative_sampling_target_only, called outside the CUDA graphs (SGLang
# even lets set_internal_state change them on a running server).
CFG_IGNORED = ("JIT_CACHE_SAVE", "JIT_CACHE_DIR", "PREFETCH_CHECKPOINTS", "DROP_CACHE_AFTER_LOAD",
               "SPEC_ACCEPT_SINGLE", "SPEC_ACCEPT_ACC")
# Server environment variables that select or configure kernels.
ENV_PREFIXES = ("SGLANG_", "FLASHINFER_", "TRITON_", "TORCHINDUCTOR_", "TORCH_CUDA_ARCH_LIST")
# Run with the SGLang venv's python; prints one line, PROBE_MARKER + JSON.
PROBE_MARKER = "JIT_CACHE_RUNTIME "
RUNTIME_PROBE = r"""
import importlib.metadata, json, os, platform, subprocess
import torch
info = {
    "python": platform.python_version(),
    "torch_cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
}
for dist in ("torch", "sglang", "sgl-kernel", "triton", "flashinfer-python", "flashinfer-cubin",
             "flashinfer-jit-cache", "tilelang", "apache-tvm-ffi"):
    try:
        info[dist] = importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        info[dist] = None
def _out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ""
releases = [l.strip() for l in _out([os.environ.get("CUDACXX", "nvcc"), "--version"]).splitlines() if "release" in l]
info["nvcc"] = releases[-1] if releases else None
driver = _out(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).split()
info["driver"] = driver[0] if driver else None
print("JIT_CACHE_RUNTIME " + json.dumps(info, sort_keys=True))
"""


def parse_probe(stdout):
    """The runtime dict printed by RUNTIME_PROBE."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(PROBE_MARKER):
            return json.loads(line[len(PROBE_MARKER):])
    raise ValueError("runtime probe printed no JIT_CACHE_RUNTIME line")


def config_snapshot(cfg, env, model_config=None):
    """The launch settings that can change compiled kernels, as plain JSON data.

    ``model_config``: the target checkpoint's config.json bytes, hashed so a
    different model does not reuse the cache.
    """
    snapshot = {
        "cfg": {k: cfg[k] for k in sorted(cfg) if k not in CFG_IGNORED},
        "env": {k: env[k] for k in sorted(env) if k.startswith(ENV_PREFIXES)},
    }
    if model_config is not None:
        snapshot["model"] = {"config.json sha256": hashlib.sha256(model_config).hexdigest()}
    return snapshot


def config_hash(snapshot):
    blob = json.dumps(snapshot, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def cache_roots(cache_dir, home):
    """{key inside the tar: directory on this machine} for every cached directory."""
    roots = {f"cache/{name}": Path(cache_dir, name) for name in CACHE_SUBDIRS}
    roots[TILELANG_KEY] = Path(home, ".tilelang")
    return roots


def make_manifest(runtime, snapshot, roots, label=None):
    return {
        "kind": KIND,
        "format": FORMAT,
        "runtime": dict(runtime),
        "config_hash": config_hash(snapshot),
        "config": snapshot,
        "roots": {key: str(path) for key, path in sorted(roots.items())},
        "label": label,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def mismatches(saved, current):
    """Why a saved manifest cannot be used here; empty when it matches."""
    problems = []
    for key in ("kind", "format"):
        if saved.get(key) != current.get(key):
            problems.append(f"{key}: saved {saved.get(key)!r}, here {current.get(key)!r}")
    if problems:
        return problems
    saved_rt, current_rt = saved.get("runtime", {}), current.get("runtime", {})
    for key in sorted(set(saved_rt) | set(current_rt)):
        if key not in INFO_ONLY and saved_rt.get(key) != current_rt.get(key):
            problems.append(f"{key}: saved {saved_rt.get(key)!r}, here {current_rt.get(key)!r}")
    if saved.get("config_hash") != current.get("config_hash"):
        differ = []
        saved_config, current_config = saved.get("config", {}), current.get("config", {})
        for section in sorted(set(saved_config) | set(current_config)):
            a, b = saved_config.get(section, {}), current_config.get(section, {})
            differ += [f"{section}.{k}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
        problems.append(
            f"config_hash: saved {saved.get('config_hash')}, here {current.get('config_hash')}"
            + (f" ({', '.join(differ)})" if differ else "")
        )
    return problems


def _skipped(name):
    # Lock files would block the next build (torch's FileBaton waits while one
    # exists); tmp.pid_* are Triton's half-written entries and .staging-* are
    # SGLang JIT builds still in progress.
    return (name == "lock" or name.endswith(".lock") or name.startswith("tmp.pid_")
            or name.startswith(".staging-"))


def iter_cache_files(roots):
    """Yield (arcname, path) for every regular cache file, in a stable order."""
    for key, root in sorted(roots.items()):
        root = Path(root)
        if not root.is_dir() or root.is_symlink():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if not _skipped(d))
            for name in sorted(filenames):
                path = Path(dirpath, name)
                if _skipped(name) or path.is_symlink() or not path.is_file():
                    continue
                yield f"{key}/{path.relative_to(root).as_posix()}", path


def _write_atomic(path, data):
    tmp = path.with_name(path.name + ".partial")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _manifest_bytes(manifest):
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()


def _size(path):
    """The file's size, or 0 if it has gone since it was listed."""
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def save(tar_path, roots, manifest, reserve=1 << 30):
    """Write the cache files under ``roots`` to ``tar_path`` plus a manifest beside it.

    Returns the manifest as written (with file and byte counts). The tar holds
    each root's files under its key, with their modification times preserved,
    and then the manifest at its root. A file removed after the listing (a
    build finishing) is left out. Refuses (OSError) when the disk would keep
    less than ``reserve`` bytes free. An error while writing the tar removes
    its partial file, so nothing half-written is left in the output.
    """
    tar_path = Path(tar_path)
    files = list(iter_cache_files(roots))
    content = sum(_size(p) for _, p in files)
    free = shutil.disk_usage(tar_path.parent).free
    if content + reserve > free:
        raise OSError(f"{content} bytes of cache would leave less than {reserve} of {free} free bytes")
    counts = {key: 0 for key in roots}
    added, content = 0, 0
    tmp = tar_path.with_name(tar_path.name + ".partial")
    try:
        with tarfile.open(tmp, "w", format=tarfile.PAX_FORMAT) as tar:
            for arcname, path in files:
                try:
                    size = path.stat().st_size
                    # Raises before writing anything when the file has gone.
                    tar.add(str(path), arcname=arcname, recursive=False)
                except FileNotFoundError:
                    continue
                counts[next(k for k in roots if arcname.startswith(k + "/"))] += 1
                added, content = added + 1, content + size
            # Last, so its counts are of the files actually stored.
            manifest = dict(manifest, files=added, content_bytes=content, files_per_root=counts)
            inner = _manifest_bytes(manifest)
            info = tarfile.TarInfo(MANIFEST_NAME)
            info.size, info.mtime, info.mode = len(inner), int(time.time()), 0o644
            tar.addfile(info, io.BytesIO(inner))
        manifest["tar_bytes"] = tmp.stat().st_size
        os.replace(tmp, tar_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _write_atomic(tar_path.with_name(MANIFEST_NAME), _manifest_bytes(manifest))
    return manifest


def candidate_dirs(path, input_root="/kaggle/input"):
    """``path`` plus its other Kaggle mount layout.

    Datasets mount as /kaggle/input/datasets/<owner>/<slug> on some kernels and
    as /kaggle/input/<slug> on others (the same fallback the notebook's
    _resolve_input uses).
    """
    path, input_root = PurePosixPath(path), PurePosixPath(input_root)
    candidates = [path]
    try:
        rel = path.relative_to(input_root).parts
    except ValueError:
        return candidates
    if len(rel) >= 3 and rel[0] == "datasets":
        candidates.append(input_root.joinpath(*rel[2:]))
    return candidates


def _read_json(path):
    return json.loads(Path(path).read_text())


def _tar_manifest(tar_path):
    with tarfile.open(tar_path) as tar:
        member = tar.getmember(MANIFEST_NAME)
        return json.loads(tar.extractfile(member).read())


def locate(path):
    """Find a saved cache under a mounted directory.

    Returns None, or {"kind": "tar" | "tree", "manifest", "path"}. "tar" is the
    archive as uploaded; "tree" is the root of an archive Kaggle unpacked on
    upload (it holds the manifest and the cached directories).
    """
    for base in map(Path, candidate_dirs(path)):
        if not base.is_dir():
            continue
        tars = sorted(p for p in base.rglob(TAR_NAME) if p.is_file())
        if len(tars) > 1:
            raise ValueError(f"several {TAR_NAME} under {base}: {tars}")
        if tars:
            sidecar = tars[0].with_name(MANIFEST_NAME)
            manifest = _read_json(sidecar) if sidecar.is_file() else _tar_manifest(tars[0])
            return {"kind": "tar", "manifest": manifest, "path": tars[0]}
        trees = []
        for manifest_path in sorted(base.rglob(MANIFEST_NAME)):
            if not manifest_path.is_file():
                continue
            manifest = _read_json(manifest_path)
            keys = manifest.get("roots", {})
            if any((manifest_path.parent / key).is_dir() for key in keys):
                trees.append({"kind": "tree", "manifest": manifest, "path": manifest_path.parent})
        if len(trees) > 1:
            raise ValueError(f"several unpacked caches under {base}: {[t['path'] for t in trees]}")
        if trees:
            return trees[0]
    return None


def _check_members(members, keys):
    """Refuse anything but plain files and directories under a known root."""
    for m in members:
        parts = PurePosixPath(m.name).parts
        if m.name.startswith("/") or ".." in parts or not (m.isfile() or m.isdir()):
            raise ValueError(f"unsafe tar member: {m.name!r}")
        if m.name != MANIFEST_NAME and not any(m.name == k or m.name.startswith(k + "/") for k in keys):
            raise ValueError(f"tar member outside the cache roots: {m.name!r}")


def _stage_tar(tar_path, staging, keys):
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        _check_members(members, keys)
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(staging, members=[m for m in members if m.name != MANIFEST_NAME], **kwargs)


def _stage_tree(tree, staging, keys):
    for key in keys:
        src = Path(tree, key)
        if src.is_dir():
            shutil.copytree(src, Path(staging, key), symlinks=False, dirs_exist_ok=True)


def _install(staged, dest):
    """Move a staged root into place, merging into an existing directory."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_dir() and not any(dest.iterdir()):
        dest.rmdir()
    if not dest.exists():
        shutil.move(str(staged), str(dest))
        return
    for dirpath, _, filenames in os.walk(staged):
        target_dir = dest / Path(dirpath).relative_to(staged)
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            shutil.move(str(Path(dirpath, name)), str(target_dir / name))


def restore(source, roots, current, staging):
    """Restore a matching cache from ``source`` into ``roots``.

    Returns (restored, message). Nothing is installed unless the manifest
    matches ``current`` and the whole archive unpacked cleanly into
    ``staging``, so a mismatched, truncated or unsafe cache leaves the launch
    cold. Installing then moves each staged root into place.
    """
    found = locate(source)
    if found is None:
        tried = ", ".join(map(str, candidate_dirs(source)))
        return False, f"no {TAR_NAME} or unpacked cache under {tried}"
    saved = found["manifest"]
    problems = mismatches(saved, current)
    if problems:
        return False, f"{found['path']} does not match this runtime: " + "; ".join(problems)
    if found["kind"] == "tar" and saved.get("tar_bytes") not in (None, found["path"].stat().st_size):
        return False, f"{found['path']} is {found['path'].stat().st_size} bytes, manifest says {saved['tar_bytes']}"
    staging = Path(staging)
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    keys = sorted(roots)
    try:
        (_stage_tar if found["kind"] == "tar" else _stage_tree)(found["path"], staging, keys)
        files = sum(len(f) for key in keys for _, _, f in os.walk(staging / key))
        for key in keys:
            if (staging / key).is_dir():
                _install(staging / key, Path(roots[key]))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return True, f"restored {files} files from {found['path']} ({found['kind']}, saved {saved.get('created')})"


def unsaved_dirs(cache_dir, roots):
    """[(directory, bytes)] for files under ``cache_dir`` that a snapshot leaves out.

    The save run prints this, so kernels the server caches somewhere else show up.
    """
    saved = [Path(p) for p in roots.values()]
    report = []
    for child in sorted(Path(cache_dir).iterdir()) if Path(cache_dir).is_dir() else []:
        if not child.is_dir() or child in saved:
            continue
        size = sum(p.stat().st_size for p in child.rglob("*")
                   if p.is_file() and not p.is_symlink() and not any(p.is_relative_to(s) for s in saved))
        if size:
            report.append((str(child), size))
    return report
