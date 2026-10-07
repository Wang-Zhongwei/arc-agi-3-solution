"""Pennyroyal SGLang server for Qwen3.8 Flash Next on Kaggle, as a TAAF setup command.

The serving logic is the launcher cell of dfranzen's milestone-2 notebook (as run
in arc3-v207). The notebook globals it expects are derived here from
ARC3_PENNYROYAL_SPEC, which inference.framework.kaggle embeds from the config's
kaggle section. Beyond Daniel's AutoRound target + separate INT4 draft, it also
serves a compressed-tensors target with an embedded MTP head (lordhansolo's mixed
NVFP4/FP8 model), optionally with an FP8 PLE table swapped in
(`ple_fp8_model_source`) and lordhansolo's shard window prefetch
(`sglang.WINDOW_PREFETCH`); see pennyroyal_views.py. Tuning flags for that model
(`sglang.EMBED_TOKENS_HOST`, `MOE_MARLIN`, `LOWM_FP8`, `LOWM_MAX_M`) patch the installed
SGLang first; see pennyroyal_patches.py. Run by the TAAF notebook before the benchmark loads;
the server outlives this process (new session), and the analyzer settings plus
the endpoint are handed to the run through TAAF_KAGGLE_SETUP_ENV.
"""
import glob
import json
import os
import sys
import threading
import time
from pathlib import Path

SPEC = json.loads(os.environ["ARC3_PENNYROYAL_SPEC"])
NOTEBOOK_START_TIME = time.time()
SERVER_STARTUP_TIMEOUT = int(SPEC.get("server_startup_timeout_s", 12 * 60))
WORKING_DIR = os.environ.get("TAAF_KAGGLE_WORKING_DIR", "/kaggle/working")
SERVED_MODEL_NAME = SPEC.get("served_model_name", "flashnext")
SERVED_MODEL_PORT = int(SPEC.get("port", 8001))


def _dataset_dir(ref: str) -> str:
    """TAAF maps each attached dataset to its mount; fall back to both layouts."""
    mounts = json.loads(os.environ.get("TAAF_KAGGLE_INPUT_PATHS", "{}"))
    owner, slug = ref.split("/", 1)
    for path in (mounts.get(ref), f"/kaggle/input/datasets/{owner}/{slug}", f"/kaggle/input/{slug}"):
        if path and os.path.isdir(path):
            return path
    raise FileNotFoundError(f"dataset {ref} is not mounted; /kaggle/input has {os.listdir('/kaggle/input')}")


def _model_dir(ref: str) -> str:
    """owner/model/framework/variation/version -> its /kaggle/input/models mount."""
    owner, model, framework, variation, version = ref.split("/")
    exact = f"/kaggle/input/models/{owner}/{model}/{framework.lower()}/{variation}/{version}"
    if os.path.isdir(exact):
        return exact
    hits = sorted(p for p in glob.glob(f"/kaggle/input/**/{model}/**/config.json", recursive=True))
    if hits:
        return str(Path(hits[0]).parent)
    raise FileNotFoundError(f"model {ref} is not mounted at {exact}")


WHEELHOUSE_DIR = _dataset_dir(SPEC["wheelhouse_dataset_source"])
MODEL_DIR = _model_dir(SPEC["model_source"])
DRAFT_MODEL_DIR = _model_dir(SPEC["draft_model_source"]) if SPEC.get("draft_model_source") else ""
PLE_FP8_DIR = _model_dir(SPEC["ple_fp8_model_source"]) if SPEC.get("ple_fp8_model_source") else ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pennyroyal_views as views

STARTUP_TIMES = {}


def mark_startup(stage: str) -> None:
    """Seconds from launcher start, written to pennyroyal-startup.json at the end."""
    STARTUP_TIMES[stage] = round(time.time() - NOTEBOOK_START_TIME, 1)
    print(f"[startup] {stage} at {STARTUP_TIMES[stage]}s", flush=True)

from concurrent.futures import as_completed, ThreadPoolExecutor

def precache(*paths, times=1, delay=0, threads=16) -> None:
    time.sleep(delay)
    print(f'Precaching: Started...')
    start_time = time.time()

    CHUNK_SIZE = 32 * 1024 * 1024  # 128 MB

    chunks = []  # list of (path, offset, length)
    total_size = 0
    file_count = 0

    def _add_file(i: int, path: str) -> None:
        nonlocal total_size, file_count
        size = os.path.getsize(path)
        total_size += size
        file_count += 1
        for offset in range(0, max(size, 1), CHUNK_SIZE):
            length = min(CHUNK_SIZE, size - offset)
            chunks.append((i, path, offset, length))

    for i, path in enumerate(paths):
        try:
            if os.path.isfile(path):
                _add_file(i, path)
                continue
            for root, _, files in os.walk(path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    if os.path.isfile(file_path):
                        _add_file(i, file_path)
        except: pass

    chunks.sort()

    _local = threading.local()

    def _read_chunk(chunk: tuple) -> None:
        try:
            buf = _local.buf
        except AttributeError:
            _local.buf = buf = bytearray(CHUNK_SIZE)
        i, path, offset, _ = chunk
        try:
            with open(path, 'rb') as f:
                f.seek(offset)
                remaining = CHUNK_SIZE
                for i in range(1024):
                    n = f.readinto(buf)
                    if not n:
                        break
                    remaining -= n
                    if remaining <= 0: break
        except: pass

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for _ in range(times):
            list(pool.map(_read_chunk, chunks))

    elapsed = time.time() - start_time
    print(f'Precaching: loaded {file_count} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n')


precache_wheels_thread = threading.Thread(target=precache, args=(WHEELHOUSE_DIR,))
precache_wheels_thread.start()
precache_wheels_thread.join()
mark_startup("wheels_precached")

# Model read tuning from kaggle.sglang. 6 precache + 8 prefetch threads load about
# 20% faster than Daniel's cell's 3 + 4; see docs/serving/startup-read-threads.md.
# Early precache (during the install instead of 60 s after launch) outruns the
# loader and gets evicted, since 185 GB exceeds RAM, so it stays off.
_READ = SPEC.get("sglang", {})
MODEL_PRECACHE_THREADS = int(_READ.get("MODEL_PRECACHE_THREADS", 6))
MODEL_PRECACHE_EARLY = bool(_READ.get("MODEL_PRECACHE_EARLY", False))

# Daemon: this process exits once the server is up, and must not wait on it.
# MODEL_PRECACHE_THREADS = 0 leaves the read to sglang's own prefetch.
precache_model_thread = threading.Thread(
    target=precache, args=(MODEL_DIR, *([DRAFT_MODEL_DIR] if DRAFT_MODEL_DIR else [])),
    kwargs={'delay': 0 if MODEL_PRECACHE_EARLY else 60, 'threads': MODEL_PRECACHE_THREADS}, daemon=True,
) if MODEL_PRECACHE_THREADS > 0 else None
if precache_model_thread and MODEL_PRECACHE_EARLY:
    precache_model_thread.start()

# Paste this entire file into the launcher cell.
# Required notebook variables: MODEL_DIR, DRAFT_MODEL_DIR, WHEELHOUSE_DIR,
# WORKING_DIR, SERVED_MODEL_NAME, SERVED_MODEL_PORT.
# MODEL_DIR: Intel AutoRound INT4 target (BF16 PLE or albucino FP8 PLE).
# DRAFT_MODEL_DIR: downloaded albucino runtime/mtp-int4-g32, or its dataset root.
import os, sys, glob, time, shutil, subprocess, urllib.request, urllib.error
import copy, hashlib, json, re, shlex, socket, zipfile
from pathlib import Path

PREFIX = "/tmp/sgl-intel"       # Separate from the previous fork's /tmp/sgl venv.
LOG = str(Path(WORKING_DIR) / "serve.log")
CFG = dict(
    CTX=(116+12+8)*1024,
    MEMFRAC=0.96,
    MAXREQ=10,
    CUDAGRAPH_MAXBS=10,
    MAMBA_CACHE=60,
    CHUNK=8192,
    MAX_PREFILL=16384,
    SPEC=True,
    FRSPEC=True,
    AUTOTUNE=True,
    # Preserve the original Kaggle checkpoint prefetch/cache behavior.
    PREFETCH_CHECKPOINTS=True,
    DROP_CACHE_AFTER_LOAD=True,
    # sglang's one-shard lookahead prefetch readers (sglang's own default is 4).
    PREFETCH_THREADS=8,
    LINEAR_BACKEND="flashinfer",
    KVDTYPE="fp8_e4m3",
    SSM_DTYPE="bfloat16",
    MAMBA_RADIX="extra_buffer",
    GDN_MTP_CACHE_MODE="none",
    SERVED_NAME=SERVED_MODEL_NAME,
    SPEC_ACCEPT_SINGLE=1.0,
    SPEC_ACCEPT_ACC=1.0,
    SPEC_STEPS=3,
    OFFLOAD_EMBED_TOKENS=False,
    WIDE_BF16_GEMM=False,
    LOG_BF16_GEMM_SHAPES=False,
    # The harness's thinking budget (custom_params.thinking_budget) is enforced
    # by sglang's reasoner grammar, which only filters tokens in strict mode.
    # Strict mode also keeps tool-call and end-of-turn tokens out of thinking.
    STRICT_THINKING=int(SPEC.get("env", {}).get("LOCAL_ANALYZER_THINKING_TOKEN_BUDGET") or 0) > 0,
    # lordhansolo's loader-following prefetch window (pennyroyal_views). It
    # replaces both the in-process lookahead (PREFETCH_CHECKPOINTS) and the
    # whole-directory precache, which would also read shards the view leaves out.
    WINDOW_PREFETCH=False,
    WINDOW_PREFETCH_GIB=8,
    WINDOW_PREFETCH_THREADS=4,
    # Tuning for lordhansolo's model (pennyroyal_patches.py). Off by default;
    # any of the first three installs the source patches before launch.
    EMBED_TOKENS_HOST=False,
    MOE_MARLIN=False,
    LOWM_FP8=False,
    LOWM_MAX_M=32,
    # If the tuned server exits before it is ready (a failed self-check, a
    # kernel that does not compile), relaunch once with the tuning off.
    TUNEUP_FALLBACK=True,
    # Run pennyroyal_marlin_debug.py on the GPU, write marlin-debug.json, and
    # stop before starting a server (the notebook then fails at setup).
    MARLIN_DEBUG_ONLY=False,
    # The MTP draft's never-loaded LM head goes on the meta device (+1.18 GiB KV).
    MTP_SHARED_HEAD=False,
    # Serve the checkpoint's own BF16 PLE table even if its config declares FP8
    # (lordhansolo's checkpoint does; his vLLM run kept BF16).
    PLE_BF16=False,
    # Draft vocabulary as a JSON of half-open token_ranges, relative to
    # ARC3-Inference/ (lordhansolo's configs/draft_vocab_32k.json). Replaces FRSPEC.
    DRAFT_VOCAB="",
    # SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK: one Mamba slot less per running request.
    MAMBA_SKIP_DECODE_LOCK=False,
)
CFG.update(SPEC.get("sglang", {}))
if CFG["WINDOW_PREFETCH"]: CFG["PREFETCH_CHECKPOINTS"] = False
CFG["SPEC"] = CFG["SPEC"] and (bool(DRAFT_MODEL_DIR) or views.has_embedded_mtp(MODEL_DIR))
VENV = f"{PREFIX}/venv"
PYTHON = f"{VENV}/bin/python"
SGLANG = f"{VENV}/bin/sglang"
TOKEN_MAP_SHA = "becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25"
# Intel's tokenizer, and primitive-ai's (lordhansolo's model, RadixArk): the latter
# re-serializes merges and adds \p{M} to the pre-tokenizer split, but its vocab
# and added tokens are id-for-id identical, so the 64K map applies to both.
TOKENIZER_SHAS = {
    "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
}
EXPERT_TARGET = r"re:^mtp\.layers\.0\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$"
DENSE_IGNORE = r"re:^(?!mtp\.layers\.0\.mlp\.experts(?:\.|$)).*"


def run(cmd, *, check=True, env=None):
    cmd = list(map(str, cmd))
    print("$", shlex.join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if result.stdout.strip(): print(result.stdout.strip()[-6000:])
    if result.stderr.strip() and result.returncode: print(result.stderr.strip()[-6000:])
    if check and result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {shlex.join(cmd)}")
    return result


def find_unique(root, name, *, directory=False, required=True):
    root = Path(root)
    direct = root if root.name == name else root / name
    valid = lambda p: p.is_dir() if directory else p.is_file()
    if valid(direct): return direct
    hits = sorted(p for p in root.rglob(name) if valid(p))
    if len(hits) > 1:
        raise RuntimeError(f"Multiple {name} matches under {root}; use a more specific directory: {hits}")
    if hits: return hits[0]
    if required: raise FileNotFoundError(f"{name} not found under {root}")
    return None


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()


def indexed_shards(root):
    root = Path(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    if not names: raise RuntimeError(f"Empty weight index: {root}")
    for name in names:
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"Invalid shard path: {name}")
        p = root / rel
        if not p.is_file() or p.stat().st_size < 8:
            raise RuntimeError(f"Missing/empty shard: {p}")
    return index, names


def prepare_draft_view(target, source_root, work_root):
    """Adapt metadata only; link original dataset weights without copying them."""
    target, source_root, work_root = map(Path, (target, source_root, work_root))
    configs = [source_root / "config.json"] if (source_root / "config.json").is_file() else []
    configs += sorted(p for p in source_root.rglob("config.json") if p not in configs)
    candidates = []
    for path in configs:
        cfg = json.loads(path.read_text())
        q = cfg.get("quantization_config", {})
        if q.get("quant_method") == "compressed-tensors" and "mtp_routed_experts" in q.get("config_groups", {}):
            candidates.append((path.parent, cfg))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one INT4 g32 MTP checkpoint under DRAFT_MODEL_DIR={source_root}; found {len(candidates)}")
    source, cfg = candidates[0]
    index, shards = indexed_shards(source)
    if not any(k.startswith("mtp.layers.0.mlp.experts.") for k in index["weight_map"]):
        raise RuntimeError("Draft index does not contain the expected MTP experts")
    cfg = copy.deepcopy(cfg)
    q = cfg["quantization_config"]
    group = q["config_groups"]["mtp_routed_experts"]
    w = group["weights"]
    if (w["num_bits"], w["group_size"], w["symmetric"]) != (4, 32, True):
        raise RuntimeError("Expected symmetric INT4 group32 draft experts")
    if group["targets"] not in (["RoutedExperts"], [EXPERT_TARGET]):
        raise RuntimeError(f"Unexpected draft target rules: {group['targets']}")
    if q.get("ignore", []) not in ([], [DENSE_IGNORE]):
        raise RuntimeError("Unexpected draft ignore rules")
    group["targets"] = [EXPERT_TARGET]
    q["ignore"] = [DENSE_IGNORE]
    # Content/path-specific directory makes notebook reruns idempotent without
    # overwriting an unrelated view. We never modify the read-only datasets.
    identity = json.dumps({"source": str(source.resolve()), "target": str(target.resolve()), "config": cfg}, sort_keys=True)
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
    view = work_root / f"draft-view-{suffix}"
    view.mkdir(parents=True, exist_ok=True)
    config_path = view / "config.json"
    config_text = json.dumps(cfg, indent=2) + "\n"
    if config_path.exists() or config_path.is_symlink():
        if config_path.is_symlink() or config_path.read_text() != config_text:
            raise RuntimeError(f"Unexpected existing draft-view config: {config_path}")
    else:
        config_path.write_text(config_text)
    links = {name: source / name for name in shards}
    links["model.safetensors.index.json"] = source / "model.safetensors.index.json"
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "generation_config.json"):
        if (target / name).is_file(): links[name] = target / name
    for name, src in links.items():
        dest = view / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            if not dest.is_symlink() or dest.resolve() != src.resolve():
                raise RuntimeError(f"Unexpected existing draft-view file: {dest}")
        else:
            dest.symlink_to(src.resolve())
    print(f"Draft source: {source}\nSGLang draft view: {view} ({len(shards)} linked weight shards)")
    return view


# ---- validate notebook inputs / identify the offline bundle ----
MODEL_DIR = str(Path(MODEL_DIR).expanduser().resolve())
assert Path(MODEL_DIR).is_dir(), f"model dir not found: {MODEL_DIR}"
assert CFG["CUDAGRAPH_MAXBS"] >= CFG["MAXREQ"], "CUDAGRAPH_MAXBS must cover MAXREQ"
assert CFG["CHUNK"] > 0 and CFG["CHUNK"] % 64 == 0
# Refuse to start a second server on an occupied port; do not kill other jobs.
with socket.socket() as probe:
    if probe.connect_ex(("127.0.0.1", int(SERVED_MODEL_PORT))) == 0:
        raise RuntimeError(f"Port {SERVED_MODEL_PORT} is occupied; stop the existing server before rerunning")
Path(PREFIX).mkdir(parents=True, exist_ok=True)
Path(LOG).parent.mkdir(parents=True, exist_ok=True)
WHEELS = find_unique(WHEELHOUSE_DIR, "wheels", directory=True)
LOCK = find_unique(WHEELHOUSE_DIR, "requirements.lock")
sglang_wheels = sorted(WHEELS.glob("sglang-*.whl"))
assert len(sglang_wheels) == 1, f"Expected one SGLang wheel, found {sglang_wheels}"
wheel = sglang_wheels[0]
install_marker = Path(PREFIX) / "installed-bundle.txt"

# ---- one-time offline install, isolated from the old mratsim environment ----
install_env = dict(os.environ, UV_OFFLINE="1", UV_PYTHON_DOWNLOADS="never", PYTHONNOUSERSITE="1")
install_env.pop("PYTHONPATH", None)
if not Path(SGLANG).exists() or not install_marker.is_file():
    print("=== installing Pennyroyal (offline) ===")
    # Bootstrap uv directly from its wheel; no global pip changes/ensurepip.
    uv_wheels = sorted(WHEELS.glob("uv-*.whl"))
    assert len(uv_wheels) == 1, "Expected one bundled uv wheel"
    uv_binary = Path(PREFIX) / "bin/uv"
    uv_binary.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(uv_wheels[0]) as archive:
        members = [n for n in archive.namelist() if Path(n).name == "uv" and not n.endswith("/")]
        assert len(members) == 1, "Cannot identify the Linux uv executable in its wheel"
        binary = archive.read(members[0])
        assert binary.startswith(b"\x7fELF"), "Bundled uv executable is not a Linux ELF binary"
        uv_binary.write_bytes(binary)
    uv_binary.chmod(0o755)
    uv = [str(uv_binary)]
    if not Path(PYTHON).exists():
        run(uv + ["venv", "--python", sys.executable, VENV], env=install_env)
    install = uv + ["pip", "install", "--python", PYTHON, "--no-index", "--find-links", str(WHEELS)]
    run(install + ["-r", str(LOCK)], env=install_env)
    run(install + ["--reinstall", "--no-deps", str(wheel)], env=install_env)
    install_marker.write_text('install')
else:
    print(f"Matching offline bundle already installed at {VENV}")
mark_startup("bundle_installed")

# ---- tuning patches for lordhansolo's model (no-ops unless their env flags are set) ----
TUNEUP_ENV = {
    "SGLANG_QWEN4_EXP_HOST_EMBED_TOKENS": "1" if CFG["EMBED_TOKENS_HOST"] else "",
    "SGLANG_CT_NVFP4_MOE_MARLIN": "1" if CFG["MOE_MARLIN"] else "",
    "SGLANG_SM120_LOWM_FP8_GEMM": "1" if CFG["LOWM_FP8"] else "",
    "SGLANG_QWEN4_EXP_MTP_SHARED_HEAD": "1" if CFG["MTP_SHARED_HEAD"] else "",
}
TUNEUP_ENV = {k: v for k, v in TUNEUP_ENV.items() if v}
TUNEUP = {"requested": sorted(TUNEUP_ENV), "low_m_max": int(CFG["LOWM_MAX_M"]), "patched": [], "fallback": None}
if TUNEUP_ENV or int(CFG["LOWM_MAX_M"]) != 32 or CFG["MARLIN_DEBUG_ONLY"]:
    import pennyroyal_patches
    site_packages = sorted(Path(VENV).glob("lib/python*/site-packages"))
    try:
        TUNEUP["patched"] = pennyroyal_patches.apply(site_packages[0])
        print("Tuning patches applied:", TUNEUP["patched"] or "already present")
    except Exception as exc:
        print(f"Tuning patches not applied, serving untuned: {exc}")
        TUNEUP["fallback"] = f"patch: {exc}"
        TUNEUP_ENV, CFG["LOWM_MAX_M"] = {}, 32
mark_startup("patches_applied")

# ---- bundled CUDA toolkit and linker symlinks ----
cu13 = sorted(Path(VENV).glob("lib/python*/site-packages/nvidia/cu13"))
if not cu13 or not (cu13[0] / "bin/nvcc").is_file():
    raise RuntimeError("Bundled CUDA 13 nvcc missing; check that the complete Pennyroyal wheelhouse was uploaded")
CUDA_HOME = str(cu13[0])
libdir = Path(CUDA_HOME) / "lib"
lib64 = Path(CUDA_HOME) / "lib64"
if not lib64.exists() and not lib64.is_symlink(): lib64.symlink_to("lib")
for so in sorted(libdir.glob("*.so.*")):
    link = libdir / re.sub(r"\.so\..*$", ".so", so.name)
    if not link.exists() and not link.is_symlink(): link.symlink_to(so.name)
assert (libdir / "libcudart.so").exists(), "libcudart.so missing: JIT linker cannot find CUDA runtime"
if not (libdir / "libcuda.so").exists():
    candidates = []
    for directory in ("/usr/local/nvidia/lib64", "/usr/local/nvidia/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
        candidates += sorted(p for p in Path(directory).glob("libcuda.so*") if p.is_file())
    if not candidates:
        code = 'import torch; torch.cuda.init(); print(open("/proc/self/maps").read())'
        maps = run([PYTHON, "-c", code], env=install_env).stdout
        candidates = [Path(p) for p in re.findall(r"(/[^\s]*libcuda\.so[^\s]*)", maps)]
    if not candidates: raise RuntimeError("GPU driver libcuda.so not found")
    link = libdir / "libcuda.so"
    if link.is_symlink(): link.unlink()
    link.symlink_to(candidates[0].resolve())
CXX = next((c for c in ("g++-15", "g++-14", "g++-13", "g++-12", "g++-11", "g++") if shutil.which(c)), None)
assert CXX, "Host C++ compiler not found"
CC = CXX.replace("g++", "gcc")
assert shutil.which(CC), f"Matching C compiler missing: {CC}"

# ---- runtime environment ----
env = dict(os.environ)
env.pop("PYTHONPATH", None)
cache = f"{PREFIX}/cache"
for name in ("huggingface", "torch", "torchinductor", "triton", "flashinfer", "cuda", "sglang/jit"):
    Path(cache, name).mkdir(parents=True, exist_ok=True)
env.update({
    "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1", "UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_HOME": CUDA_HOME, "CUDACXX": f"{CUDA_HOME}/bin/nvcc",
    "PATH": f"{CUDA_HOME}/bin:{VENV}/bin:" + env.get("PATH", ""),
    "LD_LIBRARY_PATH": f"{CUDA_HOME}/lib64:{CUDA_HOME}/lib:" + env.get("LD_LIBRARY_PATH", ""),
    "CC": CC, "CXX": CXX, "CUDAHOSTCXX": CXX,
    "TORCH_CUDA_ARCH_LIST": "12.0",
    "HF_HOME": f"{cache}/huggingface", "XDG_CACHE_HOME": cache,
    "TORCH_HOME": f"{cache}/torch", "CUDA_CACHE_PATH": f"{cache}/cuda",
    "TORCHINDUCTOR_CACHE_DIR": f"{cache}/torchinductor", "TRITON_CACHE_DIR": f"{cache}/triton",
    "FLASHINFER_WORKSPACE_BASE": f"{cache}/flashinfer",
    "SGLANG_CACHE_DIR": f"{cache}/sglang", "SGLANG_JIT_CACHE_DIR": f"{cache}/sglang/jit",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false",
    "MAX_JOBS": "8", "CMAKE_BUILD_PARALLEL_LEVEL": "8",
    "FLASHINFER_NINJA_JOBS": "8", "FLASHINFER_NVCC_THREADS": "2", "TORCHINDUCTOR_COMPILE_THREADS": "8",
    "SGLANG_ENABLE_SM120_LOWM_BF16_GEMM": "1", "SGLANG_SM120_ONLINE_MXFP8": "0",
    "SGLANG_SM120_LOWM_FP8_WEIGHT": "0", "SGLANG_SM120_LM_HEAD_FP8": "0",
    "SGLANG_MAMBA_CONV_DTYPE": "bfloat16", "SGLANG_NUMA_BIND_V2": "false",
    "SGLANG_MM_PREPROCESS_DEVICE": "cpu", "NUMPY_MADVISE_HUGEPAGE": "0",
    "ARC3_OFFLOAD_EMBED_TOKENS": "1" if CFG["OFFLOAD_EMBED_TOKENS"] else "0",
    "ARC3_WIDE_BF16_GEMM": "1" if CFG["WIDE_BF16_GEMM"] else "0",
    "SGLANG_BF16_GEMM_LOG_SHAPES": "1" if CFG["LOG_BF16_GEMM_SHAPES"] else "0",
})
if CFG["OFFLOAD_EMBED_TOKENS"] or CFG["WIDE_BF16_GEMM"]:
    patch_args = [PYTHON, Path(__file__).with_name("pennyroyal_patch.py")]
    if CFG["OFFLOAD_EMBED_TOKENS"]:
        patch_args.append("--embedding")
    if CFG["WIDE_BF16_GEMM"]:
        patch_args.append("--wide-gemm")
    run(patch_args, env=env)
if CFG["MAMBA_SKIP_DECODE_LOCK"]:
    env["SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK"] = "1"
run([str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], env=env)
print("Host compiler:", CXX)
# Report the installed engine and device before loading weights.
run([PYTHON, "-c", '''import pathlib, sglang, torch
print("sglang", sglang.__version__, "| torch", torch.__version__)
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
'''], env=env)

# ---- validate target and create the read-only views SGLang serves ----
target_cfg = json.loads(Path(MODEL_DIR, "config.json").read_text())
q = target_cfg.get("quantization_config", {})
QUANT = q.get("quant_method")
if QUANT == "auto-round":
    assert q.get("bits") == 4, "AutoRound MODEL_DIR must be the Intel/albucino INT4 checkpoint"
elif QUANT == "compressed-tensors":
    print("Compressed-tensors target, format:", q.get("format"), "| groups:",
          {k: g.get("format") for k, g in q.get("config_groups", {}).items()})
else:
    raise RuntimeError(f"Unsupported target quantization {QUANT!r}; expected auto-round or compressed-tensors")
if PLE_FP8_DIR:
    SERVE_DIR = str(views.build_target_view(MODEL_DIR, PLE_FP8_DIR, PREFIX))
elif CFG["PLE_BF16"]:
    SERVE_DIR = str(views.build_ple_dtype_view(MODEL_DIR, PREFIX, "bfloat16"))
else:
    SERVE_DIR = MODEL_DIR
serve_cfg = json.loads(Path(SERVE_DIR, "config.json").read_text())
_, target_shards = indexed_shards(SERVE_DIR)
print("Target:", SERVE_DIR, "| shards:", len(target_shards), "| PLE dtype:", serve_cfg.get("text_config", {}).get("ple_embedding_dtype", "checkpoint default (typically BF16)"))
DRAFT_VIEW = None
if CFG["SPEC"] and DRAFT_MODEL_DIR:
    assert Path(DRAFT_MODEL_DIR).is_dir(), f"draft dir not found: {DRAFT_MODEL_DIR}"
    DRAFT_VIEW = prepare_draft_view(SERVE_DIR, DRAFT_MODEL_DIR, PREFIX)
elif CFG["SPEC"]:
    # The target's own MTP head; its experts are quantized under the target's
    # compressed-tensors config, which the draft view keeps.
    DRAFT_VIEW = views.build_mtp_draft_view(SERVE_DIR, PREFIX)
mark_startup("views_ready")

# ---- assemble Pennyroyal / AutoRound launch arguments ----
graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]})
graph_bs = [n for n in graph_bs if n <= CFG["CUDAGRAPH_MAXBS"]]
args = [SGLANG, "serve", "--model-path", SERVE_DIR, "--load-format", "safetensors",
    "--model-loader-extra-config", '{"enable_multithread_load":false}',
    "--served-model-name", str(CFG["SERVED_NAME"]), "--host", "127.0.0.1",
    "--port", str(SERVED_MODEL_PORT), "--tensor-parallel-size", "1",
    "--dtype", "bfloat16", "--quantization", QUANT,
    "--kv-cache-dtype", CFG["KVDTYPE"], "--mem-fraction-static", str(CFG["MEMFRAC"]),
    "--context-length", str(CFG["CTX"]), "--page-size", "64",
    "--max-running-requests", str(CFG["MAXREQ"]), "--chunked-prefill-size", str(CFG["CHUNK"]),
    "--max-prefill-tokens", str(CFG["MAX_PREFILL"]),
    "--cuda-graph-max-bs-decode", str(CFG["CUDAGRAPH_MAXBS"]),
    "--cuda-graph-bs-decode", *map(str, graph_bs),
    "--mamba-ssm-dtype", CFG["SSM_DTYPE"], "--max-mamba-cache-size", str(CFG["MAMBA_CACHE"]),
    "--mamba-radix-cache-strategy", CFG["MAMBA_RADIX"], "--mamba-track-interval", "64",
    "--mamba-backend", CFG["LINEAR_BACKEND"],
    "--linear-attn-decode-backend", CFG["LINEAR_BACKEND"],
    "--linear-attn-prefill-backend", CFG["LINEAR_BACKEND"],
    "--moe-runner-backend", "auto", "--ple-offload-embedding", "--trust-remote-code",
    "--mm-feature-transport", "cpu", "--image-processor-backend", "pil",
    "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
    "--default-chat-template-kwargs", '{"preserve_thinking":true}',
    "--watchdog-timeout", "1800", "--schedule-policy", "lpm", "--warmups", "structured_output",
    "--enable-cache-report", "--enable-metrics", "--enable-request-time-stats-logging"]
if CFG["PREFETCH_CHECKPOINTS"]: args += ["--weight-loader-prefetch-checkpoints"]
if CFG["DROP_CACHE_AFTER_LOAD"]: args += ["--weight-loader-drop-cache-after-load"]
args += ["--weight-loader-prefetch-num-threads", str(int(CFG["PREFETCH_THREADS"]))]
tmpl = Path(SERVE_DIR, "chat_template.jinja")
if tmpl.is_file(): args += ["--chat-template", str(tmpl)]
if not CFG["AUTOTUNE"]: args += ["--disable-flashinfer-autotune"]
if CFG["GDN_MTP_CACHE_MODE"]: args += ["--gdn-mtp-cache-mode", CFG["GDN_MTP_CACHE_MODE"]]
if CFG["STRICT_THINKING"]: args += ["--enable-strict-thinking"]
if CFG["SPEC"]:
    args += ["--speculative-algorithm", "NEXTN", "--speculative-num-steps", str(CFG["SPEC_STEPS"]),
        "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", str(CFG["SPEC_STEPS"]+1),
        "--speculative-draft-model-path", str(DRAFT_VIEW),
        "--speculative-draft-model-quantization", "compressed-tensors", "--speculative-moe-runner-backend", "auto",
        "--speculative-draft-kv-cache-dtype", CFG["KVDTYPE"],
        "--speculative-accept-threshold-single", str(CFG["SPEC_ACCEPT_SINGLE"]),
        "--speculative-accept-threshold-acc", str(CFG["SPEC_ACCEPT_ACC"])]
    if CFG["DRAFT_VOCAB"]:
        # lordhansolo's draft vocabulary: half-open [start, end) id ranges built
        # for primitive-ai's tokenizer, saved as the id list SGLang's token map loads.
        vocab_path = Path(__file__).resolve().parents[2] / CFG["DRAFT_VOCAB"]
        vocab = json.loads(vocab_path.read_text())
        if vocab.get("tokenizer_sha256") != sha256(Path(SERVE_DIR, "tokenizer.json")):
            raise RuntimeError(f"{vocab_path} was built for another tokenizer")
        draft_ids = sorted({i for start, end in vocab["token_ranges"] for i in range(start, end)})
        ids_json = Path(PREFIX, f"{vocab_path.stem}-ids.json")
        ids_json.write_text(json.dumps(draft_ids))
        token_map = Path(PREFIX, f"{vocab_path.stem}-{len(draft_ids)}.pt")
        run([PYTHON, "-c", "import json, sys, torch; "
             "torch.save(torch.tensor(json.load(open(sys.argv[1])), dtype=torch.int64), sys.argv[2])",
             ids_json, token_map], env=env)
        print(f"Draft vocabulary: {len(draft_ids)} ids from {vocab_path}")
        args += ["--speculative-token-map", str(token_map)]
    elif CFG["FRSPEC"]:
        tok = find_unique(WHEELHOUSE_DIR, "hot_tokens_64k.pt", required=False)
        if tok is None: tok = find_unique(WHEELHOUSE_DIR, "flash-next-64k.pt")
        assert sha256(tok) == TOKEN_MAP_SHA, "Wrong FR-Spec map; use the map from the pinned Pennyroyal bundle"
        assert sha256(Path(SERVE_DIR, "tokenizer.json")) in TOKENIZER_SHAS, "Tokenizer differs from the FR-Spec map; set CFG['FRSPEC']=False"
        args += ["--speculative-token-map", str(tok)]

# ---- optional: NVFP4 Marlin bisect on the free GPU, then stop ----
if CFG["MARLIN_DEBUG_ONLY"]:
    report = Path(WORKING_DIR, "marlin-debug.json")
    script = Path(__file__).resolve().parent / "pennyroyal_marlin_debug.py"
    try:
        # -I: keep this directory's modules (kaggle.py, run.py, ...) off sys.path.
        result = subprocess.run([PYTHON, "-I", str(script), MODEL_DIR, str(report)], env=env, timeout=1800)
        print(f"Marlin debug exited {result.returncode}")
    except subprocess.TimeoutExpired:
        print("Marlin debug timed out after 1800 s")
    mark_startup("marlin_debug_done")
    Path(WORKING_DIR, "pennyroyal-startup.json").write_text(json.dumps(
        {"seconds_from_launcher_start": STARTUP_TIMES, "marlin_debug_only": True, "tuneup": TUNEUP}, indent=1))
    raise SystemExit(f"MARLIN_DEBUG_ONLY: results in {report}; not starting the server")

# ---- launch detached and wait for health ----
print("\n" + shlex.join(args) + "\n")


def launch(server_env, prefetch_report):
    """Start the server detached; return (process, log offset, window prefetcher)."""
    log_offset = Path(LOG).stat().st_size if Path(LOG).exists() else 0
    window_prefetch = None
    if CFG["WINDOW_PREFETCH"]:
        phases = [("target", views.loader_shard_order(SERVE_DIR))]
        if DRAFT_VIEW is not None:
            phases.append(("draft", views.loader_shard_order(DRAFT_VIEW)))
        window_prefetch = views.ShardWindowPrefetcher(
            phases, LOG, Path(WORKING_DIR, prefetch_report), log_offset=log_offset,
            threads=int(CFG["WINDOW_PREFETCH_THREADS"]), window_bytes=int(float(CFG["WINDOW_PREFETCH_GIB"]) * 1024**3),
            deadline_seconds=SERVER_STARTUP_TIMEOUT)
        window_prefetch.start()
    with open(LOG, "ab", buffering=0) as logf:
        proc = subprocess.Popen(args, env=server_env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"pid {proc.pid} -> {LOG}")
    Path(WORKING_DIR, "sglang-server.pid").write_text(str(proc.pid))
    return proc, log_offset, window_prefetch


def show_log_tail(log_offset):
    with open(LOG, "rb") as f:
        f.seek(0, 2)
        f.seek(max(log_offset, f.tell()-20000))
        print(f.read().decode(errors="replace"))


def wait_ready(proc, log_offset):
    """'ready', 'exited' (server died) or 'deadline' (still loading at the timeout)."""
    url, t0 = f"http://127.0.0.1:{SERVED_MODEL_PORT}/health", time.time()
    last_report = -30
    while time.time() - NOTEBOOK_START_TIME < SERVER_STARTUP_TIMEOUT:
        if proc.poll() is not None:
            show_log_tail(log_offset)
            return "exited"
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print(f"\nREADY after {int(time.time()-t0)}s -> {url.replace('/health','/v1')}")
                    mark_startup("server_ready")
                    return "ready"
        except Exception:
            pass
        elapsed = int(time.time()-t0)
        if elapsed-last_report >= 30:
            print(f"  {elapsed}s ...", flush=True); last_report=elapsed
        time.sleep(3)
    print(f"\nDEADLINE at {int(time.time()-NOTEBOOK_START_TIME)}s from notebook start; "
          f"releasing the benchmark with the server still loading")
    return "deadline"


def stop_prefetch(window_prefetch):
    if window_prefetch is not None:
        window_prefetch.stop()
        window_prefetch.join(timeout=5)


if not CFG["WINDOW_PREFETCH"] and precache_model_thread and not MODEL_PRECACHE_EARLY:
    precache_model_thread.start()
tuned_env = dict(env, **TUNEUP_ENV)
if int(CFG["LOWM_MAX_M"]) != 32:
    tuned_env["SGLANG_SM120_LOWM_MAX_M"] = str(int(CFG["LOWM_MAX_M"]))
print("Tuning env:", {k: v for k, v in tuned_env.items() if k not in env} or "none")
proc, log_offset, window_prefetch = launch(tuned_env, "pennyroyal-shard-prefetch.json")
mark_startup("server_launched")
status = wait_ready(proc, log_offset)
stop_prefetch(window_prefetch)
if status == "exited" and tuned_env != env and CFG["TUNEUP_FALLBACK"]:
    TUNEUP["fallback"] = "tuned server exited before ready; relaunched untuned"
    print(f"\n{TUNEUP['fallback']}\n", flush=True)
    proc, log_offset, window_prefetch = launch(env, "pennyroyal-shard-prefetch-untuned.json")
    mark_startup("server_relaunched_untuned")
    status = wait_ready(proc, log_offset)
    stop_prefetch(window_prefetch)

Path(WORKING_DIR, "pennyroyal-startup.json").write_text(json.dumps(
    {"seconds_from_launcher_start": STARTUP_TIMES, "target": SERVE_DIR, "draft": str(DRAFT_VIEW or ""),
     "quantization": QUANT, "ple_fp8_source": PLE_FP8_DIR,
     "ple_dtype": serve_cfg.get("text_config", serve_cfg).get("ple_embedding_dtype"),
     "draft_vocab": CFG["DRAFT_VOCAB"] or ("fr-spec-64k" if CFG["FRSPEC"] else "full"),
     "mamba_cache": CFG["MAMBA_CACHE"], "mamba_radix": CFG["MAMBA_RADIX"],
     "mamba_skip_decode_lock": bool(CFG["MAMBA_SKIP_DECODE_LOCK"]),
     "chunked_prefill": CFG["CHUNK"], "window_prefetch": bool(CFG["WINDOW_PREFETCH"]),
     "prefetch_checkpoints": bool(CFG["PREFETCH_CHECKPOINTS"]), "server_status": status,
     "max_running_requests": CFG["MAXREQ"], "cuda_graph_bs": graph_bs, "mem_fraction_static": CFG["MEMFRAC"],
     "tuneup": TUNEUP}, indent=1))
if status == "exited":
    # Games against a dead endpoint only burn the session; fail the setup step.
    raise SystemExit(f"SGLang server exited before it was ready; see {LOG}")

# ---- hand the endpoint and analyzer settings to the run ----
base_url = f"http://127.0.0.1:{SERVED_MODEL_PORT}/v1"
setup_env = {
    **{str(k): str(v) for k, v in SPEC.get("env", {}).items()},
    "LOCAL_ANALYZER_BASE_URL": base_url,
    "OPENAI_BASE_URL": base_url,
    "LOCAL_ANALYZER_PROVIDER": "vllm",
    "OPENAI_PROVIDER": "vllm",
    "LOCAL_ANALYZER_MODEL_ID": SERVED_MODEL_NAME,
    "INFERENCE_ANALYZER_MODEL": SERVED_MODEL_NAME,
}
setup_env_path = Path(os.environ["TAAF_KAGGLE_SETUP_ENV"])
existing = json.loads(setup_env_path.read_text()) if setup_env_path.exists() else {}
existing.update(setup_env)
setup_env_path.write_text(json.dumps(existing, indent=2))
print("Harness env: " + json.dumps(setup_env, sort_keys=True), flush=True)
