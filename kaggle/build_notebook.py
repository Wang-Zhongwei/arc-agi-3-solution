"""Build a Kaggle notebook from this branch, on Franzen's Milestone 2 notebook.

The notebook copies Tufa's source bundle (dfranzen/taaf-kaggle-source-bundle-copy)
and applies one patch cell at run time. That patch is regenerated here as
``git diff bundle-base..HEAD`` over the harness's runtime paths (RUNTIME_PATHS), so
every change is an ordinary commit and the notebook always runs exactly the
committed code. Docs, tests, the viewer, scripts and the local analysis tools
(inference/tools) never run on Kaggle, so they are left out of the patch, the code
hash and the dirty-tree check. At the ``franzen-m2`` tag the output reproduces
Franzen's submission.

Arms are defined in kaggle/arms.json. Each experiment arm runs in its own
kernel, mellamoa/v4-<arm>; only submissions reuse one kernel, mellamoa/v4-submission,
as successive versions.

``bundle-base`` is a tag on the commit holding Tufa's Kaggle source bundle exactly as
shipped in dfranzen/taaf-kaggle-source-bundle-copy. Only ARC3-Inference is patched:
Franzen's published repo also edits tufa-arc-agi-framework, but his notebook applied
only ARC3-Inference hunks, so the framework stays as bundled to match what he ran. Usage (from the repo root):

    python kaggle/build_notebook.py --arm control --push --new-batch
    python kaggle/build_notebook.py --arm temp06 --push
    python kaggle/build_notebook.py --arm control --push --submission
    python kaggle/build_notebook.py --bench levers --push      # serving benchmark, no games
    python kaggle/build_notebook.py --arm stack-v2 --push --replica 2   # repeat a run

Each push is recorded in kaggle/runs.json (version, commit, fingerprint). The
builder refuses an arm whose notebook would be identical to a recorded run, and
refuses to build at different runtime code from the open batch, so arms always run
the same code as their control. runs.json is committed per branch, so the builder
also reads it from every local and remote-tracking branch: a duplicate recorded on
any branch is refused, and a branch with no open batch of its own may not build
(without --new-batch) while another branch's open batch ran different code. A --submission build skips both checks (submitting
the notebook an arm already ran is the point) and never opens or closes a batch.
A deliberate repeat passes ``--replica N``.

Every edit to the template is an exact string replacement that must match once,
so a template change fails the build instead of silently producing a different run.
"""
from __future__ import annotations

import argparse
import base64
import copy
import gzip
import hashlib
import importlib.util
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BASE_REF = "bundle-base"
# The harness code a Kaggle run executes: what the notebook's patch carries, what the
# code hash covers and what must be committed before a build. Everything else under
# ARC3-Inference (README.md, CONFIGURATION.md, Makefile, tests/, viewer/, scripts/,
# bench/ and the local analysis tools in inference/tools) is never imported on Kaggle.
RUNTIME_PATHS = (
    "ARC3-Inference/inference",
    ":(exclude)ARC3-Inference/inference/tools",
    "ARC3-Inference/configs",
    "ARC3-Inference/pyproject.toml",
    "ARC3-Inference/uv.lock",
)
PATCH_PATHS = RUNTIME_PATHS
# Until 2026-10-08 the patch covered all of ARC3-Inference, so runs.json records from
# then hold a code hash and fingerprint of that wider patch. The builder still
# computes them, only to compare with those records.
LEGACY_PATCH_PATHS = ("ARC3-Inference",)
PATCH_HEADER = "%%writefile /kaggle/harness-changes.patch\n"
PATCH_PATH = "/kaggle/harness-changes.patch"
# Kaggle rejects kernel sources of 1 MB or more ("The kernel source must be
# less than 1 megabytes in size"); keep a margin for the JSON wrapper.
KERNEL_SOURCE_LIMIT = 950_000
OWNER = "mellamoa"
# Experiments each get their own kernel, v4-<arm>; submissions are versions of one kernel.
SUBMISSION_SLUG = "v4-submission"
# Harness env var that only makes a repeat run's notebook differ (--replica); nothing reads it.
REPLICA_ENV = "TAAF_REPLICA"

# The template applies only ARC3-Inference hunks, which is all PATCH_PATHS produces.
# The call still anchors the env cell's other edits, so it is kept unchanged.
APPLY_OLD = '["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/harness-changes.patch"]'
APPLY_NEW = APPLY_OLD
ENV_UPDATE = "os.environ.update({k: str(v) for k,v in setup_env.items()})"
DEMO_EXCLUDED = (
    "demo_excluded_games = [] if TRUE_SUBMISSION else ['bp35', 'cd82', 'cn04', 'dc22', 'g50t', "
    "'ka59', 'lf52', 'ls20', 'm0r0', 's5i5', 'sk48', 'sp80', 'su15', 'tn36', 'wa30']"
)
ALL_GAMES = (
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85", "ls20", "m0r0", "r11l",
    "re86", "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
)
# Named game panels for short runs (arms.json "demo_games"). Franzen's own 10-game set
# ("franzen") has no hard games, so it cannot show an effect on games the model is stuck on.
# "balanced": the 5 hardest games by fraction of levels cleared over 7 full runs
# (analysis/levels.csv) plus 5 easy ones that still vary between runs (never-missed
# sb26/lp85/ar25 are left out: they cannot show a difference).
GAME_PANELS = {
    "balanced": ("sk48", "bp35", "tn36", "sp80", "lf52", "ft09", "vc33", "tu93", "cd82", "r11l"),
}
# Franzen hard-codes one Kaggle mount layout; resolve each input under both.
INPUT_PATHS_ANCHOR = "SERVED_MODEL_NAME = 'flashnext'\n"
INPUT_RESOLVER = """
# Inputs mount as /kaggle/input/datasets/<owner>/<slug> (models: models/<owner>/<slug>/...)
# or, on some kernels, /kaggle/input/<slug>/...; use whichever exists, else say what is mounted.
INPUT_ROOT = Path('/kaggle/input')


def _resolve_input(path):
    p = Path(path)
    if p.exists():
        return str(p)
    rel = p.relative_to(INPUT_ROOT).parts  # (kind, [owner,] slug, *rest)
    # competitions/<slug>/... has no owner; datasets/ and models/ have one
    skip = 1 if rel and rel[0] == 'competitions' else 2
    candidates = [INPUT_ROOT.joinpath(*rel[skip:])] if len(rel) > skip else []
    for candidate in candidates:
        if candidate.exists():
            print(f'input {path} found at {candidate}')
            return str(candidate)
    mounted = sorted(str(q) for q in INPUT_ROOT.glob('*/*/*'))[:60]
    raise FileNotFoundError(f'input not mounted: {path} (also tried {candidates}); mounted: {mounted}')

WHEELHOUSE_DIR  = _resolve_input(WHEELHOUSE_DIR)
MODEL_DIR       = _resolve_input(MODEL_DIR)
DRAFT_MODEL_DIR = _resolve_input(DRAFT_MODEL_DIR)
ORIG_BUNDLE_DIR = _resolve_input(ORIG_BUNDLE_DIR)
"""
# The draft model's KV dtype accepts only fp8/bf16 (nvfp4 kills the server at
# argument parsing), so it gets its own CFG key, defaulting to the template's.
DRAFT_KV_OLD = '"--speculative-draft-kv-cache-dtype", CFG["KVDTYPE"],'
DRAFT_KV_NEW = '"--speculative-draft-kv-cache-dtype", CFG["DRAFT_KVDTYPE"],'
DRAFT_KV_DEFAULT = (
    'CFG.setdefault("DRAFT_KVDTYPE", CFG["KVDTYPE"])  # before arm overrides\n'
    'CFG.setdefault("ATTN_BACKEND", "")  # "" keeps SGLang\'s default (flashinfer)\n'
    'CFG.setdefault("TEXT_CONFIG_OVERRIDE", {})  # {} serves MODEL_DIR unchanged\n'
    'CFG.setdefault("SPEC_ADAPTIVE", {})  # {} keeps a fixed SPEC_STEPS\n'
)
# FP4 KV ("KV4 MHA") needs --attention-backend triton/torch_native/flex/trtllm_mha;
# Franzen passes none, so the arm sets CFG["ATTN_BACKEND"] and this appends it.
ATTN_ARGS_ANCHOR = 'if CFG["PREFETCH_CHECKPOINTS"]: args += ["--weight-loader-prefetch-checkpoints"]'
# A text_config override (e.g. num_experts_per_tok) is nested, which a flat
# --json-model-override-args may not reach, so the target is served from a view
# of MODEL_DIR: every file symlinked except a patched config.json. The draft
# view is built from MODEL_DIR and keeps the checkpoint's config.
ATTN_ARGS = (
    'if CFG["ATTN_BACKEND"]: args += ["--attention-backend", CFG["ATTN_BACKEND"]]\n'
    'if CFG["TEXT_CONFIG_OVERRIDE"]:\n'
    '    _view = Path(PREFIX) / "target-view"\n'
    '    _view.mkdir(parents=True, exist_ok=True)\n'
    '    for _src in Path(MODEL_DIR).iterdir():\n'
    '        _dst = _view / _src.name\n'
    '        if _src.name != "config.json" and not _dst.is_symlink():\n'
    '            _dst.symlink_to(_src.resolve())\n'
    '    _cfg = json.loads(Path(MODEL_DIR, "config.json").read_text())\n'
    '    _cfg["text_config"].update(CFG["TEXT_CONFIG_OVERRIDE"])\n'
    '    (_view / "config.json").write_text(json.dumps(_cfg, indent=2))\n'
    '    args[args.index("--model-path") + 1] = str(_view)\n'
    '    print("Target view", _view, "text_config override:", CFG["TEXT_CONFIG_OVERRIDE"])\n'
    # SGLang's adaptive speculation picks num_steps per batch size from the
    # accepted-draft EMA. Its default config probes 7 steps at batch 1, past the
    # QSA cap of 4 draft tokens (SPEC_STEPS 3), so the arm supplies candidates.
    'if CFG["SPEC_ADAPTIVE"]:\n'
    '    _acfg = Path(PREFIX) / "adaptive_spec.json"\n'
    '    _acfg.write_text(json.dumps(CFG["SPEC_ADAPTIVE"]))\n'
    '    args += ["--speculative-adaptive", "--speculative-adaptive-config", str(_acfg)]\n'
    '    print("Adaptive speculation:", CFG["SPEC_ADAPTIVE"])\n'
)
# Franzen's wait loop breaks out when the server exits during startup, and the
# benchmark then runs against a dead server (fz-tput-all v3: 0 tokens). Fail fast.
SERVER_DIED_OLD = "    if proc.poll() is not None:\n        show_log_tail()\n        break\n"
SERVER_DIED_NEW = (
    "    if proc.poll() is not None:\n        show_log_tail()\n"
    "        raise RuntimeError(f'SGLang exited with code {proc.returncode} during startup; see serve.log')\n"
)
# Competition inputs Franzen also hard-codes; wrapped in the same resolver.
COMPETITION_WHEELS = '"/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"'
# The serving cell builds the SGLang process env, then prints nvcc's version.
SERVE_ENV_ANCHOR = 'run([str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], env=env)'
# The serving cell defines CFG (context, streams, Mamba slots, ...) and then this line.
SERVE_ANCHOR = 'VENV = f"{PREFIX}/venv"'
# The serving cell is built from this file, so SGLang serving is edited in git
# like the harness. It started as a verbatim copy of the template's cell.
SERVE_LAUNCHER = REPO / "serving" / "kaggle" / "launch_sglang.py"
# Patches to the installed SGLang, embedded into the launcher (see its
# "runtime patches" block). An empty folder leaves the launcher unchanged.
RUNTIME_PATCHES = REPO / "serving" / "runtime_patches"
RUNTIME_PATCHES_ANCHOR = "SGLANG_RUNTIME_PATCHES = {}\n"
# The JIT cache helpers (issue #6), embedded into the launcher as the module `jit_cache`.
JIT_CACHE_MODULE = REPO / "serving" / "kaggle" / "jit_cache.py"
JIT_CACHE_ANCHOR = "jit_cache = None\n"


def _jit_cache_block() -> str:
    """Code that defines the module `jit_cache` from serving/kaggle/jit_cache.py."""
    source = JIT_CACHE_MODULE.read_text(encoding="utf-8")
    if "'''" in source or source.rstrip("\n").endswith("\\"):
        raise SystemExit(f"{JIT_CACHE_MODULE} cannot be embedded in a r'''...''' string")
    return (
        "import types\n"
        "jit_cache = types.ModuleType(\"jit_cache\")\n"
        f"exec(compile(r'''{source}''', \"serving/kaggle/jit_cache.py\", \"exec\"), jit_cache.__dict__)\n"
    )


def _serving_launcher(arm_patches: dict[str, str] | None = None) -> str:
    """The launcher text with serving/kaggle/jit_cache.py and serving/runtime_patches/ embedded.

    ``arm_patches``: {name: text} of serving/bench_patches/ files an arm picked
    (arms.json "runtime_patches"). They join the same table, and the launcher
    applies it in name order, so they must sort after every shared patch, as
    a bench config's patches run after the shared ones.
    """
    text = SERVE_LAUNCHER.read_text(encoding="utf-8")
    text = _replace_once(text, JIT_CACHE_ANCHOR, _jit_cache_block(), "the jit_cache placeholder")
    files = sorted(
        p for p in RUNTIME_PATCHES.glob("*") if p.suffix in (".patch", ".py") and p.is_file()
    ) if RUNTIME_PATCHES.is_dir() else []
    payload = {p.name: p.read_text(encoding="utf-8") for p in files}
    source = "serving/runtime_patches/"
    if arm_patches:
        if set(payload) & set(arm_patches):
            raise SystemExit(f"arm runtime patches reuse shared patch names: {sorted(set(payload) & set(arm_patches))}")
        if payload and max(payload) > min(arm_patches):
            raise SystemExit(f"arm runtime patch {min(arm_patches)!r} would apply before shared {max(payload)!r}; "
                             "rename it so it sorts after serving/runtime_patches/")
        payload.update(arm_patches)
        source += " and the arm's serving/bench_patches/"
    if not payload:
        return text
    encoded = base64.b64encode(gzip.compress(json.dumps(payload).encode("utf-8"), 9, mtime=0)).decode("ascii")
    lines = "\n".join(encoded[i:i + 120] for i in range(0, len(encoded), 120))
    return _replace_once(
        text, RUNTIME_PATCHES_ANCHOR,
        "SGLANG_RUNTIME_PATCHES = json.loads(gzip.decompress(base64.b64decode(\n\"\"\"\n"
        f"{lines}\n\"\"\")))  # embedded from {source}: {', '.join(payload)}\n",
        "the runtime patches placeholder",
    )
# Franzen's wait loop prints serve.log only if the server process dies, so a
# server that hangs while loading leaves nothing in the visible log. These edits
# print its newest line on every progress tick and keep streaming it after the
# startup deadline until the server is healthy or exits.
WAIT_TICK_OLD = 'print(f"  {elapsed}s ...", flush=True); last_report=elapsed'
WAIT_TICK_NEW = (
    'print(f"  {elapsed}s ... serve.log: {_serve_log_last_line()}", flush=True); last_report=elapsed'
)
WAIT_DEADLINE_OLD = (
    'f"releasing the benchmark with the server still loading")'
)
WAIT_DEADLINE_NEW = (
    'f"releasing the benchmark with the server still loading")\n'
    "    show_log_tail()\n"
    "    threading.Thread(target=_stream_serve_log_until_ready, daemon=True).start()"
)
WAIT_HELPERS_ANCHOR = 'url, t0 = f"http://127.0.0.1:{SERVED_MODEL_PORT}/health", time.time()'
WAIT_HELPERS = """import threading


def _serve_log_last_line():
    try:
        with open(LOG, "rb") as f:
            f.seek(0, 2)
            f.seek(max(log_offset, f.tell() - 4000))
            lines = [l for l in f.read().decode(errors="replace").splitlines() if l.strip()]
        return lines[-1][-300:] if lines else "(empty)"
    except OSError as exc:
        return f"(unreadable: {exc})"


def _stream_serve_log_until_ready():
    # Runs after the deadline released the benchmark: report the server's own
    # progress every 30 s until /health answers or the process exits.
    global SERVER_READY_TIME
    seen = Path(LOG).stat().st_size if Path(LOG).exists() else 0
    while proc.poll() is None:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    SERVER_READY_TIME = time.time()
                    print(f"[serve] READY after {int(time.time()-t0)}s", flush=True)
                    return
        except Exception:
            pass
        with open(LOG, "rb") as f:
            f.seek(seen)
            fresh = f.read()
            seen = f.tell()
        for line in fresh.decode(errors="replace").splitlines()[-15:]:
            if line.strip():
                print(f"[serve] {line[-300:]}", flush=True)
        time.sleep(30)
    print(f"[serve] server process exited with {proc.returncode}", flush=True)
    show_log_tail()


"""
# The score-timeline logger reports time since the server answered /health, so the
# wait loop stamps that moment (the streamer above does the same after a deadline release).
READY_OLD = 'print(f"\\nREADY after {int(time.time()-t0)}s -> {url.replace(\'/health\',\'/v1\')}")'
READY_NEW = "SERVER_READY_TIME = time.time(); " + READY_OLD
# A read-only score-timeline logger (kaggle/score_logger.py) runs in its own cell right
# after the system monitor: every 60 s one JSON line per game to score_timeline.jsonl,
# every 5 min a "[score] ..." line in the log. Off in a competition rerun, which hides
# the baselines the score needs and whose output nobody sees.
MONITOR_ANCHOR = "def start_monitor("
SCORE_LOGGER_SOURCE = HERE / "score_logger.py"
SCORE_LOGGER_PATH = "/kaggle/working/score_timeline.jsonl"
SCORE_LOGGER_MARKDOWN = (
    "## 8b. Log the score timeline\n\n"
    "Every minute, append each game's levels, actions, tokens and official score to "
    f"`{SCORE_LOGGER_PATH}`, with wall-clock time since the notebook started, since the model "
    "server was ready and since the benchmark started; every 5 minutes print a `[score]` line. "
    "Read-only: it never touches the solver, the games or the server."
)
SCORE_LOGGER_START = """

if not TRUE_SUBMISSION:
    _stop_score_logger = start_score_logger(
        lambda: globals().get("bm"),
        WORKING_DIR / "score_timeline.jsonl",
        notebook_start=NOTEBOOK_START_TIME,
        get_server_ready=lambda: globals().get("SERVER_READY_TIME"),
    )
    print("score timeline -> /kaggle/working/score_timeline.jsonl (every 60 s, [score] line every 5 min)")
"""


def score_logger_cells() -> list[dict]:
    """The markdown and code cells that start the score-timeline logger."""
    code = SCORE_LOGGER_SOURCE.read_text(encoding="utf-8").rstrip("\n") + SCORE_LOGGER_START
    return [
        {"cell_type": "markdown", "metadata": {"tags": []}, "source": SCORE_LOGGER_MARKDOWN},
        {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": code},
    ]


DEMO_MINUTES = "bm.solver.max_runtime_s_per_game = 25*60 #532*60 * bm.solver.concurrency // 110"

# ---- serving benchmark (--bench): kaggle/bench.json, run by serving/kaggle/bench.py ----
# A bench notebook keeps every setup cell, but its serving cell only loads the
# benchmark and its game cell runs it: SGLang is launched once per config
# through the same launcher text, with BENCH_* hooks for that config.
BENCH_SPEC = HERE / "bench.json"
BENCH_MODULE = REPO / "serving" / "kaggle" / "bench.py"
BENCH_PATCHES = REPO / "serving" / "bench_patches"
BENCH_CORPUS = HERE / "bench_corpus.json.gz"
GAME_RUN = "await bm.run("
RUNTIME_PATCH_LOOP = "if SGLANG_RUNTIME_PATCHES:\n"
PRECACHE_START = "precache_model_thread.start()\n"
BENCH_HOOKS = (
    (SERVE_ANCHOR,
     "# Bench: this config's CFG overrides (kaggle/bench.json), set by the bench driver.\n"
     "assert set(BENCH_CFG) <= set(CFG), f\"unknown CFG keys: {set(BENCH_CFG) - set(CFG)}\"\n"
     "CFG.update(BENCH_CFG)\n" + SERVE_ANCHOR,
     "the serving CFG anchor"),
    (RUNTIME_PATCH_LOOP,
     "# Bench: start from the installed SGLang as shipped, then add this config's patches.\n"
     "BENCH_PREPARE_SITE_PACKAGES(VENV, PREFIX)\n"
     "SGLANG_RUNTIME_PATCHES = {**SGLANG_RUNTIME_PATCHES, **BENCH_RUNTIME_PATCHES}\n" + RUNTIME_PATCH_LOOP,
     "the runtime patch loop"),
    (SERVE_ENV_ANCHOR,
     "env.update(BENCH_ENV)  # bench: this config's server env\n" + SERVE_ENV_ANCHOR,
     "the serving env anchor"),
    (PRECACHE_START,
     "args = BENCH_ARGS(args)  # bench: this config's launch argument edits\n" + PRECACHE_START,
     "the server launch"),
)
BENCH_RUN_CELL = (
    "# Serving benchmark: launch SGLang once per config, drive the load, and write\n"
    "# bench_results.json / bench_results.md to WORKING_DIR. No games are played.\n"
    "print('Starting serving benchmark (no games)...')\n"
    "bench_results = bench.run_bench(BENCH_PLAN, launcher=BENCH_LAUNCHER, corpus=BENCH_CORPUS,\n"
    "                                nb=globals(), working_dir=WORKING_DIR)\n"
)


def _bench_module():
    """serving/kaggle/bench.py, imported by path (its folder also holds the launcher)."""
    spec = importlib.util.spec_from_file_location("serving_bench", BENCH_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _launcher_cfg_keys(launcher: str) -> set[str]:
    """CFG keys the launcher defines: its CFG = dict(...) plus the builder's setdefaults."""
    block = re.search(r"^CFG = dict\((.*?)^\)", launcher, re.S | re.M)
    if block is None:
        raise SystemExit("launcher: CFG = dict(...) not found")
    keys = set(re.findall(r"^\s+([A-Z][A-Z0-9_]*)=", block[1], re.M))
    return keys | set(re.findall(r'CFG\.setdefault\("([A-Z][A-Z0-9_]*)"', launcher))


def _bench_patches(names, what: str) -> dict[str, str]:
    """{name: text} for files of serving/bench_patches/; ``what`` names the user in errors."""
    patches = {}
    for patch in names:
        path = BENCH_PATCHES / patch
        if path.suffix not in (".py", ".patch") or not path.is_file():
            raise SystemExit(f"{what}: runtime patch {path} is missing (or not .py/.patch)")
        patches[patch] = path.read_text(encoding="utf-8")
    return patches


def bench_plan(name: str) -> dict:
    """The plan for one bench of kaggle/bench.json, with its runtime patches embedded."""
    try:
        plan = _bench_module().resolve_plan(json.loads(BENCH_SPEC.read_text(encoding="utf-8")), name)
    except ValueError as exc:
        raise SystemExit(f"kaggle/bench.json: {exc}")
    plan["patches"] = _bench_patches(plan.pop("patches_needed"), f"bench {name!r}")
    return plan


def apply_bench(nb: dict, plan: dict, corpus: bytes) -> dict:
    """Turn a built notebook into a serving benchmark that plays no games."""
    nb = copy.deepcopy(nb)
    code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    serving = [cell for cell in code_cells if SERVE_ANCHOR in _cell_text(cell)]
    games = [cell for cell in code_cells if GAME_RUN in _cell_text(cell)]
    if len(serving) != 1 or len(games) != 1:
        raise SystemExit(f"template: expected 1 serving and 1 game cell, found {len(serving)} and {len(games)}")
    launcher = _cell_text(serving[0])
    unknown = {
        key for config in plan["configs"] for key in config["overrides"]["cfg"]
    } - _launcher_cfg_keys(launcher)
    if unknown:
        raise SystemExit(f"kaggle/bench.json: CFG keys the launcher does not define: {sorted(unknown)}")
    for old, new, what in BENCH_HOOKS:
        launcher = _replace_once(launcher, old, new, what)
    payload = {
        "module": BENCH_MODULE.read_text(encoding="utf-8"),
        "plan": plan,
        "launcher": launcher,
        "corpus": base64.b64encode(corpus).decode("ascii"),
    }
    module = _bench_module()
    _set_cell_text(
        serving[0],
        f"# Serving benchmark {plan['bench']!r} (kaggle/build_notebook.py --bench): this notebook plays no games.\n"
        "# This cell only loads it; the run cell below launches SGLang once per config. Embedded:\n"
        "# serving/kaggle/bench.py, the plan from kaggle/bench.json with serving/bench_patches/,\n"
        "# the launcher (serving/kaggle/launch_sglang.py with the builder's edits and BENCH_* hooks)\n"
        "# and the prompt corpus kaggle/bench_corpus.json.gz.\n"
        "import base64, gzip, json, sys, types\n"
        "_bench_blob = json.loads(gzip.decompress(base64.b64decode(\n\"\"\"\n"
        f"{module.encode_blob(payload)}\n\"\"\")))\n"
        "bench = types.ModuleType('serving_bench')\n"
        "sys.modules['serving_bench'] = bench\n"
        "exec(compile(_bench_blob['module'], 'serving/kaggle/bench.py', 'exec'), bench.__dict__)\n"
        "BENCH_PLAN, BENCH_LAUNCHER = _bench_blob['plan'], _bench_blob['launcher']\n"
        "BENCH_CORPUS = base64.b64decode(_bench_blob['corpus'])\n"
        "bench.describe(BENCH_PLAN)\n",
    )
    _set_cell_text(games[0], BENCH_RUN_CELL)
    return nb


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout


def _diff(paths: tuple[str, ...], commit: str = "HEAD") -> str:
    patch = _git("diff", "--binary", "--no-color", f"{BASE_REF}..{commit}", "--", *paths)
    if not patch.endswith("\n"):
        patch += "\n"
    return patch


def build_patch() -> str:
    """Return the committed runtime diff from the bundle to HEAD; refuse uncommitted runtime edits.

    The check covers exactly the paths the patch carries: an uncommitted doc or test
    edit cannot reach the notebook either way, so it does not block a build.
    """
    dirty = _git("status", "--porcelain", "--", *PATCH_PATHS).strip()
    if dirty:
        raise SystemExit(f"Uncommitted changes would not reach the notebook:\n{dirty}")
    return _diff(PATCH_PATHS)


def build_legacy_patch() -> str:
    """The pre-2026-10-08 patch (all of ARC3-Inference), for comparing with old records only."""
    return _diff(LEGACY_PATCH_PATHS)


def _with_runtime_code(runs: list[dict]) -> list[dict]:
    """Copies of ``runs`` where a record made before ``runtime_code`` existed gets it from its
    commit, when that commit is in this clone. A doc-only commit after an old batch's control
    then still counts as the batch's code. Only open-batch records need it (the duplicate
    check compares fingerprints). Nothing is written back to runs.json."""
    cache: dict[str, str | None] = {}
    out = []
    for run in runs:
        commit = run.get("commit")
        if "runtime_code" not in run and commit and run.get("batch_open", True):
            if commit not in cache:
                known = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}"],
                                       cwd=REPO, capture_output=True).returncode == 0
                cache[commit] = _code_hash(_diff(PATCH_PATHS, commit)) if known else None
            if cache[commit] is not None:
                run = {**run, "runtime_code": cache[commit]}
        out.append(run)
    return out


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"template: expected 1 occurrence of {what}, found {count}")
    return text.replace(old, new)


def _cell_text(cell: dict) -> str:
    source = cell["source"]
    return source if isinstance(source, str) else "".join(source)


def _set_cell_text(cell: dict, text: str) -> None:
    cell["source"] = text  # the template stores each cell as one string


def build_notebook(
    template: dict,
    patch: str,
    *,
    env: dict[str, str],
    serve: dict[str, object] | None = None,
    serve_env: dict[str, str] | None = None,
    runtime_patches: dict[str, str] | None = None,
    demo_games: str,
    demo_minutes: float | None,
) -> dict:
    nb = copy.deepcopy(template)
    code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    for cell in code_cells:
        cell["outputs"] = []
        cell["execution_count"] = None

    patch_cells = [cell for cell in code_cells if _cell_text(cell).startswith(PATCH_HEADER)]
    if len(patch_cells) != 1:
        raise SystemExit(f"template: expected 1 patch cell, found {len(patch_cells)}")
    # The documented patch is ~1 MB as text, over Kaggle's source limit, so the
    # cell carries it gzipped and base64-encoded and writes the same file.
    encoded = base64.b64encode(gzip.compress(patch.encode("utf-8"), 9, mtime=0)).decode("ascii")
    lines = "\n".join(encoded[i:i + 120] for i in range(0, len(encoded), 120))
    _set_cell_text(
        patch_cells[0],
        "# Harness patch (git diff bundle-base..HEAD), gzipped + base64 to fit Kaggle's 1 MB limit.\n"
        "import base64, gzip, pathlib\n"
        f"pathlib.Path({PATCH_PATH!r}).write_bytes(gzip.decompress(base64.b64decode(\n\"\"\"\n{lines}\n\"\"\")))\n"
        f"print('Writing {PATCH_PATH}')\n",
    )

    serving_cells = [cell for cell in code_cells if SERVE_ANCHOR in _cell_text(cell)]
    if len(serving_cells) != 1:
        raise SystemExit(f"template: expected 1 serving cell, found {len(serving_cells)}")
    _set_cell_text(serving_cells[0], _serving_launcher(runtime_patches))

    edits = 0
    edits_competition: list[int] = []
    for cell in code_cells:
        text = _cell_text(cell)
        if APPLY_OLD in text:
            text = _replace_once(text, APPLY_OLD, APPLY_NEW, "the git apply call")
            text = _replace_once(text, INPUT_PATHS_ANCHOR, INPUT_PATHS_ANCHOR + INPUT_RESOLVER,
                                 "the input path block")
            if env:
                overrides = "".join(f"setup_env[{key!r}] = {value!r}\n" for key, value in sorted(env.items()))
                text = _replace_once(
                    text, ENV_UPDATE, f"# Arm overrides (build_notebook.py --env)\n{overrides}{ENV_UPDATE}",
                    "the setup_env update",
                )
            edits += 1
        if WAIT_TICK_OLD in text:
            text = _replace_once(text, WAIT_TICK_OLD, WAIT_TICK_NEW, "the server wait tick")
            text = _replace_once(text, WAIT_DEADLINE_OLD, WAIT_DEADLINE_NEW, "the startup deadline")
            text = _replace_once(text, WAIT_HELPERS_ANCHOR, WAIT_HELPERS + WAIT_HELPERS_ANCHOR,
                                 "the health-check loop")
        if COMPETITION_WHEELS in text and "_resolve_input" not in text:
            # Resolve the competition path the same way; two cells use it.
            text = text.replace(COMPETITION_WHEELS, f"_resolve_input({COMPETITION_WHEELS})")
            edits_competition.append(1)
        if serve_env and SERVE_ENV_ANCHOR in text:
            overrides = "".join(f"    {k!r}: {v!r},\n" for k, v in sorted(serve_env.items()))
            text = _replace_once(
                text, SERVE_ENV_ANCHOR,
                "# Arm overrides of the SGLang process env (build_notebook.py serve_env).\n"
                f"env.update({{\n{overrides}}})\n{SERVE_ENV_ANCHOR}",
                "the serving env anchor",
            )
        if SERVE_ANCHOR in text:
            text = _replace_once(text, READY_OLD, READY_NEW, "the server READY print")
            text = _replace_once(text, DRAFT_KV_OLD, DRAFT_KV_NEW, "the draft KV dtype argument")
            text = _replace_once(text, ATTN_ARGS_ANCHOR, ATTN_ARGS + ATTN_ARGS_ANCHOR, "the server args")
            text = _replace_once(text, SERVER_DIED_OLD, SERVER_DIED_NEW, "the server-exit check")
            text = _replace_once(text, SERVE_ANCHOR, DRAFT_KV_DEFAULT + SERVE_ANCHOR, "the serving CFG anchor")
            if serve:
                overrides = "".join(f"    {key!r}: {value!r},\n" for key, value in sorted(serve.items()))
                text = _replace_once(
                    text, SERVE_ANCHOR,
                    "# Arm overrides (build_notebook.py --serve); only existing keys.\n"
                    f"_SERVE_OVERRIDES = {{\n{overrides}}}\n"
                    "assert set(_SERVE_OVERRIDES) <= set(CFG), set(_SERVE_OVERRIDES) - set(CFG)\n"
                    f"CFG.update(_SERVE_OVERRIDES)\n{SERVE_ANCHOR}",
                    "the serving CFG anchor",
                )
            edits += 1
        if DEMO_EXCLUDED in text:
            if demo_games != "franzen":
                panel = ALL_GAMES if demo_games == "all" else GAME_PANELS[demo_games]
                excluded = [g for g in ALL_GAMES if g not in panel]
                text = _replace_once(text, DEMO_EXCLUDED, f"demo_excluded_games = {excluded!r}",
                                     "the demo game filter")
            if demo_minutes is not None:
                text = _replace_once(
                    text, DEMO_MINUTES,
                    f"bm.solver.max_runtime_s_per_game = int({demo_minutes} * 60)",
                    "the demo time limit",
                )
            edits += 1
        _set_cell_text(cell, text)
    if edits != 3:
        raise SystemExit(f"template: expected the env, serving and solver cells, matched {edits}")
    if len(edits_competition) != 2:
        raise SystemExit(f"template: expected 2 cells using the competition path, got {len(edits_competition)}")
    monitors = [i for i, cell in enumerate(nb["cells"])
                if cell["cell_type"] == "code" and MONITOR_ANCHOR in _cell_text(cell)]
    if len(monitors) != 1:
        raise SystemExit(f"template: expected 1 system monitor cell, found {len(monitors)}")
    nb["cells"][monitors[0] + 1:monitors[0] + 1] = score_logger_cells()
    return nb


def build_metadata(
    template: dict, slug: str, title: str, notebook_name: str, datasets: list[str] | tuple = (),
) -> dict:
    """Kernel metadata from the template's, plus an arm's extra dataset sources ("owner/slug")."""
    meta = {
        key: template[key]
        for key in (
            "language", "kernel_type", "enable_gpu", "enable_tpu", "enable_internet",
            "dataset_sources", "kernel_sources", "competition_sources", "model_sources",
            "docker_image", "machine_shape",
        )
        if key in template
    }
    meta.update({"id": f"{OWNER}/{slug}", "title": title, "code_file": notebook_name, "is_private": True})
    if datasets:
        bad = [d for d in datasets if not re.fullmatch(r"[\w.-]+/[\w.-]+", d)]
        if bad:
            raise SystemExit(f"arm datasets must be 'owner/slug': {bad}")
        sources = list(meta.get("dataset_sources", []))
        meta["dataset_sources"] = sources + [d for d in datasets if d not in sources]
    return meta


def _arm_config(name: str) -> dict:
    arms = json.loads((HERE / "arms.json").read_text(encoding="utf-8"))
    if name not in arms["arms"]:
        raise SystemExit(f"unknown arm {name!r}; defined: {', '.join(arms['arms'])}")
    arm = {**arms["defaults"], **arms["arms"][name]}
    return {
        "env": {str(k): str(v) for k, v in arm.get("env", {}).items()},
        "serve": dict(arm.get("serve", {})),
        "serve_env": {str(k): str(v) for k, v in arm.get("serve_env", {}).items()},
        # Files of serving/bench_patches/ applied to the installed SGLang, after
        # serving/runtime_patches/, e.g. a bench config's patches that proved faster.
        "runtime_patches": [str(p) for p in arm.get("runtime_patches", [])],
        # Extra Kaggle datasets to mount ("owner/slug"), e.g. a JIT cache (serving/CUSTOM.md).
        "datasets": [str(d) for d in arm.get("datasets", [])],
        # What must hold (e.g. a bench result) before the arm is pushed; --gate-passed confirms it.
        "gate": str(arm["gate"]) if arm.get("gate") else None,
        "demo_games": arm.get("demo_games", "franzen"),  # "franzen", "all" or a GAME_PANELS key
        "demo_minutes": arm.get("demo_minutes"),
    }


def _fingerprint(notebook: dict, datasets: list[str] | tuple = ()) -> str:
    """The notebook as built without its arm label, plus any extra mounted datasets:
    equal fingerprints are the same run. Without extra datasets it is the notebook's hash alone."""
    run = {"notebook": notebook, "datasets": list(datasets)} if datasets else notebook
    return hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()[:16]


def _load_runs() -> list[dict]:
    path = HERE / "runs.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


RUNS_FILE = "kaggle/runs.json"


def _branch_runs() -> dict[str, list[dict]]:
    """kaggle/runs.json as committed on every local and remote-tracking branch (fetch first
    to see other clones' records). Branches without the file or with an empty one are left out."""
    refs = [ref for ref in _git("for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes").split()
            if not ref.endswith("/HEAD")]
    if not refs:
        return {}
    query = "".join(f"{ref}:{RUNS_FILE}\n" for ref in refs).encode()
    out = subprocess.run(["git", "cat-file", "--batch"], cwd=REPO, input=query, check=True,
                         capture_output=True).stdout
    found, pos = {}, 0
    for ref in refs:
        end = out.index(b"\n", pos)
        header = out[pos:end].decode().split()
        pos = end + 1
        if len(header) != 3 or header[1] != "blob":  # "<ref>:<path> missing"
            continue
        size = int(header[2])
        blob, pos = out[pos:pos + size], pos + size + 1
        try:
            runs = json.loads(blob)
        except ValueError:
            continue
        if isinstance(runs, list) and runs:
            found[ref.removeprefix("refs/heads/").removeprefix("refs/remotes/")] = runs
    return found


def _other_branch_runs(local: list[dict], branch_runs: dict[str, list[dict]]) -> list[dict]:
    """Runs recorded on other branches and not in this checkout's runs.json, one per kernel
    version, each with the ``branches`` it is recorded on. A run marked failed or closed in
    any copy counts as failed or closed. Only this builder's kernels (v4-*) count: a remote of
    an older project can carry its own runs.json."""
    local_keys = {(run.get("kernel"), run.get("version")) for run in local}
    merged: dict[tuple, dict] = {}
    for branch, runs in sorted(branch_runs.items()):
        for run in runs:
            key = (run.get("kernel"), run.get("version"))
            if key in local_keys or not str(run.get("kernel", "")).startswith(f"{OWNER}/v4-"):
                continue
            if key not in merged:
                merged[key] = {**run, "branches": [branch]}
                continue
            seen = merged[key]
            seen["branches"].append(branch)
            if run.get("status") == "failed":
                seen["status"] = "failed"
            if not run.get("batch_open", True):
                seen["batch_open"] = False
    return list(merged.values())


def _code_hash(patch: str) -> str:
    """Hash of a patch. For build_patch() that is the code a run executes: commits that
    touch only kaggle/, docs, tests, the viewer or inference/tools leave it unchanged."""
    return hashlib.sha256(patch.encode()).hexdigest()[:16]


def _check_gate(arm: str, gate: str | None, *, push: bool, gate_passed: bool) -> None:
    """Refuse to push a gated arm unless --gate-passed says its gate holds; building is free."""
    if gate_passed and not gate:
        raise SystemExit(f"--gate-passed: arm {arm!r} has no gate")
    if gate and push and not gate_passed:
        raise SystemExit(f"arm {arm!r} is gated on {gate}. Push it with --gate-passed once that holds.")


def _replica_env(replica: int) -> dict[str, str]:
    """The env that marks run ``replica`` of an arm or bench: none for the first run.

    A repeat (2, 3, ...) gets the inert TAAF_REPLICA=N, so its notebook, and with it its
    fingerprint, differs from the earlier runs' and the duplicate check lets it through.
    The first run builds exactly as it did before the option existed."""
    if replica < 1:
        raise SystemExit(f"--replica must be 1 or more, got {replica}")
    return {} if replica == 1 else {REPLICA_ENV: str(replica)}


def _is_submission(run: dict) -> bool:
    return bool(run.get("submission")) or run.get("kernel") == f"{OWNER}/{SUBMISSION_SLUG}"


def _same_code(run: dict, code: str, legacy_code: str | None) -> bool:
    """Whether a recorded run executed ``code``. Records written since the runtime-only
    patch carry ``runtime_code``; older ones only ``code``, the hash of the wider patch."""
    if "runtime_code" in run:
        return run["runtime_code"] == code
    return run["code"] == (code if legacy_code is None else legacy_code)


def _check_unique(
    arm: str, fingerprint: str, code: str, runs: list[dict], *, new_batch: bool,
    legacy_fingerprint: str | None = None, legacy_code: str | None = None,
    other_runs: list[dict] = (), bench: bool = False,
) -> None:
    """Refuse a duplicate of a recorded run, or code that differs from the open batch.

    ``runs`` is this branch's runs.json; ``other_runs`` are the runs only other branches
    record (see _other_branch_runs). ``legacy_fingerprint``/``legacy_code`` are this build's
    hashes under the old whole-ARC3-Inference patch, so records made before the runtime-only
    patch still match.
    """
    def counted(records):
        # A run that never reached the games (e.g. the server failed to start) does not count,
        # and a submission is not an experiment: it neither duplicates one nor joins a batch.
        return [run for run in records if run.get("status") != "failed" and not _is_submission(run)]

    runs, other_runs = counted(runs), counted(other_runs)
    fingerprints = {fingerprint} | ({legacy_fingerprint} if legacy_fingerprint else set())
    for run in runs + other_runs:
        if run.get("fingerprint") in fingerprints:
            where = f" on {', '.join(run['branches'])}" if run.get("branches") else ""
            raise SystemExit(
                f"{arm!r} would run exactly what {run['arm']!r} ran as version {run.get('version')} "
                f"({run['fingerprint']}{where}); not building a duplicate."
            )
    if bench:
        return  # a bench plays no games, so the harness code of a batch does not apply
    if new_batch:
        return
    batch = [run for run in runs if run.get("batch_open", True)]  # failed runs already dropped
    if batch and not any(_same_code(run, code, legacy_code) for run in batch):
        commits = ", ".join(sorted({run["commit"] for run in batch}))
        raise SystemExit(
            f"the open batch ran different code (commit {commits}): arms built now would not be "
            "comparable with its control. Rebuild the control with --new-batch, or check out "
            "the batch's code."
        )
    elsewhere = [run for run in other_runs if run.get("batch_open", True)]
    if not batch and elsewhere and not any(_same_code(run, code, legacy_code) for run in elsewhere):
        newest = sorted(elsewhere, key=lambda run: run.get("pushed_at", ""), reverse=True)
        batches = list(dict.fromkeys(f"{run['arm']} @{run['commit']} on {', '.join(run['branches'])}"
                                     for run in newest))
        batches = "; ".join(batches[:4]) + (f"; and {len(batches) - 4} more" if len(batches) > 4 else "")
        raise SystemExit(
            f"this branch has no open batch, and open batches on other branches ran different code "
            f"({batches}): "
            f"{arm!r} built here would not be comparable with that control. Start a batch here with "
            "--new-batch, running its control first, or build on the batch's branch."
        )


def main_bench(args) -> None:
    """Build (and with --push, push and record) the bench notebook v4-bench-<name>."""
    plan = bench_plan(args.bench)
    if not BENCH_CORPUS.is_file():
        raise SystemExit(f"{BENCH_CORPUS} is missing; build it with kaggle/make_bench_corpus.py")
    corpus = BENCH_CORPUS.read_bytes()
    patch = build_patch()
    legacy_patch = build_legacy_patch()
    head = _git("rev-parse", "--short", "HEAD").strip()
    template = json.loads((HERE / "template.ipynb").read_text(encoding="utf-8"))
    template_meta = json.loads((HERE / "template-metadata.json").read_text(encoding="utf-8"))
    label = f"bench-{args.bench}"

    def _build(env: dict[str, str]) -> dict:
        nb = build_notebook(template, patch, env=env, demo_games="franzen", demo_minutes=None)
        return apply_bench(nb, plan, corpus)

    replica = _replica_env(args.replica)
    fingerprint = _fingerprint(_build(replica))
    runs = _load_runs()
    other_runs = _other_branch_runs(runs, _branch_runs())
    code = _code_hash(patch)
    legacy_code = _code_hash(legacy_patch)
    _check_unique(label, fingerprint, code, runs, new_batch=False, bench=True,
                  other_runs=other_runs)
    notebook = _build({**replica, "TAAF_ARM": label})
    slug = f"v4-{label}"
    out = HERE / "out" / slug
    out.mkdir(parents=True, exist_ok=True)
    notebook_name = f"{slug}.ipynb"
    source = json.dumps(notebook, indent=1) + "\n"
    if len(source.encode("utf-8")) >= KERNEL_SOURCE_LIMIT:
        raise SystemExit(f"notebook is {len(source.encode())} bytes; Kaggle rejects sources >= 1 MB")
    (out / notebook_name).write_text(source, encoding="utf-8")
    metadata = build_metadata(template_meta, slug, slug, notebook_name)
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    configs = [c["name"] + ("" if c["enabled"] else " (disabled)") for c in plan["configs"]]
    print(f"built bench {args.bench!r} ({fingerprint}) from {BASE_REF}..{head}: configs {', '.join(configs)}; "
          + (f"replica {args.replica}; " if args.replica > 1 else "")
          + f"{len(source.encode())} bytes -> {out / notebook_name}")
    if not args.push:
        return
    pushed = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(out), "--accelerator", metadata["machine_shape"]],
        capture_output=True, text=True,
    )
    output = (pushed.stdout + pushed.stderr).strip()
    print(output)
    match = re.search(r"version (\d+)", output, re.IGNORECASE)
    if pushed.returncode != 0 or "error" in output.lower() or match is None:
        raise SystemExit("push failed; nothing recorded")
    runs.append({
        "arm": label, "kind": "bench", "kernel": f"{OWNER}/{slug}", "version": int(match.group(1)),
        "commit": head, "code": legacy_code, "runtime_code": code, "fingerprint": fingerprint,
        "pushed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # never part of an arm batch: it runs no harness code
        "batch_open": False,
        "bench": args.bench, "configs": [c["name"] for c in plan["configs"] if c["enabled"]],
        **({"replica": args.replica} if args.replica > 1 else {}),
    })
    (HERE / "runs.json").write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
    print(f"recorded {label!r} as {OWNER}/{slug} version {match.group(1)} in kaggle/runs.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--arm", help="an arm defined in kaggle/arms.json")
    target.add_argument("--bench", help="a serving benchmark defined in kaggle/bench.json (no games)")
    parser.add_argument("--push", action="store_true",
                        help="push (experiments: kernel v4-<arm>) and record it in runs.json")
    kind = parser.add_mutually_exclusive_group()
    kind.add_argument("--submission", action="store_true",
                      help="build/push as a new version of the one submission kernel, v4-submission; "
                           "skips the duplicate and batch checks and leaves the batch as it is")
    kind.add_argument("--new-batch", action="store_true",
                      help="start a new batch at HEAD (closes the previous batch in runs.json)")
    parser.add_argument("--gate-passed", action="store_true",
                        help="confirm that a gated arm's gate (arms.json \"gate\", e.g. a bench result) holds; "
                             "required to push a gated arm")
    parser.add_argument("--replica", type=int, default=1, metavar="N",
                        help="run N of an arm or bench that already ran (N >= 2): adds the inert env var "
                             f"{REPLICA_ENV}=N so the duplicate check lets the repeat through")
    args = parser.parse_args()
    if args.bench:
        if args.submission or args.new_batch or args.gate_passed:
            raise SystemExit("--bench takes none of --submission, --new-batch and --gate-passed")
        return main_bench(args)

    config = _arm_config(args.arm)
    _check_gate(args.arm, config["gate"], push=args.push, gate_passed=args.gate_passed)
    runtime_patches = _bench_patches(config["runtime_patches"], f"arm {args.arm!r}")
    patch = build_patch()
    legacy_patch = build_legacy_patch()
    head = _git("rev-parse", "--short", "HEAD").strip()
    template = json.loads((HERE / "template.ipynb").read_text(encoding="utf-8"))
    template_meta = json.loads((HERE / "template-metadata.json").read_text(encoding="utf-8"))

    def _build(env: dict[str, str], harness_patch: str = patch) -> dict:
        return build_notebook(
            template, harness_patch, env=env, serve=config["serve"], serve_env=config["serve_env"],
            runtime_patches=runtime_patches,
            demo_games=config["demo_games"], demo_minutes=config["demo_minutes"],
        )

    run_env = {**config["env"], **_replica_env(args.replica)}
    fingerprint = _fingerprint(_build(run_env), config["datasets"])
    runs = _load_runs()
    other_runs = _other_branch_runs(runs, _branch_runs())
    code = _code_hash(patch)
    legacy_code = _code_hash(legacy_patch)
    legacy_fingerprint = _fingerprint(_build(run_env, legacy_patch), config["datasets"])
    if args.submission:
        same = [run for run in runs + other_runs if run.get("fingerprint") in {fingerprint, legacy_fingerprint}
                and not _is_submission(run) and run.get("status") != "failed"]
        for run in same:
            where = f" (recorded on {', '.join(run['branches'])})" if run.get("branches") else ""
            print(f"submitting the notebook {run['arm']!r} ran as {run.get('kernel')} "
                  f"version {run.get('version')}{where}")
        if not same:
            print("note: no recorded experiment ran this exact notebook")
    else:
        _check_unique(
            args.arm, fingerprint, code, _with_runtime_code(runs), new_batch=args.new_batch,
            legacy_fingerprint=legacy_fingerprint, legacy_code=legacy_code,
            other_runs=_with_runtime_code(other_runs),
        )
    # The arm name rides along as an inert env var so a version's own output says what it was.
    notebook = _build({**run_env, "TAAF_ARM": args.arm})
    slug = SUBMISSION_SLUG if args.submission else f"v4-{args.arm}"
    out = HERE / "out" / slug
    out.mkdir(parents=True, exist_ok=True)
    notebook_name = f"{slug}.ipynb"
    source = json.dumps(notebook, indent=1) + "\n"
    if len(source.encode("utf-8")) >= KERNEL_SOURCE_LIMIT:
        raise SystemExit(f"notebook is {len(source.encode())} bytes; Kaggle rejects sources >= 1 MB")
    (out / notebook_name).write_text(source, encoding="utf-8")
    metadata = build_metadata(template_meta, slug, slug, notebook_name, datasets=config["datasets"])
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"built arm {args.arm!r} ({fingerprint}) from {BASE_REF}..{head}: "
          f"env {config['env'] or 'none'}, serve {config['serve'] or 'none'}"
          + (f", runtime patches {config['runtime_patches']}" if config["runtime_patches"] else "")
          + (f", datasets {config['datasets']}" if config["datasets"] else "")
          + (f", replica {args.replica}" if args.replica > 1 else "")
          + (f"; gated on {config['gate']}" if config["gate"] else ""))
    if not args.push:
        return

    pushed = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(out), "--accelerator", metadata["machine_shape"]],
        capture_output=True, text=True,
    )
    output = (pushed.stdout + pushed.stderr).strip()
    print(output)
    match = re.search(r"version (\d+)", output, re.IGNORECASE)
    if pushed.returncode != 0 or "error" in output.lower() or match is None:
        raise SystemExit("push failed; nothing recorded")
    if args.new_batch:
        for run in runs:
            run["batch_open"] = False
    runs.append({
        # "code" stays the old whole-ARC3-Inference hash so builders on branches not yet
        # rebased onto this one still read the record correctly.
        "arm": args.arm, "kernel": f"{OWNER}/{slug}", "version": int(match.group(1)), "commit": head,
        "code": legacy_code, "runtime_code": code, "fingerprint": fingerprint,
        "pushed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # A submission never belongs to an experiment batch.
        "batch_open": not args.submission, **({"submission": True} if args.submission else {}),
        **config,
        **({"replica": args.replica} if args.replica > 1 else {}),
    })
    (HERE / "runs.json").write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
    print(f"recorded {args.arm!r} as {OWNER}/{slug} version {match.group(1)} in kaggle/runs.json")


if __name__ == "__main__":
    main()
