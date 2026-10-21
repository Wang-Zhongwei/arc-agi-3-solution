"""The throughput-stack arms render the SGLang launch their arms.json entry says, and nothing else.

Each test builds an arm's serving cell with kaggle/build_notebook.py and runs
the parts of it that compute the launch: CFG and the arm's overrides, the
input checks, the server env and the argv. Installing SGLang, the CUDA
toolkit, the draft view and the launch itself are skipped; the FR-Spec hashes
and the socket probe are stubbed. Dry builds only: nothing is pushed.

Run from ARC3-Inference:  python -m pytest -q ../kaggle/tests/test_throughput_arms.py
"""
import importlib.util
import json
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bn = _load("build_notebook", REPO / "kaggle" / "build_notebook.py")
jc = _load("jit_cache_source", REPO / "serving" / "kaggle" / "jit_cache.py")
TEMPLATE = json.loads((REPO / "kaggle" / "template.ipynb").read_text(encoding="utf-8"))
ARMS = json.loads((REPO / "kaggle" / "arms.json").read_text(encoding="utf-8"))["arms"]
PATCH = "diff --git a/ARC3-Inference/x b/ARC3-Inference/x\n"  # stands in for the harness diff
EXTRA_GRAPHS = [3, 5, 6]
TPUT_ENV = {
    "ARC3_LEDGER_ON_TRIM": "1",
    "ARC3_MAX_ACTIVE_STREAMS": "11",
    "ARC3_STUB_DEAD_REASONING": "1",
    "ARC3_VERBATIM_HISTORY": "1",
    "ARC3_PREFIX_REUSE_DIAG": "1",
}
# Cut points in the serving cell; every one is in Franzen's cell and in ours.
VALIDATE = "# ---- validate notebook inputs"
INSTALL = "# ---- one-time offline install"
RUNTIME_ENV = "# ---- runtime environment ----"
ASSEMBLE = "# ---- assemble Pennyroyal"
LAUNCH = "# ---- launch detached"


def _template_serving_cell():
    """Franzen's serving cell as the template ships it (what stack-v2's builder edited)."""
    cells = [bn._cell_text(c) for c in TEMPLATE["cells"]
             if c["cell_type"] == "code" and bn.SERVE_ANCHOR in bn._cell_text(c)]
    assert len(cells) == 1
    return cells[0]


def _serving_cell(arm, *, serve=None):
    config = bn._arm_config(arm)
    nb = bn.build_notebook(
        TEMPLATE, PATCH, env={**config["env"], "TAAF_ARM": arm},
        serve=config["serve"] if serve is None else serve, serve_env=config["serve_env"],
        runtime_patches=bn._bench_patches(config["runtime_patches"], f"arm {arm!r}"),
        demo_games=config["demo_games"], demo_minutes=config["demo_minutes"],
    )
    cells = [bn._cell_text(c) for c in nb["cells"]
             if c["cell_type"] == "code" and bn.SERVE_ANCHOR in bn._cell_text(c)]
    assert len(cells) == 1
    return cells[0]


def _between(cell, start, end):
    return cell[cell.index(start):cell.index(end)]


class _Socket:
    """socket.socket() whose probe finds the port free."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def connect_ex(self, address):
        return 111


def _render(cell, tmp_path):
    """Run the launch-computing parts of a serving cell; return (CFG, argv, server env)."""
    model = tmp_path / "model"
    wheelhouse = tmp_path / "wheelhouse"
    (wheelhouse / "wheels").mkdir(parents=True, exist_ok=True)
    (wheelhouse / "wheels" / "sglang-2.5.3-py3-none-any.whl").write_bytes(b"")
    (wheelhouse / "requirements.lock").write_text("")
    (wheelhouse / "hot_tokens_64k.pt").write_bytes(b"")
    model.mkdir(exist_ok=True)
    for name in ("config.json", "tokenizer.json", "chat_template.jinja"):
        (model / name).write_text("{}")
    namespace = {"WORKING_DIR": str(tmp_path / "working"), "SERVED_MODEL_NAME": "flashnext"}
    exec(compile(cell[:cell.index(bn.SERVE_ANCHOR)], "serving cell: CFG", "exec"), namespace)
    # Everything the cell writes goes under tmp_path; the paths are the same for every arm.
    namespace.update(PREFIX=str(tmp_path / "sgl-intel"), LOG=str(tmp_path / "serve.log"))
    exec(compile(_between(cell, bn.SERVE_ANCHOR, VALIDATE), "serving cell: helpers", "exec"), namespace)
    namespace.update(
        MODEL_DIR=str(model), DRAFT_MODEL_DIR=str(tmp_path / "draft"), WHEELHOUSE_DIR=str(wheelhouse),
        SERVED_MODEL_PORT=8000, socket=types.SimpleNamespace(socket=_Socket),
        sha256=lambda path: (namespace["TOKEN_MAP_SHA"] if str(path).endswith(".pt")
                             else namespace["TOKENIZER_SHA"]),
    )
    exec(compile(_between(cell, VALIDATE, INSTALL), "serving cell: input checks", "exec"), namespace)
    cuda_home = namespace["VENV"] + "/lib/python3.12/site-packages/nvidia/cu13"
    namespace.update(
        os=types.SimpleNamespace(environ={"PATH": "/usr/bin", "HOME": "/root", "PYTHONPATH": "/x"}),
        CUDA_HOME=cuda_home, CC="gcc-13", CXX="g++-13",
    )
    exec(compile(_between(cell, RUNTIME_ENV, bn.SERVE_ENV_ANCHOR), "serving cell: env", "exec"), namespace)
    namespace["DRAFT_VIEW"] = str(tmp_path / "sgl-intel" / "draft-view-0") if namespace["CFG"]["SPEC"] else None
    exec(compile(_between(cell, ASSEMBLE, LAUNCH), "serving cell: argv", "exec"), namespace)
    return namespace["CFG"], namespace["args"], namespace["env"]


def _option(args, flag):
    """The values after ``flag`` up to the next option."""
    i = args.index(flag) + 1
    values = []
    while i < len(args) and not str(args[i]).startswith("--"):
        values.append(args[i])
        i += 1
    return values


def _drop_option(args, flag):
    i = args.index(flag)
    return args[:i] + args[i + 1 + len(_option(args, flag)):]


def test_control_launch_is_franzens(tmp_path, monkeypatch):
    """With the new keys at their defaults, control launches exactly as stack-v2 did."""
    ours = _render(_serving_cell("control"), tmp_path)
    monkeypatch.setattr(bn, "_serving_launcher", lambda arm_patches=None: _template_serving_cell())
    franzen = _render(_serving_cell("control"), tmp_path)
    assert ours[1] == franzen[1]
    assert ours[2] == franzen[2]
    assert _option(ours[1], "--cuda-graph-bs-decode") == ["1", "2", "4", "7", "8", "9", "10"]
    assert ours[0]["CUDAGRAPH_BS_EXTRA"] == []


@pytest.mark.parametrize("arm", ["stack-v2", "gate11", "stub-dead-reasoning", "prefix-reuse-60"])
def test_harness_only_arms_launch_like_control(arm, tmp_path):
    assert _render(_serving_cell(arm), tmp_path)[1:] == _render(_serving_cell("control"), tmp_path)[1:]


def test_tput_safe_adds_only_the_extra_graphs(tmp_path):
    control_cfg, control_args, control_env = _render(_serving_cell("control"), tmp_path)
    cfg, args, env = _render(_serving_cell("tput-safe"), tmp_path)
    assert _option(args, "--cuda-graph-bs-decode") == [str(n) for n in range(1, 11)]
    assert _drop_option(args, "--cuda-graph-bs-decode") == _drop_option(control_args, "--cuda-graph-bs-decode")
    assert env == control_env
    assert {k: v for k, v in cfg.items() if cfg[k] != control_cfg[k]} == {"CUDAGRAPH_BS_EXTRA": EXTRA_GRAPHS}
    assert bn._arm_config("tput-safe")["env"] == TPUT_ENV
    assert bn._arm_config("tput-safe")["runtime_patches"] == []


def test_tput_safe_lowm_adds_only_the_low_m_patches(tmp_path):
    safe, lowm = _serving_cell("tput-safe"), _serving_cell("tput-safe-lowm")
    assert _render(lowm, tmp_path)[1:] == _render(safe, tmp_path)[1:]
    assert bn._arm_config("tput-safe-lowm")["env"] == TPUT_ENV
    def table(cell):
        if "SGLANG_RUNTIME_PATCHES = {}\n" in cell:
            return {}
        namespace = {}
        start = cell.index("SGLANG_RUNTIME_PATCHES = ")
        exec("import base64, gzip, json\n" + cell[start:cell.index("\n", cell.index('""")))', start))], namespace)
        return namespace["SGLANG_RUNTIME_PATCHES"]

    # tput-safe carries only the shared serving/runtime_patches/; lowm adds the two bench patches
    shared = {p.name: p.read_text(encoding="utf-8") for p in sorted(bn.RUNTIME_PATCHES.glob("*"))
              if p.suffix in (".py", ".patch")}
    assert table(safe) == shared
    assert table(lowm) == {**shared, **{
        name: (bn.BENCH_PATCHES / name).read_text(encoding="utf-8") for name in ("m64.py", "nosplitk.py")}}


def test_tput_accept05_changes_only_the_acceptance_thresholds(tmp_path):
    _, safe_args, safe_env = _render(_serving_cell("tput-safe"), tmp_path)
    _, args, env = _render(_serving_cell("tput-accept05"), tmp_path)
    assert _option(args, "--speculative-accept-threshold-single") == ["0.5"]
    assert _option(args, "--speculative-accept-threshold-acc") == ["0.5"]
    assert _option(safe_args, "--speculative-accept-threshold-single") == ["1.0"]
    for flag in ("--speculative-accept-threshold-single", "--speculative-accept-threshold-acc"):
        args, safe_args = _drop_option(args, flag), _drop_option(safe_args, flag)
    assert args == safe_args and env == safe_env
    assert bn._arm_config("tput-accept05")["env"] == TPUT_ENV


def test_panel_tput_safe_is_tput_safe_on_four_streams(tmp_path):
    assert _render(_serving_cell("panel-tput-safe"), tmp_path)[1:] == _render(_serving_cell("tput-safe"), tmp_path)[1:]
    panel = bn._arm_config("panel-tput-safe")
    assert panel["env"] == {**TPUT_ENV, "ARC3_MAX_ACTIVE_STREAMS": "4"}
    assert (panel["demo_games"], panel["demo_minutes"]) == ("balanced", 20)


def _jit_snapshot(arm, tmp_path):
    cfg, _, env = _render(_serving_cell(arm), tmp_path)
    return jc.config_snapshot(cfg, env, b'{"model_type": "qwen3_next"}')


def test_tput_coldstart_save_saves_the_cache_tput_safe_can_restore(tmp_path):
    """A JIT cache restores only under the same kernel-relevant CFG and server env, so tput-safe
    needs a cache saved with its extra graphs: coldstart-save's would not match it."""
    tput = jc.config_hash(_jit_snapshot("tput-safe", tmp_path))
    assert jc.config_hash(_jit_snapshot("tput-coldstart-save", tmp_path)) == tput
    assert jc.config_hash(_jit_snapshot("coldstart-save", tmp_path)) != tput
    # the acceptance thresholds compile nothing, so tput-accept05 can restore the same cache
    assert jc.config_hash(_jit_snapshot("tput-accept05", tmp_path)) == tput
    assert bn._arm_config("tput-coldstart-save")["serve"]["JIT_CACHE_SAVE"] is True


@pytest.mark.parametrize("extra", [[11], [0], ["3"]])
def test_extra_graph_sizes_must_be_batch_sizes_up_to_the_max(extra, tmp_path):
    with pytest.raises(AssertionError, match="CUDAGRAPH_BS_EXTRA"):
        _render(_serving_cell("control", serve={"CUDAGRAPH_BS_EXTRA": extra}), tmp_path)


def test_tput_gates_bench_runs_the_configs_that_gate_the_arms():
    plan = bn.bench_plan("tput-gates")
    assert [c["name"] for c in plan["configs"]] == ["C0", "C3", "C1"]
    assert all(c["enabled"] for c in plan["configs"])
    assert set(plan["patches"]) == set(bn._arm_config("tput-safe-lowm")["runtime_patches"])
    c1 = next(c for c in plan["configs"] if c["name"] == "C1")
    accept = {k: v for k, v in bn._arm_config("tput-accept05")["serve"].items() if k.startswith("SPEC_ACCEPT")}
    assert c1["overrides"]["cfg"] == accept
    # The low-M gate ("no slower at batch 8 or below") compares C3 with C0 batch by batch,
    # so both must run the same sweeps, covering batches 7 and 8.
    sweeps = {c["name"]: {p["name"] for p in c["phases"] if p["name"].startswith("sweep")}
              for c in plan["configs"]}
    assert sweeps["C3"] == sweeps["C0"] == {"sweep1", "sweep4", "sweep7", "sweep8"}
