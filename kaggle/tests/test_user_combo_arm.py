"""Every key of the user-combo arm reaches the code that reads it.

The arm's notebook is built with kaggle/build_notebook.py (dry: nothing is
pushed), then:
- serving: the launch-computing parts of its serving cell run (as in
  test_throughput_arms.py) and the argv, the server env and the target model
  view are checked against arms.json;
- harness: the env cell's setup_env (the template's, plus the arm's overrides,
  priority scheduling on) is applied to a fresh interpreter that imports the
  harness the way the notebook's game cell does, and the values the harness
  resolved are checked: the import-time solver constants, the call-time flag
  readers, a constructed ToolAgent and its system prompt.

Run from ARC3-Inference:  python -m pytest -q ../kaggle/tests/test_user_combo_arm.py
"""
import json
import subprocess
import sys
import types
from pathlib import Path

from test_throughput_arms import (
    ASSEMBLE, INSTALL, LAUNCH, RUNTIME_ENV, VALIDATE, PATCH, TEMPLATE, _between, _Socket, bn,
)

REPO = Path(__file__).resolve().parents[2]
ARM = "user-combo"
ARMS = json.loads((REPO / "kaggle" / "arms.json").read_text(encoding="utf-8"))["arms"]


def _built():
    config = bn._arm_config(ARM)
    return config, bn.build_notebook(
        TEMPLATE, PATCH, env={**config["env"], "TAAF_ARM": ARM},
        serve=config["serve"], serve_env=config["serve_env"],
        runtime_patches=bn._bench_patches(config["runtime_patches"], f"arm {ARM!r}"),
        demo_games=config["demo_games"], demo_minutes=config["demo_minutes"],
    )


def _cell(nb, needle):
    cells = [bn._cell_text(c) for c in nb["cells"] if c["cell_type"] == "code" and needle in bn._cell_text(c)]
    assert len(cells) == 1, needle
    return cells[0]


def _render(cell, tmp_path):
    """The launch the serving cell computes: (CFG, argv, server env, model dir)."""
    model = tmp_path / "model"
    wheelhouse = tmp_path / "wheelhouse"
    (wheelhouse / "wheels").mkdir(parents=True, exist_ok=True)
    (wheelhouse / "wheels" / "sglang-2.5.3-py3-none-any.whl").write_bytes(b"")
    (wheelhouse / "requirements.lock").write_text("")
    (wheelhouse / "hot_tokens_64k.pt").write_bytes(b"")
    model.mkdir(exist_ok=True)
    for name in ("tokenizer.json", "chat_template.jinja", "model.safetensors"):
        (model / name).write_text("{}")
    (model / "config.json").write_text(json.dumps({"text_config": {"num_experts_per_tok": 10}}))
    namespace = {"WORKING_DIR": str(tmp_path / "working"), "SERVED_MODEL_NAME": "flashnext"}
    exec(compile(cell[:cell.index(bn.SERVE_ANCHOR)], "serving cell: CFG", "exec"), namespace)
    namespace.update(PREFIX=str(tmp_path / "sgl-intel"), LOG=str(tmp_path / "serve.log"))
    exec(compile(_between(cell, bn.SERVE_ANCHOR, VALIDATE), "serving cell: helpers", "exec"), namespace)
    namespace.update(
        MODEL_DIR=str(model), DRAFT_MODEL_DIR=str(tmp_path / "draft"), WHEELHOUSE_DIR=str(wheelhouse),
        SERVED_MODEL_PORT=8000, socket=types.SimpleNamespace(socket=_Socket),
        sha256=lambda path: (namespace["TOKEN_MAP_SHA"] if str(path).endswith(".pt")
                             else namespace["TOKENIZER_SHA"]),
    )
    exec(compile(_between(cell, VALIDATE, INSTALL), "serving cell: input checks", "exec"), namespace)
    namespace.update(
        os=types.SimpleNamespace(environ={"PATH": "/usr/bin", "HOME": "/root", "PYTHONPATH": "/x"}),
        CUDA_HOME=namespace["VENV"] + "/lib/python3.12/site-packages/nvidia/cu13", CC="gcc-13", CXX="g++-13",
    )
    exec(compile(_between(cell, RUNTIME_ENV, bn.SERVE_ENV_ANCHOR), "serving cell: env", "exec"), namespace)
    namespace["DRAFT_VIEW"] = str(tmp_path / "sgl-intel" / "draft-view-0")
    exec(compile(_between(cell, ASSEMBLE, LAUNCH), "serving cell: argv", "exec"), namespace)
    return namespace["CFG"], namespace["args"], namespace["env"], model


def _arg(args, flag):
    return args[args.index(flag) + 1]


def test_serve_keys_reach_the_launch(tmp_path):
    config, nb = _built()
    cfg, args, env, model = _render(_cell(nb, bn.SERVE_ANCHOR), tmp_path)
    serve = ARMS[ARM]["serve"]
    assert {key: cfg[key] for key in serve} == serve
    assert _arg(args, "--max-running-requests") == "14"
    assert _arg(args, "--cuda-graph-max-bs-decode") == "14"
    graphs = args[args.index("--cuda-graph-bs-decode") + 1:]
    graphs = [int(x) for x in graphs[:next((i for i, x in enumerate(graphs) if x.startswith("--")), len(graphs))]]
    assert graphs == list(range(1, 15))
    assert _arg(args, "--max-mamba-cache-size") == "84"
    assert _arg(args, "--context-length") == "106496"
    assert int(_arg(args, "--context-length")) >= int(config["env"]["LOCAL_ANALYZER_CONTEXT_WINDOW"])
    assert _arg(args, "--mem-fraction-static") == "0.96"
    # 13 experts per token: the target is served from a view whose config says so ...
    view = Path(_arg(args, "--model-path"))
    assert view != model and json.loads((view / "config.json").read_text())["text_config"] == {"num_experts_per_tok": 13}
    assert (view / "model.safetensors").resolve() == (model / "model.safetensors").resolve()
    # ... the checkpoint and the MTP drafter keep theirs (the draft view is built from DRAFT_MODEL_DIR)
    assert json.loads((model / "config.json").read_text())["text_config"] == {"num_experts_per_tok": 10}
    assert _arg(args, "--speculative-draft-model-path") == str(tmp_path / "sgl-intel" / "draft-view-0")
    # base_norm routing reaches the SGLang process env, where the patch reads it
    assert env["SGLANG_ARC_MOE_ROUTING"] == "base_norm" and env["SGLANG_ARC_MOE_BASE_K"] == "10"


def test_routing_patch_is_embedded_and_reads_the_server_env():
    _, nb = _built()
    cell = _cell(nb, bn.SERVE_ANCHOR)
    assert "embedded from serving/runtime_patches/: 20-moe-routing.py" in cell
    patch = (REPO / "serving" / "runtime_patches" / "20-moe-routing.py").read_text()
    assert '_arc_os.environ.get("SGLANG_ARC_MOE_ROUTING"' in patch
    assert '_arc_os.environ.get("SGLANG_ARC_MOE_BASE_K"' in patch
    # the draft's MoE blocks are skipped: SGLang 2.5.3 sets this False on every draft TopK
    assert '(_ARC_DRAFT or getattr(cfg, "allow_routed_experts_capture", True))' in patch


def _notebook_env(nb) -> dict[str, str]:
    source = _cell(nb, "setup_env = {")
    start = source.index("setup_env = {")
    end = source.index(bn.ENV_UPDATE)
    namespace = {"SERVER_BASE_URL": "http://127.0.0.1:9/v1", "SERVED_MODEL_NAME": "flashnext",
                 "USE_PRIORITY_SCHEDULING": True}
    exec(source[start:end], namespace)  # noqa: S102 - the built notebook's own cell
    return {k: str(v) for k, v in namespace["setup_env"].items()}


PROBE = r"""
import json
from inference.framework import solver
from inference.agent import tool_agent as ta, prompts
from inference.agent.action_names import reset_exposed
agent = ta.ToolAgent(model="flashnext", base_url="http://127.0.0.1:9/v1", provider="vllm")
system = agent._system_prompt
print(json.dumps({
    "solver": {k: getattr(solver, k) for k in
               ("ANIMATION_ENABLED", "ANIM_TRUTH", "ANIM_RING_SIZE", "ANIM_RING_ENABLED", "ANIM_DIGEST_ENABLED",
                "BATCH_NOOP_BLOCK", "STALE_STATE_BLOCK")},
    "truth": ta._anim_truth_enabled(), "ring": ta._anim_ring_size(), "digest": ta._anim_digest_enabled(),
    "digest_chars": ta._anim_digest_chars(), "gate": ta._anim_digest_gate(),
    "ledger": ta._ledger_on_trim(), "ledger_object": agent._ledger is not None,
    "reset": reset_exposed(), "streams": ta._max_active_streams(),
    "gate_slots": ta._priority_gate()._slots if hasattr(ta._priority_gate(), "_slots") else None,
    "temperature": ta._LOCAL_ANALYZER_TEMPERATURE, "window": ta._LOCAL_ANALYZER_CONTEXT_WINDOW,
    "budget": agent._context_budget_tokens, "drain": ta._context_drain_tokens(),
    "opener_dedup": sorted(ta._opener_dedup_modes()),
    "prompt": {name: getattr(prompts, name) in system for name in
               ("ANIMATION_TRUTH_ADDENDUM", "ANIMATION_RING_ADDENDUM", "ANIMATION_DIGEST_ADDENDUM",
                "RESET_INFO_ADDENDUM", "LEDGER_ON_TRIM_ADDENDUM")},
    "inference": solver.__file__,
}))
"""


def test_env_keys_reach_the_harness():
    _, nb = _built()
    env = _notebook_env(nb)
    arm_env = ARMS[ARM]["env"]
    assert {key: env[key] for key in arm_env} == arm_env  # the arm's values win over the template's
    root = REPO / "ARC3-Inference"
    proc = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=root, capture_output=True, text=True, timeout=120,
        env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "PYTHONPATH": str(root), **env},
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["inference"].startswith(str(root))
    assert got["solver"] == {
        "ANIMATION_ENABLED": True, "ANIM_TRUTH": True, "ANIM_RING_SIZE": 8, "ANIM_RING_ENABLED": True,
        "ANIM_DIGEST_ENABLED": True, "BATCH_NOOP_BLOCK": True, "STALE_STATE_BLOCK": True,
    }
    assert (got["truth"], got["ring"], got["digest"], got["digest_chars"], got["gate"]) == (True, 8, True, 600, "novel")
    assert got["ledger"] and got["ledger_object"]
    assert got["reset"] is True
    assert got["streams"] == 14
    assert got["temperature"] == 0.65
    assert got["window"] == 98304 and got["drain"] == 43008
    assert got["opener_dedup"] == ["resume_image", "static"]  # ARC3_OPENER_DEDUP=1 since 44f2abc1
    assert all(got["prompt"].values()), got["prompt"]
