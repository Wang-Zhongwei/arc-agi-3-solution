"""The prompt-facts-10x arm is user-combo-10x plus four harness flags, nothing else.

Its notebook is built with kaggle/build_notebook.py (dry: nothing is pushed)
next to user-combo-10x's: every cell must match except the env overrides,
which differ by exactly the four flags and the arm label. The built env cell
is then applied to a fresh interpreter that constructs the ToolAgent the way
the notebook's game cell does, and the system prompt must carry each flag's
text.

Run from ARC3-Inference:  python -m pytest -q ../kaggle/tests/test_prompt_facts_arm.py
"""
import json
import subprocess
import sys
from pathlib import Path

from test_throughput_arms import PATCH, TEMPLATE, bn

REPO = Path(__file__).resolve().parents[2]
ARM, BASE = "prompt-facts-10x", "user-combo-10x"
ARMS = json.loads((REPO / "kaggle" / "arms.json").read_text(encoding="utf-8"))["arms"]
FLAGS = {
    "ARC3_BUDGET_TRUTH": "1",
    "ARC3_PROMPT_HYGIENE": "1",
    "ARC3_IMAGE_GEOMETRY": "1",
    "ARC3_PREFER_TOOL_CALLS": "1",
}


def _built(arm):
    config = bn._arm_config(arm)
    return bn.build_notebook(
        TEMPLATE, PATCH, env={**config["env"], "TAAF_ARM": arm},
        serve=config["serve"], serve_env=config["serve_env"],
        runtime_patches=bn._bench_patches(config["runtime_patches"], f"arm {arm!r}"),
        demo_games=config["demo_games"], demo_minutes=config["demo_minutes"],
    )


def _env_cell_index(nb):
    hits = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and "setup_env = {" in bn._cell_text(c)]
    assert len(hits) == 1
    return hits[0]


def test_arm_is_user_combo_10x_plus_the_flags():
    arm, base = ARMS[ARM], ARMS[BASE]
    assert arm["env"] == {**base["env"], **FLAGS}
    assert arm["serve"] == base["serve"]
    assert set(arm) == {"note", "env", "serve"} and set(base) == {"note", "env", "serve"}
    for fact in (BASE, "56.94", "52.70", "49.09"):
        assert fact in arm["note"]


def test_notebooks_differ_only_by_the_flags():
    nb, base = _built(ARM), _built(BASE)
    assert len(nb["cells"]) == len(base["cells"])
    env_cell = _env_cell_index(nb)
    assert env_cell == _env_cell_index(base)
    for i, (cell, base_cell) in enumerate(zip(nb["cells"], base["cells"])):
        if i != env_cell:
            assert cell == base_cell, f"cell {i} differs"
    lines = set(bn._cell_text(nb["cells"][env_cell]).splitlines())
    base_lines = set(bn._cell_text(base["cells"][env_cell]).splitlines())
    added = {f"setup_env[{key!r}] = {value!r}" for key, value in FLAGS.items()}
    assert lines - base_lines == added | {f"setup_env['TAAF_ARM'] = {ARM!r}"}
    assert base_lines - lines == {f"setup_env['TAAF_ARM'] = {BASE!r}"}


PROBE = r"""
import json
from inference.agent import tool_agent as ta, prompts
agent = ta.ToolAgent(model="flashnext", base_url="http://127.0.0.1:9/v1", provider="vllm")
system = agent._system_prompt
print(json.dumps({
    "flags": [ta._budget_truth(), ta._prompt_hygiene(), ta._image_geometry(), ta._prefer_tool_calls()],
    "geometry": prompts.image_geometry_line(10) in system,
    "prefer": prompts.PREFER_TOOL_CALLS_LINE in system,
    "bar": "is a step bar. It usually counts down a per-level action budget" in system,
    "ledger_last": system.index("Memory that survives history trimming:") > system.index("`frame_diff(before=None"),
    "split": "1x1 tokens.\n- Some games" in system,
    "ephemeral": "ephemeral" in system,
    "game_over": ta._game_over_record_wording() == prompts.GAME_OVER_RECORD_LEDGER,
}))
"""


def _notebook_env(nb):
    source = bn._cell_text(nb["cells"][_env_cell_index(nb)])
    namespace = {"SERVER_BASE_URL": "http://127.0.0.1:9/v1", "SERVED_MODEL_NAME": "flashnext",
                 "USE_PRIORITY_SCHEDULING": True}
    exec(source[source.index("setup_env = {"):source.index(bn.ENV_UPDATE)], namespace)  # noqa: S102
    return {k: str(v) for k, v in namespace["setup_env"].items()}


def test_flags_reach_the_system_prompt():
    env = _notebook_env(_built(ARM))
    assert {key: env[key] for key in FLAGS} == FLAGS
    root = REPO / "ARC3-Inference"
    proc = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=root, capture_output=True, text=True, timeout=120,
        env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "PYTHONPATH": str(root), **env},
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got == {"flags": [True] * 4, "geometry": True, "prefer": True, "bar": True, "ledger_last": True,
                   "split": True, "ephemeral": False, "game_over": True}
