"""The memory-v2 arms are pf-keep-10x plus their own flags, nothing else.

Each arm's env is pf-keep-10x's plus exactly its listed memory-v2 flags, with
pf-keep-10x's serving. Their notebooks are built with kaggle/build_notebook.py
(dry: nothing is pushed) next to pf-keep-10x's: every cell must match except
the env overrides, which differ by exactly those flags and the arm label. The
built env cell is then applied to a fresh interpreter that constructs the
ToolAgent the way the notebook's game cell does: only pf-keep-append-10x
changes the system prompt (one (old, new) row in the memory section), and no
arm changes the tool schema.

Run from ARC3-Inference:  python -m pytest -q ../kaggle/tests/test_mem_v2_arms.py
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from test_prompt_facts_arm import _built, _env_cell_index, _notebook_env
from test_throughput_arms import bn

REPO = Path(__file__).resolve().parents[2]
ARMS = json.loads((REPO / "kaggle" / "arms.json").read_text(encoding="utf-8"))["arms"]
BASE = "pf-keep-10x"
RECAP = {"ARC3_MEM_RECAP_CHARS": "800", "ARC3_MEM_COMPRESS_ROUTES_FIRST": "1"}
FLAGS = {
    "pf-keep-append-10x": {"ARC3_MEM_APPEND_COMMIT": "1"},
    "pf-keep-recap-10x": RECAP,
    "pf-keep-memv2-10x": {**RECAP, "ARC3_MEM_ROUTE_FORMAT": "readable", "ARC3_MEM_HEAD_WORDS": "1",
                          "ARC3_MEM_RECAP_ASK_V2": "1", "ARC3_MEM_ATTEMPT_COUNTS": "1",
                          "ARC3_MEM_SEEN_GAME_ACTION": "1", "ARC3_MEM_EFFECT_WORDS": "1"},
    "pf-keep-stuck-10x": {"ARC3_MEM_STUCK_FACT": "1"},
}


@pytest.mark.parametrize("arm", sorted(FLAGS))
def test_arm_is_pf_keep_plus_its_flags(arm):
    got, base = ARMS[arm], ARMS[BASE]
    assert got["env"] == {**base["env"], **FLAGS[arm]}
    assert not set(FLAGS[arm]) & set(base["env"])
    assert got["serve"] == base["serve"]
    assert set(got) == {"note", "env", "serve"}
    assert BASE in got["note"] and "--new-batch" in got["note"]
    for flag in FLAGS[arm]:
        assert flag in got["note"]


@pytest.mark.parametrize("arm", sorted(FLAGS))
def test_notebooks_differ_only_by_the_flags(arm):
    nb, base = _built(arm), _built(BASE)
    assert len(nb["cells"]) == len(base["cells"])
    env_cell = _env_cell_index(nb)
    assert env_cell == _env_cell_index(base)
    for i, (cell, base_cell) in enumerate(zip(nb["cells"], base["cells"])):
        if i != env_cell:
            assert cell == base_cell, f"cell {i} differs"
    lines = set(bn._cell_text(nb["cells"][env_cell]).splitlines())
    base_lines = set(bn._cell_text(base["cells"][env_cell]).splitlines())
    added = {f"setup_env[{key!r}] = {value!r}" for key, value in FLAGS[arm].items()}
    assert lines - base_lines == added | {f"setup_env['TAAF_ARM'] = {arm!r}"}
    assert base_lines - lines == {f"setup_env['TAAF_ARM'] = {BASE!r}"}
    assert {k: _notebook_env(nb)[k] for k in FLAGS[arm]} == FLAGS[arm]


PROBE = r"""
import json
from inference.agent import tool_agent as ta
from inference.agent import prompts
agent = ta.ToolAgent(model="flashnext", base_url="http://127.0.0.1:9/v1", provider="vllm")
agent._ensure_session = lambda state_path: None
print(json.dumps({
    "system": agent._system_prompt,
    "tools": agent._tools(None),
    "rows": [list(r) for r in prompts.MEMORY_LAYOUT_APPEND_COMMIT_ROWS],
}))
"""


def _probe(arm):
    env = _notebook_env(_built(arm))
    root = REPO / "ARC3-Inference"
    proc = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=root, capture_output=True, text=True, timeout=120,
        env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "PYTHONPATH": str(root), **env},
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_only_the_append_arm_changes_the_system_prompt():
    base = _probe(BASE)
    (old, new), = base["rows"]
    assert base["system"].count(old) == 1
    for arm in sorted(FLAGS):
        got = _probe(arm)
        assert got["tools"] == base["tools"], arm
        want = base["system"].replace(old, new) if arm == "pf-keep-append-10x" else base["system"]
        assert got["system"] == want, arm
