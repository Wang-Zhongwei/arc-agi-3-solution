"""The pf-mem arms are a memory arm plus the four prompt-facts flags, nothing else.

pf-solved-10x is mem-solved-10x plus prompt-facts-10x's four flags, and
pf-keep-10x is mem-recap-keep-10x plus the same four. Their notebooks are
built with kaggle/build_notebook.py (dry: nothing is pushed) next to the
memory parent's: every cell must match except the env overrides, which
differ by exactly the four flags and the arm label. The built env cell is
then applied to a fresh interpreter that constructs the ToolAgent the way
the notebook's game cell does: its system prompt must be prompt-facts-10x's
with the ledger section swapped for the memory parent's memory section, and
its tool schema must be prompt-facts-10x's.

Run from ARC3-Inference:  python -m pytest -q ../kaggle/tests/test_pf_mem_arms.py
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from test_prompt_facts_arm import FLAGS, _built, _env_cell_index, _notebook_env
from test_throughput_arms import bn

REPO = Path(__file__).resolve().parents[2]
ARMS = json.loads((REPO / "kaggle" / "arms.json").read_text(encoding="utf-8"))["arms"]
PF = "prompt-facts-10x"
PAIRS = [("pf-solved-10x", "mem-solved-10x"), ("pf-keep-10x", "mem-recap-keep-10x")]


@pytest.mark.parametrize("arm,memory_arm", PAIRS)
def test_arm_is_the_memory_arm_plus_the_flags(arm, memory_arm):
    got, parent = ARMS[arm], ARMS[memory_arm]
    assert got["env"] == {**parent["env"], **FLAGS}
    assert {k: v for k, v in ARMS[PF]["env"].items() if k not in FLAGS} == ARMS["user-combo-10x"]["env"]
    assert got["serve"] == parent["serve"] == ARMS[PF]["serve"]
    assert set(got) == {"note", "env", "serve"}
    for fact in (memory_arm, PF):
        assert fact in got["note"]


@pytest.mark.parametrize("arm,memory_arm", PAIRS)
def test_notebooks_differ_only_by_the_flags(arm, memory_arm):
    nb, base = _built(arm), _built(memory_arm)
    assert len(nb["cells"]) == len(base["cells"])
    env_cell = _env_cell_index(nb)
    assert env_cell == _env_cell_index(base)
    for i, (cell, base_cell) in enumerate(zip(nb["cells"], base["cells"])):
        if i != env_cell:
            assert cell == base_cell, f"cell {i} differs"
    lines = set(bn._cell_text(nb["cells"][env_cell]).splitlines())
    base_lines = set(bn._cell_text(base["cells"][env_cell]).splitlines())
    added = {f"setup_env[{key!r}] = {value!r}" for key, value in FLAGS.items()}
    assert lines - base_lines == added | {f"setup_env['TAAF_ARM'] = {arm!r}"}
    assert base_lines - lines == {f"setup_env['TAAF_ARM'] = {memory_arm!r}"}


PROBE = r"""
import json
from inference.agent import tool_agent as ta
agent = ta.ToolAgent(model="flashnext", base_url="http://127.0.0.1:9/v1", provider="vllm")
agent._ensure_session = lambda state_path: None
print(json.dumps({
    "system": agent._system_prompt,
    "section": ta._memory_system_section(),
    "tools": agent._tools(None),
    "game_over": list(ta._game_over_record_wording()),
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


@pytest.mark.parametrize("arm,memory_arm", PAIRS)
def test_system_prompt_is_prompt_facts_with_the_memory_section(arm, memory_arm):
    got, pf, mem = _probe(arm), _probe(PF), _probe(memory_arm)
    assert got["section"] == mem["section"] != pf["section"]
    assert pf["system"].count(pf["section"]) == 1
    assert got["system"] == pf["system"].replace(pf["section"], mem["section"])
    assert got["system"].index(mem["section"]) > got["system"].index("`frame_diff(before=None")
    assert got["tools"] == pf["tools"]
    assert got["game_over"] == pf["game_over"]


def test_cap8k_arm_is_pf_solved_with_only_the_output_cap():
    got, base = ARMS["pf-solved-cap8k-10x"], ARMS["pf-solved-10x"]
    assert got["env"] == {**base["env"], "LOCAL_ANALYZER_MAX_OUTPUT": "8192"}
    assert got["serve"] == base["serve"]
    assert set(got) == {"note", "env", "serve"}


def test_cap8k_notebook_differs_from_pf_solved_only_by_the_cap():
    nb, base = _built("pf-solved-cap8k-10x"), _built("pf-solved-10x")
    env_cell = _env_cell_index(nb)
    for i, (cell, base_cell) in enumerate(zip(nb["cells"], base["cells"])):
        if i != env_cell:
            assert cell == base_cell, f"cell {i} differs"
    lines = set(bn._cell_text(nb["cells"][env_cell]).splitlines())
    base_lines = set(bn._cell_text(base["cells"][env_cell]).splitlines())
    assert lines - base_lines == {"setup_env['LOCAL_ANALYZER_MAX_OUTPUT'] = '8192'",
                                  "setup_env['TAAF_ARM'] = 'pf-solved-cap8k-10x'"}
    assert base_lines - lines == {"setup_env['TAAF_ARM'] = 'pf-solved-10x'"}
