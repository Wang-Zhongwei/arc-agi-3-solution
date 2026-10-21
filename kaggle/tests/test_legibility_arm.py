"""pf-keep-legible-10x is pf-keep-10x plus ARC3_LEGIBILITY=1, nothing else.

Its notebook is built with kaggle/build_notebook.py (dry: nothing is pushed)
next to pf-keep-10x's: every cell must match except the env overrides, which
differ by exactly the umbrella flag and the arm label. The built env cell is
then applied to a fresh interpreter that constructs the ToolAgent the way the
notebook's game cell does: its system prompt must be pf-keep-10x's with the
bundle's rows applied (prompts.LEGIBILITY_SYSTEM_ROWS), and its tool schema
must be pf-keep-10x's.

Run from ARC3-Inference:  python -m pytest -q ../kaggle/tests/test_legibility_arm.py
"""
import json
import subprocess
import sys
from pathlib import Path

from test_prompt_facts_arm import _built, _env_cell_index, _notebook_env
from test_throughput_arms import bn

REPO = Path(__file__).resolve().parents[2]
ARMS = json.loads((REPO / "kaggle" / "arms.json").read_text(encoding="utf-8"))["arms"]
ARM, BASE = "pf-keep-legible-10x", "pf-keep-10x"
FLAG = {"ARC3_LEGIBILITY": "1"}


def test_arm_is_pf_keep_plus_the_umbrella():
    got, base = ARMS[ARM], ARMS[BASE]
    assert got["env"] == {**base["env"], **FLAG}
    assert got["serve"] == base["serve"]
    assert set(got) == {"note", "env", "serve"}
    for fact in (BASE, "55.79", "ARC3_SEG_BBOX", "ARC3_ANIM_SCHEMA", "ARC3_RETAINED_SOURCE",
                 "ARC3_STALE_TRUTH", "ARC3_RESET_BAR_FACT", "ARC3_TURN_BASED_FACT"):
        assert fact in got["note"], fact


def test_notebook_differs_only_by_the_umbrella():
    nb, base = _built(ARM), _built(BASE)
    assert len(nb["cells"]) == len(base["cells"])
    env_cell = _env_cell_index(nb)
    assert env_cell == _env_cell_index(base)
    for i, (cell, base_cell) in enumerate(zip(nb["cells"], base["cells"])):
        if i != env_cell:
            assert cell == base_cell, f"cell {i} differs"
    lines = set(bn._cell_text(nb["cells"][env_cell]).splitlines())
    base_lines = set(bn._cell_text(base["cells"][env_cell]).splitlines())
    assert lines - base_lines == {"setup_env['ARC3_LEGIBILITY'] = '1'", f"setup_env['TAAF_ARM'] = {ARM!r}"}
    assert base_lines - lines == {f"setup_env['TAAF_ARM'] = {BASE!r}"}


PROBE = r"""
import json
from inference.agent import legibility, prompts
from inference.agent import tool_agent as ta
agent = ta.ToolAgent(model="flashnext", base_url="http://127.0.0.1:9/v1", provider="vllm")
agent._ensure_session = lambda state_path: None
print(json.dumps({
    "system": agent._system_prompt,
    "tools": agent._tools(None),
    "enabled": sorted(legibility.enabled_flags()),
    "bundle": sorted(legibility.BUNDLE),
    "rows": {name: rows for name, rows in prompts.LEGIBILITY_SYSTEM_ROWS.items()},
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


def test_system_prompt_is_pf_keep_with_the_bundle_rows():
    got, base = _probe(ARM), _probe(BASE)
    assert got["enabled"] == got["bundle"] and base["enabled"] == []
    expected = base["system"]
    for name, rows in got["rows"].items():
        if name not in got["bundle"]:
            continue
        for row in rows:
            old, new = next((o, n) for o, n in row if expected.count(o) == 1)
            expected = expected.replace(old, new)
    assert got["system"] == expected != base["system"]
    assert got["tools"] == base["tools"]
