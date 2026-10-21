"""pf-keep-v2-all-10x is pf-keep-10x plus every memory-v2 and legibility factor, nothing else.

The user asked for the memory-v2 and legibility arms to be tested in one
combined run. Its env must be pf-keep-10x's plus exactly the flags of the four
single-factor arms it combines (pf-keep-append-10x, pf-keep-memv2-10x,
pf-keep-stuck-10x, pf-keep-legible-10x), with pf-keep-10x's serving, so each
factor keeps its own flag for bisection. Its notebook is built with
kaggle/build_notebook.py (dry: nothing is pushed) next to pf-keep-10x's: every
cell must match except the env overrides, which differ by exactly those flags
and the arm label. The built env cell is then applied to a fresh interpreter
that constructs the ToolAgent the way the notebook's game cell does: its
system prompt must be pf-keep-10x's with the append-commit row and the
legibility bundle's rows applied, and its tool schema must be pf-keep-10x's.

Run from ARC3-Inference:  python -m pytest -q ../kaggle/tests/test_pf_keep_v2_arm.py
"""
import json
import subprocess
import sys
from pathlib import Path

from test_prompt_facts_arm import _built, _env_cell_index, _notebook_env
from test_throughput_arms import bn

REPO = Path(__file__).resolve().parents[2]
ARMS = json.loads((REPO / "kaggle" / "arms.json").read_text(encoding="utf-8"))["arms"]
ARM, BASE = "pf-keep-v2-all-10x", "pf-keep-10x"
PARTS = ("pf-keep-append-10x", "pf-keep-memv2-10x", "pf-keep-stuck-10x", "pf-keep-legible-10x")
# written out, so a later edit to one of the single-factor arms cannot change
# this arm's meaning unnoticed
FLAGS = {
    "ARC3_MEM_APPEND_COMMIT": "1",
    "ARC3_MEM_RECAP_CHARS": "800",
    "ARC3_MEM_COMPRESS_ROUTES_FIRST": "1",
    "ARC3_MEM_ROUTE_FORMAT": "readable",
    "ARC3_MEM_HEAD_WORDS": "1",
    "ARC3_MEM_RECAP_ASK_V2": "1",
    "ARC3_MEM_ATTEMPT_COUNTS": "1",
    "ARC3_MEM_SEEN_GAME_ACTION": "1",
    "ARC3_MEM_EFFECT_WORDS": "1",
    "ARC3_MEM_STUCK_FACT": "1",
    "ARC3_LEGIBILITY": "1",
}
# factors that exist but are not part of this run
LEFT_OUT = (
    "ARC3_MEM_TRIM_STRIP", "ARC3_MEM_APPEND_NO_HANDOVER", "ARC3_MEM_RECAP_FROZEN_FACT",
    "ARC3_DIFF_SAME_CELLS", "ARC3_DIGEST_COORD_FACT", "ARC3_EDGE_NOOP_FACT",
    "ARC3_ACTION_INDEX_FACT", "ARC3_LEGEND_CASE_PAIRS", "ARC3_FRAME_RUNS",
)


def _delta(arm: str) -> dict:
    base = ARMS[BASE]["env"]
    return {k: v for k, v in ARMS[arm]["env"].items() if base.get(k) != v}


def test_env_is_pf_keep_plus_exactly_the_combined_flags():
    got, base = ARMS[ARM], ARMS[BASE]
    assert got["env"] == {**base["env"], **FLAGS}
    assert not set(FLAGS) & set(base["env"])
    # pf-keep-10x's keys first, in their order, then the new flags
    assert list(got["env"]) == [*base["env"], *FLAGS]
    assert got["serve"] == base["serve"]
    assert set(got) == {"note", "env", "serve"}
    for flag in LEFT_OUT:
        assert flag not in got["env"], flag


def test_flags_are_the_union_of_the_single_factor_arms():
    union: dict = {}
    for arm in PARTS:
        delta = _delta(arm)
        assert ARMS[arm]["env"] == {**ARMS[BASE]["env"], **delta}, arm
        assert not set(delta) & set(union), arm  # the factors do not overlap
        union.update(delta)
    assert union == FLAGS


def test_note_names_the_base_every_flag_and_how_to_build():
    note = ARMS[ARM]["note"]
    for fact in (BASE, "55.79", "--new-batch", "one run", "bisected", *FLAGS, *PARTS):
        assert fact in note, fact
    assert note.isascii()


def test_notebook_differs_only_by_the_flags():
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
    assert {k: _notebook_env(nb)[k] for k in FLAGS} == FLAGS


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
    "append_rows": [list(r) for r in prompts.MEMORY_LAYOUT_APPEND_COMMIT_ROWS],
    "legibility_rows": {name: rows for name, rows in prompts.LEGIBILITY_SYSTEM_ROWS.items()},
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


def test_system_prompt_is_pf_keep_with_the_append_row_and_the_bundle_rows():
    got, base = _probe(ARM), _probe(BASE)
    assert got["enabled"] == got["bundle"] and base["enabled"] == []
    (old, new), = got["append_rows"]
    assert base["system"].count(old) == 1
    expected = base["system"].replace(old, new)
    for name, rows in got["legibility_rows"].items():
        if name not in got["bundle"]:
            continue
        for row in rows:
            hits = [(o, n) for o, n in row if expected.count(o) == 1]
            assert hits, (name, row[0][0][:60])
            expected = expected.replace(*hits[0])
    assert got["system"] == expected
    assert got["system"].count(new) == 1 and old not in got["system"]
    assert got["tools"] == base["tools"]


def test_system_prompt_composes_the_two_single_factor_prompts():
    """The combined prompt is the legibility arm's with the append row: the
    two system-prompt factors touch disjoint text."""
    got, legible, append = _probe(ARM), _probe("pf-keep-legible-10x"), _probe("pf-keep-append-10x")
    (old, new), = got["append_rows"]
    assert got["system"] == legible["system"].replace(old, new)
    assert legible["system"].count(old) == 1 and append["system"].count(new) == 1
