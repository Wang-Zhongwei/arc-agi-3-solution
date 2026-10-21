"""arm/mem-unified through the real game loop.

tests/anim_loop_sim.py (the real solver session, ToolAgent.analyze(), the
python sandbox; a fake SGLang that renders every request through the Qwen3.8
chat template) with the object and rule scene (ANIM_SIM_OBJECTS=1: a yellow
pixel and two sky-blue squares that a level-2 click moves until the squares
touch; replies stating confirmed rules on both levels).

Checks:
- flags off (unset, or ARC3_MEM_RULES / ARC3_MEM_MOVED / ARC3_TOUCHING_SPLIT
  set to 0) every request is byte-identical to arm/pf-keep-v2 @ d624b0f9
  under pf-keep-10x, pf-keep-v2-all-10x and mem-unified-10x's base env
  (tests/fixtures/pf_keep_v2_golden.json, tests/mem_unified_golden.py);
- mem-unified-10x and mem-unified-core-10x are that base env plus their flags;
- with the arms' flags on, in a 30k, 45k and 98k window: the same game is
  played; the system prompt and tools are static and the base env's; every
  request extends the previous one byte for byte except at a trim (new text
  only in the newest message, the cleared level's entry frozen in its
  `<cleared_level>` block); each fact reaches the model where it happens.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mem_unified_golden import BASE_ENV, CONFIGS as GOLDEN_CONFIGS, SCRIPT
from memory_v2_golden import fingerprint
from test_anim_loop import TIGHT, check_static_system, run_sim, texts

HERE = Path(__file__).resolve().parent
ARMS = json.loads((HERE.parents[1] / "kaggle" / "arms.json").read_text())["arms"]
GOLDEN = json.loads((HERE / "fixtures" / "pf_keep_v2_golden.json").read_text())
NEW_FLAGS = ("ARC3_MEM_RULES", "ARC3_MEM_MOVED", "ARC3_TOUCHING_SPLIT")
FLAGS_OFF = {flag: "0" for flag in NEW_FLAGS}
WINDOWS = {"tight": TIGHT, "mid": {"LOCAL_ANALYZER_CONTEXT_WINDOW": "45000"}, "wide": {}}
ON = {f"{arm}-{window}": {**ARMS[arm]["env"], **SCRIPT, **extra}
      for arm in ("mem-unified-10x", "mem-unified-core-10x") for window, extra in WINDOWS.items()}
OFF = {**{name: env for name, env in GOLDEN_CONFIGS.items()},
       **{f"{name}-flags-0": {**env, **FLAGS_OFF} for name, env in GOLDEN_CONFIGS.items()}}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mem-unified")
    return {name: run_sim(tmp, name, env) for name, env in {**OFF, **ON}.items()}


def test_arms_are_the_base_env_plus_their_flags():
    full, core = ARMS["mem-unified-10x"], ARMS["mem-unified-core-10x"]
    assert full["env"] == {**BASE_ENV, "ARC3_MEM_RULES": "1", "ARC3_MEM_MOVED": "1", "ARC3_TOUCHING_SPLIT": "1"}
    assert core["env"] == {**BASE_ENV, "ARC3_MEM_RULES": "1"}
    assert full["serve"] == core["serve"] == ARMS["pf-keep-10x"]["serve"]


@pytest.mark.parametrize("name", sorted(OFF))
def test_flags_off_requests_are_pf_keep_v2s(runs, name):
    want = GOLDEN["configs"][name.removesuffix("-flags-0")]
    got = fingerprint(runs[name])
    for key in ("system", "tools", "evictions", "levels_completed"):
        assert got[key] == want[key], key
    assert len(got["requests"]) == len(want["requests"])
    mismatched = [i for i, (a, b) in enumerate(zip(got["requests"], want["requests"])) if a != b]
    assert not mismatched, f"first differing request: {mismatched[0]}"


def _text(message) -> str:
    return "\n".join(texts(message))


def _system(run) -> str:
    return "".join(texts(run["requests"][0]["messages"][0]))


@pytest.mark.parametrize("name", sorted(ON))
def test_same_game_static_system_and_tools(runs, name):
    run, base = runs[name], runs[f"base-{name.rsplit('-', 1)[1]}"]
    assert run["levels_completed"] == base["levels_completed"] == 1
    assert run["engine_actions"] == base["engine_actions"]
    assert check_static_system(run) == []
    assert _system(run) == _system(base)
    assert run["requests"][0]["tools"] == base["requests"][0]["tools"]


@pytest.mark.parametrize("name", sorted(ON))
def test_each_request_extends_the_previous_between_trims(runs, name):
    reqs = runs[name]["requests"]
    for i in range(1, len(reqs)):
        a, b = reqs[i - 1], reqs[i]
        if b["evictions"] != a["evictions"]:
            continue
        assert b["prompt"].startswith(a["prompt"]), f"request {i}: prompt does not extend request {i - 1}"
        assert b["messages"][:len(a["messages"])] == a["messages"], f"request {i}: an earlier message changed"


RULE_L1 = '"So DOWN from the start row always ends the attempt in GAME OVER, confirmed."'
RULE_L1_NEWER = '"Confirmed: RIGHT always stretches the block two cells right, then it settles there ✓."'
RULE_L1_OLDER = "Confirmed: RIGHT always stretches the block two cells to the right, then it settles"
RULE_L2 = '"Verified: a click on row 19 moves the yellow pixel three cells right ✓."'
MOVED = ("Moved since this level started, not by an arrow key: the yellow pixel at (16,14) is now at (16,17). "
         "It moved at action 29, MOUSE(row=19, col=12).")
TOUCHING = ("Same-colour objects that touch, listed separately: sky blue S x2 — a 2x2 at rows 16-17, cols 5-6 "
            "and a 2x2 at rows 16-17, cols 7-8, touching (segmentation shows them as one 4x2 node). Sky blue pixel "
            "total: 8, the same as at the level start (two 2x2 squares).")


def _newest(run) -> list[str]:
    return [_text(r["messages"][-1]) if r["messages"][-1]["role"] == "user" else "" for r in run["requests"]]


def _all_user(run) -> str:
    return "\n".join(_text(m) for r in run["requests"] for m in r["messages"] if m["role"] == "user")


@pytest.mark.parametrize("name", sorted(ON))
def test_facts_reach_the_model(runs, name):
    run, full = runs[name], name.startswith("mem-unified-10x")
    newest = _newest(run)
    every = _all_user(run)
    # the level-1 rules: in <this_level> while on level 1, then frozen in the entry
    this_level = [t for t in newest if "Rules you confirmed on this level: " + RULE_L1 in t]
    assert this_level and RULE_L1_NEWER in every and RULE_L1_OLDER not in every.split("Level 2 so far")[-1]
    entry = re.findall(r"  Rules you confirmed: (.*)\n", every)
    assert entry and all(e == f"{RULE_L1} | {RULE_L1_NEWER}" for e in entry)
    assert any("Rules you confirmed on this level: " + RULE_L2 in t for t in newest)
    # the question, the "Hmm" sentence and the recap reasoning add nothing
    assert "Does SPACE" not in every and "the block must move with UP" not in every
    moved = [t for t in newest if MOVED in t]
    touching = [t for t in newest if TOUCHING in t]
    if full:
        assert moved and all(t.index(MOVED) > t.index("<this_level>") for t in moved)
        assert touching and all(t.index(TOUCHING) < t.index("Current grid image:") for t in touching)
    else:
        assert not moved and not touching and "Moved since" not in every and "Same-colour objects" not in every


def test_mid_window_stages_the_rules_in_the_cleared_level_block(runs):
    for arm in ("mem-unified-10x", "mem-unified-core-10x"):
        blocks = [b for t in _newest(runs[f"{arm}-mid"])
                  for b in re.findall(r"<cleared_level>\n(.*?)\n</cleared_level>", t, re.S)]
        assert len(blocks) == 1 and f"  Rules you confirmed: {RULE_L1} | {RULE_L1_NEWER}" in blocks[0]
