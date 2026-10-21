"""memory-v2 and level-path on main's user-combo-10x stack, through the real game loop.

Each run is tests/anim_loop_sim.py (the real solver session, ToolAgent.analyze(),
the python sandbox and the priority gate; a fake SGLang that renders every
request through the model's chat template) under the Kaggle template's
setup_env plus an arm's env read from kaggle/arms.json. The window is cut to
30k (`TIGHT`) so history is trimmed several times, the reply script adds a
model RESET (EXPOSE_RESET=on is in the stack), and three replies that do not
act write memory tags (ANIM_SIM_MEMORY_TAGS: a world-model fact, a hypothesis
drafted one way in the reasoning and stated the other way in the reply, and a
bare tag in python code).

Checks:
- with every memory flag off, user-combo-10x's requests are byte-identical to
  main's (MEMORY_MAIN_BASELINE_REF, default the commit this branch left main
  at, or MEMORY_MAIN_BASELINE_ROOT=<another checkout's ARC3-Inference>), and
  so are the memory knobs that do nothing without a memory flag;
- with each memory arm on, every rendered prompt extends the previous one
  unless history was evicted in between, the system prompt and tools never
  change, the game still plays through, and the card, delta and route text
  reach the model; the tag in code is recorded and stripped instead of
  failing the call (ARC3_MEM_CODE_TAGS), and with the card on the reply's
  hypothesis wins over the reasoning's draft (ARC3_MEM_REPLY_WINS).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from anim_loop_sim import MEMORY_CODE_NOTE
from test_anim_loop import TIGHT, check_extends, check_static_system, comparable, run_sim, texts

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
# main when arm/memory-main branched from it
BASE_REF = os.environ.get("MEMORY_MAIN_BASELINE_REF", "7f8195bd")
ARMS = json.loads((REPO / "kaggle" / "arms.json").read_text())["arms"]
MEMORY_ARMS = ("mem-solved-10x", "mem-v2-10x", "level-path-10x", "mem-v2-fixed-10x")
# the audit's corrected arm: placement trim, so no delta lines; recap in the sim
FIXED = "mem-v2-fixed-10x"
LOOP = {**TIGHT, "ANIM_SIM_RESET": "1", "ANIM_SIM_MEMORY_TAGS": "1"}
# memory knobs that change nothing while every memory flag is off
INERT_KNOBS = {
    "ARC3_MEMORY_PLACEMENT": "both", "ARC3_MEM_STATE_BORDER": "2", "ARC3_MEM_CARD_CHARS": "1200",
    "ARC3_MEM_DELTA_CHARS": "200", "ARC3_MEM_DELTA_GAME_CHARS": "500", "ARC3_MEM_SEEN_GAME_CHARS": "100",
    "ARC3_MEM_LINES_AFTER_TOKENS": "0", "ARC3_MEM_PATH_GAME_CHARS": "100", "ARC3_MEM_PATH_AFTER_TOKENS": "0",
}


def _env(arm: str, **extra: str) -> dict[str, str]:
    # the fixed arm's run also has the model recap level 1 after the clear
    debrief = {"ANIM_SIM_DEBRIEF": "1"} if arm == FIXED else {}
    return {**ARMS[arm]["env"], **LOOP, **debrief, **extra}


CONFIGS = {
    "combo-off": _env("user-combo-10x"),
    "combo-knobs-off": _env("user-combo-10x", **INERT_KNOBS),
    "combo-wide-off": {**ARMS["user-combo-10x"]["env"], "ANIM_SIM_RESET": "1", "ANIM_SIM_MEMORY_TAGS": "1"},
    **{arm: _env(arm) for arm in MEMORY_ARMS},
}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("memloop")
    return {name: run_sim(tmp, name, extra) for name, extra in CONFIGS.items()}


@pytest.fixture(scope="module")
def base_root(tmp_path_factory) -> Path:
    explicit = os.environ.get("MEMORY_MAIN_BASELINE_ROOT", "")
    if explicit:
        return Path(explicit)
    out = tmp_path_factory.mktemp("main")
    try:
        archive = subprocess.run(["git", "-C", str(REPO), "archive", BASE_REF, "ARC3-Inference/inference"],
                                 check=True, capture_output=True, timeout=120).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"no checkout of {BASE_REF} ({exc}); set MEMORY_MAIN_BASELINE_ROOT")
    subprocess.run(["tar", "-x", "-C", str(out)], input=archive, check=True)
    return out / "ARC3-Inference"


def _user_texts(run) -> list[str]:
    out, seen = [], set()
    for request in run["requests"]:
        for message in request["messages"]:
            if message["role"] != "user":
                continue
            for text in texts(message):
                if text not in seen:
                    seen.add(text)
                    out.append(text)
    return out


# ---------------------------------------------------------------- flags off: main's bytes

@pytest.mark.parametrize("name", ["combo-off", "combo-wide-off"])
def test_flags_off_matches_main(runs, tmp_path, base_root, name):
    base = run_sim(tmp_path, f"main-{name}", CONFIGS[name], root=base_root)
    assert comparable(base) == comparable(runs[name])
    assert base["engine_actions"] == runs[name]["engine_actions"]


def test_inert_knobs_change_nothing(runs):
    assert comparable(runs["combo-knobs-off"]) == comparable(runs["combo-off"])


def _cards(run) -> list[str]:
    return [t for t in _user_texts(run) if t.startswith("<memory>\n")]


def _code_tag_failed(run) -> bool:
    """Whether the python call carrying the bare tag came back as a SyntaxError."""
    for request in run["requests"]:
        messages = request["messages"]
        for i, message in enumerate(messages[:-1]):
            calls = message.get("tool_calls") or []
            if any(MEMORY_CODE_NOTE in c["function"]["arguments"] for c in calls):
                result = "".join(texts(messages[i + 1]))
                return "SyntaxError" in result or "invalid syntax" in result
    raise AssertionError("the reply with the tag in its code was never sent back")


def test_the_tight_window_trims_and_the_ledger_shows(runs):
    # a flags-off golden that never trims or never shows the ledger would prove little
    run = runs["combo-off"]
    reqs = run["requests"]
    assert sum(1 for a, b in zip(reqs, reqs[1:]) if b["evictions"] != a["evictions"]) >= 2
    cards = _cards(run)
    assert any("The 2x2 block of colour 9 is the player" in card for card in cards)
    # main: the tag in code is a SyntaxError and never reaches the ledger
    assert _code_tag_failed(run)
    assert not any(MEMORY_CODE_NOTE in card for card in cards)


# ---------------------------------------------------------------- flags on

@pytest.mark.parametrize("name", MEMORY_ARMS)
def test_prompts_extend_between_trims(runs, name):
    run = runs[name]
    assert check_extends(run) == []
    assert check_static_system(run) == []
    reqs = run["requests"]
    assert sum(1 for a, b in zip(reqs, reqs[1:]) if b["evictions"] != a["evictions"]) >= 2, "no trims"


@pytest.mark.parametrize("name", MEMORY_ARMS)
def test_game_plays_through(runs, name):
    run = runs[name]
    assert run["replies_used"] >= run["script_len"]
    assert run["levels_completed"] == 1
    engine = run["engine_actions"]
    assert engine.count("RESET") == 2  # the auto-RESET after the death and the model's own
    assert sum(1 for x in run["invocations"] if x["resumed"]) >= 2


@pytest.mark.parametrize("name", MEMORY_ARMS)
def test_memory_reaches_the_model(runs, name):
    texts_ = _user_texts(runs[name])
    cards = _cards(runs[name])
    assert cards, "no head card after a trim"
    assert any("<solved>L1:" in card for card in cards), "the solved line never reached a card"
    # placement trim (the fixed arm) sends no delta lines
    assert any("+ solved L1:" in t for t in texts_) == (name != FIXED)
    system = runs[name]["requests"][0]["messages"][0]
    system_text = "".join(texts(system))
    assert "`<solved>` lines are written by the harness" in system_text
    has_route = any("Shortest route on record from this level's start board" in t for t in texts_)
    has_actions = any('<actions level="' in card for card in cards)
    has_hazard = any("<hazard>L1: attempt 1 ended in GAME OVER" in card for card in cards)
    if name == "mem-solved-10x":
        assert not has_route and not has_actions and not has_hazard
    else:
        assert has_actions and has_hazard
        assert has_route == (name == "level-path-10x")


@pytest.mark.parametrize("name", MEMORY_ARMS)
def test_tags_in_code_are_kept_and_the_call_runs(runs, name):
    assert not _code_tag_failed(runs[name])
    assert any(MEMORY_CODE_NOTE in card for card in _cards(runs[name]))


@pytest.mark.parametrize("name", ["mem-v2-10x", "level-path-10x", FIXED])
def test_the_reply_wins_over_the_reasoning_draft(runs, name):
    # card mode keeps only the newer of two overlapping entries; with
    # ARC3_MEM_REPLY_WINS the reply is the newer one (ARC3_MEM_DEBRIEF labels
    # each entry with its level)
    cards = _cards(runs[name])
    label = "L1: " if name == FIXED else ""
    assert any(f'status="refuted">{label}SPACE opens the door on contact<' in card for card in cards)
    assert not any('SPACE opens the door</hypothesis>' in card for card in cards)


def test_memory_arms_share_the_stack(runs):
    # the three arms differ from user-combo-10x only in memory flags
    combo = ARMS["user-combo-10x"]
    for arm in MEMORY_ARMS:
        extra = {k: v for k, v in ARMS[arm]["env"].items() if k not in combo["env"]}
        assert {k: v for k, v in ARMS[arm]["env"].items() if k in combo["env"]} == combo["env"]
        assert all(k.startswith(("ARC3_MEM_", "ARC3_MEMORY_")) for k in extra), extra
        assert extra["ARC3_MEM_REPLY_WINS"] == "1"
        assert ARMS[arm]["serve"] == combo["serve"]
        assert "demo_games" not in ARMS[arm] and "demo_minutes" not in ARMS[arm]  # all games, 121 min


def test_fixed_arm_carries_the_recap_and_no_how(runs):
    from anim_loop_sim import DEBRIEF_RECAP
    from inference.agent.prompts import MEMORY_DEBRIEF_LINE, MEMORY_NOTES_SCOPE_LINE

    run = runs[FIXED]
    cards = _cards(run)
    assert any(f"| debrief: {DEBRIEF_RECAP}</solved>" in card for card in cards), cards
    assert not any("| how:" in card for card in cards)
    system = "".join(texts(run["requests"][0]["messages"][0]))
    assert MEMORY_DEBRIEF_LINE in system and MEMORY_NOTES_SCOPE_LINE not in system
    # the model's notes sit before the harness's action rows on every card
    for card in cards:
        if "<actions" in card and "<world_model>" in card:
            assert card.index("<world_model>") < card.index("<actions")
    # no route lines and no delta lines in the fixed arm
    assert not any("Shortest route on record" in t or "<memory_update>" in t for t in _user_texts(run))


def test_only_the_fixed_arm_writes_a_debrief(runs):
    for name in ("mem-solved-10x", "mem-v2-10x", "level-path-10x"):
        assert not any("| debrief:" in card for card in _cards(runs[name]))
