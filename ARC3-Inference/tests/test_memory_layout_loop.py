"""ARC3_MEM_EVERY_CALL, ARC3_MEM_LEGIBLE and ARC3_LEVEL_COMPACT through the real game loop.

Each run is tests/anim_loop_sim.py (the real solver session, ToolAgent.analyze(),
the python sandbox and the priority gate; a fake SGLang that renders every
request through the Qwen3.8 chat template) with an arm's env from
kaggle/arms.json, a 30k window so history is trimmed, a model RESET, memory
tags in three replies and the model's recap of level 1 in its first reasoning
on level 2 (ANIM_SIM_DEBRIEF).

Checks (the user's layout, 2026-10-10):
- the new knobs change nothing while ARC3_MEM_EVERY_CALL is off or has no
  ARC3_MEM_SOLVED to stand on (flags off stay main's bytes: see
  test_memory_main_loop.py's identity test, which this branch keeps);
- every request extends the previous one except where history was evicted
  or `<previous_levels>` was redrawn, and the redraw happens only at a level
  commit or a trim;
- `<previous_levels>` is its own message right after the system prompt, in
  every request from the commit on, and never next to a stale state line;
  no `<memory>` card is glued to an old turn message;
- every full turn message ends with `<this_level>`, under its cap, about the
  current level only (no cleared level's notes in it);
- the blocks are legible: no compressed codes;
- ARC3_LEVEL_COMPACT: the commit and the drop are one prefix break, the
  cleared level keeps at most ARC3_LEVEL_COMPACT_KEEP turns, and no trim
  follows within the next few requests (no double trim).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from anim_loop_sim import DEBRIEF_RECAP, MEMORY_CODE_NOTE
from test_anim_loop import TIGHT, check_static_system, comparable, run_sim, texts

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARMS = json.loads((REPO / "kaggle" / "arms.json").read_text())["arms"]
ARM = "mem-layout-10x"
LOOP = {**TIGHT, "ANIM_SIM_RESET": "1", "ANIM_SIM_MEMORY_TAGS": "1", "ANIM_SIM_DEBRIEF": "1"}
NEW_KNOBS = {"ARC3_MEM_LEGIBLE": "1", "ARC3_LEVEL_COMPACT": "1", "ARC3_LEVEL_COMPACT_KEEP": "1",
             "ARC3_MEM_HEAD_CHARS": "900", "ARC3_MEM_LEVEL_CHARS": "400"}
COMBO = {**ARMS["user-combo-10x"]["env"], **LOOP}
FIXED = {**ARMS["mem-v2-fixed-10x"]["env"], **LOOP}
LAYOUT = {**ARMS[ARM]["env"], **LOOP}

CONFIGS = {
    "combo-off": COMBO,
    # the new knobs with EVERY_CALL off: inert on user-combo-10x
    "combo-knobs": {**COMBO, **NEW_KNOBS},
    # EVERY_CALL without ARC3_MEM_SOLVED: inert
    "combo-every-call": {**COMBO, "ARC3_MEM_EVERY_CALL": "1", "ARC3_LEVEL_COMPACT": "1"},
    "fixed": FIXED,
    # ARC3_LEVEL_COMPACT without EVERY_CALL: inert on the fixed arm
    "fixed-compact": {**FIXED, "ARC3_LEVEL_COMPACT": "1"},
    "layout": LAYOUT,
    "layout-no-compact": {k: v for k, v in LAYOUT.items() if k != "ARC3_LEVEL_COMPACT"},
    "layout-compact": {**LAYOUT, "ARC3_MEM_LEGIBLE": "0"},
}
STATE_RE = re.compile(r"Current state: step \d+, level (\d+)")
CODES_RE = re.compile(r"\b\d+ acts\b|\[[UDLRS](?: |\d|\])|\| game |\bM\(\d+,\d+\)|MOUSE@|: first (?:moved|changed)")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("memlayout")
    return {name: run_sim(tmp, name, env) for name, env in CONFIGS.items()}


def _head(request) -> str:
    messages = request["messages"]
    if len(messages) > 1 and messages[1]["role"] == "user":
        text = "".join(texts(messages[1]))
        if text.startswith("<previous_levels>"):
            return text
    return ""


def _breaks(run) -> list[int]:
    reqs = run["requests"]
    return [i for i in range(1, len(reqs)) if not reqs[i]["prompt"].startswith(reqs[i - 1]["prompt"])]


def _full_openers(run) -> list[tuple[int, str]]:
    """(request index, text) of the turn message of each analyze() call that
    is not a resumption (a resumption re-sends no `<this_level>`)."""
    out = []
    for call in run["invocations"]:
        if call["resumed"] or not call["requests"]:
            continue
        i = call["first_request"]
        last = run["requests"][i]["messages"][-1]
        assert last["role"] == "user"
        out.append((i, "\n".join(texts(last))))
    return out


# ---------------------------------------------------------------- inert knobs


@pytest.mark.parametrize("name", ["combo-knobs", "combo-every-call"])
def test_new_knobs_are_inert_without_the_layout(runs, name):
    assert comparable(runs[name]) == comparable(runs["combo-off"])


def test_compact_alone_is_inert_on_the_fixed_arm(runs):
    assert comparable(runs["fixed-compact"]) == comparable(runs["fixed"])


# ---------------------------------------------------------------- the layout


@pytest.mark.parametrize("name", ["layout", "layout-no-compact", "layout-compact"])
def test_game_plays_through_and_system_is_static(runs, name):
    run = runs[name]
    assert run["replies_used"] >= run["script_len"]
    assert run["levels_completed"] == 1
    assert check_static_system(run) == []
    system = "".join(texts(run["requests"][0]["messages"][0]))
    assert "`<previous_levels>` message right after these instructions" in system
    assert "ends with a `<this_level>` block" in system


@pytest.mark.parametrize("name", ["layout", "layout-no-compact", "layout-compact"])
def test_every_request_extends_except_at_trims_and_commits(runs, name):
    run = runs[name]
    reqs = run["requests"]
    for i in _breaks(run):
        a, b = reqs[i - 1], reqs[i]
        assert b["evictions"] != a["evictions"] or _head(a) != _head(b), f"request {i}: unexplained prefix break"
        if b["evictions"] == a["evictions"]:
            # a head redraw with nothing evicted: only the head changed
            assert b["messages"][2:len(a["messages"]) + 1] == a["messages"][1:], i
    heads = [_head(r) for r in reqs]
    changes = [i for i in range(1, len(heads)) if heads[i] != heads[i - 1]]
    # one level is cleared in the sim: <previous_levels> appears once and is
    # redrawn only where history was evicted anyway
    assert len(changes) >= 1
    for i in changes[1:]:
        assert reqs[i]["evictions"] != reqs[i - 1]["evictions"], f"request {i}: head redrawn without a trim"


def test_compact_costs_one_break_per_clear_and_no_double_trim(runs):
    run = runs["layout"]
    reqs = run["requests"]
    heads = [_head(r) for r in reqs]
    commit = next(i for i, h in enumerate(heads) if h)
    assert reqs[commit]["evictions"] > reqs[commit - 1]["evictions"], "the commit should drop level 1's turns"
    # after the commit + drop + drain, the next few requests extend the prefix
    for i in range(commit + 1, min(commit + 6, len(reqs))):
        assert reqs[i]["prompt"].startswith(reqs[i - 1]["prompt"]), f"request {i} broke right after the clear"
    # the cleared level keeps at most ARC3_LEVEL_COMPACT_KEEP (2) turns
    levels = [int(m.group(1)) for message in reqs[commit]["messages"]
              for m in [STATE_RE.search("".join(texts(message)))] if m and message["role"] == "user"]
    assert sum(1 for level in levels if level == 1) <= 2
    # fewer breaks than without compaction is not guaranteed in a 30k window,
    # but compaction never adds one: no break closer than 3 requests to another
    breaks = _breaks(run)
    assert all(b - a >= 3 for a, b in zip(breaks, breaks[1:])), breaks


@pytest.mark.parametrize("name", ["layout", "layout-no-compact", "layout-compact"])
def test_previous_levels_is_its_own_message_and_never_next_to_state(runs, name):
    reqs = runs[name]["requests"]
    commit = next(i for i, r in enumerate(reqs) if _head(r))
    for request in reqs[commit:]:
        head = _head(request)
        assert head and head.endswith("</previous_levels>")
        assert "Current state:" not in head
        for message in request["messages"][2:]:
            text = "\n".join(texts(message))
            assert "<previous_levels>" not in text
            # no trim-only card glued to an old turn message
            assert not text.startswith("<memory>\n")
    for request in reqs[:commit]:
        assert not any("<previous_levels>" in "\n".join(texts(m)) for m in request["messages"][1:])


@pytest.mark.parametrize("name", ["layout", "layout-no-compact"])
def test_previous_levels_holds_the_recap_and_the_notes(runs, name):
    head = next(h for h in map(_head, runs[name]["requests"]) if h)
    assert head.startswith("<previous_levels>\nLevel 1: cleared in ")
    assert f"Your recap: {DEBRIEF_RECAP}" in head
    assert "World model: The 2x2 block of colour 9 is the player" in head
    assert "Refuted: SPACE opens the door on contact" in head
    assert f"Level note: {MEMORY_CODE_NOTE}" in head


@pytest.mark.parametrize("name", ["layout", "layout-no-compact", "layout-compact"])
def test_every_turn_message_ends_with_this_level(runs, name):
    openers = _full_openers(runs[name])
    assert len(openers) >= 8
    for index, text in openers:
        level = int(STATE_RE.search(text).group(1))
        tail = text[text.rfind("<this_level>"):]
        assert text.rstrip().endswith("</this_level>"), index
        assert tail.startswith(f"<this_level>\nLevel {level} so far:"), index
        assert len(tail) <= 1100
    # the notes of level 1 are shown on level 1 and not once it is cleared
    level1 = [t for _, t in openers if int(STATE_RE.search(t).group(1)) == 1]
    level2 = [t for _, t in openers if int(STATE_RE.search(t).group(1)) == 2]
    assert any("Your note (world model): The 2x2 block" in t for t in level1)
    assert not any("The 2x2 block" in t[t.rfind("<this_level>"):] for t in level2)


def test_legible_blocks_have_no_codes(runs):
    run = runs["layout"]
    blocks = set()
    for request in run["requests"]:
        for message in request["messages"]:
            for text in texts(message):
                blocks.update(re.findall(r"<(previous_levels|this_level)>(.*?)</\1>", text, re.S))
    assert blocks
    for _, body in blocks:
        assert not CODES_RE.search(body), body
    # and the compact form does use them (the regex can see codes)
    compact = runs["layout-compact"]
    bodies = ["".join(texts(m)) for r in compact["requests"] for m in r["messages"]]
    assert any(CODES_RE.search(b) for b in bodies if "<previous_levels>" in b or "<this_level>" in b)


def test_this_level_is_not_resent_on_resumption(runs):
    run = runs["layout"]
    resumed = [call for call in run["invocations"] if call["resumed"] and call["requests"]]
    assert resumed
    for call in resumed:
        last = run["requests"][call["first_request"]]["messages"][-1]
        assert "<this_level>" not in "\n".join(texts(last))
