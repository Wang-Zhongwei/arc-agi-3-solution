"""ARC3_MEM_RECAP_ASK and ARC3_LEVEL_COMPACT_NEEDS_RECAP.

On mem-layout-10x (2026-10-11, 49.56) the `<previous_levels>` block reached
every request, but only 24 of 106 cleared levels carried the model's recap:
the rest showed the route alone, and ARC3_LEVEL_COMPACT dropped those levels'
turns anyway. These flags ask for the recap at the level start, read it with
`extract_recap`, and drop a cleared level's turns only once it has one.

Unit checks use reasoning text from that run; the loop checks run
tests/anim_loop_sim.py (the real game loop) with the mem-layout-10x env.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from anim_loop_sim import ASKED_RECAP_BODY, DEBRIEF_RECAP
from test_anim_loop import TIGHT, comparable, run_sim, texts
from test_memory_layout_loop import STATE_RE, _head

from inference.agent.memory_facts import extract_debrief, extract_recap
from inference.agent.memory_ledger import MemoryLedger
from inference.agent.prompts import (
    MEMORY_LAYOUT_COMPACT_LINE,
    MEMORY_LAYOUT_COMPACT_RECAP_LINE,
    MEMORY_LAYOUT_RECAP_ASK_LINE,
    MEMORY_LAYOUT_RECAP_LINE,
    level_recap_ask,
    memory_layout_addendum,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARMS = json.loads((REPO / "kaggle" / "arms.json").read_text())["arms"]

# ---------------------------------------------------------------- extract_recap
# first reasonings after a clear on mem-layout-10x (shortened)

ASKED = ("Level 3 recap: the goal is to push every box onto a socket; UP/DOWN move the green piece 3 cells "
         "and a pushed box slides until blocked. Now level 4. Let me inspect the board.")
LABELLED_AFTER_TURN = (  # tn36, level 3: the label comes after "Now level 4"
    "Level 3 solved! The plug went U R R R R U and inserted -> LEVEL COMPLETED. Now level 4. Let me record "
    "notes for level 3 and inspect level 4. Level 3 mechanics learned (worth recording): - Right keyboard: "
    "3 lanes x N columns. Ball sweeps left to right, one press per column. Level 4 board: - Board: block rows")
NEW_NUMBER = (  # s5i5, level 1, named by the new board's number
    "Level 2 solved mechanics: bars grow/shrink by 3 per click on the increment/decrement cell of the box; "
    "goal: extend the bar so its red dot lands on the cross center (plugging the socket). New level 2 board: "
    "there's a purple structure")
NO_RULE_WORD = (  # lp85, level 1: no word from the precise extractor's list
    "Level 1 solved: rotate ring so yellow block reaches the yellow bracket slot. Level 2: There appear to be "
    "multiple rings")
CELEBRATION = "Level 6 solved! Now level 7. Let me inspect the board."
COUNTS = "Level 3 solved in 20 actions. Now level 4. Level 4 layout: - Codebook: 8 pairs"
HARNESS = ("Level 4 solved?? No wait - level 3 completed (level_completed) and now we're on level 4. Wait, the "
           "block now shows all white?! Let me check.")
NO_RECAP = "Level 2: Now there's an orange X-shaped cross with white center. Let me inspect the segmentation."


def test_asked_recap_is_read_up_to_the_new_board():
    assert extract_recap(ASKED, 3) == ("the goal is to push every box onto a socket; UP/DOWN move the green "
                                       "piece 3 cells and a pushed box slides until blocked.")


def test_a_labelled_recap_after_the_turn_to_the_new_level_is_kept():
    assert extract_debrief(LABELLED_AFTER_TURN, 4, precise=True) == ""
    assert extract_recap(LABELLED_AFTER_TURN, 3) == "Right keyboard: 3 lanes x N columns. Ball sweeps left to right, one press per column."


def test_a_clear_named_by_the_new_number_is_kept():
    assert extract_debrief(NEW_NUMBER, 2, precise=True) == ""
    recap = extract_recap(NEW_NUMBER, 1)
    assert recap.startswith("mechanics: bars grow/shrink by 3 per click")
    assert "purple structure" not in recap


def test_a_recap_without_a_listed_rule_word_is_kept():
    assert extract_debrief(NO_RULE_WORD, 2, precise=True) == ""
    assert extract_recap(NO_RULE_WORD, 1) == "rotate ring so yellow block reaches the yellow bracket slot."


@pytest.mark.parametrize("text", [CELEBRATION, COUNTS, HARNESS, NO_RECAP, ""])
def test_no_recap_from_celebrations_counts_or_harness_talk(text):
    level = 6 if text is CELEBRATION else 3
    assert extract_recap(text, level) == ""


def test_the_precise_recap_is_still_found():
    text = f"{DEBRIEF_RECAP} Now level 2."
    assert extract_recap(text, 1) == extract_debrief(text, 2, precise=True) != ""


def test_recap_is_cut_at_a_sentence():
    long = "Level 2 recap: " + " ".join(f"rule {i} holds when the bar is full." for i in range(60))
    recap = extract_recap(long, 2)
    assert len(recap) <= 420 and recap.endswith(".")


# ---------------------------------------------------------------- prompts and ledger


def test_ask_names_the_cleared_level():
    text = level_recap_ask(3)
    assert "`Level 3 recap:`" in text and "goal of level 3" in text and "<previous_levels>" in text


def test_addendum_without_the_flags_is_unchanged():
    base = memory_layout_addendum(debrief=True, legible=True, compact_keep=2)
    assert MEMORY_LAYOUT_RECAP_LINE in base and MEMORY_LAYOUT_COMPACT_LINE.replace("{k}", "2") in base
    assert base == memory_layout_addendum(debrief=True, legible=True, compact_keep=2,
                                          recap_ask=False, compact_needs_recap=False)
    asked = memory_layout_addendum(debrief=True, legible=True, compact_keep=2,
                                   recap_ask=True, compact_needs_recap=True)
    assert MEMORY_LAYOUT_RECAP_ASK_LINE in asked and MEMORY_LAYOUT_RECAP_LINE not in asked
    assert MEMORY_LAYOUT_COMPACT_RECAP_LINE.replace("{k}", "2") in asked


def _ledger(recap_ask: bool) -> MemoryLedger:
    ledger = MemoryLedger(card_mode=True, solved_mode=True, debrief_mode=True, recap_ask=recap_ask)
    ledger.level = 1
    ledger.on_level_change(2)
    return ledger


def test_ledger_reads_the_recap_with_extract_recap_under_the_flag():
    off, on = _ledger(False), _ledger(True)
    for ledger in (off, on):
        ledger.note_think(NO_RULE_WORD)
    assert not off.layout_has_recap(1)
    assert on.layout_has_recap(1)
    assert on._debriefs[1] == "rotate ring so yellow block reaches the yellow bracket slot."


# ---------------------------------------------------------------- the game loop

LOOP = {**TIGHT, "ANIM_SIM_RESET": "1", "ANIM_SIM_MEMORY_TAGS": "1"}
COMBO = {**ARMS["user-combo-10x"]["env"], **LOOP}
LAYOUT = {**ARMS["mem-layout-10x"]["env"], **LOOP}
# the arm's own 98k window: no trim in the sim, so an eviction is a compaction
WIDE = {**ARMS["mem-layout-10x"]["env"], "ANIM_SIM_RESET": "1", "ANIM_SIM_MEMORY_TAGS": "1"}
FLAGS = {"ARC3_MEM_RECAP_ASK": "1", "ARC3_LEVEL_COMPACT_NEEDS_RECAP": "1"}
CONFIGS = {
    "combo-off": COMBO,
    "combo-flags": {**COMBO, **FLAGS},
    "layout-debrief": {**LAYOUT, "ANIM_SIM_DEBRIEF": "1"},
    "layout-debrief-flags-off": {**LAYOUT, "ANIM_SIM_DEBRIEF": "1", "ARC3_MEM_RECAP_ASK": "0",
                                 "ARC3_LEVEL_COMPACT_NEEDS_RECAP": "0"},
    "asked": {**LAYOUT, **FLAGS, "ANIM_SIM_RECAP_ASKED": "1"},
    "silent": {**LAYOUT, **FLAGS},
    "silent-ungated": {**LAYOUT, "ARC3_MEM_RECAP_ASK": "1"},
    "wide-asked": {**WIDE, **FLAGS, "ANIM_SIM_RECAP_ASKED": "1"},
    "wide-silent": {**WIDE, **FLAGS},
    "wide-silent-ungated": {**WIDE, "ARC3_MEM_RECAP_ASK": "1"},
    "keep-asked": {**ARMS["mem-recap-keep-10x"]["env"], "ANIM_SIM_RESET": "1", "ANIM_SIM_MEMORY_TAGS": "1",
                   "ANIM_SIM_RECAP_ASKED": "1"},
}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("recapask")
    return {name: run_sim(tmp, name, env) for name, env in CONFIGS.items()}


def _level2_opener(run) -> str:
    for request in run["requests"]:
        last = request["messages"][-1]
        text = "\n".join(texts(last))
        if last["role"] == "user" and "Current state: step" in text and STATE_RE.search(text).group(1) == "2":
            return text
    raise AssertionError("no level-2 turn message")


def _level1_messages(request) -> int:
    """User messages whose state line is on level 1."""
    found = [STATE_RE.search("".join(texts(m))) for m in request["messages"] if m["role"] == "user"]
    return sum(1 for m in found if m and m.group(1) == "1")


def _commit(run) -> int:
    return next(i for i, r in enumerate(run["requests"]) if _head(r))


def test_flags_are_inert_without_the_layout(runs):
    assert comparable(runs["combo-flags"]) == comparable(runs["combo-off"])


def test_flags_off_leave_the_layout_arm_unchanged(runs):
    assert comparable(runs["layout-debrief-flags-off"]) == comparable(runs["layout-debrief"])


@pytest.mark.parametrize("name", ["asked", "silent", "silent-ungated"])
def test_the_level_start_message_asks_for_the_recap(runs, name):
    run = runs[name]
    assert run["levels_completed"] == 1
    opener = _level2_opener(run)
    assert level_recap_ask(1) in opener
    assert opener.index("You have completed the previous level") < opener.index("`Level 1 recap:`")
    system = "".join(texts(run["requests"][0]["messages"][0]))
    assert MEMORY_LAYOUT_RECAP_ASK_LINE.strip() in system
    # asked once: no other message carries it
    asks = sum("`Level 1 recap:`" in "\n".join(texts(m)) for m in run["requests"][-1]["messages"][1:])
    assert asks <= 1


def test_the_asked_recap_reaches_previous_levels_and_compacts(runs):
    for name in ("asked", "wide-asked"):
        run = runs[name]
        head = _head(run["requests"][_commit(run)])
        assert f"Your recap: {ASKED_RECAP_BODY}" in head, name
    run = runs["wide-asked"]
    reqs = run["requests"]
    commit = _commit(run)
    assert all(r["evictions"] == 0 for r in reqs[:commit]), "no trim in the wide window before the clear"
    assert reqs[commit]["evictions"] > 0, "a recapped level's turns should be dropped at its commit"
    kept = _level1_messages(reqs[commit])
    silent = runs["wide-silent"]
    assert kept < _level1_messages(silent["requests"][_commit(silent)]), "fewer level-1 messages than when kept"
    system = "".join(texts(reqs[0]["messages"][0]))
    assert MEMORY_LAYOUT_COMPACT_RECAP_LINE.replace("{k}", "2").strip() in system


def test_a_level_without_a_recap_keeps_its_turns(runs):
    gated, ungated = runs["wide-silent"], runs["wide-silent-ungated"]
    for run in (gated, ungated):
        assert "Your recap:" not in _head(run["requests"][_commit(run)])
    # without the gate the clear compacts; with it nothing is evicted
    assert ungated["requests"][_commit(ungated)]["evictions"] > 0
    assert all(r["evictions"] == 0 for r in gated["requests"])
    commit = _commit(gated)
    assert _level1_messages(gated["requests"][commit]) == _level1_messages(gated["requests"][commit - 1])


def test_the_keep_arm_saves_the_recap_and_drops_no_turns(runs):
    run = runs["keep-asked"]
    reqs = run["requests"]
    commit = _commit(run)
    assert f"Your recap: {ASKED_RECAP_BODY}" in _head(reqs[commit])
    assert all(r["evictions"] == 0 for r in reqs), "compaction is off: a clear drops nothing"
    assert _level1_messages(reqs[commit]) == _level1_messages(reqs[commit - 1])
    system = "".join(texts(reqs[0]["messages"][0]))
    assert MEMORY_LAYOUT_RECAP_ASK_LINE.strip() in system
    assert "its turns are dropped from this conversation" not in system
