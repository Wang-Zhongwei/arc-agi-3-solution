"""The memory audit's fixes (2026-10-10): ARC3_MEM_DEBRIEF.

Each test pins one bug the audit of level-path-10x and mem-v2-10x found
(research/archive/reports/memory-main.md, "Audit"):

- the solved line's "how" was a sentence of the clearing turn's last plan
  (m0r0 L3: "74 acts ... | how: 5 moves: UP,UP,UP,RIGHT,RIGHT"), never the
  level's rule; the debrief takes the model's own recap after the clear;
- the card carried no rules or goal and cut the model's notes first; in
  debrief mode the model's text comes first and is cut last;
- model notes had no level label.

ARC3_MEM_KEEP_EDGE (solved paths keep edge-only actions) is tested in
test_memory_facts.py.
"""
from __future__ import annotations

import pytest

from inference.agent.memory_facts import debrief_about_level, extract_debrief
from inference.agent.memory_ledger import MemoryLedger
from inference.agent.prompts import (
    MEMORY_DEBRIEF_LINE, MEMORY_HEAD_CARD_RECORDED_LINE, MEMORY_NOTES_CLOSING_LINE,
    MEMORY_NOTES_SCOPE_LINE, memory_addendum,
)

# real reasonings from the runs (shortened), first reasoning after a clear
M0R0_L1 = ("Level 1 solved: the goal was to make the two blocks overlap/meet! Now level 2. "
           "Level 2 board: - Background: left half magenta.")
TN36_L1 = ("Level 1 solved: the tray of 5 T-glyphs is a counter — the number of black T's = how many "
           "cells the yellow plug drops when you press the blue button. Now level 2. Let me inspect the new board.")
# the clearing turn's last reasoning in m0r0 (req 166): a plan for the final batch
M0R0_CLEARING_PLAN = "So the plan is set. 5 moves: UP,UP,UP,RIGHT,RIGHT ✓. Then the blocks meet."


# ---------------------------------------------------------------- extract_debrief

def test_debrief_takes_the_recap_and_stops_at_the_new_level():
    assert extract_debrief(M0R0_L1, 2) == "Level 1 solved: the goal was to make the two blocks overlap/meet!"
    got = extract_debrief(TN36_L1, 2)
    assert got.startswith("Level 1 solved: the tray of 5 T-glyphs is a counter")
    assert "Now level 2" not in got and "inspect" not in got


def test_debrief_needs_a_goal_or_rule_word():
    # counts are already on the solved line
    assert extract_debrief("Level 3 solved in 14 actions. Now level 4. Let me inspect.", 4) == ""
    assert extract_debrief("Level 6 solved (21 actions). Now level 7. New board.", 7) == ""


def test_debrief_ignores_announcements_and_new_level_plans():
    # "let me record the key mechanics" is not a recap (cd82 L1 in level-path-10x)
    assert extract_debrief("🎉 Let me record the key mechanics in memory, then analyze the board.", 2) == ""
    # a rule about the NEW level is not a recap of the old one
    assert extract_debrief("Now level 3. So the rule: the ink must be fully contained.", 3) == ""
    assert extract_debrief("", 2) == "" and extract_debrief("Just moving on.", 2) == ""


def test_debrief_accepts_mechanics_labels_and_strips_tags():
    text = ("Key mechanics from L4: - Clicking a gate transfers the plug when both pools match. "
            "<world_model>pumps move water</world_model> Now level 5.")
    got = extract_debrief(text, 5)
    assert got.startswith("Key mechanics from L4: - Clicking a gate transfers the plug")
    assert "<world_model>" not in got and "pumps move water" not in got


def test_debrief_is_cut_at_whole_sentences():
    long = "Level 1 solved: the goal was to fill both pits. " + "Each ring slides until blocked. " * 20
    got = extract_debrief(long, 2)
    assert len(got) <= 300 and got.endswith(".")


# ---------------------------------------------------------------- the solved line

def _ledger(**kw) -> MemoryLedger:
    return MemoryLedger(card_mode=True, solved_mode=True, **kw)


def test_old_mode_still_writes_how_from_the_clearing_plan():
    # the bug, kept as it ran so recorded arms reproduce
    ledger = _ledger()
    ledger.note_think(M0R0_CLEARING_PLAN)
    line = ledger.add_solved(3, ["UP"] * 74, 81, 0)
    assert "| how: 5 moves: UP,UP,UP,RIGHT,RIGHT" in line


def test_debrief_mode_has_no_how_and_carries_the_recap():
    ledger = _ledger(debrief_mode=True)
    ledger.note_think(M0R0_CLEARING_PLAN)  # the clearing turn
    ledger.on_level_change(2)
    ledger.note_think(M0R0_L1)  # first reasoning on level 2
    line = ledger.add_solved(1, ["RIGHT", "UP"], 9, 0)
    assert "how:" not in line
    assert line.endswith("| debrief: Level 1 solved: the goal was to make the two blocks overlap/meet!")


def test_a_recap_after_the_solved_line_is_added_to_it():
    # the clear event can be processed before the next reasoning arrives
    ledger = _ledger(debrief_mode=True)
    ledger.on_level_change(2)
    first = ledger.add_solved(1, ["RIGHT"], 3, 0)
    assert "debrief" not in first
    ledger.note_think(TN36_L1)
    card = ledger.render_card()
    assert "<solved>L1: 1 acts [R], spent 3, 0 deaths | debrief: Level 1 solved: the tray" in card


def test_only_the_first_two_reasonings_after_a_clear_count():
    ledger = _ledger(debrief_mode=True)
    ledger.on_level_change(2)
    ledger.note_think("Let me look at the board.")
    ledger.note_think("Hmm, moving on.")
    ledger.note_think(M0R0_L1)  # third: too late, and may be about anything
    assert "debrief" not in ledger.add_solved(1, ["RIGHT"], 3, 0)


# ---------------------------------------------------------------- the card

def _full_card_ledger(card_chars: int) -> MemoryLedger:
    ledger = _ledger(debrief_mode=True, card_chars=card_chars)
    for level in range(1, 6):
        ledger.on_level_change(level + 1)
        ledger.note_think(f"Level {level} solved: the goal was to push block {level} into the socket.")
        ledger.add_solved(level, ["UP", "LEFT", "MOUSE(4,7)"] * 8, 30, 0)
    ledger.absorb("<world_model>clicking a ring moves control to it</world_model>")
    ledger.absorb('<hypothesis status="refuted">SPACE opens the door</hypothesis>')
    ledger.set_item("actions", "actions", "\n".join(f"KEY{i}: moved {i}, changed {i} | game moved {i}"
                                                    for i in range(25)), data={"level": 6})
    return ledger


def test_card_puts_model_text_before_harness_rows():
    card = _full_card_ledger(20000).render_card()
    assert card.index("<solved>L1") < card.index("<world_model>") < card.index('<actions level="6">')
    assert "<world_model>L6: clicking a ring moves control to it</world_model>" in card


def test_card_cuts_harness_rows_and_paths_before_model_text():
    card = _full_card_ledger(1000).render_card()
    assert len(card) <= 1000
    assert "<actions" not in card  # every action row goes first
    # then the paths of lines older than the newest 3
    assert "<solved>L1: 24 acts, spent 30" in card and "<solved>L2: 24 acts, spent 30" in card
    assert "<solved>L5: 24 acts [U L M(4,7)" in card
    assert card.count("| debrief: Level") == 5  # debriefs stay
    assert "clicking a ring moves control" in card and "SPACE opens the door" in card


def test_card_under_a_tiny_cap_drops_notes_then_the_oldest_debrief():
    card = _full_card_ledger(600).render_card()
    assert len(card) <= 600
    assert "[U L" not in card and "<world_model>" not in card and "<hypothesis" not in card
    assert "<solved>L1: 24 acts, spent 30, 0 deaths</solved>" in card
    assert "debrief: Level 5 solved: the goal was to push block 5 into the socket." in card


def test_notes_carry_their_level_in_debrief_mode_only():
    ledger = _ledger(debrief_mode=True)
    ledger.on_level_change(3)
    ledger.absorb('<hypothesis status="confirmed">UP moves the block</hypothesis>')
    assert ledger.confirmed == ["L3: UP moves the block"]
    # a correction still finds and replaces the entry it corrects
    ledger.absorb("<hypothesis status=\"confirmed\">Correction: UP moves the block two cells</hypothesis>")
    assert ledger.confirmed == ["L3: UP moves the block two cells"]
    plain = _ledger()
    plain.absorb('<hypothesis status="confirmed">UP moves the block</hypothesis>')
    assert plain.confirmed == ["UP moves the block"]


# ---------------------------------------------------------------- the prompt

def test_prompt_states_the_debrief_and_restores_the_ledger_framing():
    common = dict(ledger=True, card=True, solved=True, effects=True, seen=True, head_card=True, deltas=False)
    text = memory_addendum(**common, debrief=True)
    assert MEMORY_HEAD_CARD_RECORDED_LINE in text and MEMORY_DEBRIEF_LINE in text
    assert MEMORY_NOTES_CLOSING_LINE in text and MEMORY_NOTES_SCOPE_LINE not in text
    assert "how" not in MEMORY_DEBRIEF_LINE
    # off: the memory-v2 text, unchanged
    assert memory_addendum(**common) == memory_addendum(**common, debrief=False)
    assert MEMORY_NOTES_SCOPE_LINE in memory_addendum(**common)
    # debrief needs solved lines to hang on
    assert memory_addendum(**{**common, "solved": False}, debrief=True) == memory_addendum(**{**common, "solved": False})


# ---------------------------------------------------------------- recap precision (PR #148 review)

# The review found 14 of 83 recaps captured on level-path-10x, mem-v2-10x and
# user-combo-10x v1 naming another level, and 4 that were no recap. These are
# those reasonings' openings, from the logs. extract_debrief(precise=True) is
# what the ledger uses.

# (reasoning opening, cleared level, the recap is kept)
REVIEW_CASES = [
    # the model calls the clear by the new board's number, then names the cleared one: kept
    ("Level 4 solved L3 with the marker-polarity rule confirmed (W→marker color, g→other). \nNow level 4: "
     "HUD top-right shows b, R, O (three colors!).", 3, True),
    ("Level 2 solved level 1 with the same mechanic. Now level 2: the pairs are now (pink, yellow).", 1, False),
    ("Level 2 solved goal from L1: cover the gray target region with boxes delivered by the player.\n"
     "New board: \n- Player (green + white) at top-left.", 1, True),
    ("Level 2 solved... no wait, LEVEL 1 COMPLETED! The UP at (15,34) with the matching key moved the block "
     "to (10,34) → level completed! The key-match hypothesis was correct.\nNow level 2. Let me record.", 1, True),
    # another level's number only: dropped
    ("Level 2 solved understanding: the goal was to push the 5x5 block (orange+blue) into the top box. "
     "Actually, wait — let me verify.\nNow level 2.", 1, False),
    ("Level 3 solved by collision! Confirmed rule: the goal is to make the two blocks collide (head-on merge).\n"
     "Level 3 board: purple (left) and red (right) backgrounds.", 2, False),
    ("Mechanics from L2: clicking the top green arrow rotates the cross clockwise, the left red arrow "
     "rotates it counter-clockwise. Now level 4.", 3, False),
    ("Under the L2/L3 rule: the sequence reads along the red box; when it hits the frame slot it dives "
     "into the linked sub-box. Now level 4.", 3, False),
    # celebrations, not recaps: dropped
    ("LEVEL 3 COMPLETE... wait, LEVEL COMPLETED → now level 3.\nLevel 2 solved in 69 steps total.", 2, False),
    ("Level 5 complete! 🎉 (The plan worked exactly as simulated: 3 covers → blue, 1 press → shape, then into "
     "the box → LEVEL COMPLETED at move 43.)\nNow level 5.", 4, False),
    ("Level 4 completed! Wait — the FAIL message was because the level changed mid-loop (the last RIGHT "
     "completed the level).\nLevel 4 board: ", 3, False),
    ("Level 4 completed! Now level 5 (step 331). Let me record the mechanics learned.", 3, False),
]


@pytest.mark.parametrize("text,level,kept", REVIEW_CASES)
def test_recap_must_be_about_the_cleared_level(text, level, kept):
    assert bool(extract_debrief(text, level + 1, precise=True)) is kept


def test_recap_cut_mid_sentence_ends_at_its_last_sentence():
    text = ("Level 1 solved: 3 clicks on bottom blue completed it because the plug's colours aligned with the "
            "gap. The completion animation showed the wall disappearing. The final state after completion had a "
            "different board (transition to level 2).")
    got = extract_debrief(text, 2, precise=True)
    assert got.endswith("the wall disappearing.") and "transition" not in got


def test_about_level_rule():
    assert debrief_about_level("Key mechanics learned: pumps move water", 4)
    assert debrief_about_level("Level 4 solved L3 with the rule", 3)
    assert not debrief_about_level("Level 4 complete!", 3)
    assert not debrief_about_level("Mechanics from L2: ...", 3)


def test_the_ledger_skips_a_wrong_level_recap_and_takes_the_next_one():
    # uc1 ls20 L4: the first reasoning calls the clear "Level 5"; the second recaps level 4
    ledger = MemoryLedger(card_mode=True, solved_mode=True, debrief_mode=True)
    ledger.level = 4
    ledger.on_level_change(5)
    ledger.note_think("Level 5 complete! 🎉 (The plan worked: 3 covers → blue, then into the box.) Now level 5.")
    assert 4 not in ledger._debriefs and ledger._debrief_for == 4
    ledger.note_think("Level 4 solved: the dial must match the shape and colour before the block enters the box. "
                      "Now level 5.")
    assert ledger._debriefs[4].startswith("Level 4 solved: the dial must match")
