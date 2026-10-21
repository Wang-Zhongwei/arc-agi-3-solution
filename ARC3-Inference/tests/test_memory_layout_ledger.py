"""The ledger and the agent under the user's memory layout (2026-10-10).

- ARC3_MEM_EVERY_CALL: notes are kept per level (open hypotheses too, closed
  when tested), labelled by the level they name; a cleared level is frozen
  into `<previous_levels>` with its recap and notes; arm/level-debrief's
  GOAL / MECHANICS fill the entry when that branch's debrief is pinned.
- ARC3_MEM_LEGIBLE: the card's solved lines in plain words.
- The system-prompt text of both, and the turn-message helpers.

The loop-level checks are in test_memory_layout_loop.py.
"""
from __future__ import annotations

from inference.agent.memory_ledger import MemoryLedger
from inference.agent.prompts import (
    MEMORY_EFFECTS_LINE, MEMORY_SOLVED_LINE, memory_addendum, memory_layout_addendum,
)
from inference.agent.tool_agent import _opener_level, _with_turn_tail


# ---------------------------------------------------------------- the ledger


def _layout_ledger() -> MemoryLedger:
    return MemoryLedger(card_mode=True, solved_mode=True, debrief_mode=True, layout_mode=True, legible=True)


def test_notes_are_kept_per_level_and_open_ones_close_when_tested():
    ledger = _layout_ledger()
    ledger.absorb("<world_model>the yellow bar counts moves</world_model>"
                  "<hypothesis>SPACE swaps the two blocks</hypothesis>")
    assert ledger.notes_for_level(1) == [("Open", "SPACE swaps the two blocks"),
                                         ("World model", "the yellow bar counts moves")]
    ledger.absorb('<hypothesis status="refuted">SPACE swaps the two blocks</hypothesis>')
    assert ledger.notes_for_level(1)[0] == ("Refuted", "SPACE swaps the two blocks")
    assert ("Open", "SPACE swaps the two blocks") not in ledger.notes_for_level(1)
    ledger.on_level_change(2)
    ledger.absorb("<level_note>needs the key first</level_note>")
    assert ledger.notes_for_level(2) == [("Level note", "needs the key first")]
    assert len(ledger.notes_for_level(1)) == 2


def test_a_note_about_the_level_just_cleared_goes_to_that_level():
    # tn36 in user-combo-10x v1: the first turn on level 2 wrote "... completes
    # level 1"; labelled with the current level it landed in level 2's entry
    ledger = _layout_ledger()
    ledger.on_level_change(2)
    ledger.absorb('<hypothesis status="confirmed">all 5 symbols black then the ball completes level 1</hypothesis>'
                  "<world_model>L2 has four symbol columns</world_model>")
    assert ledger.notes_for_level(1) == [("Confirmed", "all 5 symbols black then the ball completes level 1")]
    assert ledger.notes_for_level(2) == [("World model", "L2 has four symbol columns")]
    # once level 1 is frozen, a note naming it is about the current level
    ledger.add_solved(1, ["UP"], 1, 0)
    ledger.layout_commit([1])
    ledger.absorb("<world_model>L1/L2 mechanics: the ball runs the panel</world_model>")
    assert ("World model", "L1/L2 mechanics: the ball runs the panel") in ledger.notes_for_level(2)


def test_commit_freezes_the_level_and_adds_a_late_recap_at_the_next_redraw():
    ledger = _layout_ledger()
    ledger.absorb('<hypothesis status="confirmed">UP moves the block</hypothesis>')
    ledger.on_level_change(2)
    ledger.add_solved(1, ["UP", "UP", "LEFT"], 5, 0)
    assert ledger.layout_pending() == [1] and ledger.layout_waiting_for_recap(1)
    assert ledger.layout_commit([1])
    assert "Level 1: cleared in 3 actions (5 spent, no deaths). Route: UP x2, LEFT." in ledger.prev_text
    assert "  Confirmed: UP moves the block" in ledger.prev_text
    assert ledger.layout_pending() == []
    # the recap arrives after a forced (trim) commit: shown at the next redraw
    ledger.note_think("Level 1 solved: the goal was to bring the block to the exit, which needs UP first. Now level 2.")
    assert "Your recap" not in ledger.prev_text
    assert ledger.layout_redraw()
    assert "  Your recap: Level 1 solved: the goal was to bring the block to the exit" in ledger.prev_text
    # notes written later on level 2 do not change level 1's frozen entry
    ledger.absorb('<hypothesis status="confirmed">LEFT opens the door</hypothesis>')
    assert not ledger.layout_redraw()


def test_level_debrief_fields_fill_goal_and_rules():
    # arm/level-debrief's MemoryLedger.debriefs: level -> (fields, free text)
    ledger = _layout_ledger()
    ledger.debriefs = {1: ({"GOAL": "fill the tray", "MECHANICS": "UP pushes the block"}, "")}
    ledger.add_solved(1, ["UP"], 1, 0)
    ledger.layout_commit([1])
    assert "  Goal: fill the tray\n  Rules: UP pushes the block" in ledger.prev_text


def test_legible_card_lines():
    ledger = MemoryLedger(card_mode=True, solved_mode=True, legible=True)
    line = ledger.add_solved(1, ["UP", "UP", "LEFT"], 5, 0)
    assert line.startswith("Level 1: cleared in 3 actions (5 spent, no deaths). Route: UP x2, LEFT.")
    assert "<solved>Level 1: cleared in 3 actions" in ledger.render_card()


# ---------------------------------------------------------------- prompts and helpers


def test_layout_addendum_states_both_blocks_and_the_tags():
    text = memory_layout_addendum(debrief=True, legible=True, compact_keep=2)
    assert "`<previous_levels>` message right after these instructions" in text
    assert "ends with a `<this_level>` block" in text
    assert "except the last 2" in text and "<hypothesis>the bar" in text
    assert "U D L R S" not in text
    assert "U D L R S" in memory_layout_addendum(debrief=False, legible=False)


def test_legible_addendum_drops_the_code_legend():
    base = dict(ledger=True, card=True, solved=True, effects=True, seen=False, head_card=True, deltas=False)
    assert MEMORY_SOLVED_LINE in memory_addendum(**base)
    legible = memory_addendum(**base, legible=True)
    assert MEMORY_SOLVED_LINE not in legible and MEMORY_EFFECTS_LINE not in legible
    assert "U D L R S" not in legible


def test_turn_tail_goes_last_and_opener_level_reads_the_state_line():
    text = {"role": "user", "content": "Current state: step 4, level 3.\nValid actions right now: UP."}
    image = {"role": "user", "content": [{"type": "text", "text": "Current state: step 9, level 2."},
                                         {"type": "image_url", "image_url": {"url": "x"}}]}
    assert _with_turn_tail(text, "<this_level>\nx\n</this_level>")["content"].endswith("\n\n<this_level>\nx\n</this_level>")
    assert _with_turn_tail(image, "B")["content"][-1] == {"type": "text", "text": "\n\nB"}
    assert _with_turn_tail(text, "") is text
    assert _opener_level(text) == 3 and _opener_level(image) == 2
    assert _opener_level({"role": "tool", "content": "Current state: step 1, level 1"}) is None
    assert _opener_level({"role": "user", "content": "Current state: step 1, level 1", "_arc3_control": "resume"}) is None
