"""memory_layout.py: memory lines in plain words (ARC3_MEM_LEGIBLE) and the two
blocks of ARC3_MEM_EVERY_CALL, `<previous_levels>` and `<this_level>`.

The user (2026-10-10): the ledger "is not legible that well" -- lines such as
`<solved>L1: 17 acts [U L D11 L4], spent 25, 0 deaths</solved>` and
`UP: not tried this level | game moved 5, changed 17, no change 1`. These
render the same facts with labels, full action names, x counts, cells as
(row, col), colour names and one fact per line.

The ledger's use of them is in test_memory_layout_ledger.py; the loop-level
checks (prefix, placement, compaction) in test_memory_layout_loop.py.
"""
from __future__ import annotations

import re

from inference.agent import memory_layout as ml
from inference.agent.memory_facts import FactRecorder

CODES_RE = re.compile(r"\b\d+ acts\b|\[[UDLRS](?: |\d|\])|\| game |\bM\(\d+,\d+\)|MOUSE@")


# ---------------------------------------------------------------- words


def test_solved_text_reads_like_the_users_example():
    path = ["RIGHT", "RIGHT", "SPACE", *["DOWN"] * 8, "SPACE", *["LEFT"] * 4]
    assert ml.solved_text(2, path, 23, 0) == (
        "Level 2: cleared in 16 actions (23 spent, no deaths). "
        "Route: RIGHT x2, SPACE, DOWN x8, SPACE, LEFT x4."
    )
    assert ml.solved_text(1, ["MOUSE(12,30)", "MOUSE(12,30)"], 5, 1).endswith(
        "(5 spent, 1 death). Route: click (12, 30) x2."
    )


def test_long_routes_end_with_a_count_of_the_rest():
    path = [f"MOUSE({r},{c})" for r in range(10) for c in range(10)]
    text = ml.route_words(path, 120)
    assert len(text) <= 120 and re.search(r"\.\.\. and \d+ more actions$", text)


def test_effect_rows_are_words_one_fact_per_line():
    rec = FactRecorder(border=0)
    rec.effects.record("UP", "UP", "shift", "a", 3)
    rec.effects.record("UP", "UP", "noop", "b", 0)
    rec.effects.record("UP", "UP", "noop", "b", 0)
    rec.effects.record("MOUSE@R", "MOUSE(12,30)", "change", "a", 4)
    rows = ml.effect_rows(rec.effects, ["UP", "DOWN", "MOUSE"], board_colours=["R", "b"], game=False)
    assert rows == [
        "UP (3 presses on this level): moved cells once, did nothing twice.",
        "Clicks on red cells (1 click on 1 cell): changed cell colours once. Last: (12, 30) changed cell colours.",
        "Not tried on this level: DOWN.",
        "Colours on the board not clicked on this level: blue.",
    ]
    assert not any(CODES_RE.search(row) for row in rows)


def test_hazard_and_first_effect_in_words():
    assert ml.hazard_text(3, "DOWN", 7, 1, 9) == (
        "Level 3: attempt 1 ended in GAME OVER after 9 actions. "
        "The last action was DOWN, from the board first reached after action 7 of that attempt."
    )
    assert ml.first_effect_words("MOUSE@Y", "change", "MOUSE(4,7)") == (
        "a click on a yellow cell (click (4, 7)) changed cell colours for the first time in this game"
    )


# ---------------------------------------------------------------- blocks


def _entry(level: int, **kw) -> ml.LevelEntry:
    return ml.LevelEntry(level=level, path=["UP"] * 5 + ["LEFT"], spent=9, deaths=level % 2, **kw)


def test_previous_levels_one_entry_per_level_with_recap_and_notes():
    text = ml.render_previous_levels([
        _entry(2, recap="Level 2 solved: the goal was to fill the tray.", notes=[("Confirmed", "UP pushes")]),
        _entry(1),
    ])
    assert text.splitlines() == [
        "<previous_levels>",
        "Level 1: cleared in 6 actions (9 spent, 1 death). Route: UP x5, LEFT.",
        "Level 2: cleared in 6 actions (9 spent, no deaths). Route: UP x5, LEFT.",
        "  Your recap: Level 2 solved: the goal was to fill the tray.",
        "  Confirmed: UP pushes",
        "</previous_levels>",
    ]


def test_previous_levels_compresses_the_oldest_first():
    entries = [_entry(n, recap="x " * 120, notes=[("Confirmed", "y " * 90)] * 4) for n in range(1, 8)]
    text = ml.render_previous_levels(entries, max_chars=2500)
    assert len(text) <= 2500
    lines = text.splitlines()
    # the newest level keeps its recap; the oldest is down to its head line
    assert any(line.startswith("  Your recap:") for line in lines[lines.index(next(
        line for line in lines if line.startswith("Level 7:"))):])
    first = next(i for i, line in enumerate(lines) if line.startswith("Level 1:"))
    assert lines[first + 1].startswith("Level 2:")


def test_this_level_block_is_capped_and_keeps_its_summary():
    text = ml.render_this_level(
        level=4, attempt=2, actions=30, deaths=1, rows=["row " * 40] * 8,
        hazards=["GAME OVER one", "GAME OVER two", "GAME OVER three"], seen="The current board was seen before.",
        notes=[("Open", "the bar counts moves")] * 5, max_chars=ml.LEVEL_CHARS,
    )
    assert len(text) <= ml.LEVEL_CHARS
    assert text.splitlines()[1] == "Level 4 so far: 30 actions, attempt 2, 1 death."
    assert "GAME OVER one" not in text and "GAME OVER three" in text
