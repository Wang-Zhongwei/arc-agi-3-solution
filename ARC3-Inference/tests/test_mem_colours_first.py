"""ARC3_MEM_COLOURS_FIRST: the untried-colours row first, kept, new colours located (arm/stall-fix-tool).

Run Ucore, vc33 level 4: `<this_level>`'s "Colours on the board not clicked
on this level" row rendered last inside the 700-char effect-row budget, and
the budget loop's `break` dropped it exactly when the only untried colour was
the orange gate (requests 51, 60, 62). The model used that row as a to-do
list on the same level. With the flag on the row comes first and outside the
budget, a row that does not fit is skipped rather than ending the rows, and
colours absent from the level's first board come first, each with where it
is now.

tests/fixtures/vc33_ucore_boards.json.gz: run Ucore's vc33 from the logs,
the initial board and actions 1-134 (level 4 starts after action 56; the
run ended at action 134).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from inference.agent import memory_layout as layout
from inference.agent.memory_facts import ActionEffects, FactRecorder, colour_groups

FIXTURE = json.loads(gzip.open(Path(__file__).parent / "fixtures" / "vc33_ucore_boards.json.gz", "rt").read())
LETTERS = "WwgGcBMPRbSYOrNp"
COLOURS = "Colours on the board not clicked on this level:"
VALID = ["RESET", "MOUSE"]

# the logged effect rows at request 51 (action 114); the colours row was dropped
I51_LOGGED = [
    "Clicks on blue cells (52 clicks on 6 cells): moved cells 4 times, changed cell colours 46 times, changed "
    "only the edge twice. Last: (62, 40) changed cell colours. Before that: (62, 46) changed only the edge.",
    "Clicks on yellow cells (1 click on 1 cell): changed only the edge once. Last: (56, 5) changed only the edge.",
    "Clicks on black cells (1 click on 1 cell): changed only the edge once. Last: (40, 43) changed only the edge.",
    "Clicks on charcoal cells (1 click on 1 cell): changed only the edge once. Last: (41, 5) changed only the edge.",
    "Clicks on light gray cells (1 click on 1 cell): changed only the edge once. Last: (48, 13) changed only the "
    "edge.",
]


def _recorder(before_action: int) -> FactRecorder:
    """The recorder as the turn message of the request at ``before_action`` saw it."""
    entries = []
    for row in FIXTURE["entries"]:
        if row["n"] >= before_action:
            break
        grid = tuple(tuple(LETTERS.index(ch) for ch in line) for line in row["board"])
        result = {"action_num": row["n"], "level_completed": row.get("level_completed", False),
                  "game_over": row.get("game_over", False)}
        entries.append(NS(action=row["action"], frame=NS(grid=grid, level=row["level"]), result=result))
    recorder = FactRecorder(border=4, keep_still=True)
    recorder.feed(entries)
    return recorder


def _rows(recorder: FactRecorder, *, on: bool) -> list[str]:
    extra = {"colours_first": True, "new_colours": recorder.new_colour_groups()} if on else {}
    return layout.effect_rows(recorder.effects, VALID, board_colours=recorder.board_colours,
                              game=False, max_chars=700, **extra)


# ---------------------------------------------------------------- vc33 level 4, from the logs

def test_vc33_request_51_old_rows_are_the_logged_rows():
    assert _rows(_recorder(114), on=False) == I51_LOGGED


@pytest.mark.parametrize("before_action, line", [
    (114, f"{COLOURS} orange (new on this level; rows 34-45, cols 27-29), white."),
    (129, f"{COLOURS} orange (new on this level; rows 43-54, cols 12-14)."),
    (135, f"{COLOURS} orange (new on this level; rows 43-54, cols 12-14)."),
])
def test_vc33_open_gate_is_listed_first(before_action, line):
    recorder = _recorder(before_action)
    old, new = _rows(recorder, on=False), _rows(recorder, on=True)
    assert not any(row.startswith(COLOURS) for row in old)  # dropped, as in the run
    assert new[0] == line
    # the other rows are the old ones, in the old order
    assert [row for row in new[1:]] == old


def test_vc33_level_start_has_no_new_colour():
    recorder = _recorder(58)
    assert recorder.new_colour_groups() == {}
    row = _rows(recorder, on=True)[0]
    assert row.startswith(COLOURS) and "new on this level" not in row


# ---------------------------------------------------------------- effect_rows

def _effects(clicks: dict[str, int]) -> ActionEffects:
    effects = ActionEffects()
    effects.new_level()
    for letter, count in clicks.items():
        for k in range(count):
            effects.record(f"MOUSE@{letter}", f"MOUSE({k},{k})", "change", f"h{letter}{k}", 3)
    return effects


def test_flag_off_rows_are_unchanged():
    effects = _effects({"b": 3})
    off = layout.effect_rows(effects, VALID, board_colours=["b", "O", "W"], game=False, max_chars=700)
    explicit = layout.effect_rows(effects, VALID, board_colours=["b", "O", "W"], game=False, max_chars=700,
                                  colours_first=False, new_colours={"O": [[(1, 1)]]})
    assert off == explicit
    assert off[-1] == f"{COLOURS} orange, white."


def test_colours_row_is_outside_the_budget_and_rows_are_skipped_not_cut():
    effects = _effects({"b": 3, "Y": 1, "B": 2})
    full = layout.effect_rows(effects, VALID, board_colours=["b", "Y", "B", "O"], game=False, max_chars=10_000)
    rows = [r for r in full if not r.startswith(COLOURS)]
    # a budget that holds the first and third row but not the second
    budget = len(rows[0]) + 1 + len(rows[2]) + 1
    assert len(rows[1]) > len(rows[2])
    old = layout.effect_rows(effects, VALID, board_colours=["b", "Y", "B", "O"], game=False, max_chars=budget)
    new = layout.effect_rows(effects, VALID, board_colours=["b", "Y", "B", "O"], game=False, max_chars=budget,
                             colours_first=True, new_colours={})
    assert old == rows[:1]
    assert new == [f"{COLOURS} orange.", rows[0], rows[2]]


def test_new_colours_come_first_with_their_place():
    effects = _effects({"b": 1})
    groups = {"O": [[(43, 12), (44, 12)]], "P": [[(5, 5), (5, 6), (5, 7)], [(9, 9)]]}
    rows = layout.effect_rows(effects, VALID, board_colours=["W", "O", "P", "b"], game=False, max_chars=700,
                              colours_first=True, new_colours=groups)
    assert rows[0] == (f"{COLOURS} orange (new on this level; rows 43-44, col 12), pink (new on this level; "
                       "4 cells in 2 groups, the largest at row 5, cols 5-7), white.")


def test_no_mouse_no_colours_row():
    effects = _effects({})
    rows = layout.effect_rows(effects, ["UP", "DOWN"], board_colours=["O"], game=False, max_chars=700,
                              colours_first=True, new_colours={"O": [[(1, 1)]]})
    assert not any(r.startswith(COLOURS) for r in rows)


def test_place_words():
    assert layout.colour_place_words([[(43, 12), (54, 14)]]) == "rows 43-54, cols 12-14"
    assert layout.colour_place_words([[(7, 3)]]) == "row 7, col 3"
    assert layout.colour_place_words([[(1, 1), (1, 2)], [(9, 9)]]) == (
        "3 cells in 2 groups, the largest at row 1, cols 1-2")
    assert layout.colour_place_words([]) == ""


# ---------------------------------------------------------------- the recorder

def _grid(cells: dict[tuple[int, int], int], size: int = 16):
    grid = [[0] * size for _ in range(size)]
    for (r, c), v in cells.items():
        grid[r][c] = v
    return tuple(tuple(row) for row in grid)


def _feed(recorder, frames):
    entries = [NS(action="" if i == 0 else "MOUSE(row=8, col=8)", frame=NS(grid=g, level=lvl),
                  result={"action_num": i, "level_completed": done})
               for i, (g, lvl, done) in enumerate(frames)]
    recorder.feed(entries)


def test_colour_groups_are_cropped_and_largest_first():
    grid = _grid({(0, 0): 12, (5, 5): 12, (5, 6): 12, (9, 9): 12, (15, 15): 12})
    assert colour_groups(grid, 12, 4) == [[(5, 5), (5, 6)], [(9, 9)]]
    assert colour_groups(grid, 12, 0)[0] == [(5, 5), (5, 6)]


def test_new_colour_groups_compares_with_the_level_start():
    start = _grid({(6, 6): 9, (7, 7): 11})
    later = _grid({(6, 6): 9, (7, 7): 11, (8, 8): 12, (8, 9): 12})
    recorder = FactRecorder(border=4)
    _feed(recorder, [(start, 1, False), (later, 1, False)])
    assert recorder.new_colour_groups() == {"O": [[(8, 8), (8, 9)]]}
    # a colour on the first board only inside the edge band is still new inside the crop
    recorder = FactRecorder(border=4)
    _feed(recorder, [(_grid({(6, 6): 9, (0, 0): 12}), 1, False), (later, 1, False)])
    assert set(recorder.new_colour_groups()) == {"O", "Y"}


def test_level_change_moves_the_start_board():
    a = _grid({(6, 6): 9})
    b = _grid({(6, 6): 9, (8, 8): 12})
    c = _grid({(6, 6): 9, (8, 8): 12, (10, 10): 14})
    recorder = FactRecorder(border=4)
    _feed(recorder, [(a, 1, False), (b, 2, True), (c, 2, False)])
    # orange was on level 2's first board; only light green is new there
    assert set(recorder.new_colour_groups()) == {"N"}
