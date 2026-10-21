"""arm/mem-repair-objects: ARC3_MEM_CONTACT_ROW, ARC3_CLICK_MISS,
ARC3_MEM_CLICKED_ALSO and ARC3_EFFECT_MOVED_FIX (object_facts.py).

tests/fixtures/mem_repair_objects_states.json.gz holds, from run SF
(kaggle-outputs/v4-stall-fix-10x/v1, the events logs), every board of m0r0 up
to action 180, cn04 up to 20, tn36 up to 36, su15 up to 57 and lf52 up to 212
(letters WwgGcBMPRbSYOrNp; a board equal to the one before is null), with each
entry's action display, level, action number and game-over flag.

The moments are the mem-repair spec's: m0r0 L5 (requests 100 and 108: the
purple pressure plates never entered), tn36 L2 (the legend boxes' yellow
never clicked), su15 L4 (the pink invader, new in this game), cn04 L2
(request 20: action 19 clicked the hole of the blue U-shape), lf52 L4 and
m0r0 L5 (arrow presses that moved a box counted as "changed cell colours").
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from inference.agent import memory_layout as layout
from inference.agent import object_facts as of
from inference.agent import tool_agent
from inference.agent.memory_facts import ActionEffects, CHANGE, CLEAR, FactRecorder, NOOP, SHIFT
from inference.agent.runtime_state import load_runtime_state

from solver_fakes import FakeGame, blank, frozen, make_session

FIXTURE = json.loads(gzip.open(Path(__file__).parent / "fixtures" / "mem_repair_objects_states.json.gz", "rt").read())
LETTERS = "WwgGcBMPRbSYOrNp"
KEYS = ["UP", "DOWN", "LEFT", "RIGHT", "MOUSE"]


def _entries(game: str) -> list:
    out, board = [], None
    for action, level, number, over, rows in FIXTURE[game]:
        if rows is not None:
            board = tuple(tuple(LETTERS.index(ch) for ch in row) for row in rows)
        out.append(NS(action=action, frame=NS(grid=board, level=level),
                      result={"action_num": number, "game_over": over}))
    return out


def _contacts(game: str, executed: int) -> of.ObjectContacts:
    """The tracker as at the turn message after action ``executed``."""
    facts = of.ObjectContacts()
    facts.feed(_entries(game)[: executed + 1])
    return facts


# ---------------------------------------------------------------- ARC3_MEM_CONTACT_ROW on SF's boards

M0R0_100 = (
    "Not touched yet on this level (never clicked, entered or bumped). New in this game: light green 4x4 at "
    "rows 26-29, cols 34-37; light green 12x4 at rows 22-25, cols 14-25. Others: orange 4x4 at rows 26-29, "
    "cols 58-61; magenta 4x4 at rows 46-49, cols 14-17; 2 purple 4x4 blocks at rows 6-9, cols 14-17 and rows "
    "50-53, cols 14-17; 2 more not listed."
)
M0R0_108 = (
    "Not touched yet on this level (never clicked, entered or bumped). New in this game: light green 4x4 at "
    "rows 26-29, cols 34-37; light green 12x4 at rows 22-25, cols 14-25. Others: orange 4x4 at rows 26-29, "
    "cols 58-61; 2 purple 4x4 blocks at rows 6-9, cols 14-17 and rows 50-53, cols 14-17; purple 12x4 at rows "
    "22-25, cols 42-53."
)
TN36_26 = (
    "Not touched yet on this level (never clicked, entered or bumped): yellow 3x2 at rows 55-56, cols 20-22; "
    "yellow 2x3 at rows 57-59, cols 8-9; charcoal 4x3 at rows 4-6, cols 45-48; charcoal 14-cell shape at rows "
    "16-19, cols 22-25; yellow 14-cell shape at rows 24-27, cols 45-48; yellow 16-cell shape at rows 7-11, cols "
    "44-49; 1 more not listed."
)
SU15_41 = (
    "Not touched yet on this level (never clicked, entered or bumped). New in this game: pink 8-cell shape at "
    "rows 19-22, cols 52-56. Others: yellow 4x4 at rows 3-6, cols 30-33; blue 69-cell shape at rows 53-61, cols 1-9."
)
CN04_20 = (
    "Not touched yet on this level (never clicked, entered or bumped). New in this game: blue 45-cell shape at "
    "rows 53-58, cols 50-58. Others: 10 red 3x3 blocks within rows 14-55, cols 14-58."
)
CN04_MISS = (
    "Action 19 MOUSE(55,54) hit background (orange) inside the outline of the blue object (rows 53-58, cols "
    "50-58); it did not touch it. Nearest blue cells: (56,54), (55,52)."
)


@pytest.mark.parametrize("game, executed, row", [
    ("m0r0", 168, M0R0_100), ("m0r0", 179, M0R0_108), ("tn36", 35, TN36_26),
    ("su15", 56, SU15_41), ("cn04", 19, CN04_20),
])
def test_contact_rows_at_the_spec_moments(game, executed, row):
    assert _contacts(game, executed).contact_row() == row


def test_m0r0_level_5_plates_stay_untouched_while_the_squares_move():
    start = _contacts("m0r0", 167).contact_row()  # the level's first board
    plates = "2 purple 4x4 blocks at rows 6-9, cols 14-17 and rows 50-53, cols 14-17"
    assert plates in start and plates in M0R0_100 and plates in M0R0_108
    assert "sky blue" not in M0R0_100  # the two squares moved: touched


def test_tn36_legend_yellow_is_listed_after_a_yellow_block_was_clicked():
    row = _contacts("tn36", 35).contact_row()
    assert "yellow 3x2 at rows 55-56, cols 20-22" in row
    assert "yellow 2x3 at rows 57-59, cols 8-9" in row


def test_cn04_click_miss_and_no_touch():
    facts = _contacts("cn04", 19)
    assert facts.miss_lines(19, 19) == [CN04_MISS]
    assert facts.miss_lines(17, 18) == []
    # the click into the U's hole changed nothing, so the U is still untouched
    assert "blue 45-cell shape" in facts.contact_row()


def test_cn04_selection_recolouring_is_never_new():
    entries = _entries("cn04")
    facts = of.ObjectContacts()
    for n in range(2, len(entries) + 1):
        facts.feed(entries[:n])
        row = facts.contact_row()
        if "New in this game: " in row:
            fresh = row.split("New in this game: ", 1)[1].split(". Others:", 1)[0]
            # only the pieces of level 2 that nobody clicked; never a recoloured selection
            assert "charcoal" not in fresh and "dark gray" not in fresh, (n, row)


# ---------------------------------------------------------------- ARC3_EFFECT_MOVED_FIX

LF52_OFF = "RIGHT (24 presses on this level): moved cells once, changed cell colours 19 times, changed only the edge 4 times."
LF52_ON = "RIGHT (24 presses on this level): moved cells 14 times, changed cell colours 6 times, changed only the edge 4 times."


def _rows(game: str, executed: int, fix: bool) -> list[str]:
    recorder = FactRecorder(border=4, keep_still=True)
    recorder.moved_fix = fix
    recorder.feed(_entries(game)[: executed + 1])
    return layout.effect_rows(recorder.effects, KEYS, game=False, max_chars=700)


def test_lf52_box_moves_count_as_moves():
    assert LF52_OFF in _rows("lf52", 211, False)
    assert LF52_ON in _rows("lf52", 211, True)


def test_m0r0_squares_crossing_the_edge_band_count_as_moves():
    off, on = _rows("m0r0", 180, False), _rows("m0r0", 180, True)
    assert "LEFT (3 presses on this level): moved cells twice, changed cell colours once." in off
    assert "LEFT (3 presses on this level): moved cells 3 times." in on
    assert "RIGHT (3 presses on this level): moved cells once, changed only the edge once, did nothing once." in on


def _grid(cells: dict, fill: int = 5, size: int = 16):
    grid = [[fill] * size for _ in range(size)]
    for (r, c), v in cells.items():
        grid[r][c] = v
    return tuple(tuple(row) for row in grid)


def test_pure_move_needs_every_change_explained():
    block = {(r, c): 9 for r in (6, 7) for c in (6, 7)}
    moved = {(r, c + 1): 9 for r, c in block}
    plate = {(r, c): 14 for r in (6, 7) for c in (8, 9)}
    gate = {(r, 11): 11 for r in range(5, 10)}
    # onto a plate of another colour: counts changed, but only the mover explains them
    assert of.pure_move(_grid({**plate, **block}), _grid({**plate, **moved}))
    # a gate that opened at the same press is not a move
    assert not of.pure_move(_grid({**plate, **block, **gate}), _grid({**plate, **moved}))
    # nothing moved
    assert not of.pure_move(_grid(block), _grid({**block, (9, 9): 3}))


def test_clicks_are_left_alone():
    recorder = FactRecorder(border=4)
    recorder.moved_fix = True
    a = _grid({(6, 6): 9, (6, 7): 9, (9, 9): 14})
    b = _grid({(6, 7): 9, (6, 8): 9})
    entries = [NS(action="", frame=NS(grid=a, level=1), result={}),
               NS(action="MOUSE(row=6, col=6)", frame=NS(grid=b, level=1), result={"action_num": 1})]
    recorder.feed(entries)
    assert recorder.effects.level_counts[("MOUSE@b", CHANGE)] == 1


# ---------------------------------------------------------------- the tracker on small boards


def _step(entries, action, cells, level=1, **result):
    entries.append(NS(action=action, frame=NS(grid=_grid(cells), level=level),
                      result={"action_num": len(entries), **result}))


def test_touches_click_enter_bump_and_merge():
    avatar = {(6, 5): 9, (6, 6): 9}
    plate = {(r, c): 14 for r in (6, 7) for c in (8, 9)}
    wall = {(r, 4): 11 for r in range(5, 10)}
    other = {(r, c): 12 for r in (9, 10) for c in (9, 10)}
    entries = [NS(action="", frame=NS(grid=_grid({**avatar, **plate, **wall, **other}), level=1), result={})]
    facts = of.ObjectContacts()
    facts.feed(entries)
    assert "light green 2x2" in facts.contact_row() and "yellow 1x5" in facts.contact_row()
    # RIGHT moves the avatar one cell: it is touched (the plate is not yet)
    _step(entries, "RIGHT", {(6, 6): 9, (6, 7): 9, **plate, **wall, **other})
    facts.feed(entries)
    assert "blue" not in facts.contact_row() and "light green" in facts.contact_row()
    # RIGHT again: onto the plate (entered)
    covered = {**plate, (6, 7): 9, (6, 8): 9}
    _step(entries, "RIGHT", {**covered, **wall, **other})
    facts.feed(entries)
    assert "light green" not in facts.contact_row()
    # back LEFT twice, then LEFT is blocked by the wall: bumped
    _step(entries, "LEFT", {(6, 6): 9, (6, 7): 9, **plate, **wall, **other})
    _step(entries, "LEFT", {**avatar, **plate, **wall, **other})
    _step(entries, "LEFT", {**avatar, **plate, **wall, **other})
    facts.feed(entries)
    assert "yellow" not in facts.contact_row()
    # a click inside the orange square's box touches it
    _step(entries, "MOUSE(row=9, col=10)", {**avatar, **plate, **wall, **other, (9, 10): 3})
    facts.feed(entries)
    assert facts.contact_row() == ""


def test_new_in_this_game_is_a_colour_no_earlier_level_showed():
    first = {(6, 6): 9, (6, 7): 9, (7, 6): 9, (7, 7): 9}
    entries = [NS(action="", frame=NS(grid=_grid(first), level=1), result={})]
    _step(entries, "MOUSE(row=1, col=1)", {**first, (9, 9): 12, (9, 10): 12, (10, 9): 12, (10, 10): 12}, level=2,
          level_completed=True)
    facts = of.ObjectContacts()
    facts.feed(entries)
    assert facts.contact_row() == (
        "Not touched yet on this level (never clicked, entered or bumped). New in this game: orange 2x2 at rows "
        "9-10, cols 9-10. Others: blue 2x2 at rows 6-7, cols 6-7."
    )


def test_level_one_has_no_new_tag_and_hud_and_walls_are_left_out():
    cells = {(r, c): 9 for r in (6, 7) for c in (6, 7)}
    cells.update({(1, c): 3 for c in range(2, 14)})  # a bar in the edge band
    cells.update({(r, 0): 11 for r in range(16)})  # a wall on the outermost column
    facts = of.ObjectContacts()
    facts.feed([NS(action="", frame=NS(grid=_grid(cells), level=1), result={})])
    assert facts.contact_row() == "Not touched yet on this level (never clicked, entered or bumped): blue 2x2 at rows 6-7, cols 6-7."


def test_click_miss_outside_the_box():
    ring = {(r, c): 9 for r in range(6, 10) for c in range(6, 10) if r in (6, 9) or c in (6, 9)}
    entries = [NS(action="", frame=NS(grid=_grid(ring), level=1), result={})]
    _step(entries, "MOUSE(row=8, col=11)", ring)
    _step(entries, "MOUSE(row=8, col=8)", ring)
    _step(entries, "MOUSE(row=13, col=13)", ring)
    facts = of.ObjectContacts()
    facts.feed(entries)
    assert facts.miss_lines(1, 3) == [
        "Action 1 MOUSE(8,11) hit background (black) 2 cells outside the bounding box of the blue object (rows 6-9, "
        "cols 6-9); it did not touch it. Nearest blue cells: (8,9), (7,9).",
        "Action 2 MOUSE(8,8) hit background (black) inside the outline of the blue object (rows 6-9, cols 6-9); it "
        "did not touch it. Nearest blue cells: (8,9), (9,8).",
    ]
    assert facts.miss_lines(1, 1, max_lines=1) == facts.miss_lines(1, 1)


# ---------------------------------------------------------------- effect_rows: lead row and clicked-also


def _effects() -> ActionEffects:
    effects = ActionEffects()
    effects.record("UP", "UP", SHIFT, "h0", 4)
    for colour, n in (("R", 3), ("Y", 1), ("p", 2)):
        for i in range(n):
            effects.record(f"MOUSE@{colour}", f"MOUSE({10 + i},{20 + i})", CHANGE, "h0", 2)
    effects.record("MOUSE@b", "MOUSE(1,1)", NOOP, "h0", 0)
    return effects


def test_flags_off_rows_are_unchanged():
    effects = _effects()
    for kwargs in ({}, {"colours_first": True}):
        base = layout.effect_rows(effects, KEYS, board_colours=["R", "Y", "O"], game=False, max_chars=120, **kwargs)
        again = layout.effect_rows(effects, KEYS, board_colours=["R", "Y", "O"], game=False, max_chars=120,
                                   clicked_also=False, **kwargs)
        assert base == again


def test_lead_row_replaces_the_colours_row():
    effects = _effects()
    rows = layout.effect_rows(effects, KEYS, board_colours=["R", "Y", "O"], game=False, max_chars=1000,
                              colours_first=True, lead_row="Not touched yet on this level: X.")
    assert rows[0] == "Not touched yet on this level: X."
    assert not any(r.startswith("Colours on the board") for r in rows)
    empty = layout.effect_rows(effects, KEYS, board_colours=["R"], game=False, max_chars=1000, lead_row="")
    assert not any(r.startswith(("Colours on the board", "Not touched")) for r in empty)


@pytest.mark.parametrize("kwargs", [{}, {"colours_first": True}, {"lead_row": "Lead."}])
def test_clicked_also_names_every_click_row_left_out(kwargs):
    effects = _effects()
    full = layout.effect_rows(effects, KEYS, board_colours=["R", "O"], game=False, max_chars=2000, **kwargs)
    assert not any(r.startswith("Also clicked") for r in
                   layout.effect_rows(effects, KEYS, board_colours=["R", "O"], game=False, max_chars=2000,
                                      clicked_also=True, **kwargs))
    tight = layout.effect_rows(effects, KEYS, board_colours=["R", "O"], game=False, max_chars=150,
                               clicked_also=True, **kwargs)
    also = [r for r in tight if r.startswith("Also clicked on this level: ")]
    assert len(also) == 1
    named = {colour.rsplit(" x", 1)[0] for colour in also[0][len("Also clicked on this level: "):-1].split(", ")}
    shown = {r.split(" cells (")[0][len("Clicks on "):] for r in tight if r.startswith("Clicks on ")}
    assert named | shown == {"red", "yellow", "purple", "blue"} and not named & shown
    assert len(full) >= len(tight) - 1
    capped = layout.effect_rows(effects, KEYS, game=False, max_chars=2000, max_click_rows=2, clicked_also=True,
                                **kwargs)
    assert "Also clicked on this level: yellow x1, blue x1." in capped


def test_clicked_also_sits_after_the_shown_click_rows():
    effects = _effects()
    rows = layout.effect_rows(effects, KEYS, game=False, max_chars=2000, max_click_rows=1, clicked_also=True)
    at = next(i for i, r in enumerate(rows) if r.startswith("Also clicked"))
    assert rows[at - 1].startswith("Clicks on red cells")
    assert rows[at + 1].startswith("Not tried on this level")


# ---------------------------------------------------------------- through the turn message

BLOCK = {(r, c): 9 for r in (5, 6) for c in (5, 6)}
RING = {(r, c): 12 for r in range(8, 12) for c in range(8, 12) if r in (8, 11) or c in (8, 11)}


def scene_script(action_name: str, data: dict, board):
    """Arrows move the blue 2x2; a click on the orange ring turns it yellow;
    anything else does nothing."""
    grid = [list(row) for row in board]
    moves = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}
    if action_name in moves:
        dr, dc = moves[action_name]
        cells = [(r, c) for r in range(16) for c in range(16) if grid[r][c] == 9]
        for r, c in cells:
            grid[r][c] = 0
        for r, c in cells:
            grid[r + dr][c + dc] = 9
    elif action_name == "ACTION6":
        r, c = int(data.get("y", 0)), int(data.get("x", 0))
        if grid[r][c] == 12:
            for i in range(16):
                for j in range(16):
                    if grid[i][j] == 12:
                        grid[i][j] = 11
    return [frozen(grid)], {}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    for key in ("ARC3_CLICK_MISS", "ARC3_MEM_CONTACT_ROW", "ARC3_MEM_CLICKED_ALSO", "ARC3_EFFECT_MOVED_FIX"):
        monkeypatch.delenv(key, raising=False)
    board = blank()
    for (r, c), v in {**BLOCK, **RING}.items():
        board[r][c] = v
    session = make_session(tmp_path, FakeGame(frozen(board), scene_script))
    session.seed_initial_history()
    session.write_runtime_state()
    agent = tool_agent.ToolAgent(model="test-model")
    agent._ensure_session(session.state_path)
    agent._step_env_callback = session.step_env
    agent._current_valid_actions = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE"]
    return session, agent


def _turn(harness, code: str) -> str:
    session, agent = harness
    dispatch = agent._run_python_tool(session.state_path, {"code": code})
    assert dispatch.step_executed, dispatch.content
    current_frame, history = load_runtime_state(session.state_path)
    return agent._build_user_prompt(
        session.action_count, valid_actions=["UP", "DOWN", "LEFT", "RIGHT", "MOUSE"], current_frame=current_frame,
        history_entries=history, previous_step_summary=agent._last_step_summary,
    )


MISS = ("Action 1 MOUSE(9,9) hit background (white) inside the outline of the orange object (rows 8-11, cols "
        "8-11); it did not touch it. Nearest orange cells: (8,9), (9,8).")


def test_click_miss_follows_the_executed_actions(harness, monkeypatch):
    monkeypatch.setenv("ARC3_CLICK_MISS", "1")
    lines = _turn(harness, "action([{'action': 'MOUSE', 'row': 9, 'col': 9}])").splitlines()
    assert MISS in lines
    assert lines[lines.index(MISS) - 1].startswith("Executed actions: MOUSE(row=9, col=9)")
    # a click on the ring changes it: no line
    assert "hit background" not in _turn(harness, "action([{'action': 'MOUSE', 'row': 8, 'col': 9}])")


def test_click_miss_not_on_a_yield_resumption(harness, monkeypatch):
    monkeypatch.setenv("ARC3_CLICK_MISS", "1")
    _turn(harness, "action([{'action': 'MOUSE', 'row': 9, 'col': 9}])")
    session, agent = harness
    agent._resume_after_yield = True
    current_frame, history = load_runtime_state(session.state_path)
    again = agent._build_user_prompt(
        session.action_count, valid_actions=["MOUSE"], current_frame=current_frame,
        history_entries=history, previous_step_summary=agent._last_step_summary,
    )
    assert "hit background" not in again


@pytest.mark.parametrize("value", [None, "0"])
def test_flags_off_add_nothing(harness, monkeypatch, value):
    if value is not None:
        for key in ("ARC3_CLICK_MISS", "ARC3_MEM_CONTACT_ROW", "ARC3_MEM_CLICKED_ALSO", "ARC3_EFFECT_MOVED_FIX"):
            monkeypatch.setenv(key, value)
    prompt = _turn(harness, "action([{'action': 'MOUSE', 'row': 9, 'col': 9}])")
    assert "hit background" not in prompt and "Not touched yet" not in prompt
    assert harness[1]._object_contacts is None


MEMORY = {"ARC3_MEM_SOLVED": "1", "ARC3_MEM_EVERY_CALL": "1", "ARC3_MEM_LEGIBLE": "1", "ARC3_MEM_EFFECTS": "1",
          "ARC3_MEM_COLOURS_FIRST": "1"}
CONTACT = ("Not touched yet on this level (never clicked, entered or bumped): orange 4x4 frame at rows 8-11, "
           "cols 8-11.")


def _memory_on(harness, monkeypatch, flags: dict) -> None:
    for key, value in {**MEMORY, **flags}.items():
        monkeypatch.setenv(key, value)
    harness[1]._ledger = tool_agent._new_memory_ledger()


def test_contact_row_replaces_the_colours_row(harness, monkeypatch):
    _memory_on(harness, monkeypatch, {"ARC3_MEM_CONTACT_ROW": "1"})
    _turn(harness, "action(['RIGHT'])")
    lines = harness[1]._mem_level_tail.splitlines()
    assert lines[:3] == ["<this_level>", "Level 1 so far: 1 action, attempt 1, no deaths.", CONTACT]
    assert not any(line.startswith("Colours on the board") for line in lines)
    # clicking inside the ring's box (on the background, changing nothing) touches nothing
    _turn(harness, "action([{'action': 'MOUSE', 'row': 9, 'col': 9}])")
    assert CONTACT in harness[1]._mem_level_tail
    # clicking the ring itself does
    _turn(harness, "action([{'action': 'MOUSE', 'row': 8, 'col': 9}])")
    assert "Not touched yet" not in harness[1]._mem_level_tail


def test_memory_on_contact_flag_off_keeps_the_colours_row(harness, monkeypatch):
    _memory_on(harness, monkeypatch, {})
    _turn(harness, "action(['RIGHT'])")
    lines = harness[1]._mem_level_tail.splitlines()
    assert lines[2].startswith("Colours on the board not clicked on this level:")
    assert harness[1]._object_contacts is None
