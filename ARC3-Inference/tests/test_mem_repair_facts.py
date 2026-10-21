"""arm/mem-repair-facts on real boards.

tests/fixtures/mem_repair_states.json.gz holds stall-fix-10x's sc25 level 6
from its event log (actions 210-271, letters WwgGcBMPRbSYOrNp): at action 229
a click hid the yellow 6x6 corner frame (the moved line's marker) and showed a
purple 6x6 frame around pad B, and from request 92 on the moved line and the
boxes line were gone from `<this_level>` (the run ended unsolved).

The expected lines are what ARC3_MOVED_PERSIST gives at the requests the
spec names (92, 93, 95 and 99: actions 234, 241, 249 and 272).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace as NS

from inference.agent import board_facts as bf

FIXTURE = json.loads(gzip.open(Path(__file__).parent / "fixtures" / "mem_repair_states.json.gz", "rt").read())
LETTERS = "WwgGcBMPRbSYOrNp"
FIRST = 210  # the action whose frame is level 6's start board

HIDDEN_229 = (
    "Hidden since action 229, MOUSE(row=60, col=30) (the click at which the 16-cell b/S object at rows 37-40 "
    "cols 45-48 became 4 cells at rows 37-38 cols 45-46): the yellow 6x6 corner frame (last around rows 33-36 "
    "cols 29-32). Appeared at that action: a purple 6x6 corner frame around rows 40-43 cols 12-15; the 4-cell "
    "b/S block moved inside it at action 233, MOUSE(row=55, col=30), from rows 37-38 cols 45-46."
)
BOXES_92 = (
    "Boxes marked by four single yellow corner pixels: rows 33-36 cols 29-32 (4x4: 12 g cells and the 4 Y "
    "corners); rows 37-40 cols 53-56 (4x4: 12 g cells and the 4 Y corners); rows 40-43 cols 12-15 (4x4: 6 B, "
    "2 g, 2 b, 2 S and the 4 Y corners; the purple 6x6 frame surrounds it)."
)
BOXES_93 = (
    "Boxes marked by four single yellow corner pixels: rows 33-36 cols 29-32 (4x4: 12 g cells and the 4 Y "
    "corners); rows 37-40 cols 53-56 (4x4: 12 g cells and the 4 Y corners); rows 40-43 cols 12-15 (4x4: a 2x2 "
    "of p at rows 41-42 cols 13-14, 6 B, 2 g and the 4 Y corners; the purple 6x6 frame surrounds it)."
)
HIDDEN_254 = (
    "Hidden since action 254, MOUSE(row=60, col=30) (the click at which the 16-cell b/S object at rows 21-24 "
    "cols 41-44 became 4 cells at rows 21-22 cols 41-42): the yellow 6x6 corner frame (last around rows 37-40 "
    "cols 53-56). Appeared at that action: a purple 6x6 corner frame around rows 40-43 cols 12-15; the 4-cell "
    "b/S block moved inside it at action 259, MOUSE(row=55, col=30), from rows 21-22 cols 41-42."
)


def _entries() -> list:
    out = []
    for row in FIXTURE["sc25_SF"]["entries"]:
        grid = tuple(tuple(LETTERS.index(ch) for ch in line) for line in row["board"])
        out.append(NS(action=row["action"], frame=NS(grid=grid, level=row["level"]),
                      result={"action_num": row["n"]}))
    return out


def _facts(action: int) -> bf.BoardFacts:
    """The facts at the request of display action ``action`` (its history
    holds actions up to ``action - 1``)."""
    facts = bf.BoardFacts()
    facts.feed(_entries()[: action - FIRST])
    return facts


def test_hidden_and_shown_frames_persist_at_the_named_requests():
    assert _facts(234).moved_lines(persist=True) == [HIDDEN_229, BOXES_92]   # request 92
    assert _facts(241).moved_lines(persist=True) == [HIDDEN_229, BOXES_93]   # request 93
    assert _facts(249).moved_lines(persist=True) == [HIDDEN_229, BOXES_92]   # request 95
    assert _facts(272).moved_lines(persist=True) == [HIDDEN_254, BOXES_92]   # request 99: attempt 2


def test_without_persist_the_lines_are_gone_there():
    for action in (234, 241, 249, 272):
        assert _facts(action).moved_lines() == []


def test_persist_keeps_the_moved_line_and_its_boxes_line():
    facts = _facts(215)  # request 89: the frame moved at action 214, nothing hidden yet
    assert facts.moved_lines(persist=True) == facts.moved_lines()
    assert facts.moved_lines()[0].startswith("Moved since this level started")


def test_after_a_reset_only_the_boxes_line_stays():
    # request 97: action 249 was a RESET, the board is the level start again
    assert [line[:40] for line in _facts(250).moved_lines(persist=True)] == [BOXES_92[:40]]


def _board(cells: dict, fill: int = 5, size: int = 20):
    rows = [[fill] * size for _ in range(size)]
    for (r, c), value in cells.items():
        rows[r][c] = value
    return tuple(tuple(row) for row in rows)


def _corners(r0, c0, r1, c1, colour):
    return {(r0, c0): colour, (r0, c1): colour, (r1, c0): colour, (r1, c1): colour}


def _history(boards_and_actions):
    return [NS(action=action, frame=NS(grid=grid, level=1), result={"action_num": n})
            for n, (grid, action) in enumerate(boards_and_actions)]


def test_a_sprites_cut_corners_are_not_a_marker_frame():
    # a 4x4 orange block whose corners are light gray: the outline is orange
    sprite = {(r, c): 12 for r in range(4, 8) for c in range(4, 8)}
    sprite.update(_corners(4, 4, 7, 7, 1))
    grid = _board(sprite)
    frame = bf.Frame(1, 4, 4, 7, 7)
    assert frame in bf.corner_frames(bf.components(grid))
    assert not bf._free_frame(grid, frame)
    assert bf._free_frame(_board(_corners(4, 4, 9, 9, 11)), bf.Frame(11, 4, 4, 9, 9))


def test_frames_hidden_at_an_arrow_key_or_not_square_are_left_out():
    start = {**_corners(3, 3, 8, 8, 11), **_corners(12, 2, 14, 16, 11)}
    gone = {**_corners(12, 2, 14, 16, 11)}
    history = _history([(_board(start), ""), (_board(gone), "LEFT")])
    facts = bf.BoardFacts()
    facts.feed(history)
    assert facts.moved_lines(persist=True) == []
    history = _history([(_board(start), ""), (_board(gone), "MOUSE(row=1, col=1)")])
    facts = bf.BoardFacts()
    facts.feed(history)
    lines = facts.moved_lines(persist=True)
    assert lines[0].startswith("Hidden since action 1, MOUSE(row=1, col=1): the yellow 6x6 corner frame "
                               "(last around rows 4-7 cols 4-7).")


def test_the_agent_asks_for_the_persistent_lines_only_with_the_flag(monkeypatch):
    from inference.agent import tool_agent as ta

    agent = ta.ToolAgent(model="flashnext", base_url="http://127.0.0.1:9/v1", provider="vllm")
    agent._board_facts = _facts(234)
    for flag in ("ARC3_MEM_SOLVED", "ARC3_MEM_EVERY_CALL", "ARC3_MEM_MOVED"):
        monkeypatch.setenv(flag, "1")
    assert agent._board_fact_lines("moved") == []
    monkeypatch.setenv("ARC3_MOVED_PERSIST", "1")
    assert agent._board_fact_lines("moved") == [HIDDEN_229, BOXES_92]
    monkeypatch.setenv("ARC3_MEM_MOVED", "0")
    assert not ta._moved_persist()


# ---------------------------------------------------------------- ARC3_CLEARED_BOARDS


def _clear_script(action_name, data, board):
    """LEFT clears the level showing the solved board first (the board the
    action produced, then the next level's); RIGHT clears it showing only
    the next level's board; MOUSE recolours the clicked cell."""
    from solver_fakes import blank, frozen

    if action_name == "ACTION6":
        cells = [list(row) for row in board]
        cells[int(data.get("y", 0))][int(data.get("x", 0))] = 11
        return [frozen(cells)], {}
    if action_name in ("ACTION3", "ACTION4"):
        solved = [list(row) for row in board]
        solved[1][1] = 14
        nxt = blank()
        nxt[0][0] = 3
        frames = [frozen(solved), frozen(nxt)] if action_name == "ACTION3" else [frozen(nxt)]
        return frames, {"level_up": True}
    return [board], {}


def _clear_session(tmp_path):
    from solver_fakes import FakeGame, blank, frozen, make_session

    return make_session(tmp_path, FakeGame(frozen(blank()), _clear_script))


def test_the_solver_keeps_the_board_a_clear_produced(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC3_CLEARED_BOARDS", "1")
    session = _clear_session(tmp_path)
    session.step_env({"actions": [{"action": "MOUSE", "row": 2, "col": 3}]})
    before = [list(row) for row in session.history_entries[-1].frame.grid]
    session.step_env({"actions": [{"action": "LEFT"}]})        # clears level 1, solved board shown
    session.step_env({"actions": [{"action": "RIGHT"}]})       # clears level 2, next board only
    reply = session.step_env({"query": "cleared_boards"})
    assert reply["executed"] is False and session.action_count == 3
    one, two = reply["boards"][1], reply["boards"][2]
    assert (one["action_num"], one["action"], two["action_num"], two["action"]) == (2, "LEFT", 3, "RIGHT")
    assert two["grid"] is None
    produced = [list(row) for row in one["grid"]]
    assert produced[1][1] == 14 and produced[2][3] == 11
    before[1][1] = 14
    assert produced == before  # the board before the clear, plus what the clear did
    # the history never held it: the clear's entry is the next level's board
    assert all(list(map(list, e.frame.grid)) != produced for e in session.history_entries)


def test_without_the_flag_nothing_is_kept(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC3_CLEARED_BOARDS", raising=False)
    session = _clear_session(tmp_path)
    session.step_env({"actions": [{"action": "LEFT"}]})
    assert session.cleared_boards == {}


_CLEARED_PROBE = (
    "print('cleared_boards' in dir())\n"
    "if 'cleared_boards' in dir():\n"
    "    b = cleared_boards[1]\n"
    "    print(sorted(cleared_boards), b.level, b.step, b.shape, b.ascii.splitlines()[1][:4], cleared_boards[2])\n"
    "    print(sum(n['pixels'] for n in b.segmentation['nodes']))\n"
)


def test_the_sandbox_reads_cleared_boards_from_the_solver(tmp_path, monkeypatch):
    from inference.agent import tool_agent as ta
    from inference.agent.python_tool_sandbox import run_sandboxed_python

    monkeypatch.setenv("ARC3_CLEARED_BOARDS", "1")
    session = _clear_session(tmp_path)
    session.step_env({"actions": [{"action": "LEFT"}]})
    session.step_env({"actions": [{"action": "RIGHT"}]})
    agent = ta.ToolAgent(model="test-model")
    agent._step_env_callback = session.step_env
    payload = agent._cleared_boards_payload()
    assert sorted(payload) == ["1", "2"] and payload["2"] is None
    assert payload["1"]["step"] == 1 and payload["1"]["level"] == 1
    base = {"current_frame": None, "history": [], "valid_actions": []}
    out = run_sandboxed_python(code=_CLEARED_PROBE, timeout_seconds=20,
                               initial_state={**base, "cleared_boards": payload},
                               action_handler=lambda actions, stale_after=None: {})
    assert not out.get("error"), out
    assert out["stdout"].splitlines() == ["True", "[1, 2] 1 1 (16, 16) WNWW None", "256"]
    out = run_sandboxed_python(code=_CLEARED_PROBE, timeout_seconds=20, initial_state=base,
                               action_handler=lambda actions, stale_after=None: {})
    assert out["stdout"].splitlines() == ["False"]
    # flag off: the agent sends nothing
    monkeypatch.setenv("ARC3_CLEARED_BOARDS", "0")
    assert not ta._cleared_boards()


VC33_LINE = ("Level 3 was cleared by action 50, click (56, 46). history's last level-3 frame (step 49) is the "
             "board BEFORE that action; the board it produced is cleared_boards[3].")


def _clear_history():
    from inference.agent.runtime_state import Frame, HistoryEntry

    grid = ((0,),)
    entries = [HistoryEntry("", Frame(grid, 47, 3), {})]
    entries.append(HistoryEntry("MOUSE(row=56, col=12)", Frame(grid, 48, 3), {"action_num": 48}))
    entries.append(HistoryEntry("MOUSE(row=56, col=24)", Frame(grid, 49, 3), {"action_num": 49}))
    entries.append(HistoryEntry("MOUSE(row=56, col=46)", Frame(grid, 50, 4),
                                {"action_num": 50, "level_completed": True}))
    entries.append(HistoryEntry("LEFT", Frame(grid, 51, 4), {"action_num": 51}))
    entries.append(HistoryEntry("LEFT", Frame(grid, 52, 5), {"action_num": 52, "level_completed": True}))
    return entries


def test_clear_lines_name_the_frame_before_and_the_board_produced():
    from inference.agent.cleared_boards import clear_line, clears

    found = clears(_clear_history())
    assert [(c.level, c.action_num, c.action, c.before_step) for c in found] == [
        (3, 50, "MOUSE(row=56, col=46)", 49), (4, 52, "LEFT", 51)]
    assert clear_line(found[0], True) == VC33_LINE
    assert clear_line(found[1], False) == (
        "Level 4 was cleared by action 52, LEFT. The board it produced was never shown; history's last "
        "level-4 frame (step 51) is one action short of the clear.")


def test_the_turn_after_a_clear_and_the_entry_carry_the_line(monkeypatch):
    from inference.agent import memory_layout as layout
    from inference.agent import tool_agent as ta
    from inference.agent.memory_ledger import MemoryLedger

    agent = ta.ToolAgent(model="test-model")
    agent._step_env_callback = lambda request: {"executed": False, "query": "cleared_boards", "boards": {
        3: {"grid": ((1,),), "action_num": 50, "action": "MOUSE(row=56, col=46)"},
        4: {"grid": None, "action_num": 52, "action": "LEFT"}}}
    agent._ledger = MemoryLedger()
    lines = agent._cleared_board_lines(_clear_history(), {"start_action_num": 50, "end_action_num": 50})
    assert lines == [VC33_LINE]
    assert agent._ledger.cleared_facts[3] == VC33_LINE and 4 in agent._ledger.cleared_facts
    # a clear the solver has no record of (or another action's) gets no line
    agent._step_env_callback = lambda request: {"boards": {}}
    assert agent._cleared_board_lines(_clear_history(), {"start_action_num": 50, "end_action_num": 52}) == []
    entry = layout.LevelEntry(level=3, path=["MOUSE(56,46)"], spent=28, deaths=0, cleared=VC33_LINE,
                              recap="Goal = align each plate.")
    text = layout.render_entry(entry)
    assert text.split("\n")[1] == f"  {VC33_LINE}"
    assert "Your recap: Goal = align each plate." in text.split("\n")[2]
    plain = layout.render_entry(layout.LevelEntry(level=3, path=["MOUSE(56,46)"], spent=28, deaths=0,
                                                  recap="Goal = align each plate."))
    assert text.replace(f"  {VC33_LINE}\n", "") == plain
