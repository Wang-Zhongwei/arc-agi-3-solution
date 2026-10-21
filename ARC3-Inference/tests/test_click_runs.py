"""ARC3_CLICK_RUNS (arm/stall-fix-clicks): the click-run outcome fact.

tests/fixtures/click_runs_states.json.gz holds, from run U
(kaggle-outputs/v4-mem-unified-10x/v1), every board of sc25 up to action 141
and of ft09 up to action 16 (letters WwgGcBMPRbSYOrNp), with each action's
analysis step (one executed sequence per step), level, game-over and
level-completed flags, and the animation summary rebuilt from
intermediate_states.pkl with the harness's own collapse_frames and
summarize_animation.

sc25 L5 is the stall read in lab2/stalls/U_sc25_L5.md: at actions 127-130 the
model mis-typed the plus pattern and held "plus does not grow the cap" for
~71k tokens. The fact fires on the turn messages after actions 130 and 133
(requests 65 and 66) and nowhere else in the game.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from inference.agent import click_runs as cr
from inference.agent import tool_agent
from inference.agent.runtime_state import load_runtime_state

from solver_fakes import FakeGame, blank, frozen, make_session

FIXTURE = json.loads(gzip.open(Path(__file__).parent / "fixtures" / "click_runs_states.json.gz", "rt").read())
LETTERS = "WwgGcBMPRbSYOrNp"

SC25_130 = (
    "Click run at actions 127-130, (50,25) (50,35) (55,30) (60,30): only the clicked cells changed, from g to N, "
    "and they stay N; nothing else inside the board area changed and no animation played. Earlier click runs "
    "that changed more than the clicked cells or animated: (50,30) (55,25) (55,35) (60,30) at actions 90-93 "
    "(the 16-cell b/S object at rows 15-18 cols 35-38 is now 4 cells at rows 15-16 cols 35-36; the yellow 6x6 "
    "corner frame (single Y pixels at (34,50) (34,55) (39,50) (39,55)) is gone; the clicked cells are as before "
    "the run); "
    "(49,24) (55,25) (55,35) (60,30) at actions 16-19 on level 1 (the 16-cell b/S object at rows 19-22 cols "
    "35-38 is now 4 cells at rows 19-20 cols 35-36; the 9-cell N object at rows 49-51 cols 29-31 is gone); "
    "(50,30) (55,30) (60,30) at actions 121-123 (the 16-cell M/r object at rows 11-14 cols 15-18 is gone; the "
    "12-cell r object at rows 16-18 cols 51-54 is gone; no r cells are left on the board; the clicked cells are "
    "as before the run). The run at 90-93 has the same length and differs from this one in 3 of its 4 cells."
)
SC25_133 = (
    "Click run at actions 131-133, (50,25) (50,30) (55,30): only the clicked cells changed: (50,25) (55,30) from "
    "N to g, (50,30) from g to N; nothing else inside the board area changed and no animation played. Earlier "
    "click runs that changed more than the clicked cells or animated: (50,25) (50,30) (55,30) at actions 97-99 (an "
    "animation played; afterwards nothing outside the clicked cells differs; the clicked cells are as before the "
    "run); (50,30) (55,30) (60,30) at actions 121-123 (the 16-cell M/r object at rows 11-14 cols 15-18 is gone; "
    "the 12-cell r object at rows 16-18 cols 51-54 is gone; no r cells are left on the board; the clicked cells "
    "are as before the run); (52,13) (49,24) (47,30) at actions 12-14 on level 1 (the 9-cell W object at rows "
    "49-51 cols 29-31 is gone). The run at 97-99 clicked the same cells in the same order."
)


def _entries(rows: list[dict]) -> list:
    out = []
    for row in rows:
        grid = tuple(tuple(LETTERS.index(ch) for ch in line) for line in row["board"])
        result = {}
        if row["n"]:
            result = {"action_num": row["n"], "action_name": row.get("name")}
            result.update({k: row[k] for k in ("animation", "game_over", "level_completed") if k in row})
        out.append(NS(action=row["action"], frame=NS(grid=grid, level=row["level"]), result=result))
    return out


def _replay(name: str) -> dict[tuple[int, int], str]:
    """The line of each turn message, as tool_agent feeds the tracker: the
    full history so far and the sequence's first and last action numbers."""
    rows = FIXTURE[name]
    entries = _entries(rows)
    sequences: list[list] = []
    for index, row in enumerate(rows[1:], 1):
        if sequences and sequences[-1][0] == row["step"]:
            sequences[-1][2] = index
        else:
            sequences.append([row["step"], index, index])
    tracker = cr.ClickRuns()
    fired = {}
    for _, first, last in sequences:
        line = tracker.fact_line(entries[: last + 1], rows[first]["n"], rows[last]["n"])
        if line:
            fired[(rows[first]["n"], rows[last]["n"])] = line
    return fired


def test_sc25_fires_after_the_mistyped_plus_and_the_spoiled_L_only():
    assert _replay("sc25_U") == {(124, 130): SC25_130, (131, 133): SC25_133}


def test_sc25_runs_and_verdicts():
    entries = _entries(FIXTURE["sc25_U"])
    tracker = cr.ClickRuns()
    tracker.sequence_starts.update(row["n"] for i, row in enumerate(FIXTURE["sc25_U"][1:], 1)
                                   if row["step"] != FIXTURE["sc25_U"][i - 1].get("step"))
    runs = {(run.first, run.last): run for run in tracker.runs(entries)}
    assert runs[(90, 93)].outside and runs[(90, 93)].animated
    # the second L: the board area ends as it was, but the click animated
    assert runs[(97, 99)].outside == 0 and runs[(97, 99)].animated and runs[(97, 99)].effective
    for span in ((127, 130), (131, 133)):
        assert not runs[span].effective and runs[span].lit
    # the beam at 139-141 animates (11 frames)
    assert runs[(139, 141)].effective


def test_ft09_paint_runs_do_not_fire():
    # every click recolours only the clicked cell; the dead 5-click run at
    # 12-16 has no earlier effective run of its length
    entries = _entries(FIXTURE["ft09_U"])
    tracker = cr.ClickRuns()
    assert _replay("ft09_U") == {}
    tracker.sequence_starts.update({1, 2, 3, 5, 9, 10, 11, 12})
    runs = {(run.first, run.last): run for run in tracker.runs(entries)}
    assert not runs[(12, 16)].effective and runs[(12, 16)].size == 5
    assert runs[(3, 4)].effective and runs[(5, 8)].effective


# ---------------------------------------------------------------- synthetic boards

SIZE = 16


def _board(cells: dict | None = None) -> tuple:
    grid = [[0] * SIZE for _ in range(SIZE)]
    for (r, c), v in (cells or {}).items():
        grid[r][c] = v
    return tuple(tuple(row) for row in grid)


def _history(steps: list[tuple[str, dict, dict]], level_at: dict | None = None, start: dict | None = None) -> list:
    """``steps``: (display, board cells, result extras) per action."""
    entries = [NS(action="", frame=NS(grid=_board(start), level=1), result={})]
    for n, (display, cells, extra) in enumerate(steps, 1):
        level = (level_at or {}).get(n, entries[-1].frame.level)
        entries.append(NS(action=display, frame=NS(grid=_board(cells), level=level),
                          result={"action_num": n, **extra}))
    return entries


def _click(r: int, c: int) -> str:
    return f"MOUSE(row={r}, col={c})"


BLOCK = {(5, 5): 9, (5, 6): 9, (6, 5): 9, (6, 6): 9}
MOVED = {(5, 7): 9, (5, 8): 9, (6, 7): 9, (6, 8): 9}


def test_a_dead_run_after_an_effective_one_of_its_length_fires():
    entries = _history([
        (_click(9, 6), {**BLOCK, (9, 6): 14}, {}),
        (_click(10, 7), MOVED, {}),                                   # the pattern: the block moves
        (_click(9, 8), {**MOVED, (9, 8): 14}, {}),
        (_click(10, 7), {**MOVED, (9, 8): 14, (10, 7): 14, (0, 3): 5}, {}),  # dead (the bar ticks)
    ], start=BLOCK)
    tracker = cr.ClickRuns()
    assert tracker.fact_line(entries[:3], 1, 2) == ""
    assert tracker.fact_line(entries, 3, 4) == (
        "Click run at actions 3-4, (9,8) (10,7): only the clicked cells changed, from W to N, and they stay N; "
        "nothing else inside the board area changed and no animation played. Earlier click runs that changed "
        "more than the clicked cells or animated: (9,6) (10,7) at actions 1-2 (the 4-cell b block moved from "
        "rows 5-6 cols 5-6 to rows 5-6 cols 7-8; the clicked cells are as before the run). The run at 1-2 has "
        "the same length and differs from this one in 1 of its 2 cells."
    )


def test_no_line_without_an_earlier_effective_run_of_the_same_length():
    entries = _history([
        (_click(9, 6), {**BLOCK, (9, 6): 14}, {}),
        (_click(10, 7), MOVED, {}),
        (_click(9, 8), {**MOVED, (9, 8): 14}, {}),
        (_click(10, 7), {**MOVED, (9, 8): 14, (10, 7): 14}, {}),
        (_click(10, 9), {**MOVED, (9, 8): 14, (10, 7): 14, (10, 9): 14}, {}),
    ], start=BLOCK)
    tracker = cr.ClickRuns()
    tracker.fact_line(entries[:3], 1, 2)
    assert tracker.fact_line(entries, 3, 5) == ""          # three clicks, the effective run had two
    # the same two dead clicks as the first sequence of the game: nothing earlier
    assert cr.ClickRuns().fact_line(entries[:5], 1, 4) == ""


def test_runs_split_at_sequences_levels_and_other_actions():
    entries = _history([
        (_click(9, 6), {(9, 6): 14}, {}),
        (_click(9, 7), {(9, 6): 14, (9, 7): 14}, {}),
        ("UP", {(9, 6): 14, (9, 7): 14}, {}),
        (_click(9, 8), {(9, 6): 14, (9, 7): 14, (9, 8): 14}, {}),
        (_click(9, 9), {(9, 9): 14}, {"level_completed": True}),
        (_click(9, 6), {(9, 9): 14, (9, 6): 14}, {}),
        (_click(9, 7), {(9, 9): 14, (9, 6): 14, (9, 7): 14}, {}),
    ], level_at={5: 2, 6: 2, 7: 2})
    tracker = cr.ClickRuns()
    tracker.sequence_starts.add(7)
    spans = [(run.first, run.last) for run in tracker.runs(entries)]
    assert spans == [(1, 2), (4, 5), (6, 6), (7, 7)]
    by_span = {(run.first, run.last): run for run in tracker.runs(entries)}
    assert by_span[(4, 5)].effective and by_span[(4, 5)].level == 1
    assert not by_span[(1, 2)].effective


def test_an_animation_makes_a_run_effective_and_the_edge_band_is_ignored():
    animated = _history([
        (_click(9, 6), {(9, 6): 14}, {}),
        (_click(9, 7), {(9, 6): 14, (9, 7): 14}, {"animation": {"frames": 3, "transient_pixels": 4}}),
    ])
    edge = _history([
        (_click(9, 6), {(9, 6): 14, (0, 0): 3}, {}),
        (_click(9, 7), {(9, 6): 14, (9, 7): 14, (0, 0): 3, (15, 12): 3, (3, 8): 3}, {}),
    ])
    inside = _history([
        (_click(9, 6), {(9, 6): 14}, {}),
        (_click(9, 7), {(9, 6): 14, (9, 7): 14, (4, 4): 3}, {}),
    ])
    assert [run.effective for run in cr.ClickRuns().runs(animated)] == [True]
    assert [run.effective for run in cr.ClickRuns().runs(edge)] == [False]
    assert [run.outside for run in cr.ClickRuns().runs(inside)] == [1]


def test_the_clicked_object_is_the_node_holding_the_clicked_cell():
    # a 3x3 button lights up as one object: its eight other cells are clicked cells too
    button = {(r, c): 2 for r in range(8, 11) for c in range(8, 11)}
    lit = {(r, c): 14 for r in range(8, 11) for c in range(8, 11)}
    entries = [NS(action="", frame=NS(grid=_board(button), level=1), result={})]
    entries.append(NS(action=_click(9, 9), frame=NS(grid=_board(lit), level=1), result={"action_num": 1}))
    entries.append(NS(action=_click(9, 9), frame=NS(grid=_board(button), level=1), result={"action_num": 2}))
    run, = cr.ClickRuns().runs(entries)
    assert run.region == frozenset(button) and run.outside == 0 and not run.effective


# ---------------------------------------------------------------- through the turn message

PATTERN = {(9, 6), (10, 7)}


def keypad_script(action_name: str, data: dict, board):
    """A click toggles a keypad cell (rows 9-10, cols 6-9) between W and N, one
    frame. When the lit cells are exactly PATTERN the keypad resets and the
    2x2 b block at rows 5-6 moves two cells right."""
    if action_name != "ACTION6":
        return [board], {}
    r, c = int(data.get("y", 0)), int(data.get("x", 0))
    grid = [list(row) for row in board]
    if 9 <= r <= 10 and 6 <= c <= 9:
        grid[r][c] = 14 if grid[r][c] == 0 else 0
        lit = {(i, j) for i in (9, 10) for j in range(6, 10) if grid[i][j] == 14}
        if lit == PATTERN:
            for i, j in lit:
                grid[i][j] = 0
            for i in (5, 6):
                row = grid[i]
                cols = [j for j in range(16) if row[j] == 9]
                for j in cols:
                    row[j] = 0
                for j in cols:
                    row[j + 2] = 9
    return [frozen(grid)], {}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    for key in ("ARC3_ANIMATION", "ARC3_CLICK_RUNS"):
        monkeypatch.delenv(key, raising=False)
    board = blank()
    for (r, c), v in BLOCK.items():
        board[r][c] = v
    session = make_session(tmp_path, FakeGame(frozen(board), keypad_script))
    session.seed_initial_history()
    session.write_runtime_state()
    agent = tool_agent.ToolAgent(model="test-model")
    agent._ensure_session(session.state_path)
    agent._step_env_callback = session.step_env
    agent._current_valid_actions = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE"]
    return session, agent


def _turn(harness, clicks: list[tuple[int, int]]) -> str:
    session, agent = harness
    code = "action([" + ", ".join(f"{{'action': 'MOUSE', 'row': {r}, 'col': {c}}}" for r, c in clicks) + "])"
    dispatch = agent._run_python_tool(session.state_path, {"code": code})
    assert dispatch.step_executed, dispatch.content
    current_frame, history = load_runtime_state(session.state_path)
    return agent._build_user_prompt(
        session.action_count, valid_actions=["MOUSE"], current_frame=current_frame,
        history_entries=history, previous_step_summary=agent._last_step_summary,
    )


LOOP_LINE = (
    "Click run at actions 3-4, (9,8) (10,7): only the clicked cells changed, from W to N, and they stay N; "
    "nothing else inside the board area changed and no animation played. Earlier click runs that changed more "
    "than the clicked cells or animated: (9,6) (10,7) at actions 1-2 (the 4-cell b block moved from rows 5-6 "
    "cols 5-6 to rows 5-6 cols 7-8; the clicked cells are as before the run). The run at 1-2 has the same length and "
    "differs from this one in 1 of its 2 cells."
)


def test_the_line_follows_the_executed_actions(harness, monkeypatch):
    monkeypatch.setenv("ARC3_CLICK_RUNS", "1")
    first = _turn(harness, [(9, 6), (10, 7)])
    assert "Click run at" not in first
    lines = _turn(harness, [(9, 8), (10, 7)]).splitlines()
    assert LOOP_LINE in lines
    at = lines.index(LOOP_LINE)
    assert lines[at - 1].startswith("Executed actions: MOUSE(row=9, col=8), MOUSE(row=10, col=7)")
    assert lines[at + 1] == "You are still on the same level."
    # the next turn is about its own sequence: a single click fires nothing
    assert "Click run at" not in _turn(harness, [(9, 9)])


def test_a_yield_resumption_repeats_nothing(harness, monkeypatch):
    monkeypatch.setenv("ARC3_CLICK_RUNS", "1")
    _turn(harness, [(9, 6), (10, 7)])
    session, agent = harness
    _turn(harness, [(9, 8), (10, 7)])
    agent._resume_after_yield = True
    current_frame, history = load_runtime_state(session.state_path)
    again = agent._build_user_prompt(
        session.action_count, valid_actions=["MOUSE"], current_frame=current_frame,
        history_entries=history, previous_step_summary=agent._last_step_summary,
    )
    assert "Click run at" not in again


@pytest.mark.parametrize("value", [None, "0"])
def test_flag_off_adds_nothing(harness, monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("ARC3_CLICK_RUNS", value)
    _turn(harness, [(9, 6), (10, 7)])
    prompt = _turn(harness, [(9, 8), (10, 7)])
    assert "Click run" not in prompt
    assert harness[1]._click_run_facts is None
