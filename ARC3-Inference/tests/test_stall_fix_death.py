"""arm/stall-fix-death: ARC3_DEATH_VERDICT and ARC3_UNDONE_FACT on real boards.

tests/fixtures/death_facts_states.json.gz holds, from the Kaggle logs
(intermediate_states.pkl frames, animation summaries recomputed with
``summarize_animation`` as the solver does):
- run U (mem-unified-10x) tu93, actions 70-97: the contact death at action 75
  (bar 54/64), the RESET, the 20-action attempt whose budget ran out at action
  96 (bar 3 -> 0) and the RESET after it;
- run Ucore (mem-unified-core-10x) r11l, actions 7-17: level 2 from its first
  board, the undone clicks 8, 10, 12 and 14, the game over at 16 (the hidden
  fifth undone move) and the RESET.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from inference.agent import death_facts as df
from inference.agent import tool_agent as ta
from inference.agent.runtime_state import Frame, HistoryEntry

FIXTURE = json.loads(gzip.open(Path(__file__).parent / "fixtures" / "death_facts_states.json.gz", "rt").read())
LETTERS = FIXTURE["letters"]
B = 4

TU93_96 = (
    "Budget bar (bottom edge): 64 cells at the start of this attempt, 0 in the death frame, about 3.2 fewer "
    "per action: the bar is empty in the death frame, so this game over is the action budget running out on "
    "action 20 of this attempt."
)
TU93_75 = "Budget bar (bottom edge): 54 of 64 cells left in the death frame, about 3.3 per action: not a budget death."
R11L_BAR = "Budget bar (left edge): 54 of 64 cells left in the death frame, about 1.1 per action: not a budget death."
R11L_UNDONE = (
    "This attempt had 4 actions that the game undid (they animated, then the board returned to how it was): "
    "actions 8, 10, 12 and 14; the game over came on action 16."
)
R11L_ACTION_8 = (
    "Action 8 MOUSE(row=30, col=30) was undone by the game: it animated, then the board returned to how it "
    "was (1st undone action in this attempt; cells that changed during the animation: rows 4-32, cols 10-47)."
)
OLD_SENTENCE = (
    "Apply the BAR RULE above to decide: bar fully depleted in the death frame means budget death (shorter "
    "route or bar restore needed); bar with budget remaining means this action itself was fatal and a "
    "DIFFERENT approach is required."
)


def entries(name: str) -> list[HistoryEntry]:
    out = []
    for row in FIXTURE[name]:
        grid = tuple(tuple(LETTERS.index(ch) for ch in line) for line in row["board"])
        out.append(HistoryEntry(action=row["action"], frame=Frame(grid=grid, step=row["n"], level=row["level"]),
                                result=dict(row["result"])))
    return out


def upto(name: str, action_num: int) -> list[HistoryEntry]:
    """The history as the harness holds it once ``action_num`` has executed."""
    return [e for e in entries(name) if e.result["action_num"] <= action_num]


# ---------------------------------------------------------------- the death verdict


def test_tu93_budget_death_is_named():
    lines, decides = df.death_verdict_lines(upto("tu93_U", 97), B)
    assert lines == [TU93_96] and decides


def test_tu93_contact_death_keeps_the_bar_rule():
    lines, decides = df.death_verdict_lines(upto("tu93_U", 76), B)
    assert lines == [TU93_75] and not decides
    reading = df.read_bar(upto("tu93_U", 76), 5, B)
    assert (reading.edge, reading.start, reading.before, reading.death, reading.actions) == ("bottom", 64, 58, 54, 3)


def test_r11l_hidden_strike_death_lists_the_undone_clicks():
    lines, decides = df.death_verdict_lines(upto("r11l_Ucore", 17), B)
    assert lines == [R11L_BAR, R11L_UNDONE] and decides


def test_no_game_over_no_verdict():
    assert df.death_verdict_lines(upto("tu93_U", 74), B) == ([], False)


# ---------------------------------------------------------------- the undone fact


def test_r11l_first_undone_click_line():
    assert df.undone_lines(upto("r11l_Ucore", 8), 8, 8, B) == [R11L_ACTION_8]


def test_undone_lines_keep_the_latest_two_with_their_order():
    lines = df.undone_lines(upto("r11l_Ucore", 15), 8, 15, B)
    assert [line.split(" was ")[0] for line in lines] == [
        "Action 12 MOUSE(row=28, col=60)", "Action 14 MOUSE(row=43, col=38)"]
    assert "(3rd undone action in this attempt;" in lines[0] and "(4th undone action" in lines[1]
    assert df.undone_lines(upto("r11l_Ucore", 15), 8, 15, B, min_ordinal=4) == lines[1:]


def test_tu93_moves_are_not_undone():
    history = upto("tu93_U", 96)
    assert not any(df.is_undone(e, B) for e in history)


# ---------------------------------------------------------------- synthetic edges


def grid(cells: dict[tuple[int, int], int], size: int = 16, fill: int = 5):
    rows = [[fill] * size for _ in range(size)]
    for (r, c), v in cells.items():
        rows[r][c] = v
    return tuple(map(tuple, rows))


def entry(n: int, action: str, g, level: int = 1, **result) -> HistoryEntry:
    base = {"action_num": n, "action_name": action.split("(")[0] if action else "",
            "gameplay_changed": True, "game_over": False}
    return HistoryEntry(action=action, frame=Frame(grid=g, step=n, level=level), result={**base, **result})


def bar(length: int, size: int = 32):
    """A bottom-row bar of colour 9 (cols 0..length-1) on a 32x32 board."""
    return grid({(size - 1, c): 9 for c in range(length)}, size)


def run_of(lengths, *, die=True, reset_len=None):
    """Seed + one action per length after the first; the last one dies."""
    history = [entry(0, "", bar(lengths[0]))]
    for n, length in enumerate(lengths[1:], 1):
        history.append(entry(n, "RIGHT", bar(length), game_over=die and n == len(lengths) - 1))
    if die:
        history.append(entry(len(lengths), "RESET", bar(lengths[0] if reset_len is None else reset_len)))
    return history


def test_synthetic_budget_and_hazard_deaths():
    lines, decides = df.death_verdict_lines(run_of([28, 21, 14, 7, 0]), B)
    assert decides and lines == [
        "Budget bar (bottom edge): 28 cells at the start of this attempt, 0 in the death frame, about 7.0 "
        "fewer per action: the bar is empty in the death frame, so this game over is the action budget "
        "running out on action 4 of this attempt."]
    lines, decides = df.death_verdict_lines(run_of([28, 24, 20, 16, 12]), B)
    assert not decides and lines == [
        "Budget bar (bottom edge): 12 of 28 cells left in the death frame, about 4.0 per action: not a "
        "budget death."]


def test_no_bar_without_steady_falls_or_a_refill():
    # two falls only
    assert df.death_verdict_lines(run_of([28, 28, 24, 24, 20]), B) == ([], False)
    # never restored: the RESET board has the empty bar
    assert df.death_verdict_lines(run_of([28, 21, 14, 7, 0], reset_len=0), B) == ([], False)
    # rises almost as often as it falls: a counter, not a budget
    assert df.death_verdict_lines(run_of([28, 24, 28, 24, 28, 24, 20, 16]), B) == ([], False)


def test_a_bar_that_refills_now_and_then_still_counts():
    lines, decides = df.death_verdict_lines(run_of([28, 25, 22, 19, 16, 13, 10, 7, 4, 1, 12, 9, 6, 3, 0]), B)
    assert decides and lines[0].startswith("Budget bar (bottom edge): 28 cells at the start")
    assert "about 2.8 fewer per action" in lines[0]  # 27 + 12 cells lost over 14 actions


def test_corner_cells_alone_are_no_bar():
    # a colour only in the bottom-right corner square that shrinks 4 -> 0
    corner = [{(15, 15): 3, (15, 14): 3, (14, 15): 3, (14, 14): 3}]
    corner += [dict(list(corner[0].items())[:k]) for k in (3, 2, 1, 0)]
    history = [entry(0, "", grid(corner[0]))]
    for n, cells in enumerate(corner[1:], 1):
        history.append(entry(n, "RIGHT", grid(cells), game_over=n == 4))
    history.append(entry(5, "RESET", grid(corner[0])))
    assert df.find_bar(history, 5, B) is None
    assert df.death_verdict_lines(history, B) == ([], False)


def anim(bbox, pixels=4):
    return {"frames": 3, "transient_pixels": pixels, "transient_bbox": list(bbox)}


def test_undone_needs_an_unchanged_board_area_and_an_interior_blink():
    g = grid({})
    assert df.is_undone(entry(1, "SPACE", g, gameplay_changed=False, animation=anim([5, 5, 8, 8])), B)
    # the board area changed
    assert not df.is_undone(entry(1, "SPACE", g, gameplay_changed=True, animation=anim([5, 5, 8, 8])), B)
    # no transient cells (a plain no-op)
    assert not df.is_undone(entry(1, "SPACE", g, gameplay_changed=False), B)
    # only the border band blinked
    assert not df.is_undone(entry(1, "SPACE", g, gameplay_changed=False, animation=anim([0, 2, 2, 13])), B)
    # a click marked where it landed (su15's ripple, lf52's box)
    flash = entry(1, "MOUSE(row=7, col=7)", g, gameplay_changed=False, animation=anim([5, 5, 9, 9]))
    assert not df.is_undone(flash, B)
    assert df.is_undone(flash, B, skip_click_flash=False)
    moved = entry(1, "MOUSE(row=7, col=7)", g, gameplay_changed=False, animation=anim([5, 9, 9, 12]))
    assert df.is_undone(moved, B)


def test_undone_lines_stay_in_the_running_attempt():
    g = grid({})
    undone = {"gameplay_changed": False, "animation": anim([5, 5, 8, 8])}
    history = [entry(0, "", g), entry(1, "SPACE", g, **undone), entry(2, "RESET", g), entry(3, "SPACE", g, **undone)]
    lines = df.undone_lines(history, 1, 3, B)
    assert len(lines) == 1 and lines[0].startswith("Action 3 SPACE was undone by the game")
    assert "(1st undone action in this attempt; cells that changed during the animation: rows 5-8, cols 5-8)" in lines[0]
    # a level change starts a new attempt too
    history = [entry(0, "", g), entry(1, "SPACE", g, **undone), entry(2, "LEFT", g, level=2),
               entry(3, "SPACE", g, level=2, **undone)]
    lines = df.undone_lines(history, 1, 3, B)
    assert len(lines) == 1 and lines[0].startswith("Action 3 SPACE") and "(1st undone" in lines[0]


def test_ordinal():
    assert [df.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 101)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "101st"]

def test_a_bar_empty_long_before_the_death_gives_no_reading():
    # empty from action 4 on, game over at action 9: the bar did not end this attempt
    assert df.death_verdict_lines(run_of([28, 21, 14, 7, 0, 0, 0, 0, 0, 0]), B) == ([], False)
    # emptied, refilled, emptied again on the fatal action: read
    lines, decides = df.death_verdict_lines(run_of([28, 21, 14, 7, 0, 0, 14, 7, 0]), B)
    assert decides and "empty in the death frame" in lines[0]


def test_one_undone_action_and_a_bar_that_has_not_moved():
    g = grid({})
    undone = {"gameplay_changed": False, "animation": anim([5, 5, 8, 8])}
    history = [entry(0, "", g), entry(1, "SPACE", g, **undone), entry(2, "RIGHT", g, game_over=True),
               entry(3, "RESET", g)]
    assert df.undone_attempt_line(history, 2, B) == (
        "This attempt had 1 action that the game undid (it animated, then the board returned to how it was): "
        "action 1; the game over came on action 2.")
    reading = df.BarReading(edge="top", colour=1, start=64, before=64, death=64, actions=1, used=0)
    assert df.bar_line(reading) == "Budget bar (top edge): 64 of 64 cells left in the death frame: not a budget death."


# ---------------------------------------------------------------- the turn message


@pytest.fixture
def agent(monkeypatch):
    for key in ("ARC3_DEATH_VERDICT", "ARC3_UNDONE_FACT", "ARC3_NOOP_GUARD_BORDER"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ARC3_ANIMATION", "1")
    return ta.ToolAgent(model="test-model")


def prompt(agent, history, summary) -> str:
    return agent._build_user_prompt(
        int(history[-1].result["action_num"]),
        valid_actions=["UP", "DOWN", "LEFT", "RIGHT", "RESET"],
        current_frame=history[-1].frame,
        history_entries=history,
        previous_step_summary=summary,
    )


TU93_SUMMARY = {"executed_count": 13, "executed_actions": ["RIGHT"] * 12 + ["UP"], "level": 4, "game_over": True,
                "fatal_action": "UP", "start_action_num": 84, "end_action_num": 96, "gameplay_changed": True,
                "board_changed": True}
R11L_SUMMARY = {"executed_count": 8, "executed_actions": ["MOUSE"] * 8, "level": 2, "game_over": True,
                "fatal_action": "MOUSE(row=55, col=8)", "start_action_num": 9, "end_action_num": 16,
                "gameplay_changed": True, "board_changed": True}


def test_flags_off_game_over_message_is_unchanged(agent, monkeypatch):
    off = prompt(agent, upto("tu93_U", 97), TU93_SUMMARY)
    assert "The game over occurred immediately after action 'UP'. " + OLD_SENTENCE in off
    assert "Budget bar" not in off
    monkeypatch.setenv("ARC3_DEATH_VERDICT", "0")
    monkeypatch.setenv("ARC3_UNDONE_FACT", "0")
    assert prompt(agent, upto("tu93_U", 97), TU93_SUMMARY) == off


def test_verdict_replaces_the_bar_rule_sentence_on_a_budget_death(agent, monkeypatch):
    off = prompt(agent, upto("tu93_U", 97), TU93_SUMMARY)
    monkeypatch.setenv("ARC3_DEATH_VERDICT", "1")
    on = prompt(agent, upto("tu93_U", 97), TU93_SUMMARY)
    assert "The game over occurred immediately after action 'UP'.\n" + TU93_96 + "\n" in on
    assert OLD_SENTENCE not in on
    # nothing else moved: the BAR RULE paragraph and the rest are as before
    assert on.replace("'UP'.\n" + TU93_96, "'UP'. " + OLD_SENTENCE) == off


def test_verdict_keeps_the_sentence_on_a_contact_death(agent, monkeypatch):
    summary = {**TU93_SUMMARY, "executed_count": 3, "start_action_num": 73, "end_action_num": 75,
               "fatal_action": "RIGHT"}
    off = prompt(agent, upto("tu93_U", 76), summary)
    monkeypatch.setenv("ARC3_DEATH_VERDICT", "1")
    on = prompt(agent, upto("tu93_U", 76), summary)
    sentence = "The game over occurred immediately after action 'RIGHT'. " + OLD_SENTENCE
    assert sentence + "\n" + TU93_75 + "\n" in on
    assert on.replace("\n" + TU93_75, "") == off


def test_verdict_on_the_r11l_strike_death(agent, monkeypatch):
    monkeypatch.setenv("ARC3_DEATH_VERDICT", "1")
    monkeypatch.setenv("ARC3_UNDONE_FACT", "1")
    on = prompt(agent, upto("r11l_Ucore", 17), R11L_SUMMARY)
    assert ("The game over occurred immediately after action 'MOUSE(row=55, col=8)'.\n"
            + R11L_BAR + "\n" + R11L_UNDONE + "\n") in on
    assert OLD_SENTENCE not in on and "was undone by the game" not in on


def test_undone_fact_in_an_ordinary_turn(agent, monkeypatch):
    summary = {"executed_count": 1, "executed_actions": ["MOUSE(row=30, col=30)"], "level": 2,
               "game_over": False, "start_action_num": 8, "end_action_num": 8, "gameplay_changed": False,
               "board_changed": True}
    off = prompt(agent, upto("r11l_Ucore", 8), summary)
    monkeypatch.setenv("ARC3_UNDONE_FACT", "1")
    on = prompt(agent, upto("r11l_Ucore", 8), summary)
    assert R11L_ACTION_8 in on.splitlines()
    assert on.replace(R11L_ACTION_8 + "\n", "") == off
    # it sits after the executed-actions lines and before the level line
    lines = on.splitlines()
    assert lines.index(R11L_ACTION_8) < lines.index("You are still on the same level.")


# ---------------------------------------------------------------- through the real sandbox and solver


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """test_animation_opener's harness: the real solver session over
    solver_fakes.flash_script (SPACE blinks a 2x2 at rows 7-8, cols 7-8 and
    reverts; MOUSE blinks the clicked cell; DOWN ends the attempt)."""
    from inference.framework import solver as solver_module
    from solver_fakes import FakeGame, blank, flash_script, frozen, make_session

    monkeypatch.setattr(solver_module, "ANIMATION_ENABLED", True)
    monkeypatch.setattr(solver_module, "BATCH_NOOP_BLOCK", False)
    monkeypatch.setattr(solver_module, "STALE_STATE_BLOCK", False)
    for key in ("ARC3_DEATH_VERDICT", "ARC3_UNDONE_FACT", "ARC3_ANIM_DIGEST", "ARC3_ANIM_RING",
                "MULTIMODAL_CONTEXT", "ARC3_ANIMATION_TIMELINE", "ARC3_NOOP_GUARD_BORDER"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ARC3_ANIMATION", "1")
    board = blank()
    board[3][3] = 9
    session = make_session(tmp_path, FakeGame(frozen(board), flash_script))
    session.seed_initial_history()
    session.write_runtime_state()
    agent = ta.ToolAgent(model="test-model")
    agent._ensure_session(session.state_path)
    agent._step_env_callback = session.step_env
    agent._current_valid_actions = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE"]
    return session, agent, monkeypatch


def _turn(harness, code: str) -> str:
    from inference.agent.runtime_state import load_runtime_state

    session, agent, _ = harness
    dispatch = agent._run_python_tool(session.state_path, {"code": code})
    assert dispatch.step_executed, dispatch.content
    current_frame, history = load_runtime_state(session.state_path)
    return agent._build_user_prompt(
        session.action_count, valid_actions=["UP", "SPACE", "MOUSE"], current_frame=current_frame,
        history_entries=history, previous_step_summary=agent._last_step_summary,
    )


SPACE_LINE = ("Action {n} SPACE was undone by the game: it animated, then the board returned to how it was "
              "({nth} undone action in this attempt; cells that changed during the animation: rows 7-8, cols 7-8).")


def test_undone_fact_through_the_sandbox(harness):
    harness[2].setenv("ARC3_UNDONE_FACT", "1")
    assert SPACE_LINE.format(n=1, nth="1st") in _turn(harness, "action(['SPACE'])").splitlines()
    # a click blinking the clicked cell is the game marking the click, not an undone move
    prompt = _turn(harness, "action([{'action': 'MOUSE', 'row': 6, 'col': 6}])")
    assert "was undone by the game" not in prompt
    # actions 3-6: three undone SPACEs around a RIGHT; the latest two are named
    prompt = _turn(harness, "action(['SPACE', 'RIGHT', 'SPACE', 'SPACE'])")
    assert [line for line in prompt.splitlines() if "was undone" in line] == [
        SPACE_LINE.format(n=5, nth="3rd"), SPACE_LINE.format(n=6, nth="4th")]


def test_death_verdict_through_the_sandbox(harness):
    harness[2].setenv("ARC3_DEATH_VERDICT", "1")
    harness[2].setenv("ARC3_UNDONE_FACT", "1")
    _turn(harness, "action(['SPACE', 'SPACE'])")
    prompt = _turn(harness, "action(['DOWN'])")
    assert ("The game over occurred immediately after action 'DOWN'.\nThis attempt had 2 actions that the game "
            "undid (they animated, then the board returned to how it was): actions 1 and 2; the game over came "
            "on action 3.\n") in prompt
    assert OLD_SENTENCE not in prompt and "was undone by the game:" not in prompt


def test_a_bar_refilled_by_the_fatal_action_gives_no_reading():
    # su15: a near-empty bar shows 50-60 cells again in the death frame
    assert df.death_verdict_lines(run_of([28, 24, 20, 16, 12, 8, 24]), B) == ([], False)
