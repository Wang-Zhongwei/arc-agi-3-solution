"""memory_facts: what the harness reads off the frames for memory-v2."""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import pytest

from inference.agent.memory_facts import (
    ANIM, CHANGE, CLEAR, DEATH, HUD, NOOP, SHIFT,
    ActionEffects, FactRecorder, SeenStates, Step,
    classify, distill_think, effect_key, hazard_line, loop_free, rle_path,
    solved_line, state_hash,
)
from inference.agent.noop_repeat_guard import action_signature
from inference.agent.runtime_state import Frame, HistoryEntry
from inference.utils.grid_utils import ARC_COLOR_CHARS

SIZE, BG, WALL, PLAYER, BAR, KEY = 20, 5, 3, 9, 12, 14
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
BANNED = re.compile(r"\b(inert|refuted|impossible|useless|cannot)\b", re.I)


def board(pos, *, ticks=0, key=None, extra=()):
    rows = [[BG] * SIZE for _ in range(SIZE)]
    for col in range(2, 2 + max(0, 16 - ticks)):
        rows[1][col] = BAR  # a budget bar in the edge band, ticking per action
    if key:
        rows[key[0]][key[1]] = KEY
    for r, c, v in extra:
        rows[r][c] = v
    rows[pos[0]][pos[1]] = PLAYER
    return tuple(tuple(r) for r in rows)


class Game:
    """Moves a block; the bar ticks; a wall at row 4 makes UP a no-op there."""

    def __init__(self, start=(12, 12), goal=None, level=1):
        self.pos, self.ticks, self.level, self.goal = start, 0, level, goal
        self.entries = [HistoryEntry(action="", frame=self.frame())]

    def frame(self):
        return Frame(grid=board(self.pos, ticks=self.ticks), step=self.ticks, level=self.level)

    def act(self, name, *, result=None):
        dr, dc = MOVES.get(name, (0, 0))
        r, c = self.pos[0] + dr, self.pos[1] + dc
        if 4 <= r < 16 and 4 <= c < 16:
            self.pos = (r, c)
        self.ticks += 1
        res = {"action_name": name, **(result or {})}
        if self.goal is not None and self.pos == self.goal:
            res["level_completed"] = True
            self.level += 1
            self.pos, self.ticks = (12, 12), 0
        self.entries.append(HistoryEntry(action=name, frame=self.frame(), result=res))

    def reset(self, start=(12, 12)):
        self.pos, self.ticks = start, 0
        self.entries.append(HistoryEntry(action="RESET", frame=self.frame(), result={"automatic": True}))


# ---------------------------------------------------------------- 2 classify

def test_classify_each_class():
    a = board((8, 8))
    assert classify({"level_completed": True}, a, a, 4) == CLEAR
    assert classify({"game_over": True}, a, board((8, 9)), 4) == DEATH
    assert classify({}, a, a, 4) == NOOP
    assert classify({}, a, board((8, 8), ticks=1), 4) == HUD
    assert classify({"animation": {"frames": 3}}, a, board((8, 8), ticks=1), 4) == ANIM
    assert classify({}, a, board((8, 9)), 4) == SHIFT
    assert classify({}, a, board((8, 8), extra=[(10, 10, KEY)]), 4) == CHANGE
    # with no border, the bar tick is an interior change of colour counts
    assert classify({}, a, board((8, 8), ticks=1), 0) == CHANGE


def test_keys_for_keyboard_and_clicks():
    grid = board((8, 8), extra=[(5, 6, 8)])
    assert effect_key(action_signature("UP"), grid) == "UP"
    sig = action_signature("MOUSE(row=5, col=6)")
    assert sig == "MOUSE(5,6)"
    assert effect_key(sig, grid) == f"MOUSE@{ARC_COLOR_CHARS[8]}"
    assert effect_key("MOUSE(99,99)", grid) == "MOUSE@?"


# ---------------------------------------------------------------- 3 loops, paths, borders

def test_hud_bar_does_not_split_states_at_border_4_but_does_at_0():
    a, b = board((8, 8)), board((8, 8), ticks=5)
    assert state_hash(a, 4) == state_hash(b, 4)
    assert state_hash(a, 0) != state_hash(b, 0)


def test_loop_free_path_of_the_fixture():
    game = Game(goal=(5, 12))
    seq = ["LEFT"] * 3 + ["UP"] * 4 + ["DOWN", "UP"] + ["RIGHT"] * 3 + ["SPACE"] + ["UP"] * 3
    rec = FactRecorder(border=4)
    for name in seq:
        game.act(name)
    events = rec.feed(game.entries)
    clears = [e for e in events if e.kind == "clear"]
    assert len(clears) == 1
    assert rle_path(clears[0].data["path"]) == "L3 U4 R3 U3"
    assert clears[0].data["spent"] == len(seq)
    assert [e.kind for e in events][-1] == "level"


def test_loop_free_cuts_back_to_the_start():
    steps = [Step("LEFT", "s", "a", SHIFT), Step("RIGHT", "a", "s", SHIFT), Step("UP", "s", "b", SHIFT),
             Step("SPACE", "b", "b", NOOP), Step("UP", "b", "c", CLEAR)]
    assert loop_free(steps, "s") == ["UP", "UP"]
    assert rle_path(["MOUSE(3,4)", "MOUSE(3,4)", "UNDO"]) == "M(3,4)x2 UNDO"


# ---------------------------------------------------------------- 4 distill and solved lines

def test_distill_takes_a_whole_scored_sentence():
    text = (
        "Hmm, let me look at the board. So moving RIGHT 3 times from (5,9) reaches the yellow goal. "
        "Wait, is that right? Ok the bar is fine."
    )
    out = distill_think(text)
    assert out == "Moving RIGHT 3 times from (5,9) reaches the yellow goal."
    assert distill_think("Let me look at the board again. The bar is fine.") == ""
    long = "So pressing UP " + "and then UP again " * 20 + "reaches the goal at (5,5)."
    assert distill_think(long) == ""  # never cut mid-sentence
    assert len(distill_think(text, max_chars=160)) <= 160
    # code and move traces are not findings
    assert distill_think("Batch = p1 + p2 + ['RIGHT','UP']. From (40,34): 'DOWN' → (45,34).") == ""
    assert distill_think("Note: the 4th DOWN takes us to (35,14).") == "The 4th DOWN takes us to (35,14)."
    # the LAST of equally scored sentences wins
    assert distill_think("So UP opens the door at 3. So DOWN opens the gate at 4.") == "DOWN opens the gate at 4."


def test_distill_skips_plans_commands_and_hedges():
    # real clearing-turn sentences (r2): plans for the last move, not how the level went
    for plan in (
        "Just try UP now (1 action) — the dial matches now!",
        "Let me just try UP twice to reach the goal at (3,3).",
        "Execute: DOWN, DOWN, RIGHT×5, SPACE, then verify pickup; then LEFT×6, UP, RIGHT, SPACE.",
        "For the MINI step: navigate to TOP, ensure the color, click (21,31).",
        "One more UP should push it into the head (rows 11-14).",
        "e) Click the PURPLE swatch (4,40).",
        "And check: alternate DOWN/UP 8-10 times, scan for level_completed.",
        "From (36,16) I need to get to (32,16) with an UP move.",
        "Next: RIGHT 3 times to reach the goal at (5,12).",
        "I'll press RIGHT 3 times to reach the goal at (5,12).",
    ):
        assert distill_think(plan) == "", plan
    # an action and a number or a finding are needed (score 3)
    assert distill_think("So the door needs 3 keys before it opens.") == ""
    assert distill_think("Mirrored: LEFT moves the right square RIGHT (+4 col).") == (
        "Mirrored: LEFT moves the right square RIGHT (+4 col)."
    )
    assert distill_think(
        "Just try UP now (1 action). In level 2, clicking a lens cell was a no-op."
    ) == "In level 2, clicking a lens cell was a no-op."


def test_solved_line_cap_and_parts():
    path = ["LEFT"] * 3 + ["UP"] * 4 + ["RIGHT"] * 3 + ["UP"] * 3
    line, kept = solved_line(1, path, 40, 0,
                             think="Then I take the key before the door: UP 4 times opens it.")
    assert line.startswith("L1: 13 acts [L3 U4 R3 U3], spent 40, 0 deaths")
    assert len(line) <= 165
    assert "how: I take the key before the door: UP 4 times opens it." in line
    long_path = ["LEFT", "UP"] * 60
    line, _ = solved_line(2, long_path, 300, 7, think="So UP " * 40 + ".", note="the key opens the door " * 5)
    assert len(line) <= 165 and "...+" in line and "spent 300, 7 deaths" in line


# ---------------------------------------------------------------- 5 seen boards

def _walk(names, start=(12, 12)):
    game = Game(start=start)
    for name in names:
        game.act(name)
    rec = FactRecorder(border=4)
    rec.feed(game.entries)
    return game, rec


def test_seen_line_worked_example():
    # L L L U U U D U: the last UP comes back to the board after action 6
    _, rec = _walk(["LEFT"] * 3 + ["UP"] * 3 + ["DOWN", "UP"])
    line = rec.seen_line(["UP", "DOWN", "LEFT", "RIGHT"])
    assert line == (
        "Seen before (board interior matches): after action #6 and #8 of this attempt. "
        "From it you tried: DOWN -> the board after action #7. Untried: UP, LEFT, RIGHT."
    )
    assert len(line) <= 240 and not BANNED.search(line)


def test_seen_line_words_a_move_by_where_it_went_now():
    # UP UP DOWN: back on the board after #1, whose UP led to a board that
    # was new then and has been visited since; it is not reported as new
    _, rec = _walk(["UP", "UP", "DOWN"])
    line = rec.seen_line(["UP", "DOWN"])
    assert "UP -> the board after action #2." in line and "new board" not in line
    # a move back to the attempt's start board, and one reached in an earlier attempt
    game, rec = _walk(["LEFT", "RIGHT", "LEFT"])
    assert "RIGHT -> the start board" in rec.seen_line(["UP"])
    game.reset()
    game.act("LEFT")
    rec.feed(game.entries)
    line = rec.seen_line(["UP"])
    assert "RIGHT -> the start board." in line  # the start board, visited again this attempt


def test_seen_line_quiet_on_new_boards_noops_and_the_start_after_reset():
    _, rec = _walk(["LEFT", "LEFT"])
    assert rec.seen_line(["UP"]) == ""
    # a no-op (bumping the wall) is not a revisit of the board it stayed on
    _, rec = _walk(["UP"] * 9)
    assert rec.seen_line(["UP"]) == ""
    game, rec = _walk(["LEFT", "RIGHT"])
    assert rec.seen_line(["UP"]).startswith("Seen before")
    game.reset()
    rec.feed(game.entries)
    assert rec.seen_line(["UP"]) == ""  # the start board after a RESET
    game.act("LEFT")
    game.act("RIGHT")
    rec.feed(game.entries)
    assert "at the start and after action #2 of attempt 1" in rec.seen_line(["UP"])


def test_seen_line_fatal_items_and_caps():
    rec = FactRecorder(border=4)
    events = []
    rec._begin_level(1, None, events, start_h="s", announce=False)
    rec.step(1, "RIGHT", "RIGHT", SHIFT, "s", "a", events=events)
    rec.step(1, "UP", "UP", DEATH, "a", "x", events=events)
    rec.reset_step(1, "s", None, manual=False, events=events)
    rec.step(1, "RIGHT", "RIGHT", SHIFT, "s", "a", events=events)
    line = rec.seen_line(["UP", "DOWN", "LEFT", "RIGHT"])
    assert "UP -> GAME OVER (attempt 1)" in line
    assert "GAME OVER" not in rec.seen_line(["UP", "DOWN"], hide_fatal=True)
    for i in range(12):
        rec.step(1, f"MOUSE({i},{i})", "MOUSE@R", SHIFT, "a", f"n{i}", events=events)
        rec.step(1, "UNDO", "UNDO", SHIFT, f"n{i}", "a", events=events)
    line = rec.seen_line(["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE"])
    assert len(line) <= 240 and "more" in line


def test_seen_states_is_lru_capped():
    seen = SeenStates(max_boards=3, max_actions=2)
    for i in range(5):
        seen.visit(f"b{i}", 1, i)
        seen.try_action(f"b{i}", "UP", "move", f"b{i+1}", 1)
        seen.try_action(f"b{i}", "DOWN", "move", f"b{i}", 1)
        seen.try_action(f"b{i}", "LEFT", "noop", f"b{i}", 1)
    assert len(seen.boards) == 3 and all(len(b.tried) == 2 for b in seen.boards.values())


# ---------------------------------------------------------------- C effects

def test_effect_rows_count_and_never_judge():
    game, rec = _walk(["UP"] * 9 + ["SPACE"] * 3 + ["LEFT"])
    rows = rec.effects_rows(["UP", "DOWN", "LEFT", "RIGHT", "SPACE"])
    text = "\n".join(rows)
    assert rows[0] == "UP: moved 8, edge only 1"
    assert "SPACE: no interior change on 3 presses across 1 board" in text
    assert "not tried yet this game: DOWN, RIGHT" in text
    assert not BANNED.search(text)


def test_click_rows_are_capped_under_adversarial_input():
    fx = ActionEffects()
    for colour in range(16):
        for i in range(40):
            fx.record(f"MOUSE@{ARC_COLOR_CHARS[colour]}", f"MOUSE({i},{colour})", [SHIFT, NOOP, CHANGE, DEATH][i % 4], f"h{i}", i)
    rows = fx.render(["MOUSE"], board_colours=list(ARC_COLOR_CHARS))
    clicks = [r for r in rows if r.startswith("MOUSE@")]
    assert len(clicks) <= 11
    assert sum(len(r) + 1 for r in rows) <= 1000
    assert not any(BANNED.search(r) for r in rows)


def test_hazard_line_states_the_board_not_a_cause():
    # a budget death and a hazard death look the same to the harness
    line = hazard_line(2, "RIGHT", 14, 2, 31)
    assert line == (
        "L2: attempt 2 ended in GAME OVER after 31 actions; "
        "last action RIGHT from the board first reached at action 14 of the attempt"
    )
    assert hazard_line(1, "MOUSE(3,4)", 0, 1, 1) == (
        "L1: attempt 1 ended in GAME OVER after 1 action; last action M(3,4) from its start board"
    )
    assert len(hazard_line(12, "MOUSE(61,15)", 148, 13, 150)) <= 140
    assert not re.search(r"\b(killed|kills|fatal|caused?|because)\b", line, re.I)


# ---------------------------------------------------------------- 13 replay anchor

EV_DIR = Path(os.environ.get(
    "ARC3_MEM_EV_DIR",
    "/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/"
    "beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/memory-measure/parsed",
))


def test_replay_anchor_revisits_and_exact_repeats():
    files = sorted(glob.glob(str(EV_DIR / "ev__fz-ledger-on-trim-r2__*.json")))
    if not files:
        pytest.skip("cached ev__*.json frames are not on this machine")
    totals = {"actions": 0, "moves": 0, "revisits": 0, "exact_repeats": 0}
    for path in files:
        rows = json.loads(Path(path).read_text())
        rec = FactRecorder()
        events: list = []
        level, cur = rows[0]["level"], rows[0]["ih"]
        rec._begin_level(level, None, events, start_h=cur, announce=False)
        for row in rows[1:]:
            if row["action_display"] == "RESET":
                level, cur = row["level"], row["ih"]
                rec.reset_step(level, cur, None, manual=False, events=events)
                continue
            sig = action_signature(row["action_display"])
            if row["level_completed"]:
                cls = CLEAR
            elif row["game_over"]:
                cls = DEATH
            else:
                cls = NOOP if row["ih"] == cur else SHIFT
            rec.step(level, sig, sig, cls, cur, row["ih"], post_level=row["level"], events=events)
            if cls == CLEAR:
                level = row["level"]
            cur = row["ih"]
        for key in totals:
            totals[key] += rec.stats[key]
    revisit = 100 * totals["revisits"] / totals["moves"]
    repeats = 100 * totals["exact_repeats"] / totals["actions"]
    assert abs(revisit - 14.2) <= 0.5, revisit
    assert abs(repeats - 9.0) <= 0.5, repeats


RUN_DIR = Path(os.environ.get(
    "ARC3_MEM_RUN_DIR",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs/fz-ledger-on-trim-r2",
))


def test_replay_anchor_from_the_recorders_own_frame_hashes():
    # the anchor above starts from cached hashes; this one feeds the run's
    # frames through FactRecorder.feed, so the border-4 crop is its own
    files = sorted(glob.glob(str(RUN_DIR / "artifacts" / "*_events.jsonl")))
    if not files:
        pytest.skip("the r2 run's events.jsonl frames are not on this machine")
    totals = {"actions": 0, "moves": 0, "revisits": 0, "exact_repeats": 0}
    for path in files:
        entries, previous_game_over = [], False
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            grid = tuple(map(tuple, row.get("board") or ()))
            if row["type"] == "initial":
                entries.append(HistoryEntry(action="", frame=Frame(grid=grid, step=0, level=row["level"])))
            elif row["type"] == "action":
                result = {k: row.get(k) for k in ("level_completed", "game_over", "run_complete")}
                if row["action_display"] == "RESET":
                    result["automatic"] = previous_game_over
                previous_game_over = bool(row.get("game_over"))
                entries.append(HistoryEntry(
                    action=row["action_display"],
                    frame=Frame(grid=grid, step=row["action_num"], level=row["level"]),
                    result=result,
                ))
        rec = FactRecorder(border=4)
        rec.feed(entries)
        for key in totals:
            totals[key] += rec.stats[key]
    revisit = 100 * totals["revisits"] / totals["moves"]
    repeats = 100 * totals["exact_repeats"] / totals["actions"]
    assert abs(revisit - 14.2) <= 0.5, revisit
    assert abs(repeats - 9.0) <= 0.5, repeats


# ---------------------------------------------------------------- ARC3_MEM_KEEP_EDGE
# The audit (2026-10-10): loop_free dropped edge-only actions, so ar25 L3's
# solved path in level-path-10x (SPACE #106 changed only the selection marker
# in the edge band) did not clear when replayed.

def test_keep_edge_keeps_the_selection_change_and_cuts_no_loop_across_it():
    # ar25 L3: SPACE changes only the selection marker in the edge band, then
    # UP/DOWN move the newly selected piece out and back
    steps = [Step("LEFT", "s", "a", SHIFT), Step("SPACE", "a", "a", HUD),
             Step("UP", "a", "b", SHIFT), Step("DOWN", "b", "a", SHIFT),
             Step("RIGHT", "a", "c", SHIFT), Step("UP", "c", "d", CLEAR)]
    assert loop_free(steps, "s") == ["LEFT", "RIGHT", "UP"]
    assert loop_free(steps, "s", keep_still=True) == ["LEFT", "SPACE", "RIGHT", "UP"]


def test_keep_edge_keeps_animations_drops_noops_and_still_cuts_later_loops():
    steps = [Step("LEFT", "s", "s", ANIM), Step("SPACE", "s", "s", NOOP),
             Step("UP", "s", "a", SHIFT), Step("DOWN", "a", "s", SHIFT),
             Step("RIGHT", "s", "b", SHIFT), Step("LEFT", "b", "s", SHIFT), Step("UP", "s", "c", CLEAR)]
    # the animation-only LEFT is kept (sc25 L1 #1); a loop back onto the board
    # it left is still cut, but never the LEFT itself
    assert loop_free(steps, "s", keep_still=True) == ["LEFT", "UP"]
    assert loop_free(steps, "s") == ["UP"]


def test_keep_edge_is_the_old_path_when_nothing_is_edge_only():
    steps = [Step("LEFT", "s", "a", SHIFT), Step("RIGHT", "a", "s", SHIFT), Step("UP", "s", "b", SHIFT),
             Step("SPACE", "b", "b", NOOP), Step("UP", "b", "c", CLEAR)]
    assert loop_free(steps, "s", keep_still=True) == loop_free(steps, "s") == ["UP", "UP"]
