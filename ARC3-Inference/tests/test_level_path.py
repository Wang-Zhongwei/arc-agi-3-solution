"""ARC3_MEM_PATH: the shortest route on record from a level's start board to
the current board, over the move links the harness observed on that level."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from inference.agent.memory_facts import (
    CLEAR, DEATH, NOOP, SHIFT, FactRecorder, LevelGraph, route_suffix,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import memory_sim  # noqa: E402

MEM_FLAGS = (*memory_sim.MEMORY_V2_FLAGS, "ARC3_MEM_PATH", "ARC3_LEDGER_ON_TRIM", "ARC3_ACTION_ECHO")
# memory_sim's level 1: the key at (6,8), the goal at (5,12), red at (11,9)
LEVEL1 = [["UP"], ["DOWN"], ["RIGHT", "RIGHT"], ["DOWN", "DOWN", "DOWN", "DOWN"],
          ["UP", "UP", "UP"], ["RIGHT", "RIGHT", "RIGHT", "RIGHT"]]


class Walk:
    """Hash-only replay through ``FactRecorder.step``: boards are names."""

    def __init__(self, start: str = "S", level: int = 1) -> None:
        self.rec = FactRecorder(route=True)
        self.events: list = []
        self.cur, self.start, self.level = start, start, level
        self.rec._begin_level(level, None, self.events, start_h=start, announce=False)

    def go(self, sig: str, post: str | None = None, cls: str = SHIFT) -> "Walk":
        post = self.cur if post is None else post
        cls = NOOP if post == self.cur and cls == SHIFT else cls
        self.rec.step(self.level, sig, sig, cls, self.cur, post, post_level=self.level + (cls == CLEAR),
                      events=self.events)
        if cls == CLEAR:
            self.level += 1
            self.start = post
        self.cur = post
        return self

    def reset(self, *, manual: bool = True) -> "Walk":
        self.rec.reset_step(self.level, self.start, None, manual=manual, events=self.events)
        self.cur = self.start
        return self

    def rows(self, since: int = 0) -> list[tuple]:
        return [(r.sig, r.d, r.change, r.before) for r in self.rec.route_rows_since(since)]

    def suffix(self, since: int) -> str:
        return route_suffix(self.rec.route_rows_since(since))

    def each(self, *moves) -> list[str]:
        """The echo suffix after each single-action call."""
        out = []
        for move in moves:
            before = self.rec.route_total
            if move == "RESET":
                self.reset()
            else:
                self.go(*move)
            out.append(self.suffix(before))
        return out


# ---------------------------------------------------------------- the graph

def test_graph_routes_are_the_fewest_links_from_the_start():
    g = LevelGraph(1, "S")
    assert g.add_move("S", "RIGHT", "A")
    assert g.add_move("A", "RIGHT", "B")
    assert g.add_move("B", "DOWN", "C")
    assert g.route("C") == ["RIGHT", "RIGHT", "DOWN"]
    assert not g.add_move("A", "RIGHT", "B")  # already on record: nothing changes
    g.add_move("S", "DOWN", "C")
    assert g.route("C") == ["DOWN"] and g.dist["C"] == 1
    assert g.route("S") == [] and g.route("nowhere") is None


def test_graph_marks_varied_pairs_and_names_them_on_the_route():
    g = LevelGraph(1, "S")
    g.add_move("S", "RIGHT", "A")
    g.add_move("A", "RIGHT", "B")
    g.add_move("A", "RIGHT", "D")
    assert ("A", "RIGHT") in g.varied
    assert g.route("D") == ["RIGHT", "RIGHT"] and g.route_varied("D") == [1]
    # a move plus a death or no change from the same pair is varied too
    g.note_other("S", "UP")
    g.add_move("S", "UP", "E")
    g.add_move("B", "LEFT", "F")
    g.note_other("B", "LEFT")
    assert {("S", "UP"), ("B", "LEFT")} <= g.varied
    assert g.route_varied("E") == [0]


def test_graph_caps_refuse_links_and_never_grow_a_distance():
    g = LevelGraph(1, "S", max_boards=3, max_sigs=2, max_outcomes=2)
    g.add_move("S", "RIGHT", "A")
    g.add_move("S", "LEFT", "B")
    assert not g.add_move("S", "UP", "B")  # a third action from S
    assert not g.add_move("A", "UP", "C")  # a fourth board
    assert g.add_move("A", "UP", "B") and g.add_move("A", "UP", "S")
    assert not g.add_move("A", "UP", "A")  # a third outcome of one pair
    assert g.full == 3 and g.links["A"]["UP"] == ["B", "S"]
    assert g.dist == {"S": 0, "A": 1, "B": 1}


def test_a_new_start_board_moves_the_source():
    g = LevelGraph(1, "S")
    g.add_move("S", "RIGHT", "A")
    g.add_move("T", "UP", "A")
    assert g.dist == {"S": 0, "A": 1}
    g.set_start("T")
    assert g.dist == {"T": 0, "A": 1} and g.route("A") == ["UP"]


# ---------------------------------------------------------------- the recorder

def test_recorder_rows_for_the_worked_sequence():
    # S-R->A, A-R->B, B-D->C, C-U->B, a manual RESET, then S-D->C
    w = Walk().go("RIGHT", "A").go("RIGHT", "B").go("DOWN", "C").go("UP", "B").reset().go("DOWN", "C")
    assert w.rows() == [
        ("RIGHT", 1, "new", None), ("RIGHT", 2, "new", None), ("DOWN", 3, "new", None),
        ("UP", 2, "known", 2), ("RESET", 0, "start", 0), ("DOWN", 1, "shorter", 3),
    ]
    rows = w.rec.route_rows
    assert [(r.attempt, r.i, r.k) for r in rows] == [(1, 1, 1), (1, 2, 2), (1, 3, 3), (1, 4, 4), (2, 0, 0), (2, 1, 1)]
    assert w.rec.attempts.spent == 6 and rows[-1].path == "D"
    assert [r.act for r in rows] == [1, 2, 3, 4, 5, 6]


def test_no_change_death_clear_and_undo_rows_add_no_link():
    w = Walk().go("RIGHT", "A").go("SPACE").go("UNDO", "S").go("UP", "X", cls=DEATH)
    assert w.rows() == [
        ("RIGHT", 1, "new", None), ("SPACE", 1, "still", 1), ("UNDO", 0, "known", 0), ("UP", None, "death", None),
    ]
    graph = w.rec.graph
    assert graph.links == {"S": {"RIGHT": ["A"]}}
    w.reset(manual=False)
    assert w.rec.route_rows[-1].automatic and w.rec.route_rows[-1].pre_d is None
    w.go("RIGHT", "A").go("DOWN", "W", cls=CLEAR)
    clear = w.rec.route_rows[-1]
    assert (clear.change, clear.pre_d, clear.path, clear.level) == ("clear", 1, "R D", 1)
    # the next level starts a new graph
    assert w.rec.graph.level == 2 and w.rec.graph.start_h == "W" and w.rec.graph.links == {}


def test_undo_reaches_boards_that_no_link_reaches():
    w = Walk().go("RIGHT", "A").go("MOUSE(4,7)", "B").go("UNDO", "C").go("UP", "D")
    assert w.rows()[-2:] == [("UNDO", None, "new", None), ("UP", None, "new", None)]
    assert w.rec.graph.route("D") is None


def _facts_walk(rec: FactRecorder, before_clear=lambda: None) -> list:
    """A death, an automatic and a manual RESET, a clear and a move on level 2."""
    events: list = []
    rec._begin_level(1, None, events, start_h="S", announce=False)
    rec.step(1, "RIGHT", "RIGHT", SHIFT, "S", "A", events=events)
    rec.step(1, "UP", "UP", DEATH, "A", "X", events=events)
    rec.reset_step(1, "S", None, manual=False, events=events)
    rec.step(1, "RIGHT", "RIGHT", SHIFT, "S", "A", events=events)
    rec.reset_step(1, "S", None, manual=True, events=events)
    rec.step(1, "DOWN", "DOWN", SHIFT, "S", "B", events=events)
    before_clear()
    rec.step(1, "LEFT", "LEFT", CLEAR, "B", "W", post_level=2, events=events)
    rec.step(2, "UP", "UP", SHIFT, "W", "P", events=events)
    return events


def _facts(rec: FactRecorder, events: list) -> tuple:
    tracker = rec.attempts
    stats = {k: v for k, v in rec.stats.items() if not k.startswith(("graph_", "route_"))}
    return ([(e.kind, e.level, e.data) for e in events], tracker.level, tracker.attempt, tracker.spent,
            tracker.deaths, rec.current_h, rec.position, stats)


@pytest.mark.parametrize("hook", ["_route_step", "_route_reset", "clear route"])
def test_a_route_error_leaves_the_memory_v2_facts_as_without_routes(monkeypatch, hook):
    def boom(*args, **kwargs):
        raise RuntimeError("route hook failed")

    plain = FactRecorder(route=False)
    expected = _facts(plain, _facts_walk(plain))
    assert [kind for kind, _, _ in expected[0]] == ["first", "death", "first", "clear", "level", "first"]
    rec = FactRecorder(route=True)
    if hook == "clear route":
        events = _facts_walk(rec, before_clear=lambda: monkeypatch.setattr(LevelGraph, "route", boom))
    else:
        monkeypatch.setattr(FactRecorder, hook, boom)
        events = _facts_walk(rec)
    assert _facts(rec, events) == expected
    assert rec.stats["route_errors"] >= 1
    if rec.graph is None:  # dropped on this level: no route on record, no line
        assert rec.route_state() == {"path": None, "len": None, "varied": [], "before_death": None}
        assert rec.route_line(0) == ""
    else:  # dropped on level 1; level 2 started a new graph
        assert rec.graph.level == 2


def test_graph_caps_are_counted_in_stats():
    w = Walk()
    for i in range(20):
        w.go(f"MOUSE({i},0)", f"B{i}").go("UNDO", "S")
    assert w.rec.stats["graph_full"] == 4  # 16 actions per board
    assert len(w.rec.graph.links["S"]) == 16


# ---------------------------------------------------------------- the echo suffix

def test_suffix_after_each_action_of_the_worked_sequence():
    w = Walk()
    assert w.each(("RIGHT", "A"), ("RIGHT", "B"), ("DOWN", "C"), ("UP", "B"), "RESET", ("DOWN", "C")) == [
        " | route 1 (new board)", " | route 2 (new board)", " | route 3 (new board)",
        " | route 2 (known board)", " | route 0 (start board)", " | route 1 (shorter than 3 on record)",
    ]


def test_suffix_for_no_change_game_over_clear_and_a_batch():
    w = Walk().go("RIGHT", "A")
    assert w.each(("SPACE",), ("UP", "X", DEATH)) == [
        " | route 1 (no board change)", " | route - (GAME OVER from a board at route 1)",
    ]
    w.reset(manual=False)  # automatic: no row shown
    assert w.suffix(w.rec.route_total - 1) == ""
    before = w.rec.route_total
    w.go("RIGHT", "A").go("RIGHT", "B").go("LEFT", "A").go("UP", "F").go("UP", "G")
    assert w.suffix(before) == " | route 0->3 (1 2 1 2 3)"
    assert w.each(("DOWN", "W", CLEAR)) == [" | route - (level cleared; 4 on record)"]
    # a batch that dies, and one with a shortcut, on level 2 (start W)
    before = w.rec.route_total
    w.go("RIGHT", "P").go("RIGHT", "Q").go("DOWN", "Z", DEATH)
    assert w.suffix(before) == " | route 0->X (1 2 X)"
    w.reset(manual=False)
    before = w.rec.route_total
    w.go("DOWN", "Q").go("UP", "W").go("RIGHT", "P")
    assert w.suffix(before) == " | route 0->1 (1 0 1; shorter route found)"


def test_batch_head_spaces_the_arrow_when_a_side_has_no_route():
    w = Walk().go("RIGHT", "A").go("UNDO", "C")  # C: no recorded link reaches it
    before = w.rec.route_total
    w.go("UP", "E").go("LEFT", "F")
    assert w.suffix(before) == " | route - -> - (- -)"
    before = w.rec.route_total
    w.go("UNDO", "S").go("RIGHT", "A")
    assert w.suffix(before) == " | route - -> 1 (0 1)"
    before = w.rec.route_total
    w.go("UNDO", "G").go("DOWN", "X", DEATH)
    assert w.suffix(before) == " | route 1->X (- X)"  # both sides set: unchanged


def test_long_batches_show_the_first_and_last_five_within_80_chars():
    w = Walk()
    before = w.rec.route_total
    for i in range(30):
        w.go("RIGHT", f"B{i}")
    text = w.suffix(before)
    assert text == " | route 0->30 (1 2 3 4 5 ... 26 27 28 29 30)"
    for i in range(30, 140):
        w.go("RIGHT", f"B{i}")
    before = w.rec.route_total
    for i in range(140, 160):
        w.go("RIGHT", f"B{i}")
    w.go("LEFT", "B5")
    text = w.suffix(before)
    assert len(text) <= 80 and text.startswith(" | route 140->6 (") and "shorter" not in text


# ---------------------------------------------------------------- the turn line

HEAD = "Shortest route on record from this level's start board to the current board: "


def test_turn_line_after_a_shortcut_on_the_second_attempt():
    w = Walk().go("RIGHT", "A").go("RIGHT", "B").go("DOWN", "C").go("UP", "B")
    assert w.rec.route_line(0) == (
        HEAD + "2 actions [R2]; this attempt so far: 4 actions. "
        "Route length after each action since the last message: 1 2 3 2."
    )
    w.reset().go("DOWN", "C")
    assert w.rec.route_line(4) == (
        HEAD + "1 action [D]; this attempt so far: 1 action. "
        "Route length after each action since the last message: 0 1. "
        "A shorter route to a known board was found at action #1 of this attempt (3 -> 1)."
    )
    assert w.rec.route_line(6) == ""  # nothing ran since


def test_turn_line_names_the_attempt_when_an_earlier_one_acted_since():
    # R, R, U in one call, then a manual RESET and D in later calls of the same turn
    w = Walk().go("RIGHT", "A").go("RIGHT", "B").go("UP", "C").reset().go("DOWN", "D")
    assert w.rec.route_line(0) == (
        HEAD + "1 action [D]; this attempt so far: 1 action. "
        "Route length after each action of attempt 2 since the last message: 0 1."
    )
    # a GAME OVER, its automatic RESET, then a move in the same turn
    w = Walk().go("RIGHT", "A").go("UP", "X", DEATH).reset(manual=False).go("DOWN", "D")
    died = (
        HEAD + "1 action [D]; this attempt so far: 1 action. Attempt 1 ended in GAME OVER after 2 actions; "
        "its last board before GAME OVER is 1 action from the start on record [R]."
    )
    assert w.rec.route_line(0, max_chars=400) == (
        died + " Route length after each action of attempt 2 since the last message: 1."
    )
    assert w.rec.route_line(0) == died  # over 300 chars the list goes first


def test_turn_line_of_the_worked_batch():
    w = Walk().go("RIGHT", "A").go("RIGHT", "B").go("LEFT", "A").go("UP", "F").go("UP", "G")
    assert w.rec.route_line(0) == (
        HEAD + "3 actions [R U2]; this attempt so far: 5 actions. "
        "Route length after each action since the last message: 1 2 1 2 3."
    )


def test_turn_line_after_a_game_over():
    w = Walk().go("RIGHT", "A").go("UP", "X", DEATH).reset(manual=False)
    assert w.rec.route_line(0) == (
        HEAD + "0 actions (this is the start board). Attempt 1 ended in GAME OVER after 2 actions; "
        "its last board before GAME OVER is 1 action from the start on record [R]."
    )


def test_turn_line_names_varied_steps_clicks_and_unreached_boards():
    w = Walk().go("RIGHT", "A").go("RIGHT", "B").go("LEFT", "A").go("RIGHT", "D")
    assert w.rec.route_line(3) == (
        HEAD + "2 actions [R2]; this attempt so far: 4 actions. "
        "Route length after each action since the last message: 2. "
        "1 step on the route had more than one outcome on this level."
    )
    w = Walk().go("MOUSE(4,7)", "A")
    assert w.rec.route_line(0).startswith(HEAD + "1 action [M(4,7)];")
    w.go("UNDO", "C")
    assert w.rec.route_line(1) == (
        HEAD + "none on record; this attempt so far: 2 actions. "
        "Route length after each action since the last message: -."
    )


def test_turn_line_is_quiet_after_a_clear_and_drops_the_list_over_its_cap():
    w = Walk().go("RIGHT", "A").go("DOWN", "W", CLEAR)
    assert w.rec.route_line(0) == ""  # no action on the new level yet
    w.go("UP", "P")
    assert w.rec.route_line(0) == (
        HEAD + "1 action [U]; this attempt so far: 1 action. "
        "Route length after each action since the last message: 1."
    )
    w = Walk()
    for i in range(120):
        w.go("RIGHT", f"B{i}")
    line = w.rec.route_line(0)
    assert len(line) <= 300 and "Route length after each action" not in line
    assert line.startswith(HEAD + "120 actions [R120]; this attempt so far: 120 actions.")


# ---------------------------------------------------------------- level_path

def test_level_path_is_the_route_as_action_takes_it():
    w = Walk().go("RIGHT", "A").go("RIGHT", "B").go("DOWN", "C").go("UP", "B").reset().go("DOWN", "C")
    assert w.rec.route_state() == {"path": ["DOWN"], "len": 1, "varied": [], "before_death": None}
    w.reset()
    assert w.rec.route_state() == {"path": [], "len": 0, "varied": [], "before_death": None}
    w = Walk().go("RIGHT", "A").go("RIGHT", "B").go("LEFT", "A").go("RIGHT", "D")
    assert w.rec.route_state() == {"path": ["RIGHT", "RIGHT"], "len": 2, "varied": [1], "before_death": None}
    w = Walk().go("MOUSE(4,7)", "A")
    assert w.rec.route_state()["path"] == [{"action": "MOUSE", "row": 4, "col": 7}]
    w.go("UNDO", "C")
    assert w.rec.route_state() == {"path": None, "len": None, "varied": [], "before_death": None}
    assert FactRecorder().route_state() is None


def test_level_path_before_death_keeps_the_whole_route_after_the_reset():
    # the audit's probe (#106): 24 boards, then GAME OVER and the automatic RESET
    w = Walk()
    moves = ["UP", "LEFT", "UP", "RIGHT", "DOWN", "LEFT"] * 4
    for i, sig in enumerate(moves):
        w.go(sig, f"B{i}")
    w.go("UP", "X", DEATH).reset(manual=False)
    state = w.rec.route_state()
    assert state["path"] == [] and state["len"] == 0
    assert state["before_death"] == moves  # whole, where the turn line cuts it at 40 chars
    assert "...+" in w.rec.route_line(0)
    # it stays through the next attempt, follows a later GAME OVER, and a new level drops it
    w.go("UP", "B0").go("LEFT", "B1")
    assert w.rec.route_state()["before_death"] == moves
    w.go("DOWN", "X", DEATH).reset(manual=False)
    assert w.rec.route_state()["before_death"] == ["UP", "LEFT"]
    w.go("RIGHT", "Y", DEATH).reset(manual=False)
    assert w.rec.route_state()["before_death"] == []  # the start board itself
    w.go("UP", "B0").go("SPACE", "W", CLEAR)
    assert w.rec.route_state()["before_death"] is None
    # a board no recorded link reaches
    w = Walk().go("RIGHT", "A").go("UNDO", "C").go("UP", "X", DEATH).reset(manual=False)
    assert w.rec.route_state()["before_death"] is None


def test_tool_text_names_level_path_only_with_the_flag(monkeypatch, tmp_path):
    import json

    from inference.agent import tool_agent as ta

    monkeypatch.delenv("ARC3_MEM_PATH", raising=False)
    agent = ta.ToolAgent(model="sim-model", base_url="http://127.0.0.1:9/v1", provider="vllm")
    state_path = tmp_path / "artifacts" / "g_p0_tool_runtime_state.json"
    off = agent._tools(state_path)
    assert off[0]["function"]["description"] == ta._PYTHON_TOOL_DESCRIPTION
    assert "level_path" not in ta._only_tool_line()
    monkeypatch.setenv("ARC3_MEM_PATH", "1")
    on = agent._tools(state_path)
    assert json.dumps(on) == json.dumps(agent._tools(state_path))  # the same object every request
    assert on[0]["function"]["description"] == ta._PYTHON_TOOL_DESCRIPTION + (
        " `level_path` is the shortest route on record from this level's start board to the current board, "
        "as a list action(...) accepts, and `level_path_len` its length; `level_path_before_death` is the route "
        "on record from the start board to the last board before this level's latest GAME OVER (None before "
        "one); all refresh after every action(...) call."
    )
    assert on[0]["function"]["parameters"] == off[0]["function"]["parameters"]
    assert ta._only_tool_line() == (
        "Only tool: `python`. It receives `current_frame`, `previous_frame`, `history`, `transitions`, "
        "`last_transition`, `valid_actions`, `last_action_call_result`, `frame_diff(before, after)`, "
        "`level_path`, `level_path_len`, `level_path_before_death`, and `action(actions)`."
    )


# ---------------------------------------------------------------- the solved line

def _solved(walk: Walk) -> tuple[str, dict]:
    from inference.agent.memory_ledger import MemoryLedger

    clear = [e for e in walk.events if e.kind == "clear"][-1]
    ledger = MemoryLedger(solved_mode=True)
    data = clear.data
    return ledger.add_solved(clear.level, data["path"], data["spent"], data["deaths"], winning=data.get("winning")), ledger


def test_solved_line_uses_a_shorter_route_on_record():
    w = Walk().go("DOWN", "C").reset()
    w.go("RIGHT", "A").go("RIGHT", "B").go("DOWN", "C").go("RIGHT", "W", CLEAR)
    line, ledger = _solved(w)
    assert line == "L1: 2 acts [D R] (4 in the winning attempt), spent 6, 0 deaths"
    # the card's compressed form of an older line keeps it too
    assert ledger._solved_lines(1) == [f"  <solved>{line}</solved>"]


def test_solved_line_is_unchanged_when_the_winning_attempt_is_as_short():
    plain = FactRecorder()
    path = Walk()
    for w in (path,):
        w.go("RIGHT", "A").go("LEFT", "S").go("UP", "B").go("RIGHT", "W", CLEAR)
    events: list = []
    plain._begin_level(1, None, events, start_h="S", announce=False)
    for sig, pre, post, cls in (("RIGHT", "S", "A", SHIFT), ("LEFT", "A", "S", SHIFT),
                                ("UP", "S", "B", SHIFT), ("RIGHT", "B", "W", CLEAR)):
        plain.step(1, sig, sig, cls, pre, post, post_level=2, events=events)
    want = [e.data for e in events if e.kind == "clear"]
    got = [e.data for e in path.events if e.kind == "clear"]
    assert got == want and "winning" not in got[0]
    assert _solved(path)[0] == "L1: 2 acts [U R], spent 4, 0 deaths"


def test_solved_line_never_keeps_an_undo_that_is_no_route_on_record():
    # the audit's case (#106): loop_free keeps UNDO back onto A, a tie in length
    w = Walk().go("RIGHT", "A").go("LEFT", "S").go("UNDO", "A").go("DOWN", "B").go("SPACE", "W", CLEAR)
    clear = [e for e in w.events if e.kind == "clear"][-1]
    assert clear.data["path"] == ["RIGHT", "DOWN", "SPACE"] and "winning" not in clear.data
    assert _solved(w)[0] == "L1: 3 acts [R D S], spent 5, 0 deaths"
    # loop_free shorter than the route: UNDO jumps back onto C, 3 links away
    w = Walk().go("RIGHT", "A").go("RIGHT", "B").go("RIGHT", "C").go("DOWN", "S")
    w.go("UNDO", "C").go("SPACE", "W", CLEAR)
    clear = [e for e in w.events if e.kind == "clear"][-1]
    assert clear.data["path"] == ["RIGHT", "RIGHT", "RIGHT", "SPACE"] and "winning" not in clear.data


# ---------------------------------------------------------------- the system prompt

PATH_LINE = (
    "- Route facts come from links the harness records on each level: a board-changing action links one "
    "board to the next (boards compared with the outer 4 cells cropped; UNDO, RESET, GAME OVER, clears and "
    "no-change actions add no link; a click is its exact cell). The route is the fewest links from this "
    "level's start board to the current board. Each action(...) call prints route N and what changed; a turn "
    "message after actions on the current level carries a Shortest route on record line, up to 16000 "
    "characters of these lines per game. In python, level_path is that route as a list action(...) accepts "
    "and level_path_len its length.\n"
)
V2_BOTH = {"ARC3_LEDGER_ON_TRIM": "1", "ARC3_MEM_SOLVED": "1", "ARC3_MEM_CARD": "1", "ARC3_MEM_EFFECTS": "1",
           "ARC3_MEM_SEEN": "1", "ARC3_MEMORY_PLACEMENT": "both"}


def _system(monkeypatch, **env) -> str:
    from inference.agent import tool_agent as ta

    for flag in (*MEM_FLAGS, "ARC3_NOOP_GUARD_BORDER"):
        monkeypatch.delenv(flag, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return ta._build_system_prompt(tool_output_tokens=1024)


def test_system_prompt_places_the_route_line_after_the_fact_lines(monkeypatch):
    from inference.agent.prompts import MEMORY_SEEN_LINE, MEMORY_SOLVED_LINE, MEMORY_SOLVED_ROUTE_LINE

    text = _system(monkeypatch, **V2_BOTH, ARC3_MEM_PATH="1")
    assert MEMORY_SEEN_LINE + PATH_LINE + "- The other memory lines are your own notes." in text
    assert MEMORY_SOLVED_ROUTE_LINE in text and MEMORY_SOLVED_LINE not in text
    assert MEMORY_SOLVED_ROUTE_LINE == (
        "- `<solved>` lines are written by the harness when a level is cleared: the shortest route on record to "
        "the winning board, then the winning action (U D L R S are UP DOWN LEFT RIGHT SPACE, M(r,c) a click at "
        "row r, column c, a number repeats the action); \"(K in the winning attempt)\" follows when that "
        "attempt, with its loops cut out, was longer; then the actions spent on the level and its deaths.\n"
    )
    off = _system(monkeypatch, **V2_BOTH)
    assert text.replace(PATH_LINE, "").replace(MEMORY_SOLVED_ROUTE_LINE, MEMORY_SOLVED_LINE) == off
    assert "level_path" not in off and "route" not in off.split("Memory kept by the harness:")[1].split("\n\n")[0]


def test_system_prompt_without_memory_v2(monkeypatch):
    from inference.agent.prompts import LEDGER_ON_TRIM_ADDENDUM, PYTHON_ADDENDUM_TAIL

    alone = _system(monkeypatch, ARC3_MEM_PATH="1")
    plain = _system(monkeypatch)
    at = plain.index(PYTHON_ADDENDUM_TAIL)
    assert alone == plain[:at] + "\n\nMemory kept by the harness:\n" + PATH_LINE + plain[at:]
    ledger = _system(monkeypatch, ARC3_LEDGER_ON_TRIM="1", ARC3_MEM_PATH="1")
    assert LEDGER_ON_TRIM_ADDENDUM + PATH_LINE in ledger
    assert _system(monkeypatch, ARC3_MEM_PATH="1", ARC3_NOOP_GUARD_BORDER="2").count("outer 2 cells cropped") == 1


def test_system_prompt_states_the_turn_line_cap_and_hold(monkeypatch):
    # the prompt is true of every turn: the per-game cap, and a hold when one is set
    knobs = dict(ARC3_MEM_PATH="1", ARC3_MEM_PATH_GAME_CHARS="8000", ARC3_MEM_PATH_AFTER_TOKENS="5000")
    for env in ({**knobs}, {**V2_BOTH, **knobs}):
        text = _system(monkeypatch, **env)
        assert text.count(
            "carries a Shortest route on record line once the level has used 5000 generated tokens, "
            "up to 8000 characters of these lines per game."
        ) == 1


# ---------------------------------------------------------------- the agent

def _agent(monkeypatch, **env):
    from inference.agent import tool_agent as ta

    for flag in MEM_FLAGS:
        monkeypatch.delenv(flag, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    agent = ta.ToolAgent.__new__(ta.ToolAgent)
    agent._ledger = ta._new_memory_ledger()
    agent._mem_recorder = None
    agent._mem_pending_events = []
    agent._current_valid_actions = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE"]
    agent._session_generated_tokens = 0
    agent._tokens_at_level_start = 0
    agent._resume_after_yield = False
    return agent


def test_actions_fed_mid_snippet_give_the_memory_the_same_events(monkeypatch):
    on = {"ARC3_LEDGER_ON_TRIM": "1", "ARC3_MEM_EFFECTS": "1", "ARC3_MEM_SEEN": "1", "ARC3_MEM_SOLVED": "1",
          "ARC3_MEMORY_PLACEMENT": "both", "ARC3_MEM_LINES_AFTER_TOKENS": "0"}
    plain_game, path_game = memory_sim.FakeGame(), memory_sim.FakeGame()
    plain = _agent(monkeypatch, **on)
    path = _agent(monkeypatch, **on, ARC3_MEM_PATH="1")
    for turn, actions in enumerate(LEVEL1):
        for game, agent in ((plain_game, plain), (path_game, path)):
            if agent is path:
                monkeypatch.setenv("ARC3_MEM_PATH", "1")
                agent._route_feed(game.history)  # the snippet's initial state
                for name in actions:
                    done = game.step([name])
                    agent._route_feed(game.history)  # after each action(...) call
                    if done and (done[-1]["game_over"] or done[-1]["level_completed"]):
                        break  # a terminal result ends the snippet's actions
            else:
                monkeypatch.delenv("ARC3_MEM_PATH", raising=False)
                game.step(actions)
            if turn % 2:
                agent._mem_observe(game.history)  # a turn message
    assert path._mem_recorder.route and not plain._mem_recorder.route
    assert path._mem_pending_events == []
    assert {k: v.text for k, v in path._ledger._items.items()} == {k: v.text for k, v in plain._ledger._items.items()}
    assert [e.text for e in path._ledger._queue] == [e.text for e in plain._ledger._queue]
    assert "solved:1" in plain._ledger._items  # a death, then the clear (hazards end with the level)


def test_path_alone_feeds_the_recorder_without_a_ledger(monkeypatch):
    agent = _agent(monkeypatch, ARC3_MEM_PATH="1")
    assert agent._ledger is None
    game = memory_sim.FakeGame()
    game.step(["UP", "DOWN", "LEFT"])
    agent._route_feed(game.history)
    assert agent._mem_pending_events  # a game-first event per key, kept for the next message
    agent._mem_observe(game.history)
    assert agent._mem_pending_events == [] and agent._mem_recorder.route_total == 3
    monkeypatch.delenv("ARC3_MEM_PATH")
    agent._mem_recorder = None
    agent._mem_observe(game.history)
    assert agent._mem_recorder is None  # nothing on: nothing recorded


# ---------------------------------------------------------------- through the sandbox

def _sandbox_agent(monkeypatch, tmp_path, *, echo: bool = True, level: int = 3, **env):
    """A real ToolAgent whose action(...) plays memory_sim's game in the real sandbox."""
    from inference.agent import tool_agent as ta
    from inference.agent.runtime_state import write_runtime_state
    from inference.framework.solver import _build_action_echo

    for flag in MEM_FLAGS:
        monkeypatch.delenv(flag, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    game = memory_sim.FakeGame(level=level)
    state_path = tmp_path / "artifacts" / "sim_p0_tool_runtime_state.json"
    write_runtime_state(state_path, current_frame=game.frame(), history=game.history)
    agent = ta.ToolAgent(model="sim-model", base_url="http://127.0.0.1:9/v1", provider="vllm")
    agent._ensure_session(state_path)

    def step_env(arguments):
        names = [str(a["action"]) for a in arguments["actions"]]
        executed = game.step(names)
        write_runtime_state(state_path, current_frame=game.frame(), history=game.history)
        final = memory_sim.batch_payload(executed, names)
        if echo:
            final["action_echo"] = _build_action_echo(names, executed, None)
        return final

    agent._step_env_callback = step_env
    agent._current_valid_actions = game.valid_actions()
    return agent, state_path, game


def _stdout(result) -> str:
    import json

    return json.loads(result.content).get("stdout", "")


def test_the_worked_batch_echo_through_the_sandbox(monkeypatch, tmp_path):
    agent, state_path, _ = _sandbox_agent(monkeypatch, tmp_path, ARC3_MEM_PATH="1")
    out = _stdout(agent._run_python_tool(state_path, {"code": '_ = action(["RIGHT", "RIGHT", "LEFT", "UP", "UP"])\n'}))
    assert out == "[action] batch of 5 actions, 5 executed: RIGHT, RIGHT, LEFT, UP, UP | route 0->3 (1 2 1 2 3)\n"
    out = _stdout(agent._run_python_tool(state_path, {"code": 'action("DOWN")\n_ = action("DOWN")\n'}))
    assert out == (
        "[action] DOWN -> effect | route 2 (known board)\n"
        "[action] DOWN -> effect | route 1 (known board)\n"
    )


def test_without_the_echo_the_route_prints_its_own_line(monkeypatch, tmp_path):
    agent, state_path, _ = _sandbox_agent(monkeypatch, tmp_path, echo=False, ARC3_MEM_PATH="1")
    out = _stdout(agent._run_python_tool(state_path, {"code": '_ = action("RIGHT")\n'}))
    assert out == "[route] route 1 (new board)\n"


def test_flag_off_prints_the_echo_unchanged(monkeypatch, tmp_path):
    agent, state_path, _ = _sandbox_agent(monkeypatch, tmp_path)
    out = _stdout(agent._run_python_tool(state_path, {"code": '_ = action(["RIGHT", "UP"])\n'}))
    assert out == "[action] batch of 2 actions, 2 executed: RIGHT, UP\n"
    assert agent._mem_recorder is None


def test_turn_lines_order_hold_cap_and_resume(monkeypatch):
    on = {"ARC3_LEDGER_ON_TRIM": "1", "ARC3_MEM_EFFECTS": "1", "ARC3_MEM_SEEN": "1",
          "ARC3_MEMORY_PLACEMENT": "both", "ARC3_MEM_LINES_AFTER_TOKENS": "0", "ARC3_MEM_PATH": "1"}
    agent = _agent(monkeypatch, **on)
    agent._route_cursor = agent._route_chars_used = 0
    game = memory_sim.FakeGame()
    game.step(["UP", "DOWN", "UP", "DOWN"])  # back on the start board: a Seen line
    agent._mem_observe(game.history)
    assert agent._mem_turn_lines(game.valid_actions()) == [
        "Seen before (board interior matches): at the start and after action #2 and #4 of this attempt. "
        "From it you tried: UP -> the board after action #3. Untried: DOWN, LEFT, RIGHT, SPACE.",
        HEAD + "0 actions (this is the start board); this attempt so far: 4 actions. "
        "Route length after each action since the last message: 1 0 1 0.",
        "<memory_update>\n+ UP: first moved; DOWN: first moved\n</memory_update>",
    ]
    assert agent._route_cursor == 4
    game.step(["UP"])
    agent._mem_observe(game.history)
    agent._resume_after_yield = True
    assert agent._mem_turn_lines(game.valid_actions()) == [] and agent._route_cursor == 4
    agent._resume_after_yield = False
    route = [line for line in agent._mem_turn_lines(game.valid_actions()) if line.startswith("Shortest")]
    assert route and route[0].endswith("Route length after each action since the last message: 1.")
    used = agent._route_chars_used
    monkeypatch.setenv("ARC3_MEM_PATH_GAME_CHARS", str(used + 10))
    game.step(["DOWN"])
    agent._mem_observe(game.history)
    assert not any(line.startswith("Shortest") for line in agent._mem_turn_lines(game.valid_actions()))
    assert agent._route_chars_used == used and agent._route_cursor == 6
    monkeypatch.delenv("ARC3_MEM_PATH_GAME_CHARS")
    monkeypatch.setenv("ARC3_MEM_PATH_AFTER_TOKENS", "100")
    game.step(["UP"])
    agent._mem_observe(game.history)
    assert not any(line.startswith("Shortest") for line in agent._mem_turn_lines(game.valid_actions()))


def test_path_alone_puts_its_line_in_the_turn_message(monkeypatch):
    agent = _agent(monkeypatch, ARC3_MEM_PATH="1")
    agent._route_cursor = agent._route_chars_used = 0
    game = memory_sim.FakeGame()
    game.step(["UP", "UP"])
    agent._mem_observe(game.history)
    assert agent._mem_turn_lines(game.valid_actions()) == [
        HEAD + "2 actions [U2]; this attempt so far: 2 actions. "
        "Route length after each action since the last message: 1 2."
    ]
    monkeypatch.delenv("ARC3_MEM_PATH")
    game.step(["UP"])
    agent._mem_observe(game.history)
    assert agent._mem_turn_lines(game.valid_actions()) == []


def test_level_path_globals_refresh_at_the_start_and_after_each_action(monkeypatch, tmp_path):
    agent, state_path, _ = _sandbox_agent(monkeypatch, tmp_path, echo=False, ARC3_MEM_PATH="1")
    code = (
        "print(level_path, level_path_len, level_path_varied, level_path_before_death)\n"
        "r = action(['RIGHT', 'UP'])\n"
        "print(level_path, level_path_len)\n"
        "r = action('DOWN')\n"
        "print(level_path, level_path_len)\n"
    )
    out = _stdout(agent._run_python_tool(state_path, {"code": code}))
    assert out.splitlines() == [
        "[] 0 [] None",
        "[route] route 0->2 (1 2)",
        "['RIGHT', 'UP'] 2",
        "[route] route 1 (known board)",
        "['RIGHT'] 1",
    ]


def test_flag_off_sends_no_level_path(monkeypatch, tmp_path):
    agent, state_path, _ = _sandbox_agent(monkeypatch, tmp_path)
    code = "try:\n    level_path\nexcept NameError:\n    print('no level_path')\n"
    assert _stdout(agent._run_python_tool(state_path, {"code": code})) == "no level_path\n"


def test_each_action_writes_one_route_row(monkeypatch, tmp_path):
    import json

    from inference.agent.runtime_state import load_runtime_state

    agent, state_path, game = _sandbox_agent(monkeypatch, tmp_path, ARC3_MEM_PATH="1")
    agent._run_python_tool(state_path, {"code": "_ = action(['RIGHT', 'RIGHT', 'LEFT'])\n_ = action('UP')\n"})
    log_path = tmp_path / "level_path.jsonl"  # one game in the run: the run-level name
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [(r["act"], r["sig"], r["d"], r["k"], r["change"], r["path"]) for r in rows] == [
        (1, "RIGHT", 1, 1, "new", "R"), (2, "RIGHT", 2, 2, "new", "R2"),
        (3, "LEFT", 1, 3, "known", "R"), (4, "UP", 2, 4, "new", "R U"),
    ]
    assert set(rows[0]) == {"act", "level", "attempt", "i", "sig", "board", "d", "k", "change", "varied", "path"}
    assert len(rows[0]["board"]) == 10 and rows[0]["level"] == 3 and rows[0]["varied"] == []
    agent._mem_observe(load_runtime_state(state_path)[1])
    assert len(log_path.read_text().splitlines()) == 4  # a turn message writes nothing twice


def test_flag_off_writes_no_route_file(monkeypatch, tmp_path):
    agent, state_path, _ = _sandbox_agent(monkeypatch, tmp_path)
    agent._run_python_tool(state_path, {"code": "_ = action('RIGHT')\n"})
    assert not list(tmp_path.rglob("*level_path*"))


# ---------------------------------------------------------------- through analyze()

def _dump(fail: int = -1, **extra) -> dict:
    import json
    import os
    import subprocess

    env = memory_sim.scenario_env("kaggle-memv2-both")
    env.update({"ARC3_MEM_PATH": "1", **extra})
    proc = subprocess.run(
        [sys.executable, "tests/memory_sim.py", "dump", str(fail)], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=300, preexec_fn=lambda: os.nice(10),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _route_of(message: dict) -> str:
    return next((line for line in message["text"].splitlines() if line.startswith(HEAD)), "")


def test_route_line_sits_with_the_turn_facts_and_comes_back_after_a_rollback():
    dump = _dump(fail=19)
    openers = []
    start = 0
    for count in dump["turn_requests"]:
        if count:
            openers.append(dump["requests"][start][-1])
        start += count
    with_line = [m for m in openers if _route_of(m)]
    assert len(with_line) >= 20
    for message in with_line:
        text = message["text"]
        at = text.index(HEAD)
        assert text.index("Valid actions right now:") < at < text.index("Only tool: `python`")
        if "Seen before" in text:
            assert text.index("Seen before") < at
        if "<memory_update>" in text:
            assert at < text.index("<memory_update>")
    failed = sum(dump["turn_requests"][:19])
    # the failed turn executed nothing: the next opener shows the same rows again
    assert _route_of(dump["requests"][failed][-1]) == _route_of(dump["requests"][failed + 1][-1]) != ""


# ---------------------------------------------------------------- replay anchor

R2_EVENTS = memory_sim.TRANSCRIPT_ROOT / "fz-ledger-on-trim-r2" / "artifacts"


class StartsRecorder(FactRecorder):
    """Records every start board each level's graph had: the one it began
    with and each one a RESET moved it to, whether or not a search ran."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.starts: dict[int, set[str]] = {}

    def _note_start(self) -> None:
        if self.graph is not None:
            self.starts.setdefault(self.graph.level, set()).add(self.graph.start_h)

    def _begin_level(self, *args, **kwargs) -> None:
        super()._begin_level(*args, **kwargs)
        self._note_start()

    def _route_reset(self, *args, **kwargs) -> None:
        super()._route_reset(*args, **kwargs)
        self._note_start()

    def levels_with_two_starts(self) -> int:
        return sum(1 for boards in self.starts.values() if len(boards) > 1)


def test_the_replay_sees_a_start_board_that_moved_before_any_link():
    rec = StartsRecorder(route=True)
    events: list = []
    rec._begin_level(1, None, events, start_h="S", announce=False)
    rec.step(1, "SPACE", "SPACE", NOOP, "S", "S", events=events)  # no link, no search
    rec.reset_step(1, "T", None, manual=True, events=events)
    assert rec.levels_with_two_starts() == 1 and rec.starts == {1: {"S", "T"}}


def _replay_r2() -> dict:
    """Feed the r2 ledger run's frames through FactRecorder(route=True), one
    game at a time, comparing each route with the attempt's loop-free walk."""
    import glob
    import json
    import time

    from inference.agent.memory_facts import loop_free
    from inference.agent.runtime_state import Frame, HistoryEntry

    out = {"cur_rows": 0, "shorter_than_loop_free": 0, "clears": 0, "route_clears": [],
           "graph_full": 0, "start_boards": 0, "bfs_seconds": 0.0, "actions": 0, "route_errors": 0}

    class Replay(StartsRecorder):
        def step(self, level, sig, key, cls, pre_h, post_h, **kwargs):
            super().step(level, sig, key, cls, pre_h, post_h, **kwargs)
            row = self.route_rows[-1]
            out["actions"] += 1
            if row.change in (DEATH, CLEAR) or row.d is None:
                return
            out["cur_rows"] += 1
            if row.d < len(loop_free(self.attempts.steps, self.attempts.start_h)):
                out["shorter_than_loop_free"] += 1

    started = time.process_time()
    for path in sorted(glob.glob(str(R2_EVENTS / "*_events.jsonl"))):
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
        rec = Replay(border=4, route=True)
        real_bfs = LevelGraph.bfs

        def timed_bfs(graph, _real=real_bfs):
            t0 = time.process_time()
            _real(graph)
            out["bfs_seconds"] += time.process_time() - t0

        LevelGraph.bfs = timed_bfs
        try:
            events = rec.feed(entries)
        finally:
            LevelGraph.bfs = real_bfs
        game = Path(path).name.split("-")[0]
        for event in events:
            if event.kind == "clear":
                out["clears"] += 1
                if "winning" in event.data:
                    out["route_clears"].append((game, event.level, len(event.data["path"]), event.data["winning"]))
        out["graph_full"] += rec.stats["graph_full"]
        out["start_boards"] += rec.levels_with_two_starts()
        out["route_errors"] += rec.stats["route_errors"]
    out["cpu_seconds"] = time.process_time() - started
    return out


def test_replay_anchor_on_the_r2_run():
    if not list(R2_EVENTS.glob("*_events.jsonl")):
        pytest.skip("the r2 run's events.jsonl frames are not on this machine (ARC3_MEM_REPLAY_RUNS)")
    got = _replay_r2()
    print(json.dumps(got))
    assert got["clears"] == 112
    assert [(g, level) for g, level, _, _ in got["route_clears"]] == [("g50t", 1), ("sc25", 3)]
    share = 100 * got["shorter_than_loop_free"] / got["cur_rows"]
    assert abs(share - 2.2) <= 0.3, share
    assert got["graph_full"] == 0 and got["start_boards"] == 0 and got["route_errors"] == 0
    assert got["cpu_seconds"] < 60, got["cpu_seconds"]
