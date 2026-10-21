"""arm/mem-unified's three facts on real boards and replies.

tests/fixtures/mem_unified_states.json.gz holds, from the Kaggle logs:
- the boards of run A's sc25 level 6 up to request 100 (actions 209-246),
  run B's m0r0 level 1 up to request 19 and uc1's m0r0 level 6 up to request
  116 (the state lab's round-2 states sc25_L6_ringA, m0r0_L1_touch and
  m0r0_L6_touch), letters WwgGcBMPRbSYOrNp;
- run A's reasoning and reply at r11l calls 95, 101 and 118 and m0r0 calls
  64 and 113 (requests 94, 100, 117, 63 and 112), whose rules the lab quoted.

The expected lines are the lab's own texts (states/round2_*.json), which the
model read in the sampled cells.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from inference.agent import board_facts as bf
from inference.agent import memory_layout as layout
from inference.agent.confirmed_rules import RuleLedger, rule_score, rule_sentences
from inference.agent.memory_ledger import MemoryLedger

FIXTURE = json.loads(gzip.open(Path(__file__).parent / "fixtures" / "mem_unified_states.json.gz", "rt").read())
LETTERS = "WwgGcBMPRbSYOrNp"

SC25_LINES = [
    "Moved since this level started, not by an arrow key: the yellow 6x6 corner frame (single Y pixels at "
    "(36,52), (36,57), (41,52), (41,57), around rows 37-40 cols 53-56) is now at (32,28), (32,33), (37,28), "
    "(37,33), around rows 33-36 cols 29-32. It moved at action 241, MOUSE(row=55, col=30), the click at which "
    "the 16-cell b/S block moved from rows 33-36 cols 17-20 to rows 37-40 cols 53-56.",
    "Boxes marked by four single yellow corner pixels: rows 33-36 cols 29-32 (4x4: 12 g cells and the 4 Y "
    "corners; the 6x6 frame surrounds it); rows 37-40 cols 53-56 (4x4: 12 g cells and the 4 Y corners); "
    "rows 40-43 cols 12-15 (4x4: a 2x2 of p at rows 41-42 cols 13-14, 6 B, 2 g and the 4 Y corners).",
]
M0R0_B_LINE = (
    "Same-colour objects that touch, listed separately: sky blue S x2 — a 5x5 at rows 9-13, cols 29-33 and "
    "a 5x5 at rows 9-13, cols 34-38, touching (segmentation shows them as one 10x5 node). Sky blue pixel total: "
    "50, the same as at the level start (two 5x5 squares)."
)
M0R0_UC1_LINE = (
    "Same-colour objects that touch, listed separately: sky blue S x2 — a 4x4 at rows 34-37, cols 14-17 and "
    "a 4x4 at rows 38-41, cols 14-17, touching (segmentation shows them as one 4x8 node). Sky blue pixel total: "
    "32, the same as at the level start (two 4x4 squares)."
)


def _entries(name: str) -> list:
    out = []
    for row in FIXTURE[name]["entries"]:
        grid = tuple(tuple(LETTERS.index(ch) for ch in line) for line in row["board"])
        out.append(NS(action=row["action"], frame=NS(grid=grid, level=row["level"]),
                      result={"action_num": row["n"]}))
    return out


def _facts(name: str) -> bf.BoardFacts:
    facts = bf.BoardFacts()
    facts.feed(_entries(name))
    return facts


# ---------------------------------------------------------------- ARC3_MEM_MOVED


def test_sc25_moved_frame_is_the_lab_text():
    facts = _facts("sc25_A")
    assert facts.start == 0 and facts.level == 6
    assert facts.moved_lines() == SC25_LINES
    assert facts.touching_lines() == []


def test_moved_lines_are_capped_and_empty_at_a_level_start():
    facts = _facts("sc25_A")
    assert len(facts.moved_lines(max_lines=3)) <= 3
    first = bf.BoardFacts()
    first.feed(_entries("sc25_A")[:1])
    assert first.moved_lines() == [] and first.touching_lines() == []


def test_a_marker_recoloured_into_place_is_not_a_move():
    # uc1 m0r0 L6: the dot moved by arrow keys while selected (yellow) and
    # turned blue again at a deselecting click; that click did not move it
    assert _facts("m0r0_uc1").moved_lines() == []


def test_arrow_key_moves_and_clicked_markers_are_left_out():
    blank = [[0] * 12 for _ in range(12)]

    def grid(cells):
        rows = [list(r) for r in blank]
        for (r, c), v in cells.items():
            rows[r][c] = v
        return tuple(map(tuple, rows))

    block = {(r, c): 9 for r in (8, 9) for c in (7, 8, 9)}  # 6 cells: not a marker
    start = grid({(2, 2): 11, **block})

    def run(action, cells):
        entries = [NS(action="", frame=NS(grid=start, level=1), result={}),
                   NS(action=action, frame=NS(grid=grid(cells), level=1), result={"action_num": 1})]
        facts = bf.BoardFacts()
        facts.feed(entries)
        return facts.moved_lines()

    assert run("LEFT", {(2, 4): 11, **block}) == []
    assert run("MOUSE(row=2, col=4)", {(2, 4): 11, **block}) == []      # the marker sits on the click
    line, = run("MOUSE(row=8, col=8)", {(2, 4): 11, **block})
    assert line == ("Moved since this level started, not by an arrow key: the yellow pixel at (2,2) is now at "
                    "(2,4). It moved at action 1, MOUSE(row=8, col=8).")
    line, = run("SPACE", {(2, 4): 11, **{(r + 1, c): 9 for r, c in block}})
    assert line.endswith("It moved at action 1, SPACE, the action at which the 6-cell b block moved from "
                         "rows 8-9 cols 7-9 to rows 9-10 cols 7-9.")


def test_corner_frames_need_unambiguous_pixels():
    nodes = bf.components(tuple(tuple(1 if (r % 3 == 0 and c % 3 == 0) else 0 for c in range(10)) for r in range(10)))
    assert bf.corner_frames(nodes) == []  # a lattice of dots marks no box
    one = tuple(tuple(1 if (r, c) in {(1, 1), (1, 5), (4, 1), (4, 5)} else 0 for c in range(8)) for r in range(8))
    assert bf.corner_frames(bf.components(one)) == [bf.Frame(1, 1, 1, 4, 5)]


# ---------------------------------------------------------------- ARC3_TOUCHING_SPLIT


def test_m0r0_touching_lines_are_the_lab_texts():
    assert _facts("m0r0_B").touching_lines() == [M0R0_B_LINE]
    assert _facts("m0r0_uc1").touching_lines() == [M0R0_UC1_LINE]


def test_touching_needs_an_unchanged_total_and_objects_of_four_cells():
    def grid(cells):
        return tuple(tuple(cells.get((r, c), 0) for c in range(12)) for r in range(12))

    two = {(r, c): 10 for r in (2, 3) for c in (2, 3, 6, 7)}
    start = NS(action="", frame=NS(grid=grid(two), level=1), result={})
    touching = {(r, c): 10 for r in (2, 3) for c in (2, 3, 4, 5)}
    facts = bf.BoardFacts()
    facts.feed([start, NS(action="RIGHT", frame=NS(grid=grid(touching), level=1), result={"action_num": 1})])
    assert facts.touching_lines() == [
        "Same-colour objects that touch, listed separately: sky blue S x2 — a 2x2 at rows 2-3, cols 2-3 and a "
        "2x2 at rows 2-3, cols 4-5, touching (segmentation shows them as one 4x2 node). Sky blue pixel total: 8, "
        "the same as at the level start (two 2x2 squares)."]
    grown = {**touching, (4, 2): 10}
    facts = bf.BoardFacts()
    facts.feed([start, NS(action="RIGHT", frame=NS(grid=grid(grown), level=1), result={"action_num": 1})])
    assert facts.touching_lines() == []  # the colour total changed
    pixels = {(2, 2): 10, (2, 6): 10}
    facts = bf.BoardFacts()
    facts.feed([NS(action="", frame=NS(grid=grid(pixels), level=1), result={}),
                NS(action="RIGHT", frame=NS(grid=grid({(2, 2): 10, (2, 3): 10}), level=1), result={"action_num": 1})])
    assert facts.touching_lines() == []  # two single pixels are no objects


# ---------------------------------------------------------------- ARC3_MEM_RULES

R11L_L4_CALL95 = "So each knot must go to the ring with the SAME color set ✓✓."
R11L_L4_CALL101 = "So the match is by color SET, not counts ✓."
R11L_L5_CALL118 = ("So the mechanic: moving the white knot onto a colored blob absorbs the blob's colors (the knot "
                   "becomes half W/half R as the blob was), and the blob disappears.")
R11L_L5_TELEPORT = "and the knot's path doesn't matter (teleport) ✓."
M0R0_L3_CALL64 = ("So the selected (yellow) dot is a movable object, moving in 4-cell steps, while the blocks are "
                  "frozen.")


def _reply(key: str) -> str:
    reply = FIXTURE["replies"][key]
    return reply["reasoning"] + "\n" + (reply["content"] or "")


@pytest.mark.parametrize("key, rule", [
    ("r11l_A_94", R11L_L4_CALL95),
    ("r11l_A_100", R11L_L4_CALL101),
    ("r11l_A_117", R11L_L5_CALL118),
    ("r11l_A_117", R11L_L5_TELEPORT),
    ("m0r0_A_63", M0R0_L3_CALL64),
])
def test_the_rules_the_lab_quoted_are_captured(key, rule):
    assert rule in rule_sentences(_reply(key))


def test_an_unmarked_rule_is_not_captured():
    # m0r0 call 113's "when a dot is selected, only the dot moves (blocks
    # frozen)." carries no confirmation mark: by design the ledger keeps only
    # what the model marked as confirmed
    assert not any("only the dot moves" in s for s in rule_sentences(_reply("m0r0_A_112")))


def test_questions_plans_hedges_and_coordinates_are_dropped():
    text = ("Does SPACE always do nothing ✓? Let me confirm that UP always moves it ✓. "
            "The block probably always moves right, confirmed. "
            "The dot moved from (1,2) to (3,4) so it always moves ✓. "
            "Hmm, the red door always opens when the key is taken ✓.")
    assert rule_sentences(text) == ["the red door always opens when the key is taken ✓."]


def test_ledger_keeps_the_best_three_per_level_and_newer_restatements():
    ledger = RuleLedger()
    ledger.absorb(4, _reply("r11l_A_94"))
    ledger.absorb(4, _reply("r11l_A_100"))
    rules = ledger.rules_for(4)
    assert len(rules) == 3 and R11L_L4_CALL95 in rules
    assert rules == sorted(rules, key=lambda s: [n for n, t in ledger.levels[4] if t == s][0])
    ledger.absorb(2, "Confirmed: RIGHT always stretches the block two cells to the right ✓.")
    ledger.absorb(2, "Confirmed: RIGHT always stretches the block two cells right, then it settles ✓.")
    assert ledger.rules_for(2) == ["Confirmed: RIGHT always stretches the block two cells right, then it settles ✓."]
    assert rule_score(R11L_L4_CALL95) > rule_score("Then 142 (RIGHT) → (4,6) ✓ mirrored, (6,2) ✓ normal.")


def test_rules_ride_in_the_frozen_entry_and_this_level_line():
    ledger = MemoryLedger(card_mode=True, solved_mode=True, debrief_mode=True, recap_ask=True)
    ledger.layout_mode = ledger.legible = True
    ledger.rules = RuleLedger()
    ledger.rules.absorb(4, _reply("r11l_A_94"))
    ledger.add_solved(4, ["MOUSE(1,2)"], 3, 0)
    block, = ledger.layout_stage([4])
    assert f'  Rules you confirmed: "' in block and R11L_L4_CALL95 in block
    ledger.rules.absorb(4, "Confirmed: the knot always teleports to the clicked endpoint's mean ✓.")
    assert ledger.prev_entries[4].confirmed != ledger.rules.rules_for(4)  # frozen at the stage
    line = layout.this_level_rules_line(ledger.rules.rules_for(4))
    assert line.startswith('Rules you confirmed on this level: "') and line.endswith('"')
    assert layout.this_level_rules_line([]) == ""


def test_this_level_extra_lines_follow_the_notes_outside_the_cap():
    kwargs = dict(level=2, attempt=1, actions=3, deaths=0, rows=["UP (3 presses on this level): x."] * 40,
                  hazards=[], seen="", notes=[])
    plain = layout.render_this_level(**kwargs)
    with_extra = layout.render_this_level(**kwargs, extra=["EXTRA ONE", "", "EXTRA TWO"])
    assert with_extra == plain.replace("\n</this_level>", "\nEXTRA ONE\nEXTRA TWO\n</this_level>")
    assert layout.render_this_level(**kwargs, extra=[]) == plain
