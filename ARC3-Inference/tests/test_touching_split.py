"""ARC3_TOUCHING_SPLIT: a turn-message line when same-colour objects of the
level start touch and segmentation shows them as one node."""
import json
from pathlib import Path

import pytest

from inference.agent import board_facts as bf
from inference.agent import tool_agent
from inference.agent.runtime_state import Frame, HistoryEntry
from inference.utils.grid_utils import ARC_COLOR_CHARS

BOARDS = json.loads((Path(__file__).parent / "fixtures" / "touching_split_boards.json").read_text(encoding="utf-8"))


def _grid(rows):
    return tuple(tuple(ARC_COLOR_CHARS.index(ch) for ch in row) for row in rows)


def _blank(cells=None, size=12):
    rows = [[0] * size for _ in range(size)]
    for (r, c), v in (cells or {}).items():
        rows[r][c] = v
    return tuple(map(tuple, rows))


def _block(r0, c0, h, w, value=10):
    return {(r, c): value for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _facts(*grids, level=1):
    facts = bf.BoardFacts()
    facts.feed([HistoryEntry("RIGHT", Frame(grid=g, step=i, level=level)) for i, g in enumerate(grids)])
    return facts


@pytest.mark.parametrize("key", ["ours_m0r0_l1", "lab_m0r0_l6"])
def test_logged_boards_give_the_logged_line(key):
    case = BOARDS[key]
    assert _facts(_grid(case["start"]), _grid(case["now"])).touching_lines() == [case["line"]]


def test_no_line_at_the_level_start_or_without_a_touch():
    apart = _blank({**_block(2, 2, 2, 2), **_block(2, 7, 2, 2)})
    assert _facts(apart).touching_lines() == []
    assert _facts(apart, apart).touching_lines() == []


def test_two_squares_pushed_together_give_one_line():
    apart = _blank({**_block(2, 2, 2, 2), **_block(2, 7, 2, 2)})
    touching = _blank({**_block(2, 2, 2, 2), **_block(2, 4, 2, 2)})
    assert _facts(apart, touching).touching_lines() == [
        "Same-colour objects that touch, listed separately: sky blue S x2 — a 2x2 at rows 2-3, cols 2-3 and "
        "a 2x2 at rows 2-3, cols 4-5, touching (segmentation shows them as one 4x2 node). Sky blue pixel "
        "total: 8, the same as at the level start (two 2x2 squares)."
    ]


def test_a_grown_or_merged_colour_and_single_pixels_give_no_line():
    apart = _blank({**_block(2, 2, 2, 2), **_block(2, 7, 2, 2)})
    grown = _blank({**_block(2, 2, 2, 2), **_block(2, 4, 2, 2), **_block(6, 6, 1, 1)})
    assert _facts(apart, grown).touching_lines() == []  # the colour total changed
    pixels = _blank({(2, 2): 10, (2, 6): 10})
    assert _facts(pixels, _blank({(2, 2): 10, (2, 3): 10})).touching_lines() == []  # no objects


def test_a_new_level_restarts_the_comparison():
    apart = _blank({**_block(2, 2, 2, 2), **_block(2, 7, 2, 2)})
    touching = _blank({**_block(2, 2, 2, 2), **_block(2, 4, 2, 2)})
    facts = bf.BoardFacts()
    entries = [HistoryEntry("RIGHT", Frame(grid=apart, step=0, level=1)),
               HistoryEntry("RIGHT", Frame(grid=touching, step=1, level=2))]
    facts.feed(entries)
    assert facts.start == 1 and facts.touching_lines() == []


def _prompt(monkeypatch, tmp_path, *, enabled, resuming=False):
    if enabled:
        monkeypatch.setenv("ARC3_TOUCHING_SPLIT", "1")
    else:
        monkeypatch.delenv("ARC3_TOUCHING_SPLIT", raising=False)
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", "off")
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    agent._ensure_session(tmp_path / "state.json")
    agent._resume_after_yield = resuming
    case = BOARDS["ours_m0r0_l1"]
    start = Frame(grid=_grid(case["start"]), step=1, level=1)
    now = Frame(grid=_grid(case["now"]), step=20, level=1)
    entries = [HistoryEntry("INIT", start), HistoryEntry("UP", now)]
    summary = {"executed_count": 1, "executed_actions": ["UP"]}
    text = agent._build_user_prompt(now.step, valid_actions=["UP", "DOWN", "LEFT", "RIGHT"], current_frame=now,
                                    history_entries=entries, previous_step_summary=summary)
    return text, case["line"]


def test_turn_message_carries_the_line_after_the_valid_actions(monkeypatch, tmp_path):
    text, line = _prompt(monkeypatch, tmp_path, enabled=True)
    lines = text.splitlines()
    valid = next(i for i, t in enumerate(lines) if t.startswith("Valid actions right now:"))
    assert line in lines[valid + 1:]
    assert lines.index(line) > valid


@pytest.mark.parametrize("enabled, resuming", [(False, False), (True, True)])
def test_no_line_when_off_or_on_a_yield_resumption(monkeypatch, tmp_path, enabled, resuming):
    text, line = _prompt(monkeypatch, tmp_path, enabled=enabled, resuming=resuming)
    assert line not in text and "Same-colour objects that touch" not in text
