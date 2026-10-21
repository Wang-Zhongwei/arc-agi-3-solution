"""ARC3_CLEARED_BOARDS: which frame of a cleared level is which.

Pure python, no I/O. The solver's history has, for a level-clearing action,
the next level's first board, so the last frame of level N in `history` is
the board BEFORE the action that cleared it. A model that reads it as the
winning board sees every win one move short: vc33 L4 (stall-fix-10x, 28
near-idle requests) concluded its correct level recap was "wrong /
hallucinated" (request 48) after reading the earlier levels' last frames,
and a code grep finds models reading those frames in 18 game-runs of 9
runs. The solver keeps the board the clearing action produced
(``solver._cleared_board``, exposed as ``cleared_boards[N]`` in the
sandbox); this module writes the line that says so, for the turn message
after the clear and that level's ``<previous_levels>`` entry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

_CLICK_RE = re.compile(r"MOUSE\s*\(\s*row\s*=\s*(-?\d+)\s*,\s*col\s*=\s*(-?\d+)\s*\)", re.I)


@dataclass(frozen=True)
class Clear:
    """One level clear read from the history."""

    level: int          # the level the action cleared
    action_num: int     # the clearing action's number
    action: str         # its display, ``MOUSE(row=56, col=46)`` or ``LEFT``
    before_step: int    # step of history's last frame of ``level``


def clears(entries: Sequence[Any]) -> list[Clear]:
    """Every level clear in the solver's history (``HistoryEntry`` with
    ``action``, ``frame`` and ``result``), oldest first."""
    out: list[Clear] = []
    for index in range(1, len(entries)):
        result = getattr(entries[index], "result", None) or {}
        if not (isinstance(result, dict) and result.get("level_completed")):
            continue
        before = getattr(entries[index - 1], "frame", None)
        number = result.get("action_num")
        if before is None or not isinstance(number, int):
            continue
        out.append(Clear(int(before.level), number, str(getattr(entries[index], "action", "") or ""),
                         int(before.step)))
    return out


def action_words(display: str) -> str:
    """``click (56, 46)`` for a click (row, col), else the action's name."""
    match = _CLICK_RE.search(display)
    return f"click ({match.group(1)}, {match.group(2)})" if match else display


def clear_line(clear: Clear, produced: bool) -> str:
    """``Level 3 was cleared by action 50, click (56, 46). history's last
    level-3 frame (step 49) is the board BEFORE that action; the board it
    produced is cleared_boards[3].`` (``produced`` False: the action showed
    no board of its own before the next level's)."""
    head = f"Level {clear.level} was cleared by action {clear.action_num}, {action_words(clear.action)}."
    if produced:
        return (f"{head} history's last level-{clear.level} frame (step {clear.before_step}) is the board "
                f"BEFORE that action; the board it produced is cleared_boards[{clear.level}].")
    return (f"{head} The board it produced was never shown; history's last level-{clear.level} frame "
            f"(step {clear.before_step}) is one action short of the clear.")
