"""What the model reads: plain character grids and per-cell deltas.

Three renderings, all one character per cell so a coding model can read them the way it reads
source, and none of them nested JSON:

- `grid_text`   the settled frame. Background is `.`, every other colour its ARC letter.
- `change_text` the same shape with only the cells that changed showing their NEW colour, so the
                shape of the change is visible at a glance.
- `cell_deltas` the exact list, `{"step": .., "action": .., "deltas": [{"r":..,"c":..,"old":..,
                "new":..}]}`, with the raw colour numbers, for anything that needs arithmetic.

Row and column indices are printed because coordinates are load-bearing here: a claim names
`pos[t3].r`, a click names x and y, and counting 64 rows by eye is exactly the kind of work the
harness exists to do for the model.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional, Sequence

from .types import COLOUR_CHARS, COLOUR_NAMES, Grid

BACKGROUND_CHAR = "."
MAX_DELTA_CELLS = 120


def colour_char(value: int, background: Optional[int]) -> str:
    if background is not None and int(value) == int(background):
        return BACKGROUND_CHAR
    return COLOUR_CHARS[max(0, min(15, int(value)))]


def _ruler(width: int, indent: int) -> list[str]:
    """Two header lines giving each column's tens and units digit."""
    pad = " " * indent
    tens = "".join(str((c // 10) % 10) if c % 5 == 0 else " " for c in range(width))
    units = "".join(str(c % 10) for c in range(width))
    return [pad + tens, pad + units]


def grid_text(grid: Grid, background: Optional[int] = None, *, indices: bool = True,
              only: Optional[Iterable[tuple[int, int]]] = None, blank_rows: bool = False) -> str:
    """The grid as characters. `only` restricts what is drawn; everything else becomes `.`."""
    if not grid:
        return "(empty)"
    keep = None if only is None else set(only)
    height, width = len(grid), len(grid[0])
    indent = 3 if indices else 0
    lines = _ruler(width, indent) if indices else []
    for r, row in enumerate(grid):
        chars = []
        drawn = False
        for c, value in enumerate(row):
            if keep is not None and (r, c) not in keep:
                chars.append(BACKGROUND_CHAR)
                continue
            ch = colour_char(value, background)
            drawn = drawn or ch != BACKGROUND_CHAR
            chars.append(ch)
        if blank_rows and not drawn:
            continue
        lines.append((f"{r:2d} " if indices else "") + "".join(chars))
    return "\n".join(lines)


def change_text(before: Grid, after: Grid, background: Optional[int] = None, *,
                indices: bool = True) -> str:
    """Only the cells that changed, showing the colour they became."""
    changed = changed_cells(before, after)
    if not changed:
        return "(nothing changed)"
    return grid_text(after, background, indices=indices, only={(r, c) for r, c, _o, _n in changed})


def changed_cells(before: Grid, after: Grid) -> list[tuple[int, int, int, int]]:
    """[(row, col, old, new)] for every cell that differs, in reading order."""
    out: list[tuple[int, int, int, int]] = []
    for r, (row_b, row_a) in enumerate(zip(before, after)):
        for c, (old, new) in enumerate(zip(row_b, row_a)):
            if old != new:
                out.append((r, c, int(old), int(new)))
    return out


def cell_deltas(before: Grid, after: Grid, *, step: int, action: str,
                effect: str = "", max_cells: int = MAX_DELTA_CELLS) -> str:
    """The per-cell change list as one JSON object on one line.

    Capped: a level transition repaints the whole board, and 4096 entries would be the entire
    prompt. The change grid still shows the full shape when the list is cut.
    """
    changed = changed_cells(before, after)
    shown = changed[:max_cells]
    payload: dict[str, Any] = {
        "step": int(step),
        "action": action if not effect else f"{action} ({effect})",
        "changed_cells": len(changed),
        "deltas": [{"r": r, "c": c, "old": o, "new": n} for r, c, o, n in shown],
    }
    if len(changed) > len(shown):
        payload["omitted"] = len(changed) - len(shown)
    return json.dumps(payload, separators=(",", ":"))


def legend(grid: Grid, background: Optional[int] = None, *, max_entries: int = 12) -> str:
    """Only the colours actually on the board, as `letter=name`, plus what `.` stands for."""
    seen: dict[int, int] = {}
    for row in grid:
        for value in row:
            seen[int(value)] = seen.get(int(value), 0) + 1
    parts = []
    if background is not None:
        parts.append(f"{BACKGROUND_CHAR}={COLOUR_NAMES.get(int(background), background)} (background)")
    for value, _count in sorted(seen.items(), key=lambda kv: -kv[1]):
        if background is not None and value == int(background):
            continue
        parts.append(f"{COLOUR_CHARS[max(0, min(15, value))]}={COLOUR_NAMES.get(value, value)}")
        if len(parts) >= max_entries:
            break
    return " ".join(parts)


def crop(grid: Grid, box: tuple[int, int, int, int], background: Optional[int] = None,
         *, margin: int = 2) -> str:
    """A small window of the board with absolute indices, for looking closely at one place."""
    r0, c0, r1, c1 = box
    r0 = max(0, r0 - margin); c0 = max(0, c0 - margin)
    r1 = min(len(grid) - 1, r1 + margin); c1 = min(len(grid[0]) - 1, c1 + margin)
    lines = ["    " + "".join(str(c % 10) for c in range(c0, c1 + 1))]
    for r in range(r0, r1 + 1):
        lines.append(f"{r:3d} " + "".join(colour_char(grid[r][c], background) for c in range(c0, c1 + 1)))
    return "\n".join(lines)
