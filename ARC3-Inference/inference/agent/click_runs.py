"""Click-run outcome facts (ARC3_CLICK_RUNS).

Pure python (standard library and ``board_facts``), no I/O. ``tool_agent`` feeds the
solver's full history (``HistoryEntry`` with ``action``, ``frame`` and
``result``) at each new turn message and places the line.

A click run is a stretch of consecutive MOUSE clicks inside one executed
sequence (the actions of one turn): it ends at any other action, a RESET, a
level change, a game over and at the end of the sequence. The clicked cells
of a run are each clicked cell plus the segmentation node holding it before
and after its click (a keypad button lights up as one object), when that
node has at most ``NODE_CELLS`` cells. A run is **effective** when, at any
of its clicks, a cell inside the board area (more than ``BAND`` cells from
the edge, where budget bars sit) changed outside the clicked cells, or a
click played a multi-frame animation, or the run ended the level or the
attempt.

When the last run of the sequence just executed has two or more clicks and
is not effective, and the game (any level) already had an effective run of
the same length, the turn message gets one fact line: the dead run, what its
clicked cells did, and up to ``MAX_LISTED`` earlier effective runs of two or
more clicks (one per click sequence, most similar first: same length, then
fewest differing cells, then newest), each with a short effect phrase from
the board diff. sc25 L5 in run U: the model typed a 4-click keypad pattern
with 3 cells wrong (actions 127-130), saw only the 4 cells light up,
concluded "plus does not grow the cap" and held that for ~71k tokens, while
its own plus at actions 90-93 had shrunk the cap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from inference.agent.board_facts import (
    MAX_GROUP_CELLS,
    Node,
    _group_words,
    colour_name,
    components,
    corner_frames,
    letter,
    moved_groups,
    rows_cols,
)

Grid = Sequence[Sequence[int]]
Cell = tuple[int, int]

BAND = 4
NODE_CELLS = 64
MIN_CLICKS = 2
MAX_LISTED = 3
MAX_PHRASES = 2
_CLICK_RE = re.compile(r"MOUSE\s*\(\s*row\s*=\s*(-?\d+)\s*,\s*col\s*=\s*(-?\d+)\s*\)", re.I)


def click_cell(display: Any) -> Cell | None:
    """``(50, 30)`` for ``MOUSE(row=50, col=30)``; None for any other action."""
    match = _CLICK_RE.search(str(display or ""))
    return (int(match.group(1)), int(match.group(2))) if match else None


def _cells(cells: Sequence[Cell]) -> str:
    return " ".join(f"({r},{c})" for r, c in cells)


def _span(first: int, last: int) -> str:
    return f"action {first}" if first == last else f"actions {first}-{last}"


def _inside(cell: Cell, height: int, width: int, band: int) -> bool:
    r, c = cell
    return band <= r < height - band and band <= c < width - band


def node_at(grid: Grid, cell: Cell, limit: int = NODE_CELLS) -> set[Cell] | None:
    """The 4-connected same-colour cells around ``cell``; None when the cell is
    off the grid or the node has more than ``limit`` cells."""
    height = len(grid)
    width = len(grid[0]) if height else 0
    r0, c0 = cell
    if not (0 <= r0 < height and 0 <= c0 < width):
        return None
    colour = grid[r0][c0]
    seen = {cell}
    stack = [cell]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in seen or not (0 <= nr < height and 0 <= nc < width) or grid[nr][nc] != colour:
                continue
            seen.add((nr, nc))
            if len(seen) > limit:
                return None
            stack.append((nr, nc))
    return seen


def changed_cells(before: Grid, after: Grid) -> set[Cell]:
    out: set[Cell] = set()
    for r, (row_a, row_b) in enumerate(zip(before, after)):
        if row_a == row_b:
            continue
        for c, (a, b) in enumerate(zip(row_a, row_b)):
            if a != b:
                out.add((r, c))
    return out


@dataclass(frozen=True)
class ClickRun:
    """One run: history indexes ``start``..``end`` (the clicks), the board
    before it at ``start - 1``."""

    start: int
    end: int
    first: int
    last: int
    level: int
    clicks: tuple[Cell, ...]
    region: frozenset
    outside: int
    lit: bool
    animated: bool
    ended: bool

    @property
    def effective(self) -> bool:
        return bool(self.outside) or self.animated or self.ended

    @property
    def size(self) -> int:
        return len(self.clicks)


def _grid(entry: Any) -> Grid | None:
    frame = getattr(entry, "frame", None)
    grid = getattr(frame, "grid", None)
    return grid if grid else None


def _level(entry: Any) -> int:
    frame = getattr(entry, "frame", None)
    return int(getattr(frame, "level", 0) or 0)


def _result(entry: Any) -> dict:
    result = getattr(entry, "result", None)
    return result if isinstance(result, dict) else {}


def _action_num(entry: Any, index: int) -> int:
    number = _result(entry).get("action_num")
    return int(number) if isinstance(number, int) else index


def _animated(entry: Any) -> bool:
    animation = _result(entry).get("animation")
    if not isinstance(animation, dict):
        return False
    try:
        return int(animation.get("frames") or 0) >= 2
    except (TypeError, ValueError):
        return False


class ClickRuns:
    """The game's click runs, recomputed from the history each turn; runs
    already measured are kept by their action span."""

    def __init__(self) -> None:
        self.sequence_starts: set[int] = set()
        self._measured: dict[tuple[int, int], ClickRun] = {}
        self._phrases: dict[tuple[int, int], str] = {}
        self._length = 0

    # ---- runs

    def runs(self, entries: Sequence[Any]) -> list[ClickRun]:
        spans: list[tuple[int, int]] = []
        start = None
        for index in range(1, len(entries)):
            entry = entries[index]
            cell = click_cell(getattr(entry, "action", ""))
            joins = (
                cell is not None and start is not None
                and _action_num(entry, index) not in self.sequence_starts
                and _level(entries[index - 1]) == _level(entries[start - 1])
                and not _ends(entries[index - 1])
            )
            if joins:
                continue
            if start is not None:
                spans.append((start, index - 1))
            start = index if cell is not None else None
        if start is not None:
            spans.append((start, len(entries) - 1))
        return [self._measure(entries, a, b) for a, b in spans]

    def _measure(self, entries: Sequence[Any], start: int, end: int) -> ClickRun:
        key = (_action_num(entries[start], start), _action_num(entries[end], end))
        known = self._measured.get(key)
        if known is not None:
            return known
        clicks = tuple(click_cell(getattr(entries[i], "action", "")) for i in range(start, end + 1))
        region: set[Cell] = set()
        for offset, index in enumerate(range(start, end + 1)):
            cell = clicks[offset]
            region.add(cell)
            for grid in (_grid(entries[index - 1]), _grid(entries[index])):
                if grid is not None:
                    region |= node_at(grid, cell) or set()
        outside: set[Cell] = set()
        lit = reshaped = False
        for index in range(start, end + 1):
            pre, post = _grid(entries[index - 1]), _grid(entries[index])
            if pre is None or post is None:
                continue
            if len(pre) != len(post) or len(pre[0]) != len(post[0]):
                reshaped = True
                continue
            height, width = len(post), len(post[0])
            changed = changed_cells(pre, post)
            lit = lit or bool(changed & region)
            outside |= {cell for cell in changed if cell not in region and _inside(cell, height, width, BAND)}
        run = ClickRun(
            start=start, end=end, first=key[0], last=key[1], level=_level(entries[start - 1]),
            clicks=clicks, region=frozenset(region), outside=len(outside), lit=lit,
            animated=any(_animated(entries[i]) for i in range(start, end + 1)),
            ended=reshaped or any(_ends(entries[i]) for i in range(start, end + 1)),
        )
        self._measured[key] = run
        return run

    # ---- the fact

    def fact_line(self, entries: Sequence[Any], first_action: int | None, last_action: int | None) -> str:
        """The line for the sequence of actions ``first_action``..``last_action``
        just executed ("" when it does not apply). Records the sequence start."""
        if first_action is None or last_action is None or len(entries) < 2:
            return ""
        if len(entries) < self._length:
            # a shorter history is another game's: start over
            self.__init__()
        self._length = len(entries)
        self.sequence_starts.add(int(first_action))
        runs = self.runs(entries)
        mine = [run for run in runs if first_action <= run.last <= last_action]
        if not mine:
            return ""
        dead = mine[-1]
        if dead.size < MIN_CLICKS or dead.effective:
            return ""
        earlier = [run for run in runs if run.last < dead.first and run.effective and run.size >= MIN_CLICKS]
        if not any(run.size == dead.size for run in earlier):
            # replay of runs U and Ucore: any earlier effective run fired 16
            # times, 10 of them in ft09, where every click only recolours
            # the clicked cell and the effective runs are level clears
            return ""
        return self._line(entries, dead, _ranked(dead, earlier))

    def _line(self, entries: Sequence[Any], dead: ClickRun, listed: list[ClickRun]) -> str:
        text = f"Click run at {_span(dead.first, dead.last)}, {_cells(dead.clicks)}: {_dead_words(entries, dead)}"
        parts = []
        for run in listed:
            where = _span(run.first, run.last)
            if run.level != dead.level:
                where += f" on level {run.level}"
            parts.append(f"{_cells(run.clicks)} at {where} ({self._effect(entries, run)})")
        text += " Earlier click runs that changed more than the clicked cells or animated: " + "; ".join(parts) + "."
        top = listed[0]
        if top.size == dead.size:
            text += f" {_compare_words(dead, top)}"
        return text

    def _effect(self, entries: Sequence[Any], run: ClickRun) -> str:
        key = (run.first, run.last)
        if key not in self._phrases:
            try:
                self._phrases[key] = effect_phrase(entries, run)
            except Exception:  # noqa: BLE001 - a phrase is an extra
                self._phrases[key] = "the board changed outside the clicked cells"
        return self._phrases[key]


def _ends(entry: Any) -> bool:
    result = _result(entry)
    name = str(result.get("action_name") or getattr(entry, "action", "") or "").strip().upper()
    return bool(result.get("game_over") or result.get("level_completed") or name == "RESET")


def _differ(a: ClickRun, b: ClickRun) -> int:
    return len(set(a.clicks) ^ set(b.clicks))


def _ranked(dead: ClickRun, earlier: Sequence[ClickRun]) -> list[ClickRun]:
    """At most MAX_LISTED, one per click sequence (the newest), most similar
    first: same length, then fewest differing cells, then newest."""
    newest: dict[tuple[Cell, ...], ClickRun] = {}
    for run in earlier:
        newest[run.clicks] = run
    ranked = sorted(newest.values(), key=lambda run: (
        run.size != dead.size, abs(run.size - dead.size), _differ(dead, run), -run.last))
    return ranked[:MAX_LISTED]


def _dead_words(entries: Sequence[Any], run: ClickRun) -> str:
    """What the dead run's clicked cells did, from before its first click to
    after its last: ``only the clicked cells changed, from g to N, and they
    stay N`` (each distinct clicked cell by its own colour)."""
    tail = "nothing else inside the board area changed and no animation played."
    before, after = _grid(entries[run.start - 1]), _grid(entries[run.end])
    if before is None or after is None:
        return f"only the clicked cells changed; {tail}"
    cells = list(dict.fromkeys(run.clicks))
    moves: dict[tuple[str, str], list[Cell]] = {}
    for r, c in cells:
        if 0 <= r < len(before) and 0 <= c < len(before[0]):
            moves.setdefault((letter(before[r][c]), letter(after[r][c])), []).append((r, c))
    changed = {key: group for key, group in moves.items() if key[0] != key[1]}
    if not changed:
        if any(before[r][c] != after[r][c] for r, c in run.region):
            return f"only the clicked cells changed; {tail}"
        if run.lit:
            return f"the clicked cells changed and are back to their colours from before the run; {tail}"
        return "nothing inside the board area changed and no animation played."
    if len(changed) == 1 and len(next(iter(changed.values()))) == len(cells):
        (old, new), = changed
        return f"only the clicked cells changed, from {old} to {new}, and they stay {new}; {tail}"
    parts = [f"{_cells(group)} from {old} to {new}" for (old, new), group in changed.items()]
    same = [cell for key, group in moves.items() if key[0] == key[1] for cell in group]
    if same:
        parts.append(f"{_cells(same)} as before")
    return f"only the clicked cells changed: {', '.join(parts)}; {tail}"


def _compare_words(dead: ClickRun, top: ClickRun) -> str:
    where = _span(top.first, top.last).replace("actions ", "").replace("action ", "")
    if top.clicks == dead.clicks:
        return f"The run at {where} clicked the same cells in the same order."
    cells = set(top.clicks)
    differ = len(cells - set(dead.clicks))
    if differ == 0 and cells == set(dead.clicks):
        return f"The run at {where} clicked the same cells in another order."
    if differ == 0:
        return f"The run at {where} has the same length and all of its cells are in this one."
    if differ == len(cells):
        return f"The run at {where} has the same length and none of its cells are in this one."
    return f"The run at {where} has the same length and differs from this one in {differ} of its {len(cells)} cells."


# ---------------------------------------------------------------- effect phrases


def effect_phrase(entries: Sequence[Any], run: ClickRun) -> str:
    """What an effective run did outside its clicked cells, from the board
    before its first click to the board after its last one: moved blocks,
    corner frames and objects gone, shrunk, grown or new (the
    ``MAX_PHRASES`` that changed the most cells), and colours no longer on
    the board."""
    if run.ended:
        result = _result(entries[run.end])
        if result.get("level_completed"):
            return "the level was completed"
        if result.get("game_over"):
            return "the attempt ended (GAME OVER)"
    before, after = _grid(entries[run.start - 1]), _grid(entries[run.end])
    if before is None or after is None or len(before) != len(after) or len(before[0]) != len(after[0]):
        return "the board changed outside the clicked cells"
    height, width = len(after), len(after[0])
    changed = {cell for cell in changed_cells(before, after)
               if cell not in run.region and _inside(cell, height, width, BAND)}
    if not changed:
        if run.animated:
            text = "an animation played; afterwards nothing outside the clicked cells differs"
            return text + ("; the clicked cells are as before the run" if _clicked_back(before, after, run) else "")
        return "cells outside the clicked cells changed and changed back"
    nodes_before, nodes_after = components(before), components(after)
    items: list[tuple[int, str]] = []
    for group in moved_groups(nodes_before, nodes_after):
        if not (group.cells_before & changed):
            continue
        dr, dc = group.after[0] - group.before[0], group.after[1] - group.before[1]
        moved_to = {(r + dr, c + dc) for r, c in group.cells_before}
        items.append((len((group.cells_before | moved_to) & changed), _group_words(group)))
        changed -= group.cells_before | moved_to
    items.extend(_frame_items(nodes_before, nodes_after, changed))
    items.extend(_object_items(before, after, nodes_before, nodes_after, changed))
    items.sort(key=lambda item: -item[0])
    phrases = [text for _, text in items[:MAX_PHRASES]]
    gone = _gone_colours(before, after, run.region)
    if gone:
        phrases.append(f"no {'/'.join(gone)} cells are left on the board")
    if _clicked_back(before, after, run):
        phrases.append("the clicked cells are as before the run")
    return "; ".join(phrases) if phrases else "the board changed outside the clicked cells"


def _clicked_back(before: Grid, after: Grid, run: ClickRun) -> bool:
    """Every clicked cell has its colour from before the run, and some
    clicked cell changed during it."""
    if any(before[r][c] != after[r][c] for r, c in run.region):
        return False
    return run.lit


def _frame_items(nodes_before: Sequence[Node], nodes_after: Sequence[Node],
                 changed: set[Cell]) -> list[tuple[int, str]]:
    """Corner frames (four single same-colour pixels at a rectangle's
    corners) gone or new; their corners leave ``changed``."""
    before = {(f.colour, f.corners): f for f in corner_frames(nodes_before)}
    after = {(f.colour, f.corners): f for f in corner_frames(nodes_after)}
    items: list[tuple[int, str]] = []
    for key, frame in before.items():
        if key not in after and set(frame.corners) & changed:
            items.append((4, f"the {colour_name(frame.colour)} {frame.width}x{frame.height} corner frame "
                             f"(single {letter(frame.colour)} pixels at {_cells(frame.corners)}) is gone"))
            changed -= set(frame.corners)
    for key, frame in after.items():
        if key not in before and set(frame.corners) & changed:
            items.append((4, f"a {colour_name(frame.colour)} {frame.width}x{frame.height} corner frame "
                             f"appeared (single {letter(frame.colour)} pixels at {_cells(frame.corners)})"))
            changed -= set(frame.corners)
    return items


def _objects(nodes: Sequence[Node], cells: set[Cell]) -> list[set[Cell]]:
    """The nodes of at most MAX_GROUP_CELLS cells holding any of ``cells``,
    merged when they touch; a node with under a quarter of its cells in
    ``cells`` and more than NODE_CELLS cells is a backdrop and left out."""
    touched = [n for n in nodes if n.size <= MAX_GROUP_CELLS and n.cells & cells
               and not (n.size > NODE_CELLS and 4 * len(n.cells & cells) < n.size)]
    groups: list[set[Cell]] = []
    for node in touched:
        merged = set(node.cells)
        keep = []
        for group in groups:
            if any((r + dr, c + dc) in group for r, c in node.cells for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                merged |= group
            else:
                keep.append(group)
        groups = [*keep, merged]
    return groups


def _box(cells: set[Cell]) -> str:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return rows_cols(min(rows), min(cols), max(rows), max(cols))


def _letters(grid: Grid, cells: set[Cell]) -> str:
    return "/".join(letter(v) for v in sorted({int(grid[r][c]) for r, c in cells}))


def _object_items(before: Grid, after: Grid, nodes_before: Sequence[Node], nodes_after: Sequence[Node],
                  changed: set[Cell]) -> list[tuple[int, str]]:
    """Objects that lost cells (gone, or now smaller or larger where they
    overlap their own colours after) and objects new on changed cells; equal
    single pixels gone or new together are one item. Weight: changed cells."""
    small_after = [n for n in nodes_after if n.size <= MAX_GROUP_CELLS]
    covered: set[Cell] = set()
    fates: dict[tuple[str, str, int], list[set[Cell]]] = {}
    items: list[tuple[int, str]] = []
    for obj in _objects(nodes_before, changed):
        colours = {int(before[r][c]) for r, c in obj}
        now: set[Cell] = set()
        for node in small_after:
            if node.colour in colours and node.cells & obj:
                now |= node.cells
        weight = len((obj | now) & changed)
        covered |= obj | now
        name = f"the {len(obj)}-cell {_letters(before, obj)} object at {_box(obj)}"
        if not now:
            fates.setdefault(("gone", _letters(before, obj), len(obj)), []).append(obj)
        elif len(now) != len(obj):
            items.append((weight, f"{name} is now {len(now)} cells at {_box(now)}"))
        elif now != obj:
            items.append((weight, f"{name} now covers {_box(now)}"))
        else:
            items.append((weight, f"{name} changed colours"))
    for obj in _objects(nodes_after, changed - covered):
        if obj & covered:
            continue
        fates.setdefault(("new", _letters(after, obj), len(obj)), []).append(obj)
    for (fate, letters, size), objs in fates.items():
        cells = set().union(*objs)
        if len(objs) == 1:
            text = (f"the {size}-cell {letters} object at {_box(cells)} is gone" if fate == "gone"
                    else f"a {size}-cell {letters} object appeared at {_box(cells)}")
        elif size == 1:
            where = _cells(sorted(cells)) if len(cells) <= 4 else _box(cells)
            text = (f"{len(objs)} single {letters} pixels are gone ({where})" if fate == "gone"
                    else f"{len(objs)} single {letters} pixels appeared ({where})")
        else:
            text = (f"{len(objs)} {size}-cell {letters} objects are gone ({_box(cells)})" if fate == "gone"
                    else f"{len(objs)} {size}-cell {letters} objects appeared ({_box(cells)})")
        items.append((len(cells), text))
    return items


def _gone_colours(before: Grid, after: Grid, region: frozenset) -> list[str]:
    """Colours present before the run (outside the clicked cells) and absent
    from the whole board after it."""
    height, width = len(before), len(before[0])
    had = {int(before[r][c]) for r in range(height) for c in range(width)
           if (r, c) not in region and _inside((r, c), height, width, BAND)}
    has = {int(v) for row in after for v in row}
    return [letter(v) for v in sorted(had - has)]
