"""Touching same-colour objects, as one turn-message line (``ARC3_TOUCHING_SPLIT``).

Segmentation joins touching same-colour objects into one 4-connected node, and
the model reads such a node as a merged object. In our porting-advaith v3 run,
m0r0 level 1 step 20, it wrote "The two blocks have MERGED into one object",
and its snippet failed on ``blocks[1]`` before it worked out that the blocks were
only adjacent. ``touching_lines`` states it up front: a node of colour X on the
current board that splits along one axis into k >= 2 equal boxes, each with the
shape of a distinct same-colour object of the level start, while X's pixel total
is unchanged.

Ported from the advaith/dan-fran-arc fork (``inference/agent/board_facts.py``,
the ``ARC3_TOUCHING_SPLIT`` part only), with its wording unchanged. Their state
lab, m0r0 level 6: the right next move in 10 of 20 samples with the line, 5 of
20 without. Pure python, standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from inference.utils.grid_utils import ARC_COLOR_CHARS

# the colour names of the system prompt's legend (grid_utils.ARC_COLOR_LEGEND)
_NAMES = {
    "W": "white", "w": "light gray", "g": "gray", "G": "dark gray", "c": "charcoal", "B": "black",
    "M": "magenta", "P": "pink", "R": "red", "b": "blue", "S": "sky blue", "Y": "yellow",
    "O": "orange", "r": "dark red", "N": "light green", "p": "purple",
}
# each touching object has at least this many cells (two touching single pixels are no objects)
MIN_PART_CELLS = 4
# nodes larger than this are walls or the background, never "an object"
MAX_GROUP_CELLS = 400
MAX_LINES = 2
_COUNT_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

Grid = Sequence[Sequence[int]]


def letter(value: int) -> str:
    return ARC_COLOR_CHARS[max(0, min(15, int(value)))]


def colour_name(value: int) -> str:
    return _NAMES[letter(value)]


@dataclass(frozen=True)
class Node:
    """One segmentation node: a 4-connected component of one colour."""

    colour: int
    cells: frozenset
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def filled(self) -> bool:
        return self.size == self.width * self.height

    def shape(self) -> frozenset:
        return frozenset((r - self.r0, c - self.c0) for r, c in self.cells)

    def key(self) -> tuple:
        return (self.colour, self.cells)


def _node(colour: int, cells) -> Node:
    cells = frozenset(cells)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return Node(int(colour), cells, min(rows), min(cols), max(rows), max(cols))


def components(grid: Grid) -> list[Node]:
    """Every 4-connected same-colour component, in reading order of its first cell."""
    height = len(grid)
    width = len(grid[0]) if height else 0
    flat = [int(v) for row in grid for v in row]
    seen = bytearray(height * width)
    nodes: list[Node] = []
    for start in range(height * width):
        if seen[start]:
            continue
        value = flat[start]
        seen[start] = 1
        stack = [start]
        cells = []
        while stack:
            i = stack.pop()
            r, c = divmod(i, width)
            cells.append((r, c))
            if c > 0 and not seen[i - 1] and flat[i - 1] == value:
                seen[i - 1] = 1
                stack.append(i - 1)
            if c + 1 < width and not seen[i + 1] and flat[i + 1] == value:
                seen[i + 1] = 1
                stack.append(i + 1)
            if r > 0 and not seen[i - width] and flat[i - width] == value:
                seen[i - width] = 1
                stack.append(i - width)
            if r + 1 < height and not seen[i + width] and flat[i + width] == value:
                seen[i + width] = 1
                stack.append(i + width)
        nodes.append(_node(value, cells))
    return nodes


def rows_cols(r0: int, c0: int, r1: int, c1: int, sep: str = " ") -> str:
    """``rows 33-36 cols 29-32`` (``sep`` ", " gives ``rows 9-13, cols 29-33``)."""
    rows = f"row {r0}" if r0 == r1 else f"rows {r0}-{r1}"
    cols = f"col {c0}" if c0 == c1 else f"cols {c0}-{c1}"
    return f"{rows}{sep}{cols}"


def _split(node: Node, originals: Sequence[Node]) -> list[Node] | None:
    """``node`` as k >= 2 equal boxes along one axis whose cells have the
    shapes of k distinct same-colour ``originals``; None otherwise."""
    for k in range(2, 7):
        for axis in ("cols", "rows"):
            span = node.width if axis == "cols" else node.height
            if span % k:
                continue
            step = span // k
            parts: list[Node] = []
            for i in range(k):
                if axis == "cols":
                    lo, hi = node.c0 + i * step, node.c0 + (i + 1) * step - 1
                    cells = [(r, c) for r, c in node.cells if lo <= c <= hi]
                else:
                    lo, hi = node.r0 + i * step, node.r0 + (i + 1) * step - 1
                    cells = [(r, c) for r, c in node.cells if lo <= r <= hi]
                if not cells:
                    break
                parts.append(_node(node.colour, cells))
            if len(parts) != k or min(p.size for p in parts) < MIN_PART_CELLS:
                continue
            pool = list(originals)
            for part in parts:
                match = next((o for o in pool if o.size == part.size and o.shape() == part.shape()), None)
                if match is None:
                    break
                pool.remove(match)
            else:
                return parts
    return None


def _kind(part: Node) -> str:
    if part.filled:
        return "square" if part.width == part.height else "rectangle"
    return "object"


def _touching_words(node: Node, parts: Sequence[Node], total: int) -> str:
    name, code, k = colour_name(node.colour), letter(node.colour), len(parts)
    listed = [f"a {p.width}x{p.height} at {rows_cols(p.r0, p.c0, p.r1, p.c1, ', ')}" for p in parts]
    joined = ", ".join(listed[:-1]) + f" and {listed[-1]}"
    kinds = {(p.width, p.height, _kind(p)) for p in parts}
    if len(kinds) == 1:
        (w, h, kind), = kinds
        start = f"{_COUNT_WORDS.get(k, str(k))} {w}x{h} {kind}s"
    else:
        start = ", ".join(f"a {p.width}x{p.height} {_kind(p)}" for p in parts[:-1])
        start += f" and a {parts[-1].width}x{parts[-1].height} {_kind(parts[-1])}"
    return (f"Same-colour objects that touch, listed separately: {name} {code} x{k} — {joined}, touching "
            f"(segmentation shows them as one {node.width}x{node.height} node). {name.capitalize()} pixel total: "
            f"{total}, the same as at the level start ({start}).")


class BoardFacts:
    """Fed the solver's full history (``HistoryEntry`` with ``frame``): it notes
    where the current level started, and the lines are computed from the
    history's own frames when asked."""

    def __init__(self) -> None:
        self.cursor = 0
        self.level: int | None = None
        self.start = 0
        self._entries: Sequence[Any] = ()
        self._cache: dict[int, list[Node]] = {}

    def feed(self, entries: Sequence[Any]) -> None:
        if self.cursor > len(entries):
            self.__init__()
        for index in range(self.cursor, len(entries)):
            frame = getattr(entries[index], "frame", None)
            if frame is None:
                continue
            level = int(getattr(frame, "level", 0) or 0)
            if self.level is None or level != self.level:
                self.level, self.start = level, index
                self._cache.clear()
        self.cursor = len(entries)
        self._entries = entries

    def _grid(self, index: int) -> Grid | None:
        frame = getattr(self._entries[index], "frame", None) if 0 <= index < len(self._entries) else None
        return getattr(frame, "grid", None)

    def _nodes(self, index: int) -> list[Node]:
        nodes = self._cache.get(index)
        if nodes is None:
            grid = self._grid(index)
            nodes = components(grid) if grid else []
            if len(self._cache) > 8:
                self._cache.pop(next(iter(self._cache)))
            self._cache[index] = nodes
        return nodes

    def touching_lines(self, max_lines: int = MAX_LINES) -> list[str]:
        last = len(self._entries) - 1
        if last <= self.start or self._grid(last) is None or self._grid(self.start) is None:
            return []
        start_nodes = self._nodes(self.start)
        now_nodes = self._nodes(last)
        start_keys = {n.key() for n in start_nodes}
        totals_start: dict[int, int] = {}
        totals_now: dict[int, int] = {}
        for n in start_nodes:
            totals_start[n.colour] = totals_start.get(n.colour, 0) + n.size
        for n in now_nodes:
            totals_now[n.colour] = totals_now.get(n.colour, 0) + n.size
        lines: list[str] = []
        for node in now_nodes:
            if node.key() in start_keys or node.size > MAX_GROUP_CELLS:
                continue
            if totals_now.get(node.colour) != totals_start.get(node.colour):
                continue
            parts = _split(node, [n for n in start_nodes if n.colour == node.colour])
            if parts:
                lines.append(_touching_words(node, parts, totals_now[node.colour]))
            if len(lines) >= max_lines:
                break
        return lines
