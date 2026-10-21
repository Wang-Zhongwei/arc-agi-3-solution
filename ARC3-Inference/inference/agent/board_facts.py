"""Object facts from the frames: moved markers and touching same-colour objects.

Pure python, standard library only, no I/O. Two flags read them
(``tool_agent`` places the lines):

- **ARC3_MEM_MOVED** (``moved_lines``): small marker objects (segmentation
  nodes of at most ``MARKER_CELLS`` cells, and "corner frames": four single
  same-colour pixels at the corners of an axis-aligned rectangle) whose
  position differs from the level start, with the action at which each last
  moved and the other objects that moved at that action, then the boxes
  marked by four single corner pixels of a moved frame's colour with their
  colour counts. Moves at arrow keys (a player avatar) and markers whose new
  cells hold the clicked cell (a cursor or a selection) are left out. State
  lab, sc25 L6 (A call 101): the 6x6 yellow frame marking the big-teleport
  pad moved at a click and the tool result said only "43 cells changed and
  changed back"; with the fact 16/20 samples took the winning teleport, 0/20
  without.
- **ARC3_TOUCHING_SPLIT** (``touching_lines``): one segmentation node of
  colour X that is k >= 2 same-colour objects of the level start (its box
  splits along one axis into k equal boxes whose cells have those objects'
  shapes) while X's pixel total is unchanged. m0r0: B read two touching 5x5
  squares (one 10x5 node) as "merged" at L1 and scored 0; with the fact
  10/20 samples won uc1's L6, 5/20 without.

Segmentation is the sandbox's: 4-connected components of one colour.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from inference.utils.grid_utils import ARC_COLOR_CHARS

# the colour names the system prompt's legend gives (memory_layout.COLOUR_NAMES)
_NAMES = {
    "W": "white", "w": "light gray", "g": "gray", "G": "dark gray", "c": "charcoal", "B": "black",
    "M": "magenta", "P": "pink", "R": "red", "b": "blue", "S": "sky blue", "Y": "yellow",
    "O": "orange", "r": "dark red", "N": "light green", "p": "purple",
}
MARKER_CELLS = 4
# ARC3_TOUCHING_SPLIT: each touching object has at least this many cells
# (two touching single pixels, bp35's and su15's, are no "objects")
MIN_PART_CELLS = 4
# a colour with more single pixels than this is a texture, not corner marks
MAX_CORNER_PIXELS = 40
MAX_LINES = 3
MAX_BOXES = 4
# moved groups larger than this are walls or the background, never "an object"
MAX_GROUP_CELLS = 400
ARROWS = frozenset({"UP", "DOWN", "LEFT", "RIGHT"})
# ARC3_MOVED_PERSIST: hidden/appeared lines (one per action) and frames per
# hidden/appeared sentence shown at most
MAX_EVENT_LINES = 2
MAX_EVENT_FRAMES = 2
# a marker frame is a square of this side (corner pixels of other shapes and
# sizes were coincidences of a pattern in the 25 public games)
MARKER_FRAME_SIDES = range(4, 9)
# cells this close to the edge are the HUD (click_runs.BAND)
PERSIST_BAND = 4
_CLICK_RE = re.compile(r"row\s*=\s*(-?\d+)\s*,\s*col\s*=\s*(-?\d+)", re.I)
_COUNT_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

Grid = Sequence[Sequence[int]]
Cell = tuple[int, int]


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


def is_node(grid: Grid, colour: int, cells: frozenset) -> bool:
    """Whether ``cells`` are exactly one segmentation node of ``colour`` in ``grid``."""
    height = len(grid)
    width = len(grid[0]) if height else 0
    for r, c in cells:
        if not (0 <= r < height and 0 <= c < width) or int(grid[r][c]) != colour:
            return False
    for r, c in cells:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) not in cells and 0 <= nr < height and 0 <= nc < width and int(grid[nr][nc]) == colour:
                return False
    return True


# ---------------------------------------------------------------- words


def rows_cols(r0: int, c0: int, r1: int, c1: int, sep: str = " ") -> str:
    """``rows 33-36 cols 29-32`` (``sep`` ", " gives ``rows 9-13, cols 29-33``)."""
    rows = f"row {r0}" if r0 == r1 else f"rows {r0}-{r1}"
    cols = f"col {c0}" if c0 == c1 else f"cols {c0}-{c1}"
    return f"{rows}{sep}{cols}"


def _cell(cell: Cell) -> str:
    return f"({cell[0]},{cell[1]})"


def _click_cell(display: str) -> Cell | None:
    match = _CLICK_RE.search(str(display or ""))
    return (int(match.group(1)), int(match.group(2))) if match else None


# ---------------------------------------------------------------- markers


@dataclass(frozen=True)
class Frame:
    """Four single same-colour pixels at the corners of a rectangle."""

    colour: int
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def corners(self) -> tuple[Cell, Cell, Cell, Cell]:
        return ((self.r0, self.c0), (self.r0, self.c1), (self.r1, self.c0), (self.r1, self.c1))

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    def size_key(self) -> tuple:
        return (self.colour, self.width, self.height)

    def contains(self, cell: Cell) -> bool:
        return self.r0 <= cell[0] <= self.r1 and self.c0 <= cell[1] <= self.c1


def corner_frames(nodes: Sequence[Node]) -> list[Frame]:
    """Rectangles whose four corners are single pixels of one colour, each
    pixel in exactly one such rectangle (a lattice of dots gives none)."""
    by_colour: dict[int, set[Cell]] = {}
    for node in nodes:
        if node.size == 1:
            by_colour.setdefault(node.colour, set()).update(node.cells)
    frames: list[Frame] = []
    for colour, pixels in sorted(by_colour.items()):
        if not 4 <= len(pixels) <= MAX_CORNER_PIXELS:
            continue
        rows: dict[int, set[int]] = {}
        for r, c in pixels:
            rows.setdefault(r, set()).add(c)
        found: list[Frame] = []
        row_keys = sorted(rows)
        for i, r0 in enumerate(row_keys):
            for r1 in row_keys[i + 1:]:
                if r1 - r0 < 2:
                    continue
                common = sorted(rows[r0] & rows[r1])
                for j, c0 in enumerate(common):
                    for c1 in common[j + 1:]:
                        if c1 - c0 >= 2:
                            found.append(Frame(colour, r0, c0, r1, c1))
        uses: dict[Cell, int] = {}
        for frame in found:
            for corner in frame.corners:
                uses[corner] = uses.get(corner, 0) + 1
        frames.extend(f for f in found if all(uses[corner] == 1 for corner in f.corners))
    return frames


@dataclass(frozen=True)
class Marker:
    colour: int
    cells: frozenset
    frame: Frame | None = None

    def bbox(self) -> tuple[int, int, int, int]:
        if self.frame is not None:
            f = self.frame
            return f.r0, f.c0, f.r1, f.c1
        rows = [r for r, _ in self.cells]
        cols = [c for _, c in self.cells]
        return min(rows), min(cols), max(rows), max(cols)

    def match_key(self) -> tuple:
        if self.frame is not None:
            return ("frame", *self.frame.size_key())
        r0, c0, _, _ = self.bbox()
        return ("node", self.colour, frozenset((r - r0, c - c0) for r, c in self.cells))

    def holds(self, cell: Cell) -> bool:
        if self.frame is not None:
            return self.frame.contains(cell)
        return cell in self.cells


def markers(nodes: Sequence[Node]) -> list[Marker]:
    """Small nodes (frame corners excluded) and corner frames."""
    frames = corner_frames(nodes)
    corner_cells = {corner for frame in frames for corner in frame.corners}
    out = [Marker(n.colour, n.cells) for n in nodes
           if n.size <= MARKER_CELLS and not (n.size == 1 and next(iter(n.cells)) in corner_cells)]
    out.extend(Marker(f.colour, frozenset(f.corners), f) for f in frames)
    return out


def _present(grid: Grid, marker: Marker) -> bool:
    if marker.frame is None:
        return is_node(grid, marker.colour, marker.cells)
    return all(is_node(grid, marker.colour, frozenset([corner])) for corner in marker.frame.corners)


@dataclass(frozen=True)
class Group:
    """Nodes that moved together at one action (same offset, touching)."""

    colours: tuple[int, ...]
    size: int
    before: tuple[int, int, int, int]
    after: tuple[int, int, int, int]
    cells_before: frozenset


def moved_groups(pre: Sequence[Node], post: Sequence[Node]) -> list[Group]:
    """What moved between two boards: each node gone from ``pre`` paired with
    a node of the same colour and shape new in ``post`` (nearest first),
    grouped by offset and contact, largest first."""
    post_keys = {n.key() for n in post}
    pre_keys = {n.key() for n in pre}
    gone = [n for n in pre if n.key() not in post_keys and n.size <= MAX_GROUP_CELLS]
    new = [n for n in post if n.key() not in pre_keys and n.size <= MAX_GROUP_CELLS]
    by_shape: dict[tuple, list[Node]] = {}
    for n in new:
        by_shape.setdefault((n.colour, n.shape()), []).append(n)
    pairs: list[tuple[Node, tuple[int, int]]] = []
    for n in gone:
        options = by_shape.get((n.colour, n.shape()))
        if not options:
            continue
        best = min(options, key=lambda m: (abs(m.r0 - n.r0) + abs(m.c0 - n.c0), m.r0, m.c0))
        options.remove(best)
        pairs.append((n, (best.r0 - n.r0, best.c0 - n.c0)))
    groups: list[list[tuple[Node, tuple[int, int]]]] = []
    for pair in pairs:
        node, offset = pair
        touching = [g for g in groups if g[0][1] == offset and any(_touch(node, other) for other, _ in g)]
        merged = [pair]
        for g in touching:
            merged.extend(g)
            groups.remove(g)
        groups.append(merged)
    out: list[Group] = []
    for g in groups:
        cells = frozenset().union(*(node.cells for node, _ in g))
        dr, dc = g[0][1]
        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        r1 = max(r for r, _ in cells)
        c1 = max(c for _, c in cells)
        if len(cells) > MAX_GROUP_CELLS:
            continue
        colours = tuple(sorted({node.colour for node, _ in g}))
        out.append(Group(colours, len(cells), (r0, c0, r1, c1), (r0 + dr, c0 + dc, r1 + dr, c1 + dc), cells))
    out.sort(key=lambda grp: (-grp.size, grp.before))
    return out


def _touch(a: Node, b: Node) -> bool:
    if a.r0 > b.r1 + 1 or b.r0 > a.r1 + 1 or a.c0 > b.c1 + 1 or b.c0 > a.c1 + 1:
        return False
    return any((r + dr, c + dc) in b.cells for r, c in a.cells for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))


def _group_words(group: Group) -> str:
    colours = "/".join(letter(c) for c in group.colours)
    return (f"the {group.size}-cell {colours} block moved from {rows_cols(*group.before)} "
            f"to {rows_cols(*group.after)}")


def _marker_words(marker: Marker) -> str:
    """``the yellow pixel at (16,14)`` or ``the yellow 6x6 corner frame (...)``."""
    name = colour_name(marker.colour)
    if marker.frame is not None:
        f = marker.frame
        where = ", ".join(_cell(c) for c in f.corners)
        return f"the {name} {f.width}x{f.height} corner frame (single {letter(marker.colour)} pixels at {where}"
    if len(marker.cells) == 1:
        return f"the {name} pixel at {_cell(next(iter(marker.cells)))}"
    r0, c0, r1, c1 = marker.bbox()
    if len(marker.cells) == (r1 - r0 + 1) * (c1 - c0 + 1):
        return f"the {name} {c1 - c0 + 1}x{r1 - r0 + 1} block at {rows_cols(r0, c0, r1, c1)}"
    return f"the {len(marker.cells)}-cell {name} object at {rows_cols(r0, c0, r1, c1)}"


def _marker_frame(frame: Frame) -> bool:
    """ARC3_MOVED_PERSIST: a square corner frame of side 4-8."""
    return frame.width == frame.height and frame.width in MARKER_FRAME_SIDES


def _free_frame(grid: Grid | None, frame: Frame) -> bool:
    """At least half of the frame's outline between its corners is the
    board's most common colour: four marks around a box, not the cut
    corners of a sprite (whose outline is the sprite's colour)."""
    if not grid:
        return False
    counts: dict[int, int] = {}
    for row in grid:
        for value in row:
            counts[int(value)] = counts.get(int(value), 0) + 1
    field = max(counts, key=lambda value: counts[value])
    cells = [(frame.r0, c) for c in range(frame.c0 + 1, frame.c1)]
    cells += [(frame.r1, c) for c in range(frame.c0 + 1, frame.c1)]
    cells += [(r, frame.c0) for r in range(frame.r0 + 1, frame.r1)]
    cells += [(r, frame.c1) for r in range(frame.r0 + 1, frame.r1)]
    return 2 * sum(1 for r, c in cells if int(grid[r][c]) == field) >= len(cells)


def _frame_name(frame: Frame) -> str:
    """``yellow 6x6 corner frame``."""
    return f"{colour_name(frame.colour)} {frame.width}x{frame.height} corner frame"


def _inside_words(frame: Frame) -> str:
    """``rows 33-36 cols 29-32``: the cells inside the frame's corners."""
    return rows_cols(frame.r0 + 1, frame.c0 + 1, frame.r1 - 1, frame.c1 - 1)


def _now_words(old: Marker, new: Marker) -> str:
    if new.frame is not None:
        f, o = new.frame, old.frame
        inner_old = rows_cols(o.r0 + 1, o.c0 + 1, o.r1 - 1, o.c1 - 1)
        inner_new = rows_cols(f.r0 + 1, f.c0 + 1, f.r1 - 1, f.c1 - 1)
        where = ", ".join(_cell(c) for c in f.corners)
        return f"{_marker_words(old)}, around {inner_old}) is now at {where}, around {inner_new}"
    if len(new.cells) == 1:
        return f"{_marker_words(old)} is now at {_cell(next(iter(new.cells)))}"
    return f"{_marker_words(old)} is now at {rows_cols(*new.bbox())}"


def _box_words(grid: Grid, frame: Frame, surrounding: Frame | None) -> str:
    corners = set(frame.corners)
    counts: dict[int, list[Cell]] = {}
    for r in range(frame.r0, frame.r1 + 1):
        for c in range(frame.c0, frame.c1 + 1):
            if (r, c) not in corners:
                counts.setdefault(int(grid[r][c]), []).append((r, c))
    blocks: list[str] = []
    tallies: list[tuple[int, int]] = []
    for colour, cells in counts.items():
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
        if (r1 > r0 and c1 > c0 and len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)
                and len(counts) > 1):
            blocks.append((r0, c0, f"a {c1 - c0 + 1}x{r1 - r0 + 1} of {letter(colour)} at {rows_cols(r0, c0, r1, c1)}"))
        else:
            tallies.append((len(cells), colour))
    parts = [text for _, _, text in sorted(blocks)]
    tallies.sort(key=lambda t: (-t[0], t[1]))
    if not parts and len(tallies) == 1:
        parts.append(f"{tallies[0][0]} {letter(tallies[0][1])} cells")
    else:
        parts.extend(f"{n} {letter(colour)}" for n, colour in tallies)
    content = ", ".join(parts)
    text = (f"{rows_cols(frame.r0, frame.c0, frame.r1, frame.c1)} ({frame.width}x{frame.height}: "
            f"{content} and the 4 {letter(frame.colour)} corners")
    if surrounding is not None:
        name = "" if surrounding.colour == frame.colour else f"{colour_name(surrounding.colour)} "
        text += f"; the {name}{surrounding.width}x{surrounding.height} frame surrounds it"
    return text + ")"


def boxes_line(grid: Grid, frames: Sequence[Frame], colour: int, *, any_colour: bool = False) -> str:
    """``Boxes marked by four single yellow corner pixels: rows ... (4x4: ...); ...``.

    ``any_colour`` (ARC3_MOVED_PERSIST): a frame of any colour one cell
    around a box surrounds it (named with its colour when it differs) and
    is not itself a box; otherwise only frames of ``colour`` count."""
    mine = sorted((f for f in frames if f.colour == colour), key=lambda f: (f.r0, f.c0, f.r1, f.c1))
    pool = sorted(frames, key=lambda f: (f.r0, f.c0, f.r1, f.c1)) if any_colour else mine
    inner = {(f.r0, f.c0, f.r1, f.c1): f for f in pool}
    surrounds = {key: f for f in pool
                 if (key := (f.r0 + 1, f.c0 + 1, f.r1 - 1, f.c1 - 1)) in inner}
    listed = [f for f in mine if (f.r0 + 1, f.c0 + 1, f.r1 - 1, f.c1 - 1) not in inner]
    if not listed:
        return ""
    words = [_box_words(grid, f, surrounds.get((f.r0, f.c0, f.r1, f.c1))) for f in listed[:MAX_BOXES]]
    return f"Boxes marked by four single {colour_name(colour)} corner pixels: " + "; ".join(words) + "."


# ---------------------------------------------------------------- the tracker


class BoardFacts:
    """Fed the solver's full history (``HistoryEntry`` with ``action``,
    ``frame`` and ``result``) like ``memory_facts.FactRecorder``: it notes
    where the current level started; the lines are computed from the
    history's own frames when asked, so nothing per action is copied."""

    def __init__(self) -> None:
        self.cursor = 0
        self.level: int | None = None
        self.start = 0
        self._entries: Sequence[Any] = ()
        self._cache: dict[int, list[Node]] = {}
        # ARC3_MOVED_PERSIST: this level's corner frames and moved groups by
        # history index (a board's never change once it is in the history)
        self._frame_sets: dict[int, frozenset] = {}
        self._group_sets: dict[int, list[Group]] = {}

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
                self._frame_sets.clear()
                self._group_sets.clear()
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

    def _act(self, index: int) -> int:
        result = getattr(self._entries[index], "result", None) or {}
        number = result.get("action_num") if isinstance(result, dict) else None
        return int(number) if isinstance(number, int) else index

    def _display(self, index: int) -> str:
        return str(getattr(self._entries[index], "action", "") or "")

    def _last_move(self, marker: Marker) -> int | None:
        """The history index of the action after which ``marker`` (at its
        current place) first stood there in its latest stay."""
        for index in range(len(self._entries) - 1, self.start, -1):
            post, pre = self._grid(index), self._grid(index - 1)
            if post is None or pre is None:
                continue
            if _present(post, marker) and not _present(pre, marker):
                return index
        return None

    def _left_elsewhere(self, index: int, moved: Marker) -> bool:
        """At action ``index`` a marker like ``moved`` left another place (a
        move), rather than ``moved`` being recoloured or drawn where it is."""
        post = self._grid(index)
        for other in markers(self._nodes(index - 1)):
            if (other.match_key() == moved.match_key() and other.cells != moved.cells
                    and not _present(post, other)):
                return True
        return False

    # ---- ARC3_MEM_MOVED

    def moved_lines(self, max_lines: int = MAX_LINES, *, persist: bool = False) -> list[str]:
        """The moved-marker lines for the current board ([] when none).

        ``persist`` (ARC3_MOVED_PERSIST): also the corner frames hidden or
        new since this attempt started, and the boxes line whenever a box of
        a frame colour this level has moved, hidden or shown is on the board
        (``_persist_lines``)."""
        last = len(self._entries) - 1
        if last <= self.start or self._grid(last) is None or self._grid(self.start) is None:
            return []
        now_markers = markers(self._nodes(last))
        found = self._moved_markers(now_markers)
        if persist:
            return self._persist_lines(last, now_markers, found, max_lines)
        if not found:
            return []
        frames_now = [m.frame for m in now_markers if m.frame is not None]
        frame_colour = next((m.colour for _, _, m in found if m.frame is not None), None)
        boxes = boxes_line(self._grid(last), frames_now, frame_colour) if frame_colour is not None else ""
        room = max_lines - (1 if boxes else 0)
        lines = [self._moved_line(index, old, moved) for index, old, moved in found[:max(1, room)]]
        return [*lines, boxes] if boxes else lines

    def _moved_markers(self, now_markers: Sequence[Marker]) -> list[tuple[int, Marker, Marker]]:
        """(history index of the move, marker at the level start, marker now)
        for each marker whose position differs from the level start, newest
        move first; arrow-key moves and markers holding the clicked cell are
        left out."""
        start_markers = markers(self._nodes(self.start))
        start_keys = {(m.colour, m.cells) for m in start_markers}
        now_keys = {(m.colour, m.cells) for m in now_markers}
        gone = [m for m in start_markers if (m.colour, m.cells) not in now_keys]
        new = [m for m in now_markers if (m.colour, m.cells) not in start_keys]
        found: list[tuple[int, Marker, Marker]] = []
        for old in gone:
            same = [m for m in new if m.match_key() == old.match_key()]
            if len(same) != 1:
                continue
            moved = same[0]
            new.remove(moved)
            index = self._last_move(moved)
            if index is None or not self._left_elsewhere(index, moved):
                continue
            display = self._display(index)
            if display.strip().upper() in ARROWS:
                continue
            click = _click_cell(display)
            if click is not None and moved.holds(click):
                continue
            found.append((index, old, moved))
        found.sort(key=lambda item: (-item[0], item[2].bbox()))
        return found

    def _moved_line(self, index: int, old: Marker, moved: Marker) -> str:
        display = self._display(index)
        kind = "click" if _click_cell(display) is not None else "action"
        others = [g for g in moved_groups(self._nodes(index - 1), self._nodes(index))
                  if not (g.cells_before & old.cells) and not _overlaps(g, moved)]
        text = (f"Moved since this level started, not by an arrow key: {_now_words(old, moved)}. "
                f"It moved at action {self._act(index)}, {display}")
        if others:
            text += f", the {kind} at which " + " and ".join(_group_words(g) for g in others[:2])
        return text + "."

    # ---- ARC3_MOVED_PERSIST

    def _frames_at(self, index: int) -> frozenset:
        found = self._frame_sets.get(index)
        if found is None:
            found = self._frame_sets[index] = frozenset(corner_frames(self._nodes(index)))
        return found

    def _groups_at(self, index: int) -> list[Group]:
        found = self._group_sets.get(index)
        if found is None:
            found = self._group_sets[index] = moved_groups(self._nodes(index - 1), self._nodes(index))
        return found

    def _attempt_start(self, last: int) -> int:
        """The history index this attempt started from: the level start or
        the latest RESET on this level (a game over's automatic one too)."""
        for index in range(last, self.start, -1):
            if self._display(index).strip().upper() == "RESET":
                return index
        return self.start

    def _frame_events(self, first: int, last: int) -> dict[Frame, tuple[str, int]]:
        """Each corner frame's latest change after history index ``first``:
        ("hidden", i) or ("shown", i), i the index of the action; a frame
        gone at an action at which one of its colour and size is new is a
        move ("moved", i), which the moved line reports."""
        latest: dict[Frame, tuple[str, int]] = {}
        for index in range(first + 1, last + 1):
            pre, post = self._frames_at(index - 1), self._frames_at(index)
            if pre == post:
                continue
            gone, new = pre - post, post - pre
            moves = {f.size_key() for f in gone} & {f.size_key() for f in new}
            for frame in gone:
                latest[frame] = ("moved" if frame.size_key() in moves else "hidden", index)
            for frame in new:
                latest[frame] = ("moved" if frame.size_key() in moves else "shown", index)
        return latest

    def _change_words(self, index: int, skip: set[Cell]) -> str:
        """What else changed inside the board area at action ``index``: the
        largest moved block, else an object that changed size, else the
        largest other change ("" when nothing did)."""
        before, after = self._grid(index - 1), self._grid(index)
        if not before or not after or len(before) != len(after) or len(before[0]) != len(after[0]):
            return ""
        height, width = len(after), len(after[0])
        changed = {(r, c) for r in range(PERSIST_BAND, height - PERSIST_BAND)
                   for c in range(PERSIST_BAND, width - PERSIST_BAND)
                   if before[r][c] != after[r][c] and (r, c) not in skip}
        if not changed:
            return ""
        for group in self._groups_at(index):
            if group.cells_before & changed:
                return _group_words(group)
        from inference.agent.click_runs import _object_items  # click_runs imports this module

        items = _object_items(before, after, self._nodes(index - 1), self._nodes(index), set(changed))
        resized = [item for item in items if " is now " in item[1]]
        if not items:
            return ""
        return max(resized or items, key=lambda item: item[0])[1].replace(" is now ", " became ", 1)

    def _covered(self, frame: Frame, index: int) -> bool:
        """Whether a block that moved at history index ``index`` now lies on
        every corner of ``frame``."""
        for group in self._groups_at(index):
            r0, c0, r1, c1 = group.after
            if all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in frame.corners):
                return True
        return False

    def _landed(self, frame: Frame, first: int, last: int) -> tuple[int, Group] | None:
        """(history index, block) of the first block that moved into
        ``frame``'s inside from outside it at or after index ``first``."""
        for index in range(max(first, self.start + 1), last + 1):
            for group in self._groups_at(index):
                r0, c0, r1, c1 = group.after
                b0, d0, b1, d1 = group.before
                inside = frame.r0 < r0 and r1 < frame.r1 and frame.c0 < c0 and c1 < frame.c1
                was = frame.r0 < b0 and b1 < frame.r1 and frame.c0 < d0 and d1 < frame.c1
                if inside and not was:
                    return index, group
        return None

    def _shown_words(self, shown: Sequence[Frame], index: int, last: int, what: str) -> str:
        """``a purple 6x6 corner frame around ...; the 4-cell b/S block moved
        inside it at action 233, ..., from ...`` (a block that moved in at
        ``index`` itself is left to ``what``, which names it)."""
        words: list[str] = []
        landings: dict[str, int] = {}
        for frame in shown:
            words.append(f"a {_frame_name(frame)} around {_inside_words(frame)}")
            found = self._landed(frame, index, last)
            if found is None or (found[0] == index and what == _group_words(found[1])):
                continue
            at, group = found
            colours = "/".join(letter(c) for c in group.colours)
            text = (f"the {group.size}-cell {colours} block moved inside at action {self._act(at)}, "
                    f"{self._display(at)}, from {rows_cols(*group.before)}")
            landings[text] = landings.get(text, 0) + 1
        for text, count in landings.items():
            words.append(text.replace(" inside at ", " inside it at " if count == 1 else " inside them at ", 1))
        return "; ".join(words)

    def _event_lines(self, last: int) -> tuple[list[str], set[int]]:
        """``Hidden since action N, <action> (<what else changed>): <frames>.
        Appeared at that action: <frames>.``, one line per action, newest
        first, for the frames hidden or shown since this attempt started;
        and the colours of those frames."""
        first = self._attempt_start(last)
        at_start = self._frames_at(first)
        by_action: dict[int, tuple[list[Frame], list[Frame]]] = {}
        for frame, kind, index in self._marker_events(first, last):
            if kind == "shown" and frame in at_start:
                continue  # back where the attempt started with it
            if kind == "hidden" and self._covered(frame, index):
                continue  # under a block that moved onto it
            by_action.setdefault(index, ([], []))[0 if kind == "hidden" else 1].append(frame)
        colours = {f.colour for hidden, shown in by_action.values() for f in (*hidden, *shown)}
        lines: list[str] = []
        for index in sorted(by_action, reverse=True)[:MAX_EVENT_LINES]:
            hidden, shown = (sorted(fs, key=lambda f: (f.r0, f.c0)) for fs in by_action[index])
            display = self._display(index)
            kind = "click" if _click_cell(display) is not None else "action"
            skip = {corner for f in (*hidden, *shown) for corner in f.corners}
            what = self._change_words(index, skip)
            head = f"action {self._act(index)}, {display}" + (f" (the {kind} at which {what})" if what else "")
            parts: list[str] = []
            if hidden:
                words = "; ".join(f"the {_frame_name(f)} (last around {_inside_words(f)})"
                                  for f in hidden[:MAX_EVENT_FRAMES])
                parts.append(f"Hidden since {head}: {words}.")
            if shown:
                words = self._shown_words(shown[:MAX_EVENT_FRAMES], index, last, what)
                parts.append(f"Appeared at {'that action' if hidden else head}: {words}.")
            lines.append(" ".join(parts))
        return lines, colours

    def _persist_lines(self, last: int, now_markers: Sequence[Marker],
                       found: list[tuple[int, Marker, Marker]], max_lines: int) -> list[str]:
        """ARC3_MOVED_PERSIST: the moved lines, the hidden/appeared lines and
        the boxes line of the first frame colour with a box on the board
        among the moved frames', the hidden or shown frames' and the colours
        of frames that changed earlier on this level."""
        events, colours = self._event_lines(last)
        frames_now = [m.frame for m in now_markers if m.frame is not None and _marker_frame(m.frame)]
        order = [m.colour for _, _, m in found if m.frame is not None and _marker_frame(m.frame)]
        order += sorted(colours) + sorted(self._level_frame_colours(last))
        boxes = ""
        for colour in dict.fromkeys(order):
            boxes = boxes_line(self._grid(last), frames_now, colour, any_colour=True)
            if boxes:
                break
        room = max(1, max_lines - (1 if boxes else 0))
        moved = [self._moved_line(index, old, new) for index, old, new in found[:room]]
        return [*moved, *events, *([boxes] if boxes else [])]

    def _marker_events(self, first: int, last: int) -> list[tuple[Frame, str, int]]:
        """(frame, "hidden" or "shown", history index) of the marker frames
        whose latest change after ``first`` hid or showed them: square, of
        side 4-8, four marks around a box (``_free_frame``), at an action
        other than an arrow key (a player avatar passing over or past them)."""
        out: list[tuple[Frame, str, int]] = []
        for frame, (kind, index) in self._frame_events(first, last).items():
            if kind == "moved" or not _marker_frame(frame):
                continue
            if self._display(index).strip().upper() in ARROWS:
                continue
            if not _free_frame(self._grid(index - 1 if kind == "hidden" else index), frame):
                continue
            out.append((frame, kind, index))
        return out

    def _level_frame_colours(self, last: int) -> set[int]:
        """Colours of the marker frames hidden or shown at any action on
        this level so far."""
        return {frame.colour for frame, _, _ in self._marker_events(self.start, last)}

    # ---- ARC3_TOUCHING_SPLIT

    def touching_lines(self, max_lines: int = 2) -> list[str]:
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


def _overlaps(group: Group, marker: Marker) -> bool:
    r0, c0, r1, c1 = group.after
    return any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in marker.cells)


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
