"""Object-level contact facts for the mem-repair arm (arm/mem-repair-objects).

Pure python (standard library and ``board_facts``), no I/O. ``tool_agent``
feeds the solver's full history (``HistoryEntry`` with ``action``, ``frame``
and ``result``) at each new turn message, like ``board_facts.BoardFacts``.

- **ARC3_MEM_CONTACT_ROW** (``ObjectContacts.contact_row``): the objects on
  the current board that no action on this level has touched, first in
  ``<this_level>`` in place of the click-framed, colour-level "Colours on the
  board not clicked on this level" row. An object is an 8-connected
  same-colour component (su15's diagonal pink invader is one) of
  ``MIN_CELLS``..``MAX_CELLS`` cells that is not the background (the crop's
  most common colour), does not reach the grid's outermost row or column
  (walls, panels, terrain) and is not wholly inside the edge band (HUD and
  budget bars). It is touched when a click fell inside its bounding box
  (unless the click hit the background and changed nothing: ARC3_CLICK_MISS
  says so), when it held the clicked cell after the click, when the model
  moved it (any move at a key press; at a click, a move holding the clicked
  cell), when such a moved object entered its cells, when a key press merged
  movers into it, or when it lay where a blocked avatar (an object that moved
  at a key press) would have gone (bumped). Touches are kept as cells per
  colour; an object is touched when it overlaps a touched cell of its
  colour. Identical objects (colour and shape) are grouped; at most
  ``MAX_ITEMS`` groups and ``ROW_CHARS`` chars, objects new in this game
  first, then smaller, then fewer copies. New in this game: its colour was
  on no board of an earlier level and no action recoloured its cells (they
  held another non-background colour one frame earlier). m0r0 L5 (SF): the
  purple pressure plate went unentered for ~40 requests while the colours
  row led the model to click gates; tn36 L2: the row dropped "yellow" after
  one yellow block was clicked, though the yellow in the two legend boxes
  never was.
- **ARC3_CLICK_MISS** (``ObjectContacts.miss_lines``): a click that changed
  nothing inside the board area and hit the background inside, or within
  ``MISS_REACH`` cells of, an object's bounding box: which object, and its
  nearest cells. cn04 L2 (SF action 19): a click into the hole of the blue
  U-shape was doubted for ~11 requests.
- **ARC3_EFFECT_MOVED_FIX** (``pure_move``): an action whose only interior
  changes are objects translating (each changed cell lies in a moved
  object's old or new cells) is a move, even when the colour counts changed
  because the object crossed the edge band or covered a floor of another
  colour (m0r0 L5's squares, lf52's box).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from inference.agent.board_facts import Node, colour_name, rows_cols
from inference.agent.memory_facts import parse_click
from inference.agent.noop_repeat_guard import action_signature

Grid = Sequence[Sequence[int]]
Cell = tuple[int, int]

BORDER = 4
MIN_CELLS = 4
MAX_CELLS = 256
MAX_ITEMS = 6
ROW_CHARS = 360
MISS_REACH = 3
# ARC3_EFFECT_MOVED_FIX: changed cells a move may leave unexplained (a drop
# shadow's corner: lf52's box), at most this many and this share
MOVE_SLACK_CELLS = 2
MOVE_SLACK = 0.05
MAX_MISS_LINES = 2
ARROWS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
HEAD = "Not touched yet on this level (never clicked, entered or bumped): "
HEAD_NEW = "Not touched yet on this level (never clicked, entered or bumped). New in this game: "


# ---------------------------------------------------------------- objects


def objects(grid: Grid) -> list[Node]:
    """Every 8-connected same-colour component (a diagonal sprite such as
    su15's pink invader is one object), in reading order of its first cell."""
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
            for dr in (-1, 0, 1):
                nr = r + dr
                if nr < 0 or nr >= height:
                    continue
                for dc in (-1, 0, 1):
                    nc = c + dc
                    if (dr or dc) and 0 <= nc < width:
                        j = nr * width + nc
                        if not seen[j] and flat[j] == value:
                            seen[j] = 1
                            stack.append(j)
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        nodes.append(Node(value, frozenset(cells), min(rows), min(cols), max(rows), max(cols)))
    return nodes


def background(grid: Grid, border: int = BORDER) -> int | None:
    """The most common value inside the ``border`` crop (the whole grid when small)."""
    height = len(grid)
    width = len(grid[0]) if height else 0
    lo = border if height > 2 * border and width > 2 * border else 0
    counts = Counter(int(v) for row in grid[lo:height - lo] for v in row[lo:width - lo])
    return counts.most_common(1)[0][0] if counts else None


def interior_same(pre: Grid, post: Grid, border: int = BORDER) -> bool:
    height = len(pre)
    width = len(pre[0]) if height else 0
    if len(post) != height or any(len(row) != width for row in post):
        return False
    lo = border if height > 2 * border and width > 2 * border else 0
    return all(tuple(pre[r][lo:width - lo]) == tuple(post[r][lo:width - lo]) for r in range(lo, height - lo))


def _value(grid: Grid, cell: Cell) -> int | None:
    r, c = cell
    if 0 <= r < len(grid) and 0 <= c < len(grid[r]):
        return int(grid[r][c])
    return None


def _in_band(node: Node, height: int, width: int, border: int) -> bool:
    """Every cell within ``border`` of a grid edge (a HUD or budget bar), or
    a cell on the grid's outermost row or column (a wall, panel or column
    of terrain running into the frame)."""
    if node.r0 == 0 or node.c0 == 0 or node.r1 >= height - 1 or node.c1 >= width - 1:
        return True
    return all(r < border or r >= height - border or c < border or c >= width - border for r, c in node.cells)


def _shape(node: Node) -> frozenset:
    return frozenset((r - node.r0, c - node.c0) for r, c in node.cells)


def _kind_words(node: Node) -> tuple[str, str]:
    """(one, several): ``purple 4x4`` / ``purple 4x4 blocks`` for a filled
    rectangle (width x height), ``blue 9x9 frame(s)`` for a one-cell
    rectangle outline, ``pink 8-cell shape(s)`` for anything else."""
    name = colour_name(node.colour)
    w, h = node.width, node.height
    if node.size == w * h:
        return f"{name} {w}x{h}", f"{name} {w}x{h} blocks"
    if w >= 3 and h >= 3 and node.size == 2 * (w + h) - 4 and all(
        r in (node.r0, node.r1) or c in (node.c0, node.c1) for r, c in node.cells
    ):
        return f"{name} {w}x{h} frame", f"{name} {w}x{h} frames"
    return f"{name} {node.size}-cell shape", f"{name} {node.size}-cell shapes"


def _where(node: Node) -> str:
    return rows_cols(node.r0, node.c0, node.r1, node.c1, ", ")


# ---------------------------------------------------------------- moves


def _moved_pairs(pre: Sequence[Node], post: Sequence[Node]) -> list[tuple[Node, Node]]:
    """Each object gone from ``pre`` paired with the nearest object of the
    same colour and shape new in ``post`` (``board_facts.moved_groups``'
    pairing, ungrouped)."""
    post_keys = {n.key() for n in post}
    pre_keys = {n.key() for n in pre}
    gone = [n for n in pre if n.key() not in post_keys and n.size <= MAX_CELLS]
    new: dict[tuple, list[Node]] = {}
    for n in post:
        if n.key() not in pre_keys and n.size <= MAX_CELLS:
            new.setdefault((n.colour, _shape(n)), []).append(n)
    pairs: list[tuple[Node, Node]] = []
    for n in gone:
        options = new.get((n.colour, _shape(n)))
        if not options:
            continue
        best = min(options, key=lambda m: (abs(m.r0 - n.r0) + abs(m.c0 - n.c0), m.r0, m.c0))
        options.remove(best)
        pairs.append((n, best))
    return pairs


def pure_move(pre: Grid, post: Grid, border: int = BORDER) -> bool:
    """ARC3_EFFECT_MOVED_FIX: True when an object of 2+ cells translated
    and every interior cell that changed is explained by a translation by
    one of the moved objects' offsets d: a non-background colour moved in
    (``post[p] == pre[p - d]``) or a non-background colour moved out
    (``post[p + d] == pre[p]``). So the colour counts changed only because
    the moving thing (with its outline or shadow, which segmentation may not
    pair) crossed the edge band, covered or uncovered another colour. A cell
    that became background without its colour reappearing d away (a gate
    that opened) is not explained. Up to ``MOVE_SLACK_CELLS`` changed cells
    (and ``MOVE_SLACK`` of them) may stay unexplained: lf52's box casts a
    drop shadow whose corner cells change as it moves."""
    height = len(pre)
    width = len(pre[0]) if height else 0
    if len(post) != height or any(len(row) != width for row in post):
        return False
    lo = border if height > 2 * border and width > 2 * border else 0
    changed = [
        (r, c)
        for r in range(lo, height - lo)
        for c in range(lo, width - lo)
        if pre[r][c] != post[r][c]
    ]
    if not changed:
        return False
    offsets = {(b.r0 - a.r0, b.c0 - a.c0) for a, b in _moved_pairs(objects(pre), objects(post)) if a.size >= 2}
    if not offsets:
        return False

    bg = background(pre, border)

    def explained(r: int, c: int) -> bool:
        now, was = post[r][c], pre[r][c]
        for dr, dc in offsets:
            r0, c0, r1, c1 = r - dr, c - dc, r + dr, c + dc
            if now != bg and 0 <= r0 < height and 0 <= c0 < width and now == pre[r0][c0]:
                return True
            if was != bg and 0 <= r1 < height and 0 <= c1 < width and post[r1][c1] == was:
                return True
        return False

    slack = min(MOVE_SLACK_CELLS, int(len(changed) * MOVE_SLACK))
    misses = 0
    for r, c in changed:
        if not explained(r, c):
            misses += 1
            if misses > slack:
                return False
    return True


# ---------------------------------------------------------------- the tracker


class ObjectContacts:
    """Fed the solver's full history like ``board_facts.BoardFacts``; keeps,
    per level, the cells of every touched object (by colour), the avatars
    (objects that moved at a key press, with their step per key), the shapes
    seen on earlier levels and the click-miss facts per action number."""

    def __init__(self, border: int = BORDER) -> None:
        self.border = int(border)
        self.cursor = 0
        self.level: int | None = None
        self.first_level: int | None = None
        self.touched: dict[int, set[Cell]] = {}
        # (colour, shape) -> {key: {(dr, dc), ...}}, for the whole game (two
        # identical mirrored avatars step differently at one key)
        self.avatars: dict[tuple, dict[str, set[Cell]]] = {}
        # colours on any board of this level / of the earlier levels
        self.level_colours: set[int] = set()
        self.game_colours: set[int] = set()
        self.recoloured: set[tuple] = set()
        self.misses: dict[int, str] = {}
        self.grid: Grid | None = None
        self._nodes: list[Node] = []
        self._nodes_of: Any = None
        self._size = (0, 0)

    # ---- feeding

    def feed(self, entries: Sequence[Any]) -> None:
        if self.cursor > len(entries):
            self.__init__(self.border)
        if self.cursor == 0 and entries:
            frame = getattr(entries[0], "frame", None)
            if frame is not None and getattr(frame, "grid", None):
                self._begin_level(int(getattr(frame, "level", 0) or 0), frame.grid)
            self.cursor = 1
        for index in range(max(1, self.cursor), len(entries)):
            self.cursor = index + 1
            pre = getattr(entries[index - 1], "frame", None)
            post = getattr(entries[index], "frame", None)
            if pre is None or post is None or not getattr(pre, "grid", None) or not getattr(post, "grid", None):
                continue
            entry = entries[index]
            result = getattr(entry, "result", None) or {}
            act = result.get("action_num") if isinstance(result, dict) else None
            self._step(
                str(getattr(entry, "action", "") or ""), pre.grid, post.grid,
                int(getattr(pre, "level", 0) or 0), int(getattr(post, "level", 0) or 0),
                act=int(act) if isinstance(act, int) else index,
                over=bool(isinstance(result, dict) and result.get("game_over")),
            )

    def _nodes_for(self, grid: Grid) -> list[Node]:
        if self._nodes_of is grid:
            return self._nodes
        self._size = (len(grid), len(grid[0]) if len(grid) else 0)
        nodes = objects(grid)
        self._nodes_of, self._nodes = grid, nodes
        return nodes

    def _note_colours(self, grid: Grid) -> None:
        for row in grid:
            self.level_colours.update(int(v) for v in row)

    def _begin_level(self, level: int, grid: Grid) -> None:
        if self.level is not None:
            self.game_colours |= self.level_colours
        if self.first_level is None:
            self.first_level = level
        self.level = level
        self.touched = {}
        self.level_colours = set()
        self.recoloured = set()
        self.grid = grid
        self._nodes_for(grid)
        self._note_colours(grid)

    def _touch(self, node: Node) -> None:
        self.touched.setdefault(node.colour, set()).update(node.cells)

    def _step(self, action: str, pre: Grid, post: Grid, pre_level: int, post_level: int,
              *, act: int, over: bool) -> None:
        pre_nodes = self._nodes_for(pre)
        if self.level is None:
            self._begin_level(pre_level, pre)
        if post_level != pre_level or post_level != self.level:
            self._begin_level(post_level, post)
            return
        sig = action_signature(action)
        post_nodes = self._nodes_for(post)
        self.grid = post
        if sig == "RESET":
            return
        bg = background(pre, self.border)
        self._note_colours(post)
        pre_keys = {n.key() for n in pre_nodes}
        # a recolouring: the new object's cells held another non-background colour
        for n in post_nodes:
            if n.colour == bg or n.size < MIN_CELLS or n.key() in pre_keys:
                continue
            held = sum(1 for r, c in n.cells if int(pre[r][c]) not in (bg, n.colour))
            if 2 * held >= n.size:
                self.recoloured.add((n.colour, _shape(n)))
        click = parse_click(sig)
        pairs = _moved_pairs([n for n in pre_nodes if n.colour != bg], [n for n in post_nodes if n.colour != bg])
        if click is not None:
            r, c = click
            idle = not over and interior_same(pre, post, self.border)
            on_bg = _value(pre, click) == bg
            if not (idle and on_bg):
                # a click on the background that changed nothing touched nothing
                # (ARC3_CLICK_MISS says which object it fell inside)
                for n in pre_nodes:
                    if n.colour != bg and n.r0 <= r <= n.r1 and n.c0 <= c <= n.c1:
                        self._touch(n)
            for n in post_nodes:
                if n.colour != bg and click in n.cells:
                    self._touch(n)
            # objects the click itself moved (they hold the clicked cell before or after)
            mine = [(a, b) for a, b in pairs if click in a.cells or click in b.cells]
            if idle and on_bg:
                line = self._miss_line(act, click, pre_nodes, bg)
                if line:
                    self.misses[act] = line
                    if len(self.misses) > 64:
                        self.misses.pop(next(iter(self.misses)))
        else:
            mine = list(pairs)
            if sig in ARROWS:
                self._learn_and_bump(sig, pre_nodes, post_nodes, pairs, bg)
                self._touch_merged(pre, post, pre_keys, post_nodes, pairs, bg)
        for a, b in mine:
            self._touch(a)
            self._touch(b)
            moved = {a.key()}
            for n in pre_nodes:
                if n.colour != bg and n.key() not in moved and not n.cells.isdisjoint(b.cells):
                    self._touch(n)

    def _touch_merged(self, pre: Grid, post: Grid, pre_keys: set, post_nodes: Sequence[Node],
                      pairs: Sequence[tuple[Node, Node]], bg: int | None) -> None:
        """At a key press, a new object of a mover's colour over changed
        cells is the movers' product (two avatars or a selected piece
        merged): it is touched."""
        colours = {a.colour for a, _ in pairs}
        colours |= {colour for colour, _ in self.avatars}
        for n in post_nodes:
            if n.colour == bg or n.colour not in colours or n.key() in pre_keys:
                continue
            if any(pre[r][c] != post[r][c] for r, c in n.cells):
                self._touch(n)

    def _learn_and_bump(self, key: str, pre_nodes: Sequence[Node], post_nodes: Sequence[Node],
                        pairs: Sequence[tuple[Node, Node]], bg: int | None) -> None:
        """At a key press: learn each mover's step for ``key``; a known
        avatar that did not move was blocked, and what lies where it would
        have gone is bumped."""
        for a, b in pairs:
            steps = self.avatars.setdefault((a.colour, _shape(a)), {}).setdefault(key, set())
            if len(steps) < 4:
                steps.add((b.r0 - a.r0, b.c0 - a.c0))
        if not self.avatars:
            return
        post_keys = {n.key() for n in post_nodes}
        unit = ARROWS[key]
        for n in pre_nodes:
            known = self.avatars.get((n.colour, _shape(n)))
            if known is None or n.key() not in post_keys:
                continue
            hit = []
            for step in sorted(known.get(key, ())) or [unit]:
                target = {(r + step[0], c + step[1]) for r, c in n.cells} - n.cells
                hit += [m for m in pre_nodes if m.colour != bg and m is not n and not m.cells.isdisjoint(target)]
            if not hit and unit not in known.get(key, ()):
                target = {(r + unit[0], c + unit[1]) for r, c in n.cells} - n.cells
                hit = [m for m in pre_nodes if m.colour != bg and m is not n and not m.cells.isdisjoint(target)]
            for m in hit:
                self._touch(m)

    # ---- ARC3_CLICK_MISS

    def _miss_line(self, act: int, click: Cell, nodes: Sequence[Node], bg: int | None) -> str:
        r, c = click
        height, width = self._size
        best: tuple | None = None
        for n in nodes:
            if n.colour == bg or n.size < MIN_CELLS or n.size > MAX_CELLS or _in_band(n, height, width, self.border):
                continue
            gap = max(n.r0 - r, r - n.r1, 0, n.c0 - c, c - n.c1)
            if gap > MISS_REACH:
                continue
            nearest = min((abs(rr - r) + abs(cc - c)) for rr, cc in n.cells)
            rank = (gap, (n.r1 - n.r0 + 1) * (n.c1 - n.c0 + 1) if gap == 0 else nearest, n.r0, n.c0)
            if best is None or rank < best[0]:
                best = (rank, n, gap)
        if best is None:
            return ""
        _, n, gap = best
        name = colour_name(n.colour)
        cells = sorted(n.cells, key=lambda cell: (abs(cell[0] - r) + abs(cell[1] - c), cell))[:2]
        where = (f"inside the outline of the {name} object ({_where(n)})" if gap == 0
                 else f"{gap} cell{'s' if gap > 1 else ''} outside the bounding box of the {name} object ({_where(n)})")
        nearest = ", ".join(f"({rr},{cc})" for rr, cc in cells)
        return (f"Action {act} MOUSE({r},{c}) hit background ({colour_name(bg)}) {where}; "
                f"it did not touch it. Nearest {name} cells: {nearest}.")

    def miss_lines(self, start: Any, end: Any, max_lines: int = MAX_MISS_LINES) -> list[str]:
        """The click-miss lines of actions ``start``..``end`` (the sequence
        just executed), the latest ``max_lines``."""
        try:
            lo, hi = int(start), int(end)
        except (TypeError, ValueError):
            return []
        acts = [a for a in self.misses if lo <= a <= hi]
        return [self.misses[a] for a in sorted(acts)[-max_lines:]]

    # ---- ARC3_MEM_CONTACT_ROW

    def untouched(self) -> list[Node]:
        grid = self.grid
        if grid is None:
            return []
        height = len(grid)
        width = len(grid[0]) if height else 0
        bg = background(grid, self.border)
        out = []
        for n in self._nodes_for(grid):
            if n.colour == bg or not MIN_CELLS <= n.size <= MAX_CELLS or _in_band(n, height, width, self.border):
                continue
            if not n.cells.isdisjoint(self.touched.get(n.colour, ())):
                continue
            out.append(n)
        return out

    def _is_new(self, colour: int, shape: frozenset) -> bool:
        """New in this game: its colour was on no board of an earlier level
        (so neither was its colour and shape), and no action recoloured its
        cells. A new shape of a known colour is not tagged: on SF's 25 games
        those were 75% of the colour-and-shape tags, mostly terrain whose
        size changes per level (r11l's stretching arms, tr87's glyphs,
        sc25's grey tracks)."""
        if self.level is None or self.level == self.first_level or not self.game_colours:
            return False
        if (colour, shape) in self.recoloured:
            return False
        return colour not in self.game_colours

    def contact_row(self, max_items: int = MAX_ITEMS, max_chars: int = ROW_CHARS) -> str:
        """``Not touched yet on this level (never clicked, entered or bumped):
        purple 4x4 at rows 50-53, cols 14-17; ...``; with objects new in this
        game, ``Not touched yet on this level (never clicked, entered or
        bumped). New in this game: ...; .... Others: ....`` ("" when every
        object was touched). New objects first, then smaller (items, markers
        and plates before walls and bars), then fewer copies."""
        groups: dict[tuple, list[Node]] = {}
        for n in self.untouched():
            groups.setdefault((n.colour, _shape(n)), []).append(n)
        if not groups:
            return ""
        ranked = sorted(
            ((self._is_new(*key), sorted(nodes, key=lambda n: (n.r0, n.c0))) for key, nodes in groups.items()),
            key=lambda item: (not item[0], item[1][0].size, len(item[1]), item[1][0].r0, item[1][0].c0),
        )
        fresh: list[str] = []
        others: list[str] = []

        def render(fresh: list[str], others: list[str], rest: int) -> str:
            tail = f"; {rest} more not listed" if rest else ""
            if not fresh:
                return HEAD + "; ".join(others) + tail + "."
            text = HEAD_NEW + "; ".join(fresh)
            if others:
                text += ". Others: " + "; ".join(others)
            return text + tail + "."

        for new, nodes in ranked:
            text = _group_words(nodes)
            trial = ([*fresh, text], others) if new else (fresh, [*others, text])
            shown = len(fresh) + len(others) + 1
            if (fresh or others) and (shown > max_items
                                      or len(render(*trial, len(ranked) - shown)) > max_chars):
                break
            fresh, others = trial
        return render(fresh, others, len(ranked) - len(fresh) - len(others))


def _group_words(nodes: Sequence[Node]) -> str:
    """``purple 4x4 at rows 50-53, cols 14-17``; ``2 purple 4x4 blocks at
    ... and ...``; ``12 charcoal 4x4 blocks within rows 4-31, cols 33-60``."""
    one, several = _kind_words(nodes[0])
    if len(nodes) == 1:
        return f"{one} at {_where(nodes[0])}"
    if len(nodes) == 2:
        return f"2 {several} at {_where(nodes[0])} and {_where(nodes[1])}"
    r0 = min(n.r0 for n in nodes)
    c0 = min(n.c0 for n in nodes)
    r1 = max(n.r1 for n in nodes)
    c1 = max(n.c1 for n in nodes)
    return f"{len(nodes)} {several} within {rows_cols(r0, c0, r1, c1, ', ')}"
