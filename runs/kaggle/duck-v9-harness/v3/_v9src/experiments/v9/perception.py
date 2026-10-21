"""Deterministic perception: one frame in, one immutable reading out.

Everything here is a pure function of its inputs plus an explicit `PerceptionState`
(the HUD history and the level's background colour). No randomness, no wall clock, no
set iteration order leaking into an output: every list is sorted by an explicit key and
every hash is FNV-1a over a `repr` of a sorted tuple. Two runs over the same frames
produce byte-identical `Percept`s -- `tests/test_perception.py::test_perceive_is_deterministic`
asserts exactly that.

The reading order, and why it is that order:

1. **settle** -- the playable board is the LAST animation layer of a frame. Earlier
   layers are kept in `Percept.raw` because a mid-animation layer sometimes shows an
   object the settled frame hides (S15 in the corpus).
2. **background** -- the most common colour of the *first* frame of the level, then held
   constant for the level. It must not be recomputed per frame: a whole-screen blackout
   would otherwise redefine the background and make every real object read as "gone".
3. **global transform** -- a camera scroll or a palette permutation is one fact about the
   frame, not N object events. Detecting it first stops the tracker from reporting a
   world-ending catastrophe every time the camera pans.
4. **segment** -- 4-connected same-colour components (adapted from Duck's
   `inference/utils/segmentation.py::segment_layer`, with hashlib replaced by FNV-1a so
   the same code can run inside the stdlib-only REPL sandbox).
5. **HUD bands** -- thin strips flush against a border that keep changing on their own.
6. **veil** -- cells whose colour is an overlay, not evidence. Veiled cells are UNKNOWN:
   never "the object was destroyed", never "the background is now black".
7. **region tagging** then **merge_sprites** then **scalars_of**.

Steps 5-7 are ordered slightly differently from the plan's prose (region tagging before
sprite merging) because a sprite must not be built out of HUD or veiled components.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .types import Band, Blob, Cell, Grid, Percept, colour_letter

# --- hashing -----------------------------------------------------------------------------------

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = 0xFFFFFFFFFFFFFFFF

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))


def fnv1a(data: bytes | str) -> str:
    """Deterministic 64-bit FNV-1a of `data`, as 16 lowercase hex characters.

    Why not hashlib: perception code is spliced verbatim into the model's read-only REPL
    child, whose importer allows exactly 14 stdlib modules and hashlib is not one of them
    (CLAUDE.md rule 6). FNV-1a is 6 lines of arithmetic, so the same source runs in both
    places and a hash computed in the sandbox is comparable with one computed in the
    harness. Copied from experiments/repeat_block/repeat_block.py:88.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    h = _FNV_OFFSET
    for byte in data:
        h = ((h ^ byte) * _FNV_PRIME) & _MASK64
    return "%016x" % h


def normalised_cells(cells: Iterable[Cell]) -> tuple[Cell, ...]:
    """A cell set translated so its bounding box starts at the origin, in sorted order.

    This is what makes a shape signature translation-invariant: the same shape anywhere on
    the board normalises to the same tuple. `normalised_rle` is the encoding actually
    hashed; this is the readable form, for tests and for the REPL child's `objects` view.
    """
    cells = tuple(cells)
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


_ROW_END = b"\xff\xff"


def normalised_rle(cells: Iterable[Cell]) -> bytes:
    """A canonical, compact, translation-invariant encoding of a cell set.

    Per row of the shape (relative to the bounding box), the horizontal runs it contains, as
    (start, length) pairs of two bytes each, terminated by 0xFFFF. The encoding is a
    bijection with the normalised cell set, so equal encodings mean equal shapes.

    Why not `repr` of the sorted cell tuple: FNV-1a is a per-byte Python loop, so the length
    of the payload is the whole cost of hashing. A 64x64 background component reprs to about
    30 KB and run-length-encodes to about 0.5 KB, and hashing was 22% of `perceive` before
    this. Run lengths and starts are bounded by the board size, so 0xFFFF can never collide
    with a real value.
    """
    rows: dict[int, list[int]] = {}
    for r, c in cells:
        rows.setdefault(r, []).append(c)
    r0 = min(rows)
    c0 = min(min(cs) for cs in rows.values())
    buf = bytearray()
    for r in sorted(rows):
        buf += (r - r0).to_bytes(2, "big")
        cs = sorted(rows[r])
        start = prev = cs[0]
        for c in cs[1:]:
            if c == prev + 1:
                prev = c
                continue
            buf += (start - c0).to_bytes(2, "big") + (prev - start + 1).to_bytes(2, "big")
            start = prev = c
        buf += (start - c0).to_bytes(2, "big") + (prev - start + 1).to_bytes(2, "big")
        buf += _ROW_END
    return bytes(buf)


def shape_signature(cells: Iterable[Cell], colour: int) -> str:
    """Colour + normalised shape. Two blobs share it iff they are the same picture."""
    return fnv1a(bytes((int(colour) & 0xFF,)) + normalised_rle(cells))


def form_signature(cells: Iterable[Cell]) -> str:
    """Normalised shape only: colour-blind, so identity survives a recolour."""
    return fnv1a(normalised_rle(cells))


def grid_fingerprint(grid: Grid) -> str:
    """FNV-1a over the settled grid's bytes plus its dimensions.

    Dimensions are folded in so a 1x4 and a 4x1 board of the same values differ.
    """
    h = len(grid)
    w = len(grid[0]) if h else 0
    payload = bytearray(b"%d,%d;" % (h, w))
    for row in grid:
        payload.extend(bytes((int(v) & 0xFF) for v in row))
    return fnv1a(bytes(payload))


# --- background --------------------------------------------------------------------------------


def background_of(grid: Grid) -> int:
    """The most common colour of `grid` (ties broken by the lowest colour value).

    Ported from experiments/v8_loop/v8_loop.py::background, with the tie-break made
    explicit: `Counter.most_common` ties resolve by insertion order, which depends on
    scan order and is therefore not a fact about the frame.
    """
    counts = Counter(int(v) for row in grid for v in row)
    if not counts:
        return 0
    best = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    return int(best[0])


def colour_cells_of(grid: Grid) -> dict[int, frozenset[Cell]]:
    """colour -> the exact set of cells holding it. The base fact every scalar derives from."""
    acc: dict[int, set[Cell]] = {}
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            acc.setdefault(int(v), set()).add((r, c))
    return {k: frozenset(v) for k, v in sorted(acc.items())}


# --- segmentation ------------------------------------------------------------------------------


def _components(grid: Grid) -> tuple[list[dict[str, Any]], list[list[int]]]:
    """4-connected same-value components in reading order, plus the component-id grid.

    Reading-order scanning means component 0 is the one whose top-most-left-most cell comes
    first, which is unique on a 64x64 board -- so the component order is a fact about the
    frame, not about the traversal.
    """
    height = len(grid)
    width = len(grid[0]) if height else 0
    comp_id = [[-1] * width for _ in range(height)]
    comps: list[dict[str, Any]] = []
    for sr in range(height):
        for sc in range(width):
            if comp_id[sr][sc] != -1:
                continue
            value = int(grid[sr][sc])
            cid = len(comps)
            cells: list[Cell] = []
            stack = [(sr, sc)]
            comp_id[sr][sc] = cid
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in _ORTH:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and comp_id[nr][nc] == -1 \
                            and int(grid[nr][nc]) == value:
                        comp_id[nr][nc] = cid
                        stack.append((nr, nc))
            comps.append({"value": value, "cells": cells, "start": (sr, sc)})
    return comps, comp_id


def _holes(cells: frozenset[Cell], bbox: tuple[int, int, int, int]) -> int:
    """Number of enclosed empty regions inside a component.

    Flood the component's complement inward from a one-cell border around its bounding
    box; complement cells never reached are enclosed, and each 4-connected group of them
    is one hole. A solid rectangle short-circuits to zero, which is the common case and
    keeps the whole-board background component cheap.
    """
    r0, c0, r1, c1 = bbox
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if len(cells) == area:
        return 0
    reached: set[Cell] = set()
    stack: list[Cell] = []
    for r in range(r0 - 1, r1 + 2):
        for c in (c0 - 1, c1 + 1):
            if (r, c) not in reached:
                reached.add((r, c))
                stack.append((r, c))
    for c in range(c0 - 1, c1 + 2):
        for r in (r0 - 1, r1 + 1):
            if (r, c) not in reached:
                reached.add((r, c))
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in _ORTH:
            nr, nc = r + dr, c + dc
            if r0 - 1 <= nr <= r1 + 1 and c0 - 1 <= nc <= c1 + 1 \
                    and (nr, nc) not in reached and (nr, nc) not in cells:
                reached.add((nr, nc))
                stack.append((nr, nc))
    holes = 0
    seen: set[Cell] = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in cells or (r, c) in reached or (r, c) in seen:
                continue
            holes += 1
            seen.add((r, c))
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                for dr, dc in _ORTH:
                    nr, nc = cr + dr, cc + dc
                    if r0 <= nr <= r1 and c0 <= nc <= c1 and (nr, nc) not in cells \
                            and (nr, nc) not in reached and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
    return holes


def segment(grid: Grid, *, background: Optional[int]) -> list[Blob]:
    """Every 4-connected same-colour component of `grid`, as `Blob`s in reading order.

    Adapted from upstream/duck-harness/ARC3-Inference/inference/utils/segmentation.py's
    `segment_layer`. Dropped: the Moore contour trace and the containment/children tree
    (the tracker matches on hashes and cell sets, and the contour costs an O(n) flood per
    component). Kept and extended: reading-order ids, pixel counts, the translation-
    invariant hash (now FNV-1a, and split into a colour-carrying `shape_hash` and a
    colour-blind `form_hash` so a recolour does not look like a death and a birth), and a
    hole count.

    Components whose colour equals `background` get `region="background"`; everything else
    starts as `region="board"` and may be retagged hud/veiled by `perceive`.
    """
    comps, _ = _components(grid)
    blobs: list[Blob] = []
    for comp in comps:
        cells = frozenset(comp["cells"])
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        bbox = (min(rs), min(cs), max(rs), max(cs))
        colour = int(comp["value"])
        rle = normalised_rle(cells)          # computed once; both hashes derive from it
        blobs.append(Blob(
            colour=colour,
            cells=cells,
            bbox=bbox,
            pixels=len(cells),
            shape_hash=fnv1a(bytes((colour & 0xFF,)) + rle),
            form_hash=fnv1a(rle),
            region="background" if background is not None and colour == background else "board",
            holes=_holes(cells, bbox),
        ))
    return blobs


# --- multi-colour sprites ----------------------------------------------------------------------


def merge_sprites(blobs: list[Blob], *, max_pixels: int = 64) -> list[Blob]:
    """Union 4-adjacent small board components into multi-colour sprite blobs.

    A player avatar is very often two or three colours welded together (a body, an eye, a
    highlight). Mono-colour segmentation shreds it, and the tracker then follows three
    fragments that always move together. This builds the union as an extra `Blob` so both
    readings are available.

    Method (deliberately the simplest thing that is deterministic): consider only board
    components of at most 16 pixels, walk candidate adjacent pairs in reading order, and
    union a pair when the resulting group would still be under `max_pixels`. The returned
    list is the input list (unchanged, same order) followed by one sprite `Blob` per group
    of two or more parts, sprites sorted by bounding box.

    A sprite's `colour` is its dominant part's colour and its `shape_hash` is FNV-1a over
    the normalised (cell, colour) pairs -- so it is stable under translation and distinct
    from any mono blob's; `form_hash` is over the normalised cell set alone, so it survives
    a recolour of one part. `parts` holds the indices of the mono blobs it unions, which is
    how every other module tells a sprite from a mono blob and avoids double-counting.

    Known limits, stated rather than hidden: the pixel cap is applied greedily in reading
    order, so a chain of small parts can stop merging part-way and the grouping is not the
    maximum-cardinality one; a sprite whose parts are 8-adjacent but not 4-adjacent is not
    found; and a large object with a small satellite (>16px part) is never merged.
    """
    small = [(i, b) for i, b in enumerate(blobs)
             if b.region == "board" and not b.parts and b.pixels <= 16]
    if len(small) < 2:
        return list(blobs)

    owner: dict[Cell, int] = {}
    for i, b in small:
        for cell in b.cells:
            owner[cell] = i

    pairs: set[tuple[int, int]] = set()
    for i, b in small:
        for r, c in sorted(b.cells):
            for dr, dc in _ORTH:
                j = owner.get((r + dr, c + dc))
                if j is not None and j != i:
                    pairs.add((min(i, j), max(i, j)))

    parent = {i: i for i, _ in small}
    size = {i: b.pixels for i, b in small}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in sorted(pairs):
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if size[ri] + size[rj] > max_pixels:
            continue
        lo, hi = (ri, rj) if ri < rj else (rj, ri)
        parent[hi] = lo
        size[lo] += size[hi]

    groups: dict[int, list[int]] = {}
    for i, _ in small:
        groups.setdefault(find(i), []).append(i)

    sprites: list[Blob] = []
    for root in sorted(groups):
        members = sorted(groups[root])
        if len(members) < 2:
            continue
        cells: set[Cell] = set()
        painted: list[tuple[Cell, int]] = []
        for i in members:
            b = blobs[i]
            cells |= b.cells
            painted.extend((cell, b.colour) for cell in b.cells)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        bbox = (min(rs), min(cs), max(rs), max(cs))
        r0, c0 = bbox[0], bbox[1]
        norm = tuple(sorted(((r - r0, c - c0), col) for (r, c), col in painted))
        dominant = max(members, key=lambda i: (blobs[i].pixels, -blobs[i].colour))
        sprites.append(Blob(
            colour=blobs[dominant].colour,
            cells=frozenset(cells),
            bbox=bbox,
            pixels=len(cells),
            shape_hash=fnv1a(repr(("sprite", norm))),
            form_hash=form_signature(cells),
            region="board",
            holes=_holes(frozenset(cells), bbox),
            parts=tuple(members),
        ))
    sprites.sort(key=lambda b: (b.bbox, b.shape_hash))
    return list(blobs) + sprites


# --- HUD ---------------------------------------------------------------------------------------


class HudTracker:
    """Finds the HUD by where the board keeps changing, not by what the change looks like.

    Ported from experiments/frame_diff/frame_diff.py:337 and narrowed to the v9 definition
    of an edge band: a strip of consecutive rows (or columns) that

      * changed in at least `min_frac` of the last `window` observations,
      * is flush against a board border (its first row is 0, or its last is height-1),
      * is at most `max_thickness` cells thick, and
      * is at least `min_length` cells long.

    Why bands rather than components or cells: a timer bar that shrinks by one cell per
    action is a different component with a different hash on every frame, and no single
    cell of it ever changes twice. Persistence per component or per cell finds nothing --
    which is what happened on the recorded traces before frame_diff was rewritten this way.
    What is stable about a HUD is *where* it is.

    The band's long-axis extent is taken from the union of (a) the cells that have changed
    inside the strip over the window and (b) the strip's currently non-background cells.
    Using changes alone under-measures a 40-cell bar that only shrinks at its tip, which
    would then fail the `min_length` test and leave the HUD unclassified.

    Two settings were changed after the first live re86 run, and both were wrong for a
    reason worth recording:

    * `min_frac` was 0.75. re86's bottom row is a counter that ticks on most actions but
      not all of them, so its row crossed and re-crossed the 0.75 line and the band blinked
      in and out of existence -- one frame the row-63 bar was HUD, the next it was a
      "controllable" object. A HUD band is now a strip that changes on at least *half* of
      recent actions, and the edge-flush and thinness gates carry the discrimination.
    * bands are now **sticky**. A HUD does not move house: once a strip has qualified, it
      stays a band while it keeps changing at all within the window (`established`). What
      changes about a HUD is its content, not its address, and a detector that forgets the
      address every time the counter pauses is worse than useless -- it hands the tracker a
      new "object" on the border every few steps.
    """

    def __init__(self, height: int = 64, width: int = 64, *, window: int = 8,
                 min_frac: float = 0.5, min_observations: int = 3,
                 max_thickness: Optional[int] = None, min_length: int = 4) -> None:
        self.height = int(height)
        self.width = int(width)
        if max_thickness is None:
            # Proportional, not a flat three cells. A four-row status bar failed the test, so no
            # band was established, so `_is_hud` stayed False for it and a 140-pixel bar on the
            # bottom edge was classified controllable and then picked as the body -- the exact
            # failure the band detector exists to prevent, reappearing one row past the limit.
            max_thickness = max(3, int(0.08 * min(self.height, self.width)))
        self.window = int(window)
        self.min_frac = float(min_frac)
        self.min_observations = int(min_observations)
        self.max_thickness = int(max_thickness)
        self.min_length = int(min_length)
        self.history: list[frozenset[Cell]] = []
        # edge -> (lo, hi) of the strip along its short axis, once it has ever qualified.
        self.established: dict[str, tuple[int, int]] = {}

    def observe(self, cells: Iterable[Cell]) -> None:
        """Record the cells that changed across one executed action."""
        self.history.append(frozenset((int(r), int(c)) for r, c in cells))
        if len(self.history) > self.window:
            self.history = self.history[-self.window:]

    def ready(self) -> bool:
        return len(self.history) >= self.min_observations

    def _active(self, axis: int) -> list[int]:
        counts: dict[int, int] = {}
        for mask in self.history:
            for idx in {cell[axis] for cell in mask}:
                counts[idx] = counts.get(idx, 0) + 1
        need = self.min_frac * len(self.history)
        return sorted(idx for idx, n in counts.items() if n >= need)

    @staticmethod
    def _runs(indices: list[int]) -> list[list[int]]:
        runs: list[list[int]] = []
        for idx in indices:
            if runs and idx == runs[-1][1] + 1:
                runs[-1][1] = idx
            else:
                runs.append([idx, idx])
        return runs

    _EDGE_AXIS = {"top": 0, "bottom": 0, "left": 1, "right": 1}

    def _qualify(self, height: int, width: int) -> None:
        """Promote any strip that meets the change/flush/thickness test into `established`.

        An already-established edge absorbs an overlapping new strip (a HUD that grows a row
        is the same HUD), as long as the union stays within `max_thickness`.
        """
        for axis in (0, 1):
            limit = height if axis == 0 else width
            for lo, hi in self._runs(self._active(axis)):
                if hi - lo + 1 > self.max_thickness:
                    continue
                if lo == 0:
                    edge = "top" if axis == 0 else "left"
                elif hi == limit - 1:
                    edge = "bottom" if axis == 0 else "right"
                else:
                    continue
                known = self.established.get(edge)
                if known is not None:
                    merged = (min(lo, known[0]), max(hi, known[1]))
                    if merged[1] - merged[0] + 1 <= self.max_thickness:
                        lo, hi = merged
                self.established[edge] = (lo, hi)

    def bands(self, grid: Grid, background: int) -> list[Band]:
        """The current edge bands, as `Band`s, sorted by (edge, rows, cols).

        `grid` and `background` are needed for two of the four `Band` fields (`filled`, and
        the current half of `length`); the geometry itself comes from the change history.

        An established band is re-emitted while its strip changed at least once in the
        window. That is the stickiness described in the class docstring: a HUD that pauses
        for a couple of actions is still a HUD, but one that has gone completely quiet for a
        whole window has stopped being evidence of anything and is released.
        """
        if not self.ready() or not grid:
            return []
        height = len(grid)
        width = len(grid[0]) if height else 0
        self._qualify(height, width)

        out: list[Band] = []
        for edge in sorted(self.established):
            lo, hi = self.established[edge]
            axis = self._EDGE_AXIS[edge]
            thickness = hi - lo + 1
            if not any(any(lo <= cell[axis] <= hi for cell in mask) for mask in self.history):
                continue
            extent = {cell[1 - axis] for mask in self.history for cell in mask
                      if lo <= cell[axis] <= hi}
            span = width if axis == 0 else height
            for a in range(lo, hi + 1):
                for b in range(span):
                    r, c = (a, b) if axis == 0 else (b, a)
                    if 0 <= r < height and 0 <= c < width and int(grid[r][c]) != background:
                        extent.add(b)
            if not extent:
                continue
            o0, o1 = min(extent), max(extent)
            length = o1 - o0 + 1
            if length < self.min_length or length <= thickness:
                continue
            rows = (lo, hi) if axis == 0 else (o0, o1)
            cols = (o0, o1) if axis == 0 else (lo, hi)
            filled = 0
            for r in range(rows[0], rows[1] + 1):
                for c in range(cols[0], cols[1] + 1):
                    if 0 <= r < height and 0 <= c < width and int(grid[r][c]) != background:
                        filled += 1
            out.append(Band(edge=edge, rows=rows, cols=cols, filled=filled, length=length))
        out.sort(key=lambda b: (b.edge, b.rows, b.cols))
        return out


def band_cells(bands: Iterable[Band]) -> frozenset[Cell]:
    """Every cell covered by any band rectangle. Used for the `hud_only` delta tag."""
    cells: set[Cell] = set()
    for band in bands:
        for r in range(band.rows[0], band.rows[1] + 1):
            for c in range(band.cols[0], band.cols[1] + 1):
                cells.add((r, c))
    return frozenset(cells)


# --- veils -------------------------------------------------------------------------------------

VEIL_JUMP_FRAC = 0.25
VEIL_DOMINANCE_FRAC = 0.90


def detect_veil(prev: Optional[Percept], grid: Grid, blobs: list[Blob], background: int,
                *, cur_cells: Optional[dict[int, frozenset[Cell]]] = None) -> frozenset[Cell]:
    """Cells that are overlay, not evidence. They are UNKNOWN -- never "gone", never "empty".

    Two triggers, both from the plan's failure analysis of dc22 (a radial vignette and a
    whole-frame blackout) and m0r0 (the background repainted into quadrants):

    * **jump** -- a colour whose cell set changed by more than 25% of the board in one
      step. A repaint that large is a curtain being drawn, not 1000 objects being born.
    * **dominance** -- a colour covering 90% or more of the board. Whatever is under it is
      simply not observable this frame.

    The level's background colour is exempt from both. It is by construction the most
    common colour, so it would trip the dominance rule on every sparse board and mark the
    entire frame unknown; and its own cell set necessarily mirrors every large repaint. A
    blackout is still caught, because the background is the level's constant (fixed at the
    level's first frame) and the blackout colour is therefore not it.

    Pass `prev=None` to skip the jump rule -- `perceive` does this when a global transform
    already explains the frame, and at every level boundary. `cur_cells` is an optional
    already-computed colour -> cells index for `grid`; `perceive` passes the one it built for
    the `Percept`, since building it twice was 9% of a turn.
    """
    height = len(grid)
    width = len(grid[0]) if height else 0
    total = height * width
    if total == 0:
        return frozenset()
    if cur_cells is None:
        cur_cells = colour_cells_of(grid)

    veiled: set[int] = set()
    for colour, cells in cur_cells.items():
        if colour == background:
            continue
        if len(cells) >= VEIL_DOMINANCE_FRAC * total:
            veiled.add(colour)
    if prev is not None and prev.grid and len(prev.grid) == height:
        prev_cells = prev.colour_cells or colour_cells_of(prev.grid)
        empty: frozenset[Cell] = frozenset()
        for colour in set(cur_cells) | set(prev_cells):
            if colour == background:
                continue
            a = prev_cells.get(colour, empty)
            b = cur_cells.get(colour, empty)
            if len(a ^ b) > VEIL_JUMP_FRAC * total:
                veiled.add(colour)

    out: set[Cell] = set()
    for colour in sorted(veiled):
        out |= cur_cells.get(colour, frozenset())
    return frozenset(out)


# --- global transforms -------------------------------------------------------------------------

SCROLL_SEARCH = 4
TRANSFORM_FRAC = 0.90
SCROLL_MAX_STATIONARY = 0.50
"""A camera pan leaves at most half the board's content where it was; above that, do not
even search. Without this guard the 80-offset search ran on every ordinary frame of a real
game and cost more than segmentation did."""


def detect_global_transform(prev: Optional[Percept], grid: Grid) -> Optional[dict[str, Any]]:
    """One fact that explains the whole frame, or None.

    Two kinds:

    * `{"kind": "scroll", "dr":, "dc":, "frac":}` -- a single (dr, dc) under which at least
      90% of the previous frame's non-background cells land on a same-coloured cell of the
      new frame. This is a camera pan; without it every object on screen reports a MOVE.
    * `{"kind": "palette", "map": {old: new}}` -- a colour bijection that rewrites the
      entire frame cell for cell. This is a theme change; without it every object on screen
      reports a RECOLOUR.

    Cheap-exit design: identical frames return None immediately, and the scroll search only
    runs when fewer than half the previous frame's non-background cells are still where they
    were. On a routine turn (one object moved) that check is a single O(cells) pass and the
    81-offset search never runs, which is what keeps `perceive` in the low milliseconds on a
    64x64 board.
    """
    if prev is None or not prev.grid or not grid:
        return None
    height = len(grid)
    width = len(grid[0]) if height else 0
    if len(prev.grid) != height or (height and len(prev.grid[0]) != width):
        return None
    before, after = prev.grid, grid

    same = True
    for r in range(height):
        if before[r] != after[r]:
            same = False
            break
    if same:
        return None

    # palette bijection over the whole frame
    fwd: dict[int, int] = {}
    rev: dict[int, int] = {}
    ok = True
    for r in range(height):
        brow, arow = before[r], after[r]
        for c in range(width):
            a, b = int(brow[c]), int(arow[c])
            if fwd.setdefault(a, b) != b or rev.setdefault(b, a) != a:
                ok = False
                break
        if not ok:
            break
    if ok and any(k != v for k, v in fwd.items()):
        return {"kind": "palette", "map": {int(k): int(v) for k, v in sorted(fwd.items())}}

    bg = prev.background
    moving = [(r, c, int(before[r][c])) for r in range(height) for c in range(width)
              if int(before[r][c]) != bg]
    if not moving:
        return None

    def frac(dr: int, dc: int) -> float:
        hit = 0
        for r, c, v in moving:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and int(after[nr][nc]) == v:
                hit += 1
        return hit / len(moving)

    if frac(0, 0) >= SCROLL_MAX_STATIONARY:
        return None
    best: Optional[tuple[float, int, int]] = None
    for dr in range(-SCROLL_SEARCH, SCROLL_SEARCH + 1):
        for dc in range(-SCROLL_SEARCH, SCROLL_SEARCH + 1):
            if dr == 0 and dc == 0:
                continue
            f = frac(dr, dc)
            if f < TRANSFORM_FRAC:
                continue
            key = (-f, abs(dr) + abs(dc), dr, dc)
            if best is None or key < (-best[0], abs(best[1]) + abs(best[2]), best[1], best[2]):
                best = (f, dr, dc)
    if best is None:
        return None
    return {"kind": "scroll", "dr": int(best[1]), "dc": int(best[2]), "frac": round(best[0], 4)}


# --- scalars -----------------------------------------------------------------------------------


def scalars_of(grid: Grid, blobs: list[Blob], bands: list[Band], background: int,
               *, veil: frozenset[Cell] = frozenset()) -> dict[str, int]:
    """The closed scalar registry for one frame: the only names rules and predictions may use.

    Exactly these names are produced here:

      `cells[<letter>]`   cells of that ARC colour letter on the board
      `objs[<letter>]`    mono board components of that colour (sprites are not counted
                          again; HUD, background and veiled components are excluded)
      `objs_total`        sum of `objs[...]`
      `hud[<edge>].filled`, `hud[<edge>].len`   per detected edge band
      `veil_frac`         veiled cells per mille of the board, as an int

    `step` and `level` are stamped by `perceive`; `pos[<tid>].r/.c` come from the tracker
    and `consumed[...]`/`remaining[...]` from the fact layer, because none of those are
    functions of a single frame.
    """
    out: dict[str, int] = {}
    height = len(grid)
    width = len(grid[0]) if height else 0
    total = height * width

    # "on the board", as the line above says: a raw whole-grid Counter also moves every time a
    # HUD counter repaints or a veil flickers, and `cells[...]` is in the closed registry that
    # predictions are graded against -- so those changes were charged as changes nobody claimed,
    # on the same steps the board itself was readable. HUD and veiled cells are excluded here for
    # the same reason `objs[...]` excludes them.
    skip: set = set(veil)
    for band in bands:
        for r in range(band.rows[0], band.rows[1] + 1):
            for c in range(band.cols[0], band.cols[1] + 1):
                skip.add((r, c))
    counts = Counter(int(grid[r][c]) for r in range(height) for c in range(width)
                     if (r, c) not in skip)
    for colour in sorted(counts):
        out["cells[%s]" % colour_letter(colour)] = int(counts[colour])

    obj_counts: Counter[int] = Counter()
    for blob in blobs:
        if blob.region != "board" or blob.parts:
            continue
        obj_counts[blob.colour] += 1
    for colour in sorted(obj_counts):
        out["objs[%s]" % colour_letter(colour)] = int(obj_counts[colour])
    out["objs_total"] = int(sum(obj_counts.values()))

    by_edge: dict[str, Band] = {}
    for band in bands:
        cur = by_edge.get(band.edge)
        if cur is None or band.length > cur.length:
            by_edge[band.edge] = band
    for edge in sorted(by_edge):
        out["hud[%s].filled" % edge] = int(by_edge[edge].filled)
        out["hud[%s].len" % edge] = int(by_edge[edge].length)

    out["veil_frac"] = int(round(1000.0 * len(veil) / total)) if total else 0
    return out


# --- the reading --------------------------------------------------------------------------------


class EmptyFrame(ValueError):
    """Raised by `perceive` when the engine hands back a frame with no layers.

    This is not a bug and not a corrupt frame: the engine returns an empty frame list for a
    step it refused, which is what happens after GAME_OVER (plan, "Verified facts"). There
    is genuinely nothing to perceive, and inventing a percept would put a fabricated board
    into the ledger. It is a distinct exception type rather than a bare ValueError so the
    caller can catch exactly this and end the game cleanly:

        try:
            percept = perceive(all_layers(frame), ...)
        except EmptyFrame:
            break     # the engine refused the step; the game is over

    Left uncaught it costs a whole pass, which is why it has its own name.
    """


@dataclass
class PerceptionState:
    """What perception must remember between frames: the HUD history and the level's ground.

    `background` is deliberately a per-level constant, fixed from the level's first frame.
    Recomputing it per frame makes a whole-screen overlay redefine "empty" and turns every
    object on the board into a disappearance -- the exact dc22 failure the plan records.
    """
    hud: Optional[HudTracker] = None
    background: Optional[int] = None
    level: Optional[int] = None
    dims: tuple[int, int] = (0, 0)

    def begin_level(self, level: int, grid: Grid) -> None:
        """Fix the background for a new level. The HUD history is kept: the HUD is a
        property of the game, not of the level, and re-learning it costs 3 frames."""
        self.level = int(level)
        self.background = background_of(grid)


def perceive(layers: list[Grid], *, step: int, level: int, action: Optional[str],
             prev: Optional[Percept], state: PerceptionState) -> Percept:
    """Read one frame into an immutable `Percept`. Pure, given `state`.

    `layers` is the raw frame list from the engine: animation layers followed by the settled
    board. Everything downstream reads `layers[-1]`; the rest are kept in `Percept.raw`
    because a mid-animation layer occasionally disambiguates N identical movers.

    `prev` is last step's percept, used only for the jump-veil rule, the global transform
    and the HUD change history -- and only when it belongs to the same level and has the
    same dimensions, so a level boundary can never manufacture a delta.
    """
    if not layers or not layers[-1]:
        raise EmptyFrame("perceive: empty frame (no layers) -- the engine refused the step")
    settled = [[int(v) for v in row] for row in layers[-1]]
    height = len(settled)
    width = len(settled[0]) if height else 0

    if state.hud is None or state.dims != (height, width):
        state.hud = HudTracker(height, width)
        state.dims = (height, width)
    if state.background is None or state.level != level:
        state.begin_level(level, settled)
    background = int(state.background if state.background is not None else 0)

    continuous = (prev is not None and prev.level == level and prev.grid
                  and len(prev.grid) == height and len(prev.grid[0]) == width)
    base = prev if continuous else None

    transform = detect_global_transform(base, settled)

    if base is not None:
        changed = [(r, c) for r in range(height) for c in range(width)
                   if int(base.grid[r][c]) != settled[r][c]]
        state.hud.observe(changed)

    blobs = segment(settled, background=background)
    bands = state.hud.bands(settled, background)
    hud = band_cells(bands)
    colour_cells = colour_cells_of(settled)

    # A transform already explains the frame, so the jump rule must not fire on it.
    veil = detect_veil(None if transform is not None else base, settled, blobs, background,
                       cur_cells=colour_cells)

    tagged: list[Blob] = []
    for blob in blobs:
        region = blob.region
        if region == "board":
            if veil and blob.cells <= veil:
                region = "veiled"
            elif hud and blob.cells <= hud:
                region = "hud"
        if region != blob.region:
            blob = Blob(colour=blob.colour, cells=blob.cells, bbox=blob.bbox, pixels=blob.pixels,
                        shape_hash=blob.shape_hash, form_hash=blob.form_hash, region=region,
                        holes=blob.holes, parts=blob.parts)
        tagged.append(blob)

    tagged = merge_sprites(tagged)

    scalars = scalars_of(settled, tagged, bands, background, veil=veil)
    scalars["step"] = int(step)
    scalars["level"] = int(level)

    return Percept(
        step=int(step),
        level=int(level),
        action=action,
        raw=[[[int(v) for v in row] for row in layer] for layer in layers],
        grid=settled,
        background=background,
        hud_bands=bands,
        veil=veil,
        blobs=tagged,
        colour_cells=colour_cells,
        scalars=scalars,
        fingerprint=grid_fingerprint(settled),
        transform=transform,
    )
