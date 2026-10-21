"""Layer 1 of perception, finished: turn the parsed components into named symbols.

Connected-component labelling already runs in `perception.segment` (flood fill, no model
involved). What was missing is the last deterministic step: naming each component's *shape*, so
the symbol table reads like an inventory rather than a list of pixel counts. An 8B model asked to
find a cross in a 64x64 text grid will lose the column; told `t7 is a 3x3 cross of red at
(20,30)`, it has nothing left to lose track of.

Shape names are decided by exact geometry, never by similarity: a shape is a `square` only if its
bounding box is square and every cell in it is filled.
"""

from __future__ import annotations

from typing import Iterable, Optional

from .types import Blob, Track, colour_letter

Cell = tuple[int, int]


def shape_name(cells: Iterable[Cell], bbox: tuple[int, int, int, int]) -> str:
    """One of dot, hline, vline, square, rect, cross, L, ring, blob — decided exactly."""
    cells = frozenset(cells)
    n = len(cells)
    r0, c0, r1, c1 = bbox
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if n == 1:
        return "dot"
    if h == 1:
        return "hline"
    if w == 1:
        return "vline"
    filled = n == h * w
    if filled:
        return "square" if h == w else "rect"
    local = {(r - r0, c - c0) for r, c in cells}
    if h == w and h % 2 == 1:
        mid = h // 2
        cross = {(mid, c) for c in range(w)} | {(r, mid) for r in range(h)}
        if local == cross:
            return "cross"
    border = {(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)}
    if local == border:
        return "ring"
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    for cr, cc in corners:
        arm = {(cr, c) for c in range(w)} | {(r, cc) for r in range(h)}
        if local == arm:
            return "L"
    return "blob"


def describe_blob(blob: Blob) -> str:
    r0, c0, r1, c1 = blob.bbox
    return (f"{colour_letter(blob.colour)} {r1 - r0 + 1}x{c1 - c0 + 1} "
            f"{shape_name(blob.cells, blob.bbox)} at ({r0},{c0})")


def describe_track(track: Track, *, self_tid: Optional[str] = None) -> str:
    """`t7 = R 3x3 cross at (20,30), controllable` — the symbol the model reasons over."""
    r0, c0, r1, c1 = track.last_bbox
    shape = shape_name(track.blob.cells, track.last_bbox) if track.blob is not None else "?"
    bits = [f"{colour_letter(track.colour)} {r1 - r0 + 1}x{c1 - c0 + 1} {shape} at ({r0},{c0})"]
    if track.tid == self_tid:
        bits.append("THE BODY YOU CONTROL")
    elif track.klass not in ("unknown", ""):
        bits.append(track.klass)
    if track.label:
        bits.append(f'"{track.label}"')
    if track.state.value != "LIVE":
        bits.append(track.state.value.lower())
    return f"{track.tid} = " + ", ".join(bits)


def class_counts(tracks: Iterable[Track]) -> dict[str, int]:
    """How many live things of each colour-and-shape class, e.g. {'Y dot': 12, 'c square': 8}."""
    out: dict[str, int] = {}
    for t in tracks:
        if t.blob is None:
            continue
        key = f"{colour_letter(t.colour)} {shape_name(t.blob.cells, t.last_bbox)}"
        out[key] = out.get(key, 0) + 1
    return out


def symbol_table(tracks: Iterable[Track], *, self_tid: Optional[str] = None, max_rows: int = 14) -> str:
    """The symbol inventory: named things first, repeated scenery folded into a count."""
    tracks = list(tracks)
    singles, folded = [], {}
    for t in tracks:
        if t.blob is None:
            continue
        key = f"{colour_letter(t.colour)} {shape_name(t.blob.cells, t.last_bbox)}"
        interesting = (t.tid == self_tid or t.label or t.klass in ("controllable", "autonomous")
                       or t.state.value != "LIVE")
        (singles if interesting else folded.setdefault(key, [])).append(t) if interesting else folded.setdefault(key, []).append(t)
    lines = [describe_track(t, self_tid=self_tid) for t in singles[:max_rows]]
    for key, group in sorted(folded.items()):
        if len(group) >= 3:
            spots = ", ".join(f"({t.last_bbox[0]},{t.last_bbox[1]})" for t in group[:4])
            more = f" and {len(group) - 4} more" if len(group) > 4 else ""
            lines.append(f"{len(group)}x {key} at {spots}{more}")
        else:
            lines.extend(describe_track(t, self_tid=self_tid) for t in group)
        if len(lines) >= max_rows + 4:
            break
    return "\n".join(lines) if lines else "(nothing segmented)"
