"""Layer 2 of the symbolic pipeline: relations between symbols, computed with ARC-DSL.

    [raw grid] -> (CCL parser) -> [symbols] -> RELATIONS -> (model fills a form) -> (solver)

Connected-component labelling gives us *what* is on the board; this gives the *arithmetic between
the pieces*, which is the part an 8B model reading a text grid gets wrong: how far, which way, and
what shift would bring the body up against a thing. All of it is Michael Hodel's ARC-DSL
(`vendor/arcdsl/`, MIT), used unmodified: `manhattan`, `position`, `gravitate`, `adjacent`,
`centerofmass`, `ulcorner`, `shape`, `normalize`, `hmatching`, `vmatching`.

Nothing here asks a model anything. It is the last deterministic step before the form.
"""

from __future__ import annotations

from typing import Iterable, Optional

from .symbols import shape_name
from .types import Track, colour_letter
from .vendor.arcdsl import dsl

Cell = tuple[int, int]


def as_object(cells: Iterable[Cell], colour: int = 1):
    """Our cells -> an ARC-DSL Object: a frozenset of (value, (row, col))."""
    return frozenset((int(colour), (int(r), int(c))) for r, c in cells)


def track_object(track: Track):
    if track.blob is None:
        return frozenset()
    return as_object(track.blob.cells, track.colour)


def _direction_words(vector: tuple[int, int]) -> str:
    dr, dc = vector
    parts = []
    if dr < 0:
        parts.append("above")
    elif dr > 0:
        parts.append("below")
    if dc < 0:
        parts.append("left")
    elif dc > 0:
        parts.append("right")
    return " and ".join(parts) if parts else "same place"


def relation(body, other) -> dict[str, object]:
    """Every relation the solver or the model might want, from ARC-DSL primitives."""
    if not body or not other:
        return {}
    out: dict[str, object] = {
        "distance": int(dsl.manhattan(body, other)),
        "direction": tuple(int(v) for v in dsl.position(body, other)),
        "adjacent": bool(dsl.adjacent(body, other)),
        "same_row_band": bool(dsl.hmatching(body, other)),
        "same_col_band": bool(dsl.vmatching(body, other)),
    }
    try:
        # gravitate: the shift that brings `body` up against `other` without overlapping it.
        out["gravitate"] = tuple(int(v) for v in dsl.gravitate(body, other))
    except Exception:  # noqa: BLE001 - ARC-DSL raises when the two do not share a band
        out["gravitate"] = None
    return out


def relations_text(self_track: Optional[Track], others: Iterable[Track], *, max_rows: int = 8) -> str:
    """`t7 (Y dot at (5,5)): 12 away, below and right, same column band, shift (+3,0) to touch`."""
    if self_track is None or self_track.blob is None:
        return ""
    body = track_object(self_track)
    rows = []
    scored = []
    for other in others:
        if other.tid == self_track.tid or other.blob is None:
            continue
        rel = relation(body, track_object(other))
        if not rel:
            continue
        scored.append((rel["distance"], other, rel))
    scored.sort(key=lambda t: (t[0], t[1].tid))
    for _distance, other, rel in scored[:max_rows]:
        r0, c0, _r1, _c1 = other.last_bbox
        shape = shape_name(other.blob.cells, other.last_bbox)
        bits = [f"{rel['distance']} away", _direction_words(rel["direction"])]
        if rel["adjacent"]:
            bits.append("TOUCHING")
        if rel["same_row_band"]:
            bits.append("same rows")
        if rel["same_col_band"]:
            bits.append("same columns")
        if rel.get("gravitate"):
            dr, dc = rel["gravitate"]
            if (dr, dc) != (0, 0):
                bits.append(f"move ({dr:+d},{dc:+d}) to touch it")
        rows.append(f"{other.tid} ({colour_letter(other.colour)} {shape} at ({r0},{c0})): "
                    + ", ".join(bits))
    return "\n".join(rows)


def board_facts(grid) -> str:
    """Whole-board facts from ARC-DSL: size, palette, the dominant colour, how many components."""
    g = tuple(tuple(int(v) for v in row) for row in grid)
    if not g:
        return ""
    palette = sorted(int(v) for v in dsl.palette(g))
    return (f"{dsl.height(g)}x{dsl.width(g)}, background {colour_letter(int(dsl.mostcolor(g)))}, "
            f"colours present {' '.join(colour_letter(v) for v in palette)}, "
            f"{len(dsl.objects(g, True, False, True))} non-background components")
