"""Animation awareness.

``arcengine`` renders a frame after every internal ``step()``, so a single
action can come back as a short animation. ``GameState.frame`` is only the last
of those, and the harness consumed nothing else - so every intermediate frame
was discarded before the agent could see it. On some games the whole effect of
an action lives there: the board before and the board after are identical, and
reading the final frame cannot possibly reveal what happened.

Two shapes occur. Where cells change and change BACK, the information is
genuinely hidden and worth spending prompt tokens on. Where every cell changes
at most once, the frames are motion interpolation toward a state the final
frame already shows, and there is nothing to find - those animations are
reported nowhere, because a line about them would cost tokens to say that
nothing is missing.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any, Collection, Sequence

from inference.utils.grid_utils import ARC_COLOR_CHARS, format_grid_ascii

Grid = tuple[tuple[int, ...], ...]

# Retrieval budget.
#
# Why a diff timeline rather than the frames themselves: one 64x64 grid renders
# as roughly 4,100 characters, and a single action can return dozens of frames.
# Whole frames are never affordable against a ~1,024-token tool budget, so
# retrieval collapses identical frames and reports only the cells that changed
# between the rest. A single frame can still be read verbatim, cropped.
# No step limit: the timeline crosses the pipe as data and the model prints
# only what it chooses, so dropping steps served a prompt budget that no longer
# applies. It also cost a real misreading - a step count reported as omitted
# was read as "those steps had no changes", the exact inverse.
ANIMATION_MAX_CELLS_PER_STEP = 24
ANIMATION_MAX_TOTAL_CELLS = 80


def normalize_frames(raw_frames: Any) -> list[Grid]:
    """Engine frame list to hashable, comparable grids."""
    frames: list[Grid] = []
    for frame in raw_frames or ():
        rows = frame.tolist() if hasattr(frame, "tolist") else frame
        frames.append(tuple(tuple(int(cell) for cell in row) for row in rows or ()))
    return frames


def collapse_frames(before: Grid | None, frames: Sequence[Grid]) -> list[Grid]:
    """The pre-action board, then each distinct state the action passed through.

    Runs of identical grids collapse to one entry; non-adjacent repeats stay
    separate, because a grid at positions 2 and 7 with something between them
    is two distinct moments and the timeline should show both transitions.

    ``before`` is prepended and is NOT optional. A cell showing X before the
    action, Y during it and X again at the end arrives from the engine as
    [Y, Y, X] - one transition, which would not register as transient. With the
    board before it is [X, Y, Y, X], which gives two. Without this the very
    case the feature exists for is invisible.
    """
    collapsed: list[Grid] = []
    if before is not None:
        collapsed.append(before)
    for grid in frames:
        if collapsed and collapsed[-1] == grid:
            continue
        collapsed.append(grid)
    return collapsed


def _transient_cells(chain: Sequence[Grid]) -> list[tuple[int, int]]:
    """Cells whose value changes two or more times across the chain.

    A cell that changes once and stays changed is visible in the final frame,
    so the agent loses nothing by never seeing the frames between. A cell that
    changes twice has shown something that is no longer there - a rejected
    click, a consumed attempt, a sprite crossing a background cell. Those are
    exactly the pixels no frame the agent can reach will ever contain.
    """
    changes: defaultdict[tuple[int, int], int] = defaultdict(int)
    for before, after in zip(chain, chain[1:]):
        rows = max(len(before), len(after))
        for row in range(rows):
            before_row = before[row] if row < len(before) else ()
            after_row = after[row] if row < len(after) else ()
            for col in range(max(len(before_row), len(after_row))):
                old = before_row[col] if col < len(before_row) else None
                new = after_row[col] if col < len(after_row) else None
                if old != new:
                    changes[(row, col)] += 1
    return sorted(cell for cell, count in changes.items() if count >= 2)


def summarize_animation(chain: Sequence[Grid]) -> dict[str, Any] | None:
    """Compact metadata for one action, or ``None`` if it did not animate.

    ``None`` for the single-frame case means ordinary actions carry no extra
    tokens at all - the key is simply absent from the action result.

    ``frames`` counts engine frames after collapsing and excludes the prepended
    pre-action board, so it answers "how many distinct states did this action
    pass through" rather than describing the internal representation.
    """
    if len(chain) <= 2:
        return None
    transient = _transient_cells(chain)
    summary: dict[str, Any] = {
        "frames": len(chain) - 1,
        "transient_pixels": len(transient),
    }
    if transient:
        rows = [cell[0] for cell in transient]
        cols = [cell[1] for cell in transient]
        # Inclusive [top, left, bottom, right]: enough to point at the region
        # without spending tokens on a coordinate list.
        summary["transient_bbox"] = [min(rows), min(cols), max(rows), max(cols)]
    return summary


def _bbox_phrase(bbox: Sequence[int] | None) -> str:
    """Where the transient cells are, collapsing degenerate ranges.

    A single cell reads as "at (20,30)", a single row or column as a span in
    the other axis only; "rows 20-20, cols 30-30" is noise the model has to
    parse before noticing it means one cell.
    """
    if not bbox:
        return ""
    top, left, bottom, right = bbox
    if top == bottom and left == right:
        return f" at ({top},{left})"
    if top == bottom:
        return f" in row {top}, cols {left}-{right}"
    if left == right:
        return f" in col {left}, rows {top}-{bottom}"
    return f" around rows {top}-{bottom}, cols {left}-{right}"


def describe_animation(
    summary: dict[str, Any] | None,
    *,
    action_display: str | None,
    gameplay_changed: bool | None,
    game_over: bool = False,
    timeline: bool = True,
    border_cells: int = 0,
) -> str:
    """The opener line, or "" when there is nothing worth saying.

    ``border_cells`` (ARC3_EDGE_NOOP_FACT): cells in the edge band that the
    action changed for good, named when the board area ended unchanged.

    Silent unless cells changed and changed back. Where every cell changed at
    most once the final frame already shows the outcome, and a line saying so
    would spend tokens to report that nothing is missing.
    """
    if not summary:
        return ""
    transient = int(summary.get("transient_pixels") or 0)
    if transient <= 0:
        return ""
    frames = summary.get("frames")
    cells = "1 cell changed and changed back" if transient == 1 else (
        f"{transient} cells changed and changed back"
    )
    where = _bbox_phrase(summary.get("transient_bbox"))
    name = f" (`{action_display}`)" if action_display else ""
    # point at whichever accessor is actually available
    where_to_look = (
        "`last_animation_timeline`"
        if timeline
        else "`last_animation_frames`"
    )

    if game_over:
        # Follows the BAR RULE block, which has just walked the model through
        # history[-3] and history[-2]. This says what those two frames cannot
        # contain, which is precisely what "transient" means.
        return (
            f"That fatal action animated over {frames} frames, and inside it "
            f"{cells}{where}. Those cells are in neither "
            f"`history[-3].frame` nor `history[-2].frame` - read {where_to_look} to see "
            "what happened between them."
        )
    if gameplay_changed is False:
        if border_cells > 0:
            stayed = "1 border cell" if border_cells == 1 else f"{border_cells} border cells"
            return (
                f"The last executed action{name} animated over {frames} frames and left the "
                f"board area unchanged, but it was not a no-op: {cells}{where}, and "
                f"{stayed} changed and stayed changed. "
                "Whatever else this action did is visible only there - "
                f"read {where_to_look} before concluding it failed."
            )
        return (
            f"The last executed action{name} animated over {frames} frames and left the "
            f"board area unchanged, but it was not a no-op: {cells}{where}. "
            "Whatever this action did is visible only there - "
            f"read {where_to_look} before concluding it failed."
        )
    return (
        f"The last executed action{name} animated over {frames} frames. "
        f"{cells[0].upper()}{cells[1:]}{where}. {where_to_look} shows them."
    )


def _color_char(value: int | None) -> str:
    if value is None:
        return "?"
    return ARC_COLOR_CHARS[max(0, min(15, int(value)))]


def _cell_at(grid: Grid, row: int, col: int) -> int | None:
    if row >= len(grid):
        return None
    line = grid[row]
    return line[col] if col < len(line) else None


def _diff_cells(before: Grid, after: Grid) -> list[tuple[int, int, int | None, int | None]]:
    rows = max(len(before), len(after))
    changes: list[tuple[int, int, int | None, int | None]] = []
    for row in range(rows):
        cols = max(
            len(before[row]) if row < len(before) else 0,
            len(after[row]) if row < len(after) else 0,
        )
        for col in range(cols):
            old = _cell_at(before, row, col)
            new = _cell_at(after, row, col)
            if old != new:
                changes.append((row, col, old, new))
    return changes


def _bbox_text(cells: Sequence[tuple[int, int, Any, Any]]) -> str:
    rows = [cell[0] for cell in cells]
    cols = [cell[1] for cell in cells]
    return f"rows {min(rows)}-{max(rows)}, cols {min(cols)}-{max(cols)}"


def _format_changes(
    cells: Sequence[tuple[int, int, int | None, int | None]], budget: int
) -> list[str]:
    """Changed cells grouped by colour transition, one compact line each.

    Strings rather than nested coordinate lists because the caller renders with
    json.dumps(indent=2), which would put every [row, col] pair on four lines.
    """
    grouped: dict[str, list[str]] = {}
    for row, col, old, new in cells[:budget]:
        key = f"{_color_char(old)}>{_color_char(new)}"
        grouped.setdefault(key, []).append(f"({row},{col})")
    lines = [f"{key} @ {' '.join(points)}" for key, points in grouped.items()]
    if len(cells) > budget:
        lines.append(f"... {len(cells) - budget} further changed cells omitted")
    return lines


def build_animation_frames(record: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Every distinct state the action passed through, as FrameView payloads.

    The pre-action board is dropped here: the model already has it as
    `previous_frame`, and including it would make index 0 mean something
    different from the step numbers in the timeline.

    Sends the grid as well as the ascii so `segmentation` works on the far
    side. That doubles the payload, but it crosses a pipe rather than the
    prompt - what reaches the model is only what its own code prints.
    """
    if not record:
        return []
    chain = list(record.get("chain") or ())
    if len(chain) <= 2:
        return []
    step = record.get("step") or 0
    level = record.get("level") or 0
    return [
        {
            "ascii": format_grid_ascii(grid),
            "step": step,
            "level": level,
            "shape": (len(grid), max((len(row) for row in grid), default=0)),
            "grid": [list(row) for row in grid],
        }
        for grid in chain[1:]
    ]


def build_animation_view(
    record: dict[str, Any] | None, *, action_num: bool = False
) -> dict[str, Any]:
    """The diff timeline: which cells changed at each step.

    The cheap first look. Reading a frame verbatim is no longer this
    function's job - `last_transition.frames()` hands back FrameView objects
    and the model crops their ascii the same way it crops current_frame, with
    its own helpers rather than a region parameter it has to guess.

    ``action_num`` (the ring, ARC3_ANIM_RING) adds the action's number to the
    header, so a timeline read later says which action it belongs to.
    """
    empty = {
        "error": (
            "No animation is available: no executed action has animated yet."
            if action_num
            else "No animation is available: the last executed action returned a single frame."
        )
    }
    if not record:
        return empty
    chain: list[Grid] = list(record.get("chain") or ())
    if len(chain) <= 2:
        return empty

    header: dict[str, Any] = {
        "action": record.get("action_display"),
        # excludes the prepended pre-action board, matching the metadata
        "frames": len(chain) - 1,
    }
    if action_num:
        header = {"action_num": record.get("step"), **header}

    steps: list[dict[str, Any]] = []
    for position in range(1, len(chain)):
        cells = _diff_cells(chain[position - 1], chain[position])
        if cells:
            steps.append({"_position": position, "_cells": cells})

    rendered: list[dict[str, Any]] = []
    remaining = ANIMATION_MAX_TOTAL_CELLS
    for step in steps:
        cells = step["_cells"]
        entry: dict[str, Any] = {
            "step": step["_position"],
            "changed": len(cells),
            "bbox": _bbox_text(cells),
        }
        budget = min(ANIMATION_MAX_CELLS_PER_STEP, max(0, remaining))
        if len(cells) > ANIMATION_MAX_CELLS_PER_STEP or budget == 0:
            # Too broad to enumerate usefully: a colour-transition census says
            # more per token than an arbitrarily truncated cell list.
            census: dict[str, int] = {}
            for _, _, old, new in cells:
                key = f"{_color_char(old)}>{_color_char(new)}"
                census[key] = census.get(key, 0) + 1
            entry["transitions"] = census
        else:
            entry["changes"] = _format_changes(cells, budget)
            remaining -= min(len(cells), budget)
        rendered.append(entry)

    return {**header, "steps": rendered}


# ---------------------------------------------------------------------------
# Animation type (ARC3_ANIM_TRUTH's verdict word, ARC3_ANIM_DIGEST's headers).
#
# lordhansolo's classifier (competitors/kaggle-datasets/lordhansolo-taaf-
# kaggle-source, agent/animation.py), on our chain: S0 = board before, then
# each distinct state, the last one settled.
#   blink - the chain alternates between two boards (a flash that reverts is
#           the one-cycle case: S0, flash, S0).
#   tween - one sprite slides in a straight line from where it was first seen
#           to where it sits on the settled board.
#   path  - one sprite moves through positions that are not such a line, or
#           is gone from the settled board.
#   other - anything else.
# A sprite is a non-background, four-connected, one-colour shape that occurs
# exactly once, outside the static scenery, in every intermediate state and
# passes through at least three positions; parts moving in lockstep count as
# one sprite. Nothing here reads frame_diff, so issue #41 (recolour reported
# as a move) does not apply. Only reached with one of those two flags on.
# ---------------------------------------------------------------------------

ANIMATION_TYPE_MAX_STATES = 16
_TRACK_MIN_POINTS = 3
_PATH_POINT_LIMIT = 8


def _mask_border(chain: Sequence[Grid], border: int) -> list[Grid]:
    """Every state with its outer ``border`` cells taken from the settled board,
    re-collapsed: a HUD bar ticking there must not make a blink look like
    three boards or add a sprite."""
    if border <= 0 or not chain:
        return list(chain)
    final = chain[-1]
    rows = len(final)
    cols = max((len(row) for row in final), default=0)
    if rows <= 2 * border or cols <= 2 * border:
        return list(chain)
    masked: list[Grid] = []
    for grid in chain:
        if not _same_shape(grid, final):
            masked.append(grid)
            continue
        out = []
        for index, (row, final_row) in enumerate(zip(grid, final)):
            if index < border or index >= rows - border:
                out.append(final_row)
            else:
                out.append(final_row[:border] + row[border:cols - border] + final_row[cols - border:])
        masked.append(tuple(out))
    collapsed: list[Grid] = []
    for grid in masked:
        if not collapsed or collapsed[-1] != grid:
            collapsed.append(grid)
    return collapsed


def _shape_components(
    grid: Grid, background: int | None, region: tuple[int, int, int, int]
) -> dict[tuple[int, frozenset], list[tuple[int, int]]]:
    """{(colour, shape offsets): [top-left, ...]} of the non-background blobs
    inside ``region`` (inclusive top, left, bottom, right)."""
    top, left, bottom, right = region
    by_color: dict[int, set[tuple[int, int]]] = {}
    for row in range(top, bottom + 1):
        line = grid[row] if row < len(grid) else ()
        for col in range(left, min(right + 1, len(line))):
            value = line[col]
            if value != background:
                by_color.setdefault(int(value), set()).add((row, col))
    found: dict[tuple[int, frozenset], list[tuple[int, int]]] = {}
    for color, cells in by_color.items():
        for component in _components(cells):
            r0 = min(cell[0] for cell in component)
            c0 = min(cell[1] for cell in component)
            shape = frozenset((r - r0, c - c0) for r, c in component)
            found.setdefault((color, shape), []).append((r0, c0))
    return found


def _nearest(copies: list[tuple[int, int]], anchor: tuple[int, int]) -> tuple[int, int] | None:
    """The copy nearest ``anchor``; None if none or if two tie (a guess)."""
    if not copies:
        return None
    ranked = sorted((abs(r - anchor[0]) + abs(c - anchor[1]), (r, c)) for r, c in copies)
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None
    return ranked[0][1]


def _straight_monotonic(points: list[tuple[int, int]]) -> bool:
    start, end = points[0], points[-1]
    dr, dc = end[0] - start[0], end[1] - start[1]
    if (dr, dc) == (0, 0):
        return False
    previous = 0
    for row, col in points[1:]:
        orow, ocol = row - start[0], col - start[1]
        if dr * ocol != dc * orow:
            return False
        progress = dr * orow + dc * ocol
        if progress < previous or progress > dr * dr + dc * dc:
            return False
        previous = progress
    return True


def classify_animation(chain: Sequence[Grid], border: int = 0) -> dict[str, Any]:
    """{"type": "blink"|"tween"|"path"|"other", ...} for an animated chain.

    blink: ``cycles`` (states showing the other board). tween: ``color``,
    ``size`` ("HxW"), ``from``/``to`` top-left. path: ``color``, ``size``,
    ``points`` (at most 8, top-left, time order). {} below three states.
    """
    chain = _mask_border(list(chain), border)
    if len(chain) < 3:
        return {}
    settled = chain[-1]
    if len(set(chain)) == 2:
        return {"type": "blink", "cycles": sum(1 for grid in chain[1:] if grid != settled)}
    sample = cap_chain(chain, ANIMATION_TYPE_MAX_STATES)
    before, transient = sample[0], sample[1:-1]
    changed = [
        (row, col)
        for older, newer in zip(sample, sample[1:])
        for row, col, _, _ in _step_changes(older, newer)
    ]
    if not changed or not transient:
        return {"type": "other"}
    height = len(settled)
    width = max((len(row) for row in settled), default=0)
    # the blobs that matter all touch a changed cell; a margin keeps a sprite
    # whole when only part of it changed
    margin = 4
    region = (
        max(0, min(r for r, _ in changed) - margin), max(0, min(c for _, c in changed) - margin),
        min(height - 1, max(r for r, _ in changed) + margin),
        min(width - 1, max(c for _, c in changed) + margin),
    )
    background = _background(settled)
    before_parts = _shape_components(before, background, region)
    after_parts = _shape_components(settled, background, region)
    static = {
        (key, pos) for key, positions in before_parts.items() for pos in positions
    } & {(key, pos) for key, positions in after_parts.items() for pos in positions}
    per_frame = []
    for grid in transient:
        dynamic: dict[Any, list[tuple[int, int]]] = {}
        for key, positions in _shape_components(grid, background, region).items():
            moving = [pos for pos in positions if (key, pos) not in static]
            if moving:
                dynamic[key] = moving
        per_frame.append(dynamic)
    shared = set.intersection(*[set(frame) for frame in per_frame])
    tracks = []
    for key in sorted(shared, key=lambda item: (item[0], sorted(item[1]))):
        if any(len(frame[key]) != 1 for frame in per_frame):
            continue
        positions = [frame[key][0] for frame in per_frame]

        def _copies(parts):
            all_copies = parts.get(key, [])
            moved = [pos for pos in all_copies if (key, pos) not in static]
            return moved or all_copies

        start = _nearest(_copies(before_parts), positions[0])
        end = _nearest(_copies(after_parts), positions[-1])
        points: list[tuple[int, int]] = []
        for point in (start, *positions, end):
            if point is not None and (not points or points[-1] != point):
                points.append(point)
        if len(points) >= _TRACK_MIN_POINTS:
            tracks.append((key, positions, end, points))
    groups: dict[Any, list] = {}
    for index, track in enumerate(tracks):
        origin = track[1][0]
        displacement = tuple((r - origin[0], c - origin[1]) for r, c in track[1])
        alone = len(displacement) < 2
        groups.setdefault((displacement, index if alone else -1), []).append(track)
    if len(groups) != 1:
        return {"type": "other"}
    key, _, end, points = max(next(iter(groups.values())), key=lambda track: len(track[0][1]))
    color, shape = key
    size = f"{max(r for r, _ in shape) + 1}x{max(c for _, c in shape) + 1}"
    if end is not None and _straight_monotonic(points):
        return {"type": "tween", "color": _color_char(color), "size": size,
                "from": points[0], "to": points[-1]}
    return {"type": "path", "color": _color_char(color), "size": size,
            "points": [points[i] for i in even_subsample_indices(len(points), _PATH_POINT_LIMIT)]}


def describe_animation_type(kind: dict[str, Any] | None) -> str:
    """One short phrase: "blink x2", "tween R 2x2 (3,4)>(3,9) d(0,+5)",
    "path R 1x1 (1,1)>(1,5)>(4,5)", "other"; "" for {}."""
    if not kind:
        return ""
    name = kind.get("type")
    if name == "blink":
        return f"blink x{kind.get('cycles')}"
    if name == "tween":
        (r0, c0), (r1, c1) = kind["from"], kind["to"]
        return (f"tween {kind['color']} {kind['size']} ({r0},{c0})>({r1},{c1}) "
                f"d({r1 - r0:+d},{c1 - c0:+d})")
    if name == "path":
        return f"path {kind['color']} {kind['size']} " + ">".join(f"({r},{c})" for r, c in kind["points"])
    return "other"


# ---------------------------------------------------------------------------
# Animation digest (ARC3_ANIM_DIGEST). Everything below is only reached when
# that flag is on; nothing above calls it.
#
# The record above is one action deep and the opener only POINTS at it. The
# digest closes both gaps: the solver keeps a small ring of recent animated
# actions, and the opener carries a short, capped text account of the ones
# that are new or hid something. Text rather than images because the model
# can compute on text and the history keeps it verbatim, so it is appended
# once and never rewritten (prefix-cache safe).
# ---------------------------------------------------------------------------

# Boards kept per ring record, S0 included (so 31 states). sb26 returns up
# to 111 raw frames per action; 32 keeps a ring of 8 near 1 MB while every
# animation measured elsewhere (sk48: 2, cd82: 15) is stored whole.
ANIMATION_RING_MAX_FRAMES = 32

_HEX_DIGITS = "0123456789abcdef"


def even_subsample_indices(count: int, limit: int) -> list[int]:
    """Indices of an even subsample of ``count`` items, first and last kept.

    The same rule as Wang's frame cap (``round(i*(n-1)/(limit-1))``), so the
    extremes of a long motion survive however long it ran. With
    ``count > limit`` the stride exceeds 1 and the rounded indices are
    distinct, so exactly ``limit`` come back.
    """
    if count <= 0 or limit <= 0:
        return []
    if count <= limit:
        return list(range(count))
    if limit == 1:
        return [count - 1]
    return sorted({round(i * (count - 1) / (limit - 1)) for i in range(limit)})


def cap_chain(chain: Sequence[Grid], limit: int = ANIMATION_RING_MAX_FRAMES) -> list[Grid]:
    """At most ``limit`` states of an already collapsed chain, first/last kept.

    Applied AFTER ``collapse_frames``: a held pose collapses to one entry
    first, so it cannot eat the budget the cap is protecting.
    """
    return [chain[index] for index in even_subsample_indices(len(chain), limit)]


def encode_grid_hex(grid: Grid) -> tuple[str, ...]:
    """One string per row, one hex digit per cell (colours are 0-15)."""
    return tuple("".join(_HEX_DIGITS[max(0, min(15, int(v)))] for v in row) for row in grid)


def decode_grid_hex(rows: Sequence[str]) -> Grid:
    return tuple(tuple(int(ch, 16) for ch in row) for row in rows)


def decode_chain(frames: Sequence[Any] | None) -> list[Grid]:
    """Ring frames back to grids; accepts hex rows or grids already."""
    chain: list[Grid] = []
    for frame in frames or ():
        rows = list(frame or ())
        if rows and isinstance(rows[0], str):
            chain.append(decode_grid_hex(rows))
        else:
            chain.append(tuple(tuple(int(v) for v in row) for row in rows))
    return chain


def _same_shape(before: Grid, after: Grid) -> bool:
    return len(before) == len(after) and all(
        len(b) == len(a) for b, a in zip(before, after)
    )


def _step_changes(before: Grid, after: Grid) -> list[tuple[int, int, int | None, int | None]]:
    """``_diff_cells`` with a whole-row shortcut.

    Consecutive animation frames share almost every row, so comparing row
    tuples first skips nearly all the per-cell work. Same output as
    ``_diff_cells``; the slow path handles ragged or resized grids.
    """
    if not _same_shape(before, after):
        return _diff_cells(before, after)
    changes: list[tuple[int, int, int | None, int | None]] = []
    for row, (old_row, new_row) in enumerate(zip(before, after)):
        if old_row == new_row:
            continue
        for col, (old, new) in enumerate(zip(old_row, new_row)):
            if old != new:
                changes.append((row, col, old, new))
    return changes


def _transient_cell_set(chain: Sequence[Grid]) -> set[tuple[int, int]]:
    """``_transient_cells`` as a set, via the row shortcut."""
    counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    for before, after in zip(chain, chain[1:]):
        for row, col, _, _ in _step_changes(before, after):
            counts[(row, col)] += 1
    return {cell for cell, count in counts.items() if count >= 2}


def action_name_without_coords(action_display: str | None) -> str:
    """``MOUSE(row=3, col=4)`` -> ``MOUSE``; other names unchanged."""
    return re.sub(r"\s*\(.*\)\s*$", "", str(action_display or "")).strip()


def animation_signature(chain: Sequence[Grid], action_display: str | None = None) -> str:
    """A fingerprint of WHAT an animation did, independent of WHERE.

    The action name without coordinates plus, per step, the number of changed
    cells and a sorted colour-transition census. Positions are left out on
    purpose: the same mechanic triggered by a click on another cell should
    read as a repeat, not as something new.
    """
    steps: list[list[Any]] = []
    for before, after in zip(chain, chain[1:]):
        cells = _step_changes(before, after)
        census: dict[str, int] = {}
        for _, _, old, new in cells:
            key = f"{_color_char(old)}>{_color_char(new)}"
            census[key] = census.get(key, 0) + 1
        steps.append([len(cells), sorted(census.items())])
    blob = json.dumps(
        [action_name_without_coords(action_display), steps], separators=(",", ":")
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _peak_transient_index(chain: Sequence[Grid]) -> int | None:
    """The intermediate state showing the most transient cells, or None."""
    if len(chain) <= 2:
        return None
    transient = _transient_cell_set(chain)
    if not transient:
        return None
    final = chain[-1]
    best_index, best_count = None, 0
    for index in range(1, len(chain) - 1):
        grid = chain[index]
        count = sum(
            1 for row, col in transient if _cell_at(grid, row, col) != _cell_at(final, row, col)
        )
        if count > best_count:
            best_index, best_count = index, count
    return best_index


def select_keyframes(chain: Sequence[Grid], k: int = 4) -> list[int]:
    """Chain indices of up to ``k`` intermediate states worth showing.

    Only intermediates (S1..S(n-1)): S0 is the board the model already saw
    and S(n) is the one it is looking at. Even subsample with the first and
    last kept (Wang), then the peak-transient state swapped in for the
    nearest interior pick when the subsample missed it - the moment the most
    hidden cells are visible is the one frame that must not be dropped.
    """
    if k <= 0 or len(chain) <= 2:
        return []
    inner = list(range(1, len(chain) - 1))
    if len(inner) <= k:
        return inner
    picks = [inner[index] for index in even_subsample_indices(len(inner), k)]
    peak = _peak_transient_index(chain)
    if peak is not None and peak not in picks:
        candidates = picks[1:-1] if len(picks) >= 3 else picks
        nearest = min(candidates, key=lambda index: (abs(index - peak), index))
        picks[picks.index(nearest)] = peak
        picks.sort()
    return picks


def _components(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Four-connected components of a cell set."""
    remaining = set(cells)
    components: list[list[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        component = [seed]
        while stack:
            row, col = stack.pop()
            for neighbour in ((row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
                    component.append(neighbour)
        components.append(component)
    return components


def _background(grid: Grid) -> int | None:
    counts: dict[int, int] = {}
    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]


def overlay_trails(chain: Sequence[Grid], border: int = 0) -> list[dict[str, Any]]:
    """Lens A: where things were that are no longer there.

    For every state but the last, the cells that differ from the FINAL state
    and are not background, split into same-colour four-connected blobs.
    Blobs of equal colour and bounding-box size are chained across states
    into one trail, so a highlight sweeping over five tokens reads as one
    thing at five positions rather than five unrelated changes. This never
    claims identity or motion - it lists where a blob of that colour and
    size was seen, in time order - which is why it stays sound on the case
    object matching gets wrong (identical objects recoloured in place).

    The outer ``border`` cells are masked, as the no-op guard does, because
    the budget bar ticks there on every action and would lead every digest.
    Each trail: ``color``, ``height``, ``width``, ``pixels`` (largest blob)
    and ``points`` [(state, top, left)], sorted most informative first:
    coherent trails (at most three blobs per state) before scatter, then by
    how many states they span and blob size.
    """
    if len(chain) < 2:
        return []
    final = chain[-1]
    rows = len(final)
    cols = max((len(row) for row in final), default=0)
    if rows <= 2 * border or cols <= 2 * border:
        border = 0
    background = _background(final)
    points: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    pixels: dict[tuple[int, int, int], int] = {}
    for state, grid in enumerate(chain[:-1]):
        by_color: dict[int, set[tuple[int, int]]] = {}
        for row in range(border, rows - border):
            line = grid[row] if row < len(grid) else ()
            final_line = final[row]
            if line == final_line:
                continue
            for col in range(border, cols - border):
                value = line[col] if col < len(line) else None
                if value is None or value == background:
                    continue
                if value == (final_line[col] if col < len(final_line) else None):
                    continue
                by_color.setdefault(int(value), set()).add((row, col))
        for color, cells in by_color.items():
            for component in _components(cells):
                top = min(cell[0] for cell in component)
                left = min(cell[1] for cell in component)
                height = max(cell[0] for cell in component) - top + 1
                width = max(cell[1] for cell in component) - left + 1
                key = (color, height, width)
                points.setdefault(key, []).append((state, top, left))
                pixels[key] = max(pixels.get(key, 0), len(component))
    trails = [
        {
            "color": color,
            "height": height,
            "width": width,
            "pixels": pixels[(color, height, width)],
            "points": sorted(found),
        }
        for (color, height, width), found in points.items()
    ]
    def _crowded(trail: dict[str, Any]) -> bool:
        # more than three same-looking blobs in one state reads as scatter,
        # not as one thing moving; it goes after the coherent trails
        per_state: dict[int, int] = {}
        for state, _, _ in trail["points"]:
            per_state[state] = per_state.get(state, 0) + 1
        return max(per_state.values()) > 3

    trails.sort(
        key=lambda trail: (
            _crowded(trail),
            -len({point[0] for point in trail["points"]}),
            -trail["pixels"],
            trail["points"][0],
        )
    )
    return trails


def _render_trail(trail: dict[str, Any], max_points: int) -> str:
    """One trail as a line: positions in time order, held positions merged."""
    by_state: dict[int, list[tuple[int, int]]] = {}
    for state, top, left in trail["points"]:
        by_state.setdefault(state, []).append((top, left))
    segments: list[list[Any]] = []  # [first state, last state, positions]
    for state in sorted(by_state):
        where = tuple(sorted(by_state[state]))
        if segments and segments[-1][2] == where and segments[-1][1] == state - 1:
            segments[-1][1] = state
        else:
            segments.append([state, state, where])
    name = f"{_color_char(trail['color'])} {trail['height']}x{trail['width']}"
    contiguous = all(
        later[0] == earlier[1] + 1 for earlier, later in zip(segments, segments[1:])
    )
    simple = contiguous and all(
        len(seg[2]) == 1 and seg[0] == seg[1] for seg in segments
    )
    more = len(segments) - max_points
    shown = segments[:max_points]
    if simple:
        positions = " > ".join(f"({seg[2][0][0]},{seg[2][0][1]})" for seg in shown)
        if more > 0:
            positions += f" > ... (+{more} more)"
        first, last = segments[0][0], segments[-1][1]
        when = f"step S{first}" if first == last else f"steps S{first}-S{last}"
        return f"- {name} seen at {positions}; {when}"
    parts = []
    for first, last, where in shown:
        label = f"S{first}" if first == last else f"S{first}-S{last}"
        spots = "+".join(f"({top},{left})" for top, left in where[:3])
        if len(where) > 3:
            spots += f"+{len(where) - 3} more"
        parts.append(f"{label} {spots}")
    text = f"- {name} seen at " + " > ".join(parts)
    if more > 0:
        text += f" > ... (+{more} more)"
    return text


def _record_chain(record: dict[str, Any]) -> list[Grid]:
    return decode_chain(record.get("frames"))


def _transient_count(record: dict[str, Any]) -> int:
    summary = record.get("summary") or {}
    try:
        return int(summary.get("transient_pixels") or 0)
    except (TypeError, ValueError):
        return 0


def _subsampled(record: dict[str, Any]) -> bool:
    """The stored chain holds fewer states than the action passed through."""
    stored = len(_record_chain(record)) - 1
    summary = record.get("summary") or {}
    try:
        total = int(summary.get("frames") or stored)
    except (TypeError, ValueError):
        return False
    return stored >= 1 and total > stored


def _qualifies(record: dict[str, Any], gate: str) -> bool:
    if gate == "all":
        return True
    return _transient_count(record) > 0 or record.get("gameplay_changed") is False


def _entry_lines(
    record: dict[str, Any],
    *,
    border: int,
    settled: str,
    max_trails: int,
    max_points: int,
    readable: bool = True,
) -> list[str]:
    """Header line plus trail lines for one action shown in full.

    The header leads with the lordhansolo-style type (blink with cycles,
    tween with sprite, colour and from>to, path with up to 8 points): the
    most actionable fact and the line the budget keeps longest.
    """
    chain = _record_chain(record)
    summary = record.get("summary") or {}
    name = f"action {record.get('action_num')} `{record.get('action_display')}`"
    if len(chain) < 2:
        if readable:
            return [f"{name}: frames not kept; `animation_for({record.get('action_num')})`."]
        return [f"{name}: frames not kept."]
    stored = len(chain) - 1
    total = int(summary.get("frames") or stored)
    states = f"{stored} states (S0 = board before; S{stored} = {settled})"
    if total > stored:
        states += f", subsampled from {total}"
    kind = describe_animation_type(classify_animation(chain, border))
    header = f"{name}: {kind + '; ' if kind else ''}{states}."
    if record.get("gameplay_changed") is False:
        header += " Board area unchanged."
    transient = _transient_count(record)
    if transient == 1:
        header += " 1 cell changed and changed back."
    elif transient > 1:
        header += f" {transient} cells changed and changed back."
    lines = [header]
    trails = overlay_trails(chain, border)
    for trail in trails[:max_trails]:
        lines.append(_render_trail(trail, max_points))
    if len(trails) > max_trails:
        lines.append(f"- (+{len(trails) - max_trails} more trails)")
    if not trails:
        counts = [
            f"S{index} {len(_step_changes(chain[index - 1], chain[index]))}"
            for index in range(1, len(chain))
        ]
        shown = ", ".join(counts[:max_points])
        if len(counts) > max_points:
            shown += ", ..."
        lines.append(
            "- no non-background blob differs from the settled board inside the "
            f"border; changed cells per step: {shown}"
        )
    return lines


def build_animation_digest(
    records: Sequence[dict[str, Any]],
    budget_chars: int = 900,
    border: int = 0,
    seen: dict[str, int] | None = None,
    *,
    gate: str = "novel",
    last_action_num: int | None = None,
    game_over: bool = False,
    timeline: bool = True,
    max_entries: int = 3,
    max_trails: int = 3,
    max_points: int = 8,
    readable: Collection[int] | None = None,
) -> tuple[str, dict[str, int]]:
    """The opener's "Animation digest" block, and the signatures it showed.

    ``records`` are ring records (action_num, action_display, summary,
    signature, level, frames, gameplay_changed) for the animated actions of
    one sequence, oldest first. ``seen`` maps signatures already shown on
    this level to the action that showed them.

    Gate: ``transient`` shows an action whose animation hid something
    (transient pixels) or that left the board area unchanged; ``novel``
    (default) additionally needs an unseen signature, and turns a repeat into
    a one-line pointer; ``all`` shows every animated action. At most
    ``max_entries`` actions are shown in full - the first, the one with the
    most transient cells, and the last - and the rest are counted.

    ``readable`` holds the action numbers ``animation_for`` can still return
    (the solver's ring); the digest names ``animation_for(n)`` only for those.
    ``None`` means every action is readable.

    Returns ("", {}) when nothing qualifies. The text never exceeds
    ``budget_chars``: trail lines are dropped from the end first.
    """
    gate = gate if gate in ("novel", "transient", "all") else "novel"
    seen = dict(seen or {})
    records = [record for record in records if isinstance(record, dict)]
    if not records or budget_chars <= 0:
        return "", {}
    readable_nums = None if readable is None else {int(number) for number in readable}

    def _readable(record: dict[str, Any]) -> bool:
        return readable_nums is None or int(record.get("action_num") or 0) in readable_nums

    full: list[dict[str, Any]] = []
    pointers: dict[str, list[tuple[dict[str, Any], int]]] = {}
    first_in_call: dict[str, int] = {}
    for record in records:
        if not _qualifies(record, gate):
            continue
        signature = str(record.get("signature") or "")
        if gate == "novel":
            original = seen.get(signature, first_in_call.get(signature))
            if signature and original is not None:
                pointers.setdefault(signature, []).append((record, original))
                continue
        full.append(record)
        if signature:
            first_in_call.setdefault(signature, int(record.get("action_num") or 0))

    chosen: list[dict[str, Any]] = []
    if full:
        # positions, not records: dict equality would compare whole frame lists
        peak = max(range(len(full)), key=lambda i: (_transient_count(full[i]), -i))
        picked: list[int] = []
        for index in (0, peak, len(full) - 1):
            if index not in picked and len(picked) < max(1, max_entries):
                picked.append(index)
        chosen = [full[index] for index in sorted(picked)]

    # A pointer must name an original the model can find: one shown on an
    # earlier turn (``seen``) or in full in this digest. A repeat of an action
    # that was only counted is counted too, and stays novel for the next turn.
    chosen_nums = {int(record.get("action_num") or 0) for record in chosen}
    pointers = {
        signature: repeats
        for signature, repeats in pointers.items()
        if signature in seen or repeats[0][1] in chosen_nums
    }

    pointer_lines: list[str] = []
    represented = len(chosen)
    for signature, repeats in list(pointers.items())[:3]:
        record, original = repeats[-1]
        extra = len(repeats) - 1
        line = (
            f"action {record.get('action_num')} `{record.get('action_display')}` repeats "
            f"the animation of action {original} (same signature)"
        )
        if extra:
            line += f", as do {extra} earlier action(s) in this sequence"
        line += f"; `animation_for({record.get('action_num')})`." if _readable(record) else "."
        pointer_lines.append(line)
        represented += len(repeats)
    if not chosen and not pointer_lines:
        return "", {}
    omitted = len(records) - represented

    if not chosen:
        lines = ["Animation digest: " + pointer_lines[0], *pointer_lines[1:]]
        if omitted > 0:
            lines.append(
                f"(+{omitted} more animated action(s); `animations` lists the readable ones.)"
            )
        return _fit("\n".join(lines), budget_chars), {}

    # (priority, signature, text): trail lines go first when over budget,
    # then pointers, then later headers; the first header goes last.
    body: list[tuple[int, str, str]] = []
    for position, record in enumerate(chosen):
        is_last = last_action_num is not None and record.get("action_num") == last_action_num
        settled = (
            ("death frame" if game_over else "current board") if is_last else "board after it"
        )
        entry = _entry_lines(
            record, border=border, settled=settled,
            max_trails=max_trails, max_points=max_points, readable=_readable(record),
        )
        signature = str(record.get("signature") or "")
        body.append((3 if position == 0 else 2, signature, entry[0]))
        body.extend((0, "", line) for line in entry[1:])
    body.extend((1, "", line) for line in pointer_lines)

    merge_intro = len(chosen) == 1 and not pointer_lines and omitted <= 0
    if not merge_intro:
        animated = "animated action" if len(records) == 1 else "animated actions"
        intro = (
            f"Animation digest ({gate}), {len(chosen)} of {len(records)} {animated} "
            "in the sequence:"
        )
    tail: list[str] = []
    if omitted > 0:
        tail.append(f"(+{omitted} more animated action(s); `animations` lists the readable ones.)")
    # last_animation_* hold the FULL chain; the S-numbers above index the
    # stored one, so a subsampled record points at animation_for(n), which
    # returns the stored states
    only_last = (
        len(chosen) == 1
        and last_action_num is not None
        and chosen[0].get("action_num") == last_action_num
        and not game_over
        and not _subsampled(chosen[0])
    )
    view = "timeline" if timeline else "frames"
    heuristic = "Trails are heuristic (top-left of each blob not in the settled board). "
    if only_last and timeline:
        tail.append(
            heuristic
            + "Exact cells: last_animation_timeline; step k = last_animation_frames[k-1]."
        )
    elif only_last:
        tail.append(heuristic + "Exact states: last_animation_frames[k-1] = step S<k>.")
    else:
        kept = [record for record in chosen if _readable(record)]
        lost = [str(record.get("action_num")) for record in chosen if not _readable(record)]
        gone = (
            f"Action {lost[0]}'s frames are" if len(lost) == 1
            else f"Frames of actions {', '.join(lost)} are"
        )
        if kept:
            call = (
                f"animation_for({kept[0].get('action_num')})" if len(kept) == 1
                else "animation_for(n)"
            )
            line = heuristic + f"Exact: `{call}['{view}']`; S<k> = `{call}['frames'][k-1]`."
            if lost:
                line += f" {gone} no longer kept."
        else:
            line = heuristic + f"{gone} no longer kept."
        tail.append(line)

    def render(lines: list[tuple[int, str, str]], marker: bool, short: bool = False) -> str:
        texts = [line for _, _, line in lines]
        if merge_intro:
            head = [f"Animation digest ({gate}): {texts[0]}", *texts[1:]] if texts else []
        else:
            head = [intro, *texts]
        if marker:
            head.append("- (digest shortened to fit its budget)")
        footer = [line.replace(heuristic, "") for line in tail] if short else tail
        return "\n".join([*head, *footer])

    text = render(body, False)
    shortened = False
    while len(text) > budget_chars and len(body) > 1:
        # drop the lowest-priority line, latest first
        lowest = min(priority for priority, _, _ in body)
        index = max(i for i, (priority, _, _) in enumerate(body) if priority == lowest)
        del body[index]
        shortened = True
        text = render(body, True)
    if len(text) > budget_chars:
        text = render(body, shortened, short=True)
    shown = {signature for priority, signature, _ in body if priority >= 2}
    new_signatures = {
        str(record.get("signature")): int(record.get("action_num") or 0)
        for record in chosen
        if record.get("signature") and str(record.get("signature")) in shown
    }
    return _fit(text, budget_chars), new_signatures


def _fit(text: str, budget_chars: int) -> str:
    """Hard cap, for the case where even the header is over budget."""
    if len(text) <= budget_chars:
        return text
    if budget_chars <= 3:
        return text[:budget_chars]
    return text[: budget_chars - 3] + "..."
