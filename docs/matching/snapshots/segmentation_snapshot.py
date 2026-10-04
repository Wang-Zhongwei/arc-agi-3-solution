"""Connected-component segmentation of a single frame layer.

This module is intentionally self-contained -- standard library only, no project
imports, no ``from __future__`` import -- so its source can be spliced verbatim into
the Python-tool sandbox bootstrap, where project packages are not importable.
"""

import hashlib
import math

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
# clockwise Moore-neighbour offsets, starting at NW
_CW = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
_CW_INDEX = {off: i for i, off in enumerate(_CW)}


def _trace_outer_contour(cells, start):
    """Moore-neighbour trace of a 4-connected component's outer perimeter, clockwise."""
    if len(cells) == 1:
        return [start]

    contour = [start]
    b = start
    prev = (start[0], start[1] - 1)  # W neighbour: outside the component since start is reading-order-min
    second = None
    for _ in range(8 * len(cells) + 16):
        idx = _CW_INDEX[(prev[0] - b[0], prev[1] - b[1])]
        nxt = None
        for k in range(1, 9):
            off = _CW[(idx + k) % 8]
            cand = (b[0] + off[0], b[1] + off[1])
            if cand in cells:
                nxt = cand
                back = _CW[(idx + k - 1) % 8]
                new_prev = (b[0] + back[0], b[1] + back[1])
                break
        if nxt is None:
            break
        if second is None:
            second = nxt
        elif b == start and nxt == second:  # Jacob's stopping criterion
            break
        contour.append(nxt)
        prev, b = new_prev, nxt

    if len(contour) > 1 and contour[-1] == contour[0]:
        contour.pop()
    return contour


def _corner_points(contour):
    """Reduce a traced contour loop to only the points where its direction changes."""
    if len(contour) <= 2:
        return list(contour)
    m = len(contour)
    corners = []
    for i in range(m):
        prev, cur, nxt = contour[i - 1], contour[i], contour[(i + 1) % m]
        d_in = (cur[0] - prev[0], cur[1] - prev[1])
        d_out = (nxt[0] - cur[0], nxt[1] - cur[1])
        if d_in != d_out:
            corners.append(cur)
    return corners


def _normalize_cells(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _rotate_cells_90(norm_cells):
    """Rotate a normalized cell set 90 degrees clockwise; returns normalized."""
    max_r = max(r for r, _ in norm_cells)
    return sorted((c, max_r - r) for r, c in norm_cells)


def _object_scale(cells):
    """Largest exact uniform square-block factor of the tight occupancy mask."""
    norm = _normalize_cells(cells)
    height = max(r for r, _ in norm) + 1
    width = max(c for _, c in norm) + 1
    common = math.gcd(height, width)
    for scale in range(common, 1, -1):
        if common % scale or len(norm) % (scale * scale):
            continue
        blocks = {}
        for r, c in norm:
            block = (r // scale, c // scale)
            blocks[block] = blocks.get(block, 0) + 1
        # Every occupied block must be full; all other blocks are entirely zero.
        if all(count == scale * scale for count in blocks.values()):
            return scale
    return 1


def _mask_encoding(cells):
    height = max(r for r, _ in cells) + 1
    width = max(c for _, c in cells) + 1
    occupied = set(cells)
    return height, width, tuple(int((r, c) in occupied)
                               for r in range(height) for c in range(width))


def _reflect_cells(cells):
    width = max(c for _, c in cells) + 1
    return sorted((r, width - 1 - c) for r, c in cells)


def _d4_variants(cells):
    for mirrored in (False, True):
        variant = _reflect_cells(cells) if mirrored else cells
        for angle in (0, 90, 180, 270):
            yield mirrored, angle, variant
            variant = _rotate_cells_90(variant)


def _mask_hash(encoding, color=None):
    payload = encoding if color is None else (color, encoding)
    return hashlib.sha1(repr(payload).encode()).hexdigest()[:16]


def _silhouette_hash(cells):
    """Fill holes unreachable from the padded exterior by four-neighbor steps."""
    occupied = set(_normalize_cells(cells))
    height = max(r for r, _ in occupied) + 1
    width = max(c for _, c in occupied) + 1
    reached = {(-1, -1)}
    stack = [(-1, -1)]
    while stack:
        r, c = stack.pop()
        for dr, dc in _ORTH:
            point = (r + dr, c + dc)
            if (-1 <= point[0] <= height and -1 <= point[1] <= width
                    and point not in occupied and point not in reached):
                reached.add(point)
                stack.append(point)
    filled = [(r, c) for r in range(height) for c in range(width)
              if (r, c) not in reached]
    return _mask_hash(min(_mask_encoding(v) for _, _, v in _d4_variants(filled)))


def _object_signature(cells, color):
    """D4 hashes and pose relative to the canonical reduced, color-free mask."""
    norm = _normalize_cells(cells)
    canonical = min(_mask_encoding(v) for _, _, v in _d4_variants(norm))
    scale = _object_scale(norm)
    reduced = sorted({(r // scale, c // scale) for r, c in norm})
    reference = min((v for _, _, v in _d4_variants(reduced)), key=_mask_encoding)
    matches = [(mirror, angle) for mirror, angle, v in _d4_variants(reference)
               if v == reduced]
    flags = {mirror for mirror, _ in matches}
    mirrored = None if len(flags) == 2 else matches[0][0]
    rotation = min(angle for mirror, angle in matches if mirror == bool(mirrored))
    symmetry = sum(v == reduced for mirror, _, v in _d4_variants(reduced) if not mirror)
    if symmetry == 4:
        rotation = None
    return (_mask_hash(canonical, color), rotation, symmetry,
            _mask_hash(_mask_encoding(norm), color), _mask_hash(canonical), mirrored,
            _mask_hash(_mask_encoding(reference)))


def segment_layer(layer, color_chars):
    """Segment one frame layer into connected-component nodes.

    Pass a single layer (if the frame has multiple) and ``color_chars``, the ARC
    color-symbol mapping (indexed by integer color value -> single-char label). The
    layer is partitioned into 4-connected components of equal integer value via flood
    fill, and each component becomes a node. Nodes are ordered by their
    top-most-left-most cell -- unique within a 64x64 layer -- and that order is the
    node ``id``.

    Each node is a dict with:
      - ``id``: index in the top-left ordering.
      - ``color``: the component's ARC color character (looked up in ``color_chars``).
      - ``hash``: colored-mask signature ignoring position, rotation and reflection.
      - ``shape_hash``: the same equivalence without color.
      - ``form_hash``: shape signature after exact integer scale reduction.
      - ``silhouette_hash``: shape signature after filling enclosed holes; retains size.
      - ``pose_hash``: exact colored-mask signature ignoring position only.
      - ``rotation`` and ``mirrored``: canonical-to-observed pose; reflect left-right
        first, then rotate clockwise. None denotes symmetry ambiguity.
      - ``pixels``: number of cells in the component.
      - ``scale``: largest exact square-block factor of its own tight binary mask.
      - ``bbox``: inclusive ``(r0, c0, r1, c1)`` bounds of the component's own
        cells, in frame coordinates.
      - ``boundary``: the component's outer perimeter as an ordered, clockwise list of
        ``[row, col]`` corner points -- a Moore-neighbour trace reduced to only the
        vertices where the contour changes direction (enclosed holes are not traced).
      - ``children``: ids of components directly enclosed by this node. A is a child of
        B only if B is the innermost component that fully surrounds A (every path from A
        to the grid edge crosses B), which yields a clean nesting tree.
      - ``parent``: id of the innermost enclosing component, or ``None`` if no
        component encloses this one. This is the inverse of ``children``.

    Returns a dict with:
      - ``nodes``: list of the node dicts above, in id order.
      - ``adjacency_list``: sorted list of ``[i, j]`` id pairs for components that share
        a 4-connected edge (includes parent/child pairs, since they physically touch).
    """
    height = len(layer)
    width = len(layer[0]) if height else 0

    # connected components, 4-connectivity. Reading-order scan => component ids are
    # already ordered by top-most-left-most cell (each layer is 64x64, so it is unique).
    comp_id = [[-1] * width for _ in range(height)]
    components = []  # each: {"value": int, "cells": set[(r, c)], "start": (r, c)}
    for sr in range(height):
        for sc in range(width):
            if comp_id[sr][sc] != -1:
                continue
            value = layer[sr][sc]
            cid = len(components)
            cells = set()
            stack = [(sr, sc)]
            comp_id[sr][sc] = cid
            while stack:
                r, c = stack.pop()
                cells.add((r, c))
                for dr, dc in _ORTH:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and comp_id[nr][nc] == -1 and layer[nr][nc] == value:
                        comp_id[nr][nc] = cid
                        stack.append((nr, nc))
            components.append({"value": int(value), "cells": cells, "start": (sr, sc)})

    n = len(components)

    # adjacency between components: any two components with 4-adjacent cells
    adj_pairs = set()
    for r in range(height):
        for c in range(width):
            cid = comp_id[r][c]
            if r + 1 < height and comp_id[r + 1][c] != cid:
                other = comp_id[r + 1][c]
                adj_pairs.add((min(cid, other), max(cid, other)))
            if c + 1 < width and comp_id[r][c + 1] != cid:
                other = comp_id[r][c + 1]
                adj_pairs.add((min(cid, other), max(cid, other)))
    adjacency_list = sorted([a, b] for a, b in adj_pairs)

    # containment: for each component b, flood-fill its complement inward from the grid
    # border; any component whose cells are never reached is enclosed by b.
    enclosers = [set() for _ in range(n)]
    for b in range(n):
        reached = [[False] * width for _ in range(height)]
        stack = []
        for r in range(height):
            for c in (0, width - 1):
                if comp_id[r][c] != b and not reached[r][c]:
                    reached[r][c] = True
                    stack.append((r, c))
        for c in range(width):
            for r in (0, height - 1):
                if comp_id[r][c] != b and not reached[r][c]:
                    reached[r][c] = True
                    stack.append((r, c))
        while stack:
            r, c = stack.pop()
            for dr, dc in _ORTH:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and not reached[nr][nc] and comp_id[nr][nc] != b:
                    reached[nr][nc] = True
                    stack.append((nr, nc))
        for a in range(n):
            if a == b:
                continue
            ar, ac = components[a]["start"]
            if not reached[ar][ac]:
                enclosers[a].add(b)

    # parent = innermost encloser. enclosers are transitive, so along a nesting chain the
    # innermost component is the one that is itself most deeply enclosed.
    children = [[] for _ in range(n)]
    parents = [None] * n
    for a in range(n):
        if enclosers[a]:
            parent = max(enclosers[a], key=lambda e: (len(enclosers[e]), -e))
            parents[a] = parent
            children[parent].append(a)
    for child_list in children:
        child_list.sort()

    nodes = []
    for cid in range(n):
        comp = components[cid]
        color = color_chars[max(0, min(15, comp["value"]))]
        boundary = _corner_points(_trace_outer_contour(comp["cells"], comp["start"]))
        signature = _object_signature(comp["cells"], color)
        rows = [r for r, _ in comp["cells"]]
        cols = [c for _, c in comp["cells"]]
        nodes.append(
            {
                "id": cid,
                "color": color,
                "hash": signature[0],
                "rotation": signature[1],
                "rotational_symmetry": signature[2],
                "pose_hash": signature[3],
                "shape_hash": signature[4],
                "mirrored": signature[5],
                "form_hash": signature[6],
                "silhouette_hash": _silhouette_hash(comp["cells"]),
                "pixels": len(comp["cells"]),
                "scale": _object_scale(comp["cells"]),
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "boundary": [[r, c] for r, c in boundary],
                "children": children[cid],
                "parent": parents[cid],
            }
        )

    return {"nodes": nodes, "adjacency_list": adjacency_list}
