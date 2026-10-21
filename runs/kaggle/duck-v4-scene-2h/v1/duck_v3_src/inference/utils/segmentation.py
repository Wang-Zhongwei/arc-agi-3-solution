"""Connected-component segmentation of a single frame layer.

This module is intentionally self-contained -- standard library only, no project
imports, no ``from __future__`` import -- so its source can be spliced verbatim into
the Python-tool sandbox bootstrap, where project packages are not importable.
"""

import hashlib

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


def _object_hash(cells, color):
    """Translation-invariant signature of an object: its color plus its cell shape,
    normalized so the top-left of its bounding box is the origin. Same shape + color
    => same hash regardless of position, so objects can be matched across frames."""
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    payload = repr((color, norm)).encode()
    return hashlib.sha1(payload).hexdigest()[:16]


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
      - ``hash``: a translation-invariant signature of the object -- its color plus its
        cell shape normalized to a top-left origin -- so the same shape gets the same
        hash regardless of position (lets objects be matched across frames, or when
        several similar objects appear in one frame).
      - ``pixels``: number of cells in the component.
      - ``boundary``: the component's outer perimeter as an ordered, clockwise list of
        ``[row, col]`` corner points -- a Moore-neighbour trace reduced to only the
        vertices where the contour changes direction (enclosed holes are not traced).
      - ``children``: ids of components directly enclosed by this node. A is a child of
        B only if B is the innermost component that fully surrounds A (every path from A
        to the grid edge crosses B), which yields a clean nesting tree.

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
    for a in range(n):
        if enclosers[a]:
            parent = max(enclosers[a], key=lambda e: (len(enclosers[e]), -e))
            children[parent].append(a)
    for child_list in children:
        child_list.sort()

    nodes = []
    for cid in range(n):
        comp = components[cid]
        color = color_chars[max(0, min(15, comp["value"]))]
        boundary = _corner_points(_trace_outer_contour(comp["cells"], comp["start"]))
        rows_ = [r for r, _ in comp["cells"]]
        cols_ = [c for _, c in comp["cells"]]
        r0, r1, c0, c1 = min(rows_), max(rows_), min(cols_), max(cols_)
        nodes.append(
            {
                "id": cid,
                "color": color,
                "hash": _object_hash(comp["cells"], color),
                "pixels": len(comp["cells"]),
                "boundary": [[r, c] for r, c in boundary],
                "bbox": [r0, c0, r1, c1],
                "centre": [(r0 + r1) // 2, (c0 + c1) // 2],
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
                "children": children[cid],
            }
        )

    return {"nodes": nodes, "adjacency_list": adjacency_list, "objects": composite_objects(nodes, adjacency_list, height * width)}


def composite_objects(nodes, adjacency_list, board_cells, *, speck_cells=4, background_fraction=0.3):
    """Group raw components into the objects a player would name.

    Two merges, applied with union-find over the nodes (background components, > background_fraction of the
    board, are never merged): (1) a component and every component it encloses (a bordered box and its
    interior are one piece); (2) small components (<= speck_cells cells) that touch each other (checkerboards,
    stripes, dotted trails are one textured region, not forty specks). Solid single-colour objects stay as
    they are. Each object: id, colors (sorted letters), pixels, bbox, centre, height, width, parts (node ids),
    kind ('solid' | 'composite' | 'textured').
    """
    n = len(nodes)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    background = {node["id"] for node in nodes if node["pixels"] > background_fraction * board_cells}
    by_id = {node["id"]: node for node in nodes}
    for node in nodes:
        if node["id"] in background or not node.get("children"):
            continue
        # A thin border around an interior is one piece; a large region enclosing small things is a container
        # (an arena holding targets, a legend panel holding icons) and must not swallow them.
        inner = sum(by_id[c]["pixels"] for c in node["children"] if c in by_id and c not in background)
        if inner == 0 or node["pixels"] > 3 * inner + 4:
            continue
        for child in node["children"]:
            if child not in background:
                union(node["id"], child)
    small = {node["id"] for node in nodes if node["pixels"] <= speck_cells and node["id"] not in background}
    for a, b in adjacency_list:
        if a in small and b in small:
            union(a, b)
    groups: dict[int, list[dict]] = {}
    for node in nodes:
        groups.setdefault(find(node["id"]), []).append(node)
    objects = []
    for root in sorted(groups):
        members = groups[root]
        r0 = min(m["bbox"][0] for m in members); c0 = min(m["bbox"][1] for m in members)
        r1 = max(m["bbox"][2] for m in members); c1 = max(m["bbox"][3] for m in members)
        colors = sorted({m["color"] for m in members})
        if len(members) == 1:
            kind = "solid"
        elif all(m["id"] in small for m in members):
            kind = "textured"
        else:
            kind = "composite"
        objects.append({
            "id": len(objects),
            "colors": colors,
            "pixels": sum(m["pixels"] for m in members),
            "bbox": [r0, c0, r1, c1],
            "centre": [(r0 + r1) // 2, (c0 + c1) // 2],
            "height": r1 - r0 + 1,
            "width": c1 - c0 + 1,
            "parts": [m["id"] for m in members],
            "kind": kind,
            "background": any(m["id"] in background for m in members),
        })
    return objects
