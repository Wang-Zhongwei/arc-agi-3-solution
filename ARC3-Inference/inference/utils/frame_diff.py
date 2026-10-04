"""Object-level frame diffs shared by host and isolated Python sandbox.

Keep this module self-contained: its source is injected into the sandbox.
"""


def _hungarian(cost):
    # Minimum-cost assignment (Jonker-Volgonant style with potentials).
    # cost: n x m matrix with n <= m. Returns list row -> column.
    n = len(cost)
    m = len(cost[0]) if n else 0
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)      # p[j] = row assigned to column j (1-based; 0 = none)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    return assignment


def _hungarian_candidates(candidates):
    if not candidates:
        return []
    rows = sorted({i for _, i, _, _ in candidates})
    cols = sorted({j for _, _, j, _ in candidates})
    edges = {(i, j): (rank, change) for rank, i, j, change in candidates}
    # Maximize valid cardinality, then minimize total center distance. Dummies
    # represent unmatched rows; forbidden edges always cost more than dummies.
    penalty = (len(rows) + 1) * (max(rank for rank, *_ in candidates) + 1)
    cost = [[edges.get((i, j), (2*penalty, None))[0] for j in cols]
            + [penalty]*len(rows) for i in rows]
    assignment = _hungarian(cost)
    return [(rows[r], cols[c], edges[rows[r], cols[c]][1])
            for r, c in enumerate(assignment) if c < len(cols) and (rows[r], cols[c]) in edges]


def _node_mask(grid, node, origin):
    # Recover the component from an occupied boundary cell, not its bbox corner
    # (which may be empty). Coordinates in the resulting mask are bbox-local.
    points = node.get("boundary") or []
    if not points:
        return frozenset()
    r, c = points[0][0] - origin[0], points[0][1] - origin[1]
    value = grid[r][c]
    cells = {(r, c)}
    stack = [(r, c)]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < len(grid) and 0 <= nc < len(grid[nr])
                    and (nr, nc) not in cells and grid[nr][nc] == value):
                cells.add((nr, nc))
                stack.append((nr, nc))
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _rotate_mask(mask):
    height = max(r for r, _ in mask) + 1
    return frozenset((c, height - 1 - r) for r, c in mask)


def _reflections(mask):
    height = max(r for r, _ in mask) + 1
    width = max(c for _, c in mask) + 1
    # Axis order resolves shapes with multiple valid reflections.
    yield "-", frozenset((height - 1 - r, c) for r, c in mask)
    yield "|", frozenset((r, width - 1 - c) for r, c in mask)
    yield "/", frozenset((width - 1 - c, height - 1 - r) for r, c in mask)
    yield "\\", frozenset((c, r) for r, c in mask)


def _reduced_mask(mask, scale):
    reduced = frozenset((r // scale, c // scale) for r, c in mask)
    # Every occupied block must be full for this to be exact uniform scaling.
    return reduced if len(mask) == len(reduced) * scale * scale else None


def _center2(node):
    r0, c0, r1, c1 = node["bbox"]
    return r0 + r1, c0 + c1


def _single_change(a, b, ma, mb):
    same_color = a["color"] == b["color"]
    if same_color and a["pose_hash"] == b["pose_hash"] and ma == mb:
        if a["bbox"] != b["bbox"]:
            return "moved", {"hash": a["hash"]}
        return None
    ca, cb = _center2(a), _center2(b)
    near = all(abs(x - y) <= 2 for x, y in zip(ca, cb))
    if same_color and near and a["hash"] == b["hash"]:
        rotated = ma
        for angle in (90, 180, 270):
            rotated = _rotate_mask(rotated)
            if rotated == mb:
                return "rotated", {"hash": a["hash"], "rotated_by": angle}
        for axis, reflected in _reflections(ma):
            if reflected == mb:
                return "mirrored", {"hash": a["hash"], "axis": axis}
    if same_color and near and a["form_hash"] == b["form_hash"] and a["scale"] != b["scale"]:
        reduced = _reduced_mask(ma, a["scale"])
        if reduced is not None and reduced == _reduced_mask(mb, b["scale"]):
            return "scaled", {"form_hash": a["form_hash"],
                              "before_scale": a["scale"], "after_scale": b["scale"],
                              "scale_factor": b["scale"] / a["scale"]}
    if (not same_color and a["shape_hash"] == b["shape_hash"]
            and a["bbox"] == b["bbox"] and ma == mb):
        return "changed_color", {"shape_hash": a["shape_hash"],
                                 "before_color": a["color"], "after_color": b["color"]}
    return None


def _unique_nearest(candidates):
    """Reject shared equal-rank candidates in the resize fallback."""
    best_a, best_b = {}, {}
    for rank, i, j, change in candidates:
        for best, key in ((best_a, i), (best_b, j)):
            if key not in best or rank < best[key][0]:
                best[key] = (rank, 1)
            elif rank == best[key][0]:
                best[key] = (rank, best[key][1] + 1)
    return [(i, j, change) for rank, i, j, change in candidates
            if best_a[i] == (rank, 1) and best_b[j] == (rank, 1)]


def _bbox_area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def compute_frame_diff(grid_a, grid_b, nodes_a, nodes_b, max_group_match=None, *, origin=(0, 0)):
    """Compare same-bounds grids and nodes; coordinates retain the crop origin.

    Hungarian assignment maximizes valid pair count, then minimizes distance.
    Each pair gets one change. Exact candidates must verify against masks;
    unmatched same-color geometry can use the documented resize heuristic.
    max_group_match bounds matching within large identical-form groups.
    """
    changed = sum(va != vb for ra, rb in zip(grid_a, grid_b) for va, vb in zip(ra, rb))
    result = {"changed_cell_count": changed, **{key: [] for key in (
        "moved", "rotated", "mirrored", "scaled", "changed_color", "resized", "appeared", "disappeared")}}
    if not changed:
        return result
    masks_a = [_node_mask(grid_a, n, origin) for n in nodes_a]
    masks_b = [_node_mask(grid_b, n, origin) for n in nodes_b]
    used_a, used_b = set(), set()
    # Remove unchanged nodes before matching duplicate shapes at other positions.
    unchanged = {}
    for j, b in enumerate(nodes_b):
        unchanged.setdefault((tuple(b["bbox"]), b["pose_hash"], b["color"]), []).append(j)
    for i, a in enumerate(nodes_a):
        key = (tuple(a["bbox"]), a["pose_hash"], a["color"])
        for j in unchanged.get(key, []):
            if j not in used_b and masks_a[i] == masks_b[j]:
                used_a.add(i)
                used_b.add(j)
                break

    def record(i, j, change):
        bucket, fields = change
        a, b = nodes_a[i], nodes_b[j]
        result[bucket].append({"before_id": a["id"], "after_id": b["id"],
                               "before_bbox": tuple(a["bbox"]), "after_bbox": tuple(b["bbox"]),
                               **fields})
        used_a.add(i)
        used_b.add(j)

    groups_a, groups_b = {}, {}
    for i, a in enumerate(nodes_a):
        if i not in used_a:
            groups_a.setdefault(a["form_hash"], []).append(i)
    for j, b in enumerate(nodes_b):
        if j not in used_b:
            groups_b.setdefault(b["form_hash"], []).append(j)
    blocked_a, blocked_b = set(), set()
    for form, group_a in groups_a.items():
        group_b = groups_b.get(form, [])
        if max_group_match is not None and max(len(group_a), len(group_b)) > max_group_match:
            blocked_a.update(group_a)
            blocked_b.update(group_b)
            continue
        candidates = []
        for i in group_a:
            for j in group_b:
                change = _single_change(nodes_a[i], nodes_b[j], masks_a[i], masks_b[j])
                if change is not None:
                    distance = sum(abs(x - y) for x, y in zip(_center2(nodes_a[i]), _center2(nodes_b[j])))
                    candidates.append((distance, i, j, change))
        accepted = _hungarian_candidates(candidates)
        matched_a = {i for i, _, _ in accepted}
        matched_b = {j for _, j, _ in accepted}
        blocked_a.update(i for _, i, _, _ in candidates if i not in matched_a)
        blocked_b.update(j for _, _, j, _ in candidates if j not in matched_b)
        for i, j, change in accepted:
            record(i, j, change)

    # Geometry-change fallback, excluding uniform-form transforms/combinations.
    candidates = []
    for i, a in enumerate(nodes_a):
        if i in used_a or i in blocked_a:
            continue
        for j, b in enumerate(nodes_b):
            if j in used_b or j in blocked_b or a["color"] != b["color"] or a["form_hash"] == b["form_hash"]:
                continue
            ba, bb = a["bbox"], b["bbox"]
            height = max(0, min(ba[2], bb[2]) - max(ba[0], bb[0]) + 1)
            width = max(0, min(ba[3], bb[3]) - max(ba[1], bb[1]) + 1)
            area_a, area_b = _bbox_area(ba), _bbox_area(bb)
            overlap = height * width / min(area_a, area_b)
            if overlap >= 0.5:
                similarity = min(area_a, area_b) / max(area_a, area_b)
                candidates.append(((-overlap, -similarity), i, j))
    # Greedy by overlap, then area similarity; tied candidates sharing a node
    # remain ambiguous instead of depending on list order.
    candidates.sort()
    index = 0
    while index < len(candidates):
        end = index + 1
        while end < len(candidates) and candidates[end][0] == candidates[index][0]:
            end += 1
        tied = [(rank, i, j, None) for rank, i, j in candidates[index:end]
                if i not in used_a and j not in used_b and i not in blocked_a and j not in blocked_b]
        accepted = _unique_nearest(tied)
        matched_a = {i for i, _, _ in accepted}
        matched_b = {j for _, j, _ in accepted}
        blocked_a.update(i for _, i, _, _ in tied if i not in matched_a)
        blocked_b.update(j for _, _, j, _ in tied if j not in matched_b)
        for i, j, _ in accepted:
            a, b = nodes_a[i], nodes_b[j]
            record(i, j, ("resized", {"color": a["color"],
                                    "before_pixels": a["pixels"], "after_pixels": b["pixels"]}))
        index = end
    for bucket, nodes, used in (("appeared", nodes_b, used_b), ("disappeared", nodes_a, used_a)):
        result[bucket] = [{"id": n["id"], "color": n["color"], "hash": n["hash"],
                           "bbox": tuple(n["bbox"]), "pixels": n["pixels"]}
                          for i, n in enumerate(nodes) if i not in used]
    return result
