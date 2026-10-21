"""Candidate 2: lexicon-style training-free heuristic (reimplemented from Kaggle discussion 746295's description).

(a) cell size = largest k in {8,6,5,4,3,2} whose lattice explains >=95% of colour edges (outside 2-px border bands)
(b) background = modal colour + uniform edge-band colours; background colour regions are kept as plain 4-CC segments
(c) objects = 4-connected groups of non-background cells of ANY colour (groups multi-colour sprites)
(d) cuts: library designs (exact pixel patterns of isolated blobs seen earlier in this game, D4 variants)
    are template-matched inside larger blobs and cut out; optional periodic-tile cutting (PERIOD=1)
(e) names: D4-canonical hash of the pattern at native cell size, numbered per game in first-seen order
Deterministic, numpy+scipy only.
"""
import hashlib, os, time
import numpy as np
from scipy import ndimage

PERIOD = int(os.environ.get("LEX_PERIOD", "0"))
CONN8 = int(os.environ.get("LEX_CONN8", "0"))
BIG = int(os.environ.get("LEX_BIG", "100000"))
MODE = os.environ.get("LEX_MODE", "group")  # group: multi-colour blobs then cut; cc: colour CCs then merge library matches  # blobs larger than this fall back to colour CCs
TIMED = None
CROSS = ndimage.generate_binary_structure(2, 1)
SQUARE = ndimage.generate_binary_structure(2, 2)


def cell_size(fr):
    core = fr[2:-2, 2:-2]
    vx = np.nonzero(core[:, 1:] != core[:, :-1]); vy = np.nonzero(core[1:, :] != core[:-1, :])
    ex = vx[1] + 1 + 2; ey = vy[0] + 1 + 2  # absolute boundary positions (edge lies before this index)
    if len(ex) + len(ey) < 12:
        return 1, (0, 0)
    for k in (8, 6, 5, 4, 3, 2):
        bx = np.bincount(ex % k, minlength=k); by = np.bincount(ey % k, minlength=k)
        if (bx.max() + by.max()) >= 0.95 * (len(ex) + len(ey)):
            return k, (int(bx.argmax()), int(by.argmax()))
    return 1, (0, 0)


def background_colours(fr):
    vals, cnt = np.unique(fr, return_counts=True)
    bg = {int(vals[cnt.argmax()])}
    for band in (fr[0], fr[-1], fr[:, 0], fr[:, -1]):
        if (band == band[0]).all():
            bg.add(int(band[0]))
    return bg


def canon_name(patch, k):
    """D4-canonical hash of a (-1 = transparent) colour patch downsampled by cell size k."""
    if k > 1 and patch.shape[0] % k == 0 and patch.shape[1] % k == 0:
        patch = patch[::k, ::k]
    vs = []
    for m in (patch, patch[:, ::-1]):
        for r in range(4):
            q = np.rot90(m, r)
            vs.append(q.tobytes() + bytes(q.shape))
    return hashlib.blake2b(min(vs), digest_size=6).hexdigest()


def patch_of(fr, mask):
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    p = np.where(mask[y0:y1, x0:x1], fr[y0:y1, x0:x1], -1)
    return p, (y0, x0)


def match_design(fr, blob, d):
    """Positions (y,x) where design d (h,w; -1 transparent) matches fr exactly with all opaque px inside blob."""
    h, w = d.shape
    ys, xs = np.nonzero(blob)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if y1 - y0 < h or x1 - x0 < w:
        return []
    H, W = y1 - y0 - h + 1, x1 - x0 - w + 1
    M = np.ones((H, W), bool)
    sub = np.where(blob, fr, -9)[y0:y1, x0:x1]
    for dy, dx in zip(*np.nonzero(d >= 0)):
        M &= sub[dy:dy + H, dx:dx + W] == d[dy, dx]
        if not M.any():
            return []
    return [(y0 + a, x0 + b) for a, b in zip(*np.nonzero(M))]


def period_cuts(fr, blob):
    """Split a multi-colour blob with a clear 2-D period into period cells. Returns list of masks or None."""
    p, (y0, x0) = patch_of(fr, blob)
    cuts = []
    for axis in (1, 0):
        q = p if axis == 1 else p.T
        best = None
        for per in range(2, 17):
            if q.shape[1] < 2 * per:
                break
            a, b = q[:, :-per], q[:, per:]
            both = (a >= 0) & (b >= 0)
            if both.sum() < 3 * per * 2:
                continue
            agree = (a[both] == b[both]).mean()
            if agree >= 0.9:
                # non-trivial: some smaller shift disagrees
                low = []
                for s in range(1, per):
                    a2, b2 = q[:, :-s], q[:, s:]
                    bb = (a2 >= 0) & (b2 >= 0)
                    low.append((a2[bb] == b2[bb]).mean() if bb.any() else 1)
                if min(low) < 0.8:
                    best = per
                    break
        if best is None:
            cuts.append(None)
            continue
        e = np.nonzero((q[:, 1:] != q[:, :-1]) & (q[:, 1:] >= 0) & (q[:, :-1] >= 0))[1] + 1
        off = int(np.bincount(e % best, minlength=best).argmax()) if len(e) else 0
        cuts.append((best, off))
    if cuts[0] is None and cuts[1] is None:
        return None
    cx = cuts[0]; cy = cuts[1]
    ys, xs = np.nonzero(blob)
    cell_x = ((xs - x0 - (cx[1] if cx else 0)) // cx[0]) if cx else np.zeros_like(xs)
    cell_y = ((ys - y0 - (cy[1] if cy else 0)) // cy[0]) if cy else np.zeros_like(ys)
    out = {}
    for y, x, a, b in zip(ys, xs, cell_y, cell_x):
        out.setdefault((a, b), []).append((y, x))
    masks = []
    for pts in out.values():
        m = np.zeros(fr.shape, bool); yy, xx = zip(*pts); m[list(yy), list(xx)] = True
        masks.append(m)
    return masks


def segment(frame, prev=None, state=None):
    if MODE == "cc":
        return segment_cc(frame, prev, state)
    return segment_group(frame, prev, state)


def segment_cc(frame, prev=None, state=None):
    """Colour CCs, then multi-colour library designs (seen isolated earlier in this game) are merged where they match."""
    global TIMED
    t0 = time.perf_counter()
    fr = np.asarray(frame, dtype=np.int64)
    if state is None:
        state = {"lib": {}, "names": {}}
    k, off = cell_size(fr)
    state["cell"] = k
    bg = background_colours(fr)
    isbg = np.isin(fr, list(bg))
    lib = state["lib"]
    claimed = np.zeros(fr.shape, bool)
    lab = np.full(fr.shape, -1, np.int64); n = 0
    full = np.ones(fr.shape, bool)
    for key, d in sorted(lib.items(), key=lambda kv: (-int((kv[1] >= 0).sum()), kv[0])):
        for (y, x) in match_design(fr, ~claimed, d):
            m = np.zeros(fr.shape, bool); m[y:y + d.shape[0], x:x + d.shape[1]] = d >= 0
            if not (m & claimed).any():
                lab[m] = n; n += 1; claimed |= m
    for c in np.unique(fr):
        l, m = ndimage.label((fr == c) & ~claimed, structure=CROSS)
        lab[l > 0] = l[l > 0] - 1 + n; n += m
    blobs, nb = ndimage.label(~isbg, structure=CROSS)
    sizes = np.bincount(blobs.ravel())
    for b in range(1, nb + 1):
        if 4 <= sizes[b] <= 256:
            mask = blobs == b
            if len(np.unique(fr[mask])) < 2:
                continue
            p, _ = patch_of(fr, mask)
            for mm in (p, p[:, ::-1]):
                for r in range(4):
                    q = np.ascontiguousarray(np.rot90(mm, r))
                    lib.setdefault(q.tobytes() + bytes(q.shape), q)
    names = {}
    for s in np.unique(lab):
        if s < 0:
            continue
        p, _ = patch_of(fr, lab == s)
        names[int(s)] = state["names"].setdefault(canon_name(p, k), f"o{len(state['names'])}")
    TIMED = time.perf_counter() - t0
    return lab, names, {}, state


def segment_group(frame, prev=None, state=None):
    global TIMED
    t0 = time.perf_counter()
    fr = np.asarray(frame, dtype=np.int64)
    if state is None:
        state = {"lib": {}, "names": {}}
    k, off = cell_size(fr)
    state["cell"] = k
    bg = background_colours(fr)
    isbg = np.isin(fr, list(bg))
    lab = np.full(fr.shape, -1, np.int64); n = 0
    # background colour regions: plain colour components
    for c in bg:
        l, m = ndimage.label(fr == c, structure=CROSS)
        lab[l > 0] = l[l > 0] - 1 + n; n += m
    blobs, nb = ndimage.label(~isbg, structure=SQUARE if CONN8 else CROSS)
    lib = state["lib"]
    sizes = np.bincount(blobs.ravel())
    for b in range(1, nb + 1):
        mask = blobs == b
        pieces = [mask]
        if sizes[b] > BIG:
            pieces = []
            for c in np.unique(fr[mask]):
                l, m = ndimage.label(mask & (fr == c), structure=CROSS)
                pieces += [l == i for i in range(1, m + 1)]
        else:
            # library cuts (largest designs first)
            rest = mask.copy(); cut = []
            if len(lib) and sizes[b] > 3:
                for key, d in sorted(lib.items(), key=lambda kv: -int((kv[1] >= 0).sum())):
                    npx = int((d >= 0).sum())
                    if npx >= rest.sum() or npx < 4:
                        continue
                    for (y, x) in match_design(fr, rest, d):
                        m = np.zeros(fr.shape, bool); m[y:y + d.shape[0], x:x + d.shape[1]] = d >= 0
                        if (m & rest).sum() == npx:
                            cut.append(m); rest &= ~m
            if cut:
                l, m = ndimage.label(rest, structure=CROSS)
                pieces = cut + [l == i for i in range(1, m + 1)]
            if PERIOD:
                new = []
                for pc in pieces:
                    if len(np.unique(fr[pc])) > 1 and pc.sum() >= 16:
                        pm = period_cuts(fr, pc)
                        new += pm if pm else [pc]
                    else:
                        new.append(pc)
                pieces = new
            # isolated blob -> library
            if not cut and 3 <= sizes[b] <= 256:
                p, _ = patch_of(fr, mask)
                for mm in (p, p[:, ::-1]):
                    for r in range(4):
                        q = np.ascontiguousarray(np.rot90(mm, r))
                        lib.setdefault(q.tobytes() + bytes(q.shape), q)
        for pc in pieces:
            if pc.any():
                lab[pc] = n; n += 1
    names = {}
    for s in range(n):
        m = lab == s
        if not m.any():
            continue
        p, _ = patch_of(fr, m)
        h = canon_name(p, k)
        names[s] = state["names"].setdefault(h, f"o{len(state['names'])}")
    TIMED = time.perf_counter() - t0
    return lab, names, {}, state
