"""Design B behavioural entity tracker: an action-conditioned diff model on plain colour components.

Host-side, pure numpy/scipy, online and causal. One `update(frame, action, click, new_level)` per executed
action; returns result-dict fields for that action's tool result, and `tail_line()` gives at most one short
line. Frozen after dev tuning (split.json); parameters in P.

Per update:
  1. Plain colour components (4-connected, every non-background colour), as the harness segments.
  2. View shift: when a large part of the frame changed, the best axis-aligned global translation is removed
     before matching. A scrolled board is then unchanged in world coordinates; components that kept their
     screen position (HUD, frame) get an unknown displacement, so a scroll never credits or groups anything.
  3. Matching prev -> new components, in passes: unchanged; recolour in place; exact translation (same colour
     and shape, Hungarian on distance, long jumps only for unique shapes); turned shapes (same colour, D4-equal);
     repainted-and-moved (same shape, other colour); blobs (leftovers grouped by adjacency, equal colour
     histogram); deformed (same colour, overlapping, similar size); absorb/split in place (a recolour that merges
     with or splits from neighbours). A rigid-set pass gives a turned multi-part sprite one displacement.
     Unmatched old components are hidden (they may re-appear); unmatched new ones start tracks.
  4. Moved blobs -> entities: components within 2 px whose whole motion histories agree since they co-existed
     are grouped (a box the player moved without is never merged); a divergent step breaks the group.
  5. Action-effect statistics per track: arrow presses present / moved / moved in the pressed direction, step
     sizes and directions, moves on non-arrow actions (self-moving), shape/colour changes (bars, counters).
  6. Selection: the controlled entity is the one that moved with the last arrow press; committed after 2
     agreeing presses (1 if it was the only translating entity and moved exactly along the arrow; tiny
     entities need one more). A committed piece is replaced only after a switch action (SPACE/click) with the
     old piece sitting still, or when it moved against the press, or after 4 rival presses. Switch actions are
     learned per game; after a learned one, the selection follows the colour the selected piece lost (the
     selection marker) to the one sibling-sized entity that gained it. Appearance is carried to the next level
     and through RESET; a colour-patch template and a 2-step dead-reckoned hold bridge merges and occlusion; a
     sprite that transforms in place keeps the claim.
  7. Facts: blocker (committed mover with >= 3 agreeing moves did not move on a press in a direction it has
     moved before, and a compact, non-HUD entity lies within one step); vanish-on-contact (components next to or
     under the mover disappeared and their colour is gone from the spot once uncovered, not a recolour/merge, not
     part of the mover); selection switch; view scroll.

Names E1, E2, ... are given on first mention (the published name covering most pixels) and never reused
within a level.
"""
from __future__ import annotations

import collections
import time

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

CROSS = ndimage.generate_binary_structure(2, 1)
ARROWS = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}
DIRNAME = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT"}
ACTNAME = {"ACTION5": "ACTION5", "ACTION6": "click", "ACTION7": "ACTION7"}
EXACT = ("same", "move", "recol", "jump", "blob")

# tunables (frozen after dev tuning; see eval_tracker.py)
P = dict(
    RMAX=16,            # max displacement (px, Chebyshev) for matching repeated shapes
    VS_MIN_CHANGED=200,  # view-shift test only when >= this many pixels changed ...
    VS_FRAC=0.3,        # ... and >= this fraction of non-background pixels changed
    VS_MAXD=16,         # axis-aligned shifts tested
    LINK_GAP=2,         # components within this Chebyshev distance can be grouped by co-motion
    COMMIT_AGREE=2,     # agreeing arrow moves to commit an entity as controllable
    AGREE_RATIO=0.6,    # agreeing / moved
    BLOCK_MIN=1,        # observations before a blocker fact is committed
    CARRY=1,            # carry the controlled appearance over to the next level (1 = committed claim)
    TMPL=1,             # template fallback for the controlled sprite when its components merge / hide
    TMPL_MIN=0.85,      # fraction of template pixels that must match
    RIVAL=4,            # consecutive agreeing presses by a rival while the selection sat still -> switch
    TMPL_NMIN=8,        # smallest template (pixels) the fallback may use
    LINK_AGREE=0.8,     # co-motion agreement over co-observed steps needed to group two components
    LINK_STRIKES=1,     # consecutive divergent steps that break a group link
    LINK_DIFF=0,        # at most this many divergent co-observed steps to group two components
    SOLE1=1,            # commit after one press if it is the only entity that changed and it moved exactly along the arrow
    INHERIT=1,          # selection passes to a sprite that appears where the selected one vanished (transform in place)
    BLOCK_AGREE=3,      # agreeing moves the controlled entity needs before blocker facts are stated
    BLOCK_MAXPX=1024,   # blockers larger than this (1/4 of the frame) are treated as background
    VAN_MINPX=4,        # smallest vanished entity
    BLOCK_DIRSEEN=1,    # blocker facts only for directions the controlled entity has moved in before
    BLOCK_HUDCHG=6,     # an entity that changed shape/colour on >= this many steps is never called a blocker
    BLOCK_WALK=0,       # never call a blocker an entity whose colours the mover has walked onto
    HOLD=2,             # steps the last committed claim is kept (dead-reckoned) while the sprite is hidden
    TINY=3,             # entities this small (px) need one more agreeing move and never commit on one press
)


class Comp:
    __slots__ = ("c", "y0", "x0", "h", "w", "m", "n", "sk", "cy", "cx", "idx", "tid")

    def __init__(self, c, y0, x0, m, idx):
        self.c, self.y0, self.x0, self.m, self.idx = int(c), int(y0), int(x0), m, idx
        self.h, self.w = m.shape
        self.n = int(m.sum())
        self.sk = (self.c, self.h, self.w, m.tobytes())
        ys, xs = np.nonzero(m)
        self.cy, self.cx = self.y0 + ys.mean(), self.x0 + xs.mean()
        self.tid = -1

    def full(self, shape=(64, 64)):
        out = np.zeros(shape, bool)
        out[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w] = self.m
        return out


def extract(fr, bg):
    comps = []
    lab_all = np.full(fr.shape, -1, np.int32)
    for c in np.unique(fr):
        if c == bg:
            continue
        lab, _ = ndimage.label(fr == c, structure=CROSS)
        for i, sl in enumerate(ndimage.find_objects(lab)):
            m = lab[sl] == (i + 1)
            k = len(comps)
            comps.append(Comp(c, sl[0].start, sl[1].start, m, k))
            lab_all[sl][m] = k
    return comps, lab_all


def view_shift(a, b, bg, nchanged, p=P):
    """Best axis-aligned global translation (dy, dx) with new[y+dy, x+dx] == old[y, x]; (0, 0) if none."""
    nonbg = int((a != bg).sum())
    if nchanged < max(p["VS_MIN_CHANGED"], p["VS_FRAC"] * nonbg):
        return (0, 0)
    H, W = a.shape
    best, bd = nchanged, (0, 0)
    for d in range(1, p["VS_MAXD"] + 1):
        for dy, dx in ((d, 0), (-d, 0), (0, d), (0, -d)):
            ya, yb = max(0, -dy), min(H, H - dy)
            xa, xb = max(0, -dx), min(W, W - dx)
            ov = (yb - ya) * (xb - xa)
            mism = int((a[ya:yb, xa:xb] != b[ya + dy:yb + dy, xa + dx:xb + dx]).sum()) * (H * W) / ov
            if mism < best:
                best, bd = mism, (dy, dx)
    return bd if best < 0.5 * nchanged else (0, 0)


def d4key(m):
    vs = []
    for q in (m, m[:, ::-1]):
        for r in range(4):
            z = np.rot90(q, r)
            vs.append(bytes(z.shape) + z.tobytes())
    return min(vs)


class Track:
    __slots__ = ("tid", "comp", "prev", "hidden", "hid_at", "born", "moves", "unk", "present", "moved", "agree",
                 "steps", "dirs", "last_move", "auto", "changes")

    def __init__(self, tid, comp, born):
        self.tid, self.comp, self.hidden, self.hid_at, self.born = tid, comp, False, -1, born
        self.prev = None
        self.auto = 0       # moves on non-arrow actions (autonomous or click-driven)
        self.changes = 0    # steps on which its shape or colour changed (bars, counters, animations)
        self.moves = {}     # step -> (dy, dx) world displacement, exact matches only, nonzero only
        self.unk = set()    # steps where the displacement is unknown (deformed / hidden)
        self.present = self.moved = self.agree = 0   # arrow-press statistics
        self.steps = collections.Counter()           # |step| along the arrow axis on agreeing moves
        self.dirs = set()                            # arrow actions this track moved along
        self.last_move = -1


class EntityTracker:
    def __init__(self, params=None):
        self.p = dict(P)
        if params:
            self.p.update(params)
        self.carry_sig = None   # appearance signature of the last committed controlled entity (cross-level)
        self.carry_sigs = []    # all committed controlled appearances this game, most recent first
        self.ctrl_sigs = set()  # appearance signatures of committed controllable entities (whole game)
        self.switch_acts = collections.Counter()   # non-arrow action -> times followed by a change of mover (game)
        self.reset_level(None)

    # ------------------------------------------------------------------ level state
    def reset_level(self, frame):
        self.t = 0
        self.tracks = {}
        self.next_tid = 0
        self.comps = []
        self.lab = None
        self.frame = None
        self.prev_frame = None
        self.bg = None
        self.pub = {}            # tid -> public name number
        self.next_pub = 1
        self.links = {}          # (tidA, tidB) -> True (grouped)
        self.strikes = collections.Counter()
        self.ever_linked = set()
        self.sel = set()         # member tids of the selected entity
        self.sel_conf = None     # "committed" | "tentative" | None
        self.sel_src = None
        self.committed_sel = set()   # member tids of the last committed selection
        self.tmpl = None             # colour patch of the committed selection from its last clean frame
        self.tmpl_at = None          # (y0, x0, patch) placement while in template mode
        self.turns = False           # the controlled sprite has been seen to turn (D4 variants needed)
        self.rival = collections.Counter()
        self.walkable = set()        # colours the controlled entity has moved onto
        self._last_ctrl_name = None
        self._hold = None            # (mask, step, member tids) of the last committed claim
        self._tail = None
        self._scroll_said = False
        self.last_mover = set()      # member tids of the entity that moved on the last moving arrow press
        self.pending_switch = None                  # (action, step) of a non-arrow action since last arrow move
        self.pending_pred = None
        self.blocks = collections.Counter()        # (blocker tid, mover tid, arrow) -> count
        self.block_steps = collections.defaultdict(list)
        self.block_pub = set()
        self.vanish_pend = []    # (tids, mask, mover tid, step)
        self.facts = []          # committed fact records (for the ledger / eval)
        self.events = []         # new facts this update
        self.view = (0, 0)
        if frame is not None:
            self._init_frame(frame)

    def _init_frame(self, frame):
        vals, cnt = np.unique(frame, return_counts=True)
        self.bg = int(vals[cnt.argmax()])
        self.comps, self.lab = extract(frame, self.bg)
        for c in self.comps:
            self._new_track(c)
        self.frame = frame
        # carry-over (next level, or restart after RESET): the last controlled appearance
        hits = []
        if self.carry_sig is not None and self.p["CARRY"]:
            for sig in [self.carry_sig] + [x for x in self.carry_sigs if x != self.carry_sig]:
                hits = self._find_sig(sig)
                if hits:
                    break
            if len(hits) == 1:
                mem = sorted(hits[0])
                for a, b in zip(mem, mem[1:]):
                    self.links[(min(a, b), max(a, b))] = True
                self.sel, self.sel_conf, self.sel_src = set(mem), "committed", "carry"
                self.committed_sel = set(mem)

    def _find_sig(self, sig):
        """Groups of adjacent current components whose (shape up to D4, colour) multiset equals sig."""
        need = collections.Counter(sig)
        item = {c.idx: d4key(c.m) + bytes([c.c]) for c in self.comps if (d4key(c.m) + bytes([c.c])) in need}
        if not item:
            return []
        first = sig[0]
        byidx = {c.idx: c for c in self.comps}
        hits = []
        for i, it in item.items():
            if it != first:
                continue
            got, left, frontier = [i], need.copy(), [i]
            left[it] -= 1
            while frontier and sum(left.values()) > 0:
                cur = byidx[frontier.pop()]
                for j, jt in item.items():
                    if j in got or left[jt] <= 0:
                        continue
                    o = byidx[j]
                    if (o.y0 - (cur.y0 + cur.h) <= 2 and cur.y0 - (o.y0 + o.h) <= 2 and
                            o.x0 - (cur.x0 + cur.w) <= 2 and cur.x0 - (o.x0 + o.w) <= 2):
                        got.append(j); left[jt] -= 1; frontier.append(j)
            if sum(left.values()) == 0:
                hits.append({byidx[j].tid for j in got})
        return hits

    def _new_track(self, comp):
        tr = Track(self.next_tid, comp, self.t)
        self.next_tid += 1
        comp.tid = tr.tid
        self.tracks[tr.tid] = tr
        return tr

    # ------------------------------------------------------------------ entities
    def entities(self):
        """root tid -> list of alive member tids (union-find over co-motion links)."""
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x
        alive = [t for t, tr in self.tracks.items() if not tr.hidden]
        for (a, b) in self.links:
            if self.tracks[a].hidden or self.tracks[b].hidden:
                continue
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        out = collections.defaultdict(list)
        for t in alive:
            out[find(t)].append(t)
        return out

    def ent_of(self, tid, ents=None):
        ents = ents if ents is not None else self.entities()
        for r, mem in ents.items():
            if tid in mem:
                return r, mem
        return None, []

    def mask_of(self, members):
        m = np.zeros((64, 64), bool)
        for t in members:
            c = self.tracks[t].comp
            if c is not None and not self.tracks[t].hidden:
                m[c.y0:c.y0 + c.h, c.x0:c.x0 + c.w] |= c.m
        return m

    def _sig(self, members):
        return tuple(sorted(d4key(self.tracks[t].comp.m) + bytes([self.tracks[t].comp.c]) for t in members
                            if not self.tracks[t].hidden))

    def name(self, members):
        """Public name of an entity: the earliest published name among its members, else publish one."""
        weight = collections.Counter()
        for t in members:
            if t in self.pub and t in self.tracks:
                weight[self.pub[t]] += self.tracks[t].comp.n
        if weight:
            n = max(weight, key=lambda k: (weight[k], -k))
        else:
            n = self.next_pub
            self.next_pub += 1
        for t in members:
            self.pub.setdefault(t, n)
        return f"E{n}"

    # ------------------------------------------------------------------ matching
    def _match(self, P_, N_, vs, reset):
        sy, sx = vs
        RMAX = self.p["RMAX"]
        res = {}            # prev idx -> (new idx, kind, (dy, dx))
        usedN = set()

        def take(p, j, kind, d):
            res[p.idx] = (j, kind, d)
            usedN.add(j)
        nkey = collections.defaultdict(list)
        for n in N_:
            nkey[(n.y0, n.x0, n.sk)].append(n.idx)
        # 1 unchanged in world
        for p in P_:
            for j in nkey.get((p.y0 + sy, p.x0 + sx, p.sk), ()):
                if j not in usedN:
                    take(p, j, "same", (0, 0)); break
        # 1b screen-fixed under a view shift (HUD, frame, or a camera-followed sprite): identity kept,
        #    displacement unknown (no motion credit and no co-motion links)
        if vs != (0, 0):
            for p in P_:
                if p.idx in res:
                    continue
                for j in nkey.get((p.y0, p.x0, p.sk), ()):
                    if j not in usedN:
                        take(p, j, "screen", (-sy, -sx)); break
        # 2 recolour in place (same shape and position, other colour)
        nshape = collections.defaultdict(list)
        for n in N_:
            if n.idx not in usedN:
                nshape[(n.y0, n.x0, n.h, n.w, n.m.tobytes())].append(n.idx)
        for p in P_:
            if p.idx in res:
                continue
            for j in nshape.get((p.y0 + sy, p.x0 + sx, p.h, p.w, p.m.tobytes()), ()):
                if j not in usedN:
                    take(p, j, "recol", (0, 0)); break
        # 3 exact translation within RMAX (Hungarian per shape); unique shapes may jump further
        byP = collections.defaultdict(list)
        for p in P_:
            if p.idx not in res:
                byP[p.sk].append(p)
        byN = collections.defaultdict(list)
        for n in N_:
            if n.idx not in usedN:
                byN[n.sk].append(n)
        for sk, plist in byP.items():
            nlist = byN.get(sk)
            if not nlist:
                continue
            C = np.full((len(plist), len(nlist)), 1e6)
            for i, p in enumerate(plist):
                for j, n in enumerate(nlist):
                    dy, dx = n.y0 - p.y0 - sy, n.x0 - p.x0 - sx
                    if max(abs(dy), abs(dx)) <= RMAX:
                        C[i, j] = abs(dy) + abs(dx)
                    elif (len(plist) == 1 and len(nlist) == 1) or reset:
                        C[i, j] = 1e3 + abs(dy) + abs(dx)
            ri, ci = linear_sum_assignment(C)
            for i, j in zip(ri, ci):
                if C[i, j] >= 1e6:
                    continue
                p, n = plist[i], nlist[j]
                take(p, n.idx, "jump" if C[i, j] >= 1e3 else "move", (n.y0 - p.y0 - sy, n.x0 - p.x0 - sx))
        # 3b turned shapes: same colour, same pixel count, same shape up to rotation/reflection, within
        #    RMAX (bounding-box centres); the rigid pass later gives a turned multi-part sprite one shift
        remP = [p for p in P_ if p.idx not in res]
        remN = [n for n in N_ if n.idx not in usedN]
        if remP and remN:
            dkN = collections.defaultdict(list)
            for n in remN:
                dkN[(n.c, n.n, d4key(n.m))].append(n)
            byk = collections.defaultdict(list)
            for p in remP:
                k = (p.c, p.n, d4key(p.m))
                if k in dkN:
                    byk[k].append(p)
            for k, plist in byk.items():
                nlist = [n for n in dkN[k] if n.idx not in usedN]
                if not nlist:
                    continue
                C = np.full((len(plist), len(nlist)), 1e6)
                for i, p_ in enumerate(plist):
                    for j, n in enumerate(nlist):
                        dy = (n.y0 + n.h / 2) - (p_.y0 + p_.h / 2) - sy
                        dx = (n.x0 + n.w / 2) - (p_.x0 + p_.w / 2) - sx
                        if max(abs(dy), abs(dx)) <= RMAX:
                            C[i, j] = abs(dy) + abs(dx)
                ri, ci = linear_sum_assignment(C)
                for i, j in zip(ri, ci):
                    if C[i, j] >= 1e6:
                        continue
                    p_, n = plist[i], nlist[j]
                    dy = (n.y0 + n.h / 2) - (p_.y0 + p_.h / 2) - sy
                    dx = (n.x0 + n.w / 2) - (p_.x0 + p_.w / 2) - sx
                    take(p_, n.idx, "blob", (int(round(dy)), int(round(dx))))
        # 3c repainted and moved: same shape, other colour, within RMAX
        remP = [p for p in P_ if p.idx not in res]
        remN = [n for n in N_ if n.idx not in usedN]
        if remP and remN:
            shN = collections.defaultdict(list)
            for n in remN:
                shN[(n.h, n.w, n.m.tobytes())].append(n)
            byk = collections.defaultdict(list)
            for p_ in remP:
                k = (p_.h, p_.w, p_.m.tobytes())
                if k in shN and p_.n >= 3:
                    byk[k].append(p_)
            for k, plist in byk.items():
                nlist = [n for n in shN[k] if n.idx not in usedN]
                if not nlist:
                    continue
                C = np.full((len(plist), len(nlist)), 1e6)
                for i, p_ in enumerate(plist):
                    for j, n in enumerate(nlist):
                        dy, dx = n.y0 - p_.y0 - sy, n.x0 - p_.x0 - sx
                        if max(abs(dy), abs(dx)) <= RMAX:
                            C[i, j] = abs(dy) + abs(dx)
                ri, ci = linear_sum_assignment(C)
                for i, j in zip(ri, ci):
                    if C[i, j] < 1e6:
                        p_, n = plist[i], nlist[j]
                        take(p_, n.idx, "move", (n.y0 - p_.y0 - sy, n.x0 - p_.x0 - sx))
        # 4 blobs: leftover components grouped by adjacency, matched by identical colour histogram
        #   (a sprite that turned to face its move, or a multi-colour sprite whose parts re-split)
        remP = [p for p in P_ if p.idx not in res]
        remN = [n for n in N_ if n.idx not in usedN]
        if remP and remN:
            bP, bN = self._blobs(remP), self._blobs(remN)
            sigN = collections.defaultdict(list)
            for k, bl in enumerate(bN):
                sigN[bl[0]].append(k)
            pairs = []
            for i, bl in enumerate(bP):
                for k in sigN.get(bl[0], ()):
                    bn = bN[k]
                    dy, dx = bn[1] - bl[1] - sy, bn[2] - bl[2] - sx
                    if max(abs(dy), abs(dx)) <= RMAX:
                        pairs.append((abs(dy) + abs(dx), i, k, (int(round(dy)), int(round(dx)))))
            pairs.sort()
            doneP, doneN = set(), set()
            for _, i, k, d in pairs:
                if i in doneP or k in doneN:
                    continue
                doneP.add(i); doneN.add(k)
                # pair member components by colour, then size
                pc = sorted(bP[i][3], key=lambda c: (c.c, -c.n))
                nc = sorted(bN[k][3], key=lambda c: (c.c, -c.n))
                for p, n in zip(pc, nc):
                    take(p, n.idx, "blob", d)
        # 5 deformed: same colour, bounding boxes overlap (after view shift), similar size
        remP = [p for p in P_ if p.idx not in res]
        remN = [n for n in N_ if n.idx not in usedN]
        if remP and remN:
            C = np.full((len(remP), len(remN)), 1e6)
            for i, p in enumerate(remP):
                for j, n in enumerate(remN):
                    if n.c != p.c or not (0.5 <= n.n / p.n <= 2.0):
                        continue
                    oy = min(p.y0 + sy + p.h, n.y0 + n.h) - max(p.y0 + sy, n.y0)
                    ox = min(p.x0 + sx + p.w, n.x0 + n.w) - max(p.x0 + sx, n.x0)
                    if oy <= 0 or ox <= 0:
                        continue
                    C[i, j] = abs(n.cy - p.cy - sy) + abs(n.cx - p.cx - sx)
            ri, ci = linear_sum_assignment(C)
            for i, j in zip(ri, ci):
                if C[i, j] >= 1e6:
                    continue
                p, n = remP[i], remN[j]
                take(p, n.idx, "deform", (int(round(n.cy - p.cy - sy)), int(round(n.cx - p.cx - sx))))
        # 6 absorb / split in place (recolour that merges with or splits from same-colour neighbours):
        #   an old component >= 80% covered by one new component, or a new component >= 80% inside
        #   one old component (any colour). The best-overlapping pair keeps the track.
        self._absorbed, self._split = {}, collections.defaultdict(list)
        remP = [p for p in P_ if p.idx not in res]
        remN = [n for n in N_ if n.idx not in usedN]
        if remP and remN:
            labN = np.full((64, 64), -1, np.int32)
            for n in N_:
                labN[n.y0:n.y0 + n.h, n.x0:n.x0 + n.w][n.m] = n.idx
            labP = np.full((64, 64), -1, np.int32)
            for p in remP:
                ys, xs = np.nonzero(p.m)
                ys, xs = ys + p.y0 + sy, xs + p.x0 + sx
                ok = (ys >= 0) & (ys < 64) & (xs >= 0) & (xs < 64)
                labP[ys[ok], xs[ok]] = p.idx
            ov = collections.Counter()
            both = (labP >= 0) & (labN >= 0)
            for u, v in zip(labP[both].tolist(), labN[both].tolist()):
                ov[(u, v)] += 1
            Pn = {p.idx: p for p in remP}
            Nn = {n.idx: n for n in N_}
            cand = []
            for (u, v), k in ov.items():
                if k >= 0.8 * Pn[u].n or k >= 0.8 * Nn[v].n:
                    cand.append((k, u, v))
            cand.sort(reverse=True)
            for k, u, v in cand:
                if u in res or v in usedN:
                    continue
                take(Pn[u], v, "absorb", None)
            for k, u, v in cand:
                if u not in res and v in usedN and k >= 0.8 * Pn[u].n:
                    self._absorbed[u] = v          # old comp merged into an already-owned new comp
                elif u in res and res[u][0] != v and v not in usedN and k >= 0.8 * Nn[v].n:
                    self._split[u].append(v)       # new comp split off an owned old comp
                    usedN.add(v)
        return res, usedN

    def _rigid(self, P_, N_, res):
        """Adjacent moved components with different displacements that moved as one rigid sprite (their
        union keeps its bounding-box size, or swaps it: a turn) get the union's bounding-box-centre
        displacement. Rigid sets are connected sets of pairwise-rigid adjacent components."""
        mv = [p for p in P_ if p.idx in res and res[p.idx][1] in ("move", "blob", "jump", "deform")
              and res[p.idx][2] not in (None, (0, 0))]
        if len(mv) < 2 or len(mv) > 40:
            return

        def bb(cs):
            return (min(c.y0 for c in cs), max(c.y0 + c.h for c in cs), min(c.x0 for c in cs), max(c.x0 + c.w for c in cs))

        def rigid(cs):
            news = [N_[res[c.idx][0]] for c in cs]
            o, n = bb(cs), bb(news)
            return sorted((o[1] - o[0], o[3] - o[2])) == sorted((n[1] - n[0], n[3] - n[2])), o, n
        parent = {c.idx: c.idx for c in mv}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i, a_ in enumerate(mv):
            for b_ in mv[i + 1:]:
                if res[a_.idx][2] == res[b_.idx][2]:
                    continue
                # bounding boxes within 1 px, then pixel adjacency
                if a_.y0 > b_.y0 + b_.h or b_.y0 > a_.y0 + a_.h or a_.x0 > b_.x0 + b_.w or b_.x0 > a_.x0 + a_.w:
                    continue
                ma = ndimage.binary_dilation(a_.full(), structure=np.ones((3, 3), bool))
                if not (ma & b_.full()).any():
                    continue
                if rigid([a_, b_])[0]:
                    parent[find(a_.idx)] = find(b_.idx)
        sets = collections.defaultdict(list)
        for c in mv:
            sets[find(c.idx)].append(c)
        for cs in sets.values():
            if len(cs) < 2:
                continue
            ok, o, n = rigid(cs)
            if not ok:
                continue
            d = (int(round((n[0] + n[1]) / 2 - (o[0] + o[1]) / 2 - self.view[0])),
                 int(round((n[2] + n[3]) / 2 - (o[2] + o[3]) / 2 - self.view[1])))
            for c in cs:
                j, kind, _ = res[c.idx]
                res[c.idx] = (j, "blob", d)

    @staticmethod
    def _blobs(comps):
        """Group components within 1 px (8-neighbourhood gap <= 1); return [(sig, cy, cx, comps)]
        with sig = sorted (colour, pixels) and (cy, cx) the bounding-box centre."""
        lab = np.full((64, 64), -1, np.int32)
        for k, c in enumerate(comps):
            lab[c.y0:c.y0 + c.h, c.x0:c.x0 + c.w][c.m] = k
        reg, _ = ndimage.label(ndimage.binary_dilation(lab >= 0, structure=np.ones((3, 3), bool)),
                               structure=np.ones((3, 3), bool))
        groups = collections.defaultdict(list)
        for c in comps:
            groups[int(reg[c.y0 + int(np.argmax(c.m.any(1))), c.x0 + int(np.argmax(c.m[np.argmax(c.m.any(1))]))])].append(c)
        out = []
        for g, cs in groups.items():
            col = collections.Counter()
            for c in cs:
                col[c.c] += c.n
            y0 = min(c.y0 for c in cs); y1 = max(c.y0 + c.h for c in cs)
            x0 = min(c.x0 for c in cs); x1 = max(c.x0 + c.w for c in cs)
            out.append((tuple(sorted(col.items())), (y0 + y1) / 2, (x0 + x1) / 2, cs))
        return out

    # ------------------------------------------------------------------ main update
    def update(self, frame, action, click=None, new_level=False):
        """Feed the frame observed after `action`. Returns the result-dict fields for this action."""
        t0 = time.perf_counter()
        frame = np.asarray(frame)
        self.events = []
        if new_level or self.frame is None or action == "RESET":
            m, conf = self.claim() if self.frame is not None else (None, None)
            if conf == "committed":
                ents = self.entities()
                r = self._root_of(self.committed_sel or self.sel, ents)
                if r is not None:
                    self.carry_sig = self._sig(ents[r])
            if action == "RESET" and not new_level and self.frame is not None:
                keep = (self.t, self.facts, self.blocks, self.block_steps, self.block_pub, self.next_pub)
                self.reset_level(frame)
                self.t, self.facts, self.blocks, self.block_steps, self.block_pub, self.next_pub = keep
            else:
                self.reset_level(frame)
            self.last_ms = (time.perf_counter() - t0) * 1000
            return self.fields()
        self.t += 1
        t = self.t
        prev = self.frame
        vals, cnt = np.unique(frame, return_counts=True)
        if (frame == self.bg).sum() < 0.25 * frame.size:
            self.bg = int(vals[cnt.argmax()])
        nch = int((prev != frame).sum())
        if nch == 0:
            self.prev_frame = prev
            self.frame = frame
            alive = [tid for tid, tr in self.tracks.items() if not tr.hidden]
            for tid in alive:
                self.tracks[tid].prev = self.tracks[tid].comp
            self._after(action, {tid: (0, 0) for tid in alive}, {tid: "same" for tid in alive}, t)
            self._update_hold(action)
            self.last_ms = (time.perf_counter() - t0) * 1000
            return self.fields()
        vs = view_shift(prev, frame, self.bg, nch, self.p)
        self.view = vs
        if vs != (0, 0) and not self._scroll_said:
            self._scroll_said = True
            self.events.append(("scroll", action, vs))
        N_, lab = extract(frame, self.bg)
        P_ = self.comps
        res, usedN = self._match(P_, N_, vs, reset=(action == "RESET"))
        moved = {}       # tid -> (dy, dx) or None (unknown)
        kinds = {}
        self._rigid(P_, N_, res)
        if vs != (0, 0):
            # anything that kept its screen position under a scroll is screen-fixed: motion unknown
            for k, (j, kind, d) in list(res.items()):
                if d is not None and d == (-vs[0], -vs[1]):
                    res[k] = (j, "screen", None)
        split_new = []
        for p in P_:
            tr = self.tracks[p.tid]
            tr.prev = p
            if p.idx in res:
                j, kind, d = res[p.idx]
                n = N_[j]
                n.tid = tr.tid
                tr.comp = n
                kinds[tr.tid] = kind
                if kind in ("deform", "recol", "absorb"):
                    tr.changes += 1
                if kind in EXACT:
                    moved[tr.tid] = d
                    if d != (0, 0):
                        tr.moves[t] = d
                else:
                    moved[tr.tid] = None
                    tr.unk.add(t)
                for j2 in self._split.get(p.idx, ()):
                    split_new.append((tr.tid, N_[j2]))
            else:
                tr.hidden, tr.hid_at = True, t
                tr.unk.add(t)
                kinds[tr.tid] = "absorbed" if p.idx in self._absorbed else "gone"
        for owner, n in split_new:
            nt = self._new_track(n)
            kinds[nt.tid] = "splitnew"
            moved[nt.tid] = None
            nt.unk.add(t)
            self.links[(min(owner, nt.tid), max(owner, nt.tid))] = True
        # re-appearance of hidden tracks (exact shape near the last position), else new tracks
        hidden = [tr for tr in self.tracks.values() if tr.hidden and tr.hid_at < t]
        hid_by = collections.defaultdict(list)
        for tr in hidden:
            hid_by[tr.comp.sk].append(tr)
        for n in N_:
            if n.idx in usedN or n.tid >= 0:
                continue
            best = None
            for tr in hid_by.get(n.sk, ()):
                dd = max(abs(n.y0 - tr.comp.y0 - vs[0]), abs(n.x0 - tr.comp.x0 - vs[1]))
                if dd <= self.p["RMAX"] and (best is None or dd < best[0]):
                    best = (dd, tr)
            if best is not None:
                tr = best[1]
                hid_by[n.sk].remove(tr)
                tr.hidden = False
                tr.comp = n
                n.tid = tr.tid
                kinds[tr.tid] = "reapp"
                moved[tr.tid] = None
                tr.unk.add(t)
            else:
                self._new_track(n)
                kinds[n.tid] = "new"
                moved[n.tid] = None
        self.prev_frame = prev
        self.comps, self.lab, self.frame = N_, lab, frame
        if action != "RESET":
            self._link(P_, moved, t)
        self._after(action, moved, kinds, t)
        self._update_hold(action)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return self.fields()

    # ------------------------------------------------------------------ co-motion grouping
    def _pairs(self, lab, g):
        """Pairs of component indices within Chebyshev distance g in a label image."""
        H, W = lab.shape
        out = set()
        for dy in range(0, g + 1):
            for dx in range(-g, g + 1):
                if dy == 0 and dx <= 0:
                    continue
                ya, yb = max(0, -dy), min(H, H - dy)
                xa, xb = max(0, -dx), min(W, W - dx)
                a = lab[ya:yb, xa:xb]
                b = lab[ya + dy:yb + dy, xa + dx:xb + dx]
                mk = (a >= 0) & (b >= 0) & (a != b)
                if mk.any():
                    for u, v in zip(a[mk].tolist(), b[mk].tolist()):
                        out.add((u, v) if u < v else (v, u))
        return out

    def _same_history(self, A, B, t):
        start = max(A.born, B.born)
        unk = A.unk | B.unk
        steps = set(A.moves) | set(B.moves)
        n_known = (t - start + 1) - len([u for u in unk if u >= start])
        if n_known <= 0:
            return True
        diff = sum(1 for s_ in steps if s_ >= start and s_ not in unk and A.moves.get(s_, (0, 0)) != B.moves.get(s_, (0, 0)))
        return diff <= min(self.p["LINK_DIFF"], (1 - self.p["LINK_AGREE"]) * n_known)

    def _link(self, P_, moved, t):
        movers = {tid for tid, d in moved.items() if d not in (None, (0, 0))}
        # strike links whose motion diverged this step; break after LINK_STRIKES in a row
        for key in list(self.links):
            a, b = key
            da, db = moved.get(a), moved.get(b)
            if da is None or db is None:
                continue
            if da != db:
                self.strikes[key] += 1
                if self.strikes[key] >= self.p["LINK_STRIKES"]:
                    del self.links[key]
                    self.strikes.pop(key, None)
            elif da != (0, 0):
                self.strikes.pop(key, None)
        if not movers:
            return
        # candidate pairs: components adjacent in the previous frame, at least one moved
        prev_lab = np.full((64, 64), -1, np.int32)
        idx2tid = {}
        for p in P_:
            prev_lab[p.y0:p.y0 + p.h, p.x0:p.x0 + p.w][p.m] = p.idx
            idx2tid[p.idx] = p.tid
        for (u, v) in self._pairs(prev_lab, self.p["LINK_GAP"]):
            a, b = idx2tid[u], idx2tid[v]
            if a not in movers and b not in movers:
                continue
            key = (min(a, b), max(a, b))
            if key in self.links:
                continue
            if moved.get(a) == moved.get(b) and self._same_history(self.tracks[a], self.tracks[b], t):
                self.links[key] = True
                self.ever_linked.add(key)

    # ------------------------------------------------------------------ roles and facts
    def _ent_motion(self, members, moved):
        ds = collections.Counter()
        for m in members:
            d = moved.get(m)
            if d is not None:
                ds[d] += self.tracks[m].comp.n
        if not ds:
            return None
        return ds.most_common(1)[0][0]

    def _controllable(self, members):
        """(agree, moved, present) of the best member track."""
        best = max(members, key=lambda m: (self.tracks[m].agree, self.tracks[m].moved))
        tr = self.tracks[best]
        return tr.agree, tr.moved, tr.present

    def _is_ctrl(self, members, sole=False):
        ag, mv, pr = self._controllable(members)
        if mv == 0 or ag / mv < self.p["AGREE_RATIO"]:
            return False
        if max(self.tracks[m].auto for m in members) > ag:
            return False  # moves on its own (non-arrow actions) more than with the arrows
        tiny = sum(self.tracks[m].comp.n for m in members) <= self.p["TINY"]
        if ag >= self.p["COMMIT_AGREE"] + (1 if tiny else 0):
            return True
        if tiny:
            return False
        if ag >= 1 and sole and (self._sig(members) in self.ctrl_sigs or self.pending_switch is not None
                                 and self.switch_acts[self.pending_switch[0]] >= 1):
            return True
        return False

    def _controlled(self, ents=None):
        if not self.sel:
            return None
        ents = ents if ents is not None else self.entities()
        best, bo = None, 0
        for r, mem in ents.items():
            o = len(self.sel & set(mem))
            if o > bo:
                best, bo = r, o
        if best is None:
            return None
        self.sel = self._core(self.sel, ents[best])
        return best, sorted(self.sel), self.sel_conf

    def _core(self, core, members):
        """Alive core members plus linked members that co-moved with the core at least once."""
        alive = {m for m in core if m in members}
        moves = {}
        for m in alive:
            for st, d in self.tracks[m].moves.items():
                moves[st] = d
        out = set(alive)
        for m in members:
            if m in alive:
                continue
            mv = self.tracks[m].moves
            if any(moves.get(st) == d for st, d in mv.items()):
                out.add(m)
        return out if out else set(members)

    def _after(self, action, moved, kinds, t):
        self._mv = moved
        sel_before = set(self.sel)
        ents = self.entities()
        self._last_d = {}
        if self.p["INHERIT"]:
            self._inherit(ents, kinds)
        tid2root = {m: r for r, mem in ents.items() for m in mem}
        if action in ARROWS:
            ey, ex = ARROWS[action]
            agreeing = []
            for r, mem in ents.items():
                # statistics per member track (present on this press)
                for m in mem:
                    tr = self.tracks[m]
                    if m not in moved:
                        continue
                    tr.present += 1
                    d = moved[m]
                    if d is None or d == (0, 0):
                        continue
                    tr.moved += 1
                    along = d[0] * ey + d[1] * ex
                    across = abs(d[0] * ex) + abs(d[1] * ey)
                    if along > 0 and across <= along / 2:
                        tr.agree += 1
                        tr.steps[along] += 1
                        tr.dirs.add(action)
                        tr.last_move = t
                d = self._ent_motion(mem, moved)
                self._last_d[r] = d
                if d is not None and d != (0, 0):
                    along = d[0] * ey + d[1] * ex
                    across = abs(d[0] * ex) + abs(d[1] * ey)
                    if along > 0 and across <= along / 2:
                        agreeing.append(r)
            tm = self._tmpl_step(ents, kinds, moved, action)
            if tm is not None and tm != (0, 0):
                along = tm[0] * ey + tm[1] * ex
                if along > 0 and abs(tm[0] * ex) + abs(tm[1] * ey) <= along / 2:
                    self.pending_switch = None
                    self.rival.clear()
                    tm = "agree"
            self._select(agreeing, ents, action, t, kinds, tmpl_agree=(tm == "agree"))
            self._blocked(action, moved, kinds, ents, t)
        else:
            self._tmpl_step(ents, kinds, moved, action)
            if action != "RESET":
                for m, d in moved.items():
                    if d not in (None, (0, 0)) and m not in sel_before:
                        self.tracks[m].auto += 1
                self._non_arrow(action, moved, kinds, ents, t, sel_before)
        self._vanish(action, moved, kinds, ents, t)
        self._learn_walkable(action, moved, ents)

    def _root_of(self, members, ents):
        best, bo = None, 0
        for r, mem in ents.items():
            o = len(members & set(mem))
            if o > bo:
                best, bo = r, o
        return best

    def _select(self, agreeing, ents, action, t, kinds, tmpl_agree=False):
        if tmpl_agree or not agreeing:
            return
        cur = self._controlled(ents)
        cur_r = cur[0] if cur else None
        last = self._root_of(self.last_mover, ents) if self.last_mover else None
        com = self._root_of(self.committed_sel, ents) if self.committed_sel else None
        ctrl = {r for r in agreeing if self._is_ctrl(ents[r], sole=len(agreeing) == 1)}
        if self.p["SOLE1"] and len(agreeing) == 1 and not ctrl \
                and sum(self.tracks[m].comp.n for m in ents[agreeing[0]]) > self.p["TINY"]:
            r = agreeing[0]
            others = [m for m, k in kinds.items() if k in ("move", "blob", "jump") and m not in ents[r]
                      and self._mv.get(m) not in (None, (0, 0))]
            d = self._last_d.get(r)
            ag, mv, pr = self._controllable(ents[r])
            if not others and d is not None and (d[0] == 0 or d[1] == 0) and ag >= 1 and ag == mv \
                    and not any(self.tracks[m].auto for m in ents[r]):
                ctrl.add(r)
        if cur_r in agreeing:
            pick = cur_r
            self.rival.clear()
        elif com in agreeing:
            pick = com
        elif last in agreeing:
            pick = last
        else:
            pick = max(agreeing, key=lambda r: (r in ctrl, self._controllable(ents[r])[0],
                                                sum(self.tracks[m].comp.n for m in ents[r]), -r))
            contradicted = False
            if com is not None:
                dcom = self._last_d.get(com)
                ey, ex = ARROWS[action]
                if dcom not in (None, (0, 0)) and dcom[0] * ey + dcom[1] * ex <= 0:
                    contradicted = True  # the committed piece moved, but not with the press
            if com is not None and not contradicted and not self._switch_ok(pick, com, ents, kinds):
                # a committed piece did not move and this is no clean switch: keep it, unless a rival
                # keeps agreeing with the arrows while it sits still
                key = min(ents[pick])
                for k in list(self.rival):
                    if k != key:
                        del self.rival[k]
                self.rival[key] += 1
                if self.rival[key] < self.p["RIVAL"]:
                    return
            self.rival.clear()
        switched = last is not None and pick != last and last not in agreeing
        if switched and self.pending_switch is not None:
            self.switch_acts[self.pending_switch[0]] += 1
            self.events.append(("switch", self.pending_switch[0], tuple(ents[pick])))
        commit = pick in ctrl or (switched and self.pending_switch is not None and bool(self.committed_sel))
        if pick != com:
            self.tmpl, self.tmpl_at = None, None
        self.sel = set(ents[pick])
        self.sel_conf = "committed" if commit else "tentative"
        self.sel_src = "arrow"
        if commit:
            self.committed_sel = set(ents[pick])
            self._remember_sig(ents[pick])
        self.last_mover = set(ents[pick])
        self.pending_switch = None

    def _tmpl_build(self, mem):
        m = self.mask_of(mem)
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            return
        y0, x0, y1, x1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
        patch = np.where(m[y0:y1, x0:x1], self.frame[y0:y1, x0:x1], -1).astype(np.int16)
        self.tmpl = {"patch": patch, "y0": int(y0), "x0": int(x0), "n": int(m.sum())}

    def _tmpl_find(self, expect):
        T = self.tmpl
        fr = self.frame
        P0 = T["patch"]
        variants = [P0]
        if self.turns:
            for q in (P0, P0[:, ::-1]):
                for r in range(4):
                    z = np.rot90(q, r)
                    if not any(z.shape == v.shape and np.array_equal(z, v) for v in variants):
                        variants.append(z)
        cy, cx = T["y0"] + P0.shape[0] / 2, T["x0"] + P0.shape[1] / 2
        R = self.p["RMAX"]
        rng = np.arange(-R, R + 1)
        DY, DX = np.meshgrid(rng, rng, indexing="ij")
        DY, DX = DY.ravel(), DX.ravel()
        bonus = np.zeros(DY.shape)
        for ey, ex in expect:
            bonus[(DY == ey) & (DX == ex)] = 0.02
        best = None
        for v in variants:
            h, w = v.shape
            ys, xs = np.nonzero(v >= 0)
            cols = v[ys, xs]
            by, bx = int(round(cy - h / 2)), int(round(cx - w / 2))
            Y = by + DY[:, None] + ys[None, :]
            X = bx + DX[:, None] + xs[None, :]
            inb = (Y >= 0) & (Y < 64) & (X >= 0) & (X < 64)
            vals = fr[np.clip(Y, 0, 63), np.clip(X, 0, 63)]
            sc = ((vals == cols[None, :]) & inb).mean(1)
            val = sc + bonus - (np.abs(DY) + np.abs(DX)) * 1e-4
            k = int(np.argmax(val))
            if best is None or val[k] > best[0]:
                best = (val[k], sc[k], by + DY[k], bx + DX[k], v, (int(DY[k] + by + h / 2 - cy), int(DX[k] + bx + w / 2 - cx)))
        if best is None or best[1] < self.p["TMPL_MIN"]:
            return None
        return best

    def _tmpl_mask(self):
        y0, x0, v = self.tmpl_at
        m = np.zeros((64, 64), bool)
        h, w = v.shape
        ys, xs = np.nonzero(v >= 0)
        ys, xs = ys + y0, xs + x0
        ok = (ys >= 0) & (ys < 64) & (xs >= 0) & (xs < 64)
        m[ys[ok], xs[ok]] = True
        return m

    def _tmpl_step(self, ents, kinds, moved, action):
        """Keep the committed selection's template; when its components broke up, find it again.
        Returns the template displacement in template mode, else None."""
        if not self.p["TMPL"] or not self.committed_sel:
            return None
        r = self._root_of(self.committed_sel, ents)
        mem = ents[r] if r is not None else []
        if any(kinds.get(m) == "blob" for m in mem):
            self.turns = True
        clean = (r is not None and all(kinds.get(m, "same") in EXACT for m in mem) and
                 (self.tmpl is None or 0.8 <= sum(self.tracks[m].comp.n for m in mem) / self.tmpl["n"] <= 1.25))
        if clean and self.tmpl_at is None:
            self._tmpl_build(mem)
            return None
        if self.tmpl is None or self.tmpl["n"] < self.p["TMPL_NMIN"]:
            return None
        expect = [(0, 0)]
        if action in ARROWS:
            ey, ex = ARROWS[action]
            steps = collections.Counter()
            for m in self.committed_sel:
                if m in self.tracks:
                    steps.update(self.tracks[m].steps)
            for st in (list(steps) or [1]):
                expect.append((ey * st, ex * st))
        prev_at = self.tmpl_at
        found = self._tmpl_find(expect)
        if found is None:
            self.tmpl_at = None
            return None
        _, sc, y0, x0, v, d = found
        if prev_at is not None:
            d = (int(y0 - prev_at[0]), int(x0 - prev_at[1]))
        self.tmpl_at = (int(y0), int(x0), v)
        # re-bind: current components lying entirely inside the placed template that reproduce it exactly
        tm = self._tmpl_mask()
        inside = [c for c in self.comps if (tm[c.y0:c.y0 + c.h, c.x0:c.x0 + c.w] | ~c.m).all()]
        cov = np.zeros((64, 64), bool)
        for c in inside:
            cov[c.y0:c.y0 + c.h, c.x0:c.x0 + c.w] |= c.m
        if inside and (cov == tm).all():
            tids = sorted(c.tid for c in inside)
            for a, b in zip(tids, tids[1:]):
                self.links[(min(a, b), max(a, b))] = True
            self.committed_sel = set(tids)
            if self.sel_conf == "committed" or not self.sel:
                self.sel = set(tids)
            self.tmpl_at = None
            self._tmpl_build(set(tids))
        return d

    def _remember_sig(self, mem):
        sig = self._sig(mem)
        self.ctrl_sigs.add(sig)
        if sig in self.carry_sigs:
            self.carry_sigs.remove(sig)
        self.carry_sigs.insert(0, sig)
        del self.carry_sigs[8:]

    def _inherit(self, ents, kinds):
        """The committed selection vanished this step (no member alive) and exactly one entity that is new
        or changed this step overlaps where it was: the sprite transformed in place, keep the claim on it."""
        if not self.committed_sel or self.tmpl_at is not None:
            return
        if any(m in self.tracks and not self.tracks[m].hidden for m in self.committed_sel):
            return
        gone = [m for m in self.committed_sel if kinds.get(m) in ("gone", "absorbed")]
        if not gone:
            return
        old = np.zeros((64, 64), bool)
        for m in gone:
            c = self.tracks[m].prev or self.tracks[m].comp
            old[c.y0:c.y0 + c.h, c.x0:c.x0 + c.w] |= c.m
        old = ndimage.binary_dilation(old, structure=np.ones((3, 3), bool))
        cands = [r for r, mem in ents.items() if any(kinds.get(m) in ("new", "splitnew", "deform", "absorb") for m in mem)
                 and (self.mask_of(mem) & old).any() and sum(self.tracks[m].comp.n for m in mem) <= 4 * old.sum()]
        if len(cands) == 1:
            self.committed_sel = set(ents[cands[0]])
            self.sel = set(ents[cands[0]])
            self.sel_conf = "committed"
            self.sel_src = "inherit"
            self.tmpl = None
            self.events.append(("inherit", tuple(ents[cands[0]])))

    def _switch_ok(self, pick, com, ents, kinds):
        """A new mover replaces a committed selection only after a non-arrow action, when the old piece
        sat unchanged, the new one never moved on its own and is not touching the old one."""
        if self.pending_switch is None:
            return False
        if any(kinds.get(m) not in ("same", None) for m in ents[com]):
            return False
        mem = ents[pick]
        if any(self.tracks[m].auto > 0 for m in mem):
            return False
        ag, mv, pr = self._controllable(mem)
        if mv > 1 and (ag - 1) / (mv - 1) < self.p["AGREE_RATIO"]:
            return False
        a = ndimage.binary_dilation(self.mask_of(ents[com]), structure=np.ones((5, 5), bool))
        if (a & self.mask_of(mem)).any():
            return False
        return True

    def _hist(self, tids, before):
        h = collections.Counter()
        for m in tids:
            tr = self.tracks.get(m)
            if tr is None:
                continue
            c = tr.prev if before else (None if tr.hidden else tr.comp)
            if c is not None:
                h[c.c] += c.n
        return h

    def _non_arrow(self, action, moved, kinds, ents, t, sel_before):
        """Selection-switch handling for SPACE/click-like actions: marker transfer. Pixels of the selected
        sprite changed away from a colour that it no longer shows, and exactly one other entity (not touching
        it, not self-moving) has changed pixels inside its bounding box that took on that colour. Committed
        once this action has been seen to switch the mover in this game, tentative before."""
        if not sel_before or self.prev_frame is None:
            return
        self.pending_switch = (action, t)
        prev, cur = self.prev_frame, self.frame
        diff = prev != cur
        before = np.zeros((64, 64), bool)
        for m in sel_before:
            tr = self.tracks.get(m)
            c = tr.prev if tr is not None else None
            if c is not None:
                before[c.y0:c.y0 + c.h, c.x0:c.x0 + c.w] |= c.m
        ch = diff & before
        if not ch.any():
            return
        lost = set(np.unique(prev[ch]).tolist()) - set(np.unique(cur[before]).tolist())
        if not lost:
            return
        near = ndimage.binary_dilation(before, structure=np.ones((3, 3), bool))
        nsel = max(int(before.sum()), 1)
        cands = []
        for r, mem in ents.items():
            mk = self.mask_of(mem)
            if not mk.any() or (mk & near).any() or any(self.tracks[m].auto > 0 for m in mem):
                continue
            if mk.sum() > min(self.p["BLOCK_MAXPX"], 4 * max(nsel, 16)):
                continue  # far larger than the selected piece: board / background, not a sibling
            ys, xs = np.nonzero(mk)
            box = np.zeros((64, 64), bool)
            box[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
            g = diff & box & ~before & np.isin(cur, list(lost))
            if g.any():
                cands.append(r)
        if len(cands) == 1:
            learned = self.switch_acts[action] >= 1
            self.sel = set(ents[cands[0]])
            self.sel_conf = "committed" if learned else "tentative"
            self.sel_src = "switch"
            if learned:
                self.tmpl, self.tmpl_at = None, None
                self.committed_sel = set(ents[cands[0]])
            self.events.append(("switch_pred", action, tuple(ents[cands[0]]), learned))

    def _sig_like(self, a, b):
        """Same size class and colour set (a sibling piece), ignoring the selection marker."""
        ca = {self.tracks[m].comp.c for m in a}
        cb = {self.tracks[m].comp.c for m in b}
        na = sum(self.tracks[m].comp.n for m in a)
        nb = sum(self.tracks[m].comp.n for m in b)
        return len(ca & cb) >= 1 and 0.5 <= na / max(nb, 1) <= 2.0

    def _learn_walkable(self, action, moved, ents):
        if action not in ARROWS or self.prev_frame is None:
            return
        c = self._controlled(ents)
        if c is None or c[2] != "committed" or self._ent_motion(c[1], moved) in (None, (0, 0)):
            return
        now = self.mask_of(c[1])
        before = np.zeros((64, 64), bool)
        for m in c[1]:
            pc = self.tracks[m].prev
            if pc is not None:
                before |= pc.full()
        onto = now & ~before
        if onto.any():
            self.walkable |= set(np.unique(self.prev_frame[onto]).tolist())

    def _blocked(self, action, moved, kinds, ents, t):
        """The committed controlled entity (>= BLOCK_AGREE agreeing moves) did not move or change on this arrow
        press, and a compact non-background entity lies within one of its steps in the pressed direction."""
        if self.tmpl_at is not None:
            return
        ctrl = self._controlled(ents)
        if ctrl is None or ctrl[2] != "committed":
            return
        r, mem, _ = ctrl
        if any(kinds.get(m) != "same" for m in mem):
            return  # it moved or changed look
        if self._controllable(mem)[0] < self.p["BLOCK_AGREE"]:
            return
        if self.p["BLOCK_DIRSEEN"] and not any(action in self.tracks[m].dirs for m in mem):
            return  # it has never moved this way: no evidence that this press can move it
        steps = collections.Counter()
        for m in mem:
            steps.update(self.tracks[m].steps)
        if not steps:
            return
        S = steps.most_common(1)[0][0]
        ey, ex = ARROWS[action]
        mask = self.mask_of(mem)
        sweep = np.zeros_like(mask)
        for k in range(1, S + 1):
            sh = np.roll(np.roll(mask, k * ey, 0), k * ex, 1)
            if ey > 0: sh[:k] = False
            if ey < 0: sh[-k:] = False
            if ex > 0: sh[:, :k] = False
            if ex < 0: sh[:, -k:] = False
            sweep |= sh
        sweep &= ~mask
        hit = self.lab[sweep]
        hit = hit[hit >= 0]
        if hit.size == 0:
            return
        tids = {self.comps[i].tid for i in np.unique(hit).tolist()}
        tid2root = {mm: rr for rr, ms in ents.items() for mm in ms}
        broots = {tid2root[tt] for tt in tids if tt in tid2root}
        broots.discard(r)
        for b in broots:
            bm = self.mask_of(ents[b])
            if bm.sum() > self.p["BLOCK_MAXPX"] or b in mem:
                continue
            if any(self.tracks[m].changes >= self.p["BLOCK_HUDCHG"] for m in ents[b]):
                continue  # changes shape often: a bar / counter, not a wall
            if any((min(x, y), max(x, y)) in self.ever_linked for x in ents[b] for y in mem):
                continue  # was part of the mover at some point
            bcols = {self.tracks[m].comp.c for m in ents[b]}
            if self.p["BLOCK_WALK"] and bcols <= self.walkable:
                continue  # the mover has walked onto these colours before
            key = (b, r, action)
            self.blocks[key] += 1
            self.block_steps[key].append(t)
            if self.blocks[key] >= self.p["BLOCK_MIN"] and key not in self.block_pub:
                self.block_pub.add(key)
                rec = {"type": "blocked", "step": t, "mover": r, "blocker": b, "action": action,
                       "mover_mask": mask, "blocker_mask": bm, "n": self.blocks[key],
                       "mover_name_tids": sorted(mem), "blocker_tids": sorted(ents[b])}
                self.facts.append(rec)
                self.events.append(("blocked", tuple(ents[b]), tuple(mem), action, self.blocks[key]))

    def _vanish(self, action, moved, kinds, ents, t):
        """Vanish-on-contact. On an arrow press where the committed controlled entity moved, each component
        that disappeared next to or under it (and whose co-moving members are not still there) is a
        candidate. It vanished when its colour no longer shows on its spot once the mover is off the spot
        (checked now for uncovered pixels, later for covered ones) and it has not re-appeared. Vanished
        components that touch are one fact (>= VAN_MINPX pixels)."""
        ctrl = self._controlled(ents)
        cmask = self.mask_of(ctrl[1]) if (ctrl is not None and self.tmpl_at is None) else None
        done = []
        keep = []
        for g, step, mover in self.vanish_pend:
            tr = self.tracks[g]
            if not tr.hidden:
                continue
            c = tr.comp
            spot = c.full()
            if cmask is not None and (spot & cmask).any():
                keep.append((g, step, mover))
                continue
            if (self.frame[spot] == c.c).any() or self._recoloured(spot, c, cmask):
                continue
            done.append((g, step, mover))
        self.vanish_pend = keep
        if action in ARROWS and ctrl is not None and ctrl[2] == "committed" and self.tmpl_at is None:
            r, mem, conf = ctrl
            if self._ent_motion(mem, moved) not in (None, (0, 0)):
                near = ndimage.binary_dilation(cmask, structure=np.ones((3, 3), bool))
                mcols = {self.tracks[m].comp.c for m in mem}
                pm = np.zeros((64, 64), bool)
                for m in mem:
                    pc = self.tracks[m].prev
                    if pc is not None:
                        pm |= pc.full()
                pm = ndimage.binary_dilation(pm, structure=np.ones((3, 3), bool))
                for g, k in kinds.items():
                    if k != "gone" or g in mem:
                        continue
                    linked = {b_ for (a_, b_) in self.links if a_ == g} | {a_ for (a_, b_) in self.links if b_ == g}
                    if any(x in self.tracks and not self.tracks[x].hidden for x in linked):
                        continue
                    c = self.tracks[g].comp
                    spot = c.full()
                    if not (spot & near).any():
                        continue
                    if c.c in mcols and (spot & pm).any():
                        continue  # likely a piece of the mover itself that failed to match
                    if (spot & cmask).any():
                        self.vanish_pend.append((g, t, r))
                    elif not (self.frame[spot] == c.c).any() and not self._recoloured(spot, c, cmask):
                        done.append((g, t, r))
        if not done:
            return
        for sig, cy, cx, cs in self._blobs([self.tracks[g].comp for g, _, _ in done]):
            n = sum(c.n for c in cs)
            if n < self.p["VAN_MINPX"]:
                continue
            tids = {c.tid for c in cs}
            step = min(st for g, st, _ in done if g in tids)
            mover = next(mv for g, _, mv in done if g in tids)
            vm = np.zeros((64, 64), bool)
            for c in cs:
                vm |= c.full()
            self._vanish_fact(tids, vm, mover, step, t)

    def _recoloured(self, spot, c, cmask):
        """A small current component (<= 4x the old one, not the mover) now covers >= 80% of the uncovered
        spot: the old component was recoloured / merged into a neighbour, it did not vanish."""
        under = spot & ~cmask if cmask is not None else spot
        if not under.any():
            return False
        ids = self.lab[under]
        ids = ids[ids >= 0]
        if ids.size == 0:
            return False
        vals, cnt = np.unique(ids, return_counts=True)
        j = int(vals[cnt.argmax()])
        return cnt.max() >= 0.8 * under.sum() and self.comps[j].n <= 4 * c.n

    @staticmethod
    def _local_bg(vm, excl, frame):
        """Most common colour on a ring around vm (1 then 2 px), excluding vm and excl; None if hidden."""
        for k in (3, 5):
            ring = ndimage.binary_dilation(vm, structure=np.ones((k, k), bool)) & ~vm
            if excl is not None:
                ring &= ~excl
            if ring.sum() >= 3:
                vals, cnt = np.unique(frame[ring], return_counts=True)
                return int(vals[cnt.argmax()])
        return None

    def _vanish_fact(self, grp, vm, mover, step, t):
        rec = {"type": "vanished", "step": t, "at": step, "tids": sorted(grp), "mask": vm, "mover": mover}
        self.facts.append(rec)
        self.events.append(("vanished", tuple(sorted(grp)), step))

    # ------------------------------------------------------------------ outputs
    def _update_hold(self, action):
        m, conf = self._claim_raw()
        if m is not None and conf == "committed":
            self._hold = (m, self.t, frozenset(self.committed_sel or self.sel))
            return
        if self._hold is None or self.t - self._hold[1] > self.p["HOLD"]:
            self._hold = None
            return
        hm, ht, mem = self._hold
        if action in ARROWS:
            steps = collections.Counter()
            dirs = set()
            for t_ in mem:
                if t_ in self.tracks:
                    steps.update(self.tracks[t_].steps)
                    dirs |= self.tracks[t_].dirs
            if steps and action in dirs:
                S = steps.most_common(1)[0][0]
                ey, ex = ARROWS[action]
                ys, xs = np.nonzero(hm)
                ys, xs = ys + ey * S, xs + ex * S
                ok = (ys >= 0) & (ys < 64) & (xs >= 0) & (xs < 64)
                hm = np.zeros((64, 64), bool)
                hm[ys[ok], xs[ok]] = True
        self._hold = (hm, ht, mem)
        self._held = hm

    def claim(self):
        """(mask, conf) of the controlled-entity claim; while the committed sprite is hidden, its last mask
        (dead-reckoned along known arrow steps) for up to HOLD steps."""
        m, conf = self._claim_raw()
        if (m is None or conf != "committed") and self._hold is not None and self.t - self._hold[1] <= self.p["HOLD"] \
                and self._hold[0].any():
            return self._hold[0], "committed"
        return m, conf

    def _claim_raw(self):
        """(mask, conf) of the controlled-entity claim: the selection if committed, else the last
        committed selection (still alive), else the tentative selection."""
        if self.tmpl_at is not None:
            return self._tmpl_mask(), "committed"
        ents = self.entities()
        c = self._controlled(ents)
        if c is not None and c[2] == "committed":
            return self.mask_of(c[1]), "committed"
        if self.committed_sel:
            r = self._root_of(self.committed_sel, ents)
            if r is not None:
                self.committed_sel = self._core(self.committed_sel, ents[r])
                return self.mask_of(self.committed_sel), "committed"
        if c is not None:
            return self.mask_of(c[1]), c[2]
        return None, None

    def selection(self):
        """(mask, conf) of the current selection, tentative or committed (falls back to the claim)."""
        c = self._controlled()
        if c is not None:
            if c[2] == "committed" and self.tmpl_at is not None:
                return self._tmpl_mask(), "committed"
            return self.mask_of(c[1]), c[2]
        return self.claim()

    # ------------------------------------------------------------------ delivery
    def _where(self, members):
        m = np.zeros((64, 64), bool)
        cols = set()
        for t in members:
            tr = self.tracks.get(t)
            if tr is None:
                continue
            c = tr.comp if not tr.hidden else (tr.prev or tr.comp)
            m[c.y0:c.y0 + c.h, c.x0:c.x0 + c.w] |= c.m
            cols.add(c.c)
        if not m.any():
            return ""
        ys, xs = np.nonzero(m)
        return f"r{ys.min()}-{ys.max()} c{xs.min()}-{xs.max()}"

    def _ctrl_text(self):
        """(name, 'E1 r..-.. c..-..', evidence) of the controlled entity, or (None, None, None)."""
        if self.tmpl_at is not None and self.committed_sel:
            mem = sorted(self.committed_sel)
            y0, x0, v = self.tmpl_at
            where = f"r{y0}-{y0 + v.shape[0] - 1} c{x0}-{x0 + v.shape[1] - 1}"
            conf = "committed"
        else:
            ents = self.entities()
            c = self._controlled(ents)
            if c is not None and c[2] == "committed":
                mem, conf = c[1], "committed"
            elif self.committed_sel and self._root_of(self.committed_sel, ents) is not None:
                mem, conf = sorted(self.committed_sel), "committed"
            elif c is not None:
                mem, conf = c[1], c[2]
            else:
                return None, None, None
            where = self._where(mem)
        nm = self.name(mem) + ("" if conf == "committed" else "?")
        ag = max((self.tracks[m].agree for m in mem if m in self.tracks), default=0)
        steps = collections.Counter()
        for m in mem:
            if m in self.tracks:
                steps.update(self.tracks[m].steps)
        ev = (f"{nm} moved with the arrows on {ag} press{'es' if ag != 1 else ''}" + (f", {steps.most_common(1)[0][0]}px each" if steps else "")) if ag else None
        return nm, f"{nm} {where}", ev

    def fields(self):
        """Result-dict fields for this action: the controlled entity (name + box) and up to 2 facts that are
        new this step. The tail line is a short pointer to the most important one."""
        nm, ctrl, evid = self._ctrl_text()
        out = {}
        if ctrl:
            out["controlled"] = ctrl
        ev = []
        who = nm or "the controlled entity"
        for e in self.events:
            k = e[0]
            if k == "switch":
                n2 = self.name(e[2])
                ev.append((0, f"{e[1]} switched control to {n2} {self._where(e[2])}", f"{e[1]} switched control to {n2}"))
            elif k == "switch_pred":
                n2 = self.name(e[2]) + ("" if e[3] else "?")
                ev.append((0, f"after {e[1]} the selection marker is on {n2} {self._where(e[2])}", f"after {e[1]} the selection marker is on {n2}"))
            elif k == "inherit":
                ev.append((1, f"controlled sprite changed shape: now {self.name(e[1])} {self._where(e[1])}", "the controlled sprite changed shape"))
            elif k == "vanished":
                n2 = self.name(e[1])
                ev.append((1, f"{n2} {self._where(e[1])} vanished when {who} moved onto it", f"{n2} vanished when {who} moved onto it"))
            elif k == "scroll":
                dy, dx = e[2]
                txt = f"view scrolled {abs(dy) or abs(dx)}px {'up' if dy < 0 else 'down' if dy > 0 else 'left' if dx < 0 else 'right'} on {e[1]}"
                ev.append((1, txt, txt))
            elif k == "blocked":
                n2 = self.name(e[1])
                ev.append((2, f"{n2} {self._where(e[1])} blocked {who} moving {DIRNAME[e[3]]}", f"{n2} blocked {who} moving {DIRNAME[e[3]]}"))
        committed_now = nm and not nm.endswith("?")
        if committed_now and self._last_ctrl_name != nm and evid and not any(e[0] == "inherit" for e in self.events):
            ev.append((1, evid, f"{nm} moves with the arrows"))
        ev.sort(key=lambda x: x[0])
        if ev:
            out["new_facts"] = [x[1] for x in ev[:2]]
        self._tail = ("Tracker: " + ev[0][2] + ("" if ev[0][2].endswith("?") else ".")) if ev else None
        if committed_now:
            self._last_ctrl_name = nm
        return out

    def tail_line(self):
        """At most one short line for the tool-result tail, only when something new happened."""
        return getattr(self, "_tail", None)
