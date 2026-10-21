"""Design A: evidence-accumulating per-entity ledger for ARC-AGI-3 frames (host-side, numpy/scipy only).

Per executed action the harness calls `update(action, click, frame)`; per level it calls `start_level(frame)`.

Pipeline per update:
  1. Segment the frame into plain 4-connected colour components (background = the level's modal colour).
  2. Estimate the global view shift (best translation voted by identical components, verified by pixel agreement)
     and remove it, so a scrolling board reads as static and the camera-centred player reads as moving.
  3. Match components to entities: exact (same colour+shape, world-static), exact with displacement (Hungarian
     per design), cost-based (position/colour/shape Hungarian), re-identification of recently lost entities.
  4. Ledger per entity, on arrow presses only: agree / opposite / perpendicular / still counts, step sizes.
  5. Co-motion grouping: entities that touch and move by the same displacement (never apart) report as one object.
  6. Roles stated only as behaviour: controllable set, selected entity (switch-aware: click target, the entity that
     gained the colour the old selection lost, a cursor that jumped into a controllable, learned SPACE cycle, and
     the entity that moved after a switch), blockers ("X failed to move into Y"), vanish-on-overlap (confirmed only
     after X leaves and Y does not reappear).
  7. `facts()` returns result-dict fields; `tail_line()` returns one <=40-token line.
"""
from __future__ import annotations

import collections
import time

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

ARROWS = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}  # (dy, dx)
ARROW_NAME = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT"}
CROSS = ndimage.generate_binary_structure(2, 1)
EIGHT = np.ones((3, 3), bool)

DEFAULT_CFG = dict(
    commit_agree=2,        # agreeing presses needed to commit an entity as controllable
    precision=0.8,         # agree / (agree + opposite + perpendicular) needed to stay controllable
    presence_ratio=0.0,    # agree / presses-present floor (0 = off)
    max_exact_dist=40.0,   # gate for exact-design matches with displacement (px)
    max_exact_dist_small=10.0,  # same, for designs of <= 4 px (ambiguous)
    max_cost_dist=14.0,    # gate for cost-based matches
    cost_accept=1.25,
    lost_ttl=60,           # steps a lost entity stays re-identifiable
    view_shift_min_gain=1.15,
    blocker_commit=2,
    vanish_max_area=64,
    prior_from_last_level=True,
)


class Comp:
    __slots__ = ("colour", "y0", "x0", "y1", "x1", "crop", "area", "cy", "cx", "key", "eid")

    def mask(self):
        m = np.zeros((64, 64), bool)
        m[self.y0:self.y1, self.x0:self.x1] = self.crop
        return m


class Entity:
    __slots__ = ("eid", "colour", "comp", "last", "born", "lost_at", "presses", "agree", "opp", "perp", "still",
                 "steps", "nonarrow_moves", "last_agree", "colours_hist", "prior")

    def __init__(self, eid, comp, t):
        self.eid = eid; self.colour = comp.colour; self.comp = comp; self.last = comp; self.born = t; self.lost_at = None
        self.presses = 0; self.agree = 0; self.opp = 0; self.perp = 0; self.still = 0
        self.steps = collections.Counter(); self.nonarrow_moves = 0; self.last_agree = -1
        self.colours_hist = collections.Counter(); self.prior = False

    def precision(self):
        n = self.agree + self.opp + self.perp
        return self.agree / n if n else 0.0


def segment(frame, bg):
    comps = []
    labmap = np.full(frame.shape, -1, np.int32)
    for c in np.unique(frame):
        if c == bg:
            continue
        lab, n = ndimage.label(frame == c, structure=CROSS)
        if n == 0:
            continue
        sls = ndimage.find_objects(lab)
        for i, s in enumerate(sls):
            crop = lab[s] == i + 1
            ys, xs = np.nonzero(crop)
            k = Comp()
            k.colour = int(c); k.y0, k.x0 = s[0].start, s[1].start; k.y1, k.x1 = s[0].stop, s[1].stop
            k.crop = crop; k.area = int(len(ys)); k.cy = float(ys.mean()) + k.y0; k.cx = float(xs.mean()) + k.x0
            k.key = (k.colour, crop.shape, crop.tobytes()); k.eid = None
            labmap[s][crop] = len(comps)
            comps.append(k)
    return comps, labmap


def _sign(v, eps=0.5):
    return 0 if abs(v) < eps else (1 if v > 0 else -1)


class Tracker:
    def __init__(self, **cfg):
        self.cfg = dict(DEFAULT_CFG, **cfg)
        self.prev_player_keys = None   # appearance of last level's player (colour, shape keys)
        self.prev_player_colours = None
        self.timings = []
        self._reset_state()

    # ------------------------------------------------------------------ level lifecycle
    def _reset_state(self):
        self.t = 0
        self.ents: dict[int, Entity] = {}
        self.next_id = 1
        self.comps = []; self.labmap = None; self.frame = None; self.bg = 0
        self.view_shift = (0, 0)
        self.selected = None; self.selected_tentative = False; self.switch_pending = False
        self.player_committed = None
        self.pair_co = collections.Counter(); self.pair_div = collections.Counter()
        self.blockers = collections.Counter()   # (player_eid, blocker_eid) -> count
        self.blocker_events = []                # for auditing: dicts
        self.vanish_pending = []                # dicts
        self.vanish_events = []                 # confirmed / withdrawn
        self.cycle = {}                         # selected-before-switch eid -> selected-after eid (SPACE order)
        self.last_switch = None                 # (action, eid before)
        self.moved_now = {}                     # eid -> (dy, dx) relative to the board this step
        self.events_now = []                    # human-readable new facts this step
        self.last_action = None
        self.sel_history = []

    def start_level(self, frame):
        # carry appearance of the committed player into the next level as a prior
        if self.player_committed is not None and self.player_committed in self.ents:
            grp = self.group(self.player_committed)
            self.prev_player_colours = {self.ents[e].colour for e in grp}
            self.prev_player_keys = {self.ents[e].last.key for e in grp}
        self._reset_state()
        vals, cnt = np.unique(frame, return_counts=True)
        self.bg = int(vals[cnt.argmax()])
        self.frame = frame
        self.comps, self.labmap = segment(frame, self.bg)
        for k in self.comps:
            self._new_entity(k)
        if self.cfg["prior_from_last_level"] and self.prev_player_keys:
            for e in self.ents.values():
                if e.comp.key in self.prev_player_keys:
                    e.prior = True

    def _new_entity(self, comp):
        e = Entity(self.next_id, comp, self.t); self.next_id += 1
        comp.eid = e.eid; self.ents[e.eid] = e
        return e

    # ------------------------------------------------------------------ view shift
    def _view_shift(self, prev, cur):
        votes = collections.Counter()
        byk = collections.defaultdict(list)
        for k in cur:
            byk[k.key].append(k)
        for p in prev:
            cs = byk.get(p.key)
            if cs and len(cs) <= 3:
                for c in cs:
                    d = (c.y0 - p.y0, c.x0 - p.x0)
                    if d != (0, 0):
                        votes[d] += p.area
        if not votes:
            return (0, 0)
        f0, f1 = self.prev_frame, self.frame
        fg = (f0 != self.bg)
        nfg = max(int(fg.sum()), 1)

        def agree(d):
            dy, dx = d
            a = f0[max(0, -dy):64 - max(0, dy), max(0, -dx):64 - max(0, dx)]
            b = f1[max(0, dy):64 - max(0, -dy), max(0, dx):64 - max(0, -dx)]
            m = a != self.bg
            return int(((a == b) & m).sum())

        base = agree((0, 0))
        best, bs = (0, 0), base
        for d, _ in votes.most_common(3):
            if abs(d[0]) > 24 or abs(d[1]) > 24:
                continue
            s = agree(d)
            if s > bs:
                best, bs = d, s
        if best != (0, 0) and bs >= self.cfg["view_shift_min_gain"] * base and bs >= 0.35 * nfg:
            return best
        return (0, 0)

    # ------------------------------------------------------------------ matching
    def _match(self, prev, cur):
        sy, sx = self.view_shift
        pairs = {}  # cur index -> prev comp
        used_p = set()
        # 1. exact, world-static
        loc = {}
        for i, p in enumerate(prev):
            loc[(p.key, p.y0 + sy, p.x0 + sx)] = i
        for j, c in enumerate(cur):
            i = loc.get((c.key, c.y0, c.x0))
            if i is not None and i not in used_p:
                pairs[j] = prev[i]; used_p.add(i)
        # 2. exact design with displacement
        rp = collections.defaultdict(list); rc = collections.defaultdict(list)
        for i, p in enumerate(prev):
            if i not in used_p:
                rp[p.key].append(i)
        for j, c in enumerate(cur):
            if j not in pairs:
                rc[c.key].append(j)
        pred = self._predicted_offsets()
        for key, js in rc.items():
            is_ = rp.get(key)
            if not is_:
                continue
            gate = self.cfg["max_exact_dist_small"] if prev[is_[0]].area <= 4 else self.cfg["max_exact_dist"]
            C = np.zeros((len(is_), len(js)))
            for a, i in enumerate(is_):
                p = prev[i]; oy, ox = pred.get(p.eid, (0, 0))
                for b, j in enumerate(js):
                    c = cur[j]
                    C[a, b] = np.hypot(c.cy - (p.cy + sy + oy), c.cx - (p.cx + sx + ox))
            ra, cb = linear_sum_assignment(C)
            for a, b in zip(ra, cb):
                if C[a, b] <= gate:
                    pairs[js[b]] = prev[is_[a]]; used_p.add(is_[a])
        # 3. cost-based on the rest
        P = [i for i in range(len(prev)) if i not in used_p]
        Cj = [j for j in range(len(cur)) if j not in pairs]
        if P and Cj:
            big = 1e6
            C = np.full((len(P), len(Cj)), big)
            for a, i in enumerate(P):
                p = prev[i]; oy, ox = pred.get(p.eid, (0, 0))
                for b, j in enumerate(Cj):
                    c = cur[j]
                    d = min(np.hypot(c.cy - (p.cy + sy), c.cx - (p.cx + sx)),
                            np.hypot(c.cy - (p.cy + sy + oy), c.cx - (p.cx + sx + ox)))
                    if d > self.cfg["max_cost_dist"]:
                        continue
                    ar = abs(np.log(c.area / p.area))
                    if ar > 1.2:
                        continue
                    hw = abs(np.log((c.y1 - c.y0) / (p.y1 - p.y0))) + abs(np.log((c.x1 - c.x0) / (p.x1 - p.x0)))
                    C[a, b] = d / self.cfg["max_cost_dist"] + (0.6 if c.colour != p.colour else 0.0) + 0.5 * ar + 0.25 * hw
            ra, cb = linear_sum_assignment(C)
            for a, b in zip(ra, cb):
                if C[a, b] <= self.cfg["cost_accept"]:
                    pairs[Cj[b]] = prev[P[a]]; used_p.add(P[a])
        return pairs, used_p

    def _predicted_offsets(self):
        """Expected displacement of the selected entity's group under the current arrow press (motion prior)."""
        out = {}
        a = self.last_action
        if a in ARROWS and self.selected is not None and self.selected in self.ents:
            e = self.ents[self.selected]
            step = e.steps.most_common(1)[0][0] if e.steps else 0
            if step:
                dy, dx = ARROWS[a]
                for g in self.group(self.selected):
                    out[g] = (dy * step, dx * step)
        return out

    # ------------------------------------------------------------------ main update
    def update(self, action, click, frame):
        t0 = time.perf_counter()
        self.t += 1
        self.last_action = action
        self.prev_frame = self.frame
        prev = self.comps
        self.frame = frame
        cur, labmap = segment(frame, self.bg)
        self.events_now = []
        self.moved_now = {}
        changed = bool((self.prev_frame != frame).any())
        if action == "RESET":
            self.view_shift = (0, 0)
        else:
            self.view_shift = self._view_shift(prev, cur) if changed else (0, 0)
        pairs, used_p = self._match(prev, cur)
        sy, sx = self.view_shift
        alive = set()
        disp = {}
        for j, c in enumerate(cur):
            p = pairs.get(j)
            if p is not None:
                e = self.ents[p.eid]
                c.eid = e.eid; e.comp = c; e.last = c; e.colour = c.colour; e.lost_at = None
                alive.add(e.eid)
                if p.key == c.key:
                    d = (c.y0 - p.y0 - sy, c.x0 - p.x0 - sx)
                elif (p.y1 - p.y0, p.x1 - p.x0) == (c.y1 - c.y0, c.x1 - c.x0):
                    d = (c.y0 - p.y0 - sy, c.x0 - p.x0 - sx)
                else:
                    d = (c.cy - p.cy - sy, c.cx - p.cx - sx)
                    d = (_sign(d[0], 0.75) * max(1, round(abs(d[0]))) if abs(d[0]) >= 0.75 else 0,
                         _sign(d[1], 0.75) * max(1, round(abs(d[1]))) if abs(d[1]) >= 0.75 else 0)
                disp[e.eid] = d
        # lost entities
        for i, p in enumerate(prev):
            if i not in used_p and p.eid in self.ents and p.eid not in alive:
                e = self.ents[p.eid]
                if e.lost_at is None:
                    e.lost_at = self.t
                    e.comp = None
        # re-identify unmatched current comps against recently lost entities
        lost = [e for e in self.ents.values() if e.comp is None and e.lost_at is not None and self.t - e.lost_at <= self.cfg["lost_ttl"]]
        for j, c in enumerate(cur):
            if c.eid is not None:
                continue
            best, bd = None, None
            for e in lost:
                if e.comp is not None:
                    continue
                L = e.last
                d = np.hypot(c.cy - L.cy - sy, c.cx - L.cx - sx)
                if L.key == c.key and d <= 24:
                    sc = d
                elif L.colour == c.colour and abs(np.log(c.area / L.area)) < 0.5 and d <= 8:
                    sc = d + 10
                else:
                    continue
                if bd is None or sc < bd:
                    best, bd = e, sc
            if best is not None:
                c.eid = best.eid; best.comp = c; best.last = c; best.lost_at = None
                alive.add(best.eid)
            else:
                self._new_entity(c)
        self.comps, self.labmap = cur, labmap
        for e in self.ents.values():
            if e.comp is not None:
                e.colours_hist[e.colour] += 1
        moved = {eid: d for eid, d in disp.items() if d != (0, 0)}
        self.moved_now = moved
        if action in ARROWS:
            self._arrow_evidence(action, disp, moved)
        else:
            self._nonarrow(action, click, moved, prev, pairs)
        self._co_motion(moved)
        self._vanish_check(prev, pairs, used_p)
        self._update_player()
        self.timings.append(time.perf_counter() - t0)
        return self.facts()

    # ------------------------------------------------------------------ evidence
    def _arrow_evidence(self, action, disp, moved):
        ady, adx = ARROWS[action]
        agreeing = []
        for eid, d in disp.items():
            e = self.ents[eid]
            e.presses += 1
            if d == (0, 0):
                e.still += 1
                continue
            along = d[0] * ady + d[1] * adx
            perp = abs(d[0] * adx) + abs(d[1] * ady)
            if along >= 1 and along >= perp:
                e.agree += 1; e.last_agree = self.t; e.steps[int(round(along))] += 1
                agreeing.append(eid)
            elif along <= -1 and -along >= perp:
                e.opp += 1
            else:
                e.perp += 1
        # selection update
        ctrl = self.controllable()
        sel = self.selected
        movers_ctrl = [e for e in agreeing if e in ctrl]
        if movers_ctrl:
            best = max(movers_ctrl, key=lambda e: (e in self.group(sel) if sel else False, self.ents[e].agree - 3 * self.ents[e].opp, self.ents[e].comp.area if self.ents[e].comp else 0))
            self._set_selected(best, False)
        elif agreeing and (self.switch_pending or sel is None or sel not in moved):
            # a not-yet-committed entity moved with the arrow while the selection did not: follow it
            cand = [e for e in agreeing if self.ents[e].precision() >= self.cfg["precision"]]
            if cand:
                best = max(cand, key=lambda e: (self.ents[e].prior, self.ents[e].agree, self.ents[e].comp.area if self.ents[e].comp else 0))
                if self.switch_pending or sel is None:
                    self._set_selected(best, True)
        if self.switch_pending and self.selected is not None and not self.selected_tentative and self.last_switch:
            before = self.last_switch[1]
            if before is not None and before != self.selected and self.last_switch[0] == "ACTION5":
                self.cycle[before] = self.selected
        if agreeing or movers_ctrl:
            self.switch_pending = False
        # blocker: confirmed selection, nothing moved with the arrow, no view shift
        sel = self.selected
        if sel is not None and not self.selected_tentative and not agreeing and self.view_shift == (0, 0) and sel in self.ents and self.ents[sel].comp is not None and sel in ctrl:
            self._blocker(sel, ady, adx)

    def _set_selected(self, eid, tentative):
        if eid != self.selected:
            self.sel_history.append((self.t, eid, tentative))
        self.selected = eid; self.selected_tentative = tentative

    def _nonarrow(self, action, click, moved, prev, pairs):
        for eid in moved:
            self.ents[eid].nonarrow_moves += 1
        if action not in ("ACTION5", "ACTION6", "ACTION7"):
            return
        ctrl = self.controllable()
        before = self.selected
        self.last_switch = (action, before)
        self.switch_pending = True
        guess = None
        # (a) click on/near a controllable entity
        if action == "ACTION6" and click:
            x, y = click
            if 0 <= x < 64 and 0 <= y < 64:
                hit = None
                j = self.labmap[y, x]
                if j >= 0:
                    hit = self.comps[j].eid
                if hit is None or not (set(self.group(hit)) & ctrl):
                    for eid in ctrl:
                        e = self.ents[eid]
                        if e.comp is not None and e.comp.y0 - 2 <= y < e.comp.y1 + 2 and e.comp.x0 - 2 <= x < e.comp.x1 + 2:
                            hit = eid; break
                if hit is not None:
                    gc = [g for g in self.group(hit) if g in ctrl]
                    if gc:
                        guess = gc[0]
        if guess is None and before is not None and before in self.ents:
            # (b) appearance swap: the controllable that gained a colour the old selection lost
            changed = self._appearance_changes(prev, pairs)
            lostc = changed.get(before, (set(), set()))[0] | set().union(*[changed.get(g, (set(), set()))[0] for g in self.group(before)])
            if lostc:
                for eid in ctrl:
                    if eid in self.group(before):
                        continue
                    gained = set().union(*[changed.get(g, (set(), set()))[1] for g in self.group(eid)])
                    if gained & lostc:
                        guess = eid; break
            # (c) a small entity jumped into a controllable's box (cursor)
            if guess is None:
                for eid, d in moved.items():
                    e = self.ents[eid]
                    if e.comp is None or e.comp.area > 9 or eid in ctrl or abs(d[0]) + abs(d[1]) < 3:
                        continue
                    for c in ctrl:
                        ce = self.ents[c]
                        if ce.comp is not None and c not in self.group(before) and ce.comp.y0 - 1 <= e.comp.cy <= ce.comp.y1 and ce.comp.x0 - 1 <= e.comp.cx <= ce.comp.x1:
                            guess = c; break
                    if guess is not None:
                        break
            # (d) learned SPACE cycle
            if guess is None and action == "ACTION5" and before in self.cycle and self.cycle[before] in self.ents:
                guess = self.cycle[before]
        if guess is not None and guess != before:
            self._set_selected(guess, True)

    def _appearance_changes(self, prev, pairs):
        """eid -> (colours lost, colours gained) between the previous and current frame, per group of touching comps."""
        out = {}
        pc = collections.defaultdict(set); cc = collections.defaultdict(set)
        # compare colour sets of each entity's neighbourhood: use bbox of the entity in both frames
        for eid in self.controllable() | ({self.selected} if self.selected else set()):
            e = self.ents.get(eid)
            if e is None or e.comp is None:
                continue
            k = e.comp
            y0, y1, x0, x1 = max(k.y0 - 1, 0), min(k.y1 + 1, 64), max(k.x0 - 1, 0), min(k.x1 + 1, 64)
            a = set(np.unique(self.prev_frame[y0:y1, x0:x1]).tolist()); b = set(np.unique(self.frame[y0:y1, x0:x1]).tolist())
            if a != b:
                out[eid] = (a - b, b - a)
        return out

    def _co_motion(self, moved):
        if len(moved) < 2:
            # any entity that moved alone diverges from its group partners
            for eid in moved:
                for g in self.group(eid):
                    if g != eid:
                        self.pair_div[frozenset((eid, g))] += 1
            return
        ids = [e for e in moved if self.ents[e].comp is not None]
        dil = {}
        for e in ids:
            k = self.ents[e].comp
            m = np.zeros((66, 66), bool)
            m[k.y0:k.y1 + 2, k.x0:k.x1 + 2] |= ndimage.binary_dilation(np.pad(k.crop, 1), EIGHT)
            dil[e] = m[1:65, 1:65]
        for a in ids:
            # groups partners that did not move with a diverge
            for g in self.group(a):
                if g != a and (g not in moved or moved[g] != moved[a]):
                    self.pair_div[frozenset((a, g))] += 1
            ka = self.ents[a].comp
            near = set(self.labmap[dil[a]].tolist()) - {-1}
            for j in near:
                b = self.comps[j].eid
                if b == a or b not in moved or moved[b] != moved[a]:
                    continue
                self.pair_co[frozenset((a, b))] += 1

    def group(self, eid):
        """Entities that touched and co-moved with eid at least twice and never moved apart (BFS)."""
        if eid is None:
            return []
        seen = {eid}; q = [eid]
        while q:
            a = q.pop()
            for pair, n in self.pair_co.items():
                if a in pair and n >= 2 and self.pair_div.get(pair, 0) == 0:
                    (b,) = pair - {a}
                    if b not in seen and b in self.ents and self.ents[b].comp is not None:
                        seen.add(b); q.append(b)
        return sorted(seen)

    # ------------------------------------------------------------------ roles
    def controllable(self):
        c = self.cfg
        out = set()
        for eid, e in self.ents.items():
            need = c["commit_agree"] - (1 if e.prior else 0)
            if e.agree >= max(1, need) and e.precision() >= c["precision"] and (c["presence_ratio"] <= 0 or e.agree / max(e.presses, 1) >= c["presence_ratio"]):
                out.add(eid)
        return out

    def _update_player(self):
        ctrl = self.controllable()
        if self.selected is not None and self.selected in ctrl:
            self.player_committed = self.selected
        elif self.player_committed is not None and self.player_committed not in ctrl:
            # demoted: fall back to the best remaining controllable
            self.player_committed = max(ctrl, key=lambda e: (self.ents[e].last_agree, self.ents[e].agree)) if ctrl else None
        elif self.player_committed is None and ctrl:
            self.player_committed = max(ctrl, key=lambda e: (self.ents[e].last_agree, self.ents[e].agree))

    def player(self):
        """(eid, committed) of the controllable entity most recently confirmed as selected."""
        return self.player_committed

    def mask(self, eid, group=True):
        m = np.zeros((64, 64), bool)
        if eid is None:
            return m
        for g in (self.group(eid) if group else [eid]):
            e = self.ents.get(g)
            if e is not None and e.comp is not None:
                k = e.comp
                m[k.y0:k.y1, k.x0:k.x1] |= k.crop
        return m

    # ------------------------------------------------------------------ blockers
    def _blocker(self, sel, ady, adx):
        e = self.ents[sel]
        if not e.steps:
            return
        step = e.steps.most_common(1)[0][0]
        pm = self.mask(sel)
        ys, xs = np.nonzero(pm)
        ty, tx = ys + ady * step, xs + adx * step
        inb = (ty >= 0) & (ty < 64) & (tx >= 0) & (tx < 64)
        if not inb.all():
            return  # board edge
        tgt = np.zeros((64, 64), bool); tgt[ty, tx] = True
        tgt &= ~pm
        js = self.labmap[tgt]
        js = js[js >= 0]
        if len(js) == 0:
            return
        own = set(self.group(sel))
        cnt = collections.Counter(self.comps[j].eid for j in js.tolist() if self.comps[j].eid not in own)
        if not cnt:
            return
        (b, n), = cnt.most_common(1)
        if n < 0.5 * sum(cnt.values()):
            return
        self.blockers[(sel, b)] += 1
        k = self.blockers[(sel, b)]
        ev = dict(t=self.t, player=sel, blocker=b, count=k, dir=(ady, adx), step=step,
                  player_mask=pm.copy(), blocker_mask=self.mask(b, group=False).copy(), committed=k >= self.cfg["blocker_commit"])
        self.blocker_events.append(ev)
        if k == self.cfg["blocker_commit"]:
            self.events_now.append(f"E{b} blocks E{sel}")

    # ------------------------------------------------------------------ vanish on overlap
    def _vanish_check(self, prev, pairs, used_p):
        sel = self.player_committed
        if sel is None or sel not in self.ents or self.ents[sel].comp is None:
            pm = None
        else:
            pm = self.mask(sel)
        own = set(self.group(sel)) if sel else set()
        # new pending: prev comps that vanished this step under the player
        if pm is not None:
            prevpm = np.zeros((64, 64), bool)
            for p in prev:
                if p.eid in own:
                    prevpm[p.y0:p.y1, p.x0:p.x1] |= p.crop
            ys, xs = np.nonzero(pm | prevpm)
            if len(ys):
                sweep = np.zeros((64, 64), bool); sweep[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
                for i, p in enumerate(prev):
                    if i in used_p or p.eid in own:
                        continue
                    e = self.ents.get(p.eid)
                    if e is None or e.comp is not None:
                        continue  # re-identified elsewhere
                    if p.area > self.cfg["vanish_max_area"]:
                        continue
                    fm = p.mask()
                    if (fm & pm).sum() >= 0.5 * p.area or (fm & sweep).sum() >= p.area:
                        self.vanish_pending.append(dict(t=self.t, eid=p.eid, player=sel, footprint=fm, colour=p.colour))
        # resolve pending
        still = []
        for v in self.vanish_pending:
            e = self.ents.get(v["eid"])
            fm = v["footprint"]
            if e is not None and e.comp is not None:
                v["status"] = "withdrawn"; v["t_res"] = self.t; self.vanish_events.append(v); continue
            covered = pm is not None and (fm & pm).any()
            if covered:
                if self.t - v["t"] > 40:
                    v["status"] = "expired"; v["t_res"] = self.t; self.vanish_events.append(v); continue
                still.append(v); continue
            # player has left: does the footprint show the old colour again?
            if (self.frame[fm] == v["colour"]).mean() >= 0.5:
                v["status"] = "withdrawn"; v["t_res"] = self.t; self.vanish_events.append(v); continue
            v["status"] = "confirmed"; v["t_res"] = self.t; self.vanish_events.append(v)
            self.events_now.append(f"E{v['eid']} vanished when E{v['player']} overlapped it (step {v['t']})")
        self.vanish_pending = still

    # ------------------------------------------------------------------ output
    def facts(self):
        """Result-dict fields for the action's tool result."""
        out = {}
        ctrl = self.controllable()
        p = self.player_committed
        if p is not None and p in self.ents:
            e = self.ents[p]
            out["player"] = f"E{p}"
            out["player_evidence"] = f"{e.agree}/{e.presses} presses moved it that way"
            if e.comp is not None:
                k = e.comp
                out["player_box"] = [k.y0, k.x0, k.y1 - 1, k.x1 - 1]
        if self.selected is not None and (self.selected != p or self.selected_tentative):
            out["selected"] = f"E{self.selected}" + ("?" if self.selected_tentative else "")
        others = sorted(c for c in ctrl if c not in self.group(p))
        if others:
            out["also_controllable"] = [f"E{c}" for c in others[:3]]
        mv = {f"E{e}": list(d) for e, d in self.moved_now.items() if e in ctrl or e == p}
        if mv:
            out["moved"] = mv
        if self.view_shift != (0, 0):
            out["view_shift"] = list(self.view_shift)
        bl = [f"E{b} blocks E{a} ({n}x)" for (a, b), n in self.blockers.items() if n >= self.cfg["blocker_commit"]]
        if bl:
            out["blockers"] = bl[-3:]
        vn = [f"E{v['eid']} vanished under E{v['player']} (step {v['t']})" for v in self.vanish_events if v.get("status") == "confirmed"]
        if vn:
            out["vanished"] = vn[-3:]
        return out

    def tail_line(self):
        """One short line for the end of the tool result (<= 40 tokens), new information first."""
        parts = []
        p = self.player_committed
        if p is not None and p in self.ents and self.ents[p].comp is not None:
            k = self.ents[p].comp
            d = self.moved_now.get(p)
            mv = f" moved {d[0]:+d},{d[1]:+d}" if d else (" did not move" if self.last_action in ARROWS else "")
            parts.append(f"you=E{p} r{k.y0}c{k.x0}{mv}")
        if self.selected is not None and self.selected_tentative:
            k = self.ents[self.selected].comp if self.selected in self.ents else None
            parts.append(f"selected now E{self.selected}?" + (f" r{k.y0}c{k.x0}" if k else ""))
        parts.extend(self.events_now[:2])
        return ("Tracker: " + "; ".join(parts)) if parts else ""
