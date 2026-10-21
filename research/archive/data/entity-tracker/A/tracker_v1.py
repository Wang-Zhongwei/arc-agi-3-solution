"""Design A: evidence-accumulating per-entity ledger for ARC-AGI-3 frames (host-side, numpy/scipy only).

The harness calls `start_level(frame)` at each level start and `update(action, click, frame)` after every executed
action; `facts()` gives result-dict fields for that action's tool result and `tail_line()` one short line.

Pipeline per update:
  1. Segment the frame into plain 4-connected colour components (background = the level's modal colour).
  2. Estimate the global view shift (translation voted by identical components, verified by pixel agreement) and
     remove it, so a scrolling board reads as static and a camera-centred player reads as moving.
  3. Match components to entities: exact & world-static; exact design with displacement (Hungarian per design, with
     the selected entity's expected step as a motion prior); cost-based Hungarian (position/colour/shape);
     template re-detection of a lost controllable inside a merged same-colour component; re-identification of
     recently lost entities; otherwise a new entity.
  4. Ledger per entity on arrow presses: agree / opposite / perpendicular / still / jump counts and step sizes.
  5. Parts: touching entities with identical move histories (>= 2 moves) report as one object. A carried or pushed
     object has a different history (it stood still while the player moved), so it never joins the player.
  6. Roles, stated only as behaviour:
       controllable  agree >= 2 (1 with last level's player appearance) and agree/(agree+opp+perp) >= 0.6
       player        the visible controllable with the most net agreeing presses this level (sticky)
       selected      what responds to arrows now: follows the controllable that moved with the last press; after a
                     switch action (ACTION5/6/7) it is predicted from the click target, the entity that gained the
                     colour the old selection lost, a cursor that jumped into a controllable, or the learned SPACE
                     cycle (marked '?'), and confirmed by the next press
       blockers      the player pressed toward Y, nothing moved, Y lies in its shifted footprint (counted)
       vanished      Y disappeared under the player and did not reappear after the player left (confirmed)
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
    precision=0.6,         # agree / (agree + opposite + perpendicular) needed (jumps excluded)
    jump_factor=2.5,       # a move > jump_factor x modal step and >= 8 px is a jump (respawn/teleport/slide): neutral
    max_exact_dist=40.0,   # gate for exact-design matches with displacement (px)
    max_exact_dist_small=10.0,  # same, for designs of <= 4 px (ambiguous)
    max_cost_dist=14.0,    # gate for cost-based matches
    cost_accept=1.25,
    lost_ttl=60,           # steps a lost entity stays re-identifiable
    view_shift_min_gain=1.15,
    blocker_commit=2,
    vanish_max_area=64,
    prior_from_last_level=True,
    template_recover=True,
    part_min_moves=2,
)


class Comp:
    __slots__ = ("colour", "y0", "x0", "y1", "x1", "crop", "area", "cy", "cx", "key", "eid", "virtual")

    def mask(self):
        m = np.zeros((64, 64), bool)
        m[self.y0:self.y1, self.x0:self.x1] = self.crop
        return m


def make_comp(colour, y0, x0, crop):
    ys, xs = np.nonzero(crop)
    k = Comp()
    k.colour = int(colour); k.y0, k.x0 = int(y0), int(x0); k.y1, k.x1 = k.y0 + crop.shape[0], k.x0 + crop.shape[1]
    k.crop = crop; k.area = int(len(ys)); k.cy = float(ys.mean()) + k.y0; k.cx = float(xs.mean()) + k.x0
    k.key = (k.colour, crop.shape, crop.tobytes()); k.eid = None; k.virtual = False
    return k


class Entity:
    __slots__ = ("eid", "colour", "comp", "last", "born", "lost_at", "presses", "agree", "opp", "perp", "still",
                 "steps", "nonarrow_moves", "last_agree", "prior", "jump", "moves")

    def __init__(self, eid, comp, t):
        self.eid = eid; self.colour = comp.colour; self.comp = comp; self.last = comp; self.born = t; self.lost_at = None
        self.presses = 0; self.agree = 0; self.opp = 0; self.perp = 0; self.still = 0; self.jump = 0
        self.steps = collections.Counter(); self.nonarrow_moves = 0; self.last_agree = -1
        self.prior = False
        self.moves = {}  # t -> displacement, every action

    def precision(self):
        n = self.agree + self.opp + self.perp
        return self.agree / n if n else 0.0

    def net(self):
        return self.agree - self.opp - self.perp

    def step(self):
        return self.steps.most_common(1)[0][0] if self.steps else None


def segment(frame, bg):
    comps = []
    labmap = np.full(frame.shape, -1, np.int32)
    for c in np.unique(frame):
        if c == bg:
            continue
        lab, n = ndimage.label(frame == c, structure=CROSS)
        if n == 0:
            continue
        for i, s in enumerate(ndimage.find_objects(lab)):
            k = make_comp(c, s[0].start, s[1].start, lab[s] == i + 1)
            labmap[s][k.crop] = len(comps)
            comps.append(k)
    return comps, labmap


def _sign(v, eps=0.5):
    return 0 if abs(v) < eps else (1 if v > 0 else -1)


class Tracker:
    def __init__(self, **cfg):
        self.cfg = dict(DEFAULT_CFG, **cfg)
        self.prev_player_keys = None
        self.timings = []
        self._reset_state()

    # ------------------------------------------------------------------ level lifecycle
    def _reset_state(self):
        self.t = 0
        self.ents: dict[int, Entity] = {}
        self.next_id = 1
        self.comps = []; self.labmap = None; self.frame = None; self.prev_frame = None; self.bg = 0
        self.view_shift = (0, 0)
        self.selected = None; self.selected_tentative = False; self.switch_pending = False
        self.player_eid = None
        self.blockers = collections.Counter()
        self.blocker_events = []
        self.vanish_pending = []
        self.vanish_events = []
        self.cycle = {}
        self.last_switch = None
        self.moved_now = {}
        self.events_now = []
        self.last_action = None
        self.sel_history = []

    def start_level(self, frame):
        if self.player_eid is not None and self.player_eid in self.ents:
            self.prev_player_keys = {self.ents[e].last.key for e in self.parts(self.player_eid)}
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
        nfg = max(int((f0 != self.bg).sum()), 1)

        def agree(d):
            dy, dx = d
            a = f0[max(0, -dy):64 - max(0, dy), max(0, -dx):64 - max(0, dx)]
            b = f1[max(0, dy):64 - max(0, -dy), max(0, dx):64 - max(0, -dx)]
            return int(((a == b) & (a != self.bg)).sum())

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
    def _predicted_offsets(self):
        out = {}
        a = self.last_action
        if a in ARROWS and self.selected is not None and self.selected in self.ents:
            step = self.ents[self.selected].step()
            if step:
                dy, dx = ARROWS[a]
                for g in self.parts(self.selected):
                    out[g] = (dy * step, dx * step)
        return out

    def _match(self, prev, cur):
        sy, sx = self.view_shift
        pairs = {}
        used_p = set()
        loc = {(p.key, p.y0 + sy, p.x0 + sx): i for i, p in enumerate(prev)}
        for j, c in enumerate(cur):
            i = loc.get((c.key, c.y0, c.x0))
            if i is not None and i not in used_p:
                pairs[j] = prev[i]; used_p.add(i)
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
                    C[a, b] = min(np.hypot(c.cy - (p.cy + sy + oy), c.cx - (p.cx + sx + ox)), np.hypot(c.cy - (p.cy + sy), c.cx - (p.cx + sx)) + 0.5)
            ra, cb = linear_sum_assignment(C)
            for a, b in zip(ra, cb):
                if C[a, b] <= gate:
                    pairs[js[b]] = prev[is_[a]]; used_p.add(is_[a])
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

    def _template_recover(self, lost_now, cur):
        """A lost controllable whose exact shape now sits inside a larger same-colour component (it merged with a
        touching same-colour object): carve it out as a virtual component so its identity and evidence survive."""
        out = []
        ctrl = self.controllable() | ({self.selected} if self.selected else set())
        sy, sx = self.view_shift
        pred = self._predicted_offsets()
        for e in lost_now:
            if e.eid not in ctrl:
                continue
            L = e.last
            if L.area < 3:
                continue
            h, w = L.crop.shape
            oy, ox = pred.get(e.eid, (0, 0))
            best = None
            for c in cur:
                if c.colour != L.colour or c.area <= L.area or c.eid is None and False:
                    continue
                if c.y1 - c.y0 < h or c.x1 - c.x0 < w:
                    continue
                # candidate offsets near the expected location
                for y in range(c.y0, c.y1 - h + 1):
                    for x in range(c.x0, c.x1 - w + 1):
                        dist = min(abs(y - (L.y0 + sy)) + abs(x - (L.x0 + sx)), abs(y - (L.y0 + sy + oy)) + abs(x - (L.x0 + sx + ox)))
                        if dist > 30 or (best is not None and dist >= best[0]):
                            continue
                        win = self.frame[y:y + h, x:x + w] == L.colour
                        if np.array_equal(win, L.crop):
                            best = (dist, y, x, c)
            if best is not None:
                _, y, x, host = best
                v = make_comp(L.colour, y, x, L.crop.copy()); v.virtual = True; v.eid = e.eid
                out.append((e, v, host))
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
        changed = bool((self.prev_frame != frame).any())
        self.view_shift = (0, 0) if (action == "RESET" or not changed) else self._view_shift(prev, cur)
        pairs, used_p = self._match(prev, cur)
        sy, sx = self.view_shift
        alive = set()
        disp = {}

        def assign(e, c, p):
            c.eid = e.eid; e.comp = c; e.last = c; e.colour = c.colour; e.lost_at = None
            alive.add(e.eid)
            if p is None:
                return
            if p.key == c.key or (p.y1 - p.y0, p.x1 - p.x0) == (c.y1 - c.y0, c.x1 - c.x0):
                d = (c.y0 - p.y0 - sy, c.x0 - p.x0 - sx)
            else:
                d = (c.cy - p.cy - sy, c.cx - p.cx - sx)
                d = tuple(_sign(v, 0.75) * max(1, round(abs(v))) if abs(v) >= 0.75 else 0 for v in d)
            disp[e.eid] = d

        for j, c in enumerate(cur):
            p = pairs.get(j)
            if p is not None and p.eid in self.ents:
                assign(self.ents[p.eid], c, p)
        lost_now = []
        for i, p in enumerate(prev):
            if i not in used_p and p.eid in self.ents and p.eid not in alive:
                e = self.ents[p.eid]
                if e.lost_at is None:
                    e.lost_at = self.t; e.comp = None; lost_now.append(e)
        virtual = []
        if self.cfg["template_recover"] and lost_now:
            for e, v, host in self._template_recover(lost_now, cur):
                assign(e, v, e.last if e.last is not None else None)
                virtual.append(v)
        lost = [e for e in self.ents.values() if e.comp is None and e.lost_at is not None and self.t - e.lost_at <= self.cfg["lost_ttl"]]
        for c in cur:
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
                assign(best, c, None)
            else:
                self._new_entity(c)
        self.comps, self.labmap = cur + virtual, labmap
        moved = {eid: d for eid, d in disp.items() if d != (0, 0)}
        for eid, d in moved.items():
            self.ents[eid].moves[self.t] = d
        self.moved_now = moved
        if action in ARROWS:
            self._arrow_evidence(action, disp, moved)
        else:
            self._nonarrow(action, click, moved)
        self._vanish_check(prev, used_p)
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
            mag = abs(d[0]) + abs(d[1])
            step = e.step()
            if step is not None and mag >= 8 and mag > self.cfg["jump_factor"] * step:
                e.jump += 1
            elif along >= 1 and 2 * perp <= along:
                e.agree += 1; e.last_agree = self.t; e.steps[int(round(along))] += 1
                agreeing.append(eid)
            elif along <= -1 and 2 * perp <= -along:
                e.opp += 1
            else:
                e.perp += 1
        ctrl = self.controllable()
        sel = self.selected
        movers_ctrl = [e for e in agreeing if e in ctrl]
        if movers_ctrl:
            selparts = set(self.parts(sel)) if sel else set()
            best = max(movers_ctrl, key=lambda e: (e in selparts, self.ents[e].net(), self.ents[e].comp.area if self.ents[e].comp else 0))
            self._set_selected(best, False)
        elif agreeing and (self.switch_pending or sel is None or sel not in moved):
            cand = [e for e in agreeing if self.ents[e].opp + self.ents[e].perp == 0]
            if cand:
                best = max(cand, key=lambda e: (self.ents[e].prior, self.ents[e].agree, self.ents[e].comp.area if self.ents[e].comp else 0))
                self._set_selected(best, True)
        if self.switch_pending and agreeing:
            if self.last_switch and self.last_switch[0] == "ACTION5" and self.last_switch[1] is not None and self.last_switch[1] != self.selected:
                self.cycle[self.last_switch[1]] = self.selected
            self.switch_pending = False
        sel = self.selected
        if (sel is not None and not self.selected_tentative and not agreeing and self.view_shift == (0, 0)
                and sel in self.ents and self.ents[sel].comp is not None and sel in ctrl):
            self._blocker(sel, ady, adx)

    def _set_selected(self, eid, tentative):
        if eid != self.selected:
            self.sel_history.append((self.t, eid, tentative))
        self.selected = eid; self.selected_tentative = tentative

    def _nonarrow(self, action, click, moved):
        for eid in moved:
            self.ents[eid].nonarrow_moves += 1
        if action not in ("ACTION5", "ACTION6", "ACTION7"):
            return
        ctrl = self.controllable()
        before = self.selected
        self.last_switch = (action, before)
        self.switch_pending = True
        guess = None
        if action == "ACTION6" and click:
            x, y = click
            if 0 <= x < 64 and 0 <= y < 64:
                hit = None
                j = self.labmap[y, x]
                if j >= 0:
                    hit = self.comps[j].eid
                if hit is None or not (set(self.parts(hit)) & ctrl):
                    for eid in ctrl:
                        e = self.ents[eid]
                        if e.comp is not None and e.comp.y0 - 2 <= y < e.comp.y1 + 2 and e.comp.x0 - 2 <= x < e.comp.x1 + 2:
                            hit = eid; break
                if hit is not None:
                    gc = [g for g in self.parts(hit) if g in ctrl]
                    if gc:
                        guess = gc[0]
        if guess is None and before is not None and before in self.ents:
            changed = self._appearance_changes(ctrl | {before})
            bparts = set(self.parts(before))
            lostc = set().union(*[changed.get(g, (set(), set()))[0] for g in bparts]) if bparts else set()
            if lostc:
                for eid in ctrl:
                    if eid in bparts:
                        continue
                    gained = set().union(*[changed.get(g, (set(), set()))[1] for g in self.parts(eid)])
                    if gained & lostc:
                        guess = eid; break
            if guess is None:
                for eid, d in moved.items():
                    e = self.ents[eid]
                    if e.comp is None or e.comp.area > 9 or eid in ctrl or abs(d[0]) + abs(d[1]) < 3:
                        continue
                    for c in ctrl:
                        ce = self.ents[c]
                        if ce.comp is not None and c not in bparts and ce.comp.y0 - 1 <= e.comp.cy <= ce.comp.y1 and ce.comp.x0 - 1 <= e.comp.cx <= ce.comp.x1:
                            guess = c; break
                    if guess is not None:
                        break
            if guess is None and action == "ACTION5" and before in self.cycle and self.cycle[before] in self.ents:
                guess = self.cycle[before]
        if guess is not None and guess != before:
            self._set_selected(guess, True)

    def _appearance_changes(self, eids):
        out = {}
        for eid in eids:
            e = self.ents.get(eid)
            if e is None or e.comp is None:
                continue
            k = e.comp
            y0, y1, x0, x1 = max(k.y0 - 1, 0), min(k.y1 + 1, 64), max(k.x0 - 1, 0), min(k.x1 + 1, 64)
            a = set(np.unique(self.prev_frame[y0:y1, x0:x1]).tolist()); b = set(np.unique(self.frame[y0:y1, x0:x1]).tolist())
            if a != b:
                out[eid] = (a - b, b - a)
        return out

    def parts(self, eid):
        """eid plus touching entities with an identical move history (>= part_min_moves moves) over their common life."""
        if eid is None or eid not in self.ents:
            return [] if eid is None else [eid]
        e = self.ents[eid]
        if e.comp is None or len(e.moves) < self.cfg["part_min_moves"]:
            return [eid]
        k = e.comp
        y0, y1, x0, x1 = max(k.y0 - 1, 0), min(k.y1 + 1, 64), max(k.x0 - 1, 0), min(k.x1 + 1, 64)
        near = set(self.labmap[y0:y1, x0:x1].ravel().tolist()) - {-1}
        out = [eid]
        for j in near:
            if j >= len(self.comps):
                continue
            o = self.comps[j].eid
            if o is None or o == eid or o not in self.ents:
                continue
            oe = self.ents[o]
            since = max(oe.born, e.born)
            ma = {t: d for t, d in e.moves.items() if t > since}
            mb = {t: d for t, d in oe.moves.items() if t > since}
            if len(ma) >= self.cfg["part_min_moves"] and ma == mb:
                out.append(o)
        return sorted(out)

    # ------------------------------------------------------------------ roles
    def controllable(self):
        c = self.cfg
        out = set()
        for eid, e in self.ents.items():
            need = c["commit_agree"] - (1 if e.prior else 0)
            if e.agree >= max(1, need) and e.precision() >= c["precision"]:
                out.add(eid)
        return out

    def _update_player(self):
        ctrl = [e for e in self.controllable() if self.ents[e].comp is not None]
        if not ctrl:
            if self.player_eid is not None and (self.player_eid not in self.ents or self.ents[self.player_eid].precision() < self.cfg["precision"]):
                self.player_eid = None
            return
        cur = self.player_eid
        best = max(ctrl, key=lambda e: (self.ents[e].net(), self.ents[e].last_agree))
        if cur in ctrl and self.ents[cur].net() >= self.ents[best].net():
            return
        self.player_eid = best

    def player(self):
        return self.player_eid

    def mask(self, eid, group=True):
        m = np.zeros((64, 64), bool)
        if eid is None:
            return m
        for g in (self.parts(eid) if group else [eid]):
            e = self.ents.get(g)
            if e is not None and e.comp is not None:
                k = e.comp
                m[k.y0:k.y1, k.x0:k.x1] |= k.crop
        return m

    # ------------------------------------------------------------------ blockers
    def _blocker(self, sel, ady, adx):
        e = self.ents[sel]
        step = e.step()
        if not step:
            return
        pm = self.mask(sel)
        ys, xs = np.nonzero(pm)
        ty, tx = ys + ady * step, xs + adx * step
        if not (((ty >= 0) & (ty < 64) & (tx >= 0) & (tx < 64)).all()):
            return  # board edge
        tgt = np.zeros((64, 64), bool); tgt[ty, tx] = True
        tgt &= ~pm
        js = self.labmap[tgt]
        js = js[js >= 0]
        if len(js) == 0:
            return
        own = set(self.parts(sel))
        cnt = collections.Counter(self.comps[j].eid for j in js.tolist() if self.comps[j].eid not in own)
        if not cnt:
            return
        (b, n), = cnt.most_common(1)
        if n < 0.5 * sum(cnt.values()):
            return
        self.blockers[(sel, b)] += 1
        k = self.blockers[(sel, b)]
        self.blocker_events.append(dict(t=self.t, player=sel, blocker=b, count=k, dir=(ady, adx), step=step,
                                        player_mask=pm.copy(), blocker_mask=self.mask(b, group=False).copy(),
                                        committed=k >= self.cfg["blocker_commit"]))
        if k == self.cfg["blocker_commit"]:
            self.events_now.append(f"E{b} blocks E{sel}")

    # ------------------------------------------------------------------ vanish on overlap
    def _vanish_check(self, prev, used_p):
        sel = self.selected if (self.selected is not None and not self.selected_tentative) else self.player_eid
        pm = self.mask(sel) if (sel is not None and sel in self.ents and self.ents[sel].comp is not None) else None
        own = set(self.parts(sel)) if sel else set()
        if pm is not None and pm.any():
            prevpm = np.zeros((64, 64), bool)
            for p in prev:
                if p.eid in own:
                    prevpm[p.y0:p.y1, p.x0:p.x1] |= p.crop
            ys, xs = np.nonzero(pm | prevpm)
            sweep = np.zeros((64, 64), bool); sweep[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
            for i, p in enumerate(prev):
                if i in used_p or p.eid in own:
                    continue
                e = self.ents.get(p.eid)
                if e is None or e.comp is not None or p.area > self.cfg["vanish_max_area"]:
                    continue
                fm = p.mask()
                if (fm & pm).sum() >= 0.5 * p.area or (fm & sweep).sum() >= p.area:
                    self.vanish_pending.append(dict(t=self.t, eid=p.eid, player=sel, footprint=fm, colour=p.colour))
        still = []
        for v in self.vanish_pending:
            e = self.ents.get(v["eid"])
            fm = v["footprint"]
            if e is not None and e.comp is not None:
                v["status"] = "withdrawn"; v["t_res"] = self.t; self.vanish_events.append(v); continue
            if pm is not None and (fm & pm).any():
                if self.t - v["t"] > 40:
                    v["status"] = "expired"; v["t_res"] = self.t; self.vanish_events.append(v); continue
                still.append(v); continue
            if (self.frame[fm] == v["colour"]).mean() >= 0.5:
                v["status"] = "withdrawn"; v["t_res"] = self.t; self.vanish_events.append(v); continue
            v["status"] = "confirmed"; v["t_res"] = self.t; self.vanish_events.append(v)
            self.events_now.append(f"E{v['eid']} vanished when E{v['player']} overlapped it (step {v['t']})")
        self.vanish_pending = still

    # ------------------------------------------------------------------ output
    def facts(self):
        """Result-dict fields for the action's tool result (compact, evidence-counted)."""
        out = {}
        p = self.player_eid
        if p is not None and p in self.ents:
            e = self.ents[p]
            out["player"] = f"E{p}"
            out["player_evidence"] = f"{e.agree}/{e.presses} presses moved it that way"
            if e.comp is not None:
                k = e.comp
                out["player_box"] = [k.y0, k.x0, k.y1 - 1, k.x1 - 1]
        s = self.selected
        if s is not None and (s != p or self.selected_tentative):
            out["selected"] = f"E{s}" + ("?" if self.selected_tentative else "")
            if s in self.ents and self.ents[s].comp is not None:
                k = self.ents[s].comp
                out["selected_box"] = [k.y0, k.x0, k.y1 - 1, k.x1 - 1]
        ctrl = self.controllable()
        pparts = set(self.parts(p))
        others = sorted(c for c in ctrl if c not in pparts and c != s and self.ents[c].comp is not None)
        if others:
            out["also_controllable"] = [f"E{c}" for c in others[:3]]
        mv = {f"E{e}": list(d) for e, d in self.moved_now.items() if e in ctrl or e == p or e == s}
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
        s = self.selected if self.selected is not None else self.player_eid
        if s is not None and s in self.ents and self.ents[s].comp is not None:
            k = self.ents[s].comp
            d = self.moved_now.get(s)
            mv = f" moved {d[0]:+d},{d[1]:+d}" if d else (" did not move" if self.last_action in ARROWS else "")
            tag = "selected" if s != self.player_eid else "you"
            parts.append(f"{tag}=E{s}{'?' if self.selected_tentative else ''} r{k.y0}c{k.x0}{mv}")
        parts.extend(self.events_now[:2])
        return ("Tracker: " + "; ".join(parts)) if parts else ""
