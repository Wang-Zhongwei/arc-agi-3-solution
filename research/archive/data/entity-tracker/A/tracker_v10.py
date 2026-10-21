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
import json
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
    carry_cap=6,
    carry_walkable=True,
    click_select_changed=True,  # click on an entity whose appearance changed -> tentative selection
    fields_char_cap=160,    # result-dict fields budget (~2.1 chars/token -> about 78 tokens); low priority drops first    # colours the player walked over stay walkable in later levels            # max net presses carried to the entity that looks like last level's player
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
                 "steps", "nonarrow_moves", "last_agree", "prior", "jump", "moves", "carry", "agree_steps")

    def __init__(self, eid, comp, t):
        self.eid = eid; self.colour = comp.colour; self.comp = comp; self.last = comp; self.born = t; self.lost_at = None
        self.presses = 0; self.agree = 0; self.opp = 0; self.perp = 0; self.still = 0; self.jump = 0
        self.steps = collections.Counter(); self.nonarrow_moves = 0; self.last_agree = -1
        self.prior = False; self.carry = 0; self.agree_steps = set()
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
        self.prev_player_carry = 0
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
        self.walkable = set()   # colours revealed where the player/selection has been
        self._last_said = {}

    def start_level(self, frame):
        walk = set(self.walkable) if self.cfg["carry_walkable"] else set()
        if self.player_eid is not None and self.player_eid in self.ents:
            pe = self.ents[self.player_eid]
            self.prev_player_keys = {self.ents[e].last.key for e in self.parts(self.player_eid)}
            self.prev_player_carry = max(0, min(self.cfg["carry_cap"], pe.net() + pe.carry))
        self._reset_state()
        vals, cnt = np.unique(frame, return_counts=True)
        self.bg = int(vals[cnt.argmax()])
        self.frame = frame
        self.comps, self.labmap = segment(frame, self.bg)
        for k in self.comps:
            self._new_entity(k)
        self.first_comps = list(self.comps)
        self.walkable = walk - {self.bg}
        if self.cfg["prior_from_last_level"] and self.prev_player_keys:
            exact = [e for e in self.ents.values() if e.comp.key in self.prev_player_keys]
            if not exact:  # colour toggles (selection highlight): same shape, any colour, if rare
                shapes = {(k[1], k[2]) for k in self.prev_player_keys}
                exact = [e for e in self.ents.values() if (e.comp.key[1], e.comp.key[2]) in shapes and e.comp.area >= 4]
                if len(exact) > 3:
                    exact = []
            for e in exact:
                e.prior = True; e.carry = self.prev_player_carry

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
        if action == "RESET":
            # the level restarts from its initial layout: match against the level's first frame, ids persist
            for e in self.ents.values():
                e.comp = None
            for k in self.first_comps:
                if k.eid in self.ents:
                    self.ents[k.eid].comp = k
            prev = [k for k in self.first_comps if k.eid in self.ents]
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
        # unique design re-identification at any distance (teleport, respawn, RESET)
        lk = collections.Counter(e.last.key for e in lost)
        ck = collections.Counter(c.key for c in cur if c.eid is None)
        for c in cur:
            if c.eid is None and ck[c.key] == 1 and lk[c.key] == 1 and c.area >= 3:
                e = next(e for e in lost if e.last.key == c.key)
                if e.comp is None:
                    assign(e, c, None)
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
        if action in ARROWS:
            self._transfer_identity(disp, ARROWS[action])
        moved = {eid: d for eid, d in disp.items() if d != (0, 0)} if action != "RESET" else {}
        for eid, d in moved.items():
            self.ents[eid].moves[self.t] = d
        self.moved_now = moved
        for w in {self.selected, self.player_eid}:
            if w is not None and w in moved and w in self.ents:
                for g in self.parts(w):
                    pc = next((p for p in prev if p.eid == g), None)
                    if pc is not None:
                        m = pc.mask() & ~self.mask(w)
                        self.walkable |= set(np.unique(frame[m]).tolist()) - {self.bg}
        if action in ARROWS:
            self._arrow_evidence(action, disp, moved)
        else:
            self._nonarrow(action, click, moved)
        self._vanish_check(prev, used_p)
        self._update_player()
        self.timings.append(time.perf_counter() - t0)
        return self.facts()

    def _transfer_identity(self, disp, arrow):
        """A newborn entity (no evidence yet) that moves with the arrow and has the exact design of a lost
        controllable is that controllable (respawn / re-creation): move the comp and its motion to the old id."""
        ctrl = self.controllable()
        lost = {}
        for eid in ctrl:
            e = self.ents[eid]
            if e.comp is None:
                lost.setdefault(e.last.key, []).append(e)
        if not lost:
            return
        ady, adx = arrow
        for eid in list(disp):
            n = self.ents.get(eid)
            if n is None or n.presses > 0 or n.agree > 0 or n.comp is None:
                continue
            d = disp[eid]
            along = d[0] * ady + d[1] * adx
            if along < 1:
                continue
            cands = lost.get(n.comp.key)
            if not cands:
                # same colour, similar size (a rotated / re-drawn sprite), only if unambiguous
                cands = [e for es in lost.values() for e in es if e.last.colour == n.comp.colour and abs(np.log(n.comp.area / e.last.area)) < 0.3]
                movers = [x for x in disp if x in self.ents and self.ents[x].presses == 0 and self.ents[x].comp is not None
                          and self.ents[x].comp.colour == n.comp.colour and disp[x][0] * ady + disp[x][1] * adx >= 1]
                if len(cands) != 1 or len(movers) != 1:
                    continue
            o = max(cands, key=lambda e: e.agree)
            if o.comp is not None:
                continue
            c = n.comp
            c.eid = o.eid; o.comp = c; o.last = c; o.colour = c.colour; o.lost_at = None
            disp[o.eid] = disp.pop(eid)
            o.moves.update(n.moves)
            del self.ents[eid]
            if o in cands:
                cands.remove(o)

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
                e.agree_steps.add(self.t)
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
        before = self.selected
        ctrl = {c for c in self.controllable() if c in self.parts(before) or self.independent(c, before)}
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
                    elif self.cfg["click_select_changed"] and hit in self.ents and self.ents[hit].comp is not None:
                        # a click on an object that changed appearance (highlight / marker) likely selected it
                        k = self.ents[hit].comp
                        if k.area <= 256:
                            y0, y1, x0, x1 = max(k.y0 - 1, 0), min(k.y1 + 1, 64), max(k.x0 - 1, 0), min(k.x1 + 1, 64)
                            if (self.prev_frame[y0:y1, x0:x1] != self.frame[y0:y1, x0:x1]).any():
                                guess = self.rep(hit)
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

    def independent(self, c, sel):
        """c has moved with an arrow on a press where sel (and its parts) did not move: separately controllable."""
        if sel is None or c not in self.ents:
            return True
        sm = set()
        for g in self.parts(sel):
            if g in self.ents:
                sm |= set(self.ents[g].moves)
        return any(t not in sm for t in self.ents[c].agree_steps)

    def companions(self, eid):
        """parts(eid) plus touching entities that moved on >= half of eid's moves (an attached marker or eye that
        rotates, so its displacement differs): never a blocker or a vanished object of eid."""
        out = set(self.parts(eid))
        e = self.ents.get(eid)
        if e is None or e.comp is None or len(e.moves) < 2:
            return out
        k = e.comp
        y0, y1, x0, x1 = max(k.y0 - 2, 0), min(k.y1 + 2, 64), max(k.x0 - 2, 0), min(k.x1 + 2, 64)
        near = set(self.labmap[y0:y1, x0:x1].ravel().tolist()) - {-1}
        for j in near:
            if j >= len(self.comps):
                continue
            o = self.comps[j].eid
            if o is None or o in out or o not in self.ents:
                continue
            oe = self.ents[o]
            mine = [t for t in e.moves if t > oe.born]
            if len(mine) >= 2 and sum(1 for t in mine if t in oe.moves) >= 0.5 * len(mine):
                out.add(o)
        return out

    # ------------------------------------------------------------------ roles
    def controllable(self):
        c = self.cfg
        out = set()
        for eid, e in self.ents.items():
            if e.prior and e.carry >= c["commit_agree"] and e.opp + e.perp <= e.agree:
                out.add(eid); continue
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
        def score(e):
            x = self.ents[e]
            return x.net() + x.carry
        best = max(ctrl, key=lambda e: (score(e), self.ents[e].last_agree))
        if cur in ctrl and score(cur) >= score(best):
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
        own = self.companions(sel)
        cnt = None
        for k in sorted({1, step}):   # contact first, then the full step
            ty, tx = ys + ady * k, xs + adx * k
            if not (((ty >= 0) & (ty < 64) & (tx >= 0) & (tx < 64)).all()):
                return  # board edge
            tgt = np.zeros((64, 64), bool); tgt[ty, tx] = True
            tgt &= ~pm
            js = self.labmap[tgt]
            js = js[js >= 0]
            c = collections.Counter(self.comps[j].eid for j in js.tolist()
                                    if self.comps[j].eid not in own and self.comps[j].colour not in self.walkable
                                    and self.comps[j].eid in self.ents and self.t - self.ents[self.comps[j].eid].born >= 2)
            if c:
                cnt = c; break
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
            self.events_now.append(f"E{self.rep(b)} blocks E{self.rep(sel)}")

    # ------------------------------------------------------------------ vanish on overlap
    def _vanish_check(self, prev, used_p):
        sel = self.selected if (self.selected is not None and not self.selected_tentative) else self.player_eid
        pm = self.mask(sel) if (sel is not None and sel in self.ents and self.ents[sel].comp is not None) else None
        own = self.companions(sel) if sel else set()
        if pm is not None and pm.any():
            prevpm = np.zeros((64, 64), bool)
            for p in prev:
                if p.eid in own:
                    prevpm[p.y0:p.y1, p.x0:p.x1] |= p.crop
            sweep = np.zeros((64, 64), bool)
            d = self.moved_now.get(sel)
            st = self.ents[sel].step() if sel in self.ents else None
            if d is not None and st and (d[0] == 0 or d[1] == 0) and abs(d[0]) + abs(d[1]) <= 1.5 * st:
                ys, xs = np.nonzero(pm | prevpm)
                sweep[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
            for i, p in enumerate(prev):
                if i in used_p or p.eid in own:
                    continue
                e = self.ents.get(p.eid)
                if e is None or e.comp is not None or p.area > self.cfg["vanish_max_area"]:
                    continue
                fm = p.mask()
                if (fm & pm).sum() >= 0.5 * p.area or (fm & sweep).sum() >= p.area:
                    self.vanish_pending.append(dict(t=self.t, eid=p.eid, player=sel, footprint=fm, colour=p.colour, player_mask=pm.copy()))
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
            ring = ndimage.binary_dilation(fm, EIGHT) & ~fm
            if ring.any():
                rv, rc = np.unique(self.frame[ring], return_counts=True)
                if (self.frame[fm] == rv[rc.argmax()]).mean() < 0.8:
                    v["status"] = "recoloured"; v["t_res"] = self.t; self.vanish_events.append(v); continue
            v["status"] = "confirmed"; v["t_res"] = self.t; self.vanish_events.append(v)
            self.events_now.append(f"E{v['eid']} vanished under E{self.rep(v['player'])}")
        self.vanish_pending = still

    # ------------------------------------------------------------------ output
    def rep(self, eid):
        """Representative id of an object: its largest visible part (stable names for multi-colour sprites)."""
        if eid is None:
            return None
        ps = [g for g in self.parts(eid) if g in self.ents and self.ents[g].comp is not None]
        return max(ps, key=lambda g: (self.ents[g].comp.area, -g)) if ps else eid

    def _where(self, eid):
        e = self.ents.get(eid)
        if e is None or e.comp is None:
            return "not visible"
        m = self.mask(eid); ys, xs = np.nonzero(m)
        return f"r{ys.min()}c{xs.min()} {ys.max() - ys.min() + 1}x{xs.max() - xs.min() + 1}"

    def facts(self):
        """Result-dict fields for the action's tool result: compact, evidence-counted, priority-capped."""
        out = {}
        p = self.player_eid
        if p is not None and p in self.ents:
            e = self.ents[p]
            out["player"] = f"E{self.rep(p)} {self._where(p).split(' ')[0]} {e.agree}/{e.presses} presses"
        s = self.selected
        if s is not None and (s not in self.parts(p) or self.selected_tentative):
            out["selected"] = f"E{self.rep(s)}{'?' if self.selected_tentative else ''} {self._where(s).split(' ')[0]}"
        w = s if s is not None else p
        d = self.moved_now.get(w) if w is not None else None
        if d is not None:
            out["moved"] = f"E{self.rep(w)} {d[0]:+d},{d[1]:+d}"
        bl = [f"E{self.rep(b)} blocks E{self.rep(a)} x{n}" for (a, b), n in self.blockers.items() if n >= self.cfg["blocker_commit"]]
        if bl:
            out["blockers"] = bl[-2:]
        vn = [f"E{v['eid']} gone under E{self.rep(v['player'])} s{v['t']}" for v in self.vanish_events if v.get("status") == "confirmed"]
        if vn:
            out["vanished"] = vn[-2:]
        if self.view_shift != (0, 0):
            out["view_shift"] = f"{self.view_shift[0]:+d},{self.view_shift[1]:+d}"
        ctrl = self.controllable()
        skip = set(self.parts(p)) | set(self.parts(s))
        others = sorted({self.rep(c) for c in ctrl if c not in skip and self.ents[c].comp is not None and self.independent(c, p)})
        if others:
            out["also_controllable"] = " ".join(f"E{c}" for c in others[:2])
        cap = self.cfg["fields_char_cap"]
        for key in ("also_controllable", "view_shift", "vanished", "blockers"):
            if len(json.dumps(out, separators=(",", ":"))) <= cap:
                break
            if isinstance(out.get(key), list) and len(out[key]) > 1:
                out[key] = out[key][-1:]
                if len(json.dumps(out, separators=(",", ":"))) <= cap:
                    break
            out.pop(key, None)
        return out

    def tail_line(self):
        """One short line (<= 40 tokens) only when something new happened: player committed/changed, selection
        changed, a blocker committed, a vanish confirmed."""
        parts = []
        p, s = self.player_eid, self.selected
        if p is not None and p != self._last_said.get("player"):
            e = self.ents.get(p)
            if e is not None:
                parts.append(f"you=E{self.rep(p)} (moved with {e.agree}/{e.presses} presses)")
            self._last_said["player"] = p
        key = (s, self.selected_tentative)
        if s is not None and key != self._last_said.get("selected") and s not in self.parts(p):
            parts.append(f"selected now E{self.rep(s)}{'?' if self.selected_tentative else ''} {self._where(s).split(' ')[0]}")
        self._last_said["selected"] = key
        parts.extend(self.events_now[:1])
        return ("Tracker: " + "; ".join(parts)) if parts else ""
