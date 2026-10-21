"""Score a tracker on the GT replays with the roles_eval.py player metric plus mechanism metrics.

Usage: python eval_tracker.py {A|baseline} {dev|heldout|<g1,g2>} [--out results/x.json] [--cfg key=val ...]

Per level (eligibility as roles_eval.py: >=3 frames, a GT-controlled sprite exists and moved under an arrow):
  ok[t]       committed player claim's pixels lie >=50% on GT controlled sprites at t (False if no claim / GT absent)
  commit      first action index after which ok stays True to the level end (1-based); within5 = commit <= 5
  final_ok    ok at the last action of the level
Claims:   wrong committed claims = steps with a committed claim and GT controlled visible where ok is False.
Switches: GT selection switch = a non-arrow action between two arrow presses that moved different GT-controlled
          instances; selected-object accuracy measured right after the switch action and after the next press.
Blockers: claim 'Y blocks X' correct if GT X did not move on that press and Y's GT instance lies in X's shifted footprint.
Vanish:   confirmed 'Y vanished when X overlapped it' correct if Y's GT instance no longer occupies its footprint and a
          GT-controlled sprite overlapped/swept the footprint on the vanish step.
"""
import argparse, collections, json, os, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from util import load, SPLIT, DIRS, ROLES  # noqa: E402
import importlib, os
TA = importlib.import_module(os.environ.get('TRACKER_MODULE', 'tracker'))  # noqa: E402

sys.path.insert(0, str(Path("/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/vision/bench")))
ARROW = set(DIRS)


# ---------------------------------------------------------------------------------------------- baseline (roles_eval rule)
class Baseline:
    """roles_eval.py's rule on colour components, causal, re-implemented step by step (same names, thresholds)."""

    def __init__(self):
        import roles_eval as R
        self.R = R
        self.timings = []

    def start_level(self, frame):
        self.stats = collections.defaultdict(collections.Counter)
        self.frame = frame
        self.lab = self.R.seg_cc(frame); self.segs = self.R.name_segments(frame, self.lab)
        self.best = None

    def update(self, act, click, frame):
        t0 = time.perf_counter()
        R = self.R
        lab = R.seg_cc(frame); segs = R.name_segments(frame, lab)
        if act in DIRS:
            ex, ey = DIRS[act]
            byname = collections.defaultdict(list)
            for j, (nm, cy, cx, bb) in segs.items():
                byname[nm].append((cy, cx))
            seen = set()
            for j, (nm, cy, cx, bb) in self.segs.items():
                if nm in seen:
                    continue
                seen.add(nm)
                st = self.stats[nm]; st["present"] += 1
                cands = byname.get(nm)
                if not cands:
                    continue
                dy, dx = min(((y - cy, x - cx) for y, x in cands), key=lambda d: abs(d[0]) + abs(d[1]))
                if abs(dy) + abs(dx) > 0.5:
                    st["moved"] += 1
                    if (np.sign(round(dx)), np.sign(round(dy))) == (ex, ey):
                        st["agree"] += 1
        best = None; bs = None
        for nm, st in self.stats.items():
            if st["agree"] >= 2 and st["moved"] / max(st["present"], 1) > 0.3 and st["agree"] / max(st["moved"], 1) > 0.6:
                if bs is None or st["agree"] > bs:
                    best, bs = nm, st["agree"]
        self.best = best; self.lab = lab; self.segs = segs
        self.timings.append(time.perf_counter() - t0)

    def player_mask(self):
        if self.best is None:
            return None
        js = [j for j, v in self.segs.items() if v[0] == self.best]
        if not js:
            return np.zeros((64, 64), bool)
        return np.isin(self.lab, js)

    def selected_mask(self):
        return self.player_mask()


class AWrap:
    def __init__(self, **cfg):
        self.T = TA.Tracker(**cfg)
        self.total = []
        self.timings = self.total
        self.lines = []
        self.blocker_log = []; self.vanish_log = []; self.i0 = 0

    def _flush(self):
        for ev in self.T.blocker_events:
            self.blocker_log.append(dict(ev, t=ev["t"] + self.i0))
        for v in self.T.vanish_events + [dict(v, status="pending") for v in self.T.vanish_pending]:
            self.vanish_log.append(dict(v, t=v["t"] + self.i0, t_res=v.get("t_res", v["t"]) + self.i0))
        self.T.blocker_events = []; self.T.vanish_events = []; self.T.vanish_pending = []

    def start_level(self, frame, i0=0):
        self._flush()
        self.T.start_level(frame)
        self.i0 = i0

    def update(self, act, click, frame):
        t0 = time.perf_counter()
        f = self.T.update(act, click, frame)
        tl = self.T.tail_line()
        self.total.append(time.perf_counter() - t0)  # full per-action cost: update + facts + tail line
        self.lines.append((tl, json.dumps(f, separators=(",", ":"))))

    def player_mask(self):
        p = self.T.player()
        return None if p is None else self.T.mask(p)

    def selected_mask(self):
        s = self.T.selected
        return None if s is None else self.T.mask(s)


# ---------------------------------------------------------------------------------------------- GT helpers
def key_inst_map(objs):
    """traj key (name#k) -> instance id, using the objects-list order (k-th object of that name)."""
    seen = collections.Counter(); out = {}
    for o in objs:
        k = f"{o[1]}#{seen[o[1]]}"; seen[o[1]] += 1
        out[k] = o[0]
    return out


def gt_ctrl_mask(d, t):
    m = np.zeros((64, 64), bool)
    for o in d["meta"][t]["objects"]:
        if o[1] in d["ctrl"]:
            m |= (d["inst"][t] == o[0]) & (d["ui"][t] < 0)
    return m


def moved_ctrl_keys(d, t):
    tr = d["traj"]; out = set()
    for key, sb in tr[t]["state"].items():
        sa = tr[t - 1]["state"].get(key)
        if key.rsplit("#", 1)[0] in d["ctrl"] and sa and (sa[0], sa[1]) != (sb[0], sb[1]) and sb[4] > 0:
            out.add(key)
    return out


def press_moves(d, t):
    """GT keys that moved on press t, split into agreeing with the arrow / other."""
    ex, ey = DIRS[d["meta"][t]["act"]]
    tr = d["traj"]; agree, other = set(), set()
    for key, sb in tr[t]["state"].items():
        sa = tr[t - 1]["state"].get(key)
        if not sa or sb[4] <= 0 or (sa[0], sa[1]) == (sb[0], sb[1]):
            continue
        dx, dy = sb[0] - sa[0], sb[1] - sa[1]
        along = dx * ex + dy * ey; perp = abs(dx * ey) + abs(dy * ex)
        (agree if along >= 1 and 2 * perp <= along else other).add(key)
    return agree, other


def ext_names(d, lv):
    """Extended GT for one level: GT 'controlled(dir)' names plus names that moved with the arrow on >= 2 presses
    while no GT-controlled sprite moved (a selectable piece, not a pushed box), with >= 60% of their press-moves
    agreeing. Defined before held-out scoring; used only as a secondary diagnostic."""
    st = collections.defaultdict(collections.Counter)
    for t in lv[1:]:
        if d["meta"][t]["act"] not in DIRS:
            continue
        agree, other = press_moves(d, t)
        orig = any(k.rsplit("#", 1)[0] in d["ctrl"] for k in agree | other)
        for k in agree | other:
            n = k.rsplit("#", 1)[0]
            st[n]["moves"] += 1
            if k in agree:
                st[n]["agree"] += 1
                st[n]["indep"] += (not orig)
    ext = set(d["ctrl"])
    for n, c in st.items():
        if c["agree"] >= 2 and c["agree"] >= 0.6 * c["moves"] and c["indep"] >= 2:
            ext.add(n)
    return ext


def names_mask(d, t, names):
    m = np.zeros((64, 64), bool)
    for o in d["meta"][t]["objects"]:
        if o[1] in names:
            m |= (d["inst"][t] == o[0]) & (d["ui"][t] < 0)
    return m


def keys_mask(d, t, keys):
    km = key_inst_map(d["meta"][t]["objects"])
    m = np.zeros((64, 64), bool)
    for k in keys:
        if k in km:
            m |= (d["inst"][t] == km[k]) & (d["ui"][t] < 0)
    return m


def dominant_inst(d, t, mask):
    v = d["inst"][t][mask & (d["ui"][t] < 0)]
    v = v[v >= 0]
    if len(v) == 0:
        return None
    vals, cnt = np.unique(v, return_counts=True)
    return int(vals[cnt.argmax()])


def inst_key(d, t, iid):
    for o_k, o_i in key_inst_map(d["meta"][t]["objects"]).items():
        if o_i == iid:
            return o_k
    return None


# ---------------------------------------------------------------------------------------------- main loop
def run_game(g, make):
    d = load(g)
    frames, level, meta = d["frames"], d["level"], d["meta"]
    n = len(frames)
    trk = make()
    trk.start_level(frames[0]) if not isinstance(trk, AWrap) else trk.start_level(frames[0], 0)
    per_step = {}
    sel_at = {}
    ctrl_moved = {t for t in range(1, n) if meta[t]["act"] in ARROW and level[t] == level[t - 1] and moved_ctrl_keys(d, t)}
    for t in range(1, n):
        if level[t] != level[t - 1]:
            trk.start_level(frames[t]) if not isinstance(trk, AWrap) else trk.start_level(frames[t], t)
            continue
        trk.update(meta[t]["act"], meta[t]["click"], frames[t])
        pm = trk.player_mask()
        sm = trk.selected_mask()
        per_step[t] = pm
        sel_at[t] = sm
    res = {"levels": [], "claims": collections.Counter(), "switch": collections.Counter(), "switch_detail": []}
    # ---- player metric per level
    lv_start = 0
    for i in range(1, n + 1):
        if i < n and level[i] == level[i - 1]:
            continue
        lv = list(range(lv_start, i)); lv_start = i
        if not d["ctrl"] or len(lv) < 3:
            continue
        gt_moves = sum(1 for t in lv[1:] if t in ctrl_moved)
        if gt_moves == 0:
            continue
        preds = []; preds_ext = []
        ext = ext_names(d, lv)
        res.setdefault("ext_names", {})[int(level[lv[0]])] = sorted(ext - set(d["ctrl"]))
        last_now = None; switched = False
        for t in lv[1:]:
            pm = per_step.get(t)
            gtm = gt_ctrl_mask(d, t)
            ok = False
            if pm is not None and gtm.any():
                ok = pm.any() and (pm & gtm).sum() >= 0.5 * pm.sum()
                res["claims"]["committed"] += 1
                res["claims"]["wrong"] += (not ok)
            preds.append(bool(ok))
            gx = names_mask(d, t, ext)
            okx = False
            if pm is not None and gx.any():
                okx = pm.any() and (pm & gx).sum() >= 0.5 * pm.sum()
                res["claims"]["ext_committed"] += 1
                res["claims"]["ext_wrong"] += (not okx)
            preds_ext.append(bool(okx))
            # selected claim vs the ext-GT instance(s) that moved with the latest press (skip after a switch action)
            act = meta[t]["act"]
            if act in DIRS:
                ag, _ = press_moves(d, t)
                now = {k for k in ag if k.rsplit("#", 1)[0] in ext}
                if now:
                    last_now = now; switched = False
            elif act in ("ACTION5", "ACTION6", "ACTION7", "RESET"):
                switched = True
            if last_now and not switched:
                sm = sel_at.get(t)
                g = keys_mask(d, t, last_now)
                if g.any():
                    res["claims"]["sel_n"] += 1
                    res["claims"]["sel_ok"] += bool(sm is not None and sm.any() and (sm & g).sum() >= 0.5 * sm.sum())
        commit = None
        for k in range(len(preds)):
            if all(preds[k:]):
                commit = k + 1; break
        first = next((k + 1 for k, p in enumerate(preds) if p), None)
        commit_x = next((k + 1 for k in range(len(preds_ext)) if all(preds_ext[k:])), None)
        res["levels"].append({"level": int(level[lv[0]]), "n_actions": len(preds), "arrow_actions": gt_moves,
                              "final_ok": bool(preds[-1]), "commit": commit, "first_ok": first,
                              "ok_share": round(sum(preds) / len(preds), 3),
                              "final_ok_ext": bool(preds_ext[-1]), "commit_ext": commit_x})
    # ---- selection switches
    arrows = [t for t in range(1, n) if meta[t]["act"] in ARROW]
    for a0, a1 in zip(arrows, arrows[1:]):
        if level[a1] != level[a0] or any(level[x] != level[a0] for x in range(a0, a1 + 1)):
            continue
        between = [x for x in range(a0 + 1, a1) if meta[x]["act"] in ("ACTION5", "ACTION6", "ACTION7")]
        if not between:
            continue
        s = between[-1]
        act = meta[s]["act"]
        lvx = [x for x in range(n) if level[x] == level[a0]]
        ext = ext_names(d, lvx)
        for gtname, names in (("orig", d["ctrl"]), ("ext", ext)):
            ag0, _ = press_moves(d, a0); ag1, _ = press_moves(d, a1)
            m0 = {k for k in ag0 if k.rsplit("#", 1)[0] in names}; m1 = {k for k in ag1 if k.rsplit("#", 1)[0] in names}
            if not m0 or not m1:
                continue
            kind = "switch" if m1 != m0 else "stay"
            for when, t in (("after_switch", s), ("after_next_press", a1)):
                sm = sel_at.get(t)
                gtm = keys_mask(d, t, m1)
                ok = sm is not None and sm.any() and gtm.any() and (sm & gtm).sum() >= 0.5 * sm.sum()
                res["switch"][f"{gtname}:{kind}:{act}:{when}:n"] += 1
                res["switch"][f"{gtname}:{kind}:{act}:{when}:ok"] += bool(ok)
            res["switch_detail"].append((gtname, kind, act, s, a1))
    # ---- blockers / vanish (A only)
    if isinstance(trk, AWrap):
        trk._flush()
        res["blockers"] = score_blockers(d, trk.blocker_log)
        res["vanish"] = score_vanish(d, trk.vanish_log)
    res["timings_ms"] = [x * 1000 for x in trk.timings]
    if isinstance(trk, AWrap):
        res["lines"] = trk.lines
    return res


def score_blockers(d, events):
    """Claim 'Y blocks X' at press t is correct iff the GT instance under X did not move, Y's footprint lies within
    X's GT mask shifted 1..step px in the press direction, and X's GT pixels never enter Y's footprint anywhere in
    the level (while Y's own GT instance, if any, keeps its position). Works for background walls (no instance)."""
    out = collections.Counter(); det = []
    level = d["level"]; tr = d["traj"]
    for ev in events:
        t = ev["t"]
        pm, fp = ev["player_mask"], ev["blocker_mask"]
        iid = dominant_inst(d, t, pm)
        if iid is not None:
            fp = fp & ~(d["inst"][t] == iid)  # a same-colour merge can put X's own pixels into Y's component
        key = inst_key(d, t, iid) if iid is not None else None
        is_ctrl = key is not None and key.rsplit("#", 1)[0] in d["ctrl"]
        moved = True
        if key is not None:
            sa, sb = tr[t - 1]["state"].get(key), tr[t]["state"].get(key)
            moved = not (sa and sb and (sa[0], sa[1]) == (sb[0], sb[1]))
        ok = False
        if iid is not None and not moved:
            ady, adx = ev["dir"]
            ys, xs = np.nonzero(d["inst"][t] == iid)
            ahead = False
            for k in range(1, ev["step"] + 1):
                ty, tx = ys + ady * k, xs + adx * k
                inb = (ty >= 0) & (ty < 64) & (tx >= 0) & (tx < 64)
                if fp[ty[inb], tx[inb]].any():
                    ahead = True; break
            if ahead:
                yi = dominant_inst(d, t, fp); ykey = inst_key(d, t, yi) if yi is not None else None
                y0 = tr[t]["state"].get(ykey) if ykey else None
                lv = [x for x in range(len(level)) if level[x] == level[t]]
                entered = False
                for t2 in lv:
                    if ykey:
                        y2 = tr[t2]["state"].get(ykey)
                        if y0 is None or y2 is None or (y2[0], y2[1]) != (y0[0], y0[1]):
                            continue  # Y elsewhere at t2: overlap with the old footprint says nothing
                    if ((d["inst"][t2] == iid) & fp).any():
                        entered = True; break
                ok = not entered
        tag = "committed" if ev["committed"] else "tentative"
        out[f"{tag}:n"] += 1; out[f"{tag}:ok"] += ok
        out[f"{tag}:player_is_ctrl"] += is_ctrl
        det.append((ev["t"], ev["player"], ev["blocker"], ev["count"], ok, is_ctrl))
    return {"counts": dict(out), "detail": det}


def score_vanish(d, events):
    """Confirmed 'Y vanished when X overlapped it' is correct iff Y's GT instance no longer occupies its footprint at
    confirmation and the GT instance under the tracker's X overlapped or swept the footprint on the vanish step."""
    out = collections.Counter(); det = []
    for v in events:
        st = v.get("status")
        out[f"{st}:n"] += 1
        if st != "confirmed":
            continue
        t, tr_ = v["t"], v["t_res"]
        F = v["footprint"]
        k = dominant_inst(d, t - 1, F)
        gone = k is not None and not ((d["inst"][tr_] == k) & F).any()
        pmask = v.get("player_mask")
        xi = dominant_inst(d, t, pmask) if pmask is not None and pmask.any() else None
        over = False
        if xi is not None:
            a = d["inst"][t] == xi; b = d["inst"][t - 1] == xi
            ys, xs = np.nonzero(a | b)
            sweep = np.zeros((64, 64), bool)
            if len(ys):
                sweep[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
            over = bool(((a | sweep) & F).any())
        ok = gone and over
        out["confirmed:ok"] += ok; out["confirmed:gone"] += gone; out["confirmed:overlap"] += over
        det.append((t, tr_, v["eid"], v["player"], bool(gone), over))
    return {"counts": dict(out), "detail": det}


def summarize(allres, games):
    L = [l for g in games for l in allres[g]["levels"]]
    claims = collections.Counter(); sw = collections.Counter(); bl = collections.Counter(); vn = collections.Counter()
    tms = []
    for g in games:
        r = allres[g]
        claims.update(r["claims"]); sw.update(r["switch"]); tms += r["timings_ms"]
        if "blockers" in r:
            bl.update(r["blockers"]["counts"]); vn.update(r["vanish"]["counts"])
    s = {
        "levels": len(L),
        "final_ok": sum(l["final_ok"] for l in L),
        "within5": sum(1 for l in L if l["commit"] and l["commit"] <= 5),
        "committed_ever": sum(1 for l in L if l["commit"]),
        "wrong_claim_rate": round(claims["wrong"] / max(claims["committed"], 1), 4),
        "ext_final_ok": sum(l["final_ok_ext"] for l in L),
        "ext_within5": sum(1 for l in L if l["commit_ext"] and l["commit_ext"] <= 5),
        "ext_wrong_claim_rate": round(claims["ext_wrong"] / max(claims["ext_committed"], 1), 4),
        "selected_step_acc": round(claims["sel_ok"] / max(claims["sel_n"], 1), 4),
        "claims": dict(claims),
        "per_game": {g: (len(allres[g]["levels"]), sum(l["final_ok"] for l in allres[g]["levels"]),
                         sum(1 for l in allres[g]["levels"] if l["commit"] and l["commit"] <= 5),
                         [l["commit"] for l in allres[g]["levels"]]) for g in games},
        "switch": dict(sw),
        "latency_ms_p50": round(float(np.percentile(tms, 50)), 3) if tms else None,
        "latency_ms_p95": round(float(np.percentile(tms, 95)), 3) if tms else None,
        "latency_ms_max": round(float(np.max(tms)), 3) if tms else None,
        "updates": len(tms),
    }
    if bl or vn:
        s["blockers"] = dict(bl); s["vanish"] = dict(vn)
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which"); ap.add_argument("games"); ap.add_argument("--out"); ap.add_argument("--cfg", nargs="*", default=[])
    a = ap.parse_args()
    games = SPLIT[a.games] if a.games in SPLIT else a.games.split(",")
    cfg = {}
    for kv in a.cfg:
        k, v = kv.split("="); cfg[k] = type(TA.DEFAULT_CFG[k])(eval(v))
    if a.which == "baseline":
        make = Baseline
    else:
        def make():
            w = AWrap(**cfg)
            return w
    allres = {}
    for g in games:
        allres[g] = run_game(g, make)
        L = allres[g]["levels"]
        print(g, "levels", len(L), "final_ok", sum(l["final_ok"] for l in L), "within5", sum(1 for l in L if l["commit"] and l["commit"] <= 5),
              "commits", [l["commit"] for l in L], "claims", dict(allres[g]["claims"]), flush=True)
    s = summarize(allres, games)
    print(json.dumps({k: v for k, v in s.items() if k != "per_game"}, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"summary": s, "games": {g: {k: v for k, v in r.items() if k != "lines"} for g, r in allres.items()}}, open(a.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        if a.which != "baseline":
            with open(str(a.out).replace(".json", ".lines.jsonl"), "w") as f:
                for g, r in allres.items():
                    for ln, fd in r.get("lines", []):
                        f.write(json.dumps({"g": g, "tail": ln, "facts": fd}) + "\n")
