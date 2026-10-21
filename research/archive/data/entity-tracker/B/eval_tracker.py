"""Stage-0 evaluation of the Design B tracker against engine ground truth, with the roles_eval.py baseline
scored by the same code on the same levels.

Metrics (eligible levels = roles_eval.py's: a GT-controlled sprite visibly moves under an arrow, >= 3 frames):
  final_ok      committed claim correct at the level's last frame
  commit<=5     committed claim correct at every action from the 5th to the level end (roles_eval 'commit')
  wrong rate    committed-claim steps whose mask is < 50% on GT controlled pixels / committed steps
                (steps where GT controlled pixels are empty are not adjudicable and are excluded)
  switch        GT selection switches: consecutive moving arrow presses with disjoint GT mover sets and a
                non-arrow action between; claim scored at the frame after the last switch action
                ('immediate') and after the first arrow press that moved the new piece ('after arrow')
  blocker / vanish precision (see gt_blocker / gt_vanish)
  latency       per-update wall time (time.perf_counter), whole update incl. rendering

Usage: python eval_tracker.py dev|heldout [--final] [--games g1,g2] [--tag name] [--set K=V ...]
"""
import argparse, collections, json, os, sys, time
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tracker as TR  # noqa: E402

V = Path("/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/vision")
FULL = V / "bench" / "full"
ROLES = json.load(open(V / "gt" / "roles_behaviour.json"))
SPLIT = json.load(open(HERE / "split.json"))
DIRS = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}  # (ex, ey) as roles_eval
CROSS = ndimage.generate_binary_structure(2, 1)


# ---------------------------------------------------------------------------- ground truth
class GT:
    def __init__(self, g):
        z = np.load(FULL / f"{g}.npz")
        self.g = g
        self.frames = z["frames"].astype(np.int64)
        self.inst, self.ui, self.level = z["inst"], z["ui"], z["level"]
        self.meta = json.load(open(FULL / f"{g}.meta.json"))
        self.traj = json.load(open(V / "gt" / "data" / f"{g}.traj.json"))
        self.ctrl_names = {n for n, r, _, _ in ROLES[g] if "controlled(dir)" in r}
        # steps where a GT-controlled visible sprite moved under an arrow (roles_eval definition)
        self.ctrl_moved = set()
        self.movers = {}
        for t in range(1, len(self.traj)):
            if self.traj[t]["act"] not in DIRS:
                continue
            mv = set()
            for key, sb in self.traj[t]["state"].items():
                sa = self.traj[t - 1]["state"].get(key)
                if key.rsplit("#", 1)[0] in self.ctrl_names and sa and (sa[0], sa[1]) != (sb[0], sb[1]) and sb[4] > 0:
                    self.ctrl_moved.add(t)
                    mv.add(key)
            if mv and self.level[t] == self.level[t - 1]:
                self.movers[t] = mv
        self._keys = {}
        # arrow-driven sprites per level: moved in the pressed direction on >= 2 presses, >= 60% agreeing
        mv, ag, indep = collections.Counter(), collections.Counter(), collections.Counter()
        for t in range(1, len(self.traj)):
            a = self.traj[t]["act"]
            if a not in DIRS or self.level[t] != self.level[t - 1]:
                continue
            ex, ey = DIRS[a]
            ctrl_near = None
            for key, sb in self.traj[t]["state"].items():
                sa = self.traj[t - 1]["state"].get(key)
                if not sa or (sa[0], sa[1]) == (sb[0], sb[1]):
                    continue
                dx, dy = sb[0] - sa[0], sb[1] - sa[1]
                L = int(self.level[t])
                mv[(L, key)] += 1
                along = dx * ex + dy * ey
                if along > 0 and abs(dx * ey) + abs(dy * ex) <= along / 2:
                    ag[(L, key)] += 1
                    if key.rsplit("#", 1)[0] in self.ctrl_names:
                        continue
                    if ctrl_near is None:
                        cm = np.zeros((64, 64), bool)
                        for k2, key2 in self.keys(t - 1).items():
                            s2a = self.traj[t - 1]["state"].get(key2); s2b = self.traj[t]["state"].get(key2)
                            if key2.rsplit("#", 1)[0] in self.ctrl_names and s2a and s2b and (s2a[0], s2a[1]) != (s2b[0], s2b[1]):
                                cm |= self.inst[t - 1] == k2
                        ctrl_near = ndimage.binary_dilation(cm, structure=np.ones((5, 5), bool))
                    km = self.keys_mask(t - 1, {key})
                    if km.any() and not (km & ctrl_near).any():
                        indep[(L, key)] += 1
        self.arrow_keys = collections.defaultdict(set)
        for (L, key), n in mv.items():
            if ag[(L, key)] >= 2 and ag[(L, key)] >= 0.6 * n and (indep[(L, key)] >= 1 or key.rsplit("#", 1)[0] in self.ctrl_names):
                self.arrow_keys[L].add(key)

    def arrow_mask(self, t):
        return self.keys_mask(t, self.arrow_keys.get(int(self.level[t]), set()))

    def keys(self, t):
        """instance index k -> traj key name#j at frame t."""
        if t not in self._keys:
            cnt = collections.Counter(); out = {}
            for o in self.meta[t]["objects"]:
                out[o[0]] = o[1] + "#" + str(cnt[o[1]]); cnt[o[1]] += 1
            self._keys[t] = out
        return self._keys[t]

    def ctrl_mask(self, t):
        m = np.zeros((64, 64), bool)
        for o in self.meta[t]["objects"]:
            if o[1] in self.ctrl_names:
                m |= (self.inst[t] == o[0]) & (self.ui[t] < 0)
        return m

    def keys_mask(self, t, keys):
        m = np.zeros((64, 64), bool)
        for k, key in self.keys(t).items():
            if key in keys:
                m |= (self.inst[t] == k) & (self.ui[t] < 0)
        return m

    def levels(self):
        out = []
        s = 0
        for i in range(1, len(self.frames) + 1):
            if i < len(self.frames) and self.level[i] == self.level[i - 1]:
                continue
            out.append(list(range(s, i))); s = i
        return out

    def eligible(self, lv):
        if not self.ctrl_names or len(lv) < 3:
            return False
        return any(t in self.ctrl_moved for t in lv[1:] if self.meta[t]["act"] in DIRS)


def on(mask, gt):
    return mask is not None and mask.any() and gt.any() and (mask & gt).sum() >= 0.5 * mask.sum()


# ---------------------------------------------------------------------------- runners
def run_tracker(gt, params=None):
    tr = TR.EntityTracker(params)
    out = []
    nf_prev = 0
    for i in range(len(gt.frames)):
        newlv = i == 0 or gt.level[i] != gt.level[i - 1]
        m = gt.meta[i]
        t0 = time.perf_counter()
        f = tr.update(gt.frames[i], m["act"], m["click"], new_level=newlv)
        line = tr.tail_line()
        cm, conf = tr.claim()
        sm, sconf = tr.selection()
        dt = (time.perf_counter() - t0) * 1000
        if newlv:
            nf_prev = 0
        new_facts = tr.facts[nf_prev:]
        nf_prev = len(tr.facts)
        out.append({"claim": cm if conf == "committed" else None, "sel": sm, "sel_conf": sconf, "ms": dt,
                    "fields": f, "tail": line, "events": list(tr.events), "new_facts": new_facts})
    return out


def run_baseline(gt):
    """roles_eval.py cc rule (count rule, MIN_AGREE 2), per-step top-1 mask; reproduced verbatim in logic."""
    import hashlib

    def canon(p):
        vs = []
        for mm in (p, p[:, ::-1]):
            for r in range(4):
                q = np.rot90(mm, r); vs.append(q.tobytes() + bytes(q.shape))
        return hashlib.blake2b(min(vs), digest_size=6).hexdigest()

    def seg_cc(fr):
        lab = np.full(fr.shape, -1, np.int64); n = 0
        vals, cnt = np.unique(fr, return_counts=True); bgc = vals[cnt.argmax()]
        for c in vals:
            if c == bgc: continue
            l, k = ndimage.label(fr == c, structure=CROSS); lab[l > 0] = l[l > 0] - 1 + n; n += k
        return lab

    def name_segments(fr, lab):
        out = {}
        for j in np.unique(lab):
            if j < 0: continue
            m = lab == j; ys, xs = np.nonzero(m)
            out[int(j)] = (canon(np.where(m, fr, -1)[ys.min():ys.max() + 1, xs.min():xs.max() + 1]), float(ys.mean()), float(xs.mean()))
        return out

    labs, segs, ms = [], [], []
    for i in range(len(gt.frames)):
        t0 = time.perf_counter()
        lab = seg_cc(gt.frames[i]); labs.append(lab); segs.append(name_segments(gt.frames[i], lab))
        ms.append((time.perf_counter() - t0) * 1000)
    out = [{"claim": None, "sel": None, "ms": ms[i], "new_facts": [], "events": []} for i in range(len(gt.frames))]
    for lv in gt.levels():
        stats = collections.defaultdict(collections.Counter)
        for t in lv[1:]:
            act = gt.meta[t]["act"]
            t0 = time.perf_counter()
            if act in DIRS:
                ex, ey = DIRS[act]
                a, b = segs[t - 1], segs[t]
                byname = collections.defaultdict(list)
                for j, (nm, cy, cx) in b.items():
                    byname[nm].append((cy, cx))
                seen = set()
                for j, (nm, cy, cx) in a.items():
                    if nm in seen: continue
                    seen.add(nm)
                    st = stats[nm]; st["present"] += 1
                    cands = byname.get(nm)
                    if not cands: continue
                    dy, dx = min(((y - cy, x - cx) for y, x in cands), key=lambda d: abs(d[0]) + abs(d[1]))
                    if abs(dy) + abs(dx) > 0.5:
                        st["moved"] += 1
                        if (np.sign(round(dx)), np.sign(round(dy))) == (ex, ey): st["agree"] += 1
            best = None; bs = None
            for nm, st in stats.items():
                if st["agree"] >= 2 and st["moved"] / max(st["present"], 1) > 0.3 and st["agree"] / max(st["moved"], 1) > 0.6:
                    sc = st["agree"]
                    if bs is None or sc > bs: best, bs = nm, sc
            if best is not None:
                js = [j for j, v in segs[t].items() if v[0] == best]
                if js:
                    m = np.isin(labs[t], js)
                    out[t]["claim"] = m; out[t]["sel"] = m
            out[t]["ms"] += (time.perf_counter() - t0) * 1000
    return out


# ---------------------------------------------------------------------------- scoring
def gt_blocker(gt, f):
    """A blocker fact is right when: the claimed mover is the GT controlled sprite and it did not move on
    that press; the claimed blocker is >= 50% one GT (non-HUD, non-controlled) instance; and that instance's
    pixels lie in the GT mover's path within one GT step in the pressed direction. 'strict' also needs
    the instance to be engine-collidable."""
    t = f["step_abs"]
    ctrl = gt.ctrl_mask(t) | gt.arrow_mask(t)
    if not on(f["mover_mask"], ctrl):
        return False, False, "mover-not-player"
    keys_t = gt.keys(t); keys_p = gt.keys(t - 1)
    ks = np.unique(gt.inst[t][f["mover_mask"] & ctrl]); ks = [int(k) for k in ks if k >= 0]
    for k in ks:
        key = keys_t.get(k)
        sa = gt.traj[t - 1]["state"].get(key); sb = gt.traj[t]["state"].get(key)
        if sa and sb and (sa[0], sa[1]) != (sb[0], sb[1]):
            return False, False, "player-moved"
    bm = f["blocker_mask"]
    inst_b = gt.inst[t][bm & (gt.ui[t] < 0)]
    inst_b = inst_b[inst_b >= 0]
    if bm.sum() == 0 or inst_b.size < 0.5 * bm.sum():
        return False, False, "blocker-not-object"
    cand = [int(k) for k in np.unique(inst_b)]
    if any(k in ks for k in cand) and np.isin(inst_b, ks).sum() >= 0.5 * bm.sum():
        return False, False, "blocker-is-mover"
    cand = [k for k in cand if k not in ks]
    # GT step: modal screen displacement of the GT controlled sprite along this axis in this level
    lv = gt.level[t]
    steps = collections.Counter()
    scale = gt.meta[t]["scale"] or 1
    for tt, mv in gt.movers.items():
        if gt.level[tt] != lv: continue
        for key in mv:
            sa = gt.traj[tt - 1]["state"][key]; sb = gt.traj[tt]["state"][key]
            d = abs(sb[0] - sa[0]) + abs(sb[1] - sa[1])
            if d: steps[d * scale] += 1
    S = steps.most_common(1)[0][0] if steps else 1
    ex, ey = DIRS[f["action"]]
    pm = ctrl & np.isin(gt.inst[t], ks)
    sweep = np.zeros_like(pm)
    for s_ in range(1, S + 1):
        sh = np.zeros_like(pm)
        ys, xs = np.nonzero(pm); ys2, xs2 = ys + s_ * ey, xs + s_ * ex
        ok = (ys2 >= 0) & (ys2 < 64) & (xs2 >= 0) & (xs2 < 64)
        sh[ys2[ok], xs2[ok]] = True
        sweep |= sh
    inway = [k for k in cand if (sweep & (gt.inst[t] == k) & bm).any() or (sweep & (gt.inst[t] == k)).any() and (bm & (gt.inst[t] == k)).any()]
    if not inway:
        return False, False, "not-in-path"
    objs = {o[0]: o for o in gt.meta[t]["objects"]}
    obj = objs[inway[0]]
    return True, bool(obj[4]), "ok"


def gt_vanish(gt, f):
    """A vanish fact is right when the claimed region (previous frame) is >= 50% one GT instance that is
    removed or invisible at the claim step, and the GT controlled sprite was within 1 px of it at the
    vanish step."""
    ts, tc = f["at_abs"], f["step_abs"]
    vm = f["mask"]
    inst_p = gt.inst[ts - 1][vm & (gt.ui[ts - 1] < 0)]
    if inst_p.size == 0:
        return False, "not-object"
    vals, cnt = np.unique(inst_p, return_counts=True)
    k = int(vals[cnt.argmax()])
    if k < 0 or cnt.max() < 0.5 * vm.sum():
        return False, "not-object"
    key = gt.keys(ts - 1)[k]
    # identity-safe: traj keys are name#index by list order, so a removal renames later instances.
    # Removed = fewer visible same-name sprites than before and none visible at the original position.
    name = key.rsplit("#", 1)[0]
    sa = gt.traj[ts - 1]["state"][key]
    before = [v for kk, v in gt.traj[ts - 1]["state"].items() if kk.rsplit("#", 1)[0] == name and v[3]]
    after = [v for kk, v in gt.traj[tc]["state"].items() if kk.rsplit("#", 1)[0] == name and v[3]]
    at_pos = [v for v in after if (v[0], v[1]) == (sa[0], sa[1])]
    if at_pos:
        return False, ("still-there" if any(v[4] > 0 for v in at_pos) else "occluded")
    if len(after) >= len(before):
        return False, "moved-or-replaced"
    near = ndimage.binary_dilation(vm, structure=np.ones((3, 3), bool))
    if not ((gt.ctrl_mask(ts) | gt.arrow_mask(ts)) & near).any():
        return False, "player-not-near"
    return True, "ok"


def score(gt, out, with_facts=True):
    res = {"levels": [], "wrong": 0, "committed": 0, "wrong_a": 0, "committed_a": 0, "switch": [], "blockers": [], "vanish": [], "ms": [o["ms"] for o in out]}
    for lv in gt.levels():
        if not gt.eligible(lv):
            continue
        preds, preds_a = [], []
        for t in lv[1:]:
            gm = gt.ctrl_mask(t)
            ga = gt.arrow_mask(t)
            c = out[t]["claim"]
            ok = on(c, gm)
            ok_a = on(c, ga | gm)
            preds.append(ok); preds_a.append(ok_a)
            if c is not None and c.any() and gm.any():
                res["committed"] += 1
                res["wrong"] += not ok
            if c is not None and c.any() and (ga | gm).any():
                res["committed_a"] += 1
                res["wrong_a"] += not ok_a
        def first_commit(pr):
            for k in range(len(pr)):
                if all(pr[k:]):
                    return k + 1
            return None
        res["levels"].append({"level": int(gt.level[lv[0]]), "n_actions": len(preds), "final_ok": bool(preds[-1]) if preds else False,
                              "commit": first_commit(preds), "final_ok_a": bool(preds_a[-1]) if preds_a else False,
                              "commit_a": first_commit(preds_a)})
        # selection switches
        arrows = [t for t in lv[1:] if t in gt.movers]
        for a, b in zip(arrows, arrows[1:]):
            if gt.movers[a] & gt.movers[b]:
                continue
            between = [t for t in range(a + 1, b) if gt.meta[t]["act"] not in DIRS and gt.meta[t]["act"] != "RESET"]
            if not between:
                continue
            s = between[-1]
            new = gt.movers[b]
            g_s = gt.keys_mask(s, new); g_b = gt.keys_mask(b, new)
            res["switch"].append({"s": s, "b": b, "act": gt.meta[s]["act"],
                                  "imm": on(out[s]["sel"], g_s), "imm_committed": on(out[s]["claim"], g_s),
                                  "imm_any": out[s]["sel"] is not None,
                                  "arrow": on(out[b]["sel"], g_b), "arrow_committed": on(out[b]["claim"], g_b)})
    if with_facts:
        for t, o in enumerate(out):
            for f in o.get("new_facts", []):
                if f["type"] == "blocked":
                    ok, strict, why = gt_blocker(gt, f)
                    res["blockers"].append({"t": t, "ok": ok, "strict": strict, "why": why, "n": f["n"]})
                elif f["type"] == "vanished":
                    ok, why = gt_vanish(gt, f)
                    res["vanish"].append({"t": t, "ok": ok, "why": why})
    return res


def summarize(R, label):
    L = [l for r in R.values() for l in r["levels"]]
    n = len(L)
    fin = sum(l["final_ok"] for l in L)
    c5 = sum(1 for l in L if l["commit"] and l["commit"] <= 5)
    com = sum(r["committed"] for r in R.values()); wr = sum(r["wrong"] for r in R.values())
    sw = [s for r in R.values() for s in r["switch"]]
    bl = [b for r in R.values() for b in r["blockers"]]
    va = [v for r in R.values() for v in r["vanish"]]
    ms = np.array([m for r in R.values() for m in r["ms"]])
    s = {"label": label, "levels": n, "final_ok": fin, "commit_le5": c5, "committed_steps": com, "wrong_steps": wr,
         "wrong_rate": wr / max(com, 1),
         "final_ok_a": sum(l["final_ok_a"] for l in L), "commit_le5_a": sum(1 for l in L if l["commit_a"] and l["commit_a"] <= 5),
         "committed_a": sum(r["committed_a"] for r in R.values()), "wrong_a": sum(r["wrong_a"] for r in R.values()),
         "wrong_rate_a": sum(r["wrong_a"] for r in R.values()) / max(sum(r["committed_a"] for r in R.values()), 1),
         "switch_n": len(sw), "switch_imm": sum(x["imm"] for x in sw), "switch_imm_committed": sum(x["imm_committed"] for x in sw),
         "switch_arrow": sum(x["arrow"] for x in sw), "switch_arrow_committed": sum(x["arrow_committed"] for x in sw),
         "blockers": len(bl), "blockers_ok": sum(b["ok"] for b in bl), "blockers_strict": sum(b["strict"] for b in bl),
         "vanish": len(va), "vanish_ok": sum(v["ok"] for v in va),
         "ms_p50": float(np.percentile(ms, 50)), "ms_p95": float(np.percentile(ms, 95)), "ms_max": float(ms.max()),
         "per_game": {g: {"levels": len(r["levels"]), "final_ok": sum(l["final_ok"] for l in r["levels"]),
                          "commit_le5": sum(1 for l in r["levels"] if l["commit"] and l["commit"] <= 5),
                          "commits": [l["commit"] for l in r["levels"]],
                          "wrong": f'{r["wrong"]}/{r["committed"]}', "wrong_a": f'{r["wrong_a"]}/{r["committed_a"]}',
                          "c5_a": sum(1 for l in r["levels"] if l["commit_a"] and l["commit_a"] <= 5),
                          "switch": f'{sum(x["imm"] for x in r["switch"])}/{sum(x["arrow"] for x in r["switch"])}/{len(r["switch"])}',
                          "blk": f'{sum(b["ok"] for b in r["blockers"])}/{len(r["blockers"])}',
                          "van": f'{sum(v["ok"] for v in r["vanish"])}/{len(r["vanish"])}'} for g, r in R.items()}}
    return s


def fmt(s):
    lines = [f"[{s['label']}] levels {s['levels']}: final_ok {s['final_ok']} ({s['final_ok']/max(s['levels'],1):.0%}), "
             f"commit<=5 {s['commit_le5']} ({s['commit_le5']/max(s['levels'],1):.0%}), wrong committed {s['wrong_steps']}/{s['committed_steps']} "
             f"({s['wrong_rate']:.1%}) | arrowGT: final {s['final_ok_a']} commit<=5 {s['commit_le5_a']} wrong {s['wrong_a']}/{s['committed_a']} ({s['wrong_rate_a']:.1%}) | switch imm {s['switch_imm']}/{s['switch_n']} (committed {s['switch_imm_committed']}), after-arrow "
             f"{s['switch_arrow']}/{s['switch_n']}; blockers {s['blockers_ok']}/{s['blockers']} (strict {s['blockers_strict']}); "
             f"vanish {s['vanish_ok']}/{s['vanish']}; ms p50 {s['ms_p50']:.2f} p95 {s['ms_p95']:.2f} max {s['ms_max']:.1f}"]
    for g, v in s["per_game"].items():
        lines.append(f"   {g}: {v}")
    return "\n".join(lines)


def attach_abs(out):
    """Facts carry tracker-relative steps; attach absolute frame indices for GT lookups."""
    for t, o in enumerate(out):
        for f in o.get("new_facts", []):
            f["step_abs"] = t
            if f["type"] == "vanished":
                f["at_abs"] = t - (f["step"] - f["at"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["dev", "heldout"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--games")
    ap.add_argument("--tag", default="")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--dump-text", help="write the rendered fields + tail per step to this jsonl")
    a = ap.parse_args()
    if a.split == "heldout" and not a.final:
        sys.exit("held-out scoring needs --final (score once, frozen tracker)")
    games = a.games.split(",") if a.games else SPLIT[a.split]
    assert all(g in SPLIT[a.split] for g in games)
    params = {k: float(v) if "." in v else int(v) for k, v in (x.split("=") for x in a.set)}
    R, RB = {}, {}
    dump = open(a.dump_text, "w") if a.dump_text else None
    for g in games:
        gt = GT(g)
        out = run_tracker(gt, params)
        if dump:
            for t, o in enumerate(out):
                dump.write(json.dumps({"g": g, "t": t, "fields": o["fields"], "tail": o["tail"]}) + "\n")
        attach_abs(out)
        R[g] = score(gt, out)
        if a.baseline:
            RB[g] = score(gt, run_baseline(gt), with_facts=False)
    s = summarize(R, f"tracker {a.split}{a.tag}")
    print(fmt(s))
    res_dir = HERE / "results"; res_dir.mkdir(exist_ok=True)
    json.dump({"summary": s, "raw": {g: {k: v for k, v in r.items() if k != "ms"} for g, r in R.items()}},
              open(res_dir / f"tracker_{a.split}{a.tag}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if a.baseline:
        sb = summarize(RB, f"baseline {a.split}")
        print(fmt(sb))
        json.dump({"summary": sb, "raw": {g: {k: v for k, v in r.items() if k != "ms"} for g, r in RB.items()}},
                  open(res_dir / f"baseline_{a.split}{a.tag}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
