"""Candidate 5: behavioural object + role tracker on the full recorded action stream.

For every replayed step (bench/full/<game>.npz; recorded boards == rendered frames) a segmenter gives
labels + stable D4 names. The tracker, online and causal:
  player   : per name, over arrow actions: moved share > 0.3 and moved-in-arrow-direction share > 0.6;
             top-1 = most agreeing moves (>=2). Correct when the predicted segment's pixels lie >=50% on a
             GT controlled sprite (roles_behaviour.json 'controlled(dir)').
  click    : segment under each ACTION6 click: IoU>=0.5 with the GT instance under the click (identity of the click target),
             and 'responsive' = frame changed within that segment's bbox+1.
  HUD      : a 3-px edge strip that changes on >=30% of a level's steps (>=3): its pixels that ever changed vs GT HUD (ui) pixels.
Segmenters: cc (colour components = harness), lex (cand_lexicon cc mode), oracle (GT instances, name = design), tiny (held-out folds only).
Usage: python roles_eval.py <segmenter> [--games ...]
"""
import argparse, collections, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
FULL = HERE / "full"
ROLES = json.load(open(HERE.parent / "gt" / "roles_behaviour.json"))
DIRS = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}
CROSS = ndimage.generate_binary_structure(2, 1)
RULE = os.environ.get("TRACK_RULE", "count")  # count: most agreeing moves; ratio: agreeing moves / arrow steps present
MIN_AGREE = int(os.environ.get("TRACK_MIN_AGREE", "2"))


def canon(p):
    vs = []
    for m in (p, p[:, ::-1]):
        for r in range(4):
            q = np.rot90(m, r); vs.append(q.tobytes() + bytes(q.shape))
    return hashlib.blake2b(min(vs), digest_size=6).hexdigest()


def name_segments(fr, lab):
    out = {}
    for j in np.unique(lab):
        if j < 0: continue
        m = lab == j; ys, xs = np.nonzero(m)
        out[int(j)] = (canon(np.where(m, fr, -1)[ys.min():ys.max() + 1, xs.min():xs.max() + 1]), float(ys.mean()), float(xs.mean()), (ys.min(), xs.min(), ys.max(), xs.max()))
    return out


def seg_cc(fr, **_):
    lab = np.full(fr.shape, -1, np.int64); n = 0
    vals, cnt = np.unique(fr, return_counts=True); bgc = vals[cnt.argmax()]
    for c in vals:
        if c == bgc: continue  # modal colour = background (harness models see it too)
        l, k = ndimage.label(fr == c, structure=CROSS); lab[l > 0] = l[l > 0] - 1 + n; n += k
    return lab


def make_lex():
    os.environ["LEX_MODE"] = "cc"
    import cand_lexicon as L
    st = {"s": None}

    def f(fr, newgame=False, **_):
        if newgame: st["s"] = None
        lab, _, _, st["s"] = L.segment(fr, state=st["s"])
        vals, cnt = np.unique(fr, return_counts=True); bgc = vals[cnt.argmax()]
        # drop segments of the modal colour that are huge (background)
        for j in np.unique(lab[fr == bgc]):
            if (lab == j).sum() > 400: lab[lab == j] = -1
        return lab
    return f


def make_tiny():
    import cand_tiny as T

    def f(fr, game=None, **_):
        T.setup(game)
        aff, sem, clk = T.predict(fr)
        return T.decode(fr, aff, sem)
    return f


def run_game(g, seg, segname):
    z = np.load(FULL / f"{g}.npz"); meta = json.load(open(FULL / f"{g}.meta.json"))
    frames, inst, ui, level = z["frames"].astype(np.int64), z["inst"], z["ui"], z["level"]
    ctrl_names = {n for n, r, _, _ in ROLES[g] if "controlled(dir)" in r}
    traj = json.load(open(HERE.parent / "gt" / "data" / f"{g}.traj.json"))
    # steps at which a GT-controlled, visible sprite moved under an arrow action
    ctrl_moved = set()
    for t in range(1, len(traj)):
        if traj[t]["act"] not in DIRS: continue
        for key, sb in traj[t]["state"].items():
            sa = traj[t - 1]["state"].get(key)
            if key.rsplit("#", 1)[0] in ctrl_names and sa and (sa[0], sa[1]) != (sb[0], sb[1]) and sb[4] > 0:
                ctrl_moved.add(t)
    out = {"levels": [], "click": collections.Counter(), "hud": []}
    t_seg = []
    labs = []; segs = []
    for i in range(len(frames)):
        if segname == "oracle":
            lab = np.where(ui[i] >= 0, -1, inst[i]).astype(np.int64)
        else:
            t0 = time.perf_counter(); lab = seg(frames[i], game=g, newgame=(i == 0)); t_seg.append(time.perf_counter() - t0)
        labs.append(lab); segs.append(name_segments(frames[i], lab))
    # --- player tracking per level
    lv_start = 0
    for i in range(1, len(frames) + 1):
        if i < len(frames) and level[i] == level[i - 1]:
            continue
        lv = range(lv_start, i); lv_start = i
        if not ctrl_names or len(lv) < 3:
            continue
        stats = collections.defaultdict(collections.Counter)
        gt_moves = 0; first_ok = None; preds = []
        for t in list(lv)[1:]:
            act = meta[t]["act"]
            if act in DIRS:
                ex, ey = DIRS[act]
                a, b = segs[t - 1], segs[t]
                byname = collections.defaultdict(list)
                for j, (nm, cy, cx, bb) in b.items():
                    byname[nm].append((cy, cx))
                seen = set()
                for j, (nm, cy, cx, bb) in a.items():
                    if nm in seen: continue
                    seen.add(nm)
                    st = stats[nm]; st["present"] += 1
                    cands = byname.get(nm)
                    if not cands: continue
                    dy, dx = min(((y - cy, x - cx) for y, x in cands), key=lambda d: abs(d[0]) + abs(d[1]))
                    if abs(dy) + abs(dx) > 0.5:
                        st["moved"] += 1
                        if (np.sign(round(dx)), np.sign(round(dy))) == (ex, ey): st["agree"] += 1
                gt_moves += t in ctrl_moved
            # current top-1
            best = None; bs = None
            for nm, st in stats.items():
                if st["agree"] >= MIN_AGREE and st["moved"] / max(st["present"], 1) > 0.3 and st["agree"] / max(st["moved"], 1) > 0.6:
                    sc = st["agree"] if RULE == "count" else (st["agree"] / max(st["present"], 1), st["agree"])
                    if bs is None or sc > bs: best, bs = nm, sc
            ok = False
            if best is not None:
                gtm = np.zeros((64, 64), bool)
                objs = meta[t]["objects"]
                for o in objs:
                    if o[1] in ctrl_names: gtm |= (inst[t] == o[0]) & (ui[t] < 0)
                if gtm.any():
                    js = [j for j, v in segs[t].items() if v[0] == best]
                    if js:
                        m = np.isin(labs[t], js)
                        ok = (m & gtm).sum() >= 0.5 * m.sum()
            preds.append(ok)
        if gt_moves == 0:
            continue
        # actions-to-commit: first action index after which prediction stays correct to the end of the level
        commit = None
        for k in range(len(preds)):
            if all(preds[k:]): commit = k + 1; break
        out["levels"].append({"level": int(level[lv[0]]), "n_actions": len(preds), "arrow_actions": gt_moves, "final_ok": bool(preds[-1]) if preds else False, "commit": commit})
    # --- clicks
    for t in range(1, len(frames)):
        c = meta[t]["click"]
        if not c: continue
        x, y = c
        if not (0 <= x < 64 and 0 <= y < 64): continue
        out["click"]["n"] += 1
        k = inst[t - 1][y, x]
        if k < 0 or ui[t - 1][y, x] >= 0: continue
        out["click"]["on_sprite"] += 1
        gm = (inst[t - 1] == k) & (ui[t - 1] < 0)
        j = labs[t - 1][y, x]
        if j < 0: continue
        pm = labs[t - 1] == j
        iou = (gm & pm).sum() / (gm | pm).sum()
        out["click"]["iou50"] += iou >= 0.5
        bb = segs[t - 1][int(j)][3]
        ch = frames[t - 1] != frames[t]
        out["click"]["changed"] += bool(ch.any())
        out["click"]["responsive"] += bool(ch[max(bb[0] - 1, 0):bb[2] + 2, max(bb[1] - 1, 0):bb[3] + 2].any())
    # --- HUD from per-level pixel change frequency in a 3-px border band
    band = np.zeros((64, 64), bool); band[:3] = band[-3:] = True; band[:, :3] = band[:, -3:] = True
    lv_start = 0
    for i in range(1, len(frames) + 1):
        if i < len(frames) and level[i] == level[i - 1]: continue
        lv = list(range(lv_start, i)); lv_start = i
        if len(lv) < 5: continue
        ever = np.zeros((64, 64), bool); pred = np.zeros((64, 64), bool)
        strips = [(slice(0, 3), slice(None)), (slice(61, 64), slice(None)), (slice(None), slice(0, 3)), (slice(None), slice(61, 64))]
        hits = [0, 0, 0, 0]
        for t in lv[1:]:
            d = frames[t] != frames[t - 1]; ever |= d
            for q, sl in enumerate(strips): hits[q] += bool(d[sl].any())
        for q, sl in enumerate(strips):
            if hits[q] >= max(3, 0.3 * (len(lv) - 1)):
                m = np.zeros((64, 64), bool); m[sl] = True; pred |= m & ever
        gt = np.zeros((64, 64), bool)
        for t in lv: gt |= ui[t] >= 0
        gtc = np.zeros((64, 64), bool)  # GT HUD pixels that actually changed in this level (counter/timer)
        for t in lv[1:]: gtc |= (ui[t] >= 0) & (frames[t] != frames[t - 1])
        out["hud"].append({"pred": int(pred.sum()), "gt_changing": int(gtc.sum()), "inter": int((pred & gtc).sum()), "union": int((pred | gtc).sum())})
    out["ms"] = float(np.median(t_seg) * 1000) if t_seg else 0.0
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("seg"); ap.add_argument("--games", default=",".join(sorted(p.stem for p in FULL.glob("*.npz"))))
    a = ap.parse_args(); sys.path.insert(0, str(HERE))
    seg = {"cc": lambda: seg_cc, "lex": make_lex, "oracle": lambda: None, "tiny": make_tiny}[a.seg]()
    res = {}
    for g in a.games.split(","):
        res[g] = run_game(g, seg, a.seg)
        L = res[g]["levels"]
        print(g, "levels", len(L), "final_ok", sum(l["final_ok"] for l in L), "commit<=5", sum(1 for l in L if l["commit"] and l["commit"] <= 5), "click", dict(res[g]["click"]), flush=True)
    json.dump(res, open(HERE / "results" / f"roles_{a.seg}{os.environ.get('ROLES_TAG', '')}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
