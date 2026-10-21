"""Shared evaluator for object-segmentation candidates on the ARC-AGI-3 ground-truth frames.

Candidate module interface (bench/cand_<name>.py):
    setup(game)                     optional; called before a game's frames
    segment(frame, prev=None, state=None) -> (labels, names, roles, state)
        labels: int32 64x64, -1 = background (no object), else segment id
        names:  {seg: stable_name}  (D4/translation-invariant design name, may be {})
        roles:  {seg: role}         (may be {})
Frames of a game are fed in recorded order with prev = previous sampled frame.
Usage: python eval.py <cand> [--games g1,g2] [--out results/<cand>.json] [--threads 1]
"""
import argparse, collections, importlib, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "gt" / "data"
TILE_GAMES = {"ls20", "ar25", "tn36", "tr87", "lp85"}
ALL = sorted(p.stem for p in DATA.glob("*.npz"))


def group_map(imv, objs):
    """Merge 4-adjacent instances of the same design into groups. Returns int map (-1 bg, -2 HUD)."""
    ids = [int(k) for k in np.unique(imv) if k >= 0]
    parent = {k: k for k in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in ((imv[:, :-1], imv[:, 1:]), (imv[:-1, :], imv[1:, :])):
        m = (a >= 0) & (b >= 0) & (a != b)
        for x, y in set(zip(a[m].tolist(), b[m].tolist())):
            if objs[x]["design"] == objs[y]["design"]:
                ra, rb = find(x), find(y)
                if ra != rb:
                    parent[ra] = rb
    out = imv.copy()
    for k in ids:
        out[imv == k] = find(k)
    return out


def contingency(gt, pr, mask):
    """Counts of (gt, pred) pixel pairs inside mask -> dict."""
    g = gt[mask].astype(np.int64); p = pr[mask].astype(np.int64)
    key = (g + 3) * 100000 + (p + 3)
    u, c = np.unique(key, return_counts=True)
    return {(int(k // 100000) - 3, int(k % 100000) - 3): int(n) for k, n in zip(u, c)}


def ari_vi(cont):
    n = sum(cont.values())
    if n < 2:
        return 1.0, 0.0
    a = collections.Counter(); b = collections.Counter()
    for (g, p), c in cont.items():
        a[g] += c; b[p] += c
    comb = lambda x: x * (x - 1) / 2
    sij = sum(comb(c) for c in cont.values()); sa = sum(comb(c) for c in a.values()); sb = sum(comb(c) for c in b.values())
    exp = sa * sb / comb(n); mx = (sa + sb) / 2
    ari = 1.0 if mx == exp else (sij - exp) / (mx - exp)
    h = lambda cnt: -sum(c / n * math.log(c / n) for c in cnt.values())
    mi = sum(c / n * math.log(c * n / (a[g] * b[p])) for (g, p), c in cont.items())
    return ari, h(a) + h(b) - 2 * mi


def score_frame(gt, pr, bgmask, hudmask, flags):
    """Object metrics for one frame at one GT level. flags: {gt_id: set(hard tags)}."""
    pr = pr.astype(np.int64)
    psize = collections.Counter(dict(zip(*[x.tolist() for x in np.unique(pr[pr >= 0], return_counts=True)])))
    on_void = collections.Counter(dict(zip(*[x.tolist() for x in np.unique(pr[(pr >= 0) & (bgmask | hudmask)], return_counts=True)])))
    on_bg = collections.Counter(dict(zip(*[x.tolist() for x in np.unique(pr[(pr >= 0) & bgmask], return_counts=True)])))
    kept = {p for p, s in psize.items() if on_void[p] < 0.5 * s}
    fg = gt >= 0
    cont = contingency(gt, pr, fg)
    gsize = collections.Counter()
    for (g, p), c in cont.items():
        gsize[g] += c
    best = {}  # g -> (iou, p, inter)
    per_g = collections.defaultdict(list)
    for (g, p), c in cont.items():
        per_g[g].append((p, c))
        if p < 0 or p not in kept:
            continue
        iou = c / (gsize[g] + psize[p] - c)
        if g not in best or iou > best[g][0]:
            best[g] = (iou, p, c)
    # pred -> gt pixel share for merge test
    pred_g = collections.defaultdict(dict)
    for (g, p), c in cont.items():
        pred_g[p][g] = c
    r = collections.Counter()
    r["gt"] = len(gsize); r["pred"] = len(kept)
    r["pred_px"] = sum(psize[p] for p in kept); r["leak_px"] = sum(on_bg[p] for p in kept)
    hard = collections.Counter()
    for g, gs in gsize.items():
        iou = best.get(g, (0, None, 0))[0]
        r["tp50"] += iou >= 0.5; r["tp90"] += iou >= 0.9; r["exact"] += iou >= 0.999999
        parts = [p for p, c in per_g[g] if p >= 0 and c >= max(2, 0.1 * gs)]
        r["frag"] += len(parts) >= 2
        r["shatter"] += max([c for p, c in per_g[g] if p >= 0] or [0]) < 0.5 * gs  # no segment holds half of it
        r["segs_per_gt"] += len([p for p, c in per_g[g] if p >= 0])
        bp = max(per_g[g], key=lambda t: t[1])[0]
        if bp >= 0 and any(g2 != g and c >= 0.1 * gsize[g2] for g2, c in pred_g[bp].items() if g2 >= 0):
            r["merge"] += 1
        for t in flags.get(g, ()):
            hard[t + "_n"] += 1; hard[t + "_tp"] += iou >= 0.5
    ari, vi = ari_vi({k: v for k, v in cont.items()})
    return r, hard, ari, vi, best


def load_game(g):
    z = np.load(DATA / f"{g}.npz")
    recs = [json.loads(l) for l in open(DATA / f"{g}.jsonl")]
    return z["frames"], z["inst"], z["ui"], recs


def gt_levels(frame, inst, ui, rec):
    imv = np.where(ui >= 0, -2, inst).astype(np.int64)
    objs = rec["objects"]
    grp = group_map(imv, objs)
    bg = imv == -1; hud = imv == -2
    # hard flags per instance
    fl = {}
    scale = rec["camera"]["scale"]
    for k in np.unique(imv):
        if k < 0:
            continue
        k = int(k); m = imv == k; f = set()
        if len(np.unique(frame[m])) > 1:
            f.add("multicolour")
        ys, xs = np.nonzero(m)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = ys + dy, xs + dx
            ok = (ny >= 0) & (ny < 64) & (nx >= 0) & (nx < 64)
            nb = imv[ny[ok], nx[ok]]; same = frame[ny[ok], nx[ok]] == frame[ys[ok], xs[ok]]
            hit = same & (nb >= 0) & (nb != k)
            if hit.any() and any(objs[int(q)]["design"] != objs[k]["design"] for q in np.unique(nb[hit])):
                f.add("touchdiff")
        if scale > 1:
            f.add("scaled")
        if rec["game"] in TILE_GAMES:
            f.add("tilegame")
        fl[k] = f
    gfl = collections.defaultdict(set)
    for k, f in fl.items():
        gfl[int(grp[imv == k][0])] |= f
    return imv, grp, bg, hud, fl, gfl


def vmeasure(pairs):
    """pairs: list of (pred_name, gt_design). Returns homogeneity, completeness, V."""
    n = len(pairs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    cont = collections.Counter(pairs)
    a = collections.Counter(p for p, _ in pairs); b = collections.Counter(d for _, d in pairs)
    H = lambda cnt: -sum(c / n * math.log(c / n) for c in cnt.values())
    Hc, Hk = H(b), H(a)
    Hck = -sum(c / n * math.log(c / a[p]) for (p, d), c in cont.items())
    Hkc = -sum(c / n * math.log(c / b[d]) for (p, d), c in cont.items())
    hom = 1.0 if Hc == 0 else 1 - Hck / Hc
    com = 1.0 if Hk == 0 else 1 - Hkc / Hk
    v = 0.0 if hom + com == 0 else 2 * hom * com / (hom + com)
    return hom, com, v


def run(cand, games, threads=1, stride=1):
    mod = importlib.import_module(cand)
    res = {}
    for g in games:
        frames, inst, ui, recs = load_game(g)
        if hasattr(mod, "setup"):
            mod.setup(g)
        state = None; prev = None
        agg = {lvl: collections.Counter() for lvl in ("inst", "group")}
        hard = {lvl: collections.Counter() for lvl in ("inst", "group")}
        aris = {lvl: [] for lvl in ("inst", "group")}; vis = {lvl: [] for lvl in ("inst", "group")}
        times = []; names_pairs = []; cell = collections.Counter()
        for i in range(0, len(frames), stride):
            fr = frames[i].astype(np.int64)
            t0 = time.perf_counter()
            lab, names, roles, state = mod.segment(fr, prev=prev, state=state)
            dt = time.perf_counter() - t0
            times.append(getattr(mod, "TIMED", None) if getattr(mod, "TIMED", None) is not None else dt)
            prev = fr
            lab = np.asarray(lab, dtype=np.int64)
            rec = recs[i]
            imv, grp, bg, hud, fl, gfl = gt_levels(fr, inst[i], ui[i], rec)
            for lvl, gt, f in (("inst", imv, fl), ("group", grp, gfl)):
                r, h, ari, vi, best = score_frame(gt, lab, bg, hud, f)
                agg[lvl].update(r); hard[lvl].update(h); aris[lvl].append(ari); vis[lvl].append(vi)
                if lvl == "inst" and names:
                    for k, (iou, p, _) in best.items():
                        if iou >= 0.5 and p in names:
                            names_pairs.append((str(names[p]), rec["objects"][k]["design"]))
            if isinstance(state, dict) and "cell" in state:
                gs = rec["camera"]["scale"]
                cell["n"] += 1; cell["ok"] += int(state["cell"] == gs)
        out = {}
        for lvl in ("inst", "group"):
            a = agg[lvl]
            P = a["tp50"] / max(a["pred"], 1); R = a["tp50"] / max(a["gt"], 1)
            out[lvl] = dict(P50=P, R50=R, F50=0 if P + R == 0 else 2 * P * R / (P + R),
                            R90=a["tp90"] / max(a["gt"], 1), P90=a["tp90"] / max(a["pred"], 1),
                            exact=a["exact"] / max(a["gt"], 1), frag=a["frag"] / max(a["gt"], 1), shatter=a["shatter"] / max(a["gt"], 1),
                            merge=a["merge"] / max(a["gt"], 1), segs_per_gt=a["segs_per_gt"] / max(a["gt"], 1),
                            overseg=a["pred"] / max(a["gt"], 1), leak=a["leak_px"] / max(a["pred_px"], 1),
                            ari=float(np.mean(aris[lvl])), vi=float(np.mean(vis[lvl])), gt=a["gt"], pred=a["pred"], tp50=a["tp50"],
                            hard={t: [hard[lvl][t + "_tp"], hard[lvl][t + "_n"]] for t in ("touchdiff", "multicolour", "scaled", "tilegame")})
        hom, com, v = vmeasure(names_pairs)
        ts = np.array(times) * 1000
        out["names"] = dict(hom=hom, com=com, v=v, n=len(names_pairs))
        out["ms"] = dict(median=float(np.median(ts)), p95=float(np.percentile(ts, 95)), max=float(ts.max()))
        if cell["n"]:
            out["cell_acc"] = cell["ok"] / cell["n"]
        res[g] = out
        print(f"{g} inst F50={out['inst']['F50']:.3f} grp F50={out['group']['F50']:.3f} frag={out['inst']['frag']:.2f} merge={out['inst']['merge']:.2f} ms={out['ms']['median']:.1f}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cand"); ap.add_argument("--games", default=",".join(ALL)); ap.add_argument("--out")
    ap.add_argument("--threads", type=int, default=1); ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    sys.path.insert(0, str(HERE))
    res = run(a.cand, a.games.split(","), a.threads, a.stride)
    out = a.out or str(HERE / "results" / f"{a.cand}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)
