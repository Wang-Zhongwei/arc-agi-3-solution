"""Show a dev blocker claim: frame crop with player (P) / blocker (B) / target (*), plus GT facts."""
import sys, json, numpy as np
from util import load, SPLIT, HEX
import eval_tracker as E
g = sys.argv[1]; ts = [int(x) for x in sys.argv[2].split(",")]
assert g in SPLIT["dev"]
d = load(g)
w = E.AWrap(); fr, lv, meta = d["frames"], d["level"], d["meta"]
w.start_level(fr[0], 0)
for t in range(1, len(fr)):
    if lv[t] != lv[t - 1]:
        w.start_level(fr[t], t); continue
    w.update(meta[t]["act"], meta[t]["click"], fr[t])
w._flush()
for ev in w.blocker_log:
    if ev["t"] not in ts:
        continue
    t = ev["t"]; pm, bm = ev["player_mask"], ev["blocker_mask"]
    ys, xs = np.nonzero(pm | bm)
    r0, r1, c0, c1 = max(ys.min() - 4, 0), min(ys.max() + 5, 64), max(xs.min() - 4, 0), min(xs.max() + 5, 64)
    ady, adx = ev["dir"]; st = ev["step"]
    py, px = np.nonzero(pm); tg = np.zeros((64, 64), bool); ok = (py + ady * st >= 0) & (py + ady * st < 64) & (px + adx * st >= 0) & (px + adx * st < 64)
    tg[py[ok] + ady * st, px[ok] + adx * st] = True
    print(f"t={t} act={meta[t]['act']} player=E{ev['player']} blocker=E{ev['blocker']} step={st} count={ev['count']}")
    for r in range(r0, r1):
        row = ""
        for c in range(c0, c1):
            ch = HEX[fr[t][r, c]]
            if pm[r, c]: ch = "P"
            elif bm[r, c] and tg[r, c]: ch = "#"
            elif bm[r, c]: ch = "B"
            elif tg[r, c]: ch = "*"
            row += ch
        print("  " + row + "   " + "".join(HEX[v] for v in fr[t - 1][r, c0:c1]))
    iid = E.dominant_inst(d, t, pm); key = E.inst_key(d, t, iid)
    sa, sb = d["traj"][t - 1]["state"].get(key), d["traj"][t]["state"].get(key)
    print("  GT player key", key, "pos", sa[:2] if sa else None, "->", sb[:2] if sb else None, "| blocker inst", E.dominant_inst(d, t, bm), E.inst_key(d, t, E.dominant_inst(d, t, bm)))
