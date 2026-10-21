"""Step-level trace of tracker A on one game level (dev games only)."""
import sys, numpy as np
from util import load, SPLIT, show
import tracker as TA
from eval_tracker import gt_ctrl_mask, moved_ctrl_keys

g, lvsel = sys.argv[1], int(sys.argv[2]); nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 40
pass  # post-hoc (held-out already scored)
d = load(g); fr, level, meta = d["frames"], d["level"], d["meta"]
T = TA.Tracker()
T.start_level(fr[0]); k = 0
for t in range(1, len(fr)):
    if level[t] != level[t - 1]:
        T.start_level(fr[t]); continue
    T.update(meta[t]["act"], meta[t]["click"], fr[t])
    if level[t] != lvsel:
        continue
    k += 1
    if k > nmax:
        break
    p = T.player(); pm = T.mask(p) if p else None; gtm = gt_ctrl_mask(d, t)
    ok = pm is not None and pm.any() and gtm.any() and (pm & gtm).sum() >= 0.5 * pm.sum()
    ctrl = T.controllable()
    top = sorted(T.ents.values(), key=lambda e: -(e.agree))[:4]
    desc = " ".join(f"E{e.eid}(c{e.colour},a{e.agree}/o{e.opp}/p{e.perp}/s{e.still}{'*' if e.eid in ctrl else ''}{'' if e.comp is None else f'@{e.comp.y0},{e.comp.x0} n{e.comp.area}'})" for e in top)
    gys, gxs = np.nonzero(gtm)
    print(f"{t} {meta[t]['act']:7s} vs={T.view_shift} P=E{p} sel=E{T.selected}{'?' if T.selected_tentative else ''} ok={ok} gt@{(gys.min(), gxs.min()) if len(gys) else None} mv={ {k: v for k, v in list(T.moved_now.items())[:5]} } | {desc} | {T.tail_line()}")
