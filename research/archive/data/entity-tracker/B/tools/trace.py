"""Dev-only step trace of the tracker: action, moved entities, selection, claim correctness, new facts."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tracker as TR
import eval_tracker as E
g = sys.argv[1]; assert g in E.SPLIT["dev"] or g not in E.SPLIT["heldout"]
t0, t1 = int(sys.argv[2]), int(sys.argv[3])
params = {k: float(v) if "." in v else int(v) for k, v in (x.split("=") for x in sys.argv[4:])}
gt = E.GT(g)
tr = TR.EntityTracker(params)
orig = tr._after
def hook(action, moved, kinds, t):
    ents = tr.entities()
    mv = []
    for r, mem in ents.items():
        d = tr._ent_motion(mem, moved)
        ks = {kinds.get(m) for m in mem}
        if (d not in (None, (0, 0))) or (ks - {"same", None}):
            c = tr.mask_of(mem); ys, xs = np.nonzero(c)
            if len(ys) == 0: continue
            ag, mvd, pr = tr._controllable(mem)
            mv.append(f"r{r}[{len(mem)}c n{int(c.sum())} @({ys.min()},{xs.min()}) d={d} k={','.join(sorted(k for k in ks if k))} st={ag}/{mvd}/{pr}]")
    hook.info = mv
    return orig(action, moved, kinds, t)
tr._after = hook
for i in range(t1 + 1):
    newlv = i == 0 or gt.level[i] != gt.level[i - 1]
    m = gt.meta[i]
    hook.info = []
    tr.update(gt.frames[i], m["act"], m["click"], new_level=newlv)
    if i < t0: continue
    cm, conf = tr.claim()
    gm = gt.ctrl_mask(i)
    ok = E.on(cm, gm)
    sel = tr._controlled()
    print(f"{i} L{gt.level[i]} {m['act']} {m['click'] or ''} vs={tr.view} sel={'r%d' % sel[0] if sel else None}:{conf} ok={ok} gtmv={sorted(gt.movers.get(i, []))[:2]} | " + " ".join(hook.info[:6]) + (f" ...+{len(hook.info)-6}" if len(hook.info) > 6 else ""))
    for e in tr.events:
        print("     EVENT", e)
