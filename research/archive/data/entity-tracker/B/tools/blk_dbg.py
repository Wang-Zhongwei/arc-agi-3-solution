import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval_tracker as E
g = sys.argv[1]; assert g in E.SPLIT["dev"]
whyw = sys.argv[2]; lim = int(sys.argv[3]) if len(sys.argv) > 3 else 3
gt = E.GT(g); out = E.run_tracker(gt); E.attach_abs(out)
H = "0123456789abcdef"; n = 0
for t, o in enumerate(out):
    for f in o["new_facts"]:
        if f["type"] != "blocked": continue
        ok, strict, why = E.gt_blocker(gt, f)
        if why != whyw: continue
        n += 1
        if n > lim: break
        mm, bm = f["mover_mask"], f["blocker_mask"]; u = mm | bm; ys, xs = np.nonzero(u)
        y0, y1, x0, x1 = max(ys.min()-3, 0), min(ys.max()+4, 64), max(xs.min()-3, 0), min(xs.max()+4, 64)
        print(f"t={t} act={f['action']} why={why} n={f['n']} mover={int(mm.sum())} blocker={int(bm.sum())}")
        for y in range(y0, y1):
            print("  " + "".join(("M" if mm[y, x] else "B" if bm[y, x] else H[gt.frames[t][y, x]]) for x in range(x0, x1)) + "   " + "".join(H[gt.frames[t][y, x]] for x in range(x0, x1)))
