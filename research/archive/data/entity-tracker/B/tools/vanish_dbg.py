import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval_tracker as E
g = sys.argv[1]; assert g in E.SPLIT["dev"]
want = {int(x) for x in sys.argv[2].split(",")}
gt = E.GT(g); out = E.run_tracker(gt); E.attach_abs(out)
H = "0123456789abcdef"
for t, o in enumerate(out):
    for f in o["new_facts"]:
        if f["type"] != "vanished" or t not in want: continue
        ok, why = E.gt_vanish(gt, f)
        ts = f["at_abs"]; vm = f["mask"]; ys, xs = np.nonzero(vm)
        y0, y1, x0, x1 = max(ys.min()-4, 0), min(ys.max()+5, 64), max(xs.min()-6, 0), min(xs.max()+7, 64)
        print(f"claim t={t} vanish step={ts} why={why} n={int(vm.sum())} act={gt.meta[ts]['act']}")
        for tt in (ts - 1, ts, t):
            print(" frame", tt, gt.meta[tt]["act"])
            for y in range(y0, y1):
                print("  " + "".join((H[gt.frames[tt][y, x]].upper() if vm[y, x] else H[gt.frames[tt][y, x]]) for x in range(x0, x1)))
