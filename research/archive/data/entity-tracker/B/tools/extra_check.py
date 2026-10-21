"""False-claim check on extra-dev games (outside both splits) under gt_arrow, per level."""
import sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval_tracker as E
games = sys.argv[1].split(",") if len(sys.argv) > 1 else ["ar25", "bp35", "lf52", "tr87"]
params = {k: float(v) if "." in v else int(v) for k, v in (x.split("=") for x in sys.argv[2:])}
tot_n = tot_w = 0
for g in games:
    assert g not in E.SPLIT["dev"] and g not in E.SPLIT["heldout"]
    gt = E.GT(g); out = E.run_tracker(gt, params)
    per = collections.defaultdict(lambda: [0, 0, 0])
    for t in range(1, len(out)):
        c = out[t]["claim"]
        if c is None or not c.any(): continue
        ga = gt.arrow_mask(t) | gt.ctrl_mask(t)
        L = int(gt.level[t])
        if not ga.any(): per[L][2] += 1; continue
        per[L][0] += 1; per[L][1] += not E.on(c, ga)
    n = sum(v[0] for v in per.values()); w = sum(v[1] for v in per.values())
    tot_n += n; tot_w += w
    print(g, f"wrong {w}/{n}", {L: f"{v[1]}/{v[0]} (+{v[2]} n/a)" for L, v in sorted(per.items())}, "arrow keys", {k: sorted(v) for k, v in gt.arrow_keys.items() if k in (1, 3)} if g == "tr87" else "")
print(f"total wrong {tot_w}/{tot_n}")
