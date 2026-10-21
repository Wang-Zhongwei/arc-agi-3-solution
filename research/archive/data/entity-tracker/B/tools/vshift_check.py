"""View-shift removal check on scrolling games outside the role benchmark (lf52, bp35): on steps with a detected
global shift, how many components were 'static in world' vs credited with motion, and whether any co-motion links
or a controlled claim formed from board components. Compare with view-shift removal disabled."""
import sys, json, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tracker as TR
import eval_tracker as E
for g in ("lf52", "bp35"):
    assert g not in E.SPLIT["dev"] and g not in E.SPLIT["heldout"]
    z = np.load(E.FULL / f"{g}.npz"); meta = json.load(open(E.FULL / f"{g}.meta.json"))
    F, L = z["frames"].astype(np.int64), z["level"]
    for label, params in (("view-shift on", {}), ("view-shift off", {"VS_MIN_CHANGED": 10**9})):
        tr = TR.EntityTracker(params)
        st = collections.Counter(); biggest_claim = 0; shifts = []
        orig = tr._link
        def hook(P_, moved, t, tr=tr, st=st, orig=orig):
            before = len(tr.links)
            orig(P_, moved, t)
            if tr._big:
                st["links_on_big_steps"] += len(tr.links) - before
                st["moved_comps_on_big_steps"] += sum(1 for d in moved.values() if d not in (None, (0, 0)))
                st["static_comps_on_big_steps"] += sum(1 for d in moved.values() if d == (0, 0))
        tr._link = hook
        for i in range(len(F)):
            nl = i == 0 or L[i] != L[i - 1]
            big = i > 0 and not nl and int((F[i] != F[i - 1]).sum()) > 600
            tr._big = big
            tr.update(F[i], meta[i]["act"], meta[i]["click"], new_level=nl)
            if big:
                st["big_change_steps"] += 1
                if tr.view != (0, 0):
                    st["shift_detected"] += 1; shifts.append(tr.view)
                m, conf = tr.claim()
                if m is not None:
                    biggest_claim = max(biggest_claim, int(m.sum()))
        print(g, label, dict(st), "shifts", collections.Counter(shifts).most_common(4), "largest claim px on those steps", biggest_claim)
