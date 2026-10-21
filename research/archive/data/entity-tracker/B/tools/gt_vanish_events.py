"""List GT vanish-on-contact events: a sprite visible at t-1 is removed/invisible at t, its pixels at t-1 are
within 1 px of a controlled/arrow-driven sprite at t, and its pixels changed (not an invisible bookkeeping swap)."""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval_tracker as E
games = sys.argv[1].split(",")
for g in games:
    assert g in E.SPLIT["dev"] or g not in E.SPLIT["heldout"] or "--final" in sys.argv
    gt = E.GT(g); n = 0; ev = []
    for t in range(1, len(gt.frames)):
        if gt.level[t] != gt.level[t - 1]: continue
        a, b = gt.traj[t - 1]["state"], gt.traj[t]["state"]
        pm = gt.ctrl_mask(t) | gt.arrow_mask(t)
        near = ndimage.binary_dilation(pm, structure=np.ones((3, 3), bool))
        for k, key in gt.keys(t - 1).items():
            sa = a.get(key)
            if not sa or not sa[3] or sa[4] == 0: continue
            sb = b.get(key)
            if sb is not None and sb[3]: continue
            m = (gt.inst[t - 1] == k) & (gt.ui[t - 1] < 0)
            if not m.any() or not (m & near).any(): continue
            if not (gt.frames[t - 1][m] != gt.frames[t][m]).any(): continue
            ev.append((t, gt.meta[t]["act"], key[:14], int(m.sum())))
    print(g, len(ev), ev[:12])
