import json, sys, collections
import numpy as np
V = "/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/vision"
SPLIT = json.load(open(__file__.rsplit("/", 2)[0] + "/split.json"))
R = 8
for g in (sys.argv[1:] or SPLIT["dev"]):
    z = np.load(f"{V}/bench/full/{g}.npz"); meta = json.load(open(f"{V}/bench/full/{g}.meta.json"))
    F = z["frames"].astype(np.int16); L = z["level"]
    cnt = collections.Counter(); ex = []
    for t in range(1, len(F)):
        if L[t] != L[t-1]: continue
        a, b = F[t-1], F[t]
        d0 = int((a != b).sum())
        if d0 < 20: continue
        best = (d0, 0, 0)
        for dy in range(-R, R+1):
            for dx in range(-R, R+1):
                if dy == 0 and dx == 0: continue
                ya, yb = max(0, -dy), min(64, 64-dy); xa, xb = max(0, -dx), min(64, 64-dx)
                # new[y+dy, x+dx] == old[y, x]
                ov = (yb-ya)*(xb-xa)
                mism = int((a[ya:yb, xa:xb] != b[ya+dy:yb+dy, xa+dx:xb+dx]).sum())
                # normalise to full-frame mismatch estimate
                m = mism * 4096 / ov
                if m < best[0]: best = (m, dy, dx)
        if best[1:] != (0, 0) and best[0] < 0.5 * d0:
            cnt[(best[1], best[2])] += 1; ex.append((t, meta[t]['act'], d0, int(best[0]), best[1:]))
    print(g, 'steps', len(F), 'shift steps', sum(cnt.values()), dict(cnt.most_common(6)), ex[:5])
