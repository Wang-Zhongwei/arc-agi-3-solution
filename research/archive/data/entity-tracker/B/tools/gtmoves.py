"""Dev-only: per step, list GT sprites that changed position/visibility (screen bbox) and the action."""
import json, sys
import numpy as np
V = "/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/vision"
SPLIT = json.load(open(__file__.rsplit("/", 2)[0] + "/split.json"))
g = sys.argv[1]; assert g in SPLIT["dev"] or g not in SPLIT["heldout"]
t0, t1 = int(sys.argv[2]), int(sys.argv[3])
traj = json.load(open(f"{V}/gt/data/{g}.traj.json")); meta = json.load(open(f"{V}/bench/full/{g}.meta.json"))
R = {n: r for n, r, _, _ in json.load(open(f"{V}/gt/roles_behaviour.json"))[g]}
for t in range(max(1, t0), t1 + 1):
    a, b = traj[t-1]["state"], traj[t]["state"]
    ch = []
    for k, sb in b.items():
        sa = a.get(k)
        if sa is None: ch.append(f"+{k[:10]}"); continue
        if (sa[0], sa[1]) != (sb[0], sb[1]): ch.append(f"{k[:10]}{'*' if 'controlled(dir)' in R.get(k.rsplit('#',1)[0], []) else ''} d=({sb[1]-sa[1]},{sb[0]-sa[0]})")
        elif sa[3] != sb[3]: ch.append(f"{k[:10]} vis {sa[3]}->{sb[3]}")
        elif sa[2] != sb[2] or sa[5] != sb[5]: ch.append(f"{k[:10]} look")
    for k in a:
        if k not in b: ch.append(f"-{k[:10]}")
    print(t, traj[t]["level"], meta[t]["act"], meta[t]["click"], "; ".join(ch[:8]) + (f" ...(+{len(ch)-8})" if len(ch) > 8 else ""))
