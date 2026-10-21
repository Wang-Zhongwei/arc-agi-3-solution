"""Dev-only viewer: print frames (hex palette) for a game step range, marking changed pixels."""
import json, sys
import numpy as np
V = "/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/vision"
SPLIT = json.load(open(__file__.rsplit("/", 2)[0] + "/split.json"))
g = sys.argv[1]
assert g in SPLIT["dev"], "viewer is dev-only"
t0, t1 = int(sys.argv[2]), int(sys.argv[3])
crop = sys.argv[4] if len(sys.argv) > 4 else None
z = np.load(f"{V}/bench/full/{g}.npz"); meta = json.load(open(f"{V}/bench/full/{g}.meta.json"))
F = z["frames"]; L = z["level"]
H = "0123456789abcdef"
for t in range(t0, t1 + 1):
    d = F[t] != F[t - 1] if t > 0 else np.zeros((64, 64), bool)
    print(f"--- t={t} act={meta[t]['act']} click={meta[t]['click']} level={L[t]} changed={int(d.sum())}")
    r0, r1, c0, c1 = (0, 64, 0, 64) if not crop else map(int, crop.split(","))
    for y in range(r0, r1):
        print("".join((H[F[t][y, x]].upper() if d[y, x] else H[F[t][y, x]]) if not d[y, x] else ("*" if False else H[F[t][y, x]].upper() if H[F[t][y, x]].isalpha() else "!" if False else H[F[t][y, x]]) for x in range(c0, c1)) + "   " + "".join("#" if d[y, x] else "." for x in range(c0, c1)))
