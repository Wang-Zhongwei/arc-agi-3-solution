"""Synthetic camera-follow test (dev games only): translate every frame (and GT maps) so the GT player stays at its
level-start screen position, i.e. the board scrolls under a centred player. Compares player metrics with the
unshifted run. Blocker/vanish GT checks assume a fixed camera, so only player/selected metrics are reported."""
import sys, json
import numpy as np
import util
import eval_tracker as E

games = sys.argv[1].split(",")
assert all(g in util.SPLIT["dev"] for g in games)
orig_load = util.load


def shift(a, dy, dx, fill):
    out = np.full_like(a, fill)
    ys, yd = (slice(0, 64 - dy), slice(dy, 64)) if dy >= 0 else (slice(-dy, 64), slice(0, 64 + dy))
    xs, xd = (slice(0, 64 - dx), slice(dx, 64)) if dx >= 0 else (slice(-dx, 64), slice(0, 64 + dx))
    out[yd, xd] = a[ys, xs]
    return out


def load_cam(g):
    d = orig_load(g)
    fr, inst, ui, lv = d["frames"].copy(), d["inst"].copy(), d["ui"].copy(), d["level"]
    off = (0, 0); c0 = None; bg = 0
    for t in range(len(fr)):
        if t == 0 or lv[t] != lv[t - 1]:
            c0 = None; off = (0, 0)
            v, c = np.unique(d["frames"][t], return_counts=True); bg = int(v[c.argmax()])
        m = E.gt_ctrl_mask(d, t)
        if m.any():
            ys, xs = np.nonzero(m); c = (ys.mean(), xs.mean())
            if c0 is None:
                c0 = c
            off = (int(round(c[0] - c0[0])), int(round(c[1] - c0[1])))
        dy, dx = -off[0], -off[1]
        fr[t] = shift(d["frames"][t], dy, dx, bg); inst[t] = shift(d["inst"][t], dy, dx, -1); ui[t] = shift(d["ui"][t], dy, dx, -1)
    d["frames"], d["inst"], d["ui"] = fr, inst, ui
    return d


E.load = load_cam
res = {}
for g in games:
    res[g] = E.run_game(g, lambda: E.AWrap())
s = E.summarize(res, games)
print(json.dumps({k: s[k] for k in ["levels", "final_ok", "within5", "wrong_claim_rate", "ext_within5", "selected_step_acc", "latency_ms_p50", "latency_ms_p95"]}))
print({g: [l["commit"] for l in res[g]["levels"]] for g in games})
