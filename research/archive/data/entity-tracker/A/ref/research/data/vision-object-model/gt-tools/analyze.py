"""Summarise the ground-truth set: coverage, designs, segmentation difficulty, behavioural roles."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path(sys.argv[1])
SPLIT = 1  # instance considered "visible" with >= SPLIT pixels
DIRS = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}


def cc_label(mask_colour):
    """4-connected same-colour components of a 64x64 frame (pure numpy/python BFS)."""
    H, W = mask_colour.shape
    lab = -np.ones((H, W), dtype=np.int32)
    n = 0
    for y in range(H):
        for x in range(W):
            if lab[y, x] >= 0:
                continue
            c = mask_colour[y, x]
            stack = [(y, x)]
            lab[y, x] = n
            while stack:
                yy, xx = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = yy + dy, xx + dx
                    if 0 <= ny < H and 0 <= nx < W and lab[ny, nx] < 0 and mask_colour[ny, nx] == c:
                        lab[ny, nx] = n
                        stack.append((ny, nx))
            n += 1
    return lab, n


def roles(traj):
    """Behavioural role per sprite name from the full replay trajectory."""
    stats = collections.defaultdict(lambda: collections.Counter())
    for a, b in zip(traj, traj[1:]):
        if a["level"] != b["level"] or b["act"] == "RESET":
            continue
        act = b["act"]
        for key, sb in b["state"].items():
            name = key.rsplit("#", 1)[0]
            sa = a["state"].get(key)
            st = stats[name]
            if sa is None:
                st["appear"] += 1
                continue
            (xa, ya, pa, va, na, ra), (xb, yb, pb, vb, nb, rb) = sa, sb
            st["steps"] += 1
            st["act_" + act] += 1
            dx, dy = xb - xa, yb - ya
            if dx or dy:
                st["moved"] += 1
                st["moved_" + act] += 1
                if act in DIRS:
                    ex, ey = DIRS[act]
                    if (np.sign(dx), np.sign(dy)) == (ex, ey):
                        st["agree"] += 1
            if pa != pb:
                st["repaint"] += 1
            if ra != rb:
                st["rotate"] += 1
            if va != vb:
                st["vis_toggle"] += 1
        for key in a["state"]:
            if key not in b["state"]:
                stats[key.rsplit("#", 1)[0]]["vanish"] += 1
    return stats


def main():
    games = sorted(p.stem for p in DATA.glob("*.npz"))
    tot = collections.Counter()
    rows = []
    role_rows = {}
    for g in games:
        z = np.load(DATA / f"{g}.npz")
        recs = [json.loads(l) for l in open(DATA / f"{g}.jsonl")]
        frames, inst, ui = z["frames"], z["inst"], z["ui"]
        N = len(frames)
        px_sprite = (inst >= 0) & (ui < 0)
        px_ui = ui >= 0
        designs_vis = set()
        names_vis = set()
        designs_all = set()
        n_inst = []
        touch_same = 0
        touch_bg = 0
        touch_diff = 0
        exact = 0
        n_vis_obj = 0
        frag = 0  # GT objects split into >1 colour-CC
        merged = 0  # GT objects whose colour-CCs also contain another sprite instance's pixels
        multicolour = 0
        cc_counts = []
        scales = collections.Counter()
        for f in range(N):
            objs = recs[f]["objects"]
            fr, im = frames[f], inst[f]
            scales[recs[f]["camera"]["scale"]] += 1
            # pixel map with UI pixels removed from instances (UI drawn on top)
            imv = np.where(ui[f] >= 0, -2, im)
            lab, ncc = cc_label(fr)
            cc_counts.append(ncc)
            vis = [o for o in objs if (imv == o["k"]).sum() >= SPLIT]
            n_inst.append(len(vis))
            for o in objs:
                designs_all.add(o["design"])
            for o in vis:
                designs_vis.add(o["design"])
                names_vis.add(o["name"])
                m = imv == o["k"]
                n_vis_obj += 1
                cols = np.unique(fr[m])
                if len(cols) > 1:
                    multicolour += 1
                # touching a different sprite instance (or bare background) of the same colour (4-adjacency)
                ys0, xs0 = np.nonzero(m)
                t_obj = t_bg = t_diff = False
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = ys0 + dy, xs0 + dx
                    ok = (ny >= 0) & (ny < 64) & (nx >= 0) & (nx < 64)
                    ys, xs, ny, nx = ys0[ok], xs0[ok], ny[ok], nx[ok]
                    same = fr[ny, nx] == fr[ys, xs]
                    nb = imv[ny, nx]
                    hit = same & (nb >= 0) & (nb != o["k"])
                    t_obj |= bool(hit.any())
                    if hit.any():
                        t_diff |= any(objs[int(q)]["design"] != o["design"] for q in np.unique(nb[hit]))
                    t_bg |= bool((same & (nb == -1)).any())
                touch_same += t_obj
                touch_bg += t_bg
                touch_diff += t_diff
                ccs = np.unique(lab[m])
                if len(ccs) > 1:
                    frag += 1
                cc_px = np.isin(lab, ccs)
                oth = cc_px & ~m
                if (imv[oth] >= 0).any():
                    merged += 1
                if len(ccs) == 1 and not oth.any():
                    exact += 1
        traj = json.load(open(DATA / f"{g}.traj.json"))
        st = roles(traj)
        role_rows[g] = st
        cov_s = px_sprite.mean()
        cov_u = px_ui.mean()
        rows.append(dict(game=g, frames=N, inst_per_frame=float(np.mean(n_inst)), cc_per_frame=float(np.mean(cc_counts)),
                         designs_visible=len(designs_vis), names_visible=len(names_vis), designs_all=len(designs_all),
                         sprite_px=float(cov_s), ui_px=float(cov_u), bg_px=float(1 - cov_s - cov_u),
                         multicolour=multicolour / max(n_vis_obj, 1), touch_same=touch_same / max(n_vis_obj, 1), touch_bg=touch_bg / max(n_vis_obj, 1), touch_diff=touch_diff / max(n_vis_obj, 1),
                         frag=frag / max(n_vis_obj, 1), merged=merged / max(n_vis_obj, 1), exact=exact / max(n_vis_obj, 1), vis_obj=n_vis_obj,
                         scales=dict(scales), ui_names=sorted({n for r in recs for n in r["ui_names"]})))
        tot["vis_obj"] += n_vis_obj
        tot["touch"] += touch_same
        tot["touch_bg"] += touch_bg
        tot["touch_diff"] += touch_diff
        tot["frag"] += frag
        tot["merged"] += merged
        tot["exact"] += exact
        tot["multi"] += multicolour
        tot["frames"] += N
    json.dump(rows, open(DATA.parent / "summary_games.json", "w"), indent=1)
    hdr = "game  frames inst/fr cc/fr designs(vis/all) names spr%  ui%  bg%  multicol touchObj touchBg touchDiffDesign frag merged exact scales"
    print(hdr)
    for r in rows:
        print(f"{r['game']}  {r['frames']:3d}  {r['inst_per_frame']:6.1f} {r['cc_per_frame']:6.1f}  {r['designs_visible']:4d}/{r['designs_all']:4d}   {r['names_visible']:4d} "
              f"{100*r['sprite_px']:4.0f} {100*r['ui_px']:4.0f} {100*r['bg_px']:4.0f}  {100*r['multicolour']:5.0f}%  {100*r['touch_same']:5.0f}% {100*r['touch_bg']:4.0f}% {100*r['touch_diff']:4.0f}%  {100*r['frag']:4.0f}% {100*r['merged']:4.0f}% {100*r['exact']:4.0f}%  {r['scales']}")
    v = tot["vis_obj"]
    print(f"TOTAL frames={tot['frames']} visible objects={v} multicolour={tot['multi']/v:.1%} touch_same_obj={tot['touch']/v:.1%} touch_same_bg={tot['touch_bg']/v:.1%} touch_same_obj_other_design={tot['touch_diff']/v:.1%} "
          f"colourCC-fragmented={tot['frag']/v:.1%} colourCC-merged={tot['merged']/v:.1%} colourCC-exact={tot['exact']/v:.1%}")

    # behavioural roles
    out = {}
    for g, st in role_rows.items():
        lines = []
        for name, c in st.items():
            dir_steps = sum(c["act_" + a] for a in DIRS)
            dir_moves = sum(c["moved_" + a] for a in DIRS)
            r = []
            if dir_steps and dir_moves / dir_steps > 0.3 and c["agree"] / max(dir_moves, 1) > 0.6:
                r.append("controlled(dir)")
            elif c["moved"] and c["steps"] and c["moved"] / c["steps"] > 0.05:
                r.append("mover")
            if c["moved_ACTION6"] and c["act_ACTION6"] and c["moved_ACTION6"] / c["act_ACTION6"] > 0.05:
                r.append("moves-on-click")
            if c["repaint"]:
                r.append(f"repaint({c['repaint']})")
            if c["rotate"]:
                r.append("rotates")
            if c["vis_toggle"]:
                r.append("vis-toggle")
            if c["appear"] or c["vanish"]:
                r.append(f"spawn/despawn({c['appear']}/{c['vanish']})")
            if r:
                lines.append((name, r, c["steps"], c["moved"]))
        out[g] = lines
    json.dump({g: [[n, r, s, m] for n, r, s, m in l] for g, l in out.items()}, open(DATA.parent / "roles_behaviour.json", "w"), indent=1)
    print()
    for g, l in out.items():
        ctrl = [n for n, r, _, _ in l if "controlled(dir)" in r]
        movers = [n for n, r, _, _ in l if "mover" in r]
        rep = [n for n, r, _, _ in l if any(x.startswith("repaint") for x in r)]
        click = [n for n, r, _, _ in l if "moves-on-click" in r]
        spawn = [n for n, r, _, _ in l if any(x.startswith("spawn") for x in r)]
        print(f"{g}: controlled={ctrl[:4]} movers={len(movers)} click-move={len(click)} repaint={len(rep)} spawn={len(spawn)}")


if __name__ == "__main__":
    main()
