"""Ground-truth sprite dump for ARC-AGI-3 public games.

Loads a game's source locally (arcengine), replays the action sequence recorded
in a Kaggle run's <game>_p0_events.jsonl.xz, checks every rendered frame against
the recorded board, and at sampled steps dumps:
  frame   (64x64 palette indices, the board the agent saw)
  inst    (64x64 int16: index into that frame's object list, -1 = camera background/letterbox)
  ui      (64x64 int8: index of the RenderableUserDisplay that changed the pixel, -1 = none)
  objects (per sprite instance: name, tags, class, design hash, screen bbox, colours,
           pixel count, layer, visibility, collidability, ...)

Instance map trick: every sprite's `pixels` is temporarily swapped for an array whose
opaque cells hold an ID code, and the game's own camera._raw_render is called (twice:
low and high digits), so custom cameras, layer order, rotation, mirroring and
scaling are all reproduced exactly. HUD pixels are found by wrapping Camera.render
and diffing the frame before/after each interface.

Usage: python gt_dump.py --game ls20 --events <events.jsonl.xz> --envdir <environment_files> --out <dir> --n 40
"""

import argparse
import hashlib
import inspect
import json
import lzma
import re
import sys
from pathlib import Path

import numpy as np
import arcengine
from arcengine import ActionInput, Camera, GameAction

# ---------------------------------------------------------------- render capture
_CAPTURE = {}
_orig_render = Camera.render


def _capturing_render(self, sprites):
    """Same as Camera.render, but records the pre-UI frame and per-interface pixel diffs."""
    output = np.full((self.MAX_DIMENSION, self.MAX_DIMENSION), self._letter_box, dtype=np.int8)
    view = self._raw_render(sprites)
    scale, x_off, y_off = self._calculate_scale_and_offset()
    if scale > 1:
        view = np.repeat(np.repeat(view, scale, axis=0), scale, axis=1)
    output[y_off:y_off + view.shape[0], x_off:x_off + view.shape[1]] = view
    pre = output.copy()
    ui = np.full(output.shape, -1, dtype=np.int8)
    for i, interface in enumerate(self._interfaces):
        before = output.copy()
        output = interface.render_interface(output)
        ui[np.asarray(output) != before] = i
    _CAPTURE["pre"] = pre
    _CAPTURE["ui"] = ui
    _CAPTURE["ui_names"] = [type(it).__name__ for it in self._interfaces]
    _CAPTURE["camera"] = self
    return output


Camera.render = _capturing_render


def instance_map(game):
    """64x64 int16 map of sprite index (into level.get_sprites()) per pixel, -1 = none."""
    cam = game.camera
    sprites = game.current_level.get_sprites()
    saved = [s.pixels for s in sprites]
    saved_bg = cam._background
    passes = []
    try:
        cam._background = -1
        for digit in (0, 1):
            for k, s in enumerate(sprites):
                code = (k % 100) if digit == 0 else (k // 100)
                p = saved[k]
                s.pixels = np.where(p >= 0, code, p).astype(np.int8)
            passes.append(np.asarray(cam._raw_render(sprites)).astype(np.int16))
    finally:
        for s, p in zip(sprites, saved):
            s.pixels = p
        cam._background = saved_bg
    lo, hi = passes
    raw = np.where(lo >= 0, lo + 100 * hi, -1).astype(np.int16)
    scale, x_off, y_off = cam._calculate_scale_and_offset()
    if scale > 1:
        raw = np.repeat(np.repeat(raw, scale, axis=0), scale, axis=1)
    out = np.full((64, 64), -1, dtype=np.int16)
    out[y_off:y_off + raw.shape[0], x_off:x_off + raw.shape[1]] = raw
    return out, sprites, (scale, x_off, y_off)


def design_hash(pixels):
    p = np.asarray(pixels, dtype=np.int8)
    return hashlib.blake2b(p.tobytes() + str(p.shape).encode(), digest_size=6).hexdigest()


def sprite_record(k, s, inst, frame, scale, x_off, y_off, cam):
    rp = s.render()
    h, w = rp.shape
    sx = (s.x - cam.x) * scale + x_off
    sy = (s.y - cam.y) * scale + y_off
    m = inst == k
    n = int(m.sum())
    rec = {
        "k": k,
        "name": s.name,
        "tags": list(s.tags) if s.tags else [],
        "cls": type(s).__name__,
        "design": design_hash(s.pixels),
        "x": int(s.x), "y": int(s.y), "w": int(w), "h": int(h),
        "bbox_screen": [int(sx), int(sy), int(sx + w * scale), int(sy + h * scale)],  # x0,y0,x1,y1 (excl)
        "layer": int(s.layer),
        "rot": int(s.rotation), "scale": int(s.scale),
        "mirror": [bool(s.mirror_ud), bool(s.mirror_lr)],
        "visible": bool(s.is_visible),
        "collidable": bool(s.is_collidable),
        "blocking": getattr(s.blocking, "name", str(s.blocking)),
        "sprite_colours": sorted(int(c) for c in np.unique(rp[rp >= 0])),
        "n_px": n,
    }
    if n:
        ys, xs = np.nonzero(m)
        rec["bbox_vis"] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        rec["vis_colours"] = sorted(int(c) for c in np.unique(frame[m]))
    return rec



# ---------------------------------------------------------------- embedded-engine adapters
# bp35 and lf52 do not use arcengine sprites: a private scene graph is rasterised into a
# 64x64 buffer that a RenderableUserDisplay returns as the whole frame. We deep-copy the
# scene, swap each drawn entity's image for an ID-coded proxy and call the game's own
# rasteriser on the copy (the live game is never touched).
WORLD = {
    # game: (world getter, list fn, draw fn, image fn, dirty flag attr or None, per-entity cache attr or None, key attr, pos fn)
    "bp35": (lambda g: g.tehvqeiqsdu.qivnakklopy, "upiapwkxxz", "srlqyenmue", "ieikpxxuml", "hlfdukibtwe", None, "udwvbkynpbz", "knpqzpefyn"),
    "lf52": (lambda g: g.btiyglcumku.oegtnpbqims, "luszojnlqu", "vclswpkbjs", "sxwqiwdisg", "wuomahqexpl", "gmhhfvghrrq", "vwkvvkakmhh", "mepgityjcj"),
}


class _Proxy:
    def __init__(self, arr, fn):
        self._arr = arr
        setattr(self, fn, lambda: self._arr)


def world_instance_map(game, gid, frame):
    import copy
    getw, list_fn, draw_fn, img_fn, dirty, cache, key_attr, pos_fn = WORLD[gid]
    world = copy.deepcopy(getw(game))
    cls = type(world)
    items = getattr(world, list_fn)()
    imgs = [np.asarray(getattr(im, img_fn)()) for _, _, im in items]

    def draw(proxies):
        if dirty == "wuomahqexpl":
            world.wuomahqexpl = True
        if cache:
            for _, ent, _ in items:
                setattr(ent, cache, None)
        orig = getattr(cls, list_fn)
        try:
            if proxies is not None:
                setattr(cls, list_fn, lambda self: proxies if self is world else orig(self))
            return np.asarray(getattr(world, draw_fn)()).copy()
        finally:
            setattr(cls, list_fn, orig)

    colour = draw(None)
    passes = []
    for digit in (0, 1):  # codes start at 16 so they never collide with the background palette index
        proxies = []
        for k, ((lay, ent, im), a) in enumerate(zip(items, imgs)):
            code = 16 + ((k % 100) if digit == 0 else (k // 100))
            proxies.append((lay, ent, _Proxy(np.where(a >= 0, code, a).astype(np.int8), img_fn)))
        passes.append(draw(proxies).astype(np.int16))
    lo, hi = passes
    inst = np.where(lo >= 16, (lo - 16) + 100 * (hi - 16), -1).astype(np.int16)
    hud = colour != frame  # HUD bars the display draws over the scene raster
    inst[hud] = -1
    objs = []
    for k, ((lay, ent, im), a) in enumerate(zip(items, imgs)):
        m = inst == k
        n = int(m.sum())
        key = getattr(ent, key_attr, None)
        x, y = getattr(ent, pos_fn)()
        rec = {"k": k, "name": key if isinstance(key, str) else f"{type(ent).__name__}@{type(im).__name__}",
               "tags": [], "cls": type(ent).__name__, "design": design_hash(a), "x": float(x), "y": float(y),
               "w": int(a.shape[1]) if a.ndim == 2 else 0, "h": int(a.shape[0]) if a.ndim == 2 else 0,
               "layer": int(lay) if isinstance(lay, (int, np.integer)) else 0, "rot": 0, "scale": 1, "visible": True, "collidable": None,
               "sprite_colours": sorted(int(c) for c in np.unique(a[a >= 0])) if a.size else [], "n_px": n, "engine": gid}
        if n:
            ys, xs = np.nonzero(m)
            rec["bbox_vis"] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            rec["bbox_screen"] = rec["bbox_vis"]
            rec["vis_colours"] = sorted(int(c) for c in np.unique(frame[m]))
        objs.append(rec)
    # every instance pixel's frame colour must be one of that entity's image colours
    bad = 0
    for o in objs:
        if o["n_px"]:
            bad += int((~np.isin(frame[inst == o["k"]], o["sprite_colours"])).sum())
    return inst, objs, hud, bad


# ---------------------------------------------------------------- game loading / replay
def load_game(envdir, game):
    files = sorted(Path(envdir, game).glob(f"*/{game}.py"))
    if not files:
        raise FileNotFoundError(f"no source for {game} under {envdir}")
    src = files[0].read_text()
    ns = {"__name__": f"arc_agi_3.{game}"}
    exec(compile(src, str(files[0]), "exec"), ns)
    cls = ns[game[0].upper() + game[1:]]
    kwargs = {"seed": 0} if "seed" in inspect.signature(cls).parameters else {}
    return cls(**kwargs), files[0]


MOUSE = re.compile(r"MOUSE\(row=(\d+), col=(\d+)\)")


def to_action(ev):
    name = ev.get("action_name") or "RESET"
    act = GameAction[name]
    data = {}
    if act == GameAction.ACTION6:
        mm = MOUSE.match(ev["action_display"])
        data = {"x": int(mm.group(2)), "y": int(mm.group(1))}
    return ActionInput(id=act, data=data)


def read_events(path):
    out = []
    with lzma.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            if d["type"] in ("initial", "action"):
                d.pop("board_ascii", None)
                d.pop("transcript", None)
                out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--envdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()

    game, src = load_game(a.envdir, a.game)
    events = read_events(a.events)
    assert events[0]["type"] == "initial"

    steps = []  # per step: dict with frame, inst, ui, objects, meta
    mismatches = []
    traj = []  # light per-step sprite state for behaviour roles
    last_frame = None
    for i, ev in enumerate(events):
        act = ActionInput(id=GameAction.RESET) if ev["type"] == "initial" else to_action(ev)
        fd = game.perform_action(act, raw=True)
        if fd.frame:
            frame = np.asarray(fd.frame[-1], dtype=np.int8)
            pre, ui = _CAPTURE["pre"].copy(), _CAPTURE["ui"].copy()
            ui_names = list(_CAPTURE["ui_names"])
        else:  # no frames returned (game over / win): board unchanged
            frame = last_frame
        board = np.asarray(ev["board"], dtype=np.int8)
        ok = bool(np.array_equal(frame, board))
        if not ok:
            mismatches.append((i, ev.get("action_num"), int((frame != board).sum())))
        last_frame = frame

        inst, sprites, (scale, x_off, y_off) = instance_map(game)
        cam = game.camera
        # consistency: wherever no interface drew, pre-UI composite must equal final frame
        pre_ok = bool(np.array_equal(pre[ui < 0], frame[ui < 0]))
        objs = [sprite_record(k, s, inst, frame, scale, x_off, y_off, cam) for k, s in enumerate(sprites)]
        if a.game in WORLD:
            inst, objs, hud, bad = world_instance_map(game, a.game, frame)
            ui = np.where(hud, 0, -1).astype(np.int8)
            pre_ok = bad == 0 and int(hud.sum()) <= 128  # only HUD rows may differ from the scene raster
        traj.append({
            "i": i,
            "act": act.id.name,
            "level": int(game.level_index),
            "state": {o["name"] + "#" + str(sum(1 for p in objs[:o["k"]] if p["name"] == o["name"])):
                      (o["x"], o["y"], o["design"] + str(o.get("vis_colours")), o["visible"], o["n_px"], o["rot"])
                      for o in objs},
        })
        steps.append({
            "i": i, "action_num": ev.get("action_num"), "action": ev.get("action_display"),
            "level": int(game.level_index), "rec_level": ev.get("level"), "state": ev.get("state"),
            "match": ok, "pre_ok": pre_ok, "frame": frame, "inst": inst, "ui": ui, "ui_names": ui_names,
            "camera": {"x": cam.x, "y": cam.y, "w": cam.width, "h": cam.height, "scale": scale,
                       "x_off": x_off, "y_off": y_off, "bg": int(cam.background), "letterbox": int(cam.letter_box)},
            "objects": objs,
        })

    # ------------------------------------------------ sampling (balanced over levels, distinct frames)
    n = len(steps)
    first_of_level = [0] + [k for k in range(1, n) if steps[k]["level"] != steps[k - 1]["level"]]
    chosen = []
    seen = set()

    def add(k):
        key = steps[k]["frame"].tobytes()
        if k in chosen or key in seen or not steps[k]["match"]:
            return False
        chosen.append(k)
        seen.add(key)
        return True

    for k in first_of_level:
        add(k)
    budget = a.n
    grid = np.linspace(0, n - 1, num=max(budget * 3, 1)).round().astype(int)
    # spread over the trajectory; walk forward from each grid point to the next distinct frame
    cand = []
    for g in grid:
        for k in range(g, n):
            if steps[k]["frame"].tobytes() not in seen and steps[k]["match"]:
                cand.append(k)
                break
    cand = list(dict.fromkeys(cand))
    if len(chosen) < budget and cand:
        pick = np.linspace(0, len(cand) - 1, num=min(budget - len(chosen), len(cand))).round().astype(int)
        for p in pick:
            add(cand[p])
    # top up if duplicates collapsed the pick
    for k in cand:
        if len(chosen) >= budget:
            break
        add(k)
    chosen = sorted(chosen)[:budget] if len(chosen) > budget else sorted(chosen)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{a.game}.npz",
        frames=np.stack([steps[k]["frame"] for k in chosen]),
        inst=np.stack([steps[k]["inst"] for k in chosen]),
        ui=np.stack([steps[k]["ui"] for k in chosen]),
        step=np.array([steps[k]["i"] for k in chosen]),
    )
    with open(out / f"{a.game}.jsonl", "w") as f:
        for j, k in enumerate(chosen):
            s = steps[k]
            rec = {kk: s[kk] for kk in ("i", "action_num", "action", "level", "rec_level", "state", "match", "pre_ok", "ui_names", "camera")}
            rec["game"] = a.game
            rec["row"] = j
            rec["objects"] = s["objects"]
            f.write(json.dumps(rec) + "\n")
    with open(out / f"{a.game}.traj.json", "w") as f:
        json.dump(traj, f)
    summary = {
        "game": a.game, "src": str(src), "steps": n, "frames_match": n - len(mismatches),
        "mismatches": mismatches[:20], "pre_ok_all": all(s["pre_ok"] for s in steps),
        "sampled": len(chosen), "levels_seen": sorted({s["level"] for s in steps}),
    }
    print(json.dumps(summary))
    with open(out / f"{a.game}.summary.json", "w") as f:
        json.dump(summary, f)


if __name__ == "__main__":
    main()
