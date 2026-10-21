"""Full-trajectory GT dump: every replayed step's frame, instance map, HUD map and light object list.

Reuses gt_dump.py (same replay, same instance-map trick). Output per game in <out>:
  <game>.npz  frames (N,64,64 int8), inst (N,64,64 int16), ui (N,64,64 int8), level (N), match (N)
  <game>.meta.json  per step: act, click [x,y] or null, objects [[k, name, design, tags_has_click, collidable, blocking, visible]]
Usage: python gt_full.py --game g --events ev.jsonl.xz --envdir envdir --out dir
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gt"))
import numpy as np
import gt_dump as G
from arcengine import ActionInput, GameAction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True); ap.add_argument("--events", required=True)
    ap.add_argument("--envdir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    game, _ = G.load_game(a.envdir, a.game)
    events = G.read_events(a.events)
    F, I, U, L, M, meta = [], [], [], [], [], []
    last = None
    for ev in events:
        act = ActionInput(id=GameAction.RESET) if ev["type"] == "initial" else G.to_action(ev)
        fd = game.perform_action(act, raw=True)
        if fd.frame:
            frame = np.asarray(fd.frame[-1], dtype=np.int8); ui = G._CAPTURE["ui"].copy()
        else:
            frame = last
        last = frame
        board = np.asarray(ev["board"], dtype=np.int8)
        inst, sprites, (scale, xo, yo) = G.instance_map(game)
        objs = [G.sprite_record(k, s, inst, frame, scale, xo, yo, game.camera) for k, s in enumerate(sprites)]
        if a.game in G.WORLD:
            inst, objs, hud, bad = G.world_instance_map(game, a.game, frame)
            ui = np.where(hud, 0, -1).astype(np.int8)
        F.append(frame); I.append(inst); U.append(ui); L.append(int(game.level_index)); M.append(bool(np.array_equal(frame, board)))
        click = [act.data["x"], act.data["y"]] if act.id == GameAction.ACTION6 else None
        meta.append({"act": act.id.name, "click": click, "scale": int(scale),
                     "objects": [[o["k"], o["name"], o["design"], "sys_click" in (o.get("tags") or []), o.get("collidable"),
                                  o.get("blocking"), o.get("visible"), o["n_px"]] for o in objs]})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{a.game}.npz", frames=np.stack(F), inst=np.stack(I), ui=np.stack(U), level=np.array(L), match=np.array(M))
    json.dump(meta, open(out / f"{a.game}.meta.json", "w"))
    print(a.game, len(F), "steps, match", sum(M))


if __name__ == "__main__":
    main()
