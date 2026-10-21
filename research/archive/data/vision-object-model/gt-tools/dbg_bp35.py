import sys, numpy as np
sys.argv = ["x", "--game", "bp35", "--events", "", "--envdir", "", "--out", ""]
import gt_dump as G
D="/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/runs/kaggle/v4-user-combo-10x/v1/artifacts/bp35-0a0ad940_p0_events.jsonl.xz"
E="/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/wt/openrouter/runs/local/environment_files"
game,_=G.load_game(E,"bp35"); evs=G.read_events(D)
from arcengine import ActionInput, GameAction
for i,ev in enumerate(evs[:12]):
    act = ActionInput(id=GameAction.RESET) if ev["type"]=="initial" else G.to_action(ev)
    fd=game.perform_action(act, raw=True); fr=np.asarray(fd.frame[-1],dtype=np.int8)
    inst,objs,hud,bad=G.world_instance_map(game,"bp35",fr)
    ys,xs=np.nonzero(hud)
    print(i, ev.get("action_display"), "hud",int(hud.sum()), "rows",sorted(set(ys.tolist()))[:10], "bad",bad, "inst cover",(inst>=0).mean().round(2), "nobj",len(objs))
    if bad:
        for o in objs:
            if o["n_px"]:
                m=inst==o["k"]; b=(~np.isin(fr[m],o["sprite_colours"])).sum()
                if b: print("   ",o["k"],o["name"],o["sprite_colours"],np.unique(fr[m]).tolist(),b,o["n_px"])
