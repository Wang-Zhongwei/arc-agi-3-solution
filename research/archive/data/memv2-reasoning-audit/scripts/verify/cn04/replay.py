import sys, json, importlib.util
import numpy as np
GAME='/Users/advaith/Documents/studying-stuff/experiments/arc-agi-3/data/environment_files/cn04/2fe56bfb/cn04.py'
R='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/runs-out/mem-v2-10x'
spec=importlib.util.spec_from_file_location('cn04src', GAME); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from arcengine import ActionInput, GameAction
g=mod.Cn04()
b=json.load(open(R+'/benchmark.json'))
run=[x for x in b['game_runs'] if x['game_id'].startswith('cn04')][0]
evs=[json.loads(l) for l in open(R+'/artifacts/cn04-2fe56bfb_p0_events.jsonl')]
boards={e['action_num']:e.get('board') for e in evs if e.get('type')=='action'}
fd=g.perform_action(ActionInput(id=GameAction.RESET))
mism=0
STOP=int(sys.argv[1]) if len(sys.argv)>1 else 10**9
def pairinfo():
    # recompute pairing with types, like uqlndqojuf
    pos={}
    for s in g.current_level.get_sprites():
        if not s.is_visible: continue
        orig=g.hlxyvcmpk[s.name]; cur=s.pixels.copy(); s.pixels=orig; r=s.render(); s.pixels=cur
        for y in range(r.shape[0]):
            for x in range(r.shape[1]):
                if r[y,x] in (8,13): pos.setdefault((s.x+x,s.y+y),[]).append((s.name[:4],int(r[y,x])))
    return pos
for k,h in enumerate(run['history'],1):
    a=h['action']; aid=GameAction[a['id']] if a['id']!='RESET' else GameAction.RESET
    data=a.get('data') or {}
    fd=g.perform_action(ActionInput(id=aid,data=data))
    fr=fd.frame[-1] if fd.frame else None
    bd=boards.get(k)
    ok = (bd is None) or (np.array(bd).tolist()==np.array(fr).tolist())
    if not ok: mism+=1
    if k==STOP:
        print('action',k,a,'levels',fd.levels_completed,fd.state)
        for p,v in sorted(pairinfo().items()): print(p,v)
        print('win check', g.sjwqloivve())
        for s in g.current_level.get_sprites(): print(s.name, s.x, s.y, s.rotation, s.is_visible)
        break
print('mismatches',mism,'of',k, 'levels', fd.levels_completed)
