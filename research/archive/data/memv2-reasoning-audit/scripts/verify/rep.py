# generic replay: python rep.py <game4> <version> <classname> [N] [extra actions...]
import sys, json, importlib.util, numpy as np
g4, ver, cls = sys.argv[1], sys.argv[2], sys.argv[3]
N=int(sys.argv[4]) if len(sys.argv)>4 and sys.argv[4]!='all' else None
extra=sys.argv[5:]
GAME=f'/Users/advaith/Documents/studying-stuff/experiments/arc-agi-3/data/environment_files/{g4}/{ver}/{g4}.py'
R='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/runs-out/mem-v2-10x'
spec=importlib.util.spec_from_file_location(g4+'src', GAME); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from arcengine import ActionInput, GameAction
g=getattr(mod,cls)()
b=json.load(open(R+'/benchmark.json'))
run=[x for x in b['game_runs'] if x['game_id'].startswith(g4)][0]
H=run['history'] if N is None else run['history'][:N]
import glob
evs=[json.loads(l) for l in open(glob.glob(R+f'/artifacts/{g4}-*_p0_events.jsonl')[0])]
boards={e['action_num']:e.get('board') for e in evs if e.get('type')=='action'}
fd=g.perform_action(ActionInput(id=GameAction.RESET))
mism=[]
for k,h in enumerate(H,1):
    a=h['action']; aid=GameAction[a['id']] if a['id']!='RESET' else GameAction.RESET
    fd=g.perform_action(ActionInput(id=aid,data=a.get('data') or {}))
    fr=fd.frame[-1] if fd.frame else None
    bd=boards.get(k)
    if bd is not None and np.array(bd).tolist()!=np.array(fr).tolist(): mism.append(k)
print(g4, 'replayed',len(H),'of',len(run['history']),'mismatched frames',len(mism),mism[:10],'levels',fd.levels_completed,fd.state, 'level_idx', getattr(g,'level_index',None))
M={'UP':'ACTION1','DOWN':'ACTION2','LEFT':'ACTION3','RIGHT':'ACTION4','SPACE':'ACTION5','UNDO':'ACTION7'}
for x in extra:
    if x.startswith('M('):
        c,r=x[2:-1].split(','); fd=g.perform_action(ActionInput(id=GameAction.ACTION6,data={'x':int(c),'y':int(r)}))
    else:
        fd=g.perform_action(ActionInput(id=GameAction[M[x]]))
    print(' +',x,'levels',fd.levels_completed,fd.state)
