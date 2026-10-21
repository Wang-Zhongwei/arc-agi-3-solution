import sys, json, importlib.util, numpy as np
GAME='/Users/advaith/Documents/studying-stuff/experiments/arc-agi-3/data/environment_files/tn36/ef4dde99/tn36.py'
R='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/runs-out/mem-v2-10x'
spec=importlib.util.spec_from_file_location('tn36src', GAME); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from arcengine import ActionInput, GameAction
cls=[v for k,v in vars(mod).items() if isinstance(v,type) and k.lower()=='tn36'][0]
g=cls()
b=json.load(open(R+'/benchmark.json'))
run=[x for x in b['game_runs'] if x['game_id'].startswith('tn36')][0]
fd=g.perform_action(ActionInput(id=GameAction.RESET)); prev=np.array(fd.frame[-1])
want=set(map(int,sys.argv[1].split(',')))
chains={}
for k,h in enumerate(run['history'][:max(want)],1):
    a=h['action']; fd=g.perform_action(ActionInput(id=GameAction[a['id']] if a['id']!='RESET' else GameAction.RESET,data=a.get('data') or {}))
    frs=[prev]+[np.array(f) for f in fd.frame]
    if k in want:
        steps=[]
        for i in range(1,len(frs)):
            d=np.argwhere(frs[i]!=frs[i-1])
            if len(d): steps.append((len(d), tuple(d.min(0)), tuple(d.max(0))))
        chains[k]=steps
        print('action',k,a,'frames',len(fd.frame),'net change',int((frs[-1]!=frs[0]).sum()))
        for s in steps: print('   ',s)
    prev=np.array(fd.frame[-1])
if len(sys.argv)>2:
    N=int(sys.argv[2])
    g2=cls(); g2.perform_action(ActionInput(id=GameAction.RESET))
    for h in run['history'][:N]:
        a=h['action']; fd=g2.perform_action(ActionInput(id=GameAction[a['id']] if a['id']!='RESET' else GameAction.RESET,data=a.get('data') or {}))
    fr=np.array(fd.frame[-1])
    for c in (39,44,49,54): print('after',N,'col',c,[int(fr[r,c]) for r in (33,36,39,42,45,48)])
    print('action',N,run['history'][N-1]['action'])
