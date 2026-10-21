import sys, json, importlib.util, numpy as np
GAME='/Users/advaith/Documents/studying-stuff/experiments/arc-agi-3/data/environment_files/vc33/5430563c/vc33.py'
R='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/runs-out/mem-v2-10x'
spec=importlib.util.spec_from_file_location('vc33src', GAME); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from arcengine import ActionInput, GameAction
g=mod.Vc33()
b=json.load(open(R+'/benchmark.json'))
run=[x for x in b['game_runs'] if x['game_id'].startswith('vc33')][0]
g.perform_action(ActionInput(id=GameAction.RESET))
N=int(sys.argv[1])
for k,h in enumerate(run['history'][:N],1):
    a=h['action']; fd=g.perform_action(ActionInput(id=GameAction[a['id']] if a['id']!='RESET' else GameAction.RESET,data=a.get('data') or {}))
    if k>=N-5: print(k,a,fd.levels_completed)
fr=np.array(fd.frame[-1])
L='0123456789abcdef'
print('   '+''.join(str(c%10) for c in range(64)))
for r in range(64): print(f'{r:2d} '+''.join(L[v] for v in fr[r]))
