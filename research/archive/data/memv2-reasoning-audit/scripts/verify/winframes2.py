import sys, json, importlib.util, numpy as np, glob
g4,ver,cls=sys.argv[1],sys.argv[2],sys.argv[3]
GAME=f'/Users/advaith/Documents/studying-stuff/experiments/arc-agi-3/data/environment_files/{g4}/{ver}/{g4}.py'
R='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/runs-out/mem-v2-10x'
spec=importlib.util.spec_from_file_location(g4+'src', GAME); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from arcengine import ActionInput, GameAction
g=getattr(mod,cls)()
b=json.load(open(R+'/benchmark.json'))
run=[x for x in b['game_runs'] if x['game_id'].startswith(g4)][0]
fd=g.perform_action(ActionInput(id=GameAction.RESET)); prev=fd.levels_completed; pf=np.array(fd.frame[-1])
for k,h in enumerate(run['history'],1):
    a=h['action']; fd=g.perform_action(ActionInput(id=GameAction[a['id']] if a['id']!='RESET' else GameAction.RESET,data=a.get('data') or {}))
    frs=[np.array(f) for f in fd.frame]
    if fd.levels_completed>prev:
        d=[int((frs[i]!=(pf if i==0 else frs[i-1])).sum()) for i in range(len(frs))]
        print(f'{g4} action {k}: L{prev}->{fd.levels_completed}: {len(frs)} engine frames; cells changed per frame vs previous: {d}; frame0 differs from final: {int((frs[0]!=frs[-1]).sum())}')
        prev=fd.levels_completed
    pf=frs[-1]
