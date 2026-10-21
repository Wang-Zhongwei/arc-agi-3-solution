import sys, json, importlib.util, numpy as np, glob
g4,ver,cls=sys.argv[1],sys.argv[2],sys.argv[3]
GAME=f'/Users/advaith/Documents/studying-stuff/experiments/arc-agi-3/data/environment_files/{g4}/{ver}/{g4}.py'
R='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/runs-out/mem-v2-10x'
spec=importlib.util.spec_from_file_location(g4+'src', GAME); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from arcengine import ActionInput, GameAction
g=getattr(mod,cls)()
b=json.load(open(R+'/benchmark.json'))
run=[x for x in b['game_runs'] if x['game_id'].startswith(g4)][0]
evs=[json.loads(l) for l in open(glob.glob(R+f'/artifacts/{g4}-*_p0_events.jsonl')[0])]
ev={e['action_num']:e for e in evs if e.get('type')=='action'}
fd=g.perform_action(ActionInput(id=GameAction.RESET)); prev=fd.levels_completed
for k,h in enumerate(run['history'],1):
    a=h['action']; fd=g.perform_action(ActionInput(id=GameAction[a['id']] if a['id']!='RESET' else GameAction.RESET,data=a.get('data') or {}))
    if fd.levels_completed>prev:
        frs=[np.array(f) for f in fd.frame]
        e=ev.get(k,{})
        keys=[kk for kk in e.keys() if kk not in ('board',)]
        bd=np.array(e.get('board')) if e.get('board') is not None else None
        print(f'action {k}: level {prev}->{fd.levels_completed}, engine returned {len(frs)} frames; event board == last frame: {bd is not None and (bd==frs[-1]).all()}; == first frame: {bd is not None and (bd==frs[0]).all()}; event keys: {keys}')
        prev=fd.levels_completed
