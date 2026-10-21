# Generic replay: python rp.py <game-prefix> [--dump N,M,...] ; loads source, replays benchmark history, compares with events boards.
import sys, json, glob, importlib.util, os
from arcengine import ActionInput, GameAction, GameState
SRC='/Users/advaith/Documents/studying-stuff/experiments/arc-agi-3/data/environment_files'
RUN='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/runs-out/mem-v2-10x'
def load(prefix, ver=None):
    vs=sorted(glob.glob(f'{SRC}/{prefix}/*/{prefix}.py'))
    if ver: vs=[v for v in vs if ver in v]
    path=vs[0]
    sys.path.insert(0, os.path.dirname(path))
    spec=importlib.util.spec_from_file_location(prefix+'mod', path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    cls=getattr(m, prefix.capitalize())
    return m, cls, path
def history(prefix):
    b=json.load(open(f'{RUN}/benchmark.json'))
    r=[x for x in b['game_runs'] if x['game_id'].startswith(prefix)][0]
    return r
def events(gid):
    return [json.loads(l) for l in open(f'{RUN}/artifacts/{gid}_p0_events.jsonl')]
def replay(prefix, ver=None, cb=None, quiet=False):
    m, cls, path = load(prefix, ver)
    r=history(prefix); gid=r['game_id']
    evs=events(gid)
    boards={e['action_num']:e['board'] for e in evs if e.get('type')!='initial' and e.get('action_display') not in (None,'analysis')}
    g=cls()
    fd=g.perform_action(ActionInput(id=GameAction.RESET))
    init=evs[0]['board']
    res={'path':path,'init_match': fd.frame[-1]==init if fd.frame else None,'mism':[]}
    if cb: cb(0, None, g, fd)
    for k,h in enumerate(r['history'],1):
        a=h['action']; aid=getattr(GameAction, a['id'])
        fd=g.perform_action(ActionInput(id=aid, data=a.get('data') or {}))
        fr=fd.frame[-1] if fd.frame else None
        if k in boards and boards[k]!=fr: res['mism'].append(k)
        if cb: cb(k, a, g, fd)
    res['levels']=fd.levels_completed; res['state']=str(fd.state)
    return res
if __name__=='__main__':
    p=sys.argv[1]; ver=sys.argv[2] if len(sys.argv)>2 else None
    res=replay(p, ver)
    print(p, res['path'].split('/')[-2], 'init', res['init_match'], 'mismatches', len(res['mism']), res['mism'][:10], 'levels', res['levels'], res['state'])
