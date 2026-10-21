import sys, json, importlib.util, copy, itertools
GAME='/Users/advaith/Documents/studying-stuff/experiments/arc-agi-3/data/environment_files/cn04/2fe56bfb/cn04.py'
R='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/runs-out/mem-v2-10x'
spec=importlib.util.spec_from_file_location('cn04src', GAME); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from arcengine import ActionInput, GameAction
b=json.load(open(R+'/benchmark.json'))
run=[x for x in b['game_runs'] if x['game_id'].startswith('cn04')][0]
H=run['history']
def replay(n):
    g=mod.Cn04(); g.perform_action(ActionInput(id=GameAction.RESET))
    for h in H[:n]:
        a=h['action']; g.perform_action(ActionInput(id=GameAction[a['id']] if a['id']!='RESET' else GameAction.RESET, data=a.get('data') or {}))
    return g
g=replay(98)
print('selected', g.xseexqzst.name if g.xseexqzst else None, g.xseexqzst.x, g.xseexqzst.y, g.xseexqzst.rotation)
print('actions used in level', g._action_count, 'max', g.ojcsxidcz)
names={'U':GameAction.ACTION1,'D':GameAction.ACTION2,'L':GameAction.ACTION3,'R':GameAction.ACTION4,'S':GameAction.ACTION5}
for seq in ['SS','SSL','SSLU','SSLUU','SSLUUU','SSUUUL','SSRUUU','SSUUU']:
    g=replay(98)
    for ch in seq:
        fd=g.perform_action(ActionInput(id=names[ch]))
    s=g.xseexqzst
    print(seq, 'levels',fd.levels_completed, fd.state, s.x, s.y, s.rotation, 'win', g.sjwqloivve())
print('---- select p then finish')
g=replay(98)
p=[s for s in g.current_level.get_sprites() if s.name.startswith('0004')][0]
r=p.render()
import numpy as np
cells=[(p.x+x,p.y+y) for y in range(r.shape[0]) for x in range(r.shape[1]) if r[y,x]>=0 and r[y,x] not in (8,13)]
gx,gy=cells[0]
click=None
for dx in range(64):
    for dy in range(64):
        if g.camera.display_to_grid(dx,dy)==(gx,gy): click=(dx,dy); break
    if click: break
print('p cell',gx,gy,'click',click)
best=None
moves='UDLR'
for n in range(0,6):
    for mv in itertools.product(moves, repeat=n):
        for pre in ['SS']:
            seq=pre+''.join(mv)
            g=replay(98)
            fd=g.perform_action(ActionInput(id=GameAction.ACTION6, data={'x':click[0],'y':click[1]}))
            for ch in seq:
                fd=g.perform_action(ActionInput(id=names[ch]))
            if fd.levels_completed>=2 or g.rqolqpqwo:
                print('WIN with click +',seq, 'total actions', 1+len(seq)+(0 if fd.levels_completed>=2 else 1)); best=seq; break
        if best: break
    if best: break
