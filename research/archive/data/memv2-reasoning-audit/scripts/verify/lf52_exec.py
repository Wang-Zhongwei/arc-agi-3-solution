import sys, json, importlib.util, numpy as np, copy
GAME='/Users/advaith/Documents/studying-stuff/experiments/arc-agi-3/data/environment_files/lf52/271a04aa/lf52.py'
R='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/runs-out/mem-v2-10x'
spec=importlib.util.spec_from_file_location('lf52src', GAME); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from arcengine import ActionInput, GameAction
cls=[v for k,v in vars(mod).items() if isinstance(v,type) and k[0].isupper() and k.lower()=='lf52'][0]
g=cls()
b=json.load(open(R+'/benchmark.json'))
run=[x for x in b['game_runs'] if x['game_id'].startswith('lf52')][0]
fd=g.perform_action(ActionInput(id=GameAction.RESET))
for h in run['history'][:326]:
    a=h['action']; fd=g.perform_action(ActionInput(id=GameAction[a['id']] if a['id']!='RESET' else GameAction.RESET,data=a.get('data') or {}))
fr=np.array(fd.frame[-1])
L='0123456789abcdef'
for r in range(64): print(''.join(L[v] for v in fr[r]))
print(fd.levels_completed)
print('==== executing analyst path on real engine')
sys.argv=['x']
sys.path.insert(0,'/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/memv2-analysis/work/lf52-271a04aa')
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import solve5
S=json.load(open('/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/memv2-analysis/work/lf52-271a04aa/states.json'))['pre_reset']
gs=frozenset(tuple(x) for x in S['g']); cs=tuple(tuple(x) for x in S['c'])
path,n=solve5.solve(gs,cs)
def blobs(fr):
    out=[]
    seen=np.zeros_like(fr,bool)
    for r in range(64):
        for c in range(64):
            if fr[r,c]==14 and not seen[r,c]:
                # 4x4 blob top-left: rows r..r+3, cols c-1..c+2 (corner pattern .ee.)
                out.append((r,c-1)); seen[r:r+4,c-1:c+3]=True
    return out
fr=np.array(fd.frame[-1]); G=gs; C=cs
nact=0
def offset(fr,G):
    bl=blobs(fr)
    cands={}
    for (r,c) in bl:
        for (x,y) in G:
            if 6*y+6==r: cands[c-(6*x+6)]=cands.get(c-(6*x+6),0)+1
    return max(cands,key=cands.get) if cands else None, bl
MV={'UP':GameAction.ACTION1,'DOWN':GameAction.ACTION2,'LEFT':GameAction.ACTION3,'RIGHT':GameAction.ACTION4}
for a in path:
    if a[0]=='M':
        fd=g.perform_action(ActionInput(id=MV[a[1]])); nact+=1
        C,G=solve5.move_carts(C,G,solve5.DIRS[a[1]])
    else:
        (A,B)=a[1]
        off,bl=offset(np.array(fd.frame[-1]),G)
        ra,ca=6*A[1]+6+1, 6*A[0]+6+off+1
        rb,cb=6*B[1]+6+1, 6*B[0]+6+off+1
        fd=g.perform_action(ActionInput(id=GameAction.ACTION6,data={'x':ca,'y':ra})); nact+=1
        fd=g.perform_action(ActionInput(id=GameAction.ACTION6,data={'x':cb,'y':rb})); nact+=1
        for (mv,ng,cap) in solve5.jumps(C,G):
            if mv==(A,B): G=ng; break
        fr=np.array(fd.frame[-1]); off2,bl2=offset(fr,G)
        print('jump',A,'->',B,'off',off,'blobs now',len(bl2),'abstract greens',len(G),'levels',fd.levels_completed,fd.state)
print('total actions',nact,'final levels',fd.levels_completed,fd.state)
