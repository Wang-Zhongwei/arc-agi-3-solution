import sys; sys.path.insert(0,'.')
from cox import *
from load import load, mean
import os
RO='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/runs-out/'
def show(label,a,b):
    recs=records([a],1)+records([b],0)
    bb,se,nev,ns=cox(recs)
    sa=mean(g['score'] for g in get(a).values()); sb=mean(g['score'] for g in get(b).values())
    la=sum(g['lc'] for g in get(a).values()); lb=sum(g['lc'] for g in get(b).values())
    print(f'{label:34} score {sa:6.2f} vs {sb:6.2f}  levels {la} vs {lb} | logHR {bb:+.3f} se {se:.3f} z {bb/se:+.2f} events {nev}')
ref=RO+'panel-uc10x'
for p in ['panel-macros','panel-induction','panel-traj-model','panel-traj-openjev']:
    show(p+' vs panel-uc10x',RO+p,ref)
R='/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/runs/kaggle/'
for p in ['v4-panel-control/v1','v4-panel-memory-v2-both/v1','v4-panel-dprime/v1','v4-panel-easy-first/v1','v4-panel-thinking-digest/v1','v4-panel-conf-modes/v1','v4-panel-multi-traj/v1']:
    try: show(p+' vs panel-uc10x',R+p,ref)
    except Exception as e: print(p,'ERR',e)
