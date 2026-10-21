import sys; sys.path.insert(0,'.')
from collections import defaultdict
from load import *
from decomp import REFS
ALL = [r for r in REFS if r.startswith('v4-')] + ['v4-level-path-10x/v1']
obs=[]
for rel in ALL:
    r = load_run(rel)
    for g in r['games']:
        for lt in level_times(g, r['t_end']):
            obs.append((lt['active']/60, lt['cleared'], g['key'], lt['level'], rel))
rate = defaultdict(lambda: [0, 0])
for o in obs: rate[(o[2],o[3])][0]+=1; rate[(o[2],o[3])][1]+=o[1]
for name,flt in (('all',lambda o:True),('usual',lambda o: rate[(o[2],o[3])][1]/rate[(o[2],o[3])][0]>=0.5)):
    for T in (25,40):
        at=[o for o in obs if flt(o) and o[0]>=T]
        later=sum(1 for o in at if o[1]); never=len(at)-later
        print(f'{name} trigger {T}m: at risk {len(at)} ({len(at)/len(ALL):.1f}/run over {len(ALL)} runs), cleared later {later}, never cleared {never}')
