from load import *
from variance import GROUPS, RUNS
import math, statistics as st
for grp in ('uc10x','fz'):
    rs=RUNS[grp]; names=list(rs)
    out={'tok':[], 'wc':[], 'act':[]}; n=0; cl_all=0; cl_some=0
    for g in rs[names[0]]:
        maxL=max(len(rs[r][g]['levels']) for r in names)
        for L in range(1,maxL+1):
            recs=[]
            for r in names:
                lv=[l for l in rs[r][g]['levels'] if l['L']==L]
                recs.append(lv[0] if lv else None)
            att=[x for x in recs if x is not None]
            if len(att)<len(names): continue
            if all(x['cleared'] for x in att):
                cl_all+=1
                for key,f in (('tok',lambda x:x['tok']),('wc',lambda x:x['wc']),('act',lambda x:x['actions'])):
                    v=[math.log(max(1,f(x))) for x in att]
                    out[key].append(st.variance(v))
            elif any(x['cleared'] for x in att): cl_some+=1
    print(f'{grp}: (game,level) attempted in all runs: cleared in all {cl_all}, cleared in some {cl_some}')
    for key,v in out.items():
        s=math.sqrt(mean(v)); print(f'   sd of log {key} to clear, within (game,level) across runs: {s:.2f}  (x{math.exp(s):.2f} per sd)')
