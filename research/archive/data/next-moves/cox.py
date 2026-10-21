import sys; sys.path.insert(0,'.')
from load import *
from variance import GROUPS, RUNS
import numpy as np, math
from scipy import stats
cache={}
def get(r):
    for k,rs in RUNS.items():
        if r in rs: return rs[r]
    if r not in cache: cache[r]=load(r)
    return cache[r]
def records(runs, arm, unit='tok'):
    out=[]
    for r in runs:
        gs=get(r)
        for gid,g in gs.items():
            for l in g['levels']:
                t=l[unit] if unit=='tok' else l['wc']
                if t is None or t<=0: t=1.0
                out.append(((gid,l['L']), float(t), 1 if l['cleared'] else 0, arm, r))
    return out
def cox(recs):
    # stratified Cox, one binary covariate, Breslow; Newton-Raphson
    strata={}
    for s,t,e,x,r in recs: strata.setdefault(s,[]).append((t,e,x))
    S=[(np.array([a[0] for a in v]),np.array([a[1] for a in v]),np.array([a[2] for a in v],float)) for v in strata.values() if len(set(a[2] for a in v))>1]
    b=0.0
    for it in range(50):
        U=0;I=0
        for t,e,x in S:
            order=np.argsort(-t); t2,e2,x2=t[order],e[order],x[order]
            w=np.exp(b*x2); cw=np.cumsum(w); cwx=np.cumsum(w*x2); cwxx=np.cumsum(w*x2*x2)
            idx=np.where(e2==1)[0]
            for i in idx:
                # risk set: all with t>=t_i -> in descending order, positions 0..j where t2[j]>=t2[i]; ties: include all with same t
                j=np.searchsorted(-t2,-t2[i],side='right')-1
                m1=cwx[j]/cw[j]; m2=cwxx[j]/cw[j]
                U+=x2[i]-m1; I+=m2-m1*m1
        step=U/I if I>0 else 0
        b+=step
        if abs(step)<1e-8: break
    se=1/math.sqrt(I) if I>0 else float('nan')
    nev=sum(int(e.sum()) for t,e,x in S)
    return b,se,nev,len(S)
def cmp(label,cands,refs,unit='tok',sd=2.47):
    recs=records(cands,1,unit)+records(refs,0,unit)
    b,se,nev,ns=cox(recs)
    cm=mean(mean(g['score'] for g in get(r).values()) for r in cands)
    rm=mean(mean(g['score'] for g in get(r).values()) for r in refs)
    zs=(cm-rm)/(sd*math.sqrt(1/len(cands)+1/len(refs)))
    print(f'{label:28} score diff {cm-rm:+6.2f} z {zs:+5.2f} | Cox[{unit}] logHR {b:+.3f} (HR {math.exp(b):.2f}) se {se:.3f} z {b/se:+5.2f}  events {nev} strata {ns}')
    return zs,b/se
if __name__=='__main__':
    UC=GROUPS['uc10x']; FZ=GROUPS['fz']
    for unit in ('tok','wc'):
        print('==== unit',unit,'====  (HR>1 = clears levels faster per unit)')
        P=[]
        print('-- placebos --')
        for i,r in enumerate(UC): P.append(cmp('placebo uc '+r.split('/')[1],[r],[x for x in UC if x!=r],unit))
        for i,r in enumerate(FZ): P.append(cmp('placebo fz '+r,[r],[x for x in FZ if x!=r],unit))
        P.append(cmp('placebo 13x pair',[GROUPS['13x'][0]],[GROUPS['13x'][1]],unit))
        P.append(cmp('placebo ledger pair',[GROUPS['ledger'][0]],[GROUPS['ledger'][1]],unit))
        zc=[p[1] for p in P]; zsc=[p[0] for p in P]
        print(f'   placebo Cox z: rms {math.sqrt(mean(z*z for z in zc)):.2f} (should be ~1 if the model se is right);  placebo score z rms {math.sqrt(mean(z*z for z in zsc)):.2f}')
        print('-- real --')
        cmp('ledger(2) vs fz(4)',GROUPS['ledger'],FZ,unit)
        cmp('uc10x(3) vs fz(4)',UC,FZ,unit)
        cmp('uc10x(3) vs v4-control',UC,['v4-control/v1'],unit)
        cmp('13x(2) vs uc10x',GROUPS['13x'],UC,unit)
        for a in ['v4-uc-18s-64k/v1','v4-hats/v1','v4-user-combo-16x/v1','v4-user-combo-10x-accept05/v1','v4-sched-curriculum/v1','v4-uc-11x/v1-snowiers']:
            cmp(a,[a],UC,unit)
        for a in ['v4-stack-v1/v1','fz-ucb/v1','fz-ledger-streams12/v1','fz-ledger-stallswap/v1','fz-experts16-full/v1']:
            cmp(a,[a],FZ,unit)
