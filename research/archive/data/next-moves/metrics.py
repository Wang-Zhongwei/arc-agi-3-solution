from load import *
import statistics as st, math
from variance import GROUPS, RUNS
ARMS=['v4-user-combo-10x-accept05/v1','v4-hats/v1','v4-sched-curriculum/v1','v4-uc-18s-64k/v1','v4-uc-11x/v1-snowiers','v4-user-combo-16x/v1','v4-control/v1']
FZARMS=['fz-ledger-streams12/v1','fz-ledger-stallswap/v1','fz-ucb/v1','v4-stack-v1/v1','fz-experts16-full/v1']
def metrics(gs):
    games=[g for g in gs.values()]
    tok=sum(g['tokens'] for g in games)
    lv=sum(g['lc'] for g in games)
    m=dict(score=mean(g['score'] for g in games),
           completion=mean(completion(g) for g in games),
           levels=lv,
           wins=sum(1 for g in games if g['score']>=99.99),
           fullclear=sum(1 for g in games if g['lc']==g['n']),
           lv_per_M=lv/(tok/1e6),
           tokens_M=tok/1e6,
           effloss=mean(completion(g)-g['score'] for g in games))
    # score at matched wall-clock: reconstruct from history (approx, using t of each action and level clears)
    for T in (60,90):
        s=0
        for g in games:
            t_lim=T*60
            g2=dict(g); lv2=[l for l in g['levels'] if l['cleared'] and l['t_clear'] is not None and l['t_clear']<=t_lim]
            g2['levels']=lv2; s+=game_score(g2)
        m[f'score@{T}m']=s/len(games)
    # levels at matched tokens (global tokens across games ordered by time)
    ev=[]
    for g in games:
        for l in g['levels']:
            if l['cleared'] and l['t_clear'] is not None: ev.append(l['t_clear'])
    hist=sorted((t,k) for g in games for (t,k) in g['hist'])
    cum=0; marks={2e6:None,3e6:None,4e6:None}; tmarks={}
    for t,k in hist:
        cum+=k
        for M in marks:
            if M not in tmarks and cum>=M: tmarks[M]=t
    for M in marks:
        tm=tmarks.get(M)
        m[f'lv@{int(M/1e6)}M']=sum(1 for t in ev if tm is not None and t<=tm) if tm else float('nan')
    return m
if __name__=='__main__':
    allm={}
    for k,rs in RUNS.items():
        for r,gs in rs.items(): allm[r]=(k,metrics(gs))
    for r in ARMS+FZARMS:
        try: allm[r]=('arm',metrics(load(r)))
        except Exception as e: print('skip',r,e)
    keys=list(next(iter(allm.values()))[1].keys())
    # pooled within-group sd per metric
    sd={}
    for key in keys:
        ss=0;dof=0
        for k in GROUPS:
            xs=[allm[r][1][key] for r in GROUPS[k] if not math.isnan(allm[r][1][key])]
            if len(xs)>1:
                mu=mean(xs); ss+=sum((x-mu)**2 for x in xs); dof+=len(xs)-1
        sd[key]=math.sqrt(ss/dof) if dof else float('nan')
    ref={key:mean(allm[r][1][key] for r in GROUPS['uc10x']) for key in keys}
    fref={key:mean(allm[r][1][key] for r in GROUPS['fz']) for key in keys}
    print('metric        pooled-sd  uc10x-mean  CV    | arm effects vs uc10x mean in pooled-sd units')
    hdr=['accept05','hats','curric','18s64k','11x','16x','13x(2)','v4ctl']
    print(' '*48+'  '.join(f'{h:>7}' for h in hdr)+'  | ledger(2) vs fz')
    for key in keys:
        effs=[]
        for r in ARMS:
            effs.append((allm[r][1][key]-ref[key])/sd[key])
        e13=(mean(allm[r][1][key] for r in GROUPS['13x'])-ref[key])/sd[key]
        effs=effs[:6]+[e13]+effs[6:]
        el=(mean(allm[r][1][key] for r in GROUPS['ledger'])-fref[key])/sd[key]
        print(f'{key:12} {sd[key]:9.2f} {ref[key]:10.2f} {sd[key]/abs(ref[key]) if ref[key] else 0:5.2f} | '+'  '.join(f'{e:+7.2f}' for e in effs)+f'  | {el:+6.2f}')
    print()
    for r,(k,m) in allm.items():
        print(f'{r:38} {k:6} '+' '.join(f'{key}={m[key]:.2f}' for key in keys))
