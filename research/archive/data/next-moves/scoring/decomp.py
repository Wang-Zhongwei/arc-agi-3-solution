import json, collections, statistics as st
rows = json.load(open('/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/next-moves/scoring/rows.json'))
best = [r for r in rows if r['run'].startswith('v4-user-combo-10x/')]
def ls(b,a): return min(115.0,(b/a)**2*100) if a>0 else 0
def decomp(rs):
    eff_loss=0; comp_loss=0; tot=0; overshoot=0
    ratios=[]; capped=0; cleared=0; lvl_scores=[]
    nextlvl=[]
    for r in rs:
        n=r['n']; tw=n*(n+1)/2; done=r['done']
        cw=sum(range(1,done+1))/tw*100
        eff_loss += cw - r['my']; comp_loss += 100-cw; tot+=r['my']
        for i in range(done):
            a=r['apl'][i]; b=r['base'][i]
            ratios.append(a/b); cleared+=1; s=ls(b,a); lvl_scores.append(s)
            if s>=115: capped+=1
        if done<n:
            nextlvl.append((r['game'], done, n, round((done+1)/tw*100,1), r['apl'][done], r['base'][done]))
    k=len(set(r['run'] for r in rs))
    return dict(runs=k, mean=tot/25/k, eff_loss=eff_loss/25/k, comp_loss=comp_loss/25/k, cleared=cleared/k, capped_frac=capped/cleared,
                ratio_med=st.median(ratios), ratio_q=[round(x,2) for x in st.quantiles(ratios,n=10)],
                frac_ge100=sum(s>=100 for s in lvl_scores)/len(lvl_scores), frac_lt25=sum(s<25 for s in lvl_scores)/len(lvl_scores)), nextlvl
d, nl = decomp(best)
print('user-combo-10x x3:', d)
allr=[r for r in rows if r['my'] is not None and 'cancelled' not in r['run']]
d2,_=decomp(allr)
print('all v4 full runs:', d2)
# where the efficiency loss sits: per game in best runs
gl=collections.defaultdict(list)
for r in best:
    n=r['n']; tw=n*(n+1)/2; cw=sum(range(1,r['done']+1))/tw*100
    gl[r['game']].append((r['done'], n, round(cw,1), round(r['my'],1), round(cw-r['my'],1), [round(r['apl'][i]/r['base'][i],2) for i in range(r['done'])]))
for g,v in sorted(gl.items(), key=lambda kv:-sum(x[4] for x in kv[1])):
    print(g, v)
