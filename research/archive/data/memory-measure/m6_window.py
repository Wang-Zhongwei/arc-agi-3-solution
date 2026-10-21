import json,collections,statistics as st
from align import *
rows=[]
calls_between=[];allgaps=[]
for g in games():
    G=load(g); R=G['R']; calls=G['calls']; E=G['E']
    inv={h:k for k,h in G['asst'].items()}
    acts_by_step=collections.Counter(e['analysis_step'] for e in E if e['type']=='action' and e['action_display']!='RESET')
    ts=[0]+G['trims']; calls_between+=[b-a for a,b in zip(ts,ts[1:])]
    for k in G['trims']:
        pre_start=min((inv[h] for h in R[k-1]['hs'] if h in inv),default=0)
        start=min((inv[h] for h in R[k]['hs'] if h in inv),default=k)
        lvl_now=calls[k][1]['level']; lvl_start=calls[start][1]['level']
        steps=set(calls[j][1]['step'] for j in range(start,k))
        # steps executed: only those actually having actions
        nacts=sum(acts_by_step[s] for s in steps)
        rows.append(dict(g=g,k=k,start=start,retained_calls=k-start,dropped=start-pre_start,levels_back=(lvl_now or 0)-(lvl_start or 0),actions_visible=nacts,lvl_now=lvl_now))
    allgaps+=[v for s,v in acts_by_step.items()]
n=len(rows)
f=lambda xs,p: sorted(xs)[int(p*(len(xs)-1))]
print('trims',n,'calls between trims median',st.median(calls_between),'mean',round(st.mean(calls_between),1))
rc=[r['retained_calls'] for r in rows]; print('retained calls after trim: median',st.median(rc),'p25',f(rc,.25),'p75',f(rc,.75))
dr=[r['dropped'] for r in rows]; print('calls dropped per trim: median',st.median(dr),'p25',f(dr,.25),'p75',f(dr,.75))
av=[r['actions_visible'] for r in rows]; print('actions visible after trim: median',st.median(av),'p25',f(av,.25),'p75',f(av,.75),'p90',f(av,.9))
lb=collections.Counter(r['levels_back'] for r in rows); print('levels_back distribution',dict(sorted(lb.items())))
print('share of trims whose retained window reaches into a previous level',round(sum(1 for r in rows if r['levels_back']>0)/n,2))
print('actions per executing turn median',st.median(allgaps),'mean',round(st.mean(allgaps),2),'p90',f(allgaps,.9))
json.dump(rows,open('m6_window.json','w'))
