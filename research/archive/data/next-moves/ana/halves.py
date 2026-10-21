# Is there a run-level component shared by two disjoint halves of the games in one run?
# Within same-config groups, correlate the deviation of half-A mean with half-B mean across runs.
import json,lzma,os,random,statistics as st,math
R='/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/runs/kaggle/'
def load(p):
    f=R+p+'/benchmark.json'
    fh=open(f) if os.path.exists(f) else lzma.open(f+'.xz','rt')
    d=json.load(fh); return {g['game_id'].split('-')[0]: float(g['final_score']) for g in d['game_runs']}
groups={'fz':['fz-arms/v4','fz-arms/v5','fz-arms/v6','fz-streams14/v1'],
        '10x':['v4-user-combo-10x/v1','v4-user-combo-10x/v2-snowiers','v4-user-combo-10x/v3-snowiers'],
        'ledger':['fz-ledger-on-trim/v1','fz-ledger-on-trim-r2/v1'],
        '13x':['v4-user-combo/v1','v4-user-combo-13x/v1']}
data={}
for g,rs in groups.items():
    ok=[]
    for r in rs:
        try: data[r]=load(r); ok.append(r)
        except Exception as e: print('skip',r,e)
    groups[g]=ok
games=sorted(set.intersection(*[set(v) for v in data.values()]))
print('games',len(games))
random.seed(0)
cors=[];cov_list=[];varA=[];varB=[]
pairs_all=[]
for it in range(2000):
    gs=games[:]; random.shuffle(gs); A=gs[:len(gs)//2]; B=gs[len(gs)//2:]
    xa=[];xb=[]
    for g,rs in groups.items():
        if len(rs)<2: continue
        ma=[st.mean(data[r][x] for x in A) for r in rs]; mb=[st.mean(data[r][x] for x in B) for r in rs]
        ca=st.mean(ma); cb=st.mean(mb)
        xa+= [a-ca for a in ma]; xb+=[b-cb for b in mb]
    n=len(xa)
    cov=sum(a*b for a,b in zip(xa,xb))/n; va=sum(a*a for a in xa)/n; vb=sum(b*b for b in xb)/n
    cov_list.append(cov); varA.append(va); varB.append(vb)
c=st.mean(cov_list); va=st.mean(varA); vb=st.mean(varB)
print('mean within-group cov(halfA,halfB)',round(c,2),'var halfA',round(va,2),'var halfB',round(vb,2),'implied corr',round(c/math.sqrt(va*vb),3))
# per-game sd across runs within group
sds=[]
for g,rs in groups.items():
    if len(rs)<2: continue
    for x in games:
        sds.append(st.stdev([data[r][x] for r in rs]))
print('mean per-game sd across same-config runs',round(st.mean(sds),1),' rms',round(math.sqrt(st.mean([s*s for s in sds])),1))
for g,rs in groups.items(): print(g,[round(st.mean(data[r].values()),2) for r in rs])
