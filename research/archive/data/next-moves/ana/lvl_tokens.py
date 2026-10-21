import json,sys,os,collections,statistics as st
R='/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/runs/kaggle/'
runs={'10x-v1':'v4-user-combo-10x/v1','10x-v2':'v4-user-combo-10x/v2-snowiers','10x-v3':'v4-user-combo-10x/v3-snowiers',
      '13x-v1':'v4-user-combo/v1','13x-v2':'v4-user-combo-13x/v1','11x':'v4-uc-11x/v1-snowiers'}
def load(p):
    f=R+p+'/score_timeline.jsonl'
    if not os.path.exists(f):
        import glob; c=glob.glob(R+p.split('/')[0]+'/*/score_timeline.jsonl'); print('missing',f,c); return None
    rows=[json.loads(l) for l in open(f)]
    g=collections.defaultdict(list)
    for r in rows:
        if r.get('kind')=='game': g[r['game'][:4]].append(r)
    # level k cleared: first snapshot where levels>=k ; record tokens (game-cumulative) at that snapshot and previous level tokens
    out={}
    for gm,rs in g.items():
        rs.sort(key=lambda r:r['t_bench_s'])
        prev_tok=0;prev_lv=0
        for r in rs:
            while r['levels']>prev_lv:
                prev_lv+=1
                out[(gm,prev_lv)]=(r['tokens'],r['t_bench_s'])
    return out
data={k:load(v) for k,v in runs.items()}
# per (game,level) incremental token cost = tok(level k) - tok(level k-1)
def inc(d):
    o={}
    for (g,k),(t,s) in d.items():
        p=d.get((g,k-1),(0,0))[0] if k>1 else 0
        if (g,k-1) in d or k==1: o[(g,k)]=t-p
    return o
inc_d={k:inc(v) for k,v in data.items() if v}
for name in inc_d: print(name,'levels',len(inc_d[name]),'median inc tokens',st.median(inc_d[name].values()))
# paired comparison 13x vs mean 10x on game-levels cleared by both groups
tens=['10x-v1','10x-v2','10x-v3']; thir=['13x-v1','13x-v2']
keys=set()
for n in tens+thir: keys|=set(inc_d[n])
import math
rows=[]
for k in keys:
    a=[inc_d[n][k] for n in tens if k in inc_d[n]]; b=[inc_d[n][k] for n in thir if k in inc_d[n]]
    if a and b: rows.append((k,st.mean(a),st.mean(b)))
print('paired game-levels',len(rows))
lr=[math.log(b/a) for k,a,b in rows if a>0 and b>0]
print('mean log ratio 13x/10x tokens per cleared level',round(st.mean(lr),3),'median',round(st.median(lr),3),'n',len(lr))
for L in range(1,10):
    x=[math.log(b/a) for k,a,b in rows if k[1]==L and a>0 and b>0]
    if len(x)>=3: print(' level',L,'n',len(x),'mean log ratio',round(st.mean(x),3),'geo ratio',round(math.exp(st.mean(x)),2))
# phase: levels cleared before 40 min of bench vs after, 13x vs 10x tokens per level
for n in tens+thir+['11x']:
    d=data[n]; ic=inc_d[n]
    early=[ic[k] for k in ic if d[k][1]<2400]; late=[ic[k] for k in ic if d[k][1]>=2400]
    print(n,'early(<40min) levels',len(early),'med tok',st.median(early) if early else None,'| late levels',len(late),'med tok',st.median(late) if late else None)
