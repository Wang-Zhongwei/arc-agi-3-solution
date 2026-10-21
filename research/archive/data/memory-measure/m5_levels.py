import json,glob,os,collections,statistics
ROOT='/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs'
RUNS=['fz-ledger-on-trim','fz-ledger-on-trim-r2','fz-arms-v4-control','fz-arms-v5-temp06','fz-arms-v6-audit60k','fz-ucb','fz-streams14','fz-ledger-streams12']
rows=[]  # per level record
for run in RUNS:
    b=json.load(open(f'{ROOT}/{run}/benchmark.json'))
    for g in b['game_runs']:
        apl=g['actions_per_level']; base=g['base_actions_per_level']; done=g['levels_completed']
        h=g['history']
        # cumulative action idx -> level mapping
        cum=0; bounds=[]
        for k,a in enumerate(apl):
            bounds.append((cum,cum+a)); cum+=a
        for k,a in enumerate(apl):
            if a==0: continue
            s,e=bounds[k]
            seg=h[s:e]
            toks=sum(x['generated_tokens'] for x in seg)
            t0=h[s-1]['wallclock_seconds'] if s>0 else None
            t1=h[e-1]['wallclock_seconds'] if e-1<len(h) and e>0 else None
            rows.append(dict(run=run,game=g['game_id'][:4],level=k+1,actions=a,base=base[k],cleared=(k<done),n_levels=g['number_of_levels'],tokens=toks,secs=(t1-t0) if (t0 is not None and t1 is not None) else None,won=(g['state']=='won')))
json.dump(rows,open('m5_rows.json','w'))
print(len(rows))
