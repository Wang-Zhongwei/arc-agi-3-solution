import json,collections,statistics as st
R='/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/runs/kaggle/'
runs={'10x-v1':'v4-user-combo-10x/v1','10x-v2':'v4-user-combo-10x/v2-snowiers','10x-v3':'v4-user-combo-10x/v3-snowiers'}
for name,p in runs.items():
    rows=[json.loads(l) for l in open(R+p+'/score_timeline.jsonl')]
    g=collections.defaultdict(list)
    for r in rows:
        if r.get('kind')=='game': g[r['game'][:4]].append(r)
    cl_tok=cl_act=st_tok=st_act=0; cl_n=st_n=0
    st_time=0; cl_time=0
    for gm,rs in g.items():
        rs.sort(key=lambda r:r['t_bench_s'])
        # segment by level
        seg_start=None; prev=None
        for r in rs:
            if seg_start is None: seg_start=r
            if prev is not None and r['levels']>prev['levels']:
                cl_tok+=r['tokens']-seg_start['tokens']; cl_act+=r['actions']-seg_start['actions']; cl_n+=1
                seg_start=r
            prev=r
        last=rs[-1]
        if last['levels']<last['n_levels']:
            st_tok+=last['tokens']-seg_start['tokens']; st_act+=last['actions']-seg_start['actions']; st_n+=1
    print(name,'cleared segs',cl_n,'tok/action',round(cl_tok/max(cl_act,1)),'| final-uncleared segs',st_n,'tok/action',round(st_tok/max(st_act,1)),'tokens share on final-uncleared',round(st_tok/(st_tok+cl_tok),3))
