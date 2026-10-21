import json,lzma,sys,collections
runs=sys.argv[1:]
s100=collections.defaultdict(list); full=collections.defaultdict(list)
for r in runs:
    d=json.load(lzma.open(r+'/benchmark.json.xz'))
    a=[];b=[]
    for g in d['game_runs']:
        gid=g['game_id'][:4]
        sc=g['final_score']; 
        if sc is not None and sc>=99.999: a.append(gid); s100[gid].append(r)
        if g['levels_completed']==g['number_of_levels']: b.append(gid); full[gid].append(r)
    print(f"{r:40s} s100={len(a):2d} full={len(b):2d} mean={sum((g['final_score'] or 0) for g in d['game_runs'])/25:.2f}  full-only:{sorted(set(b)-set(a))}  s100:{sorted(a)}")
print()
for gid in sorted(set(full)|set(s100)):
    print(gid, 's100',len(s100[gid]),'full',len(full[gid]))
