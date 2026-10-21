import json,lzma
runs={'memv2':'/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/runs-out/mem-v2-10x/benchmark.json'}
base='/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/runs/kaggle/v4-user-combo-10x/'
for v in ('v1','v2-snowiers','v3-snowiers'): runs[v]=base+v+'/benchmark.json.xz'
for name,p in runs.items():
    f=lzma.open(p) if p.endswith('.xz') else open(p)
    b=json.load(f)
    out=[]; tot=0
    for r in b['game_runs']:
        H=r['history']; ws=[0]+[h.get('wallclock_seconds',0) for h in H]
        gaps=sorted(((ws[i]-ws[i-1]),i) for i in range(1,len(ws)))[::-1]
        big=[(round(g/60),i) for g,i in gaps if g>=1200]
        tot+=len(big)
        if r['game_id'][:4] in ('ka59','sk48','g50t','ls20') or big: out.append((r['game_id'][:4],big[:4]))
    print(name,'gaps>=20min:',tot)
    for o in out: print('  ',o)
