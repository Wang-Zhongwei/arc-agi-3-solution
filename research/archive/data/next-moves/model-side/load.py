import gzip, json, glob, os, bisect, statistics as st
from collections import defaultdict
def load(run):
    rows = [json.loads(l) for l in gzip.open(f"rows/{run.replace('/', '__')}.jsonl.gz", 'rt')]
    games = {r['game']: r for r in rows if r['kind'] == 'game'}
    reqs = defaultdict(list); acts = defaultdict(list)
    for r in rows:
        if r['kind'] == 'req': reqs[r['game']].append(r)
        elif r['kind'] == 'act': acts[r['game']].append(r)
    out = []
    for g, gm in games.items():
        R = sorted(reqs[g], key=lambda r: r['i']); A = sorted(acts[g], key=lambda a: a['n'])
        apl = gm['apl']; L = gm['levels']
        clears = []; c = 0
        for k in range(L): c += apl[k]; clears.append(c)
        cset = set(clears)
        an = [a['n'] for a in A]
        lastn = an[-1] if an else 0
        for j, r in enumerate(R):
            a0 = r['action'] or 1
            a1 = (R[j+1]['action'] or 1) if j + 1 < len(R) else lastn + 1
            lo = bisect.bisect_left(an, a0); hi = bisect.bisect_left(an, a1)
            ex = A[lo:hi]
            r['n_act'] = len(ex)
            r['n_bc'] = sum(1 for a in ex if a['bc'])
            r['n_go'] = sum(1 for a in ex if a['go'])
            r['clear'] = sum(1 for a in ex if a['n'] in cset)
            r['lvl'] = 1 + bisect.bisect_left(clears, a0)  # level being played at request time
            r['nlev'] = gm['nlev']
            r['score'] = gm['score']
            r['rt'] = r['rt'] or 0; r['ct'] = r['ct'] or 0
            out.append(r)
    return games, out
RUNS = [os.path.basename(p)[:-9].replace('__', '/') for p in sorted(glob.glob('rows/*.jsonl.gz'))]
UC10 = ['v4-user-combo-10x/v1', 'v4-user-combo-10x/v2-snowiers', 'v4-user-combo-10x/v3-snowiers']
def pct(xs, q):
    xs = sorted(xs); 
    if not xs: return float('nan')
    k = (len(xs) - 1) * q; f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)
