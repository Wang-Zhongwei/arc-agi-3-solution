from load import *
from collections import defaultdict
BINS = [0, 500, 1000, 2000, 4000, 8000, 12289]
for name, runs in {'uc10x': UC10, 'all14': RUNS}.items():
    rows = []
    for run in runs:
        games, R = load(run)
        byg = defaultdict(list)
        for r in R: byg[r['game']].append(r)
        for g, L in byg.items():
            L.sort(key=lambda r: r['i'])
            cum = 0; cums = []
            for r in L: cum += r['ct']; cums.append(cum)
            clr_idx = [j for j, r in enumerate(L) if r['clear']]
            for j, r in enumerate(L):
                # next clear at or after j
                nxt = next((k for k in clr_idx if k >= j), None)
                r['clr3'] = nxt is not None and nxt - j < 3
                r['tok_to_clear'] = (cums[nxt] - (cums[j] - r['ct'])) if nxt is not None else None
                r['same_level_clear'] = nxt is not None and L[nxt]['lvl'] == r['lvl']
                rows.append(r)
    print(f"\n== {name}: forward view ({len(rows)} requests) ==")
    print(f"{'bin':<12}{'n':>6}{'clear<=3req%':>13}{'lvl later cleared%':>19}{'median tok to clear k':>22}")
    for lo, hi in zip(BINS, BINS[1:]):
        B = [r for r in rows if lo <= r['rt'] < hi]
        tt = [r['tok_to_clear'] for r in B if r['same_level_clear']]
        print(f"{lo:>5}-{hi:<6}{len(B):6d}{100*sum(r['clr3'] for r in B)/len(B):13.1f}{100*sum(r['same_level_clear'] for r in B)/len(B):19.1f}{pct(tt,.5)/1e3:22.1f}")
    # level-clear neighbourhoods: reasoning of the 3 requests before a clear vs level average
    pre = []; lvl_all = []
    for run in runs:
        games, R = load(run)
        byg = defaultdict(list)
        for r in R: byg[r['game']].append(r)
        for g, L in byg.items():
            L.sort(key=lambda r: r['i'])
            for j, r in enumerate(L):
                if r['clear']:
                    same = [x for x in L[:j+1] if x['lvl'] == r['lvl']]
                    pre.append(max(x['rt'] for x in same[-3:]))
                    lvl_all.append(st.mean(x['rt'] for x in same))
    print(f"clears: {len(pre)}; max reasoning over the clearing request and the 2 before it: median {pct(pre,.5):.0f}, p75 {pct(pre,.75):.0f}, p90 {pct(pre,.9):.0f}; share with >8k: {100*sum(1 for x in pre if x>8000)/len(pre):.1f}%, >4k: {100*sum(1 for x in pre if x>4000)/len(pre):.1f}%")
