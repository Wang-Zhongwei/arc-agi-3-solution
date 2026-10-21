from load import *
import sys
sets = {'uc10x (3 runs)': UC10, '13x (2 runs)': ['v4-user-combo/v1', 'v4-user-combo-13x/v1'], 'all 14 runs': RUNS}
BINS = [0, 250, 500, 1000, 2000, 4000, 8000, 12289]
for name, runs in sets.items():
    R = []
    for run in runs: R += load(run)[1]
    G = sum(r['ct'] for r in R); RE = sum(r['rt'] for r in R)
    tot_clear = sum(r['clear'] for r in R)
    print(f"\n== {name}: {len(R)} requests, {G/1e6:.2f}M generated, {tot_clear} clears ==")
    print(f"{'reasoning bin':<14}{'reqs%':>6}{'gen%':>6}{'act/req':>8}{'bc/req':>7}{'0-act%':>7}{'clear%':>7}{'death%':>7}{'clears/1M':>10}{'acts/1k tok':>12}{'len%':>6}")
    for lo, hi in zip(BINS, BINS[1:]):
        B = [r for r in R if lo <= r['rt'] < hi]
        if not B: continue
        g = sum(r['ct'] for r in B); a = sum(r['n_act'] for r in B)
        print(f"{lo:>5}-{hi:<8}{100*len(B)/len(R):6.1f}{100*g/G:6.1f}{a/len(B):8.2f}{sum(r['n_bc'] for r in B)/len(B):7.2f}{100*sum(1 for r in B if r['n_act']==0)/len(B):7.1f}{100*sum(1 for r in B if r['clear'])/len(B):7.2f}{100*sum(1 for r in B if r['n_go'])/len(B):7.2f}{1e6*sum(r['clear'] for r in B)/g:10.1f}{1e3*a/g:12.2f}{100*sum(1 for r in B if r['fin']=='length')/len(B):6.1f}")
    # cap counterfactual (upper bound on savings; ignores what the capped request would then do)
    print('cap  reasoning-tokens-above-cap%  requests-affected%  clearing-requests-affected%')
    clr = [r for r in R if r['clear']]
    for cap in (2000, 3000, 4000, 6000, 8000):
        sav = sum(max(0, r['rt'] - cap) for r in R)
        print(f"{cap:>5}  {100*sav/RE:6.1f}  {100*sum(1 for r in R if r['rt']>cap)/len(R):6.1f}  {100*sum(1 for r in clr if r['rt']>cap)/max(1,len(clr)):6.1f}")
