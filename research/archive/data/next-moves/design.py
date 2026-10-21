import itertools
F=['A MEM (memory-v2-both card+solved+effects, reply-wins)','B DEBRIEF (ARC3_LEVEL_DEBRIEF=1)','C BEFORE_AFTER (ARC3_BEFORE_AFTER=1)','D FABLE facts (INDICATOR_NOTES + BUDGET_ALERT=both)',
   'E HYP tags (ARC3_HYP_OUTCOME_TAGS=1 + memory-main patch)','F SANDBOX (PERSIST_VARS+SHIMS+NOOP_REPEAT_GUARD)','G DYNTEMP (ARC3_DYN_TEMP=phase)','H ENTITY tracker / or dummy placebo column']
runs=[]
for d,c,b,a in itertools.product((-1,1),repeat=4):
    e=b*c*d; f=a*c*d; g=a*b*c; h=a*b*d
    g_=a*b*c
    runs.append(dict(A=a,B=b,C=c,D=d,E=e,F=f,G=g,H=h,block=('day1-2' if a*d>0 else 'day3-4'),acct=("mellamoa" if a*g>0 else 'snowiers')))
keys='ABCDEFGH'
# orthogonality checks
for k in keys:
    for blk in ('block','acct'):
        s=sum(r[k]*(1 if r[blk] in ('day1-2','mellamoa') else -1) for r in runs)
        assert s==0,(k,blk,s)
for k1,k2 in itertools.combinations(keys,2):
    assert sum(r[k1]*r[k2] for r in runs)==0
print('2^(8-4) resolution IV, generators E=BCD F=ACD G=ABC H=ABD; blocks on AD (=BH=CF=EG), accounts on AG (=BC=DE=FH); MEMxDEBRIEF (AB), MEMxHYP (AE), BAxFABLE (CD) stay estimable: all 8 main effects orthogonal to each other and to both blocks')
for f in F: print('  ',f)
print('run block   account   '+' '.join(keys)+'   flags on')
for i,r in enumerate(sorted(runs,key=lambda r:(r['block'],r['acct']))):
    on=''.join(k for k in keys if r[k]>0) or '(reference: all off)'
    print(f'{i+1:3} {r["block"]:7} {r["acct"]:9} '+' '.join('+' if r[k]>0 else '-' for k in keys)+f'   {on}')
