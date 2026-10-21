import json,sys
S='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/lab2/extract'
for run in sys.argv[1].split(','):
    try:
        R=[json.loads(l) for l in open(f'{S}/{run}/su15.requests.jsonl')]
        A=[json.loads(l) for l in open(f'{S}/{run}/su15.actions.jsonl')]
    except FileNotFoundError: continue
    by={}
    for r in R:
        u=r.get('usage') or {}
        d=by.setdefault(r['level'],{'req':0,'tok':0,'i0':r['i'],'i1':r['i'],'comp':0})
        d['req']+=1; d['tok']+=(u.get('prompt_tokens',0) or 0)*0+ (u.get('completion_tokens',0) or 0); d['i1']=r['i']
    ab={}
    for a in A:
        ab.setdefault(a['level'] if not a['cleared'] else a['level']-1,[]).append(a)
    print(run, 'total req',len(R),'actions',len(A), 'final level', A[-1]['level'] if A else None)
    for L in sorted(by):
        acts=[a for a in A if (a['level']==L and not a['cleared']) or (a['cleared'] and a['level']==L+1)]
        print(f'  L{L}: req {by[L]["req"]} (i {by[L]["i0"]}-{by[L]["i1"]}) compl_tok {by[L]["tok"]} acts {len(acts)} resets {sum(1 for a in acts if "RESET" in (a["a"] or ""))} undo {sum(1 for a in acts if "UNDO" in (a["a"] or "") or "ACTION7" in (a["a"] or ""))} overs {sum(a["over"] for a in acts)}')
