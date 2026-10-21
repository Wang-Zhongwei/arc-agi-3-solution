import json,sys
S='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/lab2/extract'
run=sys.argv[1]; i0,i1=int(sys.argv[2]),int(sys.argv[3]); fields=sys.argv[4].split(',') if len(sys.argv)>4 else ['last_text','reasoning','content','code']
lim=int(sys.argv[5]) if len(sys.argv)>5 else 3000
for l in open(f'{S}/{run}/su15.requests.jsonl'):
    r=json.loads(l)
    if i0<=r['i']<=i1:
        u=r.get('usage') or {}
        print(f"########## {run} i={r['i']} step={r['step']} action={r['action']} L{r['level']} wt={r['within_turn']} last_role={r['last_role']} ptok={u.get('prompt_tokens')} ctok={u.get('completion_tokens')} finish={r['finish']}")
        for f in fields:
            v=r.get(f)
            if isinstance(v,list): v='\n---\n'.join(v)
            if v:
                v=str(v)
                print(f'--- {f} ({len(v)}):'); print(v if len(v)<=lim else v[:lim//2]+'\n...[cut]...\n'+v[-lim//2:])
