import json,sys,re
S='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/lab2/extract'
run=sys.argv[1]; i0,i1=int(sys.argv[2]),int(sys.argv[3]); n=int(sys.argv[4]) if len(sys.argv)>4 else 500
cum=0
for l in open(f'{S}/{run}/su15.requests.jsonl'):
    r=json.loads(l)
    u=r.get('usage') or {}
    if r['i']<i0: continue
    if r['i']>i1: break
    cum+=u.get('completion_tokens') or 0
    rs=(r.get('reasoning') or '').replace('\n',' ')
    lt=r.get('last_text') or ''
    ex=re.findall(r'Executed actions: ([^\n]*)',lt)
    print(f"== i={r['i']} st={r['step']} act={r['action']} wt={r['within_turn']} ctok={u.get('completion_tokens')} cum={cum} {r['last_role']} fin={r['finish']} exec={ex[:1]}")
    print('  R:',rs[:n], ' ... ', rs[-n:] if len(rs)>2*n else '')
    for c in r.get('code') or []:
        print('  C:',c.replace('\n',' | ')[:400])
