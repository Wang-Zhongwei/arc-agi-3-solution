import json,re,glob,sys,importlib.util
from collections import Counter,defaultdict
sp=importlib.util.spec_from_file_location('cr','/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/wt/stall-fix/ARC3-Inference/inference/agent/confirmed_rules.py')
cr=importlib.util.module_from_spec(sp); sys.modules['cr']=cr; sp.loader.exec_module(cr)
R=json.load(open('SF_rules_labeled.json'))
RET=re.compile(r"\b(wrong|incorrect|false|disproved|refuted|contradict\w*|not the case|doesn't hold|does not hold|no longer|was mistaken|after all|revis\w*|actually not|isn't|is not|doesn't|does not|cannot be)\b|✗",re.I)
reas=defaultdict(list)
for f in glob.glob('SF/*.req.jsonl'):
    g=f.split('/')[-1][:4]
    for l in open(f):
        r=json.loads(l); t=(r.get('reasoning') or '')+'\n'+(r.get('content') or '')
        reas[g].append((r['i'],cr._sentences(t)))
TH=float(sys.argv[1]) if len(sys.argv)>1 else 0.3
hit=Counter(); tot=Counter(); ex=[]
for d in R:
    found=None
    for i,ss in reas[d['g']]:
        if i<=d['first'] or i>d['last']+5: continue
        for s in ss:
            if s!=d['rule'] and RET.search(s) and cr.overlap(s,d['rule'])>=TH:
                found=(i,s); break
        if found: break
    key='contra' if d['contra'] else d['cls']
    tot[key]+=1; hit[key]+=bool(found)
    if found and d['contra'] and len(ex)<6: ex.append((d['g'],d['rule'][:110],found[0],found[1][:200]))
print('threshold',TH,{k:f'{hit[k]}/{tot[k]}' for k in tot})
for e in ex: print(e)
