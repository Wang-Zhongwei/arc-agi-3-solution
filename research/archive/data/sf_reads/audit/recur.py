import json,re,glob,sys
from collections import Counter,defaultdict
import importlib.util
sp=importlib.util.spec_from_file_location('cr','/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/wt/stall-fix/ARC3-Inference/inference/agent/confirmed_rules.py')
cr=importlib.util.module_from_spec(sp); sys.modules['cr']=cr; sp.loader.exec_module(cr)
overlap,_sentences=cr.overlap,cr._sentences
R=json.load(open('SF_rules_labeled.json'))
reas=defaultdict(list)
for f in glob.glob('SF/*.req.jsonl'):
    g=f.split('/')[-1][:4]
    for l in open(f):
        r=json.loads(l); t=(r.get('reasoning') or '')+'\n'+(r.get('content') or '')
        reas[g].append((r['i'],_sentences(t)))
res=Counter(); tot=Counter()
for d in R:
    k=0
    for i,ss in reas[d['g']]:
        if any(s!=d['rule'] and overlap(s,d['rule'])>=0.5 for s in ss): k+=1
    d['recur']=k
    b='0' if k==0 else '1' if k==1 else '2-3' if k<=3 else '4+'
    res[(b,d['cls'])]+=1; tot[b]+=1
for b in ['0','1','2-3','4+']:
    print(b,tot[b],{c:res[(b,c)] for c in 'MTPHXE'},'M %.0f%%'%(100*res[(b,'M')]/max(1,tot[b])),'H %.0f%%'%(100*res[(b,'H')]/max(1,tot[b])))
print('contra recur>=1:',sum(1 for d in R if d['contra'] and d['recur']>=1),'of',sum(d['contra'] for d in R))
json.dump(R,open('SF_rules_labeled.json','w'),indent=0,ensure_ascii=False)
