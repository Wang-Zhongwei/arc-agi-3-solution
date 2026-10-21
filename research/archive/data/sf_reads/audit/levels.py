import json,glob,re,sys
from collections import defaultdict
run=sys.argv[1]
TL=re.compile(r'<this_level>\n(.*?)\n</this_level>',re.S)
res={}
for f in sorted(glob.glob(f'{run}/*.req.jsonl')):
    g=f.split('/')[-1][:4]
    acts=[json.loads(l) for l in open(f.replace('.req.','.act.'))]
    lvl_after={0:1}
    for a in acts: lvl_after[a['n']]=a['level']
    maxn=max(lvl_after)
    per=defaultdict(lambda:{'req':0,'ctok':0,'first':None,'last':None,'acts':0,'deaths':0})
    for a in acts:
        lb=lvl_after.get(a['n']-1,1); per[lb]['acts']+=1; per[lb]['deaths']+=a['over']
    for l in open(f):
        r=json.loads(l); n=int(r['action'] or 1)-1
        lv=lvl_after.get(min(n,maxn),1)
        d=per[lv]; d['req']+=1; d['ctok']+=((r.get('usage') or {}).get('completion_tokens') or 0)
        d['first']=r['i'] if d['first'] is None else d['first']; d['last']=r['i']
    final=acts[-1]['level'] if acts else 1
    won=any(a.get('over') is False and False for a in acts)
    res[g]={str(k):{**v,'cleared':k<final} for k,v in sorted(per.items())}
json.dump(res,open(f'{run}_levels.json','w'),indent=0)
for g,v in res.items():
    print(g,' '.join(f"L{k}:{d['req']}r/{d['ctok']//1000}k{'' if d['cleared'] else '*'}" for k,d in v.items()))
