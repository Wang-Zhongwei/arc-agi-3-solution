import json,sys,re
run,g,lo,hi,pat=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),sys.argv[5]
mx=int(sys.argv[6]) if len(sys.argv)>6 else 30
P=re.compile(pat,re.I); k=0
for l in open(f'{run}/{g}.req.jsonl'):
    r=json.loads(l)
    if not lo<=r['i']<=hi: continue
    t=(r.get('reasoning') or '')+'\n'+(r.get('content') or '')
    for s in re.split(r'(?<=[.!?])\s+|\n+',t):
        if P.search(s):
            print(r['i'],'|',s[:300]); k+=1
            if k>=mx: sys.exit()
