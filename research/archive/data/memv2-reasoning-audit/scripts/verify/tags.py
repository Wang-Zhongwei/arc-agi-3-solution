import json,re,glob,os,collections
R='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/runs-out/mem-v2-10x'
TAG=re.compile(r'<(world_model|hypothesis|level_note)[^>]*>(.*?)</\1>',re.S|re.I)
OPEN=re.compile(r'<(world_model|hypothesis|level_note)\b',re.I)
nreq=0
tot=collections.Counter(); where=collections.Counter(); distinct=set(); opens=collections.Counter()
d0=None
for p in sorted(glob.glob(R+'/*_p0_requests.jsonl')):
    g=os.path.basename(p)[:4]
    for l in open(p):
        d=json.loads(l)
        if d.get('event')=='request': nreq+=1
        if d.get('event')!='response': continue
        if d0 is None: d0=d; print(list(d.keys()))
        s=json.dumps(d)
        # find content/reasoning
        resp=d.get('response') or d
        txt=json.dumps(resp)
        for k,v in TAG.findall(json.loads(json.dumps(txt))):
            tot[g]+=1; distinct.add((k,' '.join(v.split())))
        opens[g]+=len(OPEN.findall(txt))
print(nreq, sum(tot.values()), len(distinct), dict(tot), dict(opens))
