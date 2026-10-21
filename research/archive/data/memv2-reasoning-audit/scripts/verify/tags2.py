import json,re,glob,os,collections
R='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/runs-out/mem-v2-10x'
TAG=re.compile(r'<(world_model|hypothesis|level_note)[^>]*>(.*?)</\1>',re.S|re.I)
OPEN=re.compile(r'<(world_model|hypothesis|level_note)\b[^>]*>',re.I)
def text(m):
    x=m.get('content')
    if isinstance(x,list): return '\n'.join(p.get('text','') for p in x if isinstance(p,dict))
    return str(x or '')
occ=[];opens=collections.Counter(); seen_asst=set(); nasst=0
for p in sorted(glob.glob(R+'/*_p0_requests.jsonl')):
    g=os.path.basename(p)[:4]; i=-1
    for l in open(p):
        d=json.loads(l)
        if d.get('event')!='request': continue
        i+=1
        for m in d['messages']:
            if m.get('role')!='assistant': continue
            key=json.dumps(m,sort_keys=True)
            if (g,key) in seen_asst: continue
            seen_asst.add((g,key)); nasst+=1
            for field,t in (('content',text(m)),('reasoning',str(m.get('reasoning_content') or '')),('tool',json.dumps(m.get('tool_calls') or ''))):
                for k,v in TAG.findall(t):
                    occ.append((g,i,field,k,' '.join(v.split())[:100]))
                for o in OPEN.findall(t): opens[(g,field)]+=1
print('assistant msgs',nasst)
for o in occ: print(o)
print(len(occ), len(set((x[3],x[4]) for x in occ)))
print(opens)
