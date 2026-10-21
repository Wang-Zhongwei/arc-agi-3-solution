import json,re,glob,os,collections,sys
R='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/runs-out/mem-v2-10x'
TAG=re.compile(r'<(world_model|hypothesis|level_note)[^>]*>',re.I)
ANN=re.compile(r"(\b(record|save|write|store|jot|log|add|put|keep|emit|note)\b[^.\n]{0,50}\b(memory|world[_ ]model|level[_ ]note|<hypothesis|hypothesis tag)\b|\bfor (the )?memory\b|\brecap for memory\b|<memory>|\bmemory (tag|note|entry|update)s?\b)",re.I)
def text(m):
    x=m.get('content')
    if isinstance(x,list): return '\n'.join(p.get('text','') for p in x if isinstance(p,dict))
    return str(x or '')
tot=0; games=collections.Counter(); ex=collections.defaultdict(list)
for p in sorted(glob.glob(R+'/*_p0_requests.jsonl')):
    g=os.path.basename(p)[:4]; call=0; seen=set()
    for l in open(p):
        d=json.loads(l)
        if d['event']=='response': call+=1; continue
        for m in d['messages']:
            if m.get('role')!='assistant': continue
            k=json.dumps(m,sort_keys=True)
            if k in seen: continue
            seen.add(k)
            reasoning=str(m.get('reasoning_content') or '')
            out=text(m)+json.dumps(m.get('tool_calls') or '')
            mm=ANN.search(reasoning)
            if mm and not TAG.search(out) and not TAG.search(reasoning):
                tot+=1; games[g]+=1
                s=max(0,mm.start()-80); ex[g].append((call, reasoning[s:mm.end()+60].replace('\n',' ')))
print('announcements without tag:',tot,'games:',len(games),dict(games))
for g in sorted(ex):
    for c,s in ex[g][:6]: print(g,'call~',c,'|',s[:260])
