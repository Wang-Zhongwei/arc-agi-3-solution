# Count model-written memory tags (assistant messages only: reply text, reasoning, tool-call code)
# in a run's *_p0_requests.jsonl[.xz]; prints per-game counts and totals.
import json,re,glob,os,sys,lzma,hashlib,collections
TAG=re.compile(r'<(world_model|hypothesis|level_note)[^>]*>(.*?)</\1>',re.S|re.I)
def text(m):
    x=m.get('content')
    if isinstance(x,list): return '\n'.join(p.get('text','') for p in x if isinstance(p,dict))
    return str(x or '')
EX=('the blue block is the piece','UP moves the blue block','MOUSE on a wall opens','level 2 needed the key')
def run(d):
    files=sorted(glob.glob(d+'/*_p0_requests.jsonl'))+sorted(glob.glob(d+'/*_p0_requests.jsonl.xz'))
    tot=0; distinct_tot=0; per={}; fields=collections.Counter(); asst_tot=0
    for p in files:
        g=os.path.basename(p)[:4]; seen=set(); tags=[]; nreq=0
        op=lzma.open if p.endswith('.xz') else open
        with op(p,'rt') as f:
            for l in f:
                if '"event": "request"' not in l[-400:] and '"event": "request"' not in l: continue
                d0=json.loads(l)
                if d0.get('event')!='request': continue
                nreq+=1
                for m in d0['messages']:
                    if m.get('role')!='assistant': continue
                    h=hashlib.md5(json.dumps(m,sort_keys=True).encode()).hexdigest()
                    if h in seen: continue
                    seen.add(h)
                    for field,t in (('content',text(m)),('reasoning',str(m.get('reasoning_content') or '')),('tool',json.dumps(m.get('tool_calls') or '', ensure_ascii=False))):
                        for k,v in TAG.findall(t):
                            v=' '.join(v.replace('\\n',' ').split())
                            if any(e in v for e in EX): continue
                            tags.append((field,k,v)); fields[field]+=1
        per[g]=(nreq,len(seen),len(tags),len(set((k,v) for _,k,v in tags)))
        tot+=len(tags); distinct_tot+=per[g][3]; asst_tot+=len(seen)
    print(d.split('/')[-1], 'requests',sum(v[0] for v in per.values()),'assistant msgs',asst_tot,'tag occurrences',tot,'distinct',distinct_tot,'games with tags',sum(1 for v in per.values() if v[2]),dict(fields))
    print('  per game (occurrences):', {g:v[2] for g,v in per.items()})
for d in sys.argv[1:]: run(d)
