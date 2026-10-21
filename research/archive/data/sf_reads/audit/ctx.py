"""Per request: token composition of the whole context (system, previous_levels, turn-message segments,
tool results, assistant). Tokens via the Qwen tokenizer, cached per message hash.
out: <run>_ctx.jsonl (one row per request) and <run>_turnseg.jsonl (one row per NEW turn message)."""
import json, glob, os, sys, hashlib, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg import classify_turn
from tokenizers import Tokenizer
tj = glob.glob('/Users/advaith/.cache/huggingface/hub/models--Intel--Qwen3.8-Flash-Next-W4A16-AutoRound/snapshots/*/tokenizer.json')[0]
TOK = Tokenizer.from_file(tj)
def n(s): return len(TOK.encode(s, add_special_tokens=False).ids) if s else 0
K='/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/kaggle-outputs'
RUNS={'SF':f'{K}/v4-stall-fix-10x/v1','U':f'{K}/v4-mem-unified-10x/v1'}
run=sys.argv[1]
def txt(m):
    c=m.get('content')
    if c is None: return ''
    return c if isinstance(c,str) else "\n".join(p.get('text','') for p in c if isinstance(p,dict) and p.get('type')=='text')
def nimg(m):
    c=m.get('content'); return 0 if isinstance(c,str) or c is None else sum(1 for p in c if isinstance(p,dict) and p.get('type')!='text')
def key(m):
    return hashlib.sha1(json.dumps({k:m.get(k) for k in ('role','content','tool_calls','tool_call_id','reasoning_content')},sort_keys=True).encode()).hexdigest()
cache={}
def comp(m):
    k=key(m)
    if k in cache: return k,cache[k]
    role=m.get('role'); t=txt(m); d={}
    if role=='system': d['system']=n(t)
    elif role=='user' and '<previous_levels>' in t and 'Current state: step' not in t:
        d['previous_levels']=n(t)
    elif role=='user' and 'Current state: step' in t:
        segs,lines=classify_turn(t)
        by={}
        for s,line in lines: by.setdefault(s,[]).append(line)
        for s,ls in by.items(): d['turn.'+s]=n('\n'.join(ls))
    elif role=='user': d['user.other']=n(t)
    elif role=='tool': d['tool']=n(t)
    elif role=='assistant':
        d['asst.reasoning']=n(m.get('reasoning_content') or '')
        d['asst.content']=n(t)
        d['asst.code']=sum(n(str((c.get('function') or {}).get('arguments'))) for c in m.get('tool_calls') or [])
    d['img']=nimg(m)
    cache[k]=d
    return k,d
out=open(f'{run}_ctx.jsonl','w'); out2=open(f'{run}_turnseg.jsonl','w')
for f in sorted(glob.glob(f'{RUNS[run]}/*_p0_requests.jsonl')):
    g=os.path.basename(f)[:4]; prev=set(); i=-1; last=None
    for line in open(f,encoding='utf-8'):
        x=json.loads(line)
        if x.get('event')!='request':
            if last is not None:
                last['usage']=x.get('usage'); out.write(json.dumps(last)+'\n'); last=None
            continue
        i+=1
        tot={}; ks=[]
        for m in x['messages']:
            k,d=comp(m); ks.append(k)
            for a,b in d.items(): tot[a]=tot.get(a,0)+b
            if k not in prev and m.get('role')=='user' and 'Current state: step' in txt(m):
                out2.write(json.dumps({'g':g,'i':i,'wt':x.get('request_index_within_turn'),**d})+'\n')
        prev=set(ks)
        if last is not None: out.write(json.dumps(last)+'\n')
        last={'g':g,'i':i,'wt':x.get('request_index_within_turn'),'n_msgs':len(x['messages']),**tot}
    if last is not None: out.write(json.dumps(last)+'\n')
    print(run,g,i+1,len(cache),flush=True)
