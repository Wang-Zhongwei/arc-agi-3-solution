"""Per game: new messages per request (full user text, tool text cut), the <previous_levels> versions,
the reply (reasoning/content/code), usage; plus actions with click colour. out: audit/<run>/<game>.{req,act}.jsonl"""
import json, os, sys, hashlib, glob, re
K='/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/kaggle-outputs'
RUNS={'SF':f'{K}/v4-stall-fix-10x/v1','U':f'{K}/v4-mem-unified-10x/v1'}
OUT=os.path.dirname(os.path.abspath(__file__))
def txt(m):
    c=m.get('content')
    if c is None: return ''
    return c if isinstance(c,str) else "\n".join(p.get('text','') for p in c if isinstance(p,dict) and p.get('type')=='text')
def nimg(m):
    c=m.get('content'); return 0 if isinstance(c,str) or c is None else sum(1 for p in c if isinstance(p,dict) and p.get('type')!='text')
def key(m):
    return hashlib.sha1(json.dumps({k:m.get(k) for k in ('role','content','tool_calls','tool_call_id','reasoning_content')},sort_keys=True).encode()).hexdigest()
def codes(m):
    out=[]
    for c in (m or {}).get('tool_calls') or []:
        a=(c.get('function') or {}).get('arguments')
        try:
            p=json.loads(a) if isinstance(a,str) else (a or {}); out.append(str(p.get('code',a)))
        except Exception: out.append(str(a))
    return out
PL=re.compile(r'<previous_levels>.*?</previous_levels>',re.S)
def one(run,f):
    gid=os.path.basename(f).split('_p0_')[0]; g=gid[:4]
    os.makedirs(f'{OUT}/{run}',exist_ok=True)
    out=open(f'{OUT}/{run}/{g}.req.jsonl','w')
    prevkeys=set(); pending=None; i=-1; lastpl=None; usage_q=[]
    rows=[]
    for line in open(f,encoding='utf-8'):
        x=json.loads(line)
        if x.get('event')!='request':
            if rows: 
                rows[-1]['usage']=x.get('usage'); rows[-1]['finish']=x.get('finish_reason')
            continue
        i+=1
        msgs=x['messages']; ks=[key(m) for m in msgs]
        new=[(m,k) for m,k in zip(msgs,ks) if k not in prevkeys]
        # reply of previous request
        if rows:
            rep=next((m for m,k in new if m.get('role')=='assistant'),None)
            if rep is not None:
                rows[-1]['reasoning']=rep.get('reasoning_content'); rows[-1]['content']=txt(rep); rows[-1]['code']=codes(rep)
            # flush older rows
            while len(rows)>1:
                out.write(json.dumps(rows.pop(0),ensure_ascii=False)+'\n')
        pl=None
        for m in msgs[:3]:
            mm=PL.search(txt(m))
            if mm: pl=mm.group(0); break
        rec={'i':i,'step':x.get('analysis_step'),'action':x.get('action'),'wt':x.get('request_index_within_turn'),'n_msgs':len(msgs),
             'new':[{'role':m.get('role'),'text':(txt(m) if m.get('role') in ('user','system') else txt(m)[:3000]),'img':nimg(m),'idx':msgs.index(m)} for m,k in new if m.get('role')!='assistant'],
             'pl':pl if pl!=lastpl else None,'pl_len':len(pl) if pl else 0}
        lastpl=pl
        rows.append(rec); prevkeys=set(ks)
    for r in rows: out.write(json.dumps(r,ensure_ascii=False)+'\n')
    out.close()
    # actions
    ev=f'{RUNS[run]}/artifacts/{gid}_p0_events.jsonl'
    prev_board=None
    with open(f'{OUT}/{run}/{g}.act.jsonl','w') as fo:
        for l in open(ev):
            e=json.loads(l)
            if e.get('type') not in ('action','initial'): continue
            b=e.get('board_ascii','').split('\n')
            if e['type']=='action':
                a=e.get('action_display'); col=None
                m=re.match(r'MOUSE\(row=(\d+), col=(\d+)\)',str(a))
                if m and prev_board:
                    r,c=int(m.group(1)),int(m.group(2))
                    try: col=prev_board[r][c]
                    except Exception: col=None
                fo.write(json.dumps({'n':int(e.get('action_num') or 0),'a':a,'level':int(e.get('level') or 1),'step':e.get('analysis_step'),
                    'cleared':e.get('level_completed') in (True,'True'),'over':e.get('game_over') in (True,'True'),'changed':e.get('board_changed'),'clickcol':col})+'\n')
            prev_board=b
    print(run,g,i+1,flush=True)
run=sys.argv[1]
for f in sorted(glob.glob(f'{RUNS[run]}/*_p0_requests.jsonl')):
    if len(sys.argv)>2 and os.path.basename(f)[:4] not in sys.argv[2].split(','): continue
    one(run,f)
