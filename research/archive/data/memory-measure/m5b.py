import json,collections,statistics as st,re
from align import *
rows=[]
for g in games():
    E=json.load(open(f'parsed/ev__{RUN}__{g}.json'))
    # segment into levels: level value at event is level AFTER action; level_completed event ends the previous level
    level=E[0]['level']; cur_state=E[0]['ih']
    seq=[]  # per level: list of (disp, before, after, go, reset)
    L={level:{'acts':[],'start':cur_state}}
    state=cur_state
    for e in E[1:]:
        if e['action_display']=='RESET':
            L[level]['acts'].append(('RESET',state,e['ih'],False,True)); state=e['ih']; continue
        L[level]['acts'].append((e['action_display'],state,e['ih'],bool(e['game_over']),False))
        if e['level_completed']:
            level=e['level']; state=e['ih']; L.setdefault(level,{'acts':[],'start':state})
        else:
            state=e['ih']
    for lv,d in L.items():
        acts=d['acts']
        done = bool(acts) and (lv in L and any(True for _ in [0])) 
        # level cleared if next level exists in L
        cleared=(lv+1) in L
        # final attempt = after last RESET
        idx=max([i for i,a in enumerate(acts) if a[4]],default=-1)
        final=acts[idx+1:]
        # loop-erased walk on states
        stack=[d['start'] if idx==-1 else acts[idx][2]]; path=[]
        for (disp,b,a,go,rs) in final:
            if a==stack[-1]: continue   # no-op: skip
            if a in stack:
                i=stack.index(a); stack=stack[:i+1]; path=path[:i]
            else:
                stack.append(a); path.append(disp)
        rows.append(dict(game=g[:4],level=lv,cleared=cleared,total=len(acts),deaths=sum(1 for a in acts if a[3]),final_len=len(final),lerw=len(path),path=path))
json.dump(rows,open('m5b.json','w'))
cl=[r for r in rows if r['cleared']]
print('cleared levels',len(cl))
def q(x,p): x=sorted(x); return x[int(p*(len(x)-1))]
for name,key in [('total actions incl failed attempts','total'),('final attempt length','final_len'),('loop-erased path length','lerw')]:
    v=[r[key] for r in cl]; print(f'{name:38s} median {st.median(v):5.1f} p25 {q(v,.25)} p75 {q(v,.75)} p90 {q(v,.9)} max {max(v)} mean {st.mean(v):.1f}')
tot=sum(r['final_len'] for r in cl); lw=sum(r['lerw'] for r in cl)
print('sum final',tot,'sum lerw',lw,'ratio',round(lw/tot,3), 'sum total',sum(r['total'] for r in cl))
# pinned record size: compact string
def compact(path):
    out=[];prev=None;n=0
    for p in path:
        if p==prev: n+=1
        else:
            if prev is not None: out.append(prev+('x%d'%n if n>1 else ''))
            prev=p;n=1
    if prev is not None: out.append(prev+('x%d'%n if n>1 else ''))
    return ' '.join(out)
sizes=[len(compact([re.sub(r'MOUSE\(row=(\d+), col=(\d+)\)',r'M\1,\2',x) for x in r['path']])) for r in cl]
print('compact winning-path string chars: median',st.median(sizes),'p75',q(sizes,.75),'p90',q(sizes,.9),'max',max(sizes))
print('example',compact(cl[3]['path']), cl[3]['game'],cl[3]['level'])
