import json,glob,os
RUN='fz-ledger-on-trim-r2'
def games(run=RUN):
    out=[]
    for p in sorted(glob.glob(f'parsed/{run}__req__*.json')):
        out.append(os.path.basename(p).split('__')[2][:-5])
    return out
def load(g,run=RUN):
    T=json.load(open(f'parsed/{run}__{g}.json'))
    R=json.load(open(f'parsed/{run}__req__{g}.json'))
    E=json.load(open(f'parsed/ev__{run}__{g}.json'))
    calls=[(ti,t,c) for ti,t in enumerate(T) for c in t['calls']]
    # step -> last call idx in sections with that analysis step
    step_call={}
    for k,(ti,t,c) in enumerate(calls):
        step_call[t['step']]=k
    # trims: request idx k where head changed
    trims=[k for k in range(1,len(R)) if R[k]['head_hash']!=R[k-1]['head_hash']]
    # assistant hash for call k = first new 'a' msg in request k+1 relative to request k
    asst={}
    for k in range(len(R)-1):
        prev=set(R[k]['hs'])
        for h,role in zip(R[k+1]['hs'],R[k+1]['roles']):
            if h not in prev and role=='a':
                asst[k]=h; break
    return dict(T=T,R=R,E=E,calls=calls,step_call=step_call,trims=trims,asst=asst)
def retained(G,call_prev,req_now):
    h=G['asst'].get(call_prev)
    if h is None: return None
    return h in set(G['R'][req_now]['hs'])
