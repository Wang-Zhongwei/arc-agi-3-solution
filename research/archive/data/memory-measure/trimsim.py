import json,glob,os
def est_blocks(T):
    """list of turn blocks: each block = (turn_idx, [call chars...], user chars)"""
    return T
def predict_trims(T,hi=310000,lo=150000,sysc=19588,uextra=170,use_user=True):
    # per call est chars; turn user chars attach to first call of turn
    items=[]  # (turn, chars)
    calls=[]
    for ti,t in enumerate(T):
        for j,c in enumerate(t['calls']):
            e=len(c['thinking'])+len(c['code'])+130+len(c['result'])+len(c['assistant'])
            if j==0: e+=t.get('user_len',2500)+uextra
            calls.append((ti,e))
    ctx=[]  # list of (call_idx,turn,chars)
    total=sysc; trims=[]
    for k,(ti,e) in enumerate(calls):
        # request k is built from context before adding call k's output; but user prompt for call k's turn is added at turn start
        # context at request k = ctx + (user chars of this turn if first call)
        # approximate: add the call's own chars AFTER the request; user chars before
        add_user=0
        if k==0 or calls[k-1][0]!=ti:
            add_user=0
        # request k's size: total + current turn's user prompt
        # we simply apply threshold on total before adding call k output (which includes user chars at j==0)
        # trim when total_with_user > hi
        # compute user chars for this call
        t=T[ti]
        ucs=(t.get('user_len',2500)+uextra) if (k==0 or calls[k-1][0]!=ti) else 0
        req=total+ucs
        if req>hi:
            # drop oldest whole turns until <= lo
            while ctx and req>lo:
                oti=ctx[0][1]
                dropped=0
                while ctx and ctx[0][1]==oti:
                    dropped+=ctx[0][2]; ctx.pop(0)
                total-=dropped; req-=dropped
            trims.append(k)
        # add call k's output (incl user chars when first call)
        e_call=e-(t.get('user_len',2500)+uextra if (k==0 or calls[k-1][0]!=ti) else 0)
        if ucs: ctx.append((k,ti,ucs)); total+=ucs
        ctx.append((k,ti,e_call)); total+=e_call
    return trims
if __name__=='__main__':
    run='fz-ledger-on-trim-r2'
    ok=0;tot=0;extra=0;miss=0
    for p in sorted(glob.glob(f'parsed/{run}__req__*.json')):
        g=os.path.basename(p).split('__')[2][:-5]
        R=json.load(open(p)); T=json.load(open(f'parsed/{run}__{g}.json'))
        truth=[k for k in range(1,len(R)) if R[k]['head_hash']!=R[k-1]['head_hash']]
        pred=predict_trims(T)
        m=sum(1 for t in truth if any(abs(t-q)<=1 for q in pred))
        ok+=m; tot+=len(truth); extra+=len(pred)-m; 
        print(g[:4],'truth',truth,'pred',pred)
    print('matched',ok,'of',tot,'extra pred',extra)
