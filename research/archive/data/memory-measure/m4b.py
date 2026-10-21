import json,collections,statistics as st
ROOT='/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs'
RUNS=['fz-ledger-on-trim','fz-ledger-on-trim-r2','fz-arms-v4-control','fz-arms-v5-temp06','fz-arms-v6-audit60k','fz-ucb','fz-streams14','fz-ledger-streams12']
def key(a):
    d=a.get('data') or {}
    return a['id']+(f"({d.get('x')},{d.get('y')})" if a['id']=='ACTION6' else '')
tot=collections.Counter(); ex=[]
perrun={}
for run in RUNS:
    b=json.load(open(f'{ROOT}/{run}/benchmark.json')); c=collections.Counter()
    for g in b['game_runs']:
        h=g['history']; apl=g['actions_per_level']
        # level boundaries by cumulative actions
        cum=0; lvl_of=[]
        for k,a in enumerate(apl):
            lvl_of+= [k+1]*a
        lvl_of+= [len(apl)]*(len(h)-len(lvl_of))
        # attempts: split at RESET entries
        attempts=[];cur=[];start=0
        deaths=[]
        for i,e in enumerate(h):
            if e['action']['id']=='RESET':
                # death attempt = cur (actions since last boundary)
                deaths.append((lvl_of[i],list(cur),i))
                cur=[]
            else:
                cur.append(key(e['action']))
        # attempts also restart at level boundaries: handle by filtering actions of same level
        # Re-cut: attempt for death = actions since previous RESET or level start
        # compute level starts
        starts={}; s=0
        for k,a in enumerate(apl): starts[k+1]=s; s+=a
        d2=[]
        last_reset_idx=-1
        for (lv,seq,i) in deaths:
            # attempt actions = h[max(start of level, last_reset_idx+1): i]
            st_i=max(starts.get(lv,0),last_reset_idx+1)
            seq=[key(e['action']) for e in h[st_i:i]]
            d2.append((lv,seq)); last_reset_idx=i
        bylevel=collections.defaultdict(list)
        for lv,seq in d2: bylevel[lv].append(seq)
        for lv,seqs in bylevel.items():
            c['death_levels']+=1; c['deaths']+=len(seqs)
            if len(seqs)>=2:
                c['levels_with_multi_death']+=1
                for a,bb in zip(seqs,seqs[1:]):
                    c['death_pairs']+=1
                    same_fatal = a[-1:]==bb[-1:] and len(a)>0 and len(bb)>0
                    c['pair_same_fatal_action']+=same_fatal
                    # longest common prefix
                    l=0
                    while l<min(len(a),len(bb)) and a[l]==bb[l]: l+=1
                    frac=l/max(1,min(len(a),len(bb)))
                    c['pair_prefix_ge_half']+= frac>=0.5
                    c['pair_identical']+= (a==bb)
                    # common suffix >=3
                    s=0
                    while s<min(len(a),len(bb)) and a[-1-s]==bb[-1-s]: s+=1
                    c['pair_suffix_ge3']+= s>=3
                    if (frac>=0.5 and l>=3) and len(ex)<12: ex.append((run,g['game_id'][:4],lv,len(a),len(bb),l,s))
        c['attempt_lens'] = 0
        for lv,seq in d2: c['dlen_sum']+=len(seq); c['dn']+=1
    perrun[run]=dict(c); tot.update(c)
    print(run,{k:v for k,v in c.items() if k!='attempt_lens'})
print('TOTAL',{k:v for k,v in tot.items()})
print(ex)
