"""Score cost of failed attempts (actions before the last RESET, incl. the RESETs) on cleared levels,
split by won / not-won games, and by model RESET vs post-GAME_OVER RESET."""
import json, lzma, sys, collections
def score(base, apl, done):
    n=len(base); tw=n*(n+1)/2; ts=0; mw=0
    for i in range(n):
        s=min(115.0,(base[i]/apl[i])**2*100) if i<done and apl[i]>0 else 0
        if s>0: mw+=i+1
        ts+=s*(i+1)
    return min(ts/tw, mw/tw*100)
tot=collections.Counter()
for p in sys.argv[1:]:
    d=json.load(lzma.open(p))
    for g in d['game_runs']:
        base=g['base_actions_per_level']; apl=list(g['actions_per_level']); done=g['levels_completed']
        # walk history assigning actions to levels using actions_per_level boundaries
        hist=[x['action']['id'] for x in g['history']]
        idx=0; last=[]
        for lv in range(g['number_of_levels']):
            seg=hist[idx:idx+apl[lv]]; idx+=apl[lv]
            if 'RESET' in seg:
                k=len(seg)-1-seg[::-1].index('RESET'); last.append(len(seg)-k-1)
            else: last.append(len(seg))
            tot['resets_cleared' if lv<done else 'resets_uncleared']+=seg.count('RESET')
        s1=score(base,apl,done)
        alt=[last[i] if i<done else apl[i] for i in range(len(base))]
        s2=score(base,alt,done)
        key='won' if done==g['number_of_levels'] else 'notwon'
        tot[key+'_gain']+=s2-s1; tot[key+'_n']+=1
        tot['wasted_actions_cleared']+=sum(apl[i]-last[i] for i in range(done))
        tot['actions_cleared']+=sum(apl[:done])
k=len(sys.argv[1:])
print({a:(round(b/k,2) if isinstance(b,float) else round(b/k,1)) for a,b in tot.items()}, 'per run; gains in game points; /25 for proxy')
print('proxy pts/run: won %.2f notwon %.2f'%(tot['won_gain']/k/25, tot['notwon_gain']/k/25))
