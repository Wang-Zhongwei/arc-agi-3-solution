"""Load per-game, per-level records from benchmark.json(.xz) of full 121-min runs."""
import json, lzma, os, math
from datetime import datetime
R='/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/runs/kaggle'
RO='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/runs-out'
def _open(p):
    if os.path.exists(p+'/benchmark.json'): return open(p+'/benchmark.json')
    return lzma.open(p+'/benchmark.json.xz','rt')
def lvl_score(base, act):
    if act<=0: return 0.0
    return min(1.15, (base/act)**2)
def game_score(g):
    n=g['n']; W=n*(n+1)/2; s=0; mw=0
    for l in g['levels']:
        if l['cleared'] and l['actions']>0:
            s+=l['L']*lvl_score(l['base'],l['actions']); mw+=l['L']
    return 100*min(s/W, mw/W)
def completion(g):
    n=g['n']; W=n*(n+1)/2
    return 100*sum(l['L'] for l in g['levels'] if l['cleared'] and l['actions']>0)/W
def load(name):
    p = name if name.startswith('/') else (R+'/'+name if os.path.isdir(R+'/'+name) else RO+'/'+name)
    d=json.load(_open(p))
    t0=datetime.fromisoformat(d['start_time'])
    games={}
    for g in d['game_runs']:
        gid=g['game_id'].split('-')[0]
        if g.get('pass',0) not in (0,None) and gid in games: gid=gid+'#'+str(g.get('pass'))
        apl=g['actions_per_level']; base=g['base_actions_per_level']; n=g['number_of_levels']
        h=g['history']; lc=g['levels_completed']
        gst=datetime.fromisoformat(g['started_at']) if g.get('started_at') else t0
        off=(gst-t0).total_seconds()
        # per-level clear time: cumulative tokens and wallclock at the last action of each completed level
        cum_tok=[];s=0
        for x in h: s+=x['generated_tokens']; cum_tok.append(s)
        levels=[]; idx=0; prev_tok=0; prev_wc=0.0
        for L in range(n):
            a=apl[L] if L<len(apl) else 0
            cleared = L<lc
            if cleared:
                end=idx+a-1
                tok=cum_tok[end] if end>=0 and end<len(cum_tok) else s
                wc=h[end]['wallclock_seconds'] if 0<=end<len(h) else None
                levels.append(dict(L=L+1,cleared=True,actions=a,base=base[L],tok=tok-prev_tok,wc=(wc-prev_wc) if wc is not None else None,t_clear=off+(wc or 0),cum_tok=tok))
                prev_tok=tok; prev_wc=wc if wc is not None else prev_wc; idx+=a
            else:
                if L==lc and (a>0 or idx<len(h)):
                    # attempted, censored
                    levels.append(dict(L=L+1,cleared=False,actions=len(h)-idx,base=base[L],tok=s-prev_tok,wc=(h[-1]['wallclock_seconds']-prev_wc) if h else None,t_clear=None,cum_tok=s))
                break
        games[gid]=dict(score=float(g['final_score']),lc=lc,n=n,levels=levels,tokens=s,base=base,apl=apl,off=off,
                        hist=[(off+x['wallclock_seconds'],x['generated_tokens']) for x in h])
    return games
def mean(xs): xs=list(xs); return sum(xs)/len(xs)
