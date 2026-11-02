"""Score candidate noise filters on v02 ONLY. Usage: noise_harness.py <candidate.py> [--list]
A candidate module defines CANDIDATES = {name: (phase, fn)} with phase in current|past|both.
fn(line) -> bool, True = drop. `line` exposes ONLY what the harness knows at run time:
  text (the shown sentence), next1 (next sentence of the thinking), level, prior (earlier kept lines of this
  game, oldest first, each a dict with text and level).
Base = lines kept by the frozen rule_v8. Acceptance bar per phase: >=4 lines, >=3 games, >=70% noise+bad."""
import json,sys,importlib.util,collections as C,random
from audit_paths import DATA
import rule_v8 as R
ROOT=f'{DATA}/'
ALL=[r for r in json.load(open(ROOT+'joined_v02.json')) if r['kind']=='confirmed' and r['in_section'] and r.get('claim_verdict') and r.get('same') and r.get('later')]
ok=lambda r:r['claim_verdict']=='correct' or (r['claim_verdict']=='partially_correct' and r['usefulness']!='misleading')
bad=lambda r:r['claim_verdict']=='incorrect' or r['usefulness']=='misleading'
useful=lambda r,rd:ok(r) and r[rd]['meaningful'] and not bad(r)
lab=lambda r,rd:'bad' if bad(r) else 'useful' if useful(r,rd) else 'noise'
def views(ph):
    uni=ALL if ph=='current' else [r for r in ALL if r['level']<r['max_level_seen']]
    kept=[r for r in uni if R.keep(r['shown'],r['next1'],ph)]
    seen=C.defaultdict(list); out=[]
    for r in kept:
        out.append((r,dict(text=r['shown'],next1=r['next1'],level=r['level'],prior=list(seen[r['game']]))))
        seen[r['game']].append(dict(text=r['shown'],level=r['level']))
    return out
PH={'current':'same','past':'later'}
def score(cands,show=False):
    res={}
    for name,(p,fn) in cands.items():
        for ph,rd in PH.items():
            if p not in (ph,'both'): continue
            hit=[r for r,l in views(ph) if fn(l)]
            c=C.Counter(lab(r,rd) for r in hit); n=len(hit); g=len({r['game'] for r in hit})
            prec=(c['noise']+c['bad'])/n if n else 0
            res[(name,ph)]=dict(n=n,useful=c['useful'],bad=c['bad'],noise=c['noise'],games=g,precision=round(prec,3),passes=n>=4 and g>=3 and prec>=.7,ids=[r['id'] for r in hit])
            print('%-30s %-7s n %3d useful %3d bad %2d noise %3d games %2d junk %3.0f%% %s'%(name,ph,n,c['useful'],c['bad'],c['noise'],g,100*prec,'PASS' if res[(name,ph)]['passes'] else ''))
            if show:
                for r in hit: print('    [%s] %s | %s'%(lab(r,rd),r['id'],r['shown'][:170].replace('\n',' ')))
    return res
if __name__=='__main__':
    for ph,rd in PH.items():
        print('base',ph,dict(C.Counter(lab(r,rd) for r,_ in views(ph))))
    spec=importlib.util.spec_from_file_location('cand',sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    score(m.CANDIDATES,'--list' in sys.argv)
