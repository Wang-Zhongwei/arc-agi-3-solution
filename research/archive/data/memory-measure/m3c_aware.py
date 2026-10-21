import json,re,collections
from align import *
import m3_revisit
AW=re.compile(r"(been here|same state|same (board|position|frame|configuration)|back (to|at|in) (the )?(start|same|initial|original|previous|earlier)|back where|loop|cycle|returned to|identical to (the )?(earlier|previous|initial|before|start)|nothing changed|no change|reverted|reset to|undid|undo|restored|went back|came back|already (tried|did|tested|saw|seen|visited))",re.I)
tot=collections.Counter(); ex=[]
for g in games():
    acts,G=m3_revisit.analyze(g); calls=G['calls']
    rev_calls=set(); rep_calls=set()
    for a in acts:
        if a['call'] is None: continue
        if not a['noop'] and not a['lvlup'] and a.get('revisit_attempt'): rev_calls.add(a['call'])
        if a.get('pair_repeat') and not a.get('same_call'): rep_calls.add(a['call'])
    allc=len(calls)
    tot['calls']+=allc; tot['base_aw']+=sum(1 for ti,t,c in calls if AW.search(c['thinking']))
    for k in rev_calls:
        if k+1<len(calls):
            tot['rev_calls']+=1
            th=calls[k+1][2]['thinking']
            if AW.search(th): tot['rev_aware']+=1
    for k in rep_calls:
        if k+1<len(calls):
            tot['rep_calls']+=1
            th=calls[k+1][2]['thinking']
            if AW.search(th): tot['rep_aware']+=1
            elif len(ex)<6: ex.append((g[:4],k,th[:200].replace('\n',' ')))
print(dict(tot))
print('base rate of awareness words per call',round(tot['base_aw']/tot['calls'],3))
print('after a revisit-closing call: next-call mentions loop/same-state/back-to/reverted words',round(tot['rev_aware']/tot['rev_calls'],3),'n',tot['rev_calls'])
print('after a cross-call exact (state,action) repeat: next call mentions',round(tot['rep_aware']/tot['rep_calls'],3),'n',tot['rep_calls'])
for e in ex: print(e)
