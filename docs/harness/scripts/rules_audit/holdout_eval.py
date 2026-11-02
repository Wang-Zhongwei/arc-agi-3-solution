"""Score the frozen rule on a labelled run. Usage: holdout_eval.py <verify_output> <cold_output> <claims.json> <out.json>"""
import json, sys, random, collections as C
import rule_v8 as R8
import rule_v9 as R9
vf,cf,cl,out=sys.argv[1:5]
ld=lambda p:(lambda d:d.get('result',d))(json.load(open(p)))
V={x['id']:x for x in ld(vf)['claims']}; K={x['id']:x for x in ld(cf)['lines']}; claims=json.load(open(cl))
rows=[]
for c in claims:
    f=(V.get(c['id']) or {}).get('final') or {}
    r=dict(c,**{k:f.get(k) for k in ['claim_verdict','next_role','consequence_verdict','doubt_outcome','usefulness','confidence','evidence']})
    r['same']=(K.get(c['id']) or {}).get('same'); r['later']=(K.get(c['id']) or {}).get('later'); rows.append(r)
json.dump(rows,open(out,'w'),indent=1,ensure_ascii=False)
conf=[r for r in rows if r['kind']=='confirmed']
print('claims',len(rows),'confirmed',len(conf),'missing verify',sum(r['claim_verdict'] is None for r in conf),'missing cold',sum(not(r['same'] and r['later']) for r in conf))
conf=[r for r in conf if r['claim_verdict'] and r['same'] and r['later']]
ok=lambda r:r['claim_verdict']=='correct' or (r['claim_verdict']=='partially_correct' and r['usefulness']!='misleading')
bad=lambda r:r['claim_verdict']=='incorrect' or r['usefulness']=='misleading'
useful=lambda r,rd:ok(r) and r[rd]['meaningful'] and not bad(r)
def sc(ls,rd):
    n=len(ls);u=sum(useful(r,rd) for r in ls);b=sum(bad(r) for r in ls)
    c=sum(r['claim_verdict']=='correct' for r in ls);w=sum(r['claim_verdict']=='incorrect' for r in ls)
    return dict(n=n,useful=u,bad=b,noise=n-u-b,correct=c,wrong=w)
def line(tag,s,base=None):
    n=max(1,s['n'])
    t='%-20s lines %3d useful %3d (%2.0f%%) bad %3d (%2.0f%%) noise %3d (%2.0f%%) | correct %2.0f%% wrong %2.0f%%'%(tag,s['n'],s['useful'],100*s['useful']/n,s['bad'],100*s['bad']/n,s['noise'],100*s['noise']/n,100*s['correct']/n,100*s['wrong']/n)
    if base: t+=' | kept %2.0f%% of useful, %2.0f%% of bad, %2.0f%% of noise'%(100*s['useful']/max(1,base['useful']),100*s['bad']/max(1,base['bad']),100*s['noise']/max(1,base['noise']))
    print(t)
PH=[('current',conf,'same'),('past',[r for r in conf if r['level']<r['max_level_seen']],'later')]
for ph,uni,rd in PH:
    a=sc(uni,rd); print('\n## phase %s (%s reader)'%(ph,rd)); line('v7 keep all',a)
    for f in ('H','M','A','I'):
        d=[r for r in uni if R8.reasons(r['shown'],r['next1'])[f]]
        if d: line('  %s catches'%f,sc(d,rd))
    k8=[r for r in uni if R8.keep(r['shown'],r['next1'],ph)]; line('rule v8',sc(k8,rd),a)
    for f in ('fragment','snapshot')+(('dead_coords','dead_layout','dead_literal') if ph=='past' else ()):
        d=[r for r in k8 if R9.useless_reasons(r['shown'])[f]]
        if d: line('  %s catches'%f[:12],sc(d,rd))
    k9=[r for r in uni if R9.keep(r['shown'],r['next1'],ph)]; line('rule v9',sc(k9,rd),a)
    for tag,k in (('v8',k8),('v9',k9)):
      games=sorted({r['game'] for r in uni}); random.seed(0); dbad=[];duse=[]
      for _ in range(4000):
          gs=[random.choice(games) for _ in games]; A=[];B=[]
          for g in gs:
              A+=[r for r in uni if r['game']==g]; B+=[r for r in k if r['game']==g]
          if A and B:
              sa=sc(A,rd); sb=sc(B,rd); dbad.append(100*sb['bad']/sb['n']-100*sa['bad']/sa['n']); duse.append(100*sb['useful']/sb['n']-100*sa['useful']/sa['n'])
      dbad.sort();duse.sort();q=lambda v,p:v[int(p*len(v))]
      print('   '+tag+' bootstrap by game, 95%%: bad share change %+.1f [%+.1f, %+.1f] pts; useful share change %+.1f [%+.1f, %+.1f] pts'%(q(dbad,.5),q(dbad,.025),q(dbad,.975),q(duse,.5),q(duse,.025),q(duse,.975)))
