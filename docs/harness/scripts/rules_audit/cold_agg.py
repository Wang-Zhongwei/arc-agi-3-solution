import json, re, sys, collections as C
from audit_paths import DATA
V = f'{DATA}/'
out=json.load(open(sys.argv[1])); out=out.get('result',out); L=out['lines']
R={r['id']:r for r in json.load(open(V+'verified_v02.json'))}
HEDGE=re.compile(r"^\W*(hmm+|wait|but|however|unless|actually|although|though|maybe|perhaps|or\b|not sure|hold on|let me (re-?check|double|verify|re-?read)|is that|did\b|does\b|was\b|could|might|need to (check|verify))|\?\s*$|\b(not sure|unclear|double[- ]check|re-?check|verify|uncertain|suspicious|contradict|doesn't (match|fit)|wrong)\b", re.I)
rows=[]
for x in L:
    r=dict(R[x['id']]); r['later']=x.get('later'); r['same']=x.get('same'); rows.append(r)
json.dump(rows,open(V+'cold_v02.json','w'),indent=1,ensure_ascii=False)
print('lines',len(rows),'missing later',sum(not r['later'] for r in rows),'missing same',sum(not r['same'] for r in rows))
prop=lambda r:r['kind']=='refuted' or (not HEDGE.search(r['next1'] or '') and r['level']<r['max_level_seen'])
def tab(title,sel,key):
    print('\n##',title)
    for lens in ('later','same'):
        s=[r for r in rows if sel(r) and r[lens]]; c=C.Counter(key(r[lens]) for r in s); n=len(s)
        print('  %-6s n=%3d '%(lens,n)+'  '.join('%s %d (%.0f%%)'%(k,v,100*v/n) for k,v in c.most_common()))
for name,sel in [('confirmed, all',lambda r:r['kind']=='confirmed'),('confirmed, proposed (cleared+no hedge)',lambda r:r['kind']=='confirmed' and prop(r)),('refuted in section',lambda r:r['kind']=='refuted')]:
    tab(name+': meaningful',sel,lambda x:x['meaningful'])
    tab(name+': comprehensible',sel,lambda x:x['comprehensible'])
    tab(name+': info_type',sel,lambda x:x['info_type'])
both=[r for r in rows if r['later'] and r['same']]
print('\nreader agreement meaningful %d/%d, info_type %d/%d'%(sum(r['later']['meaningful']==r['same']['meaningful'] for r in both),len(both),sum(r['later']['info_type']==r['same']['info_type'] for r in both),len(both)))
print('\n## meaningful (same-level reader) x verifier usefulness (confirmed)')
c=C.Counter((r['usefulness'],r['same']['meaningful']) for r in both if r['kind']=='confirmed')
for u in ['game_rule','level_rule','momentary_detail','vague','misleading']: print('  %-17s meaningful %3d  not %3d'%(u,c[(u,True)],c[(u,False)]))
print('\n## meaningful AND correct (confirmed), same-level reader')
for name,sel in [('all',lambda r:r['kind']=='confirmed'),('proposed',lambda r:r['kind']=='confirmed' and prop(r))]:
    s=[r for r in both if sel(r)]; print('  %-9s n=%3d meaningful+correct %.0f%%  meaningful-by-both+correct %.0f%%'%(name,len(s),100*sum(r['same']['meaningful'] and r['claim_verdict']=='correct' for r in s)/len(s),100*sum(r['same']['meaningful'] and r['later']['meaningful'] and r['claim_verdict']=='correct' for r in s)/len(s)))
