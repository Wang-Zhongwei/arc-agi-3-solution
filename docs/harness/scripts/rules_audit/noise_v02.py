import json,sys,collections as C,re
from audit_paths import DATA
import rule_v8 as R
rows=[r for r in json.load(open(DATA/'joined_v02.json')) if r['kind']=='confirmed' and r['in_section'] and r.get('claim_verdict') and r.get('same') and r.get('later')]
ok=lambda r:r['claim_verdict']=='correct' or (r['claim_verdict']=='partially_correct' and r['usefulness']!='misleading')
bad=lambda r:r['claim_verdict']=='incorrect' or r['usefulness']=='misleading'
useful=lambda r,rd:ok(r) and r[rd]['meaningful'] and not bad(r)
for ph,uni,rd in [('current',rows,'same'),('past',[r for r in rows if r['level']<r['max_level_seen']],'later')]:
    k=[r for r in uni if R.keep(r['shown'],r['next1'],ph)]
    print('\n##',ph,'all',len(uni),'kept',len(k))
    for tag,ls in [('v7',uni),('kept',k)]:
        noise=[r for r in ls if not bad(r) and not useful(r,rd)]
        print(tag,'noise',len(noise))
        print('  info_type',C.Counter(r[rd]['info_type'] for r in noise).most_common())
        print('  usefulness',C.Counter(r['usefulness'] for r in noise).most_common())
        print('  comprehensible',C.Counter(r[rd]['comprehensible'] for r in noise).most_common())
        print('  verdict',C.Counter(r['claim_verdict'] for r in noise).most_common())
        print('  games',C.Counter(r['game'] for r in noise).most_common(8))
    print('all-kept breakdown by info_type -> useful/bad/noise')
    t=C.defaultdict(C.Counter)
    for r in k:
        t[r[rd]['info_type']]['bad' if bad(r) else 'useful' if useful(r,rd) else 'noise']+=1
    for a,b in t.items(): print('  ',a,dict(b))
    if ph=='current':
        noise=[r for r in k if not bad(r) and not useful(r,rd)]
        for r in noise:
            print('-',r['id'],'L%d'%r['level'],r[rd]['info_type'],r['usefulness'],'|',r['shown'][:230].replace('\n',' '),'|| WHY:',r[rd]['reason'][:130])
