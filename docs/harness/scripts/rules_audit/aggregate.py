import json, sys, collections as C
from audit_paths import DATA
V = f'{DATA}/'
claims = {c['id']: c for c in json.load(open(V + 'claims_v02.json'))}
res = json.load(open(sys.argv[1]))
res = res.get('claims', res) if isinstance(res, dict) else res
rows = []
for r in res:
    f = r.get('final') or {}
    c = claims[r['id']]
    grp = ('confirmed' if c['kind'] == 'confirmed' else
           'refuted:in_section' if c['in_section'] else 'refuted:dropped:' + str(c.get('dropped_by')))
    rows.append(dict(c, **{k: f.get(k) for k in ['claim_verdict','next_role','consequence_verdict','doubt_outcome','usefulness','confidence','evidence']},
                     group=grp, disputed=r.get('disputed'),
                     a=(r.get('in_run') or {}).get('claim_verdict'), b=(r.get('cross_run') or {}).get('claim_verdict')))
json.dump(rows, open(V + 'verified_v02.json', 'w'), indent=1, ensure_ascii=False)
missing = set(claims) - {r['id'] for r in rows}
print('rows', len(rows), 'missing', len(missing), sorted(missing)[:20])

def tab(title, key, by, sub=None):
    print('\n##', title)
    g = C.defaultdict(C.Counter)
    for r in rows:
        if sub and not sub(r): continue
        g[by(r)][r[key]] += 1
    cats = sorted({k for c in g.values() for k in c}, key=str)
    print('%-34s %4s ' % ('group', 'n') + ' '.join('%10s' % str(k)[:10] for k in cats))
    for name in sorted(g, key=str):
        n = sum(g[name].values())
        print('%-34s %4d ' % (str(name)[:34], n) + ' '.join('%5d %3.0f%%' % (g[name][k], 100 * g[name][k] / n) for k in cats))

kind = lambda r: r['kind']
tab('claim_verdict by kind', 'claim_verdict', kind)
tab('claim_verdict by group', 'claim_verdict', lambda r: r['group'])
tab('claim_verdict by next_role', 'claim_verdict', lambda r: (r['kind'], r['next_role']))
tab('next_role by kind', 'next_role', kind)
tab('consequence_verdict (role consequence/elaboration) by kind,role', 'consequence_verdict',
    lambda r: (r['kind'], r['next_role']), sub=lambda r: r['next_role'] in ('consequence', 'elaboration'))
tab('doubt_outcome (role doubt) by kind', 'doubt_outcome', kind, sub=lambda r: r['next_role'] == 'doubt_about_claim')
tab('claim_verdict among doubted', 'claim_verdict', kind, sub=lambda r: r['next_role'] == 'doubt_about_claim')
tab('usefulness by kind', 'usefulness', kind)
tab('usefulness x verdict (confirmed)', 'usefulness', lambda r: r['claim_verdict'], sub=lambda r: r['kind'] == 'confirmed')
tab('heuristic next_class vs next_role', 'next_role', lambda r: r['next_class'])
tab('verdict by game', 'claim_verdict', lambda r: r['game'])
tab('lens agreement on verdict', 'disputed', lambda r: r['kind'])
agree = sum(r['a'] == r['b'] for r in rows); print('\nlens verdict agreement %d/%d' % (agree, len(rows)))
