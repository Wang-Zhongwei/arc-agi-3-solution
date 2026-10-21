"""Did the six stall-fix lines fire in the run's own requests? Counts each marker in NEW messages only
(per game), and keeps the first few full lines per marker so the content can be checked by eye."""
import glob, json, os, re, sys
from collections import Counter, defaultdict

d = sys.argv[1]
MARK = {
    'refused': 'NOT EXECUTED',
    'invalid_err': 'InvalidActionError',
    'colours_new': '(new on this level',
    'colours_row': 'Colours on the board not clicked',
    'bar': 'Budget bar (',
    'undone': 'undone by the game',
    'clickrun': 'Click run at ',
    'death_undid': 'that the game undid',
    'refused_turn': 'action() call was refused as',
    'gameover': 'GAME OVER',
    'bar_rule': 'BAR RULE',
}
txt = lambda c: c if isinstance(c, str) else ' '.join(p.get('text', '') for p in (c or []) if isinstance(p, dict))
short = re.compile(r"""action\(\s*\[?\s*['"]([UDLRudlr])['"]""")
tot, per, ex = Counter(), defaultdict(Counter), defaultdict(list)
nreq = Counter()
for f in sorted(glob.glob(f'{d}/*_p0_requests.jsonl')):
    g = os.path.basename(f)[:4]
    seen = 0
    for i, line in enumerate(open(f)):
        x = json.loads(line)
        if x.get('event') != 'request':
            continue
        nreq[g] += 1
        msgs = x['messages']
        new = msgs[seen:] if seen <= len(msgs) else msgs[-2:]
        seen = len(msgs)
        for m in new:
            if m.get('role') in ('tool', 'user'):
                t = txt(m.get('content'))
                for k, s in MARK.items():
                    n = t.count(s)
                    if n:
                        per[g][k] += n; tot[k] += n
                        if len(ex[k]) < 6:
                            j = t.find(s)
                            a = t.rfind('\n', 0, j) + 1
                            b = t.find('\n', j + len(s))
                            ex[k].append(f'{g} req{nreq[g]}: ' + t[a:b if b > 0 else None][:600])
            for c in m.get('tool_calls') or []:
                code = str((c.get('function') or {}).get('arguments'))
                if short.search(code):
                    per[g]['short_name_call'] += 1; tot['short_name_call'] += 1
print('TOTAL', dict(tot))
for g in sorted(per):
    print(g, dict(per[g]))
for k, v in ex.items():
    print(f'\n== {k}')
    for e in v:
        print('  ' + e.replace('\n', ' | '))
