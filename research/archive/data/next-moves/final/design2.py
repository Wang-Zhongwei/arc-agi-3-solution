# Revised 2^(8-4) resolution IV factorial for factorial-1 (after the two critiques).
# Changes vs ../design.py: no snowiers (the second block column is now a time period),
# H = output cap 8192 (env only) instead of the entity tracker (failed its Stage-0 gate),
# all-on and all-off run first, and a check of how much one missing run costs.
import itertools
import numpy as np
F = {'A': 'MEM: memory-v2-both (card + solved + effects, reply-wins; PR #148)',
     'B': 'DEBRIEF: ARC3_LEVEL_DEBRIEF=1 (PR #156)',
     'C': 'BEFORE_AFTER: ARC3_BEFORE_AFTER=1 (PR #156)',
     'D': 'FABLE facts: INDICATOR_NOTES + BUDGET_ALERT=both (PR #149)',
     'E': 'HYP tags: ARC3_HYP_OUTCOME_TAGS=1 + memory-main patch (PR #151)',
     'F': 'SANDBOX: PERSIST_VARS + SHIMS + NOOP_REPEAT_GUARD (#76), else dummy',
     'G': 'DYNTEMP: ARC3_DYN_TEMP=phase (PR #154)',
     'H': 'CAP8K: LOCAL_ANALYZER_MAX_OUTPUT=8192 vs 12288 (env only), else dummy'}
runs = []
for d, c, b, a in itertools.product((-1, 1), repeat=4):
    r = dict(A=a, B=b, C=c, D=d, E=b*c*d, F=a*c*d, G=a*b*c, H=a*b*d)
    r['period'] = {(1, 1): 'P1', (1, -1): 'P2', (-1, 1): 'P3', (-1, -1): 'P4'}[(a*d, a*r['G'])]
    runs.append(r)
keys = 'ABCDEFGH'
for k1, k2 in itertools.combinations(keys, 2):
    assert sum(r[k1]*r[k2] for r in runs) == 0
for k in keys:
    for blk in ('AD', 'AG'):
        assert sum(r[k]*r[blk[0]]*r[blk[1]] for r in runs) == 0
# cost of one missing run: se inflation of each main effect (intercept + 8 mains + 3 period dummies)
X = np.array([[1] + [r[k] for k in keys] + [r['A']*r['D'], r['A']*r['G'], r['A']*r['D']*r['A']*r['G']] for r in runs], float)
full = np.sqrt(np.diag(np.linalg.inv(X.T @ X)))[1:9]
worst = max(np.sqrt(np.diag(np.linalg.inv(np.delete(X, i, 0).T @ np.delete(X, i, 0))))[1:9].max() / full.max() for i in range(16))
print('2^(8-4) res IV, E=BCD F=ACD G=ABC H=ABD; periods P1-P4 from AD and AG (2 days each at 2 runs/day);')
print('main effects orthogonal to each other and to periods; AB (MEMxDEBRIEF), AE (MEMxHYP), CD (BAxFABLE) estimable.')
print(f'one missing run inflates the worst main-effect se by x{worst:.2f}')
for k in keys: print('  ', k, F[k])
order = sorted(runs, key=lambda r: (r['period'], -sum(r[k] for k in keys) if r['period'] == 'P1' else 0))
# P1: all-on first, all-off second (inspect all-on before pushing the other 14)
p1 = [r for r in order if r['period'] == 'P1']
p1.sort(key=lambda r: (0 if all(r[k] > 0 for k in keys) else 1 if all(r[k] < 0 for k in keys) else 2))
order = p1 + [r for r in order if r['period'] != 'P1']
print('run period ' + ' '.join(keys) + '   flags on')
for i, r in enumerate(order):
    on = ''.join(k for k in keys if r[k] > 0) or '(all off = user-combo-10x + lap)'
    print(f'{i+1:3}  {r["period"]}    ' + ' '.join('+' if r[k] > 0 else '-' for k in keys) + f'   {on}')
