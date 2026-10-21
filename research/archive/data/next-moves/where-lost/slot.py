"""Where the slot time (tokens) goes: cleared levels vs the final uncleared level by class; failed attempts."""
import statistics
from collections import defaultdict
from load import *
from decomp import ref_clear_times
need, reached = ref_clear_times()
RUNS = ['v4-user-combo-10x/v1', 'v4-user-combo-10x/v2-snowiers', 'v4-user-combo-10x/v3-snowiers', 'v4-level-path-10x/v1', 'v4-control/v1']
for rel in RUNS:
    r = load_run(rel)
    acc = defaultdict(float); tot = 0; fail_tok = 0; over_tok = 0
    for g in r['games']:
        tok = g['tokens']; tot += sum(tok) + g['final_tok']
        fa = failed_attempts(g)
        for i, (a0, a1) in enumerate(g['bounds']):
            if i > g['lc'] or i >= g['nl']: break
            t = sum(tok[a0:a1]) + (g['final_tok'] if i == g['lc'] else 0)
            if i < g['lc']:
                acc['cleared'] += t
                if g['apl'][i] > g['base'][i]: over_tok += t
            else:
                nd = need.get((g['key'], i), []); rate = len(nd) / max(1, reached.get((g['key'], i), 1))
                cls = 'frontier' if not nd else 'wall' if rate < 0.5 else 'usually-cleared'
                acc['final:' + cls] += t
            w = fa.get(i, {}).get('wasted', 0)
            fail_tok += sum(tok[a0:a0 + w])
    print(f"{rel:32s} tokens {tot/1e6:.2f}M: " + ', '.join(f"{k} {v/tot:.0%}" for k, v in sorted(acc.items())) +
          f"; in failed attempts (before last RESET/death) {fail_tok/tot:.0%}; cleared levels that went over the human baseline {over_tok/tot:.0%}")
