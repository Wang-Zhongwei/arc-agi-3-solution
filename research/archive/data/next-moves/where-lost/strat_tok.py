"""Same as strat.py but 'time' = tokens generated on the level (compute-normalised; ignores per-stream speed)."""
import strat
from load import *
def attempts_tok(rel):
    if rel in strat.CACHE: return strat.CACHE[rel]
    r = load_run(rel); d = {}
    for g in r['games']:
        for i in range(min(g['lc'] + 1, g['nl'])):
            if not g['at']: break
            a0, a1 = g['bounds'][i]
            tok = sum(g['tokens'][a0:a1]) + (g['final_tok'] if i == g['lc'] else 0)
            d[(g['key'], i)] = (tok / 1000, i < g['lc'])
    strat.CACHE[rel] = d; return d
strat.attempts = attempts_tok
if __name__ == '__main__':
    import sys
    sys.argv = ['x']
    exec(open('strat.py').read().split("if __name__ == '__main__':")[1].replace('\n    ', '\n'), {**strat.__dict__, 'compare': strat.compare})
