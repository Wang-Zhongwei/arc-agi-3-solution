"""Per-level efficiency decomposition from benchmark.json.xz files (official RHAE formula)."""
import json, lzma, sys, collections, glob, os

def level_score(b, a):
    return min(115.0, (b / a) ** 2 * 100) if a > 0 else 0.0

def game_score(n, base, apl, done):
    tw = n * (n + 1) / 2
    ts = 0.0; mw = 0
    for i in range(n):
        w = i + 1
        s = level_score(base[i], apl[i]) if i < done else 0.0
        if s > 0: mw += w
        ts += s * w
    return min(ts / tw, mw / tw * 100), ts / tw

rows = []
for path in sys.argv[1:]:
    d = json.load(lzma.open(path))
    run = os.path.relpath(os.path.dirname(path), '/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/runs/kaggle')
    for g in d['game_runs']:
        n = g['number_of_levels']; base = g['base_actions_per_level']; apl = g['actions_per_level']; done = g['levels_completed']
        hist = g['history']
        ids = collections.Counter(h['action']['id'] for h in hist)
        sc, raw = game_score(n, base, apl, done)
        rows.append(dict(run=run, game=g['game_id'][:4], n=n, done=done, state=g['state'], score=g['final_score'], my=sc, raw=raw,
                         base=base, apl=apl, resets=ids.get('RESET', 0), ids=dict(ids)))
json.dump(rows, open(os.environ.get('OUT', '/dev/null'), 'w'))
# consistency check
bad = [r for r in rows if r['score'] is not None and abs(r['score'] - r['my']) > 1e-6]; nn=[r['run'] for r in rows if r['score'] is None]; print('none-score', collections.Counter(nn))
print('rows', len(rows), 'mismatch', len(bad))
# per-run summary
by = collections.defaultdict(list)
for r in rows: by[r['run']].append(r)
for run, rs in by.items():
    won = [r for r in rs if r['done'] == r['n']]
    print(f"{run:45s} mean={sum(r['my'] for r in rs)/len(rs):6.2f} won={len(won)} won_scores={[round(r['my'],1) for r in won]} resets={sum(r['resets'] for r in rs)}")
