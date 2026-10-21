"""Wins and time-to-win (bench minutes from model-ready) per game, across full 10x runs."""
import json, sys
R = '/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/runs/kaggle'
K = '/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/kaggle-outputs'
RUNS = {
    'A': f'{R}/v4-pf-solved-10x/v1', 'B': f'{R}/v4-pf-keep-10x/v1', 'comb': f'{R}/v4-pf-keep-v2-all-10x/v1',
    'C': f'{R}/v4-pf-solved-cap8k-10x/v1-snowiers', 'keep': f'{R}/v4-mem-recap-keep-10x/v1',
    'ms1': f'{R}/v4-mem-solved-10x/v1', 'ms2': f'{R}/v4-mem-solved-10x/v2-snowiers',
    'uc1': f'{R}/v4-user-combo-10x/v1', 'uc2': f'{R}/v4-user-combo-10x/v2-snowiers', 'uc3': f'{R}/v4-user-combo-10x/v3-snowiers',
    'U': f'{K}/v4-mem-unified-10x/v1', 'Ucore': f'{K}/v4-mem-unified-core-10x/v1', 'SF': f'{K}/v4-stall-fix-10x/v1',
}
CUT = float(sys.argv[1]) if __name__ == "__main__" and len(sys.argv) > 1 else 1e9  # bench-minute cutoff


def load(d):
    rows = [json.loads(l) for l in open(d + '/score_timeline.jsonl') if l.strip()]
    games, last = {}, {}
    for r in rows:
        if r.get('kind') != 'game' or r['t_bench_s'] / 60 > CUT:
            continue
        g = r['game'][:4]
        last[g] = r
        if r['levels'] >= r['n_levels'] and g not in games:
            games[g] = r['t_bench_s'] / 60
    runs = [r for r in rows if r.get('kind') == 'run' and r['t_bench_s'] / 60 <= CUT]
    return games, last, (runs[-1]['mean_score'] if runs else None)


data = {k: load(d) for k, d in RUNS.items()}
allg = sorted({g for w, l, _ in data.values() for g in l})
print(f"{'game':6s}" + ''.join(f'{k:>7s}' for k in RUNS))
for g in allg:
    cells = []
    for k in RUNS:
        w, l, _ = data[k]
        if g in w:
            cells.append(f'{w[g]:6.0f}m')
        else:
            r = l.get(g)
            cells.append(f"{r['levels']}/{r['n_levels']}".rjust(7) if r else '      -')
    if any('m' in c for c in cells):
        print(f'{g:6s}' + ''.join(cells))
print(f"{'wins':6s}" + ''.join(f'{len(data[k][0]):7d}' for k in RUNS))
print(f"{'score':6s}" + ''.join(f'{data[k][2]:7.2f}' if data[k][2] is not None else '      -' for k in RUNS))
print(f"{'w<=60':6s}" + ''.join(f'{sum(1 for t in data[k][0].values() if t <= 60):7d}' for k in RUNS))
print(f"{'w<=90':6s}" + ''.join(f'{sum(1 for t in data[k][0].values() if t <= 90):7d}' for k in RUNS))
