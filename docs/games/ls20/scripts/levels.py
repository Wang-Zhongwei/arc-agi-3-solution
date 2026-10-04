import os, json, random, statistics, sys
from pathlib import Path

S = json.load(open(Path(__file__).with_name('summary.json')))
R = Path(os.environ.get('ARC3_RUNS_DIR', Path(__file__).resolve().parents[4] / 'ARC3-Inference/runs')) / 'kaggle'
RUNS = {'v0': 'jonathanwang2022/api-signatures-and-frame-diff/v03', 'v1': 'jonathanwang2022/api-signatures-and-frame-diff/v05',
        'v2': 'ticbudja/api-signatures-and-frame-diff/v01'}
if len(sys.argv) > 1:
    RUNS['v3'] = sys.argv[1]

print('Actions spent on level k (completed passes only); n = passes that cleared it')
for arm, s in S.items():
    gaps = {}
    for p in s['per_pass']:
        prev = 0
        for k, at in enumerate(p['level_up'], 1):
            gaps.setdefault(k, []).append(at - prev); prev = at
    print(arm, '  '.join(f"L{k}: {statistics.mean(v):5.0f} (n={len(v)})" for k, v in sorted(gaps.items())))

print('\nDeaths: (pass, level, action_num)')
for arm, rel in RUNS.items():
    out = []
    for f in sorted((R / rel / 'artifacts').glob('*_events.jsonl')):
        for l in open(f):
            e = json.loads(l)
            if e.get('type') == 'action' and e.get('game_over'):
                out.append((f.stem.split('_p')[1].split('_')[0], e.get('level'), e['action_num']))
    print(arm, out)

scores = {}
for arm, rel in RUNS.items():
    e = json.load(open(R / rel / 'evaluation.json'))
    scores[arm] = list(next(iter(e['games'].values()))['trial_scores'].values())


def perm(a, b, n=20000, seed=0):
    rnd = random.Random(seed); obs = statistics.mean(a) - statistics.mean(b); pool = a + b; hits = 0
    for _ in range(n):
        rnd.shuffle(pool)
        if abs(statistics.mean(pool[:len(a)]) - statistics.mean(pool[len(a):])) >= abs(obs) - 1e-12: hits += 1
    return obs, hits / n


print('\nScore mean ± sd:', {a: (round(statistics.mean(v), 2), round(statistics.stdev(v), 2)) for a, v in scores.items()})
for a, b in [('v1', 'v0'), ('v2', 'v0'), ('v2', 'v1')] + ([('v3', 'v2'), ('v3', 'v0')] if 'v3' in scores else []):
    d, p = perm(scores[a], scores[b]); print(f'{a} - {b}: {d:+.2f}  two-sided permutation p = {p:.3f}')
