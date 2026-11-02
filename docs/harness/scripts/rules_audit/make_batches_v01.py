import json, os, glob, sys, collections
from audit_paths import DATA_V01
V = str(DATA_V01)
run = sys.argv[1]
claims = json.load(open(f'{V}/claims_v01.json'))
cross = json.load(open(f'{V}/cross_runs_v01.json'))
bench = {g['game_id'].split('-')[0]: g for g in json.load(open(os.path.join(run, 'benchmark.json')))['game_runs']}
by_game = collections.defaultdict(list)
for c in claims: by_game[c['game']].append(c)
batches = []
for game, cs in sorted(by_game.items()):
    b = bench[game]; gid = b['game_id']
    runs = cross.get(game, [])
    n_levels = int(b.get('number_of_levels') or 0)
    cleared = {L: sum(r['levels_completed'] >= L for r in runs) for L in range(1, n_levels + 1)}
    ctx = {
        'game': game, 'game_id': gid,
        'this_run_transcript': cs[0]['transcript'],
        'this_run_events': os.path.abspath(os.path.join(run, 'artifacts', f'{gid}_p0_events.jsonl')),
        'this_run_levels_completed': b.get('levels_completed'), 'number_of_levels': n_levels,
        'this_run_actions_per_level': b.get('actions_per_level'),
        'other_runs_total': len(runs), 'other_runs_clearing_level': cleared,
        'best_other_runs': runs[:6], 'all_other_runs_index': f'{V}/cross_runs_v01.json',
    }
    keep = ['id', 'kind', 'in_section', 'dropped_by', 'level', 'step', 'analysis_step', 'thinking_line', 'shown', 'raw', 'next1', 'next2', 'next_class']
    for i in range(0, len(cs), 10):
        chunk = [{k: c.get(k) for k in keep} for c in cs[i:i + 10]]
        batches.append({'batch_id': f'{game}-{i // 10}', 'context': ctx, 'claims': chunk})
json.dump(batches, open(f'{V}/batches_v01.json', 'w'))
print(len(batches), 'batches; sizes', collections.Counter(len(b['claims']) for b in batches), 'bytes', os.path.getsize(f'{V}/batches_v01.json'))
print(json.dumps(batches[0]['context'])[:700])
