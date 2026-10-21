"""Idle gaps (>= 5 min with few tokens) per run, split by whether a level was cleared within 3 min before the gap began."""
import json, os, sys
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runs_def import RUNS
from decomp import load
RATE = 70.0
for r in sys.argv[1].split(','):
    b = load(RUNS[r]); rows = [json.loads(l) for l in open(RUNS[r] + '/score_timeline.jsonl') if l.strip()]
    bench0 = datetime.fromisoformat(b['start_time']).timestamp() if isinstance(b.get('start_time'), str) else None
    after = other = 0.0; n_after = n_other = 0
    for g in b['game_runs']:
        k = g['game_id'][:4]
        lv = sorted([(x['t_bench_s'], x['levels']) for x in rows if x.get('kind') == 'game' and x['game'][:4] == k])
        clears = [t for (t, l), (t0, l0) in zip(lv[1:], lv[:-1]) if l > l0]
        start = datetime.fromisoformat(g['started_at']).timestamp() - bench0
        prev = 0.0
        for a in g['history']:
            dt = a['wallclock_seconds'] - prev; idle = dt - a['generated_tokens'] / RATE
            if idle >= 300:
                t_gap = start + prev
                if any(t_gap - 240 <= c <= t_gap + 60 for c in clears): after += idle; n_after += 1
                else: other += idle; n_other += 1
            prev = a['wallclock_seconds']
    print(f'{r:6s} gaps right after a level clear: {n_after:3d} = {after/60:5.0f} min | other gaps: {n_other:3d} = {other/60:5.0f} min')
