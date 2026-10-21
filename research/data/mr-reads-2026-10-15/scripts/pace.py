"""Per run: levels at 30/60/90 min, tokens at those times, levels per M tokens, gen tokens per action, run tok/s."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runs_def import RUNS
from decomp import load
for r in sys.argv[1].split(','):
    rows = [json.loads(l) for l in open(RUNS[r] + '/score_timeline.jsonl') if l.strip()]
    run = [x for x in rows if x.get('kind') == 'run']
    def at(m):
        c = [x for x in run if x['t_bench_s'] <= m * 60]
        return c[-1] if c else run[0]
    s = []
    for m in (30, 60, 90, 120):
        x = at(m); s.append(f"{m}m L{x['levels']:3d} {x['tokens']/1e6:4.2f}M a{x['actions']:5d}")
    b = load(RUNS[r]); acts = sum(len(g['history']) for g in b['game_runs'])
    gen_calls = [h['generated_tokens'] for g in b['game_runs'] for h in g['history'] if h['generated_tokens'] > 0]
    x = run[-1]
    print(f"{r:6s} " + ' | '.join(s) + f" | lv/Mtok {x['levels']/(x['tokens']/1e6):.1f} tok/act {x['tokens']/max(1,x['actions']):.0f} calls-with-gen {len(gen_calls)} med gen {sorted(gen_calls)[len(gen_calls)//2]}")
