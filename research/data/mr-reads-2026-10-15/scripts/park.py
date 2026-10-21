"""Per game: wall minutes, estimated generating minutes (tokens / per-stream decode rate), and long idle gaps (>= 5 min
with < 2k tokens), from benchmark.json action histories. Gaps are between consecutive actions."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runs_def import RUNS
from decomp import load
RATE = 70.0  # tok/s per stream at ~13 running (926 tok/s / 13)
r = sys.argv[1]
b = load(RUNS[r]); tot_gap = 0; out = []
for g in b['game_runs']:
    h = g['history']; n = int(g['number_of_levels']); d = int(g['levels_completed'])
    prev_t = 0.0; gaps = []; gen = 0
    for a in h:
        dt = a['wallclock_seconds'] - prev_t; tok = a['generated_tokens']; gen += tok
        idle = dt - tok / RATE
        if idle >= 300: gaps.append(idle / 60)
        prev_t = a['wallclock_seconds']
    wall = g['final_wallclock_seconds'] / 60
    tail = (g['final_wallclock_seconds'] - prev_t) / 60  # time after the last action (open level at the end)
    tot_gap += sum(gaps)
    out.append((sum(gaps), f"{g['game_id'][:4]} {d}/{n} wall {wall:5.0f} gen~{gen/RATE/60:4.0f} idle-gaps>=5min {len(gaps)} = {sum(gaps):4.0f} min  (largest {max(gaps, default=0):3.0f}) tail-after-last-action {tail:4.0f}"))
for _, s in sorted(out, reverse=True): print(s)
print(f'total idle gaps {tot_gap:.0f} min over {len(out)} games')
