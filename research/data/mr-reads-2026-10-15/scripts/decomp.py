"""Split each run's mean score into: earned, lost to action inefficiency on cleared levels, lost to uncleared levels."""
import json, lzma, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runs_def import RUNS
def load(d):
    for f in ('benchmark.json', 'benchmark.json.xz'):
        p = os.path.join(d, f)
        if os.path.exists(p):
            return json.load(lzma.open(p) if f.endswith('.xz') else open(p))
def decomp(g):
    n = int(g['number_of_levels']); base = g['base_actions_per_level']; apl = g['actions_per_level']; done = int(g['levels_completed'])
    W = n * (n + 1) / 2; earned = eff = unc = 0.0; per = []
    for i in range(n):
        w = i + 1; a = apl[i] if i < len(apl) else 0
        if i < done and a > 0:
            ls = min(115.0, (base[i] / a) ** 2 * 100); earned += w * ls; eff += w * max(0.0, 100 - ls); per.append((i + 1, a, base[i], ls))
        else:
            unc += w * 100
    return earned / W, eff / W, unc / W, per, done, n
runs = sys.argv[1].split(',')
for r in runs:
    b = load(RUNS[r]); tot = [0, 0, 0]; wins = 0; nl = 0; ratio = []
    for g in b['game_runs']:
        e, f, u, per, done, n = decomp(g); tot[0] += e; tot[1] += f; tot[2] += u; wins += done == n; nl += done
        ratio += [a / bb for _, a, bb, _ in per]
    k = len(b['game_runs']); ratio.sort()
    print(f'{r:6s} earned {tot[0]/k:5.2f}  lost-efficiency {tot[1]/k:5.2f}  lost-uncleared {tot[2]/k:5.2f}  wins {wins}  levels {nl}  actions/baseline median {ratio[len(ratio)//2]:.2f} p75 {ratio[3*len(ratio)//4]:.2f}')
