"""Time-weighted #running-req and queue from SGLang decode lines; throughput by batch size."""
import re, sys, lzma
from datetime import datetime
from collections import defaultdict
p = sys.argv[1]
f = lzma.open(p, 'rt') if p.endswith('.xz') else open(p, errors='replace')
pat = re.compile(r'^\[(\S+ \S+)\] Decode batch, #running-req: (\d+).*?gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)')
ev = []
for line in f:
    m = pat.match(line)
    if m: ev.append((datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S').timestamp(), int(m.group(2)), float(m.group(3)), int(m.group(4))))
t0 = ev[0][0]; T = ev[-1][0] - t0
occ = defaultdict(float); q = 0.0; gaps = 0.0; tp = defaultdict(list)
for (t, r, g, qq), (t2, *_ ) in zip(ev, ev[1:]):
    dt = t2 - t
    if dt > 10: gaps += dt; continue
    occ[r] += dt; q += qq * dt; tp[r].append(g)
busy = sum(occ.values())
mean_r = sum(r * w for r, w in occ.items()) / busy
print(f'{p.split("/")[-3]}: span {T/60:.0f} min, decode-logged {busy/60:.0f} min, gaps>10s {gaps/60:.1f} min; mean running {mean_r:.2f}; mean queue {q/busy:.2f}')
print('  running: ' + ' '.join(f'{r}:{occ[r]/busy*100:.0f}%' for r in sorted(occ) if occ[r]/busy > 0.01))
print('  tok/s by running: ' + ' '.join(f'{r}:{sorted(v)[len(v)//2]:.0f}' for r, v in sorted(tp.items()) if len(v) > 20))
