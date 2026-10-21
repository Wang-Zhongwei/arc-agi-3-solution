"""Per run: response finish reasons, completion tokens wasted on cut-off replies (finish=length), and
replies with no tool call."""
import json, glob, os, sys, lzma
from collections import Counter
from runs_def import RUNS
def op(p): return lzma.open(p,'rt') if p.endswith('.xz') else open(p)
for run in sys.argv[1].split(','):
    d = RUNS[run]; fin = Counter(); comp = Counter(); n = 0
    for f in glob.glob(f'{d}/*_p0_requests.jsonl*'):
        for l in op(f):
            if '"event": "response"' not in l and '"event":"response"' not in l: continue
            x = json.loads(l)
            if x.get('event') != 'response': continue
            u = x.get('usage') or {}
            fr = x.get('finish_reason'); fin[fr] += 1; comp[fr] += u.get('completion_tokens', 0) or 0; n += 1
    tot = sum(comp.values())
    print(run, n, 'responses', dict(fin), {k: f'{v/1e6:.2f}M' for k, v in comp.items()}, f'length share of completion {comp["length"]/max(tot,1):.1%}')
