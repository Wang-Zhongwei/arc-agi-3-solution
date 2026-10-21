import sys, json, lzma, glob, os, re, statistics as st
run = sys.argv[1]
d = json.load(lzma.open(os.path.join(run, 'benchmark.json.xz'), 'rt'))
w_all, w_nz, toks = [], [], []
for g in d['game_runs']:
    for h in g['history']:
        w = h.get('wallclock_seconds') or 0
        w_all.append(w)
        if w > 0.5: w_nz.append(w); toks.append(h.get('generated_tokens') or 0)
def q(v, p): v = sorted(v); return round(v[min(len(v)-1, int(p*len(v)))], 1)
print('actions', len(w_all), 'entries with wallclock>0.5s', len(w_nz))
print('per-turn wallclock s: mean %.1f median %.1f p10 %s p90 %s' % (st.mean(w_nz), st.median(w_nz), q(w_nz,.1), q(w_nz,.9)))
print('generated tokens per such entry: mean %.0f median %.0f' % (st.mean(toks), st.median(toks)))
print('sum wallclock (stream-seconds) %.0f' % sum(w_all))
# gaps between successive turn starts from transcripts
gaps = []
for f in glob.glob(os.path.join(run, 'artifacts', '*_events.jsonl.xz')):
    ts = []
    for line in lzma.open(f, 'rt'):
        r = json.loads(line)
        if r.get('type') != 'analysis': continue
        m = re.search(r'\| (\d\d):(\d\d):(\d\d) \|', r.get('transcript', '')[:300])
        if m: ts.append(int(m[1])*3600 + int(m[2])*60 + int(m[3]))
    for a, b in zip(ts, ts[1:]):
        dlt = (b - a) % 86400
        gaps.append(dlt)
print('turn-start gaps: n %d mean %.1f median %.1f p10 %s p90 %s' % (len(gaps), st.mean(gaps), st.median(gaps), q(gaps,.1), q(gaps,.9)))
