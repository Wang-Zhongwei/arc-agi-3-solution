import json, glob, os, sys, collections
root, out, skip = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
idx = collections.defaultdict(list)
for b in glob.glob(os.path.join(root, 'kaggle/*/*/v*/benchmark.json')):
    run = os.path.dirname(b)
    if os.path.abspath(run) == skip: continue
    try: d = json.load(open(b))
    except Exception: continue
    for g in d.get('game_runs', []):
        gid = g.get('game_id', ''); game = gid.split('-')[0]
        tr = sorted(glob.glob(os.path.join(run, 'transcripts', gid + '_p*.txt')))
        if not tr: continue
        try: lc = int(g.get('levels_completed') or 0)
        except ValueError: lc = 0
        idx[game].append({'run': os.path.relpath(run, root), 'levels_completed': lc, 'number_of_levels': g.get('number_of_levels'), 'transcript': os.path.abspath(tr[0])})
for game in idx: idx[game].sort(key=lambda r: -r['levels_completed'])
json.dump(idx, open(out, 'w'), indent=1)
print({g: (len(v), max(r['levels_completed'] for r in v)) for g, v in sorted(idx.items())})
