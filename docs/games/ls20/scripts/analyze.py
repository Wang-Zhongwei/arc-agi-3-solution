import os, json, re, collections, statistics, sys
from pathlib import Path

R = Path(os.environ.get('ARC3_RUNS_DIR', Path(__file__).resolve().parents[4] / 'ARC3-Inference/runs')) / 'kaggle'
ARMS = {
    'v0': R / 'jonathanwang2022/api-signatures-and-frame-diff/v03',
    'v1': R / 'jonathanwang2022/api-signatures-and-frame-diff/v05',
    'v2': R / 'ticbudja/api-signatures-and-frame-diff/v01',
}
if len(sys.argv) > 1:
    ARMS['v3'] = R / sys.argv[1]

CALL_RE = re.compile(r'\[TOOL CALL: python\]\n(.*?)\n\[TOOL RESULT: python\]\n(.*?)(?=\n\[[A-Z][A-Z _]+\]|\Z)', re.S)
ERR_RE = re.compile(r'^(\w+(?:Error|Exception)): (.*)$', re.M)


def passes(run):
    for p in sorted(run.glob('artifacts/ls20-9607627b_p*_events.jsonl'), key=lambda x: int(x.stem.split('_p')[1].split('_')[0])):
        yield p, [json.loads(l) for l in open(p)]


summary = {}
examples = collections.defaultdict(list)
for arm, run in ARMS.items():
    agg = collections.Counter()
    per_pass = []
    err_types = collections.Counter()
    err_msgs = collections.Counter()
    for path, rows in passes(run):
        acts = [r for r in rows if r.get('type') == 'action']
        ans = [r for r in rows if r.get('type') == 'analysis']
        deaths = sum(1 for r in acts if r.get('game_over'))
        level_up = [r['action_num'] for r in acts if r.get('level_completed')]
        calls = errs = 0
        for a in ans:
            for code, res in CALL_RE.findall(a['transcript']):
                calls += 1
                if 'Traceback (most recent call last)' in res or res.lstrip().startswith('error:') or '\nerror:\n' in res:
                    errs += 1
                    m = ERR_RE.findall(res)
                    et, msg = (m[-1] if m else ('Other', res.strip()[:120]))
                    err_types[et] += 1
                    key = re.sub(r"\d+", "N", f'{et}: {msg}')[:140]
                    err_msgs[key] += 1
                    if len(examples[(arm, key)]) < 1:
                        examples[(arm, key)].append((path.name, a['analysis_step'], code[-700:], res[-500:]))
        per_pass.append(dict(pass_=path.stem.split('_p')[1].split('_')[0], turns=len(ans), actions=len(acts), deaths=deaths,
                             levels=len(level_up), level_up=level_up, calls=calls, errs=errs))
    summary[arm] = dict(per_pass=per_pass, err_types=err_types, err_msgs=err_msgs)

for arm, s in summary.items():
    pp = s['per_pass']
    tot = lambda k: sum(p[k] for p in pp)
    print(f"\n===== {arm}  turns={tot('turns')} actions={tot('actions')} deaths={tot('deaths')} levels={tot('levels')} "
          f"tool_calls={tot('calls')} tool_errors={tot('errs')} err_rate={tot('errs')/max(1,tot('calls')):.3f} "
          f"calls/turn={tot('calls')/tot('turns'):.2f}")
    for p in pp:
        print(f"  p{p['pass_']}: turns={p['turns']:3d} acts={p['actions']:3d} deaths={p['deaths']:2d} levels={p['levels']} "
              f"calls={p['calls']:3d} errs={p['errs']:3d} level_up_at={p['level_up']}")
    print('  error types:', dict(s['err_types'].most_common(8)))
    for k, v in s['err_msgs'].most_common(14):
        print(f'    {v:3d}  {k}')

json.dump({a: dict(per_pass=s['per_pass'], err_types=s['err_types'], err_msgs=s['err_msgs'].most_common(40)) for a, s in summary.items()},
          open(Path(__file__).with_name('summary.json'), 'w'), indent=1)
json.dump({f'{a}|{k}': v for (a, k), v in examples.items()}, open(Path(__file__).with_name('examples.json'), 'w'), indent=1)
