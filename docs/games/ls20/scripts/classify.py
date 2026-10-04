import os, json, re, collections, sys
from pathlib import Path

R = Path(os.environ.get('ARC3_RUNS_DIR', Path(__file__).resolve().parents[4] / 'ARC3-Inference/runs')) / 'kaggle'
ARMS = {'v0': R / 'jonathanwang2022/api-signatures-and-frame-diff/v03', 'v1': R / 'jonathanwang2022/api-signatures-and-frame-diff/v05',
        'v2': R / 'ticbudja/api-signatures-and-frame-diff/v01'}
if len(sys.argv) > 1:
    ARMS['v3'] = R / sys.argv[1]
CALL_RE = re.compile(r'\[TOOL CALL: python\]\n(.*?)\n\[TOOL RESULT: python\]\n(.*?)(?=\n\[[A-Z][A-Z _]+\]|\Z)', re.S)
ERR_RE = re.compile(r'^(\w+(?:Error|Exception)): (.*)$', re.M)
HARNESS = r'View|_Lazy|HistoryEntry|Transition|Frame|segmentation|frame_diff|last_animation|two slices|slice|crop|hud'
RULES = [
    ('harness refusal', lambda et, m, c: et in ('StaleStateActionError', 'TerminalStateActionError')),
    ('sandbox limit', lambda et, m, c: (et == 'ImportError' and 'not allowed' in m) or m.startswith("name 'open'") or m.startswith("name '__import__'")),
    ('API misuse', lambda et, m, c: bool(re.search(HARNESS, m)) or (et == 'AttributeError' and re.search(r"'(dict|str|list|int|tuple)' object has no attribute '(frame|action|result|segmentation|ascii|step|level)'", m))
        or (et in ('KeyError', 'TypeError') and re.search(r"frame_diff|\.segmentation|last_action_call_result|transitions|history\[", c[-400:]) is not None and et != 'NameError')),
    ('own code bug', lambda et, m, c: True),
]
res = {}
samples = collections.defaultdict(list)
for arm, run in ARMS.items():
    cat = collections.Counter(); calls = 0; api_msgs = collections.Counter()
    for f in run.glob('artifacts/ls20-9607627b_p*_events.jsonl'):
        for l in open(f):
            e = json.loads(l)
            if e.get('type') != 'analysis': continue
            for code, out in CALL_RE.findall(e['transcript']):
                calls += 1
                if not ('Traceback (most recent call last)' in out or '\nerror:\n' in out or out.lstrip().startswith('error:')): continue
                m = ERR_RE.findall(out); et, msg = m[-1] if m else ('Other', out.strip()[:120])
                c = next(name for name, rule in RULES if rule(et, msg, code))
                cat[c] += 1
                if c == 'API misuse':
                    api_msgs[re.sub(r'\d+', 'N', f'{et}: {msg}')[:110]] += 1
    res[arm] = (calls, cat, api_msgs)
for arm, (calls, cat, api) in res.items():
    tot = sum(cat.values())
    print(f"{arm}: calls={calls} errors={tot} ({tot/calls:.1%}) " + ', '.join(f"{k}={cat[k]} ({cat[k]/calls:.1%})" for k in ['API misuse', 'own code bug', 'harness refusal', 'sandbox limit']))
    for k, v in api.most_common(8): print(f'     {v:2d} {k}')
