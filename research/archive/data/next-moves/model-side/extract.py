"""Extract compact per-request and per-action rows from full runs.
Output: rows/<run>.jsonl.gz with records {kind:'req'|'act'|'game', ...}."""
import json, lzma, gzip, os, sys, re, glob
R = '/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/runs/kaggle'
RUNS = sys.argv[1:]
os.makedirs('rows', exist_ok=True)

def opn(p):
    return lzma.open(p, 'rt') if p.endswith('.xz') else open(p)

def comp(msgs):
    c = dict(sys_chars=0, user_text=0, n_img=0, tool_chars=0, asst_reason=0, asst_code=0, asst_content=0, n_msgs=len(msgs), n_asst=0)
    for m in msgs:
        r = m.get('role'); ct = m.get('content')
        if r == 'system':
            c['sys_chars'] += len(ct) if isinstance(ct, str) else sum(len(p.get('text', '')) for p in ct)
        elif r == 'user':
            if isinstance(ct, str): c['user_text'] += len(ct)
            else:
                for p in ct:
                    if p.get('type') == 'text': c['user_text'] += len(p.get('text', ''))
                    elif p.get('type') == 'image_url': c['n_img'] += 1
        elif r == 'tool':
            if isinstance(ct, str): c['tool_chars'] += len(ct)
            else:
                for p in ct:
                    if p.get('type') == 'text': c['tool_chars'] += len(p.get('text', ''))
                    elif p.get('type') == 'image_url': c['n_img'] += 1
        elif r == 'assistant':
            c['n_asst'] += 1
            c['asst_reason'] += len(m.get('reasoning_content') or '')
            if isinstance(ct, str): c['asst_content'] += len(ct)
            for tc in m.get('tool_calls') or []:
                c['asst_code'] += len(tc.get('function', {}).get('arguments', ''))
    return c

for run in RUNS:
    d = os.path.join(R, run)
    out = gzip.open(f"rows/{run.replace('/', '__')}.jsonl.gz", 'wt')
    bp = os.path.join(d, 'benchmark.json.xz')
    if not os.path.exists(bp): bp = os.path.join(d, 'benchmark.json')
    bench = json.load(opn(bp))
    for g in bench['game_runs']:
        hist = g['history']
        if isinstance(hist, str):
            import ast
            try: hist = ast.literal_eval(hist)
            except Exception: hist = []
        def lit(x):
            import ast
            return ast.literal_eval(x) if isinstance(x, str) else x
        out.write(json.dumps(dict(kind='game', run=run, game=g['game_id'].split('-')[0], score=float(g['final_score']),
                                  levels=int(g['levels_completed']), nlev=int(g['number_of_levels']),
                                  apl=lit(g['actions_per_level']), base=lit(g['base_actions_per_level']),
                                  hist_tokens=[h.get('generated_tokens') for h in hist],
                                  hist_wall=[h.get('wallclock_seconds') for h in hist])) + '\n')
    for rp in sorted(glob.glob(os.path.join(d, '*_requests.jsonl*'))):
        game = os.path.basename(rp).split('-')[0]
        i = 0
        with opn(rp) as f:
            for line in f:
                if '"usage"' not in line: continue
                r = json.loads(line)
                if r.get('event') != 'response' or 'usage' not in r: continue
                u = r['usage']; pd = u.get('prompt_tokens_details') or {}
                row = dict(kind='req', run=run, game=game, i=i, step=r.get('analysis_step'), action=r.get('action'),
                           riw=r.get('request_index_within_turn'), pt=u.get('prompt_tokens'), ct=u.get('completion_tokens'),
                           rt=u.get('reasoning_tokens'), cached=pd.get('cached_tokens'), img=pd.get('image_tokens'),
                           fin=r.get('finish_reason'), effort=(r.get('chat_template_kwargs') or {}).get('reasoning_effort'))
                row.update(comp(r.get('messages') or []))
                out.write(json.dumps(row) + '\n'); i += 1
    for ep in sorted(glob.glob(os.path.join(d, 'artifacts', '*_events.jsonl*'))):
        game = os.path.basename(ep).split('-')[0]
        with opn(ep) as f:
            for line in f:
                e = json.loads(line)
                if e.get('type') != 'action': continue
                out.write(json.dumps(dict(kind='act', run=run, game=game, n=int(e['action_num']),
                    step=int(e['analysis_step']) if e.get('analysis_step') not in (None, 'None') else None,
                    level=e.get('level'), lc=e.get('level_completed') in (True, 'True'), go=e.get('game_over') in (True, 'True'),
                    bc=e.get('board_changed') in (True, 'True'), name=e.get('action_display'))) + '\n')
    out.close()
    print('done', run, flush=True)
