"""Behaviour on stalled vs normal levels, from per-game event logs of the 12 full runs.
Per (run, game, level): actions, turns (analysis steps), game overs, no-change actions, model RESETs,
repeated (board, action) pairs, distinct boards visited, plus level tokens from the score timeline.
Writes stall_feats.json; prints stalled vs normal summaries."""
import glob, hashlib, json, lzma, os, sys
from collections import defaultdict
from statistics import median
sys.path.insert(0, os.path.dirname(__file__))
from runs_def import RUNS

OUT = os.path.join(os.path.dirname(__file__), 'stall_feats.json')


def op(p):
    return lzma.open(p, 'rt') if p.endswith('.xz') else open(p)


def level_tokens(d):
    per = defaultdict(list)
    for l in open(d + '/score_timeline.jsonl'):
        r = json.loads(l) if l.strip() else {}
        if r.get('kind') == 'game':
            per[r['game'][:4]].append(r)
    out = {}
    for g, rs in per.items():
        rs.sort(key=lambda r: r['t_bench_s'])
        tok0, t0, done = 0, rs[0]['t_bench_s'], 0
        for r in rs:
            while r['levels'] > done:
                done += 1
                out[(g, done)] = (r['tokens'] - tok0, True, (r['t_bench_s'] - t0) / 60)
                tok0, t0 = r['tokens'], r['t_bench_s']
        if done < rs[0]['n_levels']:
            out[(g, done + 1)] = (rs[-1]['tokens'] - tok0, False, (rs[-1]['t_bench_s'] - t0) / 60)
    return out


rows = []
if os.path.exists(OUT) and '--redo' not in sys.argv:
    rows = json.load(open(OUT))
else:
    for run, d in RUNS.items():
        lt = level_tokens(d)
        for f in sorted(glob.glob(f'{d}/artifacts/*_p0_events.jsonl*')):
            g = os.path.basename(f)[:4]
            feats = defaultdict(lambda: dict(actions=0, turns=0, overs=0, nochange=0, resets=0, repeats=0, boards=set(), seen=set()))
            prev_board = None
            for line in op(f):
                e = json.loads(line)
                t = e.get('type')
                lv = int(e.get('level') or 1)
                if t == 'initial':
                    prev_board = hashlib.md5(json.dumps(e['board']).encode()).hexdigest()
                    continue
                if t == 'analysis':
                    feats[lv]['turns'] += 1
                    continue
                if t != 'action':
                    continue
                # level the action was taken on: level before a clear
                a_lv = lv - 1 if e.get('level_completed') in (True, 'True') and lv > 1 else lv
                x = feats[a_lv]
                x['actions'] += 1
                key = (prev_board, e.get('action_display'))
                if key in x['seen']:
                    x['repeats'] += 1
                x['seen'].add(key)
                x['boards'].add(prev_board)
                x['overs'] += e.get('game_over') in (True, 'True')
                x['nochange'] += e.get('board_changed') in (False, 'False')
                x['resets'] += e.get('action_name') == 'RESET' and int(e.get('action_num') or 0) > 1
                prev_board = hashlib.md5(json.dumps(e['board']).encode()).hexdigest()
            for lv, x in feats.items():
                tok, clr, mins = lt.get((g, lv), (None, None, None))
                if tok is None:
                    continue
                rows.append(dict(run=run, game=g, lv=lv, tok=tok, cleared=clr, mins=mins,
                                 **{k: v for k, v in x.items() if k not in ('boards', 'seen')}, distinct=len(x['boards'])))
        print(run, len(rows), flush=True)
    json.dump(rows, open(OUT, 'w'))

# stall definition: tok >= max(60k, 2x median of this run-game's earlier cleared levels) (L1: 90k)
by_rg = defaultdict(list)
for r in rows:
    by_rg[(r['run'], r['game'])].append(r)
for seq in by_rg.values():
    seq.sort(key=lambda r: r['lv'])
    for i, r in enumerate(seq):
        prev = [p['tok'] for p in seq[:i] if p['cleared']]
        r['stall'] = r['tok'] >= (max(60e3, 2 * median(prev)) if prev else 90e3)


def summ(sel, name):
    n = len(sel)
    s = lambda k: sum(r[k] for r in sel)
    tok = s('tok') / 1e3
    print(f"{name:28s} n={n:4d} tok/lvl {tok/n:6.0f}k  act/10k {10*s('actions')/tok:5.1f}  turns/10k {10*s('turns')/tok:4.1f}  "
          f"act/turn {s('actions')/max(1,s('turns')):4.1f}  overs/lvl {s('overs')/n:4.1f}  nochange% {100*s('nochange')/max(1,s('actions')):4.1f}  "
          f"repeat% {100*s('repeats')/max(1,s('actions')):4.1f}  resets/lvl {s('resets')/n:4.1f}  tok/min {tok/max(1e-9,sum(r['mins'] for r in sel)):4.2f}k")


summ([r for r in rows if not r['stall'] and r['cleared']], 'normal, cleared')
summ([r for r in rows if r['stall'] and r['cleared']], 'stalled, cleared later')
summ([r for r in rows if r['stall'] and not r['cleared']], 'stalled, never cleared')
summ([r for r in rows if not r['stall'] and not r['cleared']], 'short, unfinished (run end)')
