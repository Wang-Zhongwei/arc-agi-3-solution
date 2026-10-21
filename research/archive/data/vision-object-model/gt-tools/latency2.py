import sys, json, lzma, os, statistics as st
run = sys.argv[1]
d = json.load(lzma.open(os.path.join(run, 'benchmark.json.xz'), 'rt'))
def q(v, p): v = sorted(v); return round(v[min(len(v)-1, int(p*len(v)))], 1)
turn_dt, turn_tok, act_per_turn = [], [], []
for g in d['game_runs']:
    prev_w = 0.0; n_in_turn = 0; started = False
    for h in g['history']:
        w = h.get('wallclock_seconds') or 0.0
        if (h.get('generated_tokens') or 0) > 0:
            if started: act_per_turn.append(n_in_turn)
            turn_dt.append(w - prev_w); turn_tok.append(h['generated_tokens'])
            n_in_turn = 0; started = True
        n_in_turn += 1
        prev_w = w
    if started: act_per_turn.append(n_in_turn)
print('turns', len(turn_dt))
print('turn wall dt s: mean %.1f median %.1f p10 %s p90 %s' % (st.mean(turn_dt), st.median(turn_dt), q(turn_dt,.1), q(turn_dt,.9)))
print('gen tokens/turn: mean %.0f median %.0f' % (st.mean(turn_tok), st.median(turn_tok)))
print('actions/turn: mean %.2f median %.0f' % (st.mean(act_per_turn), st.median(act_per_turn)))
