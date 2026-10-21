from load import *
print(f"{'run':<34}{'score':>6}{'lvls':>5}{'reqs':>6}{'gen M':>7}{'reas%':>6}{'tok/act':>8}{'tok/lvl k':>10}{'rt/req med':>11}{'p90':>6}{'p99':>6}{'len%':>6}{'ct/req':>7}{'nonreas/req':>12}{'act/req':>8}{'act/req p90':>12}")
for run in RUNS:
    games, R = load(run)
    gen = sum(r['ct'] for r in R); rea = sum(r['rt'] for r in R)
    acts = sum(r['n_act'] for r in R); lv = sum(g['levels'] for g in games.values())
    score = st.mean(g['score'] for g in games.values())
    rts = [r['rt'] for r in R]
    ln = sum(1 for r in R if r['fin'] == 'length')
    print(f"{run:<34}{score:6.2f}{lv:5d}{len(R):6d}{gen/1e6:7.2f}{100*rea/gen:6.1f}{gen/acts:8.0f}{gen/lv/1e3:10.1f}{pct(rts,.5):11.0f}{pct(rts,.9):6.0f}{pct(rts,.99):6.0f}{100*ln/len(R):6.2f}{gen/len(R):7.0f}{(gen-rea)/len(R):12.0f}{acts/len(R):8.2f}{pct([r['n_act'] for r in R],.9):12.0f}")
