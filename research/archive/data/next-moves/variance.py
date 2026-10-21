from load import *
import statistics as st, math, random
GROUPS={'uc10x':['v4-user-combo-10x/v1','v4-user-combo-10x/v2-snowiers','v4-user-combo-10x/v3-snowiers'],
        'fz':['fz-arms/v4','fz-arms/v5','fz-arms/v6','fz-streams14/v1'],
        'ledger':['fz-ledger-on-trim/v1','fz-ledger-on-trim-r2/v1'],
        '13x':['v4-user-combo/v1','v4-user-combo-13x/v1']}
RUNS={k:{r:load(r) for r in v} for k,v in GROUPS.items()}
if __name__=='__main__':
    # 0. sanity: recompute score from levels
    recompute=game_score
    bad=0
    for k,rs in RUNS.items():
        for r,gs in rs.items():
            for gid,g in gs.items():
                if abs(recompute(g)-g['score'])>0.05: bad+=1
    #print('score recompute mismatches:',bad)
    # 1. per-game variance vs run-mean variance
    print('\n== run-mean sd: observed vs implied by independent games ==')
    for k,rs in RUNS.items():
        names=list(rs); games=sorted(set.intersection(*[set(rs[r]) for r in names]))
        means=[mean(rs[r][g]['score'] for g in games) for r in names]
        vg=[st.variance([rs[r][g]['score'] for r in names]) for g in games]
        implied=math.sqrt(sum(vg))/len(games)
        print(f'{k:7} runs={len(names)} games={len(games)} means={[round(m,2) for m in means]} obs sd={st.stdev(means):.2f} implied-indep sd={implied:.2f}  mean per-game sd={math.sqrt(mean(vg)):.1f}')
    # pooled
    ss=0;dof=0;vgs={}
    for k,rs in RUNS.items():
        names=list(rs); games=sorted(set.intersection(*[set(rs[r]) for r in names]))
        for g in games:
            xs=[rs[r][g]['score'] for r in names]; m=mean(xs)
            vgs.setdefault(g,[]).append((sum((x-m)**2 for x in xs),len(xs)-1))
    pooled_g={g:sum(a for a,b in v)/sum(b for a,b in v) for g,v in vgs.items()}
    print('pooled per-game sd (all 4 groups, dof 7 each):')
    for g,v in sorted(pooled_g.items(),key=lambda x:-x[1]): print(f'  {g} {math.sqrt(v):5.1f}',end='')
    print('\n  implied run-mean sd from pooled per-game var:',round(math.sqrt(sum(pooled_g.values()))/25,2))
    # top games share of variance
    tot=sum(pooled_g.values()); srt=sorted(pooled_g.values(),reverse=True)
    print('  share of variance in top 5 / 7 / 10 games:',[round(sum(srt[:k])/tot,2) for k in (5,7,10)])
    # uc10x per-game table
    print('\n== uc10x per game: scores, levels ==')
    rs=RUNS['uc10x']; names=list(rs)
    for g in sorted(rs[names[0]]):
        print(f"  {g}: "+'  '.join(f"{rs[r][g]['score']:6.1f}/{rs[r][g]['lc']}of{rs[r][g]['n']}" for r in names))
