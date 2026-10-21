import json, lzma, os, sys, math, random, statistics as st
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','runs','kaggle')
RO=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/runs-out'
def load(path):
    p=path+'/benchmark.json'
    f=open(p) if os.path.exists(p) else lzma.open(p+'.xz','rt')
    d=json.load(f); return {g['game_id'].split('-')[0]: float(g['final_score']) for g in d['game_runs']}
def run(name): 
    for base in (R,):
        p=f'{base}/{name}'
        if os.path.isdir(p): return load(p)
    raise SystemExit(name)
FZ_REF=['fz-arms/v4','fz-arms/v5','fz-arms/v6','fz-streams14/v1']
UC_REF=['v4-user-combo-10x/v1','v4-user-combo-10x/v2-snowiers','v4-user-combo-10x/v3-snowiers']
data={k:run(k) for k in FZ_REF+UC_REF}
def mean(r): return sum(r.values())/len(r)
# pooled run-level sd from replicate groups
groups=[[mean(data[k]) for k in UC_REF],[mean(data[k]) for k in FZ_REF]]
extra_pairs={'ledger':['fz-ledger-on-trim/v1','fz-ledger-on-trim-r2/v1'],'13x':['v4-user-combo/v1','v4-user-combo-13x/v1']}
for k,v in extra_pairs.items():
    for x in v: data[x]=run(x)
    groups.append([mean(data[x]) for x in v])
ss=sum(sum((x-st.mean(g))**2 for x in g) for g in groups); dof=sum(len(g)-1 for g in groups); sd=math.sqrt(ss/dof)
print(f'pooled run-to-run sd {sd:.2f} (dof {dof}); groups: '+'; '.join(f'{[round(x,2) for x in g]}' for g in groups))
TCRIT={1:12.71,2:4.30,3:3.18,4:2.78,5:2.57,6:2.45,7:2.36,8:2.31,9:2.26,10:2.23}
def compare(label,cands,refs):
    for c in cands+refs:
        if c not in data: data[c]=run(c)
    cm=[mean(data[c]) for c in cands]; rm=[mean(data[r]) for r in refs]
    d=st.mean(cm)-st.mean(rm); se=sd*math.sqrt(1/len(cm)+1/len(rm)); t=d/se
    games=sorted(set.intersection(*[set(data[x]) for x in cands+refs]))
    pg=[st.mean([data[c][g] for c in cands])-st.mean([data[r][g] for r in refs]) for g in games]
    won=sum(1 for x in pg if x>1e-9); lost=sum(1 for x in pg if x<-1e-9)
    random.seed(1); boots=[]
    for _ in range(4000):
        s=[random.choice(pg) for _ in pg]; boots.append(sum(s)/len(s))
    boots.sort(); lo,hi=boots[100],boots[3899]
    every=min(cm)>max(rm) and 'all above' or (max(cm)<min(rm) and 'all below' or 'overlap')
    sig='YES' if abs(t)>TCRIT[min(dof,10)] else 'no'
    print(f"{label:<34} runs {', '.join(f'{x:.2f}' for x in cm):<16} ref mean {st.mean(rm):5.2f}  diff {d:+6.2f}  t {t:+5.2f}  sig(95%) {sig:<3}  {every:<9}  games +{won}/-{lost}  per-game 95% CI [{lo:+.1f}, {hi:+.1f}]")
print('(games played: '+', '.join(f'{k}={len(v)}' for k,v in data.items() if len(v)!=25)+')')
print('\n== placebo: identical config, one run vs the other two (should NOT be significant) ==')
for i,k in enumerate(UC_REF):
    compare(f'placebo 10x {k.split("/")[1]}',[k],[x for x in UC_REF if x!=k])
print('\n== Franzen-era arms vs Franzen control runs (44.10-46.38) ==')
compare('ledger-on-trim (2 runs)',extra_pairs['ledger'],FZ_REF)
for c in ['fz-ledger-streams12/v1','fz-ledger-stallswap/v1','fz-experts16-full/v1','fz-ucb/v1','v4-stack-v1/v1']:  # fz-experts16 and fz-tput-all v7 ran the easy 10-game set
    try: compare(c,[c],FZ_REF)
    except SystemExit as e: print('missing',e)
print('\n== v4 stack vs controls ==')
compare('user-combo-10x vs fz control',UC_REF,FZ_REF)
compare('user-combo-10x vs v4-control',UC_REF,['v4-control/v1'])
compare('v4-control vs fz control',['v4-control/v1'],FZ_REF)
compare('tput-safe vs v4-control',['v4-tput-safe/v1'],['v4-control/v1'])
print('\n== arms on the user-combo-10x stack vs its 3 runs ==')
compare('13 experts (2 runs)',extra_pairs['13x'],UC_REF)
for c in ['v4-user-combo-10x-accept05/v1','v4-user-combo-16x/v1','v4-hats/v1','v4-sched-curriculum/v1','v4-uc-18s-64k/v1','v4-uc-11x/v1-snowiers']:
    try: compare(c,[c],UC_REF)
    except SystemExit as e: print('missing',e)
