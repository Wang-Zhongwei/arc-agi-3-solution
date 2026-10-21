# Per-(game, losing run) probability that each lever flips the loss to a score-100 win.
# Runs: C control, A 10x/v1, B 10x/v2, D 10x/v3, U user-combo, T 13x, K curriculum, X accept05, H hats
import itertools, math
RUNS="CABDUTKXH"
WINS={ # score-100 wins (benchmark.json.xz final_score>=100)
 'ar25':"CABDUTKXH",'cd82':"AD",'ft09':"CABDUTX",'lp85':"CABDUTKX",'m0r0':"CUTK",
 'r11l':"CAUTKX",'sb26':"CAUXH",'sc25':"K",'tn36':"A",'tr87':"ABUT",'tu93':"CD",'vc33':"AUX"}
FULL={ # all levels cleared
 'ar25':"CABDUTKXH",'cd82':"ABDUTKH",'ft09':"CABDUTKX",'lp85':"CABDUTKXH",'m0r0':"CUTK",
 'r11l':"CAUTKX",'sb26':"CABDUTKXH",'sc25':"ABK",'tn36':"A",'tr87':"ABUTKX",'tu93':"CD",'vc33':"AUX"}
L=['FINAL','CARD','EXEC','NOVEL','PARK','TWIN','MARATHON','SANDBOX','BAR','SEP','NOOP']
F={ # (game,run): {lever:p}
('cd82','C'):dict(CARD=.1,EXEC=.1,PARK=.1),
('cd82','B'):dict(CARD=.3,EXEC=.2,NOOP=.05),
('cd82','U'):dict(CARD=.5,EXEC=.5,NOOP=.1),
('cd82','T'):dict(CARD=.5,EXEC=.3,NOOP=.15),
('cd82','K'):dict(CARD=.3,EXEC=.3,NOOP=.1),
('cd82','X'):dict(CARD=.05,PARK=.05),
('cd82','H'):dict(CARD=.5,EXEC=.2,NOOP=.15),
('ft09','K'):dict(NOVEL=.3,EXEC=.1),
('ft09','H'):dict(NOVEL=.5,MARATHON=.2,CARD=.1),
('m0r0','A'):dict(SEP=.5,EXEC=.2,CARD=.2),
('m0r0','B'):dict(NOVEL=.25,CARD=.1,PARK=.25),
('m0r0','D'):dict(CARD=.5,NOVEL=.4),
('m0r0','X'):dict(PARK=.5,SANDBOX=.3,CARD=.3),
('m0r0','H'):dict(CARD=.4,NOVEL=.3,PARK=.25),
('r11l','B'):dict(PARK=.6,MARATHON=.3,CARD=.3,FINAL=.1),
('r11l','D'):dict(PARK=.45,MARATHON=.5,CARD=.15),
('r11l','H'):dict(PARK=.3,MARATHON=.2,NOVEL=.2),
('sb26','B'):dict(EXEC=.5),('sb26','K'):dict(EXEC=.5),
('sb26','D'):dict(EXEC=.3),('sb26','T'):dict(EXEC=.3),
('sc25','C'):dict(PARK=.35,FINAL=.1,CARD=.1),
('sc25','A'):dict(EXEC=.3),('sc25','B'):dict(EXEC=.25),
('sc25','D'):dict(FINAL=.6,BAR=.1),
('sc25','U'):dict(CARD=.4,PARK=.2),
('sc25','T'):dict(PARK=.3,CARD=.2),
('sc25','X'):dict(PARK=.6,FINAL=.3),
('sc25','H'):dict(PARK=.25,MARATHON=.1),
('tn36','C'):dict(TWIN=.2,PARK=.2),
('tn36','B'):dict(PARK=.4,CARD=.3,TWIN=.3,FINAL=.05),
('tn36','D'):dict(TWIN=.35),('tn36','U'):dict(TWIN=.35),
('tn36','T'):dict(PARK=.4,BAR=.3,CARD=.2,TWIN=.3),
('tn36','K'):dict(TWIN=.15,NOVEL=.1),
('tn36','X'):dict(TWIN=.35),('tn36','H'):dict(TWIN=.3),
('tr87','C'):dict(FINAL=.4,SANDBOX=.1),
('tr87','D'):dict(FINAL=.5,SANDBOX=.4,CARD=.35),
('tr87','H'):dict(CARD=.1,SANDBOX=.1),
('tr87','K'):dict(EXEC=.1),('tr87','X'):dict(EXEC=.15),
('tu93','A'):dict(CARD=.65),('tu93','B'):dict(CARD=.3),
('tu93','U'):dict(CARD=.3,FINAL=.4),('tu93','T'):dict(BAR=.6),
('tu93','X'):dict(CARD=.15),('tu93','H'):dict(CARD=.3),
('tu93','K'):dict(PARK=.2,FINAL=.1),
('vc33','C'):dict(FINAL=.8),('vc33','B'):dict(FINAL=.5,EXEC=.4),
('vc33','D'):dict(NOVEL=.6),('vc33','T'):dict(FINAL=.1,PARK=.2,EXEC=.2),
('vc33','K'):dict(NOVEL=.45),('vc33','H'):dict(NOVEL=.55),
}
def pg(game, runs, levers):
    w=0.0
    for r in runs:
        if r in WINS[game]: w+=1; continue
        ps=[F.get((game,r),{}).get(l,0) for l in levers]
        w+=1-math.prod(1-p for p in ps)
    return w/len(runs)
def table(runs,label):
    print(f"== {label} (runs {runs})")
    base={g:pg(g,runs,[]) for g in WINS}
    E0=sum(base.values()); print(f"baseline E[wins]={E0:.2f}")
    for l in L:
        e=sum(pg(g,runs,[l]) for g in WINS); 
        per={g:round(pg(g,runs,[l])-base[g],2) for g in WINS if pg(g,runs,[l])-base[g]>1e-9}
        print(f"  {l:9s} +{e-E0:.2f}  {per}")
    return base
b=table(RUNS,"all 9 runs")
b3=table("ABD","user-combo-10x only")
def pbinom(ps):
    dist=[1.0]
    for p in ps:
        nd=[0]*(len(dist)+1)
        for k,q in enumerate(dist): nd[k]+=q*(1-p); nd[k+1]+=q*p
        dist=nd
    return dist
combos={'none':[], 'FINAL+CARD':['FINAL','CARD'], 'FINAL+CARD+EXEC':['FINAL','CARD','EXEC'],
 'FINAL+CARD+EXEC+NOVEL':['FINAL','CARD','EXEC','NOVEL'],
 'top5 (+PARK)':['FINAL','CARD','EXEC','NOVEL','PARK'],'all':L}
for runs,lab in [(RUNS,'9 runs'),("ABD",'10x only')]:
    print('== combos',lab)
    for name,ls in combos.items():
        ps={g:pg(g,runs,ls) for g in WINS}
        E=sum(ps.values()); d=pbinom(list(ps.values()))
        p12=d[12]; p10=sum(d[10:]); 
        print(f"  {name:24s} E={E:.2f} P(>=10)={p10:.3f} P(12)={p12:.4f}  " + ' '.join(f"{g}:{ps[g]:.2f}" for g in sorted(ps)))
# full-clear baseline
print('full-clear E 9 runs', sum(len(v) for v in FULL.values())/9, '10x', sum(sum(1 for r in "ABD" if r in v) for v in FULL.values())/3)
print('== FULL-CLEAR definition')
WINS.clear(); WINS.update(FULL)
for runs,lab in [(RUNS,'9 runs'),("ABD",'10x only')]:
    for name,ls in combos.items():
        ps={g:pg(g,runs,ls) for g in WINS}
        d=pbinom(list(ps.values()))
        print(f"  {lab} {name:24s} E={sum(ps.values()):.2f} P(>=10)={sum(d[10:]):.3f} P(>=11)={sum(d[11:]):.3f} P(12)={d[12]:.4f}")
print('== what reaching 12 needs')
for p in [0.85,0.9,0.95,0.97]:
    d=pbinom([p]*12); print(f' 12 games at p={p}: P(12)={d[12]:.3f}  E={12*p:.1f}')
for p,extra in [(0.9,[0.3]*4),(0.85,[0.4]*4),(0.9,[0.5]*3)]:
    d=pbinom([p]*12+extra); print(f' 12 at {p} + {len(extra)} extra at {extra[0]}: P(>=12)={sum(d[12:]):.3f} E={12*p+sum(extra):.1f}')
