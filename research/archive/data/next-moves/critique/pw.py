import math, random
from statistics import NormalDist
N=NormalDist()
# variance heterogeneity: uc10x vs fz control groups
uc=[56.94,52.70,49.09]; fz=[46.25,45.92,44.10,46.38]; led=[52.56,49.14]; x13=[49.61,47.34]
def ss(v): m=sum(v)/len(v); return sum((x-m)**2 for x in v)
for nm,v in [('uc',uc),('fz',fz),('led',led),('13x',x13)]: print(nm, 'sd %.2f'%math.sqrt(ss(v)/(len(v)-1)), 'ss %.2f'%ss(v))
print('pooled all', math.sqrt((ss(uc)+ss(fz)+ss(led)+ss(x13))/7), 'pooled current-stack(uc+13x)', math.sqrt((ss(uc)+ss(x13))/3))
from scipy import stats
F=(ss(uc)/2)/(ss(fz)/3); print('F uc/fz',F,'p(two-sided)',2*(1-stats.f.cdf(F,2,3)))
# chi2 CI on uc sd
for nm,v,df in [('uc',uc,2),('pooled',None,7)]:
    s2 = ss(uc)/2 if v else (ss(uc)+ss(fz)+ss(led)+ss(x13))/7
    lo=math.sqrt(df*s2/stats.chi2.ppf(0.975,df)); hi=math.sqrt(df*s2/stats.chi2.ppf(0.025,df)); print(nm,'95% CI sd',round(lo,2),round(hi,2))
# factorial adopt rule beta>1.5 (and logHR>0 approx prob q)
for sd in (2.47,3.34,3.93):
    se=sd/2
    p_null=1-N.cdf(1.5/se); p3=1-N.cdf((1.5-3)/se); p2=1-N.cdf((1.5-2)/se); pneg=1-N.cdf((1.5+1.5)/se)
    pw3=1-N.cdf(1.645-3/se)
    print(f'sd {sd}: se {se:.2f} power(+3, z>1.645) {pw3:.2f} | rule beta>1.5: P(adopt|0) {p_null:.2f} P(adopt|+2) {p2:.2f} P(adopt|+3) {p3:.2f} P(adopt|-1.5) {pneg:.3f}; 8 null flags -> expected false adopts {8*p_null:.2f} (x~0.6 if Cox must agree: {8*p_null*0.6:.2f})')
    # t-based with external dof 7 pooled
    t=stats.t.ppf(0.95,7); print('   with t(7) crit',round(t,2),'power +3', round(1-stats.nct.cdf(t,7,3/se),2))
# winner's curse: expected estimated effect given adopted, true 0 and true +3
for sd in (2.47,3.93):
    se=sd/2; c=1.5
    for d in (0,3):
        a=(c-d)/se; e=d+se*N.pdf(a)/(1-N.cdf(a)); print(f'sd {sd} true {d}: E[beta | adopted] {e:.2f}')
# sequential rule with pool nref and sd
def seq(delta, sd, nref, reps=20000, looks=(1,3,6), fut=(-1.0,0.0), acc_z=1.28):
    random.seed(1); runs=0; adopt=0
    for _ in range(reps):
        ref=random.gauss(0, sd/math.sqrt(nref)); xs=[]
        for i,n in enumerate(looks):
            while len(xs)<n: xs.append(random.gauss(delta,sd))
            d=sum(xs)/len(xs)-ref
            if i<len(looks)-1:
                if d<fut[i]: break
            else:
                if d/(sd*math.sqrt(1/n+1/nref))>acc_z: adopt+=1
        runs+=len(xs)
    return runs/reps, adopt/reps
for sd,nref in ((2.47,10),(2.47,4),(3.93,4),(3.93,10)):
    print('seq sd',sd,'nref',nref, {d:tuple(round(x,2) for x in seq(d,sd,nref)) for d in (0,2,3)})
# final confirmation power: S1 n=6 vs pool
for sd in (2.47,3.93):
    for nref in (4,10):
        for d in (1,2,3):
            se=sd*math.sqrt(1/6+1/nref); print(f'S1 6 runs vs pool {nref} sd {sd} effect +{d}: power one-sided .05 {1-N.cdf(1.645-d/se):.2f}')
# kill rule all-off outside 46-58 single run
for sd in (2.47,3.93):
    p=N.cdf((46-52.91)/sd)+1-N.cdf((58-52.91)/sd); print('P(false stop all-off outside 46-58) sd',sd,round(p,3))
# dummy > 2.5
for sd in (2.47,3.93): print('P(|dummy|>2.5) sd',sd, round(2*(1-N.cdf(2.5/(sd/2))),3))
# best-of-2
for s in (1.0,1.5,2.0,2.5,3.93): print('E[max2]-mu', s, round(s/math.sqrt(math.pi),2))
