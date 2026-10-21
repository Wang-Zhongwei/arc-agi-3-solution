import math, random, itertools
from statistics import NormalDist
N=NormalDist()
def pw(delta, se, alpha=0.05): return 1-N.cdf(N.inv_cdf(1-alpha)-delta/se)
print('== power, one-sided alpha 0.05, run-mean score ==')
for sd in (2.47,3.5):
    print(f'-- run sd {sd} --')
    rows=[('1 arm vs fresh control, n=3 each',sd*math.sqrt(2/3),6),('n=4 each',sd*math.sqrt(2/4),8),('n=6 each',sd*math.sqrt(2/6),12),('n=8 each',sd*math.sqrt(2/8),16),
          ('arm n=4 vs shared ref pool 10',sd*math.sqrt(1/4+1/10),4),('arm n=6 vs shared ref pool 10',sd*math.sqrt(1/6+1/10),6),
          ('2^(8-4) factorial, 16 runs, per flag',sd/2,2),('2^(8-3) factorial, 32 runs, per flag',sd/math.sqrt(8),4)]
    for lab,se,cost in rows:
        print(f'  {lab:40} se {se:4.2f}  power +2 {pw(2,se):.2f}  +3 {pw(3,se):.2f}  +4 {pw(4,se):.2f}   runs per arm {cost}')
print('== Cox-tok (per-token level-clear hazard), se = 0.115*sqrt(1/n1+1/n2); +3 pts ~ logHR 0.145-0.176 ==')
for lab,n1,n2 in (('1 vs 2',1,2),('3 vs 3',3,3),('4 vs 4',4,4),('6 vs 6',6,6),('factorial 8 on vs 8 off',8,8)):
    se=0.115*math.sqrt(1/n1+1/n2)
    print(f'  {lab:24} se {se:.3f}  power(+3pts) {pw(0.145,se):.2f}-{pw(0.176,se):.2f}  power(+2pts) {pw(0.097,se):.2f}-{pw(0.117,se):.2f}')
# sequential one-at-a-time vs ref pool of 10 known mean(se small)
def seq(delta, sd, looks=(1,3,6), fut=(-1.0,0.0), acc_z=1.28, nref=10, reps=20000):
    random.seed(1); runs=0; adopt=0
    for _ in range(reps):
        ref=random.gauss(0, sd/math.sqrt(nref))
        xs=[]; stopped=False
        for i,n in enumerate(looks):
            while len(xs)<n: xs.append(random.gauss(delta,sd))
            d=sum(xs)/len(xs)-ref
            if i<len(looks)-1:
                if d<fut[i]: stopped=True; break
            else:
                se=sd*math.sqrt(1/n+1/nref)
                if d/se>acc_z: adopt+=1
        runs+=len(xs)
    return runs/reps, adopt/reps
print('== sequential single-arm design: looks at 1,3,6 runs; futility if diff<-1 (n=1), <0 (n=3); adopt if z>1.28 at n=6; ref pool 10 ==')
for sd in (2.47,3.5):
    for d in (-3,-1,0,2,3,4):
        r,a=seq(d,sd); print(f'  sd {sd} true effect {d:+d}: expected runs {r:.2f}, P(adopt) {a:.2f}')
print('== E[max of two final picks] bonus: copies of the same config ==')
for s in (1.5,2.5,3.93):
    print(f'  private-run sd {s} LB pts -> +{s/math.sqrt(math.pi):.2f} LB pts (~+{s/math.sqrt(math.pi)/0.56:.1f} offline-equivalent)')
def emax(d,theta):
    return d*N.cdf(d/theta)+ theta*N.pdf(d/theta) # E[max(X,Y)]-mu_Y with mu_X-mu_Y=d
print('  A twice vs A + B (B worse by Delta; independent transfer error tau=1.5 LB, run sd 2.5 LB):')
s=2.5; tau=1.5
same= s/math.sqrt(math.pi)
for D in (0,0.5,1.0,2.0):
    th=math.sqrt(2*s*s+2*tau*tau)
    ab=emax(D,th)-D  # relative to mu_A
    print(f'    Delta {D} LB: A+A +{same:.2f}, A+B {ab:+.2f}')
