# Power and error rates at the three candidate run sds (pooled 2.47, current-stack 3.34, user-combo-10x 3.93).
import math
from statistics import NormalDist
N = NormalDist()
z = 1.645
print('sd   | fact16 +3 | fact32 +3 | S1 n6 vs pool4 +2 | n6 vs pool10 +2 | n8 vs pool8 +2 | n8 vs pool8 +1.5 | n12 vs pool10 +2 | n12 vs pool10 +3 | runs/arm for +3 (paired, 80%) | posterior mean of observed +3 (prior 6%, 16-run)')
for sd in (2.47, 3.34, 3.93):
    f16 = 1 - N.cdf(z - 3/(sd/2)); f32 = 1 - N.cdf(z - 3/(sd/math.sqrt(8)))
    def s1(n, m, d): return 1 - N.cdf(z - d/(sd*math.sqrt(1/n + 1/m)))
    n_arm = 2*((z + 0.8416)*sd/3)**2
    se = sd/2; p = 0.06
    lr = p*N.pdf(0)/(p*N.pdf(0) + (1-p)*N.pdf(3/se))
    print(f'{sd:.2f} | {f16:.2f} | {f32:.2f} | {s1(6,4,2):.2f} | {s1(6,10,2):.2f} | {s1(8,8,2):.2f} | {s1(8,8,1.5):.2f} | {s1(12,10,2):.2f} | {s1(12,10,3):.2f} | {n_arm:.1f} | {3*lr:.2f}')
# wins: runs per arm to detect +1 win/run at sd 2.1 (user-combo-10x wins 10/7/6)
print('runs/arm for +1 win at sd 2.08:', round(2*((z+0.8416)*2.08/1)**2, 1))
# expected value of the factorial under a base-rate prior (8 flags, real flags = +3, rule beta>1.5)
for sd in (2.47, 3.34, 3.93):
    pa = 1 - N.cdf((1.5-3)/(sd/2))
    print(f'sd {sd}: EV of 8 flags at prior 4% / 8% = {8*0.04*3*pa:.2f} / {8*0.08*3*pa:.2f}')
# all-off stop band 44-62 at sd 3.93
print('P(all-off outside 44-62 | no change, sd 3.93):', round(N.cdf((44-52.91)/3.93) + 1 - N.cdf((62-52.91)/3.93), 3))
print('P(mem-v2 single run < 44 | no effect, sd 3.93):', round(N.cdf((44-52.91)/3.93), 3))
