"""Compare the first run with the replication, and print the first-layer test."""
import json, math, os, statistics as st
from collections import defaultdict
tags = ['W_o fp8','W_v fp8','W_qkvo fp8','V fp8','K fp8 post-RoPE','KV fp8 post-RoPE','W_o int4','W_v int4','W_qk int4','W_qkvo int4','V int4+RHT','K int4+RHT pre-RoPE','K int4+RHT post-RoPE','KV int4 post-RoPE','KV int4+RHT post-RoPE','K fp8 + V int4+RHT post-RoPE','L0 fp8, rest int4+RHT post-RoPE','first3 fp8, rest int4+RHT post-RoPE','last3 fp8, rest int4+RHT post-RoPE']
ms = ['SmolLM2-135M','Qwen2.5-0.5B','Qwen3-0.6B','Qwen2.5-1.5B']
D = {(m, r): json.load(open(f'checks_{m}{s}.json')) for m in ms for r, s in ((0,''),(1,'-rep1'))}
def lay(R):
    g = defaultdict(list)
    for x in R['layouts']: g[x['tag']].append(x)
    return g
print(f"{'layout (KL run1/run2)':38s}" + "".join(f"{m[:13]:>18s}" for m in ms))
rows = []
for t in tags:
    row = f"{t:38s}"
    for m in ms:
        a = st.mean(x['kl'] for x in lay(D[m,0])[t]); b = st.mean(x['kl'] for x in lay(D[m,1])[t])
        row += f"{a:9.4f}/{b:8.4f}"; rows.append((m, math.log(a), math.log(b)))
    print(row)
for m in ms:
    x = [r for r in rows if r[0]==m]; A=[r[1] for r in x]; B=[r[2] for r in x]; n=len(x); ma,mb=sum(A)/n,sum(B)/n
    c = sum((p-ma)*(q-mb) for p,q in zip(A,B))/math.sqrt(sum((p-ma)**2 for p in A)*sum((q-mb)**2 for q in B))
    print(f"{m}: log-KL correlation {c:.3f}, largest run1/run2 ratio {math.exp(max(abs(p-q) for p,q in zip(A,B))):.2f}")
print("\nmatched output error r=.05: KL ratio coh/ind (attention-error^2 ratio), run1 | run2")
for m in ms:
    for n in ('q_proj','k_proj','v_proj','o_proj'):
        out = []
        for rep in (0,1):
            X = [x for x in D[m,rep]['matched'] if x['tensor']==n and x['r']==.05]
            c = [x for x in X if x['kind']=='coherent']; i = [x for x in X if x['kind']=='independent']
            ec, ei = st.mean(x['attn_err'] for x in c), st.mean(x['attn_err'] for x in i)
            out.append(f"{st.mean(x['kl'] for x in c)/st.mean(x['kl'] for x in i):.1f} ({(ec/ei)**2 if ei>1e-9 else float('nan'):.1f})")
        print(f"  {m:14s} {n} " + " | ".join(out))
print("\nadditivity whole/sum and L0 share of single-layer sum, run1 | run2")
for m in ms:
    for rep in (0,1):
        R = D[m,rep]; s0 = {x['tag']:x['kl'] for x in R['layouts'] if x['seq']==0}; pl = R['per_layer']
        def S(ts,f): return sum(x['kl'] for x in pl if x['tensor'] in ts and x['fmt']==f)
        def share(t,f):
            v=[x for x in pl if x['tensor']==t and x['fmt']==f]; return [x for x in v if x['layer']==0][0]['kl']/sum(x['kl'] for x in v)*100
        Q=('q_proj','k_proj','v_proj','o_proj')
        print(f"  {m:14s} run{rep+1} add W fp8 {s0['W_qkvo fp8']/S(Q,'w_fp8'):.2f} W int4 {s0['W_qkvo int4']/S(Q,'w_int4'):.2f} KV fp8 {s0['KV fp8 pre-RoPE']/(S(('k_proj','v_proj'),'kv_fp8')):.2f} K int4r {s0['K int4+RHT pre-RoPE']/S(('k_proj',),'kv_int4r'):.2f} | L0 share K fp8 {share('k_proj','kv_fp8'):.0f}% K int4r {share('k_proj','kv_int4r'):.0f}% V int4r {share('v_proj','kv_int4r'):.0f}%")
print("\nper-tensor mean over layers, KL e4 (rel err), run1 | run2")
for m in ms:
    for t,f in (('q_proj','w_int4'),('k_proj','w_int4'),('v_proj','w_int4'),('o_proj','w_int4'),('v_proj','kv_int4r'),('k_proj','kv_int4r')):
        o=[]
        for rep in (0,1):
            v=[x for x in D[m,rep]['per_layer'] if x['tensor']==t and x['fmt']==f and x['layer']!=0]
            o.append(f"{st.mean(x['kl'] for x in v)*1e4:.1f} ({st.mean(x['err'] for x in v):.3f})")
        print(f"  {m:14s} {t} {f:9s} excl L0: "+" | ".join(o))
print("\nfirst-layer test (mean KL, top-1 flips)")
for m in ms:
    p=f'first_layer_{m}.json'
    if not os.path.exists(p): print(m,'missing'); continue
    R=json.load(open(p)); g=defaultdict(list)
    for x in R['layouts']: g[x['tag']].append(x)
    print(" ",m," ; ".join(f"{t}: {st.mean(x['kl'] for x in v):.4f} {st.mean(x['top1'] for x in v)*100:.0f}%" for t,v in g.items()))
    d=R['diag']; top=sorted(d,key=lambda x:-x['kl_k_fp8'])[:3]
    for x in d[:1]+top:
        print(f"    L{x['layer']}: common share {x['common_share']:.4f} err_fp8 {x['err_fp8']:.3f} vs varying {x['err_fp8_vs_varying']:.3f} int4r vs varying {x['err_int4r_vs_varying']:.2f} KL {x['kl_k_fp8']:.4f}")
    print("    median layer common share %.3f, vs-varying fp8 err %.3f"%(st.median(x['common_share'] for x in d),st.median(x['err_fp8_vs_varying'] for x in d)))
