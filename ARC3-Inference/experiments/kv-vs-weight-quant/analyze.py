"""Compact tables from checks_<model>.json files: python analyze.py checks_A.json [checks_B.json ...]"""
import json, statistics as st, sys
from collections import defaultdict
for path in sys.argv[1:]:
    R = json.load(open(path)); L = R["L"]
    print(f"\n######## {R['model']} L={L} KVH={R['KVH']} D={R['D']} seq={R['seq']} nseq={len(set(x['seq'] for x in R['layouts']))}")
    g = defaultdict(list)
    for x in R["layouts"]: g[x["tag"]].append(x)
    print("\nlayout | mean KL | min-max over chunks | top1 flips | KL by position quartile")
    for t, v in g.items():
        k = [x["kl"] for x in v]
        b = [st.mean(x["bins"][i] for x in v) for i in range(4)]
        print(f"{t:42s} {st.mean(k):9.5f}  {min(k):8.5f}-{max(k):8.5f}  {st.mean(x['top1'] for x in v)*100:5.1f}%  " + " ".join(f"{y:.4f}" for y in b))
    if not R["per_layer"]: continue
    pl = defaultdict(list)
    for x in R["per_layer"]: pl[(x["tensor"], x["fmt"])].append(x)
    print("\ntensor fmt | mean rel err | mean KL e4 | KL/err^2 (median) | top layers by KL")
    for (n, f), v in sorted(pl.items(), key=lambda kv: -st.mean(x["kl"] for x in kv[1])):
        top = sorted(v, key=lambda x: -x["kl"])[:3]
        print(f"{n:7s} {f:9s} {st.mean(x['err'] for x in v):7.3f} {st.mean(x['kl'] for x in v)*1e4:10.2f} {st.median(x['kl']/max(x['err'],1e-9)**2 for x in v):8.2f}   " + " ".join(f"L{x['layer']}={x['kl']*1e4:.0f}" for x in top))
    # additivity on chunk 0
    s0 = {x["tag"]: x["kl"] for x in R["layouts"] if x["seq"] == 0}
    def S(ts, fmts): return sum(x["kl"] for x in R["per_layer"] if x["tensor"] in ts and x["fmt"] in fmts)
    print("\nadditivity, chunk 0: whole-model KL vs sum of single-layer KLs")
    for t, ts, f in (("W_qkvo fp8", ("q_proj","k_proj","v_proj","o_proj"), ("w_fp8",)), ("W_qkvo int4", ("q_proj","k_proj","v_proj","o_proj"), ("w_int4",)),
                     ("W_o int4", ("o_proj",), ("w_int4",)), ("W_v int4", ("v_proj",), ("w_int4",)), ("KV fp8 pre-RoPE", ("k_proj","v_proj"), ("kv_fp8",)), ("K int4+RHT pre-RoPE", ("k_proj",), ("kv_int4r",)), ("V int4+RHT", ("v_proj",), ("kv_int4r",))):
        print(f"  {t:22s} whole {s0[t]:.5f} sum {S(ts,f):.5f} ratio {s0[t]/max(S(ts,f),1e-12):.2f}")
    # matched output error
    m = defaultdict(list)
    for x in R["matched"]: m[(x["tensor"], x["r"], x["kind"])].append(x)
    print("\nmatched output error: tensor r | KL coh | KL ind | ratio | attn err coh | attn err ind")
    for n in ("q_proj","k_proj","v_proj","o_proj"):
        for r in (0.02,0.05,0.1):
            c, i = m[(n,r,"coherent")], m[(n,r,"independent")]
            kc, ki = st.mean(x["kl"] for x in c), st.mean(x["kl"] for x in i)
            print(f"  {n} {r:<5} {kc:.5f} {ki:.5f} {kc/max(ki,1e-9):6.1f}  {st.mean(x['attn_err'] for x in c):.4f} {st.mean(x['attn_err'] for x in i):.4f}")
    a = R["attention"]
    print("\nattention: mean similarity %.2f, n_eff %.1f, attenuation %.2f" % (st.mean(x["sim"] for x in a), st.mean(x["n_eff"] for x in a), st.mean(x["atten"] for x in a)))
    o = {(x["layer"],x["tensor"]):x for x in R["outliers"]}
    for n in ("k_proj","v_proj"):
        print(f"{n} max/median by layer:", " ".join(f"{o[(l,n)]['max_over_median']:.0f}" for l in range(L)))
        print(f"{n} top-channel energy:", " ".join(f"{o[(l,n)]['top_share']:.2f}" for l in range(L)))
        print(f"{n} modal-channel frac:", " ".join(f"{o[(l,n)]['modal_frac']:.2f}" for l in range(L)))
