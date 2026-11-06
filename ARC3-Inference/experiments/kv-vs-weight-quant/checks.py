"""Validity checks on sens.py. Run one model per process; writes checks_<model>.json.

A  Per-layer, per-format: real relative error at the tensor's output, KL, and KL/err^2 (sensitivity).
   Also outlier statistics of K and V (max/median, persistent channel, top-channel energy share).
B  Matched OUTPUT error (sens.py matched the weight norm only): coherent dW vs per-token noise, both
   scaled to the same ||delta output|| / ||output||. Also measures the error at the attention output.
C  Attention statistics: how much per-token noise can average out (similarity of attended values, n_eff).
D  Whole-model layouts over several text chunks, with K quantized before and after RoPE, a repeat-run
   noise floor, a BF16 reference and KL binned by position.
"""
import gc, glob, json, math, os, pathlib, re, sys, time
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, SEQ, NSEQ = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
LAYOUTS_ONLY = len(sys.argv) > 4 and sys.argv[4] == "layouts"
REP = int(os.environ.get("REP", 0))   # replication: REP>0 uses different text chunks and noise seeds
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(1000 * REP)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").to(dev).eval()
layers = model.model.layers
L, cfg = len(layers), model.config
H, KVH = cfg.num_attention_heads, cfg.num_key_value_heads
D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H

root = pathlib.Path(__file__).resolve().parents[3]
text = ""
for p in sorted(glob.glob(str(root / "docs" / "**" / "*.md"), recursive=True)):
    t = open(p, errors="ignore").read()
    t = re.sub(r"```.*?```", "", t, flags=re.S)
    text += re.sub(r"\|[^\n]*\|", "", t) + "\n"
all_ids = tok(text, return_tensors="pt").input_ids[0]
stride = (len(all_ids) - SEQ) // max(NSEQ - 1, 1)
offset = (stride // 2) * (REP > 0)
chunks = [all_ids[offset + i * stride: offset + i * stride + SEQ].unsqueeze(0).to(dev) for i in range(NSEQ - (REP > 0))]
assert all(c.shape[1] == SEQ for c in chunks)
print(f"model={MODEL} L={L} H={H} KVH={KVH} D={D} seqs={NSEQ}x{SEQ} text={len(all_ids)} dev={dev}", flush=True)

# ---------- quantizers (same as sens.py) ----------
F8MAX = 448.0
def fq_fp8(x, s): return (x / s).clamp(-F8MAX, F8MAX).to(torch.float8_e4m3fn).to(x.dtype) * s
def w_fp8(W): return fq_fp8(W, (W.abs().amax(1, keepdim=True) / F8MAX).clamp_min(1e-12))
def w_int4(W, g=128):
    o, i = W.shape
    while i % g: g //= 2
    Wg = W.view(o, i // g, g)
    s = (Wg.abs().amax(2, keepdim=True) / 7).clamp_min(1e-12)
    return (torch.round(Wg / s).clamp(-8, 7) * s).view(o, i)
def hadamard(n):
    h = torch.ones(1, 1)
    while h.shape[0] < n: h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)
HAD = hadamard(D).to(dev); assert HAD.shape[0] == D
def a_fp8th(x):
    xh = x.view(x.shape[0], KVH, -1)
    return fq_fp8(xh, (xh.abs().amax(2, keepdim=True) / F8MAX).clamp_min(1e-12)).view(x.shape[0], -1)
def a_int4(x, rot=False):
    xh = x.view(x.shape[0], KVH, -1)
    if rot: xh = xh @ HAD
    lo, hi = xh.amin(2, keepdim=True), xh.amax(2, keepdim=True)
    s = ((hi - lo) / 15).clamp_min(1e-12)
    y = torch.round((xh - lo) / s).clamp(0, 15) * s + lo
    if rot: y = y @ HAD.T
    return y.view(x.shape[0], -1)
def a_int4r(x): return a_int4(x, True)
def a_bf16(x): return x.to(torch.bfloat16).to(x.dtype)

# ---------- patched RoPE so K can be quantized after rotation, as engines cache it ----------
mod = sys.modules[type(layers[0].self_attn).__module__]
_orig_rope = mod.apply_rotary_pos_emb
CUR, ROPE_FN = [None], {}
def _rope(q, k, cos, sin, unsqueeze_dim=1):
    q2, k2 = _orig_rope(q, k, cos, sin, unsqueeze_dim)
    fn = ROPE_FN.get(CUR[0])
    if fn is not None:
        B, n, T, d = k2.shape
        k2 = fn(k2.transpose(1, 2).reshape(T, -1)).view(B, T, n, d).transpose(1, 2)
    return q2, k2
mod.apply_rotary_pos_emb = _rope
for i, l in enumerate(layers):
    l.self_attn.register_forward_pre_hook(lambda m, a, kw, i=i: CUR.__setitem__(0, i), with_kwargs=True)

@torch.no_grad()
def run(ids, wmods=(), amods=(), rope=None, capture=None, **kw):
    saved, hooks = [], []
    for l, name, fn in wmods:
        m = getattr(layers[l].self_attn, name)
        saved.append((m, m.weight.data)); m.weight.data = fn(m.weight.data)
    for l, name, fn in amods:
        m = getattr(layers[l].self_attn, name)
        hooks.append(m.register_forward_hook(lambda m, i, o, fn=fn: fn(o.reshape(-1, o.shape[-1])).view_as(o)))
    ROPE_FN.clear(); ROPE_FN.update(rope or {})
    if capture:
        for key, (m, fn) in capture.items():
            hooks.append(m.register_forward_hook(fn))
    try:
        out = model(ids, **kw)
    finally:
        for m, w in saved: m.weight.data = w
        for h in hooks: h.remove()
        ROPE_FN.clear()
    return out

def kl_of(logits, ref_lp):
    lp = F.log_softmax(logits.float(), -1)
    return (ref_lp.exp() * (ref_lp - lp)).sum(-1).flatten()

R = dict(model=MODEL, L=L, H=H, KVH=KVH, D=D, seq=SEQ, nseq=NSEQ, per_layer=[], matched=[], attention=[], layouts=[], outliers=[])

# per-layer K/V FP8 scales over all chunks (best case, as in sens.py), before and after RoPE
kv_s, post_s = {}, {}
@torch.no_grad()
def calibrate():
    for ids in chunks:
        cap = {}
        for l in range(L):
            for n in ("k_proj", "v_proj"):
                def f(m, i, o, key=(l, n)): kv_s[key] = max(kv_s.get(key, 0), o.abs().amax().item() / F8MAX)
                cap[(l, n)] = (getattr(layers[l].self_attn, n), f)
        def rec(l):
            def g(x): post_s[l] = max(post_s.get(l, 0), x.abs().amax().item() / F8MAX); return x
            return g
        run(ids, capture=cap, rope={l: rec(l) for l in range(L)})
calibrate()
def a_fp8(l, n): return lambda x, s=kv_s[(l, n)]: fq_fp8(x, s)

t0 = time.time()
if not LAYOUTS_ONLY:
    ids = chunks[0]
    # capture projection inputs and outputs on the reference run
    Xin, Xout = {}, {}
    cap = {}
    for l in range(L):
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            def f(m, i, o, key=(l, n)): Xin[key] = i[0].squeeze(0).detach(); Xout[key] = o.squeeze(0).detach()
            cap[(l, n)] = (getattr(layers[l].self_attn, n), f)
    ref = run(ids, capture=cap).logits.float()
    ref_lp = F.log_softmax(ref, -1); del ref

    # ---- C. attention statistics ----
    attn = {}
    def fa(m, i, o, key=None): attn[key] = o[1].detach()
    capa = {l: (layers[l].self_attn, (lambda l: (lambda m, i, o: attn.__setitem__(l, o[1].detach().cpu())))(l)) for l in range(L)}
    run(ids, capture=capa, output_attentions=True)
    for l in range(L):
        a = attn.pop(l).to(dev).squeeze(0)                       # heads, T, T
        v = Xout[(l, "v_proj")].view(SEQ, KVH, -1).repeat_interleave(H // KVH, dim=1).transpose(0, 1)  # heads, T, d
        o = a @ v                                               # heads, T, d: attended value
        vn = v.norm(dim=-1)                                     # heads, T
        denom = (a * vn.unsqueeze(1)).sum(-1)                   # sum_i alpha_i ||v_i||
        sim = (o.norm(dim=-1) / denom.clamp_min(1e-12)).mean().item()
        n_eff = (1 / (a ** 2).sum(-1).clamp_min(1e-12)).mean().item()
        # relative error after attention if each v_i carries independent error of relative size r:
        # r * sqrt(sum alpha^2 ||v||^2 / ||o||^2); report the attenuation factor
        num = ((a ** 2) * (vn ** 2).unsqueeze(1)).sum(-1).sum().sqrt()
        atten = (num / o.norm()).item()
        R["attention"].append(dict(layer=l, sim=sim, n_eff=n_eff, atten=atten))
        del a, v, o
    print("attention stats done", time.time() - t0, flush=True)

    # ---- outlier statistics ----
    for l in range(L):
        for n in ("k_proj", "v_proj"):
            xh = Xout[(l, n)].view(SEQ, KVH, -1).abs()
            top = xh.argmax(-1)
            modal = top.flatten().cpu().mode().values.to(dev)
            R["outliers"].append(dict(layer=l, tensor=n,
                max_over_median=(xh.amax(-1) / xh.median(-1).values.clamp_min(1e-12)).mean().item(),
                top_share=((xh.amax(-1) ** 2) / (xh ** 2).sum(-1).clamp_min(1e-12)).mean().item(),
                modal_frac=(top == modal).float().mean().item()))

    # ---- A. per-layer real quantizers ----
    def rel(a, b): return ((a - b).norm() / b.norm()).item()
    for l in range(L):
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod_ = getattr(layers[l].self_attn, n)
            for fmt, fn in (("w_fp8", w_fp8), ("w_int4", w_int4)):
                err = rel(F.linear(Xin[(l, n)], fn(mod_.weight.data), mod_.bias), Xout[(l, n)])
                out = run(ids, wmods=[(l, n, fn)]).logits
                k = kl_of(out, ref_lp)
                R["per_layer"].append(dict(layer=l, tensor=n, fmt=fmt, err=err, kl=k.mean().item()))
        for n in ("k_proj", "v_proj"):
            for fmt, fn in (("kv_fp8", a_fp8(l, n)), ("kv_fp8th", a_fp8th), ("kv_int4", a_int4), ("kv_int4r", a_int4r)):
                x = Xout[(l, n)].view(SEQ, -1)
                err = rel(fn(x), x)
                out = run(ids, amods=[(l, n, fn)]).logits
                R["per_layer"].append(dict(layer=l, tensor=n, fmt=fmt, err=err, kl=kl_of(out, ref_lp).mean().item()))
        if l % 5 == 0: print(f"per-layer {l}/{L} {time.time()-t0:.0f}s", flush=True)

    # ---- B. matched OUTPUT error ----
    def coherent(l, n, r, seed):
        m = getattr(layers[l].self_attn, n)
        g = torch.Generator().manual_seed(seed + 1000 * REP)
        dW = torch.randn(m.weight.shape, generator=g).to(dev)
        s = r * Xout[(l, n)].norm() / (Xin[(l, n)] @ dW.T).norm()
        return lambda W: W + dW * s
    def independent(r, seed):
        g = torch.Generator().manual_seed(seed + 1000 * REP)
        def fn(x):
            e = torch.randn(x.shape, generator=g).to(dev)
            return x + e * (r * x.norm(dim=1, keepdim=True) / e.norm(dim=1, keepdim=True))
        return fn
    for l in (L // 4, L // 2, 3 * L // 4):
        o_ref = Xin[(l, "o_proj")]
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            for r, seeds in ((0.02, [0]), (0.05, [0, 1, 2]), (0.1, [0])):
                for seed in seeds:
                    for kind in ("coherent", "independent"):
                        got = {}
                        capt = {"o": (layers[l].self_attn.o_proj, lambda m, i, o: got.__setitem__("a", i[0].squeeze(0).detach()))}
                        if kind == "coherent":
                            out = run(ids, wmods=[(l, n, coherent(l, n, r, seed))], capture=capt).logits
                        else:
                            out = run(ids, amods=[(l, n, independent(r, seed))], capture=capt).logits
                        R["matched"].append(dict(layer=l, tensor=n, r=r, seed=seed, kind=kind,
                            attn_err=rel(got["a"], o_ref), kl=kl_of(out, ref_lp).mean().item()))
    print("matched done", time.time() - t0, flush=True)
    del Xin, Xout, ref_lp; gc.collect(); torch.mps.empty_cache() if dev == "mps" else None

# ---------- D. whole-model layouts ----------
allL = list(range(L))
first3, last3 = allL[:3], allL[-3:]
def kv(ls, fmt, tensors=("k_proj", "v_proj")):
    m = []
    for l in ls:
        for n in tensors:
            fn = {"fp8": a_fp8(l, n), "fp8th": a_fp8th, "int4": a_int4, "int4r": a_int4r, "bf16": a_bf16}[fmt]
            m.append((l, n, fn))
    return m
def W(names, fn): return [(l, n, fn) for l in allL for n in names]
def post_k(ls, fmt):
    return {l: {"fp8": (lambda x, s=post_s[l]: fq_fp8(x, s)), "fp8th": a_fp8th, "int4": a_int4, "int4r": a_int4r}[fmt] for l in ls}
rest = lambda ex: [l for l in allL if l not in ex]
LAYOUTS = {
    "repeat run (noise floor)": {},
    "KV bf16": dict(amods=kv(allL, "bf16")),
    "W_o fp8": dict(wmods=W(["o_proj"], w_fp8)),
    "W_v fp8": dict(wmods=W(["v_proj"], w_fp8)),
    "W_qk fp8": dict(wmods=W(["q_proj", "k_proj"], w_fp8)),
    "W_qkv fp8": dict(wmods=W(["q_proj", "k_proj", "v_proj"], w_fp8)),
    "W_qkvo fp8": dict(wmods=W(["q_proj", "k_proj", "v_proj", "o_proj"], w_fp8)),
    "W_o int4": dict(wmods=W(["o_proj"], w_int4)),
    "W_v int4": dict(wmods=W(["v_proj"], w_int4)),
    "W_qk int4": dict(wmods=W(["q_proj", "k_proj"], w_int4)),
    "W_qkvo int4": dict(wmods=W(["q_proj", "k_proj", "v_proj", "o_proj"], w_int4)),
    "K fp8 pre-RoPE": dict(amods=kv(allL, "fp8", ["k_proj"])),
    "K fp8 post-RoPE": dict(rope=post_k(allL, "fp8")),
    "V fp8": dict(amods=kv(allL, "fp8", ["v_proj"])),
    "KV fp8 pre-RoPE": dict(amods=kv(allL, "fp8")),
    "KV fp8 post-RoPE": dict(amods=kv(allL, "fp8", ["v_proj"]), rope=post_k(allL, "fp8")),
    "KV fp8 token-head post-RoPE": dict(amods=kv(allL, "fp8th", ["v_proj"]), rope=post_k(allL, "fp8th")),
    "V int4+RHT": dict(amods=kv(allL, "int4r", ["v_proj"])),
    "K int4+RHT pre-RoPE": dict(amods=kv(allL, "int4r", ["k_proj"])),
    "K int4+RHT post-RoPE": dict(rope=post_k(allL, "int4r")),
    "K int4 post-RoPE": dict(rope=post_k(allL, "int4")),
    "KV int4+RHT post-RoPE": dict(amods=kv(allL, "int4r", ["v_proj"]), rope=post_k(allL, "int4r")),
    "KV int4 post-RoPE": dict(amods=kv(allL, "int4", ["v_proj"]), rope=post_k(allL, "int4")),
    "K fp8 + V int4+RHT post-RoPE": dict(amods=kv(allL, "int4r", ["v_proj"]), rope=post_k(allL, "fp8")),
    "L0 fp8, rest int4+RHT post-RoPE": dict(amods=kv(allL[1:], "int4r", ["v_proj"]) + kv(allL[:1], "fp8", ["v_proj"]), rope={**post_k(allL[1:], "int4r"), **post_k(allL[:1], "fp8")}),
    "first3 fp8, rest int4+RHT post-RoPE": dict(amods=kv(rest(first3), "int4r", ["v_proj"]) + kv(first3, "fp8", ["v_proj"]), rope={**post_k(rest(first3), "int4r"), **post_k(first3, "fp8")}),
    "last3 fp8, rest int4+RHT post-RoPE": dict(amods=kv(rest(last3), "int4r", ["v_proj"]) + kv(last3, "fp8", ["v_proj"]), rope={**post_k(rest(last3), "int4r"), **post_k(last3, "fp8")}),
    "first3+last3 fp8, mid int4+RHT post-RoPE": dict(amods=kv(rest(first3 + last3), "int4r", ["v_proj"]) + kv(first3 + last3, "fp8", ["v_proj"]), rope={**post_k(rest(first3 + last3), "int4r"), **post_k(first3 + last3, "fp8")}),
}
for si, ids in enumerate(chunks):
    ref_lp = F.log_softmax(run(ids).logits.float(), -1)
    ref_top = ref_lp.argmax(-1)
    for tag, kw in LAYOUTS.items():
        out = run(ids, **kw).logits
        k = kl_of(out, ref_lp)
        bins = [b.mean().item() for b in k.chunk(4)]
        R["layouts"].append(dict(tag=tag, seq=si, kl=k.mean().item(), p95=k.quantile(0.95).item(),
                                 top1=(out.argmax(-1) != ref_top).float().mean().item(), bins=bins))
    del ref_lp, ref_top; gc.collect(); torch.mps.empty_cache() if dev == "mps" else None
    print(f"layouts seq {si} {time.time()-t0:.0f}s", flush=True)

name = MODEL.split("/")[-1] + (f"-T{SEQ}" if LAYOUTS_ONLY else "") + (f"-rep{REP}" if REP else "")
json.dump(R, open(pathlib.Path(__file__).parent / f"checks_{name}.json", "w"), indent=1)
print(f"total {time.time()-t0:.0f}s", flush=True)
