"""Is the first-layer K cost one tensor, and why? Usage: first_layer.py model seq nseq; writes first_layer_<model>.json.

Layouts leave only layer 0's K at higher precision. Diagnostics per layer: relative quantization error, the share
of K's energy that is common to all tokens, and the same error measured against the part that varies by token.
"""
import glob, json, math, pathlib, re, sys
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL, SEQ, NSEQ = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dev = "mps"; tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
layers = model.model.layers; L = len(layers); cfg = model.config; KVH = cfg.num_key_value_heads
D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
root = pathlib.Path(__file__).resolve().parents[3]; text = ""
for p in sorted(glob.glob(str(root / "docs" / "**" / "*.md"), recursive=True)):
    t = re.sub(r"```.*?```", "", open(p, errors="ignore").read(), flags=re.S); text += re.sub(r"\|[^\n]*\|", "", t) + "\n"
ids = tok(text, return_tensors="pt").input_ids[0]; stride = (len(ids) - SEQ) // max(NSEQ - 1, 1)
chunks = [ids[i * stride: i * stride + SEQ].unsqueeze(0).to(dev) for i in range(NSEQ)]
F8 = 448.0
def h(n):
    x = torch.ones(1, 1)
    while x.shape[0] < n: x = torch.cat([torch.cat([x, x], 1), torch.cat([x, -x], 1)], 0)
    return x / math.sqrt(n)
HAD = h(D).to(dev)
def fq(x, s): return (x / s).clamp(-F8, F8).to(torch.float8_e4m3fn).to(x.dtype) * s
def int4r(x):
    y = x.view(x.shape[0], KVH, -1) @ HAD; lo, hi = y.amin(2, keepdim=True), y.amax(2, keepdim=True); s = ((hi - lo) / 15).clamp_min(1e-12)
    return ((torch.round((y - lo) / s).clamp(0, 15) * s + lo) @ HAD.T).view(x.shape[0], -1)
def fp8(x): return fq(x, x.abs().amax() / F8)   # per-layer scale from this chunk
mod = sys.modules[type(layers[0].self_attn).__module__]; orig = mod.apply_rotary_pos_emb; CUR, K = [0], {}
def rope(q, k, c, s, unsqueeze_dim=1):
    q2, k2 = orig(q, k, c, s, unsqueeze_dim); fn = K.get(CUR[0])
    if fn is not None:
        B, n, T, d = k2.shape; k2 = fn(k2.transpose(1, 2).reshape(T, -1)).view(B, T, n, d).transpose(1, 2)
    return q2, k2
mod.apply_rotary_pos_emb = rope
for i, l in enumerate(layers): l.self_attn.register_forward_pre_hook(lambda m, a, kw, i=i: CUR.__setitem__(0, i), with_kwargs=True)
@torch.no_grad()
def run(x, kf, vf):
    K.clear(); K.update({l: f for l, f in kf.items() if f is not None}); hs = []
    for l, f in vf.items():
        if f is not None: hs.append(layers[l].self_attn.v_proj.register_forward_hook(lambda m, i, o, f=f: f(o.reshape(-1, o.shape[-1])).view_as(o)))
    try: return model(x).logits.float()
    finally:
        K.clear(); [q.remove() for q in hs]
A = range(L)
def lay(k, v, k0=None, v0=None): return ({l: (k0 if l == 0 else k) for l in A}, {l: (v0 if l == 0 else v) for l in A})
LAY = {"KV int4+RHT": lay(int4r, int4r, int4r, int4r), "L0 K exact, rest KV int4+RHT": lay(int4r, int4r, None, int4r),
       "L0 K fp8, rest KV int4+RHT": lay(int4r, int4r, fp8, int4r), "KV fp8": lay(fp8, fp8, fp8, fp8), "L0 K exact, rest KV fp8": lay(fp8, fp8, None, fp8)}
R = dict(model=MODEL, layouts=[], diag=[])
with torch.no_grad():
    for si, x in enumerate(chunks):
        lp = F.log_softmax(run(x, {}, {}), -1)
        for t, (kf, vf) in LAY.items():
            o = run(x, kf, vf); kl = (lp.exp() * (lp - F.log_softmax(o, -1))).sum(-1).mean().item()
            R["layouts"].append(dict(tag=t, seq=si, kl=kl, top1=(o.argmax(-1) != lp.argmax(-1)).float().mean().item()))
    x = chunks[0]; lp = F.log_softmax(run(x, {}, {}), -1); cap = {}
    hk = [layers[l].self_attn.k_proj.register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(l, o.squeeze(0).detach())) for l in A]
    run(x, {}, {}); [q.remove() for q in hk]
    for l in A:
        k = cap[l]; varying = k - k.mean(0, keepdim=True); e = fp8(k) - k; e4 = int4r(k) - k
        o = run(x, {l: fp8}, {}); kl = (lp.exp() * (lp - F.log_softmax(o, -1))).sum(-1).mean().item()
        R["diag"].append(dict(layer=l, common_share=1 - (varying.norm() ** 2 / k.norm() ** 2).item(), err_fp8=(e.norm() / k.norm()).item(),
            err_fp8_vs_varying=(e.norm() / varying.norm()).item(), err_int4r_vs_varying=(e4.norm() / varying.norm()).item(), kl_k_fp8=kl))
json.dump(R, open(pathlib.Path(__file__).parent / f"first_layer_{MODEL.split('/')[-1]}.json", "w"), indent=1)
print("done", flush=True)
