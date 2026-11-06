"""Toy sensitivity study: FP8/int4 on attention projection weights vs on the KV cache.

Model: a small pretrained Llama-style LM (separate q/k/v/o projections, RoPE, GQA).
Data: real English text (markdown docs from the repo), teacher-forced.
Metric: KL(ref || perturbed) per token, mean and p95; top-1 disagreement.
Each perturbation is applied to ONE tensor of ONE layer (or to a set of layers), the rest stays FP32.

KV quantization is applied to the k_proj / v_proj outputs (pre-RoPE K, like KVQuant), per (token, head).
Weight quantization: FP8 E4M3 per output channel (like vLLM FP8 W8), int4 symmetric group-128 (like AWQ/GPTQ).
"""
import json, math, sys, time, glob, re
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M"
SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
NSEQ = int(sys.argv[3]) if len(sys.argv) > 3 else 3
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
layers = model.model.layers
L = len(layers)
cfg = model.config
H, KVH, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads

# ---------- data: real text from the repo docs ----------
docs = sorted(glob.glob(str(__import__("pathlib").Path(__file__).resolve().parents[3] / "docs" / "**" / "*.md"), recursive=True))
text = ""
for p in docs:
    t = open(p, errors="ignore").read()
    t = re.sub(r"```.*?```", "", t, flags=re.S)        # drop code blocks
    t = re.sub(r"\|[^\n]*\|", "", t)                    # drop tables
    text += t + "\n"
ids = tok(text, return_tensors="pt").input_ids[0]
ids = ids[: SEQ * NSEQ].view(NSEQ, SEQ).to(dev)
print(f"model={MODEL} L={L} H={H} KVH={KVH} D={D} seqs={NSEQ}x{SEQ} dev={dev}", flush=True)

# ---------- quantizers ----------
F8MAX = 448.0
def fq_fp8(x, scale):                      # fake FP8 E4M3 with a given scale tensor (broadcastable)
    y = (x / scale).clamp(-F8MAX, F8MAX).to(torch.float8_e4m3fn).to(x.dtype)
    return y * scale

def w_fp8_per_channel(W):                  # [out, in], one scale per output channel
    s = W.abs().amax(dim=1, keepdim=True) / F8MAX
    return fq_fp8(W, s.clamp_min(1e-12))

def w_int4_group(W, g=128):                # symmetric int4, groups of g along the input dim
    o, i = W.shape
    while i % g: g //= 2                   # hidden 576 is not a multiple of 128 -> group 64
    Wg = W.view(o, i // g, g)
    s = (Wg.abs().amax(dim=2, keepdim=True) / 7).clamp_min(1e-12)
    return (torch.round(Wg / s).clamp(-8, 7) * s).view(o, i)

def act_fp8_per_tensor(x, scale):          # vLLM-style fp8_e4m3 KV with one scale per layer
    return fq_fp8(x, scale)

def act_fp8_per_token_head(x, nh):         # FP8 with one scale per (token, head)
    T, _ = x.shape
    xh = x.view(T, nh, -1)
    s = (xh.abs().amax(dim=2, keepdim=True) / F8MAX).clamp_min(1e-12)
    return fq_fp8(xh, s).view(T, -1)

def act_int4_per_token_head(x, nh, hadamard=False):   # asymmetric int4, scale+zero per (token, head)
    T, _ = x.shape
    xh = x.view(T, nh, -1)
    if hadamard:
        xh = xh @ HAD
    lo, hi = xh.amin(dim=2, keepdim=True), xh.amax(dim=2, keepdim=True)
    s = ((hi - lo) / 15).clamp_min(1e-12)
    q = torch.round((xh - lo) / s).clamp(0, 15)
    y = q * s + lo
    if hadamard:
        y = y @ HAD.T
    return y.view(T, -1)

def hadamard(n):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)
HAD = hadamard(D).to(dev)

def coherent_noise(W, rel):                # one fixed dW with ||dW|| = rel * ||W||
    dW = torch.randn_like(W); return W + dW * (rel * W.norm() / dW.norm())

def independent_noise(x, rel):            # fresh noise per token, scaled per token
    e = torch.randn_like(x); return x + e * (rel * x.norm(dim=1, keepdim=True) / e.norm(dim=1, keepdim=True))

# ---------- running the model with perturbations ----------
@torch.no_grad()
def run(weight_mods=(), act_mods=()):
    """weight_mods: list of (layer, name, fn) replacing the weight temporarily.
       act_mods: list of (layer, name, fn) applied to the module output (fn(x2d))."""
    saved, hooks = [], []
    for l, name, fn in weight_mods:
        mod = getattr(layers[l].self_attn, name)
        saved.append((mod, mod.weight.data.clone()))
        mod.weight.data = fn(mod.weight.data)
    for l, name, fn in act_mods:
        mod = getattr(layers[l].self_attn, name)
        hooks.append(mod.register_forward_hook(lambda m, i, o, fn=fn: fn(o.reshape(-1, o.shape[-1])).view_as(o)))
    try:
        out = model(ids, output_hidden_states=True)
    finally:
        for mod, w in saved: mod.weight.data = w
        for h in hooks: h.remove()
    return out.logits.float(), [h.float() for h in out.hidden_states]

ref_logits, ref_hs = run()
ref_lp = F.log_softmax(ref_logits, -1)
ref_top = ref_logits.argmax(-1)

def metrics(logits, hs=None, l=None):
    lp = F.log_softmax(logits, -1)
    kl = (ref_lp.exp() * (ref_lp - lp)).sum(-1).flatten()
    m = {"kl_mean": kl.mean().item(), "kl_p95": kl.quantile(0.95).item(),
         "top1_dis": (logits.argmax(-1) != ref_top).float().mean().item()}
    if hs is not None and l is not None:
        # error growth from the output of block l to the final hidden state (relative to layer norms)
        d_in = (hs[l + 1] - ref_hs[l + 1]).norm() / ref_hs[l + 1].norm()
        d_out = (hs[-1] - ref_hs[-1]).norm() / ref_hs[-1].norm()
        m["rel_err_after_block"] = d_in.item(); m["rel_err_final"] = d_out.item()
        m["growth"] = (d_out / d_in).item() if d_in > 0 else float("nan")
    return m

# calibrated per-layer FP8 KV scales (best case for per-tensor FP8)
kv_scale = {}
@torch.no_grad()
def calib():
    hs = {}
    def mk(l, n):
        def f(m, i, o): hs[(l, n)] = o.abs().amax().item() / F8MAX
        return f
    hk = [getattr(layers[l].self_attn, n).register_forward_hook(mk(l, n)) for l in range(L) for n in ("k_proj", "v_proj")]
    model(ids)
    for h in hk: h.remove()
    return hs
kv_scale = calib()

results = []
def rec(tag, layer, m):
    m.update(tag=tag, layer=layer); results.append(m)
    print(f"{tag:28s} L{str(layer):>4s} KL={m['kl_mean']:.5f} p95={m['kl_p95']:.4f} top1dis={m['top1_dis']:.4f}"
          + (f" growth={m['growth']:.2f}" if 'growth' in m else ""), flush=True)

t0 = time.time()
# ---------- A. per-layer, one tensor at a time ----------
for l in range(L):
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        for fmt, fn in (("w_fp8", w_fp8_per_channel), ("w_int4", w_int4_group)):
            lg, hs = run(weight_mods=[(l, name, fn)]); rec(f"{name}:{fmt}", l, metrics(lg, hs, l))
    for name in ("k_proj", "v_proj"):
        kv = "K" if name == "k_proj" else "V"
        s = kv_scale[(l, name)]
        for fmt, fn in ((f"kv{kv}_fp8", lambda x, s=s: act_fp8_per_tensor(x, s)),
                        (f"kv{kv}_fp8th", lambda x: act_fp8_per_token_head(x, KVH)),
                        (f"kv{kv}_int4", lambda x: act_int4_per_token_head(x, KVH)),
                        (f"kv{kv}_int4rht", lambda x: act_int4_per_token_head(x, KVH, True))):
            lg, hs = run(act_mods=[(l, name, fn)]); rec(fmt, l, metrics(lg, hs, l))
print(f"per-layer done in {time.time()-t0:.0f}s", flush=True)

# ---------- B. matched relative error: coherent (weight) vs independent (per-token) noise ----------
for l in (L // 4, L // 2, 3 * L // 4):
    for rel in (0.02, 0.05, 0.1):
        for name in ("v_proj", "o_proj", "k_proj"):
            lg, hs = run(weight_mods=[(l, name, lambda W, r=rel: coherent_noise(W, r))]); rec(f"{name}:coh{rel}", l, metrics(lg, hs, l))
            lg, hs = run(act_mods=[(l, name, lambda x, r=rel: independent_noise(x, r))]); rec(f"{name}:ind{rel}", l, metrics(lg, hs, l))

# ---------- C. whole-model configs (what a serving choice looks like) ----------
allL = list(range(L)); mid = list(range(3, L - 3)); edge = [l for l in allL if l not in mid]
def kvmods(ls, fmt):
    mods = []
    for l in ls:
        for name in ("k_proj", "v_proj"):
            s = kv_scale[(l, name)]
            if fmt == "fp8": mods.append((l, name, lambda x, s=s: act_fp8_per_tensor(x, s)))
            elif fmt == "fp8th": mods.append((l, name, lambda x: act_fp8_per_token_head(x, KVH)))
            elif fmt == "int4": mods.append((l, name, lambda x: act_int4_per_token_head(x, KVH)))
            elif fmt == "int4rht": mods.append((l, name, lambda x: act_int4_per_token_head(x, KVH, True)))
    return mods
configs = {
    "ALL W_qkv fp8": dict(weight_mods=[(l, n, w_fp8_per_channel) for l in allL for n in ("q_proj", "k_proj", "v_proj")]),
    "ALL W_o fp8": dict(weight_mods=[(l, "o_proj", w_fp8_per_channel) for l in allL]),
    "ALL W_qkvo fp8": dict(weight_mods=[(l, n, w_fp8_per_channel) for l in allL for n in ("q_proj", "k_proj", "v_proj", "o_proj")]),
    "ALL W_qkvo int4": dict(weight_mods=[(l, n, w_int4_group) for l in allL for n in ("q_proj", "k_proj", "v_proj", "o_proj")]),
    "ALL KV fp8": dict(act_mods=kvmods(allL, "fp8")),
    "ALL KV fp8 per-token-head": dict(act_mods=kvmods(allL, "fp8th")),
    "ALL KV int4": dict(act_mods=kvmods(allL, "int4")),
    "ALL KV int4+rht": dict(act_mods=kvmods(allL, "int4rht")),
    "MID KV int4, edge fp8": dict(act_mods=kvmods(mid, "int4") + kvmods(edge, "fp8")),
    "MID KV int4+rht, edge fp8": dict(act_mods=kvmods(mid, "int4rht") + kvmods(edge, "fp8")),
    "EDGE KV int4, mid fp8": dict(act_mods=kvmods(edge, "int4") + kvmods(mid, "fp8")),
    "ALL KV int4 + W_qkvo fp8": dict(act_mods=kvmods(allL, "int4"), weight_mods=[(l, n, w_fp8_per_channel) for l in allL for n in ("q_proj", "k_proj", "v_proj", "o_proj")]),
}
for tag, kw in configs.items():
    lg, hs = run(**kw); rec(tag, "all", metrics(lg))

json.dump(results, open(f"results_{MODEL.split('/')[-1]}.json", "w"), indent=1)
print(f"total {time.time()-t0:.0f}s", flush=True)
