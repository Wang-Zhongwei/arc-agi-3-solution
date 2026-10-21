"""Training-time patches for qwen4_exp decoder layers: grouped MoE and token chunking.

Two exact rewrites of transformers' forward, for long single-sequence training:

- grouped experts: the routed (token, expert) rows are sorted by expert and run as
  grouped GEMMs (`torch._grouped_mm`, one call per projection per tile of experts)
  instead of a Python loop over 512 experts. AutoRound's own grouped path rejects its
  packed INT4 layers, so tiles of experts are dequantized to BF16 on the fly (the
  weights serving uses). The last layer's dequantized tiles are cached: its token chunks and
  the backward's recompute of them reuse one dequantization per pass instead of one per
  chunk (~5 GB of BF16 experts held at a time; exact).
- fast_gdn: the GDN mixer's convolution writes contiguous q, k, v; q and k keep their heads (fla's
  grouped value attention instead of repeat_interleave copies); the gated RMSNorm is compiled.
- fast_lora: peft's LoRA update (FP32 adapter on a BF16 base output) as one compiled function
  instead of FP32 intermediates the size of the output.
- PLE layer: its per-token part in checkpointed token chunks, only the short convolution on the
  whole sequence (no full-sequence FP32 temporaries; exact).
- PLE n-gram lookup: the host-side table gather runs once per step; the PLE checkpoint's
  recompute reuses the forward's result (exact).
- token chunks: everything in a decoder layer except the sequence mixer (GDN or QSA)
  and the PLE short convolution is per token (hyper-connections, router, experts,
  shared expert, injections). It runs in chunks of tokens, each under its own
  checkpoint, so that part's memory no longer grows with the sequence length.
- compiled hyper-connections: the gated residuals around the mixer and the MLP (an FP32
  grouped RMSNorm over N x 10,240, a low-rank mix, sigmoids, a 4-way weighted mean and the
  injection back into the 4 streams) are dozens of small elementwise kernels in eager mode.
  `compile_hc` runs them as two pure functions under torch.compile, shared by all layers
  (weights are arguments), so each becomes a few fused kernels around the small GEMMs.
- recompute "parts": transformers checkpoints every decoder layer, and the token chunks
  inside it are checkpointed again, so their forward runs three times per step. With
  "parts" the layer-level checkpoint is off and the mixer gets its own: every part is
  recomputed once. The layer then keeps the mixer's input and output for backward
  besides the layer input (N x 2,560 each next to N x 10,240).

Scope: batch 1, no cache (training). Anything else runs transformers' forward.
"""

from __future__ import annotations

import math
import types
import weakref

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

EXPERTS_IMPLEMENTATION = "rft_grouped"
_DEFAULTS = dict(token_chunk=8192, expert_tile=128, recompute="layer", compile_hc=False, cache_experts=True)
RECOMPUTE_MODES = ("layer", "parts")


# ---------------------------------------------------------------------------
# INT4 dequantization (AutoRound / GPTQ packing)
# ---------------------------------------------------------------------------

def dequant_int4(qweight, scales, qzeros, group_size: int, bits: int = 4, dtype=torch.bfloat16):
    """(in, out) weights of a GPTQ-packed layer: scale * (q - (zero + 1)), as AutoRound's kernel computes.

    qweight [in * bits / 32, out] packs inputs along dim 0; qzeros [groups, out * bits / 32]
    packs outputs along dim 1; scales [groups, out]."""
    maxq = (1 << bits) - 1
    shifts = torch.arange(0, 32, bits, device=qweight.device, dtype=torch.int32)
    q = ((qweight.unsqueeze(1) >> shifts.view(1, -1, 1)) & maxq).reshape(-1, qweight.shape[1])
    z = ((qzeros.unsqueeze(-1) >> shifts.view(1, 1, -1)) & maxq).reshape(qzeros.shape[0], -1) + 1
    group = torch.arange(q.shape[0], device=q.device) // group_size
    return (scales[group].float() * (q - z[group]).float()).to(dtype)


def _dequant_tile(layers, dtype=torch.bfloat16):
    """[t, in, out] weights of t packed linears with the same shape, in one dequant call."""
    qweight = torch.cat([layer.qweight for layer in layers], dim=1)
    scales = torch.cat([layer.scales for layer in layers], dim=1)
    qzeros = torch.cat([layer.qzeros for layer in layers], dim=1)
    first = layers[0]
    bits = first.bits
    in_features = qweight.shape[0] * 32 // bits
    out_features = first.qweight.shape[1]
    if qweight.is_cuda:
        try:
            from auto_round_extension.triton.triton_utils_zp.dequant import dequant248

            g_idx = getattr(first, "g_idx", None)
            if g_idx is None:
                g_idx = torch.arange(in_features, device=qweight.device, dtype=torch.int32) // first.group_size
            weights = dequant248(qweight, scales, qzeros, g_idx, bits, (1 << bits) - 1, input_dtype=dtype)
        except ImportError:
            weights = dequant_int4(qweight, scales, qzeros, first.group_size, bits, dtype)
    else:
        weights = dequant_int4(qweight, scales, qzeros, first.group_size, bits, dtype)
    return weights.view(in_features, len(layers), out_features).permute(1, 0, 2).contiguous()


def expert_weights(experts, start: int, stop: int):
    """BF16 (gate_up [t, hidden, 2 * inter], down [t, inter, hidden]) for experts start..stop-1.

    Works on transformers' fused 3D parameters and on AutoRound's unfused containers
    (children "0".."E-1" holding gate_proj / up_proj / down_proj packed INT4 linears)."""
    if isinstance(getattr(experts, "gate_up_proj", None), torch.Tensor) and experts.gate_up_proj.dim() == 3:
        return experts.gate_up_proj[start:stop].transpose(1, 2), experts.down_proj[start:stop].transpose(1, 2)
    containers = [getattr(experts, str(e)) for e in range(start, stop)]
    first = containers[0].gate_proj
    if hasattr(first, "qweight"):
        gate = _dequant_tile([c.gate_proj for c in containers])
        up = _dequant_tile([c.up_proj for c in containers])
        down = _dequant_tile([c.down_proj for c in containers])
    else:  # plain nn.Linear experts ([out, in] weights)
        gate = torch.stack([c.gate_proj.weight.t() for c in containers])
        up = torch.stack([c.up_proj.weight.t() for c in containers])
        down = torch.stack([c.down_proj.weight.t() for c in containers])
    return torch.cat([gate, up], dim=-1), down


class _GatherRows(torch.autograd.Function):
    """x[index] where every row of x appears exactly `k` times in index ([n * k] rows from [n, d]).

    Autograd's backward for x[index] is a sort-based index_put with atomic accumulation; with
    `inverse` (the token-order position of each gathered row: index[inverse] = arange(n * k) // k) it
    is a gather and a fixed-order FP32 sum over the k copies instead."""

    @staticmethod
    def forward(ctx, x, index, inverse, k):
        ctx.save_for_backward(inverse)
        ctx.k = k
        return x[index]

    @staticmethod
    def backward(ctx, grad):
        (inverse,) = ctx.saved_tensors
        rows = grad[inverse].float().view(-1, ctx.k, grad.shape[-1])
        return rows.sum(dim=1).to(grad.dtype), None, None, None


def gather_rows(x, index, inverse, k: int):
    return _GatherRows.apply(x, index, inverse, k) if torch.is_grad_enabled() and x.requires_grad else x[index]


class _PermuteRows(torch.autograd.Function):
    """x[perm] for a permutation: the backward is the gather by the inverse permutation (no atomics)."""

    @staticmethod
    def forward(ctx, x, perm, inverse):
        ctx.save_for_backward(perm)
        return x[inverse]

    @staticmethod
    def backward(ctx, grad):
        (perm,) = ctx.saved_tensors
        return grad[perm], None, None


class _LoopGroupedMM(torch.autograd.Function):
    """Grouped GEMM as a loop of per-group matmuls (GPUs without torch._grouped_mm, e.g. A100).

    transformers' fallback also computes every expert's weight gradient in backward; the experts
    here are frozen (dequantized copies), so only the input gradient is computed."""

    @staticmethod
    def forward(ctx, x, weight, offsets):
        ctx.save_for_backward(weight, offsets)
        out = x.new_empty(x.shape[0], weight.shape[2])
        start = 0
        for i, end in enumerate(offsets.tolist()):
            if end > start:
                torch.mm(x[start:end], weight[i], out=out[start:end])
            start = end
        return out

    @staticmethod
    def backward(ctx, grad):
        weight, offsets = ctx.saved_tensors
        dx = grad.new_zeros(grad.shape[0], weight.shape[1])
        start = 0
        for i, end in enumerate(offsets.tolist()):
            if end > start:
                torch.mm(grad[start:end], weight[i].t(), out=dx[start:end])
            start = end
        return dx, None, None


def grouped_mm(x, weight, offsets):
    """x [rows sorted by group, in] @ weight [groups, in, out]; torch's grouped GEMM where it runs."""
    from transformers.integrations.moe import _can_use_grouped_mm, _grouped_mm

    if x.is_cuda and weight.dtype != torch.bfloat16:
        # torch._grouped_mm (Hopper) takes BF16 only; FP32 runs are the tests' references: a plain loop, with
        # weight gradients
        x, out, start = x.to(weight.dtype), [], 0
        for g, end in enumerate(offsets.tolist()):
            out.append(x[start:end] @ weight[g])
            start = end
        return torch.cat(out)
    if weight.requires_grad or _can_use_grouped_mm(x, weight, offsets):
        return _grouped_mm(x, weight, offsets)
    from inference.rl import grouped_gemm

    if grouped_gemm.available(x, weight):  # one Triton launch for all groups (A100, SM120)
        return grouped_gemm.GroupedMM.apply(x.to(weight.dtype), weight, offsets)
    return _LoopGroupedMM.apply(x.to(weight.dtype), weight, offsets)


# the BF16 expert tiles of the experts module used last: {"owner": weakref, (start, stop): (gate_up, down)}
_EXPERT_CACHE: dict = {}


def clear_expert_cache() -> None:
    _EXPERT_CACHE.clear()


def cached_expert_weights(experts, start: int, stop: int):
    """expert_weights, kept for the experts module used last (one layer's BF16 experts at a time).

    A decoder layer calls its experts once per token chunk, and backward recomputes the chunks of one
    layer after another, so each pass over a layer dequantizes its experts once instead of per chunk."""
    if not getattr(experts, "_rft_cache_experts", _DEFAULTS["cache_experts"]) \
            or any(p.requires_grad for p in experts.parameters()):
        # trainable BF16 experts: views of the parameters, taken in the current grad mode (a view cached under
        # no_grad, as prefix stop-grad's prefix runs, would carry no gradient into a recompute's graph)
        return expert_weights(experts, start, stop)
    owner = _EXPERT_CACHE.get("owner")
    if owner is None or owner() is not experts:
        _EXPERT_CACHE.clear()
        _EXPERT_CACHE["owner"] = weakref.ref(experts)
    if (start, stop) not in _EXPERT_CACHE:
        _EXPERT_CACHE[(start, stop)] = _assemble(experts, start, stop)
    return _EXPERT_CACHE[(start, stop)]


def _assemble(experts, start: int, stop: int):
    """expert_weights for many experts, dequantized one tile at a time into preallocated tensors, so the
    transient memory is one tile's (all 512 at once needed ~7 GB of temporaries next to the 5 GB result)."""
    tile = getattr(experts, "_rft_expert_tile", _DEFAULTS["expert_tile"])
    trainable = any(p.requires_grad for p in experts.parameters())  # BF16 parameters: views, keep autograd
    if stop - start <= tile or trainable:
        return expert_weights(experts, start, stop)
    gate_up = down = None
    for first in range(start, stop, tile):
        last = min(stop, first + tile)
        g, d = expert_weights(experts, first, last)
        if gate_up is None:
            gate_up = g.new_empty((stop - start, *g.shape[1:]))
            down = d.new_empty((stop - start, *d.shape[1:]))
        gate_up[first - start:last - start] = g
        down[first - start:last - start] = d
        del g, d
    return gate_up, down


def moe_combine(rows, weights):
    """Each token's k expert outputs ([1, n, k, d] in token order) times their routing weights ([1, n, k]),
    summed in FP32 in a fixed order. A BF16 index_add here adds in a different order every run (CUDA
    atomics), so the forward differed between identical steps and near-tied routing flipped further down."""
    return (rows.float() * weights.unsqueeze(-1).float()).sum(dim=2)


def grouped_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Routed experts as grouped GEMMs over tiles of experts; same result as the per-expert loop."""
    _grouped_mm = grouped_mm

    tile = getattr(self, "_rft_expert_tile", _DEFAULTS["expert_tile"])
    n, k = top_k_index.shape
    flat_ids = top_k_index.reshape(-1)
    order = torch.argsort(flat_ids, stable=True)
    inverse = torch.argsort(order)
    rows = gather_rows(hidden_states, order // k, inverse, k)
    compiled = getattr(self, "_rft_compile_moe", False) and rows.is_cuda

    def combine(out):  # undo the sort (a gather both ways), then weight and sum
        rows = _PermuteRows.apply(out, order, inverse).view(1, n, k, -1)
        if compiled:
            return static_call("moe_combine", n, (rows, top_k_weights[None]))[0].to(hidden_states.dtype)
        return moe_combine(rows, top_k_weights[None])[0].to(hidden_states.dtype)

    if getattr(self, "_rft_cache_experts", _DEFAULTS["cache_experts"]):
        # all experts are cached anyway: one grouped GEMM per projection, offsets computed on the GPU
        # (no host sync, no concatenation of per-tile outputs)
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat_ids.device)
        counts.scatter_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=torch.int32))
        offsets = torch.cumsum(counts, 0, dtype=torch.int32)
        gate_up, down = cached_expert_weights(self, 0, self.num_experts)
        return combine(_grouped_mm(self._apply_gate(_grouped_mm(rows, gate_up, offsets)), down, offsets))
    counts = torch.bincount(flat_ids, minlength=self.num_experts)
    bounds = torch.cumsum(counts, 0).tolist()  # one host sync per call
    outputs, first_row = [], 0
    for start in range(0, self.num_experts, tile):
        stop = min(self.num_experts, start + tile)
        last_row = bounds[stop - 1]
        if last_row > first_row:
            gate_up, down = cached_expert_weights(self, start, stop)
            offsets = torch.cumsum(counts[start:stop], 0).to(torch.int32)
            gated = self._apply_gate(_grouped_mm(rows[first_row:last_row], gate_up, offsets))
            outputs.append(_grouped_mm(gated, down, offsets))
        first_row = last_row
    return combine(torch.cat(outputs) if outputs else rows[:0])


# ---------------------------------------------------------------------------
# Hyper-connections as pure functions (compiled once for all layers)
# ---------------------------------------------------------------------------

def _gated_residual(x, norm_weight, down, up, inject, eps: float, hc: int):
    """Qwen4ExpTextGatedResidual.forward without the module: (mixed input [.., hidden], injection weights [.., hc])."""
    xf = x.float().unflatten(-1, (hc, -1))
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    normed = (normed.flatten(-2) * (1.0 + norm_weight.float())).to(x.dtype)
    mix = F.silu(F.linear(normed, down) / hc)
    mix = torch.sigmoid(F.linear(mix, up))
    mixed = (mix.unflatten(-1, (hc, -1)) * normed.unflatten(-1, (hc, -1))).mean(dim=-2)
    weights = 2 * torch.sigmoid(F.linear(normed, inject) / hc)
    return mixed, weights


def _hc_args(module):
    return (module.hc_norm.weight, module.input_mix_weight_down.weight, module.input_mix_weight_up.weight,
            module.block_inject_weight.weight)


def hc_before_mixer(x, norm_weight, down, up, inject, eps: float, hc: int):
    return _gated_residual(x, norm_weight, down, up, inject, eps, hc)


def hc_after_mixer(x, mixer_out, mixer_weights, norm_weight, down, up, inject, eps: float, hc: int):
    """Inject the mixer output into the streams, then the MLP's gated residual: (streams, mlp input, weights)."""
    x = x + (mixer_out.unsqueeze(-2) * mixer_weights.unsqueeze(-1)).flatten(-2)
    mlp_in, weights = _gated_residual(x, norm_weight, down, up, inject, eps, hc)
    return x, mlp_in, weights


def hc_inject(x, out, weights):
    return x + (out.unsqueeze(-2) * weights.unsqueeze(-1)).flatten(-2)


_COMPILED: dict = {}


# Compiled variants are finite but more than dynamo's default 8 per function: token buckets x grad modes (prefix
# stop-grad, checkpoint recompute, no-grad validation) for the per-token functions, LoRA module shapes x grad modes
# for lora_add. Past the limit dynamo runs the function eagerly: the 10-20 training run fell to ~600-1,000 tok/s.
RECOMPILE_LIMIT = 64


def _hc_fn(name: str, compiled: bool, dynamic: bool = False):
    fn = globals()[name]
    if not compiled:
        return fn
    if (name, dynamic) not in _COMPILED:
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, RECOMPILE_LIMIT)
        _COMPILED[name, dynamic] = torch.compile(fn, dynamic=dynamic)
    return _COMPILED[name, dynamic]


def _bucket(n: int) -> int:
    """Token count a compiled per-token function runs at: the next power of two, at least 512."""
    return max(512, 1 << (n - 1).bit_length())


def _own_batch_stride(t):
    """A [1, n, ...] chunk of a longer sequence keeps the whole sequence's batch stride, and a compiled graph
    guards on it: one recompile per sample length. Dropping and re-adding the size-1 dim (views) gives the
    chunk its own stride."""
    return t.squeeze(0).unsqueeze(0) if t.dim() > 1 and t.shape[0] == 1 else t


def _pad_tokens(t, dim: int, size: int):
    extra = size - t.shape[dim]
    return t if extra == 0 else F.pad(t, [0, 0] * (t.dim() - dim - 1) + [0, extra])


def static_call(name: str, n: int, tokens: tuple, weights: tuple = (), **constants):
    """A compiled per-token function on `tokens` ([1, n, ...]) zero-padded to _bucket(n), outputs cut back.

    Static graphs, one per bucket (at most 5 for 8,192-token chunks), keep each chunk's checkpoint
    recompute on the same kernels as its forward. Automatic dynamic shapes would switch the 8,192
    chunks from a static graph in forward to a dynamic one in recompute once a shorter last chunk
    recompiled, and fully dynamic graphs were ~12% slower (GDN layer 1.03 -> 1.17 s on an A100).
    Padding is exact: every function here works token by token."""
    size = _bucket(n)
    out = _hc_fn(name, True)(*(_pad_tokens(_own_batch_stride(t), 1, size) for t in tokens), *weights, **constants)
    return tuple(o[:, :n] for o in out) if isinstance(out, tuple) else out[:, :n]



# ---------------------------------------------------------------------------
# GDN short convolution, channels last
# ---------------------------------------------------------------------------

def short_conv(x, weight, bias, silu: bool):
    """Depthwise causal convolution over [B, T, C] (channels last) with FP32 accumulation, then SiLU.

    transformers' fallback (no causal_conv1d package) runs F.conv1d on a [B, C, T] view of the
    projection, so the input and the output are each copied to another layout (forward, recompute
    and backward). Four shifted multiply-adds in the projection's own layout need no copy and
    compile to one kernel. Same result up to BF16 rounding of the sum."""
    k, t = weight.shape[-1], x.shape[1]
    padded = F.pad(x, (0, 0, k - 1, 0))
    w = weight.reshape(weight.shape[0], k).float()
    out = padded[:, 0:t].float() * w[:, 0]
    for i in range(1, k):
        out = out + padded[:, i:i + t].float() * w[:, i]
    if bias is not None:
        out = out + bias.float()
    out = out.to(x.dtype)
    return F.silu(out) if silu else out


_FAST_CONV = {"on": False}


def _causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
    """Drop-in for modeling_qwen4_exp.causal_conv1d_fn ([B, C, T] in and out) on CUDA when enabled."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    if not (_FAST_CONV["on"] and hidden_states.is_cuda and activation in (None, "silu")):
        return m._rft_hf_causal_conv1d_fn(hidden_states, weight, bias=bias, activation=activation, **kwargs)
    # the whole sequence (not chunked): one dynamic graph for every length
    out = _hc_fn("short_conv", True, dynamic=True)(hidden_states.transpose(1, 2), weight, bias, activation == "silu")
    return out.transpose(1, 2)


# ---------------------------------------------------------------------------
# GDN mixer: fused convolution + split, grouped value heads, fused gated norm
# ---------------------------------------------------------------------------

def conv_split(mixed, weight, bias, key_dim: int, key_heads: int, value_heads: int, head_k: int, head_v: int):
    """The GDN short convolution (+ SiLU) of the qkv projection, written straight into contiguous q, k, v
    ([1, N, heads, dim]); fla otherwise copies each split view to make it contiguous."""
    x = short_conv(mixed, weight, bias, silu=True)
    q, k, v = torch.split(x, [key_dim, key_dim, value_heads * head_v], dim=-1)
    n = x.shape[1]
    return (q.reshape(1, n, key_heads, head_k).contiguous(), k.reshape(1, n, key_heads, head_k).contiguous(),
            v.reshape(1, n, value_heads, head_v).contiguous())


def gated_rms_norm(x, gate, weight, eps: float, sigmoid: bool):
    """Qwen4ExpTextRMSNormGated.forward without the module, op for op."""
    dtype = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    h = weight * h.to(dtype)
    g = gate.to(torch.float32)
    h = h * (torch.sigmoid(g) if sigmoid else F.silu(g))
    return h.to(dtype)


_FAST_GDN = {"on": False}


def gdn_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    """transformers' GatedDeltaNet forward for training (`fast_gdn`), with three changes that keep its math:
    the convolution writes contiguous q, k, v (conv_split); q and k stay at their 16 heads and fla's grouped
    value attention maps each to its 3 value heads (transformers materializes repeat_interleave copies of
    both, and sums their gradients back); the gated RMSNorm is one compiled function. Anything else (a cache,
    no fla, CPU) runs transformers' forward."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError:
        chunk_gated_delta_rule = None
    if (not _FAST_GDN["on"] or cache_params is not None or chunk_gated_delta_rule is None
            or not hidden_states.is_cuda or hidden_states.shape[0] != 1
            or self.norm.activation not in ("sigmoid", "silu") or self.activation != "silu"):
        return type(self)._hf_forward(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask,
                                      **kwargs)
    hidden_states = m.apply_mask_to_padding_states(hidden_states, attention_mask)
    n = hidden_states.shape[1]
    mixed_qkv = self.in_proj_qkv(hidden_states)
    z = self.in_proj_z(hidden_states)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)
    q, k, v = _hc_fn("conv_split", True, dynamic=True)(mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias,
                                                       self.key_dim, self.num_k_heads, self.num_v_heads,
                                                       self.head_k_dim, self.head_v_dim)
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    core, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
                                     use_qk_l2norm_in_kernel=True)
    out = _hc_fn("gated_rms_norm", True, dynamic=True)(core.reshape(-1, self.head_v_dim),
                                                       z.reshape(-1, self.head_v_dim), self.norm.weight,
                                                       self.norm.variance_epsilon, self.norm.activation == "sigmoid")
    return self.out_proj(out.reshape(1, n, -1))


def gdn_mixer(self, hidden_states, state=None, return_state: bool = False):
    """The GDN mixer of one sequence [1, n, hidden], optionally continuing a prefix's `state` (the last K - 1 rows
    of the convolution's input and the recurrent state) and returning this sequence's state: prefix stop-grad
    runs the prefix without gradient and the suffix from its state, as a cache would. fast_gdn's kernels on
    CUDA with fla, transformers' functions otherwise (GDN layers without attention masks, batch 1)."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError:
        chunk_gated_delta_rule = None
    fast = (_FAST_GDN["on"] and chunk_gated_delta_rule is not None and hidden_states.is_cuda
            and self.norm.activation in ("sigmoid", "silu") and self.activation == "silu")
    n = hidden_states.shape[1]
    mixed_qkv = self.in_proj_qkv(hidden_states)
    z = self.in_proj_z(hidden_states)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)
    conv_in = mixed_qkv if state is None else torch.cat([state["conv"], mixed_qkv], dim=1)
    skip = conv_in.shape[1] - n  # rows of the prefix's convolution input in front
    if fast:
        q, k, v = _hc_fn("conv_split", True, dynamic=True)(conv_in, self.conv1d.weight.squeeze(1), self.conv1d.bias,
                                                           self.key_dim, self.num_k_heads, self.num_v_heads,
                                                           self.head_k_dim, self.head_v_dim)
        q, k, v = q[:, skip:], k[:, skip:], v[:, skip:]
    else:
        conv = m.causal_conv1d_fn(conv_in.transpose(1, 2), self.conv1d.weight.squeeze(1), self.conv1d.bias,
                                  activation=self.activation).transpose(1, 2)[:, skip:]
        q, k, v = torch.split(conv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(1, n, -1, self.head_k_dim)
        k = k.reshape(1, n, -1, self.head_k_dim)
        v = v.reshape(1, n, -1, self.head_v_dim)
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    initial = None if state is None else state["recurrent"]
    if fast:
        core, final = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=initial,
                                             output_final_state=return_state, use_qk_l2norm_in_kernel=True)
        out = _hc_fn("gated_rms_norm", True, dynamic=True)(core.reshape(-1, self.head_v_dim),
                                                           z.reshape(-1, self.head_v_dim), self.norm.weight,
                                                           self.norm.variance_epsilon, self.norm.activation == "sigmoid")
    else:
        if self.num_v_heads // self.num_k_heads > 1:
            q = q.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            k = k.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        core, final = m.torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=initial,
                                                     output_final_state=return_state, use_qk_l2norm_in_kernel=True)
        out = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    out = self.out_proj(out.reshape(1, n, -1))
    if not return_state:
        return out
    # a copy, not a view: the view kept the whole prefix projection alive until backward (p x 10,240 BF16 per GDN
    # layer, 12 GiB at 32K: prefix stop-grad ran out of memory)
    return out, dict(conv=conv_in[:, -(self.conv_kernel_size - 1):].detach().clone(), recurrent=final.detach())


# ---------------------------------------------------------------------------
# LoRA branch, fused
# ---------------------------------------------------------------------------

def lora_add(result, x, a, b, scaling: float):
    """PEFT's vanilla LoRA update, op for op: the adapter runs in its own dtype (FP32 by default), the BF16
    base output is promoted for the sum and the sum is rounded back once."""
    return (result + F.linear(F.linear(x.to(a.dtype), a), b) * scaling).to(result.dtype)


_FAST_LORA = {"on": False}


def lora_linear_forward(self, x, *args, **kwargs):
    """peft's lora.Linear forward with the update as one compiled function on CUDA (`fast_lora`).

    PEFT materializes the adapter output and the promoted sum in FP32 ([N, 10,240] for in_proj_qkv)
    and casts the result back: three full passes over the output per LoRA module per pass. Compiled,
    the casts, scaling and sum fuse around the two small GEMMs. Anything but a single vanilla adapter
    without dropout or bias (merged, disabled, variants, mixed batches) keeps PEFT's forward."""
    names = [n for n in self.active_adapters if n in self.lora_A.keys()]
    if (not _FAST_LORA["on"] or args or kwargs or self.disable_adapters or self.merged or len(names) != 1
            or names[0] in self.lora_variant or not isinstance(self.lora_dropout[names[0]], torch.nn.Identity)
            or self.lora_A[names[0]].bias is not None or self.lora_B[names[0]].bias is not None
            or not getattr(self, "cast_input_dtype_enabled", True)):
        return type(self)._rft_peft_forward(self, x, *args, **kwargs)
    name = names[0]
    fn = _hc_fn("lora_add", x.is_cuda, dynamic=True)  # token counts vary per sample: one dynamic graph
    return fn(self.base_layer(x), x, self.lora_A[name].weight, self.lora_B[name].weight, float(self.scaling[name]))


def patch_lora(on: bool) -> None:
    """Route peft's lora.Linear through lora_linear_forward (a no-op unless `on`)."""
    from peft.tuners.lora.layer import Linear

    if not hasattr(Linear, "_rft_peft_forward"):
        Linear._rft_peft_forward = Linear.forward
    Linear.forward = lora_linear_forward
    _FAST_LORA["on"] = on


# ---------------------------------------------------------------------------
# PLE n-gram lookup, once per step
# ---------------------------------------------------------------------------

def ngram_embedding_forward(self, input_ids, past_key_values):
    """transformers' n-gram lookup, reused by the checkpoint recompute, or taken from provide_ngram.

    The lookup depends only on the token ids, and its table is frozen and memory-mapped on the host
    (ple_mmap): a 64K pass gathers rows on the CPU for ~0.46 s, and the PLE checkpoint ran it again in
    backward. The forward keeps its output (N x 2,560, 335 MB at 64K) until the recompute takes it."""
    if past_key_values is not None or self.ngram_embedding.weight.requires_grad:
        return type(self)._hf_forward(self, input_ids, past_key_values)  # a trainable table saves its indices
    provided = self.__dict__.pop("_rft_ngram_provided", None)
    if provided is not None and tuple(provided.shape[:2]) == tuple(input_ids.shape):
        out = provided.to(input_ids.device)
    elif not torch.is_grad_enabled():
        return type(self)._hf_forward(self, input_ids, past_key_values)
    else:
        cached = getattr(self, "_rft_ngram_cache", None)
        self._rft_ngram_cache = None
        if cached is not None and cached[0]() is input_ids:
            return cached[1]
        out = type(self)._hf_forward(self, input_ids, past_key_values)
    if torch.is_grad_enabled():
        self._rft_ngram_cache = (weakref.ref(input_ids), out)
    return out


_CPU_TWINS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _ngram_modules(model) -> list:
    return [m for m in model.modules() if type(m).__name__ == "Qwen4ExpTextNGramEmbedding"]


def precompute_ngram(model, input_ids: torch.Tensor) -> list[torch.Tensor]:
    """Every n-gram module's output for host `input_ids` ([1, N]), on the host: what ngram_embedding_forward
    would compute. train_rft's prefetch thread calls this, so the table lookup (and any page faults on the
    memory-mapped table) overlaps the GPU's work on the previous batch. A CPU twin of each module holds host
    copies of its small id buffers and shares its table."""
    out = []
    for module in _ngram_modules(model):
        twin = _CPU_TWINS.get(module)
        if twin is None:
            twin = object.__new__(type(module))
            twin.__dict__.update(module.__dict__)
            twin._buffers = {k: (v.cpu() if v is not None else None) for k, v in module._buffers.items()}
            twin._modules = dict(module._modules)
            twin._parameters = dict(module._parameters)
            _CPU_TWINS[module] = twin
        with torch.no_grad():
            out.append(getattr(type(module), "_hf_forward", type(module).forward)(twin, input_ids.cpu(), None))
    return out


def provide_ngram(model, rows: list[torch.Tensor] | None) -> None:
    """Hand precompute_ngram's output to the next forward of each n-gram module (used once, then the cache)."""
    for module, row in zip(_ngram_modules(model), rows or []):
        module.__dict__["_rft_ngram_provided"] = row

def ple_forward(self, hidden_states, input_ids, past_key_values, conv_mask=None):
    """transformers' PLE layer with its per-token part in checkpointed token chunks (training, batch 1).

    Only the short convolution spans tokens. Run whole, the PLE block's norms built several full-sequence
    FP32 [N, 10,240] temporaries in backward (4.69 GiB each at 123K tokens: the size of the allocation that
    ran recompute "parts" out of memory). Chunked, those exist one chunk at a time; same operations per token."""
    if past_key_values is not None or conv_mask is not None or hidden_states.shape[0] != 1:
        return type(self)._hf_forward(self, hidden_states, input_ids, past_key_values, conv_mask=conv_mask)
    hc, hidden = self.hc_count, self.hidden_size
    embeddings = self.ple_embedding(input_ids, None)

    def per_token(h, emb):
        key_normed = self.norm_key(self.key_proj(emb)).unflatten(-1, (hc, hidden))
        value = self.value_proj(emb)
        query_normed = self.norm_query(h).unflatten(-1, (hc, hidden))
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(hidden)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = torch.sigmoid(gate) * value.unsqueeze(-2)
        return gated.flatten(-2), self.norm_conv(gated.flatten(-2))

    gated, normed = _chunked(per_token, hidden_states, embeddings, size=getattr(self, "_rft_token_chunk", 8192))
    return gated + self._short_conv(normed, None)


# ---------------------------------------------------------------------------
# Prefix stop-grad
# ---------------------------------------------------------------------------

_PREFIX = {"p": 0}


def set_prefix(p: int) -> None:
    """Positions before `p` run without gradient in the next forward/backward (0 = off): the LoRA learns only
    from the computation at and after p, as if the prefix were a cache (truncated backpropagation). p is the
    position whose output predicts the first trained token."""
    _PREFIX["p"] = int(p)


def split_decoder_forward(self, hidden_states, p: int, position_embeddings, attention_mask, conv_mask,
                          ple_input_ids, kwargs):
    """chunked_decoder_forward with positions before p computed without gradient (prefix stop-grad).

    Per-token parts and the mixer run on the prefix under no_grad (no checkpoints, nothing saved); the suffix
    runs as usual, its GDN continuing the prefix's convolution and recurrent state and its QSA attending to the
    prefix's keys and values as constants. PLE has no trainable weights: it runs on the whole sequence, and
    its gradient reaches only suffix positions since the prefix's hidden states carry none."""
    size = getattr(self, "_rft_token_chunk", _DEFAULTS["token_chunk"])
    parts = getattr(self, "_rft_recompute", "layer") == "parts"
    if self.ple is not None:
        ple = lambda h, ids: self.ple(h, ids, None, conv_mask=conv_mask)
        hidden_states = hidden_states + checkpoint(ple, hidden_states, ple_input_ids, use_reentrant=False)
    before_mixer, after_mixer = _per_token_functions(self, hidden_states.is_cuda)
    h_p, h_s = hidden_states[:, :p], hidden_states[:, p:]
    with torch.no_grad():
        mixed_p, inject_p = _chunked(before_mixer, h_p, size=size)
    mixed_s, inject_s = _chunked(before_mixer, h_s, size=size)
    if self.layer_type == "linear_attention":
        with torch.no_grad():
            out_p, state = gdn_mixer(self.linear_attn, mixed_p, return_state=True)
        mixer = lambda m: gdn_mixer(self.linear_attn, m, state=state)
        out_s = checkpoint(mixer, mixed_s, use_reentrant=False) if parts else mixer(mixed_s)
    else:
        from inference.rl import qsa

        out_p, out_s = qsa.qsa_split(self.self_attn, mixed_p, mixed_s, position_embeddings, checkpointed=parts)
    with torch.no_grad():
        y_p = _chunked(after_mixer, h_p, out_p, inject_p, size=size)
    return torch.cat([y_p, _chunked(after_mixer, h_s, out_s, inject_s, size=size)], dim=1)


# ---------------------------------------------------------------------------
# Token-chunked decoder layer
# ---------------------------------------------------------------------------

def _chunked(fn, *tensors, size: int):
    """fn over chunks of dim 1 (tokens), each checkpointed when gradients are on; outputs concatenated."""
    n = tensors[0].shape[1]
    if n <= size:
        pieces = [checkpoint(fn, *tensors, use_reentrant=False) if torch.is_grad_enabled() else fn(*tensors)]
    else:
        # split, not slicing: each slice's backward would build a zero tensor of the whole sequence and the
        # chunks' gradients would be added up one by one; split's backward concatenates them once
        pieces = []
        for part in zip(*(t.split(size, dim=1) for t in tensors)):
            pieces.append(checkpoint(fn, *part, use_reentrant=False) if torch.is_grad_enabled() else fn(*part))
    if isinstance(pieces[0], tuple):
        return tuple(torch.cat(p, dim=1) for p in zip(*pieces))
    return torch.cat(pieces, dim=1)


def mixer_forward(self, hyper_input):
    """The model's final hyper-connection mixer (Qwen4ExpTextGatedResidual, use_combine=False) in token chunks.

    It runs on every position and its grouped RMSNorm upcasts the whole [N, 4 x 2,560] stream tensor to FP32:
    4.98 GiB per temporary at 130,561 tokens, which ran the 10-20 rehearsal's longest sample out of GPU memory.
    Every operation in it works token by token, so chunks give the same values."""
    forward = type(self).forward
    if hyper_input.dim() != 3 or hyper_input.shape[0] != 1:
        return forward(self, hyper_input)
    size = getattr(self, "_rft_token_chunk", None) or _DEFAULTS["token_chunk"]
    return _chunked(lambda x: forward(self, x), hyper_input, size=size)


def _final_mixers(model) -> list:
    return [mod.hyper_connection_mixer for mod in model.modules()
            if type(mod).__name__ == "Qwen4ExpTextModel" and hasattr(mod, "hyper_connection_mixer")]


def _per_token_functions(self, cuda: bool):
    """The decoder layer's per-token work before the mixer (attention hyper-connection: mixer input, injection
    weights) and after it (inject, MLP hyper-connection, MoE, inject), compiled with compile_hc on CUDA."""
    compiled = getattr(self, "_rft_compile_hc", False) and cuda
    attn_hc, mlp_hc = self.attn_hyper_connection, self.mlp_hyper_connection
    hc_kw = dict(eps=attn_hc.hc_norm.eps, hc=attn_hc.hc_count)

    def before_mixer(h):
        if compiled:
            return static_call("hc_before_mixer", h.shape[1], (h,), _hc_args(attn_hc), **hc_kw)
        mixed, _, weights = attn_hc(h)
        return mixed, weights

    def after_mixer(h, mixer_out, weights):
        if compiled:
            n = h.shape[1]
            h, mlp_in, mlp_weights = static_call("hc_after_mixer", n, (h, mixer_out, weights), _hc_args(mlp_hc),
                                                 **hc_kw)
            return static_call("hc_inject", n, (h, self.mlp(mlp_in), mlp_weights))
        h = h + (mixer_out.unsqueeze(-2) * weights.unsqueeze(-1)).flatten(-2)
        mlp_in, hyper, mlp_weights = mlp_hc(h)
        mlp_out = self.mlp(mlp_in)
        return hyper + (mlp_out.unsqueeze(-2) * mlp_weights.unsqueeze(-1)).flatten(-2)

    return before_mixer, after_mixer


def chunked_decoder_forward(self, hidden_states, position_embeddings, attention_mask=None, conv_mask=None,
                            past_key_values=None, ple_input_ids=None, **kwargs):
    if past_key_values is not None or hidden_states.shape[0] != 1:
        return type(self)._hf_forward(self, hidden_states, position_embeddings, attention_mask=attention_mask,
                                      conv_mask=conv_mask, past_key_values=past_key_values,
                                      ple_input_ids=ple_input_ids, **kwargs)
    p = _PREFIX["p"]
    if torch.is_grad_enabled() and 0 < p < hidden_states.shape[1]:
        return split_decoder_forward(self, hidden_states, p, position_embeddings, attention_mask, conv_mask,
                                     ple_input_ids, kwargs)
    size = getattr(self, "_rft_token_chunk", _DEFAULTS["token_chunk"])
    if self.ple is not None:  # its short convolution spans tokens: one checkpoint, not chunks
        ple = lambda h, ids: self.ple(h, ids, None, conv_mask=conv_mask)
        extra = checkpoint(ple, hidden_states, ple_input_ids, use_reentrant=False) if torch.is_grad_enabled() \
            else ple(hidden_states, ple_input_ids)
        hidden_states = hidden_states + extra

    before_mixer, after_mixer = _per_token_functions(self, hidden_states.is_cuda)
    mixed, inject = _chunked(before_mixer, hidden_states, size=size)

    def mixer(h):
        if self.layer_type == "linear_attention":
            return self.linear_attn(h, cache_params=None, attention_mask=conv_mask, **kwargs)
        return self.self_attn(h, position_embeddings, attention_mask=attention_mask, past_key_values=None,
                              **kwargs)[0]

    if getattr(self, "_rft_recompute", "layer") == "parts" and torch.is_grad_enabled():
        mixed = checkpoint(mixer, mixed, use_reentrant=False)
    else:
        mixed = mixer(mixed)
    return _chunked(after_mixer, hidden_states, mixed, inject, size=size)


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

def configure(model, token_chunk: int | None = None, expert_tile: int | None = None,
              recompute: str | None = None, compile_hc: bool | None = None, compile_moe: bool | None = None,
              cache_experts: bool | None = None) -> None:
    """Set the chunk sizes (lower them after an OOM), the recompute mode and compiled hyper-connections
    on every patched module."""
    if recompute is not None and recompute not in RECOMPUTE_MODES:
        raise ValueError(f"recompute must be one of {RECOMPUTE_MODES}, got {recompute!r}")
    for module in model.modules():
        name = type(module).__name__
        if name == "Qwen4ExpTextPLELayer" and token_chunk:
            module._rft_token_chunk = token_chunk
        if name == "Qwen4ExpTextModel" and token_chunk and hasattr(module, "hyper_connection_mixer"):
            module.hyper_connection_mixer._rft_token_chunk = token_chunk
        if name == "Qwen4ExpTextDecoderLayer":
            if token_chunk:
                module._rft_token_chunk = token_chunk
            if compile_hc is not None:
                module._rft_compile_hc = compile_hc
            if recompute:
                if not hasattr(module, "_rft_layer_checkpointing"):  # what gradient_checkpointing_enable set
                    module._rft_layer_checkpointing = getattr(module, "gradient_checkpointing", False)
                module._rft_recompute = recompute if module._rft_layer_checkpointing else "layer"
                module.gradient_checkpointing = module._rft_layer_checkpointing and recompute == "layer"
        elif name == "Qwen4ExpTextExperts":
            if expert_tile:
                module._rft_expert_tile = expert_tile
            if compile_moe is not None:
                module._rft_compile_moe = compile_moe
            if cache_experts is not None:
                module._rft_cache_experts = cache_experts


def current_recompute(model) -> str:
    """The recompute mode of the patched decoder layers ("layer" when not set)."""
    for module in model.modules():
        if type(module).__name__ == "Qwen4ExpTextDecoderLayer":
            return getattr(module, "_rft_recompute", "layer")
    return "layer"


def patch_for_training(model, token_chunk: int = _DEFAULTS["token_chunk"],
                       expert_tile: int = _DEFAULTS["expert_tile"], recompute: str = _DEFAULTS["recompute"],
                       compile_hc: bool = _DEFAULTS["compile_hc"], compile_moe: bool = False,
                       cache_experts: bool = _DEFAULTS["cache_experts"], fast_conv: bool = False,
                       fast_lora: bool = False, fast_gdn: bool = False) -> dict:
    """Route qwen4_exp experts through grouped GEMMs and chunk decoder layers over tokens.

    Call it after gradient_checkpointing_enable(): `recompute="parts"` replaces the layer-level
    checkpoint it sets. Safe to call again (e.g. after reloading this module)."""
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    ALL_EXPERTS_FUNCTIONS.register(EXPERTS_IMPLEMENTATION, grouped_experts_forward)
    configs = set()
    for module in model.modules():
        if isinstance(module, m.Qwen4ExpTextExperts):
            configs.add(id(module.config))
            if not hasattr(module, "_rft_previous_implementation"):
                previous = module.config._experts_implementation
                module._rft_previous_implementation = "eager" if previous == EXPERTS_IMPLEMENTATION else previous
    for module in model.modules():
        if isinstance(module, m.Qwen4ExpTextExperts):
            module.config._experts_implementation = EXPERTS_IMPLEMENTATION
    layer = m.Qwen4ExpTextDecoderLayer
    if not hasattr(layer, "_hf_forward"):
        layer._hf_forward = layer.forward
    layer.forward = chunked_decoder_forward
    ple_layer = m.Qwen4ExpTextPLELayer
    if not hasattr(ple_layer, "_hf_forward"):
        ple_layer._hf_forward = ple_layer.forward
    ple_layer.forward = ple_forward
    ngram = m.Qwen4ExpTextNGramEmbedding
    if not hasattr(ngram, "_hf_forward"):
        ngram._hf_forward = ngram.forward
    ngram.forward = ngram_embedding_forward
    for mixer in _final_mixers(model):  # this instance only: the decoder layers' gated residuals have their own path
        mixer.forward = types.MethodType(mixer_forward, mixer)
    if not hasattr(m, "_rft_hf_causal_conv1d_fn"):
        m._rft_hf_causal_conv1d_fn = m.causal_conv1d_fn
    m.causal_conv1d_fn = _causal_conv1d_fn
    _FAST_CONV["on"] = fast_conv
    gdn = m.Qwen4ExpTextGatedDeltaNet
    if not hasattr(gdn, "_hf_forward"):
        gdn._hf_forward = gdn.forward
    gdn.forward = gdn_forward
    _FAST_GDN["on"] = fast_gdn
    try:
        patch_lora(fast_lora)
    except ImportError:  # no peft
        pass
    configure(model, token_chunk, expert_tile, recompute, compile_hc, compile_moe, cache_experts)
    return dict(experts_implementation=EXPERTS_IMPLEMENTATION, configs=len(configs), token_chunk=token_chunk,
                expert_tile=expert_tile, recompute=current_recompute(model), compile_hc=compile_hc,
                compile_moe=compile_moe, cache_experts=cache_experts, fast_conv=fast_conv, fast_lora=fast_lora,
                fast_gdn=fast_gdn)


def unpatch(model=None) -> None:
    """Back to transformers' decoder layers, short convolution and (with `model`) the experts' previous
    implementation, with layer-level checkpointing: the reference the patches are checked against."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    if model is not None:
        configure(model, recompute="layer")
        for module in model.modules():
            if isinstance(module, m.Qwen4ExpTextExperts) and hasattr(module, "_rft_previous_implementation"):
                module.config._experts_implementation = module._rft_previous_implementation
        clear_expert_cache()
        for mixer in _final_mixers(model):
            mixer.__dict__.pop("forward", None)

    layer = m.Qwen4ExpTextDecoderLayer
    if hasattr(layer, "_hf_forward"):
        layer.forward = layer._hf_forward
    if hasattr(m, "_rft_hf_causal_conv1d_fn"):
        m.causal_conv1d_fn = m._rft_hf_causal_conv1d_fn
    if hasattr(m.Qwen4ExpTextNGramEmbedding, "_hf_forward"):
        m.Qwen4ExpTextNGramEmbedding.forward = m.Qwen4ExpTextNGramEmbedding._hf_forward
    if hasattr(m.Qwen4ExpTextPLELayer, "_hf_forward"):
        m.Qwen4ExpTextPLELayer.forward = m.Qwen4ExpTextPLELayer._hf_forward
    _FAST_LORA["on"] = False  # peft's own LoRA forward
    _FAST_GDN["on"] = False


@torch.no_grad()
def check_experts(model, tokens: int = 4096) -> dict:
    """Grouped vs the experts' previous implementation on random routing for one layer: max relative error."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    experts = next(module for module in model.modules() if isinstance(module, m.Qwen4ExpTextExperts))
    device = next(iter([*experts.parameters(), *experts.buffers()])).device
    hidden = experts.hidden_dim
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16) * 0.5
    k = getattr(experts.config, "num_experts_per_tok", 2)
    index = torch.stack([torch.randperm(experts.num_experts, device=device)[:k] for _ in range(tokens)])
    weights = torch.softmax(torch.randn(tokens, k, device=device), -1).to(torch.bfloat16)
    implementation = experts.config._experts_implementation
    experts.config._experts_implementation = getattr(experts, "_rft_previous_implementation", "eager")
    try:
        reference = experts(x, index, weights).float()
    finally:
        experts.config._experts_implementation = implementation
    ours = grouped_experts_forward(experts, x, index, weights).float()
    return dict(max_rel_error=float((ours - reference).abs().max() / reference.abs().max().clamp_min(1e-6)),
                reference_implementation=getattr(experts, "_rft_previous_implementation", "eager"))
