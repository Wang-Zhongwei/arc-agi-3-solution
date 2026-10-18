"""Training-time patches for qwen4_exp decoder layers: grouped MoE and token chunking.

Two exact rewrites of transformers' forward, for long single-sequence training:

- grouped experts: the routed (token, expert) rows are sorted by expert and run as
  grouped GEMMs (`torch._grouped_mm`, one call per projection per tile of experts)
  instead of a Python loop over 512 experts. AutoRound's own grouped path rejects its
  packed INT4 layers, so tiles of experts are dequantized to BF16 on the fly (the
  weights serving uses) and freed after use.
- token chunks: everything in a decoder layer except the sequence mixer (GDN or QSA)
  and the PLE short convolution is per token (hyper-connections, router, experts,
  shared expert, injections). It runs in chunks of tokens, each under its own
  checkpoint, so that part's memory no longer grows with the sequence length.

Scope: batch 1, no cache (training). Anything else runs transformers' forward.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

EXPERTS_IMPLEMENTATION = "rft_grouped"
_DEFAULTS = dict(token_chunk=8192, expert_tile=128)


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


def grouped_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Routed experts as grouped GEMMs over tiles of experts; same result as the per-expert loop."""
    from transformers.integrations.moe import _grouped_mm

    tile = getattr(self, "_rft_expert_tile", _DEFAULTS["expert_tile"])
    n, k = top_k_index.shape
    flat_ids = top_k_index.reshape(-1)
    order = torch.argsort(flat_ids, stable=True)
    tokens = order // k
    counts = torch.bincount(flat_ids, minlength=self.num_experts)
    bounds = torch.cumsum(counts, 0).tolist()  # one host sync per call
    rows = hidden_states[tokens]
    outputs, first_row = [], 0
    for start in range(0, self.num_experts, tile):
        stop = min(self.num_experts, start + tile)
        last_row = bounds[stop - 1]
        if last_row > first_row:
            gate_up, down = expert_weights(self, start, stop)
            offsets = torch.cumsum(counts[start:stop], 0).to(torch.int32)
            gated = self._apply_gate(_grouped_mm(rows[first_row:last_row], gate_up, offsets))
            outputs.append(_grouped_mm(gated, down, offsets))
        first_row = last_row
    out = torch.cat(outputs) if outputs else rows[:0]
    out = out * top_k_weights.reshape(-1)[order].unsqueeze(-1).to(out.dtype)
    result = torch.zeros(n, hidden_states.shape[-1], dtype=out.dtype, device=out.device).index_add(0, tokens, out)
    return result.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Token-chunked decoder layer
# ---------------------------------------------------------------------------

def _chunked(fn, *tensors, size: int):
    """fn over chunks of dim 1 (tokens), each checkpointed when gradients are on; outputs concatenated."""
    n = tensors[0].shape[1]
    if n <= size:
        pieces = [checkpoint(fn, *tensors, use_reentrant=False) if torch.is_grad_enabled() else fn(*tensors)]
    else:
        pieces = []
        for start in range(0, n, size):
            part = [t[:, start:start + size] for t in tensors]
            pieces.append(checkpoint(fn, *part, use_reentrant=False) if torch.is_grad_enabled() else fn(*part))
    if isinstance(pieces[0], tuple):
        return tuple(torch.cat(p, dim=1) for p in zip(*pieces))
    return torch.cat(pieces, dim=1)


def chunked_decoder_forward(self, hidden_states, position_embeddings, attention_mask=None, conv_mask=None,
                            past_key_values=None, ple_input_ids=None, **kwargs):
    if past_key_values is not None or hidden_states.shape[0] != 1:
        return type(self)._hf_forward(self, hidden_states, position_embeddings, attention_mask=attention_mask,
                                      conv_mask=conv_mask, past_key_values=past_key_values,
                                      ple_input_ids=ple_input_ids, **kwargs)
    size = getattr(self, "_rft_token_chunk", _DEFAULTS["token_chunk"])
    if self.ple is not None:  # its short convolution spans tokens: one checkpoint, not chunks
        ple = lambda h, ids: self.ple(h, ids, None, conv_mask=conv_mask)
        extra = checkpoint(ple, hidden_states, ple_input_ids, use_reentrant=False) if torch.is_grad_enabled() \
            else ple(hidden_states, ple_input_ids)
        hidden_states = hidden_states + extra

    def before_mixer(h):
        mixed, _, weights = self.attn_hyper_connection(h)
        return mixed, weights

    mixed, inject = _chunked(before_mixer, hidden_states, size=size)
    if self.layer_type == "linear_attention":
        mixed = self.linear_attn(mixed, cache_params=None, attention_mask=conv_mask, **kwargs)
    else:
        mixed, _ = self.self_attn(mixed, position_embeddings, attention_mask=attention_mask, past_key_values=None,
                                  **kwargs)

    def after_mixer(h, mixer_out, weights):
        h = h + (mixer_out.unsqueeze(-2) * weights.unsqueeze(-1)).flatten(-2)
        mlp_in, hyper, mlp_weights = self.mlp_hyper_connection(h)
        mlp_out = self.mlp(mlp_in)
        return hyper + (mlp_out.unsqueeze(-2) * mlp_weights.unsqueeze(-1)).flatten(-2)

    return _chunked(after_mixer, hidden_states, mixed, inject, size=size)


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

def configure(model, token_chunk: int | None = None, expert_tile: int | None = None) -> None:
    """Set the chunk sizes on every patched module (lower them after an OOM)."""
    for module in model.modules():
        name = type(module).__name__
        if name == "Qwen4ExpTextDecoderLayer" and token_chunk:
            module._rft_token_chunk = token_chunk
        elif name == "Qwen4ExpTextExperts" and expert_tile:
            module._rft_expert_tile = expert_tile


def patch_for_training(model, token_chunk: int = _DEFAULTS["token_chunk"],
                       expert_tile: int = _DEFAULTS["expert_tile"]) -> dict:
    """Route qwen4_exp experts through grouped GEMMs and chunk decoder layers over tokens.

    Safe to call again (e.g. after reloading this module)."""
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
    configure(model, token_chunk, expert_tile)
    return dict(experts_implementation=EXPERTS_IMPLEMENTATION, configs=len(configs), token_chunk=token_chunk,
                expert_tile=expert_tile)


def unpatch(model=None) -> None:
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    layer = m.Qwen4ExpTextDecoderLayer
    if hasattr(layer, "_hf_forward"):
        layer.forward = layer._hf_forward


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
