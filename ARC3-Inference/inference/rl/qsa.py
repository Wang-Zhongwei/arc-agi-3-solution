"""Training-time QSA attention for qwen4_exp, replacing transformers' reference version.

Each full-attention layer first lets an indexer pick, per query, the top
`indexer_budget // indexer_compress_ratio` blocks of `compress_ratio` past
tokens (plus the incomplete tail block) and then attends only to those tokens.
transformers implements this with a Python loop over queries and a dense
[N, N] mask, which runs out of memory at 32K tokens. This version computes the
same selection and attention in query chunks:

- indexer: block keys are pooled once for the whole sequence; scores for a
  chunk of queries are one matmul, then topk over blocks (no gradient: the
  indexer is frozen and topk is not differentiable);
- attention: each query attends to its own ~budget gathered keys/values; each
  chunk is checkpointed so the gathered keys/values are not kept for backward.

Scope: batch 1, no cache, causal mask only (training on packed single
sequences). Anything else falls back to the original forward.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def _rope(x, cos, sin):
    """The model's own rotary embedding (it rotates only the first cos.shape[-1] dims).

    x [L, heads, d] with cos/sin [L, rotary_dim]."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    return apply_rotary_pos_emb(x, cos=cos, sin=sin, unsqueeze_dim=1)


@torch.no_grad()
def indexer_queries_and_keys(indexer, hidden_states, cos, sin):
    """Index queries [N, H, d] and pooled block keys [N // ratio, d], both float32, for one sequence."""
    n = hidden_states.shape[1]
    ratio, d = indexer.compress_ratio, indexer.index_head_dim
    qk = indexer.index_qk_proj(hidden_states[0])
    q, raw_keys = torch.split(qk, [indexer.index_n_heads * d, indexer.index_kv_heads * d], dim=-1)
    q = indexer.q_layernorm(q.view(n, -1, d))
    q = _rope(q, cos[0], sin[0]).float()
    raw_keys = raw_keys.view(n, d)
    blocks = n // ratio
    pooled = raw_keys[: blocks * ratio].view(blocks, ratio, d).float().mean(dim=1).to(raw_keys.dtype)
    pooled = indexer.k_layernorm(pooled)
    starts = torch.arange(blocks, device=q.device) * ratio
    return q, _rope(pooled[:, None, :], cos[0, starts], sin[0, starts])[:, 0].float()


@torch.no_grad()
def select_tokens(indexer, hidden_states, cos, sin, chunk_size: int) -> torch.Tensor:
    """Indices [N, S] of the tokens each query attends to (-1 = padding), S = budget + ratio - 1.

    Blocks tied at the cut-off score (typically several at 0 after the relu) may be
    chosen differently than the reference's per-query topk; the selection is the
    same up to those ties."""
    n = hidden_states.shape[1]
    ratio, topk, d = indexer.compress_ratio, indexer.block_topk, indexer.index_head_dim
    q, block_keys = indexer_queries_and_keys(indexer, hidden_states, cos, sin)
    blocks = block_keys.shape[0]

    width = indexer.token_budget + ratio - 1
    out = torch.full((n, width), -1, dtype=torch.long, device=q.device)
    block_offsets = torch.arange(ratio, device=q.device)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        positions = torch.arange(start, stop, device=q.device)
        complete = (positions + 1) // ratio  # complete blocks visible to each query
        if blocks:
            scores = torch.relu(torch.einsum("chd,bd->chb", q[start:stop], block_keys)).sum(dim=1) / math.sqrt(d)
            scores.masked_fill_(torch.arange(blocks, device=q.device)[None, :] >= complete[:, None], float("-inf"))
            k = min(topk, blocks)
            chosen = scores.topk(k, dim=-1).indices  # [C, k]
            valid = torch.arange(k, device=q.device)[None, :] < complete[:, None].clamp(max=k)
            tokens = (chosen[..., None] * ratio + block_offsets).view(stop - start, k * ratio)
            tokens = torch.where(valid.repeat_interleave(ratio, dim=1), tokens, -1)
            out[start:stop, : k * ratio] = tokens
        # the incomplete tail block is always visible
        tail = complete[:, None] * ratio + block_offsets[: ratio - 1]
        tail = torch.where(tail <= positions[:, None], tail, -1)
        filled = min(topk, blocks) * ratio if blocks else 0
        out[start:stop, filled: filled + ratio - 1] = tail
    return out


def _attend_chunk(q, k, v, idx, scaling):
    """q [C, H, D]; k, v [N, Hkv, D]; idx [C, S] (-1 = none) -> [C, H, D].

    Each query is one SDPA batch row whose H // Hkv heads sharing a key/value head
    are laid out as that row's query positions (the same math as GQA). SDPA's
    `enable_gqa` would instead expand the gathered keys per head, and with a mask
    the math kernel also upcasts them: 24 GiB for a 512-query chunk."""
    c, h, d = q.shape
    hkv = k.shape[1]
    safe = idx.clamp(min=0)
    # index_select backpropagates through an atomic index_add; advanced indexing (k[safe]) goes through a
    # sort-based index_put, which took 66% of a QSA layer's forward + backward at 64K tokens
    flat = safe.reshape(-1)
    keys = k.index_select(0, flat).view(c, -1, hkv, d).transpose(1, 2)  # [C, Hkv, S, D]
    values = v.index_select(0, flat).view(c, -1, hkv, d).transpose(1, 2)
    mask = (idx >= 0)[:, None, None, :]
    out = F.scaled_dot_product_attention(q.reshape(c, hkv, h // hkv, d), keys, values, attn_mask=mask,
                                         scale=scaling)
    return out.reshape(c, h, d)


def qsa_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
    if past_key_values is not None or hidden_states.shape[0] != 1:
        return type(self)._hf_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values,
                                      **kwargs)
    chunk = getattr(self, "_qsa_chunk_size", 256)
    n = hidden_states.shape[1]
    cos, sin = (x[:, -n:, :] for x in position_embeddings)
    selected = select_tokens(self.indexer, hidden_states, cos, sin, chunk)

    hidden_shape = (n, -1, self.head_dim)
    query, gate = torch.chunk(self.q_proj(hidden_states[0]).view(n, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(n, -1)
    query = self.q_norm(query.view(hidden_shape))
    key = self.k_norm(self.k_proj(hidden_states[0]).view(hidden_shape))
    value = self.v_proj(hidden_states[0]).view(hidden_shape)
    query = _rope(query, cos[0], sin[0])
    key = _rope(key, cos[0], sin[0])

    pieces = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        args = (query[start:stop], key, value, selected[start:stop], self.scaling)
        if torch.is_grad_enabled():
            pieces.append(checkpoint(_attend_chunk, *args, use_reentrant=False))
        else:
            pieces.append(_attend_chunk(*args))
    output = torch.cat(pieces).reshape(n, -1) * torch.sigmoid(gate)
    return self.o_proj(output)[None], None


def patch_qsa_attention(chunk_size: int = 256) -> None:
    """Route qwen4_exp full-attention layers through qsa_forward. Safe to call again after
    reloading this module: transformers' forward is kept on the class, not here."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    cls = m.Qwen4ExpTextAttention
    if not hasattr(cls, "_hf_forward"):
        cls._hf_forward = cls.forward
    cls.forward = qsa_forward
    cls._qsa_chunk_size = chunk_size


def unpatch_qsa_attention() -> None:
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    cls = m.Qwen4ExpTextAttention
    if hasattr(cls, "_hf_forward"):
        cls.forward = cls._hf_forward
