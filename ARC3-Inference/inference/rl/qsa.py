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
- attention on CUDA: the fused Triton kernel in qsa_kernel.py reads each query's
  selected blocks straight from K and V (forward and backward, FP32 dK/dV); a
  checkpoint's recompute reuses the forward's selection instead of rerunning the indexer;
- attention elsewhere (and with `fused=False`): each query attends to its own ~budget
  gathered keys/values; each chunk is checkpointed so the gathered keys/values are not
  kept for backward. The key/value gradients are scattered back with an atomic index_add
  per chunk, in BF16 unless `kv_grad_fp32` gathers from FP32 copies.

Scope: batch 1, no cache, causal mask only (training on packed single
sequences). Anything else falls back to the original forward.
"""

from __future__ import annotations

import math
import weakref

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from inference.rl import qsa_kernel

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


def top_blocks(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the k highest scores per row, ties going to the later block.

    After the relu many blocks score exactly 0, so whenever fewer than k blocks score above 0 the
    rest of the selection is a pick among ties, and topk's pick depends on the kernel and the batch
    shape: transformers' per-query loop, our 256- and 2,048-query chunks all chose different blocks,
    and attention over different keys gave the same 8K step LoRA gradients with cosine 0.66-0.83.
    A stable descending sort over the blocks in reverse order is deterministic and prefers the
    most recent blocks among ties."""
    order = torch.sort(scores.flip(-1), dim=-1, descending=True, stable=True).indices[..., :k]
    return scores.shape[-1] - 1 - order


@torch.no_grad()
def select_blocks(indexer, hidden_states, cos, sin, chunk_size: int) -> torch.Tensor:
    """Block ids [N, K + 1] (int32, -1 = none) each query attends to, K = min(topk, complete blocks).

    Columns 0..K-1 hold the query's top-K complete blocks; column K is its own incomplete
    block, whose tokens after the query are masked by the consumer. Blocks tied at the
    cut-off score (typically several at 0 after the relu) may be chosen differently than
    the reference's per-query topk; the selection is the same up to those ties."""
    n = hidden_states.shape[1]
    ratio, topk, d = indexer.compress_ratio, indexer.block_topk, indexer.index_head_dim
    q, block_keys = indexer_queries_and_keys(indexer, hidden_states, cos, sin)
    blocks = block_keys.shape[0]
    k = min(topk, blocks)
    out = torch.full((n, k + 1), -1, dtype=torch.int32, device=q.device)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        positions = torch.arange(start, stop, device=q.device)
        complete = (positions + 1) // ratio  # complete blocks visible to each query
        if k:
            scores = torch.relu(torch.einsum("chd,bd->chb", q[start:stop], block_keys)).sum(dim=1) / math.sqrt(d)
            scores.masked_fill_(torch.arange(blocks, device=q.device)[None, :] >= complete[:, None], float("-inf"))
            chosen = top_blocks(scores, k)  # [C, k], the visible blocks first
            valid = torch.arange(k, device=q.device)[None, :] < complete[:, None].clamp(max=k)
            out[start:stop, :k] = torch.where(valid, chosen, -1).to(torch.int32)
        out[start:stop, k] = complete.to(torch.int32)  # the incomplete tail block is always visible
    return out


def blocks_to_tokens(blocks: torch.Tensor, ratio: int, width: int) -> torch.Tensor:
    """Token indices [N, width] (-1 = padding) of select_blocks' selection: the chosen blocks' tokens,
    then the tail block's tokens up to the query."""
    n, k = blocks.shape[0], blocks.shape[1] - 1
    offsets = torch.arange(ratio, device=blocks.device)
    chosen = blocks[:, :k].long()
    tokens = (chosen[..., None] * ratio + offsets).view(n, k * ratio)
    tokens = torch.where(chosen.repeat_interleave(ratio, dim=1) >= 0, tokens, -1)
    tail = blocks[:, k:].long() * ratio + offsets[: ratio - 1]
    tail = torch.where(tail <= torch.arange(n, device=blocks.device)[:, None], tail, -1)
    out = torch.full((n, width), -1, dtype=torch.long, device=blocks.device)
    out[:, : k * ratio] = tokens
    out[:, k * ratio: k * ratio + ratio - 1] = tail
    return out


@torch.no_grad()
def select_tokens(indexer, hidden_states, cos, sin, chunk_size: int) -> torch.Tensor:
    """Indices [N, S] of the tokens each query attends to (-1 = padding), S = budget + ratio - 1."""
    ratio = indexer.compress_ratio
    blocks = select_blocks(indexer, hidden_states, cos, sin, chunk_size)
    return blocks_to_tokens(blocks, ratio, indexer.token_budget + ratio - 1)


def _selection(self, hidden_states, position_embeddings, cos, sin, chunk: int) -> torch.Tensor:
    """select_blocks, run once per forward: a checkpoint's recompute of this layer reuses the forward's
    selection instead of running the indexer again. The cache is keyed by the position embeddings
    (the same objects in the recompute) and a sample of the hidden states."""
    sample = hidden_states[0, :: max(1, hidden_states.shape[1] // 64), :16]
    cached = getattr(self, "_qsa_selection", None)
    self._qsa_selection = None
    if cached is not None and cached[0]() is position_embeddings[0] and cached[1].shape == sample.shape \
            and torch.equal(cached[1], sample):
        return cached[2]
    blocks = select_blocks(self.indexer, hidden_states, cos, sin, chunk)
    if torch.is_grad_enabled():
        self._qsa_selection = (weakref.ref(position_embeddings[0]), sample.detach().clone(), blocks)
    return blocks


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
    # k, v may be FP32 copies (kv_grad_fp32): the gathered rows are cast to q's dtype for SDPA
    flat = safe.reshape(-1)
    keys = k.index_select(0, flat).to(q.dtype).view(c, -1, hkv, d).transpose(1, 2)  # [C, Hkv, S, D]
    values = v.index_select(0, flat).to(q.dtype).view(c, -1, hkv, d).transpose(1, 2)
    mask = (idx >= 0)[:, None, None, :]
    out = F.scaled_dot_product_attention(q.reshape(c, hkv, h // hkv, d), keys, values, attn_mask=mask,
                                         scale=scaling)
    return out.reshape(c, h, d)


SELECTION_CHUNK = 2048


def qsa_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
    if past_key_values is not None or hidden_states.shape[0] != 1:
        return type(self)._hf_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values,
                                      **kwargs)
    chunk = getattr(self, "_qsa_chunk_size", 256)
    n = hidden_states.shape[1]
    cos, sin = (x[:, -n:, :] for x in position_embeddings)
    fused = getattr(self, "_qsa_fused", True) and qsa_kernel.available(hidden_states)

    hidden_shape = (n, -1, self.head_dim)
    query, gate = torch.chunk(self.q_proj(hidden_states[0]).view(n, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(n, -1)
    query = self.q_norm(query.view(hidden_shape))
    key = self.k_norm(self.k_proj(hidden_states[0]).view(hidden_shape))
    value = self.v_proj(hidden_states[0]).view(hidden_shape)
    query = _rope(query, cos[0], sin[0])
    key = _rope(key, cos[0], sin[0])
    if fused:
        # the selection's query chunks only bound the [chunk, blocks] score matrix (2,048 x 16K FP32 = 128 MB at
        # 64K); top-k is per query, so the size changes nothing but the number of launches
        blocks = _selection(self, hidden_states, position_embeddings, cos, sin, SELECTION_CHUNK)
        output = qsa_kernel.qsa_attention(query, key, value, blocks, self.scaling, self.indexer.compress_ratio)
        return self.o_proj(output.reshape(n, -1) * torch.sigmoid(gate))[None], None

    selected = select_tokens(self.indexer, hidden_states, cos, sin, SELECTION_CHUNK)  # the fused path's selection
    if getattr(self, "_qsa_kv_grad_fp32", False) and torch.is_grad_enabled():
        # gather from FP32 copies: every query's contribution to a key's gradient then adds up in
        # FP32, within a chunk's index_add and across chunks (BF16 sums were 1.7% off on one chunk)
        key, value = key.float(), value.float()

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


def patch_qsa_attention(chunk_size: int = 256, kv_grad_fp32: bool = False, fused: bool = True) -> None:
    """Route qwen4_exp full-attention layers through qsa_forward. Safe to call again after
    reloading this module: transformers' forward is kept on the class, not here.
    `fused` uses the Triton kernel (qsa_kernel) on CUDA; otherwise the chunked PyTorch path."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    cls = m.Qwen4ExpTextAttention
    if not hasattr(cls, "_hf_forward"):
        cls._hf_forward = cls.forward
    cls.forward = qsa_forward
    cls._qsa_chunk_size = chunk_size
    cls._qsa_kv_grad_fp32 = kv_grad_fp32
    cls._qsa_fused = fused


def unpatch_qsa_attention() -> None:
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    cls = m.Qwen4ExpTextAttention
    if hasattr(cls, "_hf_forward"):
        cls.forward = cls._hf_forward


def reference_indexer_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
    """transformers' indexer output (an [B, 1, N, N] selection mask) built from select_blocks.

    For end-to-end comparisons against the unpatched model: its per-query topk breaks the many ties
    at score 0 its own way, so the reference attends to other blocks than any implementation of ours.
    Everything downstream of the selection then runs transformers' code on the same selection."""
    if past_key_values is not None or hidden_states.shape[0] != 1:
        return type(self)._hf_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values)
    n = hidden_states.shape[1]
    cos, sin = (x[:, -n:, :] for x in position_embeddings)
    tokens = select_tokens(self, hidden_states, cos, sin, SELECTION_CHUNK)
    kv_length = attention_mask.shape[-1]
    mask = torch.zeros((1, n, kv_length + 1), device=hidden_states.device, dtype=torch.bool)
    mask = mask.scatter(-1, torch.where(tokens >= 0, tokens, kv_length)[None], True)[..., :kv_length].unsqueeze(1)
    if attention_mask.is_floating_point():
        mask = torch.where(mask, attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min)
    return mask


def patch_reference_indexer() -> None:
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    cls = m.Qwen4ExpTextQSAIndexer
    if not hasattr(cls, "_hf_forward"):
        cls._hf_forward = cls.forward
    cls.forward = reference_indexer_forward


def unpatch_reference_indexer() -> None:
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    cls = m.Qwen4ExpTextQSAIndexer
    if hasattr(cls, "_hf_forward"):
        cls.forward = cls._hf_forward
