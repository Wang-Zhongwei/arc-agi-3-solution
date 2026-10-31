"""Fine-tune the DFlash2 block drafter (incoai/Qwen3.8-Flash-Next-DFlash2) on the frozen target's own replies
(mtp_data samples), for k = 7 draft tokens: block 8, the block size the checkpoint was trained with.

What is served (vLLM qwen3_dflash2.py + the DFlash2 speculator), rebuilt here in torch:
- context: the target's completed layers 5, 15, 25, 35, 45 (zero-based), each the mean of the four
  hyper-connection streams, concatenated to [T, 5 x 2560]; c = hidden_norm(fc(features)). Every draft layer's key
  and value of position j are k_norm(k_proj(c_j)) with RoPE at j and v_proj(c_j).
- block: once the target has accepted through position p, the query rows are [x[p+1], mask x 7] at positions
  p+1..p+8. Each layer: input_layernorm, two-tap grouped conv, attention over the context keys j <= p inside the
  2048-token sliding window plus the block's own 8 keys (non-causal), conv, post_attention_layernorm, conv, MLP,
  conv; then the final norm. Row s (1..7) drafts x[p+1+s].
- selector: each row's top 16 tokens under the lm_head (the target's) are its candidates, with unary logits U;
  S_s(a, b) = U_s(b) + <A(a) * H(h_s), B(b)>, a the previous token (x[p+1] for s = 1). The walk draws
  b ~ softmax(S_s(a, .) / T) at the request's temperature and rejection sampling against the target checks it.

Per sample: one forward of the frozen target, then `--blocks` anchors p drawn among the positions whose first draft
x[p+2] is a trained reply token, all in one pass (a block sees only its own context and rows, so blocks are as
independent as when served). Targets: the harness's sampling distribution of the target at each drafted position
(T 0.7, top-k 20, top-p 0.95). Loss per drafted row s, weighted by --position-weights:
  unary  -sum_x p(x) log softmax(U / T)(x) over the vocabulary (keeps the right tokens among the 16 candidates)
  path   -sum_b p(b) log softmax(S_s(x[p+s], .) / T)(b) over the candidates, the true previous token as a
Acceptance as served: a_s = sum_b min(p(b), q(b)), q the path distribution after the true previous token. Rate k
is the mean of a_1 ... a_k over blocks whose seven drafted tokens are all trained (vLLM's "Per-position acceptance
rate"); acceptance length 1 + the sum of the rates.

Trained: every drafter tensor (fc, norms, layers with their convs, the selector and its codebooks). The target's
embedding and lm_head, which the drafter shares when served, stay frozen.

    python -m inference.rft.dflash_train --model W4A16 --draft DFLASH2_DIR --data mtp_v1 --out OUT [--eval-only]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from inference.rft import layer_patches, mtp_train, train_rft
from inference.rft.qsa import _ThroughE4M3, set_kv_fp8


# ---------------------------------------------------------------------------
# The drafter
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Neox-style rotary embedding over the whole head: x [n, heads, d], positions [n]."""
    d = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    freqs = positions.float()[:, None] * inv
    cos, sin = freqs.cos()[:, None], freqs.sin()[:, None]
    x1, x2 = x.float().chunk(2, -1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(x.dtype)


def kv_fp8(x: torch.Tensor, on: bool) -> torch.Tensor:
    """vLLM's fp8_e4m3 KV cache (scale 1.0): the stored value, gradient straight through."""
    return _ThroughE4M3.apply(x, 1.0) if on else x


class GroupedConv(nn.Module):
    """DFlash2's dynamic two-tap convolution inside each block: tap t of row r reads row r - t of the same block,
    with coefficient base_kernel[side, t] + a per-group delta from kernel_projection of the conv's input. `prepare`
    convolves with side 0 and returns side 1's deltas for `convolve(..., 1)` after the sublayer."""

    def __init__(self, hidden: int, taps: int, group: int):
        super().__init__()
        self.taps, self.group, self.groups = taps, group, hidden // group
        self.base_kernel = nn.Parameter(torch.zeros(2, taps, hidden))
        self.kernel_projection = nn.Linear(hidden, 2 * taps * self.groups, bias=False)

    def prepare(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [n, block, H] -> (side-0 conv of x, side-1 deltas [n, block, taps, groups])."""
        deltas = self.kernel_projection(x).unflatten(-1, (2, self.taps, self.groups))
        return self.convolve(x, deltas[:, :, 0], 0), deltas[:, :, 1]

    def convolve(self, x: torch.Tensor, deltas: torch.Tensor, side: int) -> torch.Tensor:
        coefficients = self.base_kernel[side].float().unflatten(-1, (self.groups, self.group)) + deltas.float()[..., None]
        xs = x.float().unflatten(-1, (self.groups, self.group))
        out = coefficients[:, :, 0] * xs
        for tap in range(1, self.taps):
            out = out + coefficients[:, :, tap] * F.pad(xs[:, :-tap], (0, 0, 0, 0, tap, 0))
        return out.flatten(-2).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, eps: float, theta: float):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim, self.theta = heads, kv_heads, head_dim, theta
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)

    def kv(self, x: torch.Tensor, positions: torch.Tensor, fp8: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys and values as the cache stores them: [m, kv_heads, d] each."""
        k = rope(self.k_norm(self.k_proj(x).unflatten(-1, (self.kv_heads, self.head_dim))), positions, self.theta)
        v = self.v_proj(x).unflatten(-1, (self.kv_heads, self.head_dim))
        return kv_fp8(k, fp8), kv_fp8(v, fp8)

    def forward(self, x, positions, context_kv, attend, fp8: bool) -> torch.Tensor:
        """x [N, H] query rows; context_kv the context's (k, v); attend(q, k, v) -> [N, heads, d]."""
        q = rope(self.q_norm(self.q_proj(x).unflatten(-1, (self.heads, self.head_dim))), positions, self.theta)
        k, v = self.kv(x, positions, fp8)
        out = attend(q, torch.cat([context_kv[0], k]), torch.cat([context_kv[1], v]))
        return self.o_proj(out.flatten(-2))


class MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden, eps, draft = config["hidden_size"], config["rms_norm_eps"], config["dflash_config"]
        self.input_layernorm = RMSNorm(hidden, eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.self_attn = Attention(hidden, config["num_attention_heads"], config["num_key_value_heads"],
                                   config["head_dim"], eps, config["rope_parameters"]["rope_theta"])
        self.mlp = MLP(hidden, config["intermediate_size"])
        self.attention_conv = GroupedConv(hidden, draft["conv_kernel_size"], draft["conv_group_size"])
        self.mlp_conv = GroupedConv(hidden, draft["conv_kernel_size"], draft["conv_group_size"])

    def forward(self, x, residual, context, context_positions, positions, attend, fp8: bool):
        """x [n, block, H]; vLLM's fused add-and-norm order (residual = x + residual, then the norm)."""
        residual = x if residual is None else x + residual
        h, deltas = self.attention_conv.prepare(self.input_layernorm(residual))
        context_kv = self.self_attn.kv(context, context_positions, fp8)
        h = self.self_attn(h.flatten(0, 1), positions, context_kv, attend, fp8).view_as(h)
        h = self.attention_conv.convolve(h, deltas, 1)
        residual = h + residual
        h, deltas = self.mlp_conv.prepare(self.post_attention_layernorm(residual))
        return self.mlp_conv.convolve(self.mlp(h), deltas, 1), residual


class CandidateSelector(nn.Module):
    def __init__(self, hidden: int, vocab: int, rank: int):
        super().__init__()
        self.hidden_projection = nn.Linear(hidden, rank, bias=False)
        self.predecessor_codebook = nn.Parameter(torch.zeros(vocab, rank))
        self.successor_codebook = nn.Parameter(torch.zeros(vocab, rank))

    def pair_scores(self, candidates: torch.Tensor, hidden: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        """<A(previous) * H(h), B(b)> for each candidate b: candidates [m, K], hidden [m, H], previous [m] -> [m, K]."""
        gated = self.predecessor_codebook[previous] * self.hidden_projection(hidden)
        return torch.einsum("mr,mkr->mk", gated, self.successor_codebook[candidates]).float()


class DFlash2Draft(nn.Module):
    """The drafter; parameter names are the checkpoint's."""

    def __init__(self, config: dict[str, Any], block: int):
        super().__init__()
        draft = config["dflash_config"]
        hidden = config["hidden_size"]
        self.block, self.window = block, config["sliding_window"]
        self.mask_token_id, self.top_k = draft["mask_token_id"], draft["selector_top_k"]
        self.target_layer_ids = list(draft["target_layer_ids"])
        self.output_multiplier = float(draft.get("output_multiplier", 1.0))
        self.softcap = float(draft.get("final_logit_softcapping") or 0.0)
        self.fc = nn.Linear(hidden * len(self.target_layer_ids), hidden, bias=False)
        self.hidden_norm = RMSNorm(hidden, config["rms_norm_eps"])
        self.layers = nn.ModuleList([Layer(config) for _ in range(config["num_hidden_layers"])])
        self.norm = RMSNorm(hidden, config["rms_norm_eps"])
        self.candidate_selector = CandidateSelector(hidden, config["vocab_size"], draft["selector_rank"])

    def forward(self, embeds: torch.Tensor, anchors: torch.Tensor, features: torch.Tensor, attention: str = "flex",
                fp8: bool = False) -> torch.Tensor:
        """embeds [n, block, H] (x[p+1], then mask embeddings), anchors [n] (p), features [T, 5H] -> [n, block, H]."""
        n, block, _ = embeds.shape
        context = self.hidden_norm(self.fc(features))
        context_positions = torch.arange(features.shape[0], device=features.device)
        positions = (anchors[:, None] + 1 + torch.arange(block, device=anchors.device)).flatten()
        attend = block_attention(anchors, features.shape[0], block, self.window, attention)
        x, residual = embeds, None
        for layer in self.layers:
            x, residual = layer(x, residual, context, context_positions, positions, attend, fp8)
        return self.norm(x + residual)

    def logits(self, lm_head: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        """The candidate logits: the shared lm_head, times output_multiplier, soft-capped when configured."""
        logits = lm_head(hidden).float() * self.output_multiplier
        return self.softcap * torch.tanh(logits / self.softcap) if self.softcap else logits


def block_allowed(q_idx, kv_idx, anchors: torch.Tensor, context_len: int, block: int, window: int):
    """Whether query row q_idx (row q_idx % block of block q_idx // block, anchor p) reads key kv_idx of
    [context 0..T-1, query rows]: a context key j <= p with position - j < window (the sliding window), or a row of
    its own block (non-causal)."""
    blk = q_idx // block
    p = anchors[blk]
    position = p + 1 + q_idx % block
    in_context = (kv_idx < context_len) & (kv_idx <= p) & (position - kv_idx < window)
    own = (kv_idx >= context_len) & ((kv_idx - context_len) // block == blk)
    return in_context | own


def block_attention(anchors: torch.Tensor, context_len: int, block: int, window: int, impl: str):
    """attend(q [N, h, d], k [T + N, kv_h, d], v) under block_allowed; `flex` on GPU, `dense` (an explicit mask with
    SDPA) for tests."""
    num_rows = anchors.shape[0] * block

    def allowed(q_idx, kv_idx):
        return block_allowed(q_idx, kv_idx, anchors, context_len, block, window)

    if impl == "dense":
        q_idx = torch.arange(num_rows, device=anchors.device)[:, None]
        kv_idx = torch.arange(context_len + num_rows, device=anchors.device)[None]
        mask = allowed(q_idx, kv_idx)

        def attend(q, k, v):
            out = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                                 v.transpose(0, 1)[None], attn_mask=mask, enable_gqa=True)
            return out[0].transpose(0, 1)
        return attend

    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    # compiled, so the mask is evaluated per 128 x 128 tile instead of materialized (16K x 147K rows)
    mask = _compiled("mask", create_block_mask)(lambda b, h, q_idx, kv_idx: allowed(q_idx, kv_idx), None, None,
                                               num_rows, context_len + num_rows, device=anchors.device)
    flex = _compiled("flex", flex_attention)

    def attend(q, k, v):
        out = flex(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], block_mask=mask,
                   enable_gqa=True)
        return out[0].transpose(0, 1)
    return attend


_COMPILED = {}


def _compiled(name: str, fn):
    if name not in _COMPILED:
        _COMPILED[name] = torch.compile(fn, dynamic=True)
    return _COMPILED[name]


def load_draft(draft_dir: Path, block: int) -> DFlash2Draft:
    """The drafter from a DFlash2 checkpoint directory (config.json, model.safetensors), in BF16."""
    from safetensors.torch import load_file

    config = json.loads((draft_dir / "config.json").read_text())
    model = DFlash2Draft(config, block)
    model.load_state_dict(load_file(draft_dir / "model.safetensors"), strict=True)
    return model.to(torch.bfloat16)


def export_draft(model: DFlash2Draft, draft_dir: Path, out: Path) -> None:
    """A draft directory for vLLM: the checkpoint's config.json and the trained tensors under their own names."""
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    tensors = {name: p.detach().to(torch.bfloat16).contiguous().cpu() for name, p in model.state_dict().items()}
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    shutil.copy2(draft_dir / "config.json", out / "config.json")


# ---------------------------------------------------------------------------
# Anchors, loss and acceptance
# ---------------------------------------------------------------------------

def draw_anchors(labels: torch.Tensor, count: int, block: int, generator: torch.Generator) -> torch.Tensor:
    """Up to `count` distinct anchors p, sorted, with x[p+2] trained and the block p+1..p+block inside the sample."""
    last = labels.shape[0] - block - 1  # p + block <= T - 1
    if last < 0:
        return torch.zeros(0, dtype=torch.long)
    valid = torch.nonzero(labels[2:last + 3] != -100).squeeze(1)
    if valid.numel() > count:
        valid = valid[torch.randperm(valid.numel(), generator=generator)[:count]]
    return valid.sort().values


def candidate_probs(ids: torch.Tensor, probs: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """The target's probability of each candidate: ids/probs [m, k] (its sampling support), candidates [m, K]."""
    return ((candidates[:, :, None] == ids[:, None, :]) * probs[:, None, :]).sum(-1)


def row_terms(draft: DFlash2Draft, lm_head, hidden, previous, ids, probs, temperature: float, chunk: int = 2048):
    """Per drafted row: (unary loss, path loss) with gradient, and acceptance a = sum_b min(p(b), q(b)) and the greedy
    path pick's p without. hidden [m, H], previous [m] (the token before the row's), ids/probs [m, k] the target's."""
    def piece(h, prev, i, p, candidates):
        logits = draft.logits(lm_head, h)
        unary = -(p * (logits / temperature).log_softmax(-1).gather(-1, i)).sum(-1)
        scores = logits.gather(-1, candidates) + draft.candidate_selector.pair_scores(candidates, h, prev)
        log_q = (scores / temperature).log_softmax(-1)
        return unary, -(candidate_probs(i, p, candidates) * log_q).sum(-1), log_q

    unary, path, accept, greedy = [], [], [], []
    for start in range(0, hidden.shape[0], chunk):
        sl = slice(start, start + chunk)
        h = hidden[sl]
        with torch.no_grad():
            candidates = draft.logits(lm_head, h.detach()).topk(draft.top_k, dim=-1).indices
        args = (h, previous[sl], ids[sl], probs[sl], candidates)
        u, pa, log_q = checkpoint(piece, *args, use_reentrant=False) if torch.is_grad_enabled() else piece(*args)
        unary.append(u)
        path.append(pa)
        with torch.no_grad():
            pc = candidate_probs(ids[sl], probs[sl], candidates)
            accept.append(torch.minimum(pc, log_q.detach().exp()).sum(-1))
            greedy.append(pc.gather(-1, log_q.argmax(-1, keepdim=True)).squeeze(-1))
    return torch.cat(unary), torch.cat(path), torch.cat(accept), torch.cat(greedy)


def chain_rates(accept: torch.Tensor, trained: torch.Tensor) -> dict[str, Any]:
    """accept/trained [n, steps]: per-position rates over the fully trained blocks (cumulative products)."""
    full = trained.all(-1)
    if not bool(full.any()):
        return dict(anchors=0)
    rates = accept[full].float().cumprod(-1).mean(0).tolist()
    return dict(anchors=int(full.sum()), rates=rates, length=1 + sum(rates))


# ---------------------------------------------------------------------------
# One sample
# ---------------------------------------------------------------------------

class Target:
    """The frozen target, with hooks that keep the drafter's features: each tapped layer's output, the mean of its
    hyper-connection streams."""

    def __init__(self, model, layers, layer_ids: list[int], streams: int):
        self.model, self.layer_ids, self.captured = model, layer_ids, {}
        for i in layer_ids:
            layers[i].register_forward_hook(lambda module, args, out, i=i: self.keep(i, out, streams))

    def keep(self, i: int, out, streams: int) -> None:
        h = out[0] if isinstance(out, tuple) else out
        self.captured[i] = h[0].unflatten(-1, (streams, -1)).mean(-2)

    @torch.no_grad()
    def __call__(self, inputs: dict[str, torch.Tensor], ngram) -> tuple[torch.Tensor, torch.Tensor]:
        layer_patches.provide_ngram(self.model, ngram)
        last = self.model.model(**inputs, use_cache=False).last_hidden_state
        layer_patches.provide_ngram(self.model, None)
        features = torch.cat([self.captured.pop(i) for i in self.layer_ids], -1)
        return features, last


def run_sample(target: Target, draft: DFlash2Draft, lm_head, embed, item: dict[str, Any], device, args,
               train: bool, generator: torch.Generator, loss_scale: float = 1.0) -> dict[str, Any]:
    """Target forward, then the drafter on --blocks blocks; loss and acceptance on the trained drafted rows."""
    labels = item["labels"][0]
    input_ids = item["input_ids"].to(device)
    inputs = {k: v.to(device) for k, v in item.items() if k != "labels" and not k.startswith("_")}
    started = time.perf_counter()
    set_kv_fp8(args.fp8_kv in ("target", "both"))  # vLLM's fp8_e4m3 target caches, simulated (qsa.py)
    features, last = target(inputs, item.get("_ngram"))
    set_kv_fp8(False)
    target_s = time.perf_counter() - started
    block, steps = draft.block, draft.block - 1
    anchors = draw_anchors(labels, args.blocks, block, generator).to(device)
    if anchors.numel() == 0:
        return dict(tokens=int(input_ids.shape[1]), rows=0, blocks=0, losses=[None] * steps, chain=dict(anchors=0),
                    target_s=round(target_s, 2), total_s=round(time.perf_counter() - started, 2))
    offsets = torch.arange(1, block, device=device)
    drafted = anchors[:, None] + 1 + offsets  # [n, steps]: the positions rows 1..steps draft
    trained = labels.to(device)[drafted] != -100
    ids, probs = mtp_train.target_distribution(lm_head, last[0, (drafted - 1)[trained]], args.temperature,
                                               args.top_k, args.top_p)
    tokens = input_ids[0]
    block_ids = torch.full((anchors.numel(), block), draft.mask_token_id, device=device)
    block_ids[:, 0] = tokens[anchors + 1]
    weights = torch.tensor(args.position_weights, device=device).expand_as(trained)[trained]
    with torch.set_grad_enabled(train):
        hidden = draft(embed(block_ids), anchors, features, attention=args.attention,
                       fp8=args.fp8_kv in ("draft", "both"))
        rows = hidden[:, 1:][trained]
        unary, path, accept, greedy = row_terms(draft, lm_head, rows, tokens[drafted - 1][trained], ids, probs,
                                                args.temperature)
        if train:
            loss = (weights * (unary + args.path_weight * path)).sum() * loss_scale / max(1, rows.shape[0])
            loss.backward()
    position = torch.nonzero(trained)[:, 1]
    losses = [round(float((unary + path).detach()[position == s].mean()), 5) if bool((position == s).any()) else None
              for s in range(steps)]
    full_accept = torch.zeros(trained.shape, device=device)
    full_accept[trained] = accept
    full_greedy = torch.zeros(trained.shape, device=device)
    full_greedy[trained] = greedy
    chain = chain_rates(full_accept, trained)
    greedy_chain = chain_rates(full_greedy, trained)
    if greedy_chain.get("anchors"):
        chain["greedy_length"] = greedy_chain["length"]
    return dict(tokens=int(input_ids.shape[1]), rows=int(rows.shape[0]), blocks=int(anchors.numel()), losses=losses,
                unary=round(float(unary.detach().mean()), 5), path=round(float(path.detach().mean()), 5),
                chain=chain, target_s=round(target_s, 2), total_s=round(time.perf_counter() - started, 2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="the target checkpoint (Intel W4A16 AutoRound)")
    parser.add_argument("--draft", type=Path, required=True, help="the DFlash2 checkpoint (config.json, model.safetensors)")
    parser.add_argument("--data", type=Path, required=True, help="mtp_data output: train.jsonl, eval.jsonl, images/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draft-tokens", type=int, default=7, help="k; the block is k + 1 rows (served: 7)")
    parser.add_argument("--blocks", type=int, default=2048, help="anchors drawn per sample")
    parser.add_argument("--position-weights", default="1,0.85,0.72,0.61,0.52,0.44,0.38",
                        help="loss weight of drafted rows 1..k (0.85^(s-1): later rows count when the earlier ones pass)")
    parser.add_argument("--path-weight", type=float, default=1.0, help="weight of the selector's path loss")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--attention", choices=("flex", "dense"), default="flex")
    parser.add_argument("--fp8-kv", choices=("none", "target", "draft", "both"), default="both",
                        help="round K/V to e4m3 as the served fp8_e4m3 caches do: the target's, the drafter's, or both")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--accumulate", type=int, default=1, help="samples per optimizer step")
    parser.add_argument("--train-samples", type=int, default=1400,
                        help="samples drawn from train.jsonl, round-robin over games (one pass, no repeats)")
    parser.add_argument("--eval-samples", type=int, default=48)
    parser.add_argument("--eval-every", type=int, default=100, help="optimizer steps")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--best-margin", type=float, default=0.002,
                        help="eval acceptance length a checkpoint must beat the best by to replace it")
    parser.add_argument("--max-tokens", type=int, default=131072)
    parser.add_argument("--max-hours", type=float, default=20.0)
    parser.add_argument("--token-chunk", type=int, default=8192)
    parser.add_argument("--no-qsa-kernel", dest="qsa_kernel", action="store_false")
    parser.add_argument("--no-compile-layers", dest="compile_layers", action="store_false",
                        help="without compile_hc, compile_moe, fast_conv, fast_gdn (train_rft's switch)")
    parser.add_argument("--no-fast-gdn", dest="fast_gdn", action="store_false")
    parser.add_argument("--eval-only", action="store_true", help="measure the drafter as loaded and stop")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", help="also log to this wandb project (key: WANDB_API_KEY; offline without it)")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    args = parser.parse_args(argv)
    args.position_weights = [float(w) for w in args.position_weights.split(",")]
    if len(args.position_weights) != args.draft_tokens:
        raise SystemExit("--position-weights needs one weight per draft token")
    args.out.mkdir(parents=True, exist_ok=True)
    log_file = open(args.out / "train_log.jsonl", "a")
    run = mtp_train.start_wandb(args)

    def log(**record):
        record["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()
        print(json.dumps(record), flush=True)
        if run is not None:
            try:
                metrics = mtp_train.wandb_metrics(record)
                if metrics:
                    run.log(metrics)
            except Exception as exc:  # noqa: BLE001 - monitoring must never stop training
                print(f"wandb log failed: {exc}", flush=True)

    started = time.time()
    model, processor, info = mtp_train.load_target(args)
    device = model.get_input_embeddings().weight.device
    language = model.model.language_model
    draft = load_draft(args.draft, args.draft_tokens + 1).to(device)
    draft.train()
    text_config = model.config.get_text_config()
    target = Target(model, language.layers, draft.target_layer_ids, text_config.hc_count)
    params = [(n, p) for n, p in draft.named_parameters()]
    log(event="loaded", load_s=round(time.time() - started, 1), trainable=sum(p.numel() for _, p in params),
        target=info["experts_check"], args={k: str(v) for k, v in vars(args).items()})

    def prepare(sample):
        item = train_rft.tokenize_sample(processor, sample, args.data)
        if item["input_ids"].shape[1] > args.max_tokens:
            return None
        item["_ngram"] = layer_patches.precompute_ngram(model, item["input_ids"])
        return item

    lm_head, embed = model.lm_head, language.embed_tokens
    eval_samples = train_rft.load_samples(args.data / "eval.jsonl")
    random.Random(args.seed).shuffle(eval_samples)
    eval_samples = eval_samples[:args.eval_samples]
    steps = args.draft_tokens

    def evaluate(tag):
        totals, chains = {"rows": 0, "loss": [0.0] * steps, "counted": [0] * steps}, []
        for i, sample in enumerate(eval_samples):
            try:
                item = prepare(sample)
                if item is None:
                    continue
                generator = torch.Generator().manual_seed(args.seed * 100003 + i)  # the same blocks at every eval
                r = run_sample(target, draft, lm_head, embed, item, device, args, train=False, generator=generator)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            totals["rows"] += r["rows"]
            for k, x in enumerate(r["losses"]):
                if x is not None:
                    totals["loss"][k] += x
                    totals["counted"][k] += 1
            if r["chain"].get("anchors"):
                chains.append(r["chain"])
        anchors = sum(c["anchors"] for c in chains)
        rates = [sum(c["rates"][k] * c["anchors"] for c in chains) / max(1, anchors) for k in range(steps)]
        greedy = sum(c.get("greedy_length", 0) * c["anchors"] for c in chains) / max(1, anchors)
        record = dict(tag=tag, samples=len(chains), rows=totals["rows"], anchors=anchors,
                      loss=[round(x / max(1, n), 5) for x, n in zip(totals["loss"], totals["counted"])],
                      rates=[round(x, 4) for x in rates], length=round(1 + sum(rates), 4),
                      greedy_length=round(greedy, 4))
        log(event="eval", step=state["step"], **record)
        return record

    state = dict(position=0, step=0, best_length=None, best_step=None)  # evaluate() logs state["step"]
    if args.eval_only:
        evaluate("as loaded")
        return 0

    masters = [p.detach().float().clone().requires_grad_() for _, p in params]
    optimizer = torch.optim.AdamW(masters, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    train_samples = train_rft.load_samples(args.data / "train.jsonl")
    order = mtp_train.game_round_robin(train_samples, args.train_samples, args.seed)
    total_steps = math.ceil(len(order) / args.accumulate)
    latest = args.out / "latest"
    if args.resume and (latest / "state.json").exists():
        from safetensors.torch import load_file
        weights = load_file(latest / "draft.safetensors")
        with torch.no_grad():
            for (name, p), m in zip(params, masters):
                m.copy_(weights[name].float())
                p.copy_(m)
        optimizer.load_state_dict(torch.load(latest / "optimizer.pt", map_location=device))
        state = json.loads((latest / "state.json").read_text())
        log(event="resumed", **state)

    def save(name):
        from safetensors.torch import save_file
        target_dir = args.out / name
        target_dir.mkdir(parents=True, exist_ok=True)
        save_file({n: p.detach().cpu().contiguous() for n, p in params}, target_dir / "draft.safetensors")
        torch.save(optimizer.state_dict(), target_dir / "optimizer.pt")
        (target_dir / "state.json").write_text(json.dumps(state))
        log(event="saved", name=name, step=state["step"])

    def consider(record):
        """Keep the drafter as best/ when its eval acceptance length beats the best so far by best_margin."""
        if state["best_length"] is None or record["length"] > state["best_length"] + args.best_margin:
            state["best_length"], state["best_step"] = record["length"], state["step"]
            save("best")

    if state["step"] == 0:
        consider(evaluate("step 0"))  # the drafter as loaded: training must beat it to be exported
    generator = torch.Generator().manual_seed(args.seed + state["position"])
    prefetch = train_rft.Prefetch(lambda i: prepare(train_samples[order[i]]))
    if state["position"] < len(order):
        prefetch.start(state["position"])
    pending = 0
    while state["position"] < len(order):
        if time.time() - started > args.max_hours * 3600:
            log(event="time limit", **state)
            break
        try:
            item = prefetch.get()
        except RuntimeError as exc:
            item = None
            log(event="skip", position=state["position"], error=str(exc)[-300:])
        index = state["position"]
        state["position"] += 1
        if state["position"] < len(order):
            prefetch.start(state["position"])
        if item is None:
            continue
        try:
            r = run_sample(target, draft, lm_head, embed, item, device, args, train=True, generator=generator,
                           loss_scale=1.0 / args.accumulate)
        except torch.OutOfMemoryError:
            for _, p in params:
                p.grad = None
            torch.cuda.empty_cache()
            log(event="oom", position=index, tokens=int(item["input_ids"].shape[1]))
            continue
        pending += 1
        sample = train_samples[order[index]]
        log(event="sample", position=index, id=sample["id"], **r)
        if pending < args.accumulate and state["position"] < len(order):
            continue
        for (_, p), m in zip(params, masters):
            m.grad = p.grad.float() if p.grad is not None else torch.zeros_like(m)
        norm = torch.nn.utils.clip_grad_norm_(masters, args.grad_clip)
        lr = args.lr * train_rft.lr_at(state["step"], total_steps, args.warmup, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for (_, p), m in zip(params, masters):
                p.copy_(m)
                p.grad = None
        pending = 0
        state["step"] += 1
        log(event="step", step=state["step"], lr=lr, grad_norm=round(float(norm), 4))
        if state["step"] % args.eval_every == 0:
            consider(evaluate(f"step {state['step']}"))
        if state["step"] % args.save_every == 0:
            save("latest")
    save("latest")
    if state["step"] % args.eval_every:
        consider(evaluate(f"final step {state['step']}"))
    from safetensors.torch import load_file
    best = load_file(args.out / "best" / "draft.safetensors")  # export the best drafter, not the last
    with torch.no_grad():
        for name, p in params:
            p.copy_(best[name].to(p.device))
    export_draft(draft, args.draft, args.out / "draft")
    log(event="exported", path=str(args.out / "draft"), best_step=state["best_step"], best_length=state["best_length"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - logged for the job's output
        traceback.print_exc()
        sys.exit(1)
