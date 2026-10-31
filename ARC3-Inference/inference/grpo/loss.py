"""GRPO's per-token pieces: the sampler's logprob of a token, the clipped objective, stochastic BF16 rounding, Muon.

- `processed_logprobs`: log pi(token) under the distribution vLLM samples from with `--logprobs-mode
  processed_logprobs`: logits / temperature, then top-k, then top-p, renormalized over the kept tokens (vLLM's
  `apply_top_k_top_p`: ties at the k-th value are kept; top-p drops the tokens whose ascending cumulative probability
  is <= 1 - p, keeping at least the largest). The kept set has no gradient; the logprob does. A token outside the
  trainer's kept set gets -inf and `kept` False (it is then not trained). lm_head + softmax run per chunk under
  activation checkpointing, so [chunk, vocab] logits exist one chunk at a time (vocab 248,320).
- `grpo_objective`: r = exp(logp - behavior); per token min(r A, clip(r, 1 - eps_low, 1 + eps_high) A); tokens with r
  outside [ratio_min, ratio_max] or outside the kept set are dropped. Returns the sum (to be divided by the step's
  trained tokens) and statistics.
- `stochastic_bf16`: unbiased rounding of FP32 to BF16 (add uniform noise below BF16's last bit, then truncate).
- `Muon`: momentum orthogonalized by Newton-Schulz for 2-D parameters, the update scaled to AdamW's RMS
  (0.2 sqrt(max(m, n)), as Moonlight); an option of train_grpo, not the default.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.checkpoint import checkpoint


def _full_vocab_logprob(scaled: torch.Tensor, targets: torch.Tensor, top_p: float) -> torch.Tensor:
    """Without top-k: top-p over the whole vocabulary (sorted), as vLLM."""
    if top_p and top_p < 1.0:
        with torch.no_grad():
            sorted_logits, order = scaled.sort(dim=-1)  # ascending
            cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            drop_sorted = cumulative <= 1.0 - top_p
            drop_sorted[:, -1] = False
            drop = torch.zeros_like(drop_sorted).scatter(1, order, drop_sorted)
        scaled = scaled.masked_fill(drop, float("-inf"))
    return scaled.log_softmax(dim=-1).gather(1, targets[:, None]).squeeze(1)


def _kept_logprob(logits: torch.Tensor, targets: torch.Tensor, temperature: float, top_k: int,
                  top_p: float) -> torch.Tensor:
    """[c, V] logits, [c] target ids -> [c] processed logprobs (-inf outside the kept set).

    Top-p acts only on what top-k kept, so only those values are needed, not a sort of the vocabulary: ascending
    cumulative probability (vLLM's order) is the descending order's reversed cumsum. BF16 logits often tie at the k-th
    value (a third of the rows in a random test) and vLLM keeps every tie, so the width is the chunk's largest
    count of values >= the k-th, not k."""
    scaled = logits.float() / temperature
    vocab = scaled.shape[-1]
    if not top_k or top_k <= 0 or top_k >= vocab:
        return _full_vocab_logprob(scaled, targets, top_p)
    with torch.no_grad():
        kth = scaled.topk(top_k, dim=-1).values[:, -1:]
        width = int((scaled >= kth).sum(dim=-1).max())
    vals, idx = scaled.topk(width, dim=-1)  # descending
    keep = vals >= kth
    if top_p and top_p < 1.0:
        with torch.no_grad():
            probs = vals.detach().masked_fill(~keep, float("-inf")).softmax(dim=-1)
            ascending_cumulative = probs.flip(-1).cumsum(-1).flip(-1)
            drop = ascending_cumulative <= 1.0 - top_p
            drop[:, 0] = False  # the largest is always kept
            keep = keep & ~drop
    lse = vals.masked_fill(~keep, float("-inf")).logsumexp(dim=-1)
    target = scaled.gather(1, targets[:, None]).squeeze(1)
    # Membership by value, not index: when top-p's cut falls inside a group of tied values, which tied tokens vLLM
    # keeps depends on its sort's tie order, but the sampled one was kept, and the normalizer depends only on how
    # many are kept (all tied values are equal).
    with torch.no_grad():
        smallest_kept = vals.detach().masked_fill(~keep, float("inf")).min(dim=-1).values
        hit = target.detach() >= smallest_kept
    return torch.where(hit, target - lse, torch.full_like(lse, float("-inf")))


def processed_logprobs(lm_head, hidden: torch.Tensor, targets: torch.Tensor, temperature: float = 0.7,
                       top_k: int = 20, top_p: float = 0.95, chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """[K, d] hidden states and the [K] tokens they predicted -> ([K] FP32 logprobs, [K] bool kept)."""
    def piece(h, t):
        return _kept_logprob(lm_head(h), t, temperature, top_k, top_p)

    out = []
    for start in range(0, targets.numel(), chunk):
        h, t = hidden[start:start + chunk], targets[start:start + chunk]
        out.append(checkpoint(piece, h, t, use_reentrant=False) if h.requires_grad else piece(h, t))
    logp = torch.cat(out) if out else hidden.new_zeros(0, dtype=torch.float32)
    return logp, torch.isfinite(logp.detach())


def grpo_objective(logp: torch.Tensor, behavior: torch.Tensor, kept: torch.Tensor, advantage: float,
                   eps_low: float = 0.2, eps_high: float = 0.28, ratio_min: float = 0.5,
                   ratio_max: float = 2.0) -> tuple[torch.Tensor, dict[str, Any]]:
    """Sum over the trained tokens of the clipped objective's negative (a loss), and statistics."""
    behavior = behavior.to(logp.device, torch.float32)
    kept = kept & torch.isfinite(behavior)
    zero = torch.zeros_like(behavior)
    log_ratio = torch.where(kept, logp - torch.where(kept, behavior, zero), zero)  # dropped: ratio 1, then masked
    ratio = log_ratio.exp()
    use = kept & (ratio.detach() >= ratio_min) & (ratio.detach() <= ratio_max)
    a = float(advantage)
    unclipped = ratio * a
    clipped = ratio.clamp(1.0 - eps_low, 1.0 + eps_high) * a
    objective = torch.minimum(unclipped, clipped)
    loss = -(objective * use).sum()
    with torch.no_grad():
        n = int(kept.numel())
        lr = log_ratio[kept]
        stats = dict(tokens=n, used=int(use.sum()), outside_kept=int((~kept).sum()),
                     outside_ratio=int((kept & ~use).sum()),
                     clipped=int(((unclipped != clipped) & use).sum()),
                     log_ratio_mean=float(lr.mean()) if lr.numel() else 0.0,
                     log_ratio_abs_mean=float(lr.abs().mean()) if lr.numel() else 0.0,
                     kl_k3=float((lr.exp() - 1 - lr).mean()) if lr.numel() else 0.0,
                     logp_mean=float(logp[kept].mean()) if kept.any() else 0.0)
    return loss, stats


def stochastic_bf16(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Round FP32 to BF16 with P(up) = the dropped fraction: E[result] = x. Finite inputs only."""
    bits = x.float().contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, device=bits.device, generator=generator)
    return ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)


def _newton_schulz(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate U V^T of g (Keller Jordan's quintic iteration), in BF16 as the reference implementation."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return x.T if transpose else x


class Muon(torch.optim.Optimizer):
    """Muon for 2-D parameters (LoRA A and B), the update's RMS matched to AdamW's (Moonlight: 0.2 sqrt(max(m, n)))."""

    def __init__(self, params, lr: float = 1e-5, momentum: float = 0.95, nesterov: bool = True,
                 weight_decay: float = 0.0, ns_steps: int = 5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay,
                                      ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(f"Muon takes 2-D parameters, got {tuple(p.shape)}")
                state = self.state[p]
                buf = state.setdefault("momentum_buffer", torch.zeros_like(p, dtype=torch.float32))
                g = p.grad.float()
                buf.mul_(group["momentum"]).add_(g)
                update = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                o = _newton_schulz(update, group["ns_steps"]).float()
                o.mul_(0.2 * max(p.shape) ** 0.5)
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(o.to(p.dtype), alpha=-group["lr"])
        return None
