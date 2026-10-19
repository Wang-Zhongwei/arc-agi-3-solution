"""Shared helpers for the QSA kernel tests."""

import torch

RATIO = 4


def random_blocks(n, topk, generator):
    """select_blocks' layout with random scores: top-k complete blocks (visible ones first), then the tail."""
    complete = (torch.arange(n) + 1) // RATIO
    blocks = n // RATIO
    k = min(topk, blocks)
    out = torch.full((n, k + 1), -1, dtype=torch.int32)
    if k:
        scores = torch.rand(n, blocks, generator=generator)
        scores[torch.arange(blocks)[None, :] >= complete[:, None]] = float("-inf")
        chosen = scores.topk(k, dim=-1).indices
        valid = torch.arange(k)[None, :] < complete[:, None].clamp(max=k)
        out[:, :k] = torch.where(valid, chosen, -1).to(torch.int32)
    out[:, k] = complete.to(torch.int32)
    return out
