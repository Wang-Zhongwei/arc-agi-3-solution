"""Experimental SM120 BF16 verification GEMMs. No weight/activation quantization.

Candidate tile plans follow Lord's ARC3 overlay (arc3_sm120_gemm.py); they need
measurement in Daniel's engine. Rows are actual graph shapes, not mean batch
size. Unlisted shapes keep Pennyroyal's dispatch. Accumulate/reduce in FP32.
"""

import torch
import triton
import triton.language as tl

ROWS = (36, 40, 44, 48, 52, 56, 60, 64)
# (N, K): (tile M, tile N, tile K, splits)
PLANS = {
    (336, 10240): (32, 32, 64, 16),
    (10240, 320): (64, 64, 64, 1),
    (512, 2560): (32, 32, 64, 16),
    (1280, 2560): (32, 64, 64, 8),
    (2560, 640): (32, 32, 64, 1),
    (640, 2560): (32, 32, 128, 8),
    (96, 2560): (32, 32, 64, 16),
}


def eligible(m, n, k):
    return m in ROWS and (n, k) in PLANS


@triton.jit
def _partial(X, W, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, S: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    inner = split * BK + tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for chunk in range(tl.cdiv(K, BK * S)):
        kk = inner + chunk * BK * S
        x = tl.load(X + rows[:, None] * K + kk[None, :],
                    (rows[:, None] < M) & (kk[None, :] < K), 0)
        w = tl.load(W + cols[None, :] * K + kk[:, None],
                    (cols[None, :] < N) & (kk[:, None] < K), 0)
        acc = tl.dot(x, w, acc)
    tl.store(P + split * M * N + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _reduce(P, Y, MN: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    for split in range(S):
        acc += tl.load(P + split * MN + idx, idx < MN, 0)
    tl.store(Y + idx, acc, idx < MN)


def gemm(x, weight):
    # Existing dispatch handles SM120/deterministic/bias selection. Preserve
    # torch's behavior for unexpected layout or dtype instead of copying it.
    if (not x.is_cuda or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or not x.is_contiguous() or not weight.is_contiguous()):
        return torch.nn.functional.linear(x, weight)
    matrix = x.view(-1, x.shape[-1])
    m, k = matrix.shape
    n = weight.shape[0]
    if not eligible(m, n, k):
        return torch.nn.functional.linear(x, weight)
    bm, bn, bk, splits = PLANS[n, k]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    partials = out if splits == 1 else torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
    _partial[(triton.cdiv(m, bm), triton.cdiv(n, bn), splits)](
        matrix, weight, partials, m, n, k, bm, bn, bk, splits, num_warps=4)
    if splits > 1:
        _reduce[(triton.cdiv(m * n, 256),)](partials, out, m * n, splits, 256)
    return out.view(*x.shape[:-1], n)
