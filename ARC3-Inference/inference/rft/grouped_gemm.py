"""Grouped GEMM for the routed experts in Triton, for GPUs without torch._grouped_mm.

torch._grouped_mm needs SM90+; on an A100 or an RTX PRO 6000 (SM120) the experts ran as a loop of
512 small matmuls per projection (6,163 matmul launches in one GDN layer at 14K tokens, 54% of its
GPU time). One launch computes every expert's rows here: rows are sorted by expert (offsets = the
cumulative row counts), each program takes one BLOCK_M tile of one expert's rows and one BLOCK_N
tile of the output, and finds its expert by a binary search over the cumulative tile counts (computed
on the GPU: no host sync). The same kernel computes dX = dY @ W^T by reading W with swapped strides;
the experts are frozen, so no weight gradient is computed.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs
    triton = None


def available(*tensors) -> bool:
    return triton is not None and all(t.is_cuda for t in tensors)


if triton is not None:
    @triton.autotune(configs=[triton.Config({"BLOCK_N": n, "BLOCK_K": k}, num_warps=w, num_stages=s)
                              for n, k, w, s in [(128, 64, 4, 3), (128, 64, 8, 3), (64, 64, 4, 4), (128, 32, 4, 4),
                                                 (256, 64, 8, 3), (64, 128, 4, 3)]],
                     key=["N", "K", "BLOCK_M"])
    @triton.jit
    def _grouped_kernel(X, W, Y, ENDS, TILE_CUM, E, N, K,
                        sx_m, sx_k, sw_e, sw_k, sw_n, sy_m, sy_n,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        tile = tl.program_id(0)
        lo = tl.zeros([], tl.int32)
        hi = tl.zeros([], tl.int32) + E
        while lo < hi:  # the first expert whose cumulative tile count exceeds this tile
            mid = (lo + hi) // 2
            if tl.load(TILE_CUM + mid) > tile:
                hi = mid
            else:
                lo = mid + 1
        if lo < E:
            e = lo
            first_tile = tl.load(TILE_CUM + e - 1, mask=e > 0, other=0)
            start = tl.load(ENDS + e - 1, mask=e > 0, other=0)
            rows = start + (tile - first_tile) * BLOCK_M + tl.arange(0, BLOCK_M)
            cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
            rmask = rows < tl.load(ENDS + e)
            cmask = cols < N
            acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
            w_base = W + e.to(tl.int64) * sw_e
            for k0 in range(0, K, BLOCK_K):
                ks = k0 + tl.arange(0, BLOCK_K)
                kmask = ks < K
                a = tl.load(X + rows[:, None].to(tl.int64) * sx_m + ks[None, :] * sx_k,
                            mask=rmask[:, None] & kmask[None, :], other=0.0)
                b = tl.load(w_base + ks[:, None] * sw_k + cols[None, :] * sw_n,
                            mask=kmask[:, None] & cmask[None, :], other=0.0)
                acc += tl.dot(a, b)
            tl.store(Y + rows[:, None].to(tl.int64) * sy_m + cols[None, :] * sy_n, acc.to(Y.dtype.element_ty),
                     mask=rmask[:, None] & cmask[None, :])


def grouped_matmul(x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor, transpose: bool = False):
    """Rows of x sorted by group, offsets = cumulative row counts per group [G]:
    out[rows of g] = x[rows of g] @ weight[g] ([G, K, N]), or @ weight[g].T with transpose=True."""
    g = weight.shape[0]
    k_dim, n_dim = (weight.shape[2], weight.shape[1]) if transpose else (weight.shape[1], weight.shape[2])
    sw_e, sw_k, sw_n = (weight.stride(0), weight.stride(2), weight.stride(1)) if transpose else weight.stride()
    m_total = x.shape[0]
    y = x.new_empty((m_total, n_dim))
    if m_total == 0:
        return y
    ends = offsets.to(torch.int32)
    counts = ends - torch.cat([ends.new_zeros(1), ends[:-1]])
    block_m = 64 if m_total // g < 256 else 128
    tile_cum = torch.cumsum((counts + block_m - 1) // block_m, 0, dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(m_total, block_m) + g, triton.cdiv(n_dim, meta["BLOCK_N"]))
    _grouped_kernel[grid](x, weight, y, ends, tile_cum, g, n_dim, k_dim,
                          x.stride(0), x.stride(1), sw_e, sw_k, sw_n, y.stride(0), y.stride(1), BLOCK_M=block_m)
    return y


class GroupedMM(torch.autograd.Function):
    """x @ weight[group] for frozen weights: dX = dY @ weight[group]^T with the same kernel."""

    @staticmethod
    def forward(ctx, x, weight, offsets):
        ctx.save_for_backward(weight, offsets)
        return grouped_matmul(x.contiguous(), weight, offsets)

    @staticmethod
    def backward(ctx, grad):
        weight, offsets = ctx.saved_tensors
        return grouped_matmul(grad.contiguous(), weight, offsets, transpose=True), None, None
