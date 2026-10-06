# SPDX-License-Identifier: Apache-2.0
"""SM120 low-M GEMM for FP8 weights with one scale per output row.

Lord's mixed checkpoint stores attention and linear-attention projections as
compressed-tensors FP8 (channel strategy, dynamic per-token activations). At
decode and target-verify sizes (M <= 56: 14 streams x 4 verify tokens) the stock
path runs a per-token activation quant kernel and then a CUTLASS/cuBLAS FP8 GEMM
tuned for large M. Here activations stay BF16, each FP8 weight tile is widened
to BF16 in registers, and the weight streams once with split-K, the same shape
of kernel as the low-M BF16 GEMM (Daniel's patch 1).

Installed into the Pennyroyal venv by ARC3-Inference's pennyroyal_patches.py;
enabled with SGLANG_SM120_LOWM_FP8_GEMM=1. Each (kind, M bucket, N, K) is
checked once against a dense reference outside CUDA-graph capture
(SGLANG_SM120_LOWM_SELFCHECK=0 skips it); a bucket that fails is logged as an
error and served by the stock path from then on.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MAX_M = int(os.environ.get("SGLANG_SM120_LOWM_MAX_M", "32"))
_MIN_WEIGHT_BYTES = 256 * 1024
_MAX_WEIGHT_BYTES = 128 * 1024 * 1024
_NUM_SMS = 188  # RTX PRO 6000 Blackwell
_SELFCHECK = os.environ.get("SGLANG_SM120_LOWM_SELFCHECK", "1") != "0"
_SELFCHECK_TOL = 2e-2
_checked: dict = {}  # (kind, M bucket, N, K) -> passed


@triton.jit
def _sm120_lowm_fp8_gemm_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.arange(0, BLOCK_M)
    n_mask = rn < N
    m_mask = rm < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_per = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per
    k_end = tl.minimum(k_start + k_per, K)
    for k0 in tl.range(k_start, k_end, BLOCK_K, num_stages=NUM_STAGES):
        rk = k0 + tl.arange(0, BLOCK_K)
        k_mask = rk < k_end
        xb = tl.load(
            x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        wb = tl.load(
            w_ptr + rn[:, None] * stride_wn + rk[None, :] * stride_wk,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        acc += tl.dot(xb, tl.trans(wb), out_dtype=tl.float32)
    scale = tl.load(s_ptr + rn, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc * scale[None, :]
    if OUT_BF16:
        tl.store(
            out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on,
            acc.to(tl.bfloat16),
            mask=m_mask[:, None] & n_mask[None, :],
        )
    else:
        tl.atomic_add(
            out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
            sem="relaxed",
        )


def _tactic(m: int, n: int, k: int) -> tuple[int, int, int, int, int]:
    """(split_k, block_n, block_k, num_warps, num_stages), chosen by rule.

    Not measured yet. M > 32 (the 56-token verify batch) uses 64-wide N tiles
    so the 64-row activation tile is not re-read for every 16-32 columns; split
    K until the grid covers about two waves of the 188 SMs, keeping each split
    at least two 128-wide K steps.
    """
    block_n = 64 if m > 32 else 32
    block_k = 128
    n_ctas = triton.cdiv(n, block_n)
    split_k = 1
    while (
        split_k < 16
        and n_ctas * split_k < 2 * _NUM_SMS
        and k // (split_k * 2) >= 2 * block_k
    ):
        split_k *= 2
    num_stages = 3 if m > 32 else 4
    return (split_k, block_n, block_k, 4, num_stages)


def use_sm120_lowm_fp8_gemm(m: int, n: int, k: int) -> bool:
    return (
        0 < m <= _MAX_M
        and k % 16 == 0
        and _MIN_WEIGHT_BYTES <= n * k <= _MAX_WEIGHT_BYTES
    )


def _launch(
    x_2d: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    m, k = x_2d.shape
    n = weight.shape[0]
    split_k, block_n, block_k, num_warps, num_stages = _tactic(m, n, k)
    block_m = max(16, triton.next_power_of_2(m))
    grid = (triton.cdiv(n, block_n), split_k)
    out_bf16 = split_k == 1
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x_2d.device)
    acc = out if out_bf16 else torch.zeros((m, n), dtype=torch.float32, device=x_2d.device)
    _sm120_lowm_fp8_gemm_kernel[grid](
        x_2d,
        weight,
        scale,
        acc,
        m,
        n,
        k,
        x_2d.stride(0),
        x_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        acc.stride(0),
        acc.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_M=block_m,
        SPLIT_K=split_k,
        NUM_STAGES=num_stages,
        OUT_BF16=out_bf16,
        num_warps=num_warps,
    )
    if not out_bf16:
        out.copy_(acc)
    return out


def sm120_lowm_fp8_gemm(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor | None:
    """out = (x @ weight.T) * weight_scale for bf16 x[..., K], fp8 weight[N, K]
    (row-major) and one fp32 scale per output row; bias-free, M <= _MAX_M.
    None if this shape failed its self-check: the caller uses the stock path."""
    x_2d = x.reshape(-1, x.shape[-1])
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    scale = weight_scale.reshape(-1)
    if not maybe_selfcheck(
        "fp8",
        x_2d.shape[0],
        weight,
        lambda xs: _launch(xs, weight, scale),
        lambda xs: (xs.float() @ weight.float().T) * scale.float()[None, :],
    ):
        return None
    out = _launch(x_2d, weight, scale)
    return out.view(*x.shape[:-1], weight.shape[0])


def maybe_selfcheck(
    kind: str,
    m: int,
    weight: torch.Tensor,
    run: Callable[[torch.Tensor], torch.Tensor],
    reference: Callable[[torch.Tensor], torch.Tensor],
) -> bool:
    """Compare a low-M kernel with a dense reference once per shape bucket;
    False if that bucket failed, so the caller serves it with the stock path.

    Uses its own random activations, so warmup batches of zeros cannot hide a
    wrong result. Not run while a CUDA graph is being captured (the check
    synchronizes); CUDA-graph warmup runs each captured size eagerly first.
    """
    if not _SELFCHECK:
        return True
    n, k = weight.shape
    key = (kind, max(16, triton.next_power_of_2(m)), n, k)
    if key in _checked:
        return _checked[key]
    if torch.cuda.is_current_stream_capturing():
        return True
    gen = torch.Generator(device=weight.device).manual_seed(1234 + m)
    xs = torch.randn((m, k), generator=gen, device=weight.device, dtype=torch.float32)
    xs = xs.to(torch.bfloat16)
    try:
        got = run(xs).float()
        want = reference(xs).float()
        err = (got - want).abs().max().item()
        peak = want.abs().max().item()
        rel = err / max(peak, 1e-6)
    except Exception:  # a tile that does not compile is a failed bucket too
        logger.exception("SM120 low-M %s GEMM failed to run for m=%d n=%d k=%d", kind, m, n, k)
        err = peak = rel = float("nan")
    tactic = _tactic(m, n, k) if kind == "fp8" else None
    _checked[key] = rel <= _SELFCHECK_TOL
    if not _checked[key]:
        logger.error(
            "SM120 low-M %s GEMM self-check FAILED for m=%d n=%d k=%d: max error "
            "%.4g vs max |ref| %.4g (rel %.3g), tactic %s; this shape bucket uses "
            "the stock path",
            kind,
            m,
            n,
            k,
            err,
            peak,
            rel,
            tactic,
        )
        return False
    logger.info(
        "SM120 low-M %s GEMM self-check ok: m=%d n=%d k=%d rel_err=%.2e tactic=%s",
        kind,
        m,
        n,
        k,
        rel,
        tactic,
    )
    return True
