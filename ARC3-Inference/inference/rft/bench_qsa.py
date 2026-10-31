"""Time the fused QSA kernels at one layer's real shape and split the dK/dV kernel's time.

One QSA layer at the checkpoint's head layout (24 query heads, 2 KV heads, D = 256, 4-token blocks, top 512)
on a realistic selection (bench_layer.realistic_blocks: the real model's recency share and the overlap of 16
neighbouring queries). Times the forward, the dQ kernel and the union dK/dV kernel, then reruns a copy of the
dK/dV kernel that stores its tiles instead of adding them atomically, or writes nothing, which splits its time
into the atomic adds, the writes and the rest. The copy's atomic mode is checked against the original first.
CUDA only, no checkpoint; one JSON object on stdout.

  PYTHONPATH=. python -m inference.rft.bench_qsa [--n 64893] [--repeats 5]
"""

from __future__ import annotations

import argparse
import json

import torch
import triton
import triton.language as tl

from inference.rft import qsa_kernel
from inference.rft.bench_layer import realistic_blocks, selection_stats

H, HKV, D, RATIO, TOPK = 24, 2, 256, 4, 512


@triton.autotune(configs=[triton.Config({"BLOCK_N": n}, num_warps=w, num_stages=1) for n in (16, 32) for w in (4, 8)],
                 key=["G", "D", "R", "QT", "WRITE"], reset_to_zero=["DK", "DV"])
@triton.jit
def _dkv_union_variant(Q, K, V, UB, UM, DO, LSE, DELTA, DK, DV, SINK, scale, scale2, N, U, NK, Q_OFF, KEY_START,
                       sq_n, sq_h, sk_n, sk_h, sv_n, sv_h, su_t, sdo_n, sdo_h, sl_n,
                       sdk_n, sdk_h, sdv_n, sdv_h,
                       G: tl.constexpr, R: tl.constexpr, D: tl.constexpr, QT: tl.constexpr,
                       HC: tl.constexpr, BLOCK_N: tl.constexpr, WRITE: tl.constexpr):
    """qsa_kernel._dkv_union_kernel with a selectable write: 0 atomic add (the original), 1 plain store (wrong
    sums, the same bytes written), 2 nothing (a per-program checksum keeps the compute from being dropped)."""
    t = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    ROWS: tl.constexpr = QT * HC
    offs_r = tl.arange(0, ROWS)
    rq = offs_r // HC
    rh = offs_r % HC
    query = t * QT + rq
    qmask = query < N
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    total = tl.zeros([1], tl.float32)  # a tensor, not a Python float: loop-carried values keep one type
    for start in range(0, U * R, BLOCK_N):
        j = start + offs_n
        uj = j // R
        blk = tl.load(UB + t * su_t + uj, mask=uj < U, other=-1)
        bits = tl.load(UM + t * su_t + uj, mask=uj < U, other=0)
        tok = blk.to(tl.int64) * R + (j % R)
        tvalid = (blk >= 0) & (tok < NK) & (tok >= KEY_START)
        if tl.max(tvalid.to(tl.int32), 0) > 0:
            kt = tl.load(K + tok[:, None] * sk_n + g * sk_h + offs_d[None, :], mask=tvalid[:, None], other=0.0)
            vt = tl.load(V + tok[:, None] * sv_n + g * sv_h + offs_d[None, :], mask=tvalid[:, None], other=0.0)
            dk = tl.zeros([BLOCK_N, D], tl.float32)
            dv = tl.zeros([BLOCK_N, D], tl.float32)
            selected = ((bits[None, :] >> rq[:, None]) & 1) == 1
            allowed = selected & tvalid[None, :] & (tok[None, :] <= query[:, None] + Q_OFF) & qmask[:, None]
            for c in range(0, G // HC):
                heads = g * G + c * HC + rh
                q = tl.load(Q + query[:, None] * sq_n + heads[:, None] * sq_h + offs_d[None, :],
                            mask=qmask[:, None], other=0.0)
                do = tl.load(DO + query[:, None] * sdo_n + heads[:, None] * sdo_h + offs_d[None, :],
                             mask=qmask[:, None], other=0.0)
                lse = tl.load(LSE + query * sl_n + heads, mask=qmask, other=0.0)
                delta = tl.load(DELTA + query * sl_n + heads, mask=qmask, other=0.0)
                s = tl.dot(q, tl.trans(kt)) * scale2
                p = tl.where(allowed, tl.math.exp2(s - lse[:, None]), 0.0)
                dp = tl.dot(do, tl.trans(vt))
                ds = tl.where(allowed, p * (dp - delta[:, None]), 0.0)
                dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
                dv += tl.dot(tl.trans(p.to(do.dtype)), do)
            if WRITE == 0:
                tl.atomic_add(DK + tok[:, None] * sdk_n + g * sdk_h + offs_d[None, :], dk * scale,
                              mask=tvalid[:, None], sem="relaxed")
                tl.atomic_add(DV + tok[:, None] * sdv_n + g * sdv_h + offs_d[None, :], dv,
                              mask=tvalid[:, None], sem="relaxed")
            elif WRITE == 1:
                tl.store(DK + tok[:, None] * sdk_n + g * sdk_h + offs_d[None, :], dk * scale, mask=tvalid[:, None])
                tl.store(DV + tok[:, None] * sdv_n + g * sdv_h + offs_d[None, :], dv, mask=tvalid[:, None])
            else:
                total += tl.sum(tl.sum(dk, axis=1), axis=0) + tl.sum(tl.sum(dv, axis=1), axis=0)
    if WRITE == 2:
        tl.store(SINK + t * 2 + g + tl.arange(0, 1), total)


def time_ms(fn, repeats: int) -> float:
    """Median of `repeats` timed calls after one untimed call (Triton's autotuning runs on the first)."""
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return round(sorted(times)[len(times) // 2], 2)


def rel(a, b) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=64893, help="tokens (the 64K sample of the 10-20 runs)")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    device = "cuda"
    n, g = args.n, H // HKV
    scale = D ** -0.5
    gen = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(n, H, D, generator=gen, device=device).bfloat16()
    k = torch.randn(n, HKV, D, generator=gen, device=device).bfloat16()
    v = torch.randn(n, HKV, D, generator=gen, device=device).bfloat16()
    dout = torch.randn(n, H, D, generator=gen, device=device).bfloat16()
    blocks = realistic_blocks(n, TOPK, RATIO, device=device)
    out, lse = qsa_kernel.qsa_forward(q, k, v, blocks, scale, RATIO)
    delta = (out.float() * dout.float()).sum(-1)

    qt = qsa_kernel.UNION_QUERIES
    hc = 4 if g % 4 == 0 else (2 if g % 2 == 0 else 1)  # as qsa_kernel.qsa_backward
    ub, um = qsa_kernel.union_tiles(blocks, qt)
    union_tokens = int((ub >= 0).sum()) * RATIO
    selected_tokens = int((blocks >= 0).sum()) * RATIO
    sink = torch.zeros(ub.shape[0] * HKV, device=device, dtype=torch.float32)

    def dq_only():
        dq = torch.empty_like(q)
        qsa_kernel._bwd_kernel[(n, HKV)](
            q, k, v, blocks, dout, lse, delta, dq, sink, sink, scale, scale * qsa_kernel.LOG2E, blocks.shape[1], 0, 0,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), blocks.stride(0),
            dout.stride(0), dout.stride(1), lse.stride(0), dq.stride(0), dq.stride(1), 0, 0, 0, 0,
            G=g, R=RATIO, D=D, BLOCK_H=qsa_kernel._block_h(g), WITH_DKV=False)

    def dkv(write: int, dk, dv):
        _dkv_union_variant[(ub.shape[0], HKV)](
            q, k, v, ub, um, dout, lse, delta, dk, dv, sink, scale, scale * qsa_kernel.LOG2E, n, ub.shape[1],
            k.shape[0], 0, 0,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), ub.stride(0),
            dout.stride(0), dout.stride(1), lse.stride(0), dk.stride(0), dk.stride(1), dv.stride(0), dv.stride(1),
            G=g, R=RATIO, D=D, QT=qt, HC=hc, WRITE=write)

    # the copy's atomic mode must reproduce the production kernel before its variants mean anything
    _, dk_ref, dv_ref = qsa_kernel.qsa_backward(q, k, v, blocks, out, lse, dout, scale, RATIO, union=True)
    dk32 = torch.zeros(k.shape, device=device, dtype=torch.float32)
    dv32 = torch.zeros(v.shape, device=device, dtype=torch.float32)
    dkv(0, dk32, dv32)  # first call autotunes; reset_to_zero restores the buffers after tuning
    dk32.zero_(), dv32.zero_()
    dkv(0, dk32, dv32)
    check = dict(dk=rel(dk32.to(k.dtype), dk_ref), dv=rel(dv32.to(v.dtype), dv_ref))
    assert max(check.values()) < 1e-3, check

    times = dict(
        forward=time_ms(lambda: qsa_kernel.qsa_forward(q, k, v, blocks, scale, RATIO), args.repeats),
        union_tiles=time_ms(lambda: qsa_kernel.union_tiles(blocks, qt), args.repeats),
        dq=time_ms(dq_only, args.repeats),
        backward_union=time_ms(lambda: qsa_kernel.qsa_backward(q, k, v, blocks, out, lse, dout, scale, RATIO,
                                                               union=True), args.repeats),
        backward_per_query=time_ms(lambda: qsa_kernel.qsa_backward(q, k, v, blocks, out, lse, dout, scale, RATIO,
                                                                   union=False), args.repeats),
        dkv_atomic=time_ms(lambda: dkv(0, dk32, dv32), args.repeats),
        dkv_store=time_ms(lambda: dkv(1, dk32, dv32), args.repeats),
        dkv_none=time_ms(lambda: dkv(2, dk32, dv32), args.repeats),
    )
    union_flops = union_tokens * qt * H * D * 2 * 4  # s, dP, dK, dV over every (tile query, union token, head)
    useful_flops = selected_tokens * H * D * 2 * 4
    atomics = union_tokens * D * 2 * HKV
    result = dict(
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, triton=triton.__version__,
        n=n, union_queries=qt, heads_per_chunk=hc, check_vs_production=check,
        selection=selection_stats(blocks), union_width=int(ub.shape[1]),
        union_over_selected=round(union_tokens * qt / max(selected_tokens, 1), 2),
        tflop_union=round(union_flops / 1e12, 2), tflop_useful=round(useful_flops / 1e12, 2),
        atomics_g=round(atomics / 1e9, 1), times_ms=times,
        dkv_split=dict(
            atomic_adds_ms=round(times["dkv_atomic"] - times["dkv_store"], 1),
            stores_ms=round(times["dkv_store"] - times["dkv_none"], 1),
            compute_and_loads_ms=times["dkv_none"],
            union_tflops_per_s=round(union_flops / 1e12 / (times["dkv_none"] / 1e3), 1)),
    )
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
