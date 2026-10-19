"""Fused QSA sparse attention (Triton): forward and backward without gathering keys.

Each query attends to its own selection of `ratio`-token blocks ([N, NB] block ids,
-1 = none, the last column being the query's own incomplete block). One program
handles one query and one KV head, i.e. the G query heads sharing that head (12 here,
padded to 16 rows): it reads the selected keys and values straight from K and V,
runs an online softmax and stores the output and the log-sum-exp (base 2).

Backward recomputes the probabilities from the LSE. dQ comes from a per-query kernel
like the forward, written once. dK and dV come from tiles of 16 neighbouring queries:
each tile walks the union of its queries' blocks (with a bit mask of which query chose
which block), reads every key once and adds its gradient with one FP32 atomic per
element into FP32 buffers. Neighbours overlap a lot (16 queries select ~3.5x one
query's tokens), so this needs ~4.6x fewer atomics and reads than per query.

The PyTorch path this replaces (qsa._attend_chunk) gathered [chunk, 2051, 2, 256]
keys and values per chunk of queries, ran SDPA on 12-row problems and scattered
the gradients back with a BF16 index_add.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs (the Mac): qsa.py keeps its PyTorch path
    triton = None

LOG2E = 1.4426950408889634


def available(*tensors) -> bool:
    return triton is not None and all(t.is_cuda for t in tensors)


if triton is not None:
    def _configs(block_ns):
        return [triton.Config({"BLOCK_N": n}, num_warps=w, num_stages=s)
                for n in block_ns for w in (4, 8) for s in (1, 2)]

    @triton.autotune(configs=_configs((32, 64)), key=["G", "D", "R"])
    @triton.jit
    def _fwd_kernel(Q, K, V, B, O, LSE, scale2, NB,
                    sq_n, sq_h, sk_n, sk_h, sv_n, sv_h, sb_n, so_n, so_h, sl_n,
                    G: tl.constexpr, R: tl.constexpr, D: tl.constexpr, BLOCK_H: tl.constexpr,
                    BLOCK_N: tl.constexpr):
        qi = tl.program_id(0).to(tl.int64)
        g = tl.program_id(1)
        offs_h = tl.arange(0, BLOCK_H)
        hmask = offs_h < G
        heads = g * G + offs_h
        offs_d = tl.arange(0, D)
        offs_n = tl.arange(0, BLOCK_N)
        q = tl.load(Q + qi * sq_n + heads[:, None] * sq_h + offs_d[None, :], mask=hmask[:, None], other=0.0)
        m_i = tl.full([BLOCK_H], -1e30, tl.float32)  # finite: a fully masked tile must not give inf - inf
        l_i = tl.zeros([BLOCK_H], tl.float32)
        acc = tl.zeros([BLOCK_H, D], tl.float32)
        for start in range(0, NB * R, BLOCK_N):
            j = start + offs_n
            bj = j // R
            blk = tl.load(B + qi * sb_n + bj, mask=bj < NB, other=-1)
            tok = blk.to(tl.int64) * R + (j % R)
            valid = (blk >= 0) & (tok <= qi)
            kt = tl.load(K + tok[:, None] * sk_n + g * sk_h + offs_d[None, :], mask=valid[:, None], other=0.0)
            s = tl.dot(q, tl.trans(kt)) * scale2
            s = tl.where(valid[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.math.exp2(s - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            vt = tl.load(V + tok[:, None] * sv_n + g * sv_h + offs_d[None, :], mask=valid[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
            m_i = m_new
        acc = acc / l_i[:, None]
        tl.store(O + qi * so_n + heads[:, None] * so_h + offs_d[None, :], acc.to(O.dtype.element_ty),
                 mask=hmask[:, None])
        tl.store(LSE + qi * sl_n + heads, m_i + tl.math.log2(l_i), mask=hmask)

    @triton.autotune(configs=_configs((16, 32, 64)), key=["G", "D", "R", "WITH_DKV"], reset_to_zero=["DK", "DV"])
    @triton.jit
    def _bwd_kernel(Q, K, V, B, DO, LSE, DELTA, DQ, DK, DV, scale, scale2, NB,
                    sq_n, sq_h, sk_n, sk_h, sv_n, sv_h, sb_n, sdo_n, sdo_h, sl_n, sdq_n, sdq_h,
                    sdk_n, sdk_h, sdv_n, sdv_h,
                    G: tl.constexpr, R: tl.constexpr, D: tl.constexpr, BLOCK_H: tl.constexpr,
                    BLOCK_N: tl.constexpr, WITH_DKV: tl.constexpr):
        qi = tl.program_id(0).to(tl.int64)
        g = tl.program_id(1)
        offs_h = tl.arange(0, BLOCK_H)
        hmask = offs_h < G
        heads = g * G + offs_h
        offs_d = tl.arange(0, D)
        offs_n = tl.arange(0, BLOCK_N)
        q = tl.load(Q + qi * sq_n + heads[:, None] * sq_h + offs_d[None, :], mask=hmask[:, None], other=0.0)
        do = tl.load(DO + qi * sdo_n + heads[:, None] * sdo_h + offs_d[None, :], mask=hmask[:, None], other=0.0)
        lse = tl.load(LSE + qi * sl_n + heads, mask=hmask, other=0.0)
        delta = tl.load(DELTA + qi * sl_n + heads, mask=hmask, other=0.0)
        dq = tl.zeros([BLOCK_H, D], tl.float32)
        for start in range(0, NB * R, BLOCK_N):
            j = start + offs_n
            bj = j // R
            blk = tl.load(B + qi * sb_n + bj, mask=bj < NB, other=-1)
            tok = blk.to(tl.int64) * R + (j % R)
            valid = (blk >= 0) & (tok <= qi)
            kt = tl.load(K + tok[:, None] * sk_n + g * sk_h + offs_d[None, :], mask=valid[:, None], other=0.0)
            vt = tl.load(V + tok[:, None] * sv_n + g * sv_h + offs_d[None, :], mask=valid[:, None], other=0.0)
            s = tl.dot(q, tl.trans(kt)) * scale2
            p = tl.math.exp2(s - lse[:, None])
            p = tl.where(valid[None, :] & hmask[:, None], p, 0.0)
            dp = tl.dot(do, tl.trans(vt))
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(kt.dtype), kt)
            if WITH_DKV:
                dk = tl.dot(tl.trans(ds.to(q.dtype)), q) * scale
                dv = tl.dot(tl.trans(p.to(do.dtype)), do)
                tl.atomic_add(DK + tok[:, None] * sdk_n + g * sdk_h + offs_d[None, :], dk, mask=valid[:, None],
                              sem="relaxed")
                tl.atomic_add(DV + tok[:, None] * sdv_n + g * sdv_h + offs_d[None, :], dv, mask=valid[:, None],
                              sem="relaxed")
        tl.store(DQ + qi * sdq_n + heads[:, None] * sdq_h + offs_d[None, :], (dq * scale).to(DQ.dtype.element_ty),
                 mask=hmask[:, None])

    @triton.autotune(configs=[triton.Config({"BLOCK_N": n}, num_warps=w, num_stages=1)
                              for n in (16, 32) for w in (4, 8)],
                     key=["G", "D", "R", "QT"], reset_to_zero=["DK", "DV"])
    @triton.jit
    def _dkv_union_kernel(Q, K, V, UB, UM, DO, LSE, DELTA, DK, DV, scale, scale2, N, U,
                          sq_n, sq_h, sk_n, sk_h, sv_n, sv_h, su_t, sdo_n, sdo_h, sl_n,
                          sdk_n, sdk_h, sdv_n, sdv_h,
                          G: tl.constexpr, R: tl.constexpr, D: tl.constexpr, QT: tl.constexpr,
                          HC: tl.constexpr, BLOCK_N: tl.constexpr):
        """dK, dV from a tile of QT consecutive queries over the union of their selected blocks.

        UB [T, U] union block ids (-1 = none), UM [T, U] bit j set when the tile's query j selected
        the block. Rows are (query, head) pairs, HC heads at a time; K/V tiles are read once per
        tile and their gradients added with one FP32 atomic per element."""
        t = tl.program_id(0).to(tl.int64)
        g = tl.program_id(1)
        ROWS: tl.constexpr = QT * HC
        offs_r = tl.arange(0, ROWS)
        rq = offs_r // HC  # query within the tile
        rh = offs_r % HC  # head within the chunk
        query = t * QT + rq
        qmask = query < N
        offs_d = tl.arange(0, D)
        offs_n = tl.arange(0, BLOCK_N)
        for start in range(0, U * R, BLOCK_N):
            j = start + offs_n
            uj = j // R
            blk = tl.load(UB + t * su_t + uj, mask=uj < U, other=-1)
            bits = tl.load(UM + t * su_t + uj, mask=uj < U, other=0)
            tok = blk.to(tl.int64) * R + (j % R)
            # the tail block of a query ending a block is the next one, past the last query for the final tile:
            # never read or write keys at or after N
            tvalid = (blk >= 0) & (tok < N)
            if tl.max(tvalid.to(tl.int32), 0) > 0:
                kt = tl.load(K + tok[:, None] * sk_n + g * sk_h + offs_d[None, :], mask=tvalid[:, None], other=0.0)
                vt = tl.load(V + tok[:, None] * sv_n + g * sv_h + offs_d[None, :], mask=tvalid[:, None], other=0.0)
                dk = tl.zeros([BLOCK_N, D], tl.float32)
                dv = tl.zeros([BLOCK_N, D], tl.float32)
                selected = ((bits[None, :] >> rq[:, None]) & 1) == 1
                allowed = selected & tvalid[None, :] & (tok[None, :] <= query[:, None]) & qmask[:, None]
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
                tl.atomic_add(DK + tok[:, None] * sdk_n + g * sdk_h + offs_d[None, :], dk * scale,
                              mask=tvalid[:, None], sem="relaxed")
                tl.atomic_add(DV + tok[:, None] * sdv_n + g * sdv_h + offs_d[None, :], dv,
                              mask=tvalid[:, None], sem="relaxed")


def _block_h(g: int) -> int:
    return max(16, triton.next_power_of_2(g))


def qsa_forward(q, k, v, blocks, scale: float, ratio: int):
    """q [N, H, D], k/v [N, Hkv, D], blocks [N, NB] int32 -> (out [N, H, D], lse [N, H] base 2)."""
    n, h, d = q.shape
    hkv = k.shape[1]
    out = torch.empty_like(q)
    lse = torch.empty(n, h, device=q.device, dtype=torch.float32)
    _fwd_kernel[(n, hkv)](q, k, v, blocks, out, lse, scale * LOG2E, blocks.shape[1],
                          q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
                          blocks.stride(0), out.stride(0), out.stride(1), lse.stride(0),
                          G=h // hkv, R=ratio, D=d, BLOCK_H=_block_h(h // hkv))
    return out, lse


@torch.no_grad()
def union_tiles(blocks: torch.Tensor, qt: int):
    """Per tile of `qt` consecutive queries: the union of their selected blocks [T, U] (-1 = none)
    and, per union block, a bit mask of the tile's queries that selected it [T, U] (int32)."""
    n, nb = blocks.shape
    tiles = -(-n // qt)
    padded = torch.full((tiles * qt, nb), -1, dtype=torch.int32, device=blocks.device)
    padded[:n] = blocks
    local = torch.arange(qt, device=blocks.device, dtype=torch.int32).repeat_interleave(nb)
    flat = padded.view(tiles, qt * nb)
    big = torch.iinfo(torch.int32).max
    keys = torch.where(flat >= 0, flat * qt + local, big).sort(dim=1).values  # a query selects a block once
    valid = keys != big
    block = torch.where(valid, keys // qt, -1)
    first = valid & torch.cat([torch.ones_like(valid[:, :1]), block[:, 1:] != block[:, :-1]], dim=1)
    slot = first.cumsum(dim=1) - 1
    width = int(first.sum(dim=1).max().clamp_min(1))
    union = torch.full((tiles, width), -1, dtype=torch.int32, device=blocks.device)
    rows = torch.arange(tiles, device=blocks.device)[:, None].expand_as(slot)
    union[rows[first], slot[first]] = block[first]
    bits = torch.zeros((tiles, width), dtype=torch.int32, device=blocks.device)
    bit = torch.bitwise_left_shift(torch.ones_like(keys), keys % qt)
    bits.index_put_((rows[valid], slot[valid]), bit[valid], accumulate=True)  # distinct bits: sum = or
    return union, bits


UNION_QUERIES = 16  # queries per dK/dV tile: 16 neighbours select ~3.5x one query's tokens (10-19 overlap job)


def qsa_backward(q, k, v, blocks, out, lse, dout, scale: float, ratio: int, union: bool = True):
    n, h, d = q.shape
    hkv = k.shape[1]
    g = h // hkv
    dout = dout.contiguous()
    delta = (out.float() * dout.float()).sum(-1)  # [N, H], rowsum(O * dO)
    dq = torch.empty_like(q)
    dk = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
    dv = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
    _bwd_kernel[(n, hkv)](q, k, v, blocks, dout, lse, delta, dq, dk, dv, scale, scale * LOG2E, blocks.shape[1],
                          q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
                          blocks.stride(0), dout.stride(0), dout.stride(1), lse.stride(0), dq.stride(0), dq.stride(1),
                          dk.stride(0), dk.stride(1), dv.stride(0), dv.stride(1),
                          G=g, R=ratio, D=d, BLOCK_H=_block_h(g), WITH_DKV=not union)
    if union:
        qt = UNION_QUERIES
        heads_per_chunk = 4 if g % 4 == 0 else (2 if g % 2 == 0 else 1)
        ub, um = union_tiles(blocks, qt)
        _dkv_union_kernel[(ub.shape[0], hkv)](
            q, k, v, ub, um, dout, lse, delta, dk, dv, scale, scale * LOG2E, n, ub.shape[1],
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), ub.stride(0),
            dout.stride(0), dout.stride(1), lse.stride(0), dk.stride(0), dk.stride(1), dv.stride(0), dv.stride(1),
            G=g, R=ratio, D=d, QT=qt, HC=heads_per_chunk)
    return dq, dk.to(k.dtype), dv.to(v.dtype)


UNION_DKV = True  # dK/dV per tile of neighbouring queries (False: per query, one atomic per element each)


class QSAAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, blocks, scale, ratio):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out, lse = qsa_forward(q, k, v, blocks, scale, ratio)
        ctx.save_for_backward(q, k, v, blocks, out, lse)
        ctx.scale, ctx.ratio = scale, ratio
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, blocks, out, lse = ctx.saved_tensors
        dq, dk, dv = qsa_backward(q, k, v, blocks, out, lse, dout, ctx.scale, ctx.ratio, union=UNION_DKV)
        return dq, dk, dv, None, None, None


def qsa_attention(q, k, v, blocks, scale: float, ratio: int):
    """Sparse attention of each query over its selected blocks; differentiable in q, k and v."""
    return QSAAttention.apply(q, k, v, blocks.to(torch.int32).contiguous(), scale, ratio)
