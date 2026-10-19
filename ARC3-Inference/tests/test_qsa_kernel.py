"""inference/rl/qsa_kernel.py (Triton, CUDA only) computes the same sparse attention as a dense FP32
reference with the same selection, at the checkpoint's head layout (24 query heads, 2 KV heads, D = 256).

The PyTorch path it replaces (qsa._attend_chunk) runs on the same BF16 inputs as a yardstick: the kernel
must be at least as close to the reference as that path."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the QSA kernel needs CUDA", allow_module_level=True)
pytest.importorskip("triton")

from inference.rl import qsa, qsa_kernel  # noqa: E402

from test_qsa_kernel_common import RATIO, random_blocks  # noqa: E402


def dense_reference(q, k, v, tokens, scale):
    """FP32 attention of each query over its selected tokens, via a dense [N, N] mask."""
    n, h, d = q.shape
    group = h // k.shape[1]
    mask = torch.zeros(n, n + 1, dtype=torch.bool, device=q.device)
    mask.scatter_(1, torch.where(tokens >= 0, tokens, n), True)
    mask = mask[:, :n]
    keys, values = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, keys) * scale
    probs = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, values)


def run(fn, q, k, v, upstream):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    out = fn(q, k, v)
    (out.float() * upstream).sum().backward()
    return out.detach().float(), q.grad.float(), k.grad.float(), v.grad.float()


def rel(a, b):
    return float((a - b).norm() / b.norm())


@pytest.mark.parametrize("union", [True, False])
@pytest.mark.parametrize("n,topk", [(1027, 64), (2048, 16)])
def test_kernel_matches_dense_reference(n, topk, union, monkeypatch):
    monkeypatch.setattr(qsa_kernel, "UNION_DKV", union)
    g = torch.Generator().manual_seed(n)
    h, hkv, d = 24, 2, 256
    scale = d ** -0.5
    q = torch.randn(n, h, d, generator=g).cuda()
    k = torch.randn(n, hkv, d, generator=g).cuda()
    v = torch.randn(n, hkv, d, generator=g).cuda()
    upstream = torch.randn(n, h, d, generator=g).cuda()
    blocks = random_blocks(n, topk, g).cuda()
    tokens = qsa.blocks_to_tokens(blocks, RATIO, topk * RATIO + RATIO - 1)

    bf = [t.bfloat16() for t in (q, k, v)]
    reference = run(lambda *x: dense_reference(*x, tokens, scale), *[t.float() for t in bf], upstream)
    ours = run(lambda *x: qsa_kernel.qsa_attention(*x, blocks, scale, RATIO), *bf, upstream)
    chunked = run(lambda qq, kk, vv: torch.cat([qsa._attend_chunk(qq[i:i + 256], kk, vv, tokens[i:i + 256], scale)
                                                for i in range(0, n, 256)]), *bf, upstream)
    for name, a, b, c in zip(("out", "dq", "dk", "dv"), ours, reference, chunked):
        assert torch.isfinite(a).all(), name
        assert rel(a, b) < 1e-2, (name, rel(a, b))
        assert rel(a, b) <= 1.2 * rel(c, b) + 1e-3, (name, rel(a, b), rel(c, b))


def test_lse_matches_reference():
    g = torch.Generator().manual_seed(0)
    n, h, hkv, d = 512, 24, 2, 256
    q = torch.randn(n, h, d, generator=g).cuda().bfloat16()
    k = torch.randn(n, hkv, d, generator=g).cuda().bfloat16()
    v = torch.randn(n, hkv, d, generator=g).cuda().bfloat16()
    blocks = random_blocks(n, 32, g).cuda()
    _, lse = qsa_kernel.qsa_forward(q, k, v, blocks, d ** -0.5, RATIO)
    tokens = qsa.blocks_to_tokens(blocks, RATIO, 32 * RATIO + RATIO - 1)
    keys = k.float().repeat_interleave(h // hkv, dim=1)
    scores = torch.einsum("qhd,khd->qhk", q.float(), keys) * d ** -0.5
    mask = torch.zeros(n, n + 1, dtype=torch.bool, device="cuda")
    mask.scatter_(1, torch.where(tokens >= 0, tokens, n), True)
    reference = torch.logsumexp(scores.masked_fill(~mask[:, None, :n], float("-inf")), dim=-1) / torch.log(
        torch.tensor(2.0))
    torch.testing.assert_close(lse, reference, rtol=1e-3, atol=1e-2)
